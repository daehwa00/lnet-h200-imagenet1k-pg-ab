"""Audit-preserving nonlinear corrections for staged modal descriptors."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations

import torch
from torch import Tensor, nn


class FixedAffineMain(nn.Module):
    """Freeze a trained BatchNorm-plus-linear affine decision surface."""

    def __init__(
        self,
        running_mean: Tensor,
        running_var: Tensor,
        weight: Tensor,
        bias: Tensor,
        *,
        eps: float,
    ) -> None:
        super().__init__()
        input_dim = running_mean.numel()
        if running_var.shape != running_mean.shape:
            message = "affine standardizer statistics must have matching shapes"
            raise ValueError(message)
        if weight.ndim != 2 or weight.shape[1] != input_dim:
            message = "affine weight is incompatible with its standardizer"
            raise ValueError(message)
        if bias.shape != (weight.shape[0],):
            message = "affine bias is incompatible with its weight"
            raise ValueError(message)
        self.input_dim = input_dim
        self.output_dim = weight.shape[0]
        self.eps = float(eps)
        self.register_buffer("running_mean", running_mean.detach().clone())
        self.register_buffer("running_var", running_var.detach().clone())
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("bias", bias.detach().clone())

    def standardized(self, descriptor: Tensor) -> Tensor:
        return (descriptor - self.running_mean) * torch.rsqrt(self.running_var + self.eps)

    def logits_from_standardized(self, standardized: Tensor) -> Tensor:
        return torch.nn.functional.linear(standardized, self.weight, self.bias)

    def forward(self, descriptor: Tensor) -> Tensor:
        return self.logits_from_standardized(self.standardized(descriptor))


class FixedStandardizedMain(nn.Module):
    """Freeze any trained head whose first operation is parameter-free BatchNorm."""

    def __init__(self, source: nn.Module, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        standardizer = getattr(source, "standardizer", None)
        if not isinstance(standardizer, nn.BatchNorm1d) or standardizer.affine:
            message = "fixed standardized main requires parameter-free BatchNorm input"
            raise TypeError(message)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = float(standardizer.eps)
        self.register_buffer("running_mean", standardizer.running_mean.detach().clone())
        self.register_buffer("running_var", standardizer.running_var.detach().clone())
        self.body = deepcopy(source).eval()
        self.body.standardizer = nn.Identity()  # type: ignore[attr-defined]
        self.body.requires_grad_(requires_grad=False)

    def standardized(self, descriptor: Tensor) -> Tensor:
        return (descriptor - self.running_mean) * torch.rsqrt(self.running_var + self.eps)

    def logits_from_standardized(self, standardized: Tensor) -> Tensor:
        return self.body(standardized)

    def forward(self, descriptor: Tensor) -> Tensor:
        return self.logits_from_standardized(self.standardized(descriptor))


class GatedCrossStageResidualHead(nn.Module):
    """Correct a fixed standardized main through low-rank stage interactions."""

    def __init__(
        self,
        main: FixedAffineMain | FixedStandardizedMain,
        *,
        stage_count: int,
        stage_dim: int,
        embedding_dim: int = 32,
        residual_width: int = 64,
        gated: bool = True,
        affine_margin_threshold: float | None = None,
        affine_margin_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if stage_count < 2 or stage_dim <= 0:
            message = "cross-stage residual requires at least two non-empty stages"
            raise ValueError(message)
        if stage_count * stage_dim != main.input_dim:
            message = "stage partition does not cover the main descriptor"
            raise ValueError(message)
        if embedding_dim <= 0 or residual_width <= 0:
            message = "residual widths must be positive"
            raise ValueError(message)
        if affine_margin_threshold is not None and not gated:
            message = "an affine-margin gate requires gating to be enabled"
            raise ValueError(message)
        if affine_margin_temperature <= 0.0:
            message = "affine-margin gate temperature must be positive"
            raise ValueError(message)
        self.main = main
        self.stage_count = stage_count
        self.stage_dim = stage_dim
        self.gated = gated
        self.affine_margin_threshold = affine_margin_threshold
        self.affine_margin_temperature = float(affine_margin_temperature)
        self.stage_embeddings = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(stage_dim, embedding_dim), nn.GELU())
                for _ in range(stage_count)
            ]
        )
        pair_count = stage_count * (stage_count - 1) // 2
        context_dim = embedding_dim * (stage_count + 2 * pair_count)
        self.context_norm = nn.RMSNorm(context_dim)
        self.residual = nn.Sequential(
            nn.Linear(context_dim, residual_width),
            nn.GELU(),
            nn.RMSNorm(residual_width),
            nn.Linear(residual_width, main.output_dim),
        )
        self.gate = (
            nn.Linear(context_dim, 1)
            if gated and affine_margin_threshold is None
            else None
        )
        # Zero starts at the exact trained affine surface. The random residual
        # supplies a gradient to beta on the first update; subsequent updates
        # then train the interaction and gate parameters.
        self.beta = nn.Parameter(torch.zeros(()))

    def interaction_context(self, standardized: Tensor) -> Tensor:
        stages = standardized.split(self.stage_dim, dim=-1)
        embeddings = [
            projection(stage)
            for projection, stage in zip(self.stage_embeddings, stages, strict=True)
        ]
        products: list[Tensor] = []
        differences: list[Tensor] = []
        for left, right in combinations(embeddings, 2):
            products.append(left * right)
            differences.append((left - right).abs())
        return self.context_norm(torch.cat((*embeddings, *products, *differences), dim=-1))

    def components(self, descriptor: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        standardized = self.main.standardized(descriptor)
        main_logits = self.main.logits_from_standardized(standardized)
        context = self.interaction_context(standardized)
        residual_logits = self.residual(context)
        if self.affine_margin_threshold is not None:
            top_two = main_logits.topk(2, dim=-1).values
            margin = top_two[:, :1] - top_two[:, 1:]
            gate = torch.sigmoid(
                (self.affine_margin_threshold - margin) / self.affine_margin_temperature
            )
        elif self.gate is None:
            gate = torch.ones(
                descriptor.shape[0], 1, device=descriptor.device, dtype=descriptor.dtype
            )
        else:
            gate = torch.sigmoid(self.gate(context))
        joint = main_logits + self.beta * gate * residual_logits
        return joint, main_logits, residual_logits, gate

    def forward(self, descriptor: Tensor) -> Tensor:
        return self.components(descriptor)[0]


__all__ = [
    "FixedAffineMain",
    "FixedStandardizedMain",
    "GatedCrossStageResidualHead",
]
