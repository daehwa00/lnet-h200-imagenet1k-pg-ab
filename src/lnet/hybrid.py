from __future__ import annotations

from typing import Final, Literal

import torch
from torch import Tensor, nn

from lnet.laplace import LaplaceParameterError, LaplaceShapeError
from lnet.tapped_prl import TapParameterization, TappedPRLBranch

BranchName = Literal["prl", "fir", "mlp"]
BranchNormalization = Literal["none", "rms", "layernorm"]
FusionVariant = Literal[
    "no_gate_sum",
    "fixed_equal",
    "fixed_learned_scalar",
    "softmax",
    "temperature_softmax",
]
BRANCH_ORDER: Final[tuple[BranchName, ...]] = ("prl", "fir", "mlp")


def _activation(name: Literal["identity", "tanh"]) -> nn.Module:
    match name:
        case "identity":
            return nn.Identity()
        case "tanh":
            return nn.Tanh()
        case unreachable:
            raise LaplaceParameterError(reason=f"unsupported activation: {unreachable}")


class HybridModalPRLBlock(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        output_dim: int,
        modes: int,
        fir_kernel_size: int = 9,
        prl_tap_kernel_size: int | None = None,
        tap_parameterization: TapParameterization = "shared_scalar",
        low_rank_rank: int = 2,
        mlp_hidden_dim: int | None = None,
        active_branches: tuple[BranchName, ...] = BRANCH_ORDER,
        fusion_variant: FusionVariant = "softmax",
        fusion_temperature: float = 1.0,
        branch_normalization: BranchNormalization = "none",
        branch_dropout_probability: float = 0.0,
        activation: Literal["identity", "tanh"] = "tanh",
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.raw_input_dim = raw_input_dim
        self.model_dim = model_dim
        self.fir_kernel_size = fir_kernel_size
        self.prl_tap_kernel_size = prl_tap_kernel_size or fir_kernel_size
        self.tap_parameterization = tap_parameterization
        self.low_rank_rank = low_rank_rank
        self.active_branches = active_branches
        self.fusion_variant = fusion_variant
        self.fusion_temperature = fusion_temperature
        self.branch_normalization = branch_normalization
        self.branch_dropout_probability = branch_dropout_probability
        if fusion_temperature <= 0.0:
            raise LaplaceParameterError(reason="fusion temperature must be positive")
        if not 0.0 <= branch_dropout_probability < 1.0:
            raise LaplaceParameterError(reason="branch dropout probability must be in [0, 1)")
        branch_mask = torch.tensor(
            [branch in active_branches for branch in BRANCH_ORDER], dtype=torch.bool
        )
        if not bool(branch_mask.any()):
            raise LaplaceParameterError(reason="at least one hybrid branch must be active")
        self._branch_mask: Tensor
        self.register_buffer("_branch_mask", branch_mask, persistent=False)
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.temporal_mixer = TappedPRLBranch(
            model_dim=model_dim,
            modes=modes,
            tap_kernel_size=self.prl_tap_kernel_size,
            tap_parameterization=tap_parameterization,
            low_rank_rank=low_rank_rank,
            dt=dt,
        )
        self.fir_branch = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=fir_kernel_size,
            padding=fir_kernel_size - 1,
        )
        hidden_dim = mlp_hidden_dim or model_dim * 2
        self.mlp_branch = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, model_dim),
        )
        match branch_normalization:
            case "none" | "rms":
                self.branch_layer_norms = nn.ModuleList()
            case "layernorm":
                self.branch_layer_norms = nn.ModuleList(
                    nn.LayerNorm(model_dim) for _ in BRANCH_ORDER
                )
            case _:
                raise _branch_normalization_error(branch_normalization)
        self.branch_gate: nn.Linear | None = None
        self.fixed_branch_logits: nn.Parameter | None = None
        match fusion_variant:
            case "no_gate_sum" | "fixed_equal":
                pass
            case "fixed_learned_scalar":
                self.fixed_branch_logits = nn.Parameter(torch.zeros(len(BRANCH_ORDER)))
            case "softmax" | "temperature_softmax":
                self.branch_gate = nn.Linear(model_dim, len(BRANCH_ORDER))
            case _:
                raise _fusion_variant_error(fusion_variant)
        self.activation = _activation(activation)
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.readout_projection = nn.Linear(model_dim, output_dim)

    def branch_weights(self, projected: Tensor) -> Tensor:
        mask = torch.reshape(self._branch_mask, (1, 1, len(BRANCH_ORDER)))
        active_branch_count = int(self._branch_mask.sum().item())
        match self.fusion_variant:
            case "no_gate_sum" | "fixed_equal":
                normalized = mask.to(dtype=projected.dtype) / float(active_branch_count)
                return normalized.expand(projected.shape[0], projected.shape[1], -1)
            case "fixed_learned_scalar":
                if self.fixed_branch_logits is None:
                    message = "fixed scalar branch logits are not initialized"
                    raise _state_error(message)
                logits = self.fixed_branch_logits.view(1, 1, -1).expand(
                    projected.shape[0],
                    projected.shape[1],
                    -1,
                )
            case "softmax":
                if self.branch_gate is None:
                    message = "softmax branch gate is not initialized"
                    raise _state_error(message)
                logits = self.branch_gate(projected)
            case "temperature_softmax":
                if self.branch_gate is None:
                    message = "temperature branch gate is not initialized"
                    raise _state_error(message)
                logits = self.branch_gate(projected) / self.fusion_temperature
            case _:
                raise _fusion_variant_error(self.fusion_variant)
        masked_logits = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
        return torch.softmax(masked_logits, dim=-1)

    def branch_outputs(self, projected: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        prl_output = self.temporal_mixer(projected)
        fir_input = projected.transpose(1, 2)
        fir_output = self.fir_branch(fir_input)[..., : projected.shape[1]].transpose(1, 2)
        mlp_output = self.mlp_branch(projected)
        return prl_output, fir_output, mlp_output

    def normalize_branch_outputs(
        self,
        branches: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        match self.branch_normalization:
            case "none":
                return branches
            case "rms":
                return (
                    _rms_normalize(branches[0]),
                    _rms_normalize(branches[1]),
                    _rms_normalize(branches[2]),
                )
            case "layernorm":
                if len(self.branch_layer_norms) != len(BRANCH_ORDER):
                    message = "branch layer norms are not initialized"
                    raise _state_error(message)
                return (
                    self.branch_layer_norms[0](branches[0]),
                    self.branch_layer_norms[1](branches[1]),
                    self.branch_layer_norms[2](branches[2]),
                )
            case _:
                raise _branch_normalization_error(self.branch_normalization)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.raw_input_dim,
            )
        if inputs.shape[-1] != self.raw_input_dim:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.raw_input_dim,
            )

        projected = self.input_projection(inputs)
        prl_output, fir_output, mlp_output = self.normalize_branch_outputs(
            self.branch_outputs(projected),
        )
        stacked = torch.stack((prl_output, fir_output, mlp_output), dim=2)
        match self.fusion_variant:
            case "no_gate_sum":
                branch_mask = self._branch_mask.view(1, 1, len(BRANCH_ORDER), 1).to(
                    dtype=stacked.dtype,
                )
                fused = torch.sum(branch_mask * stacked, dim=2)
            case "fixed_equal" | "fixed_learned_scalar" | "softmax" | "temperature_softmax":
                weights = self._dropout_adjusted_weights(self.branch_weights(projected))
                fused = torch.sum(weights * stacked, dim=2)
            case _:
                raise _fusion_variant_error(self.fusion_variant)
        residual = projected + self.output_projection(self.activation(fused))
        return self.readout_projection(residual)

    def _dropout_adjusted_weights(self, weights: Tensor) -> Tensor:
        if not self.training or self.branch_dropout_probability == 0.0:
            return weights.unsqueeze(-1)
        active = self._branch_mask.to(device=weights.device)
        random_values = torch.rand(len(BRANCH_ORDER), device=weights.device)
        keep = (random_values >= self.branch_dropout_probability) & active
        if not bool(keep.any()):
            keep = active
        return renormalized_branch_weights(weights, keep).unsqueeze(-1)


def _state_error(message: str) -> RuntimeError:
    return RuntimeError(message)


def _fusion_variant_error(name: str) -> RuntimeError:
    message = f"unsupported fusion variant: {name}"
    return RuntimeError(message)


def _branch_normalization_error(name: str) -> RuntimeError:
    message = f"unsupported branch normalization: {name}"
    return RuntimeError(message)


def _rms_normalize(branch: Tensor) -> Tensor:
    rms = torch.sqrt(branch.square().mean(dim=-1, keepdim=True) + 1.0e-8)
    return branch / rms


def renormalized_branch_weights(weights: Tensor, keep: Tensor) -> Tensor:
    keep_weights = keep.view(1, 1, len(BRANCH_ORDER)).to(device=weights.device, dtype=weights.dtype)
    masked = weights * keep_weights
    return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


class HybridModalPRLSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        modes: int,
        class_count: int,
        fir_kernel_size: int,
    ) -> None:
        super().__init__()
        self.encoder = HybridModalPRLBlock(
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            output_dim=model_dim,
            modes=modes,
            fir_kernel_size=fir_kernel_size,
        )
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs)
        pooled = torch.mean(encoded, dim=1)
        return self.classifier(pooled)
