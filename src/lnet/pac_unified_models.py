# ruff: noqa: EM101, TRY003
from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_stiefel_variants import REVISED_UNTIED_VARIANT
from .pac_tight_frame_models import (
    _BlockVariant,  # pyright: ignore[reportPrivateUsage]
    _CausalStem,  # pyright: ignore[reportPrivateUsage]
    _TightFrameBlock,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

PAC_UNIFIED_MODEL: Final[str] = "pac_stiefel_unified_coordtime_multiscale_slots_d64_m16"
PAC_UNIFIED_REGRESSOR: Final[str] = "pac_unified"
UnifiedObjective = Literal["classification", "regression"]
UNIFIED_VARIANT: Final = replace(
    REVISED_UNTIED_VARIANT,
    stem_kernel=1,
    stem_stride=1,
    use_local_convolution=False,
)


class BoundedCoordinateMixer(nn.Module):
    """Convex polynomial of a symmetric-normalized lattice operator."""

    def __init__(self, powers: int = 2) -> None:
        super().__init__()
        if powers < 0:
            raise ValueError("powers must be non-negative")
        initial = torch.linspace(1.5, -1.5, powers + 1)
        self.mixing_logits = nn.Parameter(initial)

    def forward(
        self,
        inputs: Tensor,
        lattice_shape: tuple[int, int] | None = None,
    ) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("coordinate mixer inputs must have shape [B,N,D]")
        shape = _checked_lattice_shape(inputs.shape[1], lattice_shape)
        powers = [inputs]
        current = inputs
        for _ in range(self.mixing_logits.numel() - 1):
            current = _normalized_lattice_step(current, shape)
            powers.append(current)
        weights = torch.softmax(self.mixing_logits, dim=0).to(dtype=inputs.dtype)
        return torch.stack(
            [weight * value for weight, value in zip(weights, powers, strict=True)]
        ).sum(dim=0)


class EvidenceSlotHead(nn.Module):
    """Fixed-K convex evidence summaries with a shared output projection."""

    def __init__(
        self,
        model_dim: int,
        modes: int,
        output_dim: int,
        *,
        slots: int = 4,
        lags: tuple[int, ...] = (1, 4),
        objective: UnifiedObjective = "classification",
    ) -> None:
        super().__init__()
        if slots < 1:
            raise ValueError("evidence slots must be positive")
        self.objective = objective
        self.norm = nn.RMSNorm(model_dim)
        self.queries = nn.Parameter(torch.empty(slots, model_dim))
        nn.init.orthogonal_(self.queries)
        moment_dim = 2 * modes * (1 + 2 * len(lags))
        self.classifier = nn.Linear(model_dim + moment_dim, output_dim)

    def forward(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
        observation_mask: Tensor | None = None,
    ) -> Tensor:
        normalized = self.norm(inputs)
        queries = functional.normalize(self.queries, dim=-1)
        scores = torch.einsum("bnd,kd->bkn", normalized, queries) / math.sqrt(
            normalized.shape[-1]
        )
        if observation_mask is None:
            weights = torch.softmax(scores, dim=-1)
        else:
            mask = observation_mask
            if mask.ndim == 3:
                mask = mask.squeeze(-1)
            mask = mask.to(device=inputs.device, dtype=torch.bool).unsqueeze(1)
            masked_scores = scores.masked_fill(~mask, -torch.inf)
            maxima = masked_scores.amax(dim=-1, keepdim=True)
            maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
            unnormalized = torch.exp(masked_scores - maxima) * mask.to(dtype=scores.dtype)
            weights = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        slots = torch.einsum("bkn,bnd->bkd", weights, normalized)
        moments = torch.cat((forward_moments, backward_moments), dim=-1)
        expanded_moments = moments.unsqueeze(1).expand(-1, slots.shape[1], -1)
        logits = self.classifier(torch.cat((slots, expanded_moments), dim=-1))
        if self.objective == "regression":
            return logits.mean(dim=1)
        return torch.logsumexp(logits, dim=1) - math.log(logits.shape[1])


class CoordinateTimeMultiscalePACClassifier(nn.Module):
    """Revised PAC with stable coordinate, scale, and evidence aggregation."""

    supports_observation_mask: Final[bool] = True

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        coordinate_shape: tuple[int, int] | None = None,
        objective: UnifiedObjective = "classification",
        scale_count: int = 3,
        evidence_slots: int = 4,
    ) -> None:
        super().__init__()
        if config.model_dim < 4:
            raise ValueError("model_dim must be at least 4")
        if scale_count < 1:
            raise ValueError("scale_count must be positive")
        self.model_dim = config.model_dim
        self.modes = max(1, min(config.modes, config.model_dim // 4))
        self.stem_stride = UNIFIED_VARIANT.stem_stride
        self.packed_spatial_channels: int | None = None
        effective_length = config.sequence_length
        stem_input_dim = config.raw_input_dim
        if coordinate_shape is not None and math.prod(coordinate_shape) != effective_length:
            height, width = coordinate_shape
            if height != effective_length or config.raw_input_dim % width != 0:
                raise ValueError("packed spatial input does not match its coordinate lattice")
            self.packed_spatial_channels = config.raw_input_dim // width
            effective_length = height * width
            stem_input_dim = self.packed_spatial_channels
        self.raw_coordinate_shape = _checked_lattice_shape(effective_length, coordinate_shape)
        self.coordinate_shape = _stem_lattice_shape(
            self.raw_coordinate_shape,
            self.stem_stride,
            (effective_length + self.stem_stride - 1) // self.stem_stride,
        )
        self.stem = _CausalStem(
            stem_input_dim,
            config.model_dim,
            kernel_size=UNIFIED_VARIANT.stem_kernel,
            stride=self.stem_stride,
        )
        self.geometry = BoundedCoordinateMixer(powers=2)
        self.forward_block = _TightFrameBlock(
            config.model_dim,
            self.modes,
            _BlockVariant("forward", UNIFIED_VARIANT),
        )
        self.backward_block = _TightFrameBlock(
            config.model_dim,
            self.modes,
            _BlockVariant("backward", UNIFIED_VARIANT),
        )
        self.scale_logits = nn.Parameter(torch.linspace(1.5, -1.0, scale_count))
        self.mixer_logit = nn.Parameter(torch.tensor(-2.0))
        self.head = EvidenceSlotHead(
            config.model_dim,
            self.modes,
            output_dim,
            slots=evidence_slots,
            lags=UNIFIED_VARIANT.moment_lags,
            objective=objective,
        )

    def forward(  # noqa: C901
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = inputs
        if self.packed_spatial_channels is not None:
            if self.raw_coordinate_shape is None:
                raise RuntimeError("packed spatial input is missing its coordinate shape")
            height, width = self.raw_coordinate_shape
            stem_inputs = inputs.reshape(
                inputs.shape[0],
                height,
                width,
                self.packed_spatial_channels,
            ).reshape(inputs.shape[0], height * width, self.packed_spatial_channels)
            if time_delta is not None:
                time_delta = time_delta.repeat_interleave(width, dim=1)
            if observation_mask is not None:
                observation_mask = observation_mask.repeat_interleave(width, dim=1)
            if valid_mask is not None:
                valid_mask = valid_mask.repeat_interleave(width, dim=1)
        raw_mask = observation_mask if observation_mask is not None else valid_mask
        if raw_mask is not None:
            if raw_mask.ndim == 2:
                raw_mask = raw_mask.unsqueeze(-1)
            stem_inputs = stem_inputs * raw_mask.to(
                device=stem_inputs.device,
                dtype=stem_inputs.dtype,
            )
        features = self.stem(stem_inputs)
        active_delta = _stem_reduce_metadata(time_delta, self.stem_stride, "sum")
        active_observation = _stem_reduce_metadata(observation_mask, self.stem_stride, "max")
        active_valid = _stem_reduce_metadata(valid_mask, self.stem_stride, "max")
        active_shape = self.coordinate_shape
        branch_inputs = features
        branch_logits: list[Tensor] = []
        for level in range(self.scale_logits.numel()):
            mixed = self.geometry(branch_inputs, active_shape)
            if active_valid is not None:
                mixed = mixed * active_valid.to(device=mixed.device, dtype=mixed.dtype)
            encoded, forward_moments = self.forward_block(
                mixed,
                time_delta=active_delta,
                observation_mask=active_observation,
                valid_mask=active_valid,
            )
            encoded, backward_moments = self.backward_block(
                encoded,
                time_delta=active_delta,
                observation_mask=active_observation,
                valid_mask=active_valid,
            )
            beta = torch.sigmoid(self.mixer_logit).to(dtype=encoded.dtype)
            encoded = (1.0 - beta) * encoded + beta * torch.tanh(encoded)
            branch_logits.append(
                self.head(encoded, forward_moments, backward_moments, active_valid)
            )
            if level + 1 == self.scale_logits.numel() or branch_inputs.shape[1] <= 1:
                break
            previous_shape = active_shape
            branch_inputs, active_shape = _pool_features(branch_inputs, active_shape)
            active_delta = _pool_metadata(active_delta, previous_shape, "sum")
            active_observation = _pool_metadata(active_observation, previous_shape, "max")
            active_valid = _pool_metadata(active_valid, previous_shape, "max")
        scale_weights = torch.softmax(self.scale_logits[: len(branch_logits)], dim=0)
        return torch.stack(
            [
                weight.to(dtype=branch.dtype) * branch
                for weight, branch in zip(scale_weights, branch_logits, strict=True)
            ]
        ).sum(dim=0)

    def first_frame_matrix(self) -> Tensor:
        return self.forward_block.frame_matrix()

    def post_optimizer_step(self) -> None:
        self.forward_block.retract_frame()
        self.backward_block.retract_frame()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        self.backward_block.finalize_frame()


class CoordinateTimePACSequenceRegressor(nn.Module):
    """Causal variable-step PAC used to isolate the temporal-coordinate claim."""

    def __init__(self, config: PACExperimentConfig) -> None:
        super().__init__()
        self.has_time_metadata = config.raw_input_dim >= 4
        content_dim = config.raw_input_dim - 2 if self.has_time_metadata else config.raw_input_dim
        self.input_projection = nn.Linear(content_dim, config.model_dim)
        self.block1 = _TightFrameBlock(
            config.model_dim,
            config.modes,
            _BlockVariant("forward", UNIFIED_VARIANT),
        )
        self.block2 = _TightFrameBlock(
            config.model_dim,
            config.modes,
            _BlockVariant("forward", UNIFIED_VARIANT),
        )
        self.final_norm = nn.RMSNorm(config.model_dim)
        self.output_projection = nn.Linear(config.model_dim, config.output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.has_time_metadata:
            time_delta = inputs[..., -2]
            observation_mask = inputs[..., -1]
            content = inputs[..., :-2]
        else:
            time_delta = None
            observation_mask = None
            content = inputs
        features = functional.silu(self.input_projection(content))
        features, _ = self.block1(
            features,
            time_delta=time_delta,
            observation_mask=observation_mask,
        )
        features, _ = self.block2(
            features,
            time_delta=time_delta,
            observation_mask=observation_mask,
        )
        return self.output_projection(self.final_norm(features))

    def post_optimizer_step(self) -> None:
        self.block1.retract_frame()
        self.block2.retract_frame()

    def finalize_constraints(self) -> None:
        self.block1.finalize_frame()
        self.block2.finalize_frame()


def build_unified_pac_classifier(
    name: str,
    config: PACExperimentConfig,
    output_dim: int,
    *,
    coordinate_shape: tuple[int, int] | None = None,
    objective: UnifiedObjective = "classification",
) -> nn.Module | None:
    if name != PAC_UNIFIED_MODEL:
        return None
    active = replace(config, model_dim=64, modes=16)
    return CoordinateTimeMultiscalePACClassifier(
        active,
        output_dim,
        coordinate_shape=coordinate_shape,
        objective=objective,
    )


def _checked_lattice_shape(
    length: int,
    lattice_shape: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if lattice_shape is None:
        return None
    height, width = lattice_shape
    if height < 1 or width < 1 or height * width != length:
        raise ValueError("coordinate lattice shape must exactly match sequence length")
    return height, width


def _stem_lattice_shape(
    shape: tuple[int, int] | None,
    stride: int,
    output_length: int,
) -> tuple[int, int] | None:
    if shape is None:
        return None
    height, width = shape
    if width % stride == 0 and height * (width // stride) == output_length:
        return height, width // stride
    return None


def _normalized_lattice_step(
    inputs: Tensor,
    shape: tuple[int, int] | None,
) -> Tensor:
    if shape is None or min(shape) == 1:
        return _normalized_chain_step(inputs)
    height, width = shape
    grid = inputs.reshape(inputs.shape[0], height, width, inputs.shape[-1])
    degree = torch.ones(height, width, device=inputs.device, dtype=inputs.dtype)
    degree[1:] += 1
    degree[:-1] += 1
    degree[:, 1:] += 1
    degree[:, :-1] += 1
    scaled = grid / torch.sqrt(degree).view(1, height, width, 1)
    summed = scaled.clone()
    summed[:, 1:] += scaled[:, :-1]
    summed[:, :-1] += scaled[:, 1:]
    summed[:, :, 1:] += scaled[:, :, :-1]
    summed[:, :, :-1] += scaled[:, :, 1:]
    output = summed / torch.sqrt(degree).view(1, height, width, 1)
    return output.reshape_as(inputs)


def _normalized_chain_step(inputs: Tensor) -> Tensor:
    length = inputs.shape[1]
    if length == 1:
        return inputs
    degree = torch.full((length,), 3.0, device=inputs.device, dtype=inputs.dtype)
    degree[0] = 2.0
    degree[-1] = 2.0
    scaled = inputs / torch.sqrt(degree).view(1, length, 1)
    summed = scaled.clone()
    summed[:, 1:] += scaled[:, :-1]
    summed[:, :-1] += scaled[:, 1:]
    return summed / torch.sqrt(degree).view(1, length, 1)


def _pool_features(
    inputs: Tensor,
    shape: tuple[int, int] | None,
) -> tuple[Tensor, tuple[int, int] | None]:
    if shape is not None and min(shape) > 1:
        height, width = shape
        grid = inputs.reshape(inputs.shape[0], height, width, inputs.shape[-1]).permute(0, 3, 1, 2)
        pooled = functional.avg_pool2d(grid, kernel_size=2, stride=2, ceil_mode=True)
        next_shape = pooled.shape[-2], pooled.shape[-1]
        return pooled.permute(0, 2, 3, 1).reshape(inputs.shape[0], -1, inputs.shape[-1]), next_shape
    pooled = functional.avg_pool1d(
        inputs.transpose(1, 2),
        kernel_size=2,
        stride=2,
        ceil_mode=True,
    ).transpose(1, 2)
    return pooled, None


def _stem_reduce_metadata(
    values: Tensor | None,
    stride: int,
    reduction: Literal["sum", "max"],
) -> Tensor | None:
    if values is None:
        return None
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    if values.ndim != 3 or values.shape[-1] != 1:
        raise ValueError("time metadata must have shape [B,N] or [B,N,1]")
    if values.dtype == torch.bool:
        values = values.to(dtype=torch.float32)
    return _pool_1d_metadata(values, stride, reduction)


def _pool_metadata(
    values: Tensor | None,
    shape: tuple[int, int] | None,
    reduction: Literal["sum", "max"],
) -> Tensor | None:
    if values is None:
        return None
    if shape is not None and min(shape) > 1:
        height, width = shape
        grid = values.reshape(values.shape[0], height, width, 1).permute(0, 3, 1, 2)
        if reduction == "sum":
            pooled = functional.avg_pool2d(
                grid,
                kernel_size=2,
                stride=2,
                ceil_mode=True,
                count_include_pad=False,
            ) * 4.0
        else:
            pooled = functional.max_pool2d(grid, kernel_size=2, stride=2, ceil_mode=True)
        return pooled.permute(0, 2, 3, 1).reshape(values.shape[0], -1, 1)
    return _pool_1d_metadata(values, 2, reduction)


def _pool_1d_metadata(
    values: Tensor,
    factor: int,
    reduction: Literal["sum", "max"],
) -> Tensor:
    length = values.shape[1]
    padding = (-length) % factor
    padded = functional.pad(values.transpose(1, 2), (0, padding))
    if reduction == "sum":
        return functional.avg_pool1d(padded, factor, factor).mul(factor).transpose(1, 2)
    return functional.max_pool1d(padded, factor, factor).transpose(1, 2)
