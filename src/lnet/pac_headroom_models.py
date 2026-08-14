# ruff: noqa: C901, EM101, TRY003
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn

from .pac_stiefel_variants import REVISED_UNTIED_VARIANT, StiefelVariant
from .pac_tight_frame_models import (
    _BlockVariant,  # pyright: ignore[reportPrivateUsage]
    _CausalStem,  # pyright: ignore[reportPrivateUsage]
    _dtype_aligned_rms_norm,  # pyright: ignore[reportPrivateUsage]
    _InvariantMomentHead,  # pyright: ignore[reportPrivateUsage]
    _masked_sequence_mean,  # pyright: ignore[reportPrivateUsage]
    _TightFrameBlock,  # pyright: ignore[reportPrivateUsage]
)
from .pac_unified_models import (
    UNIFIED_VARIANT,
    BoundedCoordinateMixer,
    EvidenceSlotHead,
    _checked_lattice_shape,  # pyright: ignore[reportPrivateUsage]
    _pool_features,  # pyright: ignore[reportPrivateUsage]
    _pool_metadata,  # pyright: ignore[reportPrivateUsage]
    _stem_lattice_shape,  # pyright: ignore[reportPrivateUsage]
    _stem_reduce_metadata,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

HeadroomObjective = Literal["classification", "regression"]


@dataclass(frozen=True, slots=True)
class HeadroomModuleSpec:
    name: str
    geometry_compatible_stem: bool = False
    geometry: bool = False
    multiscale: bool = False
    scale_count: int = 1
    slots: bool = False


HEADROOM_SPECS: Final[dict[str, HeadroomModuleSpec]] = {
    "B": HeadroomModuleSpec("B"),
    "C": HeadroomModuleSpec("C", geometry_compatible_stem=True),
    "G": HeadroomModuleSpec("G", geometry=True),
    "M": HeadroomModuleSpec("M", multiscale=True, scale_count=3),
    "M2": HeadroomModuleSpec("M2", multiscale=True, scale_count=2),
    "S": HeadroomModuleSpec("S", slots=True),
    "GM": HeadroomModuleSpec(
        "GM",
        geometry=True,
        multiscale=True,
        scale_count=3,
    ),
    "MS": HeadroomModuleSpec("MS", multiscale=True, scale_count=3, slots=True),
    "GS": HeadroomModuleSpec(
        "GS",
        geometry=True,
        slots=True,
    ),
    "GMS": HeadroomModuleSpec(
        "GMS",
        geometry=True,
        multiscale=True,
        scale_count=3,
        slots=True,
    ),
}


class HeadroomPACClassifier(nn.Module):
    """Development-only modular PAC used to decide the canonical architecture."""

    supports_observation_mask: Final[bool] = True
    supports_time_delta: Final[bool] = True

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        spec: HeadroomModuleSpec,
        *,
        coordinate_shape: tuple[int, int] | None = None,
        objective: HeadroomObjective = "classification",
        pac_variant: StiefelVariant | None = None,
        mode_divisor: int = 4,
    ) -> None:
        super().__init__()
        if mode_divisor < 2:
            message = "mode_divisor must be at least two for a 2M-column modal frame"
            raise ValueError(message)
        self.spec = spec
        self.model_dim = config.model_dim
        self.modes = max(1, min(config.modes, config.model_dim // mode_divisor))
        variant = pac_variant or (
            UNIFIED_VARIANT if spec.geometry_compatible_stem else REVISED_UNTIED_VARIANT
        )
        self.stem_stride = variant.stem_stride
        self.raw_coordinate_shape = _checked_lattice_shape(config.sequence_length, coordinate_shape)
        self.coordinate_shape = _stem_lattice_shape(
            self.raw_coordinate_shape,
            self.stem_stride,
            (config.sequence_length + self.stem_stride - 1) // self.stem_stride,
        )
        self.stem: nn.Module = _CausalStem(
            config.raw_input_dim,
            config.model_dim,
            kernel_size=variant.stem_kernel,
            stride=variant.stem_stride,
        )
        self.forward_block = _TightFrameBlock(
            config.model_dim,
            self.modes,
            _BlockVariant("forward", variant),
        )
        self.backward_block = _TightFrameBlock(
            config.model_dim,
            self.modes,
            _BlockVariant("backward", variant),
        )
        self.geometry = BoundedCoordinateMixer(powers=2) if spec.geometry else None
        if spec.multiscale:
            initial_scale_logits = torch.linspace(1.5, -1.0, 3)[: spec.scale_count]
            self.scale_logits = nn.Parameter(initial_scale_logits)
            self.mixer_logit = nn.Parameter(torch.tensor(-2.0))
        else:
            self.register_parameter("scale_logits", None)
            self.register_parameter("mixer_logit", None)
        if spec.slots:
            self.final_norm = None
            self.head: EvidenceSlotHead | _InvariantMomentHead = EvidenceSlotHead(
                config.model_dim,
                self.modes,
                output_dim,
                slots=4,
                lags=variant.moment_lags,
                objective=objective,
            )
        else:
            self.final_norm = nn.RMSNorm(config.model_dim)
            self.head = _InvariantMomentHead(
                config.model_dim,
                self.modes,
                output_dim,
                use_modal_moments=True,
                use_backward_moments=True,
                lags=variant.moment_lags,
            )
        self.use_fused_rmsnorm_mean_training = False
        self.use_fused_rmsnorm_mean_backward_training = False
        self.use_d32_rmsnorm_backward_training = False
        self.use_fused_efp16_inference_readout = False
        self.use_fused_efp16_inference_tail = False
        self.efp16_inference_tail_block_time = 32
        self.efp16_inference_tail_num_warps = 4

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        raw_mask = observation_mask if observation_mask is not None else valid_mask
        stem_inputs = inputs
        if raw_mask is not None:
            if raw_mask.ndim == 2:
                raw_mask = raw_mask.unsqueeze(-1)
            stem_inputs = stem_inputs * raw_mask.to(device=inputs.device, dtype=inputs.dtype)
        if self.geometry is not None:
            stem_inputs = self.geometry(stem_inputs, self.raw_coordinate_shape)
        branch_inputs = self.stem(stem_inputs)
        active_delta = _stem_reduce_metadata(time_delta, self.stem_stride, "sum")
        active_observation = _stem_reduce_metadata(observation_mask, self.stem_stride, "max")
        active_valid = _stem_reduce_metadata(valid_mask, self.stem_stride, "max")
        active_shape = self.coordinate_shape
        branch_logits: list[Tensor] = []
        level_count = self.spec.scale_count if self.spec.multiscale else 1
        for level in range(level_count):
            mixed = branch_inputs
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
            if self.spec.multiscale:
                if self.mixer_logit is None:
                    raise RuntimeError("multiscale mixer parameter is missing")
                beta = torch.sigmoid(self.mixer_logit).to(dtype=encoded.dtype)
                encoded = (1.0 - beta) * encoded + beta * torch.tanh(encoded)
            branch_logits.append(
                self._readout(encoded, forward_moments, backward_moments, active_valid)
            )
            if level + 1 == level_count or encoded.shape[1] <= 1:
                break
            previous_shape = active_shape
            branch_inputs, active_shape = _pool_features(encoded, active_shape)
            active_delta = _pool_metadata(active_delta, previous_shape, "sum")
            active_observation = _pool_metadata(active_observation, previous_shape, "max")
            active_valid = _pool_metadata(active_valid, previous_shape, "max")
        if len(branch_logits) == 1:
            return branch_logits[0]
        if self.scale_logits is None:
            raise RuntimeError("multiscale fusion parameter is missing")
        weights = torch.softmax(self.scale_logits[: len(branch_logits)], dim=0)
        return torch.stack(
            [
                weight.to(dtype=logits.dtype) * logits
                for weight, logits in zip(weights, branch_logits, strict=True)
            ]
        ).sum(dim=0)

    def _readout(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if isinstance(self.head, EvidenceSlotHead):
            return self.head(inputs, forward_moments, backward_moments, valid_mask)
        if self.final_norm is None:
            raise RuntimeError("mean readout normalization is missing")
        norm_weight = self.final_norm.weight
        classifier = self.head.classifier
        classifier_bias = classifier.bias
        can_fuse_efp16_inference = (
            self.use_fused_efp16_inference_readout
            and not self.training
            and not torch.is_grad_enabled()
            and valid_mask is None
            and inputs.is_cuda
            and inputs.dtype == torch.float32
            and inputs.is_contiguous()
            and forward_moments.is_cuda
            and forward_moments.dtype == torch.float32
            and forward_moments.is_contiguous()
            and backward_moments.is_cuda
            and backward_moments.dtype == torch.float32
            and backward_moments.is_contiguous()
            and norm_weight is not None
            and norm_weight.dtype == torch.float32
            and norm_weight.is_contiguous()
            and classifier_bias is not None
            and classifier.weight.dtype == torch.float32
            and classifier.weight.is_contiguous()
            and classifier_bias.dtype == torch.float32
            and classifier_bias.is_contiguous()
            and self.model_dim == 32
            and self.modes == 16
            and self.spec.name == "B"
            and self.head.use_modal_moments
            and self.head.use_backward_moments
            and inputs.shape[-1] == 32
            and forward_moments.shape[-1] == 80
            and backward_moments.shape[-1] == 80
            and classifier.in_features == 192
            and classifier.out_features == 5
        )
        if can_fuse_efp16_inference:
            from .pac_triton_efp16_readout import (  # noqa: PLC0415
                fused_efp16_readout_inference,
            )

            return fused_efp16_readout_inference(
                inputs,
                norm_weight,
                forward_moments,
                backward_moments,
                classifier.weight,
                classifier_bias,
                eps=self.final_norm.eps,
            )
        can_fuse_readout = (
            self.training
            and valid_mask is None
            and inputs.is_cuda
            and inputs.dtype == torch.float32
            and inputs.is_contiguous()
            and self.final_norm.weight is not None
        )
        if self.use_fused_rmsnorm_mean_training and can_fuse_readout:
            from .pac_triton_rmsnorm_mean_training import (  # noqa: PLC0415
                fused_rmsnorm_mean_training,
            )

            pooled = fused_rmsnorm_mean_training(
                inputs,
                self.final_norm.weight,
                eps=self.final_norm.eps,
            )
        elif self.use_fused_rmsnorm_mean_backward_training and can_fuse_readout:
            from .pac_triton_rmsnorm_mean_training import (  # noqa: PLC0415
                fused_rmsnorm_mean_backward_training,
            )

            pooled = fused_rmsnorm_mean_backward_training(
                inputs,
                self.final_norm.weight,
                eps=self.final_norm.eps,
            )
        elif (
            self.use_d32_rmsnorm_backward_training
            and self.training
            and torch.is_grad_enabled()
            and norm_weight is not None
        ):
            from .pac_triton_d32_rmsnorm_backward_training import (  # noqa: PLC0415
                d32_rmsnorm_backward_training,
            )

            normalized = d32_rmsnorm_backward_training(
                inputs,
                norm_weight,
                eps=self.final_norm.eps,
            )
            pooled = _masked_sequence_mean(normalized, valid_mask)
        else:
            normalized = _dtype_aligned_rms_norm(inputs, self.final_norm)
            pooled = _masked_sequence_mean(normalized, valid_mask)
        return self.head(pooled, forward_moments, backward_moments)

    def post_optimizer_step(self) -> None:
        self.forward_block.retract_frame()
        self.backward_block.retract_frame()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        self.backward_block.finalize_frame()


def build_headroom_pac_classifier(
    spec_name: str,
    config: PACExperimentConfig,
    output_dim: int,
    *,
    coordinate_shape: tuple[int, int] | None = None,
    objective: HeadroomObjective = "classification",
) -> HeadroomPACClassifier:
    try:
        spec = HEADROOM_SPECS[spec_name]
    except KeyError as error:
        message = f"unknown PAC headroom spec: {spec_name}"
        raise ValueError(message) from error
    return HeadroomPACClassifier(
        config,
        output_dim,
        spec,
        coordinate_shape=coordinate_shape,
        objective=objective,
    )
