# ruff: noqa: EM101, EM102, FBT001, TRY003
from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Final, Literal, cast

import torch
from torch import Tensor
from torch.nn import functional

from .pac_headroom_models import HEADROOM_SPECS, HeadroomObjective, HeadroomPACClassifier
from .pac_stiefel_variants import REVISED_UNTIED_VARIANT
from .pac_tight_frame_models import (
    Direction,
    _BlockVariant,  # pyright: ignore[reportPrivateUsage]
    _CausalStem,  # pyright: ignore[reportPrivateUsage]
    _masked_modal_moments,  # pyright: ignore[reportPrivateUsage]
    _modal_moments,  # pyright: ignore[reportPrivateUsage]
    _MomentVariant,  # pyright: ignore[reportPrivateUsage]
    _TightFrameBlock,  # pyright: ignore[reportPrivateUsage]
)
from .pac_triton_edge_frame_stem import edge_frame_stem_inference
from .pac_triton_edge_frame_stem_training import (
    ParameterGradientStrategy,
    edge_frame_stem_training,
)
from .pac_triton_pa2wp_stem import pa2wp_stem_inference
from .pac_triton_pa2wp_training_stem import pa2wp_training_stem
from .pac_unified_models import (
    UNIFIED_VARIANT,
    _checked_lattice_shape,  # pyright: ignore[reportPrivateUsage]
    _normalized_lattice_step,  # pyright: ignore[reportPrivateUsage]
    _pool_features,  # pyright: ignore[reportPrivateUsage]
    _pool_metadata,  # pyright: ignore[reportPrivateUsage]
    _stem_reduce_metadata,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from .pac_stiefel_variants import StiefelVariant
    from .pac_types import PACExperimentConfig

EfficientHeadroomSpec = Literal[
    "MP",
    "WP",
    "PCWP",
    "DPWP",
    "PAWP",
    "PA2WP",
    "LPWP",
    "OA",
    "AD",
    "SMR",
    "SMRFF",
    "SMRFBFB",
    "UMD",
    "EFP8",
    "EFP16",
    "EFU8",
    "C2M8",
]
EFFICIENT_HEADROOM_SPECS: Final[tuple[EfficientHeadroomSpec, ...]] = (
    "MP",
    "WP",
    "PCWP",
    "DPWP",
    "PAWP",
    "PA2WP",
    "LPWP",
    "OA",
    "AD",
    "SMR",
    "SMRFF",
    "SMRFBFB",
    "UMD",
    "EFP8",
    "EFP16",
    "EFU8",
    "C2M8",
)
FINAL_PAC_MODEL: Final = "pac_final_time_weighted_multiscale_d64_m16"
PHASE_COMPLETE_WP_PAC_MODEL: Final = "pac_headroom_phase_complete_wp_d64_m16"
DUAL_PHASE_WP_PAC_MODEL: Final = "pac_headroom_dual_phase_wp_d64_m16"
PHASE_AUGMENTED_WP_PAC_MODEL: Final = "pac_headroom_phase_augmented_wp_d64_m16"
PHASE_AUGMENTED_ENSEMBLE_WP_PAC_MODEL: Final = "pac_headroom_phase_augmented_ensemble_wp_d64_m16"
# Generic eager training keeps its screened ceiling. Captured runtimes may opt
# the largest paper cell into the deterministic split-K lane explicitly.
_PA2WP_TRAINING_STEM_MAX_BATCH_STEPS: Final = 65_536
_PA2WP_TRAINING_STEM_LARGE_MAX_BATCH_STEPS: Final = 131_072
LEARNED_PAIR_WP_PAC_MODEL: Final = "pac_headroom_learned_pair_wp_d64_m16"
OVERLAPPING_ANTIALIASED_PAC_MODEL: Final = "pac_headroom_overlapping_antialiased_d64_m16"
SPARSE_MULTISCALE_PAC_MODEL: Final = "pac_headroom_sparse_multiscale_residual_d64_m16"
SPARSE_MULTISCALE_FF_PAC_MODEL: Final = "pac_headroom_sparse_multiscale_ff_d64_m16"
SPARSE_MULTISCALE_FBFB_PAC_MODEL: Final = "pac_headroom_sparse_multiscale_fbfb_d64_m16"
UNDECIMATED_MODAL_DYADIC_PAC_MODEL: Final = "pac_headroom_undecimated_modal_dyadic_d64_m16"
UMD_VARIANT: Final = replace(REVISED_UNTIED_VARIANT, stem_stride=1)
EDGE_FRAME_VARIANT: Final = replace(REVISED_UNTIED_VARIANT, stem_stride=1)


class OverlappingAntiAliasedStem(torch.nn.Module):
    """Contractive joint low/detail projection followed by fixed blur-downsampling."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        if input_dim < 1 or model_dim < 1:
            raise ValueError("stem dimensions must be positive")
        self.projection = torch.nn.Linear(2 * input_dim, model_dim, bias=False)
        torch.nn.init.orthogonal_(self.projection.weight)
        self.register_buffer(
            "blur_kernel",
            torch.tensor([0.25, 0.5, 0.25]).reshape(1, 1, 3),
        )
        self.project_weight_()

    def forward(self, edge_features: Tensor) -> Tensor:
        if edge_features.ndim != 3 or edge_features.shape[1] < 1:
            raise ValueError("edge features must have shape [B,N-1,2*D]")
        projected = self.projection(edge_features)
        channels = projected.shape[-1]
        kernel = self.blur_kernel.to(device=projected.device, dtype=projected.dtype).expand(
            channels, 1, -1
        )
        filtered = functional.conv1d(
            projected.transpose(1, 2),
            kernel,
            stride=2,
            padding=1,
            groups=channels,
        )
        return functional.silu(filtered.transpose(1, 2))

    @torch.no_grad()
    def project_weight_(self) -> None:
        weight = self.projection.weight
        spectral_norm = torch.linalg.matrix_norm(weight.float(), ord=2)
        weight.div_(spectral_norm.clamp_min(1.0).to(dtype=weight.dtype))


class OverlappingAntiAliasedPAC(HeadroomPACClassifier):
    """All-edge low/detail analysis with one anti-aliased shared F->B PAC core."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
    ) -> None:
        super().__init__(config, output_dim, HEADROOM_SPECS["B"], objective=objective)
        self.stem = OverlappingAntiAliasedStem(config.raw_input_dim, config.model_dim)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        low, detail = _overlapping_low_detail(stem_inputs)
        edge_observation = _edge_joint_mask(observation_mask)
        edge_valid = _edge_joint_mask(valid_mask)
        edge_mask = _combined_edge_mask(edge_observation, edge_valid)
        edge_features = torch.cat((low, detail), dim=-1)
        if edge_mask is not None:
            edge_features = edge_features * edge_mask.to(
                device=edge_features.device, dtype=edge_features.dtype
            )
        encoded = self.stem(edge_features)
        active_delta = _stem_reduce_metadata(_edge_delta(time_delta), 2, "sum")
        active_observation = _overlap_support_mask(edge_observation)
        active_valid = _overlap_support_mask(edge_valid)
        if active_valid is not None:
            encoded = encoded * active_valid.to(device=encoded.device, dtype=encoded.dtype)
        encoded, forward_moments = self.forward_block(
            encoded,
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
        return self._readout(encoded, forward_moments, backward_moments, active_valid)

    def post_optimizer_step(self) -> None:
        super().post_optimizer_step()
        self.stem.project_weight_()

    def finalize_constraints(self) -> None:
        super().finalize_constraints()
        self.stem.project_weight_()


class EdgeFrameStem(torch.nn.Module):
    """Full-rate joint edge projection followed by a dilated depthwise local map."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        *,
        semi_orthogonal: bool,
    ) -> None:
        super().__init__()
        edge_dim = 2 * input_dim
        if input_dim < 1 or model_dim < 1:
            raise ValueError("stem dimensions must be positive")
        self.semi_orthogonal = semi_orthogonal
        self.fused_raw_inference = False
        self.projection = torch.nn.Linear(edge_dim, model_dim, bias=False)
        torch.nn.init.orthogonal_(self.projection.weight)
        self.local = torch.nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=5,
            dilation=4,
            padding=8,
            groups=model_dim,
        )
        self.project_weight_()

    def forward(self, edge_features: Tensor) -> Tensor:
        if edge_features.ndim != 3 or edge_features.shape[1] < 1:
            raise ValueError("edge features must have shape [B,E>=1,2*C]")
        projected = self.projection(edge_features)
        local = self.local(projected.transpose(1, 2)).transpose(1, 2)
        return functional.silu(local)

    def forward_raw_inference(self, raw_inputs: Tensor) -> Tensor:
        if self.local.bias is None:
            raise RuntimeError("edge-frame local bias is required for fused inference")
        return edge_frame_stem_inference(
            raw_inputs,
            self.projection.weight,
            self.local.weight,
            self.local.bias,
        )

    def prepare_fused_raw_inference_(self) -> None:
        if self.projection.in_features != 2:
            raise RuntimeError("fused edge-frame inference requires scalar raw inputs")
        if self.local.kernel_size != (5,) or self.local.dilation != (4,):
            raise RuntimeError("fused edge-frame inference requires the canonical local map")
        self.fused_raw_inference = True

    @torch.no_grad()
    def project_weight_(self) -> None:
        if not self.semi_orthogonal:
            return
        weight = self.projection.weight
        active = weight.float() if weight.shape[0] >= weight.shape[1] else weight.float().T
        frame, upper = torch.linalg.qr(active, mode="reduced")
        diagonal = torch.diagonal(upper)
        signs = torch.where(diagonal >= 0.0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
        projected = frame * signs.unsqueeze(0)
        if weight.shape[0] < weight.shape[1]:
            projected = projected.T
        weight.copy_(projected.to(dtype=weight.dtype))


class StrideOneConvStem(torch.nn.Module):
    """Parameter-matched unconstrained two-tap convolution control."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.projection = torch.nn.Conv1d(input_dim, model_dim, kernel_size=2, bias=False)
        self.local = torch.nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=5,
            dilation=4,
            padding=8,
            groups=model_dim,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1] < 1:
            raise ValueError("convolution inputs must have shape [B,N>=1,C]")
        active = inputs
        if active.shape[1] == 1:
            active = torch.cat((active, torch.zeros_like(active)), dim=1)
        projected = self.projection(active.transpose(1, 2)).transpose(1, 2)
        local = self.local(projected.transpose(1, 2)).transpose(1, 2)
        return functional.silu(local)


class EdgeFramePAC(HeadroomPACClassifier):
    """Degree-normalized undecimated edge frame with one full-rate PAC core."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        modes: int,
        semi_orthogonal: bool,
        objective: HeadroomObjective,
        model_dim: int = 32,
        pac_variant: StiefelVariant = EDGE_FRAME_VARIANT,
        mode_divisor: int | None = None,
    ) -> None:
        active_config = replace(config, model_dim=model_dim, modes=modes)
        super().__init__(
            active_config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective=objective,
            pac_variant=pac_variant,
            mode_divisor=(2 if modes > 8 else 4) if mode_divisor is None else mode_divisor,
        )
        self.stem = EdgeFrameStem(
            active_config.raw_input_dim,
            active_config.model_dim,
            semi_orthogonal=semi_orthogonal,
        )
        self.use_fused_efp16_stem_training = False
        self.efp16_stem_parameter_gradient_strategy: ParameterGradientStrategy = "auto"

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        active_stem = self.stem
        if (
            self.use_fused_efp16_stem_training
            and self.training
            and time_delta is None
            and observation_mask is None
            and valid_mask is None
            and inputs.shape[-1] == 1
            and inputs.shape[1] >= 2
        ):
            if active_stem.local.bias is None:
                message = "fused EFP16 training stem requires the canonical local bias"
                raise RuntimeError(message)
            encoded = edge_frame_stem_training(
                inputs,
                active_stem.projection.weight,
                active_stem.local.weight,
                active_stem.local.bias,
                parameter_gradient_strategy=self.efp16_stem_parameter_gradient_strategy,
            )
            active_delta = None
            active_observation = None
            active_valid = None
        elif (
            isinstance(active_stem, EdgeFrameStem)
            and active_stem.fused_raw_inference
            and not torch.is_grad_enabled()
            and time_delta is None
            and observation_mask is None
            and valid_mask is None
            and inputs.shape[-1] == 1
            and inputs.shape[1] >= 2
        ):
            encoded = active_stem.forward_raw_inference(inputs)
            active_delta = None
            active_observation = None
            active_valid = None
        else:
            stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
            low, detail, active_delta = _degree_normalized_edge_analysis(
                stem_inputs,
                time_delta,
            )
            active_observation = _edge_or_singleton_mask(observation_mask)
            active_valid = _edge_or_singleton_mask(valid_mask)
            edge_mask = _combined_edge_mask(active_observation, active_valid)
            edge_features = torch.cat((low, detail), dim=-1)
            if edge_mask is not None:
                edge_features = edge_features * edge_mask.to(
                    device=edge_features.device,
                    dtype=edge_features.dtype,
                )
            encoded = self.stem(edge_features)
            if active_valid is not None:
                encoded = encoded * active_valid.to(
                    device=encoded.device,
                    dtype=encoded.dtype,
                )
        encoded, forward_moments = self.forward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        final_norm = self.final_norm
        head = self.head
        classifier = getattr(head, "classifier", None)
        synthesis_frame = self.backward_block.independent_synthesis_frame
        direct_scale = self.backward_block.direct_scale
        layer_scale = self.backward_block.layer_scale
        can_fuse_inference_tail = (
            self.use_fused_efp16_inference_tail
            and not self.training
            and not torch.is_grad_enabled()
            and active_delta is None
            and active_observation is None
            and active_valid is None
            and encoded.is_cuda
            and encoded.dtype == torch.float32
            and encoded.is_contiguous()
            and forward_moments.is_cuda
            and forward_moments.dtype == torch.float32
            and forward_moments.is_contiguous()
            and self.model_dim == 32
            and self.modes == 16
            and final_norm is not None
            and final_norm.weight is not None
            and final_norm.weight.dtype == torch.float32
            and final_norm.weight.is_contiguous()
            and synthesis_frame is not None
            and synthesis_frame.dtype == torch.float32
            and synthesis_frame.is_contiguous()
            and direct_scale is not None
            and direct_scale.dtype == torch.float32
            and direct_scale.is_contiguous()
            and layer_scale is not None
            and layer_scale.dtype == torch.float32
            and layer_scale.is_contiguous()
            and classifier is not None
            and classifier.weight.dtype == torch.float32
            and classifier.weight.is_contiguous()
            and classifier.bias is not None
            and classifier.bias.dtype == torch.float32
            and classifier.bias.is_contiguous()
            and classifier.in_features == 192
            and classifier.out_features == 5
            and bool(getattr(head, "use_modal_moments", False))
            and bool(getattr(head, "use_backward_moments", False))
            and self.backward_block.synthesis_scale == 1.0
            and not self.backward_block.split_residual_scales
            and self.backward_block.canonical_identity_elision
        )
        if can_fuse_inference_tail:
            if (
                final_norm is None
                or final_norm.weight is None
                or synthesis_frame is None
                or direct_scale is None
                or layer_scale is None
                or classifier is None
                or classifier.bias is None
            ):
                message = "canonical EFP16 tail tensors are unavailable"
                raise RuntimeError(message)
            packed_modal_coordinates, backward_local, backward_moments = self.backward_block(
                encoded,
                return_inference_tail_components=True,
            )
            from .pac_triton_efp16_tail import (  # noqa: PLC0415
                fused_efp16_tail_inference,
            )

            return fused_efp16_tail_inference(
                encoded,
                backward_local,
                packed_modal_coordinates,
                synthesis_frame,
                direct_scale,
                layer_scale,
                final_norm.weight,
                forward_moments,
                backward_moments,
                classifier.weight,
                classifier.bias,
                eps=final_norm.eps,
                block_time=self.efp16_inference_tail_block_time,
                num_warps=self.efp16_inference_tail_num_warps,
            )
        encoded, backward_moments = self.backward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        return self._readout(encoded, forward_moments, backward_moments, active_valid)

    def post_optimizer_step(self) -> None:
        super().post_optimizer_step()
        self.stem.project_weight_()

    def finalize_constraints(self) -> None:
        super().finalize_constraints()
        self.stem.project_weight_()


class StrideOneConvPAC(HeadroomPACClassifier):
    """Raw two-tap convolution control with the same D=32, M=8 full-rate core."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
    ) -> None:
        active_config = replace(config, model_dim=32, modes=8)
        super().__init__(
            active_config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective=objective,
            pac_variant=EDGE_FRAME_VARIANT,
        )
        self.stem = StrideOneConvStem(active_config.raw_input_dim, active_config.model_dim)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        encoded = self.stem(stem_inputs)
        active_delta = _edge_or_singleton_delta(time_delta)
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        if active_valid is not None:
            encoded = encoded * active_valid.to(device=encoded.device, dtype=encoded.dtype)
        encoded, forward_moments = self.forward_block(
            encoded,
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
        return self._readout(encoded, forward_moments, backward_moments, active_valid)


class UndecimatedModalDyadicMixer(torch.nn.Module):
    """Full-length telescoping dyadic partition of complex modal excitations."""

    def __init__(
        self,
        modes: int,
        *,
        lattice_shape: tuple[int, int] | None = None,
        powers: tuple[int, ...] = (1, 2, 4, 8, 16),
    ) -> None:
        super().__init__()
        if modes < 1:
            raise ValueError("modes must be positive")
        if not powers or tuple(sorted(set(powers))) != powers or powers[0] < 1:
            raise ValueError("dyadic powers must be sorted, unique, and positive")
        self.powers = powers
        self.lattice_shape = lattice_shape
        self.band_weights = torch.nn.Parameter(torch.ones(len(powers) + 1, modes))

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        if real.shape != imag.shape or real.ndim != 3:
            raise ValueError("modal real and imaginary tensors must share shape [B,N,M]")
        shape = _checked_lattice_shape(real.shape[1], self.lattice_shape)
        excitation = torch.cat((real, imag), dim=-1)
        previous = excitation
        current = excitation
        bands: list[Tensor] = []
        power_index = 0
        for power in range(1, self.powers[-1] + 1):
            graph_step = _normalized_lattice_step(current, shape)
            current = 0.5 * (current + graph_step)
            if power != self.powers[power_index]:
                continue
            bands.append(previous - current)
            previous = current
            power_index += 1
            if power_index == len(self.powers):
                break
        bands.append(current)
        weights = self.band_weights.clamp(0.0, 1.0).to(dtype=real.dtype)
        modal_weights = torch.cat((weights, weights), dim=-1)
        mixed = (torch.stack(bands, dim=0) * modal_weights[:, None, None, :]).sum(dim=0)
        return mixed.chunk(2, dim=-1)

    @torch.no_grad()
    def project_weights_(self) -> None:
        self.band_weights.clamp_(0.0, 1.0)


class UndecimatedModalDyadicPAC(HeadroomPACClassifier):
    """Pair-free PAC with a modal dyadic partition inside each directional block."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        coordinate_shape: tuple[int, int] | None = None,
        objective: HeadroomObjective,
    ) -> None:
        self.packed_spatial_channels: int | None = None
        active_config = config
        if coordinate_shape is not None and math.prod(coordinate_shape) != config.sequence_length:
            height, width = coordinate_shape
            if height != config.sequence_length or config.raw_input_dim % width != 0:
                raise ValueError("packed spatial input does not match its coordinate lattice")
            self.packed_spatial_channels = config.raw_input_dim // width
            active_config = replace(
                config,
                sequence_length=height * width,
                raw_input_dim=self.packed_spatial_channels,
            )
        super().__init__(
            active_config,
            output_dim,
            HEADROOM_SPECS["B"],
            coordinate_shape=coordinate_shape,
            objective=objective,
            pac_variant=UMD_VARIANT,
        )
        self.raw_packed_shape = (
            coordinate_shape if self.packed_spatial_channels is not None else None
        )
        self.forward_block.excitation_mixer = UndecimatedModalDyadicMixer(
            self.modes,
            lattice_shape=self.coordinate_shape,
        )
        self.backward_block.excitation_mixer = UndecimatedModalDyadicMixer(
            self.modes,
            lattice_shape=self.coordinate_shape,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if self.packed_spatial_channels is not None:
            if self.raw_packed_shape is None:
                raise RuntimeError("packed spatial input is missing its lattice shape")
            height, width = self.raw_packed_shape
            inputs = inputs.reshape(
                inputs.shape[0], height, width, self.packed_spatial_channels
            ).reshape(inputs.shape[0], height * width, self.packed_spatial_channels)
            time_delta = _repeat_metadata(time_delta, width)
            observation_mask = _repeat_metadata(observation_mask, width)
            valid_mask = _repeat_metadata(valid_mask, width)
        return super().forward(
            inputs,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )

    def post_optimizer_step(self) -> None:
        super().post_optimizer_step()
        self._project_modal_weights()

    def finalize_constraints(self) -> None:
        super().finalize_constraints()
        self._project_modal_weights()

    @torch.no_grad()
    def _project_modal_weights(self) -> None:
        for block in (self.forward_block, self.backward_block):
            mixer = block.excitation_mixer
            if isinstance(mixer, UndecimatedModalDyadicMixer):
                mixer.project_weights_()


class SparseMultiscaleResidualMixer(torch.nn.Module):
    """Non-expansive low/detail diffusion bank over a chain or 2-D lattice."""

    def __init__(self, model_dim: int, *, scales: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        if model_dim < 1:
            raise ValueError("model_dim must be positive")
        if not scales or any(scale < 1 for scale in scales):
            raise ValueError("mixer scales must be positive")
        if tuple(sorted(set(scales))) != scales:
            raise ValueError("mixer scales must be sorted and unique")
        self.scales = scales
        component_count = 2 * len(scales)
        initial_logits = torch.full((component_count, model_dim), -2.0)
        channel_indices = torch.arange(model_dim)
        initial_logits[channel_indices % component_count, channel_indices] = 2.0
        self.component_logits = torch.nn.Parameter(initial_logits)
        self.residual_logit = torch.nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        inputs: Tensor,
        lattice_shape: tuple[int, int] | None = None,
    ) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("multiscale mixer inputs must have shape [B,N,D]")
        shape = _checked_lattice_shape(inputs.shape[1], lattice_shape)
        components: list[Tensor] = []
        current = inputs
        scale_index = 0
        for power in range(1, self.scales[-1] + 1):
            current = _normalized_lattice_step(current, shape)
            if power != self.scales[scale_index]:
                continue
            components.extend((current, 0.5 * (inputs - current)))
            scale_index += 1
            if scale_index == len(self.scales):
                break
        weights = torch.softmax(self.component_logits, dim=0).to(dtype=inputs.dtype)
        mixed = (torch.stack(components, dim=0) * weights[:, None, None, :]).sum(dim=0)
        beta = torch.sigmoid(self.residual_logit).to(dtype=inputs.dtype)
        return (1.0 - beta) * inputs + beta * mixed


class SparseMultiscaleResidualPAC(HeadroomPACClassifier):
    """Single-core PAC with shift-robust sparse multiscale low/detail mixing."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        coordinate_shape: tuple[int, int] | None = None,
        objective: HeadroomObjective,
        second_direction: Direction = "backward",
        extra_directions: tuple[Direction, ...] = (),
    ) -> None:
        self.packed_spatial_channels: int | None = None
        active_config = config
        if coordinate_shape is not None and math.prod(coordinate_shape) != config.sequence_length:
            height, width = coordinate_shape
            if height != config.sequence_length or config.raw_input_dim % width != 0:
                raise ValueError("packed spatial input does not match its coordinate lattice")
            self.packed_spatial_channels = config.raw_input_dim // width
            active_config = replace(
                config,
                sequence_length=height * width,
                raw_input_dim=self.packed_spatial_channels,
            )
        super().__init__(
            active_config,
            output_dim,
            HEADROOM_SPECS["C"],
            coordinate_shape=coordinate_shape,
            objective=objective,
        )
        self.raw_packed_shape = (
            coordinate_shape if self.packed_spatial_channels is not None else None
        )
        self.multiscale_mixer = SparseMultiscaleResidualMixer(config.model_dim)
        self.extra_directions = extra_directions
        self.extra_direction_blocks = torch.nn.ModuleList(
            _TightFrameBlock(
                active_config.model_dim,
                self.modes,
                _BlockVariant(direction, UNIFIED_VARIANT),
            )
            for direction in extra_directions
        )
        if second_direction == "forward":
            self.backward_block = _TightFrameBlock(
                active_config.model_dim,
                self.modes,
                _BlockVariant("forward", UNIFIED_VARIANT),
            )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if self.packed_spatial_channels is not None:
            if self.raw_packed_shape is None:
                raise RuntimeError("packed spatial input is missing its lattice shape")
            height, width = self.raw_packed_shape
            inputs = inputs.reshape(
                inputs.shape[0], height, width, self.packed_spatial_channels
            ).reshape(inputs.shape[0], height * width, self.packed_spatial_channels)
            time_delta = _repeat_metadata(time_delta, width)
            observation_mask = _repeat_metadata(observation_mask, width)
            valid_mask = _repeat_metadata(valid_mask, width)
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        encoded = self.stem(stem_inputs)
        active_delta = _stem_reduce_metadata(time_delta, self.stem_stride, "sum")
        active_observation = _stem_reduce_metadata(observation_mask, self.stem_stride, "max")
        active_valid = _stem_reduce_metadata(valid_mask, self.stem_stride, "max")
        encoded = self.multiscale_mixer(encoded, self.coordinate_shape)
        if active_valid is not None:
            encoded = encoded * active_valid.to(device=encoded.device, dtype=encoded.dtype)
        encoded, forward_moments = self.forward_block(
            encoded,
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
        if self.extra_direction_blocks:
            forward_group = [forward_moments]
            backward_group = [backward_moments]
            for direction, block in zip(
                self.extra_directions,
                self.extra_direction_blocks,
                strict=True,
            ):
                encoded, moments = block(
                    encoded,
                    time_delta=active_delta,
                    observation_mask=active_observation,
                    valid_mask=active_valid,
                )
                (forward_group if direction == "forward" else backward_group).append(moments)
            forward_moments = torch.stack(forward_group).mean(dim=0)
            backward_moments = torch.stack(backward_group).mean(dim=0)
        return self._readout(encoded, forward_moments, backward_moments, active_valid)

    def post_optimizer_step(self) -> None:
        super().post_optimizer_step()
        for block in self.extra_direction_blocks:
            block.retract_frame()

    def finalize_constraints(self) -> None:
        super().finalize_constraints()
        for block in self.extra_direction_blocks:
            block.finalize_frame()


class ModalMomentPyramidPAC(HeadroomPACClassifier):
    """Single-pass PAC with fine, Haar-low, and Haar-detail modal moments."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
    ) -> None:
        super().__init__(config, output_dim, HEADROOM_SPECS["B"], objective=objective)
        self.moment_scale_logits = torch.nn.Parameter(torch.tensor([1.5, 0.0, -0.5]))

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        branch_inputs, active_delta, active_observation, active_valid = self._stemmed(
            inputs, time_delta, observation_mask, valid_mask
        )
        encoded, fine_forward, forward_real, forward_imag = self.forward_block(
            branch_inputs,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_modal_states=True,
        )
        encoded, fine_backward, backward_real, backward_imag = self.backward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_modal_states=True,
        )
        weights = torch.softmax(self.moment_scale_logits, dim=0)
        forward_moments = _fused_dyadic_moments(
            fine_forward,
            forward_real,
            forward_imag,
            active_valid,
            self.forward_block.direction,
            self.forward_block.log_energy,
            self.forward_block.normalize_autocorrelation,
            self.forward_block.moment_lags,
            weights,
        )
        backward_moments = _fused_dyadic_moments(
            fine_backward,
            backward_real,
            backward_imag,
            active_valid,
            self.backward_block.direction,
            self.backward_block.log_energy,
            self.backward_block.normalize_autocorrelation,
            self.backward_block.moment_lags,
            weights,
        )
        return self._readout(encoded, forward_moments, backward_moments, active_valid)

    def _stemmed(
        self,
        inputs: Tensor,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        return (
            self.stem(stem_inputs),
            _stem_reduce_metadata(time_delta, self.stem_stride, "sum"),
            _stem_reduce_metadata(observation_mask, self.stem_stride, "max"),
            _stem_reduce_metadata(valid_mask, self.stem_stride, "max"),
        )


class WaveletPacketPAC(HeadroomPACClassifier):
    """Critically sampled low/detail bands processed in one shared PAC call."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
        pac_variant: StiefelVariant | None = None,
    ) -> None:
        super().__init__(
            config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective=objective,
            pac_variant=pac_variant,
        )
        self.band_logits = torch.nn.Parameter(torch.zeros(2))
        self.use_batched_phase_inference = False
        self.use_fused_pa2wp_stem_inference = False
        self.use_fused_pa2wp_stem_training = False
        self.use_large_fused_pa2wp_stem_training = False

    def prepare_fused_pa2wp_stem_training_(
        self,
        *,
        include_large_workloads: bool = False,
    ) -> WaveletPacketPAC:
        """Enable the exact training stem, optionally including captured large cells."""
        conv = cast("_CausalStem", self.stem).conv
        if (
            conv.in_channels != 1
            or conv.kernel_size != (9,)
            or conv.stride != (2,)
            or conv.bias is None
        ):
            message = "fused PA2WP training stem requires the canonical 1-channel K9/S2 stem"
            raise ValueError(message)
        self.use_fused_pa2wp_stem_training = True
        self.use_large_fused_pa2wp_stem_training = include_large_workloads
        return self

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        low, detail, pair_delta = _weighted_haar(stem_inputs, time_delta)
        pair_observation = _pair_mask(observation_mask)
        pair_valid = _pair_mask(valid_mask)
        band_inputs = torch.cat((low, detail), dim=0)
        band_delta = _duplicate_batch(pair_delta)
        band_observation = _duplicate_batch(pair_observation)
        band_valid = _duplicate_batch(pair_valid)
        encoded = self.stem(band_inputs)
        active_delta = _stem_reduce_metadata(band_delta, self.stem_stride, "sum")
        active_observation = _stem_reduce_metadata(band_observation, self.stem_stride, "max")
        active_valid = _stem_reduce_metadata(band_valid, self.stem_stride, "max")
        encoded, forward_moments = self.forward_block(
            encoded,
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
        logits = self._readout(encoded, forward_moments, backward_moments, active_valid)
        low_logits, detail_logits = logits.chunk(2, dim=0)
        weights = torch.softmax(self.band_logits, dim=0).to(dtype=logits.dtype)
        return weights[0] * low_logits + weights[1] * detail_logits


class PhaseCompleteWaveletPacketPAC(WaveletPacketPAC):
    """WP-PAC with a bounded residual mixture of the one-step-shifted pair phase."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.phase_logit = torch.nn.Parameter(torch.tensor(-3.0))
        self.phase_max = 0.25

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        phase_fraction = self.phase_max * torch.sigmoid(self.phase_logit)
        low, detail, pair_delta = _phase_complete_weighted_haar(
            stem_inputs,
            time_delta,
            phase_fraction,
        )
        pair_observation = _phase_complete_pair_mask(observation_mask)
        pair_valid = _phase_complete_pair_mask(valid_mask)
        band_inputs = torch.cat((low, detail), dim=0)
        band_delta = _duplicate_batch(pair_delta)
        band_observation = _duplicate_batch(pair_observation)
        band_valid = _duplicate_batch(pair_valid)
        encoded = self.stem(band_inputs)
        active_delta = _stem_reduce_metadata(band_delta, self.stem_stride, "sum")
        active_observation = _stem_reduce_metadata(band_observation, self.stem_stride, "max")
        active_valid = _stem_reduce_metadata(band_valid, self.stem_stride, "max")
        encoded, forward_moments = self.forward_block(
            encoded,
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
        logits = self._readout(encoded, forward_moments, backward_moments, active_valid)
        low_logits, detail_logits = logits.chunk(2, dim=0)
        weights = torch.softmax(self.band_logits, dim=0).to(dtype=logits.dtype)
        return weights[0] * low_logits + weights[1] * detail_logits


class DualPhaseWaveletPacketPAC(WaveletPacketPAC):
    """Average predictions from both adjacent-pair phases through one shared PAC core."""

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if (
            self.use_batched_phase_inference
            and not self.training
            and inputs.shape[1] >= 2
            and inputs.shape[1] % 2 == 0
            and time_delta is None
            and observation_mask is None
            and valid_mask is None
        ):
            return self._batched_phase_logits(inputs)
        ordinary = self._phase_logits(
            inputs,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        if inputs.shape[1] < 2:
            return ordinary
        shifted = self._phase_logits(
            inputs[:, 1:],
            time_delta=None if time_delta is None else time_delta[:, 1:],
            observation_mask=None if observation_mask is None else observation_mask[:, 1:],
            valid_mask=None if valid_mask is None else valid_mask[:, 1:],
        )
        return 0.5 * (ordinary + shifted)

    def prepare_batched_phase_inference_(self) -> DualPhaseWaveletPacketPAC:
        """Enable the shape-compatible, metadata-free dual-phase inference fast path."""
        self.eval()
        self.use_batched_phase_inference = True
        return self

    def prepare_fused_pa2wp_stem_inference_(self) -> DualPhaseWaveletPacketPAC:
        """Enable the raw-to-four-band specialized PA2WP inference stem."""
        conv = cast("_CausalStem", self.stem).conv
        if (
            conv.in_channels != 1
            or conv.kernel_size != (9,)
            or conv.stride != (2,)
            or conv.bias is None
        ):
            message = "fused PA2WP stem requires the canonical 1-channel K9/S2 stem"
            raise ValueError(message)
        self.prepare_batched_phase_inference_()
        self.use_fused_pa2wp_stem_inference = True
        return self

    def _batched_phase_logits(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.shape[0]
        if self.use_fused_pa2wp_stem_inference:
            conv = cast("_CausalStem", self.stem).conv
            if conv.bias is None:
                message = "fused PA2WP stem bias disappeared after preparation"
                raise RuntimeError(message)
            encoded = pa2wp_stem_inference(inputs, conv.weight, conv.bias)
        else:
            ordinary_low, ordinary_detail, _ = _weighted_haar(inputs, None)
            shifted_low, shifted_detail, _ = _weighted_haar(inputs[:, 1:], None)
            if ordinary_low.shape[1] != shifted_low.shape[1]:
                message = "batched PA2WP phases require equal analyzed lengths"
                raise ValueError(message)
            band_inputs = torch.cat(
                (ordinary_low, ordinary_detail, shifted_low, shifted_detail), dim=0
            )
            encoded = self.stem(band_inputs)
        encoded, forward_moments = self.forward_block(encoded)
        encoded, backward_moments = self.backward_block(encoded)
        logits = self._readout(encoded, forward_moments, backward_moments, None)
        ordinary_low_logits, ordinary_detail_logits, shifted_low_logits, shifted_detail_logits = (
            logits.split(batch_size, dim=0)
        )
        weights = torch.softmax(self.band_logits, dim=0).to(dtype=logits.dtype)
        ordinary = weights[0] * ordinary_low_logits + weights[1] * ordinary_detail_logits
        shifted = weights[0] * shifted_low_logits + weights[1] * shifted_detail_logits
        return 0.5 * (ordinary + shifted)

    def _phase_logits(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if (
            self.use_fused_pa2wp_stem_training
            and self.training
            and time_delta is None
            and observation_mask is None
            and valid_mask is None
            and inputs.shape[-1] == 1
            and (
                inputs.shape[0] * inputs.shape[1] <= _PA2WP_TRAINING_STEM_MAX_BATCH_STEPS
                or (
                    self.use_large_fused_pa2wp_stem_training
                    and inputs.shape[0] * inputs.shape[1]
                    <= _PA2WP_TRAINING_STEM_LARGE_MAX_BATCH_STEPS
                )
            )
        ):
            conv = cast("_CausalStem", self.stem).conv
            if (
                conv.in_channels != 1
                or conv.kernel_size != (9,)
                or conv.stride != (2,)
                or conv.bias is None
            ):
                message = "fused PA2WP training stem requires the canonical 1-channel K9/S2 stem"
                raise ValueError(message)
            encoded = pa2wp_training_stem(inputs, conv.weight, conv.bias)
            active_delta = None
            active_observation = None
            active_valid = None
        else:
            stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
            low, detail, pair_delta = _weighted_haar(stem_inputs, time_delta)
            pair_observation = _pair_mask(observation_mask)
            pair_valid = _pair_mask(valid_mask)
            band_inputs = torch.cat((low, detail), dim=0)
            band_delta = _duplicate_batch(pair_delta)
            band_observation = _duplicate_batch(pair_observation)
            band_valid = _duplicate_batch(pair_valid)
            encoded = self.stem(band_inputs)
            active_delta = _stem_reduce_metadata(band_delta, self.stem_stride, "sum")
            active_observation = _stem_reduce_metadata(band_observation, self.stem_stride, "max")
            active_valid = _stem_reduce_metadata(band_valid, self.stem_stride, "max")
        encoded, forward_moments = self.forward_block(
            encoded,
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
        logits = self._readout(encoded, forward_moments, backward_moments, active_valid)
        low_logits, detail_logits = logits.chunk(2, dim=0)
        weights = torch.softmax(self.band_logits, dim=0).to(dtype=logits.dtype)
        return weights[0] * low_logits + weights[1] * detail_logits


class PhaseAugmentedWaveletPacketPAC(DualPhaseWaveletPacketPAC):
    """Train on a random pair origin while retaining single-phase inference by default."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
        ensemble_inference: bool = False,
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.ensemble_inference = ensemble_inference

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if not self.training:
            if self.ensemble_inference:
                return super().forward(
                    inputs,
                    time_delta=time_delta,
                    observation_mask=observation_mask,
                    valid_mask=valid_mask,
                )
            return self._phase_logits(
                inputs,
                time_delta=time_delta,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
            )
        use_shifted = inputs.shape[1] > 1 and bool(torch.rand((), device=inputs.device) < 0.5)
        if use_shifted:
            inputs = inputs[:, 1:]
            time_delta = None if time_delta is None else time_delta[:, 1:]
            observation_mask = None if observation_mask is None else observation_mask[:, 1:]
            valid_mask = None if valid_mask is None else valid_mask[:, 1:]
        return self._phase_logits(
            inputs,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )


class LearnedPairWaveletPacketPAC(DualPhaseWaveletPacketPAC):
    """WP control with an unconstrained learned 2-tap, 2-band pair filter."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.pair_mix = torch.nn.Parameter(torch.eye(2))

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        low, detail, pair_delta = _weighted_haar(stem_inputs, time_delta)
        bands = torch.stack((low, detail), dim=-1)
        mixed = torch.einsum("bkcq,qr->bkcr", bands, self.pair_mix.to(dtype=bands.dtype))
        return self._band_logits(
            mixed[..., 0],
            mixed[..., 1],
            pair_delta=pair_delta,
            pair_observation=_pair_mask(observation_mask),
            pair_valid=_pair_mask(valid_mask),
        )

    def _band_logits(
        self,
        low: Tensor,
        detail: Tensor,
        *,
        pair_delta: Tensor | None,
        pair_observation: Tensor | None,
        pair_valid: Tensor | None,
    ) -> Tensor:
        band_inputs = torch.cat((low, detail), dim=0)
        band_delta = _duplicate_batch(pair_delta)
        band_observation = _duplicate_batch(pair_observation)
        band_valid = _duplicate_batch(pair_valid)
        encoded = self.stem(band_inputs)
        active_delta = _stem_reduce_metadata(band_delta, self.stem_stride, "sum")
        active_observation = _stem_reduce_metadata(band_observation, self.stem_stride, "max")
        active_valid = _stem_reduce_metadata(band_valid, self.stem_stride, "max")
        encoded, forward_moments = self.forward_block(
            encoded,
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
        logits = self._readout(encoded, forward_moments, backward_moments, active_valid)
        low_logits, detail_logits = logits.chunk(2, dim=0)
        weights = torch.softmax(self.band_logits, dim=0).to(dtype=logits.dtype)
        return weights[0] * low_logits + weights[1] * detail_logits


class AuxiliaryDistilledPAC(HeadroomPACClassifier):
    """Baseline PAC at inference with an optional detached coarse training branch."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
    ) -> None:
        super().__init__(config, output_dim, HEADROOM_SPECS["B"], objective=objective)
        self.aux_mixer_logit = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        fine, _ = self._fine(
            inputs,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        return fine

    def forward_with_auxiliary(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        fine, trace = self._fine(
            inputs,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded, active_delta, active_observation, active_valid = trace
        previous_shape = self.coordinate_shape
        coarse_inputs, _ = _pool_features(encoded.detach(), previous_shape)
        coarse_delta = _pool_metadata(active_delta, previous_shape, "sum")
        coarse_observation = _pool_metadata(active_observation, previous_shape, "max")
        coarse_valid = _pool_metadata(active_valid, previous_shape, "max")
        coarse_encoded, forward_moments = self.forward_block(
            coarse_inputs,
            time_delta=coarse_delta,
            observation_mask=coarse_observation,
            valid_mask=coarse_valid,
        )
        coarse_encoded, backward_moments = self.backward_block(
            coarse_encoded,
            time_delta=coarse_delta,
            observation_mask=coarse_observation,
            valid_mask=coarse_valid,
        )
        beta = torch.sigmoid(self.aux_mixer_logit).to(dtype=coarse_encoded.dtype)
        coarse_encoded = (1.0 - beta) * coarse_encoded + beta * torch.tanh(coarse_encoded)
        coarse = self._readout(
            coarse_encoded,
            forward_moments,
            backward_moments,
            coarse_valid,
        )
        return fine, coarse

    def _fine(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[
        Tensor,
        tuple[Tensor, Tensor | None, Tensor | None, Tensor | None],
    ]:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        encoded = self.stem(stem_inputs)
        active_delta = _stem_reduce_metadata(time_delta, self.stem_stride, "sum")
        active_observation = _stem_reduce_metadata(observation_mask, self.stem_stride, "max")
        active_valid = _stem_reduce_metadata(valid_mask, self.stem_stride, "max")
        encoded, forward_moments = self.forward_block(
            encoded,
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
        logits = self._readout(encoded, forward_moments, backward_moments, active_valid)
        return logits, (encoded, active_delta, active_observation, active_valid)


def build_efficient_headroom_classifier(  # noqa: C901, PLR0911, PLR0912
    spec: EfficientHeadroomSpec,
    config: PACExperimentConfig,
    output_dim: int,
    *,
    coordinate_shape: tuple[int, int] | None = None,
    objective: HeadroomObjective,
) -> HeadroomPACClassifier:
    if spec == "MP":
        return ModalMomentPyramidPAC(config, output_dim, objective=objective)
    if spec == "WP":
        return WaveletPacketPAC(config, output_dim, objective=objective)
    if spec == "PCWP":
        return PhaseCompleteWaveletPacketPAC(config, output_dim, objective=objective)
    if spec == "DPWP":
        return DualPhaseWaveletPacketPAC(config, output_dim, objective=objective)
    if spec == "PAWP":
        return PhaseAugmentedWaveletPacketPAC(config, output_dim, objective=objective)
    if spec == "PA2WP":
        return PhaseAugmentedWaveletPacketPAC(
            config,
            output_dim,
            objective=objective,
            ensemble_inference=True,
        )
    if spec == "LPWP":
        return LearnedPairWaveletPacketPAC(config, output_dim, objective=objective)
    if spec == "OA":
        return OverlappingAntiAliasedPAC(config, output_dim, objective=objective)
    if spec == "AD":
        return AuxiliaryDistilledPAC(config, output_dim, objective=objective)
    if spec == "SMR":
        return SparseMultiscaleResidualPAC(
            config,
            output_dim,
            coordinate_shape=coordinate_shape,
            objective=objective,
        )
    if spec == "SMRFF":
        return SparseMultiscaleResidualPAC(
            config,
            output_dim,
            coordinate_shape=coordinate_shape,
            objective=objective,
            second_direction="forward",
        )
    if spec == "SMRFBFB":
        return SparseMultiscaleResidualPAC(
            config,
            output_dim,
            coordinate_shape=coordinate_shape,
            objective=objective,
            extra_directions=("forward", "backward"),
        )
    if spec == "UMD":
        return UndecimatedModalDyadicPAC(
            config,
            output_dim,
            coordinate_shape=coordinate_shape,
            objective=objective,
        )
    if spec == "EFP8":
        return EdgeFramePAC(
            config,
            output_dim,
            modes=8,
            semi_orthogonal=True,
            objective=objective,
        )
    if spec == "EFP16":
        return EdgeFramePAC(
            config,
            output_dim,
            modes=16,
            semi_orthogonal=True,
            objective=objective,
        )
    if spec == "EFU8":
        return EdgeFramePAC(
            config,
            output_dim,
            modes=8,
            semi_orthogonal=False,
            objective=objective,
        )
    if spec == "C2M8":
        return StrideOneConvPAC(config, output_dim, objective=objective)
    raise ValueError(f"unknown efficient headroom spec: {spec}")


def _fused_dyadic_moments(
    fine: Tensor,
    states_real: Tensor,
    states_imag: Tensor,
    valid_mask: Tensor | None,
    direction: Direction,
    log_energy: bool,
    normalize_autocorrelation: bool,
    lags: tuple[int, ...],
    weights: Tensor,
) -> Tensor:
    low_real, detail_real = _haar_equal(states_real)
    low_imag, detail_imag = _haar_equal(states_imag)
    pair_valid = _pair_mask(valid_mask)
    variant = _MomentVariant(direction, log_energy, normalize_autocorrelation, lags)
    if pair_valid is None:
        low = _modal_moments(low_real, low_imag, variant)
        detail = _modal_moments(detail_real, detail_imag, variant)
    else:
        low = _masked_modal_moments(low_real, low_imag, pair_valid, variant)
        detail = _masked_modal_moments(detail_real, detail_imag, pair_valid, variant)
    active = weights.to(device=fine.device, dtype=fine.dtype)
    return active[0] * fine + active[1] * low + active[2] * detail


def _weighted_haar(
    inputs: Tensor,
    time_delta: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    padded = _pad_sequence(inputs, value=0.0)
    first = padded[:, 0::2]
    second = padded[:, 1::2]
    if time_delta is None:
        scale = 1.0 / math.sqrt(2.0)
        return scale * (first + second), scale * (first - second), None
    delta = _metadata_3d(time_delta)
    delta = _pad_sequence(delta, value=0.0)
    first_delta = delta[:, 0::2].clamp_min(0.0)
    second_delta = delta[:, 1::2].clamp_min(0.0)
    total = first_delta + second_delta
    safe_total = total.clamp_min(torch.finfo(inputs.dtype).eps)
    first_weight = torch.sqrt(first_delta / safe_total).to(dtype=inputs.dtype)
    second_weight = torch.sqrt(second_delta / safe_total).to(dtype=inputs.dtype)
    empty = total <= 0.0
    equal = inputs.new_tensor(1.0 / math.sqrt(2.0))
    first_weight = torch.where(empty, equal, first_weight)
    second_weight = torch.where(empty, equal, second_weight)
    low = first_weight * first + second_weight * second
    detail = second_weight * first - first_weight * second
    return low, detail, total


def _phase_complete_weighted_haar(
    inputs: Tensor,
    time_delta: Tensor | None,
    phase_fraction: Tensor,
) -> tuple[Tensor, Tensor, Tensor | None]:
    low, detail, pair_delta = _weighted_haar(inputs, time_delta)
    shifted_inputs = _shift_left_with_zero(inputs)
    shifted_delta = None if time_delta is None else _shift_left_with_zero(_metadata_3d(time_delta))
    shifted_low, shifted_detail, shifted_pair_delta = _weighted_haar(
        shifted_inputs,
        shifted_delta,
    )
    active = phase_fraction.to(device=inputs.device, dtype=inputs.dtype)
    mixed_low = (1.0 - active) * low + active * shifted_low
    mixed_detail = (1.0 - active) * detail + active * shifted_detail
    if pair_delta is None or shifted_pair_delta is None:
        return mixed_low, mixed_detail, None
    mixed_delta = (1.0 - active) * pair_delta + active * shifted_pair_delta
    return mixed_low, mixed_detail, mixed_delta


def _phase_complete_pair_mask(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None
    active = _metadata_3d(mask)
    ordinary = _pair_mask(active)
    shifted = _pair_mask(_shift_left_with_zero(active))
    if ordinary is None or shifted is None:
        raise RuntimeError("phase-complete masks unexpectedly disappeared")
    return torch.maximum(ordinary, shifted)


def _shift_left_with_zero(inputs: Tensor) -> Tensor:
    if inputs.ndim != 3 or inputs.shape[1] < 1:
        raise ValueError("shift inputs must have shape [B,N>=1,D]")
    return torch.cat((inputs[:, 1:], torch.zeros_like(inputs[:, :1])), dim=1)


def _overlapping_low_detail(inputs: Tensor) -> tuple[Tensor, Tensor]:
    if inputs.ndim != 3 or inputs.shape[1] < 2:
        raise ValueError("overlapping analysis requires inputs with shape [B,N>=2,D]")
    first = inputs[:, :-1]
    second = inputs[:, 1:]
    return 0.5 * (first + second), 0.5 * (second - first)


def _degree_normalized_edge_analysis(
    inputs: Tensor,
    time_delta: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    if inputs.ndim != 3 or inputs.shape[1] < 1:
        raise ValueError("edge analysis requires inputs with shape [B,N>=1,D]")
    if inputs.shape[1] == 1:
        return inputs, torch.zeros_like(inputs), _edge_or_singleton_delta(time_delta)
    degree = torch.ones(inputs.shape[1], device=inputs.device, dtype=inputs.dtype)
    if inputs.shape[1] > 2:
        degree[1:-1] = 2.0
    scaled = inputs * degree.rsqrt().view(1, -1, 1)
    first = scaled[:, :-1]
    second = scaled[:, 1:]
    if time_delta is None:
        scale = 1.0 / math.sqrt(2.0)
        return scale * (first + second), scale * (first - second), None
    delta = _metadata_3d(time_delta).clamp_min(0.0)
    first_delta = delta[:, :-1]
    second_delta = delta[:, 1:]
    total = first_delta + second_delta
    safe_total = total.clamp_min(torch.finfo(inputs.dtype).eps)
    first_weight = torch.sqrt(first_delta / safe_total).to(dtype=inputs.dtype)
    second_weight = torch.sqrt(second_delta / safe_total).to(dtype=inputs.dtype)
    empty = total <= 0.0
    equal = 1.0 / math.sqrt(2.0)
    first_weight = torch.where(empty, equal, first_weight)
    second_weight = torch.where(empty, equal, second_weight)
    low = first_weight * first + second_weight * second
    detail = second_weight * first - first_weight * second
    return low, detail, second_delta


def _edge_or_singleton_delta(time_delta: Tensor | None) -> Tensor | None:
    if time_delta is None:
        return None
    active = _metadata_3d(time_delta)
    return active if active.shape[1] == 1 else active[:, 1:]


def _edge_or_singleton_mask(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None
    active = _metadata_3d(mask)
    if active.shape[1] == 1:
        return active
    return torch.minimum(active[:, :-1], active[:, 1:])


def _edge_delta(time_delta: Tensor | None) -> Tensor | None:
    if time_delta is None:
        return None
    return _metadata_3d(time_delta)[:, 1:]


def _edge_joint_mask(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None
    active = _metadata_3d(mask)
    return torch.minimum(active[:, :-1], active[:, 1:])


def _combined_edge_mask(first: Tensor | None, second: Tensor | None) -> Tensor | None:
    if first is None:
        return second
    if second is None:
        return first
    return torch.minimum(first, second)


def _overlap_support_mask(edge_mask: Tensor | None) -> Tensor | None:
    if edge_mask is None:
        return None
    return functional.max_pool1d(
        edge_mask.transpose(1, 2), kernel_size=3, stride=2, padding=1
    ).transpose(1, 2)


def _haar_equal(inputs: Tensor) -> tuple[Tensor, Tensor]:
    padded = _pad_sequence(inputs, value=0.0)
    scale = 1.0 / math.sqrt(2.0)
    first = padded[:, 0::2]
    second = padded[:, 1::2]
    return scale * (first + second), scale * (first - second)


def _apply_raw_mask(
    inputs: Tensor,
    observation_mask: Tensor | None,
    valid_mask: Tensor | None,
) -> Tensor:
    # Missing observations and padding are independent invalidity sources.
    # Apply their intersection before any temporal convolution so padded values
    # cannot leak back into valid positions through a local receptive field.
    raw_mask = _combined_edge_mask(observation_mask, valid_mask)
    if raw_mask is None:
        return inputs
    return inputs * _metadata_3d(raw_mask).to(device=inputs.device, dtype=inputs.dtype)


def _pair_mask(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None
    active = _pad_sequence(_metadata_3d(mask), value=0.0)
    return torch.maximum(active[:, 0::2], active[:, 1::2])


def _metadata_3d(values: Tensor) -> Tensor:
    if values.ndim == 2:
        return values.unsqueeze(-1)
    if values.ndim == 3 and values.shape[-1] == 1:
        return values
    raise ValueError("metadata must have shape [B,N] or [B,N,1]")


def _pad_sequence(inputs: Tensor, *, value: float) -> Tensor:
    if inputs.shape[1] % 2 == 0:
        return inputs
    return functional.pad(inputs, (0, 0, 0, 1), value=value)


def _duplicate_batch(values: Tensor | None) -> Tensor | None:
    return None if values is None else torch.cat((values, values), dim=0)


def _repeat_metadata(values: Tensor | None, repeats: int) -> Tensor | None:
    return None if values is None else values.repeat_interleave(repeats, dim=1)
