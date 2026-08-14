from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, assert_never, overload

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from .pac_compiled_lag124_moments import lag124_modal_moments
from .pac_real2d_math import (
    compiled_discrete_pole_real2d,
    discrete_pole_real2d,
    pole_transition_real2d,
)
from .pac_recurrence import RecurrenceBackend, recurrence_real2d_directional
from .pac_stiefel_variants import (
    LEGACY_VARIANT,
    FrameParameterization,
    StiefelVariant,
    capacity_for_model,
    uses_full_modal_frame,
    variant_for_model,
)
from .pac_triton_excitation_recurrence_training import (
    fused_excitation_recurrence_moments_training,
    supports_fused_excitation_recurrence,
)
from .pac_triton_online_moments import online_modal_moments
from .pac_triton_ordered_pool import final_rmsnorm_ordered_pool
from .pac_triton_parallel_static_recurrence_lag124_training import (
    parallel_static_excitation_recurrence_lag124_moments_only_training,
    parallel_static_excitation_recurrence_lag124_moments_packed_io_training,
    parallel_static_radial_log_recurrence_lag124_moments_only_inference,
    parallel_static_radial_log_recurrence_lag124_moments_only_training,
    parallel_static_radial_log_recurrence_lag124_moments_packed_io_inference,
    parallel_static_radial_log_recurrence_lag124_moments_packed_io_training,
    parallel_static_recurrence_lag124_moments_only_training,
    parallel_static_recurrence_lag124_moments_packed_io_training,
)
from .pac_triton_radial_log_recurrence_lag124 import (
    static_radial_log_recurrence_lag124_moments_only_inference,
    static_radial_log_recurrence_lag124_moments_packed_io_inference,
)
from .pac_triton_recurrence_lag124 import (
    static_recurrence_lag124_moments_only_inference,
    static_recurrence_lag124_moments_packed_io_inference,
)
from .pac_triton_recurrence_lag124_training import (
    static_recurrence_lag124_moments_only_saved_states_training,
    static_recurrence_lag124_moments_only_training,
    static_recurrence_lag124_moments_packed_io_training,
)
from .pac_triton_recurrence_moments import (
    recurrence_moments_inference,
    static_recurrence_moments_inference,
    static_recurrence_moments_packed_inference,
    static_recurrence_moments_packed_input_inference,
    static_recurrence_moments_packed_io_inference,
)
from .pac_triton_recurrence_moments_training import (
    fused_recurrence_moments_packed_training,
    fused_recurrence_moments_training,
)
from .pac_triton_static_block_scan import (
    static_pole_block_scan_packed_input_recurrence,
    static_pole_block_scan_packed_io_recurrence,
    static_pole_block_scan_packed_output_recurrence,
    static_pole_block_scan_recurrence,
)
from .pac_triton_writer_drive_inference import (
    writer_rmsnorm_drive_inference,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .pac_types import PACExperimentConfig

Direction = Literal["forward", "backward"]
TIGHT_FRAME_MODELS: Final[tuple[str, ...]] = ("pac_tight_frame_depth2_autocorr",)
_LAGS: Final[tuple[int, ...]] = (1, 4)
_LAGS_124: Final[tuple[int, ...]] = (1, 2, 4)
_EPSILON: Final[float] = 1.0e-8
_BLOCK_SCAN_INFERENCE_THRESHOLD: Final[int] = 2048


@dataclass(frozen=True, slots=True)
class _BlockVariant:
    direction: Direction
    stiefel: StiefelVariant


@dataclass(frozen=True, slots=True)
class _MomentVariant:
    physical_direction: Direction
    log_energy: bool
    normalize_autocorrelation: bool
    lags: tuple[int, ...] = _LAGS


class TightFrameConfigError(ValueError):
    def __init__(self, model_dim: int) -> None:
        self.model_dim = model_dim
        super().__init__(f"model_dim must be at least 4, got {model_dim}")


class TightFrameClassifier(nn.Module):
    supports_observation_mask: Final[bool] = True
    supports_time_delta: Final[bool] = True

    def __init__(
        self,
        config: PACExperimentConfig,
        class_count: int,
        variant: StiefelVariant = LEGACY_VARIANT,
        *,
        full_modal_frame: bool = False,
    ) -> None:
        super().__init__()
        if config.model_dim < 4:
            raise TightFrameConfigError(config.model_dim)
        self.model_dim = config.model_dim
        mode_divisor = 2 if full_modal_frame else 4
        self.modes = max(1, min(config.modes, config.model_dim // mode_divisor))
        self.expected_token_count = (
            config.sequence_length + variant.stem_stride - 1
        ) // variant.stem_stride
        self.use_modal_moments = variant.use_modal_moments
        self.use_ordered_pool = variant.use_ordered_pool
        self.use_backward_block = variant.use_backward_block and variant.depth >= 2
        self.pooling_scales = variant.pooling_scales
        self.stem = _CausalStem(
            config.raw_input_dim,
            self.model_dim,
            kernel_size=variant.stem_kernel,
            stride=variant.stem_stride,
        )
        self.forward_block = _TightFrameBlock(
            self.model_dim,
            self.modes,
            _BlockVariant("forward", variant),
        )
        self.backward_block = _TightFrameBlock(
            self.model_dim,
            self.modes,
            _BlockVariant("backward", variant),
        )
        if not self.use_backward_block:
            self.backward_block.requires_grad_(requires_grad=False)
        self.extra_blocks = nn.ModuleList(
            _TightFrameBlock(
                self.model_dim,
                self.modes,
                _BlockVariant("forward" if index % 2 == 0 else "backward", variant),
            )
            for index in range(max(variant.depth - 2, 0))
        )
        self.final_norm = nn.RMSNorm(self.model_dim)
        pooled_dim = (
            sum(self.pooling_scales) * self.model_dim if self.use_ordered_pool else self.model_dim
        )
        self.head = _InvariantMomentHead(
            pooled_dim,
            self.modes,
            class_count,
            use_modal_moments=self.use_modal_moments,
            use_backward_moments=self.use_backward_block,
            lags=variant.moment_lags,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        raw_observation = _raw_metadata_mask(inputs, observation_mask, "observation_mask")
        stem_inputs = inputs if raw_observation is None else inputs * raw_observation
        features = self.stem(stem_inputs)
        stride = self.stem.conv.stride[0]
        active_delta = _prepare_stem_metadata(
            time_delta,
            inputs,
            features,
            stride,
            "sum",
            "time_delta",
        )
        active_observation = _prepare_stem_metadata(
            observation_mask,
            inputs,
            features,
            stride,
            "max",
            "observation_mask",
        )
        active_valid = _prepare_stem_metadata(
            valid_mask,
            inputs,
            features,
            stride,
            "max",
            "valid_mask",
        )
        features, forward_moments = self.forward_block(
            features,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        if not self.use_backward_block:
            backward_moments = torch.zeros_like(forward_moments)
        else:
            features, backward_moments = self.backward_block(
                features,
                time_delta=active_delta,
                observation_mask=active_observation,
                valid_mask=active_valid,
            )
        for block in self.extra_blocks:
            features, _ = block(
                features,
                time_delta=active_delta,
                observation_mask=active_observation,
                valid_mask=active_valid,
            )
        if self.final_norm.weight is None:
            message = "final RMSNorm must use a learned weight"
            raise RuntimeError(message)
        normalized = _dtype_aligned_rms_norm(features, self.final_norm)
        if self.use_ordered_pool and self.pooling_scales == (1, 2, 4) and active_valid is None:
            pooled = final_rmsnorm_ordered_pool(
                features,
                self.final_norm.weight.to(dtype=features.dtype),
                eps=self.final_norm.eps,
            )
        elif self.use_ordered_pool:
            pooled = (
                _ordered_pool(normalized, self.pooling_scales)
                if active_valid is None
                else _masked_ordered_pool(normalized, active_valid, self.pooling_scales)
            )
        else:
            pooled = _masked_sequence_mean(normalized, active_valid)
        return self.head(pooled, forward_moments, backward_moments)

    def first_frame_matrix(self) -> Tensor:
        return self.forward_block.frame_matrix()

    def post_optimizer_step(self) -> None:
        self.forward_block.retract_frame()
        if self.use_backward_block:
            self.backward_block.retract_frame()
        for block in self.extra_blocks:
            if not isinstance(block, _TightFrameBlock):
                message = "tight-frame extra block has an invalid module type"
                raise TypeError(message)
            block.retract_frame()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        if self.use_backward_block:
            self.backward_block.finalize_frame()
        for block in self.extra_blocks:
            if not isinstance(block, _TightFrameBlock):
                message = "tight-frame extra block has an invalid module type"
                raise TypeError(message)
            block.finalize_frame()

    @torch.no_grad()
    def materialize_frames_for_inference_(
        self,
        *,
        use_block_scan: bool = False,
        use_fused_recurrence_moments: bool = False,
    ) -> TightFrameClassifier:
        """Freeze constrained frame values and remove their training parametrizations."""
        self.forward_block.prepare_for_inference_(
            use_block_scan=use_block_scan,
            use_fused_recurrence_moments=use_fused_recurrence_moments,
        )
        if self.use_backward_block:
            self.backward_block.prepare_for_inference_(
                use_block_scan=use_block_scan,
                use_fused_recurrence_moments=use_fused_recurrence_moments,
            )
        return self

    @torch.no_grad()
    def prepare_compiled_inference_(self) -> TightFrameClassifier:
        """Materialize frames and enable the long-sequence backend that wins after compile."""
        use_block_scan = self.expected_token_count >= _BLOCK_SCAN_INFERENCE_THRESHOLD
        return self.materialize_frames_for_inference_(
            use_block_scan=use_block_scan,
            use_fused_recurrence_moments=not use_block_scan,
        )

    @torch.no_grad()
    def load_materialized_inference_state_dict_(
        self,
        state_dict: Mapping[str, Tensor],
        *,
        compiled: bool = False,
    ) -> TightFrameClassifier:
        """Load a checkpoint saved after frame materialization into a fresh model."""
        if compiled:
            self.prepare_compiled_inference_()
        else:
            self.materialize_frames_for_inference_()
        self.load_state_dict(state_dict, strict=True)
        self.eval()
        return self


class _TightFrameBlock(nn.Module):
    def __init__(  # noqa: PLR0915
        self, model_dim: int, modes: int, variant: _BlockVariant
    ) -> None:
        super().__init__()
        self.direction: Direction = variant.direction
        self.synthesis_scale = variant.stiefel.synthesis_scale
        self.align_moments = variant.stiefel.align_moments
        self.log_energy = variant.stiefel.log_energy
        self.normalize_autocorrelation = variant.stiefel.normalize_autocorrelation
        self.frame_parameterization: FrameParameterization = variant.stiefel.frame_parameterization
        self.split_residual_scales = variant.stiefel.split_residual_scales
        self.qr_retraction_interval = variant.stiefel.qr_retraction_interval
        self.use_mode_gate = variant.stiefel.use_mode_gate
        self.use_local_convolution = variant.stiefel.use_local_convolution
        self.tie_analysis_synthesis = variant.stiefel.tie_analysis_synthesis
        self.moment_lags = variant.stiefel.moment_lags
        self.damping_min = variant.stiefel.damping_min
        self.damping_max = variant.stiefel.damping_max
        self.frequency_bound = float(torch.pi)
        self.impulse_injection = False
        self.gate_max = variant.stiefel.gate_max
        self._optimizer_steps = 0
        self.recurrence_backend: RecurrenceBackend = "auto"
        self.fused_moments_backward_training = False
        self.fused_lag124_moments = False
        self.skip_redundant_lag124_moment_fusion = False
        self.fused_recurrence_moments_backward_training = False
        self.fused_excitation_recurrence_training = False
        self.d32_rmsnorm_backward_training = False
        self.two_pass_reverse_recurrence_moments_training: bool | None = None
        self.packed_recurrence_moments_training: bool | None = None
        self.mode_static_pole_training = False
        self.parallel_static_recurrence_training = False
        self.parallel_static_excitation_recurrence_training = False
        self.static_lag124_recurrence_moments_training = False
        self.saved_state_lag124_reader_training = False
        self.fused_recurrence_moments_inference = False
        self.static_pole_recurrence_moments_inference = False
        self.static_lag124_recurrence_moments_inference = False
        self.radial_log_lag124_moments = False
        self.physical_time_lag_moments = False
        self.static_radial_log_lag124_recurrence_moments_inference = False
        self.parallel_static_radial_log_recurrence_moments_inference = False
        self.parallel_static_radial_log_recurrence_moments_training = False
        self.fused_writer_rmsnorm_drive_inference = False
        self.static_pole_block_scan_inference = False
        self.static_pole_block_scan_block_size_inference = 256
        self.parallel_static_pole_recurrence_moments_inference = False
        self.parallel_static_pole_recurrence_num_warps_inference = 4
        self.packed_static_recurrence_moments_inference = False
        self.packed_static_recurrence_drive_inference = False
        self.single_warp_static_recurrence_moments_inference = False
        self.canonical_identity_elision = True
        self.use_input_norm = True
        self.excitation_mixer: nn.Module | None = None
        self.norm = nn.RMSNorm(model_dim)
        self.local = (
            nn.Conv1d(
                model_dim,
                model_dim,
                variant.stiefel.local_kernel,
                groups=model_dim,
            )
            if self.use_local_convolution
            else None
        )
        self.frame = nn.Linear(2 * modes, model_dim, bias=False)
        self.register_buffer("intervention_frame_override", None, persistent=False)
        self.register_buffer("inference_drive_frame", None, persistent=False)
        self.register_buffer("inference_decay_real", None, persistent=False)
        self.register_buffer("inference_decay_imag", None, persistent=False)
        self.register_buffer("fixed_damping_values", None, persistent=False)
        nn.init.orthogonal_(self.frame.weight)
        if self.tie_analysis_synthesis:
            self.register_parameter("independent_synthesis_frame", None)
        else:
            # This control must learn an independent synthesis operator.  A fixed
            # random buffer would only test a basis mismatch, not untied learning.
            self.independent_synthesis_frame = nn.Parameter(
                _independent_synthesis_frame(model_dim, modes)
            )
        if self.frame_parameterization == "matrix_exp":
            orthogonal(
                self.frame,
                "weight",
                orthogonal_map="matrix_exp",
                use_trivialization=True,
            )
        self.raw_decay = nn.Parameter(torch.linspace(-3.0, 1.0, modes))
        frequency_grid = torch.linspace(0.0, 0.75, modes).clamp(max=0.999)
        self.raw_frequency = nn.Parameter(torch.atanh(frequency_grid))
        # Canonical blocks use the zero initial condition.  Focused validation
        # controls may opt into a learned complex initial state without
        # changing the recurrence kernels: the forward path folds ``A z_0``
        # into the first drive seen in the selected traversal direction.
        self.initial_state_real: nn.Parameter | None
        self.initial_state_imag: nn.Parameter | None
        self.register_parameter("initial_state_real", None)
        self.register_parameter("initial_state_imag", None)
        if self.use_mode_gate:
            self.mode_gate_bias = nn.Parameter(torch.zeros(modes))
            self.mode_gate_gain = nn.Parameter(torch.zeros(modes))
        else:
            self.register_parameter("mode_gate_bias", None)
            self.register_parameter("mode_gate_gain", None)
        if self.split_residual_scales:
            self.register_parameter("direct_scale", None)
            self.register_parameter("layer_scale", None)
            self.modal_scale = nn.Parameter(
                torch.full((model_dim,), variant.stiefel.layer_scale_init)
            )
            self.local_scale = nn.Parameter(torch.zeros(model_dim))
        else:
            self.direct_scale = nn.Parameter(torch.zeros(model_dim))
            self.layer_scale = nn.Parameter(
                torch.full((model_dim,), variant.stiefel.layer_scale_init)
            )
            self.register_parameter("modal_scale", None)
            self.register_parameter("local_scale", None)

    def frame_matrix(self) -> Tensor:
        override = self.intervention_frame()
        if override is not None:
            return override
        return self.frame.weight

    def intervention_frame(self) -> Tensor | None:
        return self._buffers.get("intervention_frame_override")

    def set_intervention_frame(self, value: Tensor | None) -> None:
        self._buffers["intervention_frame_override"] = value

    def synthesis_frame_matrix(self) -> Tensor:
        if self.independent_synthesis_frame is None:
            return self.frame_matrix()
        return self.independent_synthesis_frame

    def frequency_values(self) -> Tensor:
        return self.frequency_bound * torch.tanh(self.raw_frequency)

    def set_fixed_damping_prefix(self, count: int, value: float) -> None:
        """Fix a prefix of modes to one damping value without freezing frequencies."""
        if not 0 <= count <= self.raw_decay.numel():
            message = "fixed damping count must lie between zero and the mode count"
            raise ValueError(message)
        if not 0.0 <= value < self.damping_max:
            message = "fixed damping must be non-negative and below damping_max"
            raise ValueError(message)
        fixed = torch.full_like(self.raw_decay, torch.nan)
        fixed[:count] = value
        self._buffers["fixed_damping_values"] = fixed if count else None

    def _apply_fixed_damping(self, damping: Tensor) -> Tensor:
        fixed = self._buffers.get("fixed_damping_values")
        if fixed is None:
            return damping
        return torch.where(torch.isfinite(fixed), fixed, damping)

    def enable_trainable_initial_state(self) -> None:
        """Replace the canonical zero state with a learned complex state."""
        if self.initial_state_real is not None or self.initial_state_imag is not None:
            message = "trainable initial state is already enabled"
            raise RuntimeError(message)
        self.initial_state_real = nn.Parameter(torch.zeros_like(self.raw_decay))
        self.initial_state_imag = nn.Parameter(torch.zeros_like(self.raw_decay))

    def damping_values(self) -> Tensor:
        damping = self.damping_min + (self.damping_max - self.damping_min) * torch.sigmoid(
            self.raw_decay
        )
        return self._apply_fixed_damping(damping)

    def retract_frame(self) -> None:
        match self.frame_parameterization:
            case "matrix_exp" | "unconstrained":
                return
            case "qr_retraction":
                self._optimizer_steps += 1
                if self._optimizer_steps % self.qr_retraction_interval == 0:
                    _retract_columns(self.frame.weight)
            case unreachable:
                assert_never(unreachable)

    def finalize_frame(self) -> None:
        if self.frame_parameterization == "qr_retraction":
            _retract_columns(self.frame.weight)

    @torch.no_grad()
    def prepare_for_inference_(
        self,
        *,
        use_block_scan: bool,
        use_fused_recurrence_moments: bool,
        use_static_pole_recurrence_moments: bool = False,
        use_packed_static_recurrence_moments: bool = False,
        use_packed_static_recurrence_drive: bool = False,
        use_single_warp_static_recurrence_moments: bool = False,
        use_static_lag124_recurrence_moments: bool = False,
        use_static_pole_block_scan: bool = False,
        static_pole_block_scan_block_size: int = 256,
    ) -> None:
        if parametrize.is_parametrized(self.frame, "weight"):
            parametrize.remove_parametrizations(self.frame, "weight", leave_parametrized=True)
        self.frame_parameterization = "qr_retraction"
        self.recurrence_backend = "triton_scan_blocks" if use_block_scan else "auto"
        supported_moment_lags = self.moment_lags == _LAGS or (
            self.moment_lags == _LAGS_124 and self.fused_lag124_moments
        )
        canonical_moments = (
            self.align_moments
            and self.log_energy
            and self.normalize_autocorrelation
            and supported_moment_lags
        )
        native_fused_moments = self.moment_lags == _LAGS or (
            self.moment_lags == _LAGS_124
            and self.fused_lag124_moments
            and not self.skip_redundant_lag124_moment_fusion
        )
        radial_log_lag124_moments = (
            self.radial_log_lag124_moments
            and self.align_moments
            and not self.log_energy
            and not self.normalize_autocorrelation
            and self.moment_lags == _LAGS_124
        )
        self.fused_recurrence_moments_inference = (
            use_fused_recurrence_moments and canonical_moments and native_fused_moments
        )
        self.static_lag124_recurrence_moments_inference = (
            use_static_lag124_recurrence_moments
            and use_static_pole_recurrence_moments
            and canonical_moments
            and self.moment_lags == _LAGS_124
            and self.fused_lag124_moments
            and self.excitation_mixer is None
            and self.mode_gate_bias is None
            and self.mode_gate_gain is None
        )
        self.static_radial_log_lag124_recurrence_moments_inference = (
            use_static_pole_recurrence_moments
            and radial_log_lag124_moments
            and self.excitation_mixer is None
            and self.mode_gate_bias is None
            and self.mode_gate_gain is None
        )
        self.static_pole_recurrence_moments_inference = (
            use_static_pole_recurrence_moments
            and canonical_moments
            and (
                self.fused_recurrence_moments_inference
                or self.static_lag124_recurrence_moments_inference
            )
            and self.excitation_mixer is None
            and self.mode_gate_bias is None
            and self.mode_gate_gain is None
        )
        self.static_pole_block_scan_inference = (
            use_static_pole_block_scan
            and use_block_scan
            and canonical_moments
            and self.excitation_mixer is None
            and self.mode_gate_bias is None
            and self.mode_gate_gain is None
        )
        self.static_pole_block_scan_block_size_inference = (
            static_pole_block_scan_block_size if self.static_pole_block_scan_inference else 256
        )
        self.parallel_static_pole_recurrence_moments_inference = False
        self.parallel_static_pole_recurrence_num_warps_inference = 4
        use_static_backend = (
            self.static_pole_recurrence_moments_inference
            or self.static_radial_log_lag124_recurrence_moments_inference
            or self.static_pole_block_scan_inference
        )
        self.packed_static_recurrence_moments_inference = (
            use_packed_static_recurrence_moments and use_static_backend
        )
        self.packed_static_recurrence_drive_inference = (
            use_packed_static_recurrence_drive and use_static_backend
        )
        self.single_warp_static_recurrence_moments_inference = (
            use_single_warp_static_recurrence_moments and use_static_backend
        )
        if (
            self.static_pole_recurrence_moments_inference
            or self.static_radial_log_lag124_recurrence_moments_inference
            or self.static_pole_block_scan_inference
        ):
            damping = self.damping_min + (self.damping_max - self.damping_min) * torch.sigmoid(
                self.raw_decay
            )
            damping = self._apply_fixed_damping(damping)
            frequency = self.frequency_values()
            if self.impulse_injection:
                decay_real, decay_imag = pole_transition_real2d(damping, frequency, 1.0)
                gamma_real = torch.ones_like(decay_real)
                gamma_imag = torch.zeros_like(decay_imag)
            else:
                decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
                    damping,
                    frequency,
                    1.0,
                )
            frame_real, frame_imag = self.frame_matrix().chunk(2, dim=-1)
            if self.impulse_injection:
                drive_real, drive_imag = frame_real, frame_imag
            else:
                drive_real = frame_real * gamma_real - frame_imag * gamma_imag
                drive_imag = frame_imag * gamma_real + frame_real * gamma_imag
            self._buffers["inference_drive_frame"] = torch.cat(
                (drive_real, drive_imag),
                dim=-1,
            ).contiguous()
            self._buffers["inference_decay_real"] = decay_real.contiguous()
            self._buffers["inference_decay_imag"] = decay_imag.contiguous()
        else:
            self._buffers["inference_drive_frame"] = None
            self._buffers["inference_decay_real"] = None
            self._buffers["inference_decay_imag"] = None

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
        metadata_prevalidated: bool = False,
        return_modal_states: Literal[False] = False,
        return_inference_tail_components: Literal[False] = False,
        return_training_tail_components: Literal[False] = False,
        return_moments_only: Literal[False] = False,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
        metadata_prevalidated: bool = False,
        return_modal_states: Literal[True],
        return_inference_tail_components: Literal[False] = False,
        return_training_tail_components: Literal[False] = False,
        return_moments_only: Literal[False] = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
        metadata_prevalidated: bool = False,
        return_modal_states: Literal[False] = False,
        return_inference_tail_components: Literal[True],
        return_training_tail_components: Literal[False] = False,
        return_moments_only: Literal[False] = False,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
        metadata_prevalidated: bool = False,
        return_modal_states: Literal[False] = False,
        return_inference_tail_components: Literal[False] = False,
        return_training_tail_components: Literal[True],
        return_moments_only: Literal[False] = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
        metadata_prevalidated: bool = False,
        return_modal_states: Literal[False] = False,
        return_inference_tail_components: Literal[False] = False,
        return_training_tail_components: Literal[False] = False,
        return_moments_only: Literal[True],
    ) -> Tensor: ...

    def forward(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
        metadata_prevalidated: bool = False,
        return_modal_states: bool = False,
        return_inference_tail_components: bool = False,
        return_training_tail_components: bool = False,
        return_moments_only: bool = False,
    ) -> (
        Tensor
        | tuple[Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor, Tensor]
    ):
        return_modes = sum(
            (
                return_modal_states,
                return_inference_tail_components,
                return_training_tail_components,
                return_moments_only,
            )
        )
        if return_modes > 1:
            message = "modal-state, tail-component, and moments-only returns are mutually exclusive"
            raise ValueError(message)
        if return_inference_tail_components and (
            self.training
            or torch.is_grad_enabled()
            or time_delta is not None
            or observation_mask is not None
            or valid_mask is not None
        ):
            message = "tail components require metadata-free eval/no-grad inference"
            raise RuntimeError(message)
        if return_training_tail_components and (
            not self.training
            or not torch.is_grad_enabled()
            or time_delta is not None
            or observation_mask is not None
            or valid_mask is not None
        ):
            message = "training tail components require metadata-free train/grad execution"
            raise RuntimeError(message)
        packed_modal_coordinates: Tensor | None = None
        fused_static_drive: Tensor | None = None
        static_drive_frame = self._buffers.get("inference_drive_frame")
        use_fused_writer_drive = (
            self.fused_writer_rmsnorm_drive_inference
            and not self.training
            and not torch.is_grad_enabled()
            and self.direction == "forward"
            and self.use_input_norm
            and self.local is None
            and self.norm.weight is not None
            and static_drive_frame is not None
            and inputs.is_cuda
            and inputs.dtype == torch.float32
            and inputs.ndim == 3
            and inputs.shape[0] in (1, 32, 64)
            and 1 <= inputs.shape[1] <= 2048
            and inputs.shape[2] == 64
            and static_drive_frame.shape == (64, 32)
        )
        if use_fused_writer_drive:
            norm_weight = self.norm.weight
            if norm_weight is None or static_drive_frame is None:
                message = "fused writer drive requires prepared affine buffers"
                raise RuntimeError(message)
            normalized, fused_static_drive = writer_rmsnorm_drive_inference(
                inputs,
                norm_weight,
                static_drive_frame,
            )
        elif (
            self.use_input_norm
            and self.d32_rmsnorm_backward_training
            and self.training
            and torch.is_grad_enabled()
        ):
            if self.norm.weight is None:
                message = "block RMSNorm must use a learned weight"
                raise RuntimeError(message)
            from .pac_triton_d32_rmsnorm_backward_training import (  # noqa: PLC0415
                d32_rmsnorm_backward_training,
            )

            normalized = d32_rmsnorm_backward_training(
                inputs,
                self.norm.weight,
                eps=self.norm.eps,
            )
        elif self.use_input_norm:
            normalized = _dtype_aligned_rms_norm(inputs, self.norm)
        else:
            normalized = inputs
        local = (
            functional.silu(_directional_depthwise_conv(normalized, self.local, self.direction))
            if self.local is not None
            else normalized
        )
        # A matrix-exp parametrization is otherwise evaluated once for analysis and again
        # for synthesis. Reuse the exact same constrained frame within this block call.
        frame = self.frame_matrix()
        synthesis_frame = (
            frame if self.independent_synthesis_frame is None else self.independent_synthesis_frame
        )
        static_decay_real = self._buffers.get("inference_decay_real")
        static_decay_imag = self._buffers.get("inference_decay_imag")
        use_static_pole = (
            (
                self.static_pole_recurrence_moments_inference
                or self.static_radial_log_lag124_recurrence_moments_inference
                or self.static_pole_block_scan_inference
            )
            and not torch.is_grad_enabled()
            and time_delta is None
            and observation_mask is None
            and valid_mask is None
            and static_drive_frame is not None
            and static_decay_real is not None
            and static_decay_imag is not None
        )
        moment_direction: Direction = "forward" if self.align_moments else self.direction
        lag124_moments = self.moment_lags == _LAGS_124 and self.fused_lag124_moments
        native_fused_moments = self.moment_lags == _LAGS or (
            lag124_moments and (not self.skip_redundant_lag124_moment_fusion or not inputs.is_cuda)
        )
        moments_are_lag124 = False
        if use_static_pole:
            if static_drive_frame is None or static_decay_real is None or static_decay_imag is None:
                message = "static-pole inference buffers are not prepared"
                raise RuntimeError(message)
            drive = (
                fused_static_drive
                if fused_static_drive is not None
                else torch.matmul(local, static_drive_frame)
            )
            if self.static_radial_log_lag124_recurrence_moments_inference:
                if (
                    self.parallel_static_radial_log_recurrence_moments_inference
                    and return_moments_only
                    and self.direction == "forward"
                ):
                    return parallel_static_radial_log_recurrence_lag124_moments_only_inference(
                        static_decay_real,
                        static_decay_imag,
                        drive,
                        num_warps=4,
                    )
                if return_moments_only and self.direction == "forward":
                    return static_radial_log_recurrence_lag124_moments_only_inference(
                        static_decay_real,
                        static_decay_imag,
                        drive,
                        single_warp=self.single_warp_static_recurrence_moments_inference,
                    )
                if self.parallel_static_radial_log_recurrence_moments_inference:
                    packed_modal_coordinates, moments = (
                        parallel_static_radial_log_recurrence_lag124_moments_packed_io_inference(
                            static_decay_real,
                            static_decay_imag,
                            drive,
                            reverse=self.direction == "backward",
                            num_warps=4,
                        )
                    )
                else:
                    packed_modal_coordinates, moments = (
                        static_radial_log_recurrence_lag124_moments_packed_io_inference(
                            static_decay_real,
                            static_decay_imag,
                            drive,
                            reverse=self.direction == "backward",
                            single_warp=(self.single_warp_static_recurrence_moments_inference),
                        )
                    )
                states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
                moments_are_lag124 = True
            elif self.parallel_static_pole_recurrence_moments_inference:
                packed_modal_coordinates, moments = (
                    parallel_static_recurrence_lag124_moments_packed_io_training(
                        static_decay_real,
                        static_decay_imag,
                        drive,
                        reverse=self.direction == "backward",
                        num_warps=(self.parallel_static_pole_recurrence_num_warps_inference),
                    )
                )
                states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
                moments_are_lag124 = True
            elif self.static_lag124_recurrence_moments_inference:
                if return_moments_only and self.direction == "forward":
                    return static_recurrence_lag124_moments_only_inference(
                        static_decay_real,
                        static_decay_imag,
                        drive,
                        single_warp=self.single_warp_static_recurrence_moments_inference,
                    )
                packed_modal_coordinates, moments = (
                    static_recurrence_lag124_moments_packed_io_inference(
                        static_decay_real,
                        static_decay_imag,
                        drive,
                        reverse=self.direction == "backward",
                        single_warp=self.single_warp_static_recurrence_moments_inference,
                    )
                )
                states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
                moments_are_lag124 = True
            elif self.static_pole_block_scan_inference:
                block_scan_num_warps = (
                    1 if self.single_warp_static_recurrence_moments_inference else 4
                )
                if (
                    self.packed_static_recurrence_drive_inference
                    and self.packed_static_recurrence_moments_inference
                ):
                    packed_modal_coordinates = static_pole_block_scan_packed_io_recurrence(
                        static_decay_real,
                        static_decay_imag,
                        drive,
                        reverse=self.direction == "backward",
                        num_warps=block_scan_num_warps,
                        block_size=self.static_pole_block_scan_block_size_inference,
                    )
                    states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
                elif self.packed_static_recurrence_drive_inference:
                    states_real, states_imag = static_pole_block_scan_packed_input_recurrence(
                        static_decay_real,
                        static_decay_imag,
                        drive,
                        reverse=self.direction == "backward",
                        num_warps=block_scan_num_warps,
                        block_size=self.static_pole_block_scan_block_size_inference,
                    )
                elif self.packed_static_recurrence_moments_inference:
                    input_real, input_imag = drive.chunk(2, dim=-1)
                    packed_modal_coordinates = static_pole_block_scan_packed_output_recurrence(
                        static_decay_real,
                        static_decay_imag,
                        input_real,
                        input_imag,
                        reverse=self.direction == "backward",
                        num_warps=block_scan_num_warps,
                        block_size=self.static_pole_block_scan_block_size_inference,
                    )
                    states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
                else:
                    input_real, input_imag = drive.chunk(2, dim=-1)
                    states_real, states_imag = static_pole_block_scan_recurrence(
                        static_decay_real,
                        static_decay_imag,
                        input_real,
                        input_imag,
                        reverse=self.direction == "backward",
                        num_warps=block_scan_num_warps,
                        block_size=self.static_pole_block_scan_block_size_inference,
                    )
                if lag124_moments:
                    moments = lag124_modal_moments(
                        states_real,
                        states_imag,
                        physical_direction=moment_direction,
                    )
                    moments_are_lag124 = True
                else:
                    moments = online_modal_moments(
                        states_real,
                        states_imag,
                        physical_direction=moment_direction,
                        fused_backward=False,
                    )
            elif (
                self.packed_static_recurrence_drive_inference
                and self.packed_static_recurrence_moments_inference
            ):
                packed_modal_coordinates, moments = static_recurrence_moments_packed_io_inference(
                    static_decay_real,
                    static_decay_imag,
                    drive,
                    reverse=self.direction == "backward",
                    single_warp=self.single_warp_static_recurrence_moments_inference,
                )
                states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
            elif self.packed_static_recurrence_drive_inference:
                states_real, states_imag, moments = (
                    static_recurrence_moments_packed_input_inference(
                        static_decay_real,
                        static_decay_imag,
                        drive,
                        reverse=self.direction == "backward",
                        single_warp=self.single_warp_static_recurrence_moments_inference,
                    )
                )
            elif self.packed_static_recurrence_moments_inference:
                input_real, input_imag = drive.chunk(2, dim=-1)
                packed_modal_coordinates, moments = static_recurrence_moments_packed_inference(
                    static_decay_real,
                    static_decay_imag,
                    input_real,
                    input_imag,
                    reverse=self.direction == "backward",
                    single_warp=self.single_warp_static_recurrence_moments_inference,
                )
                states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
            else:
                input_real, input_imag = drive.chunk(2, dim=-1)
                states_real, states_imag, moments = static_recurrence_moments_inference(
                    static_decay_real,
                    static_decay_imag,
                    input_real,
                    input_imag,
                    reverse=self.direction == "backward",
                    single_warp=self.single_warp_static_recurrence_moments_inference,
                )
            mode_gate: Tensor | float = 1.0
        else:
            excitation = torch.matmul(local, frame)
            excitation_real, excitation_imag = excitation.chunk(2, dim=-1)
            if self.excitation_mixer is not None:
                mixed_excitation = self.excitation_mixer(excitation_real, excitation_imag)
                if not isinstance(mixed_excitation, tuple) or len(mixed_excitation) != 2:
                    message = "excitation mixer must return real and imaginary tensors"
                    raise RuntimeError(message)
                excitation_real, excitation_imag = mixed_excitation
            needs_invariant_energy = not self.canonical_identity_elision or (
                self.mode_gate_bias is not None and self.mode_gate_gain is not None
            )
            invariant_energy = (
                torch.log1p(excitation_real.square() + excitation_imag.square())
                if needs_invariant_energy
                else None
            )
            active_delta, active_mask = _modal_time_inputs(
                inputs,
                time_delta,
                observation_mask,
                direction=self.direction,
                validate_values=not metadata_prevalidated,
            )
            active_valid_mask = _modal_mask(
                inputs,
                valid_mask,
                name="valid_mask",
                validate_values=not metadata_prevalidated,
            )
            use_fused_excitation_recurrence_training = (
                self.fused_excitation_recurrence_training
                and not self.impulse_injection
                and torch.is_grad_enabled()
                and self.initial_state_real is None
                and self.initial_state_imag is None
                and time_delta is None
                and active_mask is None
                and active_valid_mask is None
                and self.excitation_mixer is None
                and self.mode_gate_bias is None
                and self.mode_gate_gain is None
                and self.frequency_bound == float(torch.pi)
                and self.fused_recurrence_moments_backward_training
                and self.recurrence_backend == "auto"
                and self.log_energy
                and self.normalize_autocorrelation
                and native_fused_moments
                and supports_fused_excitation_recurrence(
                    excitation_real,
                    excitation_imag,
                    self.raw_decay,
                    self.raw_frequency,
                )
            )
            if use_fused_excitation_recurrence_training:
                states_real, states_imag, moments = fused_excitation_recurrence_moments_training(
                    excitation_real,
                    excitation_imag,
                    self.raw_decay,
                    self.raw_frequency,
                    damping_min=self.damping_min,
                    damping_max=self.damping_max,
                    recurrence_reverse=self.direction == "backward",
                    moment_direction=moment_direction,
                )
                mode_gate = 1.0
            else:
                use_mode_static_pole_training = (
                    self.mode_static_pole_training
                    and torch.is_grad_enabled()
                    and time_delta is None
                )
                damping = self.damping_values().view(1, 1, -1)
                frequency = self.frequency_values().view(1, 1, -1)
                if self.impulse_injection:
                    decay_real, decay_imag = pole_transition_real2d(
                        damping,
                        frequency,
                        1.0 if time_delta is None else active_delta,
                    )
                    decay_real = decay_real.expand_as(excitation_real)
                    decay_imag = decay_imag.expand_as(excitation_imag)
                    gamma_real = torch.ones_like(decay_real)
                    gamma_imag = torch.zeros_like(decay_imag)
                elif time_delta is None:
                    pole_function = (
                        compiled_discrete_pole_real2d
                        if use_mode_static_pole_training
                        else discrete_pole_real2d
                    )
                    static_pole = pole_function(
                        damping,
                        frequency,
                        1.0,
                    )
                    decay_real, decay_imag, gamma_real, gamma_imag = (
                        value.expand_as(excitation_real) for value in static_pole
                    )
                else:
                    decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
                        damping,
                        frequency,
                        active_delta,
                    )
                if self.impulse_injection:
                    input_real, input_imag = excitation_real, excitation_imag
                else:
                    input_real = gamma_real * excitation_real - gamma_imag * excitation_imag
                    input_imag = gamma_real * excitation_imag + gamma_imag * excitation_real
                if active_mask is not None:
                    input_real = active_mask * input_real
                    input_imag = active_mask * input_imag
                if self.initial_state_real is not None or self.initial_state_imag is not None:
                    if self.initial_state_real is None or self.initial_state_imag is None:
                        message = "real and imaginary initial states must be enabled together"
                        raise RuntimeError(message)
                    if active_valid_mask is not None:
                        message = (
                            "trainable initial-state screening requires unpadded fixed-length "
                            "sequences"
                        )
                        raise RuntimeError(message)
                    boundary_index = -1 if self.direction == "backward" else 0
                    decay_index = (
                        boundary_index if decay_real.shape[1] == input_real.shape[1] else 0
                    )
                    boundary_decay_real = decay_real[:, decay_index, :]
                    boundary_decay_imag = decay_imag[:, decay_index, :]
                    initial_real = self.initial_state_real.view(1, -1)
                    initial_imag = self.initial_state_imag.view(1, -1)
                    boundary_real = (
                        boundary_decay_real * initial_real - boundary_decay_imag * initial_imag
                    )
                    boundary_imag = (
                        boundary_decay_imag * initial_real + boundary_decay_real * initial_imag
                    )
                    zero_real = torch.zeros_like(input_real)
                    zero_imag = torch.zeros_like(input_imag)
                    if self.direction == "backward":
                        boundary_drive_real = torch.cat(
                            (zero_real[:, :-1], boundary_real.unsqueeze(1)),
                            dim=1,
                        )
                        boundary_drive_imag = torch.cat(
                            (zero_imag[:, :-1], boundary_imag.unsqueeze(1)),
                            dim=1,
                        )
                    else:
                        boundary_drive_real = torch.cat(
                            (boundary_real.unsqueeze(1), zero_real[:, 1:]),
                            dim=1,
                        )
                        boundary_drive_imag = torch.cat(
                            (boundary_imag.unsqueeze(1), zero_imag[:, 1:]),
                            dim=1,
                        )
                    input_real = input_real + boundary_drive_real
                    input_imag = input_imag + boundary_drive_imag
                if self.mode_gate_bias is None or self.mode_gate_gain is None:
                    mode_gate = 1.0
                else:
                    if invariant_energy is None:
                        message = "mode gating requires invariant energy"
                        raise RuntimeError(message)
                    mode_gate = self.gate_max * torch.sigmoid(
                        self.mode_gate_bias.view(1, 1, -1)
                        + self.mode_gate_gain.view(1, 1, -1) * invariant_energy
                    )
                use_fused_recurrence_moments_training = (
                    self.fused_recurrence_moments_backward_training
                    and torch.is_grad_enabled()
                    and active_valid_mask is None
                    and self.recurrence_backend == "auto"
                    and self.log_energy
                    and self.normalize_autocorrelation
                    and native_fused_moments
                )
                use_static_lag124_training = (
                    self.static_lag124_recurrence_moments_training
                    and use_mode_static_pole_training
                    and active_mask is None
                    and active_valid_mask is None
                    and lag124_moments
                    and moment_direction == "forward"
                    and self.log_energy
                    and self.normalize_autocorrelation
                    and self.canonical_identity_elision
                    and not isinstance(mode_gate, Tensor)
                )
                use_parallel_static_training = (
                    (
                        (
                            self.parallel_static_recurrence_training
                            and self.log_energy
                            and self.normalize_autocorrelation
                        )
                        or (
                            self.parallel_static_radial_log_recurrence_moments_training
                            and self.radial_log_lag124_moments
                            and not self.log_energy
                            and not self.normalize_autocorrelation
                        )
                    )
                    and use_mode_static_pole_training
                    and active_mask is None
                    and active_valid_mask is None
                    and self.moment_lags == _LAGS_124
                    and self.canonical_identity_elision
                    and not isinstance(mode_gate, Tensor)
                    and moment_direction == "forward"
                    and excitation_real.shape[1]
                    <= (
                        2048
                        if self.parallel_static_radial_log_recurrence_moments_training
                        else 1024
                    )
                )
                use_parallel_static_excitation_training = (
                    use_parallel_static_training
                    and self.parallel_static_excitation_recurrence_training
                    and not self.impulse_injection
                )
                use_fused_recurrence_moments = (
                    self.fused_recurrence_moments_inference
                    and not torch.is_grad_enabled()
                    and active_valid_mask is None
                    and native_fused_moments
                )
                if use_parallel_static_excitation_training:
                    if return_moments_only:
                        return parallel_static_excitation_recurrence_lag124_moments_only_training(
                            decay_real[0, 0],
                            decay_imag[0, 0],
                            gamma_real[0, 0],
                            gamma_imag[0, 0],
                            excitation_real,
                            excitation_imag,
                            reverse=self.direction == "backward",
                            num_warps=4,
                        )
                    packed_modal_coordinates, moments = (
                        parallel_static_excitation_recurrence_lag124_moments_packed_io_training(
                            decay_real[0, 0],
                            decay_imag[0, 0],
                            gamma_real[0, 0],
                            gamma_imag[0, 0],
                            excitation_real,
                            excitation_imag,
                            reverse=self.direction == "backward",
                            num_warps=8,
                        )
                    )
                    states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
                    moments_are_lag124 = True
                elif use_parallel_static_training:
                    packed_input = torch.cat((input_real, input_imag), dim=-1)
                    if self.parallel_static_radial_log_recurrence_moments_training:
                        if return_moments_only:
                            return (
                                parallel_static_radial_log_recurrence_lag124_moments_only_training(
                                    decay_real[0, 0],
                                    decay_imag[0, 0],
                                    packed_input,
                                    reverse=self.direction == "backward",
                                    num_warps=4,
                                )
                            )
                        packed_modal_coordinates, moments = (
                            parallel_static_radial_log_recurrence_lag124_moments_packed_io_training(
                                decay_real[0, 0],
                                decay_imag[0, 0],
                                packed_input,
                                reverse=self.direction == "backward",
                                num_warps=8,
                            )
                        )
                    else:
                        if return_moments_only:
                            return parallel_static_recurrence_lag124_moments_only_training(
                                decay_real[0, 0],
                                decay_imag[0, 0],
                                packed_input,
                                reverse=self.direction == "backward",
                                num_warps=4,
                            )
                        packed_modal_coordinates, moments = (
                            parallel_static_recurrence_lag124_moments_packed_io_training(
                                decay_real[0, 0],
                                decay_imag[0, 0],
                                packed_input,
                                reverse=self.direction == "backward",
                                num_warps=8,
                            )
                        )
                    states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
                    moments_are_lag124 = True
                elif use_static_lag124_training:
                    packed_input = torch.cat((input_real, input_imag), dim=-1)
                    if return_moments_only:
                        if self.saved_state_lag124_reader_training:
                            return static_recurrence_lag124_moments_only_saved_states_training(
                                decay_real,
                                decay_imag,
                                packed_input,
                                reverse=self.direction == "backward",
                            )
                        return static_recurrence_lag124_moments_only_training(
                            decay_real,
                            decay_imag,
                            packed_input,
                            reverse=self.direction == "backward",
                        )
                    packed_modal_coordinates, moments = (
                        static_recurrence_lag124_moments_packed_io_training(
                            decay_real,
                            decay_imag,
                            packed_input,
                            reverse=self.direction == "backward",
                        )
                    )
                    states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
                    moments_are_lag124 = True
                elif use_fused_recurrence_moments_training:
                    use_packed_by_shape = input_real.shape[0] > 1 and input_real.shape[1] <= 128
                    use_packed_training = (
                        (
                            use_packed_by_shape
                            if self.packed_recurrence_moments_training is None
                            else self.packed_recurrence_moments_training
                        )
                        and self.canonical_identity_elision
                        and not isinstance(mode_gate, Tensor)
                    )
                    if use_packed_training:
                        packed_modal_coordinates, moments = (
                            fused_recurrence_moments_packed_training(
                                decay_real,
                                decay_imag,
                                input_real,
                                input_imag,
                                recurrence_reverse=self.direction == "backward",
                                moment_direction=moment_direction,
                                use_two_pass_reverse=(
                                    self.two_pass_reverse_recurrence_moments_training
                                ),
                            )
                        )
                        states_real, states_imag = packed_modal_coordinates.chunk(2, dim=-1)
                    else:
                        states_real, states_imag, moments = fused_recurrence_moments_training(
                            decay_real,
                            decay_imag,
                            input_real,
                            input_imag,
                            recurrence_reverse=self.direction == "backward",
                            moment_direction=moment_direction,
                            use_two_pass_reverse=(
                                self.two_pass_reverse_recurrence_moments_training
                            ),
                        )
                elif use_fused_recurrence_moments:
                    states_real, states_imag, moments = recurrence_moments_inference(
                        decay_real,
                        decay_imag,
                        input_real,
                        input_imag,
                        reverse=self.direction == "backward",
                    )
                else:
                    states_real, states_imag = recurrence_real2d_directional(
                        decay_real,
                        decay_imag,
                        input_real,
                        input_imag,
                        self.recurrence_backend,
                        self.direction,
                    )
                    moment_variant = _MomentVariant(
                        moment_direction,
                        self.log_energy,
                        self.normalize_autocorrelation,
                        self.moment_lags,
                    )
                    if self.physical_time_lag_moments and isinstance(active_delta, Tensor):
                        moments = _physical_time_modal_moments(
                            states_real,
                            states_imag,
                            active_delta,
                            active_valid_mask,
                            moment_variant,
                        )
                    elif active_valid_mask is not None:
                        moments = _masked_modal_moments(
                            states_real,
                            states_imag,
                            active_valid_mask,
                            moment_variant,
                        )
                    elif (
                        self.log_energy and self.normalize_autocorrelation and native_fused_moments
                    ):
                        moments = online_modal_moments(
                            states_real,
                            states_imag,
                            physical_direction=moment_direction,
                            fused_backward=self.fused_moments_backward_training,
                        )
                    elif self.log_energy and self.normalize_autocorrelation and lag124_moments:
                        moments = lag124_modal_moments(
                            states_real,
                            states_imag,
                            physical_direction=moment_direction,
                        )
                        moments_are_lag124 = True
                    else:
                        moments = _modal_moments(
                            states_real,
                            states_imag,
                            moment_variant,
                        )
        if (
            self.moment_lags == _LAGS_124
            and self.fused_lag124_moments
            and not (self.physical_time_lag_moments and time_delta is not None)
            and valid_mask is None
            and not moments_are_lag124
        ):
            moments = lag124_modal_moments(
                states_real,
                states_imag,
                physical_direction=moment_direction,
            )
        if return_moments_only:
            return moments
        if packed_modal_coordinates is not None:
            modal_coordinates = packed_modal_coordinates
        elif isinstance(mode_gate, Tensor) or not self.canonical_identity_elision:
            modal_coordinates = torch.cat(
                (mode_gate * states_real, mode_gate * states_imag),
                dim=-1,
            )
        else:
            modal_coordinates = torch.cat((states_real, states_imag), dim=-1)
        if return_inference_tail_components:
            return modal_coordinates.contiguous(), local.contiguous(), moments
        if return_training_tail_components:
            return (
                modal_coordinates.contiguous(),
                local.contiguous(),
                moments,
                synthesis_frame,
            )
        modal = torch.matmul(modal_coordinates, synthesis_frame.transpose(0, 1))
        if self.synthesis_scale != 1.0 or not self.canonical_identity_elision:
            modal = self.synthesis_scale * modal
        if self.split_residual_scales:
            if self.modal_scale is None or self.local_scale is None:
                message = "split residual scales are not initialized"
                raise RuntimeError(message)
            output = (
                inputs
                + self.modal_scale.view(1, 1, -1) * modal
                + self.local_scale.view(1, 1, -1) * local
            )
        else:
            if self.direct_scale is None or self.layer_scale is None:
                message = "coupled residual scales are not initialized"
                raise RuntimeError(message)
            update = modal + self.direct_scale.view(1, 1, -1) * local
            output = inputs + self.layer_scale.view(1, 1, -1) * update
        if return_modal_states:
            return output, moments, states_real, states_imag
        return output, moments


class TightFrameSequenceRegressor(nn.Module):
    """Causal PAC-TF sequence regressor used for mechanism-recovery experiments."""

    def __init__(
        self,
        config: PACExperimentConfig,
        variant: StiefelVariant,
    ) -> None:
        super().__init__()
        if config.model_dim < 4:
            raise TightFrameConfigError(config.model_dim)
        self.input_projection = nn.Linear(config.raw_input_dim, config.model_dim)
        self.block1 = _TightFrameBlock(
            config.model_dim, config.modes, _BlockVariant("forward", variant)
        )
        self.block2 = _TightFrameBlock(
            config.model_dim, config.modes, _BlockVariant("forward", variant)
        )
        self.final_norm = nn.RMSNorm(config.model_dim)
        self.output_projection = nn.Linear(config.model_dim, config.output_dim)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        features = functional.silu(self.input_projection(inputs))
        features, _ = self.block1(
            features,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        features, _ = self.block2(
            features,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        return self.output_projection(_dtype_aligned_rms_norm(features, self.final_norm))

    def frequency_values(self) -> Tensor:
        return self.block1.frequency_values()

    def damping_values(self) -> Tensor:
        return self.block1.damping_values()

    def post_optimizer_step(self) -> None:
        self.block1.retract_frame()
        self.block2.retract_frame()

    def finalize_constraints(self) -> None:
        self.block1.finalize_frame()
        self.block2.finalize_frame()


class _CausalStem(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        *,
        kernel_size: int = 9,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_dim, model_dim, kernel_size, stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        padded = functional.pad(inputs.transpose(1, 2), (self.conv.kernel_size[0] - 1, 0))
        return functional.silu(self.conv(padded).transpose(1, 2))


class _InvariantMomentHead(nn.Module):
    def __init__(
        self,
        pooled_dim: int,
        modes: int,
        class_count: int,
        *,
        use_modal_moments: bool,
        use_backward_moments: bool,
        lags: tuple[int, ...] = _LAGS,
    ) -> None:
        super().__init__()
        self.use_modal_moments = use_modal_moments
        self.use_backward_moments = use_backward_moments
        moment_dim = modes * (1 + 2 * len(lags))
        moment_copies = 1 + int(use_backward_moments)
        input_dim = pooled_dim + moment_copies * moment_dim if use_modal_moments else pooled_dim
        self.classifier = nn.Linear(input_dim, class_count)

    def forward(self, inputs: Tensor, forward_moments: Tensor, backward_moments: Tensor) -> Tensor:
        if not self.use_modal_moments:
            features = inputs
        elif self.use_backward_moments:
            features = torch.cat((inputs, forward_moments, backward_moments), dim=-1)
        else:
            features = torch.cat((inputs, forward_moments), dim=-1)
        return self.classifier(features)


def modal_moments(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    normalized: bool = False,
    physical_direction: Direction = "forward",
) -> Tensor:
    return _modal_moments(
        states_real,
        states_imag,
        _MomentVariant(physical_direction, normalized, normalized),
    )


def _modal_moments(states_real: Tensor, states_imag: Tensor, variant: _MomentVariant) -> Tensor:
    states_real = _orient(states_real, variant.physical_direction)
    states_imag = _orient(states_imag, variant.physical_direction)
    energy = (states_real.square() + states_imag.square()).mean(dim=1)
    moments = [torch.log1p(energy) if variant.log_energy else energy]
    for lag in variant.lags:
        if states_real.shape[1] <= lag:
            zeros = states_real.new_zeros(states_real.shape[0], states_real.shape[2])
            moments.extend((zeros, zeros))
            continue
        current_real = states_real[:, lag:]
        current_imag = states_imag[:, lag:]
        previous_real = states_real[:, :-lag]
        previous_imag = states_imag[:, :-lag]
        correlation_real = (current_real * previous_real + current_imag * previous_imag).mean(dim=1)
        correlation_imag = (current_imag * previous_real - current_real * previous_imag).mean(dim=1)
        if variant.normalize_autocorrelation:
            current_energy = (current_real.square() + current_imag.square()).mean(dim=1)
            previous_energy = (previous_real.square() + previous_imag.square()).mean(dim=1)
            denominator = torch.sqrt(
                (current_energy * previous_energy).clamp_min(_EPSILON * _EPSILON)
            )
            correlation_real = correlation_real / denominator
            correlation_imag = correlation_imag / denominator
        moments.extend((correlation_real, correlation_imag))
    return torch.cat(moments, dim=-1)


def _masked_modal_moments(
    states_real: Tensor,
    states_imag: Tensor,
    observation_mask: Tensor,
    variant: _MomentVariant,
) -> Tensor:
    states_real = _orient(states_real, variant.physical_direction)
    states_imag = _orient(states_imag, variant.physical_direction)
    mask = _orient(observation_mask, variant.physical_direction).to(dtype=states_real.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    energy = ((states_real.square() + states_imag.square()) * mask).sum(dim=1) / count
    moments = [torch.log1p(energy) if variant.log_energy else energy]
    for lag in variant.lags:
        if states_real.shape[1] <= lag:
            zeros = states_real.new_zeros(states_real.shape[0], states_real.shape[2])
            moments.extend((zeros, zeros))
            continue
        valid = mask[:, lag:] * mask[:, :-lag]
        valid_count = valid.sum(dim=1).clamp_min(1.0)
        current_real = states_real[:, lag:]
        current_imag = states_imag[:, lag:]
        previous_real = states_real[:, :-lag]
        previous_imag = states_imag[:, :-lag]
        correlation_real = (
            (current_real * previous_real + current_imag * previous_imag) * valid
        ).sum(dim=1) / valid_count
        correlation_imag = (
            (current_imag * previous_real - current_real * previous_imag) * valid
        ).sum(dim=1) / valid_count
        if variant.normalize_autocorrelation:
            current_energy = ((current_real.square() + current_imag.square()) * valid).sum(
                dim=1
            ) / valid_count
            previous_energy = ((previous_real.square() + previous_imag.square()) * valid).sum(
                dim=1
            ) / valid_count
            denominator = torch.sqrt(
                (current_energy * previous_energy).clamp_min(_EPSILON * _EPSILON)
            )
            correlation_real = correlation_real / denominator
            correlation_imag = correlation_imag / denominator
        moments.extend((correlation_real, correlation_imag))
    return torch.cat(moments, dim=-1)


def _interpolate_modal_state(
    states_real: Tensor,
    states_imag: Tensor,
    timestamps: Tensor,
    valid: Tensor,
    queries: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Interpolate modal states and report whether each query has valid support."""
    upper_unclamped = torch.searchsorted(
        timestamps.contiguous(),
        queries.contiguous(),
        right=False,
    )
    upper = upper_unclamped.clamp(max=states_real.shape[1] - 1)
    lower = (upper_unclamped - 1).clamp(min=0, max=states_real.shape[1] - 1)
    batch = torch.arange(states_real.shape[0], device=states_real.device).view(-1, 1)
    lower_time = timestamps[batch, lower]
    upper_time = timestamps[batch, upper]
    span = upper_time - lower_time
    interpolation = torch.where(
        span > 0,
        (queries - lower_time) / span.clamp_min(torch.finfo(states_real.dtype).tiny),
        torch.zeros_like(queries),
    ).clamp(0.0, 1.0)
    previous_real = torch.lerp(
        states_real[batch, lower],
        states_real[batch, upper],
        interpolation.unsqueeze(-1),
    )
    previous_imag = torch.lerp(
        states_imag[batch, lower],
        states_imag[batch, upper],
        interpolation.unsqueeze(-1),
    )
    lower_valid = valid.squeeze(-1)[batch, lower]
    upper_valid = valid.squeeze(-1)[batch, upper]
    epsilon = 8.0 * torch.finfo(states_real.dtype).eps
    support_valid = torch.where(
        interpolation <= epsilon,
        lower_valid,
        torch.where(
            interpolation >= 1.0 - epsilon,
            upper_valid,
            lower_valid * upper_valid,
        ),
    )
    in_bounds = (queries >= timestamps[:, :1]) & (queries <= timestamps[:, -1:])
    return previous_real, previous_imag, support_valid * in_bounds.to(dtype=states_real.dtype)


def _physical_time_modal_moments(
    states_real: Tensor,
    states_imag: Tensor,
    time_delta: Tensor,
    valid_mask: Tensor | None,
    variant: _MomentVariant,
) -> Tensor:
    """Estimate modal autocorrelation at fixed physical-time lags.

    States are treated as a piecewise-linear path and integrated with interval
    widths. On a unit grid this reduces to the existing token-lag definition.
    Lag values therefore use the same normalized time unit as ``time_delta``.
    """
    states_real = _orient(states_real, variant.physical_direction)
    states_imag = _orient(states_imag, variant.physical_direction)
    delta = _orient(time_delta, variant.physical_direction).to(dtype=states_real.dtype)
    if delta.ndim == 2:
        delta = delta.unsqueeze(-1)
    if delta.shape != (*states_real.shape[:2], 1):
        message = "physical-time moments require time_delta with shape [B,N,1]"
        raise ValueError(message)

    if valid_mask is None:
        valid = torch.ones_like(delta)
    else:
        valid = _orient(valid_mask, variant.physical_direction).to(dtype=states_real.dtype)
        if valid.ndim == 2:
            valid = valid.unsqueeze(-1)
    timestamps = delta.squeeze(-1).cumsum(dim=1)

    duration_weight = delta * valid
    duration_count = duration_weight.sum(dim=1)
    # A zero-duration record has no physical integral. Retain a finite and useful
    # R0 by falling back to the valid-event mean; positive-duration records use
    # the physical quadrature exclusively.
    event_count = valid.sum(dim=1)
    use_duration = duration_count > 0
    energy_weight = torch.where(use_duration.unsqueeze(1), duration_weight, valid)
    energy_count = torch.where(use_duration, duration_count, event_count).clamp_min(1.0)
    energy = ((states_real.square() + states_imag.square()) * energy_weight).sum(
        dim=1
    ) / energy_count
    moments = [torch.log1p(energy) if variant.log_energy else energy]

    for lag in variant.lags:
        queries = timestamps - float(lag)
        previous_real, previous_imag, support_valid = _interpolate_modal_state(
            states_real,
            states_imag,
            timestamps,
            valid,
            queries,
        )
        pair_weight = delta.squeeze(-1) * valid.squeeze(-1) * support_valid
        pair_count = pair_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        current_real = states_real
        current_imag = states_imag
        correlation_real = (
            (current_real * previous_real + current_imag * previous_imag)
            * pair_weight.unsqueeze(-1)
        ).sum(dim=1) / pair_count
        correlation_imag = (
            (current_imag * previous_real - current_real * previous_imag)
            * pair_weight.unsqueeze(-1)
        ).sum(dim=1) / pair_count
        if variant.normalize_autocorrelation:
            current_energy = (
                (current_real.square() + current_imag.square()) * pair_weight.unsqueeze(-1)
            ).sum(dim=1) / pair_count
            previous_energy = (
                (previous_real.square() + previous_imag.square()) * pair_weight.unsqueeze(-1)
            ).sum(dim=1) / pair_count
            denominator = torch.sqrt(
                (current_energy * previous_energy).clamp_min(_EPSILON * _EPSILON)
            )
            correlation_real = correlation_real / denominator
            correlation_imag = correlation_imag / denominator
        moments.extend((correlation_real, correlation_imag))
    return torch.cat(moments, dim=-1)


@torch.no_grad()
def _retract_columns(weight: Tensor) -> None:
    frame, upper = torch.linalg.qr(weight, mode="reduced")
    diagonal = torch.diagonal(upper)
    signs = torch.where(diagonal >= 0.0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
    weight.copy_(frame * signs.unsqueeze(0))


def _independent_synthesis_frame(model_dim: int, modes: int) -> Tensor:
    frame = torch.empty(model_dim, 2 * modes)
    nn.init.orthogonal_(frame)
    return frame


def _dtype_aligned_rms_norm(inputs: Tensor, norm: nn.RMSNorm) -> Tensor:
    weight = norm.weight
    if weight is None or weight.dtype == inputs.dtype:
        return norm(inputs)
    return functional.rms_norm(
        inputs,
        norm.normalized_shape,
        weight.to(dtype=inputs.dtype),
        norm.eps,
    )


def _ordered_pool(inputs: Tensor, scales: tuple[int, ...]) -> Tensor:
    empty = inputs.new_zeros(inputs.shape[0], inputs.shape[2])
    summaries = [
        chunk.mean(dim=1) if chunk.shape[1] else empty
        for scale in scales
        for chunk in torch.tensor_split(inputs, scale, dim=1)
    ]
    return torch.cat(summaries, dim=-1)


def _masked_ordered_pool(
    inputs: Tensor,
    valid_mask: Tensor,
    scales: tuple[int, ...],
) -> Tensor:
    empty = inputs.new_zeros(inputs.shape[0], inputs.shape[2])
    summaries: list[Tensor] = []
    for scale in scales:
        input_chunks = torch.tensor_split(inputs, scale, dim=1)
        mask_chunks = torch.tensor_split(valid_mask, scale, dim=1)
        for chunk, mask in zip(input_chunks, mask_chunks, strict=True):
            if not chunk.shape[1]:
                summaries.append(empty)
                continue
            weight = mask.to(dtype=chunk.dtype)
            summaries.append((chunk * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0))
    return torch.cat(summaries, dim=-1)


def _masked_sequence_mean(inputs: Tensor, valid_mask: Tensor | None) -> Tensor:
    if valid_mask is None:
        return inputs.mean(dim=1)
    weight = valid_mask.to(dtype=inputs.dtype)
    return (inputs * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _raw_metadata_mask(inputs: Tensor, mask: Tensor | None, name: str) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim not in (2, 3):
        message = f"{name} must have shape [B,N] or [B,N,1]"
        raise ValueError(message)
    if mask.shape[1] != inputs.shape[1]:
        return None
    return _modal_mask(inputs, mask, name=name)


def _prepare_stem_metadata(
    values: Tensor | None,
    raw_inputs: Tensor,
    stem_features: Tensor,
    stride: int,
    reduction: Literal["sum", "max"],
    name: str,
) -> Tensor | None:
    if values is None:
        return None
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    if values.ndim != 3 or values.shape[0] != raw_inputs.shape[0] or values.shape[-1] != 1:
        message = f"{name} must have shape [B,N] or [B,N,1]"
        raise ValueError(message)
    if values.shape[1] == stem_features.shape[1]:
        return values
    if values.shape[1] != raw_inputs.shape[1]:
        message = f"{name} length must match raw inputs or recurrent tokens"
        raise ValueError(message)
    return _causal_stem_metadata(values, stride, reduction)


def _causal_stem_metadata(
    values: Tensor,
    stride: int,
    reduction: Literal["sum", "max"],
) -> Tensor:
    if stride < 1:
        message = "stem stride must be positive"
        raise ValueError(message)
    if values.dtype == torch.bool:
        values = values.to(dtype=torch.float32)
    output_count = (values.shape[1] + stride - 1) // stride
    first = values[:, :1]
    if output_count == 1:
        return first
    tail_end = 1 + (output_count - 1) * stride
    groups = values[:, 1:tail_end].reshape(
        values.shape[0],
        output_count - 1,
        stride,
        values.shape[2],
    )
    reduced = groups.sum(dim=2) if reduction == "sum" else groups.amax(dim=2)
    return torch.cat((first, reduced), dim=1)


def _orient(inputs: Tensor, direction: Direction) -> Tensor:
    match direction:
        case "forward":
            return inputs
        case "backward":
            return torch.flip(inputs, dims=(1,))
        case unreachable:
            assert_never(unreachable)


def _modal_time_inputs(
    inputs: Tensor,
    time_delta: Tensor | None,
    observation_mask: Tensor | None,
    *,
    direction: Direction,
    validate_values: bool = True,
) -> tuple[Tensor | float, Tensor | None]:
    if time_delta is None:
        active_delta: Tensor | float = 1.0
    else:
        if time_delta.shape not in (inputs.shape[:2], (*inputs.shape[:2], 1)):
            message = "time_delta must have shape [B,N] or [B,N,1]"
            raise ValueError(message)
        active_delta = time_delta.to(device=inputs.device, dtype=inputs.dtype)
        if active_delta.ndim == 2:
            active_delta = active_delta.unsqueeze(-1)
        if validate_values and (
            not torch.isfinite(active_delta).all() or bool((active_delta < 0).any())
        ):
            message = "time_delta must contain finite non-negative values"
            raise ValueError(message)
        if direction == "backward" and active_delta.shape[1] > 1:
            active_delta = torch.cat((active_delta[:, 1:], active_delta[:, -1:]), dim=1)

    active_mask = _modal_mask(
        inputs,
        observation_mask,
        name="observation_mask",
        validate_values=validate_values,
    )
    return active_delta, active_mask


def _modal_mask(
    inputs: Tensor,
    mask: Tensor | None,
    *,
    name: str,
    validate_values: bool = True,
) -> Tensor | None:
    if mask is None:
        return None
    if mask.shape not in (inputs.shape[:2], (*inputs.shape[:2], 1)):
        message = f"{name} must have shape [B,N] or [B,N,1]"
        raise ValueError(message)
    active = mask.to(device=inputs.device, dtype=inputs.dtype)
    if active.ndim == 2:
        active = active.unsqueeze(-1)
    if validate_values and (
        not torch.isfinite(active).all() or bool(((active < 0) | (active > 1)).any())
    ):
        message = f"{name} must contain finite values in [0,1]"
        raise ValueError(message)
    return active


def _directional_depthwise_conv(
    inputs: Tensor, convolution: nn.Conv1d, direction: Direction
) -> Tensor:
    channels_first = inputs.transpose(1, 2)
    padding = convolution.kernel_size[0] - 1
    match direction:
        case "forward":
            return convolution(functional.pad(channels_first, (padding, 0))).transpose(1, 2)
        case "backward":
            output = functional.conv1d(
                functional.pad(channels_first, (0, padding)),
                convolution.weight.flip(-1),
                convolution.bias,
                convolution.stride,
                0,
                convolution.dilation,
                convolution.groups,
            )
            return output.transpose(1, 2)
        case unreachable:
            assert_never(unreachable)


def build_tight_frame_classifier(
    name: str, config: PACExperimentConfig, class_count: int
) -> nn.Module | None:
    variant = variant_for_model(name)
    if variant is None:
        return None
    capacity = capacity_for_model(name)
    active_config = (
        config if capacity is None else replace(config, model_dim=capacity[0], modes=capacity[1])
    )
    return TightFrameClassifier(
        active_config,
        class_count,
        variant,
        full_modal_frame=uses_full_modal_frame(name),
    )


def build_tight_frame_regressor(name: str, config: PACExperimentConfig) -> nn.Module | None:
    source_name = {
        "pac_tf": "pac_stiefel_depth2_norm_autocorr_d64_m16",
        "pac_tf_fixed_damping": "pac_stiefel_depth2_norm_autocorr_d64_m16",
    }.get(name)
    if source_name is None:
        return None
    variant = variant_for_model(source_name)
    capacity = capacity_for_model(source_name)
    if variant is None or capacity is None:
        message = f"missing PAC-TF sequence-regression variant: {source_name}"
        raise RuntimeError(message)
    active_config = replace(config, model_dim=capacity[0], modes=capacity[1])
    model = TightFrameSequenceRegressor(active_config, variant)
    if name == "pac_tf_fixed_damping":
        model.block1.raw_decay.requires_grad_(requires_grad=False)
        model.block2.raw_decay.requires_grad_(requires_grad=False)
    return model
