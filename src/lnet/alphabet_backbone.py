"""Canonical implementation of the ALPHABET backbone.

This module deliberately keeps the complete high-level dataflow in one place:

    raw samples -> direct local stem -> tied writer -> direct reader lift
    -> terminal pole analysis -> pooled real stream + lag-(1,2,4) moments -> head

The numerically sensitive recurrence and moment kernels remain imported from
the surrounding ``lnet`` package. Duplicating those kernels here would create a second algorithm
to maintain and would make checkpoint/gradient parity harder to audit.

``AlphabetBackbone`` intentionally differs from the historical
``hco_identity_d{D}m{M}`` campaign at the stem: the projection has shape
``[D,C]`` rather than ``[D,2C]`` and the model retains all T timesteps.  Old
edge-stem checkpoints therefore require retraining rather than strict loading.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.pac_h_compact_lag124_tied import HCompactLag124TiedPAC
from lnet.pac_laplace_native_input import (
    _make_same_length_depthwise_conv1d,
    _RawForcingStem,
    _validate_dwconv_geometry,
)
from lnet.pac_real2d_math import compiled_discrete_pole_real2d, pole_transition_real2d
from lnet.pac_tight_frame_runtime import BorrowedInputCudaGraphInference
from lnet.pac_triton_direct_stem_training import (
    direct_stem_c1_inference,
    direct_stem_c1_training,
    direct_stem_c2_inference,
    direct_stem_c2_training,
)
from lnet.pac_triton_terminal_reader_local_training import terminal_reader_local_training
from lnet.pac_triton_terminal_reader_scan_inference import (
    terminal_reader_scan_inference,
)
from lnet.pac_triton_terminal_reader_scan_training import (
    encoded_reader_scan_training,
    terminal_reader_radial_log_scan_training,
    terminal_reader_scan_training,
)
from lnet.pac_triton_writer_reader_local_training import (
    writer_modal_reader_local_training,
    writer_reader_local_training,
)
from lnet.pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from torch.optim import AdamW

    from lnet.pac_cuda_outer_graph import EFP16ExactSplitOuterGraph
    from lnet.pac_efp16_exact_split_training import EFP16ExactSplitTraining
    from lnet.pac_headroom_models import HeadroomObjective
    from lnet.pac_tight_frame_runtime import InferenceCompileMode

CAPTURE_POST_OPTIMIZER_STEP_BY_DEFAULT = False


def _is_canonical_d4_depthwise(local: nn.Conv1d) -> bool:
    """Return whether a local map matches the geometry hard-coded by fused kernels."""
    return (
        local.kernel_size == (5,)
        and local.stride == (1,)
        and local.dilation == (4,)
        and local.padding == (8,)
        and local.groups == local.in_channels
        and local.in_channels == local.out_channels
    )


def _physical_time_depthwise_conv(
    inputs: Tensor,
    convolution: nn.Conv1d,
    time_delta: Tensor,
    valid_mask: Tensor | None,
) -> Tensor:
    """Evaluate a same-length depthwise kernel at fixed physical-time offsets.

    The projected signal is interpreted as its piecewise-linear interpolant.
    Integer-unit sampling therefore reproduces the ordinary dilated Conv1d,
    while irregular event grids retain the same physical receptive offsets.
    """
    if (
        convolution.stride != (1,)
        or convolution.groups != inputs.shape[-1]
        or convolution.in_channels != inputs.shape[-1]
        or convolution.out_channels != inputs.shape[-1]
    ):
        message = "physical-time stem requires a stride-one depthwise convolution"
        raise ValueError(message)
    if time_delta.shape != (*inputs.shape[:2], 1):
        message = "physical-time stem requires time_delta with shape [B,N,1]"
        raise ValueError(message)

    timestamps = time_delta.squeeze(-1).cumsum(dim=1)
    kernel_size = convolution.kernel_size[0]
    dilation = convolution.dilation[0]
    left_extent = dilation * (kernel_size - 1) // 2
    offsets = (
        torch.arange(kernel_size, device=inputs.device, dtype=inputs.dtype) * dilation - left_extent
    )
    queries = timestamps.unsqueeze(-1) + offsets.view(1, 1, -1)
    upper_unclamped = torch.searchsorted(
        timestamps.contiguous(),
        queries.flatten(1).contiguous(),
        right=False,
    ).reshape_as(queries)
    upper = upper_unclamped.clamp(max=inputs.shape[1] - 1)
    lower = (upper_unclamped - 1).clamp(min=0, max=inputs.shape[1] - 1)
    batch = torch.arange(inputs.shape[0], device=inputs.device).view(-1, 1, 1)
    lower_time = timestamps[batch, lower]
    upper_time = timestamps[batch, upper]
    duration = upper_time - lower_time
    interpolation = torch.where(
        duration > 0,
        (queries - lower_time) / duration.clamp_min(torch.finfo(inputs.dtype).tiny),
        torch.zeros_like(queries),
    ).clamp(0.0, 1.0)
    lower_values = inputs[batch, lower]
    upper_values = inputs[batch, upper]
    sampled = torch.lerp(
        lower_values,
        upper_values,
        interpolation.unsqueeze(-1),
    )
    in_bounds = (queries >= timestamps[:, :1].unsqueeze(-1)) & (
        queries <= timestamps[:, -1:].unsqueeze(-1)
    )

    if valid_mask is not None:
        active_valid = valid_mask.squeeze(-1)
        lower_valid = active_valid[batch, lower]
        upper_valid = active_valid[batch, upper]
        epsilon = 8.0 * torch.finfo(inputs.dtype).eps
        support_valid = torch.where(
            interpolation <= epsilon,
            lower_valid,
            torch.where(
                interpolation >= 1.0 - epsilon,
                upper_valid,
                lower_valid * upper_valid,
            ),
        )
        in_bounds = in_bounds & (support_valid > 0)
    sampled = sampled * in_bounds.unsqueeze(-1).to(dtype=sampled.dtype)

    weight = convolution.weight[:, 0, :]
    output = torch.einsum("bnkd,dk->bnd", sampled, weight)
    if convolution.bias is not None:
        output = output + convolution.bias.view(1, 1, -1)
    return output


def _validate_request_metadata(
    inputs: Tensor,
    time_delta: Tensor | None,
    observation_mask: Tensor | None,
    valid_mask: Tensor | None,
) -> None:
    """Validate request metadata before prepared or compiled execution."""
    if time_delta is not None:
        if time_delta.shape not in (inputs.shape[:2], (*inputs.shape[:2], 1)):
            message = "time_delta must have shape [B,N] or [B,N,1]"
            raise ValueError(message)
        if not torch.isfinite(time_delta).all() or bool((time_delta < 0).any()):
            message = "time_delta must contain finite non-negative values"
            raise ValueError(message)
    if observation_mask is not None:
        allowed_shapes = (
            inputs.shape[:2],
            (*inputs.shape[:2], 1),
            inputs.shape,
        )
        if observation_mask.shape not in allowed_shapes:
            message = (
                "observation_mask must have shape [B,N], [B,N,1], or [B,N,C] matching the raw input"
            )
            raise ValueError(message)
        if not torch.isfinite(observation_mask).all() or bool(
            ((observation_mask < 0) | (observation_mask > 1)).any()
        ):
            message = "observation_mask must contain finite values in [0,1]"
            raise ValueError(message)
    if valid_mask is not None:
        if valid_mask.shape not in (inputs.shape[:2], (*inputs.shape[:2], 1)):
            message = "valid_mask must have shape [B,N] or [B,N,1]"
            raise ValueError(message)
        if not torch.isfinite(valid_mask).all() or bool(
            ((valid_mask < 0) | (valid_mask > 1)).any()
        ):
            message = "valid_mask must contain finite values in [0,1]"
            raise ValueError(message)


class _StaticBlockScanInference(Protocol):
    static_lag124_recurrence_moments_inference: bool
    static_pole_recurrence_moments_inference: bool
    static_pole_block_scan_inference: bool
    static_pole_block_scan_block_size_inference: int
    packed_static_recurrence_moments_inference: bool
    packed_static_recurrence_drive_inference: bool
    single_warp_static_recurrence_moments_inference: bool


def _configure_static_block_scan_inference_(
    block: _StaticBlockScanInference,
    *,
    block_size: int,
) -> None:
    """Select the verified packed one-warp exact block-scan path."""
    block.static_lag124_recurrence_moments_inference = False
    block.static_pole_recurrence_moments_inference = False
    block.static_pole_block_scan_inference = True
    block.static_pole_block_scan_block_size_inference = block_size
    block.packed_static_recurrence_moments_inference = True
    block.packed_static_recurrence_drive_inference = True
    block.single_warp_static_recurrence_moments_inference = True


class AlphabetBackbone(HCompactLag124TiedPAC):
    """Canonical tied writer-reader backbone for ALPHABET.

    The class intentionally spells out the full model-level forward path.  The
    inherited modules provide the verified pole blocks, constrained-frame
    lifecycle, and task head; the inherited edge stem is replaced below.
    """

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        dwconv_kernel_size: int = 5,
        dwconv_dilation: int = 4,
        objective: HeadroomObjective = "classification",
    ) -> None:
        _validate_dwconv_geometry(dwconv_kernel_size, dwconv_dilation)
        super().__init__(config, output_dim, objective=objective)
        self.dwconv_kernel_size = dwconv_kernel_size
        self.dwconv_dilation = dwconv_dilation
        if (dwconv_kernel_size, dwconv_dilation) != (5, 4):
            self.second_local = _make_same_length_depthwise_conv1d(
                config.model_dim,
                kernel_size=dwconv_kernel_size,
                dilation=dwconv_dilation,
            )
        # The production model acts directly on every raw timestep:
        # Linear(C,D) -> same-length DWConv -> SiLU.  K5/D4 remains the
        # production fast path; other geometries use the exact PyTorch fallback.
        # The inherited edge-frame stem is intentionally replaced rather than
        # fed synthetic level/detail features, so the temporal length remains T.
        self.stem = _RawForcingStem(
            config.raw_input_dim,
            config.model_dim,
            dwconv_kernel_size=dwconv_kernel_size,
            dwconv_dilation=dwconv_dilation,
        )
        # The inherited writer-reader family installs a learned D x D map here.
        # ALPHABET connects the writer directly to the reader, so remove the
        # module entirely instead of retaining a parameter-free placeholder.
        del self.second_projection
        for block in (self.forward_block, self.backward_block):
            block.skip_redundant_lag124_moment_fusion = True
        # Opt in to the verified fullgraph/opaque-recurrence training runtime.
        # The shared trainer reads these policy attributes without importing the
        # optimization surface, keeping the campaign and runtime dependency one-way.
        self.use_efp16_exact_split_training = True
        self.use_external_exact_split_training = True
        self.require_external_exact_split_training = True
        self.external_exact_split_preserve_eager_body = False
        self.efp16_exact_split_compile_model_body = True
        self.efp16_exact_split_allow_multichannel_inputs = True
        self.efp16_exact_split_capture_post_optimizer_step = CAPTURE_POST_OPTIMIZER_STEP_BY_DEFAULT
        # The semi-orthogonal direct stem has two columns in the optimized
        # two-channel surface.  A one-kernel narrow QR preserves the
        # positive-diagonal convention while avoiding a general cuSOLVER launch.
        self.use_small_stem_qr_retraction = True
        self.use_fused_efp16_c2_stem_training = False
        self.use_fused_terminal_reader_local_training = False
        self.use_fused_terminal_reader_scan_inference = False
        self.use_parallel_terminal_reader_scan_inference = False
        self.use_state_free_parallel_terminal_reader_scan_inference = False
        self.use_small_parallel_writer_inference = False
        self.use_medium_writer_block_scan_inference = False
        self.use_large_writer_block_scan_inference = False
        self.use_large_terminal_reader_scan_inference = False
        self.use_fused_terminal_reader_scan_training = False
        self.use_fused_writer_reader_local_training = False
        self.use_fused_writer_modal_reader_local_training = False
        self.validate_metadata = True
        self.point_sample_local_convolution = False
        self._inference_prepared = False

    def train(self, mode: bool = True) -> AlphabetBackbone:  # noqa: FBT001, FBT002
        if mode and self._inference_prepared:
            message = "a materialized inference model cannot return to training"
            raise RuntimeError(message)
        if mode:
            self.validate_metadata = True
        return super().train(mode)

    @torch.no_grad()
    def post_optimizer_step(self) -> None:
        """Retract constrained parameters with the narrow CUDA stem fast path."""
        self.forward_block.retract_frame()
        self.backward_block.retract_frame()
        stem = self.stem
        if not isinstance(stem, _RawForcingStem):
            return
        weight = stem.projection.weight
        use_small_qr = (
            self.use_small_stem_qr_retraction
            and weight.is_cuda
            and weight.dtype == torch.float32
            and weight.is_contiguous()
            and weight.ndim == 2
            and weight.shape[1] == 2
            and weight.shape[0] >= weight.shape[1]
        )
        if use_small_qr:
            from lnet.pac_triton_small_qr import small_qr_retraction_  # noqa: PLC0415

            small_qr_retraction_(weight)
        else:
            stem.project_weight_()

    def prepare_classifier_exact_split_runtime(
        self,
        optimizer: AdamW,
        inputs: Tensor,
        labels: Tensor,
        *,
        grad_clip_norm: float,
    ) -> EFP16ExactSplitTraining | AlphabetTrainingRuntime:
        """Select the strongest verified runtime available on this CUDA build.

        CUDA 12.8 devices with the device-side matrix-exp SWITCH use the complete
        outer graph. CUDA 12.4 workers retain the exact-split host-dispatch path;
        they must not attempt construction of an unsupported outer graph.
        """
        from lnet.pac_native_matrix_exp_vjp import (  # noqa: PLC0415
            cuda_switch_matrix_exp_capability,
        )

        if inputs.is_cuda and cuda_switch_matrix_exp_capability()[0]:
            return prepare_aggressive_training(
                self,
                optimizer,
                inputs,
                labels,
                grad_clip_norm=grad_clip_norm,
                warmup_steps=1,
            )
        return prepare_exact_split_training(
            self,
            optimizer,
            inputs,
            labels,
            grad_clip_norm=grad_clip_norm,
            warmup_steps=1,
        )

    def prepare_external_exact_split_runtime(
        self,
        optimizer: AdamW,
        inputs: Tensor,
        targets: Tensor,
        *,
        objective: Literal["multiclass", "multilabel", "forecasting"],
        grad_clip_norm: float,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
        loss_weight: Tensor | None = None,
        metadata_prevalidated: bool = False,
    ) -> EFP16ExactSplitTraining | AlphabetTrainingRuntime:
        """Build the exact runtime for one fixed data and metadata signature."""
        loss_kind = cast(
            "Literal['cross_entropy', 'binary_cross_entropy', 'mse']",
            {
                "multiclass": "cross_entropy",
                "multilabel": "binary_cross_entropy",
                "forecasting": "mse",
            }[objective],
        )
        requires_base_runtime = loss_weight is not None or any(
            value is not None for value in (time_delta, observation_mask, valid_mask)
        )
        if self.external_exact_split_preserve_eager_body or requires_base_runtime:
            return prepare_exact_split_training(
                self,
                optimizer,
                inputs,
                targets,
                example_time_delta=time_delta,
                example_observation_mask=observation_mask,
                example_valid_mask=valid_mask,
                cross_entropy_weight=loss_weight,
                validate_metadata_values=not metadata_prevalidated,
                grad_clip_norm=grad_clip_norm,
                warmup_steps=1,
                mode_static_pole_training=False,
                fused_moments_backward_training=False,
                fused_recurrence_moments_backward_training=False,
                compile_model_body=False,
                loss_kind=loss_kind,
            )
        from lnet.pac_native_matrix_exp_vjp import (  # noqa: PLC0415
            cuda_switch_matrix_exp_capability,
        )

        if inputs.is_cuda and cuda_switch_matrix_exp_capability()[0]:
            return prepare_aggressive_training(
                self,
                optimizer,
                inputs,
                targets,
                grad_clip_norm=grad_clip_norm,
                warmup_steps=1,
                loss_kind=loss_kind,
            )
        return prepare_exact_split_training(
            self,
            optimizer,
            inputs,
            targets,
            grad_clip_norm=grad_clip_norm,
            warmup_steps=1,
            loss_kind=loss_kind,
        )

    @torch.no_grad()
    def prepare_for_inference_(
        self,
        *,
        sequence_length: int,
        batch_size: int,
        use_static_poles: bool = True,
    ) -> AlphabetBackbone:
        """Materialize a dedicated static-shape inference model in place.

        Call this only after checkpoint loading and the final device/dtype move.
        Direct prepared execution retains metadata validation. The compiled wrapper
        validates requests outside its fullgraph core.
        """
        if sequence_length < 1 or batch_size < 1:
            message = "prepared inference requires positive sequence and batch sizes"
            raise ValueError(message)
        self.eval()
        use_small_extreme_path = use_static_poles and batch_size == 1 and sequence_length == 512
        use_medium_extreme_path = (
            use_static_poles and batch_size in (32, 64) and 1 <= sequence_length <= 1024
        )
        use_large_extreme_path = (
            use_static_poles and batch_size == 32 and 1025 <= sequence_length <= 2048
        )
        use_block_scan = sequence_length >= 2048 and batch_size == 1
        block_size = 64 if use_block_scan else 256
        for block in (self.forward_block, self.backward_block):
            block.prepare_for_inference_(
                use_block_scan=use_block_scan,
                use_fused_recurrence_moments=not use_block_scan,
                use_static_pole_recurrence_moments=(use_static_poles and not use_block_scan),
                use_packed_static_recurrence_moments=use_static_poles,
                use_packed_static_recurrence_drive=(use_static_poles and use_block_scan),
                use_single_warp_static_recurrence_moments=(use_static_poles and batch_size == 1),
                use_static_lag124_recurrence_moments=(use_static_poles and not use_block_scan),
                use_static_pole_block_scan=(use_static_poles and use_block_scan),
                static_pole_block_scan_block_size=block_size,
            )
        if use_small_extreme_path:
            self.forward_block.parallel_static_pole_recurrence_moments_inference = True
            self.forward_block.parallel_static_pole_recurrence_num_warps_inference = 4
        elif use_medium_extreme_path:
            _configure_static_block_scan_inference_(
                self.forward_block,
                block_size=64,
            )
        elif use_large_extreme_path:
            _configure_static_block_scan_inference_(
                self.forward_block,
                block_size=128,
            )
        use_terminal_reader_extreme_path = (
            use_small_extreme_path or use_medium_extreme_path or use_large_extreme_path
        )
        self.use_fused_terminal_reader_scan_inference = use_terminal_reader_extreme_path
        self.use_parallel_terminal_reader_scan_inference = use_terminal_reader_extreme_path
        self.use_state_free_parallel_terminal_reader_scan_inference = (
            use_terminal_reader_extreme_path
        )
        self.use_small_parallel_writer_inference = use_small_extreme_path
        self.use_medium_writer_block_scan_inference = use_medium_extreme_path
        self.use_large_writer_block_scan_inference = use_large_extreme_path
        self.use_large_terminal_reader_scan_inference = use_large_extreme_path
        self._inference_prepared = True
        return self

    def _prepare_active_metadata(
        self,
        features: Tensor,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        _, event_observation = self._prepare_observation_mask(features, observation_mask)
        return (
            self._prepare_time_delta(features, time_delta),
            event_observation,
            self._prepare_mask(features, valid_mask, name="valid_mask"),
        )

    def _prepare_time_delta(self, features: Tensor, time_delta: Tensor | None) -> Tensor | None:
        if time_delta is None:
            return None
        if time_delta.shape not in (features.shape[:2], (*features.shape[:2], 1)):
            message = "time_delta must have shape [B,N] or [B,N,1]"
            raise ValueError(message)
        active = time_delta.to(device=features.device, dtype=features.dtype)
        if active.ndim == 2:
            active = active.unsqueeze(-1)
        if self.validate_metadata and (
            not torch.isfinite(active).all() or bool((active < 0).any())
        ):
            message = "time_delta must contain finite non-negative values"
            raise ValueError(message)
        return active

    def _prepare_observation_mask(
        self,
        features: Tensor,
        observation_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return raw-channel and event-level observation masks.

        Channelwise masks suppress only the corresponding raw values. The pole
        drive remains active when at least one channel is observed at an event.
        """
        if observation_mask is None:
            return None, None
        allowed_shapes = (
            features.shape[:2],
            (*features.shape[:2], 1),
            features.shape,
        )
        if observation_mask.shape not in allowed_shapes:
            message = (
                "observation_mask must have shape [B,N], [B,N,1], or [B,N,C] matching the raw input"
            )
            raise ValueError(message)
        raw_mask = observation_mask.to(device=features.device, dtype=features.dtype)
        if raw_mask.ndim == 2:
            raw_mask = raw_mask.unsqueeze(-1)
        if self.validate_metadata and (
            not torch.isfinite(raw_mask).all() or bool(((raw_mask < 0) | (raw_mask > 1)).any())
        ):
            message = "observation_mask must contain finite values in [0,1]"
            raise ValueError(message)
        event_mask = raw_mask if raw_mask.shape[-1] == 1 else raw_mask.amax(dim=-1, keepdim=True)
        return raw_mask, event_mask

    def _prepare_mask(
        self,
        features: Tensor,
        mask: Tensor | None,
        *,
        name: str,
    ) -> Tensor | None:
        if mask is None:
            return None
        if mask.shape not in (features.shape[:2], (*features.shape[:2], 1)):
            message = f"{name} must have shape [B,N] or [B,N,1]"
            raise ValueError(message)
        active = mask.to(device=features.device, dtype=features.dtype)
        if active.ndim == 2:
            active = active.unsqueeze(-1)
        if self.validate_metadata and (
            not torch.isfinite(active).all() or bool(((active < 0) | (active > 1)).any())
        ):
            message = f"{name} must contain finite values in [0,1]"
            raise ValueError(message)
        return active

    def _edge_stem(
        self,
        inputs: Tensor,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Apply the direct raw-input stem without changing temporal length."""
        active_stem = self.stem
        if self._can_fuse_direct_stem(inputs, time_delta, observation_mask, valid_mask):
            if not isinstance(active_stem, _RawForcingStem):
                message = "direct stem dispatch lost its canonical stem"
                raise AssertionError(message)
            if active_stem.local.bias is None:
                message = "fused ALPHABET training stem requires the canonical local bias"
                raise RuntimeError(message)
            fused_training = (
                direct_stem_c1_training if inputs.shape[-1] == 1 else direct_stem_c2_training
            )
            first_local = fused_training(
                inputs,
                active_stem.projection.weight,
                active_stem.local.weight,
                active_stem.local.bias,
            )
            return first_local, None, None, None
        if (
            isinstance(active_stem, _RawForcingStem)
            and _is_canonical_d4_depthwise(active_stem.local)
            and not torch.is_grad_enabled()
            and time_delta is None
            and observation_mask is None
            and valid_mask is None
            and inputs.is_cuda
            and inputs.dtype == torch.float32
            and inputs.shape[-1] in (1, 2)
            and inputs.shape[1] >= 1
        ):
            local_bias = active_stem.local.bias
            if local_bias is None:
                message = "fused ALPHABET inference stem requires the canonical local bias"
                raise RuntimeError(message)
            fused_inference = (
                direct_stem_c1_inference if inputs.shape[-1] == 1 else direct_stem_c2_inference
            )
            return (
                fused_inference(
                    inputs,
                    active_stem.projection.weight,
                    active_stem.local.weight,
                    local_bias,
                ),
                None,
                None,
                None,
            )
        active_delta = self._prepare_time_delta(inputs, time_delta)
        raw_observation, active_observation = self._prepare_observation_mask(
            inputs,
            observation_mask,
        )
        active_valid = self._prepare_mask(inputs, valid_mask, name="valid_mask")
        stem_inputs = inputs
        if raw_observation is not None:
            stem_inputs = stem_inputs * raw_observation
        if active_valid is not None:
            stem_inputs = stem_inputs * active_valid
        if (
            active_delta is not None
            and isinstance(active_stem, _RawForcingStem)
            and not self.point_sample_local_convolution
        ):
            projected = active_stem.projection(stem_inputs)
            local = _physical_time_depthwise_conv(
                projected,
                active_stem.local,
                active_delta,
                active_valid,
            )
            first_local = functional.silu(local)
        else:
            first_local = active_stem(stem_inputs)
        first_local = self._mask_features(first_local, active_valid)
        return first_local, active_delta, active_observation, active_valid

    def _can_fuse_direct_stem(
        self,
        inputs: Tensor,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> bool:
        """Return whether the C=1/C=2 maskless training stem can dispatch."""
        return (
            isinstance(self.stem, _RawForcingStem)
            and _is_canonical_d4_depthwise(self.stem.local)
            and self.use_fused_efp16_stem_training
            and self.use_fused_efp16_c2_stem_training
            and self.training
            and torch.is_grad_enabled()
            and time_delta is None
            and observation_mask is None
            and valid_mask is None
            and inputs.is_cuda
            and inputs.dtype == torch.float32
            and inputs.is_contiguous()
            and inputs.ndim == 3
            and inputs.shape[-1] in (1, 2)
            and inputs.shape[1] >= 1
        )

    def _writer(
        self,
        first_local: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run the tied analysis/synthesis writer and its lag moments."""
        return self.forward_block(
            first_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            metadata_prevalidated=True,
        )

    def _can_fuse_terminal_reader_scan(
        self,
        values: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> bool:
        reader = self.backward_block
        return (
            self.use_fused_terminal_reader_scan_training
            and self.training
            and torch.is_grad_enabled()
            and active_delta is None
            and active_observation is None
            and active_valid is None
            and values.is_cuda
            and values.dtype == torch.float32
            and values.is_contiguous()
            and values.shape[0] in (32, 64)
            and values.shape[2] == 64
            and 1 <= values.shape[1] <= 2048
            and self.second_local.bias is not None
            and self.second_local.weight.shape == (64, 1, 5)
            and self.second_local.dilation == (4,)
            and self.second_local.padding == (8,)
            and self.second_local.groups == 64
            and reader.mode_static_pole_training
            and reader.direction == "forward"
            and reader.use_input_norm
            and reader.local is None
            and reader.norm.weight is not None
            and reader.frame.out_features == 64
            and reader.frame.in_features == 32
            and reader.excitation_mixer is None
            and reader.mode_gate_bias is None
            and reader.mode_gate_gain is None
            and reader.moment_lags == (1, 2, 4)
            and (
                (reader.log_energy and reader.normalize_autocorrelation)
                or (
                    reader.parallel_static_radial_log_recurrence_moments_training
                    and reader.radial_log_lag124_moments
                    and not reader.log_energy
                    and not reader.normalize_autocorrelation
                )
            )
            and reader.canonical_identity_elision
        )

    def _can_fuse_terminal_reader_scan_inference(
        self,
        values: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> bool:
        reader = self.backward_block
        return (
            self.use_fused_terminal_reader_scan_inference
            and not self.training
            and not torch.is_grad_enabled()
            and active_delta is None
            and active_observation is None
            and active_valid is None
            and values.is_cuda
            and values.dtype == torch.float32
            and values.is_contiguous()
            and values.shape[0] in (1, 32, 64)
            and values.shape[2] == 64
            and 1 <= values.shape[1] <= 2048
            and self.second_local.bias is not None
            and self.second_local.weight.shape == (64, 1, 5)
            and self.second_local.dilation == (4,)
            and self.second_local.padding == (8,)
            and self.second_local.groups == 64
            and reader.direction == "forward"
            and reader.use_input_norm
            and reader.local is None
            and reader.norm.weight is not None
            and reader.frame.out_features == 64
            and reader.frame.in_features == 32
            and (
                (
                    reader.static_pole_recurrence_moments_inference
                    and reader.static_lag124_recurrence_moments_inference
                    and reader.log_energy
                    and reader.normalize_autocorrelation
                )
                or (
                    reader.static_radial_log_lag124_recurrence_moments_inference
                    and reader.radial_log_lag124_moments
                    and not reader.log_energy
                    and not reader.normalize_autocorrelation
                )
            )
            and getattr(reader, "inference_drive_frame", None) is not None
            and getattr(reader, "inference_decay_real", None) is not None
            and getattr(reader, "inference_decay_imag", None) is not None
            and reader.excitation_mixer is None
            and reader.mode_gate_bias is None
            and reader.mode_gate_gain is None
            and reader.moment_lags == (1, 2, 4)
            and reader.canonical_identity_elision
        )

    def _reader_static_poles(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        reader = self.backward_block
        damping = reader.damping_values().view(1, 1, -1)
        frequency = reader.frequency_values().view(1, 1, -1)
        if reader.impulse_injection:
            decay_real, decay_imag = pole_transition_real2d(damping, frequency, 1.0)
            gamma_real = torch.ones_like(decay_real)
            gamma_imag = torch.zeros_like(decay_imag)
        else:
            decay_real, decay_imag, gamma_real, gamma_imag = compiled_discrete_pole_real2d(
                damping,
                frequency,
                1.0,
            )
        return (
            decay_real[0, 0],
            decay_imag[0, 0],
            gamma_real[0, 0],
            gamma_imag[0, 0],
        )

    def _can_fuse_encoded_reader_scan(self, encoded: Tensor) -> bool:
        """Return whether the narrower post-local scan kernel is eligible.

        ``_can_fuse_terminal_reader_scan`` covers the full local+scan operation
        through T2048. The encoded-only producer/scan uses the same B32/B64 and
        T<=2048 envelope once writer/reader local fusion has produced ``encoded``.
        """
        return (
            encoded.shape[0] in (32, 64)
            and 1 <= encoded.shape[1] <= 2048
            and self._can_fuse_terminal_reader_scan(encoded, None, None, None)
        )

    def _encoded_reader_moments(self, encoded: Tensor) -> Tensor:
        reader = self.backward_block
        norm_weight = reader.norm.weight
        if norm_weight is None:
            message = "fused encoded reader scan requires an affine RMSNorm"
            raise RuntimeError(message)
        return encoded_reader_scan_training(
            encoded,
            norm_weight,
            reader.frame_matrix(),
            *self._reader_static_poles(),
        )

    def _terminal_reader(
        self,
        first_stream: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Locally lift the writer stream and run the read-only pole analyzer."""
        # The reader consumes the writer stream directly. An optimizer may
        # replace this boundary with an equivalent fused implementation.
        # Invalid tail states can remain non-zero because the writer continues
        # autonomous decay. Mask them before the centered reader convolution so
        # right padding cannot leak back into the last valid positions.
        second_projected = self._mask_features(first_stream, active_valid)
        second_local_bias = self.second_local.bias
        reader = self.backward_block
        reader_norm_weight = reader.norm.weight
        if self._can_fuse_terminal_reader_scan_inference(
            second_projected,
            active_delta,
            active_observation,
            active_valid,
        ):
            if second_local_bias is None or reader_norm_weight is None:
                message = "fused inference reader requires canonical affine parameters"
                raise RuntimeError(message)
            drive_frame = cast(
                "Tensor | None",
                getattr(reader, "inference_drive_frame", None),
            )
            decay_real = cast(
                "Tensor | None",
                getattr(reader, "inference_decay_real", None),
            )
            decay_imag = cast(
                "Tensor | None",
                getattr(reader, "inference_decay_imag", None),
            )
            if drive_frame is None or decay_real is None or decay_imag is None:
                message = "fused inference reader requires prepared static buffers"
                raise RuntimeError(message)
            return terminal_reader_scan_inference(
                second_projected,
                self.second_local.weight,
                second_local_bias,
                reader_norm_weight,
                drive_frame,
                decay_real,
                decay_imag,
                single_warp=reader.single_warp_static_recurrence_moments_inference,
                parallel_scan=self.use_parallel_terminal_reader_scan_inference,
                state_free_parallel_scan=(
                    self.use_state_free_parallel_terminal_reader_scan_inference
                ),
            )
        if self._can_fuse_terminal_reader_scan(
            second_projected,
            active_delta,
            active_observation,
            active_valid,
        ):
            if second_local_bias is None or reader_norm_weight is None:
                message = "fused terminal reader scan requires canonical affine parameters"
                raise RuntimeError(message)
            reader_scan = (
                terminal_reader_radial_log_scan_training
                if reader.radial_log_lag124_moments
                else terminal_reader_scan_training
            )
            return reader_scan(
                second_projected,
                self.second_local.weight,
                second_local_bias,
                reader_norm_weight,
                reader.frame_matrix(),
                *self._reader_static_poles(),
            )
        use_fused_local = (
            self.use_fused_terminal_reader_local_training
            and _is_canonical_d4_depthwise(self.second_local)
            and self.training
            and torch.is_grad_enabled()
            and active_delta is None
            and active_observation is None
            and active_valid is None
            and second_projected.is_cuda
            and second_projected.dtype == torch.float32
            and second_projected.is_contiguous()
            and second_local_bias is not None
        )
        if use_fused_local:
            if second_local_bias is None:
                message = "fused terminal reader requires the canonical local bias"
                raise RuntimeError(message)
            encoded = terminal_reader_local_training(
                second_projected,
                self.second_local.weight,
                second_local_bias,
            )
        else:
            if active_delta is not None and not self.point_sample_local_convolution:
                reader_pre_activation = _physical_time_depthwise_conv(
                    second_projected,
                    self.second_local,
                    active_delta,
                    active_valid,
                )
            else:
                reader_pre_activation = self.second_local(
                    second_projected.transpose(1, 2)
                ).transpose(1, 2)
            encoded = self._mask_features(functional.silu(reader_pre_activation), active_valid)
        second_moments = self.backward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            metadata_prevalidated=True,
            return_moments_only=True,
        )
        return encoded, second_moments

    def _can_fuse_writer_reader_local(
        self,
        first_local: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> bool:
        writer = self.forward_block
        reader_bias = self.second_local.bias
        return (
            self.use_fused_writer_reader_local_training
            and self.training
            and torch.is_grad_enabled()
            and active_delta is None
            and active_observation is None
            and active_valid is None
            and first_local.is_cuda
            and first_local.dtype == torch.float32
            and first_local.is_contiguous()
            and first_local.shape[-1] == 64
            and writer.frame.out_features == 64
            and writer.frame.in_features == 32
            and writer.synthesis_scale == 1.0
            and not writer.split_residual_scales
            and writer.direct_scale is not None
            and writer.layer_scale is not None
            and self.second_local.weight.shape == (64, 1, 5)
            and self.second_local.dilation == (4,)
            and self.second_local.padding == (8,)
            and self.second_local.groups == 64
            and reader_bias is not None
        )

    def _can_fuse_writer_modal_reader_local(
        self,
        first_local: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> bool:
        writer = self.forward_block
        reader_bias = self.second_local.bias
        return (
            self.use_fused_writer_modal_reader_local_training
            and not self.use_fused_writer_reader_local_training
            and self.training
            and torch.is_grad_enabled()
            and active_delta is None
            and active_observation is None
            and active_valid is None
            and first_local.is_cuda
            and first_local.dtype == torch.float32
            and first_local.is_contiguous()
            and first_local.shape[-1] == 64
            and writer.frame.out_features == 64
            and writer.frame.in_features == 32
            and writer.synthesis_scale == 1.0
            and not writer.split_residual_scales
            and writer.direct_scale is not None
            and writer.layer_scale is not None
            and self.second_local.weight.shape == (64, 1, 5)
            and self.second_local.dilation == (4,)
            and self.second_local.padding == (8,)
            and self.second_local.groups == 64
            and reader_bias is not None
        )

    def _fused_writer_reader_local(
        self,
        first_local: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        writer = self.forward_block
        modal_coordinates, writer_local, first_moments, synthesis_frame = writer(
            first_local,
            metadata_prevalidated=True,
            return_training_tail_components=True,
        )
        direct_scale = writer.direct_scale
        layer_scale = writer.layer_scale
        reader_bias = self.second_local.bias
        if direct_scale is None or layer_scale is None or reader_bias is None:
            message = "fused writer-reader path requires canonical residual scales and bias"
            raise RuntimeError(message)
        encoded = writer_reader_local_training(
            first_local,
            writer_local,
            modal_coordinates,
            synthesis_frame,
            direct_scale,
            layer_scale,
            self.second_local.weight,
            reader_bias,
        )
        if self._can_fuse_encoded_reader_scan(encoded):
            second_moments = self._encoded_reader_moments(encoded)
        else:
            second_moments = self.backward_block(
                encoded,
                metadata_prevalidated=True,
                return_moments_only=True,
            )
        return encoded, first_moments, second_moments

    def _fused_writer_modal_reader_local(
        self,
        first_local: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        writer = self.forward_block
        modal_coordinates, writer_local, first_moments, synthesis_frame = writer(
            first_local,
            metadata_prevalidated=True,
            return_training_tail_components=True,
        )
        direct_scale = writer.direct_scale
        layer_scale = writer.layer_scale
        reader_bias = self.second_local.bias
        if direct_scale is None or layer_scale is None or reader_bias is None:
            message = "post-synthesis fusion requires canonical residual scales and bias"
            raise RuntimeError(message)
        modal = torch.matmul(modal_coordinates, synthesis_frame.transpose(0, 1))
        encoded = writer_modal_reader_local_training(
            first_local,
            writer_local,
            modal,
            direct_scale,
            layer_scale,
            self.second_local.weight,
            reader_bias,
        )
        if self._can_fuse_encoded_reader_scan(encoded):
            second_moments = self._encoded_reader_moments(encoded)
        else:
            second_moments = self.backward_block(
                encoded,
                metadata_prevalidated=True,
                return_moments_only=True,
            )
        return encoded, first_moments, second_moments

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        first_local, active_delta, active_observation, active_valid = self._edge_stem(
            inputs,
            time_delta,
            observation_mask,
            valid_mask,
        )
        if self._can_fuse_writer_reader_local(
            first_local,
            active_delta,
            active_observation,
            active_valid,
        ):
            encoded, first_moments, second_moments = self._fused_writer_reader_local(first_local)
        elif self._can_fuse_writer_modal_reader_local(
            first_local,
            active_delta,
            active_observation,
            active_valid,
        ):
            encoded, first_moments, second_moments = self._fused_writer_modal_reader_local(
                first_local
            )
        else:
            first_stream, first_moments = self._writer(
                first_local,
                active_delta,
                active_observation,
                active_valid,
            )
            # observation_mask describes missing raw observations.  The writer
            # must suppress their external drive, but the reader consumes the
            # writer's latent trajectory rather than raw observations.  Passing
            # the same mask again would erase valid autonomous memory a second
            # time.  Physical time and padding validity still apply to both.
            encoded, second_moments = self._terminal_reader(
                first_stream,
                active_delta,
                None,
                active_valid,
            )
        return self._readout(encoded, first_moments, second_moments, active_valid)


class BenchmarkAlphabetBackbone(AlphabetBackbone):
    """Small convenience constructor for optimization and parity benchmarks."""

    def __init__(
        self,
        raw_input_dim: int,
        model_dim: int,
        modes: int,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
        sequence_length: int = 2,
    ) -> None:
        config = PACExperimentConfig(
            sample_count=1,
            validation_count=1,
            test_count=0,
            sequence_length=sequence_length,
            raw_input_dim=raw_input_dim,
            output_dim=output_dim,
            model_dim=model_dim,
            modes=modes,
        )
        super().__init__(config, output_dim, objective=objective)


class _CompiledAlphabetInference(nn.Module):
    """Fullgraph wrapper that preserves the model's metadata-aware call surface."""

    def __init__(
        self,
        model: AlphabetBackbone,
        mode: InferenceCompileMode,
        *,
        copy_output: bool,
    ) -> None:
        super().__init__()
        self.uses_cuda_graphs = mode in {"reduce-overhead", "max-autotune"}
        self.copy_output = copy_output
        # Tensor-value validation cannot live inside a fullgraph compiled core.
        # Keep it at this request wrapper and compile the already-validated body.
        previous_validation = model.validate_metadata
        model.validate_metadata = False
        try:
            if mode == "dynamic-no-cudagraph":
                compiled = torch.compile(
                    model,
                    fullgraph=True,
                    dynamic=True,
                    options={"triton.cudagraphs": False},
                )
            else:
                compiled = torch.compile(
                    model,
                    fullgraph=True,
                    dynamic=False,
                    mode=None if mode == "default" else mode,
                )
        except Exception:
            model.validate_metadata = previous_validation
            raise
        self.compiled = cast("AlphabetBackbone", compiled)

    @torch.no_grad()
    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        metadata_free = time_delta is None and observation_mask is None and valid_mask is None
        if not metadata_free:
            _validate_request_metadata(
                inputs,
                time_delta,
                observation_mask,
                valid_mask,
            )
        if self.uses_cuda_graphs and inputs.is_cuda:
            torch.compiler.cudagraph_mark_step_begin()
        if metadata_free:
            output = self.compiled(inputs)
        else:
            output = self.compiled(
                inputs,
                time_delta=time_delta,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
            )
        return output.clone() if self.copy_output else output


def prepare_compiled_inference(
    model: AlphabetBackbone,
    *,
    sequence_length: int,
    batch_size: int,
    compile_mode: InferenceCompileMode = "reduce-overhead",
    copy_output: bool = True,
    use_static_poles: bool = True,
) -> nn.Module:
    """Prepare and optionally fullgraph-compile a dedicated inference model."""
    model.prepare_for_inference_(
        sequence_length=sequence_length,
        batch_size=batch_size,
        use_static_poles=use_static_poles,
    )
    if compile_mode == "none":
        return model
    return _CompiledAlphabetInference(model, compile_mode, copy_output=copy_output)


def prepare_maskless_inference(
    model: AlphabetBackbone,
    *,
    sequence_length: int,
    batch_size: int,
    copy_output: bool = False,
    use_static_poles: bool = True,
) -> nn.Module:
    """Prepare the measured low-latency path for metadata-free inference.

    B1/T128 uses a manually captured CUDA Graph. Its first input allocation is
    borrowed for the runtime lifetime: update that tensor in place for later
    requests. With ``copy_output=False`` the returned logits are also borrowed
    until the next replay. This strict maskless runtime accepts only ``inputs``;
    use :func:`prepare_compiled_inference` when request metadata is required.
    Other shapes retain the safe compiled path.
    """
    if (batch_size, sequence_length) != (1, 128):
        return prepare_compiled_inference(
            model,
            sequence_length=sequence_length,
            batch_size=batch_size,
            compile_mode="reduce-overhead",
            copy_output=copy_output,
            use_static_poles=use_static_poles,
        )
    model.prepare_for_inference_(
        sequence_length=sequence_length,
        batch_size=batch_size,
        use_static_poles=use_static_poles,
    )
    return BorrowedInputCudaGraphInference(
        model,
        compile_mode="max-autotune-no-cudagraphs",
        copy_output=copy_output,
    )


def prepare_exact_split_training(
    model: AlphabetBackbone,
    optimizer: AdamW,
    example_inputs: Tensor,
    example_labels: Tensor,
    *,
    example_time_delta: Tensor | None = None,
    example_observation_mask: Tensor | None = None,
    example_valid_mask: Tensor | None = None,
    mode_static_pole_training: bool = True,
    memory_efficient_lag124_training: bool = False,
    parallel_static_recurrence_training: bool = False,
    parallel_static_excitation_training: bool = False,
    saved_state_lag124_training: bool = False,
    fused_terminal_reader_local_training: bool = False,
    fused_terminal_reader_scan_training: bool = False,
    fused_writer_reader_local_training: bool = False,
    fused_writer_modal_reader_local_training: bool = False,
    **options: object,
) -> EFP16ExactSplitTraining:
    """Build a dedicated exact native-matrix-exp CUDA Graph training runtime.

    Setup resets Inductor's process-local CUDA Graph tree, so other compiled
    callables in the process may recapture when they are used again. Close the
    runtime before returning the model to ordinary eager execution.

    ``memory_efficient_lag124_training`` selects the Mamba-style state-free
    reader and packed writer.  It lowers saved temporary memory but is opt-in
    because the existing fullgraph path is faster for the measured B32/T128
    cell on CUDA 12.8.
    """
    from lnet.pac_efp16_exact_split_training import (  # noqa: PLC0415
        prepare_efp16_exact_split_training,
    )
    from lnet.pac_native_matrix_exp_vjp import (  # noqa: PLC0415
        cuda_switch_matrix_exp_capability,
    )

    device_matrix_exp = example_inputs.is_cuda and cuda_switch_matrix_exp_capability()[0]

    defaults: dict[str, object] = {
        "warmup_steps": 1,
        "specialized_matrix_exp_vjp": True,
        "matrix_exp_dispatch": "cuda_switch" if device_matrix_exp else "host",
        "compile_model_body": True,
        "capture_post_optimizer_step": CAPTURE_POST_OPTIMIZER_STEP_BY_DEFAULT,
        "fused_recurrence_moments_backward_training": True,
        "allow_multichannel_inputs": True,
    }
    defaults.update(options)
    blocks = (model.forward_block, model.backward_block)
    static_pole_state = tuple(block.mode_static_pole_training for block in blocks)
    lag124_training_state = tuple(
        block.static_lag124_recurrence_moments_training for block in blocks
    )
    parallel_static_state = tuple(block.parallel_static_recurrence_training for block in blocks)
    parallel_radial_log_state = tuple(
        block.parallel_static_radial_log_recurrence_moments_training for block in blocks
    )
    parallel_excitation_state = tuple(
        block.parallel_static_excitation_recurrence_training for block in blocks
    )
    saved_state_reader_state = tuple(block.saved_state_lag124_reader_training for block in blocks)
    terminal_reader_local_state = model.use_fused_terminal_reader_local_training
    terminal_reader_scan_state = model.use_fused_terminal_reader_scan_training
    writer_reader_local_state = model.use_fused_writer_reader_local_training
    writer_modal_reader_local_state = model.use_fused_writer_modal_reader_local_training
    try:
        # BenchmarkAlphabetBackbone's damping is mode-static when request metadata is absent.
        # Enable the broadcast-view recurrence path while the exact-split body is
        # compiled and captured, then restore the caller-owned eager policy.  The
        # runtime replays the captured graph and therefore does not need this Python
        # dispatch flag to remain mutated after construction.
        for block in blocks:
            block.mode_static_pole_training = mode_static_pole_training
            block.static_lag124_recurrence_moments_training = (
                memory_efficient_lag124_training or saved_state_lag124_training
            )
            block.parallel_static_recurrence_training = parallel_static_recurrence_training
            block.parallel_static_radial_log_recurrence_moments_training = (
                parallel_static_recurrence_training and block.radial_log_lag124_moments
            )
            block.parallel_static_excitation_recurrence_training = (
                parallel_static_excitation_training
            )
            block.saved_state_lag124_reader_training = saved_state_lag124_training
        model.use_fused_terminal_reader_local_training = fused_terminal_reader_local_training
        model.use_fused_terminal_reader_scan_training = fused_terminal_reader_scan_training
        model.use_fused_writer_reader_local_training = fused_writer_reader_local_training
        model.use_fused_writer_modal_reader_local_training = (
            fused_writer_modal_reader_local_training
        )
        metadata_options: dict[str, object] = {}
        if any(
            value is not None
            for value in (
                example_time_delta,
                example_observation_mask,
                example_valid_mask,
            )
        ):
            metadata_options = {
                "example_time_delta": example_time_delta,
                "example_observation_mask": example_observation_mask,
                "example_valid_mask": example_valid_mask,
            }
        defaults.update(metadata_options)
        return prepare_efp16_exact_split_training(
            model,
            optimizer,
            example_inputs,
            example_labels,
            **defaults,  # pyright: ignore[reportArgumentType]
        )
    finally:
        for (
            block,
            previous_static,
            previous_lag124,
            previous_parallel,
            previous_parallel_radial_log,
            previous_parallel_excitation,
            previous_saved_reader,
        ) in zip(
            blocks,
            static_pole_state,
            lag124_training_state,
            parallel_static_state,
            parallel_radial_log_state,
            parallel_excitation_state,
            saved_state_reader_state,
            strict=True,
        ):
            block.mode_static_pole_training = previous_static
            block.static_lag124_recurrence_moments_training = previous_lag124
            block.parallel_static_recurrence_training = previous_parallel
            block.parallel_static_radial_log_recurrence_moments_training = (
                previous_parallel_radial_log
            )
            block.parallel_static_excitation_recurrence_training = previous_parallel_excitation
            block.saved_state_lag124_reader_training = previous_saved_reader
        model.use_fused_terminal_reader_local_training = terminal_reader_local_state
        model.use_fused_terminal_reader_scan_training = terminal_reader_scan_state
        model.use_fused_writer_reader_local_training = writer_reader_local_state
        model.use_fused_writer_modal_reader_local_training = writer_modal_reader_local_state


class AlphabetTrainingRuntime:
    """Production outer-graph owner for aggressive fixed-shape training.

    ``step`` accepts arbitrary same-shape CUDA tensors and performs asynchronous
    device-to-device staging followed by one compute-graph launch. ``step_static``
    removes the staging launches when callers fill the exposed runtime buffers,
    while ``step_leased`` captures staging from the construction tensor addresses
    into the same one-launch root graph.
    """

    def __init__(
        self,
        runtime: EFP16ExactSplitTraining,
        outer: EFP16ExactSplitOuterGraph,
    ) -> None:
        self.runtime = runtime
        self.outer = outer
        self._destroyed = False

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        """Train on arbitrary same-shape CUDA tensors."""
        return self.outer.step(inputs, labels)

    def step_static(self) -> Tensor:
        """Train from the runtime-owned buffers without input-copy launches."""
        return self.outer.step_static()

    def step_leased(self, inputs: Tensor, labels: Tensor) -> Tensor:
        """Train in one launch from the fixed construction tensor addresses."""
        return self.outer.step_leased(inputs, labels)

    @property
    def static_inputs(self) -> Tensor:
        """Return the writable fixed-shape input buffer."""
        return self.outer.static_inputs

    @property
    def static_labels(self) -> Tensor:
        """Return the writable fixed-shape label buffer."""
        return self.outer.static_labels

    @property
    def recurrence_backend(self) -> str:
        """Return the captured recurrence backend."""
        return self.runtime.recurrence_backend

    @property
    def matrix_exp_dispatch(self) -> str:
        """Return the captured matrix-exponential dispatcher."""
        return self.runtime.matrix_exp_dispatch

    @property
    def captures_post_optimizer_step(self) -> bool:
        """Return whether optimizer constraints are part of the outer graph."""
        return self.outer.captures_post_optimizer_step

    @property
    def parallel_cuda_switch_frames(self) -> bool:
        """Return whether the two frame lanes overlap in the outer graph."""
        return self.outer.parallel_cuda_switch_frames

    @property
    def training_backend(self) -> str:
        """Return the provenance label written by the shared campaign trainer."""
        return "identity_aggressive_cuda_outer_graph"

    def activate(self) -> None:
        """Restore the nested runtime after an eager validation interval."""
        self.runtime.activate()

    def close(self) -> None:
        """Detach the reversible exclusive runtime lifecycle."""
        self.runtime.close()

    def destroy(self) -> None:
        """Permanently release outer and nested CUDA graph ownership."""
        if self._destroyed:
            return
        try:
            self.outer.destroy()
        finally:
            self.runtime.destroy()
            self._destroyed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


def prepare_aggressive_training(
    model: AlphabetBackbone,
    optimizer: AdamW,
    example_inputs: Tensor,
    example_labels: Tensor,
    *,
    parallel_cuda_switch_frames: bool = True,
    parallel_cuda_switch_lane_dag: bool | None = None,
    fused_c2_stem_training: bool = True,
    memory_efficient_lag124_training: bool = False,
    parallel_static_recurrence_training: bool | None = None,
    parallel_static_excitation_training: bool = False,
    saved_state_lag124_training: bool | None = None,
    fused_terminal_reader_local_training: bool = True,
    fused_terminal_reader_scan_training: bool = True,
    fused_writer_reader_local_training: bool = False,
    fused_writer_modal_reader_local_training: bool = False,
    fused_optimizer_tail: bool = True,
    compile_training_loss: bool = True,
    training_compile_mode: str = "max-autotune-no-cudagraphs",
    matrix_exp_forward_tf32: bool = False,
    direct_skew_matrix_exp_vjp: bool | None = None,
    loss_kind: Literal["cross_entropy", "binary_cross_entropy", "mse"] = "cross_entropy",
    **options: object,
) -> AlphabetTrainingRuntime:
    """Capture the complete CUDA 12.8 training step behind production APIs.

    BenchmarkAlphabetBackbone defaults to two independent SWITCH runtimes whose forward
    and backward frame lanes overlap. Each runtime owns a dedicated capture stream
    so CUDA-library workspaces cannot alias. Set ``parallel_cuda_switch_frames``
    to false for a conservative serial diagnostic. The outer graph owns AdamW and
    ``post_optimizer_step``; those stages are disabled in the nested exact-split
    runtime and captured exactly once by this wrapper. The truncated direct-skew
    VJP and TF32 matrix products remain explicit approximate opt-ins; the
    production defaults use strict FP32 and the exact block-matrix VJP.
    """
    from lnet.pac_cuda_outer_graph import EFP16ExactSplitOuterGraph  # noqa: PLC0415

    active_parallel_static = (
        not memory_efficient_lag124_training
        if parallel_static_recurrence_training is None
        else parallel_static_recurrence_training
    )
    active_saved_state = (
        not memory_efficient_lag124_training
        if saved_state_lag124_training is None
        else saved_state_lag124_training
    )
    active_lane_dag = (
        parallel_cuda_switch_frames
        if parallel_cuda_switch_lane_dag is None
        else parallel_cuda_switch_lane_dag
    )
    active_direct_skew_vjp = (
        False if direct_skew_matrix_exp_vjp is None else direct_skew_matrix_exp_vjp
    )
    aggressive_options = dict(options)
    aggressive_options.update(
        {
            "capture_post_optimizer_step": False,
            "copy_loss": False,
            "parallel_cuda_switch_frames": parallel_cuda_switch_frames,
            "parallel_cuda_switch_lane_dag": active_lane_dag,
            "fused_c2_stem_training": fused_c2_stem_training,
            "compile_training_loss": compile_training_loss,
            "training_compile_mode": training_compile_mode,
            "matrix_exp_forward_tf32": matrix_exp_forward_tf32,
            "direct_skew_matrix_exp_vjp": active_direct_skew_vjp,
            "loss_kind": loss_kind,
        }
    )
    runtime = prepare_exact_split_training(
        model,
        optimizer,
        example_inputs,
        example_labels,
        memory_efficient_lag124_training=memory_efficient_lag124_training,
        parallel_static_recurrence_training=active_parallel_static,
        parallel_static_excitation_training=(
            parallel_static_excitation_training and active_parallel_static
        ),
        saved_state_lag124_training=active_saved_state,
        fused_terminal_reader_local_training=fused_terminal_reader_local_training,
        fused_terminal_reader_scan_training=fused_terminal_reader_scan_training,
        fused_writer_reader_local_training=fused_writer_reader_local_training,
        fused_writer_modal_reader_local_training=(fused_writer_modal_reader_local_training),
        **aggressive_options,  # pyright: ignore[reportArgumentType]
    )
    try:
        if fused_optimizer_tail:
            from lnet.pac_cuda_fused_optimizer_runtime import (  # noqa: PLC0415
                install_outer_graph_fused_optimizer_tail,
            )

            install_outer_graph_fused_optimizer_tail(runtime)
        outer = EFP16ExactSplitOuterGraph(runtime, example_inputs, example_labels)
    except Exception:
        runtime.destroy()
        raise
    return AlphabetTrainingRuntime(runtime, outer)


__all__ = [
    "CAPTURE_POST_OPTIMIZER_STEP_BY_DEFAULT",
    "AlphabetBackbone",
    "AlphabetTrainingRuntime",
    "BenchmarkAlphabetBackbone",
    "prepare_aggressive_training",
    "prepare_compiled_inference",
    "prepare_exact_split_training",
    "prepare_maskless_inference",
]
