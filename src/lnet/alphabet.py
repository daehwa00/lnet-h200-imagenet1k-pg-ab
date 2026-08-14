"""Final ALPHABET model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .alphabet_backbone import AlphabetBackbone
from .pac_triton_modal_affine_readout_inference import (
    modal_affine_readout_inference,
)
from .pac_triton_parallel_static_recurrence_lag124_training import (
    parallel_static_radial_log_recurrence_lag124_moments_only_inference,
)
from .pac_triton_terminal_reader_scan_inference import (
    terminal_reader_moments_inference,
)
from .pac_triton_writer_terminal_reader_inference import (
    writer_terminal_reader_drive_inference,
)

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACExperimentConfig


def radial_log_lag124_moments(moments: Tensor, modes: int) -> Tensor:
    """Apply log1p to R0 and the radial log map to complex R1/R2/R4."""
    if modes < 1:
        message = "modes must be positive"
        raise ValueError(message)
    if moments.shape[-1] != 7 * modes:
        message = f"expected 7M modal moments, got {moments.shape[-1]} coordinates for M={modes}"
        raise ValueError(message)

    transformed = [torch.log1p(moments[..., :modes].clamp_min(0.0))]
    epsilon = torch.finfo(moments.dtype).tiny
    for offset in (modes, 3 * modes, 5 * modes):
        real = moments[..., offset : offset + modes]
        imag = moments[..., offset + modes : offset + 2 * modes]
        radius = torch.sqrt((real.square() + imag.square()).clamp_min(epsilon))
        scale = torch.log1p(radius) / radius
        transformed.extend((scale * real, scale * imag))
    return torch.cat(transformed, dim=-1)


class _ModalAffineHead(nn.Module):
    def __init__(self, modes: int, output_dim: int) -> None:
        super().__init__()
        self.modes = modes
        self.mode_map = None
        self.classifier = nn.Linear(14 * modes, output_dim)

    def feature_group_slices(self) -> dict[str, tuple[slice, ...]]:
        return {
            "raw_modal": (slice(0, 14 * self.modes),),
            "mode_branch": (),
        }

    def forward(self, writer_moments: Tensor, reader_moments: Tensor) -> Tensor:
        return self.classifier(torch.cat((writer_moments, reader_moments), dim=-1))


class Alphabet(AlphabetBackbone):
    """Classify the final writer-reader radial-log descriptor with one affine head."""

    head: _ModalAffineHead
    final_norm: nn.RMSNorm | None

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        dwconv_kernel_size: int = 5,
        dwconv_dilation: int = 4,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            dwconv_kernel_size=dwconv_kernel_size,
            dwconv_dilation=dwconv_dilation,
            objective=objective,
        )
        original_classifier = self.head.classifier
        self.head = _ModalAffineHead(self.modes, output_dim)
        with torch.no_grad():
            self.head.classifier.weight.copy_(original_classifier.weight[:, self.model_dim :])
            if original_classifier.bias is not None and self.head.classifier.bias is not None:
                self.head.classifier.bias.copy_(original_classifier.bias)
        self.final_norm = None

        self.use_efp16_exact_split_training = False
        self.use_external_exact_split_training = True
        self.require_external_exact_split_training = True
        self.external_exact_split_preserve_eager_body = True
        self.use_fused_efp16_inference_readout = False
        self.use_fused_rmsnorm_mean_training = False
        self.use_fused_rmsnorm_mean_backward_training = False
        self.use_d32_rmsnorm_backward_training = False
        self.use_fused_terminal_reader_local_training = False
        self.use_fused_terminal_reader_scan_training = False
        self.use_fused_writer_reader_local_training = False
        self.use_fused_writer_modal_reader_local_training = False
        self.use_moments_only_terminal_reader_inference = True

        self.use_fused_writer_terminal_reader_inference = False
        self.use_fused_modal_affine_readout_inference = False

        # Equal-step calls retain token-lag moments. When time_delta is present,
        # the shared modal states are queried at physical-time lags instead.
        # The recurrence, poles, frames, RMSNorms, and synthesis are unchanged;
        # only the statistic returned from those states is unnormalized.
        for block in (self.forward_block, self.backward_block):
            block.log_energy = False
            block.normalize_autocorrelation = False
            block.fused_lag124_moments = False
            block.radial_log_lag124_moments = True
            block.physical_time_lag_moments = True

    def _terminal_reader_moments(
        self,
        first_stream: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> Tensor:
        """Skip the terminal D64 output when the head consumes only moments."""
        if (
            not self.use_moments_only_terminal_reader_inference
            or not self.use_fused_terminal_reader_scan_inference
            or self.training
            or torch.is_grad_enabled()
        ):
            return self._terminal_reader(
                first_stream,
                active_delta,
                active_observation,
                active_valid,
            )[1]
        second_projected = first_stream
        if not self._can_fuse_terminal_reader_scan_inference(
            second_projected,
            active_delta,
            active_observation,
            active_valid,
        ):
            return self._terminal_reader(
                first_stream,
                active_delta,
                active_observation,
                active_valid,
            )[1]
        reader = self.backward_block
        second_local_bias = self.second_local.bias
        reader_norm_weight = reader.norm.weight
        drive_frame = getattr(reader, "inference_drive_frame", None)
        decay_real = getattr(reader, "inference_decay_real", None)
        decay_imag = getattr(reader, "inference_decay_imag", None)
        if (
            second_local_bias is None
            or reader_norm_weight is None
            or drive_frame is None
            or decay_real is None
            or decay_imag is None
        ):
            message = "modal-only reader requires prepared static inference buffers"
            raise RuntimeError(message)
        return terminal_reader_moments_inference(
            second_projected,
            self.second_local.weight,
            second_local_bias,
            reader_norm_weight,
            drive_frame,
            decay_real,
            decay_imag,
            single_warp=reader.single_warp_static_recurrence_moments_inference,
            parallel_scan=self.use_parallel_terminal_reader_scan_inference,
            state_free_parallel_scan=(self.use_state_free_parallel_terminal_reader_scan_inference),
            radial_log=getattr(
                reader,
                "static_radial_log_lag124_recurrence_moments_inference",
                False,
            ),
        )

    @torch.no_grad()
    def prepare_for_inference_(
        self,
        *,
        sequence_length: int,
        batch_size: int,
        use_static_poles: bool = True,
    ) -> Alphabet:
        super().prepare_for_inference_(
            sequence_length=sequence_length,
            batch_size=batch_size,
            use_static_poles=use_static_poles,
        )
        self.forward_block.parallel_static_radial_log_recurrence_moments_inference = (
            use_static_poles
            and (
                (batch_size == 1 and sequence_length == 512)
                or (batch_size in (32, 64) and 1 <= sequence_length <= 2048)
            )
        )
        self.forward_block.fused_writer_rmsnorm_drive_inference = (
            use_static_poles and batch_size in (32, 64) and 1 <= sequence_length <= 2048
        )
        self.use_fused_writer_terminal_reader_inference = (
            use_static_poles and batch_size in (32, 64) and 1 <= sequence_length <= 2048
        )
        self.use_fused_modal_affine_readout_inference = (
            use_static_poles
            and batch_size in (32, 64)
            and 1 <= sequence_length <= 2048
            and self.head.mode_map is None
            and self.head.classifier.out_features == 5
        )
        return self

    def _represent_moments(
        self,
        moments: Tensor,
        block: object | None = None,
        *,
        metadata_free: bool = True,
    ) -> Tensor:
        if (
            metadata_free
            and block is not None
            and (
                (
                    getattr(
                        block,
                        "static_radial_log_lag124_recurrence_moments_inference",
                        False,
                    )
                    and not torch.is_grad_enabled()
                )
                or (
                    getattr(
                        block,
                        "parallel_static_radial_log_recurrence_moments_training",
                        False,
                    )
                    and torch.is_grad_enabled()
                )
            )
        ):
            return moments
        return radial_log_lag124_moments(moments, self.modes)

    def _can_fuse_writer_terminal_reader_inference(
        self,
        first_local: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> bool:
        writer = self.forward_block
        reader = self.backward_block
        return (
            self.use_fused_writer_terminal_reader_inference
            and not self.training
            and not torch.is_grad_enabled()
            and active_delta is None
            and active_observation is None
            and active_valid is None
            and first_local.is_cuda
            and first_local.dtype == torch.float32
            and first_local.is_contiguous()
            and first_local.ndim == 3
            and first_local.shape[0] in (32, 64)
            and 1 <= first_local.shape[1] <= 2048
            and first_local.shape[2] == 64
            and writer.fused_writer_rmsnorm_drive_inference
            and writer.direction == "forward"
            and writer.local is None
            and writer.synthesis_scale == 1.0
            and not writer.split_residual_scales
            and writer.direct_scale is not None
            and writer.layer_scale is not None
            and writer.static_radial_log_lag124_recurrence_moments_inference
            and writer.parallel_static_radial_log_recurrence_moments_inference
            and self.second_local.bias is not None
            and self.second_local.weight.shape == (64, 1, 5)
            and self.second_local.dilation == (4,)
            and self.second_local.padding == (8,)
            and self.second_local.groups == 64
            and reader.norm.weight is not None
            and getattr(reader, "inference_drive_frame", None) is not None
            and getattr(reader, "inference_decay_real", None) is not None
            and getattr(reader, "inference_decay_imag", None) is not None
            and reader.static_radial_log_lag124_recurrence_moments_inference
        )

    def _fused_writer_terminal_reader_moments(
        self,
        first_local: Tensor,
    ) -> tuple[Tensor, Tensor]:
        writer = self.forward_block
        reader = self.backward_block
        modal_coordinates, writer_local, first_moments = writer(
            first_local,
            metadata_prevalidated=True,
            return_inference_tail_components=True,
        )
        direct_scale = writer.direct_scale
        layer_scale = writer.layer_scale
        reader_bias = self.second_local.bias
        reader_norm = reader.norm.weight
        reader_drive = getattr(reader, "inference_drive_frame", None)
        decay_real = getattr(reader, "inference_decay_real", None)
        decay_imag = getattr(reader, "inference_decay_imag", None)
        if (
            direct_scale is None
            or layer_scale is None
            or reader_bias is None
            or reader_norm is None
            or reader_drive is None
            or decay_real is None
            or decay_imag is None
        ):
            message = "fused writer-reader inference lost prepared buffers"
            raise RuntimeError(message)
        packed_drive = writer_terminal_reader_drive_inference(
            first_local,
            writer_local,
            modal_coordinates,
            writer.synthesis_frame_matrix(),
            direct_scale,
            layer_scale,
            self.second_local.weight,
            reader_bias,
            reader_norm,
            reader_drive,
        )
        second_moments = parallel_static_radial_log_recurrence_lag124_moments_only_inference(
            decay_real,
            decay_imag,
            packed_drive,
            num_warps=4,
        )
        return first_moments, second_moments

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
        if self._can_fuse_writer_terminal_reader_inference(
            first_local,
            active_delta,
            active_observation,
            active_valid,
        ):
            first_moments, second_moments = self._fused_writer_terminal_reader_moments(first_local)
        else:
            first_stream, first_moments = self._writer(
                first_local,
                active_delta,
                active_observation,
                active_valid,
            )
            second_moments = self._terminal_reader_moments(
                first_stream,
                active_delta,
                None,
                active_valid,
            )
        metadata_free = active_delta is None and active_observation is None and active_valid is None
        writer_representation = self._represent_moments(
            first_moments,
            self.forward_block,
            metadata_free=metadata_free,
        )
        reader_representation = self._represent_moments(
            second_moments,
            self.backward_block,
            metadata_free=metadata_free,
        )
        classifier = self.head.classifier
        classifier_bias = classifier.bias
        if (
            self.use_fused_modal_affine_readout_inference
            and not self.training
            and not torch.is_grad_enabled()
            and writer_representation.is_cuda
            and writer_representation.dtype == torch.float32
            and writer_representation.shape[0] in (32, 64)
            and writer_representation.shape[1] == 112
            and reader_representation.shape == writer_representation.shape
            and self.head.mode_map is None
            and classifier.weight.shape == (5, 224)
            and classifier_bias is not None
        ):
            return modal_affine_readout_inference(
                writer_representation,
                reader_representation,
                classifier.weight,
                classifier_bias,
            )
        return self.head(writer_representation, reader_representation)


__all__ = [
    "Alphabet",
    "radial_log_lag124_moments",
]
