"""Forward-only terminal-reader fusion for static-pole ALPHABET inference.

The terminal reader must return its activated D64 stream for the classifier
while also producing seven lag-(1,2,4) moments per mode.  The ordinary
inference path launches depthwise convolution, SiLU, RMSNorm, frame projection,
and the recurrence separately.  This module fuses the local map through the
static packed drive into one producer kernel, then reuses the exact verified
state-free lag124 recurrence kernel.

Unlike the training fusion, this path does not allocate preactivations,
inverse-RMS values, excitation copies, or packed recurrence states for a
backward pass that inference will never execute.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, N803
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional

from .pac_triton_parallel_static_recurrence_lag124_training import (
    parallel_static_radial_log_recurrence_lag124_moments_only_inference,
    parallel_static_recurrence_lag124_moments_only_inference,
    parallel_static_recurrence_lag124_moments_only_training,
)
from .pac_triton_recurrence_lag124 import (
    reference_static_recurrence_lag124_moments_only,
    static_recurrence_lag124_moments_only_inference,
)
from .pac_triton_radial_log_recurrence_lag124 import (
    reference_static_radial_log_recurrence_lag124_moments_only,
)

_BATCHES: Final[tuple[int, ...]] = (1, 32, 64)
_CHANNELS: Final[int] = 64
_MODES: Final[int] = 16
_PACKED_MODES: Final[int] = 2 * _MODES
_KERNEL_SIZE: Final[int] = 5
_DILATION: Final[int] = 4
_PADDING: Final[int] = 8
_MAX_STEPS: Final[int] = 2048
_BLOCK_STEPS: Final[int] = 8
_RMS_EPSILON: Final[float] = torch.finfo(torch.float32).eps
_MOMENT_EPSILON: Final[float] = 1.0e-8


def _validate_inputs(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
) -> None:
    if inputs.shape[0] not in _BATCHES or not 1 <= inputs.shape[1] <= _MAX_STEPS:
        message = "terminal-reader inference fusion requires B1/B32/B64 and 1<=T<=2048"
        raise ValueError(message)
    if inputs.shape != (inputs.shape[0], inputs.shape[1], _CHANNELS):
        message = "terminal-reader inference fusion requires D64 BTD inputs"
        raise ValueError(message)
    expected_shapes = (
        (_CHANNELS, 1, _KERNEL_SIZE),
        (_CHANNELS,),
        (_CHANNELS,),
        (_CHANNELS, _PACKED_MODES),
        (_MODES,),
        (_MODES,),
    )
    tensors = (
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
        decay_real,
        decay_imag,
    )
    if tuple(tensor.shape for tensor in tensors) != expected_shapes:
        message = "terminal-reader inference fusion received a non-canonical shape"
        raise ValueError(message)
    if any(tensor.device != inputs.device for tensor in tensors):
        message = "terminal-reader inference tensors must share one device"
        raise ValueError(message)
    if any(tensor.dtype != inputs.dtype for tensor in tensors):
        message = "terminal-reader inference tensors must share one dtype"
        raise TypeError(message)


def _reference_producer(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> tuple[Tensor, Tensor]:
    local = functional.conv1d(
        inputs.transpose(1, 2),
        local_weight,
        local_bias,
        padding=_PADDING,
        dilation=_DILATION,
        groups=_CHANNELS,
    ).transpose(1, 2)
    encoded = functional.silu(local)
    normalized = functional.rms_norm(
        encoded,
        (_CHANNELS,),
        norm_weight,
        _RMS_EPSILON,
    )
    return encoded, torch.matmul(normalized, drive_frame)


def reference_terminal_reader_scan_inference(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    *,
    radial_log: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return the canonical activated reader stream and static lag124 moments."""
    _validate_inputs(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
        decay_real,
        decay_imag,
    )
    encoded, packed_drive = _reference_producer(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
    )
    if radial_log:
        moments = reference_static_radial_log_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_drive,
        )
    else:
        moments = reference_static_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_drive,
            epsilon=_MOMENT_EPSILON,
        )
    return encoded, moments


def reference_terminal_reader_moments_inference(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    *,
    radial_log: bool = False,
) -> Tensor:
    """Return only the lag124 moments for modal-only classifier heads."""
    return reference_terminal_reader_scan_inference(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
        decay_real,
        decay_imag,
        radial_log=radial_log,
    )[1]


@triton.jit
def _terminal_reader_inference_producer_kernel(
    inputs,
    local_weight,
    local_bias,
    norm_weight,
    drive_frame,
    encoded,
    packed_drive,
    steps: tl.constexpr,
    BLOCK_T: tl.constexpr,
    STORE_ENCODED: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    channel = tl.arange(0, 64)[None, :]
    valid_step = step < steps
    local = tl.load(local_bias + channel).to(tl.float32)
    local = tl.broadcast_to(local, (BLOCK_T, 64))

    for tap in range(5):
        source_step = step + tap * 4 - 8
        valid_source = valid_step & (source_step >= 0) & (source_step < steps)
        safe_source = tl.where(valid_source, source_step, 0)
        source_offset = (batch * steps + safe_source) * 64 + channel
        source = tl.load(
            inputs + source_offset,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        tap_weight = tl.load(
            local_weight + channel * 5 + tap,
        ).to(tl.float32)
        local += source * tap_weight

    activated = local * tl.sigmoid(local)
    if STORE_ENCODED:
        encoded_offset = (batch * steps + step) * 64 + channel
        tl.store(encoded + encoded_offset, activated, mask=valid_step)

    mean_square = tl.sum(activated * activated, axis=1)[:, None] * (1.0 / 64.0)
    inverse_rms = tl.rsqrt(mean_square + 1.1920928955078125e-07)
    normalized = activated * inverse_rms * tl.load(norm_weight + channel).to(
        tl.float32
    )

    frame_channel = tl.arange(0, 64)[:, None]
    packed_mode = tl.arange(0, 32)[None, :]
    active_frame = tl.load(
        drive_frame + frame_channel * 32 + packed_mode,
    ).to(tl.float32)
    drive = tl.dot(normalized, active_frame, input_precision="tf32x3")
    drive_offset = (batch * steps + step) * 32 + packed_mode
    tl.store(packed_drive + drive_offset, drive, mask=valid_step)


def _launch_producer(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
    *,
    wrapped: bool,
) -> tuple[Tensor, Tensor]:
    active_inputs = inputs.contiguous()
    active_local_weight = local_weight.contiguous()
    active_local_bias = local_bias.contiguous()
    active_norm_weight = norm_weight.contiguous()
    active_drive_frame = drive_frame.contiguous()
    batch, steps, _channels = active_inputs.shape
    encoded = torch.empty_like(active_inputs)
    packed_drive = active_inputs.new_empty((batch, steps, _PACKED_MODES))
    grid = (batch, triton.cdiv(steps, _BLOCK_STEPS))
    if wrapped:
        wrap_triton(_terminal_reader_inference_producer_kernel)[grid](
            active_inputs,
            active_local_weight,
            active_local_bias,
            active_norm_weight,
            active_drive_frame,
            encoded,
            packed_drive,
            steps,
            BLOCK_T=_BLOCK_STEPS,
            STORE_ENCODED=True,
            num_warps=4,
        )
    else:
        _terminal_reader_inference_producer_kernel[grid](
            active_inputs,
            active_local_weight,
            active_local_bias,
            active_norm_weight,
            active_drive_frame,
            encoded,
            packed_drive,
            steps,
            BLOCK_T=_BLOCK_STEPS,
            STORE_ENCODED=True,
            num_warps=4,
        )
    return encoded, packed_drive


def _launch_moments_producer(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
    *,
    wrapped: bool,
) -> Tensor:
    """Produce the packed drive without allocating or storing the D64 stream."""
    active_inputs = inputs.contiguous()
    active_local_weight = local_weight.contiguous()
    active_local_bias = local_bias.contiguous()
    active_norm_weight = norm_weight.contiguous()
    active_drive_frame = drive_frame.contiguous()
    batch, steps, _channels = active_inputs.shape
    packed_drive = active_inputs.new_empty((batch, steps, _PACKED_MODES))
    grid = (batch, triton.cdiv(steps, _BLOCK_STEPS))
    if wrapped:
        wrap_triton(_terminal_reader_inference_producer_kernel)[grid](
            active_inputs,
            active_local_weight,
            active_local_bias,
            active_norm_weight,
            active_drive_frame,
            active_inputs,
            packed_drive,
            steps,
            BLOCK_T=_BLOCK_STEPS,
            STORE_ENCODED=False,
            num_warps=4,
        )
    else:
        _terminal_reader_inference_producer_kernel[grid](
            active_inputs,
            active_local_weight,
            active_local_bias,
            active_norm_weight,
            active_drive_frame,
            active_inputs,
            packed_drive,
            steps,
            BLOCK_T=_BLOCK_STEPS,
            STORE_ENCODED=False,
            num_warps=4,
        )
    return packed_drive


@triton_op("lnet::pac_terminal_reader_scan_inference_producer", mutates_args={})
def _producer_op(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> tuple[Tensor, Tensor]:
    return _launch_producer(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
        wrapped=True,
    )


@torch.library.custom_op(
    "lnet::pac_terminal_reader_scan_inference_producer_opaque",
    mutates_args=(),
)
def _producer_opaque(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> tuple[Tensor, Tensor]:
    return _producer_op(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
    )


@_producer_opaque.register_fake
def _producer_fake(  # pyright: ignore[reportUnusedFunction]
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> tuple[Tensor, Tensor]:
    del local_weight, local_bias, norm_weight, drive_frame
    batch, steps, _channels = inputs.shape
    return torch.empty_like(inputs), inputs.new_empty((batch, steps, _PACKED_MODES))


@triton_op(
    "lnet::pac_terminal_reader_moments_inference_producer",
    mutates_args={},
)
def _moments_producer_op(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> Tensor:
    return _launch_moments_producer(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
        wrapped=True,
    )


@torch.library.custom_op(
    "lnet::pac_terminal_reader_moments_inference_producer_opaque",
    mutates_args=(),
)
def _moments_producer_opaque(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> Tensor:
    return _moments_producer_op(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
    )


@_moments_producer_opaque.register_fake
def _moments_producer_fake(  # pyright: ignore[reportUnusedFunction]
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> Tensor:
    del local_weight, local_bias, norm_weight, drive_frame
    batch, steps, _channels = inputs.shape
    return inputs.new_empty((batch, steps, _PACKED_MODES))


def _moments_from_packed_drive(
    packed_drive: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    *,
    single_warp: bool,
    parallel_scan: bool,
    state_free_parallel_scan: bool,
    radial_log: bool,
) -> Tensor:
    if radial_log:
        return parallel_static_radial_log_recurrence_lag124_moments_only_inference(
            decay_real,
            decay_imag,
            packed_drive,
            num_warps=4,
        )
    if state_free_parallel_scan:
        return parallel_static_recurrence_lag124_moments_only_inference(
            decay_real,
            decay_imag,
            packed_drive,
            epsilon=_MOMENT_EPSILON,
            num_warps=4,
        )
    if parallel_scan:
        return parallel_static_recurrence_lag124_moments_only_training(
            decay_real,
            decay_imag,
            packed_drive,
            epsilon=_MOMENT_EPSILON,
            num_warps=4,
        )
    return static_recurrence_lag124_moments_only_inference(
        decay_real,
        decay_imag,
        packed_drive,
        epsilon=_MOMENT_EPSILON,
        single_warp=single_warp,
    )


def terminal_reader_scan_inference(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    *,
    single_warp: bool = False,
    parallel_scan: bool = False,
    state_free_parallel_scan: bool = False,
    radial_log: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run the exact forward-only terminal reader for static FP32 inference."""
    _validate_inputs(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
        decay_real,
        decay_imag,
    )
    needs_gradients = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (
            inputs,
            local_weight,
            local_bias,
            norm_weight,
            drive_frame,
            decay_real,
            decay_imag,
        )
    )
    if not inputs.is_cuda or inputs.dtype != torch.float32 or needs_gradients:
        return reference_terminal_reader_scan_inference(
            inputs,
            local_weight,
            local_bias,
            norm_weight,
            drive_frame,
            decay_real,
            decay_imag,
            radial_log=radial_log,
        )
    if torch.compiler.is_compiling():
        encoded, packed_drive = _producer_opaque(
            inputs,
            local_weight,
            local_bias,
            norm_weight,
            drive_frame,
        )
    else:
        encoded, packed_drive = _launch_producer(
            inputs,
            local_weight,
            local_bias,
            norm_weight,
            drive_frame,
            wrapped=False,
        )
    moments = _moments_from_packed_drive(
        packed_drive,
        decay_real,
        decay_imag,
        single_warp=single_warp,
        parallel_scan=parallel_scan,
        state_free_parallel_scan=state_free_parallel_scan,
        radial_log=radial_log,
    )
    return encoded, moments


def terminal_reader_moments_inference(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    *,
    single_warp: bool = False,
    parallel_scan: bool = False,
    state_free_parallel_scan: bool = False,
    radial_log: bool = False,
) -> Tensor:
    """Run the terminal reader without materializing its unused D64 output."""
    _validate_inputs(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        drive_frame,
        decay_real,
        decay_imag,
    )
    needs_gradients = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (
            inputs,
            local_weight,
            local_bias,
            norm_weight,
            drive_frame,
            decay_real,
            decay_imag,
        )
    )
    if not inputs.is_cuda or inputs.dtype != torch.float32 or needs_gradients:
        return reference_terminal_reader_moments_inference(
            inputs,
            local_weight,
            local_bias,
            norm_weight,
            drive_frame,
            decay_real,
            decay_imag,
            radial_log=radial_log,
        )
    if torch.compiler.is_compiling():
        packed_drive = _moments_producer_opaque(
            inputs,
            local_weight,
            local_bias,
            norm_weight,
            drive_frame,
        )
    else:
        packed_drive = _launch_moments_producer(
            inputs,
            local_weight,
            local_bias,
            norm_weight,
            drive_frame,
            wrapped=False,
        )
    return _moments_from_packed_drive(
        packed_drive,
        decay_real,
        decay_imag,
        single_warp=single_warp,
        parallel_scan=parallel_scan,
        state_free_parallel_scan=state_free_parallel_scan,
        radial_log=radial_log,
    )


__all__ = [
    "reference_terminal_reader_moments_inference",
    "reference_terminal_reader_scan_inference",
    "terminal_reader_moments_inference",
    "terminal_reader_scan_inference",
]
