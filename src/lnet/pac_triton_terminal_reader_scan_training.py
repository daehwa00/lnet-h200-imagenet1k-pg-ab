"""Exact B32/B64 BenchmarkAlphabetBackbone terminal-reader training fusion.

The terminal local map is shared by two consumers: the readout needs its
activated D64 stream, while the reader scan needs RMS-normalized modal
excitations. A literal one-program kernel would therefore either recompute the
five-tap D64 convolution once per mode or retain the complete T x D tile across
an associative scan. Both choices cross the SM120 register/occupancy cliff.

This implementation keeps the required activated stream and one compact
``[B,T,2M]`` excitation boundary. One producer kernel fuses depthwise K5/D4/P8,
SiLU, RMSNorm, the D64->2M frame projection, and native-layout excitation
stores. The existing time-parallel recurrence consumes those excitations and
forms gamma, states, and lag-(1,2,4) moments in registers. Its custom backward
is joined to fused local/RMS/frame/gamma VJP kernels behind one opaque op.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false, reportPrivateUsage=false
# ruff: noqa: ANN001, FBT001, FBT003, N803
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional

from .pac_triton_parallel_static_recurrence_lag124_training import (
    _add_rn_fp32,
    _mul_rn_fp32,
    _parallel_lag124_backward_kernel,
    _parallel_lag124_excitation_forward_kernel,
    _reduce_batch_gradient_kernel,
    _sub_rn_fp32,
    parallel_static_excitation_recurrence_lag124_moments_only_training,
    parallel_static_radial_log_recurrence_lag124_moments_only_training,
)
from .pac_triton_terminal_reader_local_training import (
    _terminal_reader_local_grad_input_kernel,
    terminal_reader_local_training,
)

_BATCH: Final[int] = 32
_CHANNELS: Final[int] = 64
_MODES: Final[int] = 16
_PACKED_MODES: Final[int] = 32
_MOMENTS: Final[int] = 7 * _MODES
_KERNEL_SIZE: Final[int] = 5
_PRODUCER_FORWARD_BLOCK_STEPS: Final[int] = 8
_ENCODED_FORWARD_BLOCK_STEPS: Final[int] = 64
_PRODUCER_BLOCK_STEPS: Final[int] = 32
_PRODUCER_BACKWARD_WARPS: Final[int] = 4
_CONV_PARAMETER_COMPONENTS: Final[int] = 6
_FRAME_BLOCK_CHANNELS: Final[int] = 16
_FRAME_BLOCK_ITEMS: Final[int] = 32
_FRAME_SPLIT_ITEMS: Final[int] = 128
_PRODUCER_PARAMETER_COMPONENTS: Final[int] = _CHANNELS + 2 * _MODES
_RMS_EPSILON: Final[float] = torch.finfo(torch.float32).eps
_MOMENT_EPSILON: Final[float] = 1.0e-8
_MAX_STEPS: Final[int] = 2048


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def set_materialize_grads(self, value: bool) -> None: ...


@triton.jit
def _terminal_reader_producer_forward_kernel(
    inputs,
    local_weight,
    local_bias,
    norm_weight,
    frame,
    encoded,
    preactivation,
    inverse_rms,
    excitation_real,
    excitation_imag,
    steps: tl.constexpr,
    BLOCK_T: tl.constexpr,
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
        source = tl.load(inputs + source_offset, mask=valid_source, other=0.0).to(tl.float32)
        tap_weight = tl.load(local_weight + channel * 5 + tap).to(tl.float32)
        local += source * tap_weight

    activated = local * tl.sigmoid(local)
    output_offset = (batch * steps + step) * 64 + channel
    tl.store(encoded + output_offset, activated, mask=valid_step)
    tl.store(preactivation + output_offset, local, mask=valid_step)

    mean_square = tl.sum(activated * activated, axis=1)[:, None] * (1.0 / 64.0)
    inverse = tl.rsqrt(mean_square + 1.1920928955078125e-07)
    tl.store(
        inverse_rms + batch * steps + step,
        inverse,
        mask=valid_step,
    )
    normalized = activated * inverse * tl.load(norm_weight + channel).to(tl.float32)
    frame_channel = tl.arange(0, 64)[:, None]
    mode = tl.arange(0, 16)[None, :]
    frame_real = tl.load(frame + frame_channel * 32 + mode).to(tl.float32)
    frame_imag = tl.load(frame + frame_channel * 32 + mode + 16).to(tl.float32)
    modal_real = tl.dot(normalized, frame_real, input_precision="tf32x3")
    modal_imag = tl.dot(normalized, frame_imag, input_precision="tf32x3")
    excitation_offset = (batch * steps + step) * 16 + mode
    tl.store(excitation_real + excitation_offset, modal_real, mask=valid_step)
    tl.store(excitation_imag + excitation_offset, modal_imag, mask=valid_step)


@triton.jit
def _terminal_reader_producer_backward_kernel(
    grad_encoded,
    grad_packed_drive,
    inputs,
    preactivation,
    encoded,
    inverse_rms,
    excitation_real,
    excitation_imag,
    norm_weight,
    frame,
    gamma_real,
    gamma_imag,
    grad_local,
    parameter_partials,
    conv_partials,
    steps: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step_block = tl.program_id(1)
    step = step_block * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    channel = tl.arange(0, 64)[None, :]
    mode = tl.arange(0, 16)[None, :]
    valid_step = step < steps
    drive_offset = (batch * steps + step) * 32 + mode
    grad_drive_real = tl.load(
        grad_packed_drive + drive_offset,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    grad_drive_imag = tl.load(
        grad_packed_drive + drive_offset + 16,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    fixed_gr = tl.load(gamma_real + mode).to(tl.float32)
    fixed_gi = tl.load(gamma_imag + mode).to(tl.float32)
    grad_excitation_real = _add_rn_fp32(
        _mul_rn_fp32(fixed_gr, grad_drive_real),
        _mul_rn_fp32(fixed_gi, grad_drive_imag),
    )
    grad_excitation_imag = _sub_rn_fp32(
        _mul_rn_fp32(fixed_gr, grad_drive_imag),
        _mul_rn_fp32(fixed_gi, grad_drive_real),
    )

    frame_mode = tl.arange(0, 16)[:, None]
    frame_channel = tl.arange(0, 64)[None, :]
    frame_real_transposed = tl.load(
        frame + frame_channel * 32 + frame_mode,
    ).to(tl.float32)
    frame_imag_transposed = tl.load(
        frame + frame_channel * 32 + frame_mode + 16,
    ).to(tl.float32)
    grad_normalized = tl.dot(
        grad_excitation_real,
        frame_real_transposed,
        input_precision="tf32x3",
    )
    grad_normalized += tl.dot(
        grad_excitation_imag,
        frame_imag_transposed,
        input_precision="tf32x3",
    )

    value_offset = (batch * steps + step) * 64 + channel
    active_encoded = tl.load(encoded + value_offset, mask=valid_step, other=0.0).to(tl.float32)
    inverse = tl.load(
        inverse_rms + batch * steps + step,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    active_norm_weight = tl.load(norm_weight + channel).to(tl.float32)
    grad_scaled = grad_normalized * active_norm_weight
    radial = tl.sum(grad_scaled * active_encoded, axis=1)[:, None]
    grad_from_norm = inverse * (
        grad_scaled - active_encoded * (inverse * inverse * (1.0 / 64.0)) * radial
    )
    direct_gradient = tl.load(
        grad_encoded + value_offset,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    pre = tl.load(preactivation + value_offset, mask=valid_step, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(pre)
    local_gradient = (direct_gradient + grad_from_norm) * sigmoid * (1.0 + pre * (1.0 - sigmoid))
    tl.store(grad_local + value_offset, local_gradient, mask=valid_step)

    program = batch * tl.cdiv(steps, BLOCK_T) + step_block
    partial_base = program * 96
    norm_partial = tl.sum(grad_normalized * active_encoded * inverse, axis=0)
    tl.store(
        parameter_partials + partial_base + channel,
        norm_partial[None, :],
        mask=channel < 64,
    )
    active_excitation_real = tl.load(
        excitation_real + (batch * steps + step) * 16 + mode,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    active_excitation_imag = tl.load(
        excitation_imag + (batch * steps + step) * 16 + mode,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    gamma_real_partial = tl.sum(
        grad_drive_real * active_excitation_real + grad_drive_imag * active_excitation_imag,
        axis=0,
    )
    gamma_imag_partial = tl.sum(
        grad_drive_imag * active_excitation_real - grad_drive_real * active_excitation_imag,
        axis=0,
    )
    tl.store(
        parameter_partials + partial_base + 64 + mode,
        gamma_real_partial[None, :],
    )
    tl.store(
        parameter_partials + partial_base + 80 + mode,
        gamma_imag_partial[None, :],
    )
    conv_partial_base = program * 6 * 64 + channel
    for tap in range(5):
        source_step = step + tap * 4 - 8
        valid_source = valid_step & (source_step >= 0) & (source_step < steps)
        safe_source = tl.where(valid_source, source_step, 0)
        source = tl.load(
            inputs + (batch * steps + safe_source) * 64 + channel,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            conv_partials + conv_partial_base + tap * 64,
            tl.sum(local_gradient * source, axis=0)[None, :],
        )
    tl.store(
        conv_partials + conv_partial_base + 5 * 64,
        tl.sum(local_gradient, axis=0)[None, :],
    )


@triton.jit
def _encoded_reader_projection_forward_kernel(
    encoded,
    norm_weight,
    frame,
    inverse_rms,
    excitation_real,
    excitation_imag,
    steps: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    channel = tl.arange(0, 64)[None, :]
    valid_step = step < steps
    value_offset = (batch * steps + step) * 64 + channel
    active_encoded = tl.load(
        encoded + value_offset,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)

    mean_square = tl.sum(active_encoded * active_encoded, axis=1)[:, None] * (1.0 / 64.0)
    inverse = tl.rsqrt(mean_square + 1.1920928955078125e-07)
    tl.store(
        inverse_rms + batch * steps + step,
        inverse,
        mask=valid_step,
    )
    normalized = active_encoded * inverse * tl.load(norm_weight + channel).to(tl.float32)
    frame_channel = tl.arange(0, 64)[:, None]
    mode = tl.arange(0, 16)[None, :]
    frame_real = tl.load(frame + frame_channel * 32 + mode).to(tl.float32)
    frame_imag = tl.load(frame + frame_channel * 32 + mode + 16).to(tl.float32)
    modal_real = tl.dot(normalized, frame_real, input_precision="tf32x3")
    modal_imag = tl.dot(normalized, frame_imag, input_precision="tf32x3")
    excitation_offset = (batch * steps + step) * 16 + mode
    tl.store(excitation_real + excitation_offset, modal_real, mask=valid_step)
    tl.store(excitation_imag + excitation_offset, modal_imag, mask=valid_step)


@triton.jit
def _encoded_reader_projection_backward_kernel(
    grad_packed_drive,
    encoded,
    inverse_rms,
    excitation_real,
    excitation_imag,
    norm_weight,
    frame,
    gamma_real,
    gamma_imag,
    grad_encoded,
    parameter_partials,
    steps: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step_block = tl.program_id(1)
    step = step_block * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    channel = tl.arange(0, 64)[None, :]
    mode = tl.arange(0, 16)[None, :]
    valid_step = step < steps
    drive_offset = (batch * steps + step) * 32 + mode
    grad_drive_real = tl.load(
        grad_packed_drive + drive_offset,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    grad_drive_imag = tl.load(
        grad_packed_drive + drive_offset + 16,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    fixed_gr = tl.load(gamma_real + mode).to(tl.float32)
    fixed_gi = tl.load(gamma_imag + mode).to(tl.float32)
    grad_excitation_real = _add_rn_fp32(
        _mul_rn_fp32(fixed_gr, grad_drive_real),
        _mul_rn_fp32(fixed_gi, grad_drive_imag),
    )
    grad_excitation_imag = _sub_rn_fp32(
        _mul_rn_fp32(fixed_gr, grad_drive_imag),
        _mul_rn_fp32(fixed_gi, grad_drive_real),
    )

    frame_mode = tl.arange(0, 16)[:, None]
    frame_channel = tl.arange(0, 64)[None, :]
    frame_real_transposed = tl.load(
        frame + frame_channel * 32 + frame_mode,
    ).to(tl.float32)
    frame_imag_transposed = tl.load(
        frame + frame_channel * 32 + frame_mode + 16,
    ).to(tl.float32)
    grad_normalized = tl.dot(
        grad_excitation_real,
        frame_real_transposed,
        input_precision="tf32x3",
    )
    grad_normalized += tl.dot(
        grad_excitation_imag,
        frame_imag_transposed,
        input_precision="tf32x3",
    )

    value_offset = (batch * steps + step) * 64 + channel
    active_encoded = tl.load(
        encoded + value_offset,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    inverse = tl.load(
        inverse_rms + batch * steps + step,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    active_norm_weight = tl.load(norm_weight + channel).to(tl.float32)
    grad_scaled = grad_normalized * active_norm_weight
    radial = tl.sum(grad_scaled * active_encoded, axis=1)[:, None]
    active_grad_encoded = inverse * (
        grad_scaled - active_encoded * (inverse * inverse * (1.0 / 64.0)) * radial
    )
    tl.store(grad_encoded + value_offset, active_grad_encoded, mask=valid_step)

    program = batch * tl.cdiv(steps, BLOCK_T) + step_block
    partial_base = program * 96
    norm_partial = tl.sum(grad_normalized * active_encoded * inverse, axis=0)
    tl.store(
        parameter_partials + partial_base + channel,
        norm_partial[None, :],
        mask=channel < 64,
    )
    active_excitation_real = tl.load(
        excitation_real + (batch * steps + step) * 16 + mode,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    active_excitation_imag = tl.load(
        excitation_imag + (batch * steps + step) * 16 + mode,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    gamma_real_partial = tl.sum(
        grad_drive_real * active_excitation_real + grad_drive_imag * active_excitation_imag,
        axis=0,
    )
    gamma_imag_partial = tl.sum(
        grad_drive_imag * active_excitation_real - grad_drive_real * active_excitation_imag,
        axis=0,
    )
    tl.store(
        parameter_partials + partial_base + 64 + mode,
        gamma_real_partial[None, :],
    )
    tl.store(
        parameter_partials + partial_base + 80 + mode,
        gamma_imag_partial[None, :],
    )


@triton.jit
def _terminal_reader_frame_gradient_split_kernel(
    grad_packed_drive,
    encoded,
    inverse_rms,
    norm_weight,
    gamma_real,
    gamma_imag,
    partial_grad_frame,
    total_items: int,
    split_items: tl.constexpr,
) -> None:
    channel = tl.program_id(0) * 16 + tl.arange(0, 16)[:, None]
    part = tl.program_id(1)
    split = tl.program_id(2)
    mode = tl.arange(0, 16)[None, :]
    accumulator = tl.zeros((16, 16), tl.float32)
    split_start = split * split_items

    for item_offset in range(0, split_items, 32):
        item_columns = split_start + item_offset + tl.arange(0, 32)[None, :]
        valid_columns = item_columns < total_items
        active_encoded = tl.load(
            encoded + item_columns * 64 + channel,
            mask=valid_columns,
            other=0.0,
        ).to(tl.float32)
        inverse = tl.load(
            inverse_rms + item_columns,
            mask=valid_columns,
            other=0.0,
        ).to(tl.float32)
        normalized = active_encoded * inverse * tl.load(norm_weight + channel).to(tl.float32)

        item_rows = split_start + item_offset + tl.arange(0, 32)[:, None]
        valid_rows = item_rows < total_items
        drive_offset = item_rows * 32 + mode
        grad_drive_real = tl.load(
            grad_packed_drive + drive_offset,
            mask=valid_rows,
            other=0.0,
        ).to(tl.float32)
        grad_drive_imag = tl.load(
            grad_packed_drive + drive_offset + 16,
            mask=valid_rows,
            other=0.0,
        ).to(tl.float32)
        fixed_gr = tl.load(gamma_real + mode).to(tl.float32)
        fixed_gi = tl.load(gamma_imag + mode).to(tl.float32)
        grad_excitation_real = _add_rn_fp32(
            _mul_rn_fp32(fixed_gr, grad_drive_real),
            _mul_rn_fp32(fixed_gi, grad_drive_imag),
        )
        grad_excitation_imag = _sub_rn_fp32(
            _mul_rn_fp32(fixed_gr, grad_drive_imag),
            _mul_rn_fp32(fixed_gi, grad_drive_real),
        )
        grad_excitation = tl.where(
            part == 0,
            grad_excitation_real,
            grad_excitation_imag,
        )
        accumulator += tl.dot(normalized, grad_excitation, input_precision="tf32x3")

    coordinate = part * 16 + mode
    tl.store(
        partial_grad_frame + (split * 64 + channel) * 32 + coordinate,
        accumulator,
    )


@triton.jit
def _terminal_reader_frame_gradient_reduce_kernel(
    partial_grad_frame,
    grad_frame,
    frame_splits: tl.constexpr,
) -> None:
    channel = tl.program_id(0) * 8 + tl.arange(0, 8)[:, None]
    coordinate = tl.program_id(1) * 8 + tl.arange(0, 8)[None, :]
    accumulated = tl.zeros((8, 8), tl.float32)
    for split in range(frame_splits):
        accumulated += tl.load(
            partial_grad_frame + (split * 64 + channel) * 32 + coordinate,
        ).to(tl.float32)
    tl.store(grad_frame + channel * 32 + coordinate, accumulated)


@triton.jit
def _terminal_reader_parameter_reduce_kernel(
    conv_partials,
    parameter_partials,
    grad_local_weight,
    grad_local_bias,
    grad_norm_weight,
    grad_gamma_real,
    grad_gamma_imag,
    conv_splits: tl.constexpr,
    num_partials: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
) -> None:
    component = tl.program_id(0)
    scalar_offset = tl.arange(0, 1)
    partial = tl.arange(0, BLOCK_PARTIALS)
    if component < 384:
        channel = component // 6
        conv_component = component - channel * 6
        values = tl.load(
            conv_partials + partial * 6 * 64 + conv_component * 64 + channel,
            mask=partial < conv_splits,
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(values, axis=0)[None]
        if conv_component < 5:
            tl.store(
                grad_local_weight + channel * 5 + conv_component + scalar_offset,
                total,
            )
        else:
            tl.store(grad_local_bias + channel + scalar_offset, total)
    else:
        producer_component = component - 384
        values = tl.load(
            parameter_partials + partial * 96 + producer_component,
            mask=partial < num_partials,
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(values, axis=0)[None]
        if producer_component < 64:
            tl.store(grad_norm_weight + producer_component + scalar_offset, total)
        elif producer_component < 80:
            tl.store(
                grad_gamma_real + producer_component - 64 + scalar_offset,
                total,
            )
        else:
            tl.store(
                grad_gamma_imag + producer_component - 80 + scalar_offset,
                total,
            )


@triton.jit
def _encoded_reader_parameter_reduce_kernel(
    parameter_partials,
    grad_norm_weight,
    grad_gamma_real,
    grad_gamma_imag,
    num_partials: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
) -> None:
    component = tl.program_id(0)
    partial = tl.arange(0, BLOCK_PARTIALS)
    values = tl.load(
        parameter_partials + partial * 96 + component,
        mask=partial < num_partials,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(values, axis=0)[None]
    scalar_offset = tl.arange(0, 1)
    if component < 64:
        tl.store(grad_norm_weight + component + scalar_offset, total)
    elif component < 80:
        tl.store(grad_gamma_real + component - 64 + scalar_offset, total)
    else:
        tl.store(grad_gamma_imag + component - 80 + scalar_offset, total)


def _validate_inputs(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> None:
    if inputs.shape[0] not in (32, 64) or not 1 <= inputs.shape[1] <= _MAX_STEPS:
        message = "terminal-reader scan fusion requires B32/B64 and 1<=T<=2048"
        raise ValueError(message)
    if inputs.shape != (inputs.shape[0], inputs.shape[1], _CHANNELS):
        message = "terminal-reader scan fusion requires D64 BTD inputs"
        raise ValueError(message)
    expected_shapes = (
        (_CHANNELS, 1, _KERNEL_SIZE),
        (_CHANNELS,),
        (_CHANNELS,),
        (_CHANNELS, _PACKED_MODES),
        (_MODES,),
        (_MODES,),
        (_MODES,),
        (_MODES,),
    )
    tensors = (
        local_weight,
        local_bias,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    )
    if tuple(tensor.shape for tensor in tensors) != expected_shapes:
        message = "terminal-reader scan fusion received a non-canonical parameter shape"
        raise ValueError(message)
    if not inputs.is_cuda:
        message = "terminal-reader scan fusion is CUDA-only"
        raise RuntimeError(message)
    if inputs.dtype != torch.float32:
        message = "terminal-reader scan fusion supports FP32 only"
        raise TypeError(message)
    if any(tensor.device != inputs.device or tensor.dtype != inputs.dtype for tensor in tensors):
        message = "terminal-reader scan tensors must share one CUDA device and FP32 dtype"
        raise ValueError(message)


@triton_op("lnet::pac_terminal_reader_scan_forward_impl", mutates_args={})
def _forward_impl(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    radial_log: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    _validate_inputs(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    )
    active_inputs = inputs.contiguous()
    active_local_weight = local_weight.contiguous()
    active_local_bias = local_bias.contiguous()
    active_norm_weight = norm_weight.contiguous()
    active_frame = frame.contiguous()
    active_decay_real = decay_real.contiguous()
    active_decay_imag = decay_imag.contiguous()
    active_gamma_real = gamma_real.contiguous()
    active_gamma_imag = gamma_imag.contiguous()
    batch, steps, _channels = active_inputs.shape
    encoded = torch.empty_like(active_inputs)
    preactivation = torch.empty_like(active_inputs)
    inverse_rms = torch.empty((batch, steps), dtype=torch.float32, device=inputs.device)
    excitation_real = torch.empty((batch, steps, _MODES), dtype=torch.float32, device=inputs.device)
    excitation_imag = torch.empty_like(excitation_real)
    packed_states = torch.empty(
        (batch, steps, _PACKED_MODES), dtype=torch.float32, device=inputs.device
    )
    moments = torch.empty((batch, _MOMENTS), dtype=torch.float32, device=inputs.device)
    producer_grid = (batch, triton.cdiv(steps, _PRODUCER_FORWARD_BLOCK_STEPS))
    wrap_triton(_terminal_reader_producer_forward_kernel)[producer_grid](
        active_inputs,
        active_local_weight,
        active_local_bias,
        active_norm_weight,
        active_frame,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        steps,
        BLOCK_T=_PRODUCER_FORWARD_BLOCK_STEPS,
        num_warps=4,
    )
    wrap_triton(_parallel_lag124_excitation_forward_kernel)[(batch * _MODES,)](
        active_decay_real,
        active_decay_imag,
        active_gamma_real,
        active_gamma_imag,
        excitation_real,
        excitation_imag,
        packed_states,
        moments,
        steps,
        _MODES,
        _MOMENT_EPSILON,
        False,
        RADIAL_LOG=radial_log,
        BLOCK_T=triton.next_power_of_2(steps),
        num_warps=4,
    )
    return (
        encoded,
        moments,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )


@torch.library.custom_op("lnet::pac_terminal_reader_scan_training", mutates_args=())
def _forward_opaque(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _forward_impl(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        False,
    )


@_forward_opaque.register_fake
def _forward_fake(  # pyright: ignore[reportUnusedFunction]
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del local_weight, local_bias, norm_weight, frame, decay_imag, gamma_real, gamma_imag
    batch, steps, _channels = inputs.shape
    return (
        torch.empty_like(inputs),
        inputs.new_empty((batch, _MOMENTS)),
        torch.empty_like(inputs),
        inputs.new_empty((batch, steps)),
        inputs.new_empty((batch, steps, decay_real.numel())),
        inputs.new_empty((batch, steps, decay_real.numel())),
        inputs.new_empty((batch, steps, 2 * decay_real.numel())),
    )


@torch.library.custom_op(
    "lnet::pac_terminal_reader_radial_log_scan_training",
    mutates_args=(),
)
def _radial_forward_opaque(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _forward_impl(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        True,
    )


@_radial_forward_opaque.register_fake
def _radial_forward_fake(  # pyright: ignore[reportUnusedFunction]
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _forward_fake(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    )


@triton_op("lnet::pac_terminal_reader_scan_backward_impl", mutates_args={})
def _backward_impl(
    grad_encoded: Tensor,
    grad_moments: Tensor,
    inputs: Tensor,
    local_weight: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    encoded: Tensor,
    preactivation: Tensor,
    inverse_rms: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    packed_states: Tensor,
    radial_log: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    active_grad_encoded = grad_encoded.contiguous()
    active_grad_moments = grad_moments.contiguous()
    active_inputs = inputs.contiguous()
    active_local_weight = local_weight.contiguous()
    active_norm_weight = norm_weight.contiguous()
    active_frame = frame.contiguous()
    active_decay_real = decay_real.contiguous()
    active_decay_imag = decay_imag.contiguous()
    active_gamma_real = gamma_real.contiguous()
    active_gamma_imag = gamma_imag.contiguous()
    active_encoded = encoded.contiguous()
    active_preactivation = preactivation.contiguous()
    active_inverse_rms = inverse_rms.contiguous()
    active_excitation_real = excitation_real.contiguous()
    active_excitation_imag = excitation_imag.contiguous()
    active_states = packed_states.contiguous()
    batch, steps, _channels = active_inputs.shape
    total_items = batch * steps

    grad_packed_drive = torch.empty_like(active_states)
    per_batch_decay = torch.empty((batch, _PACKED_MODES), dtype=torch.float32, device=inputs.device)
    grad_decay_real = torch.empty_like(active_decay_real)
    grad_decay_imag = torch.empty_like(active_decay_imag)
    wrap_triton(_parallel_lag124_backward_kernel)[(batch * _MODES,)](
        active_decay_real,
        active_decay_imag,
        active_states,
        active_states,
        active_grad_moments,
        grad_packed_drive,
        per_batch_decay,
        steps,
        _MODES,
        _MOMENT_EPSILON,
        False,
        False,
        RADIAL_LOG=radial_log,
        BLOCK_T=triton.next_power_of_2(steps),
        num_warps=4,
    )
    wrap_triton(_reduce_batch_gradient_kernel)[(1,)](
        per_batch_decay,
        grad_decay_real,
        grad_decay_imag,
        batch,
        _MODES,
        BLOCK_M=_MODES,
        num_warps=1,
    )

    grad_local = torch.empty_like(active_inputs)
    num_producer_partials = batch * int(triton.cdiv(steps, _PRODUCER_BLOCK_STEPS))
    producer_partials = torch.empty(
        (num_producer_partials, _PRODUCER_PARAMETER_COMPONENTS),
        dtype=torch.float32,
        device=inputs.device,
    )
    conv_partials = torch.empty(
        (num_producer_partials, _CONV_PARAMETER_COMPONENTS, _CHANNELS),
        dtype=torch.float32,
        device=inputs.device,
    )
    producer_grid = (batch, triton.cdiv(steps, _PRODUCER_BLOCK_STEPS))
    wrap_triton(_terminal_reader_producer_backward_kernel)[producer_grid](
        active_grad_encoded,
        grad_packed_drive,
        active_inputs,
        active_preactivation,
        active_encoded,
        active_inverse_rms,
        active_excitation_real,
        active_excitation_imag,
        active_norm_weight,
        active_frame,
        active_gamma_real,
        active_gamma_imag,
        grad_local,
        producer_partials,
        conv_partials,
        steps,
        BLOCK_T=_PRODUCER_BLOCK_STEPS,
        num_warps=_PRODUCER_BACKWARD_WARPS,
    )

    grad_inputs = torch.empty_like(active_inputs)
    input_grid = (batch, triton.cdiv(steps, 8), triton.cdiv(_CHANNELS, 16))
    wrap_triton(_terminal_reader_local_grad_input_kernel)[input_grid](
        grad_local,
        active_local_weight,
        grad_inputs,
        steps,
        _CHANNELS,
        block_steps=8,
        block_channels=16,
        num_warps=2,
    )

    grad_local_weight = torch.empty_like(active_local_weight)
    grad_local_bias = torch.empty((_CHANNELS,), dtype=torch.float32, device=inputs.device)

    frame_splits = int(triton.cdiv(total_items, _FRAME_SPLIT_ITEMS))
    partial_grad_frame = torch.empty(
        (frame_splits, _CHANNELS, _PACKED_MODES),
        dtype=torch.float32,
        device=inputs.device,
    )
    wrap_triton(_terminal_reader_frame_gradient_split_kernel)[
        (
            triton.cdiv(_CHANNELS, _FRAME_BLOCK_CHANNELS),
            2,
            frame_splits,
        )
    ](
        grad_packed_drive,
        active_encoded,
        active_inverse_rms,
        active_norm_weight,
        active_gamma_real,
        active_gamma_imag,
        partial_grad_frame,
        total_items,
        split_items=_FRAME_SPLIT_ITEMS,
        num_warps=2,
    )
    grad_frame = torch.empty_like(active_frame)
    wrap_triton(_terminal_reader_frame_gradient_reduce_kernel)[
        (triton.cdiv(_CHANNELS, 8), triton.cdiv(_PACKED_MODES, 8))
    ](
        partial_grad_frame,
        grad_frame,
        frame_splits=frame_splits,
        num_warps=2,
    )

    grad_norm_weight = torch.empty_like(active_norm_weight)
    grad_gamma_real = torch.empty_like(active_gamma_real)
    grad_gamma_imag = torch.empty_like(active_gamma_imag)
    wrap_triton(_terminal_reader_parameter_reduce_kernel)[
        (_CHANNELS * _CONV_PARAMETER_COMPONENTS + _PRODUCER_PARAMETER_COMPONENTS,)
    ](
        conv_partials,
        producer_partials,
        grad_local_weight,
        grad_local_bias,
        grad_norm_weight,
        grad_gamma_real,
        grad_gamma_imag,
        num_producer_partials,
        num_producer_partials,
        BLOCK_PARTIALS=triton.next_power_of_2(num_producer_partials),
        num_warps=1,
    )
    return (
        grad_inputs,
        grad_local_weight,
        grad_local_bias,
        grad_norm_weight,
        grad_frame,
        grad_decay_real,
        grad_decay_imag,
        grad_gamma_real,
        grad_gamma_imag,
    )


@torch.library.custom_op("lnet::pac_terminal_reader_scan_backward", mutates_args=())
def _backward_opaque(
    grad_encoded: Tensor,
    grad_moments: Tensor,
    inputs: Tensor,
    local_weight: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    encoded: Tensor,
    preactivation: Tensor,
    inverse_rms: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    packed_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _backward_impl(
        grad_encoded,
        grad_moments,
        inputs,
        local_weight,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
        False,
    )


@_backward_opaque.register_fake
def _backward_fake(  # pyright: ignore[reportUnusedFunction]
    grad_encoded: Tensor,
    grad_moments: Tensor,
    inputs: Tensor,
    local_weight: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    encoded: Tensor,
    preactivation: Tensor,
    inverse_rms: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    packed_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        grad_encoded,
        grad_moments,
        norm_weight,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )
    return (
        torch.empty_like(inputs),
        torch.empty_like(local_weight),
        inputs.new_empty((_CHANNELS,)),
        inputs.new_empty((_CHANNELS,)),
        torch.empty_like(frame),
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(gamma_real),
        torch.empty_like(gamma_imag),
    )


@torch.library.custom_op(
    "lnet::pac_terminal_reader_radial_log_scan_backward",
    mutates_args=(),
)
def _radial_backward_opaque(
    grad_encoded: Tensor,
    grad_moments: Tensor,
    inputs: Tensor,
    local_weight: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    encoded: Tensor,
    preactivation: Tensor,
    inverse_rms: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    packed_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _backward_impl(
        grad_encoded,
        grad_moments,
        inputs,
        local_weight,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
        True,
    )


@_radial_backward_opaque.register_fake
def _radial_backward_fake(  # pyright: ignore[reportUnusedFunction]
    grad_encoded: Tensor,
    grad_moments: Tensor,
    inputs: Tensor,
    local_weight: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    encoded: Tensor,
    preactivation: Tensor,
    inverse_rms: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    packed_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _backward_fake(
        grad_encoded,
        grad_moments,
        inputs,
        local_weight,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
) -> None:
    (
        active_inputs,
        local_weight,
        _local_bias,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    ) = inputs
    (
        encoded,
        _moments,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    ) = output
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )
    ctx.save_for_backward(
        active_inputs,
        local_weight,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )


def _backward(
    ctx: _AutogradContext,
    grad_encoded: Tensor | None,
    grad_moments: Tensor | None,
    _grad_preactivation: Tensor | None,
    _grad_inverse_rms: Tensor | None,
    _grad_excitation_real: Tensor | None,
    _grad_excitation_imag: Tensor | None,
    _grad_packed_states: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        _grad_preactivation,
        _grad_inverse_rms,
        _grad_excitation_real,
        _grad_excitation_imag,
        _grad_packed_states,
    )
    (
        inputs,
        local_weight,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    ) = ctx.saved_tensors
    if grad_encoded is None:
        grad_encoded = torch.zeros_like(encoded)
    if grad_moments is None:
        grad_moments = encoded.new_zeros((encoded.shape[0], _MOMENTS))
    return _backward_opaque(
        grad_encoded,
        grad_moments,
        inputs,
        local_weight,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )


torch.library.register_autograd(
    "lnet::pac_terminal_reader_scan_training",
    _backward,
    setup_context=_setup_context,
)


def _radial_backward(
    ctx: _AutogradContext,
    grad_encoded: Tensor | None,
    grad_moments: Tensor | None,
    _grad_preactivation: Tensor | None,
    _grad_inverse_rms: Tensor | None,
    _grad_excitation_real: Tensor | None,
    _grad_excitation_imag: Tensor | None,
    _grad_packed_states: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        _grad_preactivation,
        _grad_inverse_rms,
        _grad_excitation_real,
        _grad_excitation_imag,
        _grad_packed_states,
    )
    (
        inputs,
        local_weight,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    ) = ctx.saved_tensors
    if grad_encoded is None:
        grad_encoded = torch.zeros_like(encoded)
    if grad_moments is None:
        grad_moments = encoded.new_zeros((encoded.shape[0], _MOMENTS))
    return _radial_backward_opaque(
        grad_encoded,
        grad_moments,
        inputs,
        local_weight,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        encoded,
        preactivation,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )


torch.library.register_autograd(
    "lnet::pac_terminal_reader_radial_log_scan_training",
    _radial_backward,
    setup_context=_setup_context,
)


def terminal_reader_scan_training(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the canonical activated reader stream and lag124 moments."""
    encoded, moments, *_hidden = _forward_opaque(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    )
    return encoded, moments


def terminal_reader_radial_log_scan_training(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the activated reader stream and fused radial-log moments."""
    encoded, moments, *_hidden = _radial_forward_opaque(
        inputs,
        local_weight,
        local_bias,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    )
    return encoded, moments


def reference_terminal_reader_scan_training(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    """Current fused-local plus excitation-scan training contract."""
    encoded = terminal_reader_local_training(inputs, local_weight, local_bias)
    normalized = functional.rms_norm(
        encoded,
        (_CHANNELS,),
        norm_weight,
        _RMS_EPSILON,
    )
    excitation = torch.matmul(normalized, frame)
    excitation_real, excitation_imag = excitation.chunk(2, dim=-1)
    moments = parallel_static_excitation_recurrence_lag124_moments_only_training(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
        num_warps=4,
    )
    return encoded, moments


def reference_terminal_reader_radial_log_scan_training(
    inputs: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    """Eager local/projection path with the canonical radial-log scan."""
    encoded = terminal_reader_local_training(inputs, local_weight, local_bias)
    normalized = functional.rms_norm(
        encoded,
        (_CHANNELS,),
        norm_weight,
        _RMS_EPSILON,
    )
    excitation = torch.matmul(normalized, frame)
    excitation_real, excitation_imag = excitation.chunk(2, dim=-1)
    input_real = gamma_real * excitation_real - gamma_imag * excitation_imag
    input_imag = gamma_real * excitation_imag + gamma_imag * excitation_real
    moments = parallel_static_radial_log_recurrence_lag124_moments_only_training(
        decay_real,
        decay_imag,
        torch.cat((input_real, input_imag), dim=-1),
        num_warps=4,
    )
    return encoded, moments


def _validate_encoded_reader_inputs(
    encoded: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> None:
    if encoded.shape[0] not in (32, 64) or not 1 <= encoded.shape[1] <= _MAX_STEPS:
        message = "encoded-reader scan fusion requires B32/B64 and 1<=T<=2048"
        raise ValueError(message)
    if encoded.shape != (encoded.shape[0], encoded.shape[1], _CHANNELS):
        message = "encoded-reader scan fusion requires D64 BTD inputs"
        raise ValueError(message)
    expected_shapes = (
        (_CHANNELS,),
        (_CHANNELS, _PACKED_MODES),
        (_MODES,),
        (_MODES,),
        (_MODES,),
        (_MODES,),
    )
    tensors = (
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    )
    if tuple(tensor.shape for tensor in tensors) != expected_shapes:
        message = "encoded-reader scan fusion received a non-canonical parameter shape"
        raise ValueError(message)
    if not encoded.is_cuda:
        message = "encoded-reader scan fusion is CUDA-only"
        raise RuntimeError(message)
    if encoded.dtype != torch.float32:
        message = "encoded-reader scan fusion supports FP32 only"
        raise TypeError(message)
    if any(tensor.device != encoded.device or tensor.dtype != encoded.dtype for tensor in tensors):
        message = "encoded-reader scan tensors must share one CUDA device and FP32 dtype"
        raise ValueError(message)


@triton_op("lnet::pac_encoded_reader_scan_forward_impl", mutates_args={})
def _encoded_reader_forward_impl(
    encoded: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _validate_encoded_reader_inputs(
        encoded,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    )
    active_encoded = encoded.contiguous()
    active_norm_weight = norm_weight.contiguous()
    active_frame = frame.contiguous()
    active_decay_real = decay_real.contiguous()
    active_decay_imag = decay_imag.contiguous()
    active_gamma_real = gamma_real.contiguous()
    active_gamma_imag = gamma_imag.contiguous()
    batch, steps, _channels = active_encoded.shape
    inverse_rms = torch.empty((batch, steps), dtype=torch.float32, device=encoded.device)
    excitation_real = torch.empty(
        (batch, steps, _MODES), dtype=torch.float32, device=encoded.device
    )
    excitation_imag = torch.empty_like(excitation_real)
    packed_states = torch.empty(
        (batch, steps, _PACKED_MODES),
        dtype=torch.float32,
        device=encoded.device,
    )
    moments = torch.empty((batch, _MOMENTS), dtype=torch.float32, device=encoded.device)
    producer_grid = (batch, triton.cdiv(steps, _ENCODED_FORWARD_BLOCK_STEPS))
    wrap_triton(_encoded_reader_projection_forward_kernel)[producer_grid](
        active_encoded,
        active_norm_weight,
        active_frame,
        inverse_rms,
        excitation_real,
        excitation_imag,
        steps,
        BLOCK_T=_ENCODED_FORWARD_BLOCK_STEPS,
        num_warps=4,
    )
    wrap_triton(_parallel_lag124_excitation_forward_kernel)[(batch * _MODES,)](
        active_decay_real,
        active_decay_imag,
        active_gamma_real,
        active_gamma_imag,
        excitation_real,
        excitation_imag,
        packed_states,
        moments,
        steps,
        _MODES,
        _MOMENT_EPSILON,
        False,
        RADIAL_LOG=False,
        BLOCK_T=triton.next_power_of_2(steps),
        num_warps=4,
    )
    return moments, inverse_rms, excitation_real, excitation_imag, packed_states


@torch.library.custom_op("lnet::pac_encoded_reader_scan_training", mutates_args=())
def _encoded_reader_forward_opaque(
    encoded: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _encoded_reader_forward_impl(
        encoded,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    )


@_encoded_reader_forward_opaque.register_fake
def _encoded_reader_forward_fake(  # pyright: ignore[reportUnusedFunction]
    encoded: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    del norm_weight, frame, decay_imag, gamma_real, gamma_imag
    batch, steps, _channels = encoded.shape
    return (
        encoded.new_empty((batch, _MOMENTS)),
        encoded.new_empty((batch, steps)),
        encoded.new_empty((batch, steps, decay_real.numel())),
        encoded.new_empty((batch, steps, decay_real.numel())),
        encoded.new_empty((batch, steps, 2 * decay_real.numel())),
    )


@triton_op("lnet::pac_encoded_reader_scan_backward_impl", mutates_args={})
def _encoded_reader_backward_impl(
    grad_moments: Tensor,
    encoded: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    inverse_rms: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    packed_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    active_grad_moments = grad_moments.contiguous()
    active_encoded = encoded.contiguous()
    active_norm_weight = norm_weight.contiguous()
    active_frame = frame.contiguous()
    active_decay_real = decay_real.contiguous()
    active_decay_imag = decay_imag.contiguous()
    active_gamma_real = gamma_real.contiguous()
    active_gamma_imag = gamma_imag.contiguous()
    active_inverse_rms = inverse_rms.contiguous()
    active_excitation_real = excitation_real.contiguous()
    active_excitation_imag = excitation_imag.contiguous()
    active_states = packed_states.contiguous()
    batch, steps, _channels = active_encoded.shape
    total_items = batch * steps

    grad_packed_drive = torch.empty_like(active_states)
    per_batch_decay = torch.empty(
        (batch, _PACKED_MODES),
        dtype=torch.float32,
        device=encoded.device,
    )
    grad_decay_real = torch.empty_like(active_decay_real)
    grad_decay_imag = torch.empty_like(active_decay_imag)
    wrap_triton(_parallel_lag124_backward_kernel)[(batch * _MODES,)](
        active_decay_real,
        active_decay_imag,
        active_states,
        active_states,
        active_grad_moments,
        grad_packed_drive,
        per_batch_decay,
        steps,
        _MODES,
        _MOMENT_EPSILON,
        False,
        False,
        RADIAL_LOG=False,
        BLOCK_T=triton.next_power_of_2(steps),
        num_warps=4,
    )
    wrap_triton(_reduce_batch_gradient_kernel)[(1,)](
        per_batch_decay,
        grad_decay_real,
        grad_decay_imag,
        batch,
        _MODES,
        BLOCK_M=_MODES,
        num_warps=1,
    )

    num_producer_partials = batch * int(
        triton.cdiv(steps, _PRODUCER_BLOCK_STEPS)
    )
    producer_partials = torch.empty(
        (num_producer_partials, _PRODUCER_PARAMETER_COMPONENTS),
        dtype=torch.float32,
        device=encoded.device,
    )
    grad_encoded = torch.empty_like(active_encoded)
    producer_grid = (batch, triton.cdiv(steps, _PRODUCER_BLOCK_STEPS))
    wrap_triton(_encoded_reader_projection_backward_kernel)[producer_grid](
        grad_packed_drive,
        active_encoded,
        active_inverse_rms,
        active_excitation_real,
        active_excitation_imag,
        active_norm_weight,
        active_frame,
        active_gamma_real,
        active_gamma_imag,
        grad_encoded,
        producer_partials,
        steps,
        BLOCK_T=_PRODUCER_BLOCK_STEPS,
        num_warps=4,
    )

    frame_splits = int(triton.cdiv(total_items, _FRAME_SPLIT_ITEMS))
    partial_grad_frame = torch.empty(
        (frame_splits, _CHANNELS, _PACKED_MODES),
        dtype=torch.float32,
        device=encoded.device,
    )
    wrap_triton(_terminal_reader_frame_gradient_split_kernel)[
        (
            triton.cdiv(_CHANNELS, _FRAME_BLOCK_CHANNELS),
            2,
            frame_splits,
        )
    ](
        grad_packed_drive,
        active_encoded,
        active_inverse_rms,
        active_norm_weight,
        active_gamma_real,
        active_gamma_imag,
        partial_grad_frame,
        total_items,
        split_items=_FRAME_SPLIT_ITEMS,
        num_warps=2,
    )
    grad_frame = torch.empty_like(active_frame)
    wrap_triton(_terminal_reader_frame_gradient_reduce_kernel)[
        (triton.cdiv(_CHANNELS, 8), triton.cdiv(_PACKED_MODES, 8))
    ](
        partial_grad_frame,
        grad_frame,
        frame_splits=frame_splits,
        num_warps=2,
    )

    grad_norm_weight = torch.empty_like(active_norm_weight)
    grad_gamma_real = torch.empty_like(active_gamma_real)
    grad_gamma_imag = torch.empty_like(active_gamma_imag)
    wrap_triton(_encoded_reader_parameter_reduce_kernel)[(_PRODUCER_PARAMETER_COMPONENTS,)](
        producer_partials,
        grad_norm_weight,
        grad_gamma_real,
        grad_gamma_imag,
        num_producer_partials,
        BLOCK_PARTIALS=triton.next_power_of_2(num_producer_partials),
        num_warps=1,
    )
    return (
        grad_encoded,
        grad_norm_weight,
        grad_frame,
        grad_decay_real,
        grad_decay_imag,
        grad_gamma_real,
        grad_gamma_imag,
    )


@torch.library.custom_op("lnet::pac_encoded_reader_scan_backward", mutates_args=())
def _encoded_reader_backward_opaque(
    grad_moments: Tensor,
    encoded: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    inverse_rms: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    packed_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _encoded_reader_backward_impl(
        grad_moments,
        encoded,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )


@_encoded_reader_backward_opaque.register_fake
def _encoded_reader_backward_fake(  # pyright: ignore[reportUnusedFunction]
    grad_moments: Tensor,
    encoded: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    inverse_rms: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    packed_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        grad_moments,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )
    return (
        torch.empty_like(encoded),
        torch.empty_like(norm_weight),
        torch.empty_like(frame),
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(gamma_real),
        torch.empty_like(gamma_imag),
    )


def _encoded_reader_setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
) -> None:
    (
        encoded,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    ) = inputs
    (
        _moments,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    ) = output
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )
    ctx.save_for_backward(
        encoded,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )


def _encoded_reader_backward(
    ctx: _AutogradContext,
    grad_moments: Tensor | None,
    _grad_inverse_rms: Tensor | None,
    _grad_excitation_real: Tensor | None,
    _grad_excitation_imag: Tensor | None,
    _grad_packed_states: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        _grad_inverse_rms,
        _grad_excitation_real,
        _grad_excitation_imag,
        _grad_packed_states,
    )
    (
        encoded,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    ) = ctx.saved_tensors
    if grad_moments is None:
        grad_moments = encoded.new_zeros((encoded.shape[0], _MOMENTS))
    return _encoded_reader_backward_opaque(
        grad_moments,
        encoded,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        inverse_rms,
        excitation_real,
        excitation_imag,
        packed_states,
    )


torch.library.register_autograd(
    "lnet::pac_encoded_reader_scan_training",
    _encoded_reader_backward,
    setup_context=_encoded_reader_setup_context,
)


def encoded_reader_scan_training(
    encoded: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> Tensor:
    """Return lag124 moments from an already-activated terminal stream."""
    moments, *_hidden = _encoded_reader_forward_opaque(
        encoded,
        norm_weight,
        frame,
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
    )
    return moments


def reference_encoded_reader_scan_training(
    encoded: Tensor,
    norm_weight: Tensor,
    frame: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> Tensor:
    """Eager RMSNorm/projection plus the canonical excitation scan."""
    normalized = functional.rms_norm(
        encoded,
        (_CHANNELS,),
        norm_weight,
        _RMS_EPSILON,
    )
    excitation = torch.matmul(normalized, frame)
    excitation_real, excitation_imag = excitation.chunk(2, dim=-1)
    return parallel_static_excitation_recurrence_lag124_moments_only_training(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
        num_warps=4,
    )


__all__ = [
    "encoded_reader_scan_training",
    "reference_encoded_reader_scan_training",
    "reference_terminal_reader_radial_log_scan_training",
    "reference_terminal_reader_scan_training",
    "terminal_reader_radial_log_scan_training",
    "terminal_reader_scan_training",
]
