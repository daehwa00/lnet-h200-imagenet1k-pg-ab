"""Fused writer synthesis tail and terminal-reader drive for inference."""

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

_BATCHES: Final[tuple[int, ...]] = (32, 64)
_CHANNELS: Final[int] = 64
_PACKED_MODES: Final[int] = 32
_MAX_STEPS: Final[int] = 2048
_BLOCK_STEPS: Final[int] = 16
_PADDING: Final[int] = 8
_DILATION: Final[int] = 4
_RMS_EPSILON: Final[float] = torch.finfo(torch.float32).eps


def _validate_inputs(
    writer_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_local_weight: Tensor,
    reader_local_bias: Tensor,
    reader_norm_weight: Tensor,
    reader_drive_frame: Tensor,
) -> None:
    if writer_inputs.ndim != 3 or writer_inputs.shape[0] not in _BATCHES:
        message = "writer-reader fusion requires B32/B64 BTD inputs"
        raise ValueError(message)
    batch, steps, channels = writer_inputs.shape
    if not 1 <= steps <= _MAX_STEPS or channels != _CHANNELS:
        message = "writer-reader fusion requires 1<=T<=2048 and D64"
        raise ValueError(message)
    expected_shapes = (
        (batch, steps, _CHANNELS),
        (batch, steps, _PACKED_MODES),
        (_CHANNELS, _PACKED_MODES),
        (_CHANNELS,),
        (_CHANNELS,),
        (_CHANNELS, 1, 5),
        (_CHANNELS,),
        (_CHANNELS,),
        (_CHANNELS, _PACKED_MODES),
    )
    tensors = (
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_local_weight,
        reader_local_bias,
        reader_norm_weight,
        reader_drive_frame,
    )
    if tuple(tensor.shape for tensor in tensors) != expected_shapes:
        message = "writer-reader fusion received non-canonical parameter shapes"
        raise ValueError(message)
    if any(tensor.device != writer_inputs.device for tensor in tensors):
        message = "writer-reader fusion tensors must share one device"
        raise ValueError(message)
    if any(tensor.dtype != writer_inputs.dtype for tensor in tensors):
        message = "writer-reader fusion tensors must share one dtype"
        raise TypeError(message)


def reference_writer_terminal_reader_drive_inference(
    writer_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_local_weight: Tensor,
    reader_local_bias: Tensor,
    reader_norm_weight: Tensor,
    reader_drive_frame: Tensor,
) -> Tensor:
    """Return the packed terminal-reader drive from explicit writer components."""
    _validate_inputs(
        writer_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_local_weight,
        reader_local_bias,
        reader_norm_weight,
        reader_drive_frame,
    )
    modal = torch.matmul(modal_coordinates, synthesis_frame.transpose(0, 1))
    writer_stream = writer_inputs + layer_scale.view(1, 1, -1) * (
        modal + direct_scale.view(1, 1, -1) * writer_local
    )
    reader_local = functional.conv1d(
        writer_stream.transpose(1, 2),
        reader_local_weight,
        reader_local_bias,
        padding=_PADDING,
        dilation=_DILATION,
        groups=_CHANNELS,
    ).transpose(1, 2)
    encoded = functional.silu(reader_local)
    normalized = functional.rms_norm(
        encoded,
        (_CHANNELS,),
        reader_norm_weight,
        _RMS_EPSILON,
    )
    return torch.matmul(normalized, reader_drive_frame)


@triton.jit
def _writer_terminal_reader_drive_inference_kernel(
    writer_inputs,
    writer_local,
    modal_coordinates,
    synthesis_frame,
    direct_scale,
    layer_scale,
    reader_local_weight,
    reader_local_bias,
    reader_norm_weight,
    reader_drive_frame,
    packed_drive,
    steps: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    output_start = tl.program_id(1) * BLOCK_T
    source_step = output_start - 8 + tl.arange(0, 32)[:, None]
    channel = tl.arange(0, 64)[None, :]
    valid_source = (source_step >= 0) & (source_step < steps)
    safe_source = tl.where(valid_source, source_step, 0)
    writer_offset = (batch * steps + safe_source) * 64 + channel
    source_inputs = tl.load(
        writer_inputs + writer_offset,
        mask=valid_source,
        other=0.0,
    ).to(tl.float32)
    source_local = tl.load(
        writer_local + writer_offset,
        mask=valid_source,
        other=0.0,
    ).to(tl.float32)

    mode = tl.arange(0, 32)[None, :]
    modal_offset = (batch * steps + safe_source) * 32 + mode
    coordinates = tl.load(
        modal_coordinates + modal_offset,
        mask=valid_source,
        other=0.0,
    ).to(tl.float32)
    frame_mode = tl.arange(0, 32)[:, None]
    frame_channel = tl.arange(0, 64)[None, :]
    synthesis = tl.load(
        synthesis_frame + frame_channel * 32 + frame_mode,
    ).to(tl.float32)
    modal = tl.dot(coordinates, synthesis, input_precision="ieee")
    source_stream = source_inputs + tl.load(layer_scale + channel).to(
        tl.float32
    ) * (
        modal
        + tl.load(direct_scale + channel).to(tl.float32) * source_local
    )

    local = tl.load(reader_local_bias + channel).to(tl.float32)
    local = tl.broadcast_to(local, (BLOCK_T, 64))
    output_lane = tl.arange(0, BLOCK_T)[:, None]
    gather_columns = tl.zeros((BLOCK_T, 64), dtype=tl.int32)
    for tap in range(5):
        source_index = output_lane + tap * 4 + gather_columns
        source_tile = tl.gather(source_stream, source_index, axis=0)
        tap_weight = tl.load(
            reader_local_weight + channel * 5 + tap,
        ).to(tl.float32)
        local += source_tile * tap_weight

    activated = local * tl.sigmoid(local)
    mean_square = tl.sum(activated * activated, axis=1)[:, None] * (
        1.0 / 64.0
    )
    inverse_rms = tl.rsqrt(mean_square + 1.1920928955078125e-07)
    normalized = (
        activated
        * inverse_rms
        * tl.load(reader_norm_weight + channel).to(tl.float32)
    )
    drive_channel = tl.arange(0, 64)[:, None]
    packed_mode = tl.arange(0, 32)[None, :]
    drive_frame = tl.load(
        reader_drive_frame + drive_channel * 32 + packed_mode,
    ).to(tl.float32)
    drive = tl.dot(normalized, drive_frame, input_precision="tf32x3")
    output_step = output_start + tl.arange(0, BLOCK_T)[:, None]
    valid_output = output_step < steps
    drive_offset = (batch * steps + output_step) * 32 + packed_mode
    tl.store(packed_drive + drive_offset, drive, mask=valid_output)


def _launch_cuda(
    writer_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_local_weight: Tensor,
    reader_local_bias: Tensor,
    reader_norm_weight: Tensor,
    reader_drive_frame: Tensor,
) -> Tensor:
    active_writer_inputs = writer_inputs.contiguous()
    active_writer_local = writer_local.contiguous()
    active_modal_coordinates = modal_coordinates.contiguous()
    active_synthesis_frame = synthesis_frame.contiguous()
    active_direct_scale = direct_scale.contiguous()
    active_layer_scale = layer_scale.contiguous()
    active_reader_weight = reader_local_weight.contiguous()
    active_reader_bias = reader_local_bias.contiguous()
    active_reader_norm = reader_norm_weight.contiguous()
    active_reader_frame = reader_drive_frame.contiguous()
    batch, steps, _channels = active_writer_inputs.shape
    packed_drive = active_writer_inputs.new_empty((batch, steps, _PACKED_MODES))
    grid = (batch, triton.cdiv(steps, _BLOCK_STEPS))
    wrap_triton(_writer_terminal_reader_drive_inference_kernel)[grid](
        active_writer_inputs,
        active_writer_local,
        active_modal_coordinates,
        active_synthesis_frame,
        active_direct_scale,
        active_layer_scale,
        active_reader_weight,
        active_reader_bias,
        active_reader_norm,
        active_reader_frame,
        packed_drive,
        steps,
        BLOCK_T=_BLOCK_STEPS,
        num_warps=8,
    )
    return packed_drive


@triton_op("lnet::pac_writer_terminal_reader_drive_inference", mutates_args={})
def _writer_terminal_reader_drive_inference_op(
    writer_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_local_weight: Tensor,
    reader_local_bias: Tensor,
    reader_norm_weight: Tensor,
    reader_drive_frame: Tensor,
) -> Tensor:
    _validate_inputs(
        writer_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_local_weight,
        reader_local_bias,
        reader_norm_weight,
        reader_drive_frame,
    )
    if not writer_inputs.is_cuda:
        return reference_writer_terminal_reader_drive_inference(
            writer_inputs,
            writer_local,
            modal_coordinates,
            synthesis_frame,
            direct_scale,
            layer_scale,
            reader_local_weight,
            reader_local_bias,
            reader_norm_weight,
            reader_drive_frame,
        )
    return _launch_cuda(
        writer_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_local_weight,
        reader_local_bias,
        reader_norm_weight,
        reader_drive_frame,
    )


def writer_terminal_reader_drive_inference(
    writer_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_local_weight: Tensor,
    reader_local_bias: Tensor,
    reader_norm_weight: Tensor,
    reader_drive_frame: Tensor,
) -> Tensor:
    """Fuse the unused writer output boundary into the reader packed drive."""
    _validate_inputs(
        writer_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_local_weight,
        reader_local_bias,
        reader_norm_weight,
        reader_drive_frame,
    )
    tensors = (
        writer_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_local_weight,
        reader_local_bias,
        reader_norm_weight,
        reader_drive_frame,
    )
    needs_gradients = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in tensors
    )
    if writer_inputs.dtype != torch.float32 or needs_gradients:
        return reference_writer_terminal_reader_drive_inference(*tensors)
    return _writer_terminal_reader_drive_inference_op(*tensors)


__all__ = [
    "reference_writer_terminal_reader_drive_inference",
    "writer_terminal_reader_drive_inference",
]
