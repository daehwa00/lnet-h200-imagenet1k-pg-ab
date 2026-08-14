"""Fused writer synthesis/residual and terminal-reader local training map.

The canonical BenchmarkAlphabetBackbone writer exposes packed modal coordinates and its
normalized local stream.  Ordinarily those values pass through a GEMM and a
pointwise residual before the terminal depthwise convolution reads the full
writer output back from global memory.  This operation keeps the synthesis,
residual, DWConv5-D4-P8, and SiLU producer in one Triton kernel.

The writer stream is still saved once for the exact backward.  It is not read
by the forward convolution, so the producer/consumer global-memory boundary is
elided without changing the training trajectory contract.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, N803
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional

from .pac_triton_terminal_reader_local_training import (
    _terminal_reader_local_training_backward_op,  # pyright: ignore[reportPrivateUsage]
)

_WIDTH: Final[int] = 64
_MODAL_WIDTH: Final[int] = 32
_KERNEL_SIZE: Final[int] = 5
_DILATION: Final[int] = 4
_PADDING: Final[int] = 8
_SHORT_BLOCK_TIME: Final[int] = 16
_SHORT_EXTENDED_TIME: Final[int] = 32
_SHORT_BLOCK_CHANNELS: Final[int] = 32
_LONG_BLOCK_TIME: Final[int] = 64
_LONG_EXTENDED_TIME: Final[int] = 128
_LONG_BLOCK_CHANNELS: Final[int] = 8


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _writer_reader_local_forward_kernel(  # noqa: PLR0915
    block_inputs,
    writer_local,
    modal_coordinates,
    synthesis_frame,
    direct_scale,
    layer_scale,
    reader_weight,
    reader_bias,
    encoded,
    preactivation,
    writer_stream,
    writer_update,
    steps: int,
    time_blocks: int,
    BLOCK_TIME: tl.constexpr,
    EXTENDED_TIME: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    MODAL_WIDTH: tl.constexpr,
    FULL_SEQUENCE: tl.constexpr,
) -> None:
    batch_time = tl.program_id(0)
    batch = batch_time // time_blocks
    time_block = batch_time - batch * time_blocks
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_valid = channel < WIDTH

    extended_lane = tl.arange(0, EXTENDED_TIME)
    source_time = (
        extended_lane if FULL_SEQUENCE else time_block * BLOCK_TIME - 8 + extended_lane
    )
    source_valid = (source_time >= 0) & (source_time < steps)
    safe_source = tl.where(source_valid, source_time, 0)
    modal = tl.arange(0, MODAL_WIDTH)

    coordinate_offsets = (
        (batch * steps + safe_source[:, None]) * MODAL_WIDTH + modal[None, :]
    )
    coordinates = tl.load(
        modal_coordinates + coordinate_offsets,
        mask=source_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    frame_offsets = channel[None, :] * MODAL_WIDTH + modal[:, None]
    frame = tl.load(
        synthesis_frame + frame_offsets,
        mask=channel_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    synthesized = tl.dot(coordinates, frame, input_precision="ieee")

    stream_offsets = (batch * steps + safe_source[:, None]) * WIDTH + channel[None, :]
    stream_mask = source_valid[:, None] & channel_valid[None, :]
    residual = tl.load(
        block_inputs + stream_offsets,
        mask=stream_mask,
        other=0.0,
    ).to(tl.float32)
    local = tl.load(
        writer_local + stream_offsets,
        mask=stream_mask,
        other=0.0,
    ).to(tl.float32)
    direct = tl.load(direct_scale + channel, mask=channel_valid, other=0.0).to(tl.float32)
    layer = tl.load(layer_scale + channel, mask=channel_valid, other=0.0).to(tl.float32)
    active_stream = residual + layer[None, :] * (
        synthesized + direct[None, :] * local
    )
    active_stream = tl.where(stream_mask, active_stream, 0.0)

    output_lane = tl.arange(0, BLOCK_TIME)
    output_time = time_block * BLOCK_TIME + output_lane
    output_valid = output_time < steps
    gather_columns = tl.zeros((BLOCK_TIME, BLOCK_CHANNELS), dtype=tl.int32)
    accumulated = tl.zeros((BLOCK_TIME, BLOCK_CHANNELS), dtype=tl.float32)
    for tap in range(5):
        if FULL_SEQUENCE:
            raw_source_index = output_lane[:, None] + tap * 4 - 8 + gather_columns
            source_index_valid = (raw_source_index >= 0) & (
                raw_source_index < EXTENDED_TIME
            )
            source_index = tl.where(source_index_valid, raw_source_index, 0)
        else:
            source_index = output_lane[:, None] + tap * 4 + gather_columns
        source_value = tl.gather(active_stream, source_index, axis=0)
        if FULL_SEQUENCE:
            source_value = tl.where(source_index_valid, source_value, 0.0)
        tap_weight = tl.load(
            reader_weight + channel * 5 + tap,
            mask=channel_valid,
            other=0.0,
        ).to(tl.float32)
        accumulated += source_value * tap_weight[None, :]
    bias = tl.load(reader_bias + channel, mask=channel_valid, other=0.0).to(tl.float32)
    active_preactivation = accumulated + bias[None, :]
    sigmoid = tl.sigmoid(active_preactivation)
    active_encoded = active_preactivation * sigmoid

    output_offsets = (batch * steps + output_time[:, None]) * WIDTH + channel[None, :]
    output_mask = output_valid[:, None] & channel_valid[None, :]
    tl.store(encoded + output_offsets, active_encoded, mask=output_mask)
    tl.store(preactivation + output_offsets, active_preactivation, mask=output_mask)
    if FULL_SEQUENCE:
        central_index = output_lane[:, None] + gather_columns
    else:
        central_index = output_lane[:, None] + 8 + gather_columns
    central_stream = tl.gather(active_stream, central_index, axis=0)
    tl.store(writer_stream + output_offsets, central_stream, mask=output_mask)
    central_synthesized = tl.gather(synthesized, central_index, axis=0)
    central_local = tl.gather(local, central_index, axis=0)
    central_update = central_synthesized + direct[None, :] * central_local
    tl.store(writer_update + output_offsets, central_update, mask=output_mask)


def _reference_components(
    block_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    modal = torch.matmul(modal_coordinates, synthesis_frame.transpose(0, 1))
    writer_update = modal + direct_scale.view(1, 1, -1) * writer_local
    writer_stream = block_inputs + layer_scale.view(1, 1, -1) * writer_update
    preactivation = functional.conv1d(
        writer_stream.transpose(1, 2),
        reader_weight,
        reader_bias,
        padding=_PADDING,
        dilation=_DILATION,
        groups=_WIDTH,
    ).transpose(1, 2)
    return functional.silu(preactivation), preactivation, writer_stream, writer_update


@torch.library.triton_op("lnet::pac_writer_reader_local_training_v3", mutates_args={})
def _writer_reader_local_training_op(
    block_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_inputs(
        block_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_weight,
        reader_bias,
    )
    if not block_inputs.is_cuda:
        return _reference_components(
            block_inputs,
            writer_local,
            modal_coordinates,
            synthesis_frame,
            direct_scale,
            layer_scale,
            reader_weight,
            reader_bias,
        )

    inputs = block_inputs.contiguous()
    local = writer_local.contiguous()
    coordinates = modal_coordinates.contiguous()
    frame = synthesis_frame.contiguous()
    direct = direct_scale.contiguous()
    layer = layer_scale.contiguous()
    weight = reader_weight.contiguous()
    bias = reader_bias.contiguous()
    batch, steps, _width = inputs.shape
    encoded = torch.empty_like(inputs)
    preactivation = torch.empty_like(inputs)
    writer_stream = torch.empty_like(inputs)
    writer_update = torch.empty_like(inputs)
    if _LONG_BLOCK_TIME <= steps <= 128:
        block_time = 128
        extended_time = 128
        block_channels = 16
        num_warps = 4
        full_sequence = True
    else:
        block_time = _SHORT_BLOCK_TIME
        extended_time = _SHORT_EXTENDED_TIME
        block_channels = _SHORT_BLOCK_CHANNELS
        num_warps = 4
        full_sequence = False
    time_blocks = triton.cdiv(steps, block_time)
    grid = (batch * time_blocks, triton.cdiv(_WIDTH, block_channels))
    torch.library.wrap_triton(_writer_reader_local_forward_kernel)[grid](
        inputs,
        local,
        coordinates,
        frame,
        direct,
        layer,
        weight,
        bias,
        encoded,
        preactivation,
        writer_stream,
        writer_update,
        steps,
        time_blocks,
        BLOCK_TIME=block_time,
        EXTENDED_TIME=extended_time,
        BLOCK_CHANNELS=block_channels,
        WIDTH=_WIDTH,
        MODAL_WIDTH=_MODAL_WIDTH,
        FULL_SEQUENCE=full_sequence,
        num_warps=num_warps,
    )
    return encoded, preactivation, writer_stream, writer_update


@torch.library.triton_op(
    "lnet::pac_writer_reader_local_training_backward_v3",
    mutates_args={},
)
def _writer_reader_local_training_backward_op(
    grad_encoded: Tensor,
    preactivation: Tensor,
    writer_stream: Tensor,
    writer_update: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    grad_stream, grad_weight, grad_bias = _terminal_reader_local_training_backward_op(
        grad_encoded,
        preactivation,
        writer_stream,
        reader_weight,
    )
    scaled_gradient = grad_stream * layer_scale.view(1, 1, -1)
    grad_block_inputs = grad_stream
    grad_writer_local = scaled_gradient * direct_scale.view(1, 1, -1)
    grad_modal_coordinates = torch.matmul(scaled_gradient, synthesis_frame)
    flattened_gradient = scaled_gradient.reshape(-1, _WIDTH)
    flattened_coordinates = modal_coordinates.reshape(-1, _MODAL_WIDTH)
    grad_synthesis_frame = flattened_gradient.transpose(0, 1) @ flattened_coordinates
    grad_direct_scale = (scaled_gradient * writer_local).sum(dim=(0, 1))
    grad_layer_scale = (grad_stream * writer_update).sum(dim=(0, 1))
    return (
        grad_block_inputs,
        grad_writer_local,
        grad_modal_coordinates,
        grad_synthesis_frame,
        grad_direct_scale,
        grad_layer_scale,
        grad_weight,
        grad_bias,
    )


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    (
        _block_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_weight,
        _reader_bias,
    ) = inputs
    _encoded, preactivation, writer_stream, writer_update = output
    ctx.mark_non_differentiable(preactivation, writer_stream, writer_update)
    ctx.save_for_backward(
        preactivation,
        writer_stream,
        writer_update,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_weight,
    )


def _backward(
    ctx: _AutogradContext,
    grad_encoded: Tensor,
    _grad_preactivation: Tensor | None,
    _grad_writer_stream: Tensor | None,
    _grad_writer_update: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del _grad_preactivation, _grad_writer_stream, _grad_writer_update
    (
        preactivation,
        writer_stream,
        writer_update,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_weight,
    ) = ctx.saved_tensors
    return _writer_reader_local_training_backward_op(
        grad_encoded,
        preactivation,
        writer_stream,
        writer_update,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_weight,
    )


torch.library.register_autograd(
    "lnet::pac_writer_reader_local_training_v3",
    _backward,
    setup_context=_setup_context,
)


def writer_reader_local_training(
    block_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> Tensor:
    """Return the encoded terminal stream without a forward writer-read boundary."""
    encoded, _preactivation, _writer_stream, _writer_update = (
        _writer_reader_local_training_op(
            block_inputs,
            writer_local,
            modal_coordinates,
            synthesis_frame,
            direct_scale,
            layer_scale,
            reader_weight,
            reader_bias,
        )
    )
    return encoded


def reference_writer_reader_local_training(
    block_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> Tensor:
    """Differentiable eager expression for parity tests and CPU fallback."""
    _validate_inputs(
        block_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_weight,
        reader_bias,
    )
    encoded, _preactivation, _writer_stream, _writer_update = _reference_components(
        block_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_weight,
        reader_bias,
    )
    return encoded


@triton.jit
def _writer_modal_reader_local_forward_kernel(
    block_inputs,
    writer_local,
    modal,
    direct_scale,
    layer_scale,
    reader_weight,
    reader_bias,
    encoded,
    preactivation,
    writer_stream,
    writer_update,
    steps: int,
    time_blocks: int,
    BLOCK_TIME: tl.constexpr,
    EXTENDED_TIME: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    FULL_SEQUENCE: tl.constexpr,
) -> None:
    batch_time = tl.program_id(0)
    batch = batch_time // time_blocks
    time_block = batch_time - batch * time_blocks
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_valid = channel < 64
    extended_lane = tl.arange(0, EXTENDED_TIME)
    source_time = (
        extended_lane if FULL_SEQUENCE else time_block * BLOCK_TIME - 8 + extended_lane
    )
    source_valid = (source_time >= 0) & (source_time < steps)
    safe_source = tl.where(source_valid, source_time, 0)
    source_offsets = (batch * steps + safe_source[:, None]) * 64 + channel[None, :]
    source_mask = source_valid[:, None] & channel_valid[None, :]
    residual = tl.load(
        block_inputs + source_offsets,
        mask=source_mask,
        other=0.0,
    ).to(tl.float32)
    local = tl.load(
        writer_local + source_offsets,
        mask=source_mask,
        other=0.0,
    ).to(tl.float32)
    active_modal = tl.load(
        modal + source_offsets,
        mask=source_mask,
        other=0.0,
    ).to(tl.float32)
    direct = tl.load(direct_scale + channel, mask=channel_valid, other=0.0).to(tl.float32)
    layer = tl.load(layer_scale + channel, mask=channel_valid, other=0.0).to(tl.float32)
    active_update = active_modal + direct[None, :] * local
    active_stream = residual + layer[None, :] * active_update
    active_stream = tl.where(source_mask, active_stream, 0.0)

    output_lane = tl.arange(0, BLOCK_TIME)
    output_time = time_block * BLOCK_TIME + output_lane
    output_valid = output_time < steps
    gather_columns = tl.zeros((BLOCK_TIME, BLOCK_CHANNELS), dtype=tl.int32)
    accumulated = tl.zeros((BLOCK_TIME, BLOCK_CHANNELS), dtype=tl.float32)
    for tap in range(5):
        if FULL_SEQUENCE:
            raw_source_index = output_lane[:, None] + tap * 4 - 8 + gather_columns
            source_index_valid = (raw_source_index >= 0) & (
                raw_source_index < EXTENDED_TIME
            )
            source_index = tl.where(source_index_valid, raw_source_index, 0)
        else:
            source_index = output_lane[:, None] + tap * 4 + gather_columns
        source_value = tl.gather(active_stream, source_index, axis=0)
        if FULL_SEQUENCE:
            source_value = tl.where(source_index_valid, source_value, 0.0)
        tap_weight = tl.load(
            reader_weight + channel * 5 + tap,
            mask=channel_valid,
            other=0.0,
        ).to(tl.float32)
        accumulated += source_value * tap_weight[None, :]
    bias = tl.load(reader_bias + channel, mask=channel_valid, other=0.0).to(tl.float32)
    active_preactivation = accumulated + bias[None, :]
    active_encoded = active_preactivation * tl.sigmoid(active_preactivation)

    output_offsets = (batch * steps + output_time[:, None]) * 64 + channel[None, :]
    output_mask = output_valid[:, None] & channel_valid[None, :]
    tl.store(encoded + output_offsets, active_encoded, mask=output_mask)
    tl.store(preactivation + output_offsets, active_preactivation, mask=output_mask)
    if FULL_SEQUENCE:
        central_index = output_lane[:, None] + gather_columns
    else:
        central_index = output_lane[:, None] + 8 + gather_columns
    tl.store(
        writer_stream + output_offsets,
        tl.gather(active_stream, central_index, axis=0),
        mask=output_mask,
    )
    tl.store(
        writer_update + output_offsets,
        tl.gather(active_update, central_index, axis=0),
        mask=output_mask,
    )


def _reference_modal_components(
    block_inputs: Tensor,
    writer_local: Tensor,
    modal: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    writer_update = modal + direct_scale.view(1, 1, -1) * writer_local
    writer_stream = block_inputs + layer_scale.view(1, 1, -1) * writer_update
    preactivation = functional.conv1d(
        writer_stream.transpose(1, 2),
        reader_weight,
        reader_bias,
        padding=_PADDING,
        dilation=_DILATION,
        groups=_WIDTH,
    ).transpose(1, 2)
    return functional.silu(preactivation), preactivation, writer_stream, writer_update


@torch.library.triton_op("lnet::pac_writer_modal_reader_local_training_v1", mutates_args={})
def _writer_modal_reader_local_training_op(
    block_inputs: Tensor,
    writer_local: Tensor,
    modal: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_modal_inputs(
        block_inputs,
        writer_local,
        modal,
        direct_scale,
        layer_scale,
        reader_weight,
        reader_bias,
    )
    if not block_inputs.is_cuda:
        return _reference_modal_components(
            block_inputs,
            writer_local,
            modal,
            direct_scale,
            layer_scale,
            reader_weight,
            reader_bias,
        )
    inputs = block_inputs.contiguous()
    local = writer_local.contiguous()
    active_modal = modal.contiguous()
    direct = direct_scale.contiguous()
    layer = layer_scale.contiguous()
    weight = reader_weight.contiguous()
    bias = reader_bias.contiguous()
    batch, steps, _width = inputs.shape
    encoded = torch.empty_like(inputs)
    preactivation = torch.empty_like(inputs)
    writer_stream = torch.empty_like(inputs)
    writer_update = torch.empty_like(inputs)
    if _LONG_BLOCK_TIME <= steps <= 128:
        block_time = 128
        extended_time = 128
        block_channels = 16
        num_warps = 4
        full_sequence = True
    else:
        block_time = _SHORT_BLOCK_TIME
        extended_time = _SHORT_EXTENDED_TIME
        block_channels = _SHORT_BLOCK_CHANNELS
        num_warps = 4
        full_sequence = False
    time_blocks = triton.cdiv(steps, block_time)
    grid = (batch * time_blocks, triton.cdiv(_WIDTH, block_channels))
    torch.library.wrap_triton(_writer_modal_reader_local_forward_kernel)[grid](
        inputs,
        local,
        active_modal,
        direct,
        layer,
        weight,
        bias,
        encoded,
        preactivation,
        writer_stream,
        writer_update,
        steps,
        time_blocks,
        BLOCK_TIME=block_time,
        EXTENDED_TIME=extended_time,
        BLOCK_CHANNELS=block_channels,
        FULL_SEQUENCE=full_sequence,
        num_warps=num_warps,
    )
    return encoded, preactivation, writer_stream, writer_update


@torch.library.triton_op(
    "lnet::pac_writer_modal_reader_local_training_backward_v1",
    mutates_args={},
)
def _writer_modal_reader_local_training_backward_op(
    grad_encoded: Tensor,
    preactivation: Tensor,
    writer_stream: Tensor,
    writer_update: Tensor,
    writer_local: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    grad_stream, grad_weight, grad_bias = _terminal_reader_local_training_backward_op(
        grad_encoded,
        preactivation,
        writer_stream,
        reader_weight,
    )
    scaled_gradient = grad_stream * layer_scale.view(1, 1, -1)
    return (
        grad_stream,
        scaled_gradient * direct_scale.view(1, 1, -1),
        scaled_gradient,
        (scaled_gradient * writer_local).sum(dim=(0, 1)),
        (grad_stream * writer_update).sum(dim=(0, 1)),
        grad_weight,
        grad_bias,
    )


def _writer_modal_setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    (
        _block_inputs,
        writer_local,
        _modal,
        direct_scale,
        layer_scale,
        reader_weight,
        _reader_bias,
    ) = inputs
    _encoded, preactivation, writer_stream, writer_update = output
    ctx.mark_non_differentiable(preactivation, writer_stream, writer_update)
    ctx.save_for_backward(
        preactivation,
        writer_stream,
        writer_update,
        writer_local,
        direct_scale,
        layer_scale,
        reader_weight,
    )


def _writer_modal_backward(
    ctx: _AutogradContext,
    grad_encoded: Tensor,
    _grad_preactivation: Tensor | None,
    _grad_writer_stream: Tensor | None,
    _grad_writer_update: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del _grad_preactivation, _grad_writer_stream, _grad_writer_update
    (
        preactivation,
        writer_stream,
        writer_update,
        writer_local,
        direct_scale,
        layer_scale,
        reader_weight,
    ) = ctx.saved_tensors
    return _writer_modal_reader_local_training_backward_op(
        grad_encoded,
        preactivation,
        writer_stream,
        writer_update,
        writer_local,
        direct_scale,
        layer_scale,
        reader_weight,
    )


torch.library.register_autograd(
    "lnet::pac_writer_modal_reader_local_training_v1",
    _writer_modal_backward,
    setup_context=_writer_modal_setup_context,
)


def writer_modal_reader_local_training(
    block_inputs: Tensor,
    writer_local: Tensor,
    modal: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> Tensor:
    """Fuse the writer residual and terminal local map after the synthesis GEMM."""
    encoded, _preactivation, _writer_stream, _writer_update = (
        _writer_modal_reader_local_training_op(
            block_inputs,
            writer_local,
            modal,
            direct_scale,
            layer_scale,
            reader_weight,
            reader_bias,
        )
    )
    return encoded


def reference_writer_modal_reader_local_training(
    block_inputs: Tensor,
    writer_local: Tensor,
    modal: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> Tensor:
    """Differentiable eager contract for the post-synthesis fusion."""
    _validate_modal_inputs(
        block_inputs,
        writer_local,
        modal,
        direct_scale,
        layer_scale,
        reader_weight,
        reader_bias,
    )
    encoded, _preactivation, _writer_stream, _writer_update = _reference_modal_components(
        block_inputs,
        writer_local,
        modal,
        direct_scale,
        layer_scale,
        reader_weight,
        reader_bias,
    )
    return encoded


def _validate_modal_inputs(
    block_inputs: Tensor,
    writer_local: Tensor,
    modal: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> None:
    if block_inputs.ndim != 3 or block_inputs.shape[-1] != _WIDTH:
        message = "writer-modal fusion requires block inputs shaped [batch,time,64]"
        raise ValueError(message)
    if writer_local.shape != block_inputs.shape or modal.shape != block_inputs.shape:
        message = "writer local and synthesized modal streams must match block inputs"
        raise ValueError(message)
    if direct_scale.shape != (_WIDTH,) or layer_scale.shape != (_WIDTH,):
        message = "writer residual scales must have shape [64]"
        raise ValueError(message)
    if reader_weight.shape != (_WIDTH, 1, _KERNEL_SIZE):
        message = "terminal reader weight must have shape [64,1,5]"
        raise ValueError(message)
    if reader_bias.shape != (_WIDTH,):
        message = "terminal reader bias must have shape [64]"
        raise ValueError(message)
    tensors = (
        block_inputs,
        writer_local,
        modal,
        direct_scale,
        layer_scale,
        reader_weight,
        reader_bias,
    )
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        message = "writer-modal fusion supports FP32 tensors only"
        raise TypeError(message)
    if any(tensor.device != block_inputs.device for tensor in tensors):
        message = "writer-modal fusion tensors must share one device"
        raise ValueError(message)
    if block_inputs.shape[1] < 1:
        message = "writer-modal fusion requires a positive time dimension"
        raise ValueError(message)


def _validate_inputs(  # noqa: C901
    block_inputs: Tensor,
    writer_local: Tensor,
    modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    reader_weight: Tensor,
    reader_bias: Tensor,
) -> None:
    if block_inputs.ndim != 3 or block_inputs.shape[-1] != _WIDTH:
        message = "writer-reader fusion requires block inputs shaped [batch,time,64]"
        raise ValueError(message)
    if writer_local.shape != block_inputs.shape:
        message = "writer local stream must match block inputs"
        raise ValueError(message)
    if modal_coordinates.shape != (*block_inputs.shape[:2], _MODAL_WIDTH):
        message = "writer modal coordinates must have shape [batch,time,32]"
        raise ValueError(message)
    if synthesis_frame.shape != (_WIDTH, _MODAL_WIDTH):
        message = "writer synthesis frame must have shape [64,32]"
        raise ValueError(message)
    if direct_scale.shape != (_WIDTH,) or layer_scale.shape != (_WIDTH,):
        message = "writer residual scales must have shape [64]"
        raise ValueError(message)
    if reader_weight.shape != (_WIDTH, 1, _KERNEL_SIZE):
        message = "terminal reader weight must have shape [64,1,5]"
        raise ValueError(message)
    if reader_bias.shape != (_WIDTH,):
        message = "terminal reader bias must have shape [64]"
        raise ValueError(message)
    tensors = (
        block_inputs,
        writer_local,
        modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        reader_weight,
        reader_bias,
    )
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        message = "writer-reader fusion supports FP32 tensors only"
        raise TypeError(message)
    if any(tensor.device != block_inputs.device for tensor in tensors):
        message = "writer-reader fusion tensors must share one device"
        raise ValueError(message)
    if block_inputs.shape[1] < 1:
        message = "writer-reader fusion requires a positive time dimension"
        raise ValueError(message)


__all__ = [
    "reference_writer_modal_reader_local_training",
    "reference_writer_reader_local_training",
    "writer_modal_reader_local_training",
    "writer_reader_local_training",
]
