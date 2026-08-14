from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional

_EDGE_PROJECTION_WIDTH: Final[int] = 2
_LOCAL_KERNEL_SIZE: Final[int] = 5
_LOCAL_DILATION: Final[int] = 4
_LOCAL_PADDING: Final[int] = 8
_INV_SQRT_TWO: Final[float] = 0.7071067811865476


@triton.jit
def _edge_frame_stem_kernel(
    raw_inputs,
    projection_weight,
    local_weight,
    local_bias,
    output,
    input_steps: int,
    edge_steps: int,
    channels: int,
    block_steps: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    edge = tl.program_id(1) * block_steps + tl.arange(0, block_steps)[:, None]
    channel = tl.arange(0, block_channels)[None, :]
    valid_edge = edge < edge_steps
    valid_channel = channel < channels
    valid_output = valid_edge & valid_channel
    projection_low = tl.load(
        projection_weight + channel * 2,
        mask=valid_channel,
    ).to(tl.float32)
    projection_detail = tl.load(
        projection_weight + channel * 2 + 1,
        mask=valid_channel,
    ).to(tl.float32)
    local = tl.full((block_steps, block_channels), 0.0, tl.float32)

    for tap in range(5):
        source_edge = edge + tap * 4 - 8
        valid_source = valid_edge & (source_edge >= 0) & (source_edge < edge_steps)
        safe_edge = tl.where(valid_source, source_edge, 0)
        first_index = safe_edge
        second_index = safe_edge + 1
        raw_base = batch * input_steps
        first = tl.load(raw_inputs + raw_base + first_index, mask=valid_source).to(
            tl.float32
        )
        second = tl.load(raw_inputs + raw_base + second_index, mask=valid_source).to(
            tl.float32
        )
        first_degree_scale = tl.where(
            (first_index == 0) | (first_index == input_steps - 1),
            1.0,
            0.7071067811865476,
        )
        second_degree_scale = tl.where(
            (second_index == 0) | (second_index == input_steps - 1),
            1.0,
            0.7071067811865476,
        )
        scaled_first = first * first_degree_scale
        scaled_second = second * second_degree_scale
        low = 0.7071067811865476 * (scaled_first + scaled_second)
        detail = 0.7071067811865476 * (scaled_first - scaled_second)
        projected = projection_low * low + projection_detail * detail
        tap_weight = tl.load(
            local_weight + channel * 5 + tap,
            mask=valid_channel,
        ).to(tl.float32)
        local += projected * tap_weight

    local += tl.load(local_bias + channel, mask=valid_channel).to(tl.float32)
    activated = local * tl.sigmoid(local)
    output_offset = (batch * edge_steps + edge) * channels + channel
    tl.store(output + output_offset, activated, mask=valid_output)


@torch.library.triton_op("lnet::pac_edge_frame_stem", mutates_args={})
def _edge_frame_stem_op(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    _validate_inputs(raw_inputs, projection_weight, local_weight, local_bias)
    if not raw_inputs.is_cuda:
        return reference_edge_frame_stem(
            raw_inputs,
            projection_weight,
            local_weight,
            local_bias,
        )
    raw = raw_inputs.contiguous()
    projection = projection_weight.contiguous()
    local = local_weight.contiguous()
    bias = local_bias.contiguous()
    batch, input_steps, _ = raw.shape
    edge_steps = input_steps - 1
    channels = projection.shape[0]
    output = torch.empty(
        (batch, edge_steps, channels),
        device=raw.device,
        dtype=raw.dtype,
    )
    block_steps = min(triton.next_power_of_2(edge_steps), 32)
    block_channels = triton.next_power_of_2(channels)
    grid = (batch, triton.cdiv(edge_steps, block_steps))
    torch.library.wrap_triton(_edge_frame_stem_kernel)[grid](
        raw,
        projection,
        local,
        bias,
        output,
        input_steps,
        edge_steps,
        channels,
        block_steps=block_steps,
        block_channels=block_channels,
        num_warps=8,
    )
    return output


def edge_frame_stem_inference(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Fuse EFP edge analysis, projection, dilated depthwise map, and SiLU."""
    return _edge_frame_stem_op(
        raw_inputs,
        projection_weight,
        local_weight,
        local_bias,
    )


def reference_edge_frame_stem(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    _validate_inputs(raw_inputs, projection_weight, local_weight, local_bias)
    input_steps = raw_inputs.shape[1]
    degree = torch.ones(
        input_steps,
        device=raw_inputs.device,
        dtype=raw_inputs.dtype,
    )
    if input_steps > 2:
        degree[1:-1] = 2.0
    scaled = raw_inputs * degree.rsqrt().view(1, -1, 1)
    first = scaled[:, :-1]
    second = scaled[:, 1:]
    low = _INV_SQRT_TWO * (first + second)
    detail = _INV_SQRT_TWO * (first - second)
    projected = torch.matmul(torch.cat((low, detail), dim=-1), projection_weight.T)
    local = functional.conv1d(
        projected.transpose(1, 2),
        local_weight,
        local_bias,
        padding=_LOCAL_PADDING,
        dilation=_LOCAL_DILATION,
        groups=projection_weight.shape[0],
    )
    return functional.silu(local.transpose(1, 2))


def _validate_inputs(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> None:
    if raw_inputs.ndim != 3 or raw_inputs.shape[-1] != 1 or raw_inputs.shape[1] < 2:
        message = "edge-frame stem requires raw inputs with shape [batch,time>=2,1]"
        raise ValueError(message)
    if projection_weight.ndim != 2 or projection_weight.shape[1] != _EDGE_PROJECTION_WIDTH:
        message = "edge-frame projection weight must have shape [channels,2]"
        raise ValueError(message)
    channels = projection_weight.shape[0]
    if local_weight.shape != (channels, 1, _LOCAL_KERNEL_SIZE):
        message = "edge-frame local weight must have shape [channels,1,5]"
        raise ValueError(message)
    if local_bias.shape != (channels,):
        message = "edge-frame local bias must have shape [channels]"
        raise ValueError(message)
    for tensor in (projection_weight, local_weight, local_bias):
        if tensor.device != raw_inputs.device or tensor.dtype != raw_inputs.dtype:
            message = "edge-frame stem tensors must share device and dtype"
            raise ValueError(message)
    if raw_inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        message = "edge-frame stem supports fp16, bf16, and fp32"
        raise TypeError(message)
