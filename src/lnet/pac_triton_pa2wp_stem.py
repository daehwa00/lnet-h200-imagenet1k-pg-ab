from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001
import math
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional

_STEM_KERNEL_SIZE: Final[int] = 9
_INV_SQRT_TWO: Final[float] = 1.0 / math.sqrt(2.0)


@triton.jit
def _pa2wp_stem_kernel(
    raw_inputs,
    stem_weight,
    stem_bias,
    output,
    input_steps: int,
    output_steps: int,
    channels: int,
    block_steps: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    raw_batch_count = tl.num_programs(0)
    step = tl.program_id(1) * block_steps + tl.arange(0, block_steps)[:, None]
    channel = tl.arange(0, block_channels)[None, :]
    valid_step = step < output_steps
    valid_channel = channel < channels
    a0 = tl.full((block_steps, block_channels), 0.0, tl.float32)
    a1 = tl.full((block_steps, block_channels), 0.0, tl.float32)
    a2 = tl.full((block_steps, block_channels), 0.0, tl.float32)

    for tap in range(9):
        pair = 2 * step + tap - 8
        raw_index0 = 2 * pair
        raw_index1 = raw_index0 + 1
        raw_index2 = raw_index0 + 2
        raw_base = batch * input_steps
        x0 = tl.load(
            raw_inputs + raw_base + raw_index0,
            mask=valid_step & (raw_index0 >= 0) & (raw_index0 < input_steps),
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            raw_inputs + raw_base + raw_index1,
            mask=valid_step & (raw_index1 >= 0) & (raw_index1 < input_steps),
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            raw_inputs + raw_base + raw_index2,
            mask=valid_step & (pair >= 0) & (raw_index2 >= 0) & (raw_index2 < input_steps),
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            stem_weight + channel * 9 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        a0 += x0 * weight
        a1 += x1 * weight
        a2 += x2 * weight

    bias = tl.load(stem_bias + channel, mask=valid_channel, other=0.0).to(tl.float32)
    ordinary_low = 0.7071067811865476 * (a0 + a1) + bias
    ordinary_detail = 0.7071067811865476 * (a0 - a1) + bias
    shifted_low = 0.7071067811865476 * (a1 + a2) + bias
    shifted_detail = 0.7071067811865476 * (a1 - a2) + bias
    ordinary_low *= tl.sigmoid(ordinary_low)
    ordinary_detail *= tl.sigmoid(ordinary_detail)
    shifted_low *= tl.sigmoid(shifted_low)
    shifted_detail *= tl.sigmoid(shifted_detail)

    valid_output = valid_step & valid_channel
    base = step * channels + channel
    batch_stride = output_steps * channels
    tl.store(
        output + batch * batch_stride + base,
        ordinary_low,
        mask=valid_output,
    )
    tl.store(
        output + (raw_batch_count + batch) * batch_stride + base,
        ordinary_detail,
        mask=valid_output,
    )
    tl.store(
        output + (2 * raw_batch_count + batch) * batch_stride + base,
        shifted_low,
        mask=valid_output,
    )
    tl.store(
        output + (3 * raw_batch_count + batch) * batch_stride + base,
        shifted_detail,
        mask=valid_output,
    )


@torch.library.triton_op("lnet::pac_pa2wp_stem", mutates_args={})
def _pa2wp_stem_op(raw_inputs: Tensor, stem_weight: Tensor, stem_bias: Tensor) -> Tensor:
    _validate_inputs(raw_inputs, stem_weight, stem_bias)
    if not raw_inputs.is_cuda:
        return reference_pa2wp_stem(raw_inputs, stem_weight, stem_bias)
    raw = raw_inputs.contiguous()
    weight = stem_weight.contiguous()
    bias = stem_bias.contiguous()
    batch, input_steps, _ = raw.shape
    channels = weight.shape[0]
    output_steps = (input_steps + 3) // 4
    output = torch.empty(
        (4 * batch, output_steps, channels),
        device=raw.device,
        dtype=raw.dtype,
    )
    block_steps = min(triton.next_power_of_2(output_steps), 16)
    block_channels = triton.next_power_of_2(channels)
    grid = (batch, triton.cdiv(output_steps, block_steps))
    torch.library.wrap_triton(_pa2wp_stem_kernel)[grid](
        raw,
        weight,
        bias,
        output,
        input_steps,
        output_steps,
        channels,
        block_steps=block_steps,
        block_channels=block_channels,
        num_warps=4,
    )
    return output


def pa2wp_stem_inference(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    stem_bias: Tensor,
) -> Tensor:
    """Fuse both PA2WP phases, Haar bands, packing, and the causal K9 stem."""
    return _pa2wp_stem_op(raw_inputs, stem_weight, stem_bias)


def reference_pa2wp_stem(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    stem_bias: Tensor,
) -> Tensor:
    _validate_inputs(raw_inputs, stem_weight, stem_bias)
    ordinary_low, ordinary_detail = _haar(raw_inputs)
    shifted_low, shifted_detail = _haar(raw_inputs[:, 1:])
    bands = torch.cat((ordinary_low, ordinary_detail, shifted_low, shifted_detail), dim=0)
    padded = functional.pad(bands.transpose(1, 2), (_STEM_KERNEL_SIZE - 1, 0))
    encoded = functional.conv1d(padded, stem_weight, stem_bias, stride=2)
    return functional.silu(encoded.transpose(1, 2))


def _haar(inputs: Tensor) -> tuple[Tensor, Tensor]:
    if inputs.shape[1] % 2:
        inputs = functional.pad(inputs, (0, 0, 0, 1))
    first = inputs[:, 0::2]
    second = inputs[:, 1::2]
    return (
        _INV_SQRT_TWO * (first + second),
        _INV_SQRT_TWO * (first - second),
    )


def _validate_inputs(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    stem_bias: Tensor,
) -> None:
    if raw_inputs.ndim != 3 or raw_inputs.shape[-1] != 1 or raw_inputs.shape[1] < 2:
        message = "PA2WP stem requires raw inputs with shape [batch,time>=2,1]"
        raise ValueError(message)
    if raw_inputs.shape[1] % 2:
        message = "fused PA2WP stem currently requires an even raw length"
        raise ValueError(message)
    if stem_weight.ndim != 3 or stem_weight.shape[1:] != (1, _STEM_KERNEL_SIZE):
        message = "PA2WP stem weight must have shape [channels,1,9]"
        raise ValueError(message)
    if stem_bias.shape != (stem_weight.shape[0],):
        message = "PA2WP stem bias must have shape [channels]"
        raise ValueError(message)
    for tensor in (stem_weight, stem_bias):
        if tensor.device != raw_inputs.device or tensor.dtype != raw_inputs.dtype:
            message = "PA2WP stem tensors must share device and dtype"
            raise ValueError(message)
    if raw_inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        message = "PA2WP stem supports fp16, bf16, and fp32"
        raise TypeError(message)
