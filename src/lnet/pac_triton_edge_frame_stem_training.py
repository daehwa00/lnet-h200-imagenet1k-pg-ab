from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001
import math
from typing import Final, Literal, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional

_EDGE_PROJECTION_WIDTH: Final[int] = 2
_EDGE_PROJECTION_WIDTH_C2: Final[int] = 4
_LOCAL_KERNEL_SIZE: Final[int] = 5
_LOCAL_DILATION: Final[int] = 4
_LOCAL_PADDING: Final[int] = 8
_INV_SQRT_TWO: Final[float] = 1.0 / math.sqrt(2.0)
_PARAMETER_GRADIENT_SPLIT_K_THRESHOLD: Final[int] = 1_024
_PARAMETER_GRADIENT_SPLIT_K_BLOCK_ITEMS: Final[int] = 128
_PARAMETER_GRADIENT_REDUCTION_BLOCK_SPLITS: Final[int] = 128
_PARAMETER_GRADIENT_SPLIT_K_NUM_WARPS: Final[int] = 4
_PARAMETER_GRADIENT_REDUCTION_NUM_WARPS: Final[int] = 1
_C2_PARAMETER_GRADIENT_SPLIT_K_BLOCK_ITEMS: Final[int] = 64
_C2_PARAMETER_GRADIENT_SPLIT_K_BLOCK_CHANNELS: Final[int] = 32
_C2_PARAMETER_GRADIENT_COMPONENTS: Final[int] = 10
_PARAMETER_GRADIENT_AUTO: Final[int] = 0
_PARAMETER_GRADIENT_SERIAL: Final[int] = 1
_PARAMETER_GRADIENT_SPLIT_K: Final[int] = 2
_PARAMETER_GRADIENT_ATOMIC: Final[int] = 3

ParameterGradientStrategy = Literal["auto", "serial", "split_k", "atomic"]
_PARAMETER_GRADIENT_STRATEGY_CODES: Final[dict[str, int]] = {
    "auto": _PARAMETER_GRADIENT_AUTO,
    "serial": _PARAMETER_GRADIENT_SERIAL,
    "split_k": _PARAMETER_GRADIENT_SPLIT_K,
    "atomic": _PARAMETER_GRADIENT_ATOMIC,
}


@triton.jit
def _edge_frame_stem_training_kernel(
    raw_inputs,
    projection_weight,
    local_weight,
    local_bias,
    output,
    preactivation,
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
    projection_low = tl.load(
        projection_weight + channel * 2,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail = tl.load(
        projection_weight + channel * 2 + 1,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    local = tl.full((block_steps, block_channels), 0.0, tl.float32)

    for tap in range(5):
        source_edge = edge + tap * 4 - 8
        valid_source = valid_edge & (source_edge >= 0) & (source_edge < edge_steps)
        safe_edge = tl.where(valid_source, source_edge, 0)
        first_index = safe_edge
        second_index = safe_edge + 1
        raw_base = batch * input_steps
        first = tl.load(
            raw_inputs + raw_base + first_index,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            raw_inputs + raw_base + second_index,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
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
            other=0.0,
        ).to(tl.float32)
        local += projected * tap_weight

    local += tl.load(local_bias + channel, mask=valid_channel, other=0.0).to(tl.float32)
    activated = local * tl.sigmoid(local)
    output_offset = (batch * edge_steps + edge) * channels + channel
    tl.store(output + output_offset, activated, mask=valid_edge & valid_channel)
    tl.store(preactivation + output_offset, local, mask=valid_edge & valid_channel)


@triton.jit
def _edge_frame_stem_training_grad_raw_kernel(
    grad_output,
    preactivation,
    projection_weight,
    local_weight,
    grad_raw_inputs,
    input_steps: int,
    edge_steps: int,
    channels: int,
    block_steps: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    raw_step = tl.program_id(1) * block_steps + tl.arange(0, block_steps)
    channel = tl.arange(0, block_channels)[None, :]
    valid_raw = raw_step < input_steps
    valid_channel = channel < channels
    projection_low = tl.load(
        projection_weight + channel * 2,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail = tl.load(
        projection_weight + channel * 2 + 1,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    grad_raw = tl.zeros((block_steps,), tl.float32)

    for side in range(2):
        source_edge = raw_step - side
        valid_source = valid_raw & (source_edge >= 0) & (source_edge < edge_steps)
        grad_projected = tl.full((block_steps, block_channels), 0.0, tl.float32)
        for tap in range(5):
            output_edge = source_edge[:, None] - tap * 4 + 8
            valid_output = valid_source[:, None] & (output_edge >= 0) & (output_edge < edge_steps)
            safe_output = tl.where(valid_output, output_edge, 0)
            output_offset = (batch * edge_steps + safe_output) * channels + channel
            pre = tl.load(
                preactivation + output_offset,
                mask=valid_output & valid_channel,
                other=0.0,
            ).to(tl.float32)
            upstream = tl.load(
                grad_output + output_offset,
                mask=valid_output & valid_channel,
                other=0.0,
            ).to(tl.float32)
            sigmoid = tl.sigmoid(pre)
            grad_local = upstream * sigmoid * (1.0 + pre * (1.0 - sigmoid))
            tap_weight = tl.load(
                local_weight + channel * 5 + tap,
                mask=valid_channel,
                other=0.0,
            ).to(tl.float32)
            grad_projected += grad_local * tap_weight

        grad_low = tl.sum(grad_projected * projection_low, axis=1)
        grad_detail = tl.sum(grad_projected * projection_detail, axis=1)
        contribution = 0.7071067811865476 * (
            grad_low + grad_detail if side == 0 else grad_low - grad_detail
        )
        grad_raw += tl.where(valid_source, contribution, 0.0)

    degree_scale = tl.where(
        (raw_step == 0) | (raw_step == input_steps - 1),
        1.0,
        0.7071067811865476,
    )
    raw_offset = batch * input_steps + raw_step
    tl.store(
        grad_raw_inputs + raw_offset,
        grad_raw * degree_scale,
        mask=valid_raw,
    )


@triton.jit
def _edge_frame_stem_training_grad_parameters_kernel(
    grad_output,
    preactivation,
    raw_inputs,
    projection_weight,
    local_weight,
    grad_projection_weight,
    grad_local_weight,
    grad_local_bias,
    batch_size: int,
    input_steps: int,
    edge_steps: int,
    channels: int,
    block_items: tl.constexpr,
) -> None:
    channel = tl.program_id(0)
    item = tl.arange(0, block_items)[None, :]
    scalar_offset = tl.arange(0, 1)
    tap_offset = tl.arange(0, 8)
    tap = tap_offset[:, None]
    projection_low = tl.load(projection_weight + channel * 2).to(tl.float32)
    projection_detail = tl.load(projection_weight + channel * 2 + 1).to(tl.float32)
    tap_weight = tl.load(
        local_weight + channel * 5 + tap,
        mask=tap < 5,
        other=0.0,
    ).to(tl.float32)
    accumulated_bias = tl.zeros((1,), tl.float32)
    accumulated_projection_low = tl.zeros((1,), tl.float32)
    accumulated_projection_detail = tl.zeros((1,), tl.float32)
    accumulated_local_weight = tl.zeros((8,), tl.float32)
    total_items = batch_size * edge_steps
    item_base = 0

    while item_base < total_items:
        flat_item = item_base + item
        valid_item = flat_item < total_items
        batch = flat_item // edge_steps
        output_edge = flat_item - batch * edge_steps
        output_offset = flat_item * channels + channel
        pre = tl.load(
            preactivation + output_offset,
            mask=valid_item,
            other=0.0,
        ).to(tl.float32)
        upstream = tl.load(
            grad_output + output_offset,
            mask=valid_item,
            other=0.0,
        ).to(tl.float32)
        sigmoid = tl.sigmoid(pre)
        grad_local = upstream * sigmoid * (1.0 + pre * (1.0 - sigmoid))
        accumulated_bias += tl.sum(grad_local, axis=1)

        source_edge = output_edge + tap * 4 - 8
        valid_source = valid_item & (tap < 5) & (source_edge >= 0) & (source_edge < edge_steps)
        safe_source = tl.where(valid_source, source_edge, 0)
        first_index = safe_source
        second_index = safe_source + 1
        raw_base = batch * input_steps
        first = tl.load(
            raw_inputs + raw_base + first_index,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            raw_inputs + raw_base + second_index,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
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
        local_contribution = grad_local * projected
        accumulated_local_weight += tl.sum(local_contribution, axis=1)
        projection_contribution = grad_local * tap_weight
        accumulated_projection_low += tl.sum(tl.sum(projection_contribution * low, axis=1), axis=0)
        accumulated_projection_detail += tl.sum(
            tl.sum(projection_contribution * detail, axis=1), axis=0
        )
        item_base += block_items

    tl.store(
        grad_projection_weight + channel * 2 + scalar_offset,
        accumulated_projection_low,
    )
    tl.store(
        grad_projection_weight + channel * 2 + 1 + scalar_offset,
        accumulated_projection_detail,
    )
    tl.store(
        grad_local_weight + channel * 5 + tap_offset,
        accumulated_local_weight,
        mask=tap_offset < 5,
    )
    tl.store(grad_local_bias + channel + scalar_offset, accumulated_bias)


@triton.jit
def _edge_frame_stem_training_grad_parameters_atomic_kernel(
    grad_output,
    preactivation,
    raw_inputs,
    projection_weight,
    local_weight,
    grad_projection_weight,
    grad_local_weight,
    grad_local_bias,
    input_steps: int,
    edge_steps: int,
    channels: int,
    block_steps: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    edge = tl.program_id(1) * block_steps + tl.arange(0, block_steps)
    channel = tl.arange(0, block_channels)
    valid_edge = edge < edge_steps
    valid_channel = channel < channels
    output_offset = (batch * edge_steps + edge[:, None]) * channels + channel[None, :]
    valid_output = valid_edge[:, None] & valid_channel[None, :]
    pre = tl.load(
        preactivation + output_offset,
        mask=valid_output,
        other=0.0,
    ).to(tl.float32)
    upstream = tl.load(
        grad_output + output_offset,
        mask=valid_output,
        other=0.0,
    ).to(tl.float32)
    sigmoid = tl.sigmoid(pre)
    grad_local = upstream * sigmoid * (1.0 + pre * (1.0 - sigmoid))
    projection_low = tl.load(
        projection_weight + channel * 2,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail = tl.load(
        projection_weight + channel * 2 + 1,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_low_partial = tl.full((block_steps, block_channels), 0.0, tl.float32)
    projection_detail_partial = tl.full((block_steps, block_channels), 0.0, tl.float32)

    for tap in range(5):
        source_edge = edge + tap * 4 - 8
        valid_source = valid_edge & (source_edge >= 0) & (source_edge < edge_steps)
        safe_source = tl.where(valid_source, source_edge, 0)
        first_index = safe_source
        second_index = safe_source + 1
        raw_base = batch * input_steps
        first = tl.load(
            raw_inputs + raw_base + first_index,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            raw_inputs + raw_base + second_index,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
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
        projected = (
            projection_low[None, :] * low[:, None] + projection_detail[None, :] * detail[:, None]
        )
        local_weight_value = tl.load(
            local_weight + channel * 5 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        tl.atomic_add(
            grad_local_weight + channel * 5 + tap,
            tl.sum(grad_local * projected, axis=0),
            mask=valid_channel,
        )
        projection_low_partial += grad_local * local_weight_value[None, :] * low[:, None]
        projection_detail_partial += grad_local * local_weight_value[None, :] * detail[:, None]

    tl.atomic_add(
        grad_projection_weight + channel * 2,
        tl.sum(projection_low_partial, axis=0),
        mask=valid_channel,
    )
    tl.atomic_add(
        grad_projection_weight + channel * 2 + 1,
        tl.sum(projection_detail_partial, axis=0),
        mask=valid_channel,
    )
    tl.atomic_add(
        grad_local_bias + channel,
        tl.sum(grad_local, axis=0),
        mask=valid_channel,
    )


@triton.jit
def _edge_frame_stem_training_grad_parameters_split_k_kernel(
    grad_output,
    preactivation,
    raw_inputs,
    projection_weight,
    local_weight,
    partial_gradients,
    batch_size: int,
    input_steps: int,
    edge_steps: int,
    channels: int,
    block_items: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    """Produce conflict-free parameter-gradient partials for one item tile.

    The partial layout is ``[split, 8, channel]``.  Components 0/1 hold
    projection gradients, 2:7 hold the five depthwise-convolution weights,
    and component 7 holds the bias gradient.  No two programs write the same
    address, so scheduling cannot change the result.
    """
    split = tl.program_id(0)
    flat_item = split * block_items + tl.arange(0, block_items)
    channel = tl.arange(0, block_channels)
    total_items = batch_size * edge_steps
    valid_item = flat_item < total_items
    valid_channel = channel < channels
    batch = flat_item // edge_steps
    output_edge = flat_item - batch * edge_steps
    output_offset = flat_item[:, None] * channels + channel[None, :]
    valid_output = valid_item[:, None] & valid_channel[None, :]
    pre = tl.load(
        preactivation + output_offset,
        mask=valid_output,
        other=0.0,
    ).to(tl.float32)
    upstream = tl.load(
        grad_output + output_offset,
        mask=valid_output,
        other=0.0,
    ).to(tl.float32)
    sigmoid = tl.sigmoid(pre)
    grad_local = upstream * sigmoid * (1.0 + pre * (1.0 - sigmoid))
    projection_low = tl.load(
        projection_weight + channel * 2,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail = tl.load(
        projection_weight + channel * 2 + 1,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_low_partial = tl.full((block_items, block_channels), 0.0, tl.float32)
    projection_detail_partial = tl.full((block_items, block_channels), 0.0, tl.float32)
    partial_base = split * 8 * channels + channel

    for tap in range(5):
        source_edge = output_edge + tap * 4 - 8
        valid_source = valid_item & (source_edge >= 0) & (source_edge < edge_steps)
        safe_source = tl.where(valid_source, source_edge, 0)
        first_index = safe_source
        second_index = safe_source + 1
        raw_base = batch * input_steps
        first = tl.load(
            raw_inputs + raw_base + first_index,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            raw_inputs + raw_base + second_index,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
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
        projected = (
            projection_low[None, :] * low[:, None] + projection_detail[None, :] * detail[:, None]
        )
        local_weight_value = tl.load(
            local_weight + channel * 5 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            partial_gradients + partial_base + (2 + tap) * channels,
            tl.sum(grad_local * projected, axis=0),
            mask=valid_channel,
        )
        projection_low_partial += grad_local * local_weight_value[None, :] * low[:, None]
        projection_detail_partial += grad_local * local_weight_value[None, :] * detail[:, None]

    tl.store(
        partial_gradients + partial_base,
        tl.sum(projection_low_partial, axis=0),
        mask=valid_channel,
    )
    tl.store(
        partial_gradients + partial_base + channels,
        tl.sum(projection_detail_partial, axis=0),
        mask=valid_channel,
    )
    tl.store(
        partial_gradients + partial_base + 7 * channels,
        tl.sum(grad_local, axis=0),
        mask=valid_channel,
    )


@triton.jit
def _edge_frame_stem_training_grad_parameters_split_k_reduce_kernel(
    partial_gradients,
    grad_projection_weight,
    grad_local_weight,
    grad_local_bias,
    num_splits: int,
    channels: int,
    block_splits: tl.constexpr,
) -> None:
    """Reduce split-K partials in a fixed, repeatable order."""
    channel = tl.program_id(0)
    component = tl.arange(0, 8)[:, None]
    split_offset = tl.arange(0, block_splits)[None, :]
    accumulated = tl.zeros((8,), tl.float32)
    split_base = 0

    while split_base < num_splits:
        split = split_base + split_offset
        valid_split = split < num_splits
        partial_offset = split * 8 * channels + component * channels + channel
        partial = tl.load(
            partial_gradients + partial_offset,
            mask=valid_split,
            other=0.0,
        ).to(tl.float32)
        accumulated += tl.sum(partial, axis=1)
        split_base += block_splits

    final_component = tl.arange(0, 8)
    tl.store(
        grad_projection_weight + channel * 2 + final_component,
        accumulated,
        mask=final_component < 2,
    )
    tl.store(
        grad_local_weight + channel * 5 + final_component - 2,
        accumulated,
        mask=(final_component >= 2) & (final_component < 7),
    )
    bias = tl.sum(tl.where(final_component == 7, accumulated, 0.0), axis=0)
    tl.store(grad_local_bias + channel, bias)


@torch.library.triton_op("lnet::pac_edge_frame_stem_training", mutates_args={})
def _edge_frame_stem_training_op(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    parameter_gradient_strategy: int = _PARAMETER_GRADIENT_AUTO,
) -> tuple[Tensor, Tensor]:
    _validate_parameter_gradient_strategy_code(parameter_gradient_strategy)
    _validate_inputs(raw_inputs, projection_weight, local_weight, local_bias)
    if not raw_inputs.is_cuda:
        preactivation = _reference_preactivation(
            raw_inputs,
            projection_weight,
            local_weight,
            local_bias,
        )
        return functional.silu(preactivation), preactivation
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
    preactivation = torch.empty_like(output)
    block_steps = min(triton.next_power_of_2(edge_steps), 32)
    block_channels = triton.next_power_of_2(channels)
    grid = (batch, triton.cdiv(edge_steps, block_steps))
    torch.library.wrap_triton(_edge_frame_stem_training_kernel)[grid](
        raw,
        projection,
        local,
        bias,
        output,
        preactivation,
        input_steps,
        edge_steps,
        channels,
        block_steps=block_steps,
        block_channels=block_channels,
        num_warps=8,
    )
    return output, preactivation


def edge_frame_stem_training(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
    *,
    parameter_gradient_strategy: ParameterGradientStrategy = "auto",
) -> Tensor:
    """Run the fused exact-FP32 EFP stem with first-order autograd support.

    The tensor arguments map directly to ``EdgeFrameStem``'s raw input,
    ``projection.weight``, ``local.weight``, and ``local.bias``.  The operation
    is intentionally isolated from model dispatch so callers can opt into the
    training path without changing inference behavior. ``auto`` keeps the
    original serial parameter reduction for small workloads and selects the
    deterministic split-K reduction at the explicit item threshold above.
    ``atomic`` remains available as an A/B control.
    """
    strategy_code = _parameter_gradient_strategy_code(parameter_gradient_strategy)
    output, _preactivation = _edge_frame_stem_training_op(
        raw_inputs,
        projection_weight,
        local_weight,
        local_bias,
        strategy_code,
    )
    return output


def reference_edge_frame_stem_training(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Differentiable eager definition used as the exact training contract."""
    _validate_inputs(raw_inputs, projection_weight, local_weight, local_bias)
    return functional.silu(
        _reference_preactivation(
            raw_inputs,
            projection_weight,
            local_weight,
            local_bias,
        )
    )


def _reference_preactivation(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    features = _edge_features(raw_inputs)
    projected = torch.matmul(features, projection_weight.T)
    local = functional.conv1d(
        projected.transpose(1, 2),
        local_weight,
        local_bias,
        padding=_LOCAL_PADDING,
        dilation=_LOCAL_DILATION,
        groups=projection_weight.shape[0],
    )
    return local.transpose(1, 2)


def _edge_features(raw_inputs: Tensor) -> Tensor:
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
    return torch.cat((low, detail), dim=-1)


@torch.library.triton_op("lnet::pac_edge_frame_stem_training_backward", mutates_args={})
def _edge_frame_stem_training_backward_op(
    grad_output: Tensor,
    preactivation: Tensor,
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    parameter_gradient_strategy: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_parameter_gradient_strategy_code(parameter_gradient_strategy)
    if not grad_output.is_cuda:
        return _reference_edge_frame_stem_training_backward(
            grad_output,
            preactivation,
            raw_inputs,
            projection_weight,
            local_weight,
        )
    upstream = grad_output.contiguous()
    active_preactivation = preactivation.contiguous()
    raw = raw_inputs.contiguous()
    projection = projection_weight.contiguous()
    local = local_weight.contiguous()
    batch, input_steps, _ = raw.shape
    edge_steps = input_steps - 1
    channels = projection.shape[0]
    grad_raw_inputs = torch.empty_like(raw)
    total_items = batch * edge_steps
    if parameter_gradient_strategy == _PARAMETER_GRADIENT_AUTO:
        resolved_parameter_gradient_strategy = (
            _PARAMETER_GRADIENT_SPLIT_K
            if total_items >= _PARAMETER_GRADIENT_SPLIT_K_THRESHOLD
            else _PARAMETER_GRADIENT_SERIAL
        )
    else:
        resolved_parameter_gradient_strategy = parameter_gradient_strategy
    allocation = (
        torch.zeros
        if resolved_parameter_gradient_strategy == _PARAMETER_GRADIENT_ATOMIC
        else torch.empty
    )
    grad_projection_weight = allocation(
        projection.shape,
        device=projection.device,
        dtype=projection.dtype,
    )
    grad_local_weight = allocation(
        local.shape,
        device=local.device,
        dtype=local.dtype,
    )
    grad_local_bias = allocation(
        (channels,),
        device=raw.device,
        dtype=raw.dtype,
    )
    block_steps = min(triton.next_power_of_2(input_steps), 32)
    block_channels = triton.next_power_of_2(channels)
    raw_grid = (batch, triton.cdiv(input_steps, block_steps))
    torch.library.wrap_triton(_edge_frame_stem_training_grad_raw_kernel)[raw_grid](
        upstream,
        active_preactivation,
        projection,
        local,
        grad_raw_inputs,
        input_steps,
        edge_steps,
        channels,
        block_steps=block_steps,
        block_channels=block_channels,
        num_warps=8,
    )
    if resolved_parameter_gradient_strategy == _PARAMETER_GRADIENT_ATOMIC:
        parameter_block_steps = 32
        parameter_grid = (batch, triton.cdiv(edge_steps, parameter_block_steps))
        torch.library.wrap_triton(_edge_frame_stem_training_grad_parameters_atomic_kernel)[
            parameter_grid
        ](
            upstream,
            active_preactivation,
            raw,
            projection,
            local,
            grad_projection_weight,
            grad_local_weight,
            grad_local_bias,
            input_steps,
            edge_steps,
            channels,
            block_steps=parameter_block_steps,
            block_channels=block_channels,
            num_warps=8,
        )
    elif resolved_parameter_gradient_strategy == _PARAMETER_GRADIENT_SPLIT_K:
        parameter_block_items = _PARAMETER_GRADIENT_SPLIT_K_BLOCK_ITEMS
        num_splits = int(triton.cdiv(total_items, parameter_block_items))
        partial_gradients = torch.empty(
            (num_splits, 8, channels),
            device=raw.device,
            dtype=raw.dtype,
        )
        torch.library.wrap_triton(_edge_frame_stem_training_grad_parameters_split_k_kernel)[
            (num_splits,)
        ](
            upstream,
            active_preactivation,
            raw,
            projection,
            local,
            partial_gradients,
            batch,
            input_steps,
            edge_steps,
            channels,
            block_items=parameter_block_items,
            block_channels=block_channels,
            num_warps=_PARAMETER_GRADIENT_SPLIT_K_NUM_WARPS,
        )
        torch.library.wrap_triton(_edge_frame_stem_training_grad_parameters_split_k_reduce_kernel)[
            (channels,)
        ](
            partial_gradients,
            grad_projection_weight,
            grad_local_weight,
            grad_local_bias,
            num_splits,
            channels,
            block_splits=_PARAMETER_GRADIENT_REDUCTION_BLOCK_SPLITS,
            num_warps=_PARAMETER_GRADIENT_REDUCTION_NUM_WARPS,
        )
    else:
        torch.library.wrap_triton(_edge_frame_stem_training_grad_parameters_kernel)[(channels,)](
            upstream,
            active_preactivation,
            raw,
            projection,
            local,
            grad_projection_weight,
            grad_local_weight,
            grad_local_bias,
            batch,
            input_steps,
            edge_steps,
            channels,
            block_items=128,
            num_warps=4,
        )
    return (
        grad_raw_inputs,
        grad_projection_weight,
        grad_local_weight,
        grad_local_bias,
    )


def _reference_edge_frame_stem_training_backward(
    grad_output: Tensor,
    preactivation: Tensor,
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    sigmoid = torch.sigmoid(preactivation)
    grad_local = grad_output * sigmoid * (1.0 + preactivation * (1.0 - sigmoid))
    features = _edge_features(raw_inputs)
    projected = torch.matmul(features, projection_weight.T)
    grad_projected = functional.conv_transpose1d(
        grad_local.transpose(1, 2),
        local_weight,
        padding=_LOCAL_PADDING,
        dilation=_LOCAL_DILATION,
        groups=projection_weight.shape[0],
    ).transpose(1, 2)
    grad_local_weight = torch.nn.grad.conv1d_weight(  # pyright: ignore[reportAttributeAccessIssue]
        projected.transpose(1, 2),
        local_weight.shape,
        grad_local.transpose(1, 2),
        padding=_LOCAL_PADDING,
        dilation=_LOCAL_DILATION,
        groups=projection_weight.shape[0],
    )
    grad_local_bias = grad_local.sum(dim=(0, 1))
    grad_projection_weight = torch.matmul(
        grad_projected.reshape(-1, projection_weight.shape[0]).T,
        features.reshape(-1, _EDGE_PROJECTION_WIDTH),
    )
    grad_features = torch.matmul(grad_projected, projection_weight)
    grad_low = grad_features[..., :1]
    grad_detail = grad_features[..., 1:]
    grad_scaled = torch.zeros_like(raw_inputs)
    grad_scaled[:, :-1] += _INV_SQRT_TWO * (grad_low + grad_detail)
    grad_scaled[:, 1:] += _INV_SQRT_TWO * (grad_low - grad_detail)
    degree_scale = torch.ones(
        raw_inputs.shape[1],
        device=raw_inputs.device,
        dtype=raw_inputs.dtype,
    )
    if raw_inputs.shape[1] > 2:
        degree_scale[1:-1] = _INV_SQRT_TWO
    grad_raw_inputs = grad_scaled * degree_scale.view(1, -1, 1)
    return (
        grad_raw_inputs,
        grad_projection_weight,
        grad_local_weight,
        grad_local_bias,
    )


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    parameter_gradient_strategy: int
    needs_input_grad: tuple[bool, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, int],
    output: tuple[Tensor, Tensor],
) -> None:
    preactivation = output[1]
    ctx.mark_non_differentiable(preactivation)
    ctx.parameter_gradient_strategy = inputs[4]
    ctx.save_for_backward(*inputs[:4], preactivation)


def _backward(
    ctx: _AutogradContext,
    grad_output: Tensor,
    _grad_preactivation: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
    del _grad_preactivation
    raw_inputs, projection_weight, local_weight, local_bias, preactivation = ctx.saved_tensors
    del local_bias
    gradients = _edge_frame_stem_training_backward_op(
        grad_output,
        preactivation,
        raw_inputs,
        projection_weight,
        local_weight,
        ctx.parameter_gradient_strategy,
    )
    return (*gradients, None)


torch.library.register_autograd(
    "lnet::pac_edge_frame_stem_training",
    _backward,
    setup_context=_setup_context,
)


def _parameter_gradient_strategy_code(strategy: ParameterGradientStrategy) -> int:
    try:
        return _PARAMETER_GRADIENT_STRATEGY_CODES[strategy]
    except KeyError as error:
        options = ", ".join(_PARAMETER_GRADIENT_STRATEGY_CODES)
        message = f"parameter-gradient strategy must be one of: {options}"
        raise ValueError(message) from error


def _validate_parameter_gradient_strategy_code(strategy: int) -> None:
    if strategy not in _PARAMETER_GRADIENT_STRATEGY_CODES.values():
        message = f"unsupported parameter-gradient strategy code: {strategy}"
        raise ValueError(message)


def _validate_inputs(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> None:
    if raw_inputs.ndim != 3 or raw_inputs.shape[-1] != 1 or raw_inputs.shape[1] < 2:
        message = "edge-frame stem training requires raw inputs with shape [batch,time>=2,1]"
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
    if raw_inputs.dtype != torch.float32:
        message = "edge-frame stem training supports exact FP32 only"
        raise TypeError(message)


# The canonical two-channel BenchmarkAlphabetBackbone surface has four edge features:
# two normalized levels followed by two normalized details.  Keeping its kernels
# separate from the scalar path avoids adding predicates, wider reductions, or a
# different partial-gradient layout to the already verified C=1 implementation.
@triton.jit
def _edge_frame_stem_c2_training_kernel(
    raw_inputs,
    projection_weight,
    local_weight,
    local_bias,
    output,
    preactivation,
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
    projection_low_0 = tl.load(
        projection_weight + channel * 4,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_low_1 = tl.load(
        projection_weight + channel * 4 + 1,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail_0 = tl.load(
        projection_weight + channel * 4 + 2,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail_1 = tl.load(
        projection_weight + channel * 4 + 3,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    local = tl.full((block_steps, block_channels), 0.0, tl.float32)

    for tap in range(5):
        source_edge = edge + tap * 4 - 8
        valid_source = valid_edge & (source_edge >= 0) & (source_edge < edge_steps)
        safe_edge = tl.where(valid_source, source_edge, 0)
        first_index = safe_edge
        second_index = safe_edge + 1
        first_base = (batch * input_steps + first_index) * 2
        second_base = (batch * input_steps + second_index) * 2
        first_0 = tl.load(
            raw_inputs + first_base,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        first_1 = tl.load(
            raw_inputs + first_base + 1,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        second_0 = tl.load(
            raw_inputs + second_base,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        second_1 = tl.load(
            raw_inputs + second_base + 1,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
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
        scaled_first_0 = first_0 * first_degree_scale
        scaled_first_1 = first_1 * first_degree_scale
        scaled_second_0 = second_0 * second_degree_scale
        scaled_second_1 = second_1 * second_degree_scale
        low_0 = 0.7071067811865476 * (scaled_first_0 + scaled_second_0)
        low_1 = 0.7071067811865476 * (scaled_first_1 + scaled_second_1)
        detail_0 = 0.7071067811865476 * (scaled_first_0 - scaled_second_0)
        detail_1 = 0.7071067811865476 * (scaled_first_1 - scaled_second_1)
        projected = (
            projection_low_0 * low_0
            + projection_low_1 * low_1
            + projection_detail_0 * detail_0
            + projection_detail_1 * detail_1
        )
        tap_weight = tl.load(
            local_weight + channel * 5 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        local += projected * tap_weight

    local += tl.load(local_bias + channel, mask=valid_channel, other=0.0).to(tl.float32)
    activated = local * tl.sigmoid(local)
    output_offset = (batch * edge_steps + edge) * channels + channel
    valid_output = valid_edge & valid_channel
    tl.store(output + output_offset, activated, mask=valid_output)
    tl.store(preactivation + output_offset, local, mask=valid_output)


@triton.jit
def _edge_frame_stem_c2_training_grad_raw_kernel(
    grad_output,
    preactivation,
    projection_weight,
    local_weight,
    grad_raw_inputs,
    input_steps: int,
    edge_steps: int,
    channels: int,
    block_steps: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    raw_step = tl.program_id(1) * block_steps + tl.arange(0, block_steps)
    channel = tl.arange(0, block_channels)[None, :]
    valid_raw = raw_step < input_steps
    valid_channel = channel < channels
    projection_low_0 = tl.load(
        projection_weight + channel * 4,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_low_1 = tl.load(
        projection_weight + channel * 4 + 1,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail_0 = tl.load(
        projection_weight + channel * 4 + 2,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail_1 = tl.load(
        projection_weight + channel * 4 + 3,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    grad_raw_0 = tl.zeros((block_steps,), tl.float32)
    grad_raw_1 = tl.zeros((block_steps,), tl.float32)

    for side in range(2):
        source_edge = raw_step - side
        valid_source = valid_raw & (source_edge >= 0) & (source_edge < edge_steps)
        grad_projected = tl.full((block_steps, block_channels), 0.0, tl.float32)
        for tap in range(5):
            output_edge = source_edge[:, None] - tap * 4 + 8
            valid_output = valid_source[:, None] & (output_edge >= 0) & (output_edge < edge_steps)
            safe_output = tl.where(valid_output, output_edge, 0)
            output_offset = (batch * edge_steps + safe_output) * channels + channel
            active_mask = valid_output & valid_channel
            pre = tl.load(
                preactivation + output_offset,
                mask=active_mask,
                other=0.0,
            ).to(tl.float32)
            upstream = tl.load(
                grad_output + output_offset,
                mask=active_mask,
                other=0.0,
            ).to(tl.float32)
            sigmoid = tl.sigmoid(pre)
            grad_local = upstream * sigmoid * (1.0 + pre * (1.0 - sigmoid))
            tap_weight = tl.load(
                local_weight + channel * 5 + tap,
                mask=valid_channel,
                other=0.0,
            ).to(tl.float32)
            grad_projected += grad_local * tap_weight

        grad_low_0 = tl.sum(grad_projected * projection_low_0, axis=1)
        grad_low_1 = tl.sum(grad_projected * projection_low_1, axis=1)
        grad_detail_0 = tl.sum(grad_projected * projection_detail_0, axis=1)
        grad_detail_1 = tl.sum(grad_projected * projection_detail_1, axis=1)
        side_0 = grad_low_0 + grad_detail_0 if side == 0 else grad_low_0 - grad_detail_0
        side_1 = grad_low_1 + grad_detail_1 if side == 0 else grad_low_1 - grad_detail_1
        grad_raw_0 += tl.where(valid_source, 0.7071067811865476 * side_0, 0.0)
        grad_raw_1 += tl.where(valid_source, 0.7071067811865476 * side_1, 0.0)

    degree_scale = tl.where(
        (raw_step == 0) | (raw_step == input_steps - 1),
        1.0,
        0.7071067811865476,
    )
    raw_offset = (batch * input_steps + raw_step) * 2
    tl.store(grad_raw_inputs + raw_offset, grad_raw_0 * degree_scale, mask=valid_raw)
    tl.store(grad_raw_inputs + raw_offset + 1, grad_raw_1 * degree_scale, mask=valid_raw)


@triton.jit
def _edge_frame_stem_c2_training_grad_parameters_split_k_kernel(  # noqa: PLR0915
    grad_output,
    preactivation,
    raw_inputs,
    projection_weight,
    local_weight,
    partial_gradients,
    batch_size: int,
    input_steps: int,
    edge_steps: int,
    channels: int,
    block_items: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    """Produce C=2 parameter-gradient partials without atomics or spills."""
    split = tl.program_id(0)
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    flat_item = split * block_items + tl.arange(0, block_items)
    total_items = batch_size * edge_steps
    valid_item = flat_item < total_items
    valid_channel = channel < channels
    batch = flat_item // edge_steps
    output_edge = flat_item - batch * edge_steps
    output_offset = flat_item[:, None] * channels + channel[None, :]
    valid_output = valid_item[:, None] & valid_channel[None, :]
    pre = tl.load(
        preactivation + output_offset,
        mask=valid_output,
        other=0.0,
    ).to(tl.float32)
    upstream = tl.load(
        grad_output + output_offset,
        mask=valid_output,
        other=0.0,
    ).to(tl.float32)
    sigmoid = tl.sigmoid(pre)
    grad_local = upstream * sigmoid * (1.0 + pre * (1.0 - sigmoid))
    projection_low_0 = tl.load(
        projection_weight + channel * 4,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_low_1 = tl.load(
        projection_weight + channel * 4 + 1,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail_0 = tl.load(
        projection_weight + channel * 4 + 2,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_detail_1 = tl.load(
        projection_weight + channel * 4 + 3,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    accumulated_projection_low_0 = tl.zeros((block_channels,), tl.float32)
    accumulated_projection_low_1 = tl.zeros((block_channels,), tl.float32)
    accumulated_projection_detail_0 = tl.zeros((block_channels,), tl.float32)
    accumulated_projection_detail_1 = tl.zeros((block_channels,), tl.float32)
    partial_base = split * 10 * channels + channel

    for tap in range(5):
        source_edge = output_edge + tap * 4 - 8
        valid_source = valid_item & (source_edge >= 0) & (source_edge < edge_steps)
        safe_source = tl.where(valid_source, source_edge, 0)
        first_index = safe_source
        second_index = safe_source + 1
        first_base = (batch * input_steps + first_index) * 2
        second_base = (batch * input_steps + second_index) * 2
        first_0 = tl.load(
            raw_inputs + first_base,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        first_1 = tl.load(
            raw_inputs + first_base + 1,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        second_0 = tl.load(
            raw_inputs + second_base,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        second_1 = tl.load(
            raw_inputs + second_base + 1,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
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
        scaled_first_0 = first_0 * first_degree_scale
        scaled_first_1 = first_1 * first_degree_scale
        scaled_second_0 = second_0 * second_degree_scale
        scaled_second_1 = second_1 * second_degree_scale
        low_0 = 0.7071067811865476 * (scaled_first_0 + scaled_second_0)
        low_1 = 0.7071067811865476 * (scaled_first_1 + scaled_second_1)
        detail_0 = 0.7071067811865476 * (scaled_first_0 - scaled_second_0)
        detail_1 = 0.7071067811865476 * (scaled_first_1 - scaled_second_1)
        projected = (
            projection_low_0[None, :] * low_0[:, None]
            + projection_low_1[None, :] * low_1[:, None]
            + projection_detail_0[None, :] * detail_0[:, None]
            + projection_detail_1[None, :] * detail_1[:, None]
        )
        local_weight_value = tl.load(
            local_weight + channel * 5 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            partial_gradients + partial_base + (4 + tap) * channels,
            tl.sum(grad_local * projected, axis=0),
            mask=valid_channel,
        )
        grad_projection_input = grad_local * local_weight_value[None, :]
        accumulated_projection_low_0 += tl.sum(grad_projection_input * low_0[:, None], axis=0)
        accumulated_projection_low_1 += tl.sum(grad_projection_input * low_1[:, None], axis=0)
        accumulated_projection_detail_0 += tl.sum(grad_projection_input * detail_0[:, None], axis=0)
        accumulated_projection_detail_1 += tl.sum(grad_projection_input * detail_1[:, None], axis=0)

    tl.store(
        partial_gradients + partial_base,
        accumulated_projection_low_0,
        mask=valid_channel,
    )
    tl.store(
        partial_gradients + partial_base + channels,
        accumulated_projection_low_1,
        mask=valid_channel,
    )
    tl.store(
        partial_gradients + partial_base + 2 * channels,
        accumulated_projection_detail_0,
        mask=valid_channel,
    )
    tl.store(
        partial_gradients + partial_base + 3 * channels,
        accumulated_projection_detail_1,
        mask=valid_channel,
    )
    tl.store(
        partial_gradients + partial_base + 9 * channels,
        tl.sum(grad_local, axis=0),
        mask=valid_channel,
    )


@triton.jit
def _edge_frame_stem_c2_training_grad_parameters_split_k_reduce_kernel(
    partial_gradients,
    grad_projection_weight,
    grad_local_weight,
    grad_local_bias,
    num_splits: int,
    channels: int,
    block_splits: tl.constexpr,
) -> None:
    channel = tl.program_id(0)
    component = tl.arange(0, 16)[:, None]
    split_offset = tl.arange(0, block_splits)[None, :]
    accumulated = tl.zeros((16,), tl.float32)
    split_base = 0

    while split_base < num_splits:
        split = split_base + split_offset
        valid = (split < num_splits) & (component < 10)
        partial_offset = split * 10 * channels + component * channels + channel
        partial = tl.load(partial_gradients + partial_offset, mask=valid, other=0.0).to(tl.float32)
        accumulated += tl.sum(partial, axis=1)
        split_base += block_splits

    final_component = tl.arange(0, 16)
    tl.store(
        grad_projection_weight + channel * 4 + final_component,
        accumulated,
        mask=final_component < 4,
    )
    tl.store(
        grad_local_weight + channel * 5 + final_component - 4,
        accumulated,
        mask=(final_component >= 4) & (final_component < 9),
    )
    bias = tl.sum(tl.where(final_component == 9, accumulated, 0.0), axis=0)
    tl.store(grad_local_bias + channel, bias)


@torch.library.triton_op("lnet::pac_edge_frame_stem_c2_training", mutates_args={})
def _edge_frame_stem_c2_training_op(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_c2_inputs(raw_inputs, projection_weight, local_weight, local_bias)
    if not raw_inputs.is_cuda:
        preactivation = _reference_preactivation(
            raw_inputs,
            projection_weight,
            local_weight,
            local_bias,
        )
        return functional.silu(preactivation), preactivation
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
    preactivation = torch.empty_like(output)
    block_steps = min(triton.next_power_of_2(edge_steps), 32)
    block_channels = triton.next_power_of_2(channels)
    grid = (batch, triton.cdiv(edge_steps, block_steps))
    torch.library.wrap_triton(_edge_frame_stem_c2_training_kernel)[grid](
        raw,
        projection,
        local,
        bias,
        output,
        preactivation,
        input_steps,
        edge_steps,
        channels,
        block_steps=block_steps,
        block_channels=block_channels,
        num_warps=8,
    )
    return output, preactivation


@torch.library.triton_op("lnet::pac_edge_frame_stem_c2_training_backward", mutates_args={})
def _edge_frame_stem_c2_training_backward_op(
    grad_output: Tensor,
    preactivation: Tensor,
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    compute_grad_raw: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not grad_output.is_cuda:
        return _reference_edge_frame_stem_c2_training_backward(
            grad_output,
            preactivation,
            raw_inputs,
            projection_weight,
            local_weight,
        )
    upstream = grad_output.contiguous()
    active_preactivation = preactivation.contiguous()
    raw = raw_inputs.contiguous()
    projection = projection_weight.contiguous()
    local = local_weight.contiguous()
    batch, input_steps, _ = raw.shape
    edge_steps = input_steps - 1
    channels = projection.shape[0]
    grad_raw_inputs = torch.empty_like(raw)
    grad_projection_weight = torch.empty_like(projection)
    grad_local_weight = torch.empty_like(local)
    grad_local_bias = torch.empty(
        (channels,),
        device=raw.device,
        dtype=raw.dtype,
    )
    # C=2 carries two independent raw gradients through the channel reduction.
    # A 16-step tile keeps that live set below the occupancy cliff on SM120.
    block_steps = min(triton.next_power_of_2(input_steps), 16)
    block_channels = triton.next_power_of_2(channels)
    if compute_grad_raw:
        raw_grid = (batch, triton.cdiv(input_steps, block_steps))
        torch.library.wrap_triton(_edge_frame_stem_c2_training_grad_raw_kernel)[raw_grid](
            upstream,
            active_preactivation,
            projection,
            local,
            grad_raw_inputs,
            input_steps,
            edge_steps,
            channels,
            block_steps=block_steps,
            block_channels=block_channels,
            num_warps=8,
        )
    parameter_block_items = _C2_PARAMETER_GRADIENT_SPLIT_K_BLOCK_ITEMS
    parameter_block_channels = min(
        triton.next_power_of_2(channels),
        _C2_PARAMETER_GRADIENT_SPLIT_K_BLOCK_CHANNELS,
    )
    num_splits = int(triton.cdiv(batch * edge_steps, parameter_block_items))
    partial_gradients = torch.empty(
        (num_splits, _C2_PARAMETER_GRADIENT_COMPONENTS, channels),
        device=raw.device,
        dtype=raw.dtype,
    )
    parameter_grid = (
        num_splits,
        triton.cdiv(channels, parameter_block_channels),
    )
    torch.library.wrap_triton(_edge_frame_stem_c2_training_grad_parameters_split_k_kernel)[
        parameter_grid
    ](
        upstream,
        active_preactivation,
        raw,
        projection,
        local,
        partial_gradients,
        batch,
        input_steps,
        edge_steps,
        channels,
        block_items=parameter_block_items,
        block_channels=parameter_block_channels,
        num_warps=4,
    )
    torch.library.wrap_triton(_edge_frame_stem_c2_training_grad_parameters_split_k_reduce_kernel)[
        (channels,)
    ](
        partial_gradients,
        grad_projection_weight,
        grad_local_weight,
        grad_local_bias,
        num_splits,
        channels,
        block_splits=_PARAMETER_GRADIENT_REDUCTION_BLOCK_SPLITS,
        num_warps=_PARAMETER_GRADIENT_REDUCTION_NUM_WARPS,
    )
    return (
        grad_raw_inputs,
        grad_projection_weight,
        grad_local_weight,
        grad_local_bias,
    )


def edge_frame_stem_c2_training(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Run the fused maskless FP32 edge stem for exactly two raw channels."""
    output, _preactivation = _edge_frame_stem_c2_training_op(
        raw_inputs,
        projection_weight,
        local_weight,
        local_bias,
    )
    return output


def reference_edge_frame_stem_c2_training(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Differentiable eager contract for the C=2 fused training stem."""
    _validate_c2_inputs(raw_inputs, projection_weight, local_weight, local_bias)
    return functional.silu(
        _reference_preactivation(
            raw_inputs,
            projection_weight,
            local_weight,
            local_bias,
        )
    )


def _reference_edge_frame_stem_c2_training_backward(
    grad_output: Tensor,
    preactivation: Tensor,
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    sigmoid = torch.sigmoid(preactivation)
    grad_local = grad_output * sigmoid * (1.0 + preactivation * (1.0 - sigmoid))
    features = _edge_features(raw_inputs)
    projected = torch.matmul(features, projection_weight.T)
    grad_projected = functional.conv_transpose1d(
        grad_local.transpose(1, 2),
        local_weight,
        padding=_LOCAL_PADDING,
        dilation=_LOCAL_DILATION,
        groups=projection_weight.shape[0],
    ).transpose(1, 2)
    grad_local_weight = torch.nn.grad.conv1d_weight(  # pyright: ignore[reportAttributeAccessIssue]
        projected.transpose(1, 2),
        local_weight.shape,
        grad_local.transpose(1, 2),
        padding=_LOCAL_PADDING,
        dilation=_LOCAL_DILATION,
        groups=projection_weight.shape[0],
    )
    grad_local_bias = grad_local.sum(dim=(0, 1))
    grad_projection_weight = torch.matmul(
        grad_projected.reshape(-1, projection_weight.shape[0]).T,
        features.reshape(-1, _EDGE_PROJECTION_WIDTH_C2),
    )
    grad_features = torch.matmul(grad_projected, projection_weight)
    grad_low = grad_features[..., :2]
    grad_detail = grad_features[..., 2:]
    grad_scaled = torch.zeros_like(raw_inputs)
    grad_scaled[:, :-1] += _INV_SQRT_TWO * (grad_low + grad_detail)
    grad_scaled[:, 1:] += _INV_SQRT_TWO * (grad_low - grad_detail)
    degree_scale = torch.ones(
        raw_inputs.shape[1],
        device=raw_inputs.device,
        dtype=raw_inputs.dtype,
    )
    if raw_inputs.shape[1] > 2:
        degree_scale[1:-1] = _INV_SQRT_TWO
    grad_raw_inputs = grad_scaled * degree_scale.view(1, -1, 1)
    return (
        grad_raw_inputs,
        grad_projection_weight,
        grad_local_weight,
        grad_local_bias,
    )


def _setup_context_c2(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor],
) -> None:
    preactivation = output[1]
    ctx.mark_non_differentiable(preactivation)
    ctx.save_for_backward(*inputs, preactivation)


def _backward_c2(
    ctx: _AutogradContext,
    grad_output: Tensor,
    _grad_preactivation: Tensor | None,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
    del _grad_preactivation
    raw_inputs, projection_weight, local_weight, local_bias, preactivation = ctx.saved_tensors
    del local_bias
    compute_grad_raw = ctx.needs_input_grad[0]
    gradients = _edge_frame_stem_c2_training_backward_op(
        grad_output,
        preactivation,
        raw_inputs,
        projection_weight,
        local_weight,
        compute_grad_raw,
    )
    return (gradients[0] if compute_grad_raw else None, *gradients[1:])


torch.library.register_autograd(
    "lnet::pac_edge_frame_stem_c2_training",
    _backward_c2,
    setup_context=_setup_context_c2,
)


def _validate_c2_inputs(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> None:
    if raw_inputs.ndim != 3 or raw_inputs.shape[-1] != 2 or raw_inputs.shape[1] < 2:
        message = "C=2 edge-frame stem training requires raw inputs with shape [batch,time>=2,2]"
        raise ValueError(message)
    if projection_weight.ndim != 2 or projection_weight.shape[1] != _EDGE_PROJECTION_WIDTH_C2:
        message = "C=2 edge-frame projection weight must have shape [channels,4]"
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
    if raw_inputs.dtype != torch.float32:
        message = "edge-frame stem training supports exact FP32 only"
        raise TypeError(message)
