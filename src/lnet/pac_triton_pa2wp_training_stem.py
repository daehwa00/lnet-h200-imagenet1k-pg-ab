from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001
import math
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional

_STEM_KERNEL_SIZE: Final[int] = 9
_INV_SQRT_TWO: Final[float] = 1.0 / math.sqrt(2.0)
# RTX 4090 N2048/B64 capture sweep: 256 beats 128 and 512 while the kernel
# autotuner selects four versus eight warps independently for each shape.
_PARAMETER_REDUCTION_BLOCK: Final[int] = 256
_PARAMETER_SPLIT_THRESHOLD: Final[int] = 8_192
_PARAMETER_PARTIALS_PER_CHANNEL: Final[int] = _STEM_KERNEL_SIZE + 1
_LARGE_WORKLOAD_ITEMS: Final[int] = 65_536


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _training_stem_forward_kernel(
    raw_inputs,
    stem_weight,
    stem_bias,
    output,
    batch_count: int,
    input_steps: int,
    output_steps: int,
    channels: int,
    block_steps: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step = tl.program_id(1) * block_steps + tl.arange(0, block_steps)[:, None]
    channel = tl.arange(0, block_channels)[None, :]
    valid_step = step < output_steps
    valid_channel = channel < channels
    even_accumulator = tl.full((block_steps, block_channels), 0.0, tl.float32)
    odd_accumulator = tl.full((block_steps, block_channels), 0.0, tl.float32)

    for tap in range(9):
        pair = 2 * step + tap - 8
        even_index = 2 * pair
        odd_index = even_index + 1
        raw_base = batch * input_steps
        even_value = tl.load(
            raw_inputs + raw_base + even_index,
            mask=valid_step & (even_index >= 0) & (even_index < input_steps),
            other=0.0,
        ).to(tl.float32)
        odd_value = tl.load(
            raw_inputs + raw_base + odd_index,
            mask=valid_step & (odd_index >= 0) & (odd_index < input_steps),
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            stem_weight + channel * 9 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        even_accumulator += even_value * weight
        odd_accumulator += odd_value * weight

    bias = tl.load(stem_bias + channel, mask=valid_channel, other=0.0).to(tl.float32)
    low = 0.7071067811865476 * (even_accumulator + odd_accumulator) + bias
    detail = 0.7071067811865476 * (even_accumulator - odd_accumulator) + bias
    low *= tl.sigmoid(low)
    detail *= tl.sigmoid(detail)
    valid_output = valid_step & valid_channel
    output_base = step * channels + channel
    batch_stride = output_steps * channels
    tl.store(
        output + batch * batch_stride + output_base,
        low,
        mask=valid_output,
    )
    tl.store(
        output + (batch_count + batch) * batch_stride + output_base,
        detail,
        mask=valid_output,
    )


@triton.jit
def _training_stem_grad_activation_kernel(
    raw_inputs,
    stem_weight,
    stem_bias,
    grad_output,
    grad_activation,
    batch_count: int,
    input_steps: int,
    output_steps: int,
    channels: int,
    block_steps: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step = tl.program_id(1) * block_steps + tl.arange(0, block_steps)[:, None]
    channel = tl.arange(0, block_channels)[None, :]
    valid_step = step < output_steps
    valid_channel = channel < channels
    valid_output = valid_step & valid_channel
    even_accumulator = tl.full((block_steps, block_channels), 0.0, tl.float32)
    odd_accumulator = tl.full((block_steps, block_channels), 0.0, tl.float32)

    for tap in range(9):
        pair = 2 * step + tap - 8
        even_index = 2 * pair
        odd_index = even_index + 1
        raw_base = batch * input_steps
        even_value = tl.load(
            raw_inputs + raw_base + even_index,
            mask=valid_step & (even_index >= 0) & (even_index < input_steps),
            other=0.0,
        ).to(tl.float32)
        odd_value = tl.load(
            raw_inputs + raw_base + odd_index,
            mask=valid_step & (odd_index >= 0) & (odd_index < input_steps),
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            stem_weight + channel * 9 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        even_accumulator += even_value * weight
        odd_accumulator += odd_value * weight

    bias = tl.load(stem_bias + channel, mask=valid_channel, other=0.0).to(tl.float32)
    low = 0.7071067811865476 * (even_accumulator + odd_accumulator) + bias
    detail = 0.7071067811865476 * (even_accumulator - odd_accumulator) + bias
    low_sigmoid = tl.sigmoid(low)
    detail_sigmoid = tl.sigmoid(detail)
    low_derivative = low_sigmoid * (1.0 + low * (1.0 - low_sigmoid))
    detail_derivative = detail_sigmoid * (1.0 + detail * (1.0 - detail_sigmoid))
    output_base = step * channels + channel
    batch_stride = output_steps * channels
    low_gradient = tl.load(
        grad_output + batch * batch_stride + output_base,
        mask=valid_output,
        other=0.0,
    ).to(tl.float32)
    detail_gradient = tl.load(
        grad_output + (batch_count + batch) * batch_stride + output_base,
        mask=valid_output,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        grad_activation + batch * batch_stride + output_base,
        low_gradient * low_derivative,
        mask=valid_output,
    )
    tl.store(
        grad_activation + (batch_count + batch) * batch_stride + output_base,
        detail_gradient * detail_derivative,
        mask=valid_output,
    )


@triton.jit
def _training_stem_grad_input_kernel(
    stem_weight,
    grad_activation,
    grad_input,
    batch_count: int,
    input_steps: int,
    output_steps: int,
    channels: int,
    block_inputs: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    raw_offset = tl.program_id(1) * block_inputs + tl.arange(0, block_inputs)
    raw_index = raw_offset[:, None]
    channel = tl.arange(0, block_channels)[None, :]
    valid_raw_offset = raw_offset < input_steps
    valid_raw = valid_raw_offset[:, None]
    valid_channel = channel < channels
    pair = raw_index // 2
    is_odd = raw_index - pair * 2
    accumulator = tl.full((block_inputs,), 0.0, tl.float32)
    batch_stride = output_steps * channels

    for tap in range(9):
        numerator = pair - tap + 8
        step = numerator // 2
        even_numerator = numerator - step * 2 == 0
        valid_source = valid_raw & (numerator >= 0) & even_numerator & (step < output_steps)
        safe_step = tl.where(valid_source, step, 0)
        output_base = safe_step * channels + channel
        low_gradient = tl.load(
            grad_activation + batch * batch_stride + output_base,
            mask=valid_source & valid_channel,
            other=0.0,
        ).to(tl.float32)
        detail_gradient = tl.load(
            grad_activation + (batch_count + batch) * batch_stride + output_base,
            mask=valid_source & valid_channel,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            stem_weight + channel * 9 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        band_gradient = low_gradient + tl.where(is_odd != 0, -detail_gradient, detail_gradient)
        accumulator += tl.sum(band_gradient * weight, axis=1)

    tl.store(
        grad_input + batch * input_steps + raw_offset,
        0.7071067811865476 * accumulator,
        mask=valid_raw_offset,
    )


@triton.jit
def _training_stem_grad_weight_kernel(
    raw_inputs,
    grad_activation,
    grad_weight,
    batch_count: int,
    input_steps: int,
    output_steps: int,
    channels: int,
    reduction_block: tl.constexpr,
) -> None:
    channel = tl.program_id(0)
    tap = tl.program_id(1)
    item_count = batch_count * output_steps
    offsets = tl.arange(0, reduction_block)
    accumulator = tl.full((), 0.0, tl.float32)
    start = 0
    while start < item_count:
        item = start + offsets
        valid_item = item < item_count
        batch = item // output_steps
        step = item - batch * output_steps
        pair = 2 * step + tap - 8
        even_index = 2 * pair
        odd_index = even_index + 1
        raw_base = batch * input_steps
        even_value = tl.load(
            raw_inputs + raw_base + even_index,
            mask=valid_item & (even_index >= 0) & (even_index < input_steps),
            other=0.0,
        ).to(tl.float32)
        odd_value = tl.load(
            raw_inputs + raw_base + odd_index,
            mask=valid_item & (odd_index >= 0) & (odd_index < input_steps),
            other=0.0,
        ).to(tl.float32)
        activation_offset = item * channels + channel
        batch_stride = output_steps * channels
        low_gradient = tl.load(
            grad_activation + activation_offset,
            mask=valid_item,
            other=0.0,
        ).to(tl.float32)
        detail_gradient = tl.load(
            grad_activation + batch_count * batch_stride + activation_offset,
            mask=valid_item,
            other=0.0,
        ).to(tl.float32)
        low = 0.7071067811865476 * (even_value + odd_value)
        detail = 0.7071067811865476 * (even_value - odd_value)
        accumulator += tl.sum(low_gradient * low + detail_gradient * detail, axis=0)
        start += reduction_block
    tl.store(grad_weight + channel * 9 + tap, accumulator)


@triton.jit
def _training_stem_grad_bias_kernel(
    grad_activation,
    grad_bias,
    batch_count: int,
    output_steps: int,
    channels: int,
    reduction_block: tl.constexpr,
) -> None:
    channel = tl.program_id(0)
    item_count = batch_count * output_steps
    offsets = tl.arange(0, reduction_block)
    accumulator = tl.full((), 0.0, tl.float32)
    start = 0
    while start < item_count:
        item = start + offsets
        valid_item = item < item_count
        activation_offset = item * channels + channel
        batch_stride = output_steps * channels
        low_gradient = tl.load(
            grad_activation + activation_offset,
            mask=valid_item,
            other=0.0,
        ).to(tl.float32)
        detail_gradient = tl.load(
            grad_activation + batch_count * batch_stride + activation_offset,
            mask=valid_item,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(low_gradient + detail_gradient, axis=0)
        start += reduction_block
    tl.store(grad_bias + channel, accumulator)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["batch_count", "output_steps", "channels"],
)
@triton.jit
def _training_stem_grad_parameter_partials_kernel(
    raw_inputs,
    grad_activation,
    parameter_partials,
    batch_count: int,
    input_steps: int,
    output_steps: int,
    channels: int,
    reduction_block: tl.constexpr,
) -> None:
    """Reduce one split-K tile for all K9 weights and the bias of one channel."""
    channel = tl.program_id(0)
    reduction_group = tl.program_id(1)
    item = reduction_group * reduction_block + tl.arange(0, reduction_block)
    item_count = batch_count * output_steps
    valid_item = item < item_count
    batch = item // output_steps
    step = item - batch * output_steps
    activation_offset = item * channels + channel
    batch_stride = output_steps * channels
    low_gradient = tl.load(
        grad_activation + activation_offset,
        mask=valid_item,
        other=0.0,
    ).to(tl.float32)
    detail_gradient = tl.load(
        grad_activation + batch_count * batch_stride + activation_offset,
        mask=valid_item,
        other=0.0,
    ).to(tl.float32)
    partial_base = (reduction_group * channels + channel) * 10

    # Loading the activation gradients once and reducing every tap in one CTA
    # cuts the large N2048/B64 grid from channels*K9*groups to channels*groups.
    for tap in range(9):
        pair = 2 * step + tap - 8
        even_index = 2 * pair
        odd_index = even_index + 1
        raw_base = batch * input_steps
        even_value = tl.load(
            raw_inputs + raw_base + even_index,
            mask=valid_item & (even_index >= 0) & (even_index < input_steps),
            other=0.0,
        ).to(tl.float32)
        odd_value = tl.load(
            raw_inputs + raw_base + odd_index,
            mask=valid_item & (odd_index >= 0) & (odd_index < input_steps),
            other=0.0,
        ).to(tl.float32)
        low = 0.7071067811865476 * (even_value + odd_value)
        detail = 0.7071067811865476 * (even_value - odd_value)
        partial = tl.sum(low_gradient * low + detail_gradient * detail, axis=0)
        tl.store(parameter_partials + partial_base + tap, partial)

    bias_partial = tl.sum(low_gradient + detail_gradient, axis=0)
    tl.store(
        parameter_partials + partial_base + 9,
        bias_partial,
    )


@triton.jit
def _training_stem_grad_parameter_finalize_kernel(
    parameter_partials,
    grad_weight,
    grad_bias,
    reduction_groups: int,
    channels: int,
    block_groups: tl.constexpr,
) -> None:
    """Finish the deterministic split-K reduction without global atomics."""
    channel = tl.program_id(0)
    group = tl.arange(0, block_groups)
    valid_group = group < reduction_groups
    partial_base = (group * channels + channel) * 10
    for tap in range(9):
        partial = tl.load(
            parameter_partials + partial_base + tap,
            mask=valid_group,
            other=0.0,
        ).to(tl.float32)
        tl.store(grad_weight + channel * 9 + tap, tl.sum(partial, axis=0))
    bias_partial = tl.load(
        parameter_partials + partial_base + 9,
        mask=valid_group,
        other=0.0,
    ).to(tl.float32)
    tl.store(grad_bias + channel, tl.sum(bias_partial, axis=0))


@torch.library.triton_op("lnet::pac_pa2wp_training_stem_backward", mutates_args={})
def _pa2wp_training_stem_backward_op(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    stem_bias: Tensor,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_inputs(raw_inputs, stem_weight, stem_bias)
    _validate_grad_output(raw_inputs, stem_weight, grad_output)
    if not raw_inputs.is_cuda:
        return _reference_backward(raw_inputs, stem_weight, stem_bias, grad_output)
    raw = raw_inputs.contiguous()
    weight = stem_weight.contiguous()
    bias = stem_bias.contiguous()
    active_gradient = grad_output.contiguous()
    batch_count, input_steps, _ = raw.shape
    output_steps = (input_steps + 3) // 4
    channels = weight.shape[0]
    grad_activation = torch.empty_like(active_gradient)
    grad_input = torch.empty_like(raw)
    reduction_items = batch_count * output_steps
    use_split_parameter_reduction = reduction_items > _PARAMETER_SPLIT_THRESHOLD
    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty_like(bias)
    block_steps = min(triton.next_power_of_2(output_steps), 16)
    block_channels = triton.next_power_of_2(channels)
    torch.library.wrap_triton(_training_stem_grad_activation_kernel)[
        (batch_count, triton.cdiv(output_steps, block_steps))
    ](
        raw,
        weight,
        bias,
        active_gradient,
        grad_activation,
        batch_count,
        input_steps,
        output_steps,
        channels,
        block_steps=block_steps,
        block_channels=block_channels,
        num_warps=1 if reduction_items > _LARGE_WORKLOAD_ITEMS else 4,
    )
    block_inputs = min(triton.next_power_of_2(input_steps), 32)
    torch.library.wrap_triton(_training_stem_grad_input_kernel)[
        (batch_count, triton.cdiv(input_steps, block_inputs))
    ](
        weight,
        grad_activation,
        grad_input,
        batch_count,
        input_steps,
        output_steps,
        channels,
        block_inputs=block_inputs,
        block_channels=block_channels,
        num_warps=1 if reduction_items > _LARGE_WORKLOAD_ITEMS else 4,
    )
    reduction_block = _PARAMETER_REDUCTION_BLOCK
    if use_split_parameter_reduction:
        reduction_groups = (reduction_items + reduction_block - 1) // reduction_block
        parameter_partials = torch.empty(
            (reduction_groups, channels, _PARAMETER_PARTIALS_PER_CHANNEL),
            device=raw.device,
            dtype=raw.dtype,
        )
        torch.library.wrap_triton(
            _training_stem_grad_parameter_partials_kernel  # pyright: ignore[reportArgumentType]
        )[
            (channels, reduction_groups)
        ](
            raw,
            grad_activation,
            parameter_partials,
            batch_count,
            input_steps,
            output_steps,
            channels,
            reduction_block=reduction_block,
        )
        block_groups = triton.next_power_of_2(reduction_groups)
        torch.library.wrap_triton(_training_stem_grad_parameter_finalize_kernel)[
            (channels,)
        ](
            parameter_partials,
            grad_weight,
            grad_bias,
            reduction_groups,
            channels,
            block_groups=block_groups,
            num_warps=1,
        )
    else:
        torch.library.wrap_triton(_training_stem_grad_weight_kernel)[
            (channels, _STEM_KERNEL_SIZE)
        ](
            raw,
            grad_activation,
            grad_weight,
            batch_count,
            input_steps,
            output_steps,
            channels,
            reduction_block=reduction_block,
            num_warps=8,
        )
        torch.library.wrap_triton(_training_stem_grad_bias_kernel)[(channels,)](
            grad_activation,
            grad_bias,
            batch_count,
            output_steps,
            channels,
            reduction_block=reduction_block,
            num_warps=8,
        )
    return grad_input, grad_weight, grad_bias


@torch.library.triton_op("lnet::pac_pa2wp_training_stem", mutates_args={})
def _pa2wp_training_stem_op(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    stem_bias: Tensor,
) -> Tensor:
    _validate_inputs(raw_inputs, stem_weight, stem_bias)
    if not raw_inputs.is_cuda:
        return reference_pa2wp_training_stem(raw_inputs, stem_weight, stem_bias)
    raw = raw_inputs.contiguous()
    weight = stem_weight.contiguous()
    bias = stem_bias.contiguous()
    batch_count, input_steps, _ = raw.shape
    output_steps = (input_steps + 3) // 4
    channels = weight.shape[0]
    output = torch.empty(
        (2 * batch_count, output_steps, channels),
        device=raw.device,
        dtype=raw.dtype,
    )
    block_steps = min(triton.next_power_of_2(output_steps), 16)
    block_channels = triton.next_power_of_2(channels)
    torch.library.wrap_triton(_training_stem_forward_kernel)[
        (batch_count, triton.cdiv(output_steps, block_steps))
    ](
        raw,
        weight,
        bias,
        output,
        batch_count,
        input_steps,
        output_steps,
        channels,
        block_steps=block_steps,
        block_channels=block_channels,
        num_warps=(
            1 if batch_count * output_steps > _LARGE_WORKLOAD_ITEMS else 4
        ),
    )
    return output


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor],
    output: Tensor,
) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _backward(
    ctx: _AutogradContext,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    raw_inputs, stem_weight, stem_bias = ctx.saved_tensors
    if not raw_inputs.is_cuda:
        return _reference_backward(raw_inputs, stem_weight, stem_bias, grad_output)
    return _pa2wp_training_stem_backward_op(
        raw_inputs,
        stem_weight,
        stem_bias,
        grad_output,
    )


torch.library.register_autograd(
    "lnet::pac_pa2wp_training_stem",
    _backward,
    setup_context=_setup_context,
)


def pa2wp_training_stem(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    stem_bias: Tensor,
) -> Tensor:
    """Fuse selected-phase Haar packing and causal K9/S2 stem with first-order autograd."""
    return _pa2wp_training_stem_op(raw_inputs, stem_weight, stem_bias)


def reference_pa2wp_training_stem(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    stem_bias: Tensor,
) -> Tensor:
    """Apply the selected-phase PA2WP training stem with native PyTorch operations."""
    _validate_inputs(raw_inputs, stem_weight, stem_bias)
    padded_inputs = (
        raw_inputs
        if raw_inputs.shape[1] % 2 == 0
        else functional.pad(raw_inputs, (0, 0, 0, 1))
    )
    first = padded_inputs[:, 0::2]
    second = padded_inputs[:, 1::2]
    low = _INV_SQRT_TWO * (first + second)
    detail = _INV_SQRT_TWO * (first - second)
    bands = torch.cat((low, detail), dim=0)
    causal = functional.pad(bands.transpose(1, 2), (_STEM_KERNEL_SIZE - 1, 0))
    encoded = functional.conv1d(causal, stem_weight, stem_bias, stride=2)
    return functional.silu(encoded.transpose(1, 2))


def _reference_backward(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    stem_bias: Tensor,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    with torch.enable_grad():
        raw = raw_inputs.detach().requires_grad_()
        weight = stem_weight.detach().requires_grad_()
        bias = stem_bias.detach().requires_grad_()
        output = reference_pa2wp_training_stem(raw, weight, bias)
    grad_input, grad_weight, grad_bias = torch.autograd.grad(
        output,
        (raw, weight, bias),
        grad_output,
        allow_unused=False,
    )
    return grad_input, grad_weight, grad_bias


def _validate_inputs(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    stem_bias: Tensor,
) -> None:
    if raw_inputs.ndim != 3 or raw_inputs.shape[-1] != 1 or raw_inputs.shape[1] < 1:
        message = "PA2WP training stem requires raw inputs with shape [batch,time>=1,1]"
        raise ValueError(message)
    if stem_weight.ndim != 3 or stem_weight.shape[1:] != (1, _STEM_KERNEL_SIZE):
        message = "PA2WP training stem weight must have shape [channels,1,9]"
        raise ValueError(message)
    if stem_bias.shape != (stem_weight.shape[0],):
        message = "PA2WP training stem bias must have shape [channels]"
        raise ValueError(message)
    if stem_weight.shape[0] < 1:
        message = "PA2WP training stem requires at least one output channel"
        raise ValueError(message)
    for tensor in (stem_weight, stem_bias):
        if tensor.device != raw_inputs.device or tensor.dtype != raw_inputs.dtype:
            message = "PA2WP training stem tensors must share device and dtype"
            raise ValueError(message)
    if raw_inputs.dtype != torch.float32:
        message = "PA2WP training stem supports exact FP32 tensors only"
        raise TypeError(message)


def _validate_grad_output(
    raw_inputs: Tensor,
    stem_weight: Tensor,
    grad_output: Tensor,
) -> None:
    expected = (
        2 * raw_inputs.shape[0],
        (raw_inputs.shape[1] + 3) // 4,
        stem_weight.shape[0],
    )
    if grad_output.shape != expected:
        message = f"PA2WP training stem output gradient must have shape {expected}"
        raise ValueError(message)
    if grad_output.device != raw_inputs.device or grad_output.dtype != raw_inputs.dtype:
        message = "PA2WP training stem output gradient must share input device and dtype"
        raise ValueError(message)
