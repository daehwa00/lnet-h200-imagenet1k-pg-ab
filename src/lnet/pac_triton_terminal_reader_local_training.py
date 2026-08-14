from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional

_KERNEL_SIZE: Final[int] = 5
_DILATION: Final[int] = 4
_PADDING: Final[int] = 8
_PARAMETER_BLOCK_ITEMS: Final[int] = 32
_PARAMETER_BLOCK_CHANNELS: Final[int] = 8
_PARAMETER_COMPONENTS: Final[int] = 6
_PARAMETER_REDUCTION_BLOCK_SPLITS: Final[int] = 128


@triton.jit
def _terminal_reader_local_forward_kernel(
    inputs,
    weight,
    bias,
    output,
    preactivation,
    steps: int,
    channels: int,
    block_steps: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step = tl.program_id(1) * block_steps + tl.arange(0, block_steps)[:, None]
    channel = tl.program_id(2) * block_channels + tl.arange(0, block_channels)[None, :]
    valid_step = step < steps
    valid_channel = channel < channels
    local = tl.load(
        bias + channel,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    local = tl.broadcast_to(local, (block_steps, block_channels))

    for tap in range(5):
        source_step = step + tap * 4 - 8
        valid_source = valid_step & (source_step >= 0) & (source_step < steps)
        safe_source = tl.where(valid_source, source_step, 0)
        source_offset = (batch * steps + safe_source) * channels + channel
        source = tl.load(
            inputs + source_offset,
            mask=valid_source & valid_channel,
            other=0.0,
        ).to(tl.float32)
        tap_weight = tl.load(
            weight + channel * 5 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        local += source * tap_weight

    activated = local * tl.sigmoid(local)
    output_offset = (batch * steps + step) * channels + channel
    valid_output = valid_step & valid_channel
    tl.store(output + output_offset, activated, mask=valid_output)
    tl.store(preactivation + output_offset, local, mask=valid_output)


@triton.jit
def _terminal_reader_local_grad_input_kernel(
    grad_local,
    weight,
    grad_inputs,
    steps: int,
    channels: int,
    block_steps: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    input_step = tl.program_id(1) * block_steps + tl.arange(0, block_steps)[:, None]
    channel = tl.program_id(2) * block_channels + tl.arange(0, block_channels)[None, :]
    valid_input = input_step < steps
    valid_channel = channel < channels
    gradient = tl.full((block_steps, block_channels), 0.0, tl.float32)

    for tap in range(5):
        output_step = input_step - tap * 4 + 8
        valid_output = valid_input & (output_step >= 0) & (output_step < steps)
        safe_output = tl.where(valid_output, output_step, 0)
        output_offset = (batch * steps + safe_output) * channels + channel
        local_gradient = tl.load(
            grad_local + output_offset,
            mask=valid_output & valid_channel,
            other=0.0,
        ).to(tl.float32)
        tap_weight = tl.load(
            weight + channel * 5 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        gradient += local_gradient * tap_weight

    input_offset = (batch * steps + input_step) * channels + channel
    tl.store(
        grad_inputs + input_offset,
        gradient,
        mask=valid_input & valid_channel,
    )


@triton.jit
def _terminal_reader_local_grad_parameters_split_kernel(
    grad_output,
    preactivation,
    inputs,
    grad_local,
    partial_gradients,
    total_items: int,
    steps: int,
    channels: int,
    block_items: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    split = tl.program_id(0)
    flat_item = split * block_items + tl.arange(0, block_items)[:, None]
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)[None, :]
    valid_item = flat_item < total_items
    valid_channel = channel < channels
    batch = flat_item // steps
    output_step = flat_item - batch * steps
    output_offset = flat_item * channels + channel
    pre = tl.load(
        preactivation + output_offset,
        mask=valid_item & valid_channel,
        other=0.0,
    ).to(tl.float32)
    upstream = tl.load(
        grad_output + output_offset,
        mask=valid_item & valid_channel,
        other=0.0,
    ).to(tl.float32)
    sigmoid = tl.sigmoid(pre)
    local_gradient = upstream * sigmoid * (1.0 + pre * (1.0 - sigmoid))
    tl.store(
        grad_local + output_offset,
        local_gradient,
        mask=valid_item & valid_channel,
    )
    partial_base = split * 6 * channels + channel

    for tap in range(5):
        source_step = output_step + tap * 4 - 8
        valid_source = valid_item & (source_step >= 0) & (source_step < steps)
        safe_source = tl.where(valid_source, source_step, 0)
        source_offset = (batch * steps + safe_source) * channels + channel
        source = tl.load(
            inputs + source_offset,
            mask=valid_source & valid_channel,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            partial_gradients + partial_base + tap * channels,
            tl.sum(local_gradient * source, axis=0)[None, :],
            mask=valid_channel,
        )

    tl.store(
        partial_gradients + partial_base + 5 * channels,
        tl.sum(local_gradient, axis=0)[None, :],
        mask=valid_channel,
    )


@triton.jit
def _terminal_reader_local_grad_parameters_reduce_kernel(
    partial_gradients,
    grad_weight,
    grad_bias,
    num_splits: int,
    channels: int,
    block_splits: tl.constexpr,
) -> None:
    channel = tl.program_id(0)
    component = tl.program_id(1)
    scalar_offset = tl.arange(0, 1)
    split_offset = tl.arange(0, block_splits)
    accumulated = tl.zeros((1,), tl.float32)
    split_base = 0

    while split_base < num_splits:
        split = split_base + split_offset
        valid_split = split < num_splits
        offset = split * 6 * channels + component * channels + channel
        partial = tl.load(
            partial_gradients + offset,
            mask=valid_split,
            other=0.0,
        ).to(tl.float32)
        accumulated += tl.sum(partial, axis=0)
        split_base += block_splits

    if component < 5:
        tl.store(grad_weight + channel * 5 + component + scalar_offset, accumulated)
    else:
        tl.store(grad_bias + channel + scalar_offset, accumulated)


@torch.library.triton_op("lnet::pac_terminal_reader_local_training", mutates_args={})
def _terminal_reader_local_training_op(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_inputs(inputs, weight, bias)
    if not inputs.is_cuda:
        preactivation = _reference_preactivation(inputs, weight, bias)
        return functional.silu(preactivation), preactivation

    active_inputs = inputs.contiguous()
    active_weight = weight.contiguous()
    active_bias = bias.contiguous()
    batch, steps, channels = active_inputs.shape
    output = torch.empty_like(active_inputs)
    preactivation = torch.empty_like(active_inputs)
    block_steps = min(triton.next_power_of_2(steps), 8)
    block_channels = min(triton.next_power_of_2(channels), 32)
    grid = (
        batch,
        triton.cdiv(steps, block_steps),
        triton.cdiv(channels, block_channels),
    )
    torch.library.wrap_triton(_terminal_reader_local_forward_kernel)[grid](
        active_inputs,
        active_weight,
        active_bias,
        output,
        preactivation,
        steps,
        channels,
        block_steps=block_steps,
        block_channels=block_channels,
        num_warps=4,
    )
    return output, preactivation


@torch.library.triton_op(
    "lnet::pac_terminal_reader_local_training_backward",
    mutates_args={},
)
def _terminal_reader_local_training_backward_op(
    grad_output: Tensor,
    preactivation: Tensor,
    inputs: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if not grad_output.is_cuda:
        return _reference_backward(grad_output, preactivation, inputs, weight)

    upstream = grad_output.contiguous()
    active_preactivation = preactivation.contiguous()
    active_inputs = inputs.contiguous()
    active_weight = weight.contiguous()
    batch, steps, channels = active_inputs.shape
    total_items = batch * steps
    grad_local = torch.empty_like(active_inputs)
    grad_inputs = torch.empty_like(active_inputs)
    grad_weight = torch.empty_like(active_weight)
    grad_bias = torch.empty(
        (channels,),
        device=active_inputs.device,
        dtype=active_inputs.dtype,
    )

    parameter_block_items = _PARAMETER_BLOCK_ITEMS
    parameter_block_channels = min(
        triton.next_power_of_2(channels),
        _PARAMETER_BLOCK_CHANNELS,
    )
    num_splits = int(triton.cdiv(total_items, parameter_block_items))
    partial_gradients = torch.empty(
        (num_splits, _PARAMETER_COMPONENTS, channels),
        device=active_inputs.device,
        dtype=active_inputs.dtype,
    )
    parameter_grid = (
        num_splits,
        triton.cdiv(channels, parameter_block_channels),
    )
    torch.library.wrap_triton(_terminal_reader_local_grad_parameters_split_kernel)[parameter_grid](
        upstream,
        active_preactivation,
        active_inputs,
        grad_local,
        partial_gradients,
        total_items,
        steps,
        channels,
        block_items=parameter_block_items,
        block_channels=parameter_block_channels,
        num_warps=1,
    )

    block_steps = min(triton.next_power_of_2(steps), 8)
    block_channels = min(triton.next_power_of_2(channels), 16)
    input_grid = (
        batch,
        triton.cdiv(steps, block_steps),
        triton.cdiv(channels, block_channels),
    )
    torch.library.wrap_triton(_terminal_reader_local_grad_input_kernel)[input_grid](
        grad_local,
        active_weight,
        grad_inputs,
        steps,
        channels,
        block_steps=block_steps,
        block_channels=block_channels,
        num_warps=2,
    )
    torch.library.wrap_triton(_terminal_reader_local_grad_parameters_reduce_kernel)[
        (channels, _PARAMETER_COMPONENTS)
    ](
        partial_gradients,
        grad_weight,
        grad_bias,
        num_splits,
        channels,
        block_splits=_PARAMETER_REDUCTION_BLOCK_SPLITS,
        num_warps=1,
    )
    return grad_inputs, grad_weight, grad_bias


def terminal_reader_local_training(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    """Fuse BenchmarkAlphabetBackbone's maskless reader DWConv5-D4 and SiLU.

    Inputs and outputs use the model-native contiguous ``[batch,time,channel]``
    layout. The weight and bias map directly to ``second_local.weight`` and
    ``second_local.bias``. The isolated API deliberately does not dispatch from
    the model until the candidate clears its parity and speed gate.
    """
    output, _preactivation = _terminal_reader_local_training_op(inputs, weight, bias)
    return output


def reference_terminal_reader_local_training(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    """Differentiable eager contract for the fused terminal-reader candidate."""
    _validate_inputs(inputs, weight, bias)
    return functional.silu(_reference_preactivation(inputs, weight, bias))


def _reference_preactivation(inputs: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return functional.conv1d(
        inputs.transpose(1, 2),
        weight,
        bias,
        padding=_PADDING,
        dilation=_DILATION,
        groups=inputs.shape[-1],
    ).transpose(1, 2)


def _reference_backward(
    grad_output: Tensor,
    preactivation: Tensor,
    inputs: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    sigmoid = torch.sigmoid(preactivation)
    grad_local = grad_output * sigmoid * (1.0 + preactivation * (1.0 - sigmoid))
    channels = inputs.shape[-1]
    grad_inputs = functional.conv_transpose1d(
        grad_local.transpose(1, 2),
        weight,
        padding=_PADDING,
        dilation=_DILATION,
        groups=channels,
    ).transpose(1, 2)
    grad_weight = torch.nn.grad.conv1d_weight(  # pyright: ignore[reportAttributeAccessIssue]
        inputs.transpose(1, 2),
        weight.shape,
        grad_local.transpose(1, 2),
        padding=_PADDING,
        dilation=_DILATION,
        groups=channels,
    )
    grad_bias = grad_local.sum(dim=(0, 1))
    return grad_inputs, grad_weight, grad_bias


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor],
) -> None:
    preactivation = output[1]
    active_inputs, weight, _bias = inputs
    ctx.mark_non_differentiable(preactivation)
    ctx.save_for_backward(active_inputs, weight, preactivation)


def _backward(
    ctx: _AutogradContext,
    grad_output: Tensor,
    _grad_preactivation: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    del _grad_preactivation
    inputs, weight, preactivation = ctx.saved_tensors
    return _terminal_reader_local_training_backward_op(
        grad_output,
        preactivation,
        inputs,
        weight,
    )


torch.library.register_autograd(
    "lnet::pac_terminal_reader_local_training",
    _backward,
    setup_context=_setup_context,
)


def _validate_inputs(inputs: Tensor, weight: Tensor, bias: Tensor) -> None:
    if inputs.ndim != 3:
        message = "terminal-reader local inputs must have shape [batch,time,channel]"
        raise ValueError(message)
    channels = inputs.shape[-1]
    if channels < 1 or inputs.shape[1] < 1:
        message = "terminal-reader local inputs require positive time and channel dimensions"
        raise ValueError(message)
    if weight.shape != (channels, 1, _KERNEL_SIZE):
        message = "terminal-reader local weight must have shape [channels,1,5]"
        raise ValueError(message)
    if bias.shape != (channels,):
        message = "terminal-reader local bias must have shape [channels]"
        raise ValueError(message)
    for tensor in (weight, bias):
        if tensor.device != inputs.device or tensor.dtype != inputs.dtype:
            message = "terminal-reader local tensors must share device and dtype"
            raise ValueError(message)
    if inputs.dtype != torch.float32:
        message = "terminal-reader local training supports exact FP32 only"
        raise TypeError(message)
