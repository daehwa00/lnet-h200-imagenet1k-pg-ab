"""Fused direct ``Linear(C,D) -> DWConv5(d4) -> SiLU`` stem kernels.

Q1 classification is predominantly scalar-input while several external tasks
have two raw channels.  Both C=1 and C=2 use one shape-specialized Triton
kernel family.  The projection is consumed by the following depthwise
convolution without materializing an intermediate ``[B,T,D]`` tensor.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, N803
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional

_KERNEL_SIZE: Final[int] = 5
_DILATION: Final[int] = 4
_PADDING: Final[int] = 8
_PARAMETER_BLOCK_ITEMS: Final[int] = 64
_PARAMETER_BLOCK_CHANNELS: Final[int] = 32
_REDUCTION_BLOCK_SPLITS: Final[int] = 128


class _AutogradContext(Protocol):
    needs_input_grad: tuple[bool, ...]
    saved_tensors: tuple[Tensor, ...]

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _direct_stem_c2_forward_kernel(
    raw_inputs,
    projection_weight,
    local_weight,
    local_bias,
    output,
    preactivation,
    steps: int,
    channels: int,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RAW_CHANNELS: tl.constexpr,
    STORE_PREACTIVATION: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    channel = tl.arange(0, BLOCK_D)[None, :]
    valid_step = step < steps
    valid_channel = channel < channels
    projection_0 = tl.load(
        projection_weight + channel * RAW_CHANNELS,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_1 = tl.zeros((1, BLOCK_D), tl.float32)
    if RAW_CHANNELS == 2:
        projection_1 = tl.load(
            projection_weight + channel * RAW_CHANNELS + 1,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
    local = tl.broadcast_to(
        tl.load(local_bias + channel, mask=valid_channel, other=0.0).to(tl.float32),
        (BLOCK_T, BLOCK_D),
    )

    for tap in range(5):
        source_step = step + tap * 4 - 8
        valid_source = valid_step & (source_step >= 0) & (source_step < steps)
        safe_source = tl.where(valid_source, source_step, 0)
        raw_offset = (batch * steps + safe_source) * RAW_CHANNELS
        raw_0 = tl.load(raw_inputs + raw_offset, mask=valid_source, other=0.0).to(
            tl.float32
        )
        projected = raw_0 * projection_0
        if RAW_CHANNELS == 2:
            raw_1 = tl.load(
                raw_inputs + raw_offset + 1,
                mask=valid_source,
                other=0.0,
            ).to(tl.float32)
            projected += raw_1 * projection_1
        tap_weight = tl.load(
            local_weight + channel * 5 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        local += projected * tap_weight

    activated = local * tl.sigmoid(local)
    output_offset = (batch * steps + step) * channels + channel
    valid_output = valid_step & valid_channel
    tl.store(output + output_offset, activated, mask=valid_output)
    if STORE_PREACTIVATION:
        tl.store(preactivation + output_offset, local, mask=valid_output)


@triton.jit
def _direct_stem_c2_grad_raw_kernel(
    grad_output,
    preactivation,
    projection_weight,
    local_weight,
    grad_raw_inputs,
    steps: int,
    channels: int,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RAW_CHANNELS: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    raw_step = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    channel = tl.arange(0, BLOCK_D)[None, :]
    valid_raw = raw_step < steps
    valid_channel = channel < channels
    projection_0 = tl.load(
        projection_weight + channel * RAW_CHANNELS,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_1 = tl.zeros((1, BLOCK_D), tl.float32)
    if RAW_CHANNELS == 2:
        projection_1 = tl.load(
            projection_weight + channel * RAW_CHANNELS + 1,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
    grad_raw_0 = tl.zeros((BLOCK_T,), tl.float32)
    grad_raw_1 = tl.zeros((BLOCK_T,), tl.float32)

    for tap in range(5):
        output_step = raw_step[:, None] - tap * 4 + 8
        valid_output = (
            valid_raw[:, None]
            & (output_step >= 0)
            & (output_step < steps)
            & valid_channel
        )
        safe_output = tl.where(valid_output, output_step, 0)
        output_offset = (batch * steps + safe_output) * channels + channel
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
        tap_weight = tl.load(
            local_weight + channel * 5 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        grad_projected = grad_local * tap_weight
        grad_raw_0 += tl.sum(grad_projected * projection_0, axis=1)
        if RAW_CHANNELS == 2:
            grad_raw_1 += tl.sum(grad_projected * projection_1, axis=1)

    raw_offset = (batch * steps + raw_step) * RAW_CHANNELS
    tl.store(grad_raw_inputs + raw_offset, grad_raw_0, mask=valid_raw)
    if RAW_CHANNELS == 2:
        tl.store(grad_raw_inputs + raw_offset + 1, grad_raw_1, mask=valid_raw)


@triton.jit
def _direct_stem_c2_grad_parameters_split_kernel(
    grad_output,
    preactivation,
    raw_inputs,
    projection_weight,
    local_weight,
    partial_gradients,
    batch_size: int,
    steps: int,
    channels: int,
    BLOCK_ITEMS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    RAW_CHANNELS: tl.constexpr,
    PARAMETER_COMPONENTS: tl.constexpr,
) -> None:
    split = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    flat_item = split * BLOCK_ITEMS + tl.arange(0, BLOCK_ITEMS)
    total_items = batch_size * steps
    valid_item = flat_item < total_items
    valid_channel = channel < channels
    batch = flat_item // steps
    output_step = flat_item - batch * steps
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
    projection_0 = tl.load(
        projection_weight + channel * RAW_CHANNELS,
        mask=valid_channel,
        other=0.0,
    ).to(tl.float32)
    projection_1 = tl.zeros((BLOCK_D,), tl.float32)
    if RAW_CHANNELS == 2:
        projection_1 = tl.load(
            projection_weight + channel * RAW_CHANNELS + 1,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
    grad_projection_0 = tl.zeros((BLOCK_D,), tl.float32)
    grad_projection_1 = tl.zeros((BLOCK_D,), tl.float32)
    partial_base = split * PARAMETER_COMPONENTS * channels + channel

    for tap in range(5):
        source_step = output_step + tap * 4 - 8
        valid_source = valid_item & (source_step >= 0) & (source_step < steps)
        safe_source = tl.where(valid_source, source_step, 0)
        raw_offset = (batch * steps + safe_source) * RAW_CHANNELS
        raw_0 = tl.load(raw_inputs + raw_offset, mask=valid_source, other=0.0).to(
            tl.float32
        )
        projected = raw_0[:, None] * projection_0
        if RAW_CHANNELS == 2:
            raw_1 = tl.load(
                raw_inputs + raw_offset + 1,
                mask=valid_source,
                other=0.0,
            ).to(tl.float32)
            projected += raw_1[:, None] * projection_1
        local_weight_value = tl.load(
            local_weight + channel * 5 + tap,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            partial_gradients + partial_base + (RAW_CHANNELS + tap) * channels,
            tl.sum(grad_local * projected, axis=0),
            mask=valid_channel,
        )
        grad_projection_input = grad_local * local_weight_value[None, :]
        grad_projection_0 += tl.sum(grad_projection_input * raw_0[:, None], axis=0)
        if RAW_CHANNELS == 2:
            grad_projection_1 += tl.sum(grad_projection_input * raw_1[:, None], axis=0)

    tl.store(partial_gradients + partial_base, grad_projection_0, mask=valid_channel)
    tl.store(
        partial_gradients + partial_base + channels,
        grad_projection_1,
        mask=valid_channel & (RAW_CHANNELS == 2),
    )
    tl.store(
        partial_gradients + partial_base + (RAW_CHANNELS + 5) * channels,
        tl.sum(grad_local, axis=0),
        mask=valid_channel,
    )


@triton.jit
def _direct_stem_c2_grad_parameters_reduce_tiles_kernel(
    partial_gradients,
    reduced_gradients,
    num_splits: int,
    channels: int,
    BLOCK_SPLITS: tl.constexpr,
    PARAMETER_COMPONENTS: tl.constexpr,
) -> None:
    output_split = tl.program_id(0)
    channel = tl.program_id(1)
    component = tl.arange(0, 8)[:, None]
    local_split = tl.arange(0, BLOCK_SPLITS)[None, :]
    split = output_split * BLOCK_SPLITS + local_split
    valid = (split < num_splits) & (component < PARAMETER_COMPONENTS)
    partial_offset = (
        split * PARAMETER_COMPONENTS * channels + component * channels + channel
    )
    values = tl.load(
        partial_gradients + partial_offset,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    reduced_offset = (
        output_split * PARAMETER_COMPONENTS * channels
        + tl.arange(0, 8) * channels
        + channel
    )
    tl.store(
        reduced_gradients + reduced_offset,
        tl.sum(values, axis=1),
        mask=tl.arange(0, 8) < PARAMETER_COMPONENTS,
    )


@triton.jit
def _direct_stem_c2_grad_parameters_reduce_kernel(
    partial_gradients,
    grad_projection_weight,
    grad_local_weight,
    grad_local_bias,
    num_splits: int,
    channels: int,
    BLOCK_SPLITS: tl.constexpr,
    RAW_CHANNELS: tl.constexpr,
    PARAMETER_COMPONENTS: tl.constexpr,
) -> None:
    channel = tl.program_id(0)
    component = tl.arange(0, 8)[:, None]
    split = tl.arange(0, BLOCK_SPLITS)[None, :]
    valid = (split < num_splits) & (component < PARAMETER_COMPONENTS)
    partial_offset = (
        split * PARAMETER_COMPONENTS * channels + component * channels + channel
    )
    values = tl.load(partial_gradients + partial_offset, mask=valid, other=0.0).to(tl.float32)
    accumulated = tl.sum(values, axis=1)
    final_component = tl.arange(0, 8)
    tl.store(
        grad_projection_weight + channel * RAW_CHANNELS + final_component,
        accumulated,
        mask=final_component < RAW_CHANNELS,
    )
    tl.store(
        grad_local_weight + channel * 5 + final_component - RAW_CHANNELS,
        accumulated,
        mask=(final_component >= RAW_CHANNELS)
        & (final_component < RAW_CHANNELS + 5),
    )
    bias = tl.sum(
        tl.where(final_component == RAW_CHANNELS + 5, accumulated, 0.0), axis=0
    )
    tl.store(grad_local_bias + channel, bias)


def _reference_preactivation(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    projected = functional.linear(raw_inputs, projection_weight)
    return functional.conv1d(
        projected.transpose(1, 2),
        local_weight,
        local_bias,
        padding=_PADDING,
        dilation=_DILATION,
        groups=projection_weight.shape[0],
    ).transpose(1, 2)


def _validate_inputs(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> None:
    if (
        raw_inputs.ndim != 3
        or raw_inputs.shape[-1] not in (1, 2)
        or raw_inputs.shape[1] < 1
    ):
        message = "direct stem requires raw inputs with shape [batch,time>=1,C], C in {1,2}"
        raise ValueError(message)
    raw_channels = raw_inputs.shape[-1]
    if projection_weight.ndim != 2 or projection_weight.shape[1] != raw_channels:
        message = "direct projection weight must have shape [channels,raw_channels]"
        raise ValueError(message)
    channels = projection_weight.shape[0]
    if local_weight.shape != (channels, 1, _KERNEL_SIZE):
        message = "direct stem local weight must have shape [channels,1,5]"
        raise ValueError(message)
    if local_bias.shape != (channels,):
        message = "direct stem local bias must have shape [channels]"
        raise ValueError(message)
    for tensor in (projection_weight, local_weight, local_bias):
        if tensor.device != raw_inputs.device or tensor.dtype != raw_inputs.dtype:
            message = "direct stem tensors must share device and dtype"
            raise ValueError(message)
    if raw_inputs.dtype != torch.float32:
        message = "direct stem fusion supports FP32 only"
        raise TypeError(message)


@triton_op(
    "lnet::pac_direct_stem_c2_training_hierarchical_v2",
    mutates_args={},
)
def _forward_hierarchical_v2_op(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> tuple[Tensor, Tensor]:
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
    batch, steps, _raw_channels = raw.shape
    raw_channels = raw.shape[-1]
    channels = projection.shape[0]
    output = torch.empty((batch, steps, channels), device=raw.device, dtype=raw.dtype)
    preactivation = torch.empty_like(output)
    block_steps = min(triton.next_power_of_2(steps), 32)
    block_channels = triton.next_power_of_2(channels)
    wrap_triton(_direct_stem_c2_forward_kernel)[
        (batch, triton.cdiv(steps, block_steps))
    ](
        raw,
        projection,
        local,
        bias,
        output,
        preactivation,
        steps,
        channels,
        BLOCK_T=block_steps,
        BLOCK_D=block_channels,
        RAW_CHANNELS=raw_channels,
        STORE_PREACTIVATION=True,
        num_warps=8,
    )
    return output, preactivation


@triton_op("lnet::pac_direct_stem_c2_inference", mutates_args={})
def _inference_op(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    _validate_inputs(raw_inputs, projection_weight, local_weight, local_bias)
    if not raw_inputs.is_cuda:
        return functional.silu(
            _reference_preactivation(
                raw_inputs,
                projection_weight,
                local_weight,
                local_bias,
            )
        )
    raw = raw_inputs.contiguous()
    projection = projection_weight.contiguous()
    local = local_weight.contiguous()
    bias = local_bias.contiguous()
    batch, steps, _raw_channels = raw.shape
    raw_channels = raw.shape[-1]
    channels = projection.shape[0]
    output = torch.empty((batch, steps, channels), device=raw.device, dtype=raw.dtype)
    block_steps = min(triton.next_power_of_2(steps), 32)
    block_channels = triton.next_power_of_2(channels)
    wrap_triton(_direct_stem_c2_forward_kernel)[
        (batch, triton.cdiv(steps, block_steps))
    ](
        raw,
        projection,
        local,
        bias,
        output,
        output,
        steps,
        channels,
        BLOCK_T=block_steps,
        BLOCK_D=block_channels,
        RAW_CHANNELS=raw_channels,
        STORE_PREACTIVATION=False,
        num_warps=8,
    )
    return output


def _reference_backward(
    grad_output: Tensor,
    preactivation: Tensor,
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    sigmoid = torch.sigmoid(preactivation)
    grad_local = grad_output * sigmoid * (1.0 + preactivation * (1.0 - sigmoid))
    projected = functional.linear(raw_inputs, projection_weight)
    grad_projected = functional.conv_transpose1d(
        grad_local.transpose(1, 2),
        local_weight,
        padding=_PADDING,
        dilation=_DILATION,
        groups=projection_weight.shape[0],
    ).transpose(1, 2)
    grad_local_weight = torch.nn.grad.conv1d_weight(  # pyright: ignore[reportAttributeAccessIssue]
        projected.transpose(1, 2),
        local_weight.shape,
        grad_local.transpose(1, 2),
        padding=_PADDING,
        dilation=_DILATION,
        groups=projection_weight.shape[0],
    )
    grad_local_bias = grad_local.sum(dim=(0, 1))
    grad_projection_weight = torch.matmul(
        grad_projected.reshape(-1, projection_weight.shape[0]).T,
        raw_inputs.reshape(-1, raw_inputs.shape[-1]),
    )
    grad_raw_inputs = torch.matmul(grad_projected, projection_weight)
    return grad_raw_inputs, grad_projection_weight, grad_local_weight, grad_local_bias


@triton_op(
    "lnet::pac_direct_stem_c2_training_backward_hierarchical_v2",
    mutates_args={},
)
def _backward_hierarchical_v2_op(
    grad_output: Tensor,
    preactivation: Tensor,
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    compute_grad_raw: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not grad_output.is_cuda:
        return _reference_backward(
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
    batch, steps, _raw_channels = raw.shape
    raw_channels = raw.shape[-1]
    parameter_components = raw_channels + _KERNEL_SIZE + 1
    channels = projection.shape[0]
    grad_raw_inputs = torch.empty_like(raw)
    grad_projection_weight = torch.empty_like(projection)
    grad_local_weight = torch.empty_like(local)
    grad_local_bias = torch.empty((channels,), device=raw.device, dtype=raw.dtype)
    block_steps = min(triton.next_power_of_2(steps), 16)
    block_channels = triton.next_power_of_2(channels)
    if compute_grad_raw:
        wrap_triton(_direct_stem_c2_grad_raw_kernel)[
            (batch, triton.cdiv(steps, block_steps))
        ](
            upstream,
            active_preactivation,
            projection,
            local,
            grad_raw_inputs,
            steps,
            channels,
            BLOCK_T=block_steps,
            BLOCK_D=block_channels,
            RAW_CHANNELS=raw_channels,
            num_warps=8,
        )

    num_splits = int(triton.cdiv(batch * steps, _PARAMETER_BLOCK_ITEMS))
    partial_gradients = torch.empty(
        (num_splits, parameter_components, channels),
        device=raw.device,
        dtype=raw.dtype,
    )
    wrap_triton(_direct_stem_c2_grad_parameters_split_kernel)[
        (num_splits, triton.cdiv(channels, _PARAMETER_BLOCK_CHANNELS))
    ](
        upstream,
        active_preactivation,
        raw,
        projection,
        local,
        partial_gradients,
        batch,
        steps,
        channels,
        BLOCK_ITEMS=_PARAMETER_BLOCK_ITEMS,
        BLOCK_D=_PARAMETER_BLOCK_CHANNELS,
        RAW_CHANNELS=raw_channels,
        PARAMETER_COMPONENTS=parameter_components,
        num_warps=4,
    )
    while num_splits > _REDUCTION_BLOCK_SPLITS:
        reduced_splits = int(triton.cdiv(num_splits, _REDUCTION_BLOCK_SPLITS))
        reduced_gradients = torch.empty(
            (reduced_splits, parameter_components, channels),
            device=raw.device,
            dtype=raw.dtype,
        )
        wrap_triton(_direct_stem_c2_grad_parameters_reduce_tiles_kernel)[
            (reduced_splits, channels)
        ](
            partial_gradients,
            reduced_gradients,
            num_splits,
            channels,
            BLOCK_SPLITS=_REDUCTION_BLOCK_SPLITS,
            PARAMETER_COMPONENTS=parameter_components,
            num_warps=1,
        )
        partial_gradients = reduced_gradients
        num_splits = reduced_splits
    wrap_triton(_direct_stem_c2_grad_parameters_reduce_kernel)[(channels,)](
        partial_gradients,
        grad_projection_weight,
        grad_local_weight,
        grad_local_bias,
        num_splits,
        channels,
        BLOCK_SPLITS=_REDUCTION_BLOCK_SPLITS,
        RAW_CHANNELS=raw_channels,
        PARAMETER_COMPONENTS=parameter_components,
        num_warps=1,
    )
    return grad_raw_inputs, grad_projection_weight, grad_local_weight, grad_local_bias


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor],
    output: tuple[Tensor, Tensor],
) -> None:
    preactivation = output[1]
    ctx.mark_non_differentiable(preactivation)
    ctx.save_for_backward(*inputs, preactivation)


def _backward(
    ctx: _AutogradContext,
    grad_output: Tensor,
    _grad_preactivation: Tensor | None,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
    del _grad_preactivation
    raw_inputs, projection_weight, local_weight, local_bias, preactivation = ctx.saved_tensors
    del local_bias
    compute_grad_raw = ctx.needs_input_grad[0]
    gradients = _backward_hierarchical_v2_op(
        grad_output,
        preactivation,
        raw_inputs,
        projection_weight,
        local_weight,
        compute_grad_raw,
    )
    return (gradients[0] if compute_grad_raw else None, *gradients[1:])


torch.library.register_autograd(
    "lnet::pac_direct_stem_c2_training_hierarchical_v2",
    _backward,
    setup_context=_setup_context,
)


def _require_raw_channels(raw_inputs: Tensor, expected: int) -> None:
    if raw_inputs.ndim != 3 or raw_inputs.shape[-1] != expected:
        message = f"direct C={expected} stem requires raw inputs with shape [batch,time,{expected}]"
        raise ValueError(message)


def direct_stem_c1_training(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Return the fused scalar-input direct stem with exact autograd."""
    _require_raw_channels(raw_inputs, 1)
    output, _preactivation = _forward_hierarchical_v2_op(
        raw_inputs,
        projection_weight,
        local_weight,
        local_bias,
    )
    return output


def direct_stem_c2_training(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Return the fused direct stem output while preserving exact autograd."""
    _require_raw_channels(raw_inputs, 2)
    output, _preactivation = _forward_hierarchical_v2_op(
        raw_inputs,
        projection_weight,
        local_weight,
        local_bias,
    )
    return output


def direct_stem_c1_inference(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Return the output-only scalar-input CUDA direct stem."""
    _require_raw_channels(raw_inputs, 1)
    return _inference_op(raw_inputs, projection_weight, local_weight, local_bias)


def direct_stem_c2_inference(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Return the output-only CUDA direct stem."""
    _require_raw_channels(raw_inputs, 2)
    return _inference_op(raw_inputs, projection_weight, local_weight, local_bias)


def reference_direct_stem_c1_training(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Differentiable eager contract for the direct C=1 stem."""
    _require_raw_channels(raw_inputs, 1)
    _validate_inputs(raw_inputs, projection_weight, local_weight, local_bias)
    return functional.silu(
        _reference_preactivation(
            raw_inputs,
            projection_weight,
            local_weight,
            local_bias,
        )
    )


def reference_direct_stem_c2_training(
    raw_inputs: Tensor,
    projection_weight: Tensor,
    local_weight: Tensor,
    local_bias: Tensor,
) -> Tensor:
    """Differentiable eager contract for the direct C=2 stem."""
    _require_raw_channels(raw_inputs, 2)
    _validate_inputs(raw_inputs, projection_weight, local_weight, local_bias)
    return functional.silu(
        _reference_preactivation(
            raw_inputs,
            projection_weight,
            local_weight,
            local_bias,
        )
    )


__all__ = [
    "direct_stem_c1_inference",
    "direct_stem_c1_training",
    "direct_stem_c2_inference",
    "direct_stem_c2_training",
    "reference_direct_stem_c1_training",
    "reference_direct_stem_c2_training",
]
