from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, N803, SLF001
import os
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    eps: float

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _rmsnorm_mean_forward_kernel(
    inputs,
    weight,
    output,
    n_steps: int,
    width: int,
    eps: float,
    BLOCK_TIME: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
) -> None:
    """Fuse row-wise RMSNorm with the following temporal mean.

    A program owns one batch element.  Tiled time rows expose enough parallelism
    for the width-32/64 PAC readouts while retaining the same FP32 sum-then-divide
    structure as ``torch.rms_norm(...).mean(dim=1)``.
    """
    batch = tl.program_id(0)
    time_lane = tl.arange(0, BLOCK_TIME)
    feature = tl.arange(0, BLOCK_WIDTH)
    feature_mask = feature < width
    feature_weight = tl.load(weight + feature, mask=feature_mask, other=0.0).to(tl.float32)
    pooled_sum = tl.zeros((BLOCK_WIDTH,), tl.float32)

    time_start = 0
    while time_start < n_steps:
        time = time_start + time_lane
        time_mask = time < n_steps
        offsets = (batch * n_steps + time[:, None]) * width + feature[None, :]
        mask = time_mask[:, None] & feature_mask[None, :]
        values = tl.load(inputs + offsets, mask=mask, other=0.0).to(tl.float32)
        square_sum = tl.sum(values * values, axis=1)
        inverse_rms = tl.rsqrt(square_sum / width + eps)
        normalized = values * inverse_rms[:, None] * feature_weight[None, :]
        pooled_sum += tl.sum(tl.where(time_mask[:, None], normalized, 0.0), axis=0)
        time_start += BLOCK_TIME

    tl.store(
        output + batch * width + feature,
        pooled_sum / n_steps,
        mask=feature_mask,
    )


@triton.jit
def _rmsnorm_mean_backward_kernel(
    inputs,
    weight,
    grad_output,
    grad_inputs,
    partial_grad_weight,
    n_steps: int,
    width: int,
    eps: float,
    BLOCK_TIME: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
) -> None:
    """Compute input gradients and a deterministic per-batch weight partial."""
    batch = tl.program_id(0)
    time_lane = tl.arange(0, BLOCK_TIME)
    feature = tl.arange(0, BLOCK_WIDTH)
    feature_mask = feature < width
    feature_weight = tl.load(weight + feature, mask=feature_mask, other=0.0).to(tl.float32)
    output_gradient = tl.load(
        grad_output + batch * width + feature,
        mask=feature_mask,
        other=0.0,
    ).to(tl.float32)
    weighted_output_gradient = feature_weight * output_gradient
    inverse_steps = 1.0 / n_steps
    inverse_width = 1.0 / width
    weight_gradient_sum = tl.zeros((BLOCK_WIDTH,), tl.float32)

    time_start = 0
    while time_start < n_steps:
        time = time_start + time_lane
        time_mask = time < n_steps
        offsets = (batch * n_steps + time[:, None]) * width + feature[None, :]
        mask = time_mask[:, None] & feature_mask[None, :]
        values = tl.load(inputs + offsets, mask=mask, other=0.0).to(tl.float32)
        square_sum = tl.sum(values * values, axis=1)
        inverse_rms = tl.rsqrt(square_sum * inverse_width + eps)
        projection = tl.sum(values * weighted_output_gradient[None, :], axis=1)
        correction = projection * inverse_width * inverse_rms * inverse_rms * inverse_rms
        input_gradient = inverse_steps * (
            weighted_output_gradient[None, :] * inverse_rms[:, None] - values * correction[:, None]
        )
        tl.store(grad_inputs + offsets, input_gradient, mask=mask)
        weight_terms = values * inverse_rms[:, None] * output_gradient[None, :]
        weight_gradient_sum += tl.sum(
            tl.where(time_mask[:, None], weight_terms, 0.0),
            axis=0,
        )
        time_start += BLOCK_TIME

    tl.store(
        partial_grad_weight + batch * width + feature,
        weight_gradient_sum * inverse_steps,
        mask=feature_mask,
    )


@triton.jit
def _rmsnorm_mean_weight_reduce_kernel(
    partial_grad_weight,
    grad_weight,
    batch_size: int,
    width: int,
    BLOCK_BATCH: tl.constexpr,
) -> None:
    """Reduce per-batch partials with a deterministic atomics-free tree."""
    feature = tl.program_id(0)
    batch_lane = tl.arange(0, BLOCK_BATCH)
    mask = (feature < width) & (batch_lane < batch_size)
    partial = tl.load(
        partial_grad_weight + batch_lane * width + feature,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(grad_weight + feature, tl.sum(partial, axis=0), mask=feature < width)


class _FusedRMSNormMeanTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _AutogradContext, inputs: Tensor, weight: Tensor, eps: float) -> Tensor:
        batch_size, n_steps, width = inputs.shape
        output = torch.empty((batch_size, width), dtype=inputs.dtype, device=inputs.device)
        block_width = triton.next_power_of_2(width)
        block_time = _resolve_block_time(block_width)
        _rmsnorm_mean_forward_kernel[(batch_size,)](
            inputs,
            weight,
            output,
            n_steps,
            width,
            eps,
            BLOCK_TIME=block_time,
            BLOCK_WIDTH=block_width,
            num_warps=4,
        )
        ctx.save_for_backward(inputs, weight)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx: _AutogradContext, *grad_outputs: Tensor) -> tuple[Tensor, Tensor, None]:
        (grad_output,) = grad_outputs
        return (*_rmsnorm_mean_backward(ctx, grad_output), None)


class _FusedRMSNormMeanBackwardTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _AutogradContext, inputs: Tensor, weight: Tensor, eps: float) -> Tensor:
        ctx.save_for_backward(inputs, weight)
        ctx.eps = eps
        return torch.nn.functional.rms_norm(
            inputs,
            (inputs.shape[-1],),
            weight,
            eps,
        ).mean(dim=1)

    @staticmethod
    def backward(ctx: _AutogradContext, *grad_outputs: Tensor) -> tuple[Tensor, Tensor, None]:
        (grad_output,) = grad_outputs
        return (*_rmsnorm_mean_backward(ctx, grad_output), None)


class _NativeRMSNormMeanBackwardTraining(torch.autograd.Function):
    """Bypass composite autograd while retaining ATen's exact FP32 kernels."""

    @staticmethod
    def forward(ctx: _AutogradContext, inputs: Tensor, weight: Tensor, eps: float) -> Tensor:
        normalized, inverse_rms = torch.ops.aten._fused_rms_norm.default(
            inputs,
            [inputs.shape[-1]],
            weight,
            eps,
        )
        ctx.save_for_backward(inputs, weight, inverse_rms)
        ctx.eps = eps
        return normalized.mean(dim=1)

    @staticmethod
    def backward(ctx: _AutogradContext, *grad_outputs: Tensor) -> tuple[Tensor, Tensor, None]:
        (grad_output,) = grad_outputs
        inputs, weight, inverse_rms = ctx.saved_tensors
        expanded_gradient = (grad_output / inputs.shape[1]).unsqueeze(1).expand_as(inputs)
        grad_inputs, grad_weight = torch.ops.aten._fused_rms_norm_backward.default(
            expanded_gradient,
            inputs,
            [inputs.shape[-1]],
            inverse_rms,
            weight,
            [True, True],
        )
        return grad_inputs, grad_weight, None


def _rmsnorm_mean_backward(ctx: _AutogradContext, grad_output: Tensor) -> tuple[Tensor, Tensor]:
    inputs, weight = ctx.saved_tensors
    batch_size, n_steps, width = inputs.shape
    grad_inputs = torch.empty_like(inputs)
    partial_grad_weight = torch.empty(
        (batch_size, width),
        dtype=weight.dtype,
        device=weight.device,
    )
    grad_weight = torch.empty_like(weight)
    block_width = triton.next_power_of_2(width)
    block_time = _resolve_block_time(block_width)
    _rmsnorm_mean_backward_kernel[(batch_size,)](
        inputs,
        weight,
        grad_output.contiguous(),
        grad_inputs,
        partial_grad_weight,
        n_steps,
        width,
        ctx.eps,
        BLOCK_TIME=block_time,
        BLOCK_WIDTH=block_width,
        num_warps=4,
    )
    block_batch = triton.next_power_of_2(batch_size)
    _rmsnorm_mean_weight_reduce_kernel[(width,)](
        partial_grad_weight,
        grad_weight,
        batch_size,
        width,
        BLOCK_BATCH=block_batch,
        num_warps=4,
    )
    return grad_inputs, grad_weight


def reference_rmsnorm_mean_training(
    inputs: Tensor,
    weight: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """Reference FP32 RMSNorm followed by an unmasked temporal mean."""
    _validate(inputs, weight, require_cuda=False)
    resolved_eps = torch.finfo(inputs.dtype).eps if eps is None else eps
    if resolved_eps <= 0.0:
        message = "eps must be positive"
        raise ValueError(message)
    return torch.nn.functional.rms_norm(inputs, (inputs.shape[-1],), weight, resolved_eps).mean(
        dim=1
    )


def fused_rmsnorm_mean_training(
    inputs: Tensor,
    weight: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """Apply the isolated exact-FP32 Triton training prototype.

    The operation intentionally supports only contiguous CUDA FP32 PAC readout
    tensors.  It is not wired into model dispatch; callers must opt in explicitly.
    """
    _validate(inputs, weight, require_cuda=True)
    resolved_eps = torch.finfo(inputs.dtype).eps if eps is None else eps
    if resolved_eps <= 0.0:
        message = "eps must be positive"
        raise ValueError(message)
    output = _FusedRMSNormMeanTraining.apply(inputs, weight, float(resolved_eps))
    if not isinstance(output, Tensor):
        message = "RMSNorm-mean training op returned an invalid output"
        raise TypeError(message)
    return output


def fused_rmsnorm_mean_backward_training(
    inputs: Tensor,
    weight: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """Keep the native forward order and fuse only the readout backward."""
    _validate(inputs, weight, require_cuda=True)
    resolved_eps = torch.finfo(inputs.dtype).eps if eps is None else eps
    if resolved_eps <= 0.0:
        message = "eps must be positive"
        raise ValueError(message)
    implementation = (
        _NativeRMSNormMeanBackwardTraining
        if os.environ.get("LNET_PAC_RMSNORM_MEAN_BACKWARD", "triton") == "native"
        else _FusedRMSNormMeanBackwardTraining
    )
    output = implementation.apply(inputs, weight, float(resolved_eps))
    if not isinstance(output, Tensor):
        message = "RMSNorm-mean backward training op returned an invalid output"
        raise TypeError(message)
    return output


def _resolve_block_time(block_width: int) -> int:
    configured = os.environ.get("LNET_PAC_RMSNORM_MEAN_BLOCK_TIME")
    if configured is None:
        return 8 if block_width <= 32 else 4
    block_time = int(configured)
    if block_time not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        message = "RMSNorm-mean block time must be a power of two from 1 through 512"
        raise ValueError(message)
    return block_time


def _validate(inputs: Tensor, weight: Tensor, *, require_cuda: bool) -> None:
    if inputs.ndim != 3:
        message = "inputs must have shape [batch, time, width]"
        raise ValueError(message)
    if weight.ndim != 1 or weight.shape[0] != inputs.shape[2]:
        message = "weight must have shape [width] matching inputs"
        raise ValueError(message)
    if inputs.shape[0] < 1 or inputs.shape[1] < 1 or inputs.shape[2] < 1:
        message = "batch, time, and width must be non-zero"
        raise ValueError(message)
    if inputs.device != weight.device:
        message = "inputs and weight must be on the same device"
        raise ValueError(message)
    if inputs.dtype != torch.float32 or weight.dtype != torch.float32:
        message = "the exact training prototype requires FP32 inputs and weight"
        raise TypeError(message)
    if require_cuda and (not inputs.is_cuda or not weight.is_cuda):
        message = "the Triton training prototype requires CUDA tensors"
        raise ValueError(message)
    if require_cuda and (not inputs.is_contiguous() or not weight.is_contiguous()):
        message = "the Triton training prototype requires contiguous tensors"
        raise ValueError(message)
