from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001
from typing import Final, Literal, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor

OrderedPoolBackend = Literal["auto", "reference", "triton"]

_LEVELS: Final[tuple[int, int, int]] = (1, 2, 4)
_BIN_COUNT: Final[int] = 7
_TRITON_PERSISTENT_STEP_LIMIT: Final[int] = 384


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    eps: float

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _ordered_pool_kernel(
    inputs,
    weight,
    output,
    n_steps: int,
    width: int,
    eps: float,
    block_width: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    feature_offsets = tl.arange(0, block_width)
    feature_mask = feature_offsets < width
    feature_weight = tl.load(weight + feature_offsets, mask=feature_mask, other=0.0).to(tl.float32)

    one_sum = tl.full((block_width,), 0.0, tl.float32)
    two0_sum = tl.full((block_width,), 0.0, tl.float32)
    two1_sum = tl.full((block_width,), 0.0, tl.float32)
    four0_sum = tl.full((block_width,), 0.0, tl.float32)
    four1_sum = tl.full((block_width,), 0.0, tl.float32)
    four2_sum = tl.full((block_width,), 0.0, tl.float32)
    four3_sum = tl.full((block_width,), 0.0, tl.float32)

    two_q = n_steps // 2
    two_r = n_steps - two_q * 2
    two0_end = two_q + tl.where(two_r > 0, 1, 0)
    two1_start = two0_end

    four_q = n_steps // 4
    four_r = n_steps - four_q * 4
    four0_len = four_q + tl.where(four_r > 0, 1, 0)
    four1_len = four_q + tl.where(four_r > 1, 1, 0)
    four2_len = four_q + tl.where(four_r > 2, 1, 0)
    four3_len = four_q
    four1_start = four0_len
    four2_start = four1_start + four1_len
    four3_start = four2_start + four2_len

    time_index = 0
    while time_index < n_steps:
        input_row_base = batch * n_steps * width + time_index * width
        row = tl.load(
            inputs + input_row_base + feature_offsets,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(row * row, axis=0)
        scale = tl.rsqrt(square_sum / width + eps)
        normalized = row * scale * feature_weight
        one_sum += normalized
        two0_sum += tl.where(time_index < two0_end, normalized, 0.0)
        two1_sum += tl.where(time_index >= two1_start, normalized, 0.0)
        four0_sum += tl.where(time_index < four0_len, normalized, 0.0)
        four1_sum += tl.where(
            (time_index >= four1_start) & (time_index < four2_start),
            normalized,
            0.0,
        )
        four2_sum += tl.where(
            (time_index >= four2_start) & (time_index < four3_start),
            normalized,
            0.0,
        )
        four3_sum += tl.where(time_index >= four3_start, normalized, 0.0)
        time_index += 1

    output_base = batch * 7 * width + feature_offsets
    tl.store(output + output_base, one_sum / n_steps, mask=feature_mask)
    tl.store(
        output + output_base + width,
        two0_sum / two0_end,
        mask=feature_mask,
    )
    two1_len = n_steps - two1_start
    tl.store(
        output + output_base + 2 * width,
        two1_sum / tl.maximum(two1_len, 1),
        mask=feature_mask,
    )
    tl.store(
        output + output_base + 3 * width,
        four0_sum / tl.maximum(four0_len, 1),
        mask=feature_mask,
    )
    tl.store(
        output + output_base + 4 * width,
        four1_sum / tl.maximum(four1_len, 1),
        mask=feature_mask,
    )
    tl.store(
        output + output_base + 5 * width,
        four2_sum / tl.maximum(four2_len, 1),
        mask=feature_mask,
    )
    tl.store(
        output + output_base + 6 * width,
        four3_sum / tl.maximum(four3_len, 1),
        mask=feature_mask,
    )


@torch.library.triton_op("lnet::final_rmsnorm_ordered_pool", mutates_args={})
def _triton_final_rmsnorm_ordered_pool(inputs: Tensor, weight: Tensor, *, eps: float) -> Tensor:
    batch, n_steps, width = inputs.shape
    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    output = torch.empty((batch, _BIN_COUNT * width), dtype=inputs.dtype, device=inputs.device)
    block_width = triton.next_power_of_2(width)
    grid = (batch,)
    torch.library.wrap_triton(_ordered_pool_kernel)[grid](
        contiguous_inputs,
        contiguous_weight,
        output,
        n_steps,
        width,
        eps,
        block_width=block_width,
    )
    return output


def reference_final_rmsnorm_ordered_pool(
    inputs: Tensor,
    weight: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """Apply final RMSNorm and ordered 1/2/4 temporal mean pooling with PyTorch ops."""
    _validate_inputs(inputs, weight, eps)
    resolved_eps = _resolve_eps(inputs, eps)
    normalized = _rms_norm_accumulated(inputs, weight, resolved_eps)
    empty = normalized.new_zeros(normalized.shape[0], normalized.shape[2])
    summaries: list[Tensor] = []
    for level in _LEVELS:
        summaries.extend(
            chunk.mean(dim=1) if chunk.shape[1] else empty
            for chunk in torch.tensor_split(normalized, level, dim=1)
        )
    return torch.cat(summaries, dim=-1)


def final_rmsnorm_ordered_pool(
    inputs: Tensor,
    weight: Tensor,
    *,
    eps: float | None = None,
    backend: OrderedPoolBackend = "auto",
) -> Tensor:
    """Dispatch fused final RMSNorm + ordered temporal pooling."""
    _validate_inputs(inputs, weight, eps)
    resolved_eps = _resolve_eps(inputs, eps)
    match backend:
        case "reference":
            return reference_final_rmsnorm_ordered_pool(inputs, weight, eps=resolved_eps)
        case "triton":
            _require_triton_inputs(inputs, weight)
            return _triton_final_rmsnorm_ordered_pool(inputs, weight, eps=resolved_eps)
        case "auto":
            triton_dtype = inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
            short_sequence = inputs.shape[1] <= _TRITON_PERSISTENT_STEP_LIMIT
            if inputs.is_cuda and triton_dtype and short_sequence:
                return _triton_final_rmsnorm_ordered_pool(inputs, weight, eps=resolved_eps)
            return reference_final_rmsnorm_ordered_pool(inputs, weight, eps=resolved_eps)
        case unreachable:
            message = f"unknown ordered-pool backend: {unreachable}"
            raise ValueError(message)


def _setup_context(ctx: _AutogradContext, inputs, keyword_only_inputs, output: Tensor) -> None:
    del output
    input_tensor, weight = inputs
    ctx.save_for_backward(input_tensor, weight)
    ctx.eps = float(keyword_only_inputs["eps"])


def _backward(ctx: _AutogradContext, grad_output: Tensor) -> tuple[Tensor, Tensor]:
    input_tensor, weight = ctx.saved_tensors
    with torch.enable_grad():
        reference_input = input_tensor.detach().requires_grad_()
        reference_weight = weight.detach().requires_grad_()
        reference_output = reference_final_rmsnorm_ordered_pool(
            reference_input,
            reference_weight,
            eps=ctx.eps,
        )
    input_gradient, weight_gradient = torch.autograd.grad(
        reference_output,
        (reference_input, reference_weight),
        grad_output,
        allow_unused=False,
    )
    return input_gradient, weight_gradient


torch.library.register_autograd(
    "lnet::final_rmsnorm_ordered_pool",
    _backward,
    setup_context=_setup_context,
)


def _rms_norm_accumulated(inputs: Tensor, weight: Tensor, eps: float) -> Tensor:
    accumulator_dtype = torch.float64 if inputs.dtype == torch.float64 else torch.float32
    values = inputs.to(accumulator_dtype)
    normed = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.to(accumulator_dtype)).to(inputs.dtype)


def _resolve_eps(inputs: Tensor, eps: float | None) -> float:
    if eps is None:
        return torch.finfo(inputs.dtype).eps
    return eps


def _validate_inputs(inputs: Tensor, weight: Tensor, eps: float | None) -> None:
    if inputs.ndim != 3:
        message = "inputs must have shape [batch, time, width]"
        raise ValueError(message)
    if weight.ndim != 1:
        message = "weight must have shape [width]"
        raise ValueError(message)
    if inputs.shape[2] != weight.shape[0]:
        message = "weight width must match input width"
        raise ValueError(message)
    if inputs.device != weight.device:
        message = "inputs and weight must be on the same device"
        raise ValueError(message)
    if inputs.dtype != weight.dtype:
        message = "inputs and weight must have the same dtype"
        raise ValueError(message)
    if not inputs.is_floating_point():
        message = "inputs must use a floating-point dtype"
        raise TypeError(message)
    if inputs.shape[1] == 0 or inputs.shape[2] == 0:
        message = "time and width dimensions must be non-zero"
        raise ValueError(message)
    if eps is not None and eps <= 0.0:
        message = "eps must be positive"
        raise ValueError(message)


def _require_triton_inputs(inputs: Tensor, weight: Tensor) -> None:
    if not inputs.is_cuda or not weight.is_cuda:
        message = "the Triton ordered-pool backend requires CUDA tensors"
        raise ValueError(message)
    if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        message = "the Triton ordered-pool backend supports fp16, bf16, and fp32"
        raise TypeError(message)
