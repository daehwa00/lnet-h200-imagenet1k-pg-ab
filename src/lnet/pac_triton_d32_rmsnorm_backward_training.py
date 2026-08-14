from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, ANN202, N803, PLR0915, SLF001
import os
from typing import Literal, Protocol, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional

D32RMSNormStrategy = Literal["native_tree", "tile32"]

_WIDTH = 32
_ROWS_PER_TILE = 32
_NATIVE_ROWS_PER_GROUP = 8
_NATIVE_ROWS_PER_BLOCK = 256
_NATIVE_TWO_PASS_THRESHOLD = 64 * 1024
_DEFAULT_STRATEGY: D32RMSNormStrategy = "native_tree"


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    strategy: int

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _materialize_fp32(value):
    """Prevent LLVM reassociation across a native FP32 reduction boundary."""
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=r,r",
        [value],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _d32_native_projection(
    inputs,
    weight,
    grad_output,
    inverse_rms,
    row,
    valid_row,
):
    """Mirror native vectorized grad-input's four-values-per-lane sum."""
    group = tl.arange(0, 8)
    first_feature = group * 4
    row_base = row * 32

    x0 = tl.load(inputs + row_base + first_feature, mask=valid_row, other=0.0).to(tl.float32)
    x1 = tl.load(inputs + row_base + first_feature + 1, mask=valid_row, other=0.0).to(
        tl.float32
    )
    x2 = tl.load(inputs + row_base + first_feature + 2, mask=valid_row, other=0.0).to(
        tl.float32
    )
    x3 = tl.load(inputs + row_base + first_feature + 3, mask=valid_row, other=0.0).to(
        tl.float32
    )
    g0 = tl.load(grad_output + row_base + first_feature, mask=valid_row, other=0.0).to(
        tl.float32
    )
    g1 = tl.load(grad_output + row_base + first_feature + 1, mask=valid_row, other=0.0).to(
        tl.float32
    )
    g2 = tl.load(grad_output + row_base + first_feature + 2, mask=valid_row, other=0.0).to(
        tl.float32
    )
    g3 = tl.load(grad_output + row_base + first_feature + 3, mask=valid_row, other=0.0).to(
        tl.float32
    )
    w0 = tl.load(weight + first_feature).to(tl.float32)
    w1 = tl.load(weight + first_feature + 1).to(tl.float32)
    w2 = tl.load(weight + first_feature + 2).to(tl.float32)
    w3 = tl.load(weight + first_feature + 3).to(tl.float32)

    t0 = _materialize_fp32(_materialize_fp32(g0 * w0) * x0)
    t0 = _materialize_fp32(t0 * inverse_rms)
    t1 = _materialize_fp32(_materialize_fp32(g1 * w1) * x1)
    t1 = _materialize_fp32(t1 * inverse_rms)
    t2 = _materialize_fp32(_materialize_fp32(g2 * w2) * x2)
    t2 = _materialize_fp32(t2 * inverse_rms)
    t3 = _materialize_fp32(_materialize_fp32(g3 * w3) * x3)
    t3 = _materialize_fp32(t3 * inverse_rms)
    group_sum = _materialize_fp32(t0 + t1)
    group_sum = _materialize_fp32(group_sum + t2)
    group_sum = _materialize_fp32(group_sum + t3)
    return tl.sum(group_sum, axis=0)


@triton.jit
def _d32_tile_backward_kernel(
    inputs,
    weight,
    inverse_rms,
    grad_output,
    grad_inputs,
    partial_grad_weight,
    outer_size: int,
    ROWS_PER_TILE: tl.constexpr,
) -> None:
    """Fuse grad-input with native two-pass-style 32-row gamma partials."""
    tile = tl.program_id(0)
    feature = tl.arange(0, 32)
    feature_weight = tl.load(weight + feature).to(tl.float32)
    gamma_sum = tl.zeros((32,), tl.float32)
    row_in_tile = 0
    while row_in_tile < ROWS_PER_TILE:
        row = tile * ROWS_PER_TILE + row_in_tile
        valid_row = row < outer_size
        inverse = tl.load(inverse_rms + row, mask=valid_row, other=0.0).to(tl.float32)
        projection = _d32_native_projection(
            inputs,
            weight,
            grad_output,
            inverse,
            row,
            valid_row,
        )
        offset = row * 32 + feature
        values = tl.load(inputs + offset, mask=valid_row, other=0.0).to(tl.float32)
        output_gradient = tl.load(
            grad_output + offset, mask=valid_row, other=0.0
        ).to(tl.float32)
        weighted_gradient = _materialize_fp32(feature_weight * output_gradient)
        leading = _materialize_fp32(32.0 * weighted_gradient)
        correction = _materialize_fp32(values * inverse)
        correction = _materialize_fp32(correction * projection)
        numerator = _materialize_fp32(leading - correction)
        scale = _materialize_fp32((1.0 / 32.0) * inverse)
        input_gradient = _materialize_fp32(numerator * scale)
        tl.store(grad_inputs + offset, input_gradient, mask=valid_row)

        gamma_term = _materialize_fp32(output_gradient * values)
        gamma_term = _materialize_fp32(gamma_term * inverse)
        gamma_sum = _materialize_fp32(gamma_sum + tl.where(valid_row, gamma_term, 0.0))
        row_in_tile += 1
    tl.store(partial_grad_weight + tile * 32 + feature, gamma_sum)


@triton.jit
def _d32_native_tree_backward_kernel(
    inputs,
    weight,
    inverse_rms,
    grad_output,
    grad_inputs,
    grad_weight,
    outer_size: int,
    BLOCK_ROWS: tl.constexpr,
) -> None:
    """Mirror native D32 GammaBeta's 32x8 row grouping and reduction tree."""
    feature = tl.arange(0, 32)[None, :]
    row_group = tl.arange(0, 32)[:, None]
    feature_weight = tl.load(weight + feature).to(tl.float32)
    gamma_partial = tl.zeros((32, 32), tl.float32)
    row_block = 0
    while row_block < outer_size:
        row_in_group = 0
        while row_in_group < 8:
            # CUDA's GammaBeta kernel assigns y-lane ``g`` the rows
            # g, g + 32, ..., g + 7*32 inside each 256-row block.
            row = row_block + row_group + row_in_group * 32
            valid_row = row < outer_size
            inverse = tl.load(inverse_rms + row, mask=valid_row, other=0.0).to(tl.float32)

            # Native CUDA assigns four adjacent D32 features to each active
            # grad-input reduction lane before its block-wide sum.  Express
            # the same four-value left fold for every row group.
            first_feature = tl.arange(0, 8)[None, :] * 4
            projection_offset = row * 32 + first_feature
            x0 = tl.load(inputs + projection_offset, mask=valid_row, other=0.0).to(tl.float32)
            x1 = tl.load(inputs + projection_offset + 1, mask=valid_row, other=0.0).to(
                tl.float32
            )
            x2 = tl.load(inputs + projection_offset + 2, mask=valid_row, other=0.0).to(
                tl.float32
            )
            x3 = tl.load(inputs + projection_offset + 3, mask=valid_row, other=0.0).to(
                tl.float32
            )
            g0 = tl.load(
                grad_output + projection_offset, mask=valid_row, other=0.0
            ).to(tl.float32)
            g1 = tl.load(
                grad_output + projection_offset + 1, mask=valid_row, other=0.0
            ).to(tl.float32)
            g2 = tl.load(
                grad_output + projection_offset + 2, mask=valid_row, other=0.0
            ).to(tl.float32)
            g3 = tl.load(
                grad_output + projection_offset + 3, mask=valid_row, other=0.0
            ).to(tl.float32)
            w0 = tl.load(weight + first_feature).to(tl.float32)
            w1 = tl.load(weight + first_feature + 1).to(tl.float32)
            w2 = tl.load(weight + first_feature + 2).to(tl.float32)
            w3 = tl.load(weight + first_feature + 3).to(tl.float32)
            t0 = _materialize_fp32(_materialize_fp32(g0 * w0) * x0)
            t0 = _materialize_fp32(t0 * inverse)
            t1 = _materialize_fp32(_materialize_fp32(g1 * w1) * x1)
            t1 = _materialize_fp32(t1 * inverse)
            t2 = _materialize_fp32(_materialize_fp32(g2 * w2) * x2)
            t2 = _materialize_fp32(t2 * inverse)
            t3 = _materialize_fp32(_materialize_fp32(g3 * w3) * x3)
            t3 = _materialize_fp32(t3 * inverse)
            group_sum = _materialize_fp32(t0 + t1)
            group_sum = _materialize_fp32(group_sum + t2)
            group_sum = _materialize_fp32(group_sum + t3)
            projection = tl.sum(group_sum, axis=1)[:, None]

            offset = row * 32 + feature
            values = tl.load(inputs + offset, mask=valid_row, other=0.0).to(tl.float32)
            output_gradient = tl.load(
                grad_output + offset, mask=valid_row, other=0.0
            ).to(tl.float32)
            weighted_gradient = _materialize_fp32(feature_weight * output_gradient)
            leading = _materialize_fp32(32.0 * weighted_gradient)
            correction = _materialize_fp32(values * inverse)
            correction = _materialize_fp32(correction * projection)
            numerator = _materialize_fp32(leading - correction)
            scale = _materialize_fp32((1.0 / 32.0) * inverse)
            input_gradient = _materialize_fp32(numerator * scale)
            tl.store(grad_inputs + offset, input_gradient, mask=valid_row)

            gamma_term = _materialize_fp32(output_gradient * values)
            gamma_term = _materialize_fp32(gamma_term * inverse)
            gamma_partial = _materialize_fp32(
                gamma_partial + tl.where(valid_row, gamma_term, 0.0)
            )
            row_in_group += 1
        row_block += BLOCK_ROWS
    tl.store(grad_weight + feature, tl.sum(gamma_partial, axis=0))


class _D32RMSNormBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _AutogradContext,
        inputs: Tensor,
        weight: Tensor,
        eps: float,
        strategy: int,
    ) -> Tensor:
        normalized, inverse_rms = torch.ops.aten._fused_rms_norm.default(
            inputs,
            [_WIDTH],
            weight,
            eps,
        )
        ctx.save_for_backward(inputs, weight, inverse_rms)
        ctx.strategy = strategy
        return normalized

    @staticmethod
    def backward(
        ctx: _AutogradContext, *grad_outputs: Tensor
    ) -> tuple[Tensor, Tensor, None, None]:
        (grad_output,) = grad_outputs
        inputs, weight, inverse_rms = ctx.saved_tensors
        grad_inputs, grad_weight = _d32_backward(
            inputs,
            weight,
            inverse_rms,
            grad_output,
            native_tree=ctx.strategy == 0,
        )
        return grad_inputs, grad_weight, None, None


def _d32_backward(
    inputs: Tensor,
    weight: Tensor,
    inverse_rms: Tensor,
    grad_output: Tensor,
    *,
    native_tree: bool,
) -> tuple[Tensor, Tensor]:
    outer_size = inputs.numel() // _WIDTH
    grad_inputs = torch.empty_like(inputs)
    active_grad_output = grad_output.contiguous()
    active_inverse_rms = inverse_rms.contiguous().view(-1)
    if native_tree and outer_size <= _NATIVE_TWO_PASS_THRESHOLD:
        grad_weight = torch.empty_like(weight)
        _d32_native_tree_backward_kernel[(1,)](
            inputs,
            weight,
            active_inverse_rms,
            active_grad_output,
            grad_inputs,
            grad_weight,
            outer_size,
            BLOCK_ROWS=_NATIVE_ROWS_PER_BLOCK,
            num_warps=8,
        )
        return grad_inputs, grad_weight

    tile_count = triton.cdiv(outer_size, _ROWS_PER_TILE)
    partial_grad_weight = torch.empty(
        (tile_count, _WIDTH),
        dtype=weight.dtype,
        device=weight.device,
    )
    _d32_tile_backward_kernel[(tile_count,)](
        inputs,
        weight,
        active_inverse_rms,
        active_grad_output,
        grad_inputs,
        partial_grad_weight,
        outer_size,
        ROWS_PER_TILE=_ROWS_PER_TILE,
        num_warps=1,
    )
    # This intentionally reuses ATen's sum kernel.  On the N2048/B64 shape it
    # consumes the same ordered [ceil(M/32), 32] partial layout as native
    # GammaBeta's M>>N two-pass path.
    return grad_inputs, partial_grad_weight.sum(dim=0)


def supports_d32_rmsnorm_backward_training(inputs: Tensor, weight: Tensor) -> bool:
    return (
        inputs.ndim >= 2
        and inputs.numel() > 0
        and inputs.shape[-1] == _WIDTH
        and inputs.dtype == torch.float32
        and weight.shape == (_WIDTH,)
        and weight.dtype == torch.float32
        and inputs.device == weight.device
        and inputs.is_cuda
        and inputs.is_contiguous()
        and weight.is_contiguous()
    )


def d32_rmsnorm_backward_training(
    inputs: Tensor,
    weight: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """Keep native RMSNorm forward and replace only canonical D32 backward."""
    resolved_eps = torch.finfo(inputs.dtype).eps if eps is None else eps
    if resolved_eps <= 0.0:
        message = "RMSNorm epsilon must be positive"
        raise ValueError(message)
    if not supports_d32_rmsnorm_backward_training(inputs, weight):
        return functional.rms_norm(inputs, (inputs.shape[-1],), weight, resolved_eps)
    strategy = _strategy()
    output = _D32RMSNormBackward.apply(
        inputs,
        weight,
        float(resolved_eps),
        0 if strategy == "native_tree" else 1,
    )
    if not isinstance(output, Tensor):
        message = "D32 RMSNorm backward prototype returned an invalid output"
        raise TypeError(message)
    return output


def _strategy() -> D32RMSNormStrategy:
    value = os.environ.get("LNET_PAC_D32_RMSNORM_BACKWARD", _DEFAULT_STRATEGY)
    if value not in {"native_tree", "tile32"}:
        message = "LNET_PAC_D32_RMSNORM_BACKWARD must be native_tree or tile32"
        raise ValueError(message)
    return cast("D32RMSNormStrategy", value)


__all__ = [
    "D32RMSNormStrategy",
    "d32_rmsnorm_backward_training",
    "supports_d32_rmsnorm_backward_training",
]
