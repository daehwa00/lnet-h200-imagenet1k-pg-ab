"""Single-kernel QR retraction for PAC's very narrow CUDA stem frame.

The canonical identity ALPHABET stem projects a ``D x (2 * raw_dim)`` matrix
back onto the Stiefel manifold after every optimizer step.  For the measured
``D=64, raw_dim=2`` cell, launching a general cuSOLVER QR is substantially
more expensive than the arithmetic itself.  These kernels implement the same
positive-diagonal convention with modified Gram--Schmidt for the only narrow
widths used by the optimized scalar/two-channel surfaces.

This is deliberately an opt-in helper.  Unsupported shapes and dtypes are
rejected so callers can retain ``torch.linalg.qr`` as their exact fallback.
"""

# pyright: reportMissingParameterType=false
# Triton pointer arguments do not have Python type annotations.
# ruff: noqa: ANN001

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _small_qr2_retraction_kernel(
    weight,
    rows: tl.constexpr,
    stride_row: tl.constexpr,
    block_rows: tl.constexpr,
) -> None:
    row = tl.arange(0, block_rows)
    valid = row < rows
    first = tl.load(weight + row * stride_row, mask=valid, other=0.0).to(tl.float32)
    second = tl.load(weight + row * stride_row + 1, mask=valid, other=0.0).to(tl.float32)

    first *= tl.rsqrt(tl.sum(first * first, axis=0))
    second -= tl.sum(first * second, axis=0) * first
    second *= tl.rsqrt(tl.sum(second * second, axis=0))

    tl.store(weight + row * stride_row, first, mask=valid)
    tl.store(weight + row * stride_row + 1, second, mask=valid)


@triton.jit
def _small_qr4_retraction_kernel(
    weight,
    rows: tl.constexpr,
    stride_row: tl.constexpr,
    block_rows: tl.constexpr,
) -> None:
    row = tl.arange(0, block_rows)
    valid = row < rows
    first = tl.load(weight + row * stride_row, mask=valid, other=0.0).to(tl.float32)
    second = tl.load(weight + row * stride_row + 1, mask=valid, other=0.0).to(tl.float32)
    third = tl.load(weight + row * stride_row + 2, mask=valid, other=0.0).to(tl.float32)
    fourth = tl.load(weight + row * stride_row + 3, mask=valid, other=0.0).to(tl.float32)

    first *= tl.rsqrt(tl.sum(first * first, axis=0))
    second -= tl.sum(first * second, axis=0) * first
    second *= tl.rsqrt(tl.sum(second * second, axis=0))
    third -= tl.sum(first * third, axis=0) * first
    third -= tl.sum(second * third, axis=0) * second
    third *= tl.rsqrt(tl.sum(third * third, axis=0))
    fourth -= tl.sum(first * fourth, axis=0) * first
    fourth -= tl.sum(second * fourth, axis=0) * second
    fourth -= tl.sum(third * fourth, axis=0) * third
    fourth *= tl.rsqrt(tl.sum(fourth * fourth, axis=0))

    tl.store(weight + row * stride_row, first, mask=valid)
    tl.store(weight + row * stride_row + 1, second, mask=valid)
    tl.store(weight + row * stride_row + 2, third, mask=valid)
    tl.store(weight + row * stride_row + 3, fourth, mask=valid)


@torch.no_grad()
def small_qr_retraction_(weight: Tensor) -> Tensor:
    """Retract one contiguous FP32 CUDA matrix with two or four columns."""
    if (
        not weight.is_cuda
        or weight.dtype != torch.float32
        or weight.ndim != 2
        or not weight.is_contiguous()
        or weight.shape[1] not in (2, 4)
        or weight.shape[0] < weight.shape[1]
    ):
        message = "small QR retraction requires contiguous CUDA FP32 [D>=K,K] with K in {2,4}"
        raise ValueError(message)
    rows, columns = weight.shape
    block_rows = triton.next_power_of_2(rows)
    if columns == 2:
        torch.library.wrap_triton(_small_qr2_retraction_kernel)[(1,)](
            weight,
            rows=rows,
            stride_row=weight.stride(0),
            block_rows=block_rows,
            num_warps=1,
        )
    else:
        torch.library.wrap_triton(_small_qr4_retraction_kernel)[(1,)](
            weight,
            rows=rows,
            stride_row=weight.stride(0),
            block_rows=block_rows,
            num_warps=1,
        )
    return weight


__all__ = ["small_qr_retraction_"]
