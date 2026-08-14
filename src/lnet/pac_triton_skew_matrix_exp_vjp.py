"""Fused FP32 commutator steps for the D64 skew matrix-exp VJP.

The guarded direct VJP writes a third-order ``phi(ad_A)`` series.  PyTorch's
compact formulation uses ``stack -> bmm -> subtract -> scale -> add`` for each
commutator.  This module keeps the same series and strict-FP32 arithmetic, but
fuses both matrix products and the elementwise recurrence into one Triton
kernel per order.

This is intentionally a narrow inference/training primitive: square contiguous
64x64 FP32 CUDA matrices only.  The caller owns the approximation guard and the
CUDA Graph capture lifecycle.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, N803
import math

import torch
import triton
import triton.language as tl
from torch import Tensor

_SIZE = 64
_BLOCK = 16


@triton.jit
def _commutator_accumulate_kernel(
    matrix,
    term,
    result,
    next_term,
    next_result,
    coefficient: tl.constexpr,
    BLOCK: tl.constexpr,
    SIZE: tl.constexpr,
    STORE_TERM: tl.constexpr,
) -> None:
    row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    rows = row_block * BLOCK + tl.arange(0, BLOCK)
    columns = column_block * BLOCK + tl.arange(0, BLOCK)
    accumulator_left = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    accumulator_right = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    for start in range(0, SIZE, BLOCK):
        inner = start + tl.arange(0, BLOCK)
        matrix_rows = tl.load(matrix + rows[:, None] * SIZE + inner[None, :])
        term_columns = tl.load(term + inner[:, None] * SIZE + columns[None, :])
        term_rows = tl.load(term + rows[:, None] * SIZE + inner[None, :])
        matrix_columns = tl.load(matrix + inner[:, None] * SIZE + columns[None, :])
        accumulator_left += tl.dot(matrix_rows, term_columns, input_precision="ieee")
        accumulator_right += tl.dot(term_rows, matrix_columns, input_precision="ieee")

    offsets = rows[:, None] * SIZE + columns[None, :]
    commutator = accumulator_left - accumulator_right
    previous = tl.load(result + offsets)
    if STORE_TERM:
        tl.store(next_term + offsets, commutator)
    tl.store(next_result + offsets, previous + coefficient * commutator)


def fused_direct_skew_vjp(
    matrix: Tensor,
    forward_output: Tensor,
    output_gradient: Tensor,
    *,
    order: int,
) -> Tensor:
    """Evaluate the existing order-0..3 direct VJP with fused commutators."""
    if order < 0 or order > 3:
        message = "fused direct skew VJP supports orders 0 through 3"
        raise ValueError(message)
    for value in (matrix, forward_output, output_gradient):
        if (
            value.device.type != "cuda"
            or value.dtype != torch.float32
            or value.shape != (_SIZE, _SIZE)
            or not value.is_contiguous()
        ):
            message = "fused direct skew VJP expects contiguous CUDA FP32 [64, 64] tensors"
            raise ValueError(message)

    transported = forward_output.mT @ output_gradient
    result = transported
    term = transported
    grid = (_SIZE // _BLOCK, _SIZE // _BLOCK)
    for index in range(1, order + 1):
        next_result = torch.empty_like(result)
        store_term = index != order
        next_term = torch.empty_like(term) if store_term else next_result
        _commutator_accumulate_kernel[grid](
            matrix,
            term,
            result,
            next_term,
            next_result,
            coefficient=1.0 / math.factorial(index + 1),
            BLOCK=_BLOCK,
            SIZE=_SIZE,
            STORE_TERM=store_term,
            num_warps=4,
        )
        term = next_term
        result = next_result
    return result


__all__ = ["fused_direct_skew_vjp"]
