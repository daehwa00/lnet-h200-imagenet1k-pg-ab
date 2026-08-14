"""Phase-Gated output projection with an exact relative-gate backward."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportIncompatibleMethodOverride=false, reportMissingParameterType=false
# ruff: noqa: ANN001, EM101, N803, PLR0915, TRY003
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional
from triton.language.extra import libdevice

from . import pac_triton_phase_gate as triton_phase_gate
from .pac_kernel_launch_config import (
    LaunchGeometry,
    autotuned,
    make_launch_scope,
    register_default,
)
from .pac_reduction_tiling import device_parameter_reduction_rows
from .pac_triton_phase_gate import phase_gate_reference
from .pac_triton_phase_gate_linear_fused import (
    fused_phase_gate_output_linear,
    supports_fused_phase_gate_output_linear,
)

BACKWARD_LAUNCH_NAME = "phase_gate_output_linear_backward"
_BACKWARD_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(
        num_warps=warps,
        blocks={"BLOCK_ROWS": block_rows},
    )
    for block_rows in (4, 8, 16, 32, 64)
    for warps in (4, 8)
)
_BACKWARD_DEFAULT_LAUNCH = LaunchGeometry.build(
    num_warps=4,
    blocks={"BLOCK_ROWS": 4},
)
register_default(
    BACKWARD_LAUNCH_NAME,
    _BACKWARD_DEFAULT_LAUNCH,
    candidates=_BACKWARD_LAUNCH_CANDIDATES,
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    redistribution: float
    self_gated: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _validate(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> None:
    if projected.ndim < 1 or alpha.ndim != 1 or alpha.numel() <= 0:
        raise ValueError("fused Phase-Gated projection requires positive dimensions")
    hidden = alpha.numel()
    projected_width = 2 * hidden if self_gated else 4 * hidden
    if projected.shape[-1] != projected_width:
        raise ValueError("fused Phase-Gated projection dimensions are incompatible")
    if (
        output_weight.ndim != 2
        or output_weight.shape[1] != 2 * hidden
        or output_weight.shape[0] <= 0
    ):
        raise ValueError("packed Phase-Gated output weight has incompatible dimensions")
    if alpha.device != projected.device or output_weight.device != projected.device:
        raise ValueError("fused Phase-Gated projection tensors must share one device")
    if not 0.0 < redistribution < 1.0:
        raise ValueError("Phase-Gated redistribution must be strictly between zero and one")


def supports_phase_gate_output_linear(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> bool:
    """Return whether the fused BF16 CUDA contract is available."""
    try:
        _validate(
            projected,
            alpha,
            output_weight,
            redistribution=redistribution,
            self_gated=self_gated,
        )
    except ValueError:
        return False
    return (
        projected.is_cuda
        and projected.dtype is torch.bfloat16
        and projected.numel() > 0
        and projected.is_contiguous()
        and output_weight.dtype is torch.bfloat16
        and output_weight.is_contiguous()
        and alpha.dtype is torch.float32
        and alpha.is_contiguous()
    )


@triton.jit
def _phase_gate_output_linear_backward_kernel(
    projected,
    alpha,
    grad_hidden,
    grad_projected,
    partial_grad_alpha,
    rows: int,
    HIDDEN: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    PARTIAL_ROWS: tl.constexpr,
    RHO: tl.constexpr,
    SELF_GATED: tl.constexpr,
) -> None:
    row_block = tl.program_id(0)
    row_id = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    active_hidden = tl.arange(0, BLOCK_HIDDEN)
    mode = active_hidden[None, :]
    mask = (row < rows) & (mode < HIDDEN)

    projected_width = 2 * HIDDEN if SELF_GATED else 4 * HIDDEN
    base = row * projected_width + mode
    value_real = tl.load(projected + base, mask=mask, other=0.0).to(tl.bfloat16)
    value_imag_offset = HIDDEN if SELF_GATED else 2 * HIDDEN
    value_imag = tl.load(
        projected + base + value_imag_offset,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    gate_real_offset = 0 if SELF_GATED else HIDDEN
    gate_imag_offset = HIDDEN if SELF_GATED else 3 * HIDDEN
    gate_real = tl.load(
        projected + base + gate_real_offset,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    gate_imag = tl.load(
        projected + base + gate_imag_offset,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)

    hidden_base = row * (2 * HIDDEN) + mode
    grad_value_real = tl.load(grad_hidden + hidden_base, mask=mask, other=0.0).to(tl.bfloat16)
    grad_value_imag = tl.load(
        grad_hidden + hidden_base + HIDDEN,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)

    gate_real_fp32 = gate_real.to(tl.float32)
    gate_imag_fp32 = gate_imag.to(tl.float32)
    denominator = 1.0 + gate_real_fp32 * gate_real_fp32 + gate_imag_fp32 * gate_imag_fp32
    magnitude = tl.where(mask, tl.log(denominator), 0.0)
    centered = magnitude - tl.sum(magnitude, axis=1)[:, None] / HIDDEN
    active_alpha = tl.load(alpha + mode, mask=mode < HIDDEN, other=0.0)
    tangent = libdevice.tanh(active_alpha * centered)
    relative = tl.where(mask, 1.0 + RHO * tangent, 0.0)
    mean_relative = tl.sum(relative, axis=1) / HIDDEN
    gate_fp32 = relative / mean_relative[:, None]
    gate = gate_fp32.to(tl.bfloat16)

    direct_real = (grad_value_real * gate).to(tl.bfloat16)
    direct_imag = (grad_value_imag * gate).to(tl.bfloat16)
    grad_gate = (
        (
            (grad_value_real * value_real).to(tl.bfloat16)
            + (grad_value_imag * value_imag).to(tl.bfloat16)
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    weighted_mean = tl.sum(tl.where(mask, grad_gate * gate_fp32, 0.0), axis=1) / HIDDEN
    grad_relative = (grad_gate - weighted_mean[:, None]) / mean_relative[:, None]
    grad_logits = grad_relative * RHO * (1.0 - tangent * tangent)
    grad_centered = grad_logits * active_alpha
    grad_magnitude = grad_centered - (
        tl.sum(tl.where(mask, grad_centered, 0.0), axis=1)[:, None] / HIDDEN
    )
    magnitude_real = grad_magnitude * 2.0 * gate_real_fp32 / denominator
    magnitude_imag = grad_magnitude * 2.0 * gate_imag_fp32 / denominator

    if SELF_GATED:
        tl.store(
            grad_projected + base,
            (direct_real + magnitude_real.to(tl.bfloat16)).to(tl.bfloat16),
            mask=mask,
        )
        tl.store(
            grad_projected + base + HIDDEN,
            (direct_imag + magnitude_imag.to(tl.bfloat16)).to(tl.bfloat16),
            mask=mask,
        )
    else:
        tl.store(grad_projected + base, direct_real, mask=mask)
        tl.store(
            grad_projected + base + HIDDEN,
            magnitude_real.to(tl.bfloat16),
            mask=mask,
        )
        tl.store(grad_projected + base + 2 * HIDDEN, direct_imag, mask=mask)
        tl.store(
            grad_projected + base + 3 * HIDDEN,
            magnitude_imag.to(tl.bfloat16),
            mask=mask,
        )

    partial_block = (row_block * BLOCK_ROWS) // PARTIAL_ROWS
    partial_offset = partial_block * HIDDEN + active_hidden
    partial = tl.sum(tl.where(mask, grad_logits * centered, 0.0), axis=0)
    if BLOCK_ROWS >= PARTIAL_ROWS:
        tl.store(
            partial_grad_alpha + partial_offset,
            partial,
            mask=active_hidden < HIDDEN,
        )
    else:
        tl.atomic_add(
            partial_grad_alpha + partial_offset,
            partial,
            mask=active_hidden < HIDDEN,
        )


@triton_op("lnet::phase_gate_v2_backward", mutates_args={})
def _backward_op(
    projected: Tensor,
    alpha: Tensor,
    grad_hidden: Tensor,
    redistribution: float,
    self_gated: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor]:
    hidden = alpha.numel()
    rows = projected.numel() // projected.shape[-1]
    if grad_hidden.shape != (rows, 2 * hidden):
        raise ValueError("Phase-Gated hidden gradient has incompatible dimensions")
    grad_projected = torch.empty_like(projected, memory_format=torch.contiguous_format)
    partial_rows = device_parameter_reduction_rows(projected, rows)
    partial_count = int(triton.cdiv(rows, partial_rows))
    partial_grad_alpha = torch.zeros(
        (partial_count, hidden),
        device=projected.device,
        dtype=torch.float32,
    )
    grad_alpha = torch.empty_like(alpha, memory_format=torch.contiguous_format)
    backward_kernel = autotuned(
        _phase_gate_output_linear_backward_kernel,
        BACKWARD_LAUNCH_NAME,
        key=("rows", "HIDDEN", "RHO", "SELF_GATED"),
        scope=make_launch_scope(
            _phase_gate_output_linear_backward_kernel,
            projected,
            shape={
                "rows": rows,
                "hidden": hidden,
                "self_gated": int(self_gated),
            },
        ),
        reset_to_zero=("partial_grad_alpha",),
    )

    def backward_grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(backward_kernel)[backward_grid](
        projected,
        alpha,
        grad_hidden,
        grad_projected,
        partial_grad_alpha,
        rows,
        HIDDEN=hidden,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(hidden)),
        PARTIAL_ROWS=partial_rows,
        RHO=redistribution,
        SELF_GATED=self_gated,
    )

    reduce_kernel = autotuned(
        triton_phase_gate._phase_gate_backward_reduce_kernel,  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        triton_phase_gate.BACKWARD_REDUCE_LAUNCH_NAME,
        key=("partial_count", "hidden"),
        scope=make_launch_scope(
            triton_phase_gate._phase_gate_backward_reduce_kernel,  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            partial_grad_alpha,
            shape={"partial_count": partial_count, "hidden": hidden},
        ),
    )

    def reduce_grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(hidden, metadata["BLOCK_HIDDEN"])),)

    wrap_triton(reduce_kernel)[reduce_grid](
        partial_grad_alpha,
        grad_alpha,
        partial_count,
        hidden,
    )
    return grad_projected, grad_alpha


class _PhaseGateOutputLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _AutogradContext,
        projected: Tensor,
        alpha: Tensor,
        output_weight: Tensor,
        redistribution: float,
        self_gated: bool,  # noqa: FBT001
    ) -> Tensor:
        hidden_output = triton_phase_gate._forward_op(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            projected,
            alpha,
            redistribution,
            self_gated,
        )
        output = functional.linear(hidden_output, output_weight)
        ctx.save_for_backward(projected, alpha, output_weight)
        ctx.redistribution = redistribution
        ctx.self_gated = self_gated
        return output

    @staticmethod
    def backward(
        ctx: _AutogradContext,
        grad_output: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, None, None]:
        projected, alpha, output_weight = ctx.saved_tensors
        rows = projected.numel() // projected.shape[-1]
        output_width = output_weight.shape[0]
        if grad_output is None:
            grad_output = torch.zeros(
                (*projected.shape[:-1], output_width),
                device=projected.device,
                dtype=projected.dtype,
            )
        flat_grad_output = grad_output.contiguous().reshape(rows, output_width)
        hidden_output = triton_phase_gate._forward_op(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            projected,
            alpha,
            ctx.redistribution,
            ctx.self_gated,
        )
        flat_hidden = hidden_output.reshape(rows, output_weight.shape[1])
        grad_output_weight = torch.mm(flat_grad_output.T, flat_hidden)
        grad_hidden = torch.mm(flat_grad_output, output_weight)
        grad_projected, grad_alpha = _backward_op(
            projected,
            alpha,
            grad_hidden,
            ctx.redistribution,
            ctx.self_gated,
        )
        return grad_projected, grad_alpha, grad_output_weight, None, None


def phase_gate_output_linear(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> Tensor:
    """Apply relative gating and the output projection with an exact backward."""
    if not supports_phase_gate_output_linear(
        projected,
        alpha,
        output_weight,
        redistribution=redistribution,
        self_gated=self_gated,
    ):
        if projected.is_cuda:
            raise RuntimeError(
                "fused Phase-Gated output projection requires contiguous BF16 tensors"
            )
        hidden = phase_gate_reference(
            projected,
            alpha,
            redistribution=redistribution,
            self_gated=self_gated,
        )
        return functional.linear(hidden, output_weight)
    if supports_fused_phase_gate_output_linear(
        projected,
        alpha,
        output_weight,
        redistribution=redistribution,
        self_gated=self_gated,
    ):
        return fused_phase_gate_output_linear(
            projected,
            alpha,
            output_weight,
            redistribution=redistribution,
            self_gated=self_gated,
        )
    output = _PhaseGateOutputLinear.apply(
        projected,
        alpha,
        output_weight,
        redistribution,
        self_gated,
    )
    if not isinstance(output, Tensor):
        raise TypeError("Phase-Gated output projection must return a tensor")
    return output


__all__ = [
    "BACKWARD_LAUNCH_NAME",
    "phase_gate_output_linear",
    "supports_phase_gate_output_linear",
]
