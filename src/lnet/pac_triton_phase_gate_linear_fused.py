"""Materialization-free gate and output projection for narrow Phase-Gated FFNs."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIncompatibleMethodOverride=false, reportMissingParameterType=false
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from . import pac_triton_phase_gate as triton_phase_gate
from .pac_kernel_launch_config import (
    LaunchGeometry,
    autotuned,
    make_launch_scope,
    register_default,
)
from .pac_reduction_tiling import device_parameter_reduction_rows
from .pac_triton_hardware import device_supports_single_warp_dot_tiles

FORWARD_LAUNCH_NAME = "phase_gate_output_linear_fused_forward"
BACKWARD_LAUNCH_NAME = "phase_gate_output_linear_fused_backward"
_FORWARD_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_ROWS": block_rows})
    for block_rows, warps in ((16, 4), (32, 4), (32, 8), (64, 4), (64, 8))
)
_BACKWARD_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_ROWS": block_rows})
    for block_rows, warps in ((64, 4), (64, 8), (128, 4), (128, 8))
)
register_default(
    FORWARD_LAUNCH_NAME,
    LaunchGeometry.build(num_warps=4, blocks={"BLOCK_ROWS": 32}),
    candidates=_FORWARD_CANDIDATES,
)
register_default(
    BACKWARD_LAUNCH_NAME,
    LaunchGeometry.build(num_warps=4, blocks={"BLOCK_ROWS": 64}),
    candidates=_BACKWARD_CANDIDATES,
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    redistribution: float
    self_gated: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def supports_fused_phase_gate_output_linear(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> bool:
    """Return whether one hardware-bounded dot tile can represent the operation."""
    hidden = alpha.numel()
    projected_width = 2 * hidden if self_gated else 4 * hidden
    return (
        projected.is_cuda
        and projected.dtype is torch.bfloat16
        and projected.ndim >= 1
        and projected.shape[-1] == projected_width
        and projected.numel() > 0
        and projected.is_contiguous()
        and alpha.dtype is torch.float32
        and alpha.shape == (hidden,)
        and alpha.is_contiguous()
        and output_weight.device == projected.device
        and output_weight.dtype is torch.bfloat16
        and output_weight.ndim == 2
        and output_weight.shape[1] == 2 * hidden
        and output_weight.is_contiguous()
        and device_supports_single_warp_dot_tiles(
            projected,
            2 * hidden,
            output_weight.shape[0],
        )
        and 0.0 < redistribution < 1.0
        and not torch.are_deterministic_algorithms_enabled()
    )


@triton.jit
def _gate_terms(
    projected,
    alpha,
    row,
    mode,
    mask,
    HIDDEN: tl.constexpr,
    RHO: tl.constexpr,
    SELF_GATED: tl.constexpr,
):
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
    gate_real_fp32 = gate_real.to(tl.float32)
    gate_imag_fp32 = gate_imag.to(tl.float32)
    denominator = 1.0 + gate_real_fp32 * gate_real_fp32 + gate_imag_fp32 * gate_imag_fp32
    magnitude = tl.where(mask, tl.log(denominator), 0.0)
    centered = magnitude - tl.sum(magnitude, axis=1)[:, None] / HIDDEN
    active_alpha = tl.load(alpha + mode, mask=mode < HIDDEN, other=0.0)
    tangent = libdevice.tanh(active_alpha * centered)
    relative = tl.where(mask, 1.0 + RHO * tangent, 0.0)
    active_row = tl.sum(mask.to(tl.int32), axis=1) > 0
    mean_relative = tl.where(active_row, tl.sum(relative, axis=1) / HIDDEN, 1.0)
    gate_fp32 = tl.where(mask, relative / mean_relative[:, None], 0.0)
    gate = gate_fp32.to(tl.bfloat16)
    return (
        base,
        value_real,
        value_imag,
        gate_real_fp32,
        gate_imag_fp32,
        denominator,
        centered,
        active_alpha,
        tangent,
        mean_relative,
        gate_fp32,
        gate,
    )


@triton.jit
def _fused_forward_kernel(
    projected,
    alpha,
    output_weight,
    output,
    rows: int,
    HIDDEN: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    RHO: tl.constexpr,
    SELF_GATED: tl.constexpr,
) -> None:
    row_id = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    mode = tl.arange(0, BLOCK_HIDDEN)[None, :]
    mask = (row < rows) & (mode < HIDDEN)
    (
        _,
        value_real,
        value_imag,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        gate,
    ) = _gate_terms(
        projected,
        alpha,
        row,
        mode,
        mask,
        HIDDEN,
        RHO,
        SELF_GATED,
    )
    hidden_real = (value_real * gate).to(tl.bfloat16)
    hidden_imag = (value_imag * gate).to(tl.bfloat16)
    output_coordinate = tl.arange(0, BLOCK_OUTPUT)
    output_mask = output_coordinate < OUTPUT_WIDTH
    weight_offset = output_coordinate[:, None] * (2 * HIDDEN) + mode
    weight_real = tl.load(
        output_weight + weight_offset,
        mask=output_mask[:, None] & (mode < HIDDEN),
        other=0.0,
    ).to(tl.bfloat16)
    weight_imag = tl.load(
        output_weight + weight_offset + HIDDEN,
        mask=output_mask[:, None] & (mode < HIDDEN),
        other=0.0,
    ).to(tl.bfloat16)
    active_output = tl.dot(hidden_real, tl.trans(weight_real))
    active_output += tl.dot(hidden_imag, tl.trans(weight_imag))
    output_offset = row * OUTPUT_WIDTH + output_coordinate[None, :]
    tl.store(
        output + output_offset,
        active_output.to(tl.bfloat16),
        mask=(row < rows) & output_mask[None, :],
    )


@triton.jit
def _fused_backward_kernel(
    projected,
    alpha,
    output_weight,
    grad_output,
    grad_projected,
    partial_grad_alpha,
    grad_output_weight,
    rows: int,
    HIDDEN: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    PARTIAL_ROWS: tl.constexpr,
    RHO: tl.constexpr,
    SELF_GATED: tl.constexpr,
) -> None:
    row_block = tl.program_id(0)
    row_id = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    mode = tl.arange(0, BLOCK_HIDDEN)[None, :]
    mask = (row < rows) & (mode < HIDDEN)
    (
        base,
        value_real,
        value_imag,
        gate_real,
        gate_imag,
        denominator,
        centered,
        active_alpha,
        tangent,
        mean_relative,
        gate_fp32,
        gate,
    ) = _gate_terms(
        projected,
        alpha,
        row,
        mode,
        mask,
        HIDDEN,
        RHO,
        SELF_GATED,
    )
    hidden_real = (value_real * gate).to(tl.bfloat16)
    hidden_imag = (value_imag * gate).to(tl.bfloat16)

    output_coordinate = tl.arange(0, BLOCK_OUTPUT)
    output_mask = output_coordinate < OUTPUT_WIDTH
    grad_output_offset = row * OUTPUT_WIDTH + output_coordinate[None, :]
    active_grad_output = tl.load(
        grad_output + grad_output_offset,
        mask=(row < rows) & output_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    weight_offset = output_coordinate[:, None] * (2 * HIDDEN) + mode
    weight_real = tl.load(
        output_weight + weight_offset,
        mask=output_mask[:, None] & (mode < HIDDEN),
        other=0.0,
    ).to(tl.bfloat16)
    weight_imag = tl.load(
        output_weight + weight_offset + HIDDEN,
        mask=output_mask[:, None] & (mode < HIDDEN),
        other=0.0,
    ).to(tl.bfloat16)
    grad_value_real = tl.dot(active_grad_output, weight_real).to(tl.bfloat16)
    grad_value_imag = tl.dot(active_grad_output, weight_imag).to(tl.bfloat16)

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
    magnitude_real = grad_magnitude * 2.0 * gate_real / denominator
    magnitude_imag = grad_magnitude * 2.0 * gate_imag / denominator
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
        tl.store(grad_projected + base + HIDDEN, magnitude_real.to(tl.bfloat16), mask=mask)
        tl.store(grad_projected + base + 2 * HIDDEN, direct_imag, mask=mask)
        tl.store(grad_projected + base + 3 * HIDDEN, magnitude_imag.to(tl.bfloat16), mask=mask)

    partial_block = (row_block * BLOCK_ROWS) // PARTIAL_ROWS
    partial_offset = partial_block * HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    partial_alpha = tl.sum(tl.where(mask, grad_logits * centered, 0.0), axis=0)
    if BLOCK_ROWS >= PARTIAL_ROWS:
        tl.store(
            partial_grad_alpha + partial_offset,
            partial_alpha,
            mask=tl.arange(0, BLOCK_HIDDEN) < HIDDEN,
        )
    else:
        tl.atomic_add(
            partial_grad_alpha + partial_offset,
            partial_alpha,
            mask=tl.arange(0, BLOCK_HIDDEN) < HIDDEN,
        )

    grad_weight_real = tl.dot(tl.trans(active_grad_output), hidden_real)
    grad_weight_imag = tl.dot(tl.trans(active_grad_output), hidden_imag)
    tl.atomic_add(
        grad_output_weight + weight_offset,
        grad_weight_real,
        mask=output_mask[:, None] & (mode < HIDDEN),
    )
    tl.atomic_add(
        grad_output_weight + weight_offset + HIDDEN,
        grad_weight_imag,
        mask=output_mask[:, None] & (mode < HIDDEN),
    )


def _scope(kernel: object, projected: Tensor, alpha: Tensor, output_weight: Tensor):
    return make_launch_scope(
        kernel,
        projected,
        shape={
            "rows": projected.numel() // projected.shape[-1],
            "hidden": alpha.numel(),
            "output_width": output_weight.shape[0],
        },
    )


@triton_op("lnet::phase_gate_output_linear_fused", mutates_args={})
def _forward_op(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    redistribution: float,
    self_gated: bool,
) -> Tensor:
    if not supports_fused_phase_gate_output_linear(
        projected,
        alpha,
        output_weight,
        redistribution=redistribution,
        self_gated=self_gated,
    ):
        raise RuntimeError("unsupported fused Phase-Gated output projection contract")
    rows = projected.numel() // projected.shape[-1]
    hidden = alpha.numel()
    output_width = output_weight.shape[0]
    output = torch.empty(
        (*projected.shape[:-1], output_width),
        device=projected.device,
        dtype=projected.dtype,
    )
    kernel = autotuned(
        _fused_forward_kernel,
        FORWARD_LAUNCH_NAME,
        key=("rows", "HIDDEN", "OUTPUT_WIDTH", "RHO", "SELF_GATED"),
        scope=_scope(_fused_forward_kernel, projected, alpha, output_weight),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        projected,
        alpha,
        output_weight,
        output,
        rows,
        HIDDEN=hidden,
        OUTPUT_WIDTH=output_width,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(hidden)),
        BLOCK_OUTPUT=max(16, triton.next_power_of_2(output_width)),
        RHO=redistribution,
        SELF_GATED=self_gated,
    )
    return output


@triton_op("lnet::phase_gate_output_linear_fused_backward", mutates_args={})
def _backward_op(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    grad_output: Tensor,
    redistribution: float,
    self_gated: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    rows = projected.numel() // projected.shape[-1]
    hidden = alpha.numel()
    output_width = output_weight.shape[0]
    grad_projected = torch.empty_like(projected, memory_format=torch.contiguous_format)
    partial_rows = device_parameter_reduction_rows(projected, rows)
    partial_count = int(triton.cdiv(rows, partial_rows))
    partial_grad_alpha = torch.zeros(
        (partial_count, hidden),
        device=projected.device,
        dtype=torch.float32,
    )
    grad_output_weight_fp32 = torch.zeros_like(output_weight, dtype=torch.float32)
    kernel = autotuned(
        _fused_backward_kernel,
        BACKWARD_LAUNCH_NAME,
        key=("rows", "HIDDEN", "OUTPUT_WIDTH", "RHO", "SELF_GATED"),
        scope=_scope(_fused_backward_kernel, projected, alpha, output_weight),
        reset_to_zero=("partial_grad_alpha", "grad_output_weight"),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        projected,
        alpha,
        output_weight,
        grad_output,
        grad_projected,
        partial_grad_alpha,
        grad_output_weight_fp32,
        rows,
        HIDDEN=hidden,
        OUTPUT_WIDTH=output_width,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(hidden)),
        BLOCK_OUTPUT=max(16, triton.next_power_of_2(output_width)),
        PARTIAL_ROWS=partial_rows,
        RHO=redistribution,
        SELF_GATED=self_gated,
    )
    grad_alpha = torch.empty_like(alpha)
    reduce_kernel = autotuned(
        triton_phase_gate._phase_gate_backward_reduce_kernel,  # pyright: ignore[reportPrivateUsage]
        triton_phase_gate.BACKWARD_REDUCE_LAUNCH_NAME,
        key=("partial_count", "hidden"),
        scope=make_launch_scope(
            triton_phase_gate._phase_gate_backward_reduce_kernel,  # pyright: ignore[reportPrivateUsage]
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
    return grad_projected, grad_alpha, grad_output_weight_fp32.to(output_weight.dtype)


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, float, bool],
    output: Tensor,
) -> None:
    del output
    projected, alpha, output_weight, redistribution, self_gated = inputs
    ctx.save_for_backward(projected, alpha, output_weight)
    ctx.redistribution = redistribution
    ctx.self_gated = self_gated


def _backward(
    ctx: _AutogradContext,
    grad_output: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, None, None]:
    projected, alpha, output_weight = ctx.saved_tensors
    if grad_output is None:
        grad_output = torch.zeros(
            (*projected.shape[:-1], output_weight.shape[0]),
            device=projected.device,
            dtype=projected.dtype,
        )
    gradients = _backward_op(
        projected,
        alpha,
        output_weight,
        grad_output.contiguous(),
        ctx.redistribution,
        ctx.self_gated,
    )
    return *gradients, None, None


_forward_op.register_autograd(  # pyright: ignore[reportFunctionMemberAccess]
    _backward,
    setup_context=_setup_context,
)


def fused_phase_gate_output_linear(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> Tensor:
    """Apply the narrow fused kernel after its public support check."""
    # AOTAutograd functionalizes the nested Triton backward as a mutation even
    # though the operator contract is functional.  Keep that synthetic copy
    # away from the leaf parameter while preserving the exact gradient path.
    active_alpha = alpha + 0.0
    return _forward_op(projected, active_alpha, output_weight, redistribution, self_gated)


__all__ = [
    "BACKWARD_LAUNCH_NAME",
    "FORWARD_LAUNCH_NAME",
    "fused_phase_gate_output_linear",
    "supports_fused_phase_gate_output_linear",
]
