"""Residual-aware narrow Phase-Gated output projection."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIncompatibleMethodOverride=false, reportMissingParameterType=false, reportPrivateUsage=false
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from . import pac_triton_phase_gate as triton_phase_gate
from .pac_kernel_launch_config import (
    LaunchGeometry,
    autotuned,
    make_launch_scope,
    register_default,
)
from .pac_reduction_tiling import device_parameter_reduction_rows
from .pac_triton_hardware import diagnostic_sample_rows
from .pac_triton_phase_gate_linear_fused import (
    _gate_terms,
    supports_fused_phase_gate_output_linear,
)

FORWARD_LAUNCH_NAME = "phase_gate_output_residual_fused_forward"
BACKWARD_LAUNCH_NAME = "phase_gate_output_residual_fused_backward"
BACKWARD_REDUCE_LAUNCH_NAME = "phase_gate_output_residual_fused_backward_reduce"
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
register_default(
    BACKWARD_REDUCE_LAUNCH_NAME,
    LaunchGeometry.build(num_warps=4, blocks={"BLOCK_PARTIALS": 256}),
    candidates=tuple(
        LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_PARTIALS": partials})
        for partials, warps in ((128, 4), (256, 4), (256, 8), (512, 8))
    ),
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    redistribution: float
    self_gated: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...


def _supports_row_layout(tensor: Tensor) -> bool:
    return tensor.is_contiguous() or (tensor.ndim >= 2 and tensor.transpose(-2, -1).is_contiguous())


def _row_layout(tensor: Tensor) -> tuple[int, int, int, int]:
    if tensor.ndim == 1:
        return 1, 0, 0, tensor.stride(-1)
    inner_rows = tensor.shape[-2]
    outer_row_stride = tensor.stride(-3) if tensor.ndim >= 3 else 0
    return inner_rows, outer_row_stride, tensor.stride(-2), tensor.stride(-1)


def supports_fused_phase_gate_output_residual(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    real: Tensor,
    imag: Tensor,
    gamma: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> bool:
    """Return whether the residual-aware narrow kernel owns this workload."""
    modes = real.shape[-1] if real.ndim >= 1 else 0
    return (
        supports_fused_phase_gate_output_linear(
            projected,
            alpha,
            output_weight,
            redistribution=redistribution,
            self_gated=self_gated,
        )
        and real.shape == imag.shape
        and real.shape[:-1] == projected.shape[:-1]
        and real.numel() > 0
        and real.device == projected.device
        and imag.device == real.device
        and real.dtype == imag.dtype
        and real.dtype in (torch.float32, torch.bfloat16)
        and real.stride() == imag.stride()
        and _supports_row_layout(real)
        and output_weight.shape[0] == 2 * modes
        and gamma.device == real.device
        and gamma.dtype is torch.float32
        and gamma.shape == ()
        and gamma.is_contiguous()
    )


@triton.jit
def _source_offset(
    row,
    mode,
    modes: tl.constexpr,
    inner_rows,
    outer_row_stride,
    inner_row_stride,
    mode_stride,
    CONTIGUOUS_SOURCE: tl.constexpr,
):
    if CONTIGUOUS_SOURCE:
        return row * modes + mode
    outer_row = row // inner_rows
    inner_row = row % inner_rows
    return outer_row * outer_row_stride + inner_row * inner_row_stride + mode * mode_stride


@triton.jit
def _phase_gate_output_residual_forward_kernel(
    projected,
    alpha,
    output_weight,
    real,
    imag,
    gamma,
    output_real,
    output_imag,
    diagnostic_delta,
    rows: int,
    diagnostic_rows: int,
    inner_rows: int,
    outer_row_stride: int,
    inner_row_stride: int,
    mode_stride: int,
    MODES: tl.constexpr,
    HIDDEN: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    RHO: tl.constexpr,
    SELF_GATED: tl.constexpr,
    CONTIGUOUS_SOURCE: tl.constexpr,
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
    ) = _gate_terms(projected, alpha, row, mode, mask, HIDDEN, RHO, SELF_GATED)
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
    active_delta = tl.dot(hidden_real, tl.trans(weight_real))
    active_delta += tl.dot(hidden_imag, tl.trans(weight_imag))
    active_delta = active_delta.to(tl.bfloat16)
    packed_offset = row * OUTPUT_WIDTH + output_coordinate[None, :]
    active_mask = (row < rows) & output_mask[None, :]
    tl.store(
        diagnostic_delta + packed_offset,
        active_delta,
        mask=active_mask & (row < diagnostic_rows),
    )

    residual_mode = output_coordinate % MODES
    residual_offset = _source_offset(
        row,
        residual_mode[None, :],
        MODES,
        inner_rows,
        outer_row_stride,
        inner_row_stride,
        mode_stride,
        CONTIGUOUS_SOURCE,
    )
    source_real = tl.load(real + residual_offset, mask=active_mask, other=0.0)
    source_imag = tl.load(imag + residual_offset, mask=active_mask, other=0.0)
    source = tl.where(output_coordinate[None, :] < MODES, source_real, source_imag)
    active_gamma = tl.load(gamma).to(tl.bfloat16)
    output = source + (active_delta * active_gamma).to(tl.bfloat16)
    real_mask = active_mask & (output_coordinate[None, :] < MODES)
    imag_mask = active_mask & (output_coordinate[None, :] >= MODES)
    tl.store(
        output_real + residual_offset,
        output,
        mask=real_mask,
    )
    tl.store(
        output_imag + residual_offset,
        output,
        mask=imag_mask,
    )


@triton.jit
def _phase_gate_output_residual_backward_kernel(
    projected,
    alpha,
    output_weight,
    grad_real,
    grad_imag,
    gamma,
    grad_projected,
    partial_grad_alpha,
    grad_output_weight,
    partial_grad_gamma,
    rows: int,
    inner_rows: int,
    outer_row_stride: int,
    inner_row_stride: int,
    mode_stride: int,
    MODES: tl.constexpr,
    HIDDEN: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    PARTIAL_ROWS: tl.constexpr,
    RHO: tl.constexpr,
    SELF_GATED: tl.constexpr,
    CONTIGUOUS_GRADIENT: tl.constexpr,
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
    ) = _gate_terms(projected, alpha, row, mode, mask, HIDDEN, RHO, SELF_GATED)
    hidden_real = (value_real * gate).to(tl.bfloat16)
    hidden_imag = (value_imag * gate).to(tl.bfloat16)

    output_coordinate = tl.arange(0, BLOCK_OUTPUT)
    output_mask = output_coordinate < OUTPUT_WIDTH
    residual_mode = output_coordinate % MODES
    gradient_offset = _source_offset(
        row,
        residual_mode[None, :],
        MODES,
        inner_rows,
        outer_row_stride,
        inner_row_stride,
        mode_stride,
        CONTIGUOUS_GRADIENT,
    )
    active_output_mask = (row < rows) & output_mask[None, :]
    active_grad_real = tl.load(grad_real + gradient_offset, mask=active_output_mask, other=0.0)
    active_grad_imag = tl.load(grad_imag + gradient_offset, mask=active_output_mask, other=0.0)
    active_residual_grad = tl.where(
        output_coordinate[None, :] < MODES,
        active_grad_real,
        active_grad_imag,
    )
    raw_grad_output = active_residual_grad.to(tl.bfloat16)
    active_gamma = tl.load(gamma).to(tl.bfloat16)
    active_grad_output = (raw_grad_output * active_gamma).to(tl.bfloat16)
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
    unscaled_grad_value_real = tl.dot(raw_grad_output, weight_real).to(tl.bfloat16)
    unscaled_grad_value_imag = tl.dot(raw_grad_output, weight_imag).to(tl.bfloat16)
    grad_value_real = (unscaled_grad_value_real * active_gamma).to(tl.bfloat16)
    grad_value_imag = (unscaled_grad_value_imag * active_gamma).to(tl.bfloat16)

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
        tl.store(
            grad_projected + base + 3 * HIDDEN,
            magnitude_imag.to(tl.bfloat16),
            mask=mask,
        )

    partial_block = (row_block * BLOCK_ROWS) // PARTIAL_ROWS
    partial_alpha_offset = partial_block * HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    partial_alpha = tl.sum(tl.where(mask, grad_logits * centered, 0.0), axis=0)
    if BLOCK_ROWS >= PARTIAL_ROWS:
        tl.store(
            partial_grad_alpha + partial_alpha_offset,
            partial_alpha,
            mask=tl.arange(0, BLOCK_HIDDEN) < HIDDEN,
        )
    else:
        tl.atomic_add(
            partial_grad_alpha + partial_alpha_offset,
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

    gamma_contribution = unscaled_grad_value_real.to(tl.float32) * hidden_real.to(
        tl.float32
    ) + unscaled_grad_value_imag.to(tl.float32) * hidden_imag.to(tl.float32)
    partial_gamma = tl.sum(
        tl.sum(tl.where(mask, gamma_contribution, 0.0), axis=1),
        axis=0,
    )
    if BLOCK_ROWS >= PARTIAL_ROWS:
        tl.store(partial_grad_gamma + partial_block, partial_gamma)
    else:
        tl.atomic_add(partial_grad_gamma + partial_block, partial_gamma)


@triton.jit
def _phase_gate_output_residual_backward_reduce_kernel(
    partial_grad_gamma,
    grad_gamma,
    partial_count: int,
    BLOCK_PARTIALS: tl.constexpr,
) -> None:
    accumulator = tl.zeros((), tl.float32)
    for partial_start in tl.range(
        0,
        partial_count,
        BLOCK_PARTIALS,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        partial = partial_start + tl.arange(0, BLOCK_PARTIALS)
        accumulator += tl.sum(
            tl.load(partial_grad_gamma + partial, mask=partial < partial_count, other=0.0)
        )
    tl.store(grad_gamma, accumulator)


def _forward_scope(kernel: object, projected: Tensor, real: Tensor, hidden: int):
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(real)
    return make_launch_scope(
        kernel,
        projected,
        shape={
            "rows": projected.numel() // projected.shape[-1],
            "modes": real.shape[-1],
            "hidden": hidden,
            "inner_rows": inner_rows,
            "outer_stride": outer_stride,
            "inner_stride": inner_stride,
            "mode_stride": mode_stride,
        },
    )


@triton_op("lnet::phase_gate_output_residual_fused", mutates_args={})
def _forward_op(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    real: Tensor,
    imag: Tensor,
    gamma: Tensor,
    redistribution: float,
    self_gated: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    if not supports_fused_phase_gate_output_residual(
        projected,
        alpha,
        output_weight,
        real,
        imag,
        gamma,
        redistribution=redistribution,
        self_gated=self_gated,
    ):
        raise RuntimeError("unsupported residual-aware Phase-Gated output contract")
    modes = real.shape[-1]
    rows = real.numel() // modes
    hidden = alpha.numel()
    output_width = output_weight.shape[0]
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(real)
    output_real = torch.empty_like(real)
    output_imag = torch.empty_like(imag)
    diagnostic_rows = diagnostic_sample_rows(real)
    diagnostic_delta = torch.empty(
        (diagnostic_rows, output_width),
        device=projected.device,
        dtype=projected.dtype,
    )
    kernel = autotuned(
        _phase_gate_output_residual_forward_kernel,
        FORWARD_LAUNCH_NAME,
        key=(
            "rows",
            "MODES",
            "HIDDEN",
            "OUTPUT_WIDTH",
            "RHO",
            "SELF_GATED",
            "inner_rows",
            "outer_row_stride",
            "inner_row_stride",
            "mode_stride",
        ),
        scope=_forward_scope(_phase_gate_output_residual_forward_kernel, projected, real, hidden),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        projected,
        alpha,
        output_weight,
        real,
        imag,
        gamma,
        output_real,
        output_imag,
        diagnostic_delta,
        rows,
        diagnostic_rows,
        inner_rows,
        outer_stride,
        inner_stride,
        mode_stride,
        MODES=modes,
        HIDDEN=hidden,
        OUTPUT_WIDTH=output_width,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(hidden)),
        BLOCK_OUTPUT=max(16, triton.next_power_of_2(output_width)),
        RHO=redistribution,
        SELF_GATED=self_gated,
        CONTIGUOUS_SOURCE=real.is_contiguous(),
    )
    return output_real, output_imag, diagnostic_delta


@triton_op("lnet::phase_gate_output_residual_fused_backward", mutates_args={})
def _backward_op(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    grad_real: Tensor,
    grad_imag: Tensor,
    gamma: Tensor,
    redistribution: float,
    self_gated: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if (
        grad_real.shape != grad_imag.shape
        or grad_real.stride() != grad_imag.stride()
        or not _supports_row_layout(grad_real)
    ):
        grad_real = grad_real.contiguous()
        grad_imag = grad_imag.contiguous()
    modes = grad_real.shape[-1]
    rows = grad_real.numel() // modes
    hidden = alpha.numel()
    output_width = output_weight.shape[0]
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(grad_real)
    grad_projected = torch.empty_like(projected, memory_format=torch.contiguous_format)
    partial_rows = device_parameter_reduction_rows(grad_real, rows)
    partial_count = int(triton.cdiv(rows, partial_rows))
    partial_grad_alpha = torch.zeros(
        (partial_count, hidden),
        device=projected.device,
        dtype=torch.float32,
    )
    grad_output_weight_fp32 = torch.zeros_like(output_weight, dtype=torch.float32)
    partial_grad_gamma = torch.zeros(partial_count, device=grad_real.device, dtype=torch.float32)
    kernel = autotuned(
        _phase_gate_output_residual_backward_kernel,
        BACKWARD_LAUNCH_NAME,
        key=(
            "rows",
            "MODES",
            "HIDDEN",
            "OUTPUT_WIDTH",
            "RHO",
            "SELF_GATED",
            "inner_rows",
            "outer_row_stride",
            "inner_row_stride",
            "mode_stride",
        ),
        scope=make_launch_scope(
            _phase_gate_output_residual_backward_kernel,
            projected,
            shape={
                "rows": rows,
                "modes": modes,
                "hidden": hidden,
                "output_width": output_width,
                "inner_rows": inner_rows,
                "outer_stride": outer_stride,
                "inner_stride": inner_stride,
                "mode_stride": mode_stride,
            },
        ),
        reset_to_zero=(
            "partial_grad_alpha",
            "grad_output_weight",
            "partial_grad_gamma",
        ),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        projected,
        alpha,
        output_weight,
        grad_real,
        grad_imag,
        gamma,
        grad_projected,
        partial_grad_alpha,
        grad_output_weight_fp32,
        partial_grad_gamma,
        rows,
        inner_rows,
        outer_stride,
        inner_stride,
        mode_stride,
        MODES=modes,
        HIDDEN=hidden,
        OUTPUT_WIDTH=output_width,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(hidden)),
        BLOCK_OUTPUT=max(16, triton.next_power_of_2(output_width)),
        PARTIAL_ROWS=partial_rows,
        RHO=redistribution,
        SELF_GATED=self_gated,
        CONTIGUOUS_GRADIENT=grad_real.is_contiguous(),
    )
    grad_alpha = torch.empty_like(alpha)
    alpha_reduce_kernel = autotuned(
        triton_phase_gate._phase_gate_backward_reduce_kernel,
        triton_phase_gate.BACKWARD_REDUCE_LAUNCH_NAME,
        key=("partial_count", "hidden"),
        scope=make_launch_scope(
            triton_phase_gate._phase_gate_backward_reduce_kernel,
            partial_grad_alpha,
            shape={"partial_count": partial_count, "hidden": hidden},
        ),
    )

    def alpha_reduce_grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(hidden, metadata["BLOCK_HIDDEN"])),)

    wrap_triton(alpha_reduce_kernel)[alpha_reduce_grid](
        partial_grad_alpha,
        grad_alpha,
        partial_count,
        hidden,
    )
    grad_gamma = torch.empty_like(gamma)
    reduce_kernel = autotuned(
        _phase_gate_output_residual_backward_reduce_kernel,
        BACKWARD_REDUCE_LAUNCH_NAME,
        key=("partial_count",),
        scope=make_launch_scope(
            _phase_gate_output_residual_backward_reduce_kernel,
            partial_grad_gamma,
            shape={"partial_count": partial_count},
        ),
    )
    wrap_triton(reduce_kernel)[(1,)](partial_grad_gamma, grad_gamma, partial_count)
    return (
        grad_projected,
        grad_alpha,
        grad_output_weight_fp32.to(output_weight.dtype),
        grad_gamma,
    )


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    projected, alpha, output_weight, real, imag, gamma, redistribution, self_gated = inputs
    _, _, diagnostic_delta = output
    ctx.save_for_backward(projected, alpha, output_weight, real, imag, gamma)
    ctx.mark_non_differentiable(diagnostic_delta)
    ctx.redistribution = redistribution
    ctx.self_gated = self_gated


def _backward(
    ctx: _AutogradContext,
    grad_real: Tensor | None,
    grad_imag: Tensor | None,
    grad_delta: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, None, None]:
    del grad_delta
    projected, alpha, output_weight, real, imag, gamma = ctx.saved_tensors
    if grad_real is None:
        grad_real = torch.zeros_like(real, memory_format=torch.contiguous_format)
    if grad_imag is None:
        grad_imag = torch.zeros_like(imag, memory_format=torch.contiguous_format)
    grad_projected, grad_alpha, grad_output_weight, grad_gamma = _backward_op(
        projected,
        alpha,
        output_weight,
        grad_real,
        grad_imag,
        gamma,
        ctx.redistribution,
        ctx.self_gated,
    )
    return (
        grad_projected,
        grad_alpha,
        grad_output_weight,
        grad_real,
        grad_imag,
        grad_gamma,
        None,
        None,
    )


_forward_op.register_autograd(  # pyright: ignore[reportFunctionMemberAccess]
    _backward,
    setup_context=_setup_context,
)


def fused_phase_gate_output_residual(
    projected: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    real: Tensor,
    imag: Tensor,
    gamma: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project, scale, and add the residual, returning only a diagnostic sample."""
    active_alpha = alpha + 0.0
    return _forward_op(
        projected,
        active_alpha,
        output_weight,
        real,
        imag,
        gamma,
        redistribution,
        self_gated,
    )


__all__ = [
    "BACKWARD_LAUNCH_NAME",
    "BACKWARD_REDUCE_LAUNCH_NAME",
    "FORWARD_LAUNCH_NAME",
    "fused_phase_gate_output_residual",
    "supports_fused_phase_gate_output_residual",
]
