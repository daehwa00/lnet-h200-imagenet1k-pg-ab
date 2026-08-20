"""Fused relative-energy gate for packed Phase-Gated complex FFNs."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from .pac_kernel_launch_config import (
    LaunchGeometry,
    LaunchScope,
    autotuned,
    make_launch_scope,
    register_default,
)

FORWARD_LAUNCH_NAME = "phase_gate_forward"
BACKWARD_REDUCE_LAUNCH_NAME = "phase_gate_backward_reduce"
_SOURCE_DTYPE = torch.bfloat16
_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_ROWS": block_rows})
    for block_rows, warps in ((1, 4), (2, 4), (2, 8), (4, 4), (4, 8), (8, 8))
)
_DEFAULT_LAUNCH = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_ROWS": 2})
register_default(FORWARD_LAUNCH_NAME, _DEFAULT_LAUNCH, candidates=_LAUNCH_CANDIDATES)

_REDUCE_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(
        num_warps=warps,
        blocks={"BLOCK_HIDDEN": block_hidden, "BLOCK_PARTIALS": block_partials},
    )
    for block_hidden, block_partials, warps in (
        (8, 128, 4),
        (16, 128, 4),
        (16, 256, 8),
        (32, 128, 8),
    )
)
_REDUCE_DEFAULT_LAUNCH = LaunchGeometry.build(
    num_warps=4,
    blocks={"BLOCK_HIDDEN": 16, "BLOCK_PARTIALS": 128},
)
register_default(
    BACKWARD_REDUCE_LAUNCH_NAME,
    _REDUCE_DEFAULT_LAUNCH,
    candidates=_REDUCE_LAUNCH_CANDIDATES,
)


def phase_gate_reference(
    projected: Tensor,
    alpha: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> Tensor:
    """Redistribute each row's gate mass without changing its mean gain."""
    hidden = alpha.numel()
    expected = 2 * hidden if self_gated else 4 * hidden
    if projected.shape[-1] != expected:
        message = "packed phase gate dimensions are incompatible"
        raise ValueError(message)
    if not 0.0 < redistribution < 1.0:
        message = "phase gate redistribution must be strictly between zero and one"
        raise ValueError(message)
    if self_gated:
        value_real, value_imag = projected.split(hidden, dim=-1)
        gate_real, gate_imag = value_real, value_imag
    else:
        value_real, gate_real, value_imag, gate_imag = projected.split(hidden, dim=-1)
    magnitude = torch.log1p(gate_real.float().square() + gate_imag.float().square())
    centered = magnitude - magnitude.mean(dim=-1, keepdim=True)
    relative = 1.0 + redistribution * torch.tanh(alpha.float() * centered)
    gate = (relative / relative.mean(dim=-1, keepdim=True)).to(dtype=projected.dtype)
    return torch.cat((value_real * gate, value_imag * gate), dim=-1)


def _validate(
    projected: Tensor,
    alpha: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> None:
    if projected.ndim < 1 or alpha.ndim != 1 or alpha.numel() <= 0:
        raise ValueError("phase gate requires positive packed dimensions")
    hidden = alpha.numel()
    expected = 2 * hidden if self_gated else 4 * hidden
    if projected.shape[-1] != expected:
        raise ValueError("packed phase gate dimensions are incompatible")
    if alpha.device != projected.device:
        raise ValueError("phase gate tensors must share one device")
    if not 0.0 < redistribution < 1.0:
        raise ValueError("phase gate redistribution must be strictly between zero and one")


def supports_phase_gate(
    projected: Tensor,
    alpha: Tensor,
    *,
    redistribution: float,
    self_gated: bool,
) -> bool:
    try:
        _validate(
            projected,
            alpha,
            redistribution=redistribution,
            self_gated=self_gated,
        )
    except ValueError:
        return False
    return (
        projected.is_cuda
        and projected.dtype is _SOURCE_DTYPE
        and projected.numel() > 0
        and projected.is_contiguous()
        and alpha.dtype is torch.float32
        and alpha.is_contiguous()
    )


def _scope(kernel: object, projected: Tensor, hidden: int, *, self_gated: bool) -> LaunchScope:
    rows = projected.numel() // projected.shape[-1]
    return make_launch_scope(
        kernel,
        projected,
        shape={"rows": rows, "hidden": hidden, "self_gated": int(self_gated)},
    )


@triton.jit
def _phase_gate_forward_kernel(
    projected,
    alpha,
    output,
    rows: int,
    hidden: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    RHO: tl.constexpr,
    SELF_GATED: tl.constexpr,
) -> None:
    row_id = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    mode = tl.arange(0, BLOCK_HIDDEN)[None, :]
    mask = (row < rows) & (mode < hidden)
    projected_width = 2 * hidden if SELF_GATED else 4 * hidden
    base = row * projected_width + mode
    value_real = tl.load(projected + base, mask=mask, other=0.0)
    value_imag_offset = hidden if SELF_GATED else 2 * hidden
    value_imag = tl.load(projected + base + value_imag_offset, mask=mask, other=0.0)
    gate_real_offset = 0 if SELF_GATED else hidden
    gate_imag_offset = hidden if SELF_GATED else 3 * hidden
    gate_real = tl.load(projected + base + gate_real_offset, mask=mask, other=0.0)
    gate_imag = tl.load(projected + base + gate_imag_offset, mask=mask, other=0.0)
    gate_real_fp32 = gate_real.to(tl.float32)
    gate_imag_fp32 = gate_imag.to(tl.float32)
    magnitude = tl.log(1.0 + gate_real_fp32 * gate_real_fp32 + gate_imag_fp32 * gate_imag_fp32)
    magnitude = tl.where(mask, magnitude, 0.0)
    centered = magnitude - tl.sum(magnitude, axis=1)[:, None] / hidden
    active_alpha = tl.load(alpha + mode, mask=mode < hidden, other=0.0)
    relative = 1.0 + RHO * libdevice.tanh(active_alpha * centered)
    relative = tl.where(mask, relative, 0.0)
    mean_relative = tl.sum(relative, axis=1) / hidden
    gate = (relative / mean_relative[:, None]).to(tl.bfloat16)
    output_base = row * (2 * hidden) + mode
    tl.store(output + output_base, value_real * gate, mask=mask)
    tl.store(output + output_base + hidden, value_imag * gate, mask=mask)


@triton.jit
def _phase_gate_backward_reduce_kernel(  # pyright: ignore[reportUnusedFunction]
    partial_grad_alpha,
    grad_alpha,
    partial_count: int,
    hidden: int,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
) -> None:
    active_hidden = tl.program_id(0) * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    valid_hidden = active_hidden < hidden
    accumulator = tl.zeros((BLOCK_HIDDEN,), tl.float32)
    for partial_start in tl.range(
        0,
        partial_count,
        BLOCK_PARTIALS,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        partial = partial_start + tl.arange(0, BLOCK_PARTIALS)
        offset = partial[:, None] * hidden + active_hidden[None, :]
        mask = (partial[:, None] < partial_count) & valid_hidden[None, :]
        accumulator += tl.sum(
            tl.load(partial_grad_alpha + offset, mask=mask, other=0.0),
            axis=0,
        )
    tl.store(grad_alpha + active_hidden, accumulator, mask=valid_hidden)


@triton_op("lnet::phase_gate_v2", mutates_args={})
def _forward_op(  # pyright: ignore[reportUnusedFunction]
    projected: Tensor,
    alpha: Tensor,
    redistribution: float,
    self_gated: bool,
) -> Tensor:
    _validate(
        projected,
        alpha,
        redistribution=redistribution,
        self_gated=self_gated,
    )
    if not supports_phase_gate(
        projected,
        alpha,
        redistribution=redistribution,
        self_gated=self_gated,
    ):
        raise RuntimeError("CUDA phase gate requires contiguous BF16 activations")
    hidden = alpha.numel()
    rows = projected.numel() // projected.shape[-1]
    output = torch.empty(
        (*projected.shape[:-1], 2 * hidden), device=projected.device, dtype=projected.dtype
    )
    kernel = autotuned(
        _phase_gate_forward_kernel,
        FORWARD_LAUNCH_NAME,
        key=("rows", "hidden", "RHO", "SELF_GATED"),
        scope=_scope(_phase_gate_forward_kernel, projected, hidden, self_gated=self_gated),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        projected,
        alpha,
        output,
        rows,
        hidden,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(hidden)),
        RHO=redistribution,
        SELF_GATED=self_gated,
    )
    return output


__all__ = [
    "BACKWARD_REDUCE_LAUNCH_NAME",
    "FORWARD_LAUNCH_NAME",
    "phase_gate_reference",
    "supports_phase_gate",
]
