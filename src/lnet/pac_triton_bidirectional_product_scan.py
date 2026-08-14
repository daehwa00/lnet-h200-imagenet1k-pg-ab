"""Measured bidirectional static product scan for the horizontal D4 path."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, ANN202, EM101, N803, PLR0915, TRY003
from typing import Protocol, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .pac_kernel_launch_config import (
    LaunchGeometry,
    LaunchScope,
    autotuned,
    make_launch_scope,
    register_default,
)
from .pac_product_scan_reference import bidirectional_product_scan_reference

ComplexField = tuple[Tensor, Tensor]
BidirectionalField = tuple[Tensor, Tensor, Tensor, Tensor]

FORWARD_LAUNCH_NAME = "bidirectional_product_scan_forward"
BACKWARD_LAUNCH_NAME = "bidirectional_product_scan_backward"
RECOMPUTE_BACKWARD_LAUNCH_NAME = "bidirectional_product_scan_backward_recompute"
_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(
        num_warps=warps,
        blocks={"BLOCK_MODES": block_modes},
    )
    for block_modes, warps in (
        (8, 4),
        (8, 8),
        (16, 4),
        (16, 8),
        (32, 4),
        (32, 8),
        (64, 8),
    )
)
_DEFAULT_LAUNCH_GEOMETRY = LaunchGeometry.build(
    num_warps=4,
    blocks={"BLOCK_MODES": 16},
)


register_default(
    FORWARD_LAUNCH_NAME,
    _DEFAULT_LAUNCH_GEOMETRY,
    candidates=_LAUNCH_CANDIDATES,
)
register_default(
    BACKWARD_LAUNCH_NAME,
    _DEFAULT_LAUNCH_GEOMETRY,
    candidates=_LAUNCH_CANDIDATES,
)
register_default(
    RECOMPUTE_BACKWARD_LAUNCH_NAME,
    _DEFAULT_LAUNCH_GEOMETRY,
    candidates=_LAUNCH_CANDIDATES,
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _scan_launch_scope(kernel: object, source: Tensor) -> LaunchScope:
    batch, height, width, modes = source.shape
    return make_launch_scope(
        kernel,
        source,
        shape={
            "batch": batch,
            "height": height,
            "width": width,
            "modes": modes,
        },
    )


def _validate(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
) -> None:
    coefficients = decay_real, decay_imag, gamma_real, gamma_imag
    if source_real.shape != source_imag.shape or source_real.ndim != 4:
        raise ValueError("bidirectional product scan requires matching NHWM sources")
    if any(value.shape != (1, 1, 1, source_real.shape[-1]) for value in coefficients):
        raise ValueError("bidirectional product scan requires compact 111M poles")
    if any(value.device != source_real.device for value in (*coefficients, source_imag)):
        raise ValueError("bidirectional product scan tensors must share one device")
    if any(value.dtype != decay_real.dtype for value in coefficients[1:]):
        raise TypeError("bidirectional product scan poles must share one dtype")
    if source_imag.dtype != source_real.dtype:
        raise TypeError("bidirectional product scan sources must share one dtype")


@triton.jit
def _compose_complex(
    left_ar,
    left_ai,
    left_r,
    left_i,
    right_ar,
    right_ai,
    right_r,
    right_i,
):
    return (
        right_ar * left_ar - right_ai * left_ai,
        right_ai * left_ar + right_ar * left_ai,
        right_ar * left_r - right_ai * left_i + right_r,
        right_ai * left_r + right_ar * left_i + right_i,
    )


@triton.jit
def _bidirectional_forward_kernel(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    source_real,
    source_imag,
    positive_real,
    positive_imag,
    negative_real,
    negative_imag,
    height: int,
    width: int,
    line_count: int,
    modes: int,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    line = tl.program_id(0)
    batch = line // height
    y = line - batch * height
    valid_line = line < line_count
    x = tl.arange(0, BLOCK_WIDTH)[:, None]
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)[None, :]
    active = valid_line & (x < width) & (mode < modes)
    offset = ((batch * height + y) * width + x) * modes + mode
    ar = tl.load(decay_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    sr = tl.load(source_real + offset, mask=active, other=0.0).to(tl.float32)
    si = tl.load(source_imag + offset, mask=active, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    scan_ai = tl.where(active, ai, 0.0)
    positive = tl.associative_scan(
        (scan_ar, scan_ai, gr * sr - gi * si, gr * si + gi * sr),
        axis=0,
        combine_fn=_compose_complex,
    )
    negative = tl.associative_scan(
        (scan_ar, -scan_ai, gr * sr + gi * si, gr * si - gi * sr),
        axis=0,
        combine_fn=_compose_complex,
        reverse=True,
    )
    tl.store(positive_real + offset, positive[2], mask=active)
    tl.store(positive_imag + offset, positive[3], mask=active)
    tl.store(negative_real + offset, negative[2], mask=active)
    tl.store(negative_imag + offset, negative[3], mask=active)


@triton.jit
def _bidirectional_backward_kernel(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    source_real,
    source_imag,
    positive_real,
    positive_imag,
    negative_real,
    negative_imag,
    grad_positive_real,
    grad_positive_imag,
    grad_negative_real,
    grad_negative_imag,
    grad_decay_real,
    grad_decay_imag,
    grad_gamma_real,
    grad_gamma_imag,
    grad_source_real,
    grad_source_imag,
    height: int,
    width: int,
    line_count: int,
    modes: int,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    RECOMPUTE_STATES: tl.constexpr,
    STORAGE_KIND: tl.constexpr,
) -> None:
    line = tl.program_id(0)
    batch = line // height
    y = line - batch * height
    valid_line = line < line_count
    x = tl.arange(0, BLOCK_WIDTH)[:, None]
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)[None, :]
    active = valid_line & (x < width) & (mode < modes)
    valid_mode = valid_line & (mode < modes)
    offset = ((batch * height + y) * width + x) * modes + mode
    ar = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gpr = tl.load(grad_positive_real + offset, mask=active, other=0.0).to(tl.float32)
    gpi = tl.load(grad_positive_imag + offset, mask=active, other=0.0).to(tl.float32)
    gnr = tl.load(grad_negative_real + offset, mask=active, other=0.0).to(tl.float32)
    gni = tl.load(grad_negative_imag + offset, mask=active, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    positive_adjoint = tl.associative_scan(
        (scan_ar, tl.where(active, -ai, 0.0), gpr, gpi),
        axis=0,
        combine_fn=_compose_complex,
        reverse=True,
    )
    negative_adjoint = tl.associative_scan(
        (scan_ar, tl.where(active, ai, 0.0), gnr, gni),
        axis=0,
        combine_fn=_compose_complex,
    )
    plr, pli = positive_adjoint[2], positive_adjoint[3]
    nlr, nli = negative_adjoint[2], negative_adjoint[3]
    sr = tl.load(source_real + offset, mask=active, other=0.0).to(tl.float32)
    si = tl.load(source_imag + offset, mask=active, other=0.0).to(tl.float32)
    if RECOMPUTE_STATES:
        positive_previous_offset = offset - modes
        negative_previous_offset = offset + modes
        psr = tl.load(
            source_real + positive_previous_offset,
            mask=active & (x > 0),
            other=0.0,
        ).to(tl.float32)
        psi = tl.load(
            source_imag + positive_previous_offset,
            mask=active & (x > 0),
            other=0.0,
        ).to(tl.float32)
        nsr = tl.load(
            source_real + negative_previous_offset,
            mask=active & (x < width - 1),
            other=0.0,
        ).to(tl.float32)
        nsi = tl.load(
            source_imag + negative_previous_offset,
            mask=active & (x < width - 1),
            other=0.0,
        ).to(tl.float32)
        positive_previous = tl.associative_scan(
            (scan_ar, tl.where(active, ai, 0.0), gr * psr - gi * psi, gr * psi + gi * psr),
            axis=0,
            combine_fn=_compose_complex,
        )
        negative_previous = tl.associative_scan(
            (scan_ar, tl.where(active, -ai, 0.0), gr * nsr + gi * nsi, gr * nsi - gi * nsr),
            axis=0,
            combine_fn=_compose_complex,
            reverse=True,
        )
        ppr, ppi = positive_previous[2], positive_previous[3]
        npr, npi = negative_previous[2], negative_previous[3]
        # Match the retained path, whose horizontal states cross a storage
        # boundary before the vertical adjoint consumes them.
        if STORAGE_KIND == 1:
            ppr, ppi = ppr.to(tl.bfloat16).to(tl.float32), ppi.to(tl.bfloat16).to(tl.float32)
            npr, npi = npr.to(tl.bfloat16).to(tl.float32), npi.to(tl.bfloat16).to(tl.float32)
        elif STORAGE_KIND == 2:
            ppr, ppi = ppr.to(tl.float16).to(tl.float32), ppi.to(tl.float16).to(tl.float32)
            npr, npi = npr.to(tl.float16).to(tl.float32), npi.to(tl.float16).to(tl.float32)
    else:
        positive_previous_offset = offset - modes
        negative_previous_offset = offset + modes
        ppr = tl.load(
            positive_real + positive_previous_offset,
            mask=active & (x > 0),
            other=0.0,
        ).to(tl.float32)
        ppi = tl.load(
            positive_imag + positive_previous_offset,
            mask=active & (x > 0),
            other=0.0,
        ).to(tl.float32)
        npr = tl.load(
            negative_real + negative_previous_offset,
            mask=active & (x < width - 1),
            other=0.0,
        ).to(tl.float32)
        npi = tl.load(
            negative_imag + negative_previous_offset,
            mask=active & (x < width - 1),
            other=0.0,
        ).to(tl.float32)
    tl.store(grad_source_real + offset, plr * gr + pli * gi + nlr * gr - nli * gi, mask=active)
    tl.store(grad_source_imag + offset, -plr * gi + pli * gr + nlr * gi + nli * gr, mask=active)
    cdr = tl.sum(
        tl.where(active, plr * ppr + pli * ppi + nlr * npr + nli * npi, 0.0),
        axis=0,
    )
    cdi = tl.sum(
        tl.where(active, -plr * ppi + pli * ppr + nlr * npi - nli * npr, 0.0),
        axis=0,
    )
    cgr = tl.sum(
        tl.where(active, plr * sr + pli * si + nlr * sr + nli * si, 0.0),
        axis=0,
    )
    cgi = tl.sum(
        tl.where(active, -plr * si + pli * sr + nlr * si - nli * sr, 0.0),
        axis=0,
    )
    tl.atomic_add(grad_decay_real + mode, cdr[None, :], mask=valid_mode)
    tl.atomic_add(grad_decay_imag + mode, cdi[None, :], mask=valid_mode)
    tl.atomic_add(grad_gamma_real + mode, cgr[None, :], mask=valid_mode)
    tl.atomic_add(grad_gamma_imag + mode, cgi[None, :], mask=valid_mode)


@triton_op("lnet::pac_bidirectional_product_scan", mutates_args={})
def _forward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
) -> BidirectionalField:
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    source = source_real, source_imag
    _validate(*pole, *source)
    if not source_real.is_cuda:
        return bidirectional_product_scan_reference(pole, source)
    outputs = cast(
        "BidirectionalField",
        tuple(torch.empty_like(source_real) for _ in range(4)),
    )
    batch, height, width, modes = source_real.shape
    line_count = batch * height
    forward_kernel = autotuned(
        _bidirectional_forward_kernel,
        FORWARD_LAUNCH_NAME,
        key=("height", "width", "line_count", "modes"),
        scope=_scan_launch_scope(_bidirectional_forward_kernel, source_real),
    )

    def grid(metadata: dict[str, int]) -> tuple[int, int]:
        return line_count, int(triton.cdiv(modes, metadata["BLOCK_MODES"]))

    wrap_triton(forward_kernel)[grid](
        *(value.contiguous() for value in (*pole, *source)),
        *outputs,
        height,
        width,
        line_count,
        modes,
        BLOCK_WIDTH=triton.next_power_of_2(width),
    )
    return outputs


def _storage_kind(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float16:
        return 2
    return 0


def _launch_backward(
    pole: tuple[Tensor, Tensor, Tensor, Tensor],
    source: ComplexField,
    states: BidirectionalField | None,
    output_gradients: BidirectionalField,
    *,
    launch_name: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    source_real, source_imag = source
    batch, height, width, modes = source_real.shape
    line_count = batch * height
    coefficient_gradients = cast(
        "tuple[Tensor, Tensor, Tensor, Tensor]",
        tuple(torch.zeros_like(value, memory_format=torch.contiguous_format) for value in pole),
    )
    source_gradients = torch.empty_like(source_real), torch.empty_like(source_imag)
    backward_kernel = autotuned(
        _bidirectional_backward_kernel,
        launch_name,
        key=("height", "width", "line_count", "modes"),
        restore_value=(
            "grad_decay_real",
            "grad_decay_imag",
            "grad_gamma_real",
            "grad_gamma_imag",
        ),
        scope=_scan_launch_scope(_bidirectional_backward_kernel, source_real),
    )

    def grid(metadata: dict[str, int]) -> tuple[int, int]:
        return line_count, int(triton.cdiv(modes, metadata["BLOCK_MODES"]))

    wrap_triton(backward_kernel)[grid](
        *(value.contiguous() for value in pole),
        *(value.contiguous() for value in source),
        *(value.contiguous() for value in (states or (source_real,) * 4)),
        *(value.contiguous() for value in output_gradients),
        *coefficient_gradients,
        *source_gradients,
        height,
        width,
        line_count,
        modes,
        BLOCK_WIDTH=triton.next_power_of_2(width),
        RECOMPUTE_STATES=states is None,
        STORAGE_KIND=_storage_kind(source_real.dtype),
    )
    return *coefficient_gradients, *source_gradients


@triton_op("lnet::pac_bidirectional_product_scan_backward", mutates_args={})
def _backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
    positive_real: Tensor,
    positive_imag: Tensor,
    negative_real: Tensor,
    negative_imag: Tensor,
    grad_positive_real: Tensor,
    grad_positive_imag: Tensor,
    grad_negative_real: Tensor,
    grad_negative_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    source = source_real, source_imag
    _validate(*pole, *source)
    return _launch_backward(
        pole,
        source,
        (positive_real, positive_imag, negative_real, negative_imag),
        (grad_positive_real, grad_positive_imag, grad_negative_real, grad_negative_imag),
        launch_name=BACKWARD_LAUNCH_NAME,
    )


@triton_op("lnet::pac_bidirectional_product_scan_backward_recompute", mutates_args={})
def _recomputed_backward_op(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
    grad_positive_real: Tensor,
    grad_positive_imag: Tensor,
    grad_negative_real: Tensor,
    grad_negative_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    source = source_real, source_imag
    _validate(*pole, *source)
    return _launch_backward(
        pole,
        source,
        None,
        (grad_positive_real, grad_positive_imag, grad_negative_real, grad_negative_imag),
        launch_name=RECOMPUTE_BACKWARD_LAUNCH_NAME,
    )


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, ...],
    output: BidirectionalField,
) -> None:
    ctx.save_for_backward(*inputs, *output)


def _backward(
    ctx: _AutogradContext,
    grad_positive_real: Tensor | None,
    grad_positive_imag: Tensor | None,
    grad_negative_real: Tensor | None,
    grad_negative_imag: Tensor | None,
) -> tuple[Tensor | None, ...]:
    *inputs, positive_real, positive_imag, negative_real, negative_imag = ctx.saved_tensors
    source_real = inputs[4]
    gradients = [
        torch.zeros_like(source_real) if value is None else value.contiguous()
        for value in (
            grad_positive_real,
            grad_positive_imag,
            grad_negative_real,
            grad_negative_imag,
        )
    ]
    if not source_real.is_cuda:
        differentiable = tuple(value.detach().requires_grad_() for value in inputs)
        with torch.enable_grad():
            outputs = bidirectional_product_scan_reference(
                (differentiable[0], differentiable[1], differentiable[2], differentiable[3]),
                (differentiable[4], differentiable[5]),
            )
            return torch.autograd.grad(outputs, differentiable, gradients)
    return _backward_op(
        *inputs,
        positive_real,
        positive_imag,
        negative_real,
        negative_imag,
        *gradients,
    )


torch.library.register_autograd(
    "lnet::pac_bidirectional_product_scan",
    _backward,
    setup_context=_setup_context,
)


def pac_triton_bidirectional_product_scan(
    pole: tuple[Tensor, Tensor, Tensor, Tensor],
    source: ComplexField,
) -> BidirectionalField:
    """Return positive and negative horizontal states without duplicate dispatch."""
    return _forward_op(*pole, *source)


__all__ = ["pac_triton_bidirectional_product_scan"]
