# ruff: noqa: T201

"""Standalone CUDA correctness gate for the fused static product scan."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lnet.pac_product_scan_pipeline import run_product_scan_pipeline
from lnet.pac_product_scan_reference import bidirectional_product_scan_reference
from lnet.pac_real2d_math import discrete_pole_real2d
from lnet.pac_triton_product_scan_coarse4 import (
    pac_triton_product_scan_coarse4,
    product_scan_coarse4_reference,
)

Pole = tuple[Tensor, Tensor, Tensor, Tensor]
ComplexField = tuple[Tensor, Tensor]


def _inputs(
    height: int,
    modes: int = 64,
    *,
    noncontiguous: bool = False,
) -> tuple[tuple[Pole, Pole], tuple[ComplexField, ...]]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(31 + height)
    poles = []
    damping = torch.logspace(
        math.log10(0.04),
        math.log10(0.35),
        modes,
        device=device,
    ).view(1, 1, 1, modes)
    for axis_scale in (0.75, 0.70):
        phase = torch.linspace(
            0.0,
            axis_scale * math.pi,
            modes,
            device=device,
        ).view(1, 1, 1, modes)
        pole = tuple(
            value.detach().clone().requires_grad_()
            for value in discrete_pole_real2d(damping, phase, 1.0)
        )
        poles.append(pole)

    def source() -> Tensor:
        value = torch.randn(
            (2, height, height, modes),
            generator=generator,
            device=device,
        )
        if noncontiguous:
            value = value.transpose(1, 2)
        return value.detach().requires_grad_()

    sources = tuple((source(), source()) for _ in range(2))
    return (poles[0], poles[1]), sources


def _reference(
    poles: tuple[Pole, Pole],
    sources: tuple[ComplexField, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    return product_scan_coarse4_reference(
        *poles,
        sources[0],
        sources[1],
    )


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    return float((actual - expected).norm() / expected.norm().clamp_min(1.0e-8))


def _check(
    height: int,
    *,
    noncontiguous: bool = False,
) -> None:
    poles, sources = _inputs(height, noncontiguous=noncontiguous)
    leaves = (*poles[1], *sources[0], *sources[1])
    reference = _reference(poles, sources)
    fused = pac_triton_product_scan_coarse4(
        *poles,
        sources[0],
        sources[1],
    )
    for actual, expected in zip(fused[:2], reference[:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-5)
    torch.testing.assert_close(fused[2], reference[2], rtol=2.0e-4, atol=2.0e-4)

    sum(value.square().mean() for value in reference).backward(retain_graph=True)
    expected_gradients = tuple(value.grad.detach().clone() for value in leaves)
    for value in leaves:
        value.grad = None
    sum(value.square().mean() for value in fused).backward()
    errors = []
    diagnostics = []
    for actual, expected in zip(leaves, expected_gradients, strict=True):
        error = _relative_error(actual.grad, expected)
        errors.append(error)
        diagnostics.append((error, float(actual.grad.norm()), float(expected.norm())))
    if max(errors) >= 5.0e-3:
        message = f"fused adjoint diagnostics are too large: {diagnostics}"
        raise AssertionError(message)
    print(
        f"height={height} descriptor=raw noncontiguous={noncontiguous} "
        f"max_gradient_relative_error={max(errors):.6g}"
    )


def _check_full_pipeline(height: int) -> None:
    poles, raw_sources = _inputs(height)
    pole_x, pole_y = poles
    excitation = raw_sources[0]
    positive_real, positive_imag, negative_real, negative_imag = (
        bidirectional_product_scan_reference(pole_x, excitation)
    )
    positive = positive_real, positive_imag
    negative = negative_real, negative_imag
    sources = (positive, negative)
    leaves = (*pole_x, *pole_y, *excitation)
    reference = _reference(poles, sources)
    fused = run_product_scan_pipeline(
        pole_x,
        pole_y,
        excitation,
        epilogue="coarse",
        gain_normalization="pointwise",
    )
    for actual, expected in zip(fused[:2], reference[:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-5)
    torch.testing.assert_close(fused[2], reference[2], rtol=2.0e-4, atol=2.0e-4)

    sum(value.square().mean() for value in reference).backward(retain_graph=True)
    expected_gradients = tuple(value.grad.detach().clone() for value in leaves)
    for value in leaves:
        value.grad = None
    sum(value.square().mean() for value in fused).backward()
    errors = tuple(
        _relative_error(actual.grad, expected)
        for actual, expected in zip(leaves, expected_gradients, strict=True)
    )
    if max(errors) >= 5.0e-3:
        message = f"full M64 fused pipeline adjoint errors are too large: {errors}"
        raise AssertionError(message)
    print(
        f"full_pipeline height={height} modes=64 descriptor=raw "
        f"max_gradient_relative_error={max(errors):.6g}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        message = "CUDA is required"
        raise RuntimeError(message)
    _check(6)
    _check(6, noncontiguous=True)
    _check(56)
    _check_full_pipeline(56)


if __name__ == "__main__":
    main()
