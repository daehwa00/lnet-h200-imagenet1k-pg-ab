"""Run deterministic exactness and anti-alias diagnostics for PolePyramid."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import torch

from lnet.pole_pyramid import (
    decimated_product_pole_scan_2d,
    strided_product_pole_scan_2d,
)

if TYPE_CHECKING:
    from lnet.pac_recurrence import RecurrenceBackend


class _PoleArguments(TypedDict):
    damping_x: torch.Tensor
    damping_y: torch.Tensor
    frequency_x: torch.Tensor
    frequency_y: torch.Tensor
    spacing_x: float
    spacing_y: float
    recurrence_backend: RecurrenceBackend


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _pole_arguments(
    modes: int,
    size: int,
    device: torch.device,
) -> _PoleArguments:
    return {
        "damping_x": torch.linspace(0.2, 0.8, modes, device=device),
        "damping_y": torch.linspace(0.3, 0.9, modes, device=device),
        "frequency_x": torch.linspace(-2.0, 2.0, modes, device=device),
        "frequency_y": torch.linspace(1.5, -1.5, modes, device=device),
        "spacing_x": 1.0 / size,
        "spacing_y": 1.0 / size,
        "recurrence_backend": "real2d_loop",
    }


def _endpoint_error(device: torch.device) -> float:
    generator = torch.Generator(device=device).manual_seed(20260731)
    maximum = 0.0
    for stride in (2, 4):
        for direction_x in (-1, 1):
            for direction_y in (-1, 1):
                real = torch.randn(
                    2,
                    8,
                    8,
                    3,
                    generator=generator,
                    device=device,
                )
                imag = torch.randn(
                    real.shape,
                    generator=generator,
                    device=device,
                )
                arguments = _pole_arguments(3, 8, device)
                direct = strided_product_pole_scan_2d(
                    real,
                    imag,
                    stride=stride,
                    direction_x=direction_x,
                    direction_y=direction_y,
                    **arguments,
                )
                reference = decimated_product_pole_scan_2d(
                    real,
                    imag,
                    stride=stride,
                    direction_x=direction_x,
                    direction_y=direction_y,
                    **arguments,
                )
                maximum = max(
                    maximum,
                    float((direct[0] - reference[0]).abs().max()),
                    float((direct[1] - reference[1]).abs().max()),
                )
    return maximum


def _gradient_error(device: torch.device) -> float:
    size = 8
    direct_real = torch.randn(1, size, size, 2, device=device, requires_grad=True)
    direct_imag = torch.randn_like(direct_real, requires_grad=True)
    reference_real = direct_real.detach().clone()
    reference_imag = direct_imag.detach().clone()
    reference_real.requires_grad = True
    reference_imag.requires_grad = True
    weights = torch.randn(1, 4, 4, 2, device=device)
    arguments = _pole_arguments(2, size, device)
    direct = strided_product_pole_scan_2d(
        direct_real,
        direct_imag,
        stride=2,
        direction_x=-1,
        direction_y=1,
        **arguments,
    )
    reference = decimated_product_pole_scan_2d(
        reference_real,
        reference_imag,
        stride=2,
        direction_x=-1,
        direction_y=1,
        **arguments,
    )
    (direct[0] * weights + direct[1] * weights.flip(-1)).sum().backward()
    (reference[0] * weights + reference[1] * weights.flip(-1)).sum().backward()
    gradients = (
        direct_real.grad,
        direct_imag.grad,
        reference_real.grad,
        reference_imag.grad,
    )
    if any(gradient is None for gradient in gradients):
        message = "PoleDown gradient diagnostic produced a missing gradient"
        raise RuntimeError(message)
    direct_real_gradient = direct_real.grad
    direct_imag_gradient = direct_imag.grad
    reference_real_gradient = reference_real.grad
    reference_imag_gradient = reference_imag.grad
    if (
        direct_real_gradient is None
        or direct_imag_gradient is None
        or reference_real_gradient is None
        or reference_imag_gradient is None
    ):
        message = "gradient narrowing failed after validation"
        raise RuntimeError(message)
    return max(
        float((direct_real_gradient - reference_real_gradient).abs().max()),
        float((direct_imag_gradient - reference_imag_gradient).abs().max()),
    )


def _sinusoid_energy(
    cycles: int,
    *,
    shift: int,
    device: torch.device,
) -> float:
    size = 32
    positions = torch.arange(size, device=device) / size
    signal = torch.sin(2.0 * math.pi * cycles * positions)
    signal = torch.roll(signal, shift).view(1, 1, size, 1)
    signal = signal.expand(1, size, size, 1).contiguous()
    state = strided_product_pole_scan_2d(
        signal,
        torch.zeros_like(signal),
        damping_x=torch.tensor((1.0,), device=device),
        damping_y=torch.tensor((1.0,), device=device),
        frequency_x=torch.zeros(1, device=device),
        frequency_y=torch.zeros(1, device=device),
        spacing_x=1.0 / size,
        spacing_y=1.0 / size,
        stride=2,
        direction_x=1,
        direction_y=1,
        recurrence_backend="real2d_loop",
    )
    return float((state[0][:, 4:, 4:].square() + state[1][:, 4:, 4:].square()).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endpoint_error = _endpoint_error(device)
    gradient_error = _gradient_error(device)
    low_energy = _sinusoid_energy(2, shift=0, device=device)
    high_energy = _sinusoid_energy(14, shift=0, device=device)
    shifted_energy = _sinusoid_energy(2, shift=1, device=device)
    high_to_low = high_energy / low_energy
    phase_relative = abs(shifted_energy - low_energy) / low_energy
    checks = {
        "endpoint_error_below_1e-5": endpoint_error < 1.0e-5,
        "gradient_error_below_1e-5": gradient_error < 1.0e-5,
        "high_frequency_attenuation_below_0.1": high_to_low < 0.1,
        "low_frequency_phase_energy_change_below_0.1": phase_relative < 0.1,
    }
    payload = {
        "schema": "lnet.pole_pyramid.synthetic_gate.v1",
        "device": str(device),
        "endpoint_max_abs_error": endpoint_error,
        "input_gradient_max_abs_error": gradient_error,
        "low_frequency_energy": low_energy,
        "high_frequency_energy": high_energy,
        "high_to_low_energy_ratio": high_to_low,
        "one_pixel_phase_relative_energy_change": phase_relative,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "scope_note": (
            "Exactness is endpoint-state equivalence within each linear "
            "PoleDown block, not invertible block compression."
        ),
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201
    if not payload["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
