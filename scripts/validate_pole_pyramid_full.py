"""Deterministic preflight checks for the full PolePyramid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from lnet.alphabet2d import product_pole_scan_2d
from lnet.pac_directional import direction_aligned_endpoints
from lnet.pac_real2d_math import discrete_pole_real2d
from lnet.pole_pyramid import strided_product_pole_scan_2d
from lnet.pole_pyramid_full import (
    FullPolePyramid,
    FullPolePyramidConfig,
    SharedPhysicalPoleGeometry,
)

_QUADRANTS = ((1, 1), (-1, 1), (1, -1), (-1, -1))


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate(device: torch.device) -> dict[str, object]:
    torch.manual_seed(20260731)
    geometry = SharedPhysicalPoleGeometry(16).to(device)
    fine_decay_real, fine_decay_imag, _, _ = discrete_pole_real2d(
        geometry.damping_x(), geometry.frequency_x(), 1.0
    )
    coarse_decay_real, coarse_decay_imag, _, _ = discrete_pole_real2d(
        geometry.damping_x(), geometry.frequency_x(), 2.0
    )
    pole_square_error = max(
        float(
            (
                coarse_decay_real - (fine_decay_real.square() - fine_decay_imag.square())
            ).detach()
            .abs()
            .max()
        ),
        float(
            (coarse_decay_imag - 2.0 * fine_decay_real * fine_decay_imag)
            .detach()
            .abs()
            .max()
        ),
    )

    excitation_real = torch.randn(2, 8, 8, 16, device=device, requires_grad=True)
    excitation_imag = torch.randn(2, 8, 8, 16, device=device, requires_grad=True)
    endpoint_errors = []
    for direction_x, direction_y in _QUADRANTS:
        direct_real, direct_imag = strided_product_pole_scan_2d(
            excitation_real,
            excitation_imag,
            damping_x=geometry.damping_x(),
            damping_y=geometry.damping_y(),
            frequency_x=geometry.frequency_x(),
            frequency_y=geometry.frequency_y(),
            spacing_x=1.0,
            spacing_y=1.0,
            stride=2,
            direction_x=direction_x,
            direction_y=direction_y,
        )
        full_real, full_imag = product_pole_scan_2d(
            excitation_real,
            excitation_imag,
            damping_x=geometry.damping_x(),
            damping_y=geometry.damping_y(),
            frequency_x=geometry.frequency_x(),
            frequency_y=geometry.frequency_y(),
            spacing_x=1.0,
            spacing_y=1.0,
            direction_x=direction_x,
            direction_y=direction_y,
        )
        selected_real, selected_imag = direction_aligned_endpoints(
            full_real,
            full_imag,
            direction_x=direction_x,
            direction_y=direction_y,
        )
        endpoint_errors.append(
            max(
                float((selected_real - direct_real).detach().abs().max()),
                float((selected_imag - direct_imag).detach().abs().max()),
            )
        )

    model_rows = {}
    for transport in ("pole", "average"):
        model = FullPolePyramid(FullPolePyramidConfig(transport=transport)).to(device)
        inputs = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
        logits = model(inputs)
        loss = logits.square().mean()
        loss.backward()
        model_rows[transport] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "descriptor_dim": model.descriptor_dim,
            "finite_logits": bool(torch.isfinite(logits).all()),
            "finite_input_gradient": bool(
                inputs.grad is not None and torch.isfinite(inputs.grad).all()
            ),
            "loss": float(loss.detach()),
        }

    endpoint_max_abs_error = max(endpoint_errors)
    parameter_match = (
        model_rows["pole"]["parameters"] == model_rows["average"]["parameters"]
    )
    gate_pass = (
        pole_square_error <= 1.0e-6
        and endpoint_max_abs_error <= 1.0e-5
        and parameter_match
        and all(bool(row["finite_logits"]) for row in model_rows.values())
        and all(bool(row["finite_input_gradient"]) for row in model_rows.values())
    )
    return {
        "schema": "lnet.pole_pyramid.full_preflight.v1",
        "device": str(device),
        "pole_square_max_abs_error": pole_square_error,
        "endpoint_max_abs_error": endpoint_max_abs_error,
        "parameter_match": parameter_match,
        "models": model_rows,
        "gate_pass": gate_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = validate(device)
    if args.output is not None:
        _atomic_json(args.output, payload)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
