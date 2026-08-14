#!/usr/bin/env python3
"""Calibrate fixed per-CFFN radial gains from ImageNet-100 training images."""

# ruff: noqa: ANN401, C901, I001, PLR0915, SLF001, T201

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import datasets

import run_a2d_deep4_m64_canonical8_calibrated_imagenet100 as control
from lnet.pac_complex_ffn import ComplexFFN
from lnet.complex_scan import ComplexScanConfig

SCHEMA = "lnet.a2d.m64_radial_rms_calibration.v1"
SITE_NAMES = tuple(
    f"stage{stage}.{site}" for stage in (1, 2, 3) for site in ("mode", "path", "transition", "post")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=4)
    return parser.parse_args()


def _site_projections(model: torch.nn.Module) -> dict[int, str]:
    sites: dict[int, str] = {}
    for stage_index in (1, 2, 3):
        stage = getattr(model, f"stage{stage_index}")
        combiner = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        sites[id(combiner.mode_input)] = f"stage{stage_index}.mode"
        sites[id(combiner.path_input)] = f"stage{stage_index}.path"
        sites[id(transition.ffn_input)] = f"stage{stage_index}.transition"
        sites[id(transition.post_ffn_input)] = f"stage{stage_index}.post"
    if set(sites.values()) != set(SITE_NAMES):
        message = "calibration did not resolve all twelve CFFN sites"
        raise RuntimeError(message)
    return sites


def _complex_sum_square(real: Tensor, imag: Tensor) -> float:
    return float((real.float().square() + imag.float().square()).sum().item())


def calibrate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    batches: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, float]], int]:
    """Measure Cartesian/raw-radial RMS on identical baseline hidden states."""
    if batches <= 0:
        message = "calibration batches must be positive"
        raise ValueError(message)
    projection_sites = _site_projections(model)
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    calls: dict[str, int] = defaultdict(int)
    original = ComplexFFN.run_cffn

    def observed_complex_ffn(real: Tensor, imag: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        projection = kwargs["input_projection"]
        site = projection_sites.get(id(projection))
        if site is not None:
            hidden_real, hidden_imag = projection(real, imag)
            hidden_addition = kwargs.get("hidden_addition")
            if hidden_addition is not None:
                hidden_real = hidden_real + hidden_addition[0]
                hidden_imag = hidden_imag + hidden_addition[1]
            hidden_transform = kwargs.get("hidden_transform")
            if hidden_transform is not None:
                hidden_real, hidden_imag = hidden_transform(hidden_real, hidden_imag)

            cartesian_real = functional.silu(hidden_real)
            cartesian_imag = functional.silu(hidden_imag)
            magnitude = torch.sqrt(
                hidden_real.float().square() + hidden_imag.float().square() + 1.0e-8
            )
            radial_gain = (2.0 * torch.sigmoid(magnitude - 1.0)).to(hidden_real.dtype)
            radial_real = radial_gain * hidden_real
            radial_imag = radial_gain * hidden_imag
            sums[site]["cartesian_sum_square"] += _complex_sum_square(
                cartesian_real,
                cartesian_imag,
            )
            sums[site]["radial_sum_square"] += _complex_sum_square(
                radial_real,
                radial_imag,
            )
            sums[site]["complex_count"] += hidden_real.numel()
            calls[site] += 1
        return original(real, imag, **kwargs)

    previous_disable = os.environ.get("LNET_DISABLE_PACKED_POSTCARRY_INFERENCE")
    os.environ["LNET_DISABLE_PACKED_POSTCARRY_INFERENCE"] = "1"
    ComplexFFN.run_cffn = staticmethod(observed_complex_ffn)
    images_seen = 0
    try:
        with torch.inference_mode():
            for batch_index, (images, _) in enumerate(loader):
                if batch_index >= batches:
                    break
                device_images = images.to(device, non_blocking=True)
                model(device_images)
                images_seen += int(device_images.shape[0])
    finally:
        ComplexFFN.run_cffn = staticmethod(original)
        if previous_disable is None:
            os.environ.pop("LNET_DISABLE_PACKED_POSTCARRY_INFERENCE", None)
        else:
            os.environ["LNET_DISABLE_PACKED_POSTCARRY_INFERENCE"] = previous_disable

    expected_calls = dict.fromkeys(SITE_NAMES, batches)
    if dict(calls) != expected_calls:
        message = f"unexpected CFFN calibration calls: {dict(calls)}"
        raise RuntimeError(message)

    scales: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for site in SITE_NAMES:
        row = sums[site]
        count = row["complex_count"]
        cartesian_rms = (row["cartesian_sum_square"] / count) ** 0.5
        radial_rms = (row["radial_sum_square"] / count) ** 0.5
        if not (cartesian_rms > 0.0 and radial_rms > 0.0):
            message = f"invalid activation RMS at {site}"
            raise RuntimeError(message)
        scale = cartesian_rms / radial_rms
        scales[site] = scale
        diagnostics[site] = {
            "cartesian_rms": cartesian_rms,
            "raw_radial_rms": radial_rms,
            "raw_radial_over_cartesian": radial_rms / cartesian_rms,
            "kappa": scale,
        }
    return scales, diagnostics, images_seen


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.batches <= 0:
        message = "calibration batch size and count must be positive"
        raise ValueError(message)
    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = control._build(
        control.VARIANT,
        ComplexScanConfig(
            output_dim=100,
            stem_strides=(2, 2),
        ),
    ).to(device)
    model.eval()

    training_transform, _ = control.heads.harness._transforms()
    dataset = datasets.ImageFolder(args.data_root / "train", training_transform)
    calibration_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=calibration_generator,
        num_workers=0,
        drop_last=True,
    )
    scales, diagnostics, images_seen = calibrate(
        model,
        loader,
        batches=args.batches,
        device=device,
    )
    payload = {
        "schema": SCHEMA,
        "seed": seed,
        "split": "train",
        "batch_size": int(args.batch_size),
        "batches": int(args.batches),
        "images": images_seen,
        "control_activation": "cartesian_silu",
        "candidate_activation": "2*sigmoid(abs(U)-1)*U",
        "complex_rms": "sqrt(mean(real^2 + imag^2))",
        "scales": scales,
        "diagnostics": diagnostics,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
