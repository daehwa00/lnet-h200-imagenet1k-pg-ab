#!/usr/bin/env python3
# pyright: reportExplicitAny=false
"""Run the mirrored-slope control required to make Phase S-A A4 valid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from lnet.alphabet2d_spectral_gate import (
    SpectralGateConfig,
    extract_spectral_features,
    fit_affine_head,
)
from lnet.alphabet2d_synthetic import (
    EqualPowerPhaseConfig,
    OffAxisSpectralConfig,
    make_alphabet2d_splits,
)

SEEDS = (11, 23, 47, 71, 101)
VARIANTS = ("axial2d", "product_single", "product_four")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _contract(batch_size: int) -> dict[str, Any]:
    data = OffAxisSpectralConfig(
        height=64,
        width=64,
        omega_x=math.pi / 4.0,
        omega_y=-math.pi / 3.0,
        contrast=0.4,
        matched_lag_radius=2,
    )
    features = SpectralGateConfig(modes=16, matched_lag_radius=2)
    return {
        "schema": "lnet.alphabet2d.phase_sa.mirror.v1",
        "evidence_status": "required directional control for A4",
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "data": asdict(data),
        "features": asdict(features),
        "counts": {
            "train_per_class": 5_000,
            "validation_per_class": 1_000,
            "test_per_class": 2_000,
        },
        "batch_size": batch_size,
        "source_sha256": {
            "runner": _digest(Path(__file__)),
            "gate": _digest(Path("src/lnet/alphabet2d_spectral_gate.py")),
            "generator": _digest(Path("src/lnet/alphabet2d_synthetic.py")),
        },
    }


def _initialize(root: Path, contract: dict[str, Any]) -> None:
    path = root / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            message = "existing mirror root has a different contract"
            raise RuntimeError(message)
    else:
        _atomic_json(path, contract)


def _features(
    inputs: torch.Tensor,
    variant: str,
    config: SpectralGateConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        chunks.extend(
            [
                extract_spectral_features(
                    batch.to(device),
                    variant,  # pyright: ignore[reportArgumentType]
                    config,
                ).cpu()
                for batch in inputs.split(batch_size)
            ]
        )
    return torch.cat(chunks)


def _run_seed(
    root: Path,
    contract: dict[str, Any],
    seed: int,
    device: torch.device,
) -> None:
    path = root / "results" / f"seed{seed}.json"
    if path.exists():
        return
    data = OffAxisSpectralConfig(**contract["data"])
    feature_config = SpectralGateConfig(**contract["features"])
    counts = contract["counts"]
    splits = make_alphabet2d_splits(
        "off_axis",
        train_per_class=counts["train_per_class"],
        validation_per_class=counts["validation_per_class"],
        test_per_class=counts["test_per_class"],
        seed=seed,
        off_axis_config=data,
        phase_config=EqualPowerPhaseConfig(),
    )
    results = {}
    for variant in VARIANTS:
        train = _features(
            splits.train.inputs,
            variant,
            feature_config,
            batch_size=contract["batch_size"],
            device=device,
        ).to(device)
        validation = _features(
            splits.validation.inputs,
            variant,
            feature_config,
            batch_size=contract["batch_size"],
            device=device,
        ).to(device)
        head, fit = fit_affine_head(
            train,
            splits.train.targets.to(device),
            validation,
            splits.validation.targets.to(device),
            seed=seed,
            config=feature_config,
        )
        test = _features(
            splits.test.inputs,
            variant,
            feature_config,
            batch_size=contract["batch_size"],
            device=device,
        ).to(device)
        standardized = (test - fit["mean"]) / fit["scale"]
        with torch.no_grad():
            accuracy = float(
                (
                    head(standardized).argmax(dim=-1)
                    == splits.test.targets.to(device)
                )
                .float()
                .mean()
            )
        results[variant] = {
            "test_balanced_accuracy": accuracy,
            "best_validation_accuracy": fit["best_validation_accuracy"],
        }
    _atomic_json(path, {"seed": seed, "variants": results})


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [root / "results" / f"seed{seed}.json" for seed in SEEDS]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    means = {
        variant: sum(
            row["variants"][variant]["test_balanced_accuracy"] for row in rows
        )
        / len(rows)
        for variant in VARIANTS
    }
    paired = [
        row["variants"]["product_single"]["test_balanced_accuracy"]
        - row["variants"]["product_four"]["test_balanced_accuracy"]
        for row in rows
    ]
    summary = {
        "schema": contract["schema"],
        "mean_balanced_accuracy": means,
        "paired_single_minus_four": paired,
        "mean_single_minus_four_pp": 100.0 * sum(paired) / len(paired),
        "A4_single_within_2pp_of_four_on_mirror": sum(paired) / len(paired) >= -0.02,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def main() -> None:
    args = _parse_args()
    contract = _contract(args.batch_size)
    args.root.mkdir(parents=True, exist_ok=True)
    _initialize(args.root, contract)
    if args.initialize_only:
        return
    if not set(args.run_seeds) <= set(SEEDS):
        message = "run seeds fall outside the mirror contract"
        raise ValueError(message)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        message = "mirror control requires CUDA"
        raise RuntimeError(message)
    for seed in args.run_seeds:
        _run_seed(args.root, contract, seed, device)
    summary = _summarize(args.root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
