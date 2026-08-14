#!/usr/bin/env python3
# pyright: reportExplicitAny=false
"""Run the preregistered ALPHABET-2D Phase S-A gate on local_gpu GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lnet.alphabet2d_spectral_gate import (
    SPECTRAL_GATE_VARIANTS,
    SpectralGateConfig,
    extract_spectral_features,
    fit_affine_head,
)
from lnet.alphabet2d_synthetic import (
    OffAxisSpectralConfig,
    TensorClassificationSplit,
    make_alphabet2d_splits,
    off_axis_oracle_score,
    off_axis_spectra,
)

EPSILONS = (0.2, 0.4, 0.6, 0.8)
SEEDS = (11, 23, 47, 71, 101)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(EPSILONS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Exploratory epsilon=.4 futility screen; never used for final gates.",
    )
    return parser.parse_args()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _contract(
    *,
    pilot: bool,
    batch_size: int,
) -> dict[str, Any]:
    data = OffAxisSpectralConfig(height=64, width=64, matched_lag_radius=2)
    feature = SpectralGateConfig(modes=16, matched_lag_radius=2)
    counts = {
        "train_per_class": 2_000 if pilot else 5_000,
        "validation_per_class": 500 if pilot else 1_000,
        "test_per_class": 2_000,
    }
    return {
        "schema": "lnet.alphabet2d.phase_sa.v1",
        "evidence_status": "exploratory futility screen" if pilot else "preregistered",
        "pilot": pilot,
        "epsilons": [0.4] if pilot else list(EPSILONS),
        "seeds": [11, 23, 47] if pilot else list(SEEDS),
        "variants": list(SPECTRAL_GATE_VARIANTS),
        "data": asdict(data),
        "features": asdict(feature),
        "counts": counts,
        "batch_size": batch_size,
        "head": (
            "train-standardized affine softmax; validation checkpoint; "
            "identical 16-coordinate budget"
        ),
        "product_scan": {
            "single": [[1, 1]],
            "four": [[1, 1], [-1, 1], [1, -1], [-1, -1]],
            "four_aggregation": "fixed mean; no extra head coordinates",
        },
        "source_sha256": {
            "runner": _digest(Path(__file__)),
            "gate": _digest(Path("src/lnet/alphabet2d_spectral_gate.py")),
            "product_scan": _digest(Path("src/lnet/alphabet2d.py")),
            "generator": _digest(Path("src/lnet/alphabet2d_synthetic.py")),
        },
    }


def _initialize(root: Path, contract: dict[str, Any]) -> None:
    path = root / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            message = "existing Phase S-A root has a different immutable contract"
            raise RuntimeError(message)
    else:
        _atomic_json(path, contract)


def _feature_batches(
    split: TensorClassificationSplit,
    variant: str,
    config: SpectralGateConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    chunks = []
    with torch.inference_mode():
        for inputs in split.inputs.split(batch_size):
            features = extract_spectral_features(
                inputs.to(device),
                variant,  # pyright: ignore[reportArgumentType]
                config,
            )
            if features.shape[1] < config.modes:
                features = torch.nn.functional.pad(
                    features,
                    (0, config.modes - features.shape[1]),
                )
            if features.shape[1] != config.modes:
                message = f"{variant} produced {features.shape[1]} coordinates"
                raise RuntimeError(message)
            chunks.append(features.cpu())
    return torch.cat(chunks)


def _accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean())


def _empirical_covariance_audit(inputs: Tensor, targets: Tensor, radius: int) -> dict[str, float]:
    maximum_z = 0.0
    maximum_difference = 0.0
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            shifted = torch.roll(inputs, (delta_y, delta_x), dims=(-2, -1))
            values = (inputs * shifted).mean(dim=(-3, -2, -1)).double()
            zero = values[targets == 0]
            one = values[targets == 1]
            difference = float(one.mean() - zero.mean())
            standard_error = torch.sqrt(
                zero.var(unbiased=True) / zero.numel()
                + one.var(unbiased=True) / one.numel()
            ).clamp_min(torch.finfo(torch.float64).eps)
            maximum_z = max(maximum_z, abs(difference) / float(standard_error))
            maximum_difference = max(maximum_difference, abs(difference))
    return {
        "maximum_absolute_mean_difference": maximum_difference,
        "maximum_absolute_welch_z": maximum_z,
    }


def _deterministic_audit(config: OffAxisSpectralConfig) -> dict[str, float]:
    class_zero, class_one, _ = off_axis_spectra(config)
    delta = class_one - class_zero
    covariance = torch.fft.ifft2(delta).real
    residuals = [
        abs(float(covariance[dy % config.height, dx % config.width]))
        for dy in range(-config.matched_lag_radius, config.matched_lag_radius + 1)
        for dx in range(-config.matched_lag_radius, config.matched_lag_radius + 1)
    ]
    return {
        "maximum_low_lag_residual": max(residuals),
        "maximum_x_marginal_residual": float(delta.sum(dim=0).abs().amax()),
        "maximum_y_marginal_residual": float(delta.sum(dim=1).abs().amax()),
        "minimum_psd": float(torch.minimum(class_zero.amin(), class_one.amin())),
    }


def _run_condition(
    root: Path,
    contract: dict[str, Any],
    *,
    epsilon: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    result_path = root / "results" / f"epsilon{epsilon:.1f}__seed{seed}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    base_data = OffAxisSpectralConfig(**contract["data"])
    data_config = replace(base_data, contrast=epsilon)
    feature_config = SpectralGateConfig(**contract["features"])
    counts = contract["counts"]
    splits = make_alphabet2d_splits(
        "off_axis",
        train_per_class=counts["train_per_class"],
        validation_per_class=counts["validation_per_class"],
        test_per_class=counts["test_per_class"],
        seed=seed,
        off_axis_config=data_config,
    )
    variant_results: dict[str, Any] = {}
    for variant in SPECTRAL_GATE_VARIANTS:
        train = _feature_batches(
            splits.train,
            variant,
            feature_config,
            batch_size=contract["batch_size"],
            device=device,
        ).to(device)
        validation = _feature_batches(
            splits.validation,
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
        test = _feature_batches(
            splits.test,
            variant,
            feature_config,
            batch_size=contract["batch_size"],
            device=device,
        ).to(device)
        with torch.no_grad():
            standardized = (test - fit["mean"]) / fit["scale"]
            test_accuracy = _accuracy(
                head(standardized),
                splits.test.targets.to(device),
            )
        variant_results[variant] = {
            "test_balanced_accuracy": test_accuracy,
            "best_validation_accuracy": fit["best_validation_accuracy"],
            "best_epoch": fit["best_epoch"],
            "feature_coordinates": feature_config.modes,
            "head_parameters": 2 * feature_config.modes + 2,
        }
        del train, validation, test, head
        torch.cuda.empty_cache()
    oracle_predictions = (
        off_axis_oracle_score(splits.test.inputs, data_config) > 0
    ).long()
    result = {
        "epsilon": epsilon,
        "seed": seed,
        "runtime": {
            "hostname": platform.node(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
        },
        "deterministic_generator_audit": _deterministic_audit(data_config),
        "empirical_train_covariance_audit": _empirical_covariance_audit(
            splits.train.inputs,
            splits.train.targets,
            data_config.matched_lag_radius,
        ),
        "oracle_test_balanced_accuracy": float(
            (oracle_predictions == splits.test.targets).float().mean()
        ),
        "variants": variant_results,
    }
    _atomic_json(result_path, result)
    return result


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    rows = []
    for epsilon in contract["epsilons"]:
        for seed in contract["seeds"]:
            path = root / "results" / f"epsilon{epsilon:.1f}__seed{seed}.json"
            if not path.exists():
                return None
            rows.append(json.loads(path.read_text()))
    by_epsilon: dict[str, Any] = {}
    for epsilon in contract["epsilons"]:
        selected = [row for row in rows if row["epsilon"] == epsilon]
        means = {
            variant: sum(
                row["variants"][variant]["test_balanced_accuracy"]
                for row in selected
            )
            / len(selected)
            for variant in SPECTRAL_GATE_VARIANTS
        }
        oracle = sum(row["oracle_test_balanced_accuracy"] for row in selected) / len(
            selected
        )
        product_minus_axial = [
            row["variants"]["product_single"]["test_balanced_accuracy"]
            - row["variants"]["axial2d"]["test_balanced_accuracy"]
            for row in selected
        ]
        single_minus_four = [
            row["variants"]["product_single"]["test_balanced_accuracy"]
            - row["variants"]["product_four"]["test_balanced_accuracy"]
            for row in selected
        ]
        oracle_fraction = (
            (means["product_single"] - 0.5) / (oracle - 0.5)
            if oracle > 0.5
            else math.nan
        )
        by_epsilon[str(epsilon)] = {
            "mean_balanced_accuracy": means,
            "oracle_mean_balanced_accuracy": oracle,
            "paired_product_minus_axial": product_minus_axial,
            "mean_product_minus_axial_pp": 100.0
            * sum(product_minus_axial)
            / len(product_minus_axial),
            "paired_single_minus_four": single_minus_four,
            "mean_single_minus_four_pp": 100.0
            * sum(single_minus_four)
            / len(single_minus_four),
            "chance_corrected_oracle_fraction": oracle_fraction,
        }
    final = not contract["pilot"]
    gates = {
        "A1_raw_covariance_within_2pp": all(
            abs(values["mean_balanced_accuracy"]["local_covariance"] - 0.5)
            <= 0.02
            for values in by_epsilon.values()
        ),
        "A2_product_beats_axial_5pp_epsilon_ge_0_4": all(
            values["mean_product_minus_axial_pp"] >= 5.0
            for epsilon, values in by_epsilon.items()
            if float(epsilon) >= 0.4
        ),
        "A3_product_reaches_90pct_chance_corrected_oracle_at_0_8": (
            by_epsilon.get("0.8", {}).get(
                "chance_corrected_oracle_fraction",
                math.nan,
            )
            >= 0.9
            if final
            else None
        ),
        "A4_single_within_2pp_of_four": all(
            values["mean_single_minus_four_pp"] >= -2.0
            for values in by_epsilon.values()
        ),
    }
    summary = {
        "schema": contract["schema"],
        "evidence_status": contract["evidence_status"],
        "conditions": len(rows),
        "by_epsilon": by_epsilon,
        "gates": gates,
        "kill_joint_2d_hypothesis": final
        and not gates["A2_product_beats_axial_5pp_epsilon_ge_0_4"],
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def main() -> None:
    args = _parse_args()
    contract = _contract(pilot=args.pilot, batch_size=args.batch_size)
    args.root.mkdir(parents=True, exist_ok=True)
    _initialize(args.root, contract)
    if args.initialize_only:
        return
    allowed_seeds = set(contract["seeds"])
    allowed_epsilons = set(contract["epsilons"])
    if not set(args.run_seeds) <= allowed_seeds:
        message = "--run-seeds contains values outside the immutable contract"
        raise ValueError(message)
    if not set(args.epsilons) <= allowed_epsilons:
        message = "--epsilons contains values outside the immutable contract"
        raise ValueError(message)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        message = "Phase S-A runner requires a CUDA device"
        raise RuntimeError(message)
    for epsilon in args.epsilons:
        for seed in args.run_seeds:
            _run_condition(
                args.root,
                contract,
                epsilon=epsilon,
                seed=seed,
                device=device,
            )
    summary = _summarize(args.root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
