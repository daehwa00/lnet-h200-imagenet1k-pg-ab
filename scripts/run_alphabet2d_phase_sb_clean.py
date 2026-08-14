#!/usr/bin/env python3
# pyright: reportExplicitAny=false
"""Run the clean circular-energy/modulus-cascade ALPHABET-2D S-B gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from lnet.alphabet2d_shape_gate import (
    HomometricTextonConfig,
    PairAuditThresholds,
    PairedInputAudit,
    ShapeGateConfig,
    TextonArrangementConfig,
    audit_paired_inputs,
    circular_energy_features,
    linear_cascade_features,
    make_homometric_texton_arrangement_pairs,
    make_texton_arrangement_dataset,
    paired_audit_is_valid,
    shared_path_global_features,
    shared_path_window_features,
)
from lnet.alphabet2d_spectral_gate import SpectralGateConfig, fit_affine_head
from lnet.alphabet2d_synthetic import (
    EqualPowerPhaseConfig,
    TensorClassificationSplit,
    make_equal_power_phase_dataset,
)

Task = Literal[
    "equal_power_phase",
    "texton_location_control",
    "homometric_texton_arrangement",
]
Variant = Literal[
    "energy",
    "linear_cascade",
    "modulus_global",
    "modulus_window",
    "histogram",
]
TASKS: tuple[Task, ...] = (
    "equal_power_phase",
    "texton_location_control",
    "homometric_texton_arrangement",
)
VARIANTS: tuple[Variant, ...] = (
    "energy",
    "linear_cascade",
    "modulus_global",
    "modulus_window",
    "histogram",
)
DEFAULT_V3_SEEDS = (401, 419, 443, 467, 491)
HEAD_WIDTH = 20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--contract-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_V3_SEEDS),
    )
    parser.add_argument("--run-seeds", type=int, nargs="+")
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _contract(batch_size: int, seeds: tuple[int, ...]) -> dict[str, Any]:
    if len(set(seeds)) != len(seeds) or not seeds:
        message = "v3 contract seeds must be nonempty and unique"
        raise ValueError(message)
    thresholds = PairAuditThresholds()
    return {
        "schema": "lnet.alphabet2d.phase_sb.clean.v3",
        "evidence_status": "confirmatory homometric B-2 with diagnostic location control",
        "seeds": list(seeds),
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "counts": {
            "train_per_class": 2_000,
            "validation_per_class": 500,
            "test_per_class": 2_000,
        },
        "batch_size": batch_size,
        "head_width": HEAD_WIDTH,
        "shape": asdict(ShapeGateConfig(modes=25)),
        "equal_power": asdict(EqualPowerPhaseConfig(height=64, width=64)),
        "texton": asdict(TextonArrangementConfig()),
        "homometric_texton": asdict(HomometricTextonConfig()),
        "pair_audit_thresholds": asdict(thresholds),
        "gate_thresholds": {
            "chance_absolute_margin": 0.02,
            "recovery_margin": 0.10,
        },
        "shared_path_contract": {
            "path_count": HEAD_WIDTH,
            "base_statistic": "modulus_cascade_abs_squared",
            "global_coordinates_per_path": 1,
            "window_coordinates_per_path": 1,
            "window_basis": "2x2_dct_cycle_dc_h_v_checkerboard",
        },
        "head": asdict(
            SpectralGateConfig(
                modes=20,
                head_epochs=300,
                head_patience=30,
                head_learning_rate=3.0e-2,
            )
        ),
        "selection": "validation loss; test evaluated once after checkpoint selection",
        "source_sha256": {
            "runner": _digest(Path(__file__)),
            "shape_gate": _digest(Path("src/lnet/alphabet2d_shape_gate.py")),
            "equal_power_generator": _digest(
                Path("src/lnet/alphabet2d_synthetic.py")
            ),
            "head": _digest(Path("src/lnet/alphabet2d_spectral_gate.py")),
        },
    }


def _initialize(root: Path, contract: dict[str, Any]) -> None:
    path = root / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            message = "existing clean S-B root has a different immutable contract"
            raise RuntimeError(message)
    else:
        _atomic_json(path, contract)


def _shuffle_pairs(pairs: Tensor, seed: int) -> TensorClassificationSplit:
    count = pairs.shape[0]
    inputs = pairs.permute(1, 0, 2, 3, 4).reshape(2 * count, *pairs.shape[2:])
    targets = torch.arange(2, dtype=torch.long).repeat_interleave(count)
    generator = torch.Generator(device="cpu").manual_seed(seed + 97_003)
    order = torch.randperm(targets.numel(), generator=generator)
    return TensorClassificationSplit(inputs[order], targets[order])


def _generate(
    task: Task,
    count: int,
    seed: int,
    contract: dict[str, Any],
) -> tuple[TensorClassificationSplit, dict[str, float] | None]:
    if task == "equal_power_phase":
        inputs, targets = make_equal_power_phase_dataset(
            count,
            seed=seed,
            config=EqualPowerPhaseConfig(**contract["equal_power"]),
        )
        return TensorClassificationSplit(inputs, targets), None
    if task == "texton_location_control":
        inputs, targets = make_texton_arrangement_dataset(
            count,
            seed=seed,
            config=TextonArrangementConfig(**contract["texton"]),
        )
        return TensorClassificationSplit(inputs, targets), None
    pairs = make_homometric_texton_arrangement_pairs(
        count,
        seed=seed,
        config=HomometricTextonConfig(**contract["homometric_texton"]),
    )
    audit = audit_paired_inputs(pairs)
    return _shuffle_pairs(pairs, seed), asdict(audit)


def _histogram_features(inputs: Tensor) -> Tensor:
    flattened = inputs.flatten(1)
    quantiles = torch.linspace(
        0.025,
        0.975,
        HEAD_WIDTH,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    return torch.quantile(flattened, quantiles, dim=1).T


def _variant_features(
    inputs: Tensor,
    variant: Variant,
    config: ShapeGateConfig,
) -> Tensor:
    if variant == "energy":
        return circular_energy_features(inputs, config)[:, :HEAD_WIDTH]
    if variant == "linear_cascade":
        return linear_cascade_features(inputs, config)[:, :HEAD_WIDTH]
    if variant == "modulus_global":
        return shared_path_global_features(
            inputs,
            config,
            path_count=HEAD_WIDTH,
        )
    if variant == "modulus_window":
        return shared_path_window_features(
            inputs,
            config,
            path_count=HEAD_WIDTH,
        )
    return _histogram_features(inputs)


def _extract(
    split: TensorClassificationSplit,
    variant: Variant,
    config: ShapeGateConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    chunks = []
    with torch.inference_mode():
        chunks.extend(
            [
                _variant_features(batch.to(device), variant, config).cpu()
                for batch in split.inputs.split(batch_size)
            ]
        )
    features = torch.cat(chunks)
    if features.shape[1] != HEAD_WIDTH:
        message = f"{variant} violates the {HEAD_WIDTH}-coordinate contract"
        raise RuntimeError(message)
    return features


def _paired_power_residual(split: TensorClassificationSplit) -> float:
    power = torch.fft.fft2(split.inputs.double(), norm="ortho").abs().square()
    zero = power[split.targets == 0].sort(dim=0).values
    one = power[split.targets == 1].sort(dim=0).values
    denominator = torch.maximum(zero.abs(), one.abs()).clamp_min(
        torch.finfo(torch.float64).eps
    )
    return float(((zero - one).abs() / denominator).amax())


def _run_job(
    root: Path,
    contract: dict[str, Any],
    task: Task,
    seed: int,
    device: torch.device,
) -> None:
    path = root / "results" / f"{task}__seed{seed}.json"
    if path.exists():
        return
    counts = contract["counts"]
    generated = {
        "train": _generate(task, counts["train_per_class"], seed, contract),
        "validation": _generate(
            task,
            counts["validation_per_class"],
            seed + 10_007,
            contract,
        ),
        "test": _generate(
            task,
            counts["test_per_class"],
            seed + 20_011,
            contract,
        ),
    }
    splits = {name: value[0] for name, value in generated.items()}
    pair_audits = {
        name: value[1] for name, value in generated.items() if value[1] is not None
    }
    thresholds = PairAuditThresholds(**contract["pair_audit_thresholds"])
    input_valid = all(
        paired_audit_is_valid(
            PairedInputAudit(**audit_payload),
            thresholds,
        )
        for audit_payload in pair_audits.values()
    )
    shape_config = ShapeGateConfig(**contract["shape"])
    head_config = SpectralGateConfig(**contract["head"])
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        train = _extract(
            splits["train"],
            variant,
            shape_config,
            batch_size=contract["batch_size"],
            device=device,
        ).to(device)
        validation = _extract(
            splits["validation"],
            variant,
            shape_config,
            batch_size=contract["batch_size"],
            device=device,
        ).to(device)
        head, fit = fit_affine_head(
            train,
            splits["train"].targets.to(device),
            validation,
            splits["validation"].targets.to(device),
            seed=seed,
            config=head_config,
        )
        test = _extract(
            splits["test"],
            variant,
            shape_config,
            batch_size=contract["batch_size"],
            device=device,
        ).to(device)
        with torch.no_grad():
            standardized = (test - fit["mean"]) / fit["scale"]
            accuracy = float(
                (
                    head(standardized).argmax(dim=-1)
                    == splits["test"].targets.to(device)
                )
                .float()
                .mean()
            )
        variants[variant] = {
            "test_balanced_accuracy": accuracy,
            "best_validation_accuracy": fit["best_validation_accuracy"],
            "best_epoch": fit["best_epoch"],
            "coordinates": HEAD_WIDTH,
            "head_parameters": 2 * HEAD_WIDTH + 2,
        }
        del train, validation, test, head
        torch.cuda.empty_cache()
    result = {
        "task": task,
        "seed": seed,
        "runtime": {
            "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
        },
        "test_pair_maximum_relative_power_residual": _paired_power_residual(
            splits["test"]
        ),
        "paired_input_audits_before_shuffle": pair_audits,
        "input_valid": input_valid,
        "variants": variants,
    }
    _atomic_json(path, result)


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [
        root / "results" / f"{task}__seed{seed}.json"
        for task in TASKS
        for seed in contract["seeds"]
    ]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    groups: dict[str, Any] = {}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        means = {
            variant: sum(
                row["variants"][variant]["test_balanced_accuracy"]
                for row in selected
            )
            / len(selected)
            for variant in VARIANTS
        }
        groups[task] = {
            "mean_balanced_accuracy": means,
            "paired_modulus_minus_energy": [
                row["variants"]["modulus_global"]["test_balanced_accuracy"]
                - row["variants"]["energy"]["test_balanced_accuracy"]
                for row in selected
            ],
            "paired_window_minus_global": [
                row["variants"]["modulus_window"]["test_balanced_accuracy"]
                - row["variants"]["modulus_global"]["test_balanced_accuracy"]
                for row in selected
            ],
            "maximum_pair_power_residual": max(
                row["test_pair_maximum_relative_power_residual"]
                for row in selected
            ),
            "all_inputs_valid": all(row["input_valid"] for row in selected),
        }
    b1 = groups["equal_power_phase"]
    location = groups["texton_location_control"]
    b2 = groups["homometric_texton_arrangement"]
    chance_margin = contract["gate_thresholds"]["chance_absolute_margin"]
    recovery_margin = contract["gate_thresholds"]["recovery_margin"]
    gates = {
        "B1_energy_chance_equal_power": abs(
            b1["mean_balanced_accuracy"]["energy"] - 0.5
        )
        <= chance_margin,
        "B1_linear_cascade_chance": abs(
            b1["mean_balanced_accuracy"]["linear_cascade"] - 0.5
        )
        <= chance_margin,
        "B2_modulus_recovers_10pp": (
            sum(b1["paired_modulus_minus_energy"])
            / len(b1["paired_modulus_minus_energy"])
            >= recovery_margin
        ),
        "B2_texton_energy_chance": abs(
            b2["mean_balanced_accuracy"]["energy"] - 0.5
        )
        <= chance_margin,
        "B2_texton_histogram_chance": abs(
            b2["mean_balanced_accuracy"]["histogram"] - 0.5
        )
        <= chance_margin,
        "B3_window_recovers_10pp": (
            sum(b2["paired_window_minus_global"])
            / len(b2["paired_window_minus_global"])
            >= recovery_margin
        ),
        "B3_homometric_inputs_valid": b2["all_inputs_valid"],
    }
    payload = {
        "schema": contract["schema"],
        "groups": groups,
        "gates": gates,
        "decision": {
            "shape_hypothesis": (
                "pass"
                if gates["B1_energy_chance_equal_power"]
                and gates["B1_linear_cascade_chance"]
                and gates["B2_modulus_recovers_10pp"]
                and gates["B2_texton_energy_chance"]
                and gates["B2_texton_histogram_chance"]
                and gates["B3_window_recovers_10pp"]
                and gates["B3_homometric_inputs_valid"]
                else "fail_or_invalid"
            )
        },
        "diagnostic_only": {
            "texton_location_control": location,
        },
    }
    _atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    args = _parse_args()
    contract_seeds = tuple(args.contract_seeds)
    contract = _contract(args.batch_size, contract_seeds)
    args.root.mkdir(parents=True, exist_ok=True)
    _initialize(args.root, contract)
    if args.initialize_only:
        return
    run_seeds = tuple(args.run_seeds) if args.run_seeds else contract_seeds
    if not set(run_seeds) <= set(contract_seeds):
        message = "run seeds fall outside the clean S-B contract"
        raise ValueError(message)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        message = "clean S-B requires CUDA"
        raise RuntimeError(message)
    for task in args.tasks:
        for seed in run_seeds:
            _run_job(args.root, contract, task, seed, device)
    summary = _summarize(args.root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
