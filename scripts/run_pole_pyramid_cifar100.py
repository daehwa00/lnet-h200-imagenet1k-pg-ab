# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the PolePyramid versus average-pyramid CIFAR-100 gate."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_cifar100_nano as harness

from lnet.pole_pyramid import PolePyramid, PolePyramidConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("pole_pyramid", "average_pyramid")
SEEDS = (401, 409, 419)


def _build(variant: str, config: PolePyramidConfig) -> PolePyramid:
    if variant == "pole_pyramid":
        return PolePyramid(replace(config, transport="pole"))
    if variant == "average_pyramid":
        return PolePyramid(replace(config, transport="average"))
    message = f"unknown PolePyramid variant: {variant}"
    raise ValueError(message)


def _contract(args: Namespace) -> dict[str, Any]:
    config = PolePyramidConfig()
    payload = {
        "schema": "lnet.pole_pyramid.cifar100.endpoint_gate.v1",
        "evidence_status": ("exploratory convolution-free exact-coarsening versus average control"),
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": {
            variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
            for variant in VARIANTS
        },
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": 3.0e-3,
            "modal_learning_rate_multiplier": 1.0 / 3.0,
            "pole_geometry_learning_rate_multiplier": 0.1,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "warmup plus cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "augmentation": ("RandomCrop(32,pad4)+HFlip+RandAugment(2,9)+RandomErasing"),
            "validation": "fixed stratified 5k from CIFAR-100 train",
            "test_selection": False,
            "gate": "mean pole minus average test accuracy >= 1pp",
        },
        "data_sha256": {
            name: harness._digest(args.data_root / "cifar-100-python" / name)
            for name in ("train", "test", "meta")
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_cifar100_nano.py")),
            "model": harness._digest(Path("src/lnet/pole_pyramid.py")),
            "scan": harness._digest(Path("src/lnet/alphabet2d.py")),
        },
    }
    return json.loads(json.dumps(payload))


def _build_optimizer(
    model: torch.nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    modal = []
    geometry = []
    geometry_tokens = (
        "log_damping_offset",
        "frequency_offset",
    )
    modal_tokens = (
        "analysis",
        "direction_gain",
        "point_mix",
    )
    for name, parameter in model.named_parameters():
        if any(token in name for token in geometry_tokens):
            geometry.append(parameter)
        elif any(token in name for token in modal_tokens):
            modal.append(parameter)
        elif parameter.ndim < 2 or "norm" in name or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    learning_rate = recipe["learning_rate"]
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "lr": learning_rate,
                "weight_decay": recipe["weight_decay"],
            },
            {
                "params": no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": modal,
                "lr": learning_rate * recipe["modal_learning_rate_multiplier"],
                "weight_decay": 0.0,
            },
            {
                "params": geometry,
                "lr": learning_rate * recipe["pole_geometry_learning_rate_multiplier"],
                "weight_decay": 0.0,
            },
        ]
    )


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [
        root / "results" / f"{variant}__seed{seed}.json" for variant in VARIANTS for seed in SEEDS
    ]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    means = {
        variant: sum(row["test"]["accuracy"] for row in rows if row["variant"] == variant)
        / len(SEEDS)
        for variant in VARIANTS
    }
    paired = [
        next(
            row["test"]["accuracy"]
            for row in rows
            if row["variant"] == "pole_pyramid" and row["seed"] == seed
        )
        - next(
            row["test"]["accuracy"]
            for row in rows
            if row["variant"] == "average_pyramid" and row["seed"] == seed
        )
        for seed in SEEDS
    ]
    mean_delta = sum(paired) / len(paired)
    payload = {
        "schema": contract["schema"],
        "mean_test_accuracy": means,
        "paired_pole_minus_average": paired,
        "mean_pole_minus_average_pp": 100.0 * mean_delta,
        "gate_pass": mean_delta >= 0.01,
        "decision": (
            "promote PolePyramid to ImageNet-100"
            if mean_delta >= 0.01
            else "hold PolePyramid at synthetic/CIFAR stage"
        ),
        "parameter_counts": {
            variant: sorted({row["parameters"] for row in rows if row["variant"] == variant})
            for variant in VARIANTS
        },
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.CifarNanoConfig = PolePyramidConfig
    harness.build_cifar_nano = _build
    harness._contract = _contract
    harness._build_optimizer = _build_optimizer
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
