# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportArgumentType=false, reportMissingTypeArgument=false, reportReturnType=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Train static-pole PolePyramid-A-Tiny on CIFAR-100."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cifar100_packed_data import build_loaders
import run_alphabet2d_cifar100_nano as harness

from lnet.polepyramid_a_tiny import PolePyramidATiny, PolePyramidATinyConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("static_pole_pyramid_a",)
SEEDS = (401,)


def _build(variant: str, config: PolePyramidATinyConfig) -> PolePyramidATiny:
    if variant != VARIANTS[0]:
        message = f"unknown PolePyramid-A variant: {variant}"
        raise ValueError(message)
    return PolePyramidATiny(config)


def _contract(args: Namespace) -> dict[str, Any]:
    config = PolePyramidATinyConfig()
    model = _build(VARIANTS[0], config)
    payload = {
        "schema": "lnet.polepyramid_a_tiny.cifar100.static_poles.v2",
        "evidence_status": "single-seed exploratory architecture screen",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": {VARIANTS[0]: sum(p.numel() for p in model.parameters())},
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
            "augmentation": "RandomCrop(32,pad4)+HFlip+RandAugment(2,9)+RandomErasing",
            "validation": "fixed stratified 5k from CIFAR-100 train",
            "test_selection": False,
        },
        "data_sha256": {
            "cifar100_packed.pt": harness._digest(args.data_root / "cifar100_packed.pt")
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_cifar100_nano.py")),
            "model": harness._digest(Path("src/lnet/polepyramid_a_tiny.py")),
            "scan": harness._digest(Path("src/lnet/alphabet2d.py")),
        },
    }
    return json.loads(json.dumps(payload))


def _build_optimizer(
    model: torch.nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    modal: list[torch.nn.Parameter] = []
    geometry: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if "damping_logits" in name or "phase_" in name:
            geometry.append(parameter)
        elif "analysis" in name or "direction_" in name:
            modal.append(parameter)
        elif parameter.ndim < 2 or "norm" in name or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    learning_rate = float(recipe["learning_rate"])
    return torch.optim.AdamW(
        [
            {"params": decay, "lr": learning_rate, "weight_decay": recipe["weight_decay"]},
            {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0},
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
    result_path = root / "results" / f"{VARIANTS[0]}__seed{SEEDS[0]}.json"
    if not result_path.exists():
        return None
    row = json.loads(result_path.read_text())
    payload = {
        "schema": contract["schema"],
        "test": row["test"],
        "best_epoch": row["best_epoch"],
        "best_validation_accuracy": row["best_validation_accuracy"],
        "parameters": row["parameters"],
        "training_seconds": row["training_seconds"],
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.CifarNanoConfig = PolePyramidATinyConfig
    harness.build_cifar_nano = _build
    harness._contract = _contract
    harness._build_optimizer = _build_optimizer
    harness._loaders = build_loaders
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
