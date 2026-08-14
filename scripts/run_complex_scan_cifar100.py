# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the continuous complex scan CIFAR-100 validation screen."""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cifar100_packed_data as packed_data
import run_alphabet2d_cifar100_nano as harness

from lnet.complex_scan import (
    ComplexScanBackbone,
    ComplexScanConfig,
)

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("complex_linear",)
SEEDS = (401, 402, 403)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANTS[0]:
        message = f"unknown continuous complex scan variant: {variant}"
        raise ValueError(message)
    return ComplexScanBackbone(config)


def _contract(args: Namespace) -> dict[str, Any]:
    config = ComplexScanConfig()
    models = {variant: _build(variant, config) for variant in VARIANTS}
    payload = {
        "schema": "lnet.complex_scan.backbone.cifar100.v1",
        "evidence_status": (
            "validation-only 50-epoch architecture screen"
            if args.skip_test and args.epochs == 50
            else "follow-up evaluation"
        ),
        "official_test_evaluation": not args.skip_test,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "variant_configs": {"complex_linear": asdict(config)},
        "parameter_counts": {
            variant: {
                "total": sum(parameter.numel() for parameter in model.parameters()),
                "trainable": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
            }
            for variant, model in models.items()
        },
        "architecture_invariants": {
            "intermediate_state": "complex only",
            "direction_mixing": "raw directional states then bias-free complex linear",
            "normalization": "phase-equivariant ComplexRMSNorm",
            "real_synthesis": False,
            "real_direction_mixer": False,
            "pointwise_real_mlp": False,
            "terminal": "static product-pole energy analyzer",
            "descriptor_dim": 288,
            "head": "LRQ16",
        },
        "selection_protocol": {
            "architecture_selection": "validation only",
            "screening_epochs": 50,
            "late_validation_window": [40, 50],
            "primary_contrast": "continuous complex backbone",
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
            "loader": harness._digest(Path("scripts/cifar100_packed_data.py")),
            "model": harness._digest(Path("src/lnet/complex_scan.py")),
            "recurrence": harness._digest(Path("src/lnet/pac_triton_recurrence_op.py")),
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
        elif "analysis" in name or "bridge" in name:
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


def _late_validation(history: list[dict[str, float]]) -> dict[str, float | None]:
    window = [row for row in history if 40 <= int(row["epoch"]) <= 50]
    if not window:
        return {"accuracy_mean_epoch40_50": None, "ce_mean_epoch40_50": None}
    return {
        "accuracy_mean_epoch40_50": statistics.fmean(
            row["validation_accuracy"] for row in window
        ),
        "ce_mean_epoch40_50": statistics.fmean(
            row["validation_cross_entropy"] for row in window
        ),
    }


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    result_root = root / "results"
    paths = sorted(result_root.glob("*.json")) if result_root.exists() else []
    if not paths:
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    results = {
        f"{row['variant']}__seed{row['seed']}": {
            "parameters": row["parameters"],
            "best_epoch": row["best_epoch"],
            "best_validation_accuracy": row["best_validation_accuracy"],
            "late_validation": _late_validation(row["history"]),
            "test": row["test"],
            "training_seconds": row["training_seconds"],
            "training_examples_per_second": row["complete_training_examples_per_second"],
        }
        for row in rows
    }
    payload = {
        "schema": contract["schema"],
        "selection_basis": "validation only; official test is sealed during screening",
        "official_test_evaluation": contract["official_test_evaluation"],
        "results": results,
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.CifarNanoConfig = ComplexScanConfig
    harness.build_cifar_nano = _build
    harness._contract = _contract
    harness._build_optimizer = _build_optimizer
    harness._loaders = packed_data.build_loaders
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
