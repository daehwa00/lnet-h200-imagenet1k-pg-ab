# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Train the matched Early-A/Static-T PolePyramid with TCIR on CIFAR-100."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cifar100_packed_data as packed_data
import run_alphabet2d_cifar100_nano as harness
import run_polepyramid_a_tiny_cifar100 as base

from lnet.polepyramid_a_tiny import (
    PolePyramidATerminalTiny,
    PolePyramidATerminalTinyConfig,
)

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("earlya_statict_stopgrad_tcir_affine",)
SEEDS = (401,)


def _config() -> PolePyramidATerminalTinyConfig:
    return PolePyramidATerminalTinyConfig(
        stop_gradient_gain_normalization=True,
        tcir_innovation_reweighting=True,
        tcir_radius=0.5,
    )


def _build(
    variant: str,
    config: PolePyramidATerminalTinyConfig,
) -> PolePyramidATerminalTiny:
    if variant != VARIANTS[0]:
        message = f"unknown TCIR PolePyramid variant: {variant}"
        raise ValueError(message)
    return PolePyramidATerminalTiny(config)


def _contract(args: Namespace) -> dict[str, Any]:
    config = _config()
    model = _build(VARIANTS[0], config)
    payload = {
        "schema": "lnet.polepyramid.earlya_statict.tcir.cifar100.v1",
        "evidence_status": "single-seed matched TCIR architecture screen",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": {
            VARIANTS[0]: sum(parameter.numel() for parameter in model.parameters())
        },
        "comparison_contract": {
            "reference": "earlya_statict_stopgrad_raw_affine",
            "changed_factor": "mode-wise transport-consistent innovation reweighting",
            "initial_function": "exactly M + J because every TCIR multiplier starts at one",
            "success_gate": (
                ">= +0.5 percentage points validation/test versus matched seed 401 reference"
            ),
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
            "shared_runner": harness._digest(
                Path("scripts/run_polepyramid_a_tiny_cifar100.py")
            ),
            "model": harness._digest(Path("src/lnet/polepyramid_a_tiny.py")),
        },
    }
    return json.loads(json.dumps(payload))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    result_path = root / "results" / f"{VARIANTS[0]}__seed{SEEDS[0]}.json"
    if not result_path.exists():
        return None
    row = json.loads(result_path.read_text())
    payload = {
        "schema": contract["schema"],
        "parameters": row["parameters"],
        "best_epoch": row["best_epoch"],
        "best_validation_accuracy": row["best_validation_accuracy"],
        "test": row["test"],
        "training_seconds": row["training_seconds"],
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.CifarNanoConfig = PolePyramidATerminalTinyConfig
    harness.build_cifar_nano = _build
    harness._contract = _contract
    harness._build_optimizer = base._build_optimizer
    harness._loaders = packed_data.build_loaders
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
