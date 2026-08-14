# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the deep no-excitation-gate PolePyramid-C-Tiny ImageNet-100 screen."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_imagenet100_nano as harness

from lnet.polepyramid_c_deep_tiny import PolePyramidCDeepTiny, PolePyramidCDeepTinyConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("polepyramid_c_deep_tiny",)
SEEDS = (501,)


def _build(variant: str, config: PolePyramidCDeepTinyConfig) -> PolePyramidCDeepTiny:
    if variant != "polepyramid_c_deep_tiny":
        message = f"unknown deep PolePyramid-C-Tiny variant: {variant}"
        raise ValueError(message)
    return PolePyramidCDeepTiny(config)


def _contract(args: Namespace) -> dict[str, Any]:
    config = PolePyramidCDeepTinyConfig()
    data_digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    payload = {
        "schema": "lnet.polepyramid_c_deep_tiny.imagenet100.screening.v1",
        "evidence_status": "exploratory single-seed architecture screen",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": {
            "polepyramid_c_deep_tiny": sum(
                parameter.numel() for parameter in PolePyramidCDeepTiny().parameters()
            )
        },
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": 3.0e-4,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "warmup plus cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "precision": args.precision,
            "augmentation": "RandomResizedCrop(224)+HFlip+RandAugment(2,9)+RandomErasing",
            "selection": "fixed final epoch with full learning-curve inspection",
        },
        "data": {
            "manifest_sha256": data_digest,
            "train_images": train_count,
            "validation_images": validation_count,
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_imagenet100_nano.py")),
            "model": harness._digest(Path("src/lnet/polepyramid_c_deep_tiny.py")),
            "bank": harness._digest(Path("src/lnet/alphabet2d_tiny.py")),
            "moments": harness._digest(Path("src/lnet/polepyramid_c_tiny.py")),
        },
    }
    return json.loads(json.dumps(payload))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    path = root / "results" / "polepyramid_c_deep_tiny__seed501.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    payload = {
        "schema": contract["schema"],
        "final_validation": row["final_validation"],
        "parameters": row["parameters"],
        "training_examples_per_second": row["training_examples_per_second"],
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.ImageNetNanoConfig = PolePyramidCDeepTinyConfig  # type: ignore[attr-defined]
    harness.build_imagenet_nano = _build  # type: ignore[attr-defined]
    harness._contract = _contract
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
