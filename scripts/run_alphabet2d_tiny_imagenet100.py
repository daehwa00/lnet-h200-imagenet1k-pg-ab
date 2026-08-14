# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# pyright: reportPrivateLocalImportUsage=false
"""Run the exploratory single-seed ALPHABET-2D-Tiny ImageNet-100 screen."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_imagenet100_nano as harness

from lnet.alphabet2d_tiny import Alphabet2DTiny, Alphabet2DTinyConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("alphabet2d_tiny",)
SEEDS = (501,)


def _build(variant: str, config: Alphabet2DTinyConfig) -> Alphabet2DTiny:
    if variant != "alphabet2d_tiny":
        message = f"unknown ALPHABET-2D-Tiny variant: {variant}"
        raise ValueError(message)
    return Alphabet2DTiny(config)


def _contract(args: Namespace) -> dict[str, Any]:
    config = Alphabet2DTinyConfig()
    data_digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    payload = {
        "schema": "lnet.alphabet2d_tiny.imagenet100.screening.v1",
        "evidence_status": "exploratory single-seed architecture screen",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": {
            "alphabet2d_tiny": sum(parameter.numel() for parameter in Alphabet2DTiny().parameters())
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
            "model": harness._digest(Path("src/lnet/alphabet2d_tiny.py")),
            "scan": harness._digest(Path("src/lnet/alphabet2d.py")),
        },
    }
    return json.loads(json.dumps(payload))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    path = root / "results" / "alphabet2d_tiny__seed501.json"
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
    harness.ImageNetNanoConfig = Alphabet2DTinyConfig  # type: ignore[attr-defined]
    harness.build_imagenet_nano = _build  # type: ignore[attr-defined]
    harness._contract = _contract
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
