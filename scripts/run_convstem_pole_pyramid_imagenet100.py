# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run ConvStem PolePyramid and its matched average control on ImageNet-100."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_imagenet100_nano as harness

from lnet.convstem_pole_pyramid import ConvStemPolePyramid, ConvStemPolePyramidConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("convstem_pole", "convstem_average")
SEEDS = (501, 509, 521)


def _build(variant: str, config: ConvStemPolePyramidConfig) -> ConvStemPolePyramid:
    if variant == "convstem_pole":
        return ConvStemPolePyramid(replace(config, transport="pole"))
    if variant == "convstem_average":
        return ConvStemPolePyramid(replace(config, transport="average"))
    message = f"unknown ConvStem PolePyramid variant: {variant}"
    raise ValueError(message)


def _contract(args: Namespace) -> dict[str, Any]:
    config = ConvStemPolePyramidConfig()
    data_digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    payload = {
        "schema": "lnet.convstem_pole_pyramid.imagenet100.v2",
        "evidence_status": "confirmatory matched residual-pole/PoleDown/QRC comparison",
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
            "learning_rate": 3.0e-4,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "warmup plus cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "precision": args.precision,
            "augmentation": "RandomResizedCrop(224)+HFlip+RandAugment(2,9)+RandomErasing",
            "selection": "fixed final epoch; validation is not used for selection",
            "gate": "mean pole minus average final validation accuracy >= 1pp",
        },
        "data": {
            "manifest_sha256": data_digest,
            "train_images": train_count,
            "validation_images": validation_count,
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_imagenet100_nano.py")),
            "model": harness._digest(Path("src/lnet/convstem_pole_pyramid.py")),
            "pole_blocks": harness._digest(Path("src/lnet/pole_pyramid_full.py")),
            "qrc_readout": harness._digest(Path("src/lnet/spatialalphabet_h.py")),
        },
    }
    return json.loads(json.dumps(payload))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [
        root / "results" / f"{variant}__seed{seed}.json"
        for variant in VARIANTS
        for seed in SEEDS
    ]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    means = {
        variant: sum(
            row["final_validation"]["accuracy"] for row in rows if row["variant"] == variant
        )
        / len(SEEDS)
        for variant in VARIANTS
    }
    paired = [
        next(
            row["final_validation"]["accuracy"]
            for row in rows
            if row["variant"] == "convstem_pole" and row["seed"] == seed
        )
        - next(
            row["final_validation"]["accuracy"]
            for row in rows
            if row["variant"] == "convstem_average" and row["seed"] == seed
        )
        for seed in SEEDS
    ]
    delta = sum(paired) / len(paired)
    payload = {
        "schema": contract["schema"],
        "mean_final_validation_accuracy": means,
        "paired_pole_minus_average": paired,
        "mean_pole_minus_average_pp": 100.0 * delta,
        "gate_pass": delta >= 0.01,
        "parameter_counts": contract["parameter_counts"],
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.ImageNetNanoConfig = ConvStemPolePyramidConfig
    harness.build_imagenet_nano = _build
    harness._contract = _contract
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
