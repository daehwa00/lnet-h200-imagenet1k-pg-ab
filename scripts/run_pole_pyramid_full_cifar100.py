# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the full PolePyramid versus average-pyramid CIFAR-100 gate."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_cifar100_nano as harness

from lnet.pole_pyramid_full import FullPolePyramid, FullPolePyramidConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("full_pole_pyramid", "full_average_pyramid")
SEEDS = (401, 409, 419)


def _build(variant: str, config: FullPolePyramidConfig) -> FullPolePyramid:
    if variant == "full_pole_pyramid":
        return FullPolePyramid(replace(config, transport="pole"))
    if variant == "full_average_pyramid":
        return FullPolePyramid(replace(config, transport="average"))
    message = f"unknown full PolePyramid variant: {variant}"
    raise ValueError(message)


def _contract(args: Namespace) -> dict[str, Any]:
    config = FullPolePyramidConfig()
    payload = {
        "schema": "lnet.pole_pyramid.cifar100.full_fine_detail_gate.v1",
        "evidence_status": (
            "confirmatory full fine-detail shared-physical-pole versus average control"
        ),
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
            "augmentation": "RandomCrop(32,pad4)+HFlip+RandAugment(2,9)+RandomErasing",
            "gradient_clip": 1.0,
            "descriptor_standardization": "none",
            "validation": "fixed stratified 5k from CIFAR-100 train",
            "test_selection": False,
            "gate": "mean full pole minus full average test accuracy >= 1pp",
        },
        "data_sha256": {
            name: harness._digest(args.data_root / "cifar-100-python" / name)
            for name in ("train", "test", "meta")
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_cifar100_nano.py")),
            "model": harness._digest(Path("src/lnet/pole_pyramid_full.py")),
            "scan": harness._digest(Path("src/lnet/alphabet2d.py")),
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
        variant: sum(row["test"]["accuracy"] for row in rows if row["variant"] == variant)
        / len(SEEDS)
        for variant in VARIANTS
    }
    paired = [
        next(
            row["test"]["accuracy"]
            for row in rows
            if row["variant"] == "full_pole_pyramid" and row["seed"] == seed
        )
        - next(
            row["test"]["accuracy"]
            for row in rows
            if row["variant"] == "full_average_pyramid" and row["seed"] == seed
        )
        for seed in SEEDS
    ]
    mean_delta = sum(paired) / len(paired)
    payload = {
        "schema": contract["schema"],
        "mean_test_accuracy": means,
        "paired_full_pole_minus_average": paired,
        "mean_full_pole_minus_average_pp": 100.0 * mean_delta,
        "gate_pass": mean_delta >= 0.01,
        "decision": (
            "promote full PolePyramid to ImageNet-100"
            if mean_delta >= 0.01
            else "retain as a negative/diagnostic architectural result"
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
    harness.CifarNanoConfig = FullPolePyramidConfig
    harness.build_cifar_nano = _build
    harness._contract = _contract
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
