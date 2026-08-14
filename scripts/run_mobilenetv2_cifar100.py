# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Train a CIFAR-adapted MobileNetV2-0.25x with the matched CIFAR-100 recipe."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torchvision.models import MobileNetV2, mobilenet_v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_cifar100_nano as harness
import cifar100_packed_data as packed_data

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

VARIANTS = ("mobilenetv2_025x",)
SEEDS = (401,)


@dataclass(frozen=True, slots=True)
class MobileNetV2CifarConfig:
    output_dim: int = 100
    width_mult: float = 0.25
    first_stride: int = 1


def _build(variant: str, config: MobileNetV2CifarConfig) -> MobileNetV2:
    if variant != VARIANTS[0]:
        message = f"unknown MobileNetV2 variant: {variant}"
        raise ValueError(message)
    model = mobilenet_v2(width_mult=config.width_mult, num_classes=config.output_dim)
    first_block = cast("nn.Sequential", model.features[0])
    first_conv = cast("nn.Conv2d", first_block[0])
    first_conv.stride = (config.first_stride, config.first_stride)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    config = MobileNetV2CifarConfig()
    model = _build(VARIANTS[0], config)
    payload = {
        "schema": "lnet.mobilenetv2_025x.cifar100.matched.v1",
        "evidence_status": "single-seed matched external baseline",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": {VARIANTS[0]: sum(p.numel() for p in model.parameters())},
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": 3.0e-3,
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
            "shared_runner": harness._digest(Path("scripts/run_polepyramid_a_tiny_cifar100.py")),
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
        "test": row["test"],
        "best_epoch": row["best_epoch"],
        "best_validation_accuracy": row["best_validation_accuracy"],
        "parameters": row["parameters"],
        "training_seconds": row["training_seconds"],
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def _build_optimizer(
    model: torch.nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
    )


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.CifarNanoConfig = MobileNetV2CifarConfig
    harness.build_cifar_nano = _build
    harness._contract = _contract
    harness._build_optimizer = _build_optimizer
    harness._loaders = packed_data.build_loaders
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
