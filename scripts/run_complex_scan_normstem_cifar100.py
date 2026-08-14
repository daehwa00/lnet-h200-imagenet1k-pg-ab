# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the normalized, activation-free Complex-Linear CIFAR-100 screen."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cifar100_packed_data as packed_data
import run_alphabet2d_cifar100_nano as harness
import run_complex_scan_cifar100 as complex_run

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("complex_linear_norm_no_gelu_stem",)
SEEDS = (401, 402, 403)


def _config() -> ComplexScanConfig:
    return ComplexScanConfig(stem="normalized_no_activation")


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANTS[0]:
        message = f"unknown normalized activation-free stem variant: {variant}"
        raise ValueError(message)
    return ComplexScanBackbone(config)


def _contract(args: Namespace) -> dict[str, Any]:
    config = _config()
    model = _build(VARIANTS[0], config)
    payload = {
        "schema": "lnet.complex_scan.norm_no_gelu_stem.cifar100.v1",
        "evidence_status": "validation-only 50-epoch stem ablation",
        "official_test_evaluation": not args.skip_test,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": {
            VARIANTS[0]: {
                "total": sum(parameter.numel() for parameter in model.parameters()),
                "trainable": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            }
        },
        "ablation": {
            "reference": "complex_linear seed401 50-epoch best validation 48.50%",
            "changed": "stem activation only",
            "stem": "Conv3x3 -> LayerNorm2d -> Conv3x3 -> LayerNorm2d",
            "removed": ["GELU"],
            "retained": "pole-boundary RMSNorm before real-to-complex analysis",
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


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.CifarNanoConfig = ComplexScanConfig
    harness.build_cifar_nano = _build
    harness._contract = _contract
    harness._build_optimizer = complex_run._build_optimizer
    harness._loaders = packed_data.build_loaders
    harness.main()


if __name__ == "__main__":
    main()
