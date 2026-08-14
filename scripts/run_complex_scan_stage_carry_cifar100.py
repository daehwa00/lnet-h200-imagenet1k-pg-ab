# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the all-complex S2D stage-carry CIFAR-100 validation ladder."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cifar100_packed_data as packed_data
import run_alphabet2d_cifar100_nano as harness
import run_complex_scan_augmented_cifar100 as augmented

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = (
    "s2d_pole_main",
    "s2d_carry_main",
)
SEEDS = (401, 402, 403)


def _variant_config(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanConfig:
    if variant not in VARIANTS:
        message = f"unknown complex stage-carry variant: {variant}"
        raise ValueError(message)
    merge = "carry_main" if variant.endswith("carry_main") else "pole_main"
    return replace(
        config,
        augmented_widths=(48, 64),
        carry_bases=("s2d", "s2d"),
        carry_merge=merge,
        carry_scale_initial=0.1 if merge == "carry_main" else 1.0e-2,
    )


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return ComplexScanBackbone(_variant_config(variant, config))


def _parameter_count(model: ComplexScanBackbone) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def _contract(args: Namespace) -> dict[str, Any]:
    config = ComplexScanConfig()
    models = {variant: _build(variant, config) for variant in VARIANTS}
    payload = {
        "schema": "lnet.complex_scan.stage_carry.cifar100.v1",
        "evidence_status": "validation-only 50-epoch stage-carry selection",
        "official_test_evaluation": not args.skip_test,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "variant_configs": {
            variant: asdict(_variant_config(variant, config)) for variant in VARIANTS
        },
        "parameter_counts": {variant: _parameter_count(model) for variant, model in models.items()},
        "architecture": {
            "stem": "locked normalized GELU stem",
            "pole_branch": "static product-pole endpoints plus widely-linear direction mixing",
            "carry_branch": "lossless 2x2 complex S2D coordinates",
            "carry_projection": "Wz + V*conj(z) into the augmented hidden width",
            "merge_location": "after direction mixing and before Cartesian-SiLU ACFFN",
            "pole_main_initial_scale": 1.0e-2,
            "carry_main_initial_pole_scale": 0.1,
            "augmented_hidden_widths": [48, 64],
            "augmented_expansion": 2,
            "activation": "Cartesian SiLU",
            "terminal": "static product-pole energy analyzer",
            "descriptor_dim": 288,
            "head": "LRQ16",
        },
        "references": {
            "augmented_complex_ffn_seed401_best_validation_50epoch": 0.5378,
            "augmented_complex_ffn_mean_best_validation_100epoch": 0.5960666666666666,
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
    harness._build_optimizer = augmented._build_optimizer
    harness._loaders = packed_data.build_loaders
    harness.main()


if __name__ == "__main__":
    main()
