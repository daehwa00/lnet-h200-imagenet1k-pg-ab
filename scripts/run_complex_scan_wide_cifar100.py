# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the parameter-matched Complex-AmpPhase-Wide CIFAR-100 screen."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cifar100_packed_data as packed_data
import run_alphabet2d_cifar100_nano as harness

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("complex_amp_phase_wide",)
SEEDS = (401, 402, 403)


def _config() -> ComplexScanConfig:
    return ComplexScanConfig(transition_widths=(64, 96))


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANTS[0]:
        message = f"unknown wide complex variant: {variant}"
        raise ValueError(message)
    return ComplexScanBackbone(config)


def _build_optimizer(
    model: torch.nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    modal: list[torch.nn.Parameter] = []
    geometry: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if "damping_logits" in name or (
            "phase_" in name and "phase_gate" not in name
        ):
            geometry.append(parameter)
        elif (
            "analysis" in name
            or "transition.input_projection" in name
            or "transition.output_projection" in name
        ):
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


def _contract(args: Namespace) -> dict[str, Any]:
    config = _config()
    model = _build(VARIANTS[0], config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "schema": "lnet.complex_scan.amp_phase_wide.cifar100.v1",
        "evidence_status": "validation-only 50-epoch parameter-matched capacity screen",
        "official_test_evaluation": not args.skip_test,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": {
            VARIANTS[0]: {"total": parameters, "trainable": parameters}
        },
        "architecture": {
            "stem": "locked winning normalized GELU stem",
            "complex_state_continuous": True,
            "transition_widths": [64, 96],
            "expansion_ratio": 2,
            "amplitude_gate": "exp(0.4*tanh(W*log1p(|U|^2)+b))",
            "phase_gate": "pi/12*tanh(W*log1p(|U|^2)+b)",
            "layer_scale_initial": 1.0e-2,
            "terminal": "static product-pole energy analyzer",
            "descriptor_dim": 288,
            "head": "LRQ16",
        },
        "references": {
            "complex_linear_66k_mean_best_validation_100epoch": 0.5111333333333334,
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
    harness._build_optimizer = _build_optimizer
    harness._loaders = packed_data.build_loaders
    harness.main()


if __name__ == "__main__":
    main()
