# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run widely-linear and augmented-complex CIFAR-100 validation screens."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cifar100_packed_data as packed_data
import run_alphabet2d_cifar100_nano as harness

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("widely_linear", "augmented_complex_ffn")
SEEDS = (401, 402, 403)


def _variant_config(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanConfig:
    if variant == "widely_linear":
        return replace(config, widely_linear_bridges=True)
    if variant == "augmented_complex_ffn":
        return replace(config, augmented_widths=(48, 64))
    message = f"unknown augmented complex variant: {variant}"
    raise ValueError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return ComplexScanBackbone(_variant_config(variant, config))


def _build_optimizer(
    model: torch.nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    phase_gated_projection_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, PhaseGatedComplexFFN)
        for projection in (module.input_projection, module.output_projection)
        for parameter in projection.parameters()
    }
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    modal_no_decay: list[torch.nn.Parameter] = []
    geometry: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        parameter_name = name.rsplit(".", maxsplit=1)[-1]
        if id(parameter) in phase_gated_projection_ids:
            no_decay.append(parameter)
        elif "damping_logits" in parameter_name or parameter_name.startswith("phase_"):
            geometry.append(parameter)
        elif (
            "analysis" in name
            or "widely_bridge" in name
            or "augmented.direction_mixer" in name
            or "augmented.output_projection" in name
        ):
            modal_no_decay.append(parameter)
        elif (
            parameter.ndim < 2
            or "norm" in name
            or "initial_" in name
            or name.endswith(".bias")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    learning_rate = float(recipe["learning_rate"])
    groups = [
        {"params": decay, "lr": learning_rate, "weight_decay": recipe["weight_decay"]},
        {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0},
        {
            "params": modal_no_decay,
            "lr": learning_rate * recipe["modal_learning_rate_multiplier"],
            "weight_decay": 0.0,
        },
    ]
    groups.append(
        {
            "params": geometry,
            "lr": learning_rate * recipe["pole_geometry_learning_rate_multiplier"],
            "weight_decay": 0.0,
        }
    )
    optimizer = torch.optim.AdamW(
        groups,
        fused=bool(recipe.get("fused_optimizer", False)),
    )
    projected_modules = tuple(
        module
        for module in model.modules()
        if isinstance(module, PhaseGatedComplexFFN) and module.projected_direction_rows
    )
    if projected_modules:

        def project_phase_gated_directions(
            _optimizer: torch.optim.Optimizer,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
        ) -> None:
            for module in projected_modules:
                module.project_direction_rows_()

        optimizer.register_step_post_hook(project_phase_gated_directions)
    return optimizer


def _contract(args: Namespace) -> dict[str, Any]:
    config = ComplexScanConfig()
    models = {variant: _build(variant, config) for variant in VARIANTS}
    payload = {
        "schema": "lnet.complex_scan.augmented_complex.cifar100.v1",
        "evidence_status": "validation-only 50-epoch augmented-complex ladder",
        "official_test_evaluation": not args.skip_test,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "variant_configs": {
            variant: asdict(_variant_config(variant, config)) for variant in VARIANTS
        },
        "parameter_counts": {
            variant: {
                "total": sum(parameter.numel() for parameter in model.parameters()),
                "trainable": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
            }
            for variant, model in models.items()
        },
        "architecture": {
            "stem": "locked normalized GELU stem",
            "widely_linear": "Wz + V*conj(z)",
            "augmented_hidden_widths": [48, 64],
            "augmented_expansion": 2,
            "activation": "Cartesian SiLU",
            "layer_scale_initial": 1.0e-3,
            "terminal": "static product-pole energy analyzer",
            "descriptor_dim": 288,
            "head": "LRQ16",
        },
        "references": {
            "complex_linear_mean_best_validation_100epoch": 0.5111333333333334,
            "cib_rank8_mean_best_validation_100epoch": 0.5478,
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
