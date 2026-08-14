"""Train D4-M64 with orientation-matched radial-group damping initialization."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_backbone_variants_imagenet100 as backbone
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import ComplexScanConfig, ComplexScanStage

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "D4-M64-FairInit"
VARIANTS = (VARIANT,)
SEEDS = (501, 509)
heads = backbone.heads


def _radial_group_damping(modes: int, *, like: torch.Tensor) -> torch.Tensor:
    if modes <= 0 or modes % 4:
        message = "fair M64 damping initialization requires four-orientation mode groups"
        raise ValueError(message)
    group_damping = torch.logspace(
        math.log10(0.04),
        math.log10(0.35),
        modes // 4,
        dtype=like.dtype,
        device=like.device,
    )
    return group_damping.repeat_interleave(4)


def _install_fair_damping_initialization(bank: ComplexScanStage) -> None:
    damping = _radial_group_damping(bank.modes, like=bank.damping_logits_x)
    ratio = ((damping - bank.damping_min) / (bank.damping_max - bank.damping_min)).clamp(
        1.0e-4,
        1.0 - 1.0e-4,
    )
    logits = torch.logit(ratio)
    with torch.no_grad():
        bank.damping_logits_x.copy_(logits)
        bank.damping_logits_y.copy_(logits)


def _pole_banks(model: nn.Module) -> tuple[ComplexScanStage, ...]:
    banks = tuple(getattr(model, name) for name in ("stage1", "stage2", "stage3", "terminal"))
    if not all(isinstance(bank, ComplexScanStage) for bank in banks):
        message = "D4-M64 fair initialization expected four complex scan stages"
        raise TypeError(message)
    return banks


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported fair-initialized Deep4 variant: {variant}"
        raise ValueError(message)
    model = backbone._build(backbone.UNIFORM_M64, config)
    for bank in _pole_banks(model):
        _install_fair_damping_initialization(bank)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = backbone._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][backbone.UNIFORM_M64])
    variant_config["backbone"]["damping_initialization"] = {
        "kind": "radial_group_logspace",
        "range": [0.04, 0.35],
        "radial_groups": 16,
        "orientations_per_group": 4,
        "x_y_matched": True,
        "training_parameters_independent_after_initialization": True,
    }
    payload.update(
        {
            "schema": "lnet.a2d.deep4_m64_fair_init.imagenet100.v1",
            "evidence_status": "two-seed 100-epoch damping-initialization ablation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "D4-M64 with every architecture and training setting retained; "
                    "the only change is 16 logarithmic damping values repeated over "
                    "each four-orientation radial group at initialization."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_deep4_m64_fair_init_runner"] = heads.harness._digest(
        Path(__file__)
    )
    return json.loads(json.dumps(payload))


def main() -> None:
    source = resaux_base
    residuals = a2d_base.residuals
    harness = source.heads.harness
    source.heads.VARIANTS = VARIANTS
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=backbone._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
