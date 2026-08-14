#!/usr/bin/env python3
"""Train the fair-damped D4-M64 control with a canonical eight-angle atlas."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_m64_fair_init_imagenet100 as fair_init
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import ComplexScanConfig, ComplexScanStage

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "D4-M64-C8"
VARIANTS = (VARIANT,)
SEEDS = (501,)
ORIENTATIONS = 8
RADIAL_LEVELS = 8
MAXIMUM_PHASES = (
    math.pi * 0.75,
    math.pi * 0.70,
    fair_init.backbone.deep4.STAGE3_MAXIMUM_PHASE,
    math.pi * 0.65,
)
heads = fair_init.heads


def _canonical8_phase_atlas(
    modes: int,
    maximum_phase: float,
    *,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if modes != ORIENTATIONS * RADIAL_LEVELS:
        message = "canonical-8 initialization requires exactly 64 pole modes"
        raise ValueError(message)
    radial = torch.logspace(
        math.log10(maximum_phase / 8.0),
        math.log10(maximum_phase),
        RADIAL_LEVELS,
        dtype=like.dtype,
        device=like.device,
    ).repeat_interleave(ORIENTATIONS)
    orientation = torch.linspace(
        0.0,
        math.pi / 2.0,
        ORIENTATIONS,
        dtype=like.dtype,
        device=like.device,
    ).repeat(RADIAL_LEVELS)
    return radial * torch.cos(orientation), radial * torch.sin(orientation)


def _install_canonical8_initialization(
    bank: ComplexScanStage,
    maximum_phase: float,
) -> None:
    phase_x, phase_y = _canonical8_phase_atlas(
        bank.modes,
        maximum_phase,
        like=bank.phase_x,
    )
    damping = torch.logspace(
        math.log10(0.04),
        math.log10(0.35),
        RADIAL_LEVELS,
        dtype=bank.damping_logits_x.dtype,
        device=bank.damping_logits_x.device,
    ).repeat_interleave(ORIENTATIONS)
    damping_ratio = ((damping - bank.damping_min) / (bank.damping_max - bank.damping_min)).clamp(
        1.0e-4, 1.0 - 1.0e-4
    )
    damping_logits = torch.logit(damping_ratio)
    with torch.no_grad():
        bank.phase_x.copy_(phase_x)
        bank.phase_y.copy_(phase_y)
        bank.damping_logits_x.copy_(damping_logits)
        bank.damping_logits_y.copy_(damping_logits)


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported canonical-8 Deep4 variant: {variant}"
        raise ValueError(message)
    model = fair_init._build(fair_init.VARIANT, config)
    for bank, maximum_phase in zip(
        fair_init._pole_banks(model),
        MAXIMUM_PHASES,
        strict=True,
    ):
        _install_canonical8_initialization(bank, maximum_phase)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = fair_init._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][fair_init.VARIANT])
    variant_config["backbone"]["damping_initialization"] = {
        "kind": "radial_group_logspace",
        "range": [0.04, 0.35],
        "radial_groups": RADIAL_LEVELS,
        "orientations_per_group": ORIENTATIONS,
        "x_y_matched": True,
        "training_parameters_independent_after_initialization": True,
    }
    variant_config["backbone"]["phase_initialization"] = {
        "kind": "canonical_first_quadrant",
        "orientations_degrees": [
            round(value, 6) for value in torch.linspace(0.0, 90.0, ORIENTATIONS).tolist()
        ],
        "orientations": ORIENTATIONS,
        "radial_levels": RADIAL_LEVELS,
        "radial_range_relative_to_maximum_phase": [0.125, 1.0],
        "effective_signs_supplied_by_quadrant_scans": True,
        "training_parameters_independent_after_initialization": True,
    }
    payload.update(
        {
            "schema": "lnet.a2d.deep4_m64_canonical8.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch pole-atlas ablation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "The D4-M64-FairInit architecture and training recipe with its "
                    "pole atlas regrouped from four half-plane orientations by "
                    "sixteen radii to eight first-quadrant orientations by eight "
                    "radii; damping remains matched within each radial group."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_deep4_m64_canonical8_runner"] = heads.harness._digest(
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
            wandb_model_metrics=fair_init.backbone._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
