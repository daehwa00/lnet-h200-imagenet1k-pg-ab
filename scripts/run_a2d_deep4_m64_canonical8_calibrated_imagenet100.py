#!/usr/bin/env python3
"""Train D4-M64 with a symmetric, learned-calibrated canonical pole atlas."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_m64_canonical8_imagenet100 as canonical8
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import ComplexScanConfig, ComplexScanStage

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "D4-M64-C8-Cal"
VARIANTS = (VARIANT,)
SEEDS = (501,)

# Robust stage-level summaries from the completed D4-M64 pole audit.  The
# angular atlas remains canonical and symmetric; only frequency scale and
# damping are calibrated, so no individual learned pole is copied.
FREQUENCY_SCALES = (0.96, 0.86, 0.88, 0.90)
DAMPING_SCALES = (1.15, 1.20, 1.20, 1.15)
heads = canonical8.heads


def _static_damping_config(config: ComplexScanConfig) -> ComplexScanConfig:
    """Return the shared associative D4 scan configuration."""
    return config


def _install_calibrated_canonical8_initialization(
    bank: ComplexScanStage,
    maximum_phase: float,
    frequency_scale: float,
    damping_scale: float,
) -> None:
    calibrated_maximum_phase = maximum_phase * frequency_scale
    phase_x, phase_y = canonical8._canonical8_phase_atlas(
        bank.modes,
        calibrated_maximum_phase,
        like=bank.phase_x,
    )
    damping = torch.logspace(
        math.log10(0.04 * damping_scale),
        math.log10(0.35 * damping_scale),
        canonical8.RADIAL_LEVELS,
        dtype=bank.damping_logits_x.dtype,
        device=bank.damping_logits_x.device,
    ).repeat_interleave(canonical8.ORIENTATIONS)
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
        message = f"unsupported calibrated canonical-8 Deep4 variant: {variant}"
        raise ValueError(message)
    model = canonical8.fair_init._build(
        canonical8.fair_init.VARIANT,
        _static_damping_config(config),
    )
    calibration = zip(
        canonical8.fair_init._pole_banks(model),
        canonical8.MAXIMUM_PHASES,
        FREQUENCY_SCALES,
        DAMPING_SCALES,
        strict=True,
    )
    for bank, maximum_phase, frequency_scale, damping_scale in calibration:
        _install_calibrated_canonical8_initialization(
            bank,
            maximum_phase,
            frequency_scale,
            damping_scale,
        )
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = canonical8._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][canonical8.VARIANT])
    variant_config["backbone"]["damping_mode"] = {
        "kind": "static_learnable_per_axis_per_mode",
        "coefficient_layout": "mode_static",
    }
    stage_names = ("stage1", "stage2", "stage3", "terminal")
    variant_config["backbone"]["pole_initialization_calibration"] = {
        "source": "robust stage-level summaries from completed D4-M64 seed501 audit",
        "copies_individual_learned_poles": False,
        "angles_remain_uniform_and_x_y_symmetric": True,
        "frequency_scale_by_stage": dict(zip(stage_names, FREQUENCY_SCALES, strict=True)),
        "damping_scale_by_stage": dict(zip(stage_names, DAMPING_SCALES, strict=True)),
    }
    variant_config["backbone"]["damping_initialization"]["range_by_stage"] = {
        stage: [round(0.04 * scale, 6), round(0.35 * scale, 6)]
        for stage, scale in zip(stage_names, DAMPING_SCALES, strict=True)
    }
    variant_config["backbone"]["phase_initialization"]["maximum_phase_scale_by_stage"] = dict(
        zip(stage_names, FREQUENCY_SCALES, strict=True)
    )
    payload.update(
        {
            "schema": "lnet.a2d.deep4_m64_canonical8_calibrated.imagenet100.v2",
            "evidence_status": "untrained static-damping optimization candidate",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "D4-M64-C8 with the uniform eight-angle canonical atlas retained; "
                    "stage-level frequency contraction and damping expansion are "
                    "initialized from robust completed-run audit summaries, then learned "
                    "per axis and mode."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_deep4_m64_canonical8_calibrated_runner"] = heads.harness._digest(
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
            wandb_model_metrics=canonical8.fair_init.backbone._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
