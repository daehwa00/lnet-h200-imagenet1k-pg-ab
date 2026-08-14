#!/usr/bin/env python3
"""Train P4-ProjRes with unit joint and stage residual coefficients."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_p4_projected_imagenet100 as projected
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexScanConfig,
    S2DProjectedResidualPostFusionCFFNTransition,
)

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "P4-NoScale"
VARIANTS = (VARIANT,)
SEEDS = (501,)
heads = projected.heads


def _assert_noscale(model: nn.Module) -> None:
    projected._assert_projected(model)
    for stage in projected.stemres.joint.p4._pole_banks(model)[:-1]:
        transition = stage.augmented
        if not isinstance(transition, S2DProjectedResidualPostFusionCFFNTransition):
            message = "P4-NoScale requires projected residual transitions"
            raise TypeError(message)
        if transition.joint_scale is not None or transition.pole_scale is not None:
            message = "P4-NoScale retained eta_joint or alpha_s"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported P4 no-scale variant: {variant}"
        raise ValueError(message)
    model = projected._build(projected.VARIANT, config)
    for stage in projected.stemres.joint.p4._pole_banks(model)[:-1]:
        transition = stage.augmented
        if not isinstance(transition, S2DProjectedResidualPostFusionCFFNTransition):
            message = "P4-NoScale build received the wrong transition"
            raise TypeError(message)
        transition.use_unit_scales_()
    _assert_noscale(model)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = projected._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][projected.VARIANT])
    variant_config["backbone"]["path_contract"].update(
        {
            "joint_residual_coefficient": "fixed_1_no_parameter",
            "stage_pole_coefficient": "fixed_1_no_parameter",
        }
    )
    payload.update(
        {
            "schema": "lnet.a2d.p4_noscale.imagenet100.v1",
            "evidence_status": "P4-ProjRes unit-coefficient ablation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "P4-StemRes plus projected joint residual with eta_joint and "
                    "alpha_s removed: the 64-wide nonlinear update and pole update "
                    "both enter with fixed unit coefficients."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_p4_noscale_runner"] = heads.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def main() -> None:
    calibrated = projected.stemres.joint.p4.calibrated
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
            wandb_model_metrics=calibrated.canonical8.fair_init.backbone._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
