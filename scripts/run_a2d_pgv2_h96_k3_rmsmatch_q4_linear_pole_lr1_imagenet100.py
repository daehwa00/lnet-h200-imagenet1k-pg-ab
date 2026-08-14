#!/usr/bin/env python3
"""Run the Q4-linear control with pole geometry trained at the base LR."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_pgv2_h96_k3_rmsmatch_q4_linear_imagenet100 as control
import torch

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "PGv2-H96-K3-RMSMatch-Q4Linear-PoleLR1"
VARIANTS = (VARIANT,)
SEEDS = control.SEEDS
POLE_GEOMETRY_LR_MULTIPLIER = 1.0


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = control.q4.local_reader.control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported Q4-linear pole-LR variant: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    _configure_ramp()
    control._assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["optimizer_ablation"] = {
        "changed_parameter_family": "pole damping_logits and phase coordinates only",
        "pole_geometry_learning_rate_multiplier": POLE_GEOMETRY_LR_MULTIPLIER,
        "effective_peak_learning_rate": 3.0e-3,
        "weight_decay": 0.0,
        "control_multiplier": 0.1,
        "all_other_parameter_groups": "unchanged",
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    ramp = control.q4.local_reader.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pgv2_h96.k3_rmsmatch_q4_linear_pole_lr1.imagenet100.v1"
    payload["evidence_status"] = "untrained pole-geometry learning-rate ablation"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    recipe = payload["recipe"]
    recipe["pole_geometry_learning_rate_multiplier"] = POLE_GEOMETRY_LR_MULTIPLIER
    recipe["pole_geometry_effective_peak_learning_rate"] = (
        float(recipe["learning_rate"]) * POLE_GEOMETRY_LR_MULTIPLIER
    )
    payload["architecture"] = {
        VARIANT: (
            "Exact PGv2-H96-K3-RMSMatch-Q4Linear control, including model weights, "
            "initialization, raw-Q4 Linear384-to-100 head, data recipe, and seed. "
            "Only the optimizer changes: pole damping and phase parameters use the "
            "full base learning rate instead of the control's 0.1 multiplier."
        )
    }
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pole_lr1_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    _configure_ramp()
    ramp = control.q4.local_reader.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = ramp.backbone.a2d_base.residuals
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
            model_config=ramp.PoleModelConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=control.q4.local_reader.control.control._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
