#!/usr/bin/env python3
"""Train four-stage A2D with an independent stage-residual modal head."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_resaux1_deep4_imagenet100 as deep4
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.a2d_head_design import IndependentStageResidualHead
from lnet.complex_scan import ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "A2D-Deep4-SR64"
SEEDS = (501,)
STAGE_WIDTH = 64
STAGE_DIM = 4 * deep4.STAGE_MODES
heads = deep4.heads


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported A2D Deep4 stage-residual variant: {variant}"
        raise ValueError(message)
    model = deep4._build(deep4.VARIANT, config)
    stage_residual = IndependentStageResidualHead(
        nn.BatchNorm1d(deep4.DESCRIPTOR_DIM, affine=False),
        config.output_dim,
        STAGE_WIDTH,
        input_dim=deep4.DESCRIPTOR_DIM,
        stage_dim=STAGE_DIM,
    )
    model.classifier = resaux_base.heads.A2DAffineQClassifier(
        deep4.DESCRIPTOR_DIM,
        config.output_dim,
        main="fusion",
        affine=None,
        fusion=cast("Any", stage_residual),
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=0.0,
    )
    return model


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = deep4._wandb_model_metrics(model)
    classifier = cast("Any", model).classifier
    metrics["head/stage_residual_beta"] = float(classifier.fusion.beta.detach())
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = deep4._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload.update(
        {
            "schema": "lnet.a2d.deep4_stage_residual.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch four-stage independent-residual head",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        VARIANT: {
            "backbone": {
                "name": "A2D-D4-PathMix-PostCarry-PostFFN-4Stage",
                "modes": [deep4.STAGE_MODES] * 4,
                "spatial_resolutions": [56, 28, 14, 7],
                "descriptor_dim": deep4.DESCRIPTOR_DIM,
            },
            "head": {
                "normalizer": "BatchNorm1d(768, affine=False)",
                "main": "Affine768-to-100",
                "stage_residuals": "4 independent 192-to-64-to-100 branches",
                "residual_scale_initial": 0.1,
                "affine_auxiliary_weight": 0.0,
                "lrq": False,
            },
        }
    }
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "A2D-Deep4 with its 768-coordinate four-stage Q descriptor standardized "
            "by non-affine BatchNorm, classified by one affine 768-to-100 main path "
            "plus beta-scaled independent GELU/RMSNorm 192-to-64-to-100 residuals "
            "for each of the four stages. No deep fusion, LRQ, or duplicate affine "
            "auxiliary branch."
        )
    }
    payload["source_sha256"]["a2d_deep4_stage_residual_runner"] = (
        deep4.baseline.heads.harness._digest(Path(__file__))
    )
    return json.loads(json.dumps(payload))


def main() -> None:
    source = resaux_base
    residuals = a2d_base.residuals
    harness = source.heads.harness
    source.heads.VARIANTS = (VARIANT,)
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=(VARIANT,),
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
