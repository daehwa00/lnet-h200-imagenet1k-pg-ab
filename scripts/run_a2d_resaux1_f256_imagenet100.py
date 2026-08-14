#!/usr/bin/env python3
"""Train A2D-ResAux1 with a 256-wide nonlinear fusion head."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_resaux1_imagenet100 as baseline
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexScanConfig,
    ModalFusionHead,
    ParallelFusionLRQHead,
)
from lnet.image_layers import StandardizedAffineModalHead

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "A2D-ResAux1-F256"
SEEDS = (501,)
FUSION_WIDTH = 256
AFFINE_AUXILIARY_WEIGHT = 1.0


def _build(
    variant: str,
    config: ComplexScanConfig,
) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported A2D ResAux1 F256 variant: {variant}"
        raise ValueError(message)
    model = baseline.backbone._build(baseline.backbone.VARIANT, config)
    current = model.classifier
    if not isinstance(current, ParallelFusionLRQHead):
        message = "residual A2D backbone no longer exposes Fusion+LRQ"
        raise TypeError(message)
    fusion = ModalFusionHead(
        model.descriptor_dim,
        FUSION_WIDTH,
        config.output_dim,
    )
    model.classifier = baseline.heads.A2DAffineQClassifier(
        model.descriptor_dim,
        config.output_dim,
        main="fusion",
        affine=StandardizedAffineModalHead(model.descriptor_dim, config.output_dim),
        fusion=fusion,
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=AFFINE_AUXILIARY_WEIGHT,
    )
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = baseline._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    original = payload["variant_configs"][baseline.VARIANT]
    payload.update(
        {
            "schema": "lnet.a2d.resaux1_f256.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch Fusion256 comparison",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        VARIANT: {
            "backbone": original["backbone"],
            "head": {
                "main": "Fusion256",
                "affine_auxiliary_weight": AFFINE_AUXILIARY_WEIGHT,
                "lrq": False,
            },
        }
    }
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "A2D-D4-PathMix with PostCarry and a 48-96-48 post-fusion residual "
            "CFFN, followed by descriptor-to-256 nonlinear fusion and a "
            "standardized affine Q auxiliary head with weight 1.0; no LRQ."
        )
    }
    payload["source_sha256"]["resaux1_f256_runner"] = baseline.heads.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def main() -> None:
    residuals = a2d_base.residuals
    harness = baseline.heads.harness
    baseline.heads.VARIANTS = (VARIANT,)
    baseline.heads.SEEDS = SEEDS
    baseline.structured._training_objective = baseline.heads._training_objective
    baseline.structured._after_training_batch = baseline.heads._after_training_batch
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
            prepare_model=baseline._prepare_model,
            train_epoch=baseline.structured._train_epoch,
            evaluate=baseline.heads._evaluate,
            wandb_model_metrics=baseline._wandb_model_metrics,
            summarize=baseline.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
