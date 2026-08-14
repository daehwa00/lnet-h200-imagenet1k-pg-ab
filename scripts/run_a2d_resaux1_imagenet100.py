#!/usr/bin/env python3
"""Train the residual A2D backbone with Fusion384 and affine auxiliary CE."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_affine_qhead_imagenet100 as heads
import run_a2d_d4_postcarry_postffn_imagenet100 as backbone
import run_a2d_qhead_e2e_imagenet100 as structured
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexScanConfig,
    ParallelFusionLRQHead,
)
from lnet.image_layers import StandardizedAffineModalHead

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "A2D-ResAux1"
SEEDS = (501,)
FUSION_WIDTH = 384
AFFINE_AUXILIARY_WEIGHT = 1.0


def _build(
    variant: str,
    config: ComplexScanConfig,
) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported A2D ResAux1 variant: {variant}"
        raise ValueError(message)
    model = backbone._build(backbone.VARIANT, config)
    current = model.classifier
    if not isinstance(current, ParallelFusionLRQHead):
        message = "residual A2D backbone no longer exposes Fusion384+LRQ64"
        raise TypeError(message)
    if current.fusion.hidden_dim != FUSION_WIDTH:
        message = "residual A2D backbone Fusion width changed"
        raise RuntimeError(message)
    model.classifier = heads.A2DAffineQClassifier(
        model.descriptor_dim,
        config.output_dim,
        main="fusion",
        affine=StandardizedAffineModalHead(model.descriptor_dim, config.output_dim),
        fusion=current.fusion,
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=AFFINE_AUXILIARY_WEIGHT,
    )
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = backbone._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    transition = payload["variant_configs"][backbone.VARIANT]["stage_transition"]
    payload.update(
        {
            "schema": "lnet.a2d.resaux1.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch residual-head confirmation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        VARIANT: {
            "backbone": {
                "name": "A2D-D4-PathMix-PostCarry-PostFFN",
                "stage_transition": transition,
            },
            "head": {
                "main": "Fusion384",
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
            "A2D-D4-PathMix with PostCarry outer residual and a mode-wise "
            "48-96-48 post-fusion residual CFFN, followed by Fusion384 as "
            "the inference head and a standardized affine Q head trained "
            "only through auxiliary cross entropy with weight 1.0; no LRQ."
        )
    }
    payload["recipe"].update(
        {
            "epochs": args.epochs,
            "optimizer": "AdamW (fused, pole-aware parameter groups)",
            "fused_optimizer": True,
            "learning_rate": 3.0e-3,
            "modal_learning_rate_multiplier": 1.0 / 3.0,
            "pole_geometry_learning_rate_multiplier": 0.1,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "warmup plus cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "loader_prefetch_factor": heads.harness.PREFETCH_FACTOR,
            "device_prefetch_scope": "copy_only",
            # Record the mode that actually runs, so the contract cannot claim
            # one compile mode while LNET_COMPILE_MODE selects another.
            "compile_mode": os.environ.get("LNET_COMPILE_MODE", "default"),
            "matmul_precision": "high (TF32 enabled)",
            "compiled_training_preparation": True,
            "channels_last": True,
            "augmentation": "matched A2D ImageNet-100 public recipe",
            "selection": "fixed epoch 100; no within-run validation selection",
            "resume": "epoch-boundary exact RNG restore",
        }
    )
    payload["source_sha256"].update(
        {
            "resaux1_runner": heads.harness._digest(Path(__file__)),
            "residual_backbone_runner": heads.harness._digest(
                Path("scripts/run_a2d_d4_postcarry_postffn_imagenet100.py")
            ),
            "affine_head_runner": heads.harness._digest(
                Path("scripts/run_a2d_affine_qhead_imagenet100.py")
            ),
        }
    )
    return json.loads(json.dumps(payload))


def _prepare_model(model: nn.Module, recipe: dict[str, Any]) -> nn.Module:
    residuals = a2d_base.residuals
    prepared = residuals.base._prepare_model(model, recipe)
    method = getattr(prepared, "prepare_for_compiled_training_", None)
    if not callable(method):
        message = "A2D ResAux1 backbone is missing compiled-training preparation"
        raise TypeError(message)
    return cast("nn.Module", method())


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = heads._wandb_model_metrics(model)
    metrics.update(backbone._wandb_model_metrics(model))
    return metrics


def main() -> None:
    residuals = a2d_base.residuals
    harness = heads.harness
    heads.VARIANTS = (VARIANT,)
    heads.SEEDS = SEEDS
    structured._training_objective = heads._training_objective
    structured._after_training_batch = heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.RunnerBindings(
            variants=(VARIANT,),
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=_prepare_model,
            train_epoch=structured._train_epoch,
            evaluate=heads._evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
