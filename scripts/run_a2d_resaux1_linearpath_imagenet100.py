#!/usr/bin/env python3
"""Train A2D-ResAux1 with ModeCFFN followed by affine path synthesis."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_resaux1_imagenet100 as base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    AffineQuadrantPathModeCFFNCombiner,
    ComplexScanConfig,
    FactorizedQuadrantPathModeCFFNCombiner,
)

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "A2D-ResAux1-LinearPath"
SEEDS = (501,)


def _replace_path_combiner(stage: nn.Module) -> None:
    previous = stage.quadrant_path_mode_combiner
    if not isinstance(previous, FactorizedQuadrantPathModeCFFNCombiner):
        message = "LinearPath requires the established factorized D4 combiner"
        raise TypeError(message)
    replacement = AffineQuadrantPathModeCFFNCombiner(
        modes=previous.modes,
        mode_hidden=previous.mode_input.output_modes,
    )
    replacement.copy_mode_and_synthesis_from(previous)
    stage.quadrant_path_mode_combiner = replacement


def _build(
    variant: str,
    config: ComplexScanConfig,
) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported A2D LinearPath variant: {variant}"
        raise ValueError(message)
    model = base._build(base.VARIANT, config)
    _replace_path_combiner(model.stage1)
    _replace_path_combiner(model.stage2)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = base._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    base_variant = payload["variant_configs"][base.VARIANT]
    payload.update(
        {
            "schema": "lnet.a2d.resaux1.linearpath.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch affine-path confirmation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        VARIANT: {
            **base_variant,
            "path_combination": {
                "mode_feature_extractor": "shared residual ModeCFFN 48-96-48",
                "path_feature_extractor": None,
                "synthesis": "mode-wise strict-complex bias-free D4 mixing",
                "initialization": "identity across the four product paths",
            },
        }
    }
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "A2D-ResAux1 with shared residual ModeCFFN 48-96-48 retained per "
            "path, PathCFFN removed, and each mode's four D4 product paths mixed "
            "by a bias-free strict-complex "
            "linear synthesis. PostCarry/PostFFN residuals, Fusion384, and "
            "affine auxiliary CE weight 1.0 are unchanged."
        )
    }
    payload["source_sha256"]["linearpath_runner"] = base.heads.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def main() -> None:
    residuals = a2d_base.residuals
    harness = base.heads.harness
    base.heads.VARIANTS = (VARIANT,)
    base.heads.SEEDS = SEEDS
    base.structured._training_objective = base.heads._training_objective
    base.structured._after_training_batch = base.heads._after_training_batch
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
            prepare_model=base._prepare_model,
            train_epoch=base.structured._train_epoch,
            evaluate=base.heads._evaluate,
            wandb_model_metrics=base._wandb_model_metrics,
            summarize=base.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
