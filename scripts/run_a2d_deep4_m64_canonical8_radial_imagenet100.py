#!/usr/bin/env python3
"""Train calibrated D4-M64-C8 with identity-centered radial CFFNs."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_m64_canonical8_calibrated_imagenet100 as calibrated
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexScanConfig,
    FactorizedQuadrantPathModeCFFNCombiner,
    S2DPostFusionCFFNTransition,
)

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "D4-M64-C8-RadAll"
VARIANTS = (VARIANT,)
SEEDS = (501,)
ACTIVATION = "identity_centered_magnitude"
heads = calibrated.heads


def _install_radial_activations(model: nn.Module) -> None:
    """Replace only the twelve non-terminal CFFN activations."""
    for stage_name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, stage_name)
        combiner = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        if not isinstance(combiner, FactorizedQuadrantPathModeCFFNCombiner):
            message = f"{stage_name} does not use the expected path/mode CFFN"
            raise TypeError(message)
        if not isinstance(transition, S2DPostFusionCFFNTransition):
            message = f"{stage_name} does not use the expected post-fusion CFFN"
            raise TypeError(message)
        combiner.mode_activation = ACTIVATION
        combiner.path_activation = ACTIVATION
        transition.ffn_activation = ACTIVATION
        transition.post_ffn_activation = ACTIVATION


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported radial Deep4 variant: {variant}"
        raise ValueError(message)
    model = calibrated._build(calibrated.VARIANT, config)
    _install_radial_activations(model)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = calibrated._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][calibrated.VARIANT])
    variant_config["backbone"]["complex_ffn_activation_ablation"] = {
        "scope": "all twelve CFFNs across stages 1-3",
        "sites_per_stage": [
            "mode_cffn",
            "path_cffn",
            "pole_transition_cffn",
            "post_merge_cffn",
        ],
        "control": "cartesian_silu(real) + i*cartesian_silu(imag)",
        "candidate": "2*sigmoid(abs(U)-1)*U",
        "trainable_activation_parameters": 0,
        "all_other_architecture_and_training_settings_unchanged": True,
    }
    payload.update(
        {
            "schema": "lnet.a2d.deep4_m64_canonical8_radial.imagenet100.v2",
            "evidence_status": "untrained static-damping optimization candidate",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "D4-M64-C8-Cal with all twelve non-terminal-stage CFFN "
                    "Cartesian-SiLU activations replaced by the fixed, "
                    "phase-preserving 2*sigmoid(abs(U)-1)*U activation."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_deep4_m64_canonical8_radial_runner"] = heads.harness._digest(
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
            wandb_model_metrics=calibrated.canonical8.fair_init.backbone._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
