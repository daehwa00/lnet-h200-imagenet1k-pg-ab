#!/usr/bin/env python3
"""Train P4 with explicit projected, nonlinear, and spatial residual branches."""

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
    FixedComplexRMSNorm,
    S2DCleanProjectedResidualPostFusionCFFNTransition,
    S2DProjectedResidualPostFusionCFFNTransition,
)
from lnet.pac_path_cffn import IdentityQuadrantPathModeCombiner

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "P4-CleanProjRes"
VARIANTS = (VARIANT,)
SEEDS = (501,)
CORRECTION_HIDDEN = 128
heads = projected.heads


def _replace_transition(stage: nn.Module) -> None:
    previous = stage.augmented
    if not isinstance(previous, S2DProjectedResidualPostFusionCFFNTransition):
        message = "P4-CleanProjRes requires a P4-ProjRes transition"
        raise TypeError(message)
    transition = S2DCleanProjectedResidualPostFusionCFFNTransition(
        modes=stage.modes,
        correction_hidden_modes=CORRECTION_HIDDEN,
        output_modes=previous.output_modes,
        pole_paths=4,
        post_hidden_modes=previous.post_hidden_modes,
        post_layer_scale_initial=float(previous.post_ffn_scale.detach().mean()),
        post_ffn_activation=previous.post_ffn_activation,
    )
    transition.copy_retained_state_from(previous)
    stage.augmented = transition


def _assert_clean(model: nn.Module) -> None:
    projected.stemres._assert_stem(model)
    for stage in projected.stemres.joint.p4._pole_banks(model)[:-1]:
        combiner = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        if not isinstance(combiner, IdentityQuadrantPathModeCombiner) or tuple(
            combiner.parameters()
        ):
            message = "P4-CleanProjRes retained a pre-transition path combiner"
            raise RuntimeError(message)
        if (
            not isinstance(
                transition,
                S2DCleanProjectedResidualPostFusionCFFNTransition,
            )
            or transition.input_modes != 256
            or transition.output_modes != 64
            or transition.correction_hidden_modes != CORRECTION_HIDDEN
            or transition.shortcut_projection.input_modes != 256
            or transition.shortcut_projection.output_modes != 64
            or transition.correction_input.input_modes != 256
            or transition.correction_input.output_modes != 128
            or transition.correction_output.input_modes != 128
            or transition.correction_output.output_modes != 64
            or not isinstance(transition.correction_norm, FixedComplexRMSNorm)
            or transition.pole_scale is not None
        ):
            message = "P4-CleanProjRes transition contract changed"
            raise RuntimeError(message)
        if stage.output_modes is None:
            message = "P4-CleanProjRes left the fused product-only scan path"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported P4 clean projected-residual variant: {variant}"
        raise ValueError(message)
    model = projected._build(projected.VARIANT, config)
    for stage in projected.stemres.joint.p4._pole_banks(model)[:-1]:
        _replace_transition(stage)
    _assert_clean(model)
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
            "projected_shortcut": "widely_linear_256_to_64",
            "nonlinear_correction": ("fixed_complex_rmsnorm_then_256_to_128_silu_128_to_64"),
            "correction_output_initialization": "exact_zero",
            "execution": "shared_widely_linear_and_complex_ffn_dispatchers",
            "stage_merge": "s2d_carry_plus_projected_shortcut_plus_correction",
            "stage_merge_scale": "none",
            "post_refinement": "64_to_128_silu_128_to_64_with_eta_post",
        }
    )
    payload.update(
        {
            "schema": "lnet.a2d.p4_cleanprojres.imagenet100.v1",
            "evidence_status": "clean projected residual ablation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "P4-StemRes with B=WL(X), U=WL(SiLU(WL(FixedCRMSNorm(X)))), "
                    "H=S2DCarry+B+U, zero-initialized U output, and the retained "
                    "eta_post-gated 64-to-128-to-64 refinement."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_p4_cleanprojres_runner"] = heads.harness._digest(Path(__file__))
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
