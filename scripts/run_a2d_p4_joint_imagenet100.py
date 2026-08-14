#!/usr/bin/env python3
"""Train raw-descriptor P4 with one joint path-mode CFFN and a direct S2D transition."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_p4_imagenet100 as p4
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexScanConfig,
    S2DDirectPostFusionCFFNTransition,
    S2DPostFusionCFFNTransition,
)
from lnet.pac_path_cffn import JointPathModeCFFNCombiner

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "P4-Joint128"
VARIANTS = (VARIANT,)
SEEDS = (501,)
JOINT_HIDDEN = 128
heads = p4.heads


def _replace_stage_composition(stage: nn.Module) -> None:
    previous_transition = stage.augmented
    if not isinstance(previous_transition, S2DPostFusionCFFNTransition):
        message = "P4-Joint requires the GlobalGain PostFusion transition"
        raise TypeError(message)
    previous_combiner = stage.quadrant_path_mode_combiner
    if previous_combiner is None or previous_combiner.path_count != 4:
        message = "P4-Joint requires four coarsened product paths"
        raise TypeError(message)

    stage.quadrant_path_mode_combiner = JointPathModeCFFNCombiner(
        stage.modes,
        JOINT_HIDDEN,
        layer_scale_initial=1.0e-3,
    )
    transition = S2DDirectPostFusionCFFNTransition(
        modes=stage.modes,
        output_modes=previous_transition.output_modes,
        pole_paths=4,
        pole_scale_initial=float(previous_transition.pole_scale.detach()),
        post_hidden_modes=previous_transition.post_hidden_modes,
        post_layer_scale_initial=float(previous_transition.post_ffn_scale.detach().mean()),
        post_ffn_activation=previous_transition.post_ffn_activation,
    )
    transition.copy_retained_state_from(previous_transition)
    stage.augmented = transition


def _assert_joint(model: nn.Module) -> None:
    p4._pole_banks(model)
    for stage in p4._pole_banks(model)[:-1]:
        combiner = stage.quadrant_path_mode_combiner
        if (
            not isinstance(combiner, JointPathModeCFFNCombiner)
            or combiner.path_count != 4
            or combiner.modes != 64
            or combiner.hidden_modes != JOINT_HIDDEN
        ):
            message = "P4-Joint128 stage is missing its 4x64 joint CFFN"
            raise RuntimeError(message)
        transition = stage.augmented
        if (
            not isinstance(transition, S2DDirectPostFusionCFFNTransition)
            or transition.input_modes != 256
            or transition.output_modes != 64
            or transition.direction_mixer is not None
            or transition.ffn_input is not None
            or transition.ffn_output is not None
        ):
            message = "P4-Joint retained the legacy expanded pole transition"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported P4 joint variant: {variant}"
        raise ValueError(message)
    model = p4._build(p4.VARIANT, config)
    for stage in p4._pole_banks(model)[:-1]:
        _replace_stage_composition(stage)
    _assert_joint(model)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = p4._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][p4.VARIANT])
    variant_config["backbone"]["path_contract"].update(
        {
            "mode_cffn": "joint_path_mode_256_128_256",
            "path_cffn": "absorbed_into_joint_cffn",
            "joint_cffn_hidden": JOINT_HIDDEN,
            "stage_transition": "direct_256_to_64_plus_s2d_postfusion",
        }
    )
    variant_config["backbone"]["legacy_mode_cffn_widths"] = None
    variant_config["backbone"]["legacy_augmented_widths"] = None
    payload.update(
        {
            "schema": "lnet.a2d.p4_joint.imagenet100.v2",
            "evidence_status": "joint path-mode composition architecture candidate",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "GlobalGain raw-descriptor P4 with four product paths flattened "
                    "into one residual "
                    "256-to-128-to-256 joint CFFN per non-terminal stage, a direct "
                    "256-to-64 pole projection, S2D carry, and retained 64-to-128-to-64 "
                    "PostFusion CFFN; legacy per-path ModeCFFN and expanded pole "
                    "transition are removed."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_p4_joint_runner"] = heads.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def main() -> None:
    calibrated = p4.calibrated
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
