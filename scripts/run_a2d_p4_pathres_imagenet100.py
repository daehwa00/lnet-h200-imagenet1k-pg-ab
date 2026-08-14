#!/usr/bin/env python3
"""Train product-only P4 with a joint path-mode residual before compression."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_p4_cleanprojres_imagenet100 as clean
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexRMSNorm,
    ComplexScanConfig,
    S2DCleanProjectedResidualPostFusionCFFNTransition,
    S2DJointPathResidualPostFusionCFFNTransition,
)
from lnet.pac_path_cffn import IdentityQuadrantPathModeCombiner

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "P4-PathRes"
VARIANTS = (VARIANT,)
SEEDS = (501,)
PATH_HIDDEN = 128
heads = clean.heads


def _replace_transition(stage: nn.Module) -> None:
    previous = stage.augmented
    if not isinstance(previous, S2DCleanProjectedResidualPostFusionCFFNTransition):
        message = "P4-PathRes requires the clean projected-residual source transition"
        raise TypeError(message)
    transition = S2DJointPathResidualPostFusionCFFNTransition(
        modes=stage.modes,
        path_hidden_modes=PATH_HIDDEN,
        output_modes=previous.output_modes,
        pole_paths=4,
        post_hidden_modes=previous.post_hidden_modes,
        path_activation="cartesian_silu",
        post_ffn_activation=previous.post_ffn_activation,
    )
    transition.copy_retained_state_from(previous)
    stage.augmented = transition


def _assert_pathres(model: nn.Module) -> None:
    clean.projected.stemres._assert_stem(model)
    banks = clean.projected.stemres.joint.p4._pole_banks(model)
    for bank in banks:
        if bank.product_gain_normalization != "global":
            message = "P4-PathRes requires global finite-grid gain normalization"
            raise RuntimeError(message)
    for stage in banks[:-1]:
        combiner = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        if not isinstance(combiner, IdentityQuadrantPathModeCombiner) or tuple(
            combiner.parameters()
        ):
            message = "P4-PathRes retained a pre-transition path combiner"
            raise RuntimeError(message)
        if (
            not isinstance(
                transition,
                S2DJointPathResidualPostFusionCFFNTransition,
            )
            or transition.input_modes != 256
            or transition.output_modes != 64
            or transition.path_hidden_modes != PATH_HIDDEN
            or not isinstance(transition.path_norm, ComplexRMSNorm)
            or transition.path_input.input_modes != 256
            or transition.path_input.output_modes != 128
            or transition.path_output.input_modes != 128
            or transition.path_output.output_modes != 256
            or transition.compression.input_modes != 256
            or transition.compression.output_modes != 64
            or transition.pole_input is not None
            or transition.pole_output is not None
            or transition.pole_scale is not None
            or transition.post_ffn_scale is not None
            or float(transition.path_residual_scale) != 1.0
            or float(transition.post_residual_scale) != 1.0
        ):
            message = "P4-PathRes transition contract changed"
            raise RuntimeError(message)
        if stage.output_modes is None:
            message = "P4-PathRes left the fused product-only scan path"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported P4 path-residual variant: {variant}"
        raise ValueError(message)
    model = clean._build(clean.VARIANT, config)
    for stage in clean.projected.stemres.joint.p4._pole_banks(model)[:-1]:
        _replace_transition(stage)
    _assert_pathres(model)
    return model


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = heads._wandb_model_metrics(model)
    for stage_index, stage in enumerate(
        clean.projected.stemres.joint.p4._pole_banks(model)[:-1],
        start=1,
    ):
        transition = stage.augmented
        if not isinstance(transition, S2DJointPathResidualPostFusionCFFNTransition):
            message = "P4-PathRes lost its requested stage transition"
            raise TypeError(message)
        prefix = f"pathres/stage{stage_index}"
        metrics[f"{prefix}/path_input_rms"] = float(
            transition.path_input.weight_real.detach().float().square().mean().sqrt()
        )
        metrics[f"{prefix}/path_output_rms"] = float(
            transition.path_output.weight_real.detach().float().square().mean().sqrt()
        )
        metrics[f"{prefix}/compression_rms"] = float(
            transition.compression.weight_real.detach().float().square().mean().sqrt()
        )
        if transition.carry_weight is not None:
            metrics[f"{prefix}/carry_mean"] = float(transition.carry_weight.detach().float().mean())
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = clean._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][clean.VARIANT])
    variant_config["backbone"]["path_contract"].update(
        {
            "joint_path_residual": (
                "complex_rmsnorm_then_256_to_128_silu_128_to_256_plus_identity"
            ),
            "joint_path_residual_output_initialization": "exact_zero",
            "path_compression": "widely_linear_256_to_64_after_joint_residual",
            "path_cffn": "none_identity_handoff_before_joint_residual",
            "execution": "shared_widely_linear_and_complex_ffn_dispatchers",
            "stage_merge": "s2d_carry_plus_compressed_joint_path_state",
            "stage_merge_scale": "none",
            "post_refinement": ("complex_rmsnorm_then_64_to_128_silu_128_to_64_unit_residual"),
            "post_residual_scale": "fixed_one",
        }
    )
    payload.update(
        {
            "schema": "lnet.a2d.p4_pathres.imagenet100.v1",
            "evidence_status": "joint path-mode residual before compression ablation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "GlobalGain raw-Q P4-StemRes with four product states, "
                    "a zero-initialized pre-norm 256-to-128-to-256 joint residual, "
                    "one 256-to-64 compression, unit S2D stage merge, and a "
                    "pre-normalized unit 64-to-128-to-64 PostFusion residual."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_p4_pathres_runner"] = heads.harness._digest(Path(__file__))
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
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
