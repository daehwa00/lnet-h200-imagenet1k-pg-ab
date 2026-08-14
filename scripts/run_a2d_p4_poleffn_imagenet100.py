#!/usr/bin/env python3
"""Train product-only P4 with an unnormalized pole FFN stage update."""

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
    ComplexRMSNorm,
    ComplexScanConfig,
    S2DProjectedResidualPostFusionCFFNTransition,
    S2DUnnormalizedPolePostFusionCFFNTransition,
)
from lnet.pac_path_cffn import IdentityQuadrantPathModeCombiner

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "P4-PoleFFN"
VARIANTS = (VARIANT,)
SEEDS = (501,)
POLE_HIDDEN = 128
heads = projected.heads


def _replace_transition(stage: nn.Module) -> None:
    previous = stage.augmented
    if not isinstance(previous, S2DProjectedResidualPostFusionCFFNTransition):
        message = "P4-PoleFFN requires the projected-residual source transition"
        raise TypeError(message)
    transition = S2DUnnormalizedPolePostFusionCFFNTransition(
        modes=stage.modes,
        pole_hidden_modes=POLE_HIDDEN,
        output_modes=previous.output_modes,
        pole_paths=4,
        post_hidden_modes=previous.post_hidden_modes,
        pole_activation="cartesian_silu",
        post_ffn_activation=previous.post_ffn_activation,
    )
    transition.copy_retained_state_from(previous)
    stage.augmented = transition


def _assert_poleffn(model: nn.Module) -> None:
    projected.stemres._assert_stem(model)
    banks = projected.stemres.joint.p4._pole_banks(model)
    for bank in banks:
        if bank.product_gain_normalization != "global":
            message = "P4-PoleFFN requires global finite-grid gain normalization"
            raise RuntimeError(message)
    for stage in banks[:-1]:
        combiner = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        if not isinstance(combiner, IdentityQuadrantPathModeCombiner) or tuple(
            combiner.parameters()
        ):
            message = "P4-PoleFFN retained a pre-transition path combiner"
            raise RuntimeError(message)
        if (
            not isinstance(transition, S2DUnnormalizedPolePostFusionCFFNTransition)
            or transition.input_modes != 256
            or transition.output_modes != 64
            or transition.pole_hidden_modes != POLE_HIDDEN
            or transition.pole_input.input_modes != 256
            or transition.pole_input.output_modes != 128
            or transition.pole_output.input_modes != 128
            or transition.pole_output.output_modes != 64
            or transition.ffn_norm is not None
            or transition.output_norm is not None
            or transition.pole_scale is not None
            or transition.post_ffn_scale is not None
            or not isinstance(transition.post_ffn_norm, ComplexRMSNorm)
            or float(transition.post_residual_scale) != 1.0
        ):
            message = "P4-PoleFFN transition contract changed"
            raise RuntimeError(message)
        if stage.output_modes is None:
            message = "P4-PoleFFN left the fused product-only scan path"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported P4 PoleFFN variant: {variant}"
        raise ValueError(message)
    model = projected._build(projected.VARIANT, config)
    for stage in projected.stemres.joint.p4._pole_banks(model)[:-1]:
        _replace_transition(stage)
    _assert_poleffn(model)
    return model


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = heads._wandb_model_metrics(model)
    for stage_index, stage in enumerate(
        projected.stemres.joint.p4._pole_banks(model)[:-1],
        start=1,
    ):
        transition = stage.augmented
        if not isinstance(transition, S2DUnnormalizedPolePostFusionCFFNTransition):
            message = "P4-PoleFFN lost its requested stage transition"
            raise TypeError(message)
        prefix = f"poleffn/stage{stage_index}"
        metrics[f"{prefix}/pole_input_rms"] = float(
            transition.pole_input.weight_real.detach().float().square().mean().sqrt()
        )
        metrics[f"{prefix}/pole_output_rms"] = float(
            transition.pole_output.weight_real.detach().float().square().mean().sqrt()
        )
        if transition.carry_weight is not None:
            metrics[f"{prefix}/carry_mean"] = float(transition.carry_weight.detach().float().mean())
    return metrics


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
            "descriptor": "raw_directional_product_energy",
            "gain_normalization": "global_finite_grid_mean",
            "long_range_update": ("widely_linear_256_to_128_cartesian_silu_128_to_64"),
            "long_range_pre_norm": "none",
            "long_range_shortcut": "none",
            "stage_merge": "s2d_carry_plus_pole_update_unit_coefficients",
            "post_refinement": ("complex_rmsnorm_then_64_to_128_cartesian_silu_128_to_64"),
            "post_residual_scale": "fixed_one",
        }
    )
    payload.update(
        {
            "schema": "lnet.a2d.p4_poleffn.imagenet100.v1",
            "evidence_status": "unnormalized pole-update stage ablation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "GlobalGain raw-Q P4-StemRes with four product states, "
                    "parameter-free identity handoff, an unnormalized "
                    "256-to-128-to-64 Cartesian-SiLU pole update, unit S2D stage "
                    "residual, and a pre-normalized unit 64-to-128-to-64 "
                    "PostFusion residual."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_p4_poleffn_runner"] = heads.harness._digest(Path(__file__))
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
