#!/usr/bin/env python3
"""Train A2D-D4-PathMix with a post-CFFN 48-mode S2D carry-main merge."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import run_a2d_d4_pathmix_imagenet100 as baseline
import run_double_prefc_imagenet100 as a2d_base

from lnet.complex_scan import (
    AugmentedComplexTransition,
    S2DPostCFFNCarryMainTransition,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import (
        ComplexScanBackbone,
        ComplexScanConfig,
    )


VARIANT = "a2d_postcarry"
POLE_SCALE_INITIAL = 1.0


def _variant_config(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanConfig:
    return baseline._variant_config(
        baseline.VARIANT if variant == VARIANT else variant,
        config,
    )


def _replace_transition(stage: nn.Module) -> None:
    previous = stage.augmented
    if not isinstance(previous, AugmentedComplexTransition):
        message = "A2D PostCarry requires the established augmented transition"
        raise TypeError(message)
    if stage.carry_basis != "s2d":
        message = "A2D PostCarry requires S2D carry coordinates"
        raise RuntimeError(message)
    transition = S2DPostCFFNCarryMainTransition(
        modes=stage.modes,
        hidden_modes=previous.hidden_modes,
        output_modes=previous.output_modes,
        pole_paths=previous.input_modes // stage.modes,
        expansion=previous.ffn_input.output_modes // previous.hidden_modes,
        pole_scale_initial=POLE_SCALE_INITIAL,
    )
    transition.copy_pole_branch_from(previous)
    stage.augmented = transition


def _build(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    model = baseline._build(
        baseline.VARIANT if variant == VARIANT else variant,
        config,
    )
    if variant == VARIANT:
        _replace_transition(model.stage1)
        _replace_transition(model.stage2)
    return model


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for stage_index, stage in enumerate((model.stage1, model.stage2), start=1):
        transition = stage.augmented
        if not isinstance(transition, S2DPostCFFNCarryMainTransition):
            message = "A2D PostCarry lost its requested transition"
            raise TypeError(message)
        prefix = f"postcarry/stage{stage_index}"
        metrics[f"{prefix}/alpha"] = float(transition.pole_scale.detach())
        if transition.carry_weight is not None:
            weights = transition.carry_weight.detach().float()
            metrics[f"{prefix}/carry_mean"] = float(weights.mean())
            metrics[f"{prefix}/carry_min"] = float(weights.min())
            metrics[f"{prefix}/carry_max"] = float(weights.max())
        elif transition.carry_projection is not None:
            weights = transition.carry_projection.weight_real.detach().float()
            metrics[f"{prefix}/carry_projection_rms"] = float(weights.square().mean().sqrt())
    return metrics


def _contract(args: Namespace) -> dict[str, object]:
    payload = baseline._contract(args)
    residuals = a2d_base.residuals
    base_config = residuals.ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    active = _variant_config(VARIANT, base_config)
    model = _build(VARIANT, base_config)
    payload["variant_configs"][VARIANT] = residuals.asdict(active)
    payload["variant_configs"][VARIANT]["precomplex_fc_widths"] = [96, 96, 96]
    payload["variant_configs"][VARIANT]["precomplex_fc_activations"] = [
        "gelu",
        "identity",
    ]
    payload["variant_configs"][VARIANT]["stage_transition"] = {
        "pole": (
            "WL192-96, ComplexRMSNorm, residual Cartesian-SiLU "
            "CFFN96-192-96, ComplexRMSNorm, WL96-48"
        ),
        "carry": (
            "S2D reshape to 4x48 followed by a real mode-wise 4-to-1 "
            "projection initialized to average; no bias, norm, activation, "
            "conjugate, or cross-mode mixing"
        ),
        "merge": "C48 + alpha_stage * F48",
        "alpha_shape": [],
        "alpha_initial": POLE_SCALE_INITIAL,
        "alpha_weight_decay": 0.0,
    }
    payload["parameter_counts"][VARIANT] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    payload["architecture"][VARIANT] = (
        "A2D-D4-PathMix with scan, endpoint coarsening, factorized PathMix, "
        "raw directional descriptors, and Fusion384+LRQ64 head retained.  The pole "
        "branch completes the established augmented CFFN and 96-to-48 "
        "projection independently; a bias-free real mode-wise S2D 4-to-1 "
        "carry is then the main state, and a trainable stage scalar initialized "
        "to one multiplies the completed 48-mode pole update."
    )
    payload["source_sha256"]["a2d_d4_postcarry_runner"] = residuals.harness._digest(
        Path(__file__)
    )
    return json.loads(json.dumps(payload))


def main() -> None:
    residuals = a2d_base.residuals
    residuals.main(
        variants=(VARIANT,),
        build_model=_build,
        contract=_contract,
        wandb_model_metrics=_wandb_model_metrics,
    )


if __name__ == "__main__":
    main()
