#!/usr/bin/env python3
"""Train A2D PostCarry with a joint 48-96-48 post-fusion residual CFFN."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import run_a2d_d4_postcarry_imagenet100 as postcarry
import run_double_prefc_imagenet100 as a2d_base

from lnet.complex_scan import (
    S2DPostCFFNCarryMainTransition,
    S2DPostFusionCFFNTransition,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import (
        ComplexScanBackbone,
        ComplexScanConfig,
    )


VARIANT = "a2d_postcarry_postffn"
POLE_SCALE_INITIAL = 1.0
POST_HIDDEN_MODES = 96
POST_LAYER_SCALE_INITIAL = 0.1


def _variant_config(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanConfig:
    return postcarry._variant_config(
        postcarry.VARIANT if variant == VARIANT else variant,
        config,
    )


def _replace_transition(stage: nn.Module) -> None:
    previous = stage.augmented
    if not isinstance(previous, S2DPostCFFNCarryMainTransition):
        message = "post-fusion CFFN requires the matched PostCarry transition"
        raise TypeError(message)
    transition = S2DPostFusionCFFNTransition(
        modes=stage.modes,
        hidden_modes=previous.hidden_modes,
        output_modes=previous.output_modes,
        pole_paths=previous.input_modes // stage.modes,
        expansion=previous.ffn_input.output_modes // previous.hidden_modes,
        pole_scale_initial=POLE_SCALE_INITIAL,
        post_hidden_modes=2 * previous.output_modes,
        post_layer_scale_initial=POST_LAYER_SCALE_INITIAL,
    )
    transition.copy_pole_branch_from(previous)
    if transition.carry_weight is not None and previous.carry_weight is not None:
        transition.carry_weight.data.copy_(previous.carry_weight.data)
    elif transition.carry_projection is not None and previous.carry_projection is not None:
        transition.carry_projection.load_state_dict(previous.carry_projection.state_dict())
    else:
        message = "PostCarry and PostFFN carry projections do not match"
        raise TypeError(message)
    transition.pole_scale.data.copy_(previous.pole_scale.data)
    stage.augmented = transition


def _build(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    model = postcarry._build(
        postcarry.VARIANT if variant == VARIANT else variant,
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
        if not isinstance(transition, S2DPostFusionCFFNTransition):
            message = "A2D PostCarry PostFFN lost its requested transition"
            raise TypeError(message)
        prefix = f"postcarry_postffn/stage{stage_index}"
        if transition.pole_scale is not None:
            metrics[f"{prefix}/alpha"] = float(transition.pole_scale.detach())
        beta = transition.post_ffn_scale.detach().float()
        metrics[f"{prefix}/beta_mean"] = float(beta.mean())
        metrics[f"{prefix}/beta_min"] = float(beta.min())
        metrics[f"{prefix}/beta_max"] = float(beta.max())
        if transition.carry_weight is not None:
            carry = transition.carry_weight.detach().float()
            metrics[f"{prefix}/carry_mean"] = float(carry.mean())
        elif transition.carry_projection is not None:
            carry = transition.carry_projection.weight_real.detach().float()
            metrics[f"{prefix}/carry_projection_rms"] = float(carry.square().mean().sqrt())
    return metrics


def _contract(args: Namespace) -> dict[str, object]:
    payload = postcarry.baseline._contract(args)
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
        "outer_residual": "H48 = C48 + alpha_stage * F48",
        "carry": "matched mode-wise real S2D four-to-one projection",
        "pole": "matched completed PostCarry pole branch",
        "post_fusion": ("H48 + beta_mode * WL96-48(CartesianSiLU(WL48-96(CRMSNorm48(H48))))"),
        "alpha_initial": POLE_SCALE_INITIAL,
        "beta_shape": [48],
        "beta_initial": POST_LAYER_SCALE_INITIAL,
        "post_add_operation": "none",
    }
    payload["parameter_counts"][VARIANT] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    payload["architecture"][VARIANT] = (
        "A2D-D4-PathMix PostCarry with its projection residual retained exactly, "
        "followed by a separate pre-normalized widely-linear 48-96-48 Cartesian-"
        "SiLU mode CFFN residual with mode-wise LayerScale."
    )
    payload["source_sha256"]["a2d_d4_postcarry_postffn_runner"] = residuals.harness._digest(
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
