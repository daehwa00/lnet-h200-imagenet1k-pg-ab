#!/usr/bin/env python3
"""Train P4-StemRes with a projected joint residual at each stage handoff."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_p4_stemres_imagenet100 as stemres
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexScanConfig,
    S2DDirectPostFusionCFFNTransition,
    S2DProjectedResidualPostFusionCFFNTransition,
)
from lnet.pac_path_cffn import (
    IdentityQuadrantPathModeCombiner,
    JointPathModeCFFNCombiner,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.pac_complex_layers import WidelyLinear


VARIANT = "P4-ProjRes"
VARIANTS = (VARIANT,)
SEEDS = (501,)
JOINT_HIDDEN = 128
heads = stemres.heads


def _copy_projection_rows(
    destination: WidelyLinear,
    destination_slice: slice,
    source: WidelyLinear,
) -> None:
    """Copy one output block without changing its established initialization."""
    destination_width = destination_slice.stop - destination_slice.start
    if source.input_modes != destination.input_modes or source.output_modes != destination_width:
        message = "projected residual initialization has incompatible affine widths"
        raise ValueError(message)
    with torch.no_grad():
        for name in (
            "weight_real",
            "weight_imag",
            "conjugate_real",
            "conjugate_imag",
        ):
            getattr(destination, name)[destination_slice].copy_(getattr(source, name))
        if (
            destination.bias_real is None
            or destination.bias_imag is None
            or source.bias_real is None
            or source.bias_imag is None
        ):
            message = "projected residual initialization requires affine biases"
            raise TypeError(message)
        destination.bias_real[destination_slice].copy_(source.bias_real)
        destination.bias_imag[destination_slice].copy_(source.bias_imag)


def _replace_stage_composition(stage: nn.Module) -> None:
    previous_combiner = stage.quadrant_path_mode_combiner
    previous_transition = stage.augmented
    if not isinstance(previous_combiner, JointPathModeCFFNCombiner):
        message = "P4-ProjRes requires the P4-Joint path-mode CFFN"
        raise TypeError(message)
    if not isinstance(previous_transition, S2DDirectPostFusionCFFNTransition):
        message = "P4-ProjRes requires the direct P4-Joint transition"
        raise TypeError(message)

    transition = S2DProjectedResidualPostFusionCFFNTransition(
        modes=stage.modes,
        joint_hidden_modes=JOINT_HIDDEN,
        output_modes=previous_transition.output_modes,
        pole_paths=4,
        joint_layer_scale_initial=float(previous_combiner.layer_scale.detach().mean()),
        pole_scale_initial=float(previous_transition.pole_scale.detach()),
        post_hidden_modes=previous_transition.post_hidden_modes,
        post_layer_scale_initial=float(previous_transition.post_ffn_scale.detach().mean()),
        post_ffn_activation=previous_transition.post_ffn_activation,
    )
    transition.copy_retained_state_from(previous_transition)
    base_stop = transition.output_modes
    _copy_projection_rows(
        transition.joint_input,
        slice(0, base_stop),
        previous_transition.output_projection,
    )
    _copy_projection_rows(
        transition.joint_input,
        slice(base_stop, base_stop + JOINT_HIDDEN),
        previous_combiner.input_projection,
    )
    stage.quadrant_path_mode_combiner = IdentityQuadrantPathModeCombiner(
        stage.modes,
    )
    stage.augmented = transition


def _assert_projected(model: nn.Module) -> None:
    stemres._assert_stem(model)
    banks = stemres.joint.p4._pole_banks(model)
    for stage in banks[:-1]:
        combiner = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        if (
            not isinstance(combiner, IdentityQuadrantPathModeCombiner)
            or combiner.path_count != 4
            or combiner.modes != 64
            or sum(parameter.numel() for parameter in combiner.parameters()) != 0
        ):
            message = "P4-ProjRes stage retained a pre-projection path CFFN"
            raise RuntimeError(message)
        if (
            not isinstance(transition, S2DProjectedResidualPostFusionCFFNTransition)
            or transition.input_modes != 256
            or transition.output_modes != 64
            or transition.joint_hidden_modes != JOINT_HIDDEN
            or transition.joint_input.input_modes != 256
            or transition.joint_input.output_modes != 192
            or transition.joint_output.input_modes != 128
            or transition.joint_output.output_modes != 64
            or transition.direction_mixer is not None
            or transition.ffn_input is not None
            or transition.ffn_output is not None
            or transition.output_projection is not None
        ):
            message = "P4-ProjRes stage does not implement 256-to-(64+128)-to-64"
            raise RuntimeError(message)
        if stage.output_modes is None:
            message = "P4-ProjRes left the fused product-only scan path"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported P4 projected-residual variant: {variant}"
        raise ValueError(message)
    model = stemres._build(stemres.VARIANT, config)
    for stage in stemres.joint.p4._pole_banks(model)[:-1]:
        _replace_stage_composition(stage)
    _assert_projected(model)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = stemres._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][stemres.VARIANT])
    variant_config["backbone"]["path_contract"].update(
        {
            "mode_cffn": "projected_joint_residual",
            "path_cffn": "none_identity_handoff",
            "projected_joint_input": "256_to_192_split_64_base_plus_128_hidden",
            "projected_joint_output": "128_to_64",
            "stage_transition": "base64_plus_nonlinear64_then_s2d_postfusion",
        }
    )
    payload.update(
        {
            "schema": "lnet.a2d.p4_projected.imagenet100.v1",
            "evidence_status": "P4-StemRes projected joint residual ablation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "P4-StemRes with parameter-free four-path handoff and one packed "
                    "256-to-(64 base + 128 hidden) projection followed by Cartesian "
                    "SiLU and a direct 128-to-64 residual branch per non-terminal "
                    "stage; S2D carry and the 64-to-128-to-64 PostFusion CFFN remain."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_p4_projected_runner"] = heads.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def main() -> None:
    calibrated = stemres.joint.p4.calibrated
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
