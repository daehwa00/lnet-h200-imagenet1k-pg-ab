#!/usr/bin/env python3
"""Train the uniform-P96 Stem32 model with automatic MPM8 transitions."""

# pyright: reportArgumentType=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_deep4_calibrated_uniform_p96_stemres_imagenet100 as stemres
import torch
from torch import nn

from lnet.pac_factorized_stage_transition import (
    FactorizedS2DPostFusionTransition,
    ModeResidualPathCollapse,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanStage


VARIANT = "D4-Cal-U96-Stem32-MPM8"
VARIANTS = (VARIANT,)
SEEDS = stemres.SEEDS
P = stemres.P
PATH_HIDDEN = 8


def _configure_ramp() -> None:
    stemres._configure_ramp()
    ramp = stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _install_mpm8(stage: ComplexScanStage) -> None:
    """Install the MPM8 semantics; kernel selection stays inside the modules."""
    if stage.modes != P or stage.output_modes != P or stage.carry_basis != "s2d":
        message = "MPM8 requires a uniform-P96 S2D stage"
        raise RuntimeError(message)
    stage.quadrant_path_mode_combiner = ModeResidualPathCollapse(
        P,
        mode_hidden=2 * P,
        path_hidden=PATH_HIDDEN,
    )
    stage.augmented = FactorizedS2DPostFusionTransition(P, post_hidden=2 * P)


def _assert_model(model: ComplexScanBackbone) -> None:
    stemres._assert_stem(model)
    for name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, name)
        if not isinstance(stage.quadrant_path_mode_combiner, ModeResidualPathCollapse):
            message = f"{name} is missing its MPM8 path collapse"
            raise TypeError(message)
        if not isinstance(stage.augmented, FactorizedS2DPostFusionTransition):
            message = f"{name} is missing its MPM8 post-fusion transition"
            raise TypeError(message)
        if any(
            branch is not None
            for branch in (
                stage.transition,
                stage.interaction,
                stage.widely_bridge,
                stage.bridge,
                stage.post_transition_ffn,
            )
        ):
            message = f"{name} retained a competing transition"
            raise RuntimeError(message)
    if model.terminal.output_modes is not None:
        message = "MPM8 changed the terminal descriptor stage"
        raise RuntimeError(message)


def _build(
    variant: str,
    config: stemres.uniform.base.PoleModelConfig,
) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported MPM8 variant: {variant}"
        raise ValueError(message)
    _configure_ramp()
    model = cast("ComplexScanBackbone", stemres._RAMP_BUILD(variant, config))
    model.stem = stemres.ModeScaledTwoConvStem(
        P,
        config.stem_strides,
        hidden_width=stemres.STEM_HIDDEN_WIDTH,
    )
    precomplex_fc = model.precomplex_fc
    if precomplex_fc is None:
        message = "MPM8 requires the established pre-complex mixer"
        raise TypeError(message)
    model.precomplex_fc = stemres.ResidualPreComplexMixer(precomplex_fc)
    for name in ("stage1", "stage2", "stage3"):
        _install_mpm8(getattr(model, name))
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(stemres._variant_config())
    payload["backbone"]["name"] = "A2D-Cal-U96-Stem32-MPM8"
    payload["backbone"]["stage_transition"] = {
        "mode_residual": f"{P}-{2 * P}-{P}",
        "path_collapse": f"grouped-4-{PATH_HIDDEN}-1",
        "postfusion": f"{P}-{2 * P}-{P}",
    }
    return payload


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    return {
        "model/parameters": float(sum(parameter.numel() for parameter in model.parameters())),
        "model/trainable_parameters": float(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def _contract(args: Namespace) -> dict[str, Any]:
    payload = stemres.uniform._contract(args)
    config = stemres.uniform.base.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.deep4_cal_u96_stem32_mpm8.imagenet100.v1"
    payload["evidence_status"] = "untrained automatic MPM8 candidate"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Uniform-P96 Stem32 with automatic stage1-3 mode residual, "
            "grouped per-mode 4-to-8-to-1 path collapse, S2D carry merge, "
            "and post-fusion residual; the terminal descriptor is unchanged."
        )
    }
    payload["source_sha256"]["a2d_u96_mpm8_runner"] = (
        stemres.uniform.base.heads.harness._digest(Path(__file__))
    )
    return payload


def main() -> None:
    _configure_ramp()
    ramp = stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = ramp.backbone.a2d_base.residuals
    harness = source.heads.harness
    source.heads.VARIANTS = VARIANTS
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    bindings = getattr(harness, "runner_bindings", None)
    if not callable(bindings):
        message = "MPM8 requires the shared runner bindings"
        raise TypeError(message)
    harness.main(
        bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ramp.PoleModelConfig,
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
