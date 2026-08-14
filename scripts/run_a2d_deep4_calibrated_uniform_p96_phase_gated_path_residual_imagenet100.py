#!/usr/bin/env python3
"""Train PGv2-H192 with a residual path PG and final learned D4 collapse."""

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_imagenet100 as base
import torch
from torch import nn

from lnet.pac_factorized_stage_transition import FactorizedS2DPostFusionTransition
from lnet.pac_phase_gated_transition import (
    PhaseGatedModePathResidualGWLCollapse,
    PhaseGatedModeResidualPathCollapse,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage


VARIANT = "PGv2-H192-PathPG8-GWL4to1"
VARIANTS = (VARIANT,)
SEEDS = base.SEEDS
P = base.P
MODE_HIDDEN = base.MODE_HIDDEN
PATH_HIDDEN = base.PATH_HIDDEN


def _install_path_residual(stage: ComplexScanStage) -> None:
    baseline = stage.quadrant_path_mode_combiner
    if not isinstance(baseline, PhaseGatedModeResidualPathCollapse):
        message = "path-residual transition requires the exact PGv2-H192 control"
        raise TypeError(message)
    mixer = PhaseGatedModePathResidualGWLCollapse(
        P,
        mode_hidden=MODE_HIDDEN,
        path_hidden=PATH_HIDDEN,
    )
    mixer.copy_mode_from(baseline)
    stage.quadrant_path_mode_combiner = mixer


def _assert_model(model: ComplexScanBackbone) -> None:
    base.control.stemres._assert_stem(model)
    for name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        if not isinstance(mixer, PhaseGatedModePathResidualGWLCollapse):
            message = f"{name} is missing its residual path PG transition"
            raise TypeError(message)
        if (
            mixer.mode.modes != P
            or mixer.mode.hidden_modes != MODE_HIDDEN
            or mixer.path.modes != 4
            or mixer.path.hidden_modes != PATH_HIDDEN
            or mixer.collapse.input_paths != 4
            or mixer.collapse.output_paths != 1
            or type(stage.augmented) is not FactorizedS2DPostFusionTransition
        ):
            message = f"{name} changed the requested transition contract"
            raise RuntimeError(message)
    if model.terminal.quadrant_path_mode_combiner is not None:
        message = "path-residual experiment changed the terminal descriptor"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported path-residual variant: {variant}"
        raise ValueError(message)
    model = base._build(base.VARIANT, config)
    for name in ("stage1", "stage2", "stage3"):
        _install_path_residual(getattr(model, name))
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(base._variant_config())
    payload["backbone"]["name"] = "A2D-PGv2-H192-PathPG8-GWL4to1"
    transition = payload["backbone"]["stage_transition"]
    transition["mode_residual"] = "PhaseGatedComplexFFNv2-96-192-96"
    transition["path_residual"] = "shared-PhaseGatedComplexFFNv2-4-8-4"
    transition["path_collapse"] = "GroupedWidelyLinear-4-1"
    transition["post_fusion"] = "unchanged-MPM8-96-192-96"
    return payload


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = {
        "model/parameters": float(sum(parameter.numel() for parameter in model.parameters())),
        "model/trainable_parameters": float(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    metrics.update(
        {
            f"train/{name}": float(value)
            for name, value in getattr(model, "_latest_training_diagnostics", {}).items()
        }
    )
    for index, name in enumerate(("stage1", "stage2", "stage3"), start=1):
        mixer = getattr(model, name).quadrant_path_mode_combiner
        if isinstance(mixer, PhaseGatedModePathResidualGWLCollapse):
            for axis, block in (("mode", mixer.mode), ("path", mixer.path)):
                for metric, value in block.diagnostic_metrics().items():
                    metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
                for metric, value in block.gradient_metrics().items():
                    metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = base._contract(args)
    config = base.control.stemres.uniform.base.PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pg_path_residual_gwl.imagenet100.v1"
    payload["evidence_status"] = "untrained residual path-PG candidate"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact PGv2-H192-All3e-3 control through mode mixing, followed by a shared "
            "Phase-Gated v2 residual over the four D4 paths (4-8-4) and a final "
            "per-mode GroupedWidelyLinear 4-to-1 collapse. S2D carry, PostFusion, "
            "terminal descriptor, head, optimizer, and training recipe are unchanged."
        )
    }
    digest = base.control.stemres.uniform.base.heads.harness._digest
    payload["source_sha256"]["pac_phase_gated_transition"] = digest(
        Path("src/lnet/pac_phase_gated_transition.py")
    )
    payload["source_sha256"]["path_residual_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    base._configure_ramp()
    ramp = base.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = ramp.backbone.a2d_base.residuals
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
