#!/usr/bin/env python3
"""Train the PGv2 control with only the mode hidden width reduced to 96."""

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_imagenet100 as control
import torch

from lnet.pac_phase_gated_transition import PhaseGatedModeResidualPathCollapse

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanStage


VARIANT = "D4-Cal-U96-Stem32-MPM8-PGv2-H96-Aux05"
VARIANTS = (VARIANT,)
SEEDS = control.SEEDS
P = control.P
MODE_HIDDEN = 96
PATH_HIDDEN = control.PATH_HIDDEN


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _replace_mode(stage: ComplexScanStage) -> None:
    baseline = stage.quadrant_path_mode_combiner
    if not isinstance(baseline, PhaseGatedModeResidualPathCollapse):
        message = "PGv2-H96 requires the exact PGv2 mode control"
        raise TypeError(message)
    replacement = PhaseGatedModeResidualPathCollapse(
        P,
        mode_hidden=MODE_HIDDEN,
        path_hidden=PATH_HIDDEN,
    )
    replacement.copy_path_from(baseline)
    stage.quadrant_path_mode_combiner = replacement


def _assert_model(model: ComplexScanBackbone) -> None:
    for name in ("stage1", "stage2", "stage3"):
        mixer = getattr(model, name).quadrant_path_mode_combiner
        if not isinstance(mixer, PhaseGatedModeResidualPathCollapse):
            message = f"{name} is missing its Phase-Gated mode transition"
            raise TypeError(message)
        if (
            mixer.mode.modes != P
            or mixer.mode.hidden_modes != MODE_HIDDEN
            or mixer.mode.input_projection.input_modes != P
            or mixer.mode.input_projection.output_modes != 2 * MODE_HIDDEN
            or mixer.mode.output_projection.input_modes != MODE_HIDDEN
            or mixer.mode.output_projection.output_modes != P
        ):
            message = f"{name} changed the PGv2-H96 projection contract"
            raise RuntimeError(message)


def _build(
    variant: str,
    config: control.control.stemres.uniform.base.PoleModelConfig,
) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported PGv2-H96 variant: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    for name in ("stage1", "stage2", "stage3"):
        _replace_mode(getattr(model, name))
    _configure_ramp()
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = "A2D-Cal-U96-Stem32-MPM8-PGv2-H96-Aux05"
    transition = payload["backbone"]["stage_transition"]
    transition["mode_residual"] = f"PhaseGatedComplexFFNv2-{P}-{MODE_HIDDEN}-{P}"
    transition["mode_gate"]["gate_hidden_width"] = MODE_HIDDEN
    transition["mode_gate"]["input_projection_complex_width"] = 2 * MODE_HIDDEN
    transition["mode_gate"]["value_hidden_width"] = MODE_HIDDEN
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    config = control.control.stemres.uniform.base.PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.cal_u96_mpm8.phase_gated_mode_h96.imagenet100.v1"
    payload["evidence_status"] = "untrained PGv2 mode-width ablation"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact PGv2-H192-All3e-3 control with only the Stage 1-3 Phase-Gated "
            "mode block reduced from H192 to H96: packed input projection 96-to-192 "
            "(u96 plus v96), followed by output projection 96-to-96. Path, post-fusion, "
            "stem, poles, classifier, optimizer, learning rate, and augmentation are unchanged."
        )
    }
    digest = control.control.stemres.uniform.base.heads.harness._digest
    payload["source_sha256"]["phase_gated_h96_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    _configure_ramp()
    ramp = control.control.stemres.uniform.base
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
            wandb_model_metrics=control._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
