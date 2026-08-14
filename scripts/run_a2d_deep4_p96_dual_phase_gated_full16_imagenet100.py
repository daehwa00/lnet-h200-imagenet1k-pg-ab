#!/usr/bin/env python3
"""Train Raw16 with residual phase-gated interaction on path and mode axes."""

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

from lnet.pac_full_state_transition import Full16DualPhaseGatedCollapse
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage


VARIANT = "D4-PGv2-H192-Raw16-DualPG"
VARIANTS = (VARIANT,)
SEEDS = base.SEEDS
P = base.P
MODE_HIDDEN = base.MODE_HIDDEN
PATH_HIDDEN = 32
COMPILE_MODE = "max-autotune-no-cudagraphs"


def _install_dual_pg(stage: ComplexScanStage) -> None:
    baseline = stage.quadrant_path_mode_combiner
    baseline_mode = getattr(baseline, "mode", None)
    if not isinstance(baseline_mode, PhaseGatedComplexFFN):
        message = "dual-PG transition requires the validated PGv2 baseline"
        raise TypeError(message)
    mixer = Full16DualPhaseGatedCollapse(
        P,
        mode_hidden=MODE_HIDDEN,
        path_hidden=PATH_HIDDEN,
    )
    mixer.mode.load_state_dict(baseline_mode.state_dict())
    stage.quadrant_path_mode_combiner = mixer


def _assert_model(model: ComplexScanBackbone) -> None:
    for name in ("stage1", "stage2", "stage3"):
        mixer = getattr(model, name).quadrant_path_mode_combiner
        if not isinstance(mixer, Full16DualPhaseGatedCollapse):
            message = f"{name} is missing its dual-PG transition"
            raise TypeError(message)
        if (
            mixer.path_mode.modes != 16
            or mixer.path_mode.hidden_modes != PATH_HIDDEN
            or mixer.mode.modes != P
            or mixer.mode.hidden_modes != MODE_HIDDEN
            or mixer.path_collapse.input_paths != 16
            or mixer.path_collapse.output_paths != 1
        ):
            message = f"{name} changed the dual-PG transition contract"
            raise RuntimeError(message)
    if model.terminal.quadrant_path_mode_combiner is not None:
        message = "dual-PG experiment changed the terminal descriptor"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported dual-PG variant: {variant}"
        raise ValueError(message)
    model = base._build(base.VARIANT, config)
    for name in ("stage1", "stage2", "stage3"):
        _install_dual_pg(getattr(model, name))
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(base._variant_config())
    payload["backbone"]["name"] = "A2D-PGv2-H192-Raw16-DualPG"
    transition = payload["backbone"]["stage_transition"]
    transition["coarsening"] = {
        "cell": "full normalized 2x2 product state",
        "direction_order": ["+x+y", "-x+y", "+x-y", "-x-y"],
        "local_order": ["q00", "q10", "q01", "q11"],
        "state_shape": "B,h,w,16-path,96-mode",
    }
    transition["path_residual"] = "shared PhaseGatedComplexFFNv2-16-32-16"
    transition["mode_residual"] = "shared PhaseGatedComplexFFNv2-96-192-96"
    transition["path_collapse"] = "grouped-widely-linear-16-1"
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
        if isinstance(mixer, Full16DualPhaseGatedCollapse):
            for axis, block in (("path", mixer.path_mode), ("mode", mixer.mode)):
                for metric, value in block.diagnostic_metrics().items():
                    metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
                for metric, value in block.gradient_metrics().items():
                    metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = base._contract(args)
    payload["recipe"]["compile_mode"] = COMPILE_MODE
    config = base.control.stemres.uniform.base.PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pg_dual_axis_full16.imagenet100.v1"
    payload["evidence_status"] = "untrained Raw16 dual-PG information-bound candidate"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Raw direction-relative 2x2 scan cells retain 16 complex states. A shared "
            "16-32-16 residual Phase-Gated block mixes local/path memory per mode, then "
            "the validated shared 96-192-96 residual Phase-Gated block mixes modes per "
            "state. Only the final per-mode GWL canonicalizes 16 paths to one."
        )
    }
    digest = base.control.stemres.uniform.base.heads.harness._digest
    payload["source_sha256"]["pac_full_state_transition"] = digest(
        Path("src/lnet/pac_full_state_transition.py")
    )
    payload["source_sha256"]["dual_pg_full16_runner"] = digest(Path(__file__))
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
