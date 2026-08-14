#!/usr/bin/env python3
"""Train H96 PG stages with a mode-wise strict complex-linear D4 collapse."""

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_h96_path_pg_complex_linear_imagenet100 as base
import torch
from torch import nn

from lnet.pac_modewise_path_collapse import (
    PhaseGatedModePathResidualModeWiseCollapse,
)
from lnet.pac_phase_gated_transition import (
    PhaseGatedModePathResidualComplexLinearCollapse,
    PhaseGatedS2DPostFusionTransition,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage


VARIANT = "PGv2-H96-PathPG-MWCL-PostPG"
VARIANTS = (VARIANT,)
SEEDS = base.SEEDS
P = base.P
MODE_HIDDEN = base.MODE_HIDDEN
PATH_HIDDEN = base.PATH_HIDDEN


def _replace_collapse(stage: ComplexScanStage) -> None:
    source = stage.quadrant_path_mode_combiner
    if type(source) is not PhaseGatedModePathResidualComplexLinearCollapse:
        message = "mode-wise collapse requires the exact shared-CL control"
        raise TypeError(message)
    replacement = PhaseGatedModePathResidualModeWiseCollapse(
        P,
        mode_hidden=MODE_HIDDEN,
        path_hidden=PATH_HIDDEN,
    )
    replacement.mode.load_state_dict(source.mode.state_dict())
    replacement.path.load_state_dict(source.path.state_dict())
    stage.quadrant_path_mode_combiner = replacement


def _assert_model(model: ComplexScanBackbone) -> None:
    base.base.control.control.stemres._assert_stem(model)
    for name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        if type(mixer) is not PhaseGatedModePathResidualModeWiseCollapse:
            message = f"{name} is missing its mode-wise strict collapse"
            raise TypeError(message)
        if (
            mixer.mode.modes != P
            or mixer.mode.hidden_modes != MODE_HIDDEN
            or mixer.path.modes != 4
            or mixer.path.hidden_modes != PATH_HIDDEN
            or tuple(mixer.collapse.weight_real.shape) != (P, 4)
            or not isinstance(stage.augmented, PhaseGatedS2DPostFusionTransition)
            or stage.augmented.post.hidden_modes != MODE_HIDDEN
        ):
            message = f"{name} changed the mode-wise collapse contract"
            raise RuntimeError(message)
    if model.terminal.quadrant_path_mode_combiner is not None:
        message = "mode-wise collapse experiment changed the terminal descriptor"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported mode-wise collapse variant: {variant}"
        raise ValueError(message)
    model = base._build(base.VARIANT, config)
    for name in ("stage1", "stage2", "stage3"):
        _replace_collapse(getattr(model, name))
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(base._variant_config())
    payload["backbone"]["name"] = "A2D-PGv2-H96-PathPG-MWCL-PostPG"
    transition = payload["backbone"]["stage_transition"]
    transition["path_collapse"] = "mode-wise-strict-ComplexLinear-4-1"
    transition["path_collapse_equation"] = "y[m]=sum_d c[m,d]*z[m,d]"
    transition["path_collapse_initialization"] = "per-mode exact-mean: 0.25+0i"
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
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        if isinstance(mixer, PhaseGatedModePathResidualModeWiseCollapse):
            blocks = (("mode", mixer.mode), ("path", mixer.path), ("post", stage.augmented.post))
            for axis, block in blocks:
                for metric, value in block.diagnostic_metrics().items():
                    metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
                for metric, value in block.gradient_metrics().items():
                    metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = base._contract(args)
    config = base.base.control.control.stemres.uniform.base.PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pgv2_h96.path_pg_modewise_cl.imagenet100.v1"
    payload["evidence_status"] = "untrained mode-wise strict-collapse candidate"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact PGv2-H96-PathPG-CL-PostPG control with only the shared strict "
            "ComplexLinear D4 collapse replaced by independent strict complex 4-to-1 "
            "coefficients for each of 96 pole modes. Mode PG, Path PG, S2D carry, "
            "Post PG, terminal, head, optimizer, learning rate, and augmentation are unchanged."
        )
    }
    digest = base.base.control.control.stemres.uniform.base.heads.harness._digest
    payload["source_sha256"]["pac_modewise_path_collapse"] = digest(
        Path("src/lnet/pac_modewise_path_collapse.py")
    )
    payload["source_sha256"]["modewise_collapse_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    base.base._configure_ramp()
    ramp = base.base.control.control.stemres.uniform.base
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
