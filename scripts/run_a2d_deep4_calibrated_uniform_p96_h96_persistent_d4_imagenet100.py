#!/usr/bin/env python3
"""Train PGv2-H96 while preserving the four D4 memories across scales."""

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_h96_path_pg_complex_linear_imagenet100 as control
import torch
from torch import nn

from lnet.pac_persistent_directional_scan import (
    InitialPersistentD4Stage,
    MatchingPersistentD4Stage,
    TerminalMatchingPersistentD4Stage,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "PGv2-H96-PersistentD4"
VARIANTS = (VARIANT,)
SEEDS = control.SEEDS
P = control.P
MODE_HIDDEN = control.MODE_HIDDEN
PATH_HIDDEN = control.PATH_HIDDEN


def _assert_model(model: ComplexScanBackbone) -> None:
    if not isinstance(model.stage1, InitialPersistentD4Stage):
        message = "persistent D4 model is missing its initial D4 stage"
        raise TypeError(message)
    for name in ("stage2", "stage3"):
        stage = getattr(model, name)
        if not isinstance(stage, MatchingPersistentD4Stage):
            message = f"{name} is not a matching-direction persistent stage"
            raise TypeError(message)
    if not isinstance(model.terminal, TerminalMatchingPersistentD4Stage):
        message = "persistent D4 model is missing its matching terminal stage"
        raise TypeError(message)
    for name in ("stage1", "stage2", "stage3"):
        transition = getattr(model, name).transition
        if (
            transition.mode.modes != P
            or transition.mode.hidden_modes != MODE_HIDDEN
            or transition.path.modes != 4
            or transition.path.hidden_modes != PATH_HIDDEN
            or transition.post.modes != P
            or transition.post.hidden_modes != MODE_HIDDEN
        ):
            message = f"{name} changed the requested PGv2-H96/H8/H96 contract"
            raise RuntimeError(message)
        if any("collapse" in child_name for child_name, _ in transition.named_modules()):
            message = f"{name} retained a forbidden direction-collapse module"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported persistent D4 variant: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    model.stage1 = InitialPersistentD4Stage(model.stage1)  # pyright: ignore[reportAttributeAccessIssue]
    model.stage2 = MatchingPersistentD4Stage(model.stage2)  # pyright: ignore[reportAttributeAccessIssue]
    model.stage3 = MatchingPersistentD4Stage(model.stage3)
    model.terminal = TerminalMatchingPersistentD4Stage(  # pyright: ignore[reportAttributeAccessIssue]
        model.terminal
    )
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = "A2D-PGv2-H96-PersistentD4"
    transition = payload["backbone"]["stage_transition"]
    transition["direction_memory"] = "persistent (++,-+,+-,--) across all scales"
    transition["stage1_scan"] = "one fused D4 associative product scan"
    transition["later_scans"] = "four matching-direction associative product scans"
    transition["mode_residual"] = "PhaseGatedComplexFFNv2-96-96-96"
    transition["path_residual"] = "shared-PhaseGatedComplexFFNv2-4-8-4"
    transition["path_collapse"] = "none"
    transition.pop("path_collapse_initialization", None)
    transition["carry"] = "direction-preserving shared mode-wise S2D"
    transition["postfusion"] = "PhaseGatedComplexFFNv2-96-96-96 per direction"
    transition["persistent_shapes"] = ["28x28x4x96", "14x14x4x96", "7x7x4x96"]
    transition["removed_operators"] = [
        "WidelyLinear",
        "GroupedWidelyLinear",
        "ComplexLinear-4-to-1",
    ]
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
        transition = getattr(model, name).transition
        for axis, block in (
            ("mode", transition.mode),
            ("path", transition.path),
            ("post", transition.post),
        ):
            for metric, value in block.diagnostic_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
            for metric, value in block.gradient_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    ramp = control.base.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pgv2_h96.persistent_d4.imagenet100.v1"
    payload["evidence_status"] = "untrained persistent-direction memory candidate"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact PGv2-H96 PathPG/CL/PostPG control with ComplexLinear 4-to-1 removed. "
            "Stage 1 creates D4 once; Stage 2, Stage 3, and terminal apply only the scan "
            "matching each persistent direction. Mode PG H96, Path PG H8, mode-wise S2D "
            "carry, and Post PG H96 preserve the four-direction memory axis."
        )
    }
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pac_persistent_directional_scan"] = digest(
        Path("src/lnet/pac_persistent_directional_scan.py")
    )
    payload["source_sha256"]["persistent_d4_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    control.base._configure_ramp()
    ramp = control.base.control.control.stemres.uniform.base
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
