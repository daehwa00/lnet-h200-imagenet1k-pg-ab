#!/usr/bin/env python3
"""Train shared excitation with persistent D4 side-memory across scales."""

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_h96_path_pg_modewise_complex_linear_imagenet100 as control  # noqa: E501
import torch
from torch import nn

from lnet.pac_common_persistent_directional import (
    CommonExcitationTerminal,
    CommonPersistentStage,
    InitialCommonPersistentStage,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "PGv2-H96-CommonE-PersistentR4"
VARIANTS = (VARIANT,)
SEEDS = control.SEEDS
P = control.P
MODE_HIDDEN = control.MODE_HIDDEN
PATH_HIDDEN = control.PATH_HIDDEN


def _assert_model(model: ComplexScanBackbone) -> None:
    if not isinstance(model.stage1, InitialCommonPersistentStage):
        message = "common/persistent model is missing its initial state split"
        raise TypeError(message)
    for name in ("stage2", "stage3"):
        if not isinstance(getattr(model, name), CommonPersistentStage):
            message = f"{name} is not a common-excitation/persistent-memory stage"
            raise TypeError(message)
    if not isinstance(model.terminal, CommonExcitationTerminal):
        message = "common/persistent model changed its terminal excitation readout"
        raise TypeError(message)
    for name in ("stage1", "stage2", "stage3"):
        transition = getattr(model, name).transition
        if (
            transition.mode.modes != P
            or transition.mode.hidden_modes != MODE_HIDDEN
            or transition.path.modes != 4
            or transition.path.hidden_modes != PATH_HIDDEN
            or tuple(transition.readout.weight_real.shape) != (P, 4)
            or transition.post.modes != P
            or transition.post.hidden_modes != MODE_HIDDEN
        ):
            message = f"{name} changed the H96/H8/MWCL/H96 transition contract"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported common/persistent variant: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    model.stage1 = InitialCommonPersistentStage(  # pyright: ignore[reportAttributeAccessIssue]
        model.stage1
    )
    model.stage2 = CommonPersistentStage(model.stage2)  # pyright: ignore[reportAttributeAccessIssue]
    model.stage3 = CommonPersistentStage(model.stage3)
    model.terminal = CommonExcitationTerminal(  # pyright: ignore[reportAttributeAccessIssue]
        model.terminal
    )
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = "A2D-PGv2-H96-CommonE-PersistentR4"
    transition = payload["backbone"]["stage_transition"]
    transition["state"] = {
        "common_excitation": "E_s in C^96",
        "directional_memory": "R_s in C^(4x96)",
    }
    transition["fresh_observation"] = "H_s=D4Scan(E_s)"
    transition["memory_update"] = (
        "R_(s+1)=PathPG(ModePG(H_s+direction-preserving-S2D(R_s)))"
    )
    transition["excitation_update"] = (
        "E_(s+1)=PostPG(MWCL4to1(R_(s+1))+mode-wise-S2D(E_s))"
    )
    transition["memory_fusion_parameters"] = "shared existing S2D carry weights"
    transition["path_collapse_role"] = "consensus readout only; R remains uncollapsed"
    transition["persistent_shapes"] = ["28x28x4x96", "14x14x4x96", "7x7x4x96"]
    transition["terminal"] = "fresh D4 Q from shared E4; R4 already contributes through E4"
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
        metrics[f"persistent/stage{index}/readout_weight_rms"] = float(
            transition.readout.weight_real.detach().float().square().mean().sqrt()
        )
        metrics[f"persistent/stage{index}/memory_carry_weight_rms"] = float(
            transition.carry_weight.detach().float().square().mean().sqrt()
        )
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    ramp = control.base.base.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pgv2_h96.common_e_persistent_r4.imagenet100.v1"
    payload["evidence_status"] = "untrained common-excitation persistent-D4 candidate"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Each stage scans one shared E_s into four fresh directional measurements H_s. "
            "A direction-preserving S2D carry of R_s is fused before H96 Mode PG and H8 "
            "Path PG, producing uncollapsed R_(s+1). The existing mode-wise strict 4-to-1 "
            "readout is used only to form shared E_(s+1), together with S2D(E_s) and H96 "
            "Post PG. Terminal Q remains four fresh measurements of shared E4."
        )
    }
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pac_common_persistent_directional"] = digest(
        Path("src/lnet/pac_common_persistent_directional.py")
    )
    payload["source_sha256"]["common_persistent_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    control.base.base._configure_ramp()
    ramp = control.base.base.control.control.stemres.uniform.base
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
