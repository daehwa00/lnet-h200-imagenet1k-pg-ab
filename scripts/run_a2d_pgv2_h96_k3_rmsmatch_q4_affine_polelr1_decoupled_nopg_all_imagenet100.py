#!/usr/bin/env python3
"""Train DecoupledInit after removing every transition mode PG."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_pgv2_h96_k3_rmsmatch_q4_affine_polelr1_decoupled_nostage1pg_imagenet100 as stage1_ablation
import torch

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


control = stage1_ablation.control
VARIANT = "PGv2-H96-K3-RMSMatch-Q4Affine-PoleLR1-DecoupledInit-NoPG-All"
VARIANTS = (VARIANT,)
SEEDS = control.SEEDS
_STAGE_NAMES = ("stage1", "stage2", "stage3")


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = control.base.local_reader.control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _remove_all_mode_pg(model: ComplexScanBackbone) -> None:
    for stage_name in _STAGE_NAMES:
        stage = getattr(model, stage_name)
        mixer = stage.quadrant_path_mode_combiner
        if not isinstance(mixer, stage1_ablation.PhaseGatedModeResidualPathCollapse):
            message = f"DecoupledInit {stage_name} lost its phase-gated path-collapse contract"
            raise TypeError(message)
        stage.quadrant_path_mode_combiner = stage1_ablation.PathOnlyCollapse(mixer)


def _assert_model(model: ComplexScanBackbone) -> None:
    control.base._assert_model(model)
    control._assert_initialization(model)
    for stage_name in _STAGE_NAMES:
        mixer = getattr(model, stage_name).quadrant_path_mode_combiner
        if not isinstance(mixer, stage1_ablation.PathOnlyCollapse):
            message = f"{stage_name} mode PG removal was not installed"
            raise TypeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported DecoupledInit no-PG variant: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    _remove_all_mode_pg(model)
    _configure_ramp()
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = "A2D-PGv2-H96-K3-RMSMatch-DecoupledPoleInit-NoPG-All"
    payload["backbone"]["stage1_mode_processing"] = "identity; mode PG removed"
    payload["backbone"]["stage2_mode_processing"] = "identity; mode PG removed"
    payload["backbone"]["stage3_mode_processing"] = "identity; mode PG removed"
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    ramp = control.base.local_reader.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = (
        "lnet.a2d.pgv2_h96.k3_rmsmatch.q4_affine.polelr1.decoupled_init.no_pg_all.imagenet100.v1"
    )
    payload["evidence_status"] = "untrained causal removal of every transition mode PG"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact DecoupledInit control with the Stage-1, Stage-2, and Stage-3 "
            "PhaseGatedComplexFFN modules deleted. Every GWL 4-to-8-to-1 path "
            "collapse, reader, pole, transition, Q4 descriptor, and affine head "
            "is unchanged."
        )
    }
    payload["source_sha256"]["no_pg_all_runner"] = ramp.heads.harness._digest(Path(__file__))
    return payload


def main() -> None:
    _configure_ramp()
    ramp = control.base.local_reader.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
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
            build_optimizer=control.base._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=control.base.local_reader.control.control._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
