#!/usr/bin/env python3
"""Train lossless Raw16 and pole-aligned Innov16 PGv2 transitions."""

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

from lnet.pac_full_state_transition import Full16PhaseGatedModeResidualPathCollapse
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage
    from lnet.pac_full_state_transition import FullStateBasis


RAW16 = "D4-PGv2-H192-Raw16"
INNOV16 = "D4-PGv2-H192-Innov16"
VARIANTS = (RAW16, INNOV16)
SEEDS = base.SEEDS
P = base.P
MODE_HIDDEN = base.MODE_HIDDEN
PATH_HIDDEN = 32
COMPILE_MODE = "max-autotune-no-cudagraphs"


def _basis(variant: str) -> FullStateBasis:
    if variant == RAW16:
        return "raw"
    if variant == INNOV16:
        return "innovation"
    message = f"unsupported full-state variant: {variant}"
    raise ValueError(message)


def _install_full_state_mode(stage: ComplexScanStage, *, basis: FullStateBasis) -> None:
    baseline = stage.quadrant_path_mode_combiner
    baseline_mode = getattr(baseline, "mode", None)
    if not isinstance(baseline_mode, PhaseGatedComplexFFN):
        message = "full-state transition requires the validated PGv2 baseline"
        raise TypeError(message)
    mixer = Full16PhaseGatedModeResidualPathCollapse(
        P,
        mode_hidden=MODE_HIDDEN,
        basis=basis,
        path_hidden=PATH_HIDDEN,
    )
    mixer.mode.load_state_dict(baseline_mode.state_dict())
    stage.quadrant_path_mode_combiner = mixer


def _assert_full_state_model(model: ComplexScanBackbone, *, basis: FullStateBasis) -> None:
    for name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        if not isinstance(mixer, Full16PhaseGatedModeResidualPathCollapse):
            message = f"{name} is missing its full-state transition"
            raise TypeError(message)
        if (
            mixer.basis != basis
            or mixer.mode.hidden_modes != MODE_HIDDEN
            or mixer.path_count != 16
            or mixer.path_input.input_paths != 16
            or mixer.path_input.output_paths != PATH_HIDDEN
            or mixer.path_output.input_paths != PATH_HIDDEN
            or mixer.path_output.output_paths != 16
            or mixer.path_collapse.input_paths != 16
            or mixer.path_collapse.output_paths != 1
        ):
            message = f"{name} changed the full-state transition contract"
            raise RuntimeError(message)
    if model.terminal.quadrant_path_mode_combiner is not None:
        message = "full-state experiment changed the terminal descriptor"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    basis = _basis(variant)
    model = base._build(base.VARIANT, config)
    for name in ("stage1", "stage2", "stage3"):
        _install_full_state_mode(getattr(model, name), basis=basis)
    _assert_full_state_model(model, basis=basis)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    basis = _basis(variant)
    payload = deepcopy(base._variant_config())
    payload["backbone"]["name"] = f"A2D-PGv2-H192-{variant.rsplit('-', maxsplit=1)[-1]}"
    transition = payload["backbone"]["stage_transition"]
    transition["coarsening"] = {
        "cell": "full normalized 2x2 product state",
        "direction_order": ["+x+y", "-x+y", "+x-y", "-x-y"],
        "local_order": ["q00", "q10", "q01", "q11"],
        "legacy_endpoint_local_index": 3,
        "state_shape": "B,h,w,4-direction,4-local,96-mode",
    }
    transition["full_state_basis"] = (
        "raw direction-relative cells"
        if basis == "raw"
        else "detached-pole invertible (M,Ix,Iy,Ixy) difference basis"
    )
    transition["mode_residual"] = "shared PhaseGatedComplexFFNv2-96-192-96 over 16 states"
    transition["path_collapse"] = "grouped-16-32-16-residual-then-16-1"
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
        if isinstance(mixer, Full16PhaseGatedModeResidualPathCollapse):
            for metric, value in mixer.mode.diagnostic_metrics().items():
                metrics[f"phase_gated/stage{index}/mode_{metric}"] = value
            for metric, value in mixer.mode.gradient_metrics().items():
                metrics[f"phase_gated/stage{index}/mode_{metric}"] = value
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = base._contract(args)
    # Full16 keeps four times as many Stage 1 states alive. CUDA Graph capture
    # duplicates those saved tensors in a private pool and exceeds a 24 GiB
    # device at the matched batch size. Keep Inductor/Triton max-autotuning,
    # but make the memory-safe runtime part of this experiment's contract so
    # callers do not need a launch-time override.
    payload["recipe"]["compile_mode"] = COMPILE_MODE
    config = base.control.stemres.uniform.base.PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    models = {variant: _build(variant, config) for variant in VARIANTS}
    payload["schema"] = "lnet.a2d.pg_full16.imagenet100.v1"
    payload["evidence_status"] = "untrained full-state information-bound candidates"
    payload["variants"] = list(VARIANTS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in VARIANTS}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture"] = {
        RAW16: (
            "PGv2-H192 baseline with lossless direction-relative 2x2 scan cells, one shared "
            "PG block over 16 states, and grouped 16-32-16 residual mixing before collapse."
        ),
        INNOV16: (
            "Raw16 with the same parameters and execution graph except for an invertible "
            "detached-pole (M,Ix,Iy,Ixy) basis before the shared PG block."
        ),
    }
    digest = base.control.stemres.uniform.base.heads.harness._digest
    for name in (
        "complex_scan_stage.py",
        "pac_directional.py",
        "pac_full_state_transition.py",
        "pac_product_scan_pipeline.py",
        "pac_product_scan_reference.py",
        "pac_triton_product_scan_coarse4.py",
    ):
        payload["source_sha256"][name.removesuffix(".py")] = digest(Path("src/lnet") / name)
    payload["source_sha256"]["full16_runner"] = digest(Path(__file__))
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
