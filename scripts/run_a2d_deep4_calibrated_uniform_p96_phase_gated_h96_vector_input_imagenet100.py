#!/usr/bin/env python3
"""Train PGv2-H96 with one identity-initialized pole-input coupling per scan."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_imagenet100 as control
import torch

from lnet.pac_complex_layers import PackedComplexLinear

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage


VARIANT = "PGv2-H96-VectorInput96"
VARIANTS = (VARIANT,)
SEEDS = control.SEEDS
P = control.P


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _install_vector_input(stage: ComplexScanStage) -> None:
    if stage.modes != P:
        message = "PGv2-H96 vector input requires an unchanged uniform-P96 stage"
        raise RuntimeError(message)
    # Preserve the control's subsequent data-order and augmentation RNG stream.
    # The random constructor values are immediately replaced by exact identity.
    with torch.random.fork_rng(devices=[]):
        projection = PackedComplexLinear(P, P)
    with torch.no_grad():
        projection.weight_real.copy_(torch.eye(P))
        projection.weight_imag.zero_()
    stage.pole_input_projection = projection


def _assert_model(model: ComplexScanBackbone) -> None:
    control._assert_model(model)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        stage = getattr(model, name)
        projection = stage.pole_input_projection
        if (
            not isinstance(projection, PackedComplexLinear)
            or projection.input_modes != P
            or projection.output_modes != P
        ):
            message = f"{name} changed the strict 96-to-96 pole-input contract"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported PGv2-H96 vector-input variant: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        _install_vector_input(getattr(model, name))
    _configure_ramp()
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = "A2D-PGv2-H96-VectorInput96"
    payload["backbone"]["pole_input"] = {
        "operator": "identity-initialized strict PackedComplexLinear",
        "shape": "96-to-96 before every Stage1-3 and terminal scan",
        "scope": "scan branch only; local S2D carry remains unchanged",
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    ramp = control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pgv2_h96.vector_input96.imagenet100.v1"
    payload["evidence_status"] = "single-change PGv2-H96 vector-input experiment"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact PGv2-H96-All3e-3 stem, transition, carry, Q1536, head, optimizer, "
            "and recipe. The sole change is an identity-initialized strict complex "
            "96-to-96 coupling on each scan branch so every pole can learn to read the "
            "complete excitation vector; S2D carry still receives the original input."
        )
    }
    payload["recipe"]["pole_input_optimizer"] = {
        "learning_rate": payload["recipe"]["learning_rate"],
        "weight_decay": payload["recipe"]["weight_decay"],
        "selection": "standard two-dimensional AdamW weight group",
    }
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["complex_scan_stage"] = digest(
        Path("src/lnet/complex_scan_stage.py")
    )
    payload["source_sha256"]["pgv2_h96_vector_input_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    _configure_ramp()
    ramp = control.control.control.stemres.uniform.base
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
            wandb_model_metrics=control.control._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
