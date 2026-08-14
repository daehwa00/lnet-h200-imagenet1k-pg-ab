#!/usr/bin/env python3
"""Compare RMS-matched point reading with residual-gated local reading."""

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

from lnet.pac_complex_scan_reader import (
    PackedComplexConv2dReader,
    ResidualGatedComplexConv2dReader,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage


K1_RMS_VARIANT = "PGv2-H96-K1-RMSMatch"
RESIDUAL_K3_VARIANT = "PGv2-H96-K3-ResidualGate"
VARIANT = K1_RMS_VARIANT
VARIANTS = (K1_RMS_VARIANT, RESIDUAL_K3_VARIANT)
SEEDS = control.SEEDS
P = control.P
LOCAL_KERNEL_SIZE = 3
RESIDUAL_SCALE_INITIAL = 0.01
RESIDUAL_SCALE_MAXIMUM = 0.5


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _make_reader(variant: str) -> torch.nn.Module:
    if variant == K1_RMS_VARIANT:
        reader = PackedComplexConv2dReader(
            P,
            P,
            kernel_size=1,
            match_input_rms=True,
        )
    elif variant == RESIDUAL_K3_VARIANT:
        reader = ResidualGatedComplexConv2dReader(
            P,
            kernel_size=LOCAL_KERNEL_SIZE,
            residual_scale_init=RESIDUAL_SCALE_INITIAL,
            residual_scale_max=RESIDUAL_SCALE_MAXIMUM,
        )
    else:
        message = f"unsupported PGv2-H96 scan-reader follow-up: {variant}"
        raise ValueError(message)
    reader.initialize_identity_()
    return reader


def _install_reader(stage: ComplexScanStage, variant: str) -> None:
    if stage.modes != P:
        message = "PGv2-H96 scan-reader follow-up requires unchanged uniform modes"
        raise RuntimeError(message)
    with torch.random.fork_rng(devices=[]):
        stage.pole_input_projection = _make_reader(variant)


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    control._assert_model(model)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        reader = getattr(model, name).pole_input_projection
        if variant == K1_RMS_VARIANT:
            if (
                not isinstance(reader, PackedComplexConv2dReader)
                or reader.input_modes != P
                or reader.output_modes != P
                or reader.kernel_size != 1
                or not reader.match_input_rms
            ):
                message = f"{name} changed the K1 RMS-matched reader contract"
                raise RuntimeError(message)
        elif (
            not isinstance(reader, ResidualGatedComplexConv2dReader)
            or reader.modes != P
            or reader.kernel_size != LOCAL_KERNEL_SIZE
            or reader.candidate.input_modes != P
            or reader.candidate.output_modes != P
            or not reader.candidate.match_input_rms
            or reader.residual_scale_max != RESIDUAL_SCALE_MAXIMUM
        ):
            message = f"{name} changed the residual-gated K3 reader contract"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant not in VARIANTS:
        message = f"unsupported PGv2-H96 scan-reader follow-up: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        _install_reader(getattr(model, name), variant)
    _configure_ramp()
    _assert_model(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = f"A2D-{variant}"
    if variant == K1_RMS_VARIANT:
        pole_input = {
            "operator": "unit-row strict-complex K1 convolution",
            "shape": f"{P}-to-{P} before every scan",
            "initialization": "exact identity",
            "gain": "per-token shared RMS(output)=RMS(input)",
            "purpose": "separate vector reading and RMS control from K3 spatial context",
        }
    else:
        pole_input = {
            "candidate": "unit-row strict-complex K3 with per-token input RMS matching",
            "candidate_shape": f"{P}-to-{P} before every scan",
            "gate": "fixed CRMSNorm then learned mean-one magnitude gate",
            "merge": "U=E+gamma*g*(candidate-E)",
            "gamma_initial": RESIDUAL_SCALE_INITIAL,
            "gamma_bound": RESIDUAL_SCALE_MAXIMUM,
            "initialization": "exact identity",
            "purpose": "preserve the control excitation while adding detected local evidence",
        }
    pole_input["scope"] = "scan branch only; carry, transition, Q1536, and head unchanged"
    payload["backbone"]["pole_input"] = pole_input
    return payload


def _selected_variants(args: Namespace) -> tuple[str, ...]:
    requested = tuple(getattr(args, "variants", ()))
    return requested or VARIANTS


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    selected = _selected_variants(args)
    ramp = control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    models = {variant: _build(variant, config) for variant in selected}
    payload["schema"] = "lnet.a2d.pgv2_h96.scan_reader_followup.imagenet100.v1"
    payload["evidence_status"] = "controlled scalar pole-drive sufficiency comparison"
    payload["variants"] = list(selected)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {
        variant: _variant_config(variant) for variant in selected
    }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture"] = {
        K1_RMS_VARIANT: (
            "Exact PGv2-H96 control plus an identity-initialized RMS-matched K1 "
            "strict-complex reader before each D4 scan."
        ),
        RESIDUAL_K3_VARIANT: (
            "Exact PGv2-H96 control plus an identity-preserving residual K3 reader. "
            "The RMS-matched local candidate is converted to relative evidence by a "
            "mean-one magnitude gate and added with a bounded learned scalar."
        ),
    }
    payload["recipe"]["scan_reader_optimizer"] = {
        "kernel_learning_rate": payload["recipe"]["learning_rate"],
        "kernel_weight_decay": payload["recipe"]["weight_decay"],
        "gate_and_gamma_learning_rate": payload["recipe"]["learning_rate"],
        "gate_and_gamma_weight_decay": 0.0,
    }
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["complex_scan_reader"] = digest(
        Path("src/lnet/pac_complex_scan_reader.py")
    )
    payload["source_sha256"]["mean_one_magnitude_gate"] = digest(
        Path("src/lnet/pac_mean_one_magnitude_gate.py")
    )
    payload["source_sha256"]["scan_reader_followup_runner"] = digest(Path(__file__))
    return payload


@torch.no_grad()
def _reader_metrics(reader: ResidualGatedComplexConv2dReader) -> dict[str, float]:
    values = reader.gate.diagnostic_metrics()
    values.update(reader.gate.gradient_metrics())
    values["residual_scale"] = float(reader.effective_residual_scale())
    if reader.gamma.grad is not None:
        values["residual_scale_grad"] = float(reader.gamma.grad)
    return values


def _wandb_model_metrics(model: torch.nn.Module) -> dict[str, float]:
    metrics = control.control._wandb_model_metrics(model)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        reader = getattr(model, name).pole_input_projection
        if not isinstance(reader, ResidualGatedComplexConv2dReader):
            continue
        metrics.update(
            {
                f"scan_reader/{name}/{metric}": value
                for metric, value in _reader_metrics(reader).items()
            }
        )
    return metrics


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
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
