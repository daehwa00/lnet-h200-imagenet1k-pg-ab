#!/usr/bin/env python3
"""Train PoleLR1 with a decoupled, axis-paired 96-pole initialization."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_pgv2_h96_k3_rmsmatch_q4_affine_polelr1_imagenet100 as base
import torch

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage


VARIANT = "PGv2-H96-K3-RMSMatch-Q4Affine-PoleLR1-DecoupledInit"
VARIANTS = (VARIANT,)
SEEDS = base.SEEDS
ORIENTATIONS = 8
DAMPING_OFFSETS = (0, 3, 7, 10, 10, 7, 3, 0)
LOG_ANISOTROPY = (-0.20, -0.14, -0.08, -0.03, 0.03, 0.08, 0.14, 0.20)
DAMPING_INDEX_MULTIPLIER = 5


def _configure_ramp() -> None:
    base._configure_ramp()
    ramp = base.local_reader.control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _damping(stage: ComplexScanStage, axis: str) -> torch.Tensor:
    logits = getattr(stage, f"damping_logits_{axis}")
    return stage.damping_min + (stage.damping_max - stage.damping_min) * logits.sigmoid()


def _damping_logits(stage: ComplexScanStage, damping: torch.Tensor) -> torch.Tensor:
    ratio = ((damping - stage.damping_min) / (stage.damping_max - stage.damping_min)).clamp(
        1.0e-4,
        1.0 - 1.0e-4,
    )
    return torch.logit(ratio)


def _install_decoupled_initialization(stage: ComplexScanStage) -> None:
    if stage.modes % ORIENTATIONS:
        message = "decoupled pole initialization requires complete orientation groups"
        raise ValueError(message)
    radial_levels = stage.modes // ORIENTATIONS
    if math.gcd(DAMPING_INDEX_MULTIPLIER, radial_levels) != 1:
        message = "damping permutation multiplier must be coprime with radial levels"
        raise ValueError(message)

    original_x = _damping(stage, "x").detach().reshape(radial_levels, ORIENTATIONS)
    original_y = _damping(stage, "y").detach().reshape(radial_levels, ORIENTATIONS)
    if not torch.equal(original_x, original_y) or not torch.allclose(
        original_x,
        original_x[:, :1].expand_as(original_x),
        rtol=0.0,
        atol=1.0e-7,
    ):
        message = "decoupled initialization requires the calibrated isotropic radial atlas"
        raise RuntimeError(message)

    radial = torch.arange(radial_levels, device=original_x.device).view(-1, 1)
    offsets = torch.tensor(DAMPING_OFFSETS, device=original_x.device).view(1, -1)
    damping_indices = (DAMPING_INDEX_MULTIPLIER * radial + offsets) % radial_levels
    base_damping = original_x[:, 0][damping_indices]
    anisotropy = torch.tensor(
        LOG_ANISOTROPY,
        dtype=base_damping.dtype,
        device=base_damping.device,
    ).view(1, -1)
    damping_x = base_damping * anisotropy.exp()
    damping_y = base_damping * (-anisotropy).exp()
    if damping_x.min() <= stage.damping_min or damping_y.min() <= stage.damping_min:
        message = "decoupled initialization crossed the minimum damping bound"
        raise RuntimeError(message)
    if damping_x.max() >= stage.damping_max or damping_y.max() >= stage.damping_max:
        message = "decoupled initialization crossed the maximum damping bound"
        raise RuntimeError(message)

    with torch.no_grad():
        stage.damping_logits_x.copy_(_damping_logits(stage, damping_x.flatten()))
        stage.damping_logits_y.copy_(_damping_logits(stage, damping_y.flatten()))


def _assert_initialization(model: ComplexScanBackbone) -> None:
    for name in ("stage1", "stage2", "stage3", "terminal"):
        stage = getattr(model, name)
        radial_levels = stage.modes // ORIENTATIONS
        damping_x = _damping(stage, "x").reshape(radial_levels, ORIENTATIONS)
        damping_y = _damping(stage, "y").reshape(radial_levels, ORIENTATIONS)
        if not torch.allclose(damping_x, damping_y.flip(1), rtol=2.0e-6, atol=1.0e-7):
            message = f"{name} lost its exact x/y paired symmetry"
            raise RuntimeError(message)
        if torch.equal(damping_x, damping_y):
            message = f"{name} remained isotropic after decoupled initialization"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported decoupled-init variant: {variant}"
        raise ValueError(message)
    model = base._build(base.VARIANT, config)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        _install_decoupled_initialization(getattr(model, name))
    _configure_ramp()
    base._assert_model(model)
    _assert_initialization(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(base._variant_config())
    payload["backbone"]["name"] = "A2D-PGv2-H96-K3-RMSMatch-DecoupledPoleInit"
    payload["backbone"]["pole_initialization"] = {
        "modes_per_stage": base.P,
        "radial_levels": base.P // ORIENTATIONS,
        "orientations": ORIENTATIONS,
        "frequency_atlas": "unchanged calibrated 12-radial by 8-orientation atlas",
        "damping_index_rule": (
            f"(5 * radial_index + orientation_offset) mod 12; offsets={list(DAMPING_OFFSETS)}"
        ),
        "log_anisotropy_by_orientation": list(LOG_ANISOTROPY),
        "axis_pairing": "orientation i swaps damping_x/y with orientation 7-i",
        "preserves": (
            "all 96 frequency vectors, the damping-level multiset, geometric-mean "
            "memory range, parameter count, and exact global x/y symmetry"
        ),
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = base._contract(args)
    ramp = base.local_reader.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = (
        "lnet.a2d.pgv2_h96.k3_rmsmatch_q4_affine_polelr1_decoupled_init.imagenet100.v1"
    )
    payload["evidence_status"] = "untrained deterministic decoupled-pole initialization ablation"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact PoleLR1 base model with only the initial damping atlas changed. "
            "Frequencies are unchanged; damping levels are deterministically decoupled "
            "from radial frequency and assigned paired x/y anisotropy."
        )
    }
    payload["source_sha256"]["decoupled_init_runner"] = ramp.heads.harness._digest(Path(__file__))
    return payload


def main() -> None:
    _configure_ramp()
    ramp = base.local_reader.control.control.control.stemres.uniform.base
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
            build_optimizer=base._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=base.local_reader.control.control._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
