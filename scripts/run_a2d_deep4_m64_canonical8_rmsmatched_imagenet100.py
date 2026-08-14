#!/usr/bin/env python3
"""Train D4-M64-C8 with fixed per-CFFN RMS-matched radial activations."""

# ruff: noqa: I001, SLF001

from __future__ import annotations

import run_a2d_resaux1_imagenet100 as resaux_base

import run_double_prefc_imagenet100 as a2d_base

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

import calibrate_a2d_m64_radial_scales_imagenet100 as calibration
import run_a2d_deep4_m64_canonical8_calibrated_imagenet100 as calibrated
import run_a2d_deep4_m64_canonical8_radial_imagenet100 as raw_radial
from lnet.complex_scan import ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANT = "D4-M64-C8-RMatch"
VARIANTS = (VARIANT,)
SEEDS = (501,)
CALIBRATION_ENV = "LNET_ACTIVATION_SCALE_JSON"
heads = calibrated.heads


def _calibration_payload() -> dict[str, Any]:
    path_value = os.environ.get(CALIBRATION_ENV)
    if not path_value:
        message = f"{CALIBRATION_ENV} must point to a calibration JSON file"
        raise RuntimeError(message)
    path = Path(path_value)
    payload = json.loads(path.read_text())
    if payload.get("schema") != calibration.SCHEMA:
        message = "activation calibration has an incompatible schema"
        raise RuntimeError(message)
    scales = payload.get("scales")
    if not isinstance(scales, dict) or set(scales) != set(calibration.SITE_NAMES):
        message = "activation calibration does not contain all twelve CFFN scales"
        raise RuntimeError(message)
    if not all(float(value) > 0.0 for value in scales.values()):
        message = "activation calibration scales must be positive"
        raise RuntimeError(message)
    return payload


def _register_scale(module: nn.Module, name: str, value: float) -> None:
    module.register_buffer(name, torch.tensor(float(value)), persistent=True)


def _install_matched_activations(model: nn.Module, scales: dict[str, float]) -> None:
    raw_radial._install_radial_activations(model)
    for stage_index in (1, 2, 3):
        stage = getattr(model, f"stage{stage_index}")
        combiner = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        prefix = f"stage{stage_index}"
        _register_scale(
            combiner,
            "mode_activation_scale",
            scales[f"{prefix}.mode"],
        )
        _register_scale(
            combiner,
            "path_activation_scale",
            scales[f"{prefix}.path"],
        )
        _register_scale(
            transition,
            "ffn_activation_scale",
            scales[f"{prefix}.transition"],
        )
        _register_scale(
            transition,
            "post_ffn_activation_scale",
            scales[f"{prefix}.post"],
        )


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported RMS-matched radial variant: {variant}"
        raise ValueError(message)
    payload = _calibration_payload()
    model = calibrated._build(calibrated.VARIANT, config)
    _install_matched_activations(
        model,
        {name: float(value) for name, value in payload["scales"].items()},
    )
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = calibrated._contract(args)
    calibration_payload = _calibration_payload()
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][calibrated.VARIANT])
    variant_config["backbone"]["complex_ffn_activation_ablation"] = {
        "scope": "all twelve CFFNs across stages 1-3",
        "control": "cartesian_silu(real) + i*cartesian_silu(imag)",
        "candidate": "kappa_l * 2*sigmoid(abs(U)-1)*U",
        "kappa_trainable": False,
        "kappa_by_site": calibration_payload["scales"],
        "calibration": {
            key: calibration_payload[key]
            for key in ("schema", "seed", "split", "batch_size", "batches", "images")
        },
        "all_other_architecture_and_training_settings_unchanged": True,
    }
    payload.update(
        {
            "schema": "lnet.a2d.deep4_m64_canonical8_rmsmatched.imagenet100.v2",
            "evidence_status": "untrained static-damping optimization candidate",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "D4-M64-C8-Cal with twelve phase-preserving radial CFFNs; "
                    "each activation has a fixed training-split RMS calibration "
                    "against the matched Cartesian-SiLU initialization."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_m64_rms_calibrator"] = heads.harness._digest(
        Path(calibration.__file__)
    )
    payload["source_sha256"]["a2d_m64_rmsmatched_runner"] = heads.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def main() -> None:
    source = resaux_base
    residuals = a2d_base.residuals
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
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=calibrated.canonical8.fair_init.backbone._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
