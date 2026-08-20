#!/usr/bin/env python3
"""Train the calibrated product-only Deep4 model with a p/2p/3p/4p pole ramp."""

# ruff: noqa: I001

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_backbone_variants_imagenet100 as backbone
import run_a2d_deep4_m64_canonical8_calibrated_imagenet100 as calibrated
import run_a2d_deep4_m64_canonical8_imagenet100 as canonical8
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexScanConfig as PoleModelConfig,
    ComplexScanStage as PoleBank,
)

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "D4-Cal-PR-P64"
VARIANTS = (VARIANT,)
SEEDS = (501,)
BASE_POLES = 64
ORIENTATIONS = canonical8.ORIENTATIONS
STAGE_MODES = tuple(BASE_POLES * multiplier for multiplier in range(1, 5))
SPEC = backbone.Deep4BackboneSpec(
    modes=STAGE_MODES,
    stem_width=2 * STAGE_MODES[0],
    mode_cffn_widths=tuple(2 * modes for modes in STAGE_MODES[:3]),
    augmented_widths=tuple(2 * modes for modes in STAGE_MODES[:3]),
    post_ffn_widths=tuple(2 * modes for modes in STAGE_MODES[1:]),
)
heads = backbone.heads


def _register_spec() -> None:
    """Expose the ramp to the established variable-width Deep4 builder."""
    backbone.SPECS[VARIANT] = SPEC


def _product_only_config(config: PoleModelConfig) -> PoleModelConfig:
    return config


def _canonical_phase_atlas(
    modes: int,
    maximum_phase: float,
    *,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if modes <= 0 or modes % ORIENTATIONS:
        message = "calibrated pole ramp requires complete canonical-8 radial groups"
        raise ValueError(message)
    radial_levels = modes // ORIENTATIONS
    radial = torch.logspace(
        math.log10(maximum_phase / 8.0),
        math.log10(maximum_phase),
        radial_levels,
        dtype=like.dtype,
        device=like.device,
    ).repeat_interleave(ORIENTATIONS)
    orientation = torch.linspace(
        0.0,
        math.pi / 2.0,
        ORIENTATIONS,
        dtype=like.dtype,
        device=like.device,
    ).repeat(radial_levels)
    return radial * torch.cos(orientation), radial * torch.sin(orientation)


def _install_calibrated_initialization(
    bank: PoleBank,
    maximum_phase: float,
    frequency_scale: float,
    damping_scale: float,
) -> None:
    phase_x, phase_y = _canonical_phase_atlas(
        bank.modes,
        maximum_phase * frequency_scale,
        like=bank.phase_x,
    )
    radial_levels = bank.modes // ORIENTATIONS
    damping = torch.logspace(
        math.log10(0.04 * damping_scale),
        math.log10(0.35 * damping_scale),
        radial_levels,
        dtype=bank.damping_logits_x.dtype,
        device=bank.damping_logits_x.device,
    ).repeat_interleave(ORIENTATIONS)
    ratio = ((damping - bank.damping_min) / (bank.damping_max - bank.damping_min)).clamp(
        1.0e-4,
        1.0 - 1.0e-4,
    )
    logits = torch.logit(ratio)
    with torch.no_grad():
        bank.phase_x.copy_(phase_x)
        bank.phase_y.copy_(phase_y)
        bank.damping_logits_x.copy_(logits)
        bank.damping_logits_y.copy_(logits)


def _build(variant: str, config: PoleModelConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported calibrated pole-ramp variant: {variant}"
        raise ValueError(message)
    _register_spec()
    model = backbone._build(variant, _product_only_config(config))
    calibration = zip(
        canonical8.fair_init._pole_banks(model),
        canonical8.MAXIMUM_PHASES,
        calibrated.FREQUENCY_SCALES,
        calibrated.DAMPING_SCALES,
        strict=True,
    )
    for bank, maximum_phase, frequency_scale, damping_scale in calibration:
        _install_calibrated_initialization(
            bank,
            maximum_phase,
            frequency_scale,
            damping_scale,
        )
    return model


def _variant_config() -> dict[str, Any]:
    stage_names = ("stage1", "stage2", "stage3", "terminal")
    return {
        "backbone": {
            "name": "A2D-Calibrated-Product4-PoleRamp-FullOpt",
            "base_poles": BASE_POLES,
            "pole_schedule": list(STAGE_MODES),
            "canonical_orientations": ORIENTATIONS,
            "radial_levels": [modes // ORIENTATIONS for modes in STAGE_MODES],
            "stem_width": SPEC.stem_width,
            "mode_cffn_widths": list(SPEC.mode_cffn_widths),
            "augmented_widths": list(SPEC.augmented_widths),
            "post_ffn_widths": list(SPEC.post_ffn_widths),
            "spatial_resolutions": [56, 28, 14, 7],
            "descriptor_dim": SPEC.descriptor_dim,
            "product_paths": 4,
            "frequency_scale_by_stage": dict(
                zip(stage_names, calibrated.FREQUENCY_SCALES, strict=True)
            ),
            "damping_scale_by_stage": dict(
                zip(stage_names, calibrated.DAMPING_SCALES, strict=True)
            ),
        },
        "head": {
            "main": f"Fusion{SPEC.descriptor_dim}-384-256",
            "affine_auxiliary_weight": backbone.AFFINE_AUXILIARY_WEIGHT,
            "lrq": False,
        },
    }


def _contract(args: Namespace) -> dict[str, Any]:
    payload = backbone.deep4._contract(args)
    config = PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload.update(
        {
            "schema": "lnet.a2d.deep4_calibrated_pole_ramp.imagenet100.v1",
            "evidence_status": "untrained calibrated p/2p/3p/4p FullOpt candidate",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: _variant_config()},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "The calibrated product-only Deep4 model with 64/128/192/256 "
                    "pole modes, canonical eight-angle radial groups, stage-matched "
                    "mode CFFNs, and the existing width-generic optimized kernels."
                )
            },
        }
    )
    payload["model"] = deepcopy(payload["model"])
    payload["source_sha256"]["a2d_deep4_calibrated_pole_ramp_runner"] = heads.harness._digest(
        Path(__file__)
    )
    return json.loads(json.dumps(payload))


def main() -> None:
    source = canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = backbone.a2d_base.residuals
    harness = source.heads.harness
    source.heads.VARIANTS = VARIANTS
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    runner_bindings = getattr(harness, "runner_bindings", None)
    if callable(runner_bindings):
        harness.main(
            runner_bindings(
                variants=VARIANTS,
                seeds=SEEDS,
                model_config=PoleModelConfig,
                build_model=_build,
                contract=_contract,
                build_optimizer=residuals.optimizer_source._build_optimizer,
                prepare_model=source._prepare_model,
                train_epoch=source.structured._train_epoch,
                evaluate=source.heads._evaluate,
                wandb_model_metrics=backbone._wandb_model_metrics,
                summarize=source.heads._summarize,
            )
        )
        return

    # The frozen RTX 4090 runtime predates explicit RunnerBindings. Patch the
    # same harness callbacks its calibrated M64 runner uses, without touching
    # any model or kernel implementation.
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.ImageNetNanoConfig = PoleModelConfig
    harness.build_imagenet_nano = _build
    harness._contract = _contract
    harness._build_optimizer = residuals.optimizer_source._build_optimizer
    harness._prepare_model = source._prepare_model
    harness._train_epoch = source.structured._train_epoch
    harness._evaluate = source.heads._evaluate
    harness._wandb_model_metrics = backbone._wandb_model_metrics
    harness._summarize = source.heads._summarize
    harness.main()


if __name__ == "__main__":
    main()
