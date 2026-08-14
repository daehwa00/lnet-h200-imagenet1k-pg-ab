#!/usr/bin/env python3
"""Train the two missing D4-M64 pole-width/CFFN-width factorial cells."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_backbone_variants_imagenet100 as backbone
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexScanBackbone,
    ComplexScanConfig,
    ModalFusionHead,
)
from lnet.image_layers import StandardizedAffineModalHead

if TYPE_CHECKING:
    from argparse import Namespace


P64_H96 = "D4-P64-H96"
P48_H128 = "D4-P48-H128"
VARIANTS = (P64_H96, P48_H128)
SEEDS = (501,)
# The generic runner smoke test reads the shared structured objective here.
heads = backbone.heads

SPECS = {
    P64_H96: backbone.Deep4BackboneSpec(
        modes=(64, 64, 64, 64),
        stem_width=128,
        mode_cffn_widths=(96, 96, 96),
        augmented_widths=(96, 96, 96),
        post_ffn_widths=(96, 96, 96),
    ),
    P48_H128: backbone.Deep4BackboneSpec(
        modes=(48, 48, 48, 48),
        stem_width=96,
        mode_cffn_widths=(128, 128, 128),
        augmented_widths=(128, 128, 128),
        post_ffn_widths=(128, 128, 128),
    ),
}


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        message = f"unsupported Deep4 factorial variant: {variant}"
        raise ValueError(message) from error

    active = backbone._active_config(config, spec)
    source = ComplexScanBackbone(active)
    source = a2d_base._remove_final_precomplex_gelu(source)
    backbone._install_transition(source.stage1, spec.post_ffn_widths[0])
    backbone._install_transition(source.stage2, spec.post_ffn_widths[1])
    model = backbone.VariableFourStageA2D(
        source,
        backbone._make_stage3(active, spec),
        backbone._make_terminal(active, spec.modes[3]),
        spec.descriptor_dim,
    )
    affine = StandardizedAffineModalHead(spec.descriptor_dim, config.output_dim)
    fusion_source = ModalFusionHead(
        spec.descriptor_dim,
        backbone.deep4.baseline.FIRST_WIDTH,
        config.output_dim,
    )
    fusion = backbone.deep4.baseline.DeepModalFusionHead(
        fusion_source,
        config.output_dim,
    )
    model.classifier = resaux_base.heads.A2DAffineQClassifier(
        spec.descriptor_dim,
        config.output_dim,
        main="fusion",
        affine=affine,
        fusion=fusion,
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=backbone.AFFINE_AUXILIARY_WEIGHT,
    )
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = backbone.deep4._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    models = {variant: _build(variant, config) for variant in VARIANTS}
    payload.update(
        {
            "schema": "lnet.a2d.deep4_factorial_cells.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch missing factorial cells",
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        variant: {
            "backbone": {
                "name": "A2D-D4-PathMix-PostCarry-PostFFN-4Stage",
                "modes": list(spec.modes),
                "stem_width": spec.stem_width,
                "mode_cffn_widths": list(spec.mode_cffn_widths),
                "augmented_widths": list(spec.augmented_widths),
                "post_ffn_widths": list(spec.post_ffn_widths),
                "spatial_resolutions": [56, 28, 14, 7],
                "descriptor_dim": spec.descriptor_dim,
            },
            "head": {
                "main": f"Fusion{spec.descriptor_dim}-384-256",
                "affine_auxiliary_weight": backbone.AFFINE_AUXILIARY_WEIGHT,
                "lrq": False,
            },
        }
        for variant, spec in SPECS.items()
    }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture"] = {
        P64_H96: (
            "The uniform 64-pole M64 carrier and its required 128-wide stem, "
            "with all ModeCFFN, transition, and PostFFN widths reduced to 96."
        ),
        P48_H128: (
            "The uniform 48-pole Deep4 carrier and 96-wide stem, with every "
            "ModeCFFN, transition, and PostFFN width increased to 128."
        ),
    }
    payload["source_sha256"]["deep4_factorial_cells_runner"] = backbone.heads.harness._digest(
        Path(__file__)
    )
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
            wandb_model_metrics=backbone._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
