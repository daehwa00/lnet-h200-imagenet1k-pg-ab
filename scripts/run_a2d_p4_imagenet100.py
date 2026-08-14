#!/usr/bin/env python3
"""Train the strict four-product-path A2D-P4 ImageNet-100 models."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_m64_canonical8_calibrated_imagenet100 as calibrated
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import nn

from lnet.complex_scan import (
    ComplexScanConfig,
    ComplexScanStage,
    FactorizedQuadrantPathModeCFFNCombiner,
)

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "P4-R"
VARIANTS = (VARIANT,)
SEEDS = (501,)
GAIN_NORMALIZATION = "global"
heads = calibrated.heads


class A2DP4(calibrated.canonical8.fair_init.backbone.VariableFourStageA2D):
    """D4-M64 whose learned features are exclusively two-axis product states."""

    def __init__(self, source: nn.Module) -> None:
        nn.Module.__init__(self)
        for name in (
            "config",
            "stem",
            "input_norm",
            "precomplex_fc",
            "analysis",
            "stage1",
            "stage2",
            "stage3",
            "terminal",
            "descriptor_dim",
            "classifier",
        ):
            setattr(self, name, getattr(source, name))
        _assert_d4_contract(self)


def _pole_banks(model: nn.Module) -> tuple[ComplexScanStage, ...]:
    banks = tuple(getattr(model, name) for name in ("stage1", "stage2", "stage3", "terminal"))
    if not all(isinstance(bank, ComplexScanStage) for bank in banks):
        message = "A2D-P4 requires four complex scan stages"
        raise TypeError(message)
    return banks


def _assert_d4_contract(model: nn.Module) -> None:
    banks = _pole_banks(model)
    for bank in banks:
        if bank.product_gain_normalization != GAIN_NORMALIZATION:
            message = "A2D-P4 requires global finite-grid gain normalization"
            raise RuntimeError(message)
    for bank in banks[:-1]:
        combiner = bank.quadrant_path_mode_combiner
        if (
            not isinstance(combiner, FactorizedQuadrantPathModeCFFNCombiner)
            or combiner.path_count != 4
            or combiner.output_paths != 4
            or not combiner.identity_path_handoff
            or combiner.path_input is not None
            or combiner.path_output is not None
            or combiner.path_layer_scale is not None
            or combiner.path_synthesis_real is not None
            or combiner.path_synthesis_imag is not None
        ):
            message = "A2D-P4 transitions require four identity-handoff product paths"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> A2DP4:
    if variant != VARIANT:
        message = f"unsupported A2D-P4 variant: {variant}"
        raise ValueError(message)
    active = replace(
        config,
        scan_memory_policy="recompute",
    )
    source = calibrated._build(calibrated.VARIANT, active)
    for bank in _pole_banks(source):
        bank.product_gain_normalization = GAIN_NORMALIZATION
    for bank in _pole_banks(source)[:-1]:
        combiner = bank.quadrant_path_mode_combiner
        if not isinstance(combiner, FactorizedQuadrantPathModeCFFNCombiner):
            message = "A2D-P4 requires a factorized product-path combiner"
            raise TypeError(message)
        combiner.use_identity_path_handoff_()
    return A2DP4(source)


def _contract(args: Namespace) -> dict[str, Any]:
    payload = calibrated._contract(args)
    payload["recipe"]["loader_workers"] = heads.harness._active_loader_workers(args.workers)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
        scan_memory_policy="recompute",
    )
    model = _build(VARIANT, config)
    source_config = deepcopy(payload["variant_configs"][calibrated.VARIANT])
    source_config["backbone"]["path_contract"] = {
        "product_paths": 4,
        "axis_order": "x_then_y",
        "x_scan_states_are_intermediate_only": True,
        "mode_cffn": "enabled",
        "path_cffn": "none_identity_handoff",
        "descriptor": "raw_directional_product_energy",
        "gain_normalization": "global_finite_grid_mean",
    }
    payload.update(
        {
            "schema": "lnet.a2d.p4.imagenet100.v2",
            "evidence_status": "strict product-only architecture comparison",
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: source_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "A2D-P4-M64 with four two-axis product paths per bank, "
                    "ModeCFFN-only identity path handoff, raw directional product-Q "
                    "descriptors, global finite-grid pole-gain normalization, S2D carry, "
                    "and PostFusion CFFNs."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_p4_runner"] = heads.harness._digest(Path(__file__))
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
