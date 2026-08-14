"""Shared ImageNet-100 runtime helpers for associative complex-scan models."""

# ruff: noqa: SLF001
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import run_alphabet2d_imagenet100_nano as harness
import run_complex_scan_augmented_cifar100 as optimizer_source
import run_complex_scan_followups_imagenet100 as followups
import run_complex_scan_zero_init_imagenet100 as base
import torch

from lnet.complex_scan import ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable
    from pathlib import Path

    from torch import nn


REFERENCE_VARIANT = "d4_double_precomplex_fc"
SEEDS = (501,)


def _variant_config(variant: str, config: ComplexScanConfig) -> ComplexScanConfig:
    if variant != REFERENCE_VARIANT:
        return followups._variant_config(variant, config)
    matched = followups._variant_config("capacity_dual_fusion384_lrq64", config)
    return replace(
        matched,
        use_precomplex_fc=True,
        precomplex_fc_layers=2,
    )


def _contract(args: Namespace) -> dict[str, Any]:
    payload = followups._contract(args)
    config = ComplexScanConfig(output_dim=100, stem_strides=(2, 2))
    active = _variant_config(REFERENCE_VARIANT, config)
    payload["schema"] = "lnet.complex_scan.associative_runtime.imagenet100.v1"
    payload["variants"] = [REFERENCE_VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {REFERENCE_VARIANT: asdict(active)}
    return payload


def _summarize(
    root: Path,
    contract: dict[str, Any],
    *,
    variants: tuple[str, ...],
) -> dict[str, Any] | None:
    paths = [root / "results" / f"{variant}__seed501.json" for variant in variants]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    payload = {
        "schema": contract["schema"],
        "variants": {
            row["variant"]: {
                "parameters": row["parameters"],
                "final_validation": row["final_validation"],
                "training_examples_per_second": row["complete_training_examples_per_second"],
            }
            for row in rows
        },
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main(
    *,
    variants: tuple[str, ...],
    build_model: Callable[..., nn.Module],
    contract: Callable[[Namespace], dict[str, Any]],
    wandb_model_metrics: Callable[[nn.Module], dict[str, float]] = harness._wandb_model_metrics,
) -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=variants,
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=build_model,
            contract=contract,
            build_optimizer=optimizer_source._build_optimizer,
            prepare_model=base._prepare_model,
            wandb_model_metrics=wandb_model_metrics,
            summarize=lambda root, payload: _summarize(root, payload, variants=variants),
        )
    )


__all__ = [
    "REFERENCE_VARIANT",
    "ComplexScanConfig",
    "base",
    "harness",
    "main",
    "optimizer_source",
]
