#!/usr/bin/env python3
"""Run the four uniform-K XL controls reserved for H200."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_k_family_wave_a_imagenet100 as family

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANTS = family.XL_VARIANTS
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {0: VARIANTS}
SMOKE_VARIANTS = ("XL-K96-Rich",)
SEEDS = family.SEEDS


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported H200 XL variant: {variant}")
    model = family._build(variant, config)
    runtime.configure(VARIANTS, SEEDS)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    payload = family._variant_config(variant)
    payload["experiment"]["execution_pool"] = "H200 XL-only queue"
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload = campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="uniform_k_family_wave_a_h200_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.uniform_k_family_wave_a_h200.v1",
        evidence_status="representative compiled CUDA smoke required before H200 queue activation",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={[family.POLICIES[variant][0]] * 4}, "
                f"P={list(family.POLICIES[variant][1])}, D2262, raw identity-carry "
                "merge, WLPost1.5K, PathH4, Q4 affine"
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={"XL_hardware_control": "all four XL candidates run on the same H200"},
    )
    payload["jobs_by_gpu"] = {"0": list(VARIANTS)}
    payload["smoke_variants"] = list(SMOKE_VARIANTS)
    return payload


def main() -> None:
    runtime.run(
        variants=VARIANTS,
        seeds=SEEDS,
        build_model=_build,
        contract=_contract,
    )


if __name__ == "__main__":
    main()
