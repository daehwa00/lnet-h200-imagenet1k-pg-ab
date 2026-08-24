#!/usr/bin/env python3
"""Refine stage-local pole allocation across the uniform-K family."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_k_family_wave_a_imagenet100 as family
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import torch

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


M_K48_P80 = "M-K48-P80-80-80-80"
L_K64_P2 = "L-K64-P80-96-80-80"
L_K64_P23 = "L-K64-P80-96-96-80"
S_K32_P3 = "S-K32-P48-48-64-48"
XL_K96_P2_LITE = "XL-K96-P128-144-128-128"
VARIANTS = (M_K48_P80, L_K64_P2, L_K64_P23, S_K32_P3, XL_K96_P2_LITE)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {0: VARIANTS}
SEEDS = runtime.DEFAULT_SEEDS
POLICIES: dict[str, tuple[int, tuple[int, int, int, int], str]] = {
    M_K48_P80: (48, (80, 80, 80, 80), "remove the M-family Stage-2 P96 peak"),
    L_K64_P2: (64, (80, 96, 80, 80), "isolate the L-family Stage-2 expansion"),
    L_K64_P23: (64, (80, 96, 96, 80), "couple L-family Stage-2 and Stage-3 expansion"),
    S_K32_P3: (32, (48, 48, 64, 48), "isolate the S-family Stage-3 expansion"),
    XL_K96_P2_LITE: (96, (128, 144, 128, 128), "test a smaller XL Stage-2 expansion"),
}
SPECS = {
    variant: allocation.StageAllocationSpec(
        excitation_modes=(width,) * 4,
        pole_modes=poles,
        extra_blocks=family.EXTRA_BLOCKS,
        family="uniform_k_family_p_refinement",
    )
    for variant, (width, poles, _question) in POLICIES.items()
}
REFERENCE_SPECS = {
    width: allocation.StageAllocationSpec(
        excitation_modes=(width,) * 4,
        pole_modes=(width,) * 4,
        extra_blocks=family.EXTRA_BLOCKS,
        family="uniform_k_family_p_refinement_reference",
    )
    for width in {width for width, _poles, _question in POLICIES.values()}
}


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    family.capacity._assert_model_for_spec(model, SPECS[variant], variant)
    for name in family.STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if type(transition) is not GatedPoleExcitationS2DTransition:
            raise RuntimeError(f"{variant}/{name} changed the established raw merge")
        if transition.carry_projection is not None:
            raise RuntimeError(f"{variant}/{name} lost its uniform-K identity carry")


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        width, _poles, _question = POLICIES[variant]
    except KeyError as error:
        raise ValueError(f"unsupported K-family P-refinement variant: {variant}") from error
    initial_seed = torch.initial_seed()
    model = family._build_spec(SPECS[variant], config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initial_seed)
        reference = family._build_spec(REFERENCE_SPECS[width], config)
    family._pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    width, poles, question = POLICIES[variant]
    payload = family.capacity._variant_config_for_spec(variant, SPECS[variant])
    payload["backbone"]["transition"]["merge"] = (
        "established raw projected memory plus uniform-K identity S2D carry"
    )
    payload["experiment"] = {
        "family": "uniform_k_family_p_refinement",
        "priority": VARIANTS.index(variant) + 1,
        "question": question,
        "excitation_modes": [width] * 4,
        "pole_modes": list(poles),
        "depth": [2, 2, 6, 2],
        "paired_initialization": (
            "all same-name, same-shape state tensors match the uniform-P=K reference"
        ),
        "only_controlled_axis": "pole schedule",
    }
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
        runner_source_key="uniform_k_family_p_refinement_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.uniform_k_family_p_refinement.v1",
        evidence_status="H200 XL-Rich smoke covers the same operators and wider shapes",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={[POLICIES[variant][0]] * 4}, P={list(POLICIES[variant][1])}, "
                "D2262, raw identity-carry merge, WLPost1.5K, PathH4, Q4 affine"
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={
            "M_current_best": "M-K48-Shaped / P=(80,96,80,80)",
            "L_current_controls": "L-K64-U1 and L-K64-U125",
            "S_current_controls": "S-K32-Shaped and S-K32-Rich",
            "XL_current_best": "XL-K96-Rich",
        },
    )
    payload["jobs_by_gpu"] = {"0": list(VARIANTS)}
    return payload


def main() -> None:
    runtime.run(variants=VARIANTS, seeds=SEEDS, build_model=_build, contract=_contract)


if __name__ == "__main__":
    main()
