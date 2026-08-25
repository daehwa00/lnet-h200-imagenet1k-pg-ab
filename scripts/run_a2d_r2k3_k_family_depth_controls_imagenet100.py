#!/usr/bin/env python3
"""Run paired S/L controls and S/L/XL Stage-3 depth probes on H200."""

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


S_D2262 = "S-K32-P48x4-D2262"
S_D2242 = "S-K32-P48x4-D2242"
L_D2262 = "L-K64-P80x4-D2262"
L_D2282 = "L-K64-P80x4-D2282"
XL_D2282 = "XL-K96-P128x4-D2282"
VARIANTS = (S_D2262, S_D2242, L_D2262, L_D2282, XL_D2282)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {0: VARIANTS}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = family.RESOLUTIONS
STAGE_NAMES = family.STAGE_NAMES
SameResolutionFactorialBackbone = family.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = family.POST_HIDDEN_RATIO
POLICIES: dict[
    str,
    tuple[int, tuple[int, int, int, int], tuple[int, int, int, int], str],
] = {
    S_D2262: (32, (48, 48, 48, 48), (0, 0, 4, 0), "S H200 D2262 control"),
    S_D2242: (32, (48, 48, 48, 48), (0, 0, 2, 0), "S Stage-3 depth reduction"),
    L_D2262: (64, (80, 80, 80, 80), (0, 0, 4, 0), "L H200 D2262 control"),
    L_D2282: (64, (80, 80, 80, 80), (0, 0, 6, 0), "L Stage-3 depth expansion"),
    XL_D2282: (
        96,
        (128, 128, 128, 128),
        (0, 0, 6, 0),
        "XL Stage-3 depth expansion against the completed H200 D2262 control",
    ),
}
SPECS = {
    variant: allocation.StageAllocationSpec(
        excitation_modes=(width,) * 4,
        pole_modes=poles,
        extra_blocks=extras,
        family="uniform_k_family_depth_controls",
    )
    for variant, (width, poles, extras, _question) in POLICIES.items()
}
REFERENCE_SPECS = {
    (width, poles): allocation.StageAllocationSpec(
        excitation_modes=(width,) * 4,
        pole_modes=poles,
        extra_blocks=(0, 0, 4, 0),
        family="uniform_k_family_depth_control_reference",
    )
    for width, poles in {
        (width, poles) for width, poles, _extras, _question in POLICIES.values()
    }
}


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    family.capacity._assert_model_for_spec(model, SPECS[variant], variant)
    for name in STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if type(transition) is not GatedPoleExcitationS2DTransition:
            raise RuntimeError(f"{variant}/{name} changed the established raw merge")
        if transition.carry_projection is not None:
            raise RuntimeError(f"{variant}/{name} lost its uniform-K identity carry")
    if SPECS[variant].depth[-1] != 2:
        raise RuntimeError(f"{variant} changed terminal depth")


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        width, poles, _extras, _question = POLICIES[variant]
    except KeyError as error:
        raise ValueError(f"unsupported K-family depth-control variant: {variant}") from error
    initial_seed = torch.initial_seed()
    model = family._build_spec(SPECS[variant], config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initial_seed)
        reference = family._build_spec(REFERENCE_SPECS[(width, poles)], config)
    family._pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    width, poles, _extras, question = POLICIES[variant]
    payload = family.capacity._variant_config_for_spec(variant, SPECS[variant])
    payload["backbone"]["transition"]["merge"] = (
        "established raw projected memory plus uniform-K identity S2D carry"
    )
    payload["experiment"] = {
        "family": "uniform_k_family_depth_controls",
        "priority": VARIANTS.index(variant) + 1,
        "question": question,
        "excitation_modes": [width] * 4,
        "pole_modes": list(poles),
        "depth": list(SPECS[variant].depth),
        "terminal_depth": 2,
        "paired_initialization": "same-shape tensors match the D2262 family control",
        "only_controlled_axis": "Stage-3 depth within each K/P pair",
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
        runner_source_key="uniform_k_family_depth_controls_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.uniform_k_family_depth_controls.v1",
        evidence_status="representative compiled H200 smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={[POLICIES[variant][0]] * 4}, P={list(POLICIES[variant][1])}, "
                f"D={list(SPECS[variant].depth)}, terminal depth2, "
                "raw identity-carry merge, WLPost1.5K, PathH4, Q4 affine"
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={
            "XL_D2262_H200": {
                "variant": "XL-K96-U125",
                "parameters": 2_791_524,
                "best_validation_accuracy": 0.875,
                "reuse": "completed H200 control; intentionally excluded from this queue",
            }
        },
    )
    payload["jobs_by_gpu"] = {"0": list(VARIANTS)}
    return payload


def main() -> None:
    runtime.run(variants=VARIANTS, seeds=SEEDS, build_model=_build, contract=_contract)


if __name__ == "__main__":
    main()
