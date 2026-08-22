#!/usr/bin/env python3
"""Train a paired K64/D2262 pole-allocation factorial."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false, reportUnusedFunction=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_front_capacity_reallocation_d2262_imagenet100 as capacity
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import torch

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


MAIN = "K64-P96-128-128-96-D2262"
P1_64 = "K64-P64-128-128-96-D2262"
P1_128 = "K64-P128-128-128-96-D2262"
P2_96 = "K64-P96-96-128-96-D2262"
P3_96 = "K64-P96-128-96-96-D2262"
P4_64 = "K64-P96-128-128-64-D2262"
QLAB_VARIANTS = (MAIN, P1_64, P1_128, P2_96)
H200_VARIANTS = (P3_96, P4_64)
VARIANTS = (*QLAB_VARIANTS, *H200_VARIANTS)
VARIANT = MAIN
JOBS_BY_GPU = {
    0: (MAIN, P1_128),
    1: (P1_64, P2_96),
}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = capacity.RESOLUTIONS
STAGE_NAMES = capacity.STAGE_NAMES
SameResolutionFactorialBackbone = capacity.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = capacity.POST_HIDDEN_RATIO
EXTRA_BLOCKS = capacity.EXTRA_BLOCKS
POLE_SCHEDULES = {
    MAIN: (96, 128, 128, 96),
    P1_64: (64, 128, 128, 96),
    P1_128: (128, 128, 128, 96),
    P2_96: (96, 96, 128, 96),
    P3_96: (96, 128, 96, 96),
    P4_64: (96, 128, 128, 64),
}
SPECS = {
    variant: allocation.StageAllocationSpec(
        excitation_modes=(64, 64, 64, 64),
        pole_modes=poles,
        extra_blocks=EXTRA_BLOCKS,
        family="k64_pole_allocation",
    )
    for variant, poles in POLE_SCHEDULES.items()
}


def _build_candidate(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported K64 pole-allocation variant: {variant}") from error
    return capacity._build_spec_model(spec, config)


@torch.no_grad()
def _pair_common_initialization_(
    target: ComplexScanBackbone,
    reference: ComplexScanBackbone,
) -> int:
    """Pair every state tensor whose role and shape are unchanged from MAIN."""
    source_state = reference.state_dict()
    copied = 0
    for name, target_value in target.state_dict().items():
        source_value = source_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        target_value.copy_(source_value)
        copied += 1
    if copied == 0:
        raise RuntimeError("P-allocation pairing found no common model state")
    return copied


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    capacity._assert_model_for_spec(model, spec, variant)
    for name in STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if type(transition) is not GatedPoleExcitationS2DTransition:
            raise RuntimeError(f"{variant}/{name} changed the established raw merge")
        if transition.carry_projection is not None:
            raise RuntimeError(f"{variant}/{name} lost its K64 identity carry")


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    initial_seed = torch.initial_seed()
    model = _build_candidate(variant, config)
    if variant != MAIN:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initial_seed)
            reference = _build_candidate(MAIN, config)
        _pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = capacity._variant_config_for_spec(variant, spec)
    payload["backbone"]["transition"]["merge"] = (
        "established raw projected memory plus K64 identity S2D carry"
    )
    payload["experiment"] = {
        "family": "k64_pole_allocation",
        "question": "which P in {64,96,128} is sufficient at each stage when K64 is fixed",
        "excitation_modes": [64, 64, 64, 64],
        "pole_modes": list(spec.pole_modes),
        "paired_initialization": (
            "all same-name, same-shape state tensors match the MAIN schedule exactly"
        ),
        "controls": {
            "depth": "D2262",
            "post_hidden_ratio": POST_HIDDEN_RATIO,
            "path_hidden": 4,
            "head": "terminal raw-Q affine",
        },
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
        runner_source_key="k64_p_allocation_d2262_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.k64_p_allocation_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K=[64,64,64,64], P={list(SPECS[variant].pole_modes)}, "
                "D2262, raw identity-carry merge, WLPost1.5K, PathH4, "
                "terminal raw-Q affine head"
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={
            "accuracy_anchor": "K128-P160x4-D2262-FullSR14x4",
            "P_mid_allocation": "K128-P128-192-192-128-D2242",
            "terminal_P96": "K128-PF15K-SR56-28-14-7-WLPost-PathH4-TermP96",
            "discarded_overcomplete_control": "K64-P96-160-160-96-D2262",
        },
    )
    payload["jobs_by_gpu"] = {str(gpu): list(variants) for gpu, variants in JOBS_BY_GPU.items()}
    payload["h200_variants"] = list(H200_VARIANTS)
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
