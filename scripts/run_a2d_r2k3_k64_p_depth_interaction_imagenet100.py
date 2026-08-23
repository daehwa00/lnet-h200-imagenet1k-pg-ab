#!/usr/bin/env python3
"""Test P3 reduction against Stage-3 and terminal depth interactions."""

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


P3_D2282 = "K64-P96-128-96-96-D2282"
P3_D2283 = "K64-P96-128-96-96-D2283"
P3_D2263 = "K64-P96-128-96-96-D2263"
PFULL_D2283 = "K64-P96-128-128-96-D2283"
QLAB_VARIANTS = (P3_D2282, P3_D2283)
H200_VARIANTS = (P3_D2263, PFULL_D2283)
VARIANTS = (*QLAB_VARIANTS, *H200_VARIANTS)
VARIANT = P3_D2282
JOBS_BY_GPU = {0: (P3_D2282,), 1: (P3_D2283,)}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = capacity.RESOLUTIONS
STAGE_NAMES = capacity.STAGE_NAMES
SameResolutionFactorialBackbone = capacity.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = capacity.POST_HIDDEN_RATIO
EXCITATION_MODES = (64, 64, 64, 64)
P3_REDUCED = (96, 128, 96, 96)
P3_FULL = (96, 128, 128, 96)
REFERENCE_SPEC = allocation.StageAllocationSpec(
    excitation_modes=EXCITATION_MODES,
    pole_modes=P3_FULL,
    extra_blocks=(0, 0, 4, 0),
    family="k64_p_depth_interaction_reference",
)
SPECS = {
    P3_D2282: allocation.StageAllocationSpec(
        excitation_modes=EXCITATION_MODES,
        pole_modes=P3_REDUCED,
        extra_blocks=(0, 0, 6, 0),
        family="k64_p_depth_interaction",
    ),
    P3_D2283: allocation.StageAllocationSpec(
        excitation_modes=EXCITATION_MODES,
        pole_modes=P3_REDUCED,
        extra_blocks=(0, 0, 6, 1),
        family="k64_p_depth_interaction",
    ),
    P3_D2263: allocation.StageAllocationSpec(
        excitation_modes=EXCITATION_MODES,
        pole_modes=P3_REDUCED,
        extra_blocks=(0, 0, 4, 1),
        family="k64_p_depth_interaction",
    ),
    PFULL_D2283: allocation.StageAllocationSpec(
        excitation_modes=EXCITATION_MODES,
        pole_modes=P3_FULL,
        extra_blocks=(0, 0, 6, 1),
        family="k64_p_depth_interaction",
    ),
}


def _build_spec(
    spec: allocation.StageAllocationSpec,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    return capacity._build_spec_model(spec, config)


@torch.no_grad()
def _pair_common_initialization_(
    target: ComplexScanBackbone,
    reference: ComplexScanBackbone,
) -> int:
    """Pair all role- and shape-matched tensors with the D2262 control."""
    source_state = reference.state_dict()
    copied = 0
    for name, target_value in target.state_dict().items():
        source_value = source_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        target_value.copy_(source_value)
        copied += 1
    if copied == 0:
        raise RuntimeError("P/depth interaction pairing found no common model state")
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
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported P/depth interaction variant: {variant}") from error
    initial_seed = torch.initial_seed()
    model = _build_spec(spec, config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initial_seed)
        reference = _build_spec(REFERENCE_SPEC, config)
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
        "family": "k64_p_depth_interaction",
        "question": (
            "does P3 reduction remain Pareto-optimal when Stage-3 or terminal depth grows"
        ),
        "excitation_modes": list(EXCITATION_MODES),
        "pole_modes": list(spec.pole_modes),
        "depth": list(spec.depth),
        "paired_initialization": (
            "all same-name, same-shape state tensors match the P3-full D2262 control"
        ),
        "controls": {
            "P3_full_D2262": "K64-P96-128-128-96-D2262",
            "P3_reduced_D2262": "K64-P96-128-96-96-D2262",
            "P3_full_D2282": "K64-P96-128-128-96-D2282",
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
        runner_source_key="k64_p_depth_interaction_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.k64_p_depth_interaction.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={list(EXCITATION_MODES)}, P={list(SPECS[variant].pole_modes)}, "
                f"D={list(SPECS[variant].depth)}, raw identity-carry merge, "
                "WLPost1.5K, PathH4, terminal raw-Q affine head"
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={
            "accuracy_anchor": "K128-P160x4-D2262-FullSR14x4",
            "compact_anchor": "K64-P96-128-96-96-D2262",
            "depth_anchor": "K64-P96-128-128-96-D2282",
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
