#!/usr/bin/env python3
"""Screen pole schedules for uniform-K S/M/L/XXL families on qlab."""

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


POLICIES: dict[str, tuple[int, tuple[int, int, int, int], str]] = {
    "S-K32-U1": (32, (32, 32, 32, 32), "U1.0"),
    "S-K32-U125": (32, (48, 48, 48, 48), "U1.25"),
    "S-K32-Shaped": (32, (48, 64, 48, 48), "Shaped"),
    "S-K32-Rich": (32, (48, 64, 64, 48), "Shaped-Rich"),
    "M-K48-U1": (48, (48, 48, 48, 48), "U1.0"),
    "M-K48-U125": (48, (64, 64, 64, 64), "U1.25"),
    "M-K48-Shaped": (48, (80, 96, 80, 80), "Shaped"),
    "M-K48-Rich": (48, (80, 96, 96, 80), "Shaped-Rich"),
    "L-K64-U1": (64, (64, 64, 64, 64), "U1.0"),
    "L-K64-U125": (64, (80, 80, 80, 80), "U1.25"),
    "XL-K96-U1": (96, (96, 96, 96, 96), "U1.0"),
    "XL-K96-U125": (96, (128, 128, 128, 128), "U1.25"),
    "XL-K96-Shaped": (96, (144, 192, 144, 144), "Shaped"),
    "XL-K96-Rich": (96, (144, 192, 192, 144), "Shaped-Rich"),
    "XXL-K128-Shaped": (128, (192, 256, 192, 192), "Shaped"),
}
XL_VARIANTS = (
    "XL-K96-U1",
    "XL-K96-U125",
    "XL-K96-Shaped",
    "XL-K96-Rich",
)
VARIANTS = tuple(variant for variant in POLICIES if variant not in XL_VARIANTS)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {
    0: (
        "S-K32-U1",
        "S-K32-U125",
        "S-K32-Shaped",
        "S-K32-Rich",
        "M-K48-U1",
        "M-K48-U125",
        "L-K64-U1",
    ),
    1: (
        "M-K48-Shaped",
        "M-K48-Rich",
        "L-K64-U125",
        "XXL-K128-Shaped",
    ),
}
SMOKE_VARIANTS = (
    "S-K32-Rich",
    "M-K48-Rich",
    "L-K64-U125",
    "XXL-K128-Shaped",
)
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = capacity.RESOLUTIONS
STAGE_NAMES = capacity.STAGE_NAMES
SameResolutionFactorialBackbone = capacity.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = capacity.POST_HIDDEN_RATIO
EXTRA_BLOCKS = (0, 0, 4, 0)
SPECS = {
    variant: allocation.StageAllocationSpec(
        excitation_modes=(width,) * 4,
        pole_modes=poles,
        extra_blocks=EXTRA_BLOCKS,
        family="uniform_k_family_wave_a",
    )
    for variant, (width, poles, _policy) in POLICIES.items()
}
REFERENCE_SPECS = {
    width: allocation.StageAllocationSpec(
        excitation_modes=(width,) * 4,
        pole_modes=(width,) * 4,
        extra_blocks=EXTRA_BLOCKS,
        family="uniform_k_family_reference",
    )
    for width in (32, 48, 64, 96, 128)
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
    source_state = reference.state_dict()
    copied = 0
    for name, target_value in target.state_dict().items():
        source_value = source_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        target_value.copy_(source_value)
        copied += 1
    if copied == 0:
        raise RuntimeError("K-family pairing found no common state")
    return copied


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    capacity._assert_model_for_spec(model, spec, variant)
    for name in STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if type(transition) is not GatedPoleExcitationS2DTransition:
            raise RuntimeError(f"{variant}/{name} changed the established raw merge")
        if transition.carry_projection is not None:
            raise RuntimeError(f"{variant}/{name} lost its uniform-K identity carry")


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        width, _poles, _policy = POLICIES[variant]
    except KeyError as error:
        raise ValueError(f"unsupported K-family Wave-A variant: {variant}") from error
    initial_seed = torch.initial_seed()
    model = _build_spec(SPECS[variant], config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initial_seed)
        reference = _build_spec(REFERENCE_SPECS[width], config)
    _pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    width, poles, policy = POLICIES[variant]
    payload = capacity._variant_config_for_spec(variant, SPECS[variant])
    payload["backbone"]["transition"]["merge"] = (
        "established raw projected memory plus uniform-K identity S2D carry"
    )
    payload["experiment"] = {
        "family": "uniform_k_family_wave_a",
        "size_label": variant.split("-", maxsplit=1)[0],
        "policy": policy,
        "excitation_modes": [width] * 4,
        "pole_modes": list(poles),
        "depth": [2, 2, 6, 2],
        "paired_initialization": (
            "all same-name, same-shape state tensors match the uniform-P=K reference"
        ),
        "promotion": {
            "accuracy_candidate": "highest best validation accuracy",
            "efficient_candidate": "within 0.15pp of best, then minimum parameters",
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
        runner_source_key="uniform_k_family_wave_a_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.uniform_k_family_wave_a.v1",
        evidence_status="representative compiled CUDA smoke required",
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
            "L_Shaped": "K64-P96-128-96-96-D2262",
            "L_Shaped_Rich": "K64-P96-128-128-96-D2262",
            "XXL_U1": "K128-P128x4-D2262-FullSR14x4",
            "XXL_U125": "K128-P160x4-D2262-FullSR14x4",
        },
    )
    payload["jobs_by_gpu"] = {str(gpu): list(variants) for gpu, variants in JOBS_BY_GPU.items()}
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
