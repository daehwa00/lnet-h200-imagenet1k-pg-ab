#!/usr/bin/env python3
"""Train the six-cell high-resolution K/P capacity reallocation screen."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# pyright: reportUnusedFunction=false
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_normalized_shortcut_k160_d2262_imagenet100 as normalized
import run_a2d_r2k3_progressive_k_d2262_imagenet100 as progressive
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import torch

from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    NormalizedShortcutGatedTransition,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


K128_P128 = "K128-128-160-160-P128-160-160-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
K96_P128 = "K96-128-160-160-P128-160-160-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
K128_P96 = "K128-128-160-160-P96-160-160-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
K96_P96 = "K96-128-160-160-P96-160-160-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
MID_POLE_REALLOCATION = "K96-128-160-160-P96-192-192-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
MID_EXCITATION_REALLOCATION = (
    "K96-128-192-160-P96-160-192-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
)
VARIANTS = (
    K128_P128,
    K96_P128,
    K128_P96,
    K96_P96,
    MID_POLE_REALLOCATION,
    MID_EXCITATION_REALLOCATION,
)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {
    0: (K128_P128, K128_P96, MID_POLE_REALLOCATION),
    1: (K96_P128, K96_P96, MID_EXCITATION_REALLOCATION),
}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = progressive.RESOLUTIONS
STAGE_NAMES = progressive.STAGE_NAMES
SameResolutionFactorialBackbone = progressive.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = progressive.POST_HIDDEN_RATIO
MEMORY_SCALE_INITIAL = 1.0
EXTRA_BLOCKS = progressive.EXTRA_BLOCKS


def _spec(
    excitation_modes: tuple[int, int, int, int],
    pole_modes: tuple[int, int, int, int],
) -> allocation.StageAllocationSpec:
    return allocation.StageAllocationSpec(
        excitation_modes,
        pole_modes,
        extra_blocks=EXTRA_BLOCKS,
        family="front_capacity_reallocation",
    )


SPECS = {
    K128_P128: _spec((128, 128, 160, 160), (128, 160, 160, 160)),
    K96_P128: _spec((96, 128, 160, 160), (128, 160, 160, 160)),
    K128_P96: _spec((128, 128, 160, 160), (96, 160, 160, 160)),
    K96_P96: _spec((96, 128, 160, 160), (96, 160, 160, 160)),
    MID_POLE_REALLOCATION: _spec((96, 128, 160, 160), (96, 192, 192, 160)),
    MID_EXCITATION_REALLOCATION: _spec(
        (96, 128, 192, 160),
        (96, 160, 192, 160),
    ),
}


def _build_spec_model(
    spec: allocation.StageAllocationSpec,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    model = progressive._build_spec_model(spec, config)
    normalized._restore_identity_memory(model)
    normalized._install_normalized_shortcuts(
        model,
        spec.excitation_modes,
        memory_scale_initial=MEMORY_SCALE_INITIAL,
    )
    return model


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported front-capacity variant: {variant}") from error
    model = _build_spec_model(spec, config)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model_for_spec(model, spec, variant)
    return model


def _assert_model_for_spec(
    model: ComplexScanBackbone,
    spec: allocation.StageAllocationSpec,
    variant: str,
) -> None:
    progressive._assert_model_for_spec(model, spec, variant)
    for name, input_modes, output_modes in zip(
        STAGE_NAMES[:3],
        spec.excitation_modes[:-1],
        spec.excitation_modes[1:],
        strict=True,
    ):
        transition = cast(
            "GatedPoleExcitationS2DTransition",
            getattr(model, name).augmented,
        )
        if input_modes != output_modes:
            if type(transition) is not NormalizedShortcutGatedTransition:
                raise RuntimeError(f"{variant}/{name} lost its normalized shortcut")
            if not torch.equal(
                transition.memory_scale,
                torch.full_like(transition.memory_scale, MEMORY_SCALE_INITIAL),
            ):
                raise RuntimeError(f"{variant}/{name} changed its memory scale")
        elif isinstance(transition, NormalizedShortcutGatedTransition):
            raise RuntimeError(f"{variant}/{name} normalized an equal-width shortcut")

    for owner in normalized._memory_owners(model):
        if isinstance(owner, GatedPoleExcitationS2DTransition):
            input_modes = owner.input_modes
            output_modes = owner.output_modes
        else:
            input_modes = owner.pole_modes
            output_modes = owner.modes
        projection = owner.memory_projection
        if input_modes == output_modes and projection is not None:
            raise RuntimeError(f"{variant} materialized a redundant square memory map")
        if input_modes != output_modes and projection is None:
            raise RuntimeError(f"{variant} lost a required rectangular memory map")


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported front-capacity variant: {variant}") from error
    _assert_model_for_spec(model, spec, variant)


def _variant_config_for_spec(
    variant: str,
    spec: allocation.StageAllocationSpec,
) -> dict[str, Any]:
    payload = progressive._variant_config_for_spec(variant, spec)
    payload["experiment"] = {
        "family": "front_capacity_reallocation",
        "question": (
            "can 56x56 complex excitation/pole width be reduced and its capacity "
            "reallocated to lower-resolution pole or excitation coordinates"
        ),
        "memory_scale_initial": MEMORY_SCALE_INITIAL,
        "high_resolution": {
            "excitation_modes": spec.excitation_modes[0],
            "pole_modes": spec.pole_modes[0],
        },
        "reallocation": {
            "excitation_modes": list(spec.excitation_modes[1:]),
            "pole_modes": list(spec.pole_modes[1:]),
        },
        "controls": {
            "depth": "D2262",
            "post_hidden_ratio": POST_HIDDEN_RATIO,
            "path_hidden": 4,
            "head": "terminal raw-Q affine",
        },
    }
    return payload


def _variant_config(variant: str) -> dict[str, Any]:
    return _variant_config_for_spec(variant, SPECS[variant])


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload = campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="front_capacity_reallocation_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.front_capacity_reallocation_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={list(spec.excitation_modes)}, P={list(spec.pole_modes)}, "
                "D2262, WLPost1.5K, PathH4, normalized width-changing shortcuts, "
                "terminal raw-Q affine head"
            )
            for variant, spec in SPECS.items()
        },
        parameter_counts=parameter_counts,
        references={
            "completed_uniform_k128_p160": "K128-P160x4-D2262-FullSR14x4",
            "normalized_monotone_k160": normalized.VARIANT,
        },
    )
    payload["jobs_by_gpu"] = {str(gpu): list(variants) for gpu, variants in JOBS_BY_GPU.items()}
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
