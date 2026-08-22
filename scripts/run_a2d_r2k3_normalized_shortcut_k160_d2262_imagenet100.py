#!/usr/bin/env python3
"""Train monotone K160 with a normalized stage projection shortcut."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_monotone_k_d2262_imagenet100 as control
import run_a2d_r2k3_progressive_k_d2262_imagenet100 as progressive
import torch

from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    NormalizedShortcutGatedTransition,
    normalized_shortcut_transition,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


VARIANT = "K128-128-160-160-P160x4-D2262-NormShortcut-WLPost15K-PathH4"
FULL_MEMORY_VARIANT = "K128-128-160-160-P160x4-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
VARIANTS = (VARIANT, FULL_MEMORY_VARIANT)
JOBS_BY_GPU = {1: (VARIANT,), 0: (FULL_MEMORY_VARIANT,)}
SEEDS = runtime.DEFAULT_SEEDS
STAGE_NAMES = progressive.STAGE_NAMES
RESOLUTIONS = progressive.RESOLUTIONS
SameResolutionFactorialBackbone = progressive.SameResolutionFactorialBackbone
SPEC = control.SPEC
MEMORY_SCALE_INITIAL = 0.1
MEMORY_SCALE_INITIALS = {
    VARIANT: MEMORY_SCALE_INITIAL,
    FULL_MEMORY_VARIANT: 1.0,
}
SPECS = dict.fromkeys(VARIANTS, SPEC)


def _transition_owners(
    model: ComplexScanBackbone,
) -> tuple[GatedPoleExcitationS2DTransition, ...]:
    return tuple(
        cast("GatedPoleExcitationS2DTransition", getattr(model, name).augmented)
        for name in STAGE_NAMES[:3]
    )


def _memory_owners(
    model: ComplexScanBackbone,
) -> tuple[GatedPoleExcitationS2DTransition | SameResolutionPoleScanBlock, ...]:
    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    same_resolution = cast(
        "tuple[SameResolutionPoleScanBlock, ...]",
        tuple(blocks.values()),
    )
    repeated = tuple(
        cast("SameResolutionPoleScanBlock", block)
        for group in extras.values()
        for block in cast("nn.ModuleList", group)
    )
    return (*_transition_owners(model), *same_resolution, *repeated)


def _install_normalized_shortcuts(
    model: ComplexScanBackbone,
    excitation_modes: tuple[int, int, int, int],
    *,
    memory_scale_initial: float,
) -> None:
    for name, input_modes, output_modes in zip(
        STAGE_NAMES[:3],
        excitation_modes[:-1],
        excitation_modes[1:],
        strict=True,
    ):
        stage = getattr(model, name)
        source = cast("GatedPoleExcitationS2DTransition", stage.augmented)
        if source.excitation_modes != input_modes or source.output_modes != output_modes:
            raise RuntimeError(f"{name} changed its excitation-width contract")
        if input_modes != output_modes:
            stage.augmented = normalized_shortcut_transition(
                source,
                memory_scale_initial=memory_scale_initial,
            )


def _restore_identity_memory(model: ComplexScanBackbone) -> None:
    for owner in _memory_owners(model):
        if isinstance(owner, GatedPoleExcitationS2DTransition):
            input_modes = owner.input_modes
            output_modes = owner.output_modes
        else:
            input_modes = owner.pole_modes
            output_modes = owner.modes
        if input_modes == output_modes:
            projection = owner.memory_projection
            if projection is None:
                continue
            identity = torch.eye(
                input_modes,
                device=projection.weight_real.device,
                dtype=projection.weight_real.dtype,
            )
            if not (
                torch.equal(projection.weight_real, identity)
                and not bool(torch.count_nonzero(projection.weight_imag))
            ):
                raise RuntimeError("refusing to remove a learned square memory map")
            owner.memory_projection = None


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        memory_scale_initial = MEMORY_SCALE_INITIALS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported normalized-shortcut variant: {variant}") from error
    model = progressive._build_spec_model(SPEC, config)
    _restore_identity_memory(model)
    _install_normalized_shortcuts(
        model,
        SPEC.excitation_modes,
        memory_scale_initial=memory_scale_initial,
    )
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    try:
        memory_scale_initial = MEMORY_SCALE_INITIALS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported normalized-shortcut variant: {variant}") from error
    progressive._assert_model_for_spec(model, SPEC, variant)
    transition = model.stage2.augmented
    if (
        type(transition) is not NormalizedShortcutGatedTransition
        or transition.memory_projection is not None
        or transition.carry_projection is None
        or not bool(
            torch.allclose(
                transition.memory_scale.detach(),
                torch.full_like(transition.memory_scale, memory_scale_initial),
            )
        )
    ):
        raise RuntimeError("width-changing stage lost its normalized shortcut contract")
    stage3 = cast("GatedPoleExcitationS2DTransition", model.stage3.augmented)
    if stage3.memory_projection is not None:
        raise RuntimeError("K160 stage3 memory must remain identity")
    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    if any(blocks[str(res)].memory_projection is not None for res in (14, 7)):
        raise RuntimeError("K160 same-resolution memory must remain identity")
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    extra_blocks = cast("nn.ModuleList", extras["14"])
    if any(block.memory_projection is not None for block in extra_blocks):
        raise RuntimeError("K160 repeated memory must remain identity")


def _variant_config(variant: str) -> dict[str, Any]:
    memory_scale_initial = MEMORY_SCALE_INITIALS[variant]
    payload = progressive._variant_config_for_spec(variant, SPEC)
    payload["experiment"] = {
        "family": "normalized_projection_shortcut",
        "control": control.VARIANT,
        "question": "does a ResNet-style normalized shortcut make monotone K160 train naturally",
        "width_change": {
            "shortcut": "S2D carry -> strict CL 128-to-160 -> output CRMSNorm",
            "memory": "identity P160-to-K160 -> output CRMSNorm -> layer scale",
            "memory_scale_initial": memory_scale_initial,
        },
        "same_width_k160_memory": "fixed identity; no materialized square ComplexLinear",
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
        runner_source_key="normalized_shortcut_k160_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.normalized_shortcut_k160_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                "K=[128,128,160,160], P160/D2262/PathH4 with one normalized "
                f"128-to-160 shortcut, normalized memory scale {MEMORY_SCALE_INITIALS[variant]}, "
                "and identity K160 memory maps."
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={"free_projection_control": control.VARIANT},
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
