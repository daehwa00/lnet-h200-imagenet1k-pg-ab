#!/usr/bin/env python3
"""Screen stage-wise pole allocation and low-resolution depth on ImageNet-100."""

from __future__ import annotations

# This campaign composes frozen private builders to preserve seeded ancestry.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as capacity_insight
import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as control
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as factorial
from torch import nn

from lnet.pac_phase_gated_transition import PathOnlyCollapse

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.complex_scan_types import ComplexField
    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


CONTROL = "K128-P128x4-D2222-Control"
P14_ONLY = "K128-P128-128-160-128-D2222"
P28_14 = "K128-P128-160-160-128-D2222"
P_FRONT3 = "K128-P160-160-160-128-D2222"
P_FRONT3_TERM96 = "K128-P160-160-160-96-D2222"
P14_TERM96 = "K128-P128-128-160-96-D2222"
P_MEMORY_PYRAMID = "K128-P96-128-160-96-D2222"
P96_UNIFORM = "K128-P96x4-D2222"
K96_P128 = "K96-P128x4-D2222"
DEPTH_2232 = "K128-P128x4-D2232-FullSR14x1"
DEPTH_2223 = "K128-P128x4-D2223-FullSR7x1"
DEPTH_2242 = "K128-P128x4-D2242-FullSR14x2"
DEPTH_2262 = "K128-P128x4-D2262-FullSR14x4"

ALLOCATION_VARIANTS = (
    CONTROL,
    P14_ONLY,
    P28_14,
    P_FRONT3,
    P_FRONT3_TERM96,
    P14_TERM96,
    P_MEMORY_PYRAMID,
    P96_UNIFORM,
    K96_P128,
)
DEPTH_VARIANTS = (
    DEPTH_2232,
    DEPTH_2223,
    DEPTH_2242,
    DEPTH_2262,
)
VARIANTS = (*ALLOCATION_VARIANTS, *DEPTH_VARIANTS)
VARIANT = VARIANTS[0]
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = factorial.RESOLUTIONS
STAGE_NAMES = runtime.STAGE_NAMES
POST_HIDDEN = control.POST_HIDDEN
PATH_HIDDEN = control.PATH_HIDDEN
EXTRA_BLOCK_SEED_STRIDE = 1_000_003
SameResolutionFactorialBackbone = factorial.SameResolutionFactorialBackbone

# The local workstation exposes one training GPU.  Priority first resolves the
# stage-allocation chain, then the two rectangular controls, then depth scaling.
JOBS_BY_GPU = {0: VARIANTS}


@dataclass(frozen=True, slots=True)
class StageAllocationSpec:
    excitation_modes: tuple[int, int, int, int]
    pole_modes: tuple[int, int, int, int]
    extra_blocks: tuple[int, int, int, int] = (0, 0, 0, 0)
    family: str = "pole_allocation"

    @property
    def width(self) -> int:
        return self.excitation_modes[0]

    @property
    def excitation_width(self) -> int:
        return self.width

    @property
    def resolutions(self) -> tuple[int, int, int, int]:
        return RESOLUTIONS

    @property
    def depth(self) -> tuple[int, int, int, int]:
        first, second, third, fourth = self.extra_blocks
        return 2 + first, 2 + second, 2 + third, 2 + fourth

    @property
    def descriptor_dim(self) -> int:
        return 4 * sum(self.pole_modes)

    @property
    def q4_dim(self) -> int:
        return 4 * self.pole_modes[-1]

    def as_insight_spec(self) -> capacity_insight.InsightSpec:
        return capacity_insight.InsightSpec(self.excitation_modes, self.pole_modes)


SPECS = {
    CONTROL: StageAllocationSpec((128,) * 4, (128,) * 4, family="control"),
    P14_ONLY: StageAllocationSpec((128,) * 4, (128, 128, 160, 128)),
    P28_14: StageAllocationSpec((128,) * 4, (128, 160, 160, 128)),
    P_FRONT3: StageAllocationSpec((128,) * 4, (160, 160, 160, 128)),
    P_FRONT3_TERM96: StageAllocationSpec((128,) * 4, (160, 160, 160, 96)),
    P14_TERM96: StageAllocationSpec((128,) * 4, (128, 128, 160, 96)),
    P_MEMORY_PYRAMID: StageAllocationSpec((128,) * 4, (96, 128, 160, 96)),
    P96_UNIFORM: StageAllocationSpec((128,) * 4, (96,) * 4),
    K96_P128: StageAllocationSpec((96,) * 4, (128,) * 4),
    DEPTH_2232: StageAllocationSpec(
        (128,) * 4,
        (128,) * 4,
        extra_blocks=(0, 0, 1, 0),
        family="depth_localization",
    ),
    DEPTH_2223: StageAllocationSpec(
        (128,) * 4,
        (128,) * 4,
        extra_blocks=(0, 0, 0, 1),
        family="depth_localization",
    ),
    DEPTH_2242: StageAllocationSpec(
        (128,) * 4,
        (128,) * 4,
        extra_blocks=(0, 0, 2, 0),
        family="depth_localization",
    ),
    DEPTH_2262: StageAllocationSpec(
        (128,) * 4,
        (128,) * 4,
        extra_blocks=(0, 0, 4, 0),
        family="depth_scaling",
    ),
}


class RepeatedSameResolutionBackbone(factorial.SameResolutionFactorialBackbone):
    """Append an ordered sequence of independently initialized full SR blocks."""

    def __init__(
        self,
        source: ComplexScanBackbone,
        extras: dict[int, tuple[SameResolutionPoleScanBlock, ...]],
    ) -> None:
        existing = dict(source.same_resolution_blocks.items())
        super().__init__(source, existing)
        self.extra_same_resolution_blocks = nn.ModuleDict(
            {
                str(resolution): nn.ModuleList(blocks)
                for resolution, blocks in extras.items()
            }
        )

    def _apply_at(self, resolution: int, state: ComplexField) -> ComplexField:
        state = super()._apply_at(resolution, state)
        key = str(resolution)
        if key not in self.extra_same_resolution_blocks:
            return state
        blocks = cast("nn.ModuleList", self.extra_same_resolution_blocks[key])
        for block in blocks:
            state = cast("ComplexField", block(*state))
        return state


def _templates(model: ComplexScanBackbone) -> dict[int, Any]:
    return {
        56: model.stage1,
        28: model.stage2,
        14: cast("Any", model).stage3,
        7: model.terminal,
    }


def _append_repeated_blocks(
    model: ComplexScanBackbone,
    spec: StageAllocationSpec,
) -> ComplexScanBackbone:
    templates = _templates(model)
    extras: dict[int, tuple[SameResolutionPoleScanBlock, ...]] = {}
    for stage_index, (resolution, count) in enumerate(
        zip(RESOLUTIONS, spec.extra_blocks, strict=True)
    ):
        blocks: list[SameResolutionPoleScanBlock] = []
        for repeat_index in range(count):
            block = factorial._make_block(
                width=spec.excitation_modes[stage_index],
                pole_modes=spec.pole_modes[stage_index],
                resolution=resolution,
                pole_template=templates[resolution],
                post_hidden=POST_HIDDEN,
                seed_offset=EXTRA_BLOCK_SEED_STRIDE * (repeat_index + 1),
            )
            block.path_collapse = PathOnlyCollapse(
                spec.pole_modes[stage_index],
                path_hidden=PATH_HIDDEN,
            )
            blocks.append(block)
        if blocks:
            extras[resolution] = tuple(blocks)
    if not extras:
        return model
    return RepeatedSameResolutionBackbone(model, extras)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported stage-allocation variant: {variant}") from error

    model = _build_for_spec(spec, variant, config)
    runtime.configure(VARIANTS, SEEDS)
    return model


def _build_for_spec(
    spec: StageAllocationSpec,
    label: str,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    model = capacity_insight._build_all_resolution(spec.as_insight_spec(), config)
    model = _append_repeated_blocks(model, spec)
    _assert_model_for_spec(model, spec, label)
    return model


def _assert_block(
    block: nn.Module,
    *,
    excitation_modes: int,
    pole_modes: int,
    label: str,
) -> None:
    capacity_insight._assert_block(
        block,
        excitation_modes=excitation_modes,
        pole_modes=pole_modes,
        label=label,
    )


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    _assert_model_for_spec(model, SPECS[variant], variant)


def _assert_model_for_spec(
    model: ComplexScanBackbone,
    spec: StageAllocationSpec,
    label: str,
) -> None:
    extras = getattr(model, "extra_same_resolution_blocks", None)
    if extras is not None:
        delattr(model, "extra_same_resolution_blocks")
    try:
        capacity_insight._assert_model_for_spec(model, spec.as_insight_spec(), label)
    finally:
        if extras is not None:
            model.extra_same_resolution_blocks = extras
    active_extras = extras if extras is not None else nn.ModuleDict()
    expected = {
        str(resolution): count
        for resolution, count in zip(RESOLUTIONS, spec.extra_blocks, strict=True)
        if count
    }
    if {name: len(blocks) for name, blocks in active_extras.items()} != expected:
        raise RuntimeError(f"{label} changed its repeated same-resolution depth")
    for stage_index, resolution in enumerate(RESOLUTIONS):
        key = str(resolution)
        if key not in active_extras:
            continue
        for repeat_index, block in enumerate(active_extras[key]):
            _assert_block(
                block,
                excitation_modes=spec.excitation_modes[stage_index],
                pole_modes=spec.pole_modes[stage_index],
                label=f"{label}/ExtraSR{resolution}.{repeat_index}",
            )


def _variant_config(variant: str) -> dict[str, Any]:
    return _variant_config_for_spec(variant, SPECS[variant])


def _variant_config_for_spec(
    variant: str,
    spec: StageAllocationSpec,
) -> dict[str, Any]:
    payload = deepcopy(
        capacity_insight._variant_config_for_spec(variant, spec.as_insight_spec())
    )
    payload["experiment"] = {
        "family": spec.family,
        "depth_by_resolution": dict(zip(map(str, RESOLUTIONS), spec.depth, strict=True)),
        "extra_full_blocks_by_resolution": dict(
            zip(map(str, RESOLUTIONS), spec.extra_blocks, strict=True)
        ),
        "pole_schedule_applies_to": "main scan and same-resolution block",
        "selection_role": "single-seed diagnostic screen",
    }
    payload["backbone"]["same_resolution_factorial"]["extra_repetitions_by_resolution"] = (
        dict(zip(map(str, RESOLUTIONS), spec.extra_blocks, strict=True))
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    payload = runtime.base_contract(args)
    payload["schema"] = "lnet.a2d.r2k3.stage_allocation_screen.v1"
    payload["evidence_status"] = (
        "13-cell seed501 diagnostic screen; CPU and compiled CUDA full-batch smoke required"
    )
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["screen"] = {
        "allocation_variants": list(ALLOCATION_VARIANTS),
        "depth_variants": list(DEPTH_VARIANTS),
        "selection": (
            "advance the best stage allocation and the best depth family to independent "
            "interaction and seeds 509/521 follow-ups"
        ),
    }
    payload["variant_configs"] = {
        variant: _variant_config(variant) for variant in VARIANTS
    }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload["architecture"] = dict.fromkeys(
        VARIANTS,
        (
            "K/P stage-allocation or full same-resolution depth diagnostic on the exact "
            "WL-H192 PathH4 all-resolution R2K3 control; no Lite block is used."
        ),
    )
    payload["references"] = {
        "K128_P128_D2222": {
            "variant": control.VARIANT,
            "parameters": 2_693_668,
        },
        "K128_P160x4_D2222": {
            "variant": capacity_insight.K128_P160,
            "parameters": 3_154_660,
        },
        "K160_P128x4_D2222": {
            "variant": capacity_insight.K160_P128,
            "parameters": 3_671_780,
        },
        "K160_P160x4_D2222": {
            "variant": capacity_insight.K160_P160,
            "parameters": 3_592_100,
        },
        "K128_P128x4_D2233": {
            "variant": capacity_insight.EXTRA_SR14_7,
            "parameters": 3_350_052,
        },
    }
    payload["source_sha256"]["capacity_factory"] = runtime.digest(
        Path("scripts/a2d_r2k3_capacity_factory.py")
    )
    payload["source_sha256"]["same_resolution_depth"] = runtime.digest(
        Path("src/lnet/pac_same_resolution_depth.py")
    )
    payload["source_sha256"]["stage_allocation_runner"] = runtime.digest(Path(__file__))
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
