#!/usr/bin/env python3
"""Train the full width-by-resolution factorial of same-resolution pole stages."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_capacity_factory as capacity
import a2d_r2k3_runtime as runtime
import a2d_r2k3_source_manifest as source_manifest
import run_a2d_r2k3_uniform_prenorm_gated_postfusion_imagenet100 as base
import torch
from torch import Tensor, nn

from lnet.complex_scan import ComplexScanBackbone
from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    resized_gated_transition,
)
from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanConfig, ComplexScanStage
    from lnet.complex_scan_types import ComplexField


WIDTHS = (64, 96, 128)
RESOLUTIONS = (56, 28, 14, 7)
READER_RANK = 2
KERNEL_SIZE = 3
SEEDS = runtime.DEFAULT_SEEDS
STAGE_NAMES = base.STAGE_NAMES
BASE_VARIANT_BY_WIDTH = dict(zip(WIDTHS, base.VARIANTS, strict=True))


@dataclass(frozen=True, slots=True)
class FactorialSpec:
    width: int
    resolutions: tuple[int, ...]
    post_hidden_ratio: float = capacity.DEFAULT_POST_HIDDEN_RATIO

    @property
    def post_hidden(self) -> int:
        return round(self.width * self.post_hidden_ratio)


def _resolutions_from_mask(mask: int) -> tuple[int, ...]:
    if not 0 <= mask < 2 ** len(RESOLUTIONS):
        message = f"invalid same-resolution placement mask: {mask}"
        raise ValueError(message)
    return tuple(resolution for index, resolution in enumerate(RESOLUTIONS) if mask & (1 << index))


def variant_name(width: int, resolutions: tuple[int, ...]) -> str:
    suffix = "Base" if not resolutions else "SR" + "-".join(map(str, resolutions))
    return f"K{width}-PF15K-{suffix}"


SPECS = {
    variant_name(width, resolutions): FactorialSpec(width, resolutions)
    for width in WIDTHS
    for mask in range(2 ** len(RESOLUTIONS))
    for resolutions in (_resolutions_from_mask(mask),)
}
VARIANTS = tuple(SPECS)
REFERENCE_VARIANTS = {
    variant_name(128, ()): "K128-PF15K-D1",
    variant_name(128, (7,)): "K128-PF15K-D2-FullSR7",
    variant_name(128, (14,)): "K128-PF15K-D3-FullSR14",
    variant_name(128, (14, 7)): "K128-PF15K-D4-FullSR14-7",
}
TRAIN_VARIANTS = tuple(variant for variant in VARIANTS if variant not in REFERENCE_VARIANTS)


class SameResolutionFactorialBackbone(ComplexScanBackbone):
    """Insert canonical full stages at any subset of 56/28/14/7 resolutions."""

    def __init__(
        self,
        source: ComplexScanBackbone,
        blocks: dict[int, SameResolutionPoleScanBlock],
    ) -> None:
        nn.Module.__init__(self)
        self.config = source.config
        self.stem = source.stem
        self.input_norm = source.input_norm
        self.precomplex_fc = source.precomplex_fc
        self.analysis = source.analysis
        self.stage1 = source.stage1
        self.stage2 = source.stage2
        self.stage3 = cast("Any", source).stage3
        self.terminal = source.terminal
        self.same_resolution_blocks = nn.ModuleDict(
            {str(resolution): block for resolution, block in blocks.items()}
        )
        self.descriptor_dim = source.descriptor_dim
        self.classifier = source.classifier

    def _apply_at(self, resolution: int, state: ComplexField) -> ComplexField:
        key = str(resolution)
        if key not in self.same_resolution_blocks:
            return state
        return cast("ComplexField", self.same_resolution_blocks[key](*state))

    def complex_features(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: Tensor,
    ) -> tuple[ComplexField, ComplexField, ComplexField]:
        excitation = self._initial_excitation(inputs)
        excitation = self._apply_at(56, excitation)
        state2, _ = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state2 = self._apply_at(28, state2)
        state3, _ = self.stage2(*state2)
        state3 = self._require_state(state3)
        state3 = self._apply_at(14, state3)
        state4, _ = self.stage3(*state3)
        state4 = self._require_state(state4)
        state4 = self._apply_at(7, state4)
        return state2, state3, state4

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        excitation = self._initial_excitation(inputs)
        excitation = self._apply_at(56, excitation)
        state2, descriptor1 = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state2 = self._apply_at(28, state2)
        state3, descriptor2 = self.stage2(*state2)
        state3 = self._require_state(state3)
        state3 = self._apply_at(14, state3)
        state4, descriptor3 = self.stage3(*state3)
        state4 = self._require_state(state4)
        state4 = self._apply_at(7, state4)
        _, descriptor4 = self.terminal(*state4)
        descriptor = torch.cat(
            (descriptor1, descriptor2, descriptor3, descriptor4),
            dim=-1,
        )
        if descriptor.shape[-1] != self.descriptor_dim:
            message = "same-resolution placement changed the established Raw-Q width"
            raise RuntimeError(message)
        return descriptor


def _resize_postfusion(model: ComplexScanBackbone, hidden_modes: int) -> None:
    """Change only PostFusion workspace width; preserve the merge projections."""
    for name in STAGE_NAMES[:3]:
        stage = getattr(model, name)
        source = stage.augmented
        if type(source) is not GatedPoleExcitationS2DTransition:
            message = f"{name} lost the established gated transition"
            raise TypeError(message)
        stage.augmented = resized_gated_transition(source, hidden_modes)


def _make_block(
    *,
    width: int,
    pole_modes: int | None = None,
    resolution: int,
    pole_template: ComplexScanStage,
    post_hidden: int,
    seed_offset: int = 0,
) -> SameResolutionPoleScanBlock:
    initial_seed = torch.initial_seed()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(
            (initial_seed + resolution * 1_001 + seed_offset) % (2**63 - 1)
        )
        return SameResolutionPoleScanBlock(
            width,
            pole_modes=pole_modes,
            reader_rank=READER_RANK,
            kernel_size=KERNEL_SIZE,
            pole_template=pole_template,
            post_hidden=post_hidden,
        )


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        message = f"unsupported same-resolution factorial variant: {variant}"
        raise ValueError(message) from error

    model = base._build(BASE_VARIANT_BY_WIDTH[spec.width], config)
    _resize_postfusion(model, spec.post_hidden)
    templates = {
        56: model.stage1,
        28: model.stage2,
        14: cast("Any", model).stage3,
        7: model.terminal,
    }
    blocks = {
        resolution: _make_block(
            width=spec.width,
            resolution=resolution,
            pole_template=templates[resolution],
            post_hidden=spec.post_hidden,
        )
        for resolution in spec.resolutions
    }
    if blocks:
        model = SameResolutionFactorialBackbone(model, blocks)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    if model.descriptor_dim != 16 * spec.width:
        message = f"{variant} changed its four-stage Raw-Q descriptor"
        raise RuntimeError(message)
    for name in STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if (
            type(transition) is not GatedPoleExcitationS2DTransition
            or transition.input_modes != spec.width
            or transition.output_modes != spec.width
            or transition.post_hidden != spec.post_hidden
            or transition.post_fusion.hidden_modes != spec.post_hidden
        ):
            message = f"{variant}/{name} changed its PF1.5K transition contract"
            raise RuntimeError(message)

    actual = getattr(model, "same_resolution_blocks", nn.ModuleDict())
    if set(actual) != {str(resolution) for resolution in spec.resolutions}:
        message = f"{variant} changed its same-resolution placement"
        raise RuntimeError(message)
    for block in actual.values():
        if (
            type(block) is not SameResolutionPoleScanBlock
            or block.modes != spec.width
            or type(block.path_collapse) is not PathOnlyCollapse
            or block.post_fusion.hidden_modes != spec.post_hidden
        ):
            message = f"{variant} lost a canonical full pole stage"
            raise RuntimeError(message)


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = deepcopy(base._variant_config(BASE_VARIANT_BY_WIDTH[spec.width]))
    payload["backbone"]["name"] = f"A2D-{variant}"
    payload["backbone"]["transition"]["post_fusion"] = {
        "operator": "gated complex PostFusion",
        "hidden_modes": spec.post_hidden,
        "hidden_ratio": spec.post_hidden / spec.width,
    }
    payload["backbone"]["same_resolution_factorial"] = {
        "candidate_resolutions": list(RESOLUTIONS),
        "enabled_resolutions": list(spec.resolutions),
        "reader": "pre-CRMSNorm orthogonal rank-2 strict-complex K3",
        "scan": "D4 associative product scan retaining full-resolution states",
        "collapse": "canonical shared per-mode GWL 4-to-8-to-1 Cartesian-SiLU",
        "carry": "identity excitation carry",
        "post_fusion": f"gated complex {spec.width}-to-{spec.post_hidden}-to-{spec.width}",
        "descriptor_policy": "no extra Q; established Q1-Q4 head unchanged",
    }
    return payload


def _selected_variants(args: Namespace) -> tuple[str, ...]:
    selected = getattr(args, "variants", None)
    if not selected:
        return VARIANTS
    if isinstance(selected, str):
        return (selected,)
    return tuple(cast("list[str] | tuple[str, ...]", selected))


def _contract(args: Namespace) -> dict[str, Any]:
    payload = runtime.base_contract(args)
    selected = _selected_variants(args)
    config = runtime.model_config()
    payload["schema"] = "lnet.a2d.r2k3.same_resolution_factorial.v1"
    payload["evidence_status"] = "candidate-specific CPU and compiled CUDA smoke required"
    payload["variants"] = list(selected)
    payload["seeds"] = list(SEEDS)
    payload["factorial"] = {
        "widths": list(WIDTHS),
        "resolutions": list(RESOLUTIONS),
        "cells": len(VARIANTS),
        "post_fusion_ratio": capacity.DEFAULT_POST_HIDDEN_RATIO,
    }
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in selected}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in selected
    }
    payload["references"] = {
        "uniform_pf2k": {
            str(width): (
                "/home/qlab/experiments/alphabet/"
                "r2k3-uniform-prenorm-gated-postfusion-20260815/runs/"
                f"{BASE_VARIANT_BY_WIDTH[width]}"
            )
            for width in WIDTHS
        },
        "existing_k128_pf15k": REFERENCE_VARIANTS,
    }
    payload["architecture"] = dict.fromkeys(
        selected,
        (
            "Uniform K=P R2K3 Raw-Q backbone with PF1.5K gated PostFusion and a "
            "factorial subset of canonical full same-resolution pole stages; only "
            "the spatial coarsening/carry policy differs from ordinary stages."
        ),
    )
    payload["source_sha256"]["same_resolution_stage"] = runtime.digest(
        Path("src/lnet/pac_same_resolution_depth.py")
    )
    payload["source_sha256"]["factorized_complex_scan_reader"] = runtime.digest(
        Path("src/lnet/pac_factorized_complex_scan_reader.py")
    )
    payload["source_sha256"]["gated_post_fusion"] = runtime.digest(
        Path("src/lnet/pac_gated_post_fusion.py")
    )
    payload["source_sha256"]["same_resolution_factorial_runner"] = runtime.digest(Path(__file__))
    payload["source_sha256"]["r2k3_runtime"] = runtime.digest(Path("scripts/a2d_r2k3_runtime.py"))
    payload["source_sha256"]["r2k3_capacity_factory"] = runtime.digest(
        Path("scripts/a2d_r2k3_capacity_factory.py")
    )
    repo = Path(__file__).resolve().parents[1]
    dependency_paths = source_manifest.dependency_paths(repo, (Path(__file__).stem,))
    payload["source_sha256"]["r2k3_dependency_tree"] = source_manifest.fingerprint(
        repo,
        dependency_paths,
    )
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
