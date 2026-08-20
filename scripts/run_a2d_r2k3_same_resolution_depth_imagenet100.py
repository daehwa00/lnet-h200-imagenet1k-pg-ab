#!/usr/bin/env python3
"""Train full K128 R2K3 stages with and without spatial coarsening."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_uniform_prenorm_gated_postfusion_imagenet100 as base
import torch
from torch import Tensor, nn

from lnet.complex_scan import ComplexScanBackbone
from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    resized_gated_transition,
)
from lnet.pac_same_resolution_depth import (
    D4_PATHS,
    SameResolutionPoleScanBlock,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanConfig, ComplexScanStage
    from lnet.complex_scan_types import ComplexField


BASE_VARIANT = "R2K3-K128-P128x4-PreNorm-GatedPostFusion"
D0 = "K128-PF2K-D0"
D1 = "K128-PF15K-D1"
D2 = "K128-PF15K-D2-FullSR7"
D3 = "K128-PF15K-D3-FullSR14"
D4 = "K128-PF15K-D4-FullSR14-7"
VARIANTS = (D0, D1, D2, D3, D4)
TRAIN_VARIANTS = (D2, D3, D4)
SEEDS = runtime.DEFAULT_SEEDS
MODES = 128
READER_RANK = 2
KERNEL_SIZE = 3
STAGE_NAMES = base.STAGE_NAMES


@dataclass(frozen=True, slots=True)
class DepthSpec:
    post_hidden_numerator: int
    post_hidden_denominator: int
    block14: bool = False
    block7: bool = False

    def post_hidden(self, modes: int) -> int:
        numerator = modes * self.post_hidden_numerator
        if numerator % self.post_hidden_denominator:
            message = "PostFusion ratio must produce an integral hidden width"
            raise ValueError(message)
        return numerator // self.post_hidden_denominator


SPECS = {
    D0: DepthSpec(2, 1),
    D1: DepthSpec(3, 2),
    D2: DepthSpec(3, 2, block7=True),
    D3: DepthSpec(3, 2, block14=True),
    D4: DepthSpec(3, 2, block14=True, block7=True),
}


class SameResolutionDepthBackbone(ComplexScanBackbone):
    """Insert full pole stages without changing descriptors or resolution."""

    def __init__(
        self,
        source: ComplexScanBackbone,
        *,
        block14: nn.Module | None,
        block7: nn.Module | None,
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
        self.block14 = block14
        self.block7 = block7
        self.descriptor_dim = source.descriptor_dim
        self.classifier = source.classifier

    @staticmethod
    def _apply_block(block: nn.Module | None, state: ComplexField) -> ComplexField:
        if block is None:
            return state
        return cast("ComplexField", block(*state))

    def complex_features(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: Tensor,
    ) -> tuple[ComplexField, ComplexField, ComplexField]:
        excitation = self._initial_excitation(inputs)
        state2, _ = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state3, _ = self.stage2(*state2)
        state3 = self._apply_block(self.block14, self._require_state(state3))
        state4, _ = self.stage3(*state3)
        state4 = self._apply_block(self.block7, self._require_state(state4))
        return state2, state3, state4

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        excitation = self._initial_excitation(inputs)
        state2, descriptor1 = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state3, descriptor2 = self.stage2(*state2)
        state3 = self._apply_block(self.block14, self._require_state(state3))
        state4, descriptor3 = self.stage3(*state3)
        state4 = self._apply_block(self.block7, self._require_state(state4))
        _, descriptor4 = self.terminal(*state4)
        descriptor = torch.cat(
            (descriptor1, descriptor2, descriptor3, descriptor4),
            dim=-1,
        )
        if descriptor.shape[-1] != self.descriptor_dim:
            message = "same-resolution depth changed the established Raw-Q width"
            raise RuntimeError(message)
        return descriptor


def _resize_postfusion(model: ComplexScanBackbone, hidden_modes: int) -> None:
    """Change only PostFusion workspace width; retain memory/carry projections."""
    for name in STAGE_NAMES[:3]:
        stage = getattr(model, name)
        source = stage.augmented
        if type(source) is not GatedPoleExcitationS2DTransition:
            message = f"{name} lost the established gated transition"
            raise TypeError(message)
        stage.augmented = resized_gated_transition(source, hidden_modes)


def _make_block(
    *,
    enabled: bool,
    pole_template: ComplexScanStage,
    post_hidden: int,
    seed_offset: int,
) -> nn.Module | None:
    if not enabled:
        return None
    initial_seed = torch.initial_seed()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((initial_seed + seed_offset) % (2**63 - 1))
        return SameResolutionPoleScanBlock(
            MODES,
            reader_rank=READER_RANK,
            kernel_size=KERNEL_SIZE,
            pole_template=pole_template,
            post_hidden=post_hidden,
        )


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        message = f"unsupported same-resolution depth variant: {variant}"
        raise ValueError(message) from error

    model = cast("ComplexScanBackbone", base._build(BASE_VARIANT, config))
    hidden_modes = spec.post_hidden(MODES)
    if hidden_modes != 2 * MODES:
        _resize_postfusion(model, hidden_modes)
    block14 = _make_block(
        enabled=spec.block14,
        pole_template=cast("Any", model).stage3,
        post_hidden=hidden_modes,
        seed_offset=14_014,
    )
    block7 = _make_block(
        enabled=spec.block7,
        pole_template=model.terminal,
        post_hidden=hidden_modes,
        seed_offset=7_007,
    )
    if block14 is not None or block7 is not None:
        model = SameResolutionDepthBackbone(
            model,
            block14=block14,
            block7=block7,
        )
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    if model.descriptor_dim != 4 * D4_PATHS * MODES:
        message = f"{variant} changed the K128 four-stage Raw-Q descriptor"
        raise RuntimeError(message)
    hidden_modes = spec.post_hidden(MODES)
    for name in STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if (
            type(transition) is not GatedPoleExcitationS2DTransition
            or transition.post_hidden != hidden_modes
            or transition.post_fusion.hidden_modes != hidden_modes
        ):
            message = f"{variant}/{name} changed its PostFusion-width contract"
            raise RuntimeError(message)

    expected = (spec.block14, spec.block7)
    actual = (
        getattr(model, "block14", None),
        getattr(model, "block7", None),
    )
    for resolution, enabled, block in zip((14, 7), expected, actual, strict=True):
        if not enabled:
            if block is not None:
                message = f"{variant} unexpectedly added a {resolution}x{resolution} block"
                raise RuntimeError(message)
        elif (
            type(block) is not SameResolutionPoleScanBlock
            or block.modes != MODES
            or block.post_fusion.hidden_modes != hidden_modes
        ):
            message = f"{variant} lost its full {resolution}x{resolution} pole stage"
            raise RuntimeError(message)


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = deepcopy(base._variant_config(BASE_VARIANT))
    payload["backbone"]["name"] = f"A2D-{variant}"
    payload["backbone"]["transition"]["post_fusion"] = {
        "operator": "gated complex PostFusion",
        "hidden_modes": spec.post_hidden(MODES),
        "hidden_ratio": spec.post_hidden(MODES) / MODES,
    }
    payload["backbone"]["same_resolution_depth"] = {
        "block14": "full pole stage" if spec.block14 else None,
        "block7": "full pole stage" if spec.block7 else None,
        "reader": "pre-CRMSNorm orthogonal rank-2 strict-complex K3",
        "scan": "D4 associative product scan retaining all full-resolution states",
        "collapse": "shared per-mode GWL 4-to-8-to-1 Cartesian-SiLU",
        "carry": "identity excitation carry; no S2D when resolution is unchanged",
        "post_fusion": f"gated complex {MODES}-to-{spec.post_hidden(MODES)}-to-{MODES}",
        "descriptor_policy": "no extra Q; established Q1-Q4 head remains unchanged",
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
    payload["schema"] = "lnet.a2d.r2k3.same_resolution_full_stage.v1"
    payload["evidence_status"] = "candidate-specific CPU and compiled CUDA smoke required"
    payload["variants"] = list(selected)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in selected}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in selected
    }
    payload["external_baseline"] = {
        "variant": D0,
        "source_variant": BASE_VARIANT,
        "checkpoint": (
            "/home/qlab/experiments/alphabet/"
            "r2k3-uniform-prenorm-gated-postfusion-20260815/runs/"
            f"{BASE_VARIANT}/checkpoints/{BASE_VARIANT}__seed501.pt"
        ),
        "validation_accuracy": 84.96,
        "policy": "reuse completed D0; do not retrain",
    }
    architecture = (
        "K=P=128 pre-norm R2K3 Raw-Q backbone with controlled gated-PostFusion "
        "workspace and optional full same-resolution pole stages whose only transition "
        "change is identity carry instead of coarsening plus S2D; "
        "all standard transitions, Q1-Q4 descriptor, affine Q4 head, and recipe remain fixed."
    )
    payload["architecture"] = dict.fromkeys(selected, architecture)
    payload["source_sha256"]["same_resolution_depth"] = runtime.digest(
        Path("src/lnet/pac_same_resolution_depth.py")
    )
    payload["source_sha256"]["same_resolution_depth_runner"] = runtime.digest(Path(__file__))
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
