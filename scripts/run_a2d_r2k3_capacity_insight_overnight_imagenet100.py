#!/usr/bin/env python3
"""Train the controlled K/P/depth capacity follow-up queue."""

from __future__ import annotations

# This campaign intentionally composes the established private builders.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# pyright: reportUnusedFunction=false
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_capacity_factory as capacity
import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as control
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as factorial
import run_a2d_r2k3_uniform_prenorm_gated_postfusion_imagenet100 as uniform
from torch import nn

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_gated_post_fusion import (
    GatedComplexPostFusion,
    GatedPoleExcitationS2DTransition,
)
from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.complex_scan_types import ComplexField


TERM_P96 = "K128-PF15K-SR56-28-14-7-WLPost-PathH4-TermP96"
TERM_P160 = "K128-PF15K-SR56-28-14-7-WLPost-PathH4-TermP160"
EARLY_THIN = "K128-P64-64-96-128-SR56-28-14-7-WLPostH192-PathH4"
K160_P128 = "K160-P128x4-SR56-28-14-7-WLPostH192-PathH4"
K128_P160 = "K128-P160x4-SR56-28-14-7-WLPostH192-PathH4"
K160_P160 = "K160-P160x4-SR56-28-14-7-WLPostH192-PathH4"
EXTRA_SR14_7 = "K128-P128x4-SR56-28-14-7-ExtraSR14-7-WLPostH192-PathH4"

VARIANTS = (
    TERM_P96,
    TERM_P160,
    EARLY_THIN,
    K160_P128,
    K128_P160,
    K160_P160,
    EXTRA_SR14_7,
)
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = factorial.RESOLUTIONS
WIDTHS = (128, 160)
STAGE_NAMES = runtime.STAGE_NAMES
SameResolutionFactorialBackbone = factorial.SameResolutionFactorialBackbone
POST_HIDDEN = control.POST_HIDDEN
PATH_HIDDEN = control.PATH_HIDDEN
EXTRA_BLOCK_SEED_OFFSET = 1_000_003
JOBS_BY_GPU = {
    0: (TERM_P96, TERM_P160, K160_P128, EXTRA_SR14_7),
    1: (EARLY_THIN, K128_P160, K160_P160),
}


@dataclass(frozen=True, slots=True)
class InsightSpec:
    excitation_modes: tuple[int, int, int, int]
    pole_modes: tuple[int, int, int, int]
    terminal_resize: bool = False
    extra_resolutions: tuple[int, ...] = ()

    @property
    def width(self) -> int:
        return self.excitation_modes[0]

    @property
    def excitation_width(self) -> int:
        return self.width

    @property
    def descriptor_dim(self) -> int:
        return 4 * sum(self.pole_modes)

    @property
    def q4_dim(self) -> int:
        return 4 * self.pole_modes[-1]

    @property
    def resolutions(self) -> tuple[int, ...]:
        return RESOLUTIONS

    @property
    def block_pole_modes(self) -> tuple[int, int, int, int]:
        if self.terminal_resize:
            return (128, 128, 128, 128)
        return self.pole_modes


SPECS = {
    TERM_P96: InsightSpec((128,) * 4, (128, 128, 128, 96), terminal_resize=True),
    TERM_P160: InsightSpec((128,) * 4, (128, 128, 128, 160), terminal_resize=True),
    EARLY_THIN: InsightSpec((128,) * 4, (64, 64, 96, 128)),
    K160_P128: InsightSpec((160,) * 4, (128,) * 4),
    K128_P160: InsightSpec((128,) * 4, (160,) * 4),
    K160_P160: InsightSpec((160,) * 4, (160,) * 4),
    EXTRA_SR14_7: InsightSpec((128,) * 4, (128,) * 4, extra_resolutions=(14, 7)),
}


def variant_name(width: int, resolutions: tuple[int, ...]) -> str:
    return factorial.variant_name(width, resolutions)


class ExtraSameResolutionBackbone(SameResolutionFactorialBackbone):
    """Append independent full blocks without renaming established block state."""

    def __init__(
        self,
        source: ComplexScanBackbone,
        extras: dict[int, SameResolutionPoleScanBlock],
    ) -> None:
        existing = dict(source.same_resolution_blocks.items())
        super().__init__(source, existing)
        self.extra_same_resolution_blocks = nn.ModuleDict(
            {str(resolution): block for resolution, block in extras.items()}
        )

    def _apply_at(self, resolution: int, state: ComplexField) -> ComplexField:
        state = super()._apply_at(resolution, state)
        key = str(resolution)
        if key not in self.extra_same_resolution_blocks:
            return state
        return cast("ComplexField", self.extra_same_resolution_blocks[key](*state))


def _templates(model: ComplexScanBackbone) -> dict[int, Any]:
    return {
        56: model.stage1,
        28: model.stage2,
        14: cast("Any", model).stage3,
        7: model.terminal,
    }


def _build_all_resolution(spec: InsightSpec, config: ComplexScanConfig) -> ComplexScanBackbone:
    capacity_spec = capacity.CapacitySpec(
        spec.excitation_modes,
        spec.pole_modes,
        post_hidden_ratio=2.0,
    )
    model = capacity._build_spec(capacity_spec, config)
    uniform._install_combined_transition(model)
    factorial._resize_postfusion(model, POST_HIDDEN)
    templates = _templates(model)
    blocks = {
        resolution: factorial._make_block(
            width=spec.excitation_modes[index],
            pole_modes=spec.block_pole_modes[index],
            resolution=resolution,
            pole_template=templates[resolution],
            post_hidden=POST_HIDDEN,
        )
        for index, resolution in enumerate(RESOLUTIONS)
    }
    model = SameResolutionFactorialBackbone(model, blocks)
    control._install_path_h4(model)
    return model


def _append_extra_blocks(
    model: ComplexScanBackbone,
    resolutions: tuple[int, ...],
) -> ComplexScanBackbone:
    templates = _templates(model)
    extras = {
        resolution: factorial._make_block(
            width=128,
            pole_modes=128,
            resolution=resolution,
            pole_template=templates[resolution],
            post_hidden=POST_HIDDEN,
            seed_offset=EXTRA_BLOCK_SEED_OFFSET,
        )
        for resolution in resolutions
    }
    for block in extras.values():
        block.path_collapse = PathOnlyCollapse(128, path_hidden=PATH_HIDDEN)
    return ExtraSameResolutionBackbone(model, extras)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported capacity-insight variant: {variant}") from error

    model = _build_spec_model(spec, config)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model_for_spec(model, spec, variant)
    return model


def _build_spec_model(
    spec: InsightSpec,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    if spec.terminal_resize:
        model = control._build(control.VARIANT, config)
        capacity.resize_terminal_poles_(model, target_poles=spec.pole_modes[-1])
    elif spec.extra_resolutions:
        model = control._build(control.VARIANT, config)
        model = _append_extra_blocks(model, spec.extra_resolutions)
    else:
        model = _build_all_resolution(spec, config)
    return model


def _assert_path_collapse(collapse: nn.Module, pole_modes: int, label: str) -> None:
    if (
        type(collapse) is not PathOnlyCollapse
        or collapse.modes != pole_modes
        or collapse.path_input.output_paths != PATH_HIDDEN
        or collapse.path_output.input_paths != PATH_HIDDEN
        or collapse.output_paths != 1
    ):
        raise RuntimeError(f"{label} changed the PathH4 contract")


def _assert_block(
    block: nn.Module,
    *,
    excitation_modes: int,
    pole_modes: int,
    label: str,
) -> None:
    if (
        type(block) is not SameResolutionPoleScanBlock
        or block.modes != excitation_modes
        or block.pole_modes != pole_modes
        or block.reader.input_modes != excitation_modes
        or block.reader.output_modes != pole_modes
        or block.reader.rank != capacity.READER_RANK
        or block.reader.kernel_size != capacity.KERNEL_SIZE
        or block.post_fusion.modes != excitation_modes
        or block.post_fusion.hidden_modes != POST_HIDDEN
    ):
        raise RuntimeError(f"{label} changed its full-block capacity contract")
    _assert_path_collapse(block.path_collapse, pole_modes, label)


def _assert_model(model: nn.Module, variant: str) -> None:
    _assert_model_for_spec(model, SPECS[variant], variant)


def _assert_model_for_spec(
    model: nn.Module,
    spec: InsightSpec,
    label: str,
) -> None:
    if model.descriptor_dim != spec.descriptor_dim or model.classifier.input_dim != spec.q4_dim:
        raise RuntimeError(f"{label} changed its Raw-Q descriptor/head contract")

    stages = tuple(getattr(model, name) for name in STAGE_NAMES)
    for index, (name, stage) in enumerate(zip(STAGE_NAMES, stages, strict=True)):
        excitation_modes = spec.excitation_modes[index]
        pole_modes = spec.pole_modes[index]
        reader = stage.pole_input_projection
        if (
            stage.input_modes != excitation_modes
            or stage.modes != pole_modes
            or type(reader) is not FactorizedComplexConv2dReader
            or reader.input_modes != excitation_modes
            or reader.output_modes != pole_modes
            or reader.rms_reference_modes != pole_modes
        ):
            raise RuntimeError(f"{label}/{name} changed its K-to-P reader contract")
        if name == "terminal":
            if stage.output_modes is not None:
                raise RuntimeError(f"{label} terminal unexpectedly emits an excitation")
            continue
        transition = stage.augmented
        if (
            type(transition) is not GatedPoleExcitationS2DTransition
            or transition.input_modes != pole_modes
            or transition.excitation_modes != excitation_modes
            or transition.output_modes != spec.excitation_modes[index + 1]
            or type(transition.post_fusion) is not GatedComplexPostFusion
            or transition.post_fusion.hidden_modes != POST_HIDDEN
        ):
            raise RuntimeError(f"{label}/{name} changed its WL-H192 transition contract")
        _assert_path_collapse(stage.quadrant_path_mode_combiner, pole_modes, f"{label}/{name}")

    for index, resolution in enumerate(RESOLUTIONS):
        blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
        _assert_block(
            blocks[str(resolution)],
            excitation_modes=spec.excitation_modes[index],
            pole_modes=spec.block_pole_modes[index],
            label=f"{label}/SR{resolution}",
        )

    extras = cast(
        "nn.ModuleDict",
        getattr(model, "extra_same_resolution_blocks", nn.ModuleDict()),
    )
    if set(extras) != {str(value) for value in spec.extra_resolutions}:
        raise RuntimeError(f"{label} changed its extra low-resolution depth")
    for resolution in spec.extra_resolutions:
        index = RESOLUTIONS.index(resolution)
        _assert_block(
            extras[str(resolution)],
            excitation_modes=spec.excitation_modes[index],
            pole_modes=spec.pole_modes[index],
            label=f"{label}/ExtraSR{resolution}",
        )


def _variant_config(variant: str) -> dict[str, Any]:
    return _variant_config_for_spec(variant, SPECS[variant])


def _variant_config_for_spec(
    variant: str,
    spec: InsightSpec,
) -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    backbone = payload["backbone"]
    backbone["name"] = f"A2D-{variant}"
    backbone["excitation_schedule"] = list(spec.excitation_modes)
    backbone["pole_schedule"] = list(spec.pole_modes)
    backbone["descriptor_dim"] = spec.descriptor_dim
    backbone["stem"] = {
        "convolutions": (
            f"3-to-32 stride2 then 32-to-{2 * spec.excitation_modes[0]} stride2"
        ),
        "precomplex_mixer": (
            f"residual Linear{2 * spec.excitation_modes[0]}-GELU-"
            f"Linear{2 * spec.excitation_modes[0]}"
        ),
        "interface_norm": f"RMSNorm{2 * spec.excitation_modes[0]}",
        "complex_projection": (
            f"semi-orthogonal Linear{2 * spec.excitation_modes[0]}-to-"
            f"{2 * spec.excitation_modes[0]} then real/imag split"
        ),
    }
    backbone["pole_input"]["shape_by_stage"] = [
        f"{excitation}-to-{poles}"
        for excitation, poles in zip(spec.excitation_modes, spec.pole_modes, strict=True)
    ]
    backbone["transition"]["post_fusion"] = {
        "operator": "gated complex PostFusion",
        "hidden_modes": POST_HIDDEN,
        "hidden_ratio_by_stage": [POST_HIDDEN / modes for modes in spec.excitation_modes[:3]],
    }
    backbone["descriptor"]["shape_by_stage"] = [4 * modes for modes in spec.pole_modes]
    backbone["pole_initialization"]["modes_per_stage"] = list(spec.pole_modes)
    backbone["pole_initialization"]["radial_levels"] = [modes // 8 for modes in spec.pole_modes]
    backbone["same_resolution_factorial"].update(
        {
            "excitation_modes_by_resolution": dict(
                zip(map(str, RESOLUTIONS), spec.excitation_modes, strict=True)
            ),
            "pole_modes_by_resolution": dict(
                zip(map(str, RESOLUTIONS), spec.block_pole_modes, strict=True)
            ),
            "post_fusion": f"gated complex with fixed hidden width {POST_HIDDEN}",
            "extra_repetitions": list(spec.extra_resolutions),
        }
    )
    payload["head"] = {
        "descriptor_source": f"terminal raw Q only; final {spec.q4_dim} coordinates",
        "operator": f"BatchNorm{spec.q4_dim}-affine-false-Linear100",
        "auxiliary": False,
    }
    payload["initialization_policy"] = (
        "joint fresh initialization at every requested K/P width; no nested copies or zeroed head columns"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    payload = runtime.base_contract(args)
    payload["schema"] = "lnet.a2d.r2k3.capacity_insight_overnight.v1"
    payload["evidence_status"] = "all variants require CPU and compiled CUDA full-batch smoke"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in VARIANTS}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload["architecture"] = dict.fromkeys(
        VARIANTS,
        (
            "Controlled capacity follow-up on WL-H192 PathH4 all-resolution R2K3; "
            "only the declared excitation width, pole schedule, terminal width, or "
            "additional SR14/SR7 depth differs."
        ),
    )
    payload["references"] = {
        "P128_control": {
            "variant": control.VARIANT,
            "parameters": 2_693_668,
        },
        "P192_fresh": {
            "variant": "K128-PF15K-SR56-28-14-7-WLPost-PathH4-TermP192",
            "parameters": 2_754_596,
        },
    }
    payload["source_sha256"]["capacity_factory"] = runtime.digest(
        Path("scripts/a2d_r2k3_capacity_factory.py")
    )
    payload["source_sha256"]["same_resolution_depth"] = runtime.digest(
        Path("src/lnet/pac_same_resolution_depth.py")
    )
    payload["source_sha256"]["capacity_insight_runner"] = runtime.digest(Path(__file__))
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
