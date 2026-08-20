#!/usr/bin/env python3
"""Train the all-resolution WL-H192 model with H4 path collapse."""

from __future__ import annotations

# The frozen campaign runner intentionally exposes private construction hooks.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as base

from lnet.pac_gated_post_fusion import GatedComplexPostFusion
from lnet.pac_phase_gated_transition import PathOnlyCollapse

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


BASE_VARIANT = base.variant_name(128, base.RESOLUTIONS)
VARIANT = "K128-PF15K-SR56-28-14-7-WLPost-PathH4"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
SPECS = {VARIANT: base.SPECS[BASE_VARIANT]}
STAGE_NAMES = base.STAGE_NAMES
RESOLUTIONS = base.RESOLUTIONS
SameResolutionFactorialBackbone = base.SameResolutionFactorialBackbone
PATH_HIDDEN = 4
POST_HIDDEN = 192
PATH_BLOCKS = 7
JOBS_BY_GPU = {0: VARIANTS}


def _path_collapses(model: nn.Module) -> tuple[Any, ...]:
    main = tuple(
        getattr(model, name).quadrant_path_mode_combiner
        for name in STAGE_NAMES[:3]
    )
    same_resolution = tuple(
        block.path_collapse for block in model.same_resolution_blocks.values()
    )
    return (*main, *same_resolution)


def _postfusions(model: nn.Module) -> tuple[Any, ...]:
    main = tuple(getattr(model, name).augmented for name in STAGE_NAMES[:3])
    same_resolution = tuple(model.same_resolution_blocks.values())
    return (*main, *same_resolution)


def _install_path_hidden(model: nn.Module, path_hidden: int) -> None:
    if path_hidden <= 0:
        raise ValueError("path hidden width must be positive")
    for name in STAGE_NAMES[:3]:
        stage = getattr(model, name)
        stage.quadrant_path_mode_combiner = PathOnlyCollapse(
            stage.modes,
            path_hidden=path_hidden,
        )
    for block in model.same_resolution_blocks.values():
        block.path_collapse = PathOnlyCollapse(
            block.pole_modes,
            path_hidden=path_hidden,
        )


def _install_path_h4(model: nn.Module) -> None:
    _install_path_hidden(model, PATH_HIDDEN)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported WL-H192 PathH4 variant: {variant}")
    model = base._build(BASE_VARIANT, config)
    _install_path_h4(model)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model)
    return model


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported WL-H192 PathH4 variant: {variant}")
    collapses = _path_collapses(model)
    if len(collapses) != PATH_BLOCKS or any(
        type(collapse) is not PathOnlyCollapse
        or collapse.modes != 128
        or collapse.path_count != 4
        or collapse.path_input.output_paths != PATH_HIDDEN
        or collapse.path_output.input_paths != PATH_HIDDEN
        or collapse.output_paths != 1
        for collapse in collapses
    ):
        raise RuntimeError("WL-H192 PathH4 changed its seven collapse contracts")
    postfusions = _postfusions(model)
    if len(postfusions) != PATH_BLOCKS or any(
        type(owner.post_fusion) is not GatedComplexPostFusion
        or owner.post_fusion.modes != 128
        or owner.post_fusion.hidden_modes != POST_HIDDEN
        for owner in postfusions
    ):
        raise RuntimeError("WL-H192 PathH4 changed the established WL PostFusion")
    if model.descriptor_dim != 2048 or model.classifier.input_dim != 512:
        raise RuntimeError("WL-H192 PathH4 changed the established Raw-Q head")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(base._variant_config(BASE_VARIANT))
    payload["backbone"]["name"] = f"A2D-{VARIANT}"
    collapse = "shared per-mode GWL 4-to-4-to-1 with Cartesian SiLU"
    payload["backbone"]["transition"]["path_collapse"] = collapse
    payload["backbone"]["same_resolution_factorial"]["collapse"] = collapse
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = runtime.base_contract(args)
    config = runtime.model_config()
    payload["schema"] = "lnet.a2d.r2k3.same_resolution_all_wl192_pathh4.v1"
    payload["evidence_status"] = "CPU and compiled CUDA smoke required before training"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in _build(VARIANT, config).parameters())
    }
    payload["references"] = {
        "WL_H192_PathH8": {"parameters": 2_772_516, "best_validation_accuracy": 0.8726},
        "CL_H192_PathH4": {"parameters": 2_005_540, "best_validation_accuracy": 0.8666},
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact K=P=128 all-resolution WL-H192 best model with only all seven "
            "path hidden widths reduced from eight to four."
        )
    }
    payload["source_sha256"]["phase_gated_transition"] = runtime.digest(
        Path("src/lnet/pac_phase_gated_transition.py")
    )
    payload["source_sha256"]["wl192_pathh4_runner"] = runtime.digest(Path(__file__))
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
