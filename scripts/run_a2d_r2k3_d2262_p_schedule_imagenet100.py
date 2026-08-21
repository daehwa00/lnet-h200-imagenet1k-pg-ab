#!/usr/bin/env python3
"""Train the controlled D2262 pole-schedule follow-up on ImageNet-100."""

from __future__ import annotations

# ruff: noqa: EM102, SLF001, TRY003
# This campaign composes the frozen stage-allocation builder by explicit spec.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import a2d_r2k3_source_manifest as source_manifest
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as stage

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


A = "A-K128-P160-160-160-128-D2262"
B = "B-K128-P160-160-192-128-D2262"
C = "C-K128-P160-192-160-128-D2262"
D = "D-K128-P160-192-192-128-D2262"
E = "E-K128-P128-192-192-128-D2262"
F = "F-K128-P128-160-192-128-D2262"
VARIANTS = (A, B, C, D, E, F)
SEEDS = runtime.DEFAULT_SEEDS
K_SCHEDULE = (128, 128, 128, 128)
DEPTH = (2, 2, 6, 2)
EXTRA_BLOCKS = (0, 0, 4, 0)
POST_FUSION_HIDDEN = 192
PATH_HIDDEN = 4
P_SCHEDULES = {
    A: (160, 160, 160, 128),
    B: (160, 160, 192, 128),
    C: (160, 192, 160, 128),
    D: (160, 192, 192, 128),
    E: (128, 192, 192, 128),
    F: (128, 160, 192, 128),
}
PARAMETER_COUNTS = {
    A: 4_621_476,
    B: 4_793_892,
    C: 4_678_948,
    D: 4_851_364,
    E: 4_728_356,
    F: 4_670_884,
}
QUESTIONS = {
    A: "terminal P160-to-128 effect",
    B: "P14 expansion effect",
    C: "P28 expansion effect",
    D: "P28-by-P14 expansion interaction",
    E: "whether P56 can be reduced to 128",
    F: "whether P28 needs expansion from 160 to 192",
}
SPECS = {
    variant: stage.StageAllocationSpec(
        K_SCHEDULE,
        poles,
        extra_blocks=EXTRA_BLOCKS,
        family="d2262_p_schedule",
    )
    for variant, poles in P_SCHEDULES.items()
}
JOBS_BY_GPU = {0: VARIANTS}


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported D2262 P-schedule variant: {variant}") from error
    model = stage._build_for_spec(spec, variant, config)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    if (
        spec.excitation_modes != K_SCHEDULE
        or spec.depth != DEPTH
        or spec.extra_blocks != EXTRA_BLOCKS
        or stage.POST_HIDDEN != POST_FUSION_HIDDEN
        or stage.PATH_HIDDEN != PATH_HIDDEN
    ):
        raise RuntimeError(f"{variant} changed the frozen common condition")
    stage._assert_model_for_spec(model, spec, variant)


def _variant_config(variant: str) -> dict[str, Any]:
    payload = stage._variant_config_for_spec(variant, SPECS[variant])
    payload["experiment"].update(
        {
            "common_conditions": {
                "K_schedule": list(K_SCHEDULE),
                "depth": list(DEPTH),
                "post_fusion_hidden": POST_FUSION_HIDDEN,
                "path_hidden": PATH_HIDDEN,
                "reader": "rank-2 strict-complex K3",
                "head": "terminal Q4 affine",
            },
            "question": QUESTIONS[variant],
            "varied_dimension": "pole_schedule_only",
        }
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = runtime.base_contract(args)
    payload.get("runtime", {}).pop("hostname", None)
    payload.get("recipe", {}).pop("cpu_affinity", None)
    payload["schema"] = "lnet.a2d.r2k3.d2262_p_schedule.v1"
    payload["evidence_status"] = (
        "six-cell seed501 controlled P-schedule follow-up; H200 sequential queue"
    )
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in VARIANTS}
    payload["parameter_counts"] = dict(PARAMETER_COUNTS)
    payload["screen"] = {
        "baseline": {
            "pole_schedule": [160, 160, 160, 160],
            "provenance": "existing evidence external to this six-run public campaign",
        },
        "comparisons": [
            "existing P160x4 versus A: terminal reduction",
            "A versus B: P14 main effect",
            "A versus C: P28 main effect",
            "B and C versus D: P28-by-P14 interaction",
            "D versus E: P56 reduction",
            "E versus F: P28 192-versus-160",
        ],
        "only_varied_dimension": "P schedule",
    }
    payload["source_sha256"]["stage_allocation_builder"] = runtime.digest(
        Path("scripts/run_a2d_r2k3_stage_allocation_screen_imagenet100.py")
    )
    payload["source_sha256"]["d2262_p_schedule_runner"] = runtime.digest(Path(__file__))
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
