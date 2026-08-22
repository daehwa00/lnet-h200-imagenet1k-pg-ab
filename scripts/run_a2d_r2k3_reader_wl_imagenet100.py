#!/usr/bin/env python3
"""Compare strict-complex and gated widely-linear rank-2 K3 Readers."""

from __future__ import annotations

# ruff: noqa: EM101, EM102, SLF001, TRY003
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import a2d_r2k3_source_manifest as source_manifest
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as stage

from lnet.pac_factorized_complex_scan_reader import (
    FactorizedComplexConv2dReader,
    GatedWidelyLinearFactorizedComplexConv2dReader,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


STRICT = "StrictReader-K96-128-128-128-P128-192-192-128-D2262"
WL = "WLReader-K96-128-128-128-P128-192-192-128-D2262"
VARIANTS = (STRICT, WL)
SEEDS = runtime.DEFAULT_SEEDS
K_SCHEDULE = (96, 128, 128, 128)
P_SCHEDULE = (128, 192, 192, 128)
EXTRA_BLOCKS = (0, 0, 4, 0)
DEPTH = (2, 2, 6, 2)
POST_FUSION_HIDDEN = 192
PATH_HIDDEN = 4
READER_COUNT = 12
SPEC = stage.StageAllocationSpec(
    K_SCHEDULE,
    P_SCHEDULE,
    extra_blocks=EXTRA_BLOCKS,
    family="strict_vs_widely_linear_reader",
)
PARAMETER_COUNTS = {
    STRICT: 4_578_244,
    WL: 5_669_828,
}
JOBS_BY_GPU = {0: VARIANTS}


def _reader_slots(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    slots = [
        (f"main.{name}", getattr(model, name), "pole_input_projection")
        for name in stage.STAGE_NAMES
    ]
    standard = cast("nn.ModuleDict", model.same_resolution_blocks)
    slots.extend((f"sr.{resolution}", block, "reader") for resolution, block in standard.items())
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    for resolution, blocks in extras.items():
        slots.extend(
            (f"extra.{resolution}.{index}", block, "reader")
            for index, block in enumerate(cast("nn.ModuleList", blocks))
        )
    return slots


def _install_widely_linear_readers(model: nn.Module) -> None:
    for _label, owner, attribute in _reader_slots(model):
        source = getattr(owner, attribute)
        if type(source) is not FactorizedComplexConv2dReader:
            raise TypeError("WL conversion requires an exact strict factorized Reader")
        setattr(
            owner,
            attribute,
            GatedWidelyLinearFactorizedComplexConv2dReader.from_strict(source),
        )


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported strict-versus-WL Reader variant: {variant}")
    model = stage._build_for_spec(SPEC, variant, config)
    if variant == WL:
        _install_widely_linear_readers(model)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: nn.Module, variant: str) -> None:
    if (
        SPEC.excitation_modes != K_SCHEDULE
        or SPEC.pole_modes != P_SCHEDULE
        or SPEC.extra_blocks != EXTRA_BLOCKS
        or SPEC.depth != DEPTH
        or stage.POST_HIDDEN != POST_FUSION_HIDDEN
        or stage.PATH_HIDDEN != PATH_HIDDEN
    ):
        raise RuntimeError("strict-versus-WL study changed a frozen common condition")
    slots = _reader_slots(model)
    if len(slots) != READER_COUNT:
        raise RuntimeError("strict-versus-WL study changed the Reader count")
    expected_type = (
        FactorizedComplexConv2dReader
        if variant == STRICT
        else GatedWidelyLinearFactorizedComplexConv2dReader
    )
    for label, owner, attribute in slots:
        reader = getattr(owner, attribute)
        if type(reader) is not expected_type or reader.rank != 2 or reader.kernel_size != 3:
            raise RuntimeError(f"{label} changed the frozen Reader operator contract")


def _variant_config(variant: str) -> dict[str, Any]:
    payload = stage._variant_config_for_spec(variant, SPEC)
    payload["experiment"].update(
        {
            "common_conditions": {
                "K_schedule": list(K_SCHEDULE),
                "P_schedule": list(P_SCHEDULE),
                "depth": list(DEPTH),
                "post_fusion_hidden": POST_FUSION_HIDDEN,
                "path_hidden": PATH_HIDDEN,
                "reader_rank": 2,
                "reader_kernel_size": 3,
                "head": "terminal Q4 affine",
            },
            "reader_operator": (
                "strict Wz" if variant == STRICT else "gated WL Wz + tanh(beta) V conjugate(z)"
            ),
            "reader_count": READER_COUNT,
            "varied_dimension": "Reader complex-linearity only",
            "wl_initialization": (
                None if variant == STRICT else "strict branch copied exactly; conjugate gate beta=0"
            ),
        }
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = runtime.base_contract(args)
    payload.get("runtime", {}).pop("hostname", None)
    payload.get("recipe", {}).pop("cpu_affinity", None)
    payload["schema"] = "lnet.a2d.r2k3.strict_vs_wl_reader.v1"
    payload["evidence_status"] = (
        "two-cell seed501 Reader operator comparison; H200 sequential queue"
    )
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in VARIANTS}
    payload["parameter_counts"] = dict(PARAMETER_COUNTS)
    payload["screen"] = {
        "comparison": "strict rank-2 K3 Reader versus gated WL rank-2 K3 Reader",
        "initial_function": "exactly equal because every conjugate gate starts at zero",
        "only_varied_dimension": "conjugate Reader path availability",
        "parameter_delta": PARAMETER_COUNTS[WL] - PARAMETER_COUNTS[STRICT],
        "parameter_increase_percent": 100.0
        * (PARAMETER_COUNTS[WL] - PARAMETER_COUNTS[STRICT])
        / PARAMETER_COUNTS[STRICT],
    }
    payload["source_sha256"]["stage_allocation_builder"] = runtime.digest(
        Path("scripts/run_a2d_r2k3_stage_allocation_screen_imagenet100.py")
    )
    payload["source_sha256"]["factorized_reader"] = runtime.digest(
        Path("src/lnet/pac_factorized_complex_scan_reader.py")
    )
    payload["source_sha256"]["strict_vs_wl_runner"] = runtime.digest(Path(__file__))
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
