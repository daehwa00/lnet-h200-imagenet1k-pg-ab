from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
from typing import TYPE_CHECKING, cast

import pytest
import torch

from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_stage_allocation_screen_imagenet100 as runner
from scripts import run_complex_scan_followups_imagenet100 as followups
from scripts import run_complex_scan_zero_init_imagenet100 as zero
from scripts import smoke_a2d_r2k3_same_resolution_factorial as smoke

if TYPE_CHECKING:
    from torch import nn

    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


EXPECTED_PARAMETERS = {
    runner.CONTROL: 2_693_668,
    runner.P14_ONLY: 2_816_676,
    runner.P28_14: 2_939_684,
    runner.P_FRONT3: 3_062_692,
    runner.P_FRONT3_TERM96: 3_036_260,
    runner.P14_TERM96: 2_790_244,
    runner.P_MEMORY_PYRAMID: 2_798_308,
    runner.P96_UNIFORM: 2_691_428,
    runner.K96_P128: 2_198_884,
    runner.DEPTH_2232: 3_021_860,
    runner.DEPTH_2223: 3_021_860,
    runner.DEPTH_2242: 3_350_052,
    runner.DEPTH_2262: 4_006_436,
}


def test_screen_covers_each_declared_diagnostic_once() -> None:
    assert len(runner.VARIANTS) == 13
    assert len(set(runner.VARIANTS)) == len(runner.VARIANTS)
    assert set(runner.ALLOCATION_VARIANTS).isdisjoint(runner.DEPTH_VARIANTS)
    assert set(runner.ALLOCATION_VARIANTS).union(runner.DEPTH_VARIANTS) == set(
        runner.VARIANTS
    )
    queued = tuple(variant for lane in runner.JOBS_BY_GPU.values() for variant in lane)
    assert tuple(runner.JOBS_BY_GPU) == (0,)
    assert queued == runner.VARIANTS


def test_transition_reference_is_enforced_only_for_deterministic_fp32() -> None:
    smoke._validate_transition_error("cuda", 1.0)
    with pytest.raises(RuntimeError, match="algebra changed"):
        smoke._validate_transition_error("cpu", 1.0)


def test_dataset_manifest_override_is_explicit_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(followups.MANIFEST_ENV, raising=False)
    assert followups._expected_imagenet100_manifest() == zero.IMAGENET100_MANIFEST_SHA256
    monkeypatch.setenv(followups.MANIFEST_ENV, "a" * 64)
    assert followups._expected_imagenet100_manifest() == "a" * 64
    monkeypatch.setenv(followups.MANIFEST_ENV, "not-a-digest")
    with pytest.raises(ValueError, match="SHA-256"):
        followups._expected_imagenet100_manifest()


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_variants_match_their_k_p_depth_and_parameter_contract(variant: str) -> None:
    spec = runner.SPECS[variant]
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())

    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS[
        variant
    ]
    assert model.descriptor_dim == spec.descriptor_dim
    assert model.classifier.input_dim == spec.q4_dim
    same_resolution_blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    for index, name in enumerate(runner.STAGE_NAMES):
        stage = getattr(model, name)
        assert stage.input_modes == spec.excitation_modes[index]
        assert stage.modes == spec.pole_modes[index]
        block = cast(
            "SameResolutionPoleScanBlock",
            same_resolution_blocks[str(runner.RESOLUTIONS[index])],
        )
        assert block.modes == spec.excitation_modes[index]
        assert block.pole_modes == spec.pole_modes[index]

    extras = cast(
        "nn.ModuleDict",
        getattr(model, "extra_same_resolution_blocks", torch.nn.ModuleDict()),
    )
    extra_counts = []
    for resolution in runner.RESOLUTIONS:
        key = str(resolution)
        count = len(cast("nn.ModuleList", extras[key])) if key in extras else 0
        extra_counts.append(count)
    actual_depth = tuple(2 + count for count in extra_counts)
    assert actual_depth == spec.depth


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_variant_configs_are_json_exact(variant: str) -> None:
    payload = runner._variant_config(variant)
    assert json.loads(json.dumps(payload)) == payload
    assert tuple(
        payload["experiment"]["depth_by_resolution"][str(resolution)]
        for resolution in runner.RESOLUTIONS
    ) == runner.SPECS[variant].depth


def test_repeated_sr14_blocks_are_distinct_and_execute_in_order() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.DEPTH_2242, runtime.model_config()).eval()
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    blocks = cast("nn.ModuleList", extras["14"])
    assert len(blocks) == 2
    assert blocks[0] is not blocks[1]
    assert all(
        left is not right
        for left, right in zip(blocks[0].parameters(), blocks[1].parameters(), strict=True)
    )

    calls: list[int] = []
    handles = [
        block.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.append(index)
        )
        for index, block in enumerate(blocks)
    ]
    try:
        with torch.no_grad():
            descriptor = model.raw_descriptor(torch.randn(1, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert descriptor.shape == (1, 2_048)
    assert calls == [0, 1]
