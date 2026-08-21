from __future__ import annotations

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
import json
from typing import TYPE_CHECKING, cast

import a2d_r2k3_capacity_factory as capacity
import a2d_r2k3_runtime as runtime
import pytest
import run_a2d_r2k3_d2262_p_schedule_imagenet100 as runner
import torch

from lnet.pac_phase_gated_transition import PathOnlyCollapse

if TYPE_CHECKING:
    from torch import nn

    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


EXPECTED_PARAMETERS = {
    runner.A: 4_621_476,
    runner.B: 4_793_892,
    runner.C: 4_678_948,
    runner.D: 4_851_364,
    runner.E: 4_728_356,
    runner.F: 4_670_884,
}


def test_screen_declares_exact_six_p_schedule_questions() -> None:
    assert runner.VARIANTS == (runner.A, runner.B, runner.C, runner.D, runner.E, runner.F)
    assert len(set(runner.VARIANTS)) == 6
    assert set(runner.SPECS) == set(runner.VARIANTS)
    assert set(runner.P_SCHEDULES) == set(runner.VARIANTS)
    assert set(runner.QUESTIONS) == set(runner.VARIANTS)
    assert runner.JOBS_BY_GPU == {0: runner.VARIANTS}
    assert runner.SEEDS == (501,)
    assert runner.PARAMETER_COUNTS == EXPECTED_PARAMETERS


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_variants_freeze_k_d2262_pf192_pathh4_and_q4(variant: str) -> None:
    spec = runner.SPECS[variant]
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())

    assert spec.excitation_modes == (128, 128, 128, 128)
    assert spec.pole_modes == runner.P_SCHEDULES[variant]
    assert spec.depth == (2, 2, 6, 2)
    assert spec.extra_blocks == (0, 0, 4, 0)
    assert (
        sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS[variant]
    )
    assert isinstance(model.classifier, capacity.CapacityQ4OnlyAffineClassifier)
    assert model.descriptor_dim == 4 * sum(spec.pole_modes)
    assert model.classifier.input_dim == 4 * spec.pole_modes[-1] == 512

    for index, name in enumerate(runner.stage.STAGE_NAMES):
        active = getattr(model, name)
        reader = active.pole_input_projection
        assert reader.input_modes == 128
        assert reader.output_modes == spec.pole_modes[index]
        assert reader.rank == 2
        assert reader.kernel_size == 3
        if name != "terminal":
            assert active.augmented.post_fusion.hidden_modes == 192
            collapse = active.quadrant_path_mode_combiner
            assert type(collapse) is PathOnlyCollapse
            assert collapse.path_input.output_paths == 4

    standard = cast("nn.ModuleDict", model.same_resolution_blocks)
    assert tuple(standard) == ("56", "28", "14", "7")
    for index, resolution in enumerate((56, 28, 14, 7)):
        block = cast("SameResolutionPoleScanBlock", standard[str(resolution)])
        assert block.modes == 128
        assert block.pole_modes == spec.pole_modes[index]
        assert block.post_fusion.hidden_modes == 192
        assert type(block.path_collapse) is PathOnlyCollapse
        assert block.path_collapse.path_input.output_paths == 4

    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    assert tuple(extras) == ("14",)
    blocks = cast("nn.ModuleList", extras["14"])
    assert len(blocks) == 4
    assert len({id(block) for block in blocks}) == 4
    assert all(block.pole_modes == spec.pole_modes[2] for block in blocks)
    assert all(block.post_fusion.hidden_modes == 192 for block in blocks)
    assert all(block.path_collapse.path_input.output_paths == 4 for block in blocks)


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_variant_configs_are_json_exact_and_only_vary_p(variant: str) -> None:
    payload = runner._variant_config(variant)

    assert json.loads(json.dumps(payload)) == payload
    assert payload["backbone"]["excitation_schedule"] == [128, 128, 128, 128]
    assert payload["backbone"]["pole_schedule"] == list(runner.P_SCHEDULES[variant])
    assert payload["experiment"]["depth_by_resolution"] == {
        "56": 2,
        "28": 2,
        "14": 6,
        "7": 2,
    }
    assert payload["experiment"]["varied_dimension"] == "pole_schedule_only"


def test_four_extra_sr14_blocks_execute_in_order() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.D, runtime.model_config()).eval()
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    blocks = cast("nn.ModuleList", extras["14"])
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

    assert descriptor.shape == (1, 2_688)
    assert calls == [0, 1, 2, 3]
