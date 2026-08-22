from __future__ import annotations

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
import json

import a2d_r2k3_runtime as runtime
import pytest
import run_a2d_r2k3_reader_wl_imagenet100 as runner
import torch

from lnet.pac_factorized_complex_scan_reader import (
    FactorizedComplexConv2dReader,
    GatedWidelyLinearFactorizedComplexConv2dReader,
)

EXPECTED_READER_SHAPES = {
    "main.stage1": (96, 128, True, False),
    "main.stage2": (128, 192, True, False),
    "main.stage3": (128, 192, True, False),
    "main.terminal": (128, 128, False, True),
    "sr.56": (96, 128, True, False),
    "sr.28": (128, 192, True, False),
    "sr.14": (128, 192, True, False),
    "sr.7": (128, 128, True, False),
    "extra.14.0": (128, 192, True, False),
    "extra.14.1": (128, 192, True, False),
    "extra.14.2": (128, 192, True, False),
    "extra.14.3": (128, 192, True, False),
}


def test_reader_study_freezes_exact_two_cell_contract() -> None:
    assert runner.VARIANTS == (runner.STRICT, runner.WL)
    assert runner.SEEDS == (501,)
    assert runner.K_SCHEDULE == (96, 128, 128, 128)
    assert runner.P_SCHEDULE == (128, 192, 192, 128)
    assert runner.DEPTH == (2, 2, 6, 2)
    assert runner.PARAMETER_COUNTS == {
        runner.STRICT: 4_578_244,
        runner.WL: 5_669_828,
    }


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_every_reader_uses_the_registered_operator_and_shape(variant: str) -> None:
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())
    slots = runner._reader_slots(model)

    assert {label for label, _owner, _attribute in slots} == set(EXPECTED_READER_SHAPES)
    expected_type = (
        FactorizedComplexConv2dReader
        if variant == runner.STRICT
        else GatedWidelyLinearFactorizedComplexConv2dReader
    )
    for label, owner, attribute in slots:
        reader = getattr(owner, attribute)
        expected_input, expected_output, normalized, matched = EXPECTED_READER_SHAPES[label]
        assert type(reader) is expected_type
        assert (reader.input_modes, reader.output_modes) == (expected_input, expected_output)
        assert (reader.rank, reader.kernel_size) == (2, 3)
        strict = reader if variant == runner.STRICT else reader.strict_reader
        assert (strict.input_norm is not None) is normalized
        assert strict.match_input_rms is matched
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == runner.PARAMETER_COUNTS[variant]
    )


def test_wl_cell_preserves_every_strict_tensor_and_initial_function() -> None:
    torch.manual_seed(501)
    strict_model = runner._build(runner.STRICT, runtime.model_config()).eval()
    torch.manual_seed(501)
    wl_model = runner._build(runner.WL, runtime.model_config()).eval()

    strict_slots = {
        label: getattr(owner, attribute)
        for label, owner, attribute in runner._reader_slots(strict_model)
    }
    wl_slots = {
        label: getattr(owner, attribute)
        for label, owner, attribute in runner._reader_slots(wl_model)
    }
    for label, strict in strict_slots.items():
        nested = wl_slots[label].strict_reader
        assert strict.state_dict().keys() == nested.state_dict().keys()
        assert all(
            torch.equal(value, nested.state_dict()[name])
            for name, value in strict.state_dict().items()
        )
        assert torch.count_nonzero(wl_slots[label].conjugate_gate) == 0

    inputs = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        strict_descriptor = strict_model.raw_descriptor(inputs)
        wl_descriptor = wl_model.raw_descriptor(inputs)
    assert torch.equal(strict_descriptor, wl_descriptor)


def test_optimizer_covers_wl_parameters_once_with_matched_reader_policy() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.WL, runtime.model_config())
    optimizer = runtime.build_optimizer(
        model,
        {
            "fused_optimizer": False,
            "learning_rate": 3.0e-3,
            "modal_learning_rate_multiplier": 1.0 / 3.0,
            "pole_geometry_learning_rate_multiplier": 1.0,
            "weight_decay": 0.05,
        },
    )
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters()
    }
    by_id = {
        id(parameter): group for group in optimizer.param_groups for parameter in group["params"]
    }
    for module in model.modules():
        if not isinstance(module, GatedWidelyLinearFactorizedComplexConv2dReader):
            continue
        for parameter in (
            module.conjugate_point_weight_real,
            module.conjugate_point_weight_imag,
            module.conjugate_spatial_weight_real,
            module.conjugate_spatial_weight_imag,
        ):
            assert by_id[id(parameter)]["lr"] == 3.0e-3
            assert by_id[id(parameter)]["weight_decay"] == 0.05
        assert by_id[id(module.conjugate_gate)]["lr"] == 3.0e-3
        assert by_id[id(module.conjugate_gate)]["weight_decay"] == 0.0


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_reader_variant_config_is_json_exact(variant: str) -> None:
    payload = runner._variant_config(variant)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["experiment"]["varied_dimension"] == "Reader complex-linearity only"
    assert payload["experiment"]["common_conditions"]["K_schedule"] == [96, 128, 128, 128]
    assert payload["experiment"]["common_conditions"]["P_schedule"] == [128, 192, 192, 128]
