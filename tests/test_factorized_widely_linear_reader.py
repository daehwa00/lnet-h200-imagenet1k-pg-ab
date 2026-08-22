from __future__ import annotations

from unittest import mock

import torch
from torch.nn import functional

from lnet import pac_factorized_complex_scan_reader as reader_module
from lnet.pac_factorized_complex_scan_reader import (
    FactorizedComplexConv2dReader,
    GatedWidelyLinearFactorizedComplexConv2dReader,
)


def _strict_reader() -> FactorizedComplexConv2dReader:
    torch.manual_seed(17)
    reader = FactorizedComplexConv2dReader(
        4,
        4,
        rank=2,
        kernel_size=3,
        normalize_input=True,
        match_input_rms=True,
    )
    reader.initialize_orthogonal_()
    return reader


def _field() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(29)
    return (
        torch.randn(2, 7, 6, 4, generator=generator),
        torch.randn(2, 7, 6, 4, generator=generator),
    )


def test_from_strict_is_an_exact_zero_gate_function_and_uses_one_convolution() -> None:
    strict = _strict_reader()
    real, imag = _field()
    expected = strict(real, imag)
    rng_before = torch.random.get_rng_state().clone()

    reader = GatedWidelyLinearFactorizedComplexConv2dReader.from_strict(strict)

    assert reader.strict_reader is strict
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.count_nonzero(reader.conjugate_gate) == 0
    assert torch.equal(reader.conjugate_point_weight_real, strict.point_weight_real)
    assert torch.equal(reader.conjugate_point_weight_imag, strict.point_weight_imag)
    assert torch.equal(reader.conjugate_spatial_weight_real, strict.spatial_weight_real)
    assert torch.equal(reader.conjugate_spatial_weight_imag, strict.spatial_weight_imag)
    with mock.patch.object(
        reader_module.functional,
        "conv2d",
        wraps=functional.conv2d,
    ) as packed_conv:
        actual = reader(real, imag)
    assert packed_conv.call_count == 1
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_zero_gate_learns_then_exposes_conjugate_factor_gradients() -> None:
    reader = GatedWidelyLinearFactorizedComplexConv2dReader.from_strict(_strict_reader())
    real, imag = _field()
    target_real = torch.linspace(-1.0, 1.0, real.numel()).reshape_as(real)
    target_imag = torch.linspace(1.0, -0.5, imag.numel()).reshape_as(imag)

    output_real, output_imag = reader(real, imag)
    loss = (output_real * target_real + output_imag * target_imag).sum()
    loss.backward()

    assert reader.conjugate_gate.grad is not None
    assert torch.count_nonzero(reader.conjugate_gate.grad) > 0
    reader.zero_grad(set_to_none=True)
    with torch.no_grad():
        reader.conjugate_gate.fill_(0.25)

    output_real, output_imag = reader(real, imag)
    (output_real.square().mean() + 0.7 * output_imag.square().mean()).backward()
    conjugate_gradients = (
        reader.conjugate_point_weight_real.grad,
        reader.conjugate_point_weight_imag.grad,
        reader.conjugate_spatial_weight_real.grad,
        reader.conjugate_spatial_weight_imag.grad,
    )
    assert all(gradient is not None for gradient in conjugate_gradients)
    gradient_total = sum(
        float(gradient.abs().sum()) for gradient in conjugate_gradients if gradient is not None
    )
    assert gradient_total > 0


def test_active_conjugate_branch_breaks_global_phase_equivariance() -> None:
    reader = GatedWidelyLinearFactorizedComplexConv2dReader.from_strict(_strict_reader())
    with torch.no_grad():
        reader.conjugate_gate.fill_(0.4)
    real, imag = _field()
    angle = torch.tensor(0.73)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotated_real = cosine * real - sine * imag
    rotated_imag = sine * real + cosine * imag

    output_real, output_imag = reader(real, imag)
    actual_real, actual_imag = reader(rotated_real, rotated_imag)
    equivariant_real = cosine * output_real - sine * output_imag
    equivariant_imag = sine * output_real + cosine * output_imag

    error = (actual_real - equivariant_real).abs().max() + (
        actual_imag - equivariant_imag
    ).abs().max()
    assert float(error.detach()) > 1.0e-3


def test_joint_normalization_and_state_round_trip_are_deterministic() -> None:
    reader = GatedWidelyLinearFactorizedComplexConv2dReader.from_strict(_strict_reader())
    with torch.no_grad():
        reader.conjugate_gate.copy_(torch.linspace(-0.4, 0.4, reader.output_modes))
    (weight_real, weight_imag), (conjugate_real, conjugate_imag) = (
        reader.joint_unit_energy_kernels()
    )
    row_energy = (
        weight_real.square()
        .add(weight_imag.square())
        .add(conjugate_real.square())
        .add(conjugate_imag.square())
        .sum(dim=(1, 2, 3))
    )
    torch.testing.assert_close(row_energy, torch.ones_like(row_energy))

    restored = GatedWidelyLinearFactorizedComplexConv2dReader.from_strict(_strict_reader())
    restored.load_state_dict(reader.state_dict())
    assert tuple(reader.state_dict()) == tuple(restored.state_dict())
    for name, value in reader.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    real, imag = _field()
    expected = reader(real, imag)
    actual = restored(real, imag)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
