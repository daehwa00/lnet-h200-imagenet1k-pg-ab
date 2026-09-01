from __future__ import annotations

import torch

from lnet.alphabet_lm_tensor_probe import (
    TensorAxisProbe,
    axis_spectrum_metrics,
    normalized_states,
)


def test_axis_probes_expose_the_declared_tensor_maps() -> None:
    states = torch.randn(7, 3, 5)
    for mode in ("temporal", "content", "full"):
        probe = TensorAxisProbe(3, 5, 4, mode=mode)
        output = probe(states)
        weight = probe.weight_tensor()
        reconstructed = torch.einsum("ntc,tco->no", states, weight) + probe.output.bias
        torch.testing.assert_close(output, reconstructed)


def test_low_rank_full_probe_reports_its_composed_weight() -> None:
    states = torch.randn(9, 4, 6)
    probe = TensorAxisProbe(4, 6, 7, mode="full", probe_rank=3)
    output = probe(states)
    reconstructed = (
        torch.einsum("ntc,tco->no", states, probe.weight_tensor())
        + probe.output.bias
    )
    torch.testing.assert_close(output, reconstructed)


def test_axis_spectrum_recognizes_a_rank_one_temporal_factor() -> None:
    temporal = torch.randn(6)
    content_output = torch.randn(5, 4)
    weight = temporal[:, None, None] * content_output[None, :, :]
    temporal_metrics = axis_spectrum_metrics(weight, axis=0)
    content_metrics = axis_spectrum_metrics(weight, axis=1)
    assert temporal_metrics["top1_energy"] > 0.9999
    assert temporal_metrics["stable_rank"] < 1.0001
    assert content_metrics["stable_rank"] > 1.0


def test_state_normalization_preserves_shape_and_sets_unit_rms() -> None:
    states = torch.randn(11, 4, 8) * 3.0
    normalized, scale = normalized_states(states)
    assert normalized.shape == states.shape
    assert scale > 0.0
    torch.testing.assert_close(normalized.square().mean(), torch.ones(()))
