from __future__ import annotations

from typing import Any, cast

import torch

from lnet.alphabet_lm_mamba import (
    LaplaceSSDMamba2Config,
    LaplaceSSDMamba2LM,
    LaplaceSSDMamba2Mixer,
)


def test_laplace_ssd_replaces_only_official_mamba2_mixers() -> None:
    config = LaplaceSSDMamba2Config(
        vocab_size=128,
        model_width=64,
        layers=2,
        pole_modes=4,
        expand=2,
        head_dim=32,
        context_length=64,
        minimum_half_life=2.0,
        maximum_half_life=8.0,
    )
    model = LaplaceSSDMamba2LM(config)
    backbone = cast("Any", model.model).backbone
    assert len(backbone.layers) == 2
    assert all(isinstance(block.mixer, LaplaceSSDMamba2Mixer) for block in backbone.layers)
    assert model.model.lm_head.weight is backbone.embedding.weight


def test_laplace_ssd_state_and_route_shapes_match_mamba2_heads() -> None:
    config = LaplaceSSDMamba2Config(
        vocab_size=128,
        model_width=64,
        layers=2,
        pole_modes=4,
        expand=2,
        head_dim=32,
        groups=1,
        context_length=64,
        minimum_half_life=2.0,
        maximum_half_life=8.0,
    )
    model = LaplaceSSDMamba2LM(config)
    mixer = cast("Any", model.model).backbone.layers[0].mixer
    assert mixer.nheads == 4
    assert mixer.headdim == 32
    assert mixer.memory.modes == 16
    route = torch.randn(2, 7, 1, 4)
    assert mixer._group_routes(route).shape == (2, 7, 4, 4)
    damping = mixer.memory.damping().reshape(4, 4)
    torch.testing.assert_close(damping[0], damping[-1])


def test_default_laplace_ssd_parameter_and_state_contract() -> None:
    model = LaplaceSSDMamba2LM(LaplaceSSDMamba2Config())
    mixer = cast("Any", model.model).backbone.layers[0].mixer
    assert sum(parameter.numel() for parameter in model.parameters()) == 45_437_328
    assert mixer.nheads * mixer.poles * mixer.headdim == 8_192
    assert mixer.memory.parallel_static_scan
