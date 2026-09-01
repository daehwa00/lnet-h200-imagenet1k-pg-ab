from __future__ import annotations

from typing import cast

import torch

from lnet.alphabet_lm import (
    HybridContentDenseImagePostFusionAlphabet2Block,
    HybridContentDenseImagePostFusionAlphabet2LM,
    LaplaceMambaLMConfig,
)


def _config() -> LaplaceMambaLMConfig:
    return LaplaceMambaLMConfig(
        vocab_size=64,
        model_width=32,
        layers=2,
        pole_modes=8,
        conv_width=3,
        context_length=16,
        minimum_half_life=2.0,
        maximum_half_life=8.0,
        activation_checkpoint=False,
        content_preserving_heads=2,
        content_preserving_poles_per_head=2,
        content_preserving_width_per_head=8,
        hybrid_dense_poles=4,
        hybrid_dense_width=2,
    )


def test_hybrid_has_declared_branch_states_and_independent_scales() -> None:
    model = HybridContentDenseImagePostFusionAlphabet2LM(_config())
    block = cast("HybridContentDenseImagePostFusionAlphabet2Block", model.blocks[0])
    assert block.content_state_modes == 32
    assert block.dense_state_modes == 8
    assert block.content_scale is not block.dense_scale
    damping = block.content_memory.damping().reshape(16, 2)
    torch.testing.assert_close(damping[0], damping[-1])


def test_hybrid_forward_backward_reaches_both_memory_branches() -> None:
    model = HybridContentDenseImagePostFusionAlphabet2LM(_config())
    block = cast("HybridContentDenseImagePostFusionAlphabet2Block", model.blocks[0])
    tokens = torch.randint(0, 64, (2, 9))
    logits = model(tokens)
    assert logits.shape == (2, 9, 64)
    logits.square().mean().backward()
    assert block.content_reader.weight_real.grad is not None
    assert block.content_write.weight.grad is not None
    assert block.content_memory.raw_damping.grad is not None
    assert block.dense_reader.weight_real.grad is not None
    assert block.dense_memory.raw_damping.grad is not None
    assert block.content_scale.grad is not None
    assert block.dense_scale.grad is not None


def test_default_hybrid_matches_parameter_and_state_contract() -> None:
    config = LaplaceMambaLMConfig(
        conv_width=3,
        content_preserving_poles_per_head=4,
        hybrid_dense_poles=32,
        hybrid_dense_width=8,
    )
    model = HybridContentDenseImagePostFusionAlphabet2LM(config)
    block = cast("HybridContentDenseImagePostFusionAlphabet2Block", model.blocks[0])
    assert sum(parameter.numel() for parameter in model.parameters()) == 56_989_718
    assert block.content_state_modes == 1_024
    assert block.dense_state_modes == 256
