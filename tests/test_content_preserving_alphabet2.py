from __future__ import annotations

from typing import cast

import torch
from torch import nn

from lnet.alphabet_lm import (
    ContentPreservingImagePostFusionAlphabet2Block,
    ContentPreservingImagePostFusionAlphabet2LM,
    LaplaceMambaLMConfig,
    VectorImagePostFusionAlphabet2LM,
)


def _small_config() -> LaplaceMambaLMConfig:
    return LaplaceMambaLMConfig(
        vocab_size=64,
        model_width=32,
        layers=2,
        pole_modes=8,
        head_width=3,
        conv_width=3,
        context_length=16,
        minimum_half_life=2.0,
        maximum_half_life=8.0,
        activation_checkpoint=False,
        content_preserving_heads=2,
        content_preserving_poles_per_head=4,
        content_preserving_width_per_head=8,
    )


def test_content_preserving_block_keeps_state_budget_and_group_axes() -> None:
    model = ContentPreservingImagePostFusionAlphabet2LM(_small_config())
    block = cast("ContentPreservingImagePostFusionAlphabet2Block", model.blocks[0])
    real = torch.randn(2, 9, 16)
    imag = torch.randn_like(real)
    content, write, read = block._analyze(  # pyright: ignore[reportPrivateUsage]
        real, imag
    )
    assert content[0].shape == (2, 9, 2, 8)
    assert write.shape == (2, 9, 2, 4)
    assert read[0].shape == (2, 9, 2, 4)

    captured: list[torch.Tensor] = []
    def capture_state(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        captured.append(output[0])

    handle = block.memory.register_forward_hook(capture_state)
    output = block(real, imag)
    handle.remove()
    assert output[0].shape == real.shape
    assert captured[0].shape == (2, 9, 64)


def test_content_memory_reads_history_before_the_current_write() -> None:
    model = ContentPreservingImagePostFusionAlphabet2LM(_small_config())
    block = cast("ContentPreservingImagePostFusionAlphabet2Block", model.blocks[0])
    real = torch.randn(2, 9, 16)
    imag = torch.randn_like(real)
    content, write, read = block._analyze(  # pyright: ignore[reportPrivateUsage]
        real, imag
    )
    selected = block._transport_and_read(  # pyright: ignore[reportPrivateUsage]
        content, write, read
    )
    torch.testing.assert_close(selected[0][:, 0], torch.zeros_like(selected[0][:, 0]))
    torch.testing.assert_close(selected[1][:, 0], torch.zeros_like(selected[1][:, 0]))


def test_history_residual_has_one_interpretable_layer_scale() -> None:
    model = ContentPreservingImagePostFusionAlphabet2LM(_small_config())
    block = cast("ContentPreservingImagePostFusionAlphabet2Block", model.blocks[0])
    torch.testing.assert_close(block.memory_scale, torch.tensor(0.01))


def test_every_head_starts_with_the_complete_pole_palette() -> None:
    model = ContentPreservingImagePostFusionAlphabet2LM(_small_config())
    block = cast("ContentPreservingImagePostFusionAlphabet2Block", model.blocks[0])
    damping = block.memory.damping().reshape(16, 4)
    frequency = block.memory.frequency().reshape(16, 4)
    torch.testing.assert_close(damping[0], damping[-1])
    torch.testing.assert_close(frequency[0], frequency[-1])


def test_content_preserving_lm_has_finite_forward_and_gradients() -> None:
    model = ContentPreservingImagePostFusionAlphabet2LM(_small_config())
    block = cast("ContentPreservingImagePostFusionAlphabet2Block", model.blocks[0])
    tokens = torch.randint(0, 64, (2, 9))
    logits = model(tokens)
    assert logits.shape == (2, 9, 64)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert block.feature_reader.weight_real.grad is not None
    assert block.write_router.weight.grad is not None
    assert block.read_router.weight_real.grad is not None
    assert block.memory.raw_damping.grad is not None


def test_projection_free_candidate_has_declared_capacity() -> None:
    config = LaplaceMambaLMConfig(conv_width=3)
    baseline = VectorImagePostFusionAlphabet2LM(config)
    candidate = ContentPreservingImagePostFusionAlphabet2LM(config)
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    assert baseline_parameters == 64_105_427
    assert candidate_parameters == 39_901_555
    assert candidate_parameters < 0.82 * 48_987_136
    assert sum(name.endswith("memory_scale") for name, _ in candidate.named_parameters()) == 19
