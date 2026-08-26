from __future__ import annotations

import torch

from lnet.alphabet_lm import AlphabetLM, AlphabetLMConfig


def _small() -> AlphabetLMConfig:
    return AlphabetLMConfig(
        vocab_size=64,
        modes=8,
        pole_modes=12,
        layers=2,
        post_hidden=12,
        context_length=16,
    )


def test_alphabet_lm_default_parameter_count() -> None:
    torch.manual_seed(501)
    model = AlphabetLM(AlphabetLMConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 34_794_496


def test_alphabet_lm_is_causal_and_has_finite_gradients() -> None:
    torch.manual_seed(501)
    model = AlphabetLM(_small())
    tokens = torch.randint(64, (1, 17))
    changed = tokens.clone()
    changed[:, 10:] = torch.randint(64, changed[:, 10:].shape)
    with torch.no_grad():
        expected = model(tokens[:, :-1])
        actual = model(changed[:, :-1])
    assert torch.allclose(expected[:, :10], actual[:, :10], atol=1.0e-6, rtol=0.0)
    logits = model(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
