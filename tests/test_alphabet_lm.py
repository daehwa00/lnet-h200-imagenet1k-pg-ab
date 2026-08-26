from __future__ import annotations

from pathlib import Path

import torch

from lnet.alphabet_lm import AlphabetLM, AlphabetLMConfig
from lnet.pac_triton_recurrence_op import pac_triton_recurrence_opaque_op


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


def test_h200_mamba_runtime_dependency_contract_is_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    requirements = (
        root / "h200/alphabet_lm_preflight/requirements.txt"
    ).read_text(encoding="utf-8")
    lock = (root / "h200/alphabet_lm_preflight/requirements.lock").read_text(
        encoding="utf-8"
    )
    for requirement in (
        "einops==0.8.1",
        "ninja==1.13.0",
        "packaging==26.3",
        "setuptools==84.0.0",
        "torch==2.9.1+cu130",
        "transformers==4.57.1",
        "triton==3.5.1",
        "wandb==0.22.3",
        "wheel==0.46.2",
    ):
        assert requirement in requirements
        assert requirement in lock
    for transitive in (
        "huggingface-hub==0.36.2",
        "safetensors==0.8.0",
        "tokenizers==0.22.2",
    ):
        assert transitive in lock
    assert "transformers==5." not in requirements
    assert "transformers==5." not in lock


def test_opaque_recurrence_compiles_time_major_noncontiguous_input() -> None:
    batch, steps, modes = 2, 7, 5

    def inputs() -> tuple[torch.Tensor, torch.Tensor]:
        real = torch.randn(batch, modes, steps).transpose(1, 2).detach().requires_grad_()
        imag = torch.randn(batch, modes, steps).transpose(1, 2).detach().requires_grad_()
        assert real.stride() == (steps * modes, 1, steps)
        return real, imag

    def target(input_real: torch.Tensor, input_imag: torch.Tensor) -> torch.Tensor:
        shape = (1, 1, modes)
        decay_real = torch.full(shape, 0.8).expand_as(input_real)
        decay_imag = torch.full(shape, 0.1).expand_as(input_imag)
        states_real, states_imag = pac_triton_recurrence_opaque_op(
            decay_real, decay_imag, input_real, input_imag
        )
        assert states_real.is_contiguous()
        assert states_imag.is_contiguous()
        return states_real.square().mean() + states_imag.square().mean()

    expected_inputs = inputs()
    expected_loss = target(*expected_inputs)
    expected_grads = torch.autograd.grad(expected_loss, expected_inputs)
    actual_inputs = tuple(
        tensor.detach().clone(memory_format=torch.preserve_format).requires_grad_()
        for tensor in expected_inputs
    )
    compiled_loss = torch.compile(target, fullgraph=True, dynamic=False)(*actual_inputs)
    actual_grads = torch.autograd.grad(compiled_loss, actual_inputs)
    torch.testing.assert_close(compiled_loss, expected_loss)
    torch.testing.assert_close(actual_grads, expected_grads)

    real, imag = inputs()
    shape = (1, 1, modes)
    opcheck = torch.library.opcheck(
        torch.ops.lnet.pac_real2d_recurrence_opaque.default,
        (
            torch.full(shape, 0.8).expand_as(real),
            torch.full(shape, 0.1).expand_as(imag),
            real,
            imag,
            False,
        ),
        raise_exception=True,
    )
    assert all(value == "SUCCESS" for value in opcheck.values())
