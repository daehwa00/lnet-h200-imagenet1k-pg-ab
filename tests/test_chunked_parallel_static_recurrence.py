from __future__ import annotations

import torch

from lnet.pac_triton_parallel_static_recurrence import (
    chunked_parallel_static_recurrence_packed,
)


def _serial(decay_real: torch.Tensor, decay_imag: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    modes = decay_real.numel()
    input_real, input_imag = packed.split(modes, dim=-1)
    state_real = torch.zeros_like(input_real[:, 0])
    state_imag = torch.zeros_like(input_imag[:, 0])
    output_real = []
    output_imag = []
    for step in range(packed.shape[1]):
        previous_real, previous_imag = state_real, state_imag
        state_real = decay_real * previous_real - decay_imag * previous_imag + input_real[:, step]
        state_imag = decay_imag * previous_real + decay_real * previous_imag + input_imag[:, step]
        output_real.append(state_real)
        output_imag.append(state_imag)
    return torch.cat((torch.stack(output_real, dim=1), torch.stack(output_imag, dim=1)), dim=-1)


def test_chunked_parallel_recurrence_matches_serial_across_boundaries() -> None:
    torch.manual_seed(7)
    decay_real = torch.tensor([0.7, 0.8, 0.9], requires_grad=True)
    decay_imag = torch.tensor([0.1, -0.05, 0.02], requires_grad=True)
    packed = torch.randn(2, 19, 6, requires_grad=True)
    parallel = chunked_parallel_static_recurrence_packed(
        decay_real, decay_imag, packed, chunk_size=5
    )
    reference = _serial(decay_real, decay_imag, packed)
    torch.testing.assert_close(parallel, reference, rtol=2.0e-5, atol=2.0e-6)

    gradient = torch.randn_like(parallel)
    parallel_gradients = torch.autograd.grad(
        parallel, (decay_real, decay_imag, packed), gradient, retain_graph=True
    )
    reference_gradients = torch.autograd.grad(
        reference, (decay_real, decay_imag, packed), gradient
    )
    for actual, expected in zip(parallel_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=5.0e-5, atol=5.0e-6)
