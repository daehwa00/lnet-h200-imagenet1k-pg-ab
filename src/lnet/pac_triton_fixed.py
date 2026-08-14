from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _fixed_forward_kernel(
    decay_real,
    decay_imag,
    input_real,
    input_imag,
    states_real,
    states_imag,
    n_steps: int,
    modes: int,
) -> None:
    program = tl.program_id(0)
    batch = program // modes
    mode = program - batch * modes
    base = batch * n_steps * modes + mode
    ar = tl.load(decay_real + mode)
    ai = tl.load(decay_imag + mode)
    state_real = tl.full((), 0.0, tl.float32)
    state_imag = tl.full((), 0.0, tl.float32)
    time_index = 0
    while time_index < n_steps:
        offset = base + time_index * modes
        previous_real = state_real
        previous_imag = state_imag
        state_real = ar * previous_real - ai * previous_imag + tl.load(input_real + offset)
        state_imag = ai * previous_real + ar * previous_imag + tl.load(input_imag + offset)
        tl.store(states_real + offset, state_real)
        tl.store(states_imag + offset, state_imag)
        time_index += 1


@triton.jit
def _fixed_backward_kernel(
    decay_real,
    decay_imag,
    states_real,
    states_imag,
    grad_states_real,
    grad_states_imag,
    grad_decay_real,
    grad_decay_imag,
    grad_input_real,
    grad_input_imag,
    n_steps: int,
    modes: int,
) -> None:
    program = tl.program_id(0)
    batch = program // modes
    mode = program - batch * modes
    base = batch * n_steps * modes + mode
    ar = tl.load(decay_real + mode)
    ai = tl.load(decay_imag + mode)
    lambda_real = tl.full((), 0.0, tl.float32)
    lambda_imag = tl.full((), 0.0, tl.float32)
    decay_real_sum = tl.full((), 0.0, tl.float32)
    decay_imag_sum = tl.full((), 0.0, tl.float32)
    remaining = n_steps
    while remaining > 0:
        time_index = remaining - 1
        offset = base + time_index * modes
        lambda_real += tl.load(grad_states_real + offset)
        lambda_imag += tl.load(grad_states_imag + offset)
        previous_offset = base + (time_index - 1) * modes
        has_previous = time_index > 0
        previous_real = tl.load(states_real + previous_offset, mask=has_previous, other=0.0)
        previous_imag = tl.load(states_imag + previous_offset, mask=has_previous, other=0.0)
        tl.store(grad_input_real + offset, lambda_real)
        tl.store(grad_input_imag + offset, lambda_imag)
        decay_real_sum += lambda_real * previous_real + lambda_imag * previous_imag
        decay_imag_sum += -lambda_real * previous_imag + lambda_imag * previous_real
        next_lambda_real = ar * lambda_real + ai * lambda_imag
        next_lambda_imag = -ai * lambda_real + ar * lambda_imag
        lambda_real = next_lambda_real
        lambda_imag = next_lambda_imag
        remaining -= 1
    tl.atomic_add(grad_decay_real + mode, decay_real_sum)
    tl.atomic_add(grad_decay_imag + mode, decay_imag_sum)


class _FixedTritonRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _AutogradContext,
        decay_real: Tensor,
        decay_imag: Tensor,
        input_real: Tensor,
        input_imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        real = decay_real.contiguous()
        imag = decay_imag.contiguous()
        drive_real = input_real.contiguous()
        drive_imag = input_imag.contiguous()
        states_real = torch.empty_like(drive_real)
        states_imag = torch.empty_like(drive_imag)
        batch, n_steps, modes = drive_real.shape
        _fixed_forward_kernel[(batch * modes,)](
            real, imag, drive_real, drive_imag, states_real, states_imag, n_steps, modes
        )
        ctx.save_for_backward(real, imag, states_real, states_imag)
        return states_real, states_imag

    @staticmethod
    def backward(
        ctx: _AutogradContext,
        *grad_outputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        grad_states_real, grad_states_imag = grad_outputs
        decay_real, decay_imag, states_real, states_imag = ctx.saved_tensors
        grad_decay_real = torch.zeros_like(decay_real)
        grad_decay_imag = torch.zeros_like(decay_imag)
        grad_input_real = torch.empty_like(states_real)
        grad_input_imag = torch.empty_like(states_imag)
        batch, n_steps, modes = states_real.shape
        _fixed_backward_kernel[(batch * modes,)](
            decay_real,
            decay_imag,
            states_real,
            states_imag,
            grad_states_real.contiguous(),
            grad_states_imag.contiguous(),
            grad_decay_real,
            grad_decay_imag,
            grad_input_real,
            grad_input_imag,
            n_steps,
            modes,
        )
        return grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag


def triton_fixed_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    outputs = _FixedTritonRecurrence.apply(decay_real, decay_imag, input_real, input_imag)
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        message = "Fixed Triton recurrence returned an invalid output"
        raise RuntimeError(message)
    states_real, states_imag = outputs
    if not isinstance(states_real, Tensor) or not isinstance(states_imag, Tensor):
        message = "Fixed Triton recurrence outputs must be tensors"
        raise TypeError(message)
    return states_real, states_imag
