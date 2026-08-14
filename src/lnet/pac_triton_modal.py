from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor

from .pac_triton_recurrence import triton_recurrence_backward_from_states


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _modal_forward_kernel(
    decay_real,
    decay_imag,
    input_real,
    input_imag,
    writer_real,
    writer_imag,
    states_real,
    states_imag,
    modal,
    n_steps: int,
    modes: int,
    model_dim: int,
) -> None:
    lane = tl.program_id(0)
    batch = lane // modes
    mode = lane - batch * modes
    base = batch * n_steps * modes + mode
    state_real = tl.full((), 0.0, tl.float32)
    state_imag = tl.full((), 0.0, tl.float32)
    time_index = 0
    while time_index < n_steps:
        offset = base + time_index * modes
        ar = tl.load(decay_real + offset)
        ai = tl.load(decay_imag + offset)
        ur = tl.load(input_real + offset)
        ui = tl.load(input_imag + offset)
        previous_real = state_real
        previous_imag = state_imag
        state_real = ar * previous_real - ai * previous_imag + ur
        state_imag = ai * previous_real + ar * previous_imag + ui
        tl.store(states_real + offset, state_real)
        tl.store(states_imag + offset, state_imag)
        dim = 0
        modal_base = (batch * n_steps + time_index) * model_dim
        writer_base = mode * model_dim
        while dim < model_dim:
            wr = tl.load(writer_real + writer_base + dim)
            wi = tl.load(writer_imag + writer_base + dim)
            contribution = 2.0 * (state_real * wr - state_imag * wi)
            tl.atomic_add(modal + modal_base + dim, contribution, sem="relaxed")
            dim += 1
        time_index += 1


@triton.jit
def _modal_backward_kernel(
    states_real,
    states_imag,
    writer_real,
    writer_imag,
    grad_modal,
    grad_states_real,
    grad_states_imag,
    grad_writer_real,
    grad_writer_imag,
    n_steps: int,
    modes: int,
    model_dim: int,
) -> None:
    lane = tl.program_id(0)
    batch = lane // modes
    mode = lane - batch * modes
    base = batch * n_steps * modes + mode
    time_index = 0
    while time_index < n_steps:
        offset = base + time_index * modes
        state_real = tl.load(states_real + offset)
        state_imag = tl.load(states_imag + offset)
        state_grad_real = tl.full((), 0.0, tl.float32)
        state_grad_imag = tl.full((), 0.0, tl.float32)
        dim = 0
        modal_base = (batch * n_steps + time_index) * model_dim
        writer_base = mode * model_dim
        while dim < model_dim:
            grad = tl.load(grad_modal + modal_base + dim)
            wr = tl.load(writer_real + writer_base + dim)
            wi = tl.load(writer_imag + writer_base + dim)
            state_grad_real += 2.0 * grad * wr
            state_grad_imag += -2.0 * grad * wi
            tl.atomic_add(
                grad_writer_real + writer_base + dim,
                2.0 * grad * state_real,
                sem="relaxed",
            )
            tl.atomic_add(
                grad_writer_imag + writer_base + dim,
                -2.0 * grad * state_imag,
                sem="relaxed",
            )
            dim += 1
        tl.store(grad_states_real + offset, state_grad_real)
        tl.store(grad_states_imag + offset, state_grad_imag)
        time_index += 1


class _TritonModalFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _AutogradContext,
        decay_real: Tensor,
        decay_imag: Tensor,
        input_real: Tensor,
        input_imag: Tensor,
        writer_real: Tensor,
        writer_imag: Tensor,
    ) -> Tensor:
        real = decay_real.contiguous()
        imag = decay_imag.contiguous()
        drive_real = input_real.contiguous()
        drive_imag = input_imag.contiguous()
        writer_r = writer_real.contiguous()
        writer_i = writer_imag.contiguous()
        batch, n_steps, modes = real.shape
        model_dim = writer_r.shape[1]
        states_real = torch.empty_like(real)
        states_imag = torch.empty_like(real)
        modal = torch.zeros(batch, n_steps, model_dim, dtype=real.dtype, device=real.device)
        _modal_forward_kernel[(batch * modes,)](
            real,
            imag,
            drive_real,
            drive_imag,
            writer_r,
            writer_i,
            states_real,
            states_imag,
            modal,
            n_steps,
            modes,
            model_dim,
        )
        ctx.save_for_backward(real, imag, states_real, states_imag, writer_r, writer_i)
        return modal

    @staticmethod
    def backward(
        ctx: _AutogradContext,
        *grad_outputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        (grad_modal,) = grad_outputs
        decay_real, decay_imag, states_real, states_imag, writer_real, writer_imag = (
            ctx.saved_tensors
        )
        grad_states_real = torch.empty_like(states_real)
        grad_states_imag = torch.empty_like(states_imag)
        grad_writer_real = torch.zeros_like(writer_real)
        grad_writer_imag = torch.zeros_like(writer_imag)
        batch, n_steps, modes = states_real.shape
        model_dim = writer_real.shape[1]
        _modal_backward_kernel[(batch * modes,)](
            states_real,
            states_imag,
            writer_real,
            writer_imag,
            grad_modal.contiguous(),
            grad_states_real,
            grad_states_imag,
            grad_writer_real,
            grad_writer_imag,
            n_steps,
            modes,
            model_dim,
        )
        grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag = (
            triton_recurrence_backward_from_states(
                decay_real,
                decay_imag,
                states_real,
                states_imag,
                grad_states_real,
                grad_states_imag,
            )
        )
        return (
            grad_decay_real,
            grad_decay_imag,
            grad_input_real,
            grad_input_imag,
            grad_writer_real,
            grad_writer_imag,
        )


def triton_modal_fused_output(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    writer_real: Tensor,
    writer_imag: Tensor,
) -> Tensor:
    output = _TritonModalFused.apply(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        writer_real,
        writer_imag,
    )
    if not isinstance(output, Tensor):
        message = "Triton modal fused output must be a tensor"
        raise TypeError(message)
    return output
