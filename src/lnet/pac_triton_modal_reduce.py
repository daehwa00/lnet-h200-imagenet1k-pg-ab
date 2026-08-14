from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor

from .pac_triton_recurrence import (
    triton_fused_recurrence,
    triton_recurrence_backward_from_states,
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _modal_reduce_forward_kernel(
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
    block_m: tl.constexpr,
    block_d: tl.constexpr,
    store_states: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    dim_block = tl.program_id(1)
    mode_offsets = tl.arange(0, block_m)
    dim_offsets = dim_block * block_d + tl.arange(0, block_d)
    mode_mask = mode_offsets < modes
    dim_mask = dim_offsets < model_dim
    state_real = tl.zeros((block_m,), tl.float32)
    state_imag = tl.zeros((block_m,), tl.float32)
    base = batch * n_steps * modes + mode_offsets
    time_index = 0
    while time_index < n_steps:
        offset = base + time_index * modes
        ar = tl.load(decay_real + offset, mask=mode_mask, other=1.0)
        ai = tl.load(decay_imag + offset, mask=mode_mask, other=0.0)
        ur = tl.load(input_real + offset, mask=mode_mask, other=0.0)
        ui = tl.load(input_imag + offset, mask=mode_mask, other=0.0)
        previous_real = state_real
        previous_imag = state_imag
        state_real = ar * previous_real - ai * previous_imag + ur
        state_imag = ai * previous_real + ar * previous_imag + ui
        if store_states and dim_block == 0:
            tl.store(states_real + offset, state_real, mask=mode_mask)
            tl.store(states_imag + offset, state_imag, mask=mode_mask)
        wr = tl.load(
            writer_real + mode_offsets[:, None] * model_dim + dim_offsets[None, :],
            mask=mode_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        wi = tl.load(
            writer_imag + mode_offsets[:, None] * model_dim + dim_offsets[None, :],
            mask=mode_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        reduced = tl.sum(state_real[:, None] * wr - state_imag[:, None] * wi, axis=0)
        output_offsets = (batch * n_steps + time_index) * model_dim + dim_offsets
        tl.store(modal + output_offsets, 2.0 * reduced, mask=dim_mask)
        time_index += 1


def _modal_parameter_grads(
    states_real: Tensor,
    states_imag: Tensor,
    writer_real: Tensor,
    writer_imag: Tensor,
    grad_modal: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    grad_states_real = 2.0 * torch.einsum("bnd,md->bnm", grad_modal, writer_real)
    grad_states_imag = -2.0 * torch.einsum("bnd,md->bnm", grad_modal, writer_imag)
    grad_writer_real = 2.0 * torch.einsum("bnm,bnd->md", states_real, grad_modal)
    grad_writer_imag = -2.0 * torch.einsum("bnm,bnd->md", states_imag, grad_modal)
    return grad_states_real, grad_states_imag, grad_writer_real, grad_writer_imag


def _forward_reduce(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    writer_real: Tensor,
    writer_imag: Tensor,
    *,
    store_states: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    block_m = triton.next_power_of_2(decay_real.shape[2])
    block_d = min(triton.next_power_of_2(writer_real.shape[1]), 16)
    batch, n_steps, modes = decay_real.shape
    model_dim = writer_real.shape[1]
    states_real = torch.empty_like(decay_real)
    states_imag = torch.empty_like(decay_imag)
    modal = torch.empty(batch, n_steps, model_dim, dtype=decay_real.dtype, device=decay_real.device)
    _modal_reduce_forward_kernel[(batch, triton.cdiv(model_dim, block_d))](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        input_real.contiguous(),
        input_imag.contiguous(),
        writer_real.contiguous(),
        writer_imag.contiguous(),
        states_real,
        states_imag,
        modal,
        n_steps,
        modes,
        model_dim,
        block_m,
        block_d,
        store_states,
    )
    return modal, states_real, states_imag


class _ModalReduceSave(torch.autograd.Function):
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
        modal, states_real, states_imag = _forward_reduce(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            writer_real,
            writer_imag,
            store_states=True,
        )
        ctx.save_for_backward(
            decay_real,
            decay_imag,
            states_real,
            states_imag,
            writer_real,
            writer_imag,
        )
        return modal

    @staticmethod
    def backward(ctx: _AutogradContext, *grad_outputs: Tensor) -> tuple[Tensor, ...]:
        decay_real, decay_imag, states_real, states_imag, writer_real, writer_imag = (
            ctx.saved_tensors
        )
        grad_states_real, grad_states_imag, grad_writer_real, grad_writer_imag = (
            _modal_parameter_grads(
                states_real,
                states_imag,
                writer_real,
                writer_imag,
                grad_outputs[0],
            )
        )
        grads = triton_recurrence_backward_from_states(
            decay_real, decay_imag, states_real, states_imag, grad_states_real, grad_states_imag
        )
        return (*grads, grad_writer_real, grad_writer_imag)


class _ModalReduceRecompute(torch.autograd.Function):
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
        modal, _states_real, _states_imag = _forward_reduce(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            writer_real,
            writer_imag,
            store_states=False,
        )
        ctx.save_for_backward(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            writer_real,
            writer_imag,
        )
        return modal

    @staticmethod
    def backward(ctx: _AutogradContext, *grad_outputs: Tensor) -> tuple[Tensor, ...]:
        decay_real, decay_imag, input_real, input_imag, writer_real, writer_imag = ctx.saved_tensors
        states_real, states_imag = triton_fused_recurrence(
            decay_real, decay_imag, input_real, input_imag
        )
        grad_states_real, grad_states_imag, grad_writer_real, grad_writer_imag = (
            _modal_parameter_grads(
                states_real,
                states_imag,
                writer_real,
                writer_imag,
                grad_outputs[0],
            )
        )
        grads = triton_recurrence_backward_from_states(
            decay_real, decay_imag, states_real, states_imag, grad_states_real, grad_states_imag
        )
        return (*grads, grad_writer_real, grad_writer_imag)


def triton_modal_reduce_output(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    writer_real: Tensor,
    writer_imag: Tensor,
    *,
    recompute_backward: bool,
) -> Tensor:
    function = _ModalReduceRecompute if recompute_backward else _ModalReduceSave
    output = function.apply(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        writer_real,
        writer_imag,
    )
    if not isinstance(output, Tensor):
        message = "Triton modal reduce output must be a tensor"
        raise TypeError(message)
    return output
