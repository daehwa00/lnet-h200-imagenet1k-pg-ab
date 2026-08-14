from __future__ import annotations

# pyright: reportArgumentType=false, reportMissingParameterType=false
# ruff: noqa: ANN001
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor

from .pac_triton_recurrence_op import _is_mode_static_expanded


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    reverse: int
    static_decay: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _recurrence_forward_kernel(
    decay_real,
    decay_imag,
    input_real,
    input_imag,
    states_real,
    states_imag,
    n_steps: int,
    modes: int,
    reverse: tl.constexpr,
    static_decay: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    batch = program // modes
    mode = program - batch * modes
    base = batch * n_steps * modes + mode
    state_real = tl.full((), 0.0, tl.float32)
    state_imag = tl.full((), 0.0, tl.float32)
    fixed_decay_real = tl.full((), 0.0, tl.float32)
    fixed_decay_imag = tl.full((), 0.0, tl.float32)
    if static_decay:
        fixed_decay_real = tl.load(decay_real + mode)
        fixed_decay_imag = tl.load(decay_imag + mode)
    step = 0
    while step < n_steps:
        time_index = n_steps - 1 - step if reverse else step
        offset = base + time_index * modes
        if static_decay:
            ar = fixed_decay_real
            ai = fixed_decay_imag
        else:
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
        step += 1


@triton.jit
def _recurrence_backward_kernel(
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
    reverse: tl.constexpr,
    static_decay: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    batch = program // modes
    mode = program - batch * modes
    base = batch * n_steps * modes + mode
    lambda_real = tl.full((), 0.0, tl.float32)
    lambda_imag = tl.full((), 0.0, tl.float32)
    fixed_decay_real = tl.full((), 0.0, tl.float32)
    fixed_decay_imag = tl.full((), 0.0, tl.float32)
    if static_decay:
        fixed_decay_real = tl.load(decay_real + mode)
        fixed_decay_imag = tl.load(decay_imag + mode)
    step = 0
    while step < n_steps:
        time_index = step if reverse else n_steps - 1 - step
        offset = base + time_index * modes
        lambda_real += tl.load(grad_states_real + offset)
        lambda_imag += tl.load(grad_states_imag + offset)
        previous_index = time_index + 1 if reverse else time_index - 1
        previous_offset = base + previous_index * modes
        has_previous = time_index < n_steps - 1 if reverse else time_index > 0
        previous_real = tl.load(states_real + previous_offset, mask=has_previous, other=0.0)
        previous_imag = tl.load(states_imag + previous_offset, mask=has_previous, other=0.0)
        if static_decay:
            ar = fixed_decay_real
            ai = fixed_decay_imag
        else:
            ar = tl.load(decay_real + offset)
            ai = tl.load(decay_imag + offset)
        tl.store(grad_input_real + offset, lambda_real)
        tl.store(grad_input_imag + offset, lambda_imag)
        decay_real_grad = lambda_real * previous_real + lambda_imag * previous_imag
        decay_imag_grad = -lambda_real * previous_imag + lambda_imag * previous_real
        tl.store(grad_decay_real + offset, decay_real_grad)
        tl.store(grad_decay_imag + offset, decay_imag_grad)
        next_lambda_real = ar * lambda_real + ai * lambda_imag
        next_lambda_imag = -ai * lambda_real + ar * lambda_imag
        lambda_real = next_lambda_real
        lambda_imag = next_lambda_imag
        step += 1


class _TritonFusedRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _AutogradContext,
        decay_real: Tensor,
        decay_imag: Tensor,
        input_real: Tensor,
        input_imag: Tensor,
        reverse: int,
    ) -> tuple[Tensor, Tensor]:
        static_decay = _is_mode_static_expanded(decay_real, input_real) and (
            _is_mode_static_expanded(decay_imag, input_imag)
        )
        real = decay_real if static_decay else decay_real.contiguous()
        imag = decay_imag if static_decay else decay_imag.contiguous()
        drive_real = input_real.contiguous()
        drive_imag = input_imag.contiguous()
        states_real = torch.empty_like(drive_real)
        states_imag = torch.empty_like(drive_imag)
        batch, n_steps, modes = drive_real.shape
        _recurrence_forward_kernel[(batch * modes,)](
            real,
            imag,
            drive_real,
            drive_imag,
            states_real,
            states_imag,
            n_steps,
            modes,
            reverse,
            static_decay=static_decay,
        )
        ctx.reverse = reverse
        ctx.static_decay = static_decay
        ctx.save_for_backward(real, imag, states_real, states_imag)
        return states_real, states_imag

    @staticmethod
    def backward(
        ctx: _AutogradContext,
        *grad_outputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
        grad_states_real, grad_states_imag = grad_outputs
        decay_real, decay_imag, states_real, states_imag = ctx.saved_tensors
        grad_decay_real = torch.empty_like(states_real)
        grad_decay_imag = torch.empty_like(states_imag)
        grad_input_real = torch.empty_like(states_real)
        grad_input_imag = torch.empty_like(states_imag)
        batch, n_steps, modes = states_real.shape
        _recurrence_backward_kernel[(batch * modes,)](
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
            ctx.reverse,
            static_decay=ctx.static_decay,
        )
        return grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag, None


def triton_fused_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    outputs = _TritonFusedRecurrence.apply(
        decay_real, decay_imag, input_real, input_imag, int(reverse)
    )
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        message = "Triton recurrence returned an invalid output"
        raise RuntimeError(message)
    states_real, states_imag = outputs
    if not isinstance(states_real, Tensor) or not isinstance(states_imag, Tensor):
        message = "Triton recurrence outputs must be tensors"
        raise TypeError(message)
    return states_real, states_imag


def triton_recurrence_backward_from_states(
    decay_real: Tensor,
    decay_imag: Tensor,
    states_real: Tensor,
    states_imag: Tensor,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    static_decay = _is_mode_static_expanded(decay_real, states_real) and (
        _is_mode_static_expanded(decay_imag, states_imag)
    )
    grad_decay_real = torch.empty_like(states_real)
    grad_decay_imag = torch.empty_like(states_imag)
    grad_input_real = torch.empty_like(states_real)
    grad_input_imag = torch.empty_like(states_imag)
    batch, n_steps, modes = states_real.shape
    _recurrence_backward_kernel[(batch * modes,)](
        decay_real if static_decay else decay_real.contiguous(),
        decay_imag if static_decay else decay_imag.contiguous(),
        states_real.contiguous(),
        states_imag.contiguous(),
        grad_states_real.contiguous(),
        grad_states_imag.contiguous(),
        grad_decay_real,
        grad_decay_imag,
        grad_input_real,
        grad_input_imag,
        n_steps,
        modes,
        reverse,
        static_decay=static_decay,
    )
    return grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag
