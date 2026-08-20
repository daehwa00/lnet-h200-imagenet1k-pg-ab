from __future__ import annotations

# pyright: reportMissingParameterType=false
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
def _scan_local_kernel(
    decay_real,
    decay_imag,
    input_real,
    input_imag,
    local_real,
    local_imag,
    prefix_real,
    prefix_imag,
    block_decay_real,
    block_decay_imag,
    block_shift_real,
    block_shift_imag,
    n_steps: int,
    modes: int,
    block_count: int,
    block_n,
) -> None:
    lane = tl.program_id(0)
    block = tl.program_id(1)
    batch = lane // modes
    mode = lane - batch * modes
    base = batch * n_steps * modes + mode
    summary = lane * block_count + block
    state_real = tl.full((), 0.0, tl.float32)
    state_imag = tl.full((), 0.0, tl.float32)
    affine_real = tl.full((), 1.0, tl.float32)
    affine_imag = tl.full((), 0.0, tl.float32)
    step = 0
    start = block * block_n
    while step < block_n:
        time_index = start + step
        active = time_index < n_steps
        offset = base + time_index * modes
        ar = tl.load(decay_real + offset, mask=active, other=1.0)
        ai = tl.load(decay_imag + offset, mask=active, other=0.0)
        ur = tl.load(input_real + offset, mask=active, other=0.0)
        ui = tl.load(input_imag + offset, mask=active, other=0.0)
        previous_real = state_real
        previous_imag = state_imag
        state_real = ar * previous_real - ai * previous_imag + ur
        state_imag = ai * previous_real + ar * previous_imag + ui
        previous_affine_real = affine_real
        previous_affine_imag = affine_imag
        affine_real = ar * previous_affine_real - ai * previous_affine_imag
        affine_imag = ai * previous_affine_real + ar * previous_affine_imag
        tl.store(local_real + offset, state_real, mask=active)
        tl.store(local_imag + offset, state_imag, mask=active)
        tl.store(prefix_real + offset, affine_real, mask=active)
        tl.store(prefix_imag + offset, affine_imag, mask=active)
        step += 1
    tl.store(block_decay_real + summary, affine_real)
    tl.store(block_decay_imag + summary, affine_imag)
    tl.store(block_shift_real + summary, state_real)
    tl.store(block_shift_imag + summary, state_imag)


@triton.jit
def _scan_carry_kernel(
    block_decay_real,
    block_decay_imag,
    block_shift_real,
    block_shift_imag,
    carry_real,
    carry_imag,
    block_count: int,
) -> None:
    lane = tl.program_id(0)
    state_real = tl.full((), 0.0, tl.float32)
    state_imag = tl.full((), 0.0, tl.float32)
    block = 0
    while block < block_count:
        offset = lane * block_count + block
        tl.store(carry_real + offset, state_real)
        tl.store(carry_imag + offset, state_imag)
        ar = tl.load(block_decay_real + offset)
        ai = tl.load(block_decay_imag + offset)
        ur = tl.load(block_shift_real + offset)
        ui = tl.load(block_shift_imag + offset)
        previous_real = state_real
        previous_imag = state_imag
        state_real = ar * previous_real - ai * previous_imag + ur
        state_imag = ai * previous_real + ar * previous_imag + ui
        block += 1


@triton.jit
def _scan_apply_kernel(
    local_real,
    local_imag,
    prefix_real,
    prefix_imag,
    carry_real,
    carry_imag,
    states_real,
    states_imag,
    n_steps: int,
    modes: int,
    block_count: int,
    block_n,
) -> None:
    lane = tl.program_id(0)
    block = tl.program_id(1)
    batch = lane // modes
    mode = lane - batch * modes
    base = batch * n_steps * modes + mode
    summary = lane * block_count + block
    carry_r = tl.load(carry_real + summary)
    carry_i = tl.load(carry_imag + summary)
    step = 0
    start = block * block_n
    while step < block_n:
        time_index = start + step
        active = time_index < n_steps
        offset = base + time_index * modes
        ar = tl.load(prefix_real + offset, mask=active, other=1.0)
        ai = tl.load(prefix_imag + offset, mask=active, other=0.0)
        ur = tl.load(local_real + offset, mask=active, other=0.0)
        ui = tl.load(local_imag + offset, mask=active, other=0.0)
        tl.store(states_real + offset, ar * carry_r - ai * carry_i + ur, mask=active)
        tl.store(states_imag + offset, ai * carry_r + ar * carry_i + ui, mask=active)
        step += 1


class _TritonScanBlocks(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _AutogradContext,
        decay_real: Tensor,
        decay_imag: Tensor,
        input_real: Tensor,
        input_imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        block_n = 256
        real = decay_real.contiguous()
        imag = decay_imag.contiguous()
        drive_real = input_real.contiguous()
        drive_imag = input_imag.contiguous()
        batch, n_steps, modes = real.shape
        block_count = (n_steps + block_n - 1) // block_n
        shape = (batch, n_steps, modes)
        summary_shape = (batch * modes, block_count)
        local_real = torch.empty(shape, dtype=real.dtype, device=real.device)
        local_imag = torch.empty_like(local_real)
        prefix_real = torch.empty_like(local_real)
        prefix_imag = torch.empty_like(local_real)
        states_real = torch.empty_like(local_real)
        states_imag = torch.empty_like(local_real)
        block_decay_real = torch.empty(summary_shape, dtype=real.dtype, device=real.device)
        block_decay_imag = torch.empty_like(block_decay_real)
        block_shift_real = torch.empty_like(block_decay_real)
        block_shift_imag = torch.empty_like(block_decay_real)
        carry_real = torch.empty_like(block_decay_real)
        carry_imag = torch.empty_like(block_decay_real)
        grid = (batch * modes, block_count)
        _scan_local_kernel[grid](
            real,
            imag,
            drive_real,
            drive_imag,
            local_real,
            local_imag,
            prefix_real,
            prefix_imag,
            block_decay_real,
            block_decay_imag,
            block_shift_real,
            block_shift_imag,
            n_steps,
            modes,
            block_count,
            block_n,
        )
        _scan_carry_kernel[(batch * modes,)](
            block_decay_real,
            block_decay_imag,
            block_shift_real,
            block_shift_imag,
            carry_real,
            carry_imag,
            block_count,
        )
        _scan_apply_kernel[grid](
            local_real,
            local_imag,
            prefix_real,
            prefix_imag,
            carry_real,
            carry_imag,
            states_real,
            states_imag,
            n_steps,
            modes,
            block_count,
            block_n,
        )
        ctx.save_for_backward(real, imag, states_real, states_imag)
        return states_real, states_imag

    @staticmethod
    def backward(
        ctx: _AutogradContext,
        *grad_outputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        decay_real, decay_imag, states_real, states_imag = ctx.saved_tensors
        return triton_recurrence_backward_from_states(
            decay_real,
            decay_imag,
            states_real,
            states_imag,
            grad_outputs[0],
            grad_outputs[1],
        )


def triton_scan_blocks_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    outputs = _TritonScanBlocks.apply(decay_real, decay_imag, input_real, input_imag)
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        message = "Triton scan returned an invalid output"
        raise RuntimeError(message)
    states_real, states_imag = outputs
    if not isinstance(states_real, Tensor) or not isinstance(states_imag, Tensor):
        message = "Triton scan outputs must be tensors"
        raise TypeError(message)
    return states_real, states_imag
