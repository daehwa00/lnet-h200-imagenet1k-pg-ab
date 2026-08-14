from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001, N803
from dataclasses import dataclass
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

_DEFAULT_BLOCK_N: Final[int] = 256
_VALID_BLOCK_SIZES: Final[tuple[int, ...]] = (64, 128, 256)


@dataclass(frozen=True, slots=True)
class StaticBlockScanWorkspace:
    """Logical live allocation footprint, including returned state tensors."""

    sequence_tensor_count: int
    summary_tensor_count: int
    bytes: int


def static_block_scan_workspace(
    batch_size: int,
    sequence_length: int,
    modes: int,
    *,
    element_size: int = 4,
    block_size: int = _DEFAULT_BLOCK_N,
) -> StaticBlockScanWorkspace:
    """Return the kernel's live scan allocations, including returned states."""
    if min(batch_size, sequence_length, modes, element_size) < 1:
        message = "static block-scan workspace dimensions must be positive"
        raise ValueError(message)
    _validate_block_size(block_size)
    block_count = (sequence_length + block_size - 1) // block_size
    sequence_elements = 2 * batch_size * sequence_length * modes
    summary_elements = (
        2 * batch_size * modes * block_count + 2 * batch_size * modes
    )
    return StaticBlockScanWorkspace(
        sequence_tensor_count=2,
        summary_tensor_count=4,
        bytes=(sequence_elements + summary_elements) * element_size,
    )


@triton.jit
def _static_scan_local_kernel(  # noqa: PLR0915
    decay_real,
    decay_imag,
    input_real,
    input_imag,
    states_real,
    states_imag,
    block_shift_real,
    block_shift_imag,
    block_decay_real,
    block_decay_imag,
    n_steps: int,
    modes: int,
    block_count: int,
    reverse: tl.constexpr,
    PACKED_INPUT: tl.constexpr,
    PACKED_OUTPUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    lane = tl.program_id(0)
    block = tl.program_id(1)
    batch = lane // modes
    mode = lane - batch * modes
    input_base = batch * n_steps * modes + mode
    output_base = batch * n_steps * modes + mode
    if PACKED_INPUT:
        input_base = batch * n_steps * 2 * modes + mode
    if PACKED_OUTPUT:
        output_base = batch * n_steps * 2 * modes + mode
    ar = tl.load(decay_real + mode).to(tl.float32)
    ai = tl.load(decay_imag + mode).to(tl.float32)
    state_real = tl.full((), 0.0, tl.float32)
    state_imag = tl.full((), 0.0, tl.float32)
    affine_real = tl.full((), 1.0, tl.float32)
    affine_imag = tl.full((), 0.0, tl.float32)
    start = block * BLOCK_N
    step = 0
    while step < BLOCK_N:
        traversal_index = start + step
        active = traversal_index < n_steps
        time_index = n_steps - 1 - traversal_index if reverse else traversal_index
        input_offset = input_base + time_index * modes
        output_offset = output_base + time_index * modes
        if PACKED_INPUT:
            input_offset = input_base + time_index * 2 * modes
            drive_real = tl.load(input_real + input_offset, mask=active, other=0.0).to(
                tl.float32
            )
            drive_imag = tl.load(
                input_real + input_offset + modes,
                mask=active,
                other=0.0,
            ).to(tl.float32)
        else:
            drive_real = tl.load(input_real + input_offset, mask=active, other=0.0).to(
                tl.float32
            )
            drive_imag = tl.load(input_imag + input_offset, mask=active, other=0.0).to(
                tl.float32
            )
        previous_real = state_real
        previous_imag = state_imag
        next_real = ar * previous_real - ai * previous_imag + drive_real
        next_imag = ai * previous_real + ar * previous_imag + drive_imag
        state_real = tl.where(active, next_real, state_real)
        state_imag = tl.where(active, next_imag, state_imag)
        previous_affine_real = affine_real
        previous_affine_imag = affine_imag
        next_affine_real = ar * previous_affine_real - ai * previous_affine_imag
        next_affine_imag = ai * previous_affine_real + ar * previous_affine_imag
        affine_real = tl.where(active, next_affine_real, affine_real)
        affine_imag = tl.where(active, next_affine_imag, affine_imag)
        if PACKED_OUTPUT:
            output_offset = output_base + time_index * 2 * modes
            tl.store(states_real + output_offset, state_real, mask=active)
            tl.store(states_real + output_offset + modes, state_imag, mask=active)
        else:
            tl.store(states_real + output_offset, state_real, mask=active)
            tl.store(states_imag + output_offset, state_imag, mask=active)
        step += 1

    summary = lane * block_count + block
    tl.store(block_shift_real + summary, state_real)
    tl.store(block_shift_imag + summary, state_imag)
    # Every block before a possible short tail contains BLOCK_N steps.  Keeping
    # one decay summary per lane is therefore sufficient and avoids two [B,M,K]
    # summaries.  Exactly the block-zero program writes each lane's value.
    tl.store(block_decay_real + lane, affine_real, mask=block == 0)
    tl.store(block_decay_imag + lane, affine_imag, mask=block == 0)


@triton.jit
def _static_scan_apply_kernel(  # noqa: PLR0915
    states_real,
    states_imag,
    block_shift_real,
    block_shift_imag,
    block_decay_real,
    block_decay_imag,
    decay_real,
    decay_imag,
    n_steps: int,
    modes: int,
    block_count: int,
    reverse: tl.constexpr,
    PACKED_OUTPUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    lane = tl.program_id(0)
    block = tl.program_id(1)
    batch = lane // modes
    mode = lane - batch * modes
    output_base = batch * n_steps * modes + mode
    if PACKED_OUTPUT:
        output_base = batch * n_steps * 2 * modes + mode
    block_ar = tl.load(block_decay_real + lane).to(tl.float32)
    block_ai = tl.load(block_decay_imag + lane).to(tl.float32)
    carry_real = tl.full((), 0.0, tl.float32)
    carry_imag = tl.full((), 0.0, tl.float32)
    prior = 0
    # Recompute the tiny exclusive carry prefix from at most seven summaries at
    # N=2048.  This removes the carry kernel and its two output buffers while
    # preserving the original left-to-right FP32 affine composition order.
    while prior < block:
        summary = lane * block_count + prior
        shift_real = tl.load(block_shift_real + summary).to(tl.float32)
        shift_imag = tl.load(block_shift_imag + summary).to(tl.float32)
        previous_real = carry_real
        previous_imag = carry_imag
        carry_real = block_ar * previous_real - block_ai * previous_imag + shift_real
        carry_imag = block_ai * previous_real + block_ar * previous_imag + shift_imag
        prior += 1

    ar = tl.load(decay_real + mode).to(tl.float32)
    ai = tl.load(decay_imag + mode).to(tl.float32)
    affine_real = tl.full((), 1.0, tl.float32)
    affine_imag = tl.full((), 0.0, tl.float32)
    start = block * BLOCK_N
    step = 0
    while step < BLOCK_N:
        traversal_index = start + step
        active = traversal_index < n_steps
        time_index = n_steps - 1 - traversal_index if reverse else traversal_index
        output_offset = output_base + time_index * modes
        if PACKED_OUTPUT:
            output_offset = output_base + time_index * 2 * modes
        previous_affine_real = affine_real
        previous_affine_imag = affine_imag
        affine_real = ar * previous_affine_real - ai * previous_affine_imag
        affine_imag = ai * previous_affine_real + ar * previous_affine_imag
        local_real = tl.load(states_real + output_offset, mask=active, other=0.0).to(
            tl.float32
        )
        if PACKED_OUTPUT:
            local_imag = tl.load(
                states_real + output_offset + modes,
                mask=active,
                other=0.0,
            ).to(tl.float32)
        else:
            local_imag = tl.load(states_imag + output_offset, mask=active, other=0.0).to(
                tl.float32
            )
        output_real = affine_real * carry_real - affine_imag * carry_imag + local_real
        output_imag = affine_imag * carry_real + affine_real * carry_imag + local_imag
        tl.store(states_real + output_offset, output_real, mask=active)
        if PACKED_OUTPUT:
            tl.store(states_real + output_offset + modes, output_imag, mask=active)
        else:
            tl.store(states_imag + output_offset, output_imag, mask=active)
        step += 1


def _validate_static_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
) -> None:
    if decay_real.ndim != 1 or decay_real.numel() == 0:
        message = "static block-scan decay must have shape (modes,)"
        raise ValueError(message)
    if decay_imag.shape != decay_real.shape:
        message = "static block-scan decay tensors must have matching shapes"
        raise ValueError(message)
    if input_real.ndim != 3 or input_real.shape[1] == 0:
        message = "static block-scan inputs must have shape (batch, steps, modes)"
        raise ValueError(message)
    if input_real.shape[-1] != decay_real.numel() or input_imag.shape != input_real.shape:
        message = "static block-scan input and mode dimensions must match"
        raise ValueError(message)
    tensors = (decay_real, decay_imag, input_real, input_imag)
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        message = "static block-scan prototype requires exact FP32 tensors"
        raise TypeError(message)
    if any(tensor.device != input_real.device for tensor in tensors):
        message = "static block-scan tensors must share one device"
        raise ValueError(message)


def _validate_static_packed_input(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
) -> None:
    if decay_real.ndim != 1 or decay_real.numel() == 0:
        message = "static block-scan decay must have shape (modes,)"
        raise ValueError(message)
    if decay_imag.shape != decay_real.shape:
        message = "static block-scan decay tensors must have matching shapes"
        raise ValueError(message)
    modes = decay_real.numel()
    if (
        packed_input.ndim != 3
        or packed_input.shape[1] == 0
        or packed_input.shape[-1] != 2 * modes
    ):
        message = "packed static block-scan input must have shape (batch, steps, 2*modes)"
        raise ValueError(message)
    tensors = (decay_real, decay_imag, packed_input)
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        message = "static block-scan prototype requires exact FP32 tensors"
        raise TypeError(message)
    if any(tensor.device != packed_input.device for tensor in tensors):
        message = "static block-scan tensors must share one device"
        raise ValueError(message)


def _validate_num_warps(num_warps: int) -> None:
    if num_warps not in (1, 4):
        message = "static block-scan num_warps must be 1 or 4"
        raise ValueError(message)


def _validate_block_size(block_size: int) -> None:
    if block_size not in _VALID_BLOCK_SIZES:
        message = f"static block-scan block_size must be one of {_VALID_BLOCK_SIZES}"
        raise ValueError(message)


def _reference_static_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool,
) -> tuple[Tensor, Tensor]:
    states_real = torch.empty_like(input_real)
    states_imag = torch.empty_like(input_imag)
    state_real = torch.zeros_like(input_real[:, 0, :])
    state_imag = torch.zeros_like(input_imag[:, 0, :])
    indices = range(input_real.shape[1] - 1, -1, -1) if reverse else range(input_real.shape[1])
    for time_index in indices:
        previous_real = state_real
        previous_imag = state_imag
        state_real = (
            decay_real * previous_real
            - decay_imag * previous_imag
            + input_real[:, time_index, :]
        )
        state_imag = (
            decay_imag * previous_real
            + decay_real * previous_imag
            + input_imag[:, time_index, :]
        )
        states_real[:, time_index, :] = state_real
        states_imag[:, time_index, :] = state_imag
    return states_real, states_imag


def _launch_static_block_scan(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool,
    packed_input: bool,
    packed_output: bool,
    num_warps: int,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    real = decay_real.contiguous()
    imag = decay_imag.contiguous()
    drive_real = input_real.contiguous()
    drive_imag = input_imag.contiguous()
    batch, n_steps, input_width = drive_real.shape
    modes = decay_real.shape[0]
    if packed_input and input_width != 2 * modes:
        message = "packed static block-scan input width changed after validation"
        raise RuntimeError(message)
    block_count = (n_steps + block_size - 1) // block_size
    if packed_output:
        states_real = torch.empty(
            (batch, n_steps, 2 * modes),
            dtype=torch.float32,
            device=drive_real.device,
        )
        states_imag = states_real
    else:
        states_real = torch.empty(
            (batch, n_steps, modes),
            dtype=torch.float32,
            device=drive_real.device,
        )
        states_imag = torch.empty_like(states_real)
    summary_shape = (batch * modes, block_count)
    block_shift_real = torch.empty(summary_shape, dtype=torch.float32, device=drive_real.device)
    block_shift_imag = torch.empty_like(block_shift_real)
    block_decay_real = torch.empty((batch * modes,), dtype=torch.float32, device=drive_real.device)
    block_decay_imag = torch.empty_like(block_decay_real)
    grid = (batch * modes, block_count)
    wrap_triton(_static_scan_local_kernel)[grid](
        real,
        imag,
        drive_real,
        drive_imag,
        states_real,
        states_imag,
        block_shift_real,
        block_shift_imag,
        block_decay_real,
        block_decay_imag,
        n_steps,
        modes,
        block_count,
        reverse=reverse,
        PACKED_INPUT=packed_input,
        PACKED_OUTPUT=packed_output,
        BLOCK_N=block_size,
        num_warps=num_warps,
    )
    wrap_triton(_static_scan_apply_kernel)[grid](
        states_real,
        states_imag,
        block_shift_real,
        block_shift_imag,
        block_decay_real,
        block_decay_imag,
        real,
        imag,
        n_steps,
        modes,
        block_count,
        reverse=reverse,
        PACKED_OUTPUT=packed_output,
        BLOCK_N=block_size,
        num_warps=num_warps,
    )
    return states_real, states_imag


@triton_op("lnet::pac_static_pole_block_scan", mutates_args={})
def _static_pole_block_scan_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool,
    num_warps: int = 4,
    block_size: int = _DEFAULT_BLOCK_N,
) -> tuple[Tensor, Tensor]:
    _validate_static_inputs(decay_real, decay_imag, input_real, input_imag)
    _validate_num_warps(num_warps)
    _validate_block_size(block_size)
    if not input_real.is_cuda:
        return _reference_static_recurrence(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=reverse,
        )
    return _launch_static_block_scan(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=reverse,
        packed_input=False,
        packed_output=False,
        num_warps=num_warps,
        block_size=block_size,
    )


@triton_op("lnet::pac_static_pole_block_scan_packed_input", mutates_args={})
def _static_pole_block_scan_packed_input_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    num_warps: int = 4,
    block_size: int = _DEFAULT_BLOCK_N,
) -> tuple[Tensor, Tensor]:
    _validate_static_packed_input(decay_real, decay_imag, packed_input)
    _validate_num_warps(num_warps)
    _validate_block_size(block_size)
    modes = decay_real.shape[0]
    if not packed_input.is_cuda:
        input_real, input_imag = packed_input.split(modes, dim=-1)
        return _reference_static_recurrence(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=reverse,
        )
    return _launch_static_block_scan(
        decay_real,
        decay_imag,
        packed_input,
        packed_input,
        reverse=reverse,
        packed_input=True,
        packed_output=False,
        num_warps=num_warps,
        block_size=block_size,
    )


@triton_op("lnet::pac_static_pole_block_scan_packed_output", mutates_args={})
def _static_pole_block_scan_packed_output_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool,
    num_warps: int = 4,
    block_size: int = _DEFAULT_BLOCK_N,
) -> Tensor:
    _validate_static_inputs(decay_real, decay_imag, input_real, input_imag)
    _validate_num_warps(num_warps)
    _validate_block_size(block_size)
    if not input_real.is_cuda:
        states_real, states_imag = _reference_static_recurrence(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=reverse,
        )
        return torch.cat((states_real, states_imag), dim=-1)
    packed_states, _ = _launch_static_block_scan(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=reverse,
        packed_input=False,
        packed_output=True,
        num_warps=num_warps,
        block_size=block_size,
    )
    return packed_states


@triton_op("lnet::pac_static_pole_block_scan_packed_io", mutates_args={})
def _static_pole_block_scan_packed_io_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    num_warps: int = 4,
    block_size: int = _DEFAULT_BLOCK_N,
) -> Tensor:
    _validate_static_packed_input(decay_real, decay_imag, packed_input)
    _validate_num_warps(num_warps)
    _validate_block_size(block_size)
    modes = decay_real.shape[0]
    if not packed_input.is_cuda:
        input_real, input_imag = packed_input.split(modes, dim=-1)
        states_real, states_imag = _reference_static_recurrence(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=reverse,
        )
        return torch.cat((states_real, states_imag), dim=-1)
    packed_states, _ = _launch_static_block_scan(
        decay_real,
        decay_imag,
        packed_input,
        packed_input,
        reverse=reverse,
        packed_input=True,
        packed_output=True,
        num_warps=num_warps,
        block_size=block_size,
    )
    return packed_states


def static_pole_block_scan_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool = False,
    num_warps: int = 4,
    block_size: int = _DEFAULT_BLOCK_N,
) -> tuple[Tensor, Tensor]:
    """Exact-FP32 static-pole block scan with gamma folding supplied by its caller."""
    return _static_pole_block_scan_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=reverse,
        num_warps=num_warps,
        block_size=block_size,
    )


def static_pole_block_scan_packed_input_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    num_warps: int = 4,
    block_size: int = _DEFAULT_BLOCK_N,
) -> tuple[Tensor, Tensor]:
    """Consume contiguous ``[real|imag]`` drive and return separate states."""
    return _static_pole_block_scan_packed_input_op(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        num_warps=num_warps,
        block_size=block_size,
    )


def static_pole_block_scan_packed_output_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool = False,
    num_warps: int = 4,
    block_size: int = _DEFAULT_BLOCK_N,
) -> Tensor:
    """Consume separate drive tensors and return synthesis-ready packed states."""
    return _static_pole_block_scan_packed_output_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=reverse,
        num_warps=num_warps,
        block_size=block_size,
    )


def static_pole_block_scan_packed_io_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    num_warps: int = 4,
    block_size: int = _DEFAULT_BLOCK_N,
) -> Tensor:
    """Consume and emit contiguous ``[real|imag]`` tensors."""
    return _static_pole_block_scan_packed_io_op(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        num_warps=num_warps,
        block_size=block_size,
    )
