from __future__ import annotations

# pyright: reportMissingParameterType=false, reportPrivateUsage=false
# ruff: noqa: ANN001, N803
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from .pac_triton_online_moments import reference_online_modal_moments
from .pac_triton_recurrence_op import (
    _is_mode_static_expanded,
    _mode_grid,
    _select_block_modes,
    pac_triton_recurrence_op,
)

_EPSILON: Final[float] = 1.0e-8


@triton.jit
def _recurrence_moments_kernel(  # noqa: PLR0915
    decay_real,
    decay_imag,
    input_real,
    input_imag,
    states_real,
    states_imag,
    moment_output,
    n_steps: int,
    modes: int,
    epsilon: float,
    reverse: tl.constexpr,
    static_decay: tl.constexpr,
    packed_input: tl.constexpr,
    packed_output: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    base = batch * n_steps * modes + mode
    state_real = tl.zeros((BLOCK_MODES,), tl.float32)
    state_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history1_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history1_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr1_real_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr1_imag_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr4_real_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr4_imag_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    current1_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    previous1_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    current4_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    previous4_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_real = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    if static_decay:
        fixed_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
        fixed_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    step = 0
    while step < n_steps:
        time_index = n_steps - 1 - step if reverse else step
        offset = base + time_index * modes
        if static_decay:
            ar = fixed_decay_real
            ai = fixed_decay_imag
        else:
            ar = tl.load(decay_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
            ai = tl.load(decay_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        if packed_input:
            packed_input_offset = (batch * n_steps + time_index) * 2 * modes + mode
            drive_real = tl.load(
                input_real + packed_input_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            drive_imag = tl.load(
                input_real + packed_input_offset + modes,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        else:
            drive_real = tl.load(input_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
            drive_imag = tl.load(input_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        previous_state_real = state_real
        previous_state_imag = state_imag
        state_real = ar * previous_state_real - ai * previous_state_imag + drive_real
        state_imag = ai * previous_state_real + ar * previous_state_imag + drive_imag
        if packed_output:
            packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
            tl.store(states_real + packed_offset, state_real, mask=valid_mode)
            tl.store(states_real + packed_offset + modes, state_imag, mask=valid_mode)
        else:
            tl.store(states_real + offset, state_real, mask=valid_mode)
            tl.store(states_imag + offset, state_imag, mask=valid_mode)

        current_energy = state_real * state_real + state_imag * state_imag
        energy_sum += current_energy
        valid1 = step >= 1
        valid4 = step >= 4
        if reverse:
            corr1_real = history1_real * state_real + history1_imag * state_imag
            corr1_imag = history1_imag * state_real - history1_real * state_imag
            corr4_real = history4_real * state_real + history4_imag * state_imag
            corr4_imag = history4_imag * state_real - history4_real * state_imag
            current1_energy = history1_real * history1_real + history1_imag * history1_imag
            previous1_energy = current_energy
            current4_energy = history4_real * history4_real + history4_imag * history4_imag
            previous4_energy = current_energy
        else:
            corr1_real = state_real * history1_real + state_imag * history1_imag
            corr1_imag = state_imag * history1_real - state_real * history1_imag
            corr4_real = state_real * history4_real + state_imag * history4_imag
            corr4_imag = state_imag * history4_real - state_real * history4_imag
            current1_energy = current_energy
            previous1_energy = history1_real * history1_real + history1_imag * history1_imag
            current4_energy = current_energy
            previous4_energy = history4_real * history4_real + history4_imag * history4_imag

        corr1_real_sum += tl.where(valid1, corr1_real, 0.0)
        corr1_imag_sum += tl.where(valid1, corr1_imag, 0.0)
        current1_energy_sum += tl.where(valid1, current1_energy, 0.0)
        previous1_energy_sum += tl.where(valid1, previous1_energy, 0.0)
        corr4_real_sum += tl.where(valid4, corr4_real, 0.0)
        corr4_imag_sum += tl.where(valid4, corr4_imag, 0.0)
        current4_energy_sum += tl.where(valid4, current4_energy, 0.0)
        previous4_energy_sum += tl.where(valid4, previous4_energy, 0.0)

        history4_real = history3_real
        history4_imag = history3_imag
        history3_real = history2_real
        history3_imag = history2_imag
        history2_real = history1_real
        history2_imag = history1_imag
        history1_real = state_real
        history1_imag = state_imag
        step += 1

    energy = energy_sum / n_steps
    count1 = tl.maximum(n_steps - 1, 1)
    count4 = tl.maximum(n_steps - 4, 1)
    denominator1 = tl.maximum(
        tl.sqrt((current1_energy_sum / count1) * (previous1_energy_sum / count1)),
        epsilon,
    )
    denominator4 = tl.maximum(
        tl.sqrt((current4_energy_sum / count4) * (previous4_energy_sum / count4)),
        epsilon,
    )
    corr1_real = tl.where(n_steps > 1, (corr1_real_sum / count1) / denominator1, 0.0)
    corr1_imag = tl.where(n_steps > 1, (corr1_imag_sum / count1) / denominator1, 0.0)
    corr4_real = tl.where(n_steps > 4, (corr4_real_sum / count4) / denominator4, 0.0)
    corr4_imag = tl.where(n_steps > 4, (corr4_imag_sum / count4) / denominator4, 0.0)
    moment_base = batch * 5 * modes + mode
    tl.store(moment_output + moment_base, libdevice.log1p(energy), mask=valid_mode)
    tl.store(moment_output + moment_base + modes, corr1_real, mask=valid_mode)
    tl.store(moment_output + moment_base + 2 * modes, corr1_imag, mask=valid_mode)
    tl.store(moment_output + moment_base + 3 * modes, corr4_real, mask=valid_mode)
    tl.store(moment_output + moment_base + 4 * modes, corr4_imag, mask=valid_mode)


@torch.library.triton_op("lnet::pac_real2d_recurrence_moments", mutates_args={})
def _pac_real2d_recurrence_moments_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_inputs(decay_real, decay_imag, input_real, input_imag, epsilon)
    if not decay_real.is_cuda:
        return reference_recurrence_moments(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=reverse,
            epsilon=epsilon,
        )
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
    moments = torch.empty(
        (batch, 5 * modes),
        dtype=real.dtype,
        device=real.device,
    )
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    torch.library.wrap_triton(_recurrence_moments_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        drive_real,
        drive_imag,
        states_real,
        states_imag,
        moments,
        n_steps,
        modes,
        epsilon,
        reverse=reverse,
        static_decay=static_decay,
        packed_input=False,
        packed_output=False,
        BLOCK_MODES=block_modes,
    )
    return states_real, states_imag, moments


@torch.library.triton_op("lnet::pac_static_real2d_recurrence_moments", mutates_args={})
def _pac_static_real2d_recurrence_moments_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_static_inputs(decay_real, decay_imag, input_real, input_imag, epsilon)
    if not input_real.is_cuda:
        expanded_real = decay_real.view(1, 1, -1).expand_as(input_real)
        expanded_imag = decay_imag.view(1, 1, -1).expand_as(input_imag)
        return reference_recurrence_moments(
            expanded_real,
            expanded_imag,
            input_real,
            input_imag,
            reverse=reverse,
            epsilon=epsilon,
        )
    real = decay_real.contiguous()
    imag = decay_imag.contiguous()
    drive_real = input_real.contiguous()
    drive_imag = input_imag.contiguous()
    states_real = torch.empty_like(drive_real)
    states_imag = torch.empty_like(drive_imag)
    batch, n_steps, modes = drive_real.shape
    moments = torch.empty(
        (batch, 5 * modes),
        dtype=drive_real.dtype,
        device=drive_real.device,
    )
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    torch.library.wrap_triton(_recurrence_moments_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        drive_real,
        drive_imag,
        states_real,
        states_imag,
        moments,
        n_steps,
        modes,
        epsilon,
        reverse=reverse,
        static_decay=True,
        packed_input=False,
        packed_output=False,
        BLOCK_MODES=block_modes,
        num_warps=1 if single_warp else 4,
    )
    return states_real, states_imag, moments


@torch.library.triton_op("lnet::pac_static_real2d_recurrence_moments_packed", mutates_args={})
def _pac_static_real2d_recurrence_moments_packed_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor]:
    _validate_static_inputs(decay_real, decay_imag, input_real, input_imag, epsilon)
    if not input_real.is_cuda:
        states_real, states_imag, moments = static_recurrence_moments_inference(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=reverse,
            epsilon=epsilon,
            single_warp=single_warp,
        )
        return torch.cat((states_real, states_imag), dim=-1), moments
    real = decay_real.contiguous()
    imag = decay_imag.contiguous()
    drive_real = input_real.contiguous()
    drive_imag = input_imag.contiguous()
    batch, n_steps, modes = drive_real.shape
    packed_states = torch.empty(
        (batch, n_steps, 2 * modes),
        dtype=drive_real.dtype,
        device=drive_real.device,
    )
    moments = torch.empty(
        (batch, 5 * modes),
        dtype=drive_real.dtype,
        device=drive_real.device,
    )
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    torch.library.wrap_triton(_recurrence_moments_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        drive_real,
        drive_imag,
        packed_states,
        packed_states,
        moments,
        n_steps,
        modes,
        epsilon,
        reverse=reverse,
        static_decay=True,
        packed_input=False,
        packed_output=True,
        BLOCK_MODES=block_modes,
        num_warps=1 if single_warp else 4,
    )
    return packed_states, moments


@torch.library.triton_op("lnet::pac_static_real2d_recurrence_moments_packed_input", mutates_args={})
def _pac_static_real2d_recurrence_moments_packed_input_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_static_packed_inputs(decay_real, decay_imag, packed_input, epsilon)
    modes = decay_real.shape[0]
    if not packed_input.is_cuda:
        input_real, input_imag = packed_input.split(modes, dim=-1)
        return static_recurrence_moments_inference(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=reverse,
            epsilon=epsilon,
            single_warp=single_warp,
        )
    real = decay_real.contiguous()
    imag = decay_imag.contiguous()
    drive = packed_input.contiguous()
    batch, n_steps, _ = drive.shape
    states_real = torch.empty(
        (batch, n_steps, modes),
        dtype=drive.dtype,
        device=drive.device,
    )
    states_imag = torch.empty_like(states_real)
    moments = torch.empty(
        (batch, 5 * modes),
        dtype=drive.dtype,
        device=drive.device,
    )
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    torch.library.wrap_triton(_recurrence_moments_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        drive,
        drive,
        states_real,
        states_imag,
        moments,
        n_steps,
        modes,
        epsilon,
        reverse=reverse,
        static_decay=True,
        packed_input=True,
        packed_output=False,
        BLOCK_MODES=block_modes,
        num_warps=1 if single_warp else 4,
    )
    return states_real, states_imag, moments


@torch.library.triton_op("lnet::pac_static_real2d_recurrence_moments_packed_io", mutates_args={})
def _pac_static_real2d_recurrence_moments_packed_io_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor]:
    _validate_static_packed_inputs(decay_real, decay_imag, packed_input, epsilon)
    modes = decay_real.shape[0]
    if not packed_input.is_cuda:
        input_real, input_imag = packed_input.split(modes, dim=-1)
        states_real, states_imag, moments = static_recurrence_moments_inference(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=reverse,
            epsilon=epsilon,
        )
        return torch.cat((states_real, states_imag), dim=-1), moments
    real = decay_real.contiguous()
    imag = decay_imag.contiguous()
    drive = packed_input.contiguous()
    batch, n_steps, _ = drive.shape
    packed_states = torch.empty_like(drive)
    moments = torch.empty(
        (batch, 5 * modes),
        dtype=drive.dtype,
        device=drive.device,
    )
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    torch.library.wrap_triton(_recurrence_moments_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        drive,
        drive,
        packed_states,
        packed_states,
        moments,
        n_steps,
        modes,
        epsilon,
        reverse=reverse,
        static_decay=True,
        packed_input=True,
        packed_output=True,
        BLOCK_MODES=block_modes,
        num_warps=1 if single_warp else 4,
    )
    return packed_states, moments


def recurrence_moments_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse inference recurrence and physical-time canonical modal moments."""
    return _pac_real2d_recurrence_moments_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=reverse,
        epsilon=epsilon,
    )


def static_recurrence_moments_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse fixed-pole inference recurrence with canonical modal moments."""
    return _pac_static_real2d_recurrence_moments_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
    )


def static_recurrence_moments_packed_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor]:
    """Fuse fixed-pole recurrence/moments and emit synthesis-ready [real|imag] states."""
    return _pac_static_real2d_recurrence_moments_packed_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
    )


def static_recurrence_moments_packed_io_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor]:
    """Consume and emit contiguous [real|imag] tensors around fixed-pole recurrence."""
    return _pac_static_real2d_recurrence_moments_packed_io_op(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
    )


def static_recurrence_moments_packed_input_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Consume [real|imag] drive while retaining separate recurrent state outputs."""
    return _pac_static_real2d_recurrence_moments_packed_input_op(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
    )


def reference_recurrence_moments(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_inputs(decay_real, decay_imag, input_real, input_imag, epsilon)
    states_real, states_imag = pac_triton_recurrence_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=reverse,
    )
    moments = reference_online_modal_moments(
        states_real,
        states_imag,
        physical_direction="forward",
        epsilon=epsilon,
    )
    return states_real, states_imag, moments


def _validate_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    epsilon: float,
) -> None:
    shape = decay_real.shape
    if decay_real.ndim != 3 or shape[1] == 0 or shape[2] == 0:
        message = "recurrence-moments tensors must have non-empty [batch, time, modes] shape"
        raise ValueError(message)
    for tensor in (decay_imag, input_real, input_imag):
        if tensor.shape != shape:
            message = "recurrence-moments tensors must have matching shapes"
            raise ValueError(message)
        if tensor.device != decay_real.device or tensor.dtype != decay_real.dtype:
            message = "recurrence-moments tensors must share device and dtype"
            raise ValueError(message)
    if decay_real.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        message = "recurrence-moments supports fp16, bf16, and fp32"
        raise TypeError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)


def _validate_static_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    epsilon: float,
) -> None:
    if decay_real.ndim != 1 or decay_real.shape[0] == 0:
        message = "static decay tensors must have non-empty [modes] shape"
        raise ValueError(message)
    if decay_imag.shape != decay_real.shape:
        message = "static decay tensors must have matching shapes"
        raise ValueError(message)
    if input_real.ndim != 3 or input_real.shape[1] == 0:
        message = "static recurrence inputs must have non-empty [batch, time, modes] shape"
        raise ValueError(message)
    if input_imag.shape != input_real.shape or input_real.shape[2] != decay_real.shape[0]:
        message = "static recurrence input and decay shapes must match in modes"
        raise ValueError(message)
    for tensor in (decay_imag, input_real, input_imag):
        if tensor.device != decay_real.device or tensor.dtype != decay_real.dtype:
            message = "static recurrence tensors must share device and dtype"
            raise ValueError(message)
    if decay_real.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        message = "static recurrence-moments supports fp16, bf16, and fp32"
        raise TypeError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)


def _validate_static_packed_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    epsilon: float,
) -> None:
    if decay_real.ndim != 1 or decay_real.shape[0] == 0:
        message = "static decay tensors must have non-empty [modes] shape"
        raise ValueError(message)
    if decay_imag.shape != decay_real.shape:
        message = "static decay tensors must have matching shapes"
        raise ValueError(message)
    if (
        packed_input.ndim != 3
        or packed_input.shape[1] == 0
        or packed_input.shape[2] != 2 * decay_real.shape[0]
    ):
        message = "packed static recurrence input must have [batch, time, 2 * modes] shape"
        raise ValueError(message)
    for tensor in (decay_imag, packed_input):
        if tensor.device != decay_real.device or tensor.dtype != decay_real.dtype:
            message = "static recurrence tensors must share device and dtype"
            raise ValueError(message)
    if decay_real.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        message = "static recurrence-moments supports fp16, bf16, and fp32"
        raise TypeError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)
