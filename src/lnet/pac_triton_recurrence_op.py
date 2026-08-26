from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportMissingParameterType=false
import os
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

_VALID_BLOCK_MODES: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)
_VALID_NUM_WARPS: Final[tuple[int, ...]] = (1, 2, 4, 8)


def _select_block_modes(modes: int, *, batch: int | None = None, n_steps: int | None = None) -> int:
    """Choose how many independent modes one Triton program processes.

    Keeping time in the inner scalar loop preserves the exact FP32 recurrence
    order for every mode.  The environment override is intentionally private;
    it exists so the hardware screen can compare candidates without changing
    the custom-op schema or invalidating CUDA Graph callers.
    """
    override = os.environ.get("LNET_PAC_BLOCK_MODES")
    if override is not None:
        block_modes = int(override)
        if block_modes not in _VALID_BLOCK_MODES:
            message = f"LNET_PAC_BLOCK_MODES must be one of {_VALID_BLOCK_MODES}"
            raise ValueError(message)
        return block_modes
    if batch is None or n_steps is None:
        return 1

    # The six-shape FP32 training screen has two occupancy regimes.  Short and
    # mid-length single-batch work benefits from keeping all 16 canonical
    # modes together, whereas long B64 adjoints need more independent programs.
    if n_steps <= 512:
        selected = 16 if batch == 1 else 8
    else:
        selected = 8 if batch == 1 else 2
    maximum_useful = 1 << (modes - 1).bit_length()
    return min(selected, maximum_useful)


def _mode_grid(batch: int, modes: int, block_modes: int) -> tuple[int]:
    return (batch * ((modes + block_modes - 1) // block_modes),)


def _select_recurrence_num_warps(
    *,
    batch: int,
    n_steps: int,
    modes: int,
    device: torch.device,
    backward: bool,
) -> int:
    """Select a portable default with optional direction-specific overrides."""
    del batch, n_steps, modes, device
    direction = "BACKWARD" if backward else "FORWARD"
    override_name = f"LNET_PAC_RECURRENCE_{direction}_NUM_WARPS"
    override = os.environ.get(override_name)
    active_name = override_name
    if override is None:
        active_name = "LNET_PAC_RECURRENCE_NUM_WARPS"
        override = os.environ.get(active_name)
    if override is not None:
        num_warps = int(override)
        if num_warps not in _VALID_NUM_WARPS:
            message = f"{active_name} must be one of {_VALID_NUM_WARPS}"
            raise ValueError(message)
        return num_warps

    return 4


def is_mode_static_expanded(decay: Tensor, reference: Tensor) -> bool:
    """Return whether ``decay`` is a broadcast view of one value per mode.

    Fixed-damping training creates the pole once as ``[1, 1, M]`` and expands
    it to the recurrence's logical ``[B, N, M]`` shape.  Recognising the two
    zero strides here lets Triton load each pole once without materialising the
    expanded tensor.  The logical shape remains unchanged, so the existing
    autograd contract still returns a per-step pole gradient; PyTorch's
    ExpandBackward then performs the deterministic reduction to mode-static
    pole parameters without adding an atomic reduction to the Triton kernel.
    """
    return (
        decay.shape == reference.shape
        and decay.ndim == 3
        and decay.stride(0) == 0
        and decay.stride(1) == 0
        and decay.stride(2) == 1
    )


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    reverse: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...


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
    fixed_decay_real = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    if static_decay:
        fixed_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0)
        fixed_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0)
    step = 0
    while step < n_steps:
        time_index = n_steps - 1 - step if reverse else step
        offset = base + time_index * modes
        if static_decay:
            ar = fixed_decay_real
            ai = fixed_decay_imag
        else:
            ar = tl.load(decay_real + offset, mask=valid_mode, other=0.0)
            ai = tl.load(decay_imag + offset, mask=valid_mode, other=0.0)
        ur = tl.load(input_real + offset, mask=valid_mode, other=0.0)
        ui = tl.load(input_imag + offset, mask=valid_mode, other=0.0)
        previous_real = state_real
        previous_imag = state_imag
        state_real = ar * previous_real - ai * previous_imag + ur
        state_imag = ai * previous_real + ar * previous_imag + ui
        tl.store(states_real + offset, state_real, mask=valid_mode)
        tl.store(states_imag + offset, state_imag, mask=valid_mode)
        step += 1


@triton.jit
def _state_variance_forward_kernel(
    decay_real,
    decay_imag,
    input_real,
    input_imag,
    variance_decay,
    variance_input,
    states_real,
    states_imag,
    variance_states,
    n_steps: int,
    modes: int,
    reverse: tl.constexpr,
    static_decay: tl.constexpr,
    static_variance_decay: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    """Transport a complex state and its detached gain variance together."""
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    base = batch * n_steps * modes + mode
    state_real = tl.zeros((BLOCK_MODES,), tl.float32)
    state_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    variance_state = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_real = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_variance_decay = tl.zeros((BLOCK_MODES,), tl.float32)
    if static_decay:
        fixed_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0)
        fixed_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0)
    if static_variance_decay:
        fixed_variance_decay = tl.load(variance_decay + mode, mask=valid_mode, other=0.0)
    step = 0
    while step < n_steps:
        time_index = n_steps - 1 - step if reverse else step
        offset = base + time_index * modes
        if static_decay:
            ar = fixed_decay_real
            ai = fixed_decay_imag
        else:
            ar = tl.load(decay_real + offset, mask=valid_mode, other=0.0)
            ai = tl.load(decay_imag + offset, mask=valid_mode, other=0.0)
        variance_ar = (
            fixed_variance_decay
            if static_variance_decay
            else tl.load(variance_decay + offset, mask=valid_mode, other=0.0)
        )
        ur = tl.load(input_real + offset, mask=valid_mode, other=0.0)
        ui = tl.load(input_imag + offset, mask=valid_mode, other=0.0)
        variance_u = tl.load(variance_input + offset, mask=valid_mode, other=0.0)
        previous_real = state_real
        previous_imag = state_imag
        state_real = ar * previous_real - ai * previous_imag + ur
        state_imag = ai * previous_real + ar * previous_imag + ui
        variance_state = variance_ar * variance_state + variance_u
        tl.store(states_real + offset, state_real, mask=valid_mode)
        tl.store(states_imag + offset, state_imag, mask=valid_mode)
        tl.store(variance_states + offset, variance_state, mask=valid_mode)
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
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    base = batch * n_steps * modes + mode
    lambda_real = tl.zeros((BLOCK_MODES,), tl.float32)
    lambda_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_real = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    if static_decay:
        fixed_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0)
        fixed_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0)
    step = 0
    while step < n_steps:
        time_index = step if reverse else n_steps - 1 - step
        offset = base + time_index * modes
        lambda_real += tl.load(grad_states_real + offset, mask=valid_mode, other=0.0)
        lambda_imag += tl.load(grad_states_imag + offset, mask=valid_mode, other=0.0)
        previous_index = time_index + 1 if reverse else time_index - 1
        previous_offset = base + previous_index * modes
        has_previous = time_index < n_steps - 1 if reverse else time_index > 0
        previous_real = tl.load(
            states_real + previous_offset,
            mask=valid_mode & has_previous,
            other=0.0,
        )
        previous_imag = tl.load(
            states_imag + previous_offset,
            mask=valid_mode & has_previous,
            other=0.0,
        )
        if static_decay:
            ar = fixed_decay_real
            ai = fixed_decay_imag
        else:
            ar = tl.load(decay_real + offset, mask=valid_mode, other=0.0)
            ai = tl.load(decay_imag + offset, mask=valid_mode, other=0.0)
        tl.store(grad_input_real + offset, lambda_real, mask=valid_mode)
        tl.store(grad_input_imag + offset, lambda_imag, mask=valid_mode)
        decay_real_grad = lambda_real * previous_real + lambda_imag * previous_imag
        decay_imag_grad = -lambda_real * previous_imag + lambda_imag * previous_real
        tl.store(grad_decay_real + offset, decay_real_grad, mask=valid_mode)
        tl.store(grad_decay_imag + offset, decay_imag_grad, mask=valid_mode)
        next_lambda_real = ar * lambda_real + ai * lambda_imag
        next_lambda_imag = -ai * lambda_real + ar * lambda_imag
        lambda_real = next_lambda_real
        lambda_imag = next_lambda_imag
        step += 1


def _validate_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
) -> None:
    shape = decay_real.shape
    if len(shape) != 3:
        message = "PAC recurrence tensors must have shape (batch, steps, modes)"
        raise ValueError(message)
    if shape[1] == 0 or shape[2] == 0:
        message = "PAC recurrence requires non-zero step and mode dimensions"
        raise ValueError(message)
    if decay_imag.shape != shape or input_real.shape != shape or input_imag.shape != shape:
        message = "PAC recurrence tensors must have matching shapes"
        raise ValueError(message)
    if decay_real.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        message = "PAC recurrence supports float16, bfloat16, and float32 tensors"
        raise TypeError(message)
    if (
        decay_imag.dtype != decay_real.dtype
        or input_real.dtype != decay_real.dtype
        or input_imag.dtype != decay_real.dtype
    ):
        message = "PAC recurrence tensors must have matching dtypes"
        raise TypeError(message)
    if (
        decay_imag.device != decay_real.device
        or input_real.device != decay_real.device
        or input_imag.device != decay_real.device
    ):
        message = "PAC recurrence tensors must be on the same device"
        raise ValueError(message)


def _reference_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor]:
    state_real = torch.zeros(
        input_real.shape[0],
        input_real.shape[2],
        dtype=torch.float32,
        device=input_real.device,
    )
    state_imag = torch.zeros_like(state_real)
    states_real: list[Tensor | None] = [None] * input_real.shape[1]
    states_imag: list[Tensor | None] = [None] * input_real.shape[1]
    steps = range(input_real.shape[1] - 1, -1, -1) if reverse else range(input_real.shape[1])
    for time_index in steps:
        previous_real = state_real
        previous_imag = state_imag
        ar = decay_real[:, time_index, :].float()
        ai = decay_imag[:, time_index, :].float()
        state_real = ar * previous_real - ai * previous_imag + input_real[:, time_index, :].float()
        state_imag = ai * previous_real + ar * previous_imag + input_imag[:, time_index, :].float()
        states_real[time_index] = state_real
        states_imag[time_index] = state_imag
    real = torch.stack([state for state in states_real if state is not None], dim=1)
    imag = torch.stack([state for state in states_imag if state is not None], dim=1)
    return real.to(input_real.dtype), imag.to(input_imag.dtype)


def _reference_recurrence_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    states_real: Tensor,
    states_imag: Tensor,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lambda_real = torch.zeros(
        decay_real.shape[0],
        decay_real.shape[2],
        dtype=torch.float32,
        device=decay_real.device,
    )
    lambda_imag = torch.zeros_like(lambda_real)
    grad_decay_real: list[Tensor | None] = [None] * decay_real.shape[1]
    grad_decay_imag: list[Tensor | None] = [None] * decay_real.shape[1]
    grad_input_real: list[Tensor | None] = [None] * decay_real.shape[1]
    grad_input_imag: list[Tensor | None] = [None] * decay_real.shape[1]
    steps = range(decay_real.shape[1]) if reverse else range(decay_real.shape[1] - 1, -1, -1)
    for time_index in steps:
        lambda_real = lambda_real + grad_states_real[:, time_index, :].float()
        lambda_imag = lambda_imag + grad_states_imag[:, time_index, :].float()
        has_previous = time_index < decay_real.shape[1] - 1 if reverse else time_index > 0
        if has_previous:
            previous_index = time_index + 1 if reverse else time_index - 1
            previous_real = states_real[:, previous_index, :].float()
            previous_imag = states_imag[:, previous_index, :].float()
        else:
            previous_real = torch.zeros_like(lambda_real)
            previous_imag = torch.zeros_like(lambda_imag)
        grad_input_real[time_index] = lambda_real
        grad_input_imag[time_index] = lambda_imag
        grad_decay_real[time_index] = lambda_real * previous_real + lambda_imag * previous_imag
        grad_decay_imag[time_index] = -lambda_real * previous_imag + lambda_imag * previous_real
        ar = decay_real[:, time_index, :].float()
        ai = decay_imag[:, time_index, :].float()
        next_lambda_real = ar * lambda_real + ai * lambda_imag
        next_lambda_imag = -ai * lambda_real + ar * lambda_imag
        lambda_real = next_lambda_real
        lambda_imag = next_lambda_imag
    return (
        torch.stack([grad for grad in grad_decay_real if grad is not None], dim=1).to(
            decay_real.dtype
        ),
        torch.stack([grad for grad in grad_decay_imag if grad is not None], dim=1).to(
            decay_imag.dtype
        ),
        torch.stack([grad for grad in grad_input_real if grad is not None], dim=1).to(
            grad_states_real.dtype
        ),
        torch.stack([grad for grad in grad_input_imag if grad is not None], dim=1).to(
            grad_states_imag.dtype
        ),
    )


@triton_op("lnet::pac_real2d_recurrence_backward", mutates_args={})
def _pac_real2d_recurrence_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    states_real: Tensor,
    states_imag: Tensor,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_inputs(decay_real, decay_imag, states_real, states_imag)
    _validate_inputs(decay_real, decay_imag, grad_states_real, grad_states_imag)
    if (
        not decay_real.is_cuda
        or not decay_imag.is_cuda
        or not grad_states_real.is_cuda
        or not grad_states_imag.is_cuda
    ):
        return _reference_recurrence_backward(
            decay_real,
            decay_imag,
            states_real,
            states_imag,
            grad_states_real,
            grad_states_imag,
            reverse,
        )
    static_decay = is_mode_static_expanded(decay_real, grad_states_real) and (
        is_mode_static_expanded(decay_imag, grad_states_imag)
    )
    real = decay_real if static_decay else decay_real.contiguous()
    imag = decay_imag if static_decay else decay_imag.contiguous()
    state_real = states_real.contiguous()
    state_imag = states_imag.contiguous()
    grad_state_real = grad_states_real.contiguous()
    grad_state_imag = grad_states_imag.contiguous()
    grad_decay_real = torch.empty_like(grad_state_real)
    grad_decay_imag = torch.empty_like(grad_state_imag)
    grad_input_real = torch.empty_like(grad_state_real)
    grad_input_imag = torch.empty_like(grad_state_imag)
    batch, n_steps, modes = grad_state_real.shape
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    num_warps = _select_recurrence_num_warps(
        batch=batch,
        n_steps=n_steps,
        modes=modes,
        device=grad_state_real.device,
        backward=True,
    )
    wrap_triton(_recurrence_backward_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        state_real,
        state_imag,
        grad_state_real,
        grad_state_imag,
        grad_decay_real,
        grad_decay_imag,
        grad_input_real,
        grad_input_imag,
        n_steps,
        modes,
        reverse,
        static_decay=static_decay,
        BLOCK_MODES=block_modes,
        num_warps=num_warps,
    )
    return grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag


@triton_op("lnet::pac_real2d_state_variance_recurrence", mutates_args={})
def _pac_real2d_state_variance_recurrence_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    variance_decay: Tensor,
    variance_input: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse state transport with a deliberately stop-gradient variance scan."""
    _validate_inputs(decay_real, decay_imag, input_real, input_imag)
    if variance_decay.shape != input_real.shape or variance_input.shape != input_real.shape:
        message = "PAC variance recurrence tensors must match the state shape"
        raise ValueError(message)
    if variance_decay.dtype != input_real.dtype or variance_input.dtype != input_real.dtype:
        message = "PAC variance recurrence tensors must match the state dtype"
        raise TypeError(message)
    if variance_decay.device != input_real.device or variance_input.device != input_real.device:
        message = "PAC variance recurrence tensors must be on the state device"
        raise ValueError(message)
    if not decay_real.is_cuda:
        states_real, states_imag = _reference_recurrence(
            decay_real, decay_imag, input_real, input_imag, reverse
        )
        variance_states, _ = _reference_recurrence(
            variance_decay,
            torch.zeros_like(variance_decay),
            variance_input,
            torch.zeros_like(variance_input),
            reverse,
        )
        return states_real, states_imag, variance_states
    static_decay = is_mode_static_expanded(decay_real, input_real) and (
        is_mode_static_expanded(decay_imag, input_imag)
    )
    static_variance_decay = is_mode_static_expanded(variance_decay, variance_input)
    real = decay_real if static_decay else decay_real.contiguous()
    imag = decay_imag if static_decay else decay_imag.contiguous()
    variance_real = variance_decay if static_variance_decay else variance_decay.contiguous()
    drive_real = input_real.contiguous()
    drive_imag = input_imag.contiguous()
    variance_drive = variance_input.contiguous()
    states_real = torch.empty_like(drive_real)
    states_imag = torch.empty_like(drive_imag)
    variance_states = torch.empty_like(variance_drive)
    batch, n_steps, modes = drive_real.shape
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    num_warps = _select_recurrence_num_warps(
        batch=batch,
        n_steps=n_steps,
        modes=modes,
        device=drive_real.device,
        backward=False,
    )
    wrap_triton(_state_variance_forward_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        drive_real,
        drive_imag,
        variance_real,
        variance_drive,
        states_real,
        states_imag,
        variance_states,
        n_steps,
        modes,
        reverse,
        static_decay=static_decay,
        static_variance_decay=static_variance_decay,
        BLOCK_MODES=block_modes,
        num_warps=num_warps,
    )
    return states_real, states_imag, variance_states


def _setup_state_variance_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    decay_real, decay_imag, _input_real, _input_imag, _variance_decay, _variance_input, reverse = (
        inputs
    )
    states_real, states_imag, _variance_states = output
    ctx.reverse = reverse
    ctx.mark_non_differentiable(_variance_states)
    ctx.save_for_backward(decay_real, decay_imag, states_real, states_imag)


def _state_variance_backward(
    ctx: _AutogradContext,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
    _grad_variance_states: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, None, None, None]:
    decay_real, decay_imag, states_real, states_imag = ctx.saved_tensors
    gradients = _pac_real2d_recurrence_backward_op(
        decay_real,
        decay_imag,
        states_real,
        states_imag,
        grad_states_real,
        grad_states_imag,
        ctx.reverse,
    )
    return *gradients, None, None, None


torch.library.register_autograd(
    "lnet::pac_real2d_state_variance_recurrence",
    _state_variance_backward,
    setup_context=_setup_state_variance_context,
)


@triton_op("lnet::pac_real2d_recurrence", mutates_args={})
def _pac_real2d_recurrence_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor]:
    _validate_inputs(decay_real, decay_imag, input_real, input_imag)
    if not decay_real.is_cuda:
        return _reference_recurrence(decay_real, decay_imag, input_real, input_imag, reverse)
    static_decay = is_mode_static_expanded(decay_real, input_real) and (
        is_mode_static_expanded(decay_imag, input_imag)
    )
    real = decay_real if static_decay else decay_real.contiguous()
    imag = decay_imag if static_decay else decay_imag.contiguous()
    drive_real = input_real.contiguous()
    drive_imag = input_imag.contiguous()
    states_real = torch.empty_like(drive_real)
    states_imag = torch.empty_like(drive_imag)
    batch, n_steps, modes = drive_real.shape
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    num_warps = _select_recurrence_num_warps(
        batch=batch,
        n_steps=n_steps,
        modes=modes,
        device=drive_real.device,
        backward=False,
    )
    wrap_triton(_recurrence_forward_kernel)[_mode_grid(batch, modes, block_modes)](
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
        BLOCK_MODES=block_modes,
        num_warps=num_warps,
    )
    return states_real, states_imag


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, bool],
    output: tuple[Tensor, Tensor],
) -> None:
    decay_real, decay_imag, _input_real, _input_imag, reverse = inputs
    states_real, states_imag = output
    ctx.reverse = reverse
    ctx.save_for_backward(decay_real, decay_imag, states_real, states_imag)


def _backward(
    ctx: _AutogradContext,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
    decay_real, decay_imag, states_real, states_imag = ctx.saved_tensors
    grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag = (
        _pac_real2d_recurrence_backward_op(
            decay_real,
            decay_imag,
            states_real,
            states_imag,
            grad_states_real,
            grad_states_imag,
            ctx.reverse,
        )
    )
    return grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag, None


torch.library.register_autograd(
    "lnet::pac_real2d_recurrence",
    _backward,
    setup_context=_setup_context,
)


@torch.library.custom_op(
    "lnet::pac_real2d_recurrence_opaque_backward",
    mutates_args=(),
)
def _opaque_recurrence_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    states_real: Tensor,
    states_imag: Tensor,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Keep the hand-written scan backward opaque to outer AOTAutograd."""
    return _pac_real2d_recurrence_backward_op(
        decay_real,
        decay_imag,
        states_real,
        states_imag,
        grad_states_real,
        grad_states_imag,
        reverse,
    )


@_opaque_recurrence_backward_op.register_fake
def _opaque_recurrence_backward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    states_real: Tensor,
    states_imag: Tensor,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del decay_real, decay_imag, states_real, states_imag, reverse
    return (
        torch.empty_like(grad_states_real, memory_format=torch.contiguous_format),
        torch.empty_like(grad_states_imag, memory_format=torch.contiguous_format),
        torch.empty_like(grad_states_real, memory_format=torch.contiguous_format),
        torch.empty_like(grad_states_imag, memory_format=torch.contiguous_format),
    )


@torch.library.custom_op(
    "lnet::pac_real2d_recurrence_opaque",
    mutates_args=(),
)
def _opaque_recurrence_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor]:
    """Expose the scan as one Mamba-style custom-autograd boundary."""
    return _pac_real2d_recurrence_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse,
    )


@_opaque_recurrence_op.register_fake
def _opaque_recurrence_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor]:
    del decay_real, decay_imag, reverse
    return (
        torch.empty_like(input_real, memory_format=torch.contiguous_format),
        torch.empty_like(input_imag, memory_format=torch.contiguous_format),
    )


def _opaque_backward(
    ctx: _AutogradContext,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
    decay_real, decay_imag, states_real, states_imag = ctx.saved_tensors
    gradients = _opaque_recurrence_backward_op(
        decay_real,
        decay_imag,
        states_real,
        states_imag,
        grad_states_real,
        grad_states_imag,
        ctx.reverse,
    )
    return *gradients, None


torch.library.register_autograd(
    "lnet::pac_real2d_recurrence_opaque",
    _opaque_backward,
    setup_context=_setup_context,
)


def pac_triton_recurrence_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    return _pac_real2d_recurrence_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse,
    )


def pac_triton_state_variance_recurrence_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    variance_decay: Tensor,
    variance_input: Tensor,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run a complex scan and its stop-gradient variance transport together."""
    return _pac_real2d_state_variance_recurrence_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        variance_decay,
        variance_input,
        reverse,
    )


def pac_triton_recurrence_opaque_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run the exact scan behind an outer-compiler-safe autograd boundary."""
    return _opaque_recurrence_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse,
    )
