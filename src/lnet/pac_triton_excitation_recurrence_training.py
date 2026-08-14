from __future__ import annotations

# pyright: reportMissingParameterType=false, reportPrivateUsage=false
# ruff: noqa: ANN001, FBT001, FBT003, N803, PLR0915
from typing import Literal, Protocol, assert_never, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from .pac_real2d_math import discrete_pole_real2d
from .pac_triton_recurrence_moments_training import (
    _fused_backward_op,
    _materialize_fp32,
    fused_recurrence_moments_training,
)
from .pac_triton_recurrence_op import _mode_grid, _select_block_modes

Direction = Literal["forward", "backward"]
_FORWARD = 1
_BACKWARD = -1
_DEFAULT_EPSILON = 1.0e-8


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    recurrence_reverse: bool
    moment_direction: int
    epsilon: float
    damping_min: float
    damping_max: float

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _excitation_recurrence_moments_kernel(
    excitation_real,
    excitation_imag,
    raw_decay,
    raw_frequency,
    states_real,
    states_imag,
    moment_output,
    n_steps: int,
    modes: int,
    damping_min: float,
    damping_max: float,
    epsilon: float,
    reverse: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    """Produce pole/gamma/drive in registers, then run recurrence+moments."""
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    base = batch * n_steps * modes + mode

    raw_damping = tl.load(raw_decay + mode, mask=valid_mode, other=0.0).to(tl.float32)
    raw_phase = tl.load(raw_frequency + mode, mask=valid_mode, other=0.0).to(tl.float32)
    sigmoid_decay = _materialize_fp32(1.0 / (1.0 + tl.exp(-raw_damping)))
    damping_scale = _materialize_fp32((damping_max - damping_min) * sigmoid_decay)
    damping = _materialize_fp32(damping_min + damping_scale)
    frequency = _materialize_fp32(3.141592653589793 * libdevice.tanh(raw_phase))
    scaled_decay = _materialize_fp32(tl.exp(-damping))
    phase_cosine = _materialize_fp32(tl.cos(frequency))
    phase_sine = _materialize_fp32(tl.sin(frequency))
    decay_real = _materialize_fp32(scaled_decay * phase_cosine)
    decay_imag = _materialize_fp32(scaled_decay * phase_sine)
    shifted_real = _materialize_fp32(decay_real - 1.0)
    pole_real = _materialize_fp32(-damping)
    pole_real_square = _materialize_fp32(pole_real * pole_real)
    pole_imag_square = _materialize_fp32(frequency * frequency)
    denominator = _materialize_fp32(pole_real_square + pole_imag_square)
    gamma_real_first = _materialize_fp32(shifted_real * pole_real)
    gamma_real_second = _materialize_fp32(decay_imag * frequency)
    gamma_real = _materialize_fp32((gamma_real_first + gamma_real_second) / denominator)
    gamma_imag_first = _materialize_fp32(decay_imag * pole_real)
    gamma_imag_second = _materialize_fp32(shifted_real * frequency)
    gamma_imag = _materialize_fp32((gamma_imag_first - gamma_imag_second) / denominator)
    small = tl.sqrt(pole_real_square + pole_imag_square) < 1.0e-6
    gamma_real = tl.where(small, 1.0, gamma_real)
    gamma_imag = tl.where(small, 0.0, gamma_imag)

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
    step = 0
    while step < n_steps:
        time_index = n_steps - 1 - step if reverse else step
        offset = base + time_index * modes
        excitation_r = tl.load(excitation_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
        excitation_i = tl.load(excitation_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        drive_real_first = _materialize_fp32(gamma_real * excitation_r)
        drive_real_second = _materialize_fp32(gamma_imag * excitation_i)
        drive_real = _materialize_fp32(drive_real_first - drive_real_second)
        drive_imag_first = _materialize_fp32(gamma_real * excitation_i)
        drive_imag_second = _materialize_fp32(gamma_imag * excitation_r)
        drive_imag = _materialize_fp32(drive_imag_first + drive_imag_second)

        previous_state_real = state_real
        previous_state_imag = state_imag
        state_real = (
            decay_real * previous_state_real - decay_imag * previous_state_imag + drive_real
        )
        state_imag = (
            decay_imag * previous_state_real + decay_real * previous_state_imag + drive_imag
        )
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


def reference_excitation_control(
    excitation_real: Tensor,
    excitation_imag: Tensor,
    raw_decay: Tensor,
    raw_frequency: Tensor,
    damping_min: float,
    damping_max: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The exact eager producer whose VJP is replayed by the custom backward."""
    damping_logits = raw_decay.view(1, 1, -1).expand_as(excitation_real)
    damping = damping_min + (damping_max - damping_min) * torch.sigmoid(damping_logits)
    frequency = torch.pi * torch.tanh(raw_frequency).view(1, 1, -1)
    decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(damping, frequency, 1.0)
    input_real = gamma_real * excitation_real - gamma_imag * excitation_imag
    input_imag = gamma_real * excitation_imag + gamma_imag * excitation_real
    return decay_real, decay_imag, input_real, input_imag


@triton_op("lnet::pac_excitation_recurrence_moments_training", mutates_args={})
def _fused_op(
    excitation_real: Tensor,
    excitation_imag: Tensor,
    raw_decay: Tensor,
    raw_frequency: Tensor,
    damping_min: float,
    damping_max: float,
    recurrence_reverse: bool,
    moment_direction: int,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if not excitation_real.is_cuda:
        decay_real, decay_imag, input_real, input_imag = reference_excitation_control(
            excitation_real,
            excitation_imag,
            raw_decay,
            raw_frequency,
            damping_min,
            damping_max,
        )
        return fused_recurrence_moments_training(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            recurrence_reverse=recurrence_reverse,
            moment_direction=_direction_name(moment_direction),
            epsilon=epsilon,
            use_two_pass_reverse=False,
        )
    real = excitation_real.contiguous()
    imag = excitation_imag.contiguous()
    raw_d = raw_decay.contiguous()
    raw_f = raw_frequency.contiguous()
    states_real = torch.empty_like(real)
    states_imag = torch.empty_like(imag)
    batch, n_steps, modes = real.shape
    moments = torch.empty((batch, 5 * modes), dtype=real.dtype, device=real.device)
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    wrap_triton(_excitation_recurrence_moments_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        raw_d,
        raw_f,
        states_real,
        states_imag,
        moments,
        n_steps,
        modes,
        damping_min,
        damping_max,
        epsilon,
        reverse=recurrence_reverse,
        BLOCK_MODES=block_modes,
        num_warps=1 if n_steps > 128 else 4,
    )
    return states_real, states_imag, moments


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, float, float, bool, int, float],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    (
        excitation_real,
        excitation_imag,
        raw_decay,
        raw_frequency,
        dmin,
        dmax,
        reverse,
        direction,
        eps,
    ) = inputs
    states_real, states_imag, _moments = output
    ctx.damping_min = dmin
    ctx.damping_max = dmax
    ctx.recurrence_reverse = reverse
    ctx.moment_direction = direction
    ctx.epsilon = eps
    ctx.save_for_backward(
        excitation_real,
        excitation_imag,
        raw_decay,
        raw_frequency,
        states_real,
        states_imag,
    )


def _backward(
    ctx: _AutogradContext,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
    grad_moments: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, None, None, None, None, None]:
    excitation_real, excitation_imag, raw_decay, raw_frequency, states_real, states_imag = (
        ctx.saved_tensors
    )
    with torch.enable_grad():
        active_excitation_real = excitation_real.detach().requires_grad_(True)
        active_excitation_imag = excitation_imag.detach().requires_grad_(True)
        active_raw_decay = raw_decay.detach().requires_grad_(True)
        active_raw_frequency = raw_frequency.detach().requires_grad_(True)
        producer_outputs = reference_excitation_control(
            active_excitation_real,
            active_excitation_imag,
            active_raw_decay,
            active_raw_frequency,
            ctx.damping_min,
            ctx.damping_max,
        )
    # The producer is replayed once.  The recurrence adjoint consumes its
    # decay tensors without building a higher-order graph, then the original
    # eager producer graph applies the four returned VJPs in the same order.
    with torch.no_grad():
        grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag = _fused_backward_op(
            producer_outputs[0],
            producer_outputs[1],
            states_real,
            states_imag,
            grad_states_real,
            grad_states_imag,
            grad_moments,
            ctx.recurrence_reverse,
            ctx.moment_direction,
            ctx.epsilon,
        )
    with torch.enable_grad():
        gradients = cast(
            "tuple[Tensor, Tensor, Tensor, Tensor]",
            torch.autograd.grad(
                producer_outputs,
                (
                    active_excitation_real,
                    active_excitation_imag,
                    active_raw_decay,
                    active_raw_frequency,
                ),
                grad_outputs=(
                    grad_decay_real,
                    grad_decay_imag,
                    grad_input_real,
                    grad_input_imag,
                ),
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            ),
        )
    return *gradients, None, None, None, None, None


torch.library.register_autograd(
    "lnet::pac_excitation_recurrence_moments_training",
    _backward,
    setup_context=_setup_context,
)


def supports_fused_excitation_recurrence(
    excitation_real: Tensor,
    excitation_imag: Tensor,
    raw_decay: Tensor,
    raw_frequency: Tensor,
) -> bool:
    """Guard the prototype to its canonical FP32 static-control domain."""
    return (
        excitation_real.ndim == 3
        and excitation_imag.shape == excitation_real.shape
        and excitation_real.dtype == torch.float32
        and excitation_imag.dtype == torch.float32
        and excitation_real.device == excitation_imag.device
        and raw_decay.ndim == 1
        and raw_frequency.shape == raw_decay.shape
        and raw_decay.shape[0] == excitation_real.shape[-1]
        and raw_decay.dtype == torch.float32
        and raw_frequency.dtype == torch.float32
        and raw_decay.device == excitation_real.device
        and raw_frequency.device == excitation_real.device
    )


def fused_excitation_recurrence_moments_training(
    excitation_real: Tensor,
    excitation_imag: Tensor,
    raw_decay: Tensor,
    raw_frequency: Tensor,
    *,
    damping_min: float,
    damping_max: float,
    recurrence_reverse: bool = False,
    moment_direction: Direction = "forward",
    epsilon: float = _DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse the static pole/gamma/complex-drive producer into recurrence forward."""
    if not supports_fused_excitation_recurrence(
        excitation_real, excitation_imag, raw_decay, raw_frequency
    ):
        decay_real, decay_imag, input_real, input_imag = reference_excitation_control(
            excitation_real,
            excitation_imag,
            raw_decay,
            raw_frequency,
            damping_min,
            damping_max,
        )
        return fused_recurrence_moments_training(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            recurrence_reverse=recurrence_reverse,
            moment_direction=moment_direction,
            epsilon=epsilon,
            use_two_pass_reverse=False,
        )
    if damping_min <= 0.0 or damping_max <= damping_min or epsilon <= 0.0:
        message = "damping bounds and epsilon must be positive and ordered"
        raise ValueError(message)
    return _fused_op(
        excitation_real,
        excitation_imag,
        raw_decay,
        raw_frequency,
        damping_min,
        damping_max,
        recurrence_reverse,
        _direction_code(moment_direction),
        epsilon,
    )


def _direction_code(direction: Direction) -> int:
    match direction:
        case "forward":
            return _FORWARD
        case "backward":
            return _BACKWARD
        case unreachable:
            assert_never(unreachable)


def _direction_name(value: int) -> Direction:
    return "forward" if value == _FORWARD else "backward"


__all__ = [
    "fused_excitation_recurrence_moments_training",
    "reference_excitation_control",
    "supports_fused_excitation_recurrence",
]
