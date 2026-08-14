from __future__ import annotations

# pyright: reportMissingParameterType=false, reportPrivateUsage=false
# ruff: noqa: ANN001, N803
from typing import Final, Literal, Protocol, assert_never

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional
from triton.language.extra import libdevice

from .pac_triton_recurrence_op import _mode_grid, _select_block_modes

Direction = Literal["forward", "backward"]
MomentBackend = Literal["auto", "reference", "triton"]

_EPSILON: Final[float] = 1.0e-8
_LAGS: Final[tuple[int, int]] = (1, 4)
_FORWARD: Final[int] = 1
_BACKWARD: Final[int] = -1
_FUSED_BACKWARD_MAX_BATCH_STEPS: Final[int] = 32_768


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    direction: int
    epsilon: float

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _online_moments_kernel(  # noqa: PLR0915
    states_real,
    states_imag,
    output,
    n_steps: int,
    modes: int,
    epsilon: float,
    reverse: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    input_base = batch * n_steps * modes + mode
    output_base = batch * 5 * modes + mode

    energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr1_real_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr1_imag_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr4_real_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr4_imag_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    current1_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    previous1_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    current4_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    previous4_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)

    history1_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history1_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_imag = tl.zeros((BLOCK_MODES,), tl.float32)

    time_index = 0
    while time_index < n_steps:
        offset = input_base + time_index * modes
        current_real = tl.load(states_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
        current_imag = tl.load(states_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        current_energy = current_real * current_real + current_imag * current_imag
        energy_sum += current_energy

        valid1 = time_index >= 1
        corr1_real = current_real * history1_real + current_imag * history1_imag
        corr1_imag_forward = current_imag * history1_real - current_real * history1_imag
        corr1_imag = -corr1_imag_forward if reverse else corr1_imag_forward
        corr1_real_sum += tl.where(valid1, corr1_real, 0.0)
        corr1_imag_sum += tl.where(valid1, corr1_imag, 0.0)
        current1_energy_sum += tl.where(valid1, current_energy, 0.0)
        previous1_energy_sum += tl.where(
            valid1,
            history1_real * history1_real + history1_imag * history1_imag,
            0.0,
        )

        valid4 = time_index >= 4
        corr4_real = current_real * history4_real + current_imag * history4_imag
        corr4_imag_forward = current_imag * history4_real - current_real * history4_imag
        corr4_imag = -corr4_imag_forward if reverse else corr4_imag_forward
        corr4_real_sum += tl.where(valid4, corr4_real, 0.0)
        corr4_imag_sum += tl.where(valid4, corr4_imag, 0.0)
        current4_energy_sum += tl.where(valid4, current_energy, 0.0)
        previous4_energy_sum += tl.where(
            valid4,
            history4_real * history4_real + history4_imag * history4_imag,
            0.0,
        )

        history4_real = history3_real
        history4_imag = history3_imag
        history3_real = history2_real
        history3_imag = history2_imag
        history2_real = history1_real
        history2_imag = history1_imag
        history1_real = current_real
        history1_imag = current_imag
        time_index += 1

    energy = energy_sum / n_steps
    log_energy = libdevice.log1p(energy)

    count1 = tl.maximum(n_steps - 1, 1)
    corr1_real = corr1_real_sum / count1
    corr1_imag = corr1_imag_sum / count1
    current1_energy = current1_energy_sum / count1
    previous1_energy = previous1_energy_sum / count1
    denominator1 = tl.maximum(tl.sqrt(current1_energy * previous1_energy), epsilon)
    corr1_real = tl.where(n_steps > 1, corr1_real / denominator1, 0.0)
    corr1_imag = tl.where(n_steps > 1, corr1_imag / denominator1, 0.0)

    count4 = tl.maximum(n_steps - 4, 1)
    corr4_real = corr4_real_sum / count4
    corr4_imag = corr4_imag_sum / count4
    current4_energy = current4_energy_sum / count4
    previous4_energy = previous4_energy_sum / count4
    denominator4 = tl.maximum(tl.sqrt(current4_energy * previous4_energy), epsilon)
    corr4_real = tl.where(n_steps > 4, corr4_real / denominator4, 0.0)
    corr4_imag = tl.where(n_steps > 4, corr4_imag / denominator4, 0.0)

    tl.store(output + output_base, log_energy, mask=valid_mode)
    tl.store(output + output_base + modes, corr1_real, mask=valid_mode)
    tl.store(output + output_base + 2 * modes, corr1_imag, mask=valid_mode)
    tl.store(output + output_base + 3 * modes, corr4_real, mask=valid_mode)
    tl.store(output + output_base + 4 * modes, corr4_imag, mask=valid_mode)


@triton.jit
def _online_moments_backward_kernel(  # noqa: PLR0915
    states_real,
    states_imag,
    grad_output,
    grad_states_real,
    grad_states_imag,
    n_steps: int,
    modes: int,
    epsilon: float,
    reverse: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    """Differentiate all five modal moments in one streaming kernel.

    Reversing the time axis leaves energy, real correlations, and their
    normalizers unchanged.  It only negates the imaginary correlations, so
    the backward pass can stay in physical storage order and negate the two
    corresponding output gradients.
    """
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    input_base = batch * n_steps * modes + mode
    output_base = batch * 5 * modes + mode

    energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr1_real_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr1_imag_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr4_real_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    corr4_imag_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    current1_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    previous1_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    current4_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    previous4_energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)

    history1_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history1_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_imag = tl.zeros((BLOCK_MODES,), tl.float32)

    time_index = 0
    while time_index < n_steps:
        offset = input_base + time_index * modes
        current_real = tl.load(states_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
        current_imag = tl.load(states_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        current_energy = current_real * current_real + current_imag * current_imag
        energy_sum += current_energy

        valid1 = time_index >= 1
        corr1_real_sum += tl.where(
            valid1,
            current_real * history1_real + current_imag * history1_imag,
            0.0,
        )
        corr1_imag_sum += tl.where(
            valid1,
            current_imag * history1_real - current_real * history1_imag,
            0.0,
        )
        current1_energy_sum += tl.where(valid1, current_energy, 0.0)
        previous1_energy_sum += tl.where(
            valid1,
            history1_real * history1_real + history1_imag * history1_imag,
            0.0,
        )

        valid4 = time_index >= 4
        corr4_real_sum += tl.where(
            valid4,
            current_real * history4_real + current_imag * history4_imag,
            0.0,
        )
        corr4_imag_sum += tl.where(
            valid4,
            current_imag * history4_real - current_real * history4_imag,
            0.0,
        )
        current4_energy_sum += tl.where(valid4, current_energy, 0.0)
        previous4_energy_sum += tl.where(
            valid4,
            history4_real * history4_real + history4_imag * history4_imag,
            0.0,
        )

        history4_real = history3_real
        history4_imag = history3_imag
        history3_real = history2_real
        history3_imag = history2_imag
        history2_real = history1_real
        history2_imag = history1_imag
        history1_real = current_real
        history1_imag = current_imag
        time_index += 1

    energy = energy_sum / n_steps
    energy_gradient = tl.load(grad_output + output_base, mask=valid_mode, other=0.0).to(tl.float32)
    energy_scale = (2.0 / n_steps) * energy_gradient / (1.0 + energy)
    imaginary_sign = -1.0 if reverse else 1.0

    count1 = tl.maximum(n_steps - 1, 1)
    inverse_count1 = 1.0 / count1
    corr1_real = corr1_real_sum * inverse_count1
    corr1_imag = corr1_imag_sum * inverse_count1
    current1_energy = current1_energy_sum * inverse_count1
    previous1_energy = previous1_energy_sum * inverse_count1
    root1 = tl.sqrt(current1_energy * previous1_energy)
    denominator1 = tl.maximum(root1, epsilon)
    output1_real_gradient = tl.load(
        grad_output + output_base + modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    output1_imag_gradient = imaginary_sign * tl.load(
        grad_output + output_base + 2 * modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    real1_weight = tl.where(n_steps > 1, output1_real_gradient / denominator1, 0.0)
    imag1_weight = tl.where(n_steps > 1, output1_imag_gradient / denominator1, 0.0)
    weighted1 = output1_real_gradient * corr1_real + output1_imag_gradient * corr1_imag
    root1_gradient = tl.where(
        root1 > epsilon,
        -weighted1 / (denominator1 * denominator1),
        0.0,
    )
    safe_root1 = tl.maximum(root1, epsilon)
    current1_energy_gradient = 0.5 * root1_gradient * previous1_energy / safe_root1
    previous1_energy_gradient = 0.5 * root1_gradient * current1_energy / safe_root1

    count4 = tl.maximum(n_steps - 4, 1)
    inverse_count4 = 1.0 / count4
    corr4_real = corr4_real_sum * inverse_count4
    corr4_imag = corr4_imag_sum * inverse_count4
    current4_energy = current4_energy_sum * inverse_count4
    previous4_energy = previous4_energy_sum * inverse_count4
    root4 = tl.sqrt(current4_energy * previous4_energy)
    denominator4 = tl.maximum(root4, epsilon)
    output4_real_gradient = tl.load(
        grad_output + output_base + 3 * modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    output4_imag_gradient = imaginary_sign * tl.load(
        grad_output + output_base + 4 * modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    real4_weight = tl.where(n_steps > 4, output4_real_gradient / denominator4, 0.0)
    imag4_weight = tl.where(n_steps > 4, output4_imag_gradient / denominator4, 0.0)
    weighted4 = output4_real_gradient * corr4_real + output4_imag_gradient * corr4_imag
    root4_gradient = tl.where(
        root4 > epsilon,
        -weighted4 / (denominator4 * denominator4),
        0.0,
    )
    safe_root4 = tl.maximum(root4, epsilon)
    current4_energy_gradient = 0.5 * root4_gradient * previous4_energy / safe_root4
    previous4_energy_gradient = 0.5 * root4_gradient * current4_energy / safe_root4

    time_index = 0
    while time_index < n_steps:
        offset = input_base + time_index * modes
        current_real = tl.load(states_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
        current_imag = tl.load(states_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        grad_real = energy_scale * current_real
        grad_imag = energy_scale * current_imag

        has_previous1 = time_index >= 1
        previous1_offset = offset - modes
        previous1_real = tl.load(
            states_real + previous1_offset,
            mask=valid_mode & has_previous1,
            other=0.0,
        ).to(tl.float32)
        previous1_imag = tl.load(
            states_imag + previous1_offset,
            mask=valid_mode & has_previous1,
            other=0.0,
        ).to(tl.float32)
        current1_real_grad = (
            inverse_count1 * (real1_weight * previous1_real - imag1_weight * previous1_imag)
            + 2.0 * inverse_count1 * current1_energy_gradient * current_real
        )
        current1_imag_grad = (
            inverse_count1 * (real1_weight * previous1_imag + imag1_weight * previous1_real)
            + 2.0 * inverse_count1 * current1_energy_gradient * current_imag
        )
        grad_real += tl.where(has_previous1, current1_real_grad, 0.0)
        grad_imag += tl.where(has_previous1, current1_imag_grad, 0.0)

        has_next1 = time_index < n_steps - 1
        next1_offset = offset + modes
        next1_real = tl.load(
            states_real + next1_offset,
            mask=valid_mode & has_next1,
            other=0.0,
        ).to(tl.float32)
        next1_imag = tl.load(
            states_imag + next1_offset,
            mask=valid_mode & has_next1,
            other=0.0,
        ).to(tl.float32)
        previous1_real_grad = (
            inverse_count1 * (real1_weight * next1_real + imag1_weight * next1_imag)
            + 2.0 * inverse_count1 * previous1_energy_gradient * current_real
        )
        previous1_imag_grad = (
            inverse_count1 * (real1_weight * next1_imag - imag1_weight * next1_real)
            + 2.0 * inverse_count1 * previous1_energy_gradient * current_imag
        )
        grad_real += tl.where(has_next1, previous1_real_grad, 0.0)
        grad_imag += tl.where(has_next1, previous1_imag_grad, 0.0)

        has_previous4 = time_index >= 4
        previous4_offset = offset - 4 * modes
        previous4_real = tl.load(
            states_real + previous4_offset,
            mask=valid_mode & has_previous4,
            other=0.0,
        ).to(tl.float32)
        previous4_imag = tl.load(
            states_imag + previous4_offset,
            mask=valid_mode & has_previous4,
            other=0.0,
        ).to(tl.float32)
        current4_real_grad = (
            inverse_count4 * (real4_weight * previous4_real - imag4_weight * previous4_imag)
            + 2.0 * inverse_count4 * current4_energy_gradient * current_real
        )
        current4_imag_grad = (
            inverse_count4 * (real4_weight * previous4_imag + imag4_weight * previous4_real)
            + 2.0 * inverse_count4 * current4_energy_gradient * current_imag
        )
        grad_real += tl.where(has_previous4, current4_real_grad, 0.0)
        grad_imag += tl.where(has_previous4, current4_imag_grad, 0.0)

        has_next4 = time_index < n_steps - 4
        next4_offset = offset + 4 * modes
        next4_real = tl.load(
            states_real + next4_offset,
            mask=valid_mode & has_next4,
            other=0.0,
        ).to(tl.float32)
        next4_imag = tl.load(
            states_imag + next4_offset,
            mask=valid_mode & has_next4,
            other=0.0,
        ).to(tl.float32)
        previous4_real_grad = (
            inverse_count4 * (real4_weight * next4_real + imag4_weight * next4_imag)
            + 2.0 * inverse_count4 * previous4_energy_gradient * current_real
        )
        previous4_imag_grad = (
            inverse_count4 * (real4_weight * next4_imag - imag4_weight * next4_real)
            + 2.0 * inverse_count4 * previous4_energy_gradient * current_imag
        )
        grad_real += tl.where(has_next4, previous4_real_grad, 0.0)
        grad_imag += tl.where(has_next4, previous4_imag_grad, 0.0)

        tl.store(grad_states_real + offset, grad_real, mask=valid_mode)
        tl.store(grad_states_imag + offset, grad_imag, mask=valid_mode)
        time_index += 1


@torch.library.triton_op("lnet::tight_frame_online_moments", mutates_args={})
def _triton_online_moments(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    direction: int,
    epsilon: float,
) -> Tensor:
    real = states_real.contiguous()
    imag = states_imag.contiguous()
    batch, n_steps, modes = real.shape
    output = torch.empty((batch, 5 * modes), dtype=real.dtype, device=real.device)
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    torch.library.wrap_triton(_online_moments_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        output,
        n_steps,
        modes,
        epsilon,
        reverse=direction == _BACKWARD,
        BLOCK_MODES=block_modes,
    )
    return output


@torch.library.triton_op("lnet::tight_frame_online_moments_training", mutates_args={})
def _triton_online_moments_training(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    direction: int,
    epsilon: float,
) -> Tensor:
    """Forward-identical training op whose registered backward is fused."""
    real = states_real.contiguous()
    imag = states_imag.contiguous()
    batch, n_steps, modes = real.shape
    output = torch.empty((batch, 5 * modes), dtype=real.dtype, device=real.device)
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    torch.library.wrap_triton(_online_moments_kernel)[_mode_grid(batch, modes, block_modes)](
        real,
        imag,
        output,
        n_steps,
        modes,
        epsilon,
        reverse=direction == _BACKWARD,
        BLOCK_MODES=block_modes,
    )
    return output


@torch.library.triton_op("lnet::tight_frame_online_moments_backward", mutates_args={})
def _triton_online_moments_backward(
    states_real: Tensor,
    states_imag: Tensor,
    grad_output: Tensor,
    *,
    direction: int,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    if not states_real.is_cuda:
        return _reference_moments_backward(
            states_real,
            states_imag,
            grad_output,
            direction=direction,
            epsilon=epsilon,
        )
    real = states_real.contiguous()
    imag = states_imag.contiguous()
    output_gradient = grad_output.contiguous()
    grad_real = torch.empty_like(real)
    grad_imag = torch.empty_like(imag)
    batch, n_steps, modes = real.shape
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    torch.library.wrap_triton(_online_moments_backward_kernel)[
        _mode_grid(batch, modes, block_modes)
    ](
        real,
        imag,
        output_gradient,
        grad_real,
        grad_imag,
        n_steps,
        modes,
        epsilon,
        reverse=direction == _BACKWARD,
        BLOCK_MODES=block_modes,
    )
    return grad_real, grad_imag


def reference_online_modal_moments(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    physical_direction: Direction = "forward",
    epsilon: float = _EPSILON,
) -> Tensor:
    """Compute the corrected F/G modal-moment contract with ordinary PyTorch ops."""
    _validate_inputs(states_real, states_imag, epsilon)
    oriented_real = _orient(states_real, physical_direction)
    oriented_imag = _orient(states_imag, physical_direction)
    energy = (oriented_real.square() + oriented_imag.square()).mean(dim=1)
    moments = [torch.log1p(energy)]
    for lag in _LAGS:
        if oriented_real.shape[1] <= lag:
            zeros = oriented_real.new_zeros(oriented_real.shape[0], oriented_real.shape[2])
            moments.extend((zeros, zeros))
            continue
        current_real = oriented_real[:, lag:]
        current_imag = oriented_imag[:, lag:]
        previous_real = oriented_real[:, :-lag]
        previous_imag = oriented_imag[:, :-lag]
        correlation_real = (current_real * previous_real + current_imag * previous_imag).mean(dim=1)
        correlation_imag = (current_imag * previous_real - current_real * previous_imag).mean(dim=1)
        current_energy = (current_real.square() + current_imag.square()).mean(dim=1)
        previous_energy = (previous_real.square() + previous_imag.square()).mean(dim=1)
        denominator = torch.sqrt((current_energy * previous_energy).clamp_min(epsilon * epsilon))
        moments.extend((correlation_real / denominator, correlation_imag / denominator))
    return torch.cat(moments, dim=-1)


def online_modal_moments(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    physical_direction: Direction = "forward",
    backend: MomentBackend = "auto",
    epsilon: float = _EPSILON,
    fused_backward: bool = False,
) -> Tensor:
    """Dispatch exact modal moments to a streaming Triton reduction or the reference path."""
    _validate_inputs(states_real, states_imag, epsilon)
    direction = _direction_code(physical_direction)
    match backend:
        case "reference":
            return reference_online_modal_moments(
                states_real,
                states_imag,
                physical_direction=physical_direction,
                epsilon=epsilon,
            )
        case "triton":
            _require_triton_inputs(states_real)
            operation = (
                _triton_online_moments_training if fused_backward else _triton_online_moments
            )
            return operation(
                states_real,
                states_imag,
                direction=direction,
                epsilon=epsilon,
            )
        case "auto":
            if states_real.is_cuda and states_real.dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                operation = (
                    _triton_online_moments_training if fused_backward else _triton_online_moments
                )
                return operation(
                    states_real,
                    states_imag,
                    direction=direction,
                    epsilon=epsilon,
                )
            return reference_online_modal_moments(
                states_real,
                states_imag,
                physical_direction=physical_direction,
                epsilon=epsilon,
            )
        case unreachable:
            message = f"unknown online moments backend: {unreachable}"
            raise ValueError(message)


def _setup_context(ctx: _AutogradContext, inputs, keyword_only_inputs, output: Tensor) -> None:
    del output
    states_real, states_imag = inputs
    ctx.save_for_backward(states_real, states_imag)
    ctx.direction = int(keyword_only_inputs["direction"])
    ctx.epsilon = float(keyword_only_inputs["epsilon"])


def _backward(ctx: _AutogradContext, grad_output: Tensor) -> tuple[Tensor, Tensor]:
    states_real, states_imag = ctx.saved_tensors
    return _reference_moments_backward(
        states_real,
        states_imag,
        grad_output,
        direction=ctx.direction,
        epsilon=ctx.epsilon,
    )


torch.library.register_autograd(
    "lnet::tight_frame_online_moments",
    _backward,
    setup_context=_setup_context,
)


def _training_backward(ctx: _AutogradContext, grad_output: Tensor) -> tuple[Tensor, Tensor]:
    states_real, states_imag = ctx.saved_tensors
    return _optimized_moments_backward(
        states_real,
        states_imag,
        grad_output,
        direction=ctx.direction,
        epsilon=ctx.epsilon,
    )


torch.library.register_autograd(
    "lnet::tight_frame_online_moments_training",
    _training_backward,
    setup_context=_setup_context,
)


def _reference_moments_backward(
    states_real: Tensor,
    states_imag: Tensor,
    grad_output: Tensor,
    *,
    direction: int,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    oriented_real = states_real if direction == _FORWARD else torch.flip(states_real, (1,))
    oriented_imag = states_imag if direction == _FORWARD else torch.flip(states_imag, (1,))
    _, n_steps, modes = oriented_real.shape
    energy = (oriented_real.square() + oriented_imag.square()).mean(dim=1)
    energy_scale = (2.0 / n_steps) * grad_output[:, :modes] / (1.0 + energy)
    grad_real = energy_scale.unsqueeze(1) * oriented_real
    grad_imag = energy_scale.unsqueeze(1) * oriented_imag

    output_offset = modes
    for lag in _LAGS:
        grad_correlation_real = grad_output[:, output_offset : output_offset + modes]
        output_offset += modes
        grad_correlation_imag = grad_output[:, output_offset : output_offset + modes]
        output_offset += modes
        if n_steps <= lag:
            continue

        current_real = oriented_real[:, lag:]
        current_imag = oriented_imag[:, lag:]
        previous_real = oriented_real[:, :-lag]
        previous_imag = oriented_imag[:, :-lag]
        overlap = n_steps - lag
        inverse_overlap = 1.0 / overlap

        correlation_real = (current_real * previous_real + current_imag * previous_imag).mean(dim=1)
        correlation_imag = (current_imag * previous_real - current_real * previous_imag).mean(dim=1)
        current_energy = (current_real.square() + current_imag.square()).mean(dim=1)
        previous_energy = (previous_real.square() + previous_imag.square()).mean(dim=1)
        root = torch.sqrt(current_energy * previous_energy)
        denominator = root.clamp_min(epsilon)

        real_weight = grad_correlation_real / denominator
        imag_weight = grad_correlation_imag / denominator
        current_real_grad = inverse_overlap * (
            real_weight.unsqueeze(1) * previous_real - imag_weight.unsqueeze(1) * previous_imag
        )
        current_imag_grad = inverse_overlap * (
            real_weight.unsqueeze(1) * previous_imag + imag_weight.unsqueeze(1) * previous_real
        )
        previous_real_grad = inverse_overlap * (
            real_weight.unsqueeze(1) * current_real + imag_weight.unsqueeze(1) * current_imag
        )
        previous_imag_grad = inverse_overlap * (
            real_weight.unsqueeze(1) * current_imag - imag_weight.unsqueeze(1) * current_real
        )

        weighted_correlation = (
            grad_correlation_real * correlation_real + grad_correlation_imag * correlation_imag
        )
        active_denominator = root > epsilon
        safe_root = root.clamp_min(epsilon)
        root_gradient = torch.where(
            active_denominator,
            -weighted_correlation / denominator.square(),
            torch.zeros_like(weighted_correlation),
        )
        current_energy_gradient = 0.5 * root_gradient * previous_energy / safe_root
        previous_energy_gradient = 0.5 * root_gradient * current_energy / safe_root
        current_real_grad = current_real_grad + (
            2.0 * inverse_overlap * current_energy_gradient.unsqueeze(1) * current_real
        )
        current_imag_grad = current_imag_grad + (
            2.0 * inverse_overlap * current_energy_gradient.unsqueeze(1) * current_imag
        )
        previous_real_grad = previous_real_grad + (
            2.0 * inverse_overlap * previous_energy_gradient.unsqueeze(1) * previous_real
        )
        previous_imag_grad = previous_imag_grad + (
            2.0 * inverse_overlap * previous_energy_gradient.unsqueeze(1) * previous_imag
        )

        grad_real = grad_real + functional.pad(current_real_grad, (0, 0, lag, 0))
        grad_real = grad_real + functional.pad(previous_real_grad, (0, 0, 0, lag))
        grad_imag = grad_imag + functional.pad(current_imag_grad, (0, 0, lag, 0))
        grad_imag = grad_imag + functional.pad(previous_imag_grad, (0, 0, 0, lag))

    if direction == _BACKWARD:
        return torch.flip(grad_real, (1,)), torch.flip(grad_imag, (1,))
    return grad_real, grad_imag


_compiled_reference_moments_backward = torch.compile(
    _reference_moments_backward,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)


def _optimized_moments_backward(
    states_real: Tensor,
    states_imag: Tensor,
    grad_output: Tensor,
    *,
    direction: int,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Select the fastest exact backward family by measured RTX 4090 workload."""
    if not states_real.is_cuda or states_real.dtype != torch.float32:
        return _reference_moments_backward(
            states_real,
            states_imag,
            grad_output,
            direction=direction,
            epsilon=epsilon,
        )
    batch_steps = states_real.shape[0] * states_real.shape[1]
    if batch_steps <= _FUSED_BACKWARD_MAX_BATCH_STEPS:
        return _triton_online_moments_backward(
            states_real,
            states_imag,
            grad_output,
            direction=direction,
            epsilon=epsilon,
        )
    # PyTorch 2.11 Inductor can incorrectly mix-fuse adjacent reductions for
    # large odd time dimensions (observed at 783 and 999), then assert that
    # the n and n-1 reduction extents are equal.  Execute the exact function
    # that the compiled wrapper wraps for these shapes.  This changes only the
    # optimization backend; FP32 operations and the training contract remain
    # the reference implementation.
    if states_real.shape[1] % 2 == 1:
        return _reference_moments_backward(
            states_real,
            states_imag,
            grad_output,
            direction=direction,
            epsilon=epsilon,
        )
    return _compiled_reference_moments_backward(
        states_real,
        states_imag,
        grad_output,
        direction=direction,
        epsilon=epsilon,
    )


def _validate_inputs(states_real: Tensor, states_imag: Tensor, epsilon: float) -> None:
    if states_real.ndim != 3 or states_imag.ndim != 3:
        message = "modal states must have shape [batch, time, modes]"
        raise ValueError(message)
    if states_real.shape != states_imag.shape:
        message = "real and imaginary modal states must have the same shape"
        raise ValueError(message)
    if states_real.device != states_imag.device:
        message = "real and imaginary modal states must be on the same device"
        raise ValueError(message)
    if states_real.dtype != states_imag.dtype:
        message = "real and imaginary modal states must have the same dtype"
        raise ValueError(message)
    if not states_real.is_floating_point():
        message = "modal states must use a floating-point dtype"
        raise TypeError(message)
    if states_real.shape[1] == 0 or states_real.shape[2] == 0:
        message = "time and mode dimensions must be non-zero"
        raise ValueError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)


def _require_triton_inputs(states_real: Tensor) -> None:
    if not states_real.is_cuda:
        message = "the Triton online-moments backend requires CUDA tensors"
        raise ValueError(message)
    if states_real.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        message = "the Triton online-moments backend supports fp16, bf16, and fp32"
        raise TypeError(message)


def _direction_code(direction: Direction) -> int:
    match direction:
        case "forward":
            return _FORWARD
        case "backward":
            return _BACKWARD
        case unreachable:
            assert_never(unreachable)


def _orient(inputs: Tensor, direction: Direction) -> Tensor:
    return inputs if direction == "forward" else torch.flip(inputs, (1,))
