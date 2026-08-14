from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001, C901, FBT001, N803, PLR0912, PLR0915
import os
from dataclasses import dataclass
from typing import Literal, Protocol, assert_never

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from .pac_triton_online_moments import online_modal_moments
from .pac_triton_recurrence_moments import (
    _recurrence_moments_kernel,
    recurrence_moments_inference,
)
from .pac_triton_recurrence_op import (
    _is_mode_static_expanded,
    _mode_grid,
    _select_block_modes,
    pac_triton_recurrence_op,
)

Direction = Literal["forward", "backward"]

_FORWARD = 1
_BACKWARD = -1
_DEFAULT_EPSILON = 1.0e-8


def _select_backward_num_warps(n_steps: int) -> int:
    override = os.environ.get("LNET_PAC_BACKWARD_WARPS")
    if override is None:
        return 4 if n_steps <= 128 else 1
    try:
        value = int(override)
    except ValueError:
        return 4 if n_steps <= 128 else 1
    return value if value in {1, 2, 4, 8} else (4 if n_steps <= 128 else 1)


def _use_split_backward() -> bool:
    return os.environ.get("LNET_PAC_SPLIT_BACKWARD", "").lower() in {"1", "true", "yes"}


def _select_split_block_modes(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value in {1, 2, 4, 8, 16, 32} else default


def _use_split_raw_statistics() -> bool:
    return os.environ.get("LNET_PAC_SPLIT_RAW_STATISTICS", "").lower() in {
        "1",
        "true",
        "yes",
    }


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    recurrence_reverse: bool
    moment_direction: int
    epsilon: float

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class _PackedAutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    recurrence_reverse: bool
    moment_direction: int
    epsilon: float

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _materialize_fp32(value):  # noqa: ANN202
    """Prevent LLVM from reassociating across the eliminated tensor boundary."""
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=r,r",
        [value],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reverse_recurrence_physical_moments_kernel(
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
    moments_reverse: tl.constexpr,
    packed_output: tl.constexpr,
    static_decay: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    """Run reverse recurrence, then reduce moments in physical time order."""
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
        fixed_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(
            tl.float32
        )
        fixed_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(
            tl.float32
        )
    recurrence_step = 0
    while recurrence_step < n_steps:
        time_index = n_steps - 1 - recurrence_step
        offset = base + time_index * modes
        if static_decay:
            ar = fixed_decay_real
            ai = fixed_decay_imag
        else:
            ar = tl.load(decay_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
            ai = tl.load(decay_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        drive_real = tl.load(input_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
        drive_imag = tl.load(input_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        previous_real = state_real
        previous_imag = state_imag
        state_real = ar * previous_real - ai * previous_imag + drive_real
        state_imag = ai * previous_real + ar * previous_imag + drive_imag
        if packed_output:
            packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
            tl.store(states_real + packed_offset, state_real, mask=valid_mode)
            tl.store(states_real + packed_offset + modes, state_imag, mask=valid_mode)
        else:
            tl.store(states_real + offset, state_real, mask=valid_mode)
            tl.store(states_imag + offset, state_imag, mask=valid_mode)
        recurrence_step += 1

    # The state stores and loads belong to the same program.  Keep an explicit
    # compiler barrier so the second pass cannot be folded into reverse order.
    tl.debug_barrier()

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
        if packed_output:
            packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
            current_real = tl.load(
                states_real + packed_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            current_imag = tl.load(
                states_real + packed_offset + modes,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        else:
            offset = base + time_index * modes
            current_real = tl.load(states_real + offset, mask=valid_mode, other=0.0).to(
                tl.float32
            )
            current_imag = tl.load(states_imag + offset, mask=valid_mode, other=0.0).to(
                tl.float32
            )
        current_energy = current_real * current_real + current_imag * current_imag
        energy_sum += current_energy

        valid1 = time_index >= 1
        corr1_real = current_real * history1_real + current_imag * history1_imag
        corr1_imag_forward = current_imag * history1_real - current_real * history1_imag
        corr1_imag = -corr1_imag_forward if moments_reverse else corr1_imag_forward
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
        corr4_imag = -corr4_imag_forward if moments_reverse else corr4_imag_forward
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

    moment_base = batch * 5 * modes + mode
    tl.store(moment_output + moment_base, libdevice.log1p(energy), mask=valid_mode)
    tl.store(moment_output + moment_base + modes, corr1_real, mask=valid_mode)
    tl.store(moment_output + moment_base + 2 * modes, corr1_imag, mask=valid_mode)
    tl.store(moment_output + moment_base + 3 * modes, corr4_real, mask=valid_mode)
    tl.store(moment_output + moment_base + 4 * modes, corr4_imag, mask=valid_mode)


@triton.jit
def _fused_recurrence_moments_backward_kernel(
    decay_real,
    decay_imag,
    states_real,
    states_imag,
    direct_grad_states_real,
    direct_grad_states_imag,
    grad_moments,
    grad_decay_real,
    grad_decay_imag,
    grad_input_real,
    grad_input_imag,
    n_steps: int,
    modes: int,
    epsilon: float,
    recurrence_reverse: tl.constexpr,
    moments_reverse: tl.constexpr,
    packed_states: tl.constexpr,
    static_decay: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    """Fuse modal-moment VJP production into the recurrence adjoint.

    The statistics pass is deliberately in physical storage order.  The
    second pass alone follows recurrence-adjoint order.  Keeping separate
    counters also prevents the non-terminating loop that affected the first
    experimental implementation.
    """
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    base = batch * n_steps * modes + mode
    moment_base = batch * 5 * modes + mode
    fixed_decay_real = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    if static_decay:
        fixed_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(
            tl.float32
        )
        fixed_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(
            tl.float32
        )

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

    # Pass 1: moment statistics always use physical time 0 -> N-1.
    statistics_time_index = 0
    while statistics_time_index < n_steps:
        statistics_offset = base + statistics_time_index * modes
        if packed_states:
            packed_statistics_offset = (
                (batch * n_steps + statistics_time_index) * 2 * modes + mode
            )
            current_real = tl.load(
                states_real + packed_statistics_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            current_imag = tl.load(
                states_real + packed_statistics_offset + modes,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        else:
            current_real = tl.load(
                states_real + statistics_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            current_imag = tl.load(
                states_imag + statistics_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        current_energy = current_real * current_real + current_imag * current_imag
        energy_sum += current_energy

        valid1 = statistics_time_index >= 1
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

        valid4 = statistics_time_index >= 4
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
        statistics_time_index += 1

    energy = energy_sum / n_steps
    energy_output_gradient = tl.load(
        grad_moments + moment_base,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    energy_scale = (2.0 / n_steps) * energy_output_gradient / (1.0 + energy)
    imaginary_sign = -1.0 if moments_reverse else 1.0

    count1 = tl.maximum(n_steps - 1, 1)
    inverse_count1 = 1.0 / count1
    corr1_real = corr1_real_sum * inverse_count1
    corr1_imag = corr1_imag_sum * inverse_count1
    current1_energy = current1_energy_sum * inverse_count1
    previous1_energy = previous1_energy_sum * inverse_count1
    root1 = tl.sqrt(current1_energy * previous1_energy)
    denominator1 = tl.maximum(root1, epsilon)
    output1_real_gradient = tl.load(
        grad_moments + moment_base + modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    output1_imag_gradient = imaginary_sign * tl.load(
        grad_moments + moment_base + 2 * modes,
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
        grad_moments + moment_base + 3 * modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    output4_imag_gradient = imaginary_sign * tl.load(
        grad_moments + moment_base + 4 * modes,
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

    # Pass 2: produce each state's moment VJP just before consuming it in
    # recurrence-adjoint order.  No [B,N,M] moment-gradient tensor is written.
    lambda_real = tl.zeros((BLOCK_MODES,), tl.float32)
    lambda_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    adjoint_step = 0
    while adjoint_step < n_steps:
        time_index = adjoint_step if recurrence_reverse else n_steps - 1 - adjoint_step
        offset = base + time_index * modes
        if packed_states:
            packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
            current_real = tl.load(
                states_real + packed_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            current_imag = tl.load(
                states_real + packed_offset + modes,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        else:
            current_real = tl.load(states_real + offset, mask=valid_mode, other=0.0).to(
                tl.float32
            )
            current_imag = tl.load(states_imag + offset, mask=valid_mode, other=0.0).to(
                tl.float32
            )
        moment_grad_real = energy_scale * current_real
        moment_grad_imag = energy_scale * current_imag

        has_previous1 = time_index >= 1
        if packed_states:
            previous1_offset = packed_offset - 2 * modes
            previous1_real = tl.load(
                states_real + previous1_offset,
                mask=valid_mode & has_previous1,
                other=0.0,
            ).to(tl.float32)
            previous1_imag = tl.load(
                states_real + previous1_offset + modes,
                mask=valid_mode & has_previous1,
                other=0.0,
            ).to(tl.float32)
        else:
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
        moment_grad_real += tl.where(has_previous1, current1_real_grad, 0.0)
        moment_grad_imag += tl.where(has_previous1, current1_imag_grad, 0.0)

        has_next1 = time_index < n_steps - 1
        if packed_states:
            next1_offset = packed_offset + 2 * modes
            next1_real = tl.load(
                states_real + next1_offset,
                mask=valid_mode & has_next1,
                other=0.0,
            ).to(tl.float32)
            next1_imag = tl.load(
                states_real + next1_offset + modes,
                mask=valid_mode & has_next1,
                other=0.0,
            ).to(tl.float32)
        else:
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
        moment_grad_real += tl.where(has_next1, previous1_real_grad, 0.0)
        moment_grad_imag += tl.where(has_next1, previous1_imag_grad, 0.0)

        has_previous4 = time_index >= 4
        if packed_states:
            previous4_offset = packed_offset - 8 * modes
            previous4_real = tl.load(
                states_real + previous4_offset,
                mask=valid_mode & has_previous4,
                other=0.0,
            ).to(tl.float32)
            previous4_imag = tl.load(
                states_real + previous4_offset + modes,
                mask=valid_mode & has_previous4,
                other=0.0,
            ).to(tl.float32)
        else:
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
        moment_grad_real += tl.where(has_previous4, current4_real_grad, 0.0)
        moment_grad_imag += tl.where(has_previous4, current4_imag_grad, 0.0)

        has_next4 = time_index < n_steps - 4
        if packed_states:
            next4_offset = packed_offset + 8 * modes
            next4_real = tl.load(
                states_real + next4_offset,
                mask=valid_mode & has_next4,
                other=0.0,
            ).to(tl.float32)
            next4_imag = tl.load(
                states_real + next4_offset + modes,
                mask=valid_mode & has_next4,
                other=0.0,
            ).to(tl.float32)
        else:
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
        moment_grad_real += tl.where(has_next4, previous4_real_grad, 0.0)
        moment_grad_imag += tl.where(has_next4, previous4_imag_grad, 0.0)

        # Match the unfused autograd boundary: moments backward and the direct
        # state consumer are first added into one FP32 state-gradient tensor,
        # then recurrence backward adds that rounded value to lambda.  Adding
        # each contribution to lambda separately changes FP32 association and
        # is enough to diverge over a long near-unit-pole training trajectory.
        if packed_states:
            direct_grad_real = tl.load(
                direct_grad_states_real + packed_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            direct_grad_imag = tl.load(
                direct_grad_states_real + packed_offset + modes,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        else:
            direct_grad_real = tl.load(
                direct_grad_states_real + offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            direct_grad_imag = tl.load(
                direct_grad_states_imag + offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        combined_state_grad_real = _materialize_fp32(direct_grad_real + moment_grad_real)
        combined_state_grad_imag = _materialize_fp32(direct_grad_imag + moment_grad_imag)
        lambda_real += combined_state_grad_real
        lambda_imag += combined_state_grad_imag

        previous_index = time_index + 1 if recurrence_reverse else time_index - 1
        has_recurrence_previous = time_index < n_steps - 1 if recurrence_reverse else time_index > 0
        if packed_states:
            recurrence_previous_offset = (
                (batch * n_steps + previous_index) * 2 * modes + mode
            )
            recurrence_previous_real = tl.load(
                states_real + recurrence_previous_offset,
                mask=valid_mode & has_recurrence_previous,
                other=0.0,
            ).to(tl.float32)
            recurrence_previous_imag = tl.load(
                states_real + recurrence_previous_offset + modes,
                mask=valid_mode & has_recurrence_previous,
                other=0.0,
            ).to(tl.float32)
        else:
            recurrence_previous_offset = base + previous_index * modes
            recurrence_previous_real = tl.load(
                states_real + recurrence_previous_offset,
                mask=valid_mode & has_recurrence_previous,
                other=0.0,
            ).to(tl.float32)
            recurrence_previous_imag = tl.load(
                states_imag + recurrence_previous_offset,
                mask=valid_mode & has_recurrence_previous,
                other=0.0,
            ).to(tl.float32)
        if static_decay:
            ar = fixed_decay_real
            ai = fixed_decay_imag
        else:
            ar = tl.load(decay_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
            ai = tl.load(decay_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        tl.store(grad_input_real + offset, lambda_real, mask=valid_mode)
        tl.store(grad_input_imag + offset, lambda_imag, mask=valid_mode)
        tl.store(
            grad_decay_real + offset,
            lambda_real * recurrence_previous_real + lambda_imag * recurrence_previous_imag,
            mask=valid_mode,
        )
        tl.store(
            grad_decay_imag + offset,
            -lambda_real * recurrence_previous_imag + lambda_imag * recurrence_previous_real,
            mask=valid_mode,
        )
        next_lambda_real = ar * lambda_real + ai * lambda_imag
        next_lambda_imag = -ai * lambda_real + ar * lambda_imag
        lambda_real = next_lambda_real
        lambda_imag = next_lambda_imag
        adjoint_step += 1


@triton.jit
def _split_moment_weights_kernel(
    states_real,
    states_imag,
    grad_moments,
    moment_weights,
    n_steps: int,
    modes: int,
    epsilon: float,
    moments_reverse: tl.constexpr,
    packed_states: tl.constexpr,
    raw_statistics: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    base = batch * n_steps * modes + mode
    moment_base = batch * 5 * modes + mode

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
        offset = base + time_index * modes
        if packed_states:
            packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
            current_real = tl.load(
                states_real + packed_offset, mask=valid_mode, other=0.0
            ).to(tl.float32)
            current_imag = tl.load(
                states_real + packed_offset + modes, mask=valid_mode, other=0.0
            ).to(tl.float32)
        else:
            current_real = tl.load(
                states_real + offset, mask=valid_mode, other=0.0
            ).to(tl.float32)
            current_imag = tl.load(
                states_imag + offset, mask=valid_mode, other=0.0
            ).to(tl.float32)
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
    energy_output_gradient = tl.load(
        grad_moments + moment_base, mask=valid_mode, other=0.0
    ).to(tl.float32)
    energy_scale = (2.0 / n_steps) * energy_output_gradient / (1.0 + energy)
    imaginary_sign = -1.0 if moments_reverse else 1.0

    inverse_count1 = 1.0 / tl.maximum(n_steps - 1, 1)
    corr1_real = corr1_real_sum * inverse_count1
    corr1_imag = corr1_imag_sum * inverse_count1
    current1_energy = current1_energy_sum * inverse_count1
    previous1_energy = previous1_energy_sum * inverse_count1
    root1 = tl.sqrt(current1_energy * previous1_energy)
    denominator1 = tl.maximum(root1, epsilon)
    output1_real_gradient = tl.load(
        grad_moments + moment_base + modes, mask=valid_mode, other=0.0
    ).to(tl.float32)
    output1_imag_gradient = imaginary_sign * tl.load(
        grad_moments + moment_base + 2 * modes, mask=valid_mode, other=0.0
    ).to(tl.float32)
    real1_weight = tl.where(n_steps > 1, output1_real_gradient / denominator1, 0.0)
    imag1_weight = tl.where(n_steps > 1, output1_imag_gradient / denominator1, 0.0)
    weighted1 = output1_real_gradient * corr1_real + output1_imag_gradient * corr1_imag
    root1_gradient = tl.where(
        root1 > epsilon, -weighted1 / (denominator1 * denominator1), 0.0
    )
    safe_root1 = tl.maximum(root1, epsilon)
    current1_energy_gradient = 0.5 * root1_gradient * previous1_energy / safe_root1
    previous1_energy_gradient = 0.5 * root1_gradient * current1_energy / safe_root1

    inverse_count4 = 1.0 / tl.maximum(n_steps - 4, 1)
    corr4_real = corr4_real_sum * inverse_count4
    corr4_imag = corr4_imag_sum * inverse_count4
    current4_energy = current4_energy_sum * inverse_count4
    previous4_energy = previous4_energy_sum * inverse_count4
    root4 = tl.sqrt(current4_energy * previous4_energy)
    denominator4 = tl.maximum(root4, epsilon)
    output4_real_gradient = tl.load(
        grad_moments + moment_base + 3 * modes, mask=valid_mode, other=0.0
    ).to(tl.float32)
    output4_imag_gradient = imaginary_sign * tl.load(
        grad_moments + moment_base + 4 * modes, mask=valid_mode, other=0.0
    ).to(tl.float32)
    real4_weight = tl.where(n_steps > 4, output4_real_gradient / denominator4, 0.0)
    imag4_weight = tl.where(n_steps > 4, output4_imag_gradient / denominator4, 0.0)
    weighted4 = output4_real_gradient * corr4_real + output4_imag_gradient * corr4_imag
    root4_gradient = tl.where(
        root4 > epsilon, -weighted4 / (denominator4 * denominator4), 0.0
    )
    safe_root4 = tl.maximum(root4, epsilon)
    current4_energy_gradient = 0.5 * root4_gradient * previous4_energy / safe_root4
    previous4_energy_gradient = 0.5 * root4_gradient * current4_energy / safe_root4

    weight_base = batch * 9 * modes + mode
    if raw_statistics:
        tl.store(moment_weights + weight_base, energy_sum, mask=valid_mode)
        tl.store(moment_weights + weight_base + modes, corr1_real_sum, mask=valid_mode)
        tl.store(moment_weights + weight_base + 2 * modes, corr1_imag_sum, mask=valid_mode)
        tl.store(
            moment_weights + weight_base + 3 * modes,
            current1_energy_sum,
            mask=valid_mode,
        )
        tl.store(
            moment_weights + weight_base + 4 * modes,
            previous1_energy_sum,
            mask=valid_mode,
        )
        tl.store(moment_weights + weight_base + 5 * modes, corr4_real_sum, mask=valid_mode)
        tl.store(moment_weights + weight_base + 6 * modes, corr4_imag_sum, mask=valid_mode)
        tl.store(
            moment_weights + weight_base + 7 * modes,
            current4_energy_sum,
            mask=valid_mode,
        )
        tl.store(
            moment_weights + weight_base + 8 * modes,
            previous4_energy_sum,
            mask=valid_mode,
        )
    else:
        tl.store(moment_weights + weight_base, energy_scale, mask=valid_mode)
        tl.store(moment_weights + weight_base + modes, real1_weight, mask=valid_mode)
        tl.store(moment_weights + weight_base + 2 * modes, imag1_weight, mask=valid_mode)
        tl.store(
            moment_weights + weight_base + 3 * modes,
            current1_energy_gradient,
            mask=valid_mode,
        )
        tl.store(
            moment_weights + weight_base + 4 * modes,
            previous1_energy_gradient,
            mask=valid_mode,
        )
        tl.store(moment_weights + weight_base + 5 * modes, real4_weight, mask=valid_mode)
        tl.store(moment_weights + weight_base + 6 * modes, imag4_weight, mask=valid_mode)
        tl.store(
            moment_weights + weight_base + 7 * modes,
            current4_energy_gradient,
            mask=valid_mode,
        )
        tl.store(
            moment_weights + weight_base + 8 * modes,
            previous4_energy_gradient,
            mask=valid_mode,
        )


@triton.jit
def _split_correlation_gradient(  # noqa: ANN202
    states_real,
    states_imag,
    current_real,
    current_imag,
    time_index,
    offset,
    packed_offset,
    n_steps: int,
    modes: int,
    inverse_count,
    real_weight,
    imag_weight,
    current_energy_gradient,
    previous_energy_gradient,
    lag: tl.constexpr,
    packed_states: tl.constexpr,
    valid_mode,
):
    has_previous = time_index >= lag
    if packed_states:
        previous_offset = packed_offset - 2 * lag * modes
        previous_real = tl.load(
            states_real + previous_offset, mask=valid_mode & has_previous, other=0.0
        ).to(tl.float32)
        previous_imag = tl.load(
            states_real + previous_offset + modes,
            mask=valid_mode & has_previous,
            other=0.0,
        ).to(tl.float32)
    else:
        previous_offset = offset - lag * modes
        previous_real = tl.load(
            states_real + previous_offset, mask=valid_mode & has_previous, other=0.0
        ).to(tl.float32)
        previous_imag = tl.load(
            states_imag + previous_offset, mask=valid_mode & has_previous, other=0.0
        ).to(tl.float32)
    current_real_grad = inverse_count * (
        real_weight * previous_real - imag_weight * previous_imag
    ) + 2.0 * inverse_count * current_energy_gradient * current_real
    current_imag_grad = inverse_count * (
        real_weight * previous_imag + imag_weight * previous_real
    ) + 2.0 * inverse_count * current_energy_gradient * current_imag

    has_next = time_index < n_steps - lag
    if packed_states:
        next_offset = packed_offset + 2 * lag * modes
        next_real = tl.load(
            states_real + next_offset, mask=valid_mode & has_next, other=0.0
        ).to(tl.float32)
        next_imag = tl.load(
            states_real + next_offset + modes,
            mask=valid_mode & has_next,
            other=0.0,
        ).to(tl.float32)
    else:
        next_offset = offset + lag * modes
        next_real = tl.load(
            states_real + next_offset, mask=valid_mode & has_next, other=0.0
        ).to(tl.float32)
        next_imag = tl.load(
            states_imag + next_offset, mask=valid_mode & has_next, other=0.0
        ).to(tl.float32)
    previous_real_grad = inverse_count * (
        real_weight * next_real + imag_weight * next_imag
    ) + 2.0 * inverse_count * previous_energy_gradient * current_real
    previous_imag_grad = inverse_count * (
        real_weight * next_imag - imag_weight * next_real
    ) + 2.0 * inverse_count * previous_energy_gradient * current_imag
    return (
        tl.where(has_previous, current_real_grad, 0.0),
        tl.where(has_previous, current_imag_grad, 0.0),
        tl.where(has_next, previous_real_grad, 0.0),
        tl.where(has_next, previous_imag_grad, 0.0),
    )


@triton.jit
def _split_recurrence_moments_adjoint_kernel(
    decay_real,
    decay_imag,
    states_real,
    states_imag,
    direct_grad_states_real,
    direct_grad_states_imag,
    moment_weights,
    grad_moments,
    grad_decay_real,
    grad_decay_imag,
    grad_input_real,
    grad_input_imag,
    n_steps: int,
    modes: int,
    epsilon: float,
    recurrence_reverse: tl.constexpr,
    moments_reverse: tl.constexpr,
    packed_states: tl.constexpr,
    static_decay: tl.constexpr,
    raw_statistics: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    base = batch * n_steps * modes + mode
    weight_base = batch * 9 * modes + mode
    inverse_count1 = 1.0 / tl.maximum(n_steps - 1, 1)
    inverse_count4 = 1.0 / tl.maximum(n_steps - 4, 1)
    if raw_statistics:
        moment_base = batch * 5 * modes + mode
        energy_sum = tl.load(moment_weights + weight_base, mask=valid_mode, other=0.0)
        corr1_real_sum = tl.load(
            moment_weights + weight_base + modes, mask=valid_mode, other=0.0
        )
        corr1_imag_sum = tl.load(
            moment_weights + weight_base + 2 * modes, mask=valid_mode, other=0.0
        )
        current1_energy_sum = tl.load(
            moment_weights + weight_base + 3 * modes, mask=valid_mode, other=0.0
        )
        previous1_energy_sum = tl.load(
            moment_weights + weight_base + 4 * modes, mask=valid_mode, other=0.0
        )
        corr4_real_sum = tl.load(
            moment_weights + weight_base + 5 * modes, mask=valid_mode, other=0.0
        )
        corr4_imag_sum = tl.load(
            moment_weights + weight_base + 6 * modes, mask=valid_mode, other=0.0
        )
        current4_energy_sum = tl.load(
            moment_weights + weight_base + 7 * modes, mask=valid_mode, other=0.0
        )
        previous4_energy_sum = tl.load(
            moment_weights + weight_base + 8 * modes, mask=valid_mode, other=0.0
        )
        energy = energy_sum / n_steps
        energy_output_gradient = tl.load(
            grad_moments + moment_base, mask=valid_mode, other=0.0
        ).to(tl.float32)
        energy_scale = (2.0 / n_steps) * energy_output_gradient / (1.0 + energy)
        imaginary_sign = -1.0 if moments_reverse else 1.0
        corr1_real = corr1_real_sum * inverse_count1
        corr1_imag = corr1_imag_sum * inverse_count1
        current1_energy = current1_energy_sum * inverse_count1
        previous1_energy = previous1_energy_sum * inverse_count1
        root1 = tl.sqrt(current1_energy * previous1_energy)
        denominator1 = tl.maximum(root1, epsilon)
        output1_real_gradient = tl.load(
            grad_moments + moment_base + modes, mask=valid_mode, other=0.0
        ).to(tl.float32)
        output1_imag_gradient = imaginary_sign * tl.load(
            grad_moments + moment_base + 2 * modes, mask=valid_mode, other=0.0
        ).to(tl.float32)
        real1_weight = tl.where(
            n_steps > 1, output1_real_gradient / denominator1, 0.0
        )
        imag1_weight = tl.where(
            n_steps > 1, output1_imag_gradient / denominator1, 0.0
        )
        weighted1 = output1_real_gradient * corr1_real + output1_imag_gradient * corr1_imag
        root1_gradient = tl.where(
            root1 > epsilon, -weighted1 / (denominator1 * denominator1), 0.0
        )
        safe_root1 = tl.maximum(root1, epsilon)
        current1_energy_gradient = 0.5 * root1_gradient * previous1_energy / safe_root1
        previous1_energy_gradient = 0.5 * root1_gradient * current1_energy / safe_root1
        corr4_real = corr4_real_sum * inverse_count4
        corr4_imag = corr4_imag_sum * inverse_count4
        current4_energy = current4_energy_sum * inverse_count4
        previous4_energy = previous4_energy_sum * inverse_count4
        root4 = tl.sqrt(current4_energy * previous4_energy)
        denominator4 = tl.maximum(root4, epsilon)
        output4_real_gradient = tl.load(
            grad_moments + moment_base + 3 * modes, mask=valid_mode, other=0.0
        ).to(tl.float32)
        output4_imag_gradient = imaginary_sign * tl.load(
            grad_moments + moment_base + 4 * modes, mask=valid_mode, other=0.0
        ).to(tl.float32)
        real4_weight = tl.where(
            n_steps > 4, output4_real_gradient / denominator4, 0.0
        )
        imag4_weight = tl.where(
            n_steps > 4, output4_imag_gradient / denominator4, 0.0
        )
        weighted4 = output4_real_gradient * corr4_real + output4_imag_gradient * corr4_imag
        root4_gradient = tl.where(
            root4 > epsilon, -weighted4 / (denominator4 * denominator4), 0.0
        )
        safe_root4 = tl.maximum(root4, epsilon)
        current4_energy_gradient = 0.5 * root4_gradient * previous4_energy / safe_root4
        previous4_energy_gradient = 0.5 * root4_gradient * current4_energy / safe_root4
    else:
        energy_scale = tl.load(moment_weights + weight_base, mask=valid_mode, other=0.0)
        real1_weight = tl.load(
            moment_weights + weight_base + modes, mask=valid_mode, other=0.0
        )
        imag1_weight = tl.load(
            moment_weights + weight_base + 2 * modes, mask=valid_mode, other=0.0
        )
        current1_energy_gradient = tl.load(
            moment_weights + weight_base + 3 * modes, mask=valid_mode, other=0.0
        )
        previous1_energy_gradient = tl.load(
            moment_weights + weight_base + 4 * modes, mask=valid_mode, other=0.0
        )
        real4_weight = tl.load(
            moment_weights + weight_base + 5 * modes, mask=valid_mode, other=0.0
        )
        imag4_weight = tl.load(
            moment_weights + weight_base + 6 * modes, mask=valid_mode, other=0.0
        )
        current4_energy_gradient = tl.load(
            moment_weights + weight_base + 7 * modes, mask=valid_mode, other=0.0
        )
        previous4_energy_gradient = tl.load(
            moment_weights + weight_base + 8 * modes, mask=valid_mode, other=0.0
        )
    fixed_decay_real = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    if static_decay:
        fixed_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(
            tl.float32
        )
        fixed_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(
            tl.float32
        )

    lambda_real = tl.zeros((BLOCK_MODES,), tl.float32)
    lambda_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    adjoint_step = 0
    while adjoint_step < n_steps:
        time_index = adjoint_step if recurrence_reverse else n_steps - 1 - adjoint_step
        offset = base + time_index * modes
        packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
        if packed_states:
            current_real = tl.load(
                states_real + packed_offset, mask=valid_mode, other=0.0
            ).to(tl.float32)
            current_imag = tl.load(
                states_real + packed_offset + modes, mask=valid_mode, other=0.0
            ).to(tl.float32)
        else:
            current_real = tl.load(
                states_real + offset, mask=valid_mode, other=0.0
            ).to(tl.float32)
            current_imag = tl.load(
                states_imag + offset, mask=valid_mode, other=0.0
            ).to(tl.float32)
        moment_grad_real = energy_scale * current_real
        moment_grad_imag = energy_scale * current_imag
        current1_real, current1_imag, previous1_real, previous1_imag = (
            _split_correlation_gradient(
                states_real,
                states_imag,
                current_real,
                current_imag,
                time_index,
                offset,
                packed_offset,
                n_steps,
                modes,
                inverse_count1,
                real1_weight,
                imag1_weight,
                current1_energy_gradient,
                previous1_energy_gradient,
                lag=1,
                packed_states=packed_states,
                valid_mode=valid_mode,
            )
        )
        moment_grad_real += current1_real
        moment_grad_imag += current1_imag
        moment_grad_real += previous1_real
        moment_grad_imag += previous1_imag
        current4_real, current4_imag, previous4_real, previous4_imag = (
            _split_correlation_gradient(
                states_real,
                states_imag,
                current_real,
                current_imag,
                time_index,
                offset,
                packed_offset,
                n_steps,
                modes,
                inverse_count4,
                real4_weight,
                imag4_weight,
                current4_energy_gradient,
                previous4_energy_gradient,
                lag=4,
                packed_states=packed_states,
                valid_mode=valid_mode,
            )
        )
        moment_grad_real += current4_real
        moment_grad_imag += current4_imag
        moment_grad_real += previous4_real
        moment_grad_imag += previous4_imag

        if packed_states:
            direct_grad_real = tl.load(
                direct_grad_states_real + packed_offset, mask=valid_mode, other=0.0
            ).to(tl.float32)
            direct_grad_imag = tl.load(
                direct_grad_states_real + packed_offset + modes,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        else:
            direct_grad_real = tl.load(
                direct_grad_states_real + offset, mask=valid_mode, other=0.0
            ).to(tl.float32)
            direct_grad_imag = tl.load(
                direct_grad_states_imag + offset, mask=valid_mode, other=0.0
            ).to(tl.float32)
        combined_state_grad_real = _materialize_fp32(direct_grad_real + moment_grad_real)
        combined_state_grad_imag = _materialize_fp32(direct_grad_imag + moment_grad_imag)
        lambda_real += combined_state_grad_real
        lambda_imag += combined_state_grad_imag

        previous_index = time_index + 1 if recurrence_reverse else time_index - 1
        has_recurrence_previous = (
            time_index < n_steps - 1 if recurrence_reverse else time_index > 0
        )
        if packed_states:
            recurrence_previous_offset = (
                (batch * n_steps + previous_index) * 2 * modes + mode
            )
            recurrence_previous_real = tl.load(
                states_real + recurrence_previous_offset,
                mask=valid_mode & has_recurrence_previous,
                other=0.0,
            ).to(tl.float32)
            recurrence_previous_imag = tl.load(
                states_real + recurrence_previous_offset + modes,
                mask=valid_mode & has_recurrence_previous,
                other=0.0,
            ).to(tl.float32)
        else:
            recurrence_previous_offset = base + previous_index * modes
            recurrence_previous_real = tl.load(
                states_real + recurrence_previous_offset,
                mask=valid_mode & has_recurrence_previous,
                other=0.0,
            ).to(tl.float32)
            recurrence_previous_imag = tl.load(
                states_imag + recurrence_previous_offset,
                mask=valid_mode & has_recurrence_previous,
                other=0.0,
            ).to(tl.float32)
        if static_decay:
            ar = fixed_decay_real
            ai = fixed_decay_imag
        else:
            ar = tl.load(decay_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
            ai = tl.load(decay_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        tl.store(grad_input_real + offset, lambda_real, mask=valid_mode)
        tl.store(grad_input_imag + offset, lambda_imag, mask=valid_mode)
        tl.store(
            grad_decay_real + offset,
            lambda_real * recurrence_previous_real + lambda_imag * recurrence_previous_imag,
            mask=valid_mode,
        )
        tl.store(
            grad_decay_imag + offset,
            -lambda_real * recurrence_previous_imag + lambda_imag * recurrence_previous_real,
            mask=valid_mode,
        )
        next_lambda_real = ar * lambda_real + ai * lambda_imag
        next_lambda_imag = -ai * lambda_real + ar * lambda_imag
        lambda_real = next_lambda_real
        lambda_imag = next_lambda_imag
        adjoint_step += 1


@dataclass(frozen=True, slots=True)
class _MomentWeights:
    energy_scale: Tensor
    inverse1: float
    real1: Tensor
    imag1: Tensor
    current_energy1: Tensor
    previous_energy1: Tensor
    inverse4: float
    real4: Tensor
    imag4: Tensor
    current_energy4: Tensor
    previous_energy4: Tensor


def _correlation_weights(
    states_real: Tensor,
    states_imag: Tensor,
    grad_moments: Tensor,
    *,
    lag: int,
    output_offset: int,
    imaginary_sign: float,
    epsilon: float,
) -> tuple[float, Tensor, Tensor, Tensor, Tensor]:
    n_steps = states_real.shape[1]
    modes = states_real.shape[2]
    grad_real = grad_moments[:, output_offset : output_offset + modes]
    grad_imag = imaginary_sign * grad_moments[:, output_offset + modes : output_offset + 2 * modes]
    if n_steps <= lag:
        zeros = torch.zeros_like(grad_real)
        return 1.0, zeros, zeros, zeros, zeros

    current_real = states_real[:, lag:]
    current_imag = states_imag[:, lag:]
    previous_real = states_real[:, :-lag]
    previous_imag = states_imag[:, :-lag]
    overlap = n_steps - lag
    inverse_overlap = 1.0 / overlap
    correlation_real = (current_real * previous_real + current_imag * previous_imag).mean(dim=1)
    correlation_imag = (current_imag * previous_real - current_real * previous_imag).mean(dim=1)
    current_energy = (current_real.square() + current_imag.square()).mean(dim=1)
    previous_energy = (previous_real.square() + previous_imag.square()).mean(dim=1)
    root = torch.sqrt(current_energy * previous_energy)
    denominator = root.clamp_min(epsilon)
    real_weight = grad_real / denominator
    imag_weight = grad_imag / denominator
    weighted = grad_real * correlation_real + grad_imag * correlation_imag
    root_gradient = torch.where(
        root > epsilon,
        -weighted / denominator.square(),
        torch.zeros_like(weighted),
    )
    safe_root = root.clamp_min(epsilon)
    current_energy_gradient = 0.5 * root_gradient * previous_energy / safe_root
    previous_energy_gradient = 0.5 * root_gradient * current_energy / safe_root
    return (
        inverse_overlap,
        real_weight,
        imag_weight,
        current_energy_gradient,
        previous_energy_gradient,
    )


def _moment_weights(
    states_real: Tensor,
    states_imag: Tensor,
    grad_moments: Tensor,
    *,
    direction: int,
    epsilon: float,
) -> _MomentWeights:
    n_steps = states_real.shape[1]
    modes = states_real.shape[2]
    energy = (states_real.square() + states_imag.square()).mean(dim=1)
    energy_scale = (2.0 / n_steps) * grad_moments[:, :modes] / (1.0 + energy)
    imaginary_sign = -1.0 if direction == _BACKWARD else 1.0
    inverse1, real1, imag1, current1, previous1 = _correlation_weights(
        states_real,
        states_imag,
        grad_moments,
        lag=1,
        output_offset=modes,
        imaginary_sign=imaginary_sign,
        epsilon=epsilon,
    )
    inverse4, real4, imag4, current4, previous4 = _correlation_weights(
        states_real,
        states_imag,
        grad_moments,
        lag=4,
        output_offset=3 * modes,
        imaginary_sign=imaginary_sign,
        epsilon=epsilon,
    )
    return _MomentWeights(
        energy_scale,
        inverse1,
        real1,
        imag1,
        current1,
        previous1,
        inverse4,
        real4,
        imag4,
        current4,
        previous4,
    )


def _moment_gradient_at(
    states_real: Tensor,
    states_imag: Tensor,
    time_index: int,
    weights: _MomentWeights,
) -> tuple[Tensor, Tensor]:
    n_steps = states_real.shape[1]
    current_real = states_real[:, time_index].float()
    current_imag = states_imag[:, time_index].float()
    grad_real = weights.energy_scale * current_real
    grad_imag = weights.energy_scale * current_imag

    if time_index >= 1:
        previous_real = states_real[:, time_index - 1].float()
        previous_imag = states_imag[:, time_index - 1].float()
        grad_real = (
            grad_real
            + weights.inverse1 * (weights.real1 * previous_real - weights.imag1 * previous_imag)
            + 2.0 * weights.inverse1 * weights.current_energy1 * current_real
        )
        grad_imag = (
            grad_imag
            + weights.inverse1 * (weights.real1 * previous_imag + weights.imag1 * previous_real)
            + 2.0 * weights.inverse1 * weights.current_energy1 * current_imag
        )
    if time_index < n_steps - 1:
        next_real = states_real[:, time_index + 1].float()
        next_imag = states_imag[:, time_index + 1].float()
        grad_real = (
            grad_real
            + weights.inverse1 * (weights.real1 * next_real + weights.imag1 * next_imag)
            + 2.0 * weights.inverse1 * weights.previous_energy1 * current_real
        )
        grad_imag = (
            grad_imag
            + weights.inverse1 * (weights.real1 * next_imag - weights.imag1 * next_real)
            + 2.0 * weights.inverse1 * weights.previous_energy1 * current_imag
        )

    if time_index >= 4:
        previous_real = states_real[:, time_index - 4].float()
        previous_imag = states_imag[:, time_index - 4].float()
        grad_real = (
            grad_real
            + weights.inverse4 * (weights.real4 * previous_real - weights.imag4 * previous_imag)
            + 2.0 * weights.inverse4 * weights.current_energy4 * current_real
        )
        grad_imag = (
            grad_imag
            + weights.inverse4 * (weights.real4 * previous_imag + weights.imag4 * previous_real)
            + 2.0 * weights.inverse4 * weights.current_energy4 * current_imag
        )
    if time_index < n_steps - 4:
        next_real = states_real[:, time_index + 4].float()
        next_imag = states_imag[:, time_index + 4].float()
        grad_real = (
            grad_real
            + weights.inverse4 * (weights.real4 * next_real + weights.imag4 * next_imag)
            + 2.0 * weights.inverse4 * weights.previous_energy4 * current_real
        )
        grad_imag = (
            grad_imag
            + weights.inverse4 * (weights.real4 * next_imag - weights.imag4 * next_real)
            + 2.0 * weights.inverse4 * weights.previous_energy4 * current_imag
        )
    return grad_real, grad_imag


def _reference_fused_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    states_real: Tensor,
    states_imag: Tensor,
    direct_grad_states_real: Tensor,
    direct_grad_states_imag: Tensor,
    grad_moments: Tensor,
    *,
    recurrence_reverse: bool,
    moment_direction: int,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    weights = _moment_weights(
        states_real.float(),
        states_imag.float(),
        grad_moments.float(),
        direction=moment_direction,
        epsilon=epsilon,
    )
    batch, n_steps, modes = decay_real.shape
    lambda_real = torch.zeros((batch, modes), dtype=torch.float32, device=decay_real.device)
    lambda_imag = torch.zeros_like(lambda_real)
    grad_decay_real: list[Tensor | None] = [None] * n_steps
    grad_decay_imag: list[Tensor | None] = [None] * n_steps
    grad_input_real: list[Tensor | None] = [None] * n_steps
    grad_input_imag: list[Tensor | None] = [None] * n_steps
    adjoint_steps = range(n_steps) if recurrence_reverse else range(n_steps - 1, -1, -1)
    for time_index in adjoint_steps:
        moment_real, moment_imag = _moment_gradient_at(
            states_real,
            states_imag,
            time_index,
            weights,
        )
        lambda_real = lambda_real + direct_grad_states_real[:, time_index].float() + moment_real
        lambda_imag = lambda_imag + direct_grad_states_imag[:, time_index].float() + moment_imag
        has_previous = time_index < n_steps - 1 if recurrence_reverse else time_index > 0
        if has_previous:
            previous_index = time_index + 1 if recurrence_reverse else time_index - 1
            previous_real = states_real[:, previous_index].float()
            previous_imag = states_imag[:, previous_index].float()
        else:
            previous_real = torch.zeros_like(lambda_real)
            previous_imag = torch.zeros_like(lambda_imag)
        grad_input_real[time_index] = lambda_real
        grad_input_imag[time_index] = lambda_imag
        grad_decay_real[time_index] = lambda_real * previous_real + lambda_imag * previous_imag
        grad_decay_imag[time_index] = -lambda_real * previous_imag + lambda_imag * previous_real
        ar = decay_real[:, time_index].float()
        ai = decay_imag[:, time_index].float()
        next_lambda_real = ar * lambda_real + ai * lambda_imag
        next_lambda_imag = -ai * lambda_real + ar * lambda_imag
        lambda_real = next_lambda_real
        lambda_imag = next_lambda_imag
    return (
        torch.stack([value for value in grad_decay_real if value is not None], dim=1).to(
            decay_real.dtype
        ),
        torch.stack([value for value in grad_decay_imag if value is not None], dim=1).to(
            decay_imag.dtype
        ),
        torch.stack([value for value in grad_input_real if value is not None], dim=1).to(
            direct_grad_states_real.dtype
        ),
        torch.stack([value for value in grad_input_imag if value is not None], dim=1).to(
            direct_grad_states_imag.dtype
        ),
    )


@triton_op("lnet::pac_real2d_recurrence_moments_training_backward", mutates_args={})
def _fused_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    states_real: Tensor,
    states_imag: Tensor,
    direct_grad_states_real: Tensor,
    direct_grad_states_imag: Tensor,
    grad_moments: Tensor,
    recurrence_reverse: bool,
    moment_direction: int,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not decay_real.is_cuda:
        return _reference_fused_backward(
            decay_real,
            decay_imag,
            states_real,
            states_imag,
            direct_grad_states_real,
            direct_grad_states_imag,
            grad_moments,
            recurrence_reverse=recurrence_reverse,
            moment_direction=moment_direction,
            epsilon=epsilon,
        )
    static_decay = _is_mode_static_expanded(decay_real, direct_grad_states_real) and (
        _is_mode_static_expanded(decay_imag, direct_grad_states_imag)
    )
    real = decay_real if static_decay else decay_real.contiguous()
    imag = decay_imag if static_decay else decay_imag.contiguous()
    state_real = states_real.contiguous()
    state_imag = states_imag.contiguous()
    direct_real = direct_grad_states_real.contiguous()
    direct_imag = direct_grad_states_imag.contiguous()
    moment_gradient = grad_moments.contiguous()
    grad_decay_real = torch.empty_like(direct_real)
    grad_decay_imag = torch.empty_like(direct_imag)
    grad_input_real = torch.empty_like(direct_real)
    grad_input_imag = torch.empty_like(direct_imag)
    batch, n_steps, modes = direct_real.shape
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    grid = _mode_grid(batch, modes, block_modes)
    num_warps = _select_backward_num_warps(n_steps)
    if _use_split_backward():
        stats_block_modes = _select_split_block_modes(
            "LNET_PAC_SPLIT_STATS_BLOCK_MODES", block_modes
        )
        adjoint_block_modes = _select_split_block_modes(
            "LNET_PAC_SPLIT_ADJOINT_BLOCK_MODES", block_modes
        )
        raw_statistics = _use_split_raw_statistics()
        moment_weights = torch.empty(
            (batch, 9, modes), dtype=torch.float32, device=direct_real.device
        )
        wrap_triton(_split_moment_weights_kernel)[
            _mode_grid(batch, modes, stats_block_modes)
        ](
            state_real,
            state_imag,
            moment_gradient,
            moment_weights,
            n_steps,
            modes,
            epsilon,
            moments_reverse=moment_direction == _BACKWARD,
            packed_states=False,
            raw_statistics=raw_statistics,
            BLOCK_MODES=stats_block_modes,
            num_warps=num_warps,
        )
        wrap_triton(_split_recurrence_moments_adjoint_kernel)[
            _mode_grid(batch, modes, adjoint_block_modes)
        ](
            real,
            imag,
            state_real,
            state_imag,
            direct_real,
            direct_imag,
            moment_weights,
            moment_gradient,
            grad_decay_real,
            grad_decay_imag,
            grad_input_real,
            grad_input_imag,
            n_steps,
            modes,
            epsilon,
            recurrence_reverse=recurrence_reverse,
            moments_reverse=moment_direction == _BACKWARD,
            packed_states=False,
            static_decay=static_decay,
            raw_statistics=raw_statistics,
            BLOCK_MODES=adjoint_block_modes,
            num_warps=num_warps,
        )
    else:
        wrap_triton(_fused_recurrence_moments_backward_kernel)[grid](
            real,
            imag,
            state_real,
            state_imag,
            direct_real,
            direct_imag,
            moment_gradient,
            grad_decay_real,
            grad_decay_imag,
            grad_input_real,
            grad_input_imag,
            n_steps,
            modes,
            epsilon,
            recurrence_reverse=recurrence_reverse,
            moments_reverse=moment_direction == _BACKWARD,
            packed_states=False,
            static_decay=static_decay,
            BLOCK_MODES=block_modes,
            num_warps=num_warps,
        )
    return grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag


@triton_op("lnet::pac_real2d_recurrence_moments_training_packed_backward", mutates_args={})
def _fused_packed_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    recurrence_reverse: bool,
    moment_direction: int,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    modes = decay_real.shape[2]
    if not decay_real.is_cuda:
        states_real, states_imag = packed_states.split(modes, dim=-1)
        direct_real, direct_imag = direct_grad_packed_states.split(modes, dim=-1)
        return _reference_fused_backward(
            decay_real,
            decay_imag,
            states_real,
            states_imag,
            direct_real,
            direct_imag,
            grad_moments,
            recurrence_reverse=recurrence_reverse,
            moment_direction=moment_direction,
            epsilon=epsilon,
        )
    input_shape_reference = direct_grad_packed_states[..., :modes]
    static_decay = _is_mode_static_expanded(decay_real, input_shape_reference) and (
        _is_mode_static_expanded(decay_imag, input_shape_reference)
    )
    real = decay_real if static_decay else decay_real.contiguous()
    imag = decay_imag if static_decay else decay_imag.contiguous()
    states = packed_states.contiguous()
    direct = direct_grad_packed_states.contiguous()
    moment_gradient = grad_moments.contiguous()
    grad_decay_real = torch.empty(
        (*direct.shape[:-1], modes), dtype=direct.dtype, device=direct.device
    )
    grad_decay_imag = torch.empty_like(grad_decay_real)
    grad_input_real = torch.empty_like(grad_decay_real)
    grad_input_imag = torch.empty_like(grad_decay_real)
    batch, n_steps, _ = grad_decay_real.shape
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    grid = _mode_grid(batch, modes, block_modes)
    num_warps = _select_backward_num_warps(n_steps)
    if _use_split_backward():
        stats_block_modes = _select_split_block_modes(
            "LNET_PAC_SPLIT_STATS_BLOCK_MODES", block_modes
        )
        adjoint_block_modes = _select_split_block_modes(
            "LNET_PAC_SPLIT_ADJOINT_BLOCK_MODES", block_modes
        )
        raw_statistics = _use_split_raw_statistics()
        moment_weights = torch.empty(
            (batch, 9, modes), dtype=torch.float32, device=direct.device
        )
        wrap_triton(_split_moment_weights_kernel)[
            _mode_grid(batch, modes, stats_block_modes)
        ](
            states,
            states,
            moment_gradient,
            moment_weights,
            n_steps,
            modes,
            epsilon,
            moments_reverse=moment_direction == _BACKWARD,
            packed_states=True,
            raw_statistics=raw_statistics,
            BLOCK_MODES=stats_block_modes,
            num_warps=num_warps,
        )
        wrap_triton(_split_recurrence_moments_adjoint_kernel)[
            _mode_grid(batch, modes, adjoint_block_modes)
        ](
            real,
            imag,
            states,
            states,
            direct,
            direct,
            moment_weights,
            moment_gradient,
            grad_decay_real,
            grad_decay_imag,
            grad_input_real,
            grad_input_imag,
            n_steps,
            modes,
            epsilon,
            recurrence_reverse=recurrence_reverse,
            moments_reverse=moment_direction == _BACKWARD,
            packed_states=True,
            static_decay=static_decay,
            raw_statistics=raw_statistics,
            BLOCK_MODES=adjoint_block_modes,
            num_warps=num_warps,
        )
    else:
        wrap_triton(_fused_recurrence_moments_backward_kernel)[grid](
            real,
            imag,
            states,
            states,
            direct,
            direct,
            moment_gradient,
            grad_decay_real,
            grad_decay_imag,
            grad_input_real,
            grad_input_imag,
            n_steps,
            modes,
            epsilon,
            recurrence_reverse=recurrence_reverse,
            moments_reverse=moment_direction == _BACKWARD,
            packed_states=True,
            static_decay=static_decay,
            BLOCK_MODES=block_modes,
            num_warps=num_warps,
        )
    return grad_decay_real, grad_decay_imag, grad_input_real, grad_input_imag


def _reverse_two_pass_forward(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    moment_direction: int,
    epsilon: float,
    packed_output: bool,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    static_decay = _is_mode_static_expanded(decay_real, input_real) and (
        _is_mode_static_expanded(decay_imag, input_imag)
    )
    real = decay_real if static_decay else decay_real.contiguous()
    imag = decay_imag if static_decay else decay_imag.contiguous()
    drive_real = input_real.contiguous()
    drive_imag = input_imag.contiguous()
    batch, n_steps, modes = drive_real.shape
    if packed_output:
        packed_states = torch.empty(
            (batch, n_steps, 2 * modes),
            dtype=real.dtype,
            device=real.device,
        )
        states_real = packed_states
        states_imag = packed_states
    else:
        states_real = torch.empty_like(drive_real)
        states_imag = torch.empty_like(drive_imag)
    moments = torch.empty((batch, 5 * modes), dtype=real.dtype, device=real.device)
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    wrap_triton(_reverse_recurrence_physical_moments_kernel)[
        _mode_grid(batch, modes, block_modes)
    ](
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
        moments_reverse=moment_direction == _BACKWARD,
        packed_output=packed_output,
        static_decay=static_decay,
        BLOCK_MODES=block_modes,
        num_warps=4 if n_steps <= 128 else 1,
    )
    if packed_output:
        return states_real, moments
    return states_real, states_imag, moments


@triton_op("lnet::pac_real2d_recurrence_moments_training", mutates_args={})
def _fused_training_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    recurrence_reverse: bool,
    moment_direction: int,
    epsilon: float,
    use_two_pass_reverse: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    # In physical forward order the inference kernel has the exact same FP32
    # recurrence and moment operation order as the two training forward
    # kernels.  Reverse recurrence can use the exact two-pass kernel below or
    # retain the old split path for per-shape A/B screening.
    if not recurrence_reverse and moment_direction == _FORWARD:
        return recurrence_moments_inference(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=False,
            epsilon=epsilon,
        )
    if recurrence_reverse and use_two_pass_reverse and decay_real.is_cuda:
        states_real, states_imag, moments = _reverse_two_pass_forward(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            moment_direction=moment_direction,
            epsilon=epsilon,
            packed_output=False,
        )
        return states_real, states_imag, moments
    states_real, states_imag = pac_triton_recurrence_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=recurrence_reverse,
    )
    moments = online_modal_moments(
        states_real,
        states_imag,
        physical_direction=_direction_name(moment_direction),
        backend="auto",
        epsilon=epsilon,
    )
    return states_real, states_imag, moments


@triton_op("lnet::pac_real2d_recurrence_moments_training_packed", mutates_args={})
def _fused_packed_training_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    recurrence_reverse: bool,
    moment_direction: int,
    epsilon: float,
    use_two_pass_reverse: bool,
) -> tuple[Tensor, Tensor]:
    if not decay_real.is_cuda:
        states_real, states_imag, moments = _fused_training_op(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            recurrence_reverse,
            moment_direction,
            epsilon,
            use_two_pass_reverse,
        )
        return torch.cat((states_real, states_imag), dim=-1), moments
    if recurrence_reverse and use_two_pass_reverse:
        packed_states, moments = _reverse_two_pass_forward(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            moment_direction=moment_direction,
            epsilon=epsilon,
            packed_output=True,
        )
        return packed_states, moments
    if not recurrence_reverse and moment_direction == _FORWARD:
        static_decay = _is_mode_static_expanded(decay_real, input_real) and (
            _is_mode_static_expanded(decay_imag, input_imag)
        )
        real = decay_real if static_decay else decay_real.contiguous()
        imag = decay_imag if static_decay else decay_imag.contiguous()
        drive_real = input_real.contiguous()
        drive_imag = input_imag.contiguous()
        batch, n_steps, modes = drive_real.shape
        packed_states = torch.empty(
            (batch, n_steps, 2 * modes),
            dtype=real.dtype,
            device=real.device,
        )
        moments = torch.empty((batch, 5 * modes), dtype=real.dtype, device=real.device)
        block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
        wrap_triton(_recurrence_moments_kernel)[_mode_grid(batch, modes, block_modes)](
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
            reverse=False,
            static_decay=static_decay,
            packed_input=False,
            packed_output=True,
            BLOCK_MODES=block_modes,
        )
        return packed_states, moments
    states_real, states_imag = pac_triton_recurrence_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        reverse=recurrence_reverse,
    )
    moments = online_modal_moments(
        states_real,
        states_imag,
        physical_direction=_direction_name(moment_direction),
        backend="auto",
        epsilon=epsilon,
    )
    return torch.cat((states_real, states_imag), dim=-1), moments


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, bool, int, float, bool],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    (
        decay_real,
        decay_imag,
        _input_real,
        _input_imag,
        recurrence_reverse,
        direction,
        epsilon,
        _use_two_pass_reverse,
    ) = inputs
    states_real, states_imag, _moments = output
    ctx.recurrence_reverse = recurrence_reverse
    ctx.moment_direction = direction
    ctx.epsilon = epsilon
    ctx.save_for_backward(decay_real, decay_imag, states_real, states_imag)


def _backward(
    ctx: _AutogradContext,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
    grad_moments: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, None, None, None, None]:
    decay_real, decay_imag, states_real, states_imag = ctx.saved_tensors
    gradients = _fused_backward_op(
        decay_real,
        decay_imag,
        states_real,
        states_imag,
        grad_states_real,
        grad_states_imag,
        grad_moments,
        ctx.recurrence_reverse,
        ctx.moment_direction,
        ctx.epsilon,
    )
    return *gradients, None, None, None, None


torch.library.register_autograd(
    "lnet::pac_real2d_recurrence_moments_training",
    _backward,
    setup_context=_setup_context,
)


def _setup_packed_context(
    ctx: _PackedAutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, bool, int, float, bool],
    output: tuple[Tensor, Tensor],
) -> None:
    (
        decay_real,
        decay_imag,
        _input_real,
        _input_imag,
        recurrence_reverse,
        direction,
        epsilon,
        _use_two_pass_reverse,
    ) = inputs
    packed_states, _moments = output
    ctx.recurrence_reverse = recurrence_reverse
    ctx.moment_direction = direction
    ctx.epsilon = epsilon
    ctx.save_for_backward(decay_real, decay_imag, packed_states)


def _packed_backward(
    ctx: _PackedAutogradContext,
    grad_packed_states: Tensor,
    grad_moments: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, None, None, None, None]:
    decay_real, decay_imag, packed_states = ctx.saved_tensors
    gradients = _fused_packed_backward_op(
        decay_real,
        decay_imag,
        packed_states,
        grad_packed_states,
        grad_moments,
        ctx.recurrence_reverse,
        ctx.moment_direction,
        ctx.epsilon,
    )
    return *gradients, None, None, None, None


torch.library.register_autograd(
    "lnet::pac_real2d_recurrence_moments_training_packed",
    _packed_backward,
    setup_context=_setup_packed_context,
)


def fused_recurrence_moments_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    recurrence_reverse: bool = False,
    moment_direction: Direction = "forward",
    epsilon: float = _DEFAULT_EPSILON,
    use_two_pass_reverse: bool | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run recurrence+moments with a fused training backward experiment."""
    _validate_inputs(decay_real, decay_imag, input_real, input_imag, epsilon)
    resolved_two_pass = (
        decay_real.shape[0] > 1 and decay_real.shape[1] <= 128
        if use_two_pass_reverse is None
        else use_two_pass_reverse
    )
    return _fused_training_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        recurrence_reverse,
        _direction_code(moment_direction),
        epsilon,
        resolved_two_pass,
    )


def fused_recurrence_moments_packed_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    recurrence_reverse: bool = False,
    moment_direction: Direction = "forward",
    epsilon: float = _DEFAULT_EPSILON,
    use_two_pass_reverse: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the training recurrence with real/imag states packed for synthesis."""
    _validate_inputs(decay_real, decay_imag, input_real, input_imag, epsilon)
    resolved_two_pass = (
        decay_real.shape[0] > 1 and decay_real.shape[1] <= 128
        if use_two_pass_reverse is None
        else use_two_pass_reverse
    )
    return _fused_packed_training_op(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        recurrence_reverse,
        _direction_code(moment_direction),
        epsilon,
        resolved_two_pass,
    )


def _validate_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    epsilon: float,
) -> None:
    shape = decay_real.shape
    if len(shape) != 3 or shape[1] == 0 or shape[2] == 0:
        message = "recurrence tensors must have shape [batch,time>0,modes>0]"
        raise ValueError(message)
    if any(tensor.shape != shape for tensor in (decay_imag, input_real, input_imag)):
        message = "recurrence tensors must have matching shapes"
        raise ValueError(message)
    if any(tensor.device != decay_real.device for tensor in (decay_imag, input_real, input_imag)):
        message = "recurrence tensors must share one device"
        raise ValueError(message)
    if any(tensor.dtype != decay_real.dtype for tensor in (decay_imag, input_real, input_imag)):
        message = "recurrence tensors must share one dtype"
        raise TypeError(message)
    if decay_real.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        message = "fused recurrence+moments supports fp16, bf16, and fp32"
        raise TypeError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)


def _direction_code(direction: Direction) -> int:
    match direction:
        case "forward":
            return _FORWARD
        case "backward":
            return _BACKWARD
        case unreachable:
            assert_never(unreachable)


def _direction_name(direction: int) -> Direction:
    return "forward" if direction == _FORWARD else "backward"


__all__ = [
    "fused_recurrence_moments_packed_training",
    "fused_recurrence_moments_training",
]
