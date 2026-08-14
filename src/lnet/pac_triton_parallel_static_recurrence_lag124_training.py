"""Time-parallel static recurrence fused with canonical lag-(1, 2, 4) moments.

One Triton program owns one ``(batch, mode)`` pair.  Forward evaluates the
complex affine recurrence with ``tl.associative_scan`` and reduces all seven
moments before leaving the program.  Backward forms the complete lag-moment
state VJP and feeds it directly into the reverse affine scan, avoiding both an
intermediate sequence gradient and a second sequence kernel.

The moments are always measured in physical forward time, independently of
the recurrence direction.  This is the aligned-moment convention used by the
optimized ALPHABET writer and terminal reader.  Parallel association changes
FP32 rounding, so the API is deliberately explicit and accepts only static
FP32 poles with at most 2048 steps.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false, reportPrivateUsage=false
# ruff: noqa: ANN001, ANN202, FBT001, N803
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from .pac_triton_radial_log_recurrence_lag124 import (
    reference_static_radial_log_recurrence_lag124_moments_only,
    reference_static_radial_log_recurrence_lag124_moments_packed_io,
)
from .pac_triton_recurrence_lag124 import (
    reference_static_recurrence_lag124_moments_only,
    reference_static_recurrence_lag124_moments_packed_io,
)
from .pac_triton_recurrence_lag124_training import (
    _reference_backward as _serial_reference_backward,
)

_DEFAULT_EPSILON: Final[float] = 1.0e-8
_MAX_STEPS: Final[int] = 2048
_VALID_WARPS: Final[tuple[int, ...]] = (4, 8)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    reverse: bool
    epsilon: float
    num_warps: int

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def set_materialize_grads(self, value: bool) -> None: ...


class _ExcitationAutogradContext(_AutogradContext, Protocol):
    """Saved state for the static gamma/excitation boundary fusion."""


@triton.jit
def _compose_complex_affine(
    left_ar,
    left_ai,
    left_br,
    left_bi,
    right_ar,
    right_ai,
    right_br,
    right_bi,
):
    """Compose adjacent transforms as ``right(left(state))``."""
    product_ar = right_ar * left_ar - right_ai * left_ai
    product_ai = right_ai * left_ar + right_ar * left_ai
    shift_br = right_ar * left_br - right_ai * left_bi + right_br
    shift_bi = right_ai * left_br + right_ar * left_bi + right_bi
    return product_ar, product_ai, shift_br, shift_bi


@triton.jit
def _mul_rn_fp32(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_fp32(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_fp32(left, right):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _physical_state_order(value, step, active, n_steps: tl.constexpr, reverse: tl.constexpr):
    if reverse:
        index = tl.maximum(n_steps - 1 - step, 0)
        return tl.where(active, tl.gather(value, index, axis=0), 0.0)
    return tl.where(active, value, 0.0)


@triton.jit
def _normalized_lag_moment(
    states_real,
    states_imag,
    step,
    active,
    n_steps: tl.constexpr,
    epsilon: float,
    LAG: tl.constexpr,
):
    previous_index = tl.maximum(step - LAG, 0)
    previous_real = tl.gather(states_real, previous_index, axis=0)
    previous_imag = tl.gather(states_imag, previous_index, axis=0)
    paired = active & (step >= LAG)
    correlation_real = tl.sum(
        tl.where(
            paired,
            states_real * previous_real + states_imag * previous_imag,
            0.0,
        )
    )
    correlation_imag = tl.sum(
        tl.where(
            paired,
            states_imag * previous_real - states_real * previous_imag,
            0.0,
        )
    )
    current_energy = tl.sum(
        tl.where(paired, states_real * states_real + states_imag * states_imag, 0.0)
    )
    previous_energy = tl.sum(
        tl.where(
            paired,
            previous_real * previous_real + previous_imag * previous_imag,
            0.0,
        )
    )
    count = tl.maximum(n_steps - LAG, 1)
    inverse_count = 1.0 / count
    denominator = tl.maximum(
        tl.sqrt(current_energy * previous_energy) * inverse_count,
        epsilon,
    )
    valid = n_steps > LAG
    return (
        tl.where(valid, correlation_real * inverse_count / denominator, 0.0),
        tl.where(valid, correlation_imag * inverse_count / denominator, 0.0),
    )


@triton.jit
def _radial_log_lag_moment(
    states_real,
    states_imag,
    step,
    active,
    n_steps: tl.constexpr,
    LAG: tl.constexpr,
):
    previous_index = tl.maximum(step - LAG, 0)
    previous_real = tl.gather(states_real, previous_index, axis=0)
    previous_imag = tl.gather(states_imag, previous_index, axis=0)
    paired = active & (step >= LAG)
    inverse_count = 1.0 / tl.maximum(n_steps - LAG, 1)
    correlation_real = (
        tl.sum(
            tl.where(
                paired,
                states_real * previous_real + states_imag * previous_imag,
                0.0,
            )
        )
        * inverse_count
    )
    correlation_imag = (
        tl.sum(
            tl.where(
                paired,
                states_imag * previous_real - states_real * previous_imag,
                0.0,
            )
        )
        * inverse_count
    )
    valid = n_steps > LAG
    raw_real = tl.where(valid, correlation_real, 0.0)
    raw_imag = tl.where(valid, correlation_imag, 0.0)
    radius = tl.sqrt(
        tl.maximum(
            raw_real * raw_real + raw_imag * raw_imag,
            1.1754943508222875e-38,
        )
    )
    scale = libdevice.log1p(radius) / radius
    return scale * raw_real, scale * raw_imag


@triton.jit
def _parallel_lag124_forward_kernel(
    decay_real,
    decay_imag,
    packed_input,
    packed_states,
    moments,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    epsilon: float,
    reverse: tl.constexpr,
    STORE_STATES: tl.constexpr,
    RADIAL_LOG: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    lane = tl.program_id(0)
    batch = lane // modes
    mode = lane - batch * modes
    step = tl.arange(0, BLOCK_T)
    active = step < n_steps
    time_index = n_steps - 1 - step if reverse else step
    packed_offset = (batch * n_steps + time_index) * 2 * modes + mode

    fixed_ar = tl.load(decay_real + mode).to(tl.float32)
    fixed_ai = tl.load(decay_imag + mode).to(tl.float32)
    ar = tl.where(active, fixed_ar, 1.0)
    ai = tl.where(active, fixed_ai, 0.0)
    drive_real = tl.load(
        packed_input + packed_offset,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    drive_imag = tl.load(
        packed_input + packed_offset + modes,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    _prefix_ar, _prefix_ai, traversal_real, traversal_imag = tl.associative_scan(
        (ar, ai, drive_real, drive_imag),
        axis=0,
        combine_fn=_compose_complex_affine,
    )
    if STORE_STATES:
        tl.store(packed_states + packed_offset, traversal_real, mask=active)
        tl.store(packed_states + packed_offset + modes, traversal_imag, mask=active)

    # Moment reductions always observe the state sequence in physical time.
    states_real = _physical_state_order(
        traversal_real,
        step,
        active,
        n_steps,
        reverse,
    )
    states_imag = _physical_state_order(
        traversal_imag,
        step,
        active,
        n_steps,
        reverse,
    )
    energy = (
        tl.sum(tl.where(active, states_real * states_real + states_imag * states_imag, 0.0))
        / n_steps
    )
    if RADIAL_LOG:
        lag1_real, lag1_imag = _radial_log_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            LAG=1,
        )
        lag2_real, lag2_imag = _radial_log_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            LAG=2,
        )
        lag4_real, lag4_imag = _radial_log_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            LAG=4,
        )
    else:
        lag1_real, lag1_imag = _normalized_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            epsilon,
            LAG=1,
        )
        lag2_real, lag2_imag = _normalized_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            epsilon,
            LAG=2,
        )
        lag4_real, lag4_imag = _normalized_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            epsilon,
            LAG=4,
        )
    moment_base = batch * 7 * modes + mode
    tl.store(moments + moment_base, libdevice.log1p(energy))
    tl.store(moments + moment_base + modes, lag1_real)
    tl.store(moments + moment_base + 2 * modes, lag1_imag)
    tl.store(moments + moment_base + 3 * modes, lag2_real)
    tl.store(moments + moment_base + 4 * modes, lag2_imag)
    tl.store(moments + moment_base + 5 * modes, lag4_real)
    tl.store(moments + moment_base + 6 * modes, lag4_imag)


@triton.jit
def _parallel_lag124_excitation_forward_kernel(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    excitation_real,
    excitation_imag,
    packed_states,
    moments,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    epsilon: float,
    reverse: tl.constexpr,
    RADIAL_LOG: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    """Form the complex drive in registers before the parallel scan."""
    lane = tl.program_id(0)
    batch = lane // modes
    mode = lane - batch * modes
    step = tl.arange(0, BLOCK_T)
    active = step < n_steps
    time_index = n_steps - 1 - step if reverse else step
    excitation_offset = (batch * n_steps + time_index) * modes + mode
    packed_offset = (batch * n_steps + time_index) * 2 * modes + mode

    fixed_ar = tl.load(decay_real + mode).to(tl.float32)
    fixed_ai = tl.load(decay_imag + mode).to(tl.float32)
    fixed_gr = tl.load(gamma_real + mode).to(tl.float32)
    fixed_gi = tl.load(gamma_imag + mode).to(tl.float32)
    ar = tl.where(active, fixed_ar, 1.0)
    ai = tl.where(active, fixed_ai, 0.0)
    excitation_r = tl.load(
        excitation_real + excitation_offset,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    excitation_i = tl.load(
        excitation_imag + excitation_offset,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    drive_real_first = _mul_rn_fp32(fixed_gr, excitation_r)
    drive_real_second = _mul_rn_fp32(fixed_gi, excitation_i)
    drive_real = _sub_rn_fp32(drive_real_first, drive_real_second)
    drive_imag_first = _mul_rn_fp32(fixed_gr, excitation_i)
    drive_imag_second = _mul_rn_fp32(fixed_gi, excitation_r)
    drive_imag = _add_rn_fp32(drive_imag_first, drive_imag_second)
    _prefix_ar, _prefix_ai, traversal_real, traversal_imag = tl.associative_scan(
        (ar, ai, drive_real, drive_imag),
        axis=0,
        combine_fn=_compose_complex_affine,
    )
    tl.store(packed_states + packed_offset, traversal_real, mask=active)
    tl.store(packed_states + packed_offset + modes, traversal_imag, mask=active)

    states_real = _physical_state_order(
        traversal_real,
        step,
        active,
        n_steps,
        reverse,
    )
    states_imag = _physical_state_order(
        traversal_imag,
        step,
        active,
        n_steps,
        reverse,
    )
    energy = (
        tl.sum(tl.where(active, states_real * states_real + states_imag * states_imag, 0.0))
        / n_steps
    )
    if RADIAL_LOG:
        lag1_real, lag1_imag = _radial_log_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            LAG=1,
        )
        lag2_real, lag2_imag = _radial_log_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            LAG=2,
        )
        lag4_real, lag4_imag = _radial_log_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            LAG=4,
        )
    else:
        lag1_real, lag1_imag = _normalized_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            epsilon,
            LAG=1,
        )
        lag2_real, lag2_imag = _normalized_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            epsilon,
            LAG=2,
        )
        lag4_real, lag4_imag = _normalized_lag_moment(
            states_real,
            states_imag,
            step,
            active,
            n_steps,
            epsilon,
            LAG=4,
        )
    moment_base = batch * 7 * modes + mode
    tl.store(moments + moment_base, libdevice.log1p(energy))
    tl.store(moments + moment_base + modes, lag1_real)
    tl.store(moments + moment_base + 2 * modes, lag1_imag)
    tl.store(moments + moment_base + 3 * modes, lag2_real)
    tl.store(moments + moment_base + 4 * modes, lag2_imag)
    tl.store(moments + moment_base + 5 * modes, lag4_real)
    tl.store(moments + moment_base + 6 * modes, lag4_imag)


@triton.jit
def _lag_moment_state_vjp(
    states_real,
    states_imag,
    step,
    active,
    output_real_gradient,
    output_imag_gradient,
    n_steps: tl.constexpr,
    epsilon: float,
    LAG: tl.constexpr,
):
    previous_index = tl.maximum(step - LAG, 0)
    next_index = tl.minimum(step + LAG, n_steps - 1)
    previous_real = tl.gather(states_real, previous_index, axis=0)
    previous_imag = tl.gather(states_imag, previous_index, axis=0)
    next_real = tl.gather(states_real, next_index, axis=0)
    next_imag = tl.gather(states_imag, next_index, axis=0)
    has_previous = active & (step >= LAG)
    has_next = active & (step < n_steps - LAG)

    current_energy_sum = tl.sum(
        tl.where(
            has_previous,
            states_real * states_real + states_imag * states_imag,
            0.0,
        )
    )
    previous_energy_sum = tl.sum(
        tl.where(
            has_previous,
            previous_real * previous_real + previous_imag * previous_imag,
            0.0,
        )
    )
    correlation_real_sum = tl.sum(
        tl.where(
            has_previous,
            states_real * previous_real + states_imag * previous_imag,
            0.0,
        )
    )
    correlation_imag_sum = tl.sum(
        tl.where(
            has_previous,
            states_imag * previous_real - states_real * previous_imag,
            0.0,
        )
    )
    count = tl.maximum(n_steps - LAG, 1)
    inverse_count = 1.0 / count
    current_energy = current_energy_sum * inverse_count
    previous_energy = previous_energy_sum * inverse_count
    correlation_real = correlation_real_sum * inverse_count
    correlation_imag = correlation_imag_sum * inverse_count
    root = tl.sqrt(current_energy * previous_energy)
    denominator = tl.maximum(root, epsilon)
    valid_lag = n_steps > LAG
    real_weight = tl.where(valid_lag, output_real_gradient / denominator, 0.0)
    imag_weight = tl.where(valid_lag, output_imag_gradient / denominator, 0.0)
    weighted = output_real_gradient * correlation_real + output_imag_gradient * correlation_imag
    root_gradient = tl.where(
        valid_lag & (root > epsilon),
        -weighted / (denominator * denominator),
        0.0,
    )
    safe_root = tl.maximum(root, epsilon)
    current_gradient = 0.5 * root_gradient * previous_energy / safe_root
    previous_gradient = 0.5 * root_gradient * current_energy / safe_root

    grad_real = tl.where(
        has_previous,
        inverse_count * (real_weight * previous_real - imag_weight * previous_imag)
        + 2.0 * inverse_count * current_gradient * states_real,
        0.0,
    )
    grad_imag = tl.where(
        has_previous,
        inverse_count * (real_weight * previous_imag + imag_weight * previous_real)
        + 2.0 * inverse_count * current_gradient * states_imag,
        0.0,
    )
    grad_real += tl.where(
        has_next,
        inverse_count * (real_weight * next_real + imag_weight * next_imag)
        + 2.0 * inverse_count * previous_gradient * states_real,
        0.0,
    )
    grad_imag += tl.where(
        has_next,
        inverse_count * (real_weight * next_imag - imag_weight * next_real)
        + 2.0 * inverse_count * previous_gradient * states_imag,
        0.0,
    )
    return grad_real, grad_imag


@triton.jit
def _radial_log_lag_moment_state_vjp(
    states_real,
    states_imag,
    step,
    active,
    output_real_gradient,
    output_imag_gradient,
    n_steps: tl.constexpr,
    LAG: tl.constexpr,
):
    """Apply the radial-log VJP and the raw complex-correlation VJP."""
    previous_index = tl.maximum(step - LAG, 0)
    next_index = tl.minimum(step + LAG, n_steps - 1)
    previous_real = tl.gather(states_real, previous_index, axis=0)
    previous_imag = tl.gather(states_imag, previous_index, axis=0)
    next_real = tl.gather(states_real, next_index, axis=0)
    next_imag = tl.gather(states_imag, next_index, axis=0)
    has_previous = active & (step >= LAG)
    has_next = active & (step < n_steps - LAG)
    count = tl.maximum(n_steps - LAG, 1)
    inverse_count = 1.0 / count
    correlation_real = (
        tl.sum(
            tl.where(
                has_previous,
                states_real * previous_real + states_imag * previous_imag,
                0.0,
            )
        )
        * inverse_count
    )
    correlation_imag = (
        tl.sum(
            tl.where(
                has_previous,
                states_imag * previous_real - states_real * previous_imag,
                0.0,
            )
        )
        * inverse_count
    )

    tiny = 1.1754943508222875e-38
    radius_squared = correlation_real * correlation_real + correlation_imag * correlation_imag
    radius = tl.sqrt(tl.maximum(radius_squared, tiny))
    radial_scale = libdevice.log1p(radius) / radius
    radial_slope_over_radius = tl.where(
        radius_squared > tiny,
        (radius / (1.0 + radius) - libdevice.log1p(radius)) / (radius * radius * radius),
        0.0,
    )
    projection = output_real_gradient * correlation_real + output_imag_gradient * correlation_imag
    raw_real_gradient = (
        radial_scale * output_real_gradient
        + radial_slope_over_radius * projection * correlation_real
    )
    raw_imag_gradient = (
        radial_scale * output_imag_gradient
        + radial_slope_over_radius * projection * correlation_imag
    )
    valid_lag = n_steps > LAG
    real_weight = tl.where(valid_lag, raw_real_gradient, 0.0)
    imag_weight = tl.where(valid_lag, raw_imag_gradient, 0.0)

    grad_real = tl.where(
        has_previous,
        inverse_count * (real_weight * previous_real - imag_weight * previous_imag),
        0.0,
    )
    grad_imag = tl.where(
        has_previous,
        inverse_count * (real_weight * previous_imag + imag_weight * previous_real),
        0.0,
    )
    grad_real += tl.where(
        has_next,
        inverse_count * (real_weight * next_real + imag_weight * next_imag),
        0.0,
    )
    grad_imag += tl.where(
        has_next,
        inverse_count * (real_weight * next_imag - imag_weight * next_real),
        0.0,
    )
    return grad_real, grad_imag


@triton.jit
def _adjoint_state_order(
    value,
    step,
    active,
    n_steps: tl.constexpr,
    reverse: tl.constexpr,
):
    if reverse:
        return tl.where(active, value, 0.0)
    index = tl.maximum(n_steps - 1 - step, 0)
    return tl.where(active, tl.gather(value, index, axis=0), 0.0)


@triton.jit
def _parallel_lag124_backward_kernel(
    decay_real,
    decay_imag,
    packed_states,
    direct_grad_packed_states,
    grad_moments,
    grad_packed_input,
    grad_decay_per_batch,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    epsilon: float,
    reverse: tl.constexpr,
    has_direct_state_grad: tl.constexpr,
    RADIAL_LOG: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    lane = tl.program_id(0)
    batch = lane // modes
    mode = lane - batch * modes
    step = tl.arange(0, BLOCK_T)
    active = step < n_steps
    physical_offset = (batch * n_steps + step) * 2 * modes + mode
    states_real = tl.load(
        packed_states + physical_offset,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    states_imag = tl.load(
        packed_states + physical_offset + modes,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    moment_base = batch * 7 * modes + mode
    energy_gradient = tl.load(grad_moments + moment_base).to(tl.float32)
    energy = (
        tl.sum(tl.where(active, states_real * states_real + states_imag * states_imag, 0.0))
        / n_steps
    )
    energy_scale = (2.0 / n_steps) * energy_gradient / (1.0 + energy)
    state_grad_real = energy_scale * states_real
    state_grad_imag = energy_scale * states_imag

    if RADIAL_LOG:
        lag_real, lag_imag = _radial_log_lag_moment_state_vjp(
            states_real,
            states_imag,
            step,
            active,
            tl.load(grad_moments + moment_base + modes).to(tl.float32),
            tl.load(grad_moments + moment_base + 2 * modes).to(tl.float32),
            n_steps,
            LAG=1,
        )
    else:
        lag_real, lag_imag = _lag_moment_state_vjp(
            states_real,
            states_imag,
            step,
            active,
            tl.load(grad_moments + moment_base + modes).to(tl.float32),
            tl.load(grad_moments + moment_base + 2 * modes).to(tl.float32),
            n_steps,
            epsilon,
            LAG=1,
        )
    state_grad_real += lag_real
    state_grad_imag += lag_imag
    if RADIAL_LOG:
        lag_real, lag_imag = _radial_log_lag_moment_state_vjp(
            states_real,
            states_imag,
            step,
            active,
            tl.load(grad_moments + moment_base + 3 * modes).to(tl.float32),
            tl.load(grad_moments + moment_base + 4 * modes).to(tl.float32),
            n_steps,
            LAG=2,
        )
    else:
        lag_real, lag_imag = _lag_moment_state_vjp(
            states_real,
            states_imag,
            step,
            active,
            tl.load(grad_moments + moment_base + 3 * modes).to(tl.float32),
            tl.load(grad_moments + moment_base + 4 * modes).to(tl.float32),
            n_steps,
            epsilon,
            LAG=2,
        )
    state_grad_real += lag_real
    state_grad_imag += lag_imag
    if RADIAL_LOG:
        lag_real, lag_imag = _radial_log_lag_moment_state_vjp(
            states_real,
            states_imag,
            step,
            active,
            tl.load(grad_moments + moment_base + 5 * modes).to(tl.float32),
            tl.load(grad_moments + moment_base + 6 * modes).to(tl.float32),
            n_steps,
            LAG=4,
        )
    else:
        lag_real, lag_imag = _lag_moment_state_vjp(
            states_real,
            states_imag,
            step,
            active,
            tl.load(grad_moments + moment_base + 5 * modes).to(tl.float32),
            tl.load(grad_moments + moment_base + 6 * modes).to(tl.float32),
            n_steps,
            epsilon,
            LAG=4,
        )
    state_grad_real += lag_real
    state_grad_imag += lag_imag

    if has_direct_state_grad:
        state_grad_real += tl.load(
            direct_grad_packed_states + physical_offset,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        state_grad_imag += tl.load(
            direct_grad_packed_states + physical_offset + modes,
            mask=active,
            other=0.0,
        ).to(tl.float32)

    adjoint_real = _adjoint_state_order(
        state_grad_real,
        step,
        active,
        n_steps,
        reverse,
    )
    adjoint_imag = _adjoint_state_order(
        state_grad_imag,
        step,
        active,
        n_steps,
        reverse,
    )
    fixed_ar = tl.load(decay_real + mode).to(tl.float32)
    fixed_ai = tl.load(decay_imag + mode).to(tl.float32)
    ar = tl.where(active, fixed_ar, 1.0)
    ai = tl.where(active, -fixed_ai, 0.0)
    _prefix_ar, _prefix_ai, lambda_real, lambda_imag = tl.associative_scan(
        (ar, ai, adjoint_real, adjoint_imag),
        axis=0,
        combine_fn=_compose_complex_affine,
    )

    adjoint_time = step if reverse else n_steps - 1 - step
    adjoint_offset = (batch * n_steps + adjoint_time) * 2 * modes + mode
    tl.store(grad_packed_input + adjoint_offset, lambda_real, mask=active)
    tl.store(grad_packed_input + adjoint_offset + modes, lambda_imag, mask=active)

    previous_time = adjoint_time + 1 if reverse else adjoint_time - 1
    has_previous = active & (adjoint_time < n_steps - 1 if reverse else adjoint_time > 0)
    previous_offset = (batch * n_steps + previous_time) * 2 * modes + mode
    previous_real = tl.load(
        packed_states + previous_offset,
        mask=has_previous,
        other=0.0,
    ).to(tl.float32)
    previous_imag = tl.load(
        packed_states + previous_offset + modes,
        mask=has_previous,
        other=0.0,
    ).to(tl.float32)
    decay_gradient_real = tl.where(
        active,
        lambda_real * previous_real + lambda_imag * previous_imag,
        0.0,
    )
    decay_gradient_imag = tl.where(
        active,
        -lambda_real * previous_imag + lambda_imag * previous_real,
        0.0,
    )
    summary_offset = batch * 2 * modes + mode
    tl.store(grad_decay_per_batch + summary_offset, tl.sum(decay_gradient_real))
    tl.store(
        grad_decay_per_batch + summary_offset + modes,
        tl.sum(decay_gradient_imag),
    )


@triton.jit
def _reduce_batch_gradient_kernel(
    grad_decay_per_batch,
    grad_decay_real,
    grad_decay_imag,
    batch_size: tl.constexpr,
    modes: tl.constexpr,
    BLOCK_M: tl.constexpr,
) -> None:
    mode = tl.arange(0, BLOCK_M)
    valid = mode < modes
    total_real = tl.zeros((BLOCK_M,), tl.float32)
    total_imag = tl.zeros((BLOCK_M,), tl.float32)
    for batch in range(batch_size):
        offset = batch * 2 * modes + mode
        total_real += tl.load(
            grad_decay_per_batch + offset,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        total_imag += tl.load(
            grad_decay_per_batch + offset + modes,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
    tl.store(grad_decay_real + mode, total_real, mask=valid)
    tl.store(grad_decay_imag + mode, total_imag, mask=valid)


def _validate_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    epsilon: float,
    num_warps: int,
) -> None:
    if decay_real.ndim != 1 or decay_real.numel() == 0:
        message = "parallel lag124 decay must have shape [modes]"
        raise ValueError(message)
    if decay_imag.shape != decay_real.shape:
        message = "parallel lag124 decay tensors must match"
        raise ValueError(message)
    if (
        packed_input.ndim != 3
        or packed_input.shape[1] == 0
        or packed_input.shape[-1] != 2 * decay_real.numel()
    ):
        message = "packed input must have shape [batch, steps, 2*modes]"
        raise ValueError(message)
    if packed_input.shape[1] > _MAX_STEPS:
        message = f"parallel lag124 training supports at most {_MAX_STEPS} steps"
        raise ValueError(message)
    tensors = (decay_real, decay_imag, packed_input)
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        message = "parallel lag124 training supports FP32 tensors only"
        raise TypeError(message)
    if any(tensor.device != packed_input.device for tensor in tensors):
        message = "parallel lag124 tensors must share one device"
        raise ValueError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)
    if num_warps not in _VALID_WARPS:
        message = f"num_warps must be one of {_VALID_WARPS}"
        raise ValueError(message)


def _reference_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    has_direct_state_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    direct_gradient = direct_grad_packed_states if has_direct_state_grad else None
    return _serial_reference_backward(
        decay_real,
        decay_imag,
        packed_states,
        direct_gradient,
        grad_moments,
        reverse=reverse,
        epsilon=epsilon,
    )


def _reference_radial_log_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    *,
    reverse: bool,
    has_direct_state_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reconstruct the drive and replay the differentiable CPU reference."""
    with torch.inference_mode(False), torch.enable_grad():
        modes = decay_real.numel()
        states_real, states_imag = packed_states.detach().clone().chunk(2, dim=-1)
        zeros = torch.zeros_like(states_real[:, :1])
        if reverse:
            previous_real = torch.cat((states_real[:, 1:], zeros), dim=1)
            previous_imag = torch.cat((states_imag[:, 1:], zeros), dim=1)
        else:
            previous_real = torch.cat((zeros, states_real[:, :-1]), dim=1)
            previous_imag = torch.cat((zeros, states_imag[:, :-1]), dim=1)
        fixed_real = decay_real.detach().clone().view(1, 1, modes)
        fixed_imag = decay_imag.detach().clone().view(1, 1, modes)
        input_real = states_real - fixed_real * previous_real + fixed_imag * previous_imag
        input_imag = states_imag - fixed_imag * previous_real - fixed_real * previous_imag
        packed_input = torch.cat((input_real, input_imag), dim=-1).detach()
        active_decay_real = decay_real.detach().clone().requires_grad_()
        active_decay_imag = decay_imag.detach().clone().requires_grad_()
        active_input = packed_input.requires_grad_()
        replay_states, replay_moments = (
            reference_static_radial_log_recurrence_lag124_moments_packed_io(
                active_decay_real,
                active_decay_imag,
                active_input,
                reverse=reverse,
            )
        )
        direct_gradient = (
            direct_grad_packed_states if has_direct_state_grad else torch.zeros_like(replay_states)
        )
        gradients = torch.autograd.grad(
            (replay_states, replay_moments),
            (active_decay_real, active_decay_imag, active_input),
            grad_outputs=(direct_gradient, grad_moments),
        )
    return gradients


@triton_op("lnet::pac_parallel_static_recurrence_lag124_forward_impl", mutates_args={})
def _forward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    _validate_inputs(decay_real, decay_imag, packed_input, epsilon, num_warps)
    if not packed_input.is_cuda:
        return reference_static_recurrence_lag124_moments_packed_io(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
        )
    drive = packed_input.contiguous()
    states = torch.empty_like(drive)
    batch, n_steps, packed_modes = drive.shape
    modes = packed_modes // 2
    moments = drive.new_empty((batch, 7 * modes))
    wrap_triton(_parallel_lag124_forward_kernel)[(batch * modes,)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        drive,
        states,
        moments,
        n_steps,
        modes,
        epsilon,
        reverse,
        STORE_STATES=True,
        RADIAL_LOG=False,
        BLOCK_T=triton.next_power_of_2(n_steps),
        num_warps=num_warps,
    )
    return states, moments


@triton_op(
    "lnet::pac_parallel_static_recurrence_lag124_moments_only_inference_impl",
    mutates_args={},
)
def _moments_only_inference_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
) -> Tensor:
    _validate_inputs(decay_real, decay_imag, packed_input, epsilon, num_warps)
    if not packed_input.is_cuda:
        return reference_static_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
        )
    drive = packed_input.contiguous()
    batch, n_steps, packed_modes = drive.shape
    modes = packed_modes // 2
    moments = drive.new_empty((batch, 7 * modes))
    wrap_triton(_parallel_lag124_forward_kernel)[(batch * modes,)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        drive,
        moments,
        moments,
        n_steps,
        modes,
        epsilon,
        reverse,
        STORE_STATES=False,
        RADIAL_LOG=False,
        BLOCK_T=triton.next_power_of_2(n_steps),
        num_warps=num_warps,
    )
    return moments


@torch.library.custom_op(
    "lnet::pac_parallel_static_recurrence_lag124_moments_only_inference",
    mutates_args=(),
)
def _moments_only_inference_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
) -> Tensor:
    return _moments_only_inference_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        epsilon,
        num_warps,
    )


@_moments_only_inference_opaque.register_fake
def _moments_only_inference_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
) -> Tensor:
    del decay_imag, reverse, epsilon, num_warps
    modes = decay_real.numel()
    return packed_input.new_empty((packed_input.shape[0], 7 * modes))


@triton_op(
    "lnet::pac_parallel_static_radial_log_recurrence_lag124_packed_io_impl",
    mutates_args={},
)
def _radial_log_packed_io_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    _validate_inputs(
        decay_real,
        decay_imag,
        packed_input,
        _DEFAULT_EPSILON,
        num_warps,
    )
    if not packed_input.is_cuda:
        return reference_static_radial_log_recurrence_lag124_moments_packed_io(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
        )
    drive = packed_input.contiguous()
    batch, n_steps, packed_modes = drive.shape
    modes = packed_modes // 2
    states = torch.empty_like(drive)
    moments = drive.new_empty((batch, 7 * modes))
    wrap_triton(_parallel_lag124_forward_kernel)[(batch * modes,)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        drive,
        states,
        moments,
        n_steps,
        modes,
        _DEFAULT_EPSILON,
        reverse,
        STORE_STATES=True,
        RADIAL_LOG=True,
        BLOCK_T=triton.next_power_of_2(n_steps),
        num_warps=num_warps,
    )
    return states, moments


@triton_op(
    "lnet::pac_parallel_static_radial_log_recurrence_lag124_moments_only_impl",
    mutates_args={},
)
def _radial_log_moments_only_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> Tensor:
    _validate_inputs(
        decay_real,
        decay_imag,
        packed_input,
        _DEFAULT_EPSILON,
        num_warps,
    )
    if not packed_input.is_cuda:
        return reference_static_radial_log_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
        )
    drive = packed_input.contiguous()
    batch, n_steps, packed_modes = drive.shape
    modes = packed_modes // 2
    moments = drive.new_empty((batch, 7 * modes))
    wrap_triton(_parallel_lag124_forward_kernel)[(batch * modes,)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        drive,
        moments,
        moments,
        n_steps,
        modes,
        _DEFAULT_EPSILON,
        reverse,
        STORE_STATES=False,
        RADIAL_LOG=True,
        BLOCK_T=triton.next_power_of_2(n_steps),
        num_warps=num_warps,
    )
    return moments


@torch.library.custom_op(
    "lnet::pac_parallel_static_radial_log_recurrence_lag124_packed_io",
    mutates_args=(),
)
def _radial_log_packed_io_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    return _radial_log_packed_io_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        num_warps,
    )


@_radial_log_packed_io_opaque.register_fake
def _radial_log_packed_io_fake(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    del decay_imag, reverse, num_warps
    return (
        torch.empty_like(packed_input),
        packed_input.new_empty((packed_input.shape[0], 7 * decay_real.numel())),
    )


@torch.library.custom_op(
    "lnet::pac_parallel_static_radial_log_recurrence_lag124_moments_only",
    mutates_args=(),
)
def _radial_log_moments_only_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> Tensor:
    return _radial_log_moments_only_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        num_warps,
    )


@_radial_log_moments_only_opaque.register_fake
def _radial_log_moments_only_fake(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> Tensor:
    del decay_imag, reverse, num_warps
    return packed_input.new_empty((packed_input.shape[0], 7 * decay_real.numel()))


@triton_op("lnet::pac_parallel_static_recurrence_lag124_backward_impl", mutates_args={})
def _backward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
    has_direct_state_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_inputs(decay_real, decay_imag, packed_states, epsilon, num_warps)
    if direct_grad_packed_states.shape != packed_states.shape:
        message = "direct packed-state gradient must match packed states"
        raise ValueError(message)
    modes = decay_real.numel()
    if grad_moments.shape != (packed_states.shape[0], 7 * modes):
        message = "lag124 moment gradient has an invalid shape"
        raise ValueError(message)
    if not packed_states.is_cuda:
        return _reference_backward(
            decay_real,
            decay_imag,
            packed_states,
            direct_grad_packed_states,
            grad_moments,
            reverse=reverse,
            epsilon=epsilon,
            has_direct_state_grad=has_direct_state_grad,
        )
    states = packed_states.contiguous()
    direct_gradient = direct_grad_packed_states.contiguous()
    moment_gradient = grad_moments.contiguous()
    batch, n_steps, packed_modes = states.shape
    modes = packed_modes // 2
    grad_input = torch.empty_like(states)
    per_batch = torch.empty((batch, 2 * modes), dtype=torch.float32, device=states.device)
    grad_decay_real = torch.empty_like(decay_real)
    grad_decay_imag = torch.empty_like(decay_imag)
    wrap_triton(_parallel_lag124_backward_kernel)[(batch * modes,)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        states,
        direct_gradient,
        moment_gradient,
        grad_input,
        per_batch,
        n_steps,
        modes,
        epsilon,
        reverse,
        has_direct_state_grad,
        RADIAL_LOG=False,
        BLOCK_T=triton.next_power_of_2(n_steps),
        num_warps=num_warps,
    )
    wrap_triton(_reduce_batch_gradient_kernel)[(1,)](
        per_batch,
        grad_decay_real,
        grad_decay_imag,
        batch,
        modes,
        BLOCK_M=triton.next_power_of_2(modes),
        num_warps=1,
    )
    return grad_decay_real, grad_decay_imag, grad_input


@triton_op(
    "lnet::pac_parallel_static_radial_log_recurrence_lag124_backward_impl",
    mutates_args={},
)
def _radial_log_backward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    num_warps: int,
    has_direct_state_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_inputs(
        decay_real,
        decay_imag,
        packed_states,
        _DEFAULT_EPSILON,
        num_warps,
    )
    if direct_grad_packed_states.shape != packed_states.shape:
        message = "direct packed-state gradient must match packed states"
        raise ValueError(message)
    modes = decay_real.numel()
    if grad_moments.shape != (packed_states.shape[0], 7 * modes):
        message = "radial-log lag124 moment gradient has an invalid shape"
        raise ValueError(message)
    if not packed_states.is_cuda:
        return _reference_radial_log_backward(
            decay_real,
            decay_imag,
            packed_states,
            direct_grad_packed_states,
            grad_moments,
            reverse=reverse,
            has_direct_state_grad=has_direct_state_grad,
        )
    states = packed_states.contiguous()
    direct_gradient = direct_grad_packed_states.contiguous()
    moment_gradient = grad_moments.contiguous()
    batch, n_steps, packed_modes = states.shape
    modes = packed_modes // 2
    grad_input = torch.empty_like(states)
    per_batch = torch.empty(
        (batch, 2 * modes),
        dtype=torch.float32,
        device=states.device,
    )
    grad_decay_real = torch.empty_like(decay_real)
    grad_decay_imag = torch.empty_like(decay_imag)
    wrap_triton(_parallel_lag124_backward_kernel)[(batch * modes,)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        states,
        direct_gradient,
        moment_gradient,
        grad_input,
        per_batch,
        n_steps,
        modes,
        _DEFAULT_EPSILON,
        reverse,
        has_direct_state_grad,
        RADIAL_LOG=True,
        BLOCK_T=triton.next_power_of_2(n_steps),
        num_warps=num_warps,
    )
    wrap_triton(_reduce_batch_gradient_kernel)[(1,)](
        per_batch,
        grad_decay_real,
        grad_decay_imag,
        batch,
        modes,
        BLOCK_M=triton.next_power_of_2(modes),
        num_warps=1,
    )
    return grad_decay_real, grad_decay_imag, grad_input


@torch.library.custom_op(
    "lnet::pac_parallel_static_recurrence_lag124_backward",
    mutates_args=(),
)
def _backward_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
    has_direct_state_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return _backward_impl(
        decay_real,
        decay_imag,
        packed_states,
        direct_grad_packed_states,
        grad_moments,
        reverse,
        epsilon,
        num_warps,
        has_direct_state_grad,
    )


@_backward_opaque.register_fake
def _backward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
    has_direct_state_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    del (
        direct_grad_packed_states,
        grad_moments,
        reverse,
        epsilon,
        num_warps,
        has_direct_state_grad,
    )
    return (
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(packed_states),
    )


@torch.library.custom_op(
    "lnet::pac_parallel_static_radial_log_recurrence_lag124_backward",
    mutates_args=(),
)
def _radial_log_backward_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    num_warps: int,
    has_direct_state_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return _radial_log_backward_impl(
        decay_real,
        decay_imag,
        packed_states,
        direct_grad_packed_states,
        grad_moments,
        reverse,
        num_warps,
        has_direct_state_grad,
    )


@_radial_log_backward_opaque.register_fake
def _radial_log_backward_fake(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    num_warps: int,
    has_direct_state_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    del (
        direct_grad_packed_states,
        grad_moments,
        reverse,
        num_warps,
        has_direct_state_grad,
    )
    return (
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(packed_states),
    )


@torch.library.custom_op(
    "lnet::pac_parallel_static_recurrence_lag124_training",
    mutates_args=(),
)
def _forward_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    return _forward_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        epsilon,
        num_warps,
    )


@_forward_opaque.register_fake
def _forward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    del decay_real, decay_imag, reverse, epsilon, num_warps
    modes = packed_input.shape[-1] // 2
    moments = packed_input.new_empty((packed_input.shape[0], 7 * modes))
    return torch.empty_like(packed_input), moments


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, bool, float, int],
    output: tuple[Tensor, Tensor],
) -> None:
    decay_real, decay_imag, _packed_input, reverse, epsilon, num_warps = inputs
    packed_states, _moments = output
    ctx.reverse = reverse
    ctx.epsilon = epsilon
    ctx.num_warps = num_warps
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(decay_real, decay_imag, packed_states)


def _backward(
    ctx: _AutogradContext,
    grad_packed_states: Tensor | None,
    grad_moments: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
    decay_real, decay_imag, packed_states = ctx.saved_tensors
    modes = decay_real.numel()
    has_direct_state_grad = grad_packed_states is not None
    if grad_packed_states is None:
        # The pointer is a compile-time-dead dummy in the moments-only path.
        grad_packed_states = packed_states
    if grad_moments is None:
        grad_moments = packed_states.new_zeros((packed_states.shape[0], 7 * modes))
    gradients = _backward_opaque(
        decay_real,
        decay_imag,
        packed_states,
        grad_packed_states,
        grad_moments,
        ctx.reverse,
        ctx.epsilon,
        ctx.num_warps,
        has_direct_state_grad,
    )
    return *gradients, None, None, None


torch.library.register_autograd(
    "lnet::pac_parallel_static_recurrence_lag124_training",
    _backward,
    setup_context=_setup_context,
)


def parallel_static_recurrence_lag124_moments_packed_io_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _DEFAULT_EPSILON,
    num_warps: int = 4,
) -> tuple[Tensor, Tensor]:
    """Return packed states and seven physical-time lag moments."""
    return _forward_opaque(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        epsilon,
        num_warps,
    )


def parallel_static_recurrence_lag124_moments_only_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _DEFAULT_EPSILON,
    num_warps: int = 4,
) -> Tensor:
    """Return moments while privately retaining forward states for the VJP."""
    _packed_states, moments = _forward_opaque(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        epsilon,
        num_warps,
    )
    return moments


@torch.library.custom_op(
    "lnet::pac_parallel_static_radial_log_recurrence_lag124_training",
    mutates_args=(),
)
def _radial_log_forward_training_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    return _radial_log_packed_io_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        num_warps,
    )


@_radial_log_forward_training_opaque.register_fake
def _radial_log_forward_training_fake(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    del decay_imag, reverse, num_warps
    return (
        torch.empty_like(packed_input),
        packed_input.new_empty((packed_input.shape[0], 7 * decay_real.numel())),
    )


def _setup_radial_log_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, bool, int],
    output: tuple[Tensor, Tensor],
) -> None:
    decay_real, decay_imag, _packed_input, reverse, num_warps = inputs
    packed_states, _moments = output
    ctx.reverse = reverse
    ctx.num_warps = num_warps
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(decay_real, decay_imag, packed_states)


def _radial_log_backward(
    ctx: _AutogradContext,
    grad_packed_states: Tensor | None,
    grad_moments: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, None, None]:
    decay_real, decay_imag, packed_states = ctx.saved_tensors
    modes = decay_real.numel()
    has_direct_state_grad = grad_packed_states is not None
    if grad_packed_states is None:
        grad_packed_states = packed_states
    if grad_moments is None:
        grad_moments = packed_states.new_zeros((packed_states.shape[0], 7 * modes))
    gradients = _radial_log_backward_opaque(
        decay_real,
        decay_imag,
        packed_states,
        grad_packed_states,
        grad_moments,
        ctx.reverse,
        ctx.num_warps,
        has_direct_state_grad,
    )
    return *gradients, None, None


torch.library.register_autograd(
    "lnet::pac_parallel_static_radial_log_recurrence_lag124_training",
    _radial_log_backward,
    setup_context=_setup_radial_log_context,
)


def parallel_static_radial_log_recurrence_lag124_moments_packed_io_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    num_warps: int = 4,
) -> tuple[Tensor, Tensor]:
    """Return packed states and radial-log raw lag-(1,2,4) moments."""
    return _radial_log_forward_training_opaque(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        num_warps,
    )


def parallel_static_radial_log_recurrence_lag124_moments_only_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    num_warps: int = 4,
) -> Tensor:
    """Return radial-log moments while privately retaining states for the VJP."""
    _packed_states, moments = _radial_log_forward_training_opaque(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        num_warps,
    )
    return moments


def _validate_excitation_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    epsilon: float,
    num_warps: int,
) -> None:
    if decay_real.ndim != 1 or decay_real.numel() == 0:
        message = "parallel excitation lag124 decay must have shape [modes]"
        raise ValueError(message)
    if decay_imag.shape != decay_real.shape:
        message = "parallel excitation lag124 decay tensors must match"
        raise ValueError(message)
    if gamma_real.shape != decay_real.shape or gamma_imag.shape != decay_real.shape:
        message = "parallel excitation lag124 gamma tensors must match decay"
        raise ValueError(message)
    if (
        excitation_real.ndim != 3
        or excitation_real.shape[1] == 0
        or excitation_real.shape[-1] != decay_real.numel()
        or excitation_imag.shape != excitation_real.shape
    ):
        message = "excitation inputs must have matching [batch, steps, modes] shapes"
        raise ValueError(message)
    if excitation_real.shape[1] > _MAX_STEPS:
        message = f"parallel lag124 training supports at most {_MAX_STEPS} steps"
        raise ValueError(message)
    tensors = (
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
    )
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        message = "parallel excitation lag124 training supports FP32 tensors only"
        raise TypeError(message)
    if any(tensor.device != excitation_real.device for tensor in tensors):
        message = "parallel excitation lag124 tensors must share one device"
        raise ValueError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)
    if num_warps not in _VALID_WARPS:
        message = f"num_warps must be one of {_VALID_WARPS}"
        raise ValueError(message)


def _reference_excitation_forward(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    *,
    reverse: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    input_real = gamma_real * excitation_real - gamma_imag * excitation_imag
    input_imag = gamma_real * excitation_imag + gamma_imag * excitation_real
    return reference_static_recurrence_lag124_moments_packed_io(
        decay_real,
        decay_imag,
        torch.cat((input_real, input_imag), dim=-1),
        reverse=reverse,
        epsilon=epsilon,
    )


@triton_op("lnet::pac_parallel_static_excitation_lag124_forward_impl", mutates_args={})
def _excitation_forward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    _validate_excitation_inputs(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
        epsilon,
        num_warps,
    )
    if not excitation_real.is_cuda:
        return _reference_excitation_forward(
            decay_real,
            decay_imag,
            gamma_real,
            gamma_imag,
            excitation_real,
            excitation_imag,
            reverse=reverse,
            epsilon=epsilon,
        )
    real = excitation_real.contiguous()
    imag = excitation_imag.contiguous()
    batch, n_steps, modes = real.shape
    packed_states = real.new_empty((batch, n_steps, 2 * modes))
    moments = real.new_empty((batch, 7 * modes))
    wrap_triton(_parallel_lag124_excitation_forward_kernel)[(batch * modes,)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        gamma_real.contiguous(),
        gamma_imag.contiguous(),
        real,
        imag,
        packed_states,
        moments,
        n_steps,
        modes,
        epsilon,
        reverse,
        RADIAL_LOG=False,
        BLOCK_T=triton.next_power_of_2(n_steps),
        num_warps=num_warps,
    )
    return packed_states, moments


@torch.library.custom_op(
    "lnet::pac_parallel_static_excitation_lag124_training",
    mutates_args=(),
)
def _excitation_forward_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    return _excitation_forward_impl(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
        reverse,
        epsilon,
        num_warps,
    )


@_excitation_forward_opaque.register_fake
def _excitation_forward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    reverse: bool,
    epsilon: float,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    del decay_imag, gamma_real, gamma_imag, excitation_imag, reverse, epsilon, num_warps
    modes = decay_real.numel()
    packed_states = excitation_real.new_empty(
        (excitation_real.shape[0], excitation_real.shape[1], 2 * modes)
    )
    moments = excitation_real.new_empty((excitation_real.shape[0], 7 * modes))
    return packed_states, moments


def _setup_excitation_context(
    ctx: _ExcitationAutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, float, int],
    output: tuple[Tensor, Tensor],
) -> None:
    (
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
        reverse,
        epsilon,
        num_warps,
    ) = inputs
    packed_states, _moments = output
    ctx.reverse = reverse
    ctx.epsilon = epsilon
    ctx.num_warps = num_warps
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
        packed_states,
    )


def _backward_excitation(
    ctx: _ExcitationAutogradContext,
    grad_packed_states: Tensor | None,
    grad_moments: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, None, None, None]:
    (
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
        packed_states,
    ) = ctx.saved_tensors
    modes = decay_real.numel()
    has_direct_state_grad = grad_packed_states is not None
    if grad_packed_states is None:
        grad_packed_states = packed_states
    if grad_moments is None:
        grad_moments = packed_states.new_zeros((packed_states.shape[0], 7 * modes))
    grad_decay_real, grad_decay_imag, grad_packed_input = _backward_opaque(
        decay_real,
        decay_imag,
        packed_states,
        grad_packed_states,
        grad_moments,
        ctx.reverse,
        ctx.epsilon,
        ctx.num_warps,
        has_direct_state_grad,
    )
    grad_input_real, grad_input_imag = grad_packed_input.chunk(2, dim=-1)
    grad_gamma_real = (grad_input_real * excitation_real + grad_input_imag * excitation_imag).sum(
        dim=(0, 1)
    )
    grad_gamma_imag = (grad_input_imag * excitation_real - grad_input_real * excitation_imag).sum(
        dim=(0, 1)
    )
    grad_excitation_real = gamma_real * grad_input_real + gamma_imag * grad_input_imag
    grad_excitation_imag = gamma_real * grad_input_imag - gamma_imag * grad_input_real
    return (
        grad_decay_real,
        grad_decay_imag,
        grad_gamma_real,
        grad_gamma_imag,
        grad_excitation_real,
        grad_excitation_imag,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "lnet::pac_parallel_static_excitation_lag124_training",
    _backward_excitation,
    setup_context=_setup_excitation_context,
)


def parallel_static_excitation_recurrence_lag124_moments_packed_io_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _DEFAULT_EPSILON,
    num_warps: int = 4,
) -> tuple[Tensor, Tensor]:
    """Fuse static gamma rotation and packing into the parallel writer scan."""
    return _excitation_forward_opaque(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
        reverse,
        epsilon,
        num_warps,
    )


def parallel_static_excitation_recurrence_lag124_moments_only_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _DEFAULT_EPSILON,
    num_warps: int = 4,
) -> Tensor:
    """Fuse static gamma rotation and packing into the read-only scan."""
    _packed_states, moments = _excitation_forward_opaque(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        excitation_real,
        excitation_imag,
        reverse,
        epsilon,
        num_warps,
    )
    return moments


def parallel_static_recurrence_lag124_moments_only_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _DEFAULT_EPSILON,
    num_warps: int = 4,
) -> Tensor:
    """Return parallel lag moments without allocating or storing recurrence states."""
    _validate_inputs(decay_real, decay_imag, packed_input, epsilon, num_warps)
    needs_gradients = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (decay_real, decay_imag, packed_input)
    )
    if needs_gradients:
        return reference_static_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
        )
    return _moments_only_inference_opaque(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        epsilon,
        num_warps,
    )


def parallel_static_radial_log_recurrence_lag124_moments_packed_io_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    num_warps: int = 4,
) -> tuple[Tensor, Tensor]:
    """Return time-parallel packed states and radial-log raw moments."""
    _validate_inputs(
        decay_real,
        decay_imag,
        packed_input,
        _DEFAULT_EPSILON,
        num_warps,
    )
    needs_gradients = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (decay_real, decay_imag, packed_input)
    )
    if needs_gradients:
        return reference_static_radial_log_recurrence_lag124_moments_packed_io(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
        )
    return _radial_log_packed_io_opaque(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        num_warps,
    )


def parallel_static_radial_log_recurrence_lag124_moments_only_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    num_warps: int = 4,
) -> Tensor:
    """Return time-parallel radial-log raw moments without storing states."""
    _validate_inputs(
        decay_real,
        decay_imag,
        packed_input,
        _DEFAULT_EPSILON,
        num_warps,
    )
    needs_gradients = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (decay_real, decay_imag, packed_input)
    )
    if needs_gradients:
        return reference_static_radial_log_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
        )
    return _radial_log_moments_only_opaque(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        num_warps,
    )


__all__ = [
    "parallel_static_excitation_recurrence_lag124_moments_only_training",
    "parallel_static_excitation_recurrence_lag124_moments_packed_io_training",
    "parallel_static_radial_log_recurrence_lag124_moments_only_inference",
    "parallel_static_radial_log_recurrence_lag124_moments_only_training",
    "parallel_static_radial_log_recurrence_lag124_moments_packed_io_inference",
    "parallel_static_radial_log_recurrence_lag124_moments_packed_io_training",
    "parallel_static_recurrence_lag124_moments_only_inference",
    "parallel_static_recurrence_lag124_moments_only_training",
    "parallel_static_recurrence_lag124_moments_packed_io_training",
]
