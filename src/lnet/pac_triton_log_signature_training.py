"""FP32 CUDA training candidate for recurrence plus degree-two log signature.

The forward kernel materializes only the complex recurrence trajectory while
streaming the seven level-one and twenty-one level-two coordinates.  The
analytic backward reconstructs each event in reverse, recovers its level-one
prefix from the final signature, and folds the event VJP directly into the
complex recurrence adjoint.  Increment and prefix sequences are never stored.

Metadata is treated as fixed sample geometry.  Gradients for ``valid_mask`` and
``time_delta`` are deliberately rejected; the four recurrence inputs are the
only differentiable inputs supported by this bounded candidate.
"""

from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001, FBT001, FBT003, N803, PLR0915
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

_EVENT_DIMENSION: Final[int] = 7
_SIGNATURE_DIMENSION: Final[int] = 28
_DEFAULT_EPSILON: Final[float] = 1.0e-8
_PAIR_FIRST: Final[tuple[int, ...]] = tuple(
    first for first in range(_EVENT_DIMENSION) for _ in range(first + 1, _EVENT_DIMENSION)
)
_PAIR_SECOND: Final[tuple[int, ...]] = tuple(
    second for first in range(_EVENT_DIMENSION) for second in range(first + 1, _EVENT_DIMENSION)
)


class _TrainingContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    needs_input_grad: tuple[bool, ...]
    has_valid: bool
    has_delta: bool
    epsilon: float

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _duration_kernel(
    valid_mask,
    time_delta,
    duration_output,
    support_output,
    n_steps: int,
    epsilon: float,
    HAS_VALID: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    BLOCK_STEPS: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    duration_mass = tl.full((), 0.0, tl.float32)
    offsets = tl.arange(0, BLOCK_STEPS)
    block_start = 0
    while block_start < n_steps:
        step = block_start + offsets
        active = step < n_steps
        offset = batch * n_steps + step
        valid_value = (
            tl.load(valid_mask + offset, mask=active, other=0.0).to(tl.float32)
            if HAS_VALID
            else active.to(tl.float32)
        )
        delta_value = (
            tl.load(time_delta + offset, mask=active, other=0.0).to(tl.float32)
            if HAS_DELTA
            else active.to(tl.float32)
        )
        duration_mass += tl.sum(valid_value * delta_value)
        block_start += BLOCK_STEPS
    tl.store(duration_output + batch, tl.maximum(duration_mass, epsilon))
    tl.store(support_output + batch, duration_mass > 0.0)


@triton.jit
def _forward_kernel(
    decay_real,
    decay_imag,
    drive_real,
    drive_imag,
    valid_mask,
    time_delta,
    duration,
    states_real,
    states_imag,
    signature_output,
    n_steps: int,
    modes: int,
    epsilon: float,
    STATIC_DECAY: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    active_duration = tl.load(duration + batch).to(tl.float32)

    state_real = tl.zeros((BLOCK_MODES,), tl.float32)
    state_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    level0 = tl.zeros((BLOCK_MODES,), tl.float32)
    level1 = tl.zeros((BLOCK_MODES,), tl.float32)
    level2 = tl.zeros((BLOCK_MODES,), tl.float32)
    level3 = tl.zeros((BLOCK_MODES,), tl.float32)
    level4 = tl.zeros((BLOCK_MODES,), tl.float32)
    level5 = tl.zeros((BLOCK_MODES,), tl.float32)
    level6 = tl.zeros((BLOCK_MODES,), tl.float32)

    area01 = tl.zeros((BLOCK_MODES,), tl.float32)
    area02 = tl.zeros((BLOCK_MODES,), tl.float32)
    area03 = tl.zeros((BLOCK_MODES,), tl.float32)
    area04 = tl.zeros((BLOCK_MODES,), tl.float32)
    area05 = tl.zeros((BLOCK_MODES,), tl.float32)
    area06 = tl.zeros((BLOCK_MODES,), tl.float32)
    area12 = tl.zeros((BLOCK_MODES,), tl.float32)
    area13 = tl.zeros((BLOCK_MODES,), tl.float32)
    area14 = tl.zeros((BLOCK_MODES,), tl.float32)
    area15 = tl.zeros((BLOCK_MODES,), tl.float32)
    area16 = tl.zeros((BLOCK_MODES,), tl.float32)
    area23 = tl.zeros((BLOCK_MODES,), tl.float32)
    area24 = tl.zeros((BLOCK_MODES,), tl.float32)
    area25 = tl.zeros((BLOCK_MODES,), tl.float32)
    area26 = tl.zeros((BLOCK_MODES,), tl.float32)
    area34 = tl.zeros((BLOCK_MODES,), tl.float32)
    area35 = tl.zeros((BLOCK_MODES,), tl.float32)
    area36 = tl.zeros((BLOCK_MODES,), tl.float32)
    area45 = tl.zeros((BLOCK_MODES,), tl.float32)
    area46 = tl.zeros((BLOCK_MODES,), tl.float32)
    area56 = tl.zeros((BLOCK_MODES,), tl.float32)

    fixed_decay_real = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    if STATIC_DECAY:
        fixed_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
        fixed_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)

    step = 0
    while step < n_steps:
        offset = (batch * n_steps + step) * modes + mode
        if STATIC_DECAY:
            active_decay_real = fixed_decay_real
            active_decay_imag = fixed_decay_imag
        else:
            active_decay_real = tl.load(decay_real + offset, mask=valid_mode, other=0.0).to(
                tl.float32
            )
            active_decay_imag = tl.load(decay_imag + offset, mask=valid_mode, other=0.0).to(
                tl.float32
            )
        active_drive_real = tl.load(drive_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
        active_drive_imag = tl.load(drive_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        previous_real, previous_imag = state_real, state_imag
        prediction_real = active_decay_real * previous_real - active_decay_imag * previous_imag
        prediction_imag = active_decay_imag * previous_real + active_decay_real * previous_imag
        state_real = prediction_real + active_drive_real
        state_imag = prediction_imag + active_drive_imag
        tl.store(states_real + offset, state_real, mask=valid_mode)
        tl.store(states_imag + offset, state_imag, mask=valid_mode)

        metadata_offset = batch * n_steps + step
        valid_value = tl.load(valid_mask + metadata_offset).to(tl.float32) if HAS_VALID else 1.0
        delta_value = tl.load(time_delta + metadata_offset).to(tl.float32) if HAS_DELTA else 1.0
        event_weight = valid_value * delta_value / active_duration
        prediction_power = prediction_real * prediction_real + prediction_imag * prediction_imag
        drive_power = active_drive_real * active_drive_real + active_drive_imag * active_drive_imag
        previous_power = previous_real * previous_real + previous_imag * previous_imag
        product_scale = tl.sqrt(epsilon + drive_power) * tl.sqrt(epsilon + prediction_power)
        transport_scale = tl.sqrt(epsilon + prediction_power) * tl.sqrt(epsilon + previous_power)
        increment0 = event_weight
        increment1 = event_weight * libdevice.log1p(prediction_power)
        increment2 = event_weight * libdevice.log1p(drive_power)
        increment3 = (
            event_weight
            * (active_drive_real * prediction_real + active_drive_imag * prediction_imag)
            / product_scale
        )
        increment4 = (
            event_weight
            * (active_drive_imag * prediction_real - active_drive_real * prediction_imag)
            / product_scale
        )
        increment5 = (
            event_weight
            * (prediction_real * previous_real + prediction_imag * previous_imag)
            / transport_scale
        )
        increment6 = (
            event_weight
            * (prediction_imag * previous_real - prediction_real * previous_imag)
            / transport_scale
        )

        area01 += 0.5 * (level0 * increment1 - level1 * increment0)
        area02 += 0.5 * (level0 * increment2 - level2 * increment0)
        area03 += 0.5 * (level0 * increment3 - level3 * increment0)
        area04 += 0.5 * (level0 * increment4 - level4 * increment0)
        area05 += 0.5 * (level0 * increment5 - level5 * increment0)
        area06 += 0.5 * (level0 * increment6 - level6 * increment0)
        area12 += 0.5 * (level1 * increment2 - level2 * increment1)
        area13 += 0.5 * (level1 * increment3 - level3 * increment1)
        area14 += 0.5 * (level1 * increment4 - level4 * increment1)
        area15 += 0.5 * (level1 * increment5 - level5 * increment1)
        area16 += 0.5 * (level1 * increment6 - level6 * increment1)
        area23 += 0.5 * (level2 * increment3 - level3 * increment2)
        area24 += 0.5 * (level2 * increment4 - level4 * increment2)
        area25 += 0.5 * (level2 * increment5 - level5 * increment2)
        area26 += 0.5 * (level2 * increment6 - level6 * increment2)
        area34 += 0.5 * (level3 * increment4 - level4 * increment3)
        area35 += 0.5 * (level3 * increment5 - level5 * increment3)
        area36 += 0.5 * (level3 * increment6 - level6 * increment3)
        area45 += 0.5 * (level4 * increment5 - level5 * increment4)
        area46 += 0.5 * (level4 * increment6 - level6 * increment4)
        area56 += 0.5 * (level5 * increment6 - level6 * increment5)
        level0 += increment0
        level1 += increment1
        level2 += increment2
        level3 += increment3
        level4 += increment4
        level5 += increment5
        level6 += increment6
        step += 1

    signature_base = (batch * modes + mode) * 28
    tl.store(signature_output + signature_base, level0, mask=valid_mode)
    tl.store(signature_output + signature_base + 1, level1, mask=valid_mode)
    tl.store(signature_output + signature_base + 2, level2, mask=valid_mode)
    tl.store(signature_output + signature_base + 3, level3, mask=valid_mode)
    tl.store(signature_output + signature_base + 4, level4, mask=valid_mode)
    tl.store(signature_output + signature_base + 5, level5, mask=valid_mode)
    tl.store(signature_output + signature_base + 6, level6, mask=valid_mode)
    tl.store(signature_output + signature_base + 7, area01, mask=valid_mode)
    tl.store(signature_output + signature_base + 8, area02, mask=valid_mode)
    tl.store(signature_output + signature_base + 9, area03, mask=valid_mode)
    tl.store(signature_output + signature_base + 10, area04, mask=valid_mode)
    tl.store(signature_output + signature_base + 11, area05, mask=valid_mode)
    tl.store(signature_output + signature_base + 12, area06, mask=valid_mode)
    tl.store(signature_output + signature_base + 13, area12, mask=valid_mode)
    tl.store(signature_output + signature_base + 14, area13, mask=valid_mode)
    tl.store(signature_output + signature_base + 15, area14, mask=valid_mode)
    tl.store(signature_output + signature_base + 16, area15, mask=valid_mode)
    tl.store(signature_output + signature_base + 17, area16, mask=valid_mode)
    tl.store(signature_output + signature_base + 18, area23, mask=valid_mode)
    tl.store(signature_output + signature_base + 19, area24, mask=valid_mode)
    tl.store(signature_output + signature_base + 20, area25, mask=valid_mode)
    tl.store(signature_output + signature_base + 21, area26, mask=valid_mode)
    tl.store(signature_output + signature_base + 22, area34, mask=valid_mode)
    tl.store(signature_output + signature_base + 23, area35, mask=valid_mode)
    tl.store(signature_output + signature_base + 24, area36, mask=valid_mode)
    tl.store(signature_output + signature_base + 25, area45, mask=valid_mode)
    tl.store(signature_output + signature_base + 26, area46, mask=valid_mode)
    tl.store(signature_output + signature_base + 27, area56, mask=valid_mode)


@triton.jit
def _backward_kernel(
    decay_real,
    decay_imag,
    drive_real,
    drive_imag,
    valid_mask,
    time_delta,
    duration,
    states_real,
    states_imag,
    signature,
    grad_signature,
    grad_states_real,
    grad_states_imag,
    grad_decay_real,
    grad_decay_imag,
    grad_drive_real,
    grad_drive_imag,
    n_steps: int,
    modes: int,
    epsilon: float,
    STATIC_DECAY: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    HAS_STATE_GRAD: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    signature_base = (batch * modes + mode) * 28
    active_duration = tl.load(duration + batch).to(tl.float32)

    level0 = tl.load(signature + signature_base, mask=valid_mode, other=0.0).to(tl.float32)
    level1 = tl.load(signature + signature_base + 1, mask=valid_mode, other=0.0).to(tl.float32)
    level2 = tl.load(signature + signature_base + 2, mask=valid_mode, other=0.0).to(tl.float32)
    level3 = tl.load(signature + signature_base + 3, mask=valid_mode, other=0.0).to(tl.float32)
    level4 = tl.load(signature + signature_base + 4, mask=valid_mode, other=0.0).to(tl.float32)
    level5 = tl.load(signature + signature_base + 5, mask=valid_mode, other=0.0).to(tl.float32)
    level6 = tl.load(signature + signature_base + 6, mask=valid_mode, other=0.0).to(tl.float32)
    grad_level0 = tl.load(grad_signature + signature_base, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_level1 = tl.load(grad_signature + signature_base + 1, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_level2 = tl.load(grad_signature + signature_base + 2, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_level3 = tl.load(grad_signature + signature_base + 3, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_level4 = tl.load(grad_signature + signature_base + 4, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_level5 = tl.load(grad_signature + signature_base + 5, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_level6 = tl.load(grad_signature + signature_base + 6, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area01 = tl.load(grad_signature + signature_base + 7, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area02 = tl.load(grad_signature + signature_base + 8, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area03 = tl.load(grad_signature + signature_base + 9, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area04 = tl.load(grad_signature + signature_base + 10, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area05 = tl.load(grad_signature + signature_base + 11, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area06 = tl.load(grad_signature + signature_base + 12, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area12 = tl.load(grad_signature + signature_base + 13, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area13 = tl.load(grad_signature + signature_base + 14, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area14 = tl.load(grad_signature + signature_base + 15, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area15 = tl.load(grad_signature + signature_base + 16, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area16 = tl.load(grad_signature + signature_base + 17, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area23 = tl.load(grad_signature + signature_base + 18, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area24 = tl.load(grad_signature + signature_base + 19, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area25 = tl.load(grad_signature + signature_base + 20, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area26 = tl.load(grad_signature + signature_base + 21, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area34 = tl.load(grad_signature + signature_base + 22, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area35 = tl.load(grad_signature + signature_base + 23, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area36 = tl.load(grad_signature + signature_base + 24, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area45 = tl.load(grad_signature + signature_base + 25, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area46 = tl.load(grad_signature + signature_base + 26, mask=valid_mode, other=0.0).to(
        tl.float32
    )
    grad_area56 = tl.load(grad_signature + signature_base + 27, mask=valid_mode, other=0.0).to(
        tl.float32
    )

    fixed_decay_real = tl.zeros((BLOCK_MODES,), tl.float32)
    fixed_decay_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    if STATIC_DECAY:
        fixed_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
        fixed_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    accumulated_decay_real = tl.zeros((BLOCK_MODES,), tl.float32)
    accumulated_decay_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    lambda_real = tl.zeros((BLOCK_MODES,), tl.float32)
    lambda_imag = tl.zeros((BLOCK_MODES,), tl.float32)

    reverse_step = 0
    while reverse_step < n_steps:
        step = n_steps - 1 - reverse_step
        offset = (batch * n_steps + step) * modes + mode
        has_previous = step > 0
        previous_offset = offset - modes
        previous_real = tl.load(
            states_real + previous_offset,
            mask=valid_mode & has_previous,
            other=0.0,
        ).to(tl.float32)
        previous_imag = tl.load(
            states_imag + previous_offset,
            mask=valid_mode & has_previous,
            other=0.0,
        ).to(tl.float32)
        if STATIC_DECAY:
            active_decay_real = fixed_decay_real
            active_decay_imag = fixed_decay_imag
        else:
            active_decay_real = tl.load(decay_real + offset, mask=valid_mode, other=0.0).to(
                tl.float32
            )
            active_decay_imag = tl.load(decay_imag + offset, mask=valid_mode, other=0.0).to(
                tl.float32
            )
        active_drive_real = tl.load(drive_real + offset, mask=valid_mode, other=0.0).to(tl.float32)
        active_drive_imag = tl.load(drive_imag + offset, mask=valid_mode, other=0.0).to(tl.float32)
        prediction_real = active_decay_real * previous_real - active_decay_imag * previous_imag
        prediction_imag = active_decay_imag * previous_real + active_decay_real * previous_imag

        metadata_offset = batch * n_steps + step
        valid_value = tl.load(valid_mask + metadata_offset).to(tl.float32) if HAS_VALID else 1.0
        delta_value = tl.load(time_delta + metadata_offset).to(tl.float32) if HAS_DELTA else 1.0
        event_weight = valid_value * delta_value / active_duration
        prediction_power = prediction_real * prediction_real + prediction_imag * prediction_imag
        drive_power = active_drive_real * active_drive_real + active_drive_imag * active_drive_imag
        previous_power = previous_real * previous_real + previous_imag * previous_imag
        inverse_product_scale = tl.rsqrt(epsilon + drive_power) * tl.rsqrt(
            epsilon + prediction_power
        )
        inverse_transport_scale = tl.rsqrt(epsilon + prediction_power) * tl.rsqrt(
            epsilon + previous_power
        )
        normalized_product_real = (
            active_drive_real * prediction_real + active_drive_imag * prediction_imag
        ) * inverse_product_scale
        normalized_product_imag = (
            active_drive_imag * prediction_real - active_drive_real * prediction_imag
        ) * inverse_product_scale
        normalized_transport_real = (
            prediction_real * previous_real + prediction_imag * previous_imag
        ) * inverse_transport_scale
        normalized_transport_imag = (
            prediction_imag * previous_real - prediction_real * previous_imag
        ) * inverse_transport_scale
        increment0 = event_weight
        increment1 = event_weight * libdevice.log1p(prediction_power)
        increment2 = event_weight * libdevice.log1p(drive_power)
        increment3 = event_weight * normalized_product_real
        increment4 = event_weight * normalized_product_imag
        increment5 = event_weight * normalized_transport_real
        increment6 = event_weight * normalized_transport_imag

        prefix0 = level0 - increment0
        prefix1 = level1 - increment1
        prefix2 = level2 - increment2
        prefix3 = level3 - increment3
        prefix4 = level4 - increment4
        prefix5 = level5 - increment5
        prefix6 = level6 - increment6

        grad_increment1 = grad_level1 + 0.5 * (
            grad_area01 * prefix0
            - grad_area12 * prefix2
            - grad_area13 * prefix3
            - grad_area14 * prefix4
            - grad_area15 * prefix5
            - grad_area16 * prefix6
        )
        grad_increment2 = grad_level2 + 0.5 * (
            grad_area02 * prefix0
            + grad_area12 * prefix1
            - grad_area23 * prefix3
            - grad_area24 * prefix4
            - grad_area25 * prefix5
            - grad_area26 * prefix6
        )
        grad_increment3 = grad_level3 + 0.5 * (
            grad_area03 * prefix0
            + grad_area13 * prefix1
            + grad_area23 * prefix2
            - grad_area34 * prefix4
            - grad_area35 * prefix5
            - grad_area36 * prefix6
        )
        grad_increment4 = grad_level4 + 0.5 * (
            grad_area04 * prefix0
            + grad_area14 * prefix1
            + grad_area24 * prefix2
            + grad_area34 * prefix3
            - grad_area45 * prefix5
            - grad_area46 * prefix6
        )
        grad_increment5 = grad_level5 + 0.5 * (
            grad_area05 * prefix0
            + grad_area15 * prefix1
            + grad_area25 * prefix2
            + grad_area35 * prefix3
            + grad_area45 * prefix4
            - grad_area56 * prefix6
        )
        grad_increment6 = grad_level6 + 0.5 * (
            grad_area06 * prefix0
            + grad_area16 * prefix1
            + grad_area26 * prefix2
            + grad_area36 * prefix3
            + grad_area46 * prefix4
            + grad_area56 * prefix5
        )

        grad_level0 += 0.5 * (
            grad_area01 * increment1
            + grad_area02 * increment2
            + grad_area03 * increment3
            + grad_area04 * increment4
            + grad_area05 * increment5
            + grad_area06 * increment6
        )
        grad_level1 += 0.5 * (
            -grad_area01 * increment0
            + grad_area12 * increment2
            + grad_area13 * increment3
            + grad_area14 * increment4
            + grad_area15 * increment5
            + grad_area16 * increment6
        )
        grad_level2 += 0.5 * (
            -grad_area02 * increment0
            - grad_area12 * increment1
            + grad_area23 * increment3
            + grad_area24 * increment4
            + grad_area25 * increment5
            + grad_area26 * increment6
        )
        grad_level3 += 0.5 * (
            -grad_area03 * increment0
            - grad_area13 * increment1
            - grad_area23 * increment2
            + grad_area34 * increment4
            + grad_area35 * increment5
            + grad_area36 * increment6
        )
        grad_level4 += 0.5 * (
            -grad_area04 * increment0
            - grad_area14 * increment1
            - grad_area24 * increment2
            - grad_area34 * increment3
            + grad_area45 * increment5
            + grad_area46 * increment6
        )
        grad_level5 += 0.5 * (
            -grad_area05 * increment0
            - grad_area15 * increment1
            - grad_area25 * increment2
            - grad_area35 * increment3
            - grad_area45 * increment4
            + grad_area56 * increment6
        )
        grad_level6 += 0.5 * (
            -grad_area06 * increment0
            - grad_area16 * increment1
            - grad_area26 * increment2
            - grad_area36 * increment3
            - grad_area46 * increment4
            - grad_area56 * increment5
        )
        level0 = prefix0
        level1 = prefix1
        level2 = prefix2
        level3 = prefix3
        level4 = prefix4
        level5 = prefix5
        level6 = prefix6

        weighted1 = grad_increment1 * event_weight
        weighted2 = grad_increment2 * event_weight
        weighted3 = grad_increment3 * event_weight
        weighted4 = grad_increment4 * event_weight
        weighted5 = grad_increment5 * event_weight
        weighted6 = grad_increment6 * event_weight
        inverse_prediction_power = 1.0 / (epsilon + prediction_power)
        inverse_drive_power = 1.0 / (epsilon + drive_power)
        inverse_previous_power = 1.0 / (epsilon + previous_power)

        event_prediction_real = (
            weighted1 * (2.0 * prediction_real / (1.0 + prediction_power))
            + weighted3
            * (
                active_drive_real * inverse_product_scale
                - normalized_product_real * prediction_real * inverse_prediction_power
            )
            + weighted4
            * (
                active_drive_imag * inverse_product_scale
                - normalized_product_imag * prediction_real * inverse_prediction_power
            )
            + weighted5
            * (
                previous_real * inverse_transport_scale
                - normalized_transport_real * prediction_real * inverse_prediction_power
            )
            + weighted6
            * (
                -previous_imag * inverse_transport_scale
                - normalized_transport_imag * prediction_real * inverse_prediction_power
            )
        )
        event_prediction_imag = (
            weighted1 * (2.0 * prediction_imag / (1.0 + prediction_power))
            + weighted3
            * (
                active_drive_imag * inverse_product_scale
                - normalized_product_real * prediction_imag * inverse_prediction_power
            )
            + weighted4
            * (
                -active_drive_real * inverse_product_scale
                - normalized_product_imag * prediction_imag * inverse_prediction_power
            )
            + weighted5
            * (
                previous_imag * inverse_transport_scale
                - normalized_transport_real * prediction_imag * inverse_prediction_power
            )
            + weighted6
            * (
                previous_real * inverse_transport_scale
                - normalized_transport_imag * prediction_imag * inverse_prediction_power
            )
        )
        event_drive_real = (
            weighted2 * (2.0 * active_drive_real / (1.0 + drive_power))
            + weighted3
            * (
                prediction_real * inverse_product_scale
                - normalized_product_real * active_drive_real * inverse_drive_power
            )
            + weighted4
            * (
                -prediction_imag * inverse_product_scale
                - normalized_product_imag * active_drive_real * inverse_drive_power
            )
        )
        event_drive_imag = (
            weighted2 * (2.0 * active_drive_imag / (1.0 + drive_power))
            + weighted3
            * (
                prediction_imag * inverse_product_scale
                - normalized_product_real * active_drive_imag * inverse_drive_power
            )
            + weighted4
            * (
                prediction_real * inverse_product_scale
                - normalized_product_imag * active_drive_imag * inverse_drive_power
            )
        )
        event_previous_real = weighted5 * (
            prediction_real * inverse_transport_scale
            - normalized_transport_real * previous_real * inverse_previous_power
        ) + weighted6 * (
            prediction_imag * inverse_transport_scale
            - normalized_transport_imag * previous_real * inverse_previous_power
        )
        event_previous_imag = weighted5 * (
            prediction_imag * inverse_transport_scale
            - normalized_transport_real * previous_imag * inverse_previous_power
        ) + weighted6 * (
            -prediction_real * inverse_transport_scale
            - normalized_transport_imag * previous_imag * inverse_previous_power
        )

        if HAS_STATE_GRAD:
            lambda_real += tl.load(
                grad_states_real + offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            lambda_imag += tl.load(
                grad_states_imag + offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)

        prediction_gradient_real = event_prediction_real + lambda_real
        prediction_gradient_imag = event_prediction_imag + lambda_imag
        drive_gradient_real = event_drive_real + lambda_real
        drive_gradient_imag = event_drive_imag + lambda_imag
        tl.store(grad_drive_real + offset, drive_gradient_real, mask=valid_mode)
        tl.store(grad_drive_imag + offset, drive_gradient_imag, mask=valid_mode)

        decay_gradient_real = (
            prediction_gradient_real * previous_real + prediction_gradient_imag * previous_imag
        )
        decay_gradient_imag = (
            -prediction_gradient_real * previous_imag + prediction_gradient_imag * previous_real
        )
        if STATIC_DECAY:
            accumulated_decay_real += decay_gradient_real
            accumulated_decay_imag += decay_gradient_imag
        else:
            tl.store(grad_decay_real + offset, decay_gradient_real, mask=valid_mode)
            tl.store(grad_decay_imag + offset, decay_gradient_imag, mask=valid_mode)

        lambda_real = (
            event_previous_real
            + active_decay_real * prediction_gradient_real
            + active_decay_imag * prediction_gradient_imag
        )
        lambda_imag = (
            event_previous_imag
            - active_decay_imag * prediction_gradient_real
            + active_decay_real * prediction_gradient_imag
        )
        reverse_step += 1

    if STATIC_DECAY:
        static_offset = batch * modes + mode
        tl.store(grad_decay_real + static_offset, accumulated_decay_real, mask=valid_mode)
        tl.store(grad_decay_imag + static_offset, accumulated_decay_imag, mask=valid_mode)


def _select_block_modes(batch: int, steps: int, modes: int) -> int:
    maximum_useful = 1 << (modes - 1).bit_length()
    if batch >= 8 and steps >= 512:
        selected = 1
    elif steps >= 1024:
        selected = 2
    else:
        selected = 4
    return min(selected, maximum_useful)


@triton_op("lnet::pac_log_signature_training_forward", mutates_args={})
def _training_forward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    valid_mask: Tensor,
    time_delta: Tensor,
    has_valid: bool,
    has_delta: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, steps, modes = drive_real.shape
    signature = torch.empty(
        (batch, modes, _SIGNATURE_DIMENSION),
        dtype=torch.float32,
        device=drive_real.device,
    )
    support = torch.empty((batch,), dtype=torch.bool, device=drive_real.device)
    states_real = torch.empty_like(drive_real)
    states_imag = torch.empty_like(drive_imag)
    if has_valid or has_delta:
        duration = torch.empty((batch,), dtype=torch.float32, device=drive_real.device)
        wrap_triton(_duration_kernel)[(batch,)](
            valid_mask,
            time_delta,
            duration,
            support,
            steps,
            epsilon,
            HAS_VALID=has_valid,
            HAS_DELTA=has_delta,
            BLOCK_STEPS=256,
            num_warps=4,
        )
    else:
        duration = torch.full(
            (batch,),
            float(steps),
            dtype=torch.float32,
            device=drive_real.device,
        )
        support.fill_(True)
    block_modes = _select_block_modes(batch, steps, modes)
    wrap_triton(_forward_kernel)[(batch * triton.cdiv(modes, block_modes),)](
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask,
        time_delta,
        duration,
        states_real,
        states_imag,
        signature,
        steps,
        modes,
        epsilon,
        STATIC_DECAY=decay_real.ndim == 1,
        HAS_VALID=has_valid,
        HAS_DELTA=has_delta,
        BLOCK_MODES=block_modes,
        num_warps=1,
    )
    return signature, support.view(batch, 1, 1), states_real, states_imag, duration


@triton_op("lnet::pac_log_signature_training_backward", mutates_args={})
def _training_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    valid_mask: Tensor,
    time_delta: Tensor,
    duration: Tensor,
    states_real: Tensor,
    states_imag: Tensor,
    signature: Tensor,
    grad_signature: Tensor,
    grad_states_real: Tensor,
    grad_states_imag: Tensor,
    has_valid: bool,
    has_delta: bool,
    has_state_grad: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, steps, modes = drive_real.shape
    static_decay = decay_real.ndim == 1
    if static_decay:
        decay_gradient_real = torch.empty(
            (batch, modes), dtype=torch.float32, device=drive_real.device
        )
        decay_gradient_imag = torch.empty_like(decay_gradient_real)
    else:
        decay_gradient_real = torch.empty_like(decay_real)
        decay_gradient_imag = torch.empty_like(decay_imag)
    drive_gradient_real = torch.empty_like(drive_real)
    drive_gradient_imag = torch.empty_like(drive_imag)
    block_modes = _select_block_modes(batch, steps, modes)
    wrap_triton(_backward_kernel)[(batch * triton.cdiv(modes, block_modes),)](
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask,
        time_delta,
        duration,
        states_real,
        states_imag,
        signature,
        grad_signature.contiguous(),
        grad_states_real,
        grad_states_imag,
        decay_gradient_real,
        decay_gradient_imag,
        drive_gradient_real,
        drive_gradient_imag,
        steps,
        modes,
        epsilon,
        STATIC_DECAY=static_decay,
        HAS_VALID=has_valid,
        HAS_DELTA=has_delta,
        HAS_STATE_GRAD=has_state_grad,
        BLOCK_MODES=block_modes,
        num_warps=1,
    )
    if static_decay:
        return (
            decay_gradient_real.sum(dim=0),
            decay_gradient_imag.sum(dim=0),
            drive_gradient_real,
            drive_gradient_imag,
        )
    return (
        decay_gradient_real,
        decay_gradient_imag,
        drive_gradient_real,
        drive_gradient_imag,
    )


def _setup_context(
    ctx: _TrainingContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, float],
    output: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
) -> None:
    (
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask,
        time_delta,
        has_valid,
        has_delta,
        epsilon,
    ) = inputs
    signature, _support, states_real, states_imag, duration = output
    ctx.has_valid = has_valid
    ctx.has_delta = has_delta
    ctx.epsilon = epsilon
    ctx.save_for_backward(
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask,
        time_delta,
        duration,
        states_real,
        states_imag,
        signature,
    )


def _materialized_signature_for_higher_order(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    valid_mask: Tensor,
    time_delta: Tensor,
    *,
    has_valid: bool,
    has_delta: bool,
    epsilon: float,
) -> Tensor:
    """Recompute the canonical materialized+BMM graph for double backward."""
    batch, steps, modes = drive_real.shape
    state_real = torch.zeros_like(drive_real[:, 0])
    state_imag = torch.zeros_like(state_real)
    previous_real: list[Tensor] = []
    previous_imag: list[Tensor] = []
    prediction_real: list[Tensor] = []
    prediction_imag: list[Tensor] = []
    static_decay = decay_real.ndim == 1
    for index in range(steps):
        previous_real.append(state_real)
        previous_imag.append(state_imag)
        active_decay_real = decay_real if static_decay else decay_real[:, index]
        active_decay_imag = decay_imag if static_decay else decay_imag[:, index]
        active_prediction_real = active_decay_real * state_real - active_decay_imag * state_imag
        active_prediction_imag = active_decay_imag * state_real + active_decay_real * state_imag
        prediction_real.append(active_prediction_real)
        prediction_imag.append(active_prediction_imag)
        state_real = active_prediction_real + drive_real[:, index]
        state_imag = active_prediction_imag + drive_imag[:, index]

    predictions_real = torch.stack(prediction_real, dim=1)
    predictions_imag = torch.stack(prediction_imag, dim=1)
    previous_states_real = torch.stack(previous_real, dim=1)
    previous_states_imag = torch.stack(previous_imag, dim=1)
    valid = (
        valid_mask.view(batch, steps, 1)
        if has_valid
        else torch.ones((batch, steps, 1), dtype=torch.float32, device=drive_real.device)
    )
    delta = time_delta.view(batch, steps, 1) if has_delta else torch.ones_like(valid)
    duration = (valid * delta).sum(dim=1, keepdim=True).clamp_min(epsilon)
    event_weight = valid * delta / duration
    prediction_power = predictions_real.square() + predictions_imag.square()
    drive_power = drive_real.square() + drive_imag.square()
    previous_power = previous_states_real.square() + previous_states_imag.square()
    product_scale = torch.sqrt(epsilon + drive_power) * torch.sqrt(epsilon + prediction_power)
    transport_scale = torch.sqrt(epsilon + prediction_power) * torch.sqrt(epsilon + previous_power)
    increments = torch.stack(
        (
            event_weight.expand(batch, steps, modes),
            event_weight * torch.log1p(prediction_power),
            event_weight * torch.log1p(drive_power),
            event_weight
            * (drive_real * predictions_real + drive_imag * predictions_imag)
            / product_scale,
            event_weight
            * (drive_imag * predictions_real - drive_real * predictions_imag)
            / product_scale,
            event_weight
            * (predictions_real * previous_states_real + predictions_imag * previous_states_imag)
            / transport_scale,
            event_weight
            * (predictions_imag * previous_states_real - predictions_real * previous_states_imag)
            / transport_scale,
        ),
        dim=-1,
    )
    prefix = torch.cumsum(increments, dim=1) - increments
    level_one = increments.sum(dim=1)
    if torch.backends.cuda.matmul.allow_tf32:
        areas = torch.stack(
            [
                0.5
                * (
                    prefix[..., first] * increments[..., second]
                    - prefix[..., second] * increments[..., first]
                ).sum(dim=1)
                for first, second in zip(_PAIR_FIRST, _PAIR_SECOND, strict=True)
            ],
            dim=-1,
        )
        return torch.cat((level_one, areas), dim=-1)
    left = prefix.permute(0, 2, 3, 1).reshape(
        batch * modes,
        _EVENT_DIMENSION,
        steps,
    )
    right = increments.permute(0, 2, 1, 3).reshape(
        batch * modes,
        steps,
        _EVENT_DIMENSION,
    )
    cross = torch.bmm(left, right).reshape(
        batch,
        modes,
        _EVENT_DIMENSION,
        _EVENT_DIMENSION,
    )
    area_matrix = 0.5 * (cross - cross.transpose(-1, -2))
    areas = area_matrix[..., _PAIR_FIRST, _PAIR_SECOND]
    return torch.cat((level_one, areas), dim=-1)


def _higher_order_backward(
    ctx: _TrainingContext,
    grad_signature: Tensor,
    grad_states_real: Tensor | None,
    grad_states_imag: Tensor | None,
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    valid_mask: Tensor,
    time_delta: Tensor,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    signature = _materialized_signature_for_higher_order(
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask,
        time_delta,
        has_valid=ctx.has_valid,
        has_delta=ctx.has_delta,
        epsilon=ctx.epsilon,
    )
    recurrence_inputs = (decay_real, decay_imag, drive_real, drive_imag)
    requested_indices = tuple(index for index in range(4) if ctx.needs_input_grad[index])
    requested_inputs = tuple(recurrence_inputs[index] for index in requested_indices)
    if grad_states_real is None and grad_states_imag is None:
        requested_gradients = torch.autograd.grad(
            signature,
            requested_inputs,
            grad_signature,
            create_graph=True,
        )
    else:
        batch, steps, _modes = drive_real.shape
        static_decay = decay_real.ndim == 1
        state_real = torch.zeros_like(drive_real[:, 0])
        state_imag = torch.zeros_like(state_real)
        state_real_rows: list[Tensor] = []
        state_imag_rows: list[Tensor] = []
        for index in range(steps):
            active_decay_real = decay_real if static_decay else decay_real[:, index]
            active_decay_imag = decay_imag if static_decay else decay_imag[:, index]
            next_real = (
                active_decay_real * state_real
                - active_decay_imag * state_imag
                + drive_real[:, index]
            )
            next_imag = (
                active_decay_imag * state_real
                + active_decay_real * state_imag
                + drive_imag[:, index]
            )
            state_real, state_imag = next_real, next_imag
            state_real_rows.append(state_real)
            state_imag_rows.append(state_imag)
        states_real = torch.stack(state_real_rows, dim=1).reshape(batch, steps, -1)
        states_imag = torch.stack(state_imag_rows, dim=1).reshape(batch, steps, -1)
        active_grad_states_real = (
            torch.zeros_like(states_real) if grad_states_real is None else grad_states_real
        )
        active_grad_states_imag = (
            torch.zeros_like(states_imag) if grad_states_imag is None else grad_states_imag
        )
        requested_gradients = torch.autograd.grad(
            (signature, states_real, states_imag),
            requested_inputs,
            (grad_signature, active_grad_states_real, active_grad_states_imag),
            create_graph=True,
        )
    gradients: list[Tensor | None] = [None, None, None, None]
    for index, gradient in zip(requested_indices, requested_gradients, strict=True):
        gradients[index] = gradient
    return gradients[0], gradients[1], gradients[2], gradients[3]


def _backward(
    ctx: _TrainingContext,
    grad_signature: Tensor | None,
    _grad_support: Tensor | None,
    grad_states_real: Tensor | None,
    grad_states_imag: Tensor | None,
    _grad_duration: Tensor | None,
) -> tuple[
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    None,
    None,
    None,
    None,
    None,
]:
    (
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask,
        time_delta,
        duration,
        states_real,
        states_imag,
        signature,
    ) = ctx.saved_tensors
    active_grad_signature = (
        torch.zeros_like(signature) if grad_signature is None else grad_signature
    )
    if torch.is_grad_enabled():
        higher_order_gradients = _higher_order_backward(
            ctx,
            active_grad_signature,
            grad_states_real,
            grad_states_imag,
            decay_real,
            decay_imag,
            drive_real,
            drive_imag,
            valid_mask,
            time_delta,
        )
        return *higher_order_gradients, None, None, None, None, None
    has_state_grad = grad_states_real is not None or grad_states_imag is not None
    active_grad_states_real = (
        (torch.zeros_like(states_real) if has_state_grad else states_real)
        if grad_states_real is None
        else grad_states_real.contiguous()
    )
    active_grad_states_imag = (
        (torch.zeros_like(states_imag) if has_state_grad else states_imag)
        if grad_states_imag is None
        else grad_states_imag.contiguous()
    )
    gradients = _training_backward_op(
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask,
        time_delta,
        duration,
        states_real,
        states_imag,
        signature,
        active_grad_signature,
        active_grad_states_real,
        active_grad_states_imag,
        ctx.has_valid,
        ctx.has_delta,
        has_state_grad,
        ctx.epsilon,
    )
    return *gradients, None, None, None, None, None


torch.library.register_autograd(
    "lnet::pac_log_signature_training_forward",
    _backward,
    setup_context=_setup_context,
)


def _metadata_2d(
    values: Tensor | None,
    *,
    batch: int,
    steps: int,
    reference: Tensor,
    name: str,
) -> Tensor | None:
    if values is None:
        return None
    if values.requires_grad:
        message = f"{name} gradients are unsupported by fused log-signature training"
        raise RuntimeError(message)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != (batch, steps):
        message = f"{name} must have shape [batch, steps] or [batch, steps, 1]"
        raise ValueError(message)
    if values.device != reference.device or values.dtype != torch.float32:
        message = f"{name} must be FP32 CUDA on the recurrence device"
        raise TypeError(message)
    if not values.is_contiguous():
        message = f"{name} must be contiguous"
        raise ValueError(message)
    return values


def _validated_forward_arguments(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    *,
    valid_mask: Tensor | None = None,
    time_delta: Tensor | None = None,
    epsilon: float = _DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, float]:
    if drive_real.ndim != 3 or min(drive_real.shape) < 1:
        message = "drive tensors must have non-empty shape [batch, steps, modes]"
        raise ValueError(message)
    if drive_imag.shape != drive_real.shape:
        message = "real and imaginary drive tensors must have matching shapes"
        raise ValueError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)
    tensors = (decay_real, decay_imag, drive_real, drive_imag)
    if any(not tensor.is_cuda or tensor.dtype != torch.float32 for tensor in tensors):
        message = "fused log-signature training requires FP32 CUDA tensors"
        raise TypeError(message)
    if any(not tensor.is_contiguous() for tensor in tensors):
        message = "fused log-signature training requires contiguous tensors"
        raise ValueError(message)
    if any(tensor.device != drive_real.device for tensor in tensors):
        message = "recurrence tensors must share one CUDA device"
        raise ValueError(message)
    batch, steps, modes = drive_real.shape
    static_shape = (modes,)
    singleton_static_shape = (1, 1, modes)
    dynamic_shape = drive_real.shape
    if decay_real.shape not in (static_shape, singleton_static_shape, dynamic_shape):
        message = "decay must have shape [modes], [1,1,modes], or [batch,steps,modes]"
        raise ValueError(message)
    if decay_imag.shape != decay_real.shape:
        message = "real and imaginary decay tensors must use the same layout"
        raise ValueError(message)
    active_decay_real = (
        decay_real[0, 0] if decay_real.shape == singleton_static_shape else decay_real
    )
    active_decay_imag = (
        decay_imag[0, 0] if decay_imag.shape == singleton_static_shape else decay_imag
    )
    active_valid = _metadata_2d(
        valid_mask,
        batch=batch,
        steps=steps,
        reference=drive_real,
        name="valid_mask",
    )
    active_delta = _metadata_2d(
        time_delta,
        batch=batch,
        steps=steps,
        reference=drive_real,
        name="time_delta",
    )
    placeholder = drive_real
    return (
        active_decay_real,
        active_decay_imag,
        drive_real,
        drive_imag,
        placeholder if active_valid is None else active_valid,
        placeholder if active_delta is None else active_delta,
        active_valid is not None,
        active_delta is not None,
        epsilon,
    )


def fused_recurrence_log_signature_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    *,
    valid_mask: Tensor | None = None,
    time_delta: Tensor | None = None,
    epsilon: float = _DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor]:
    """Run the bounded FP32 CUDA fused training path.

    Static decay accepts ``[M]`` or ``[1,1,M]`` and dynamic decay accepts
    ``[B,T,M]``.  The returned tensors are the ``[B,M,28]`` signature and the
    boolean ``[B,1,1]`` support flag.
    """
    arguments = _validated_forward_arguments(
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask=valid_mask,
        time_delta=time_delta,
        epsilon=epsilon,
    )
    signature, support, _states_real, _states_imag, _duration = _training_forward_op(*arguments)
    return signature, support


def fused_recurrence_log_signature_states_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    *,
    valid_mask: Tensor | None = None,
    time_delta: Tensor | None = None,
    epsilon: float = _DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return LogSig and differentiable recurrence states from one CUDA pass."""
    arguments = _validated_forward_arguments(
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask=valid_mask,
        time_delta=time_delta,
        epsilon=epsilon,
    )
    signature, support, states_real, states_imag, _duration = _training_forward_op(*arguments)
    return signature, support, states_real, states_imag


def fused_recurrence_log_signature_states_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    *,
    valid_mask: Tensor | None = None,
    time_delta: Tensor | None = None,
    epsilon: float = _DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return signature and recurrence states from one inference-only pass."""
    recurrence_inputs = (decay_real, decay_imag, drive_real, drive_imag)
    if torch.is_grad_enabled() and any(value.requires_grad for value in recurrence_inputs):
        message = "stateful fused LogSig is inference-only; disable gradient recording"
        raise RuntimeError(message)
    arguments = _validated_forward_arguments(
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask=valid_mask,
        time_delta=time_delta,
        epsilon=epsilon,
    )
    signature, support, states_real, states_imag, _duration = _training_forward_op(*arguments)
    return signature, support, states_real, states_imag


__all__ = [
    "fused_recurrence_log_signature_states_inference",
    "fused_recurrence_log_signature_states_training",
    "fused_recurrence_log_signature_training",
]
