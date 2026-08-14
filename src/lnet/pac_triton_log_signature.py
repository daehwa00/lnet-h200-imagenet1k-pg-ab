"""State-free recurrence and degree-two pole log-signature inference.

The differentiable PyTorch reference streams one complex recurrence state,
seven level-one coordinates, and twenty-one antisymmetric level-two areas per
mode.  It never constructs the state, increment, or prefix sequences.  The
CUDA inference path implements the same recurrence and accumulators in one
Triton kernel and returns only the ``[B, M, 28]`` signature and a support flag.

``drive_real`` and ``drive_imag`` are expected to include any observation-mask
effect.  ``valid_mask`` and ``time_delta`` affect event weighting only, matching
the pole-motion reader's existing semantics.
"""

from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001, N803, PLR0915
from typing import Final, Literal

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

_EVENT_DIMENSION: Final[int] = 7
_SIGNATURE_DIMENSION: Final[int] = 28
_DEFAULT_EPSILON: Final[float] = 1.0e-8
_PAIR_FIRST: Final[tuple[int, ...]] = (
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    4,
    4,
    5,
)
_PAIR_SECOND: Final[tuple[int, ...]] = (
    1,
    2,
    3,
    4,
    5,
    6,
    2,
    3,
    4,
    5,
    6,
    3,
    4,
    5,
    6,
    4,
    5,
    6,
    5,
    6,
    6,
)

LogSignatureBackend = Literal["auto", "reference", "triton"]


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
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape != (batch, steps):
        message = f"{name} must have shape [batch, steps] or [batch, steps, 1]"
        raise ValueError(message)
    if values.device != reference.device:
        message = f"{name} must share the recurrence device"
        raise ValueError(message)
    if values.dtype != reference.dtype:
        message = f"{name} must share the recurrence dtype"
        raise TypeError(message)
    return values


def _normalize_decay(
    decay: Tensor,
    drive: Tensor,
    *,
    name: str,
    preserve_expanded: bool,
) -> Tensor:
    modes = drive.shape[2]
    if decay.ndim == 1 and decay.shape == (modes,):
        return decay
    if decay.ndim == 3 and decay.shape == (1, 1, modes):
        return decay[0, 0]
    if decay.shape != drive.shape:
        message = f"{name} must have shape [modes], [1,1,modes], or [batch,steps,modes]"
        raise ValueError(message)
    if (
        decay.stride(0) == 0
        and decay.stride(1) == 0
        and decay.stride(2) == 1
        and not preserve_expanded
    ):
        return decay[0, 0]
    return decay


def _prepare_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    valid_mask: Tensor | None,
    time_delta: Tensor | None,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    if drive_real.ndim != 3 or min(drive_real.shape) < 1:
        message = "drive tensors must have non-empty shape [batch, steps, modes]"
        raise ValueError(message)
    if drive_imag.shape != drive_real.shape:
        message = "real and imaginary drive tensors must have matching shapes"
        raise ValueError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)
    if not drive_real.is_floating_point():
        message = "log-signature recurrence tensors must use a floating-point dtype"
        raise TypeError(message)
    for tensor in (decay_real, decay_imag, drive_imag):
        if tensor.device != drive_real.device:
            message = "log-signature recurrence tensors must share one device"
            raise ValueError(message)
        if tensor.dtype != drive_real.dtype:
            message = "log-signature recurrence tensors must share one dtype"
            raise TypeError(message)
    preserve_expanded = torch.is_grad_enabled() and (
        decay_real.requires_grad or decay_imag.requires_grad
    )
    active_decay_real = _normalize_decay(
        decay_real,
        drive_real,
        name="decay_real",
        preserve_expanded=preserve_expanded,
    )
    active_decay_imag = _normalize_decay(
        decay_imag,
        drive_real,
        name="decay_imag",
        preserve_expanded=preserve_expanded,
    )
    if active_decay_real.shape != active_decay_imag.shape:
        message = "real and imaginary decay tensors must use the same layout"
        raise ValueError(message)
    batch, steps, _ = drive_real.shape
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
    return active_decay_real, active_decay_imag, active_valid, active_delta


def reference_streaming_log_signature(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    *,
    valid_mask: Tensor | None = None,
    time_delta: Tensor | None = None,
    epsilon: float = _DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor]:
    """Compute the recurrence log-signature without sequence materialization.

    This implementation remains differentiable on CPU and for FP32/FP64 CUDA
    tensors.  The canonical CUDA half/bfloat16 recurrence has distinct hidden
    FP32-state and stored source-dtype semantics, so those dtypes deliberately
    stay on the model's materialized fallback instead of using this reference.
    """
    active_real, active_imag, active_valid, active_delta = _prepare_inputs(
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask,
        time_delta,
        epsilon,
    )
    if drive_real.is_cuda and drive_real.dtype in (torch.float16, torch.bfloat16):
        message = (
            "CUDA float16/bfloat16 requires the canonical materialized recurrence; "
            "the streaming reference supports CUDA float32/float64"
        )
        raise RuntimeError(message)
    working_dtype = (
        torch.float32 if drive_real.dtype in (torch.float16, torch.bfloat16) else drive_real.dtype
    )
    real_drive = drive_real
    imag_drive = drive_imag
    real_decay = active_real
    imag_decay = active_imag
    batch, steps, modes = drive_real.shape
    if active_valid is None:
        valid = torch.ones(
            (batch, steps),
            dtype=working_dtype,
            device=drive_real.device,
        )
    else:
        valid = active_valid.to(dtype=working_dtype)
    delta = torch.ones_like(valid) if active_delta is None else active_delta.to(dtype=working_dtype)
    duration_mass = (valid * delta).sum(dim=1, keepdim=True)
    duration = duration_mass.clamp_min(epsilon)

    state_real = torch.zeros(
        (batch, modes),
        dtype=drive_real.dtype,
        device=drive_real.device,
    )
    state_imag = torch.zeros_like(state_real)
    level_one = torch.zeros(
        (batch, modes, _EVENT_DIMENSION),
        dtype=working_dtype,
        device=drive_real.device,
    )
    areas = torch.zeros(
        (batch, modes, _SIGNATURE_DIMENSION - _EVENT_DIMENSION),
        dtype=working_dtype,
        device=drive_real.device,
    )
    static_decay = real_decay.ndim == 1
    for index in range(steps):
        previous_real, previous_imag = state_real, state_imag
        if static_decay:
            step_decay_real = real_decay
            step_decay_imag = imag_decay
        else:
            step_decay_real = real_decay[:, index]
            step_decay_imag = imag_decay[:, index]
        source_prediction_real = step_decay_real * previous_real - step_decay_imag * previous_imag
        source_prediction_imag = step_decay_imag * previous_real + step_decay_real * previous_imag
        active_drive_real = real_drive[:, index]
        active_drive_imag = imag_drive[:, index]
        state_real = source_prediction_real + active_drive_real
        state_imag = source_prediction_imag + active_drive_imag

        prediction_real = source_prediction_real.to(dtype=working_dtype)
        prediction_imag = source_prediction_imag.to(dtype=working_dtype)
        event_drive_real = active_drive_real.to(dtype=working_dtype)
        event_drive_imag = active_drive_imag.to(dtype=working_dtype)
        event_previous_real = previous_real.to(dtype=working_dtype)
        event_previous_imag = previous_imag.to(dtype=working_dtype)

        event_weight = (valid[:, index : index + 1] * delta[:, index : index + 1]) / duration
        prediction_power = prediction_real.square() + prediction_imag.square()
        drive_power = event_drive_real.square() + event_drive_imag.square()
        previous_power = event_previous_real.square() + event_previous_imag.square()
        product_scale = torch.sqrt(epsilon + drive_power) * torch.sqrt(epsilon + prediction_power)
        transport_scale = torch.sqrt(epsilon + prediction_power) * torch.sqrt(
            epsilon + previous_power
        )
        increment = torch.stack(
            (
                event_weight.expand(batch, modes),
                event_weight * torch.log1p(prediction_power),
                event_weight * torch.log1p(drive_power),
                event_weight
                * (event_drive_real * prediction_real + event_drive_imag * prediction_imag)
                / product_scale,
                event_weight
                * (event_drive_imag * prediction_real - event_drive_real * prediction_imag)
                / product_scale,
                event_weight
                * (prediction_real * event_previous_real + prediction_imag * event_previous_imag)
                / transport_scale,
                event_weight
                * (prediction_imag * event_previous_real - prediction_real * event_previous_imag)
                / transport_scale,
            ),
            dim=-1,
        )
        areas = areas + 0.5 * (
            level_one[..., _PAIR_FIRST] * increment[..., _PAIR_SECOND]
            - level_one[..., _PAIR_SECOND] * increment[..., _PAIR_FIRST]
        )
        level_one = level_one + increment

    signature = torch.cat((level_one, areas), dim=-1)
    has_support = (duration_mass > 0).view(batch, 1, 1)
    return signature, has_support


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
    block_start = 0
    offsets = tl.arange(0, BLOCK_STEPS)
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
def _streaming_log_signature_kernel(
    decay_real,
    decay_imag,
    drive_real,
    drive_imag,
    valid_mask,
    time_delta,
    duration_input,
    signature_output,
    support_output,
    n_steps: int,
    modes: int,
    epsilon: float,
    STATIC_DECAY: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    HAS_METADATA: tl.constexpr,
    EXTERNAL_DURATION: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes

    duration = n_steps * 1.0
    if HAS_METADATA:
        if EXTERNAL_DURATION:
            duration = tl.load(duration_input + batch).to(tl.float32)
        else:
            duration_mass = tl.full((), 0.0, tl.float32)
            metadata_step = 0
            while metadata_step < n_steps:
                metadata_offset = batch * n_steps + metadata_step
                valid_value = (
                    tl.load(valid_mask + metadata_offset).to(tl.float32) if HAS_VALID else 1.0
                )
                delta_value = (
                    tl.load(time_delta + metadata_offset).to(tl.float32) if HAS_DELTA else 1.0
                )
                duration_mass += valid_value * delta_value
                metadata_step += 1
            duration = tl.maximum(duration_mass, epsilon)
            tl.store(support_output + batch, duration_mass > 0.0, mask=mode_block == 0)

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
        fixed_decay_real = tl.load(
            decay_real + mode,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        fixed_decay_imag = tl.load(
            decay_imag + mode,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)

    step = 0
    while step < n_steps:
        recurrence_offset = (batch * n_steps + step) * modes + mode
        if STATIC_DECAY:
            active_decay_real = fixed_decay_real
            active_decay_imag = fixed_decay_imag
        else:
            active_decay_real = tl.load(
                decay_real + recurrence_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            active_decay_imag = tl.load(
                decay_imag + recurrence_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        active_drive_real = tl.load(
            drive_real + recurrence_offset,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        active_drive_imag = tl.load(
            drive_imag + recurrence_offset,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        previous_real, previous_imag = state_real, state_imag
        prediction_real = active_decay_real * previous_real - active_decay_imag * previous_imag
        prediction_imag = active_decay_imag * previous_real + active_decay_real * previous_imag
        state_real = prediction_real + active_drive_real
        state_imag = prediction_imag + active_drive_imag

        metadata_offset = batch * n_steps + step
        valid_value = tl.load(valid_mask + metadata_offset).to(tl.float32) if HAS_VALID else 1.0
        delta_value = tl.load(time_delta + metadata_offset).to(tl.float32) if HAS_DELTA else 1.0
        event_weight = valid_value * delta_value / duration
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

    if not HAS_METADATA:
        tl.store(support_output + batch, 1, mask=mode_block == 0)
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


def _select_block_modes(batch: int, steps: int, modes: int) -> int:
    maximum_useful = 1 << (modes - 1).bit_length()
    if batch == 1:
        selected = 16 if steps >= 1024 else 8
    elif batch >= 8:
        if steps <= 256:
            selected = 16
        elif steps >= 1024:
            selected = 2
        else:
            selected = 4
    else:
        selected = 8
    return min(selected, maximum_useful)


def _launch_cuda(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    valid_mask: Tensor | None,
    time_delta: Tensor | None,
    *,
    epsilon: float,
    wrapped: bool,
) -> tuple[Tensor, Tensor]:
    batch, steps, modes = drive_real.shape
    output = torch.empty(
        (batch, modes, _SIGNATURE_DIMENSION),
        dtype=torch.float32,
        device=drive_real.device,
    )
    support = torch.empty((batch,), dtype=torch.bool, device=drive_real.device)
    block_modes = _select_block_modes(batch, steps, modes)
    has_metadata = valid_mask is not None or time_delta is not None
    use_external_duration = has_metadata and (modes + block_modes - 1) // block_modes > 1
    duration = (
        torch.empty((batch,), dtype=torch.float32, device=drive_real.device)
        if use_external_duration
        else drive_real
    )
    if use_external_duration:
        duration_kernel = wrap_triton(_duration_kernel) if wrapped else _duration_kernel
        duration_kernel[(batch,)](
            drive_real if valid_mask is None else valid_mask,
            drive_real if time_delta is None else time_delta,
            duration,
            support,
            steps,
            epsilon,
            HAS_VALID=valid_mask is not None,
            HAS_DELTA=time_delta is not None,
            BLOCK_STEPS=256,
            num_warps=4,  # pyright: ignore[reportCallIssue]
        )
    kernel = (
        wrap_triton(_streaming_log_signature_kernel) if wrapped else _streaming_log_signature_kernel
    )
    kernel[(batch * triton.cdiv(modes, block_modes),)](
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        drive_real if valid_mask is None else valid_mask,
        drive_real if time_delta is None else time_delta,
        duration,
        output,
        support,
        steps,
        modes,
        epsilon,
        STATIC_DECAY=decay_real.ndim == 1,
        HAS_VALID=valid_mask is not None,
        HAS_DELTA=time_delta is not None,
        HAS_METADATA=has_metadata,
        EXTERNAL_DURATION=use_external_duration,
        BLOCK_MODES=block_modes,
        num_warps=1,  # pyright: ignore[reportCallIssue]
    )
    return output, support.view(batch, 1, 1)


@triton_op("lnet::pac_streaming_log_signature", mutates_args={})
def _streaming_log_signature_op(
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
) -> tuple[Tensor, Tensor]:
    if not drive_real.is_cuda:
        return reference_streaming_log_signature(
            decay_real,
            decay_imag,
            drive_real,
            drive_imag,
            valid_mask=valid_mask if has_valid else None,
            time_delta=time_delta if has_delta else None,
            epsilon=epsilon,
        )
    return _launch_cuda(
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask if has_valid else None,
        time_delta if has_delta else None,
        epsilon=epsilon,
        wrapped=True,
    )


def _can_use_triton(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    valid_mask: Tensor | None,
    time_delta: Tensor | None,
) -> bool:
    tensors = (decay_real, decay_imag, drive_real, drive_imag)
    metadata = tuple(value for value in (valid_mask, time_delta) if value is not None)
    return (
        drive_real.is_cuda
        and drive_real.dtype == torch.float32
        and all(tensor.is_contiguous() for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in metadata)
    )


def fused_recurrence_log_signature(
    decay_real: Tensor,
    decay_imag: Tensor,
    drive_real: Tensor,
    drive_imag: Tensor,
    *,
    valid_mask: Tensor | None = None,
    time_delta: Tensor | None = None,
    epsilon: float = _DEFAULT_EPSILON,
    backend: LogSignatureBackend = "auto",
) -> tuple[Tensor, Tensor]:
    """Return a state-free degree-two signature and ``[B,1,1]`` support flag.

    ``auto`` uses the CUDA kernel only for contiguous FP32 inference.  Autograd,
    CPU, and unmeasured layouts retain the exact differentiable streaming
    reference.  CUDA half/bfloat16 callers must retain the model's canonical
    materialized recurrence.  ``triton`` requests the CUDA inference path and
    raises if its narrow contract is not satisfied.
    """
    active_real, active_imag, active_valid, active_delta = _prepare_inputs(
        decay_real,
        decay_imag,
        drive_real,
        drive_imag,
        valid_mask,
        time_delta,
        epsilon,
    )
    if backend not in {"auto", "reference", "triton"}:
        message = "backend must be 'auto', 'reference', or 'triton'"
        raise ValueError(message)
    needs_gradients = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (
            active_real,
            active_imag,
            drive_real,
            drive_imag,
            *((active_valid,) if active_valid is not None else ()),
            *((active_delta,) if active_delta is not None else ()),
        )
    )
    supported = _can_use_triton(
        active_real,
        active_imag,
        drive_real,
        drive_imag,
        active_valid,
        active_delta,
    )
    if backend == "triton" and (needs_gradients or not supported):
        message = "Triton log-signature requires contiguous FP32 CUDA inference tensors"
        raise RuntimeError(message)
    if backend != "reference" and supported and not needs_gradients:
        if not torch.compiler.is_compiling():
            return _launch_cuda(
                active_real,
                active_imag,
                drive_real,
                drive_imag,
                active_valid,
                active_delta,
                epsilon=epsilon,
                wrapped=False,
            )
        return _streaming_log_signature_op(
            active_real,
            active_imag,
            drive_real,
            drive_imag,
            drive_real if active_valid is None else active_valid,
            drive_real if active_delta is None else active_delta,
            has_valid=active_valid is not None,
            has_delta=active_delta is not None,
            epsilon=epsilon,
        )
    return reference_streaming_log_signature(
        active_real,
        active_imag,
        drive_real,
        drive_imag,
        valid_mask=active_valid,
        time_delta=active_delta,
        epsilon=epsilon,
    )


__all__ = [
    "LogSignatureBackend",
    "fused_recurrence_log_signature",
    "reference_streaming_log_signature",
]
