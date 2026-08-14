"""Fused RMSNorm and packed modal-drive projection for static inference."""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, N803
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional

_BATCHES: Final[tuple[int, ...]] = (1, 32, 64)
_CHANNELS: Final[int] = 64
_PACKED_MODES: Final[int] = 32
_MAX_STEPS: Final[int] = 2048
_BLOCK_STEPS: Final[int] = 8
_RMS_EPSILON: Final[float] = torch.finfo(torch.float32).eps


def _validate_inputs(
    inputs: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> None:
    if inputs.ndim != 3 or inputs.shape[0] not in _BATCHES:
        message = "writer drive fusion requires B1/B32/B64 BTD inputs"
        raise ValueError(message)
    if not 1 <= inputs.shape[1] <= _MAX_STEPS or inputs.shape[2] != _CHANNELS:
        message = "writer drive fusion requires 1<=T<=2048 and D64"
        raise ValueError(message)
    if norm_weight.shape != (_CHANNELS,):
        message = "writer drive fusion requires a D64 RMSNorm weight"
        raise ValueError(message)
    if drive_frame.shape != (_CHANNELS, _PACKED_MODES):
        message = "writer drive fusion requires a 64x32 packed drive frame"
        raise ValueError(message)
    if any(tensor.device != inputs.device for tensor in (norm_weight, drive_frame)):
        message = "writer drive fusion tensors must share one device"
        raise ValueError(message)
    if any(tensor.dtype != inputs.dtype for tensor in (norm_weight, drive_frame)):
        message = "writer drive fusion tensors must share one dtype"
        raise TypeError(message)


def reference_writer_rmsnorm_drive_inference(
    inputs: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the normalized writer stream and its packed complex drive."""
    _validate_inputs(inputs, norm_weight, drive_frame)
    normalized = functional.rms_norm(
        inputs,
        (_CHANNELS,),
        norm_weight,
        _RMS_EPSILON,
    )
    return normalized, torch.matmul(normalized, drive_frame)


@triton.jit
def _writer_rmsnorm_drive_inference_kernel(
    inputs,
    norm_weight,
    drive_frame,
    normalized_output,
    packed_drive,
    steps: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    step = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    channel = tl.arange(0, 64)[None, :]
    valid_step = step < steps
    input_offset = (batch * steps + step) * 64 + channel
    values = tl.load(
        inputs + input_offset,
        mask=valid_step,
        other=0.0,
    ).to(tl.float32)
    mean_square = tl.sum(values * values, axis=1)[:, None] * (1.0 / 64.0)
    inverse_rms = tl.rsqrt(mean_square + 1.1920928955078125e-07)
    normalized = (
        values
        * inverse_rms
        * tl.load(norm_weight + channel).to(tl.float32)
    )
    tl.store(
        normalized_output + input_offset,
        normalized,
        mask=valid_step,
    )

    frame_channel = tl.arange(0, 64)[:, None]
    packed_mode = tl.arange(0, 32)[None, :]
    frame = tl.load(
        drive_frame + frame_channel * 32 + packed_mode,
    ).to(tl.float32)
    drive = tl.dot(normalized, frame, input_precision="tf32x3")
    drive_offset = (batch * steps + step) * 32 + packed_mode
    tl.store(packed_drive + drive_offset, drive, mask=valid_step)


def _launch_cuda(
    inputs: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> tuple[Tensor, Tensor]:
    active_inputs = inputs.contiguous()
    active_weight = norm_weight.contiguous()
    active_frame = drive_frame.contiguous()
    batch, steps, _channels = active_inputs.shape
    normalized = torch.empty_like(active_inputs)
    packed_drive = active_inputs.new_empty((batch, steps, _PACKED_MODES))
    grid = (batch, triton.cdiv(steps, _BLOCK_STEPS))
    wrap_triton(_writer_rmsnorm_drive_inference_kernel)[grid](
        active_inputs,
        active_weight,
        active_frame,
        normalized,
        packed_drive,
        steps,
        BLOCK_T=_BLOCK_STEPS,
        num_warps=4,
    )
    return normalized, packed_drive


@triton_op("lnet::pac_writer_rmsnorm_drive_inference", mutates_args={})
def _writer_rmsnorm_drive_inference_op(
    inputs: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_inputs(inputs, norm_weight, drive_frame)
    if not inputs.is_cuda:
        return reference_writer_rmsnorm_drive_inference(
            inputs,
            norm_weight,
            drive_frame,
        )
    return _launch_cuda(inputs, norm_weight, drive_frame)


def writer_rmsnorm_drive_inference(
    inputs: Tensor,
    norm_weight: Tensor,
    drive_frame: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fuse the canonical writer normalization and static drive projection."""
    _validate_inputs(inputs, norm_weight, drive_frame)
    needs_gradients = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (inputs, norm_weight, drive_frame)
    )
    if inputs.dtype != torch.float32 or needs_gradients:
        return reference_writer_rmsnorm_drive_inference(
            inputs,
            norm_weight,
            drive_frame,
        )
    return _writer_rmsnorm_drive_inference_op(
        inputs,
        norm_weight,
        drive_frame,
    )


__all__ = [
    "reference_writer_rmsnorm_drive_inference",
    "writer_rmsnorm_drive_inference",
]
