"""One-kernel affine readout for paired 7M radial-log descriptors."""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional

_BATCHES: Final[tuple[int, ...]] = (32, 64)
_BRANCH_WIDTH: Final[int] = 112
_FEATURE_WIDTH: Final[int] = 224
_CLASS_COUNT: Final[int] = 5


def _validate_inputs(
    writer_moments: Tensor,
    reader_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
) -> None:
    if (
        writer_moments.ndim != 2
        or writer_moments.shape[0] not in _BATCHES
        or writer_moments.shape[1] != _BRANCH_WIDTH
    ):
        message = "modal affine readout requires B32/B64 by 112 writer moments"
        raise ValueError(message)
    expected_shapes = (
        writer_moments.shape,
        (_CLASS_COUNT, _FEATURE_WIDTH),
        (_CLASS_COUNT,),
    )
    tensors = (reader_moments, classifier_weight, classifier_bias)
    if tuple(tensor.shape for tensor in tensors) != expected_shapes:
        message = "modal affine readout received non-canonical shapes"
        raise ValueError(message)
    if any(tensor.device != writer_moments.device for tensor in tensors):
        message = "modal affine readout tensors must share one device"
        raise ValueError(message)
    if any(tensor.dtype != writer_moments.dtype for tensor in tensors):
        message = "modal affine readout tensors must share one dtype"
        raise TypeError(message)


def reference_modal_affine_readout_inference(
    writer_moments: Tensor,
    reader_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
) -> Tensor:
    """Apply the exact public cat-plus-affine descriptor contract."""
    _validate_inputs(
        writer_moments,
        reader_moments,
        classifier_weight,
        classifier_bias,
    )
    return functional.linear(
        torch.cat((writer_moments, reader_moments), dim=-1),
        classifier_weight,
        classifier_bias,
    )


@triton.jit
def _modal_affine_readout_inference_kernel(
    writer_moments,
    reader_moments,
    classifier_weight,
    classifier_bias,
    output,
) -> None:
    batch = tl.program_id(0)
    feature = tl.arange(0, 128)
    feature_mask = feature < 112
    writer = tl.load(
        writer_moments + batch * 112 + feature,
        mask=feature_mask,
        other=0.0,
    ).to(tl.float32)
    reader = tl.load(
        reader_moments + batch * 112 + feature,
        mask=feature_mask,
        other=0.0,
    ).to(tl.float32)
    for class_index in tl.static_range(0, 5):
        weight_base = class_index * 224
        writer_weight = tl.load(
            classifier_weight + weight_base + feature,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        reader_weight = tl.load(
            classifier_weight + weight_base + 112 + feature,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        logit = tl.load(classifier_bias + class_index).to(tl.float32)
        logit += tl.sum(writer * writer_weight, axis=0)
        logit += tl.sum(reader * reader_weight, axis=0)
        tl.store(output + batch * 5 + class_index, logit)


@triton_op("lnet::pac_modal_affine_readout_inference", mutates_args={})
def _modal_affine_readout_inference_op(
    writer_moments: Tensor,
    reader_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
) -> Tensor:
    _validate_inputs(
        writer_moments,
        reader_moments,
        classifier_weight,
        classifier_bias,
    )
    if not writer_moments.is_cuda:
        return reference_modal_affine_readout_inference(
            writer_moments,
            reader_moments,
            classifier_weight,
            classifier_bias,
        )
    output = writer_moments.new_empty((writer_moments.shape[0], _CLASS_COUNT))
    wrap_triton(_modal_affine_readout_inference_kernel)[
        (writer_moments.shape[0],)
    ](
        writer_moments.contiguous(),
        reader_moments.contiguous(),
        classifier_weight.contiguous(),
        classifier_bias.contiguous(),
        output,
        num_warps=4,
    )
    return output


def modal_affine_readout_inference(
    writer_moments: Tensor,
    reader_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
) -> Tensor:
    """Fuse paired-moment concatenation with the canonical affine classifier."""
    _validate_inputs(
        writer_moments,
        reader_moments,
        classifier_weight,
        classifier_bias,
    )
    tensors = (
        writer_moments,
        reader_moments,
        classifier_weight,
        classifier_bias,
    )
    needs_gradients = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in tensors
    )
    if writer_moments.dtype != torch.float32 or needs_gradients:
        return reference_modal_affine_readout_inference(*tensors)
    return _modal_affine_readout_inference_op(*tensors)


__all__ = [
    "modal_affine_readout_inference",
    "reference_modal_affine_readout_inference",
]
