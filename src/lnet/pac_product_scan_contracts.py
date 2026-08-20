"""Semantic types and validation shared by D4 product scan implementations."""

from __future__ import annotations

from typing import Final, Literal

import torch
from torch import Tensor

ComplexField = tuple[Tensor, Tensor]
DirectionalState = tuple[Tensor, Tensor, Tensor]
FusedOutputs = tuple[Tensor, Tensor, Tensor]
ProductGainNormalization = Literal["pointwise", "global"]
ProductScans = tuple[
    DirectionalState,
    DirectionalState,
    DirectionalState,
    DirectionalState,
]

DEFAULT_EPSILON: Final = 1.0e-8
POINTWISE_GAIN: Final = 0
GLOBAL_GAIN: Final = 1

_GAIN_KIND_BY_NORMALIZATION: Final[dict[ProductGainNormalization, int]] = {
    "pointwise": POINTWISE_GAIN,
    "global": GLOBAL_GAIN,
}
_GAIN_NORMALIZATION_BY_KIND: Final[dict[int, ProductGainNormalization]] = {
    kind: normalization for normalization, kind in _GAIN_KIND_BY_NORMALIZATION.items()
}
SUPPORTED_STORAGE_DTYPES: Final = frozenset((torch.float32, torch.bfloat16, torch.float16))


def gain_kind(normalization: ProductGainNormalization) -> int:
    try:
        return _GAIN_KIND_BY_NORMALIZATION[normalization]
    except KeyError as error:
        raise ValueError(f"unsupported product gain normalization: {normalization}") from error


def gain_normalization(kind: int) -> ProductGainNormalization:
    try:
        return _GAIN_NORMALIZATION_BY_KIND[kind]
    except KeyError as error:
        raise ValueError(f"unsupported product gain kind: {kind}") from error


def _supports_pac_triton_product_scan4(
    real: Tensor,
    imag: Tensor,
) -> bool:
    return (
        real.is_cuda
        and real.dtype in SUPPORTED_STORAGE_DTYPES
        and imag.shape == real.shape
        and imag.device == real.device
        and imag.dtype == real.dtype
        and real.ndim == 4
    )


def supports_pac_triton_product_scan_coarse4(
    real: Tensor,
    imag: Tensor,
) -> bool:
    """Return whether the fused CUDA scan can emit 2x2 coarse endpoints."""
    return (
        _supports_pac_triton_product_scan4(real, imag)
        and real.shape[1] % 2 == 0
        and real.shape[2] % 2 == 0
    )


def supports_pac_triton_product_scan_descriptor4(
    real: Tensor,
    imag: Tensor,
) -> bool:
    """Return whether the fused CUDA scan can emit a global descriptor.

    Descriptor-only scans do not coarsen, so odd rectangular feature maps are valid.
    Recurrence values are converted to FP32 after loading for every supported
    storage dtype.
    """
    return _supports_pac_triton_product_scan4(real, imag)


def validate_product_scan(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    epsilon: float,
    gain_kind: int,
    *,
    emit_coarse: bool,
) -> None:
    source_shape = source_real_a.shape
    invalid_shape = len(source_shape) != 4 or (
        emit_coarse and (source_shape[1] % 2 or source_shape[2] % 2)
    )
    if invalid_shape:
        qualifier = "even spatial" if emit_coarse else "four-dimensional"
        raise ValueError(f"fused product scan requires {qualifier} NHWM sources")
    if epsilon <= 0.0:
        raise ValueError("fused product scan/coarsening epsilon must be positive")
    coefficients = (decay_real, decay_imag, gamma_real, gamma_imag)
    if any(value.shape != (1, 1, 1, source_shape[-1]) for value in coefficients):
        raise ValueError("fused product scan/coarsening requires compact static 111M poles")
    sources = (
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
    )
    reference = source_real_a
    if any(value.shape != source_shape for value in sources[1:]):
        raise ValueError("fused product scan/coarsening sources must have matching shapes")
    table_shape_x = (2, source_shape[2], source_shape[3])
    table_shape_y = (2, source_shape[1], source_shape[3])
    valid_variance_shapes = (
        variance_x.shape == table_shape_x and variance_y.shape == table_shape_y
    ) or (gain_kind == GLOBAL_GAIN and variance_x.shape == (0,) and variance_y.shape == (0,))
    if not valid_variance_shapes:
        raise ValueError("fused product scan received an invalid x variance tensor")
    if any(
        value.device != reference.device
        for value in (*coefficients, *sources, variance_x, variance_y)
    ):
        raise ValueError("fused product scan/coarsening tensors must share one device")
    if any(value.dtype != coefficients[0].dtype for value in coefficients[1:]):
        raise TypeError("fused product scan/coarsening poles must share one dtype")
    if any(value.dtype != reference.dtype for value in sources[1:]):
        raise TypeError("fused product scan/coarsening sources must share one dtype")
    if coefficients[0].dtype not in SUPPORTED_STORAGE_DTYPES:
        raise TypeError("fused product scan/coarsening poles use an unsupported dtype")
    if reference.dtype not in SUPPORTED_STORAGE_DTYPES:
        raise TypeError("fused product scan/coarsening sources use an unsupported dtype")
    if variance_x.dtype != torch.float32 or variance_y.dtype != torch.float32:
        raise TypeError("static variance tables must use FP32")


def validate_global_inverse_gain(
    inverse_gain: Tensor,
    *,
    modes: int,
    gain_kind: int,
    reference: Tensor,
) -> None:
    expected_shape = (modes,) if gain_kind == GLOBAL_GAIN else (0,)
    if inverse_gain.shape != expected_shape:
        raise ValueError("fused product scan received an invalid global gain tensor")
    if inverse_gain.device != reference.device or inverse_gain.dtype != torch.float32:
        raise TypeError("fused product global gain must be FP32 on the recurrence device")


__all__ = [
    "DEFAULT_EPSILON",
    "SUPPORTED_STORAGE_DTYPES",
    "ComplexField",
    "DirectionalState",
    "FusedOutputs",
    "ProductGainNormalization",
    "ProductScans",
    "gain_kind",
    "gain_normalization",
    "supports_pac_triton_product_scan_coarse4",
    "supports_pac_triton_product_scan_descriptor4",
    "validate_global_inverse_gain",
    "validate_product_scan",
]
