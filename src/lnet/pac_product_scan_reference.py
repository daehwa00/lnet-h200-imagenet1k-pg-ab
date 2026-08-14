"""Readable CPU/reference contracts for the D4 product scan."""

from __future__ import annotations

# pyright: reportCallIssue=false
# ruff: noqa: EM101, TRY003
import torch
from torch import Tensor

from .pac_directional import direction_aligned_cells
from .pac_product_scan_contracts import (
    DEFAULT_EPSILON,
    ComplexField,
    DirectionalState,
    FusedOutputs,
    ProductGainNormalization,
    ProductScans,
    gain_kind,
    validate_product_scan,
)
from .pac_product_scan_normalization import static_variance_tables

BidirectionalField = tuple[Tensor, Tensor, Tensor, Tensor]


def _causal_weights(decay: Tensor, length: int) -> Tensor:
    positions = torch.arange(length, device=decay.device)
    lag = positions[:, None] - positions[None, :]
    powers = decay[None, None, :].pow(lag.clamp_min(0)[..., None])
    return torch.where(lag[..., None] >= 0, powers, torch.zeros_like(powers))


def bidirectional_product_scan_reference(
    pole: tuple[Tensor, Tensor, Tensor, Tensor],
    source: ComplexField,
) -> BidirectionalField:
    """Evaluate the horizontal D4 scan with ordinary differentiable PyTorch."""
    decay_real, decay_imag, gamma_real, gamma_imag = (value.reshape(-1).float() for value in pole)
    source_real, source_imag = source
    source_complex = torch.complex(source_real.float(), source_imag.float())

    def scan(*, reverse: bool) -> ComplexField:
        active_source = source_complex.flip(2) if reverse else source_complex
        decay = torch.complex(decay_real, -decay_imag if reverse else decay_imag)
        gamma = torch.complex(gamma_real, -gamma_imag if reverse else gamma_imag)
        states = torch.einsum(
            "tjm,bhjm->bhtm",
            _causal_weights(decay, source_real.shape[2]),
            gamma * active_source,
        )
        if reverse:
            states = states.flip(2)
        return (
            states.real.to(source_real.dtype).clone(),
            states.imag.to(source_imag.dtype).clone(),
        )

    positive = scan(reverse=False)
    negative = scan(reverse=True)
    return *positive, *negative


def _vertical_product_scan_reference(
    pole: tuple[Tensor, Tensor, Tensor, Tensor],
    source: DirectionalState,
    *,
    reverse: bool,
) -> DirectionalState:
    decay_real, decay_imag, gamma_real, gamma_imag = (value.reshape(-1).float() for value in pole)
    source_real, source_imag, source_variance = source
    variance_decay = decay_real.square().add(decay_imag.square()).detach()
    variance_gamma = gamma_real.square().add(gamma_imag.square()).detach()
    source_complex = torch.complex(source_real.float(), source_imag.float())
    active_source = source_complex.flip(1) if reverse else source_complex
    active_variance = source_variance.float().flip(1) if reverse else source_variance.float()
    decay = torch.complex(decay_real, decay_imag)
    gamma = torch.complex(gamma_real, gamma_imag)
    state = torch.einsum(
        "yjm,bjwm->bywm",
        _causal_weights(decay, source_real.shape[1]),
        gamma * active_source,
    )
    variance = torch.einsum(
        "yjm,bjwm->bywm",
        _causal_weights(variance_decay, source_real.shape[1]),
        variance_gamma * active_variance,
    )
    if reverse:
        state = state.flip(1)
        variance = variance.flip(1)
    return (
        state.real.to(source_real.dtype).clone(),
        state.imag.to(source_imag.dtype).clone(),
        variance.to(source_variance.dtype),
    )


def _product_coarse4_reference(
    scans: ProductScans,
    *,
    epsilon: float,
    gain_normalization: ProductGainNormalization,
) -> ComplexField:
    """Reference endpoint selection used only by the fused op's CPU path."""
    directions = ((1, 1), (-1, 1), (1, -1), (-1, -1))
    real_paths = []
    imag_paths = []
    for (real, imag, variance), (direction_x, direction_y) in zip(
        scans,
        directions,
        strict=True,
    ):
        y_offset = 1 if direction_y == 1 else 0
        x_offset = 1 if direction_x == 1 else 0
        selected_variance = variance[:, y_offset::2, x_offset::2]
        if gain_normalization == "global":
            selected_variance = variance.mean((1, 2), keepdim=True)
        inverse = torch.rsqrt(selected_variance.detach().clamp_min(epsilon))
        real_paths.append(real[:, y_offset::2, x_offset::2] * inverse)
        imag_paths.append(imag[:, y_offset::2, x_offset::2] * inverse)
    return torch.stack(real_paths, dim=-2), torch.stack(imag_paths, dim=-2)


def _product_full16_reference(
    scans: ProductScans,
    *,
    epsilon: float,
    gain_normalization: ProductGainNormalization,
) -> ComplexField:
    """Gather normalized 2x2 states in direction-relative local order."""
    directions = ((1, 1), (-1, 1), (1, -1), (-1, -1))
    real_paths = []
    imag_paths = []
    for (real, imag, variance), (direction_x, direction_y) in zip(
        scans,
        directions,
        strict=True,
    ):
        active_variance = (
            variance.mean((1, 2), keepdim=True)
            if gain_normalization == "global"
            else variance
        )
        inverse = torch.rsqrt(active_variance.detach().clamp_min(epsilon))
        cells = direction_aligned_cells(
            real * inverse,
            imag * inverse,
            direction_x=direction_x,
            direction_y=direction_y,
        )
        real_paths.append(cells[0])
        imag_paths.append(cells[1])
    return torch.stack(real_paths, dim=-3), torch.stack(imag_paths, dim=-3)


def raw_product_descriptor_reference(
    *states: DirectionalState,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> Tensor:
    """Return exact normalized Q for each final product direction and mode."""
    if len(states) != 4:
        raise ValueError("raw product Q requires four final product states")
    parts = []
    for real, imag, variance in states:
        if real.shape != imag.shape or real.shape != variance.shape or real.ndim != 4:
            raise ValueError("raw product Q requires matching NHWM directional states")
        active_variance = (
            variance.mean((1, 2), keepdim=True) if gain_normalization == "global" else variance
        )
        inverse_variance = active_variance.detach().clamp_min(epsilon).reciprocal()
        energy = real.float().square().add(imag.float().square()) * inverse_variance
        parts.append(torch.log1p(energy.mean((1, 2))))
    return torch.cat(parts, dim=-1)


def product_scan_coarse4_reference(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> FusedOutputs:
    """Readable correctness oracle; this function is not used by model dispatch."""
    variance_x, variance_y = static_variance_tables(
        *(value.detach() for value in (*pole_x, *pole_y)),
        source_a[0].shape[2],
        source_a[0].shape[1],
    )
    return _product_scan_coarse4_from_tables_reference(
        pole_y,
        source_a,
        source_b,
        variance_x,
        variance_y,
        epsilon=epsilon,
        gain_normalization=gain_normalization,
    )


def product_scan_descriptor4_reference(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> Tensor:
    """Readable descriptor-only correctness oracle, including odd rectangular grids."""
    variance_x, variance_y = static_variance_tables(
        *(value.detach() for value in (*pole_x, *pole_y)),
        source_a[0].shape[2],
        source_a[0].shape[1],
    )
    return _product_scan_descriptor4_from_tables_reference(
        pole_y,
        source_a,
        source_b,
        variance_x,
        variance_y,
        epsilon=epsilon,
        gain_normalization=gain_normalization,
    )


def product_scan_full16_reference(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> FusedOutputs:
    """Readable full-cell correctness oracle; model dispatch uses the fused op."""
    variance_x, variance_y = static_variance_tables(
        *(value.detach() for value in (*pole_x, *pole_y)),
        source_a[0].shape[2],
        source_a[0].shape[1],
    )
    return _product_scan_full16_from_tables_reference(
        pole_y,
        source_a,
        source_b,
        variance_x,
        variance_y,
        epsilon=epsilon,
        gain_normalization=gain_normalization,
    )


def _product_scans_from_tables_reference(
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    variance_x: Tensor,
) -> ProductScans:
    decay_real, decay_imag, gamma_real, gamma_imag = pole_y
    batch, height, width, modes = source_a[0].shape
    source_variance_a = variance_x[0].view(1, 1, width, modes).expand(batch, height, width, modes)
    source_variance_b = variance_x[1].view(1, 1, width, modes).expand(batch, height, width, modes)
    directional_a = (*source_a, source_variance_a)
    directional_b = (*source_b, source_variance_b)
    positive_a = _vertical_product_scan_reference(pole_y, directional_a, reverse=False)
    positive_b = _vertical_product_scan_reference(pole_y, directional_b, reverse=False)
    negative_pole = (decay_real, -decay_imag, gamma_real, -gamma_imag)
    negative_a = _vertical_product_scan_reference(
        negative_pole,
        directional_a,
        reverse=True,
    )
    negative_b = _vertical_product_scan_reference(
        negative_pole,
        directional_b,
        reverse=True,
    )
    return positive_a, positive_b, negative_a, negative_b


def _product_scan_coarse4_from_tables_reference(
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    variance_x: Tensor,
    variance_y: Tensor,
    *,
    epsilon: float,
    gain_normalization: ProductGainNormalization,
) -> FusedOutputs:
    validate_product_scan(
        *pole_y,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        epsilon,
        gain_kind(gain_normalization),
        emit_coarse=True,
    )
    scans = _product_scans_from_tables_reference(pole_y, source_a, source_b, variance_x)
    coarse_real, coarse_imag = _product_coarse4_reference(
        scans,
        epsilon=epsilon,
        gain_normalization=gain_normalization,
    )
    descriptor = raw_product_descriptor_reference(
        *scans,
        epsilon=epsilon,
        gain_normalization=gain_normalization,
    )
    return coarse_real, coarse_imag, descriptor


def _product_scan_full16_from_tables_reference(
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    variance_x: Tensor,
    variance_y: Tensor,
    *,
    epsilon: float,
    gain_normalization: ProductGainNormalization,
) -> FusedOutputs:
    validate_product_scan(
        *pole_y,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        epsilon,
        gain_kind(gain_normalization),
        emit_coarse=True,
    )
    scans = _product_scans_from_tables_reference(pole_y, source_a, source_b, variance_x)
    full_real, full_imag = _product_full16_reference(
        scans,
        epsilon=epsilon,
        gain_normalization=gain_normalization,
    )
    descriptor = raw_product_descriptor_reference(
        *scans,
        epsilon=epsilon,
        gain_normalization=gain_normalization,
    )
    return full_real, full_imag, descriptor


def _product_scan_descriptor4_from_tables_reference(
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    variance_x: Tensor,
    variance_y: Tensor,
    *,
    epsilon: float,
    gain_normalization: ProductGainNormalization,
) -> Tensor:
    validate_product_scan(
        *pole_y,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        epsilon,
        gain_kind(gain_normalization),
        emit_coarse=False,
    )
    scans = _product_scans_from_tables_reference(pole_y, source_a, source_b, variance_x)
    return raw_product_descriptor_reference(
        *scans,
        epsilon=epsilon,
        gain_normalization=gain_normalization,
    )


__all__ = [
    "bidirectional_product_scan_reference",
    "product_scan_coarse4_reference",
    "product_scan_descriptor4_reference",
    "product_scan_full16_reference",
    "raw_product_descriptor_reference",
]
