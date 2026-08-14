"""Lossless 2x2 product-state transitions for information-bound experiments."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional

from .pac_grouped_path_cffn import GroupedWidelyLinear
from .pac_path_cffn import D4PathModeCombiner
from .pac_phase_gated_cffn import PhaseGatedComplexFFN
from .pac_product_scan_contracts import DEFAULT_EPSILON
from .pac_product_scan_normalization import static_variance_tables

ComplexField = tuple[Tensor, Tensor]
FullStateBasis = Literal["raw", "innovation"]
Pole = tuple[Tensor, Tensor, Tensor, Tensor]


def select_legacy_product_endpoints(full: Tensor) -> Tensor:
    """Recover the four legacy direction endpoints from a full-cell tensor."""
    if full.ndim != 6 or full.shape[-3] != 4 or full.shape[-2] != 4:
        message = "full product cells must have shape [B,h,w,4,4,M]"
        raise ValueError(message)
    return full[..., 3, :]


def _directional_decays(
    pole_x: Pole,
    pole_y: Pole,
    reference: Tensor,
    *,
    gain_normalization: Literal["pointwise", "global"],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    modes = reference.shape[-1]
    if any(value.numel() != modes for value in (*pole_x, *pole_y)):
        message = "product poles do not match the full-state modal width"
        raise ValueError(message)
    ax_real = pole_x[0].detach().reshape(modes).to(device=reference.device, dtype=reference.dtype)
    ax_imag = pole_x[1].detach().reshape(modes).to(device=reference.device, dtype=reference.dtype)
    ay_real = pole_y[0].detach().reshape(modes).to(device=reference.device, dtype=reference.dtype)
    ay_imag = pole_y[1].detach().reshape(modes).to(device=reference.device, dtype=reference.dtype)
    direction_shape = (1, 1, 1, 4, modes)
    sign_x = reference.new_tensor((1.0, -1.0, 1.0, -1.0)).view(4, 1)
    sign_y = reference.new_tensor((1.0, 1.0, -1.0, -1.0)).view(4, 1)
    if gain_normalization == "global":
        return (
            ax_real.expand(4, modes).reshape(direction_shape),
            (sign_x * ax_imag).reshape(direction_shape),
            ay_real.expand(4, modes).reshape(direction_shape),
            (sign_y * ay_imag).reshape(direction_shape),
        )
    if gain_normalization != "pointwise":
        message = f"unsupported full-state gain normalization: {gain_normalization}"
        raise ValueError(message)

    height, width = 2 * reference.shape[1], 2 * reference.shape[2]
    variance_x, variance_y = static_variance_tables(
        pole_x[0].detach(),
        pole_x[1].detach(),
        pole_x[2].detach(),
        pole_x[3].detach(),
        pole_y[0].detach(),
        pole_y[1].detach(),
        pole_y[2].detach(),
        pole_y[3].detach(),
        width,
        height,
    )
    x_ratios = []
    y_ratios = []
    for direction_x, direction_y in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        x_direction = 0 if direction_x == 1 else 1
        y_direction = 0 if direction_y == 1 else 1
        predecessor_x, endpoint_x = (0, 1) if direction_x == 1 else (1, 0)
        predecessor_y, endpoint_y = (0, 1) if direction_y == 1 else (1, 0)
        x_ratios.append(
            torch.sqrt(
                variance_x[x_direction, predecessor_x::2]
                / variance_x[x_direction, endpoint_x::2].clamp_min(DEFAULT_EPSILON)
            )
        )
        y_ratios.append(
            torch.sqrt(
                variance_y[y_direction, predecessor_y::2]
                / variance_y[y_direction, endpoint_y::2].clamp_min(DEFAULT_EPSILON)
            )
        )
    ratio_x = (
        torch.stack(x_ratios, dim=1).view(1, 1, width // 2, 4, modes).to(dtype=reference.dtype)
    )
    ratio_y = (
        torch.stack(y_ratios, dim=1).view(1, height // 2, 1, 4, modes).to(dtype=reference.dtype)
    )
    return (
        ax_real.expand(4, modes).reshape(direction_shape) * ratio_x,
        (sign_x * ax_imag).reshape(direction_shape) * ratio_x,
        ay_real.expand(4, modes).reshape(direction_shape) * ratio_y,
        (sign_y * ay_imag).reshape(direction_shape) * ratio_y,
    )


def direction_relative_pointwise_gains(
    reference: Tensor,
    *,
    pole_x: Pole,
    pole_y: Pole,
) -> Tensor:
    """Return the exact pointwise gain for every canonical 2x2 direction cell."""
    height, width = 2 * reference.shape[1], 2 * reference.shape[2]
    variance_x, variance_y = static_variance_tables(
        pole_x[0].detach(),
        pole_x[1].detach(),
        pole_x[2].detach(),
        pole_x[3].detach(),
        pole_y[0].detach(),
        pole_y[1].detach(),
        pole_y[2].detach(),
        pole_y[3].detach(),
        width,
        height,
    )
    direction_gains = []
    for direction_x, direction_y in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        x_direction = 0 if direction_x == 1 else 1
        y_direction = 0 if direction_y == 1 else 1
        predecessor_x, endpoint_x = (0, 1) if direction_x == 1 else (1, 0)
        predecessor_y, endpoint_y = (0, 1) if direction_y == 1 else (1, 0)
        x_pre = variance_x[x_direction, predecessor_x::2]
        x_end = variance_x[x_direction, endpoint_x::2]
        y_pre = variance_y[y_direction, predecessor_y::2]
        y_end = variance_y[y_direction, endpoint_y::2]
        direction_gains.append(
            torch.stack(
                (
                    torch.sqrt(y_pre[:, None] * x_pre[None, :]),
                    torch.sqrt(y_end[:, None] * x_pre[None, :]),
                    torch.sqrt(y_pre[:, None] * x_end[None, :]),
                    torch.sqrt(y_end[:, None] * x_end[None, :]),
                ),
                dim=-2,
            )
        )
    gains = torch.stack(direction_gains, dim=2).unsqueeze(0)
    return gains.to(device=reference.device, dtype=reference.dtype)


def _complex_multiply(
    left_real: Tensor,
    left_imag: Tensor,
    right_real: Tensor,
    right_imag: Tensor,
) -> ComplexField:
    return (
        left_real * right_real - left_imag * right_imag,
        left_real * right_imag + left_imag * right_real,
    )


def _complex_divide(
    numerator_real: Tensor,
    numerator_imag: Tensor,
    denominator_real: Tensor,
    denominator_imag: Tensor,
) -> ComplexField:
    inverse_norm = denominator_real.square().add(denominator_imag.square()).reciprocal()
    return (
        (numerator_real * denominator_real + numerator_imag * denominator_imag) * inverse_norm,
        (numerator_imag * denominator_real - numerator_real * denominator_imag) * inverse_norm,
    )


def raw_to_pole_aligned_innovations(
    real: Tensor,
    imag: Tensor,
    *,
    pole_x: Pole,
    pole_y: Pole,
    gain_normalization: Literal["pointwise", "global"] = "pointwise",
) -> ComplexField:
    """Apply the invertible ``(M, Ix, Iy, Ixy)`` basis with detached poles."""
    if real.shape != imag.shape or real.ndim != 6 or tuple(real.shape[-3:-1]) != (4, 4):
        message = "innovation basis requires [B,h,w,4-direction,4-local,M] tensors"
        raise ValueError(message)
    q00r, q10r, q01r, q11r = real.unbind(dim=-2)
    q00i, q10i, q01i, q11i = imag.unbind(dim=-2)
    axr, axi, ayr, ayi = _directional_decays(
        pole_x,
        pole_y,
        real,
        gain_normalization=gain_normalization,
    )
    ax_q10 = _complex_multiply(axr, axi, q10r, q10i)
    ay_q01 = _complex_multiply(ayr, ayi, q01r, q01i)
    axay = _complex_multiply(axr, axi, ayr, ayi)
    axay_q00 = _complex_multiply(*axay, q00r, q00i)

    ix = q11r - ax_q10[0], q11i - ax_q10[1]
    iy = q11r - ay_q01[0], q11i - ay_q01[1]
    ixy = (
        q11r - ax_q10[0] - ay_q01[0] + axay_q00[0],
        q11i - ax_q10[1] - ay_q01[1] + axay_q00[1],
    )
    return (
        torch.stack((q11r, ix[0], iy[0], ixy[0]), dim=-2),
        torch.stack((q11i, ix[1], iy[1], ixy[1]), dim=-2),
    )


def pole_aligned_innovations_to_raw(
    real: Tensor,
    imag: Tensor,
    *,
    pole_x: Pole,
    pole_y: Pole,
    gain_normalization: Literal["pointwise", "global"] = "pointwise",
) -> ComplexField:
    """Invert :func:`raw_to_pole_aligned_innovations` for validation."""
    if real.shape != imag.shape or real.ndim != 6 or tuple(real.shape[-3:-1]) != (4, 4):
        message = "inverse innovation basis requires [B,h,w,4-direction,4-local,M] tensors"
        raise ValueError(message)
    mr, ixr, iyr, ixyr = real.unbind(dim=-2)
    mi, ixi, iyi, ixyi = imag.unbind(dim=-2)
    axr, axi, ayr, ayi = _directional_decays(
        pole_x,
        pole_y,
        real,
        gain_normalization=gain_normalization,
    )
    q10 = _complex_divide(mr - ixr, mi - ixi, axr, axi)
    q01 = _complex_divide(mr - iyr, mi - iyi, ayr, ayi)
    axay = _complex_multiply(axr, axi, ayr, ayi)
    q00 = _complex_divide(
        ixyr + mr - ixr - iyr,
        ixyi + mi - ixi - iyi,
        *axay,
    )
    return (
        torch.stack((q00[0], q10[0], q01[0], mr), dim=-2),
        torch.stack((q00[1], q10[1], q01[1], mi), dim=-2),
    )


class Full16PhaseGatedModeResidualPathCollapse(D4PathModeCombiner):
    """Share one PG mode block over 16 states, then mix and collapse paths."""

    collapses_product_paths = True
    requires_full_product_cells = True

    def __init__(
        self,
        modes: int,
        *,
        mode_hidden: int,
        basis: FullStateBasis,
        path_hidden: int = 32,
    ) -> None:
        super().__init__()
        if basis not in {"raw", "innovation"}:
            message = f"unsupported full-state basis: {basis}"
            raise ValueError(message)
        if min(modes, mode_hidden, path_hidden) <= 0:
            message = "full-state transition dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = 16
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.basis: FullStateBasis = basis
        self.mode = PhaseGatedComplexFFN(modes, mode_hidden)
        self.path_input = GroupedWidelyLinear(modes, 16, path_hidden, bias=True)
        self.path_output = GroupedWidelyLinear(modes, path_hidden, 16, bias=True)
        self.path_collapse = GroupedWidelyLinear(modes, 16, 1, bias=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "full-state transition inputs have incompatible shapes"
            raise ValueError(message)
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            message = "full-state transition requires NHW-path-mode inputs"
            raise ValueError(message)
        mixed_real, mixed_imag = self.mode(source_real, source_imag)
        batch, height, width, _, modes = mixed_real.shape
        packed = torch.cat(
            (
                mixed_real.permute(0, 4, 3, 1, 2),
                mixed_imag.permute(0, 4, 3, 1, 2),
            ),
            dim=2,
        ).reshape(batch, 2 * modes * self.path_count, height, width)
        hidden = functional.silu(
            functional.conv2d(
                packed,
                self.path_input.packed_weight(),
                self.path_input.packed_bias(),
                groups=modes,
            )
        )
        update = functional.conv2d(
            hidden,
            self.path_output.packed_weight(),
            self.path_output.packed_bias(),
            groups=modes,
        )
        output = functional.conv2d(
            packed + update,
            self.path_collapse.packed_weight(),
            self.path_collapse.packed_bias(),
            groups=modes,
        ).reshape(batch, modes, 2, height, width)
        return (
            output[:, :, 0].permute(0, 2, 3, 1).unsqueeze(-2),
            output[:, :, 1].permute(0, 2, 3, 1).unsqueeze(-2),
        )

    def forward_full_state(
        self,
        source_real: Tensor,
        source_imag: Tensor,
        *,
        pole_x: Pole,
        pole_y: Pole,
    ) -> ComplexField:
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 6
            or tuple(source_real.shape[-3:]) != (4, 4, self.modes)
        ):
            message = "full-state transition requires NHW-direction-local-mode inputs"
            raise ValueError(message)
        active = (source_real, source_imag)
        if self.basis == "innovation":
            active = raw_to_pole_aligned_innovations(
                *active,
                pole_x=pole_x,
                pole_y=pole_y,
            )
        shape = (*source_real.shape[:-3], self.path_count, self.modes)
        return self.forward_packed(active[0].reshape(shape), active[1].reshape(shape))


class Full16DualPhaseGatedCollapse(D4PathModeCombiner):
    """Mix all 16 paths and all modes with residual PG blocks before collapse."""

    collapses_product_paths = True
    requires_full_product_cells = True

    def __init__(
        self,
        modes: int,
        *,
        mode_hidden: int,
        path_hidden: int = 32,
    ) -> None:
        super().__init__()
        if min(modes, mode_hidden, path_hidden) <= 0:
            message = "dual-PG transition dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = 16
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.path_mode = PhaseGatedComplexFFN(self.path_count, path_hidden)
        self.mode = PhaseGatedComplexFFN(modes, mode_hidden)
        self.path_collapse = GroupedWidelyLinear(modes, self.path_count, 1, bias=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "dual-PG transition inputs have incompatible shapes"
            raise ValueError(message)
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            message = "dual-PG transition requires NHW-path-mode inputs"
            raise ValueError(message)
        path_real, path_imag = self.path_mode(
            source_real.transpose(-2, -1),
            source_imag.transpose(-2, -1),
        )
        mode_real, mode_imag = self.mode(
            path_real.transpose(-2, -1),
            path_imag.transpose(-2, -1),
        )
        return self.path_collapse(mode_real, mode_imag)

    def forward_full_state(
        self,
        source_real: Tensor,
        source_imag: Tensor,
        *,
        pole_x: Pole,
        pole_y: Pole,
    ) -> ComplexField:
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 6
            or tuple(source_real.shape[-3:]) != (4, 4, self.modes)
        ):
            message = "dual-PG transition requires NHW-direction-local-mode inputs"
            raise ValueError(message)
        del pole_x, pole_y
        shape = (*source_real.shape[:-3], self.path_count, self.modes)
        return self.forward_packed(source_real.reshape(shape), source_imag.reshape(shape))


__all__ = [
    "Full16DualPhaseGatedCollapse",
    "Full16PhaseGatedModeResidualPathCollapse",
    "FullStateBasis",
    "direction_relative_pointwise_gains",
    "pole_aligned_innovations_to_raw",
    "raw_to_pole_aligned_innovations",
    "select_legacy_product_endpoints",
]
