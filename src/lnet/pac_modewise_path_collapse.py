"""Phase-equivariant D4 collapse with independent readout per pole mode."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .pac_phase_gated_transition import (
    PhaseGatedModePathMeanCollapse,
)

ComplexField = tuple[Tensor, Tensor]


class ModeWiseComplexLinearCollapse(nn.Module):
    """Collapse paths with one strict complex-linear readout per pole mode."""

    def __init__(self, modes: int, paths: int = 4) -> None:
        super().__init__()
        if min(modes, paths) <= 0:
            message = "mode-wise collapse dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.input_modes = paths
        self.output_modes = 1
        self.weight_real = nn.Parameter(torch.full((modes, paths), 1.0 / paths))
        self.weight_imag = nn.Parameter(torch.zeros(modes, paths))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        expected = (self.modes, self.input_modes)
        if real.shape != imag.shape or tuple(real.shape[-2:]) != expected:
            message = "mode-wise collapse requires matching mode-path coordinates"
            raise ValueError(message)
        compute_dtype = (
            torch.get_autocast_dtype("cuda")
            if real.is_cuda and torch.is_autocast_enabled("cuda")
            else real.dtype
        )
        coordinate_real = real.to(dtype=compute_dtype)
        coordinate_imag = imag.to(dtype=compute_dtype)
        weight_real = self.weight_real.to(dtype=compute_dtype)
        weight_imag = self.weight_imag.to(dtype=compute_dtype)
        output_real = (
            coordinate_real * weight_real - coordinate_imag * weight_imag
        ).sum(dim=-1, keepdim=True).to(dtype=compute_dtype)
        output_imag = (
            coordinate_real * weight_imag + coordinate_imag * weight_real
        ).sum(dim=-1, keepdim=True).to(dtype=compute_dtype)
        return output_real, output_imag


class PhaseGatedModePathResidualModeWiseCollapse(
    PhaseGatedModePathMeanCollapse
):
    """Use Mode PG and Path PG before a mode-wise strict D4 collapse."""

    def __init__(
        self,
        modes: int,
        *,
        mode_hidden: int,
        path_hidden: int = 8,
    ) -> None:
        super().__init__(
            modes,
            mode_hidden=mode_hidden,
            path_hidden=path_hidden,
        )
        self.collapse = ModeWiseComplexLinearCollapse(modes, self.path_count)

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            message = "mode-wise collapse requires NHW-path-mode inputs"
            raise ValueError(message)
        mode_real, mode_imag = self.mode(source_real, source_imag)
        path_real, path_imag = self.path(
            mode_real.transpose(-2, -1),
            mode_imag.transpose(-2, -1),
        )
        collapsed_real, collapsed_imag = self.collapse(path_real, path_imag)
        return collapsed_real.transpose(-2, -1), collapsed_imag.transpose(-2, -1)


__all__ = [
    "ModeWiseComplexLinearCollapse",
    "PhaseGatedModePathResidualModeWiseCollapse",
]
