"""Scale-controlled complex projections for memory and carry branches."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_layers import ComplexLinear

ComplexField = tuple[Tensor, Tensor]


class PreNormScaledComplexProjection(nn.Module):
    """Apply CRMSNorm, a free complex projection, and a learned branch scale."""

    def __init__(
        self,
        input_modes: int,
        output_modes: int,
        *,
        initial_scale: float,
    ) -> None:
        super().__init__()
        if input_modes <= 0 or output_modes <= 0 or initial_scale <= 0.0:
            raise ValueError("pre-norm projection dimensions and scale must be positive")
        self.input_modes = int(input_modes)
        self.output_modes = int(output_modes)
        self.norm = ComplexRMSNorm(input_modes)
        self.projection = ComplexLinear(input_modes, output_modes)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    @classmethod
    def from_projection(
        cls,
        source: ComplexLinear,
        *,
        initial_scale: float,
    ) -> PreNormScaledComplexProjection:
        reference = source.weight_real
        with torch.random.fork_rng(devices=[]):
            adapter = cls(
                source.input_modes,
                source.output_modes,
                initial_scale=initial_scale,
            ).to(device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            adapter.projection.weight_real.copy_(source.weight_real)
            adapter.projection.weight_imag.copy_(source.weight_imag)
        return adapter

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            raise ValueError("pre-norm complex projection inputs have incompatible shapes")
        unit = self.norm(real, imag)
        projected = self.projection(*unit)
        scale = self.scale.to(dtype=projected[0].dtype)
        return projected[0] * scale, projected[1] * scale


__all__ = ["PreNormScaledComplexProjection"]
