"""Transitions between independent pole-memory and excitation widths."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_ffn import ComplexFFN
from .pac_complex_layers import (
    ComplexLinear,
    WidelyLinear,
    semi_orthogonal_complex_linear_,
)

ComplexField = tuple[Tensor, Tensor]


class PoleExcitationS2DPostFusionTransition(ComplexFFN):
    """Join pole memory and S2D excitation carry at an output width."""

    def __init__(
        self,
        pole_modes: int,
        excitation_modes: int,
        output_modes: int,
        *,
        post_hidden: int,
    ) -> None:
        super().__init__()
        if min(pole_modes, excitation_modes, output_modes, post_hidden) <= 0:
            message = "pole/excitation transition dimensions must be positive"
            raise ValueError(message)
        self.modes = pole_modes
        self.input_modes = pole_modes
        self.excitation_modes = excitation_modes
        self.output_modes = output_modes
        self.carry_input_modes = 4 * excitation_modes
        self.post_hidden = post_hidden
        self.carry_weight = nn.Parameter(torch.full((excitation_modes, 4), 0.25))

        self.memory_projection = self._make_projection(pole_modes, output_modes)
        self.carry_projection = self._make_projection(excitation_modes, output_modes)
        self.post_norm = ComplexRMSNorm(output_modes)
        self.post_input = WidelyLinear(output_modes, post_hidden, bias=True)
        self.post_output = WidelyLinear(post_hidden, output_modes, bias=True)

    @staticmethod
    def _make_projection(
        input_modes: int,
        output_modes: int,
    ) -> ComplexLinear | None:
        if input_modes == output_modes:
            return None
        projection = ComplexLinear(input_modes, output_modes)
        semi_orthogonal_complex_linear_(projection)
        if output_modes > input_modes:
            with torch.no_grad():
                projection.weight_real.mul_((output_modes / input_modes) ** 0.5)
        return projection

    @staticmethod
    def _project(
        projection: ComplexLinear | None,
        field: ComplexField,
    ) -> ComplexField:
        return field if projection is None else projection(*field)

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.carry_input_modes:
            message = "pole/excitation transition carry has incompatible shape"
            raise ValueError(message)
        shape = (*real.shape[:-1], 4, self.excitation_modes)
        weight = self.carry_weight.transpose(0, 1)
        return (
            (real.reshape(shape) * weight).sum(dim=-2),
            (imag.reshape(shape) * weight).sum(dim=-2),
        )

    def forward(
        self,
        memory_real: Tensor,
        memory_imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if memory_real.shape != memory_imag.shape or memory_real.shape[-1] != self.input_modes:
            message = "pole/excitation transition memory has incompatible shape"
            raise ValueError(message)
        if carry_real is None or carry_imag is None:
            message = "pole/excitation transition requires S2D carry coordinates"
            raise ValueError(message)

        memory = self._project(
            self.memory_projection,
            (memory_real, memory_imag),
        )
        carry = self._project(
            self.carry_projection,
            self._carry(carry_real, carry_imag),
        )
        merged = memory[0] + carry[0], memory[1] + carry[1]
        unit = self.post_norm(*merged)
        return self.run_cffn(
            *unit,
            input_projection=self.post_input,
            output_projection=self.post_output,
            activation="cartesian_silu",
            residual_scale=unit[0].new_ones(()),
            residual_source=merged,
        )


__all__ = ["PoleExcitationS2DPostFusionTransition"]
