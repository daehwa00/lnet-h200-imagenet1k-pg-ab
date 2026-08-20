"""Value-gated PostFusion for complex pole-memory transitions."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_layers import (
    ComplexLinear,
    PackedComplexLinear,
    WidelyLinear,
    packed_complex_linear_weight,
)
from .pac_pole_excitation_transition import PoleExcitationS2DPostFusionTransition
from .pac_triton_complex_rmsnorm import packed_complex_rms_norm

ComplexField = tuple[Tensor, Tensor]


class _GatedPostFusionBase(nn.Module):
    """Shared real-gated residual for strict and widely-linear projections."""

    def __init__(self, modes: int, hidden_modes: int) -> None:
        super().__init__()
        if modes <= 0 or hidden_modes <= 0:
            message = "gated PostFusion dimensions must be positive"
            raise ValueError(message)
        self.modes = int(modes)
        self.hidden_modes = int(hidden_modes)
        self.norm = ComplexRMSNorm(modes)
        self.value = self._make_projection(modes, hidden_modes)
        self.gate = nn.Linear(self._gate_input_modes(modes), hidden_modes, bias=False)
        self.out = self._make_projection(hidden_modes, modes)

    def _make_projection(
        self,
        input_modes: int,
        output_modes: int,
    ) -> ComplexLinear | WidelyLinear:
        _ = input_modes, output_modes
        raise NotImplementedError

    def _gate_input_modes(self, modes: int) -> int:
        return 2 * modes

    def _gate_values(self, unit_real: Tensor, unit_imag: Tensor) -> Tensor:
        gate_input = torch.cat((unit_real, unit_imag), dim=-1)
        return functional.silu(self.gate(gate_input))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "gated PostFusion inputs have incompatible shapes"
            raise ValueError(message)
        unit_real, unit_imag = self.norm(real, imag)
        value_real, value_imag = self.value(unit_real, unit_imag)
        gate = self._gate_values(unit_real, unit_imag)
        update_real, update_imag = self.out(
            value_real * gate,
            value_imag * gate,
        )
        return real + update_real, imag + update_imag


class GatedComplexPostFusion(_GatedPostFusionBase):
    """Refine a complex state through widely-linear value/output projections."""

    def _make_projection(
        self,
        input_modes: int,
        output_modes: int,
    ) -> WidelyLinear:
        return WidelyLinear(input_modes, output_modes, bias=False)


class GatedStrictComplexPostFusion(_GatedPostFusionBase):
    """Run strict-complex PostFusion in one packed activation layout."""

    def _make_projection(
        self,
        input_modes: int,
        output_modes: int,
    ) -> PackedComplexLinear:
        return PackedComplexLinear(input_modes, output_modes)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "gated PostFusion inputs have incompatible shapes"
            raise ValueError(message)
        normalized = packed_complex_rms_norm(
            real,
            imag,
            self.norm.weight,
            self.norm.epsilon,
        )
        value_weight = packed_complex_linear_weight(
            self.value.weight_real,
            self.value.weight_imag,
        )
        joint = functional.linear(
            normalized,
            torch.cat((value_weight, self.gate.weight), dim=0),
        )
        value, gate_logits = joint.split(
            (2 * self.hidden_modes, self.hidden_modes),
            dim=-1,
        )
        gate = functional.silu(gate_logits)
        gated_value = value * torch.cat((gate, gate), dim=-1)
        output_weight = packed_complex_linear_weight(
            self.out.weight_real,
            self.out.weight_imag,
        )
        packed_update = functional.linear(gated_value, output_weight)
        update_real, update_imag = packed_update.split(self.modes, dim=-1)
        return real + update_real, imag + update_imag


class GatedStrictComplexMagnitudePostFusion(GatedStrictComplexPostFusion):
    """Use a positive phase-invariant mean-one gate with strict projections."""

    redistribution = 0.5

    def __init__(self, modes: int, hidden_modes: int) -> None:
        super().__init__(modes, hidden_modes)
        nn.init.zeros_(self.gate.weight)

    def _gate_input_modes(self, modes: int) -> int:
        return modes

    def _gate_values(self, unit_real: Tensor, unit_imag: Tensor) -> Tensor:
        energy = torch.log1p(unit_real.float().square() + unit_imag.float().square())
        centered = energy - energy.mean(dim=-1, keepdim=True)
        logits = self.gate(centered.to(dtype=unit_real.dtype)).float()
        gate = 1.0 + self.redistribution * torch.tanh(logits)
        gate = gate / gate.mean(dim=-1, keepdim=True)
        return gate.to(dtype=unit_real.dtype)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        return _GatedPostFusionBase.forward(self, real, imag)

class GatedPoleExcitationS2DTransition(PoleExcitationS2DPostFusionTransition):
    """Keep the established memory/carry merge and use gated PostFusion."""

    def __init__(
        self,
        pole_modes: int,
        excitation_modes: int,
        output_modes: int,
        *,
        post_hidden: int,
    ) -> None:
        super().__init__(
            pole_modes,
            excitation_modes,
            output_modes,
            post_hidden=post_hidden,
        )
        del self.post_norm
        del self.post_input
        del self.post_output
        self.post_fusion = GatedComplexPostFusion(output_modes, post_hidden)

    def forward(
        self,
        memory_real: Tensor,
        memory_imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if memory_real.shape != memory_imag.shape or memory_real.shape[-1] != self.input_modes:
            message = "gated pole/excitation transition memory has incompatible shape"
            raise ValueError(message)
        if carry_real is None or carry_imag is None:
            message = "gated pole/excitation transition requires S2D carry coordinates"
            raise ValueError(message)

        memory = self._project(
            self.memory_projection,
            (memory_real, memory_imag),
        )
        carry = self._project(
            self.carry_projection,
            self._carry(carry_real, carry_imag),
        )
        return self.post_fusion(
            memory[0] + carry[0],
            memory[1] + carry[1],
        )


def resized_gated_transition(
    source: GatedPoleExcitationS2DTransition,
    post_hidden: int,
) -> GatedPoleExcitationS2DTransition:
    """Resize only the PostFusion workspace and preserve merge projections."""
    replacement = GatedPoleExcitationS2DTransition(
        source.input_modes,
        source.excitation_modes,
        source.output_modes,
        post_hidden=post_hidden,
    )
    with torch.no_grad():
        replacement.carry_weight.copy_(source.carry_weight)
    for target, current in (
        (replacement.memory_projection, source.memory_projection),
        (replacement.carry_projection, source.carry_projection),
    ):
        if target is None and current is None:
            continue
        if target is None or current is None or type(target) is not type(current):
            message = "PostFusion resize changed a branch projection contract"
            raise TypeError(message)
        target.load_state_dict(current.state_dict())
    return replacement


__all__ = [
    "GatedComplexPostFusion",
    "GatedPoleExcitationS2DTransition",
    "GatedStrictComplexMagnitudePostFusion",
    "GatedStrictComplexPostFusion",
    "resized_gated_transition",
]
