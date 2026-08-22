"""Value-gated PostFusion for complex pole-memory transitions."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import ComplexRMSNorm, complex_rms_unit
from .pac_complex_layers import ComplexLinear, PackedComplexLinear, WidelyLinear
from .pac_pole_excitation_transition import PoleExcitationS2DPostFusionTransition
from .pac_triton_complex_rmsnorm import (
    packed_complex_rms_norm,
    supports_packed_complex_rms_norm,
)

ComplexField = tuple[Tensor, Tensor]


class _GatedPostFusionBase(nn.Module):
    """Shared real-gated residual for strict and widely-linear projections.

    The residual is evaluated in the packed ``[real | imag]`` layout: one RMS
    normalization writes the packed rows, one GEMM produces the value and gate
    together, and one GEMM produces the packed update. Both projection families
    expose ``packed_weight``, so the layout is the only thing they share.
    """

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
        for projection in (self.value, self.out):
            if getattr(projection, "bias_real", None) is not None:
                message = "gated PostFusion projections must be bias-free"
                raise ValueError(message)

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

    def _validate(self, real: Tensor, imag: Tensor) -> None:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "gated PostFusion inputs have incompatible shapes"
            raise ValueError(message)

    def _packed_normalized(self, real: Tensor, imag: Tensor) -> Tensor:
        """Return packed normalized rows through the fused kernel when it applies."""
        if supports_packed_complex_rms_norm(real, imag, self.norm.weight):
            return packed_complex_rms_norm(
                real,
                imag,
                self.norm.weight,
                self.norm.epsilon,
            )
        return torch.cat(self.norm(real, imag), dim=-1)

    def _gated_value(self, value: Tensor, gate: Tensor) -> Tensor:
        """Scale both packed coordinates by one gate without widening it."""
        scaled = value.unflatten(-1, (2, self.hidden_modes)) * gate.unsqueeze(-2)
        return scaled.flatten(-2)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        self._validate(real, imag)
        normalized = self._packed_normalized(real, imag)
        joint = functional.linear(
            normalized,
            torch.cat((self.value.packed_weight(), self.gate.weight), dim=0),
        )
        value, gate_logits = joint.split(
            (2 * self.hidden_modes, self.hidden_modes),
            dim=-1,
        )
        gated_value = self._gated_value(value, functional.silu(gate_logits))
        packed_update = functional.linear(gated_value, self.out.packed_weight())
        update_real, update_imag = packed_update.split(self.modes, dim=-1)
        return real + update_real, imag + update_imag

    def _split_forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        """Evaluate the residual one coordinate at a time for custom gates."""
        self._validate(real, imag)
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
        # The phase-invariant gate reads unit coordinates rather than packed
        # rows, so this variant keeps the split residual.
        return self._split_forward(real, imag)


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


class NormalizedShortcutGatedTransition(GatedPoleExcitationS2DTransition):
    """Define a new-width stage through a normalized projection shortcut."""

    def __init__(
        self,
        pole_modes: int,
        excitation_modes: int,
        output_modes: int,
        *,
        post_hidden: int,
        memory_scale_initial: float = 0.1,
    ) -> None:
        if excitation_modes == output_modes:
            raise ValueError("normalized projection shortcuts require a width change")
        if memory_scale_initial < 0.0:
            raise ValueError("normalized shortcut memory scale must be non-negative")
        super().__init__(
            pole_modes,
            excitation_modes,
            output_modes,
            post_hidden=post_hidden,
        )
        self.shortcut_norm = ComplexRMSNorm(output_modes)
        self.memory_norm = ComplexRMSNorm(output_modes)
        self.memory_scale = nn.Parameter(torch.full((output_modes,), float(memory_scale_initial)))

    def forward(
        self,
        memory_real: Tensor,
        memory_imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if memory_real.shape != memory_imag.shape or memory_real.shape[-1] != self.input_modes:
            raise ValueError("normalized shortcut memory has incompatible shape")
        if carry_real is None or carry_imag is None:
            raise ValueError("normalized shortcut requires S2D carry coordinates")
        memory = self._project(
            self.memory_projection,
            (memory_real, memory_imag),
        )
        carry = self._project(
            self.carry_projection,
            self._carry(carry_real, carry_imag),
        )
        shortcut = self.shortcut_norm(*carry)
        memory_unit = self.memory_norm(*memory)
        scale = self.memory_scale.to(dtype=memory_unit[0].dtype)
        merged = (
            shortcut[0] + scale * memory_unit[0],
            shortcut[1] + scale * memory_unit[1],
        )
        return self.post_fusion(*merged)


class ResNetStyleGatedTransition(GatedPoleExcitationS2DTransition):
    """Normalize the memory branch while preserving an identity shortcut."""

    normalization_epsilon = 1.0e-6

    def __init__(
        self,
        pole_modes: int,
        excitation_modes: int,
        output_modes: int,
        *,
        post_hidden: int,
        carry_gain_initial: float = 1.0,
        memory_gain_initial: float = 1.0,
    ) -> None:
        super().__init__(
            pole_modes,
            excitation_modes,
            output_modes,
            post_hidden=post_hidden,
        )
        self.memory_gain = nn.Parameter(torch.full((output_modes,), float(memory_gain_initial)))
        if self.carry_projection is None:
            self.register_parameter("carry_gain", None)
        else:
            self.carry_gain = nn.Parameter(torch.full((output_modes,), float(carry_gain_initial)))

    def forward(
        self,
        memory_real: Tensor,
        memory_imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if memory_real.shape != memory_imag.shape or memory_real.shape[-1] != self.input_modes:
            raise ValueError("ResNet-style transition memory has incompatible shape")
        if carry_real is None or carry_imag is None:
            raise ValueError("ResNet-style transition requires S2D carry coordinates")
        memory = self._project(
            self.memory_projection,
            (memory_real, memory_imag),
        )
        memory_unit = complex_rms_unit(*memory, epsilon=self.normalization_epsilon)
        memory_gain = self.memory_gain.to(dtype=memory_unit[0].dtype)
        source_carry = self._carry(carry_real, carry_imag)
        carry_projection = self.carry_projection
        carry_gain = self.carry_gain
        if carry_projection is None:
            carry = source_carry
        else:
            if carry_gain is None:
                raise RuntimeError("projected shortcut lost its learnable gain")
            projected_carry = carry_projection(*source_carry)
            carry_unit = complex_rms_unit(
                *projected_carry,
                epsilon=self.normalization_epsilon,
            )
            active_gain = carry_gain.to(dtype=carry_unit[0].dtype)
            carry = active_gain * carry_unit[0], active_gain * carry_unit[1]
        merged = (
            carry[0] + memory_gain * memory_unit[0],
            carry[1] + memory_gain * memory_unit[1],
        )
        return self.post_fusion(*merged)


class RMSMatchedShortcutGatedTransition(GatedPoleExcitationS2DTransition):
    """Project a width-changing carry without changing the memory merge contract."""

    variance_epsilon = 1.0e-6

    def __init__(
        self,
        pole_modes: int,
        excitation_modes: int,
        output_modes: int,
        *,
        post_hidden: int,
    ) -> None:
        if excitation_modes == output_modes:
            raise ValueError("RMS-matched shortcuts require a width change")
        super().__init__(
            pole_modes,
            excitation_modes,
            output_modes,
            post_hidden=post_hidden,
        )

    def _match_carry_rms(
        self,
        source: ComplexField,
        projected: ComplexField,
    ) -> ComplexField:
        source_energy = (source[0].float().square() + source[1].float().square()).mean(
            dim=-1, keepdim=True
        )
        projected_energy = (projected[0].float().square() + projected[1].float().square()).mean(
            dim=-1, keepdim=True
        )
        scale = torch.sqrt(
            (source_energy + self.variance_epsilon) / (projected_energy + self.variance_epsilon)
        ).to(dtype=projected[0].dtype)
        return projected[0] * scale, projected[1] * scale

    def forward(
        self,
        memory_real: Tensor,
        memory_imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if memory_real.shape != memory_imag.shape or memory_real.shape[-1] != self.input_modes:
            raise ValueError("RMS-matched shortcut memory has incompatible shape")
        if carry_real is None or carry_imag is None:
            raise ValueError("RMS-matched shortcut requires S2D carry coordinates")

        memory = self._project(
            self.memory_projection,
            (memory_real, memory_imag),
        )
        source_carry = self._carry(carry_real, carry_imag)
        projected_carry = self._project(self.carry_projection, source_carry)
        carry = self._match_carry_rms(source_carry, projected_carry)
        return self.post_fusion(
            memory[0] + carry[0],
            memory[1] + carry[1],
        )


def rms_matched_shortcut_transition(
    source: GatedPoleExcitationS2DTransition,
) -> RMSMatchedShortcutGatedTransition:
    """Replace one projected shortcut while preserving its learned operators."""
    if source.excitation_modes == source.output_modes:
        raise ValueError("source transition does not change excitation width")
    reference = next(source.parameters())
    with torch.random.fork_rng(devices=[]):
        replacement = RMSMatchedShortcutGatedTransition(
            source.input_modes,
            source.excitation_modes,
            source.output_modes,
            post_hidden=source.post_hidden,
        ).to(device=reference.device, dtype=reference.dtype)
    with torch.no_grad():
        replacement.carry_weight.copy_(source.carry_weight)
    for target, current in (
        (replacement.memory_projection, source.memory_projection),
        (replacement.carry_projection, source.carry_projection),
    ):
        if target is None and current is None:
            continue
        if target is None or current is None or type(target) is not type(current):
            raise TypeError("RMS-matched shortcut changed its projection contract")
        target.load_state_dict(current.state_dict())
    replacement.post_fusion.load_state_dict(source.post_fusion.state_dict())
    replacement.train(source.training)
    return replacement


def resnet_style_transition(
    source: GatedPoleExcitationS2DTransition,
    *,
    carry_gain_initial: float = 1.0,
    memory_gain_initial: float = 1.0,
) -> ResNetStyleGatedTransition:
    """Install a projected shortcut without perturbing seeded operators."""
    reference = next(source.parameters())
    with torch.random.fork_rng(devices=[]):
        replacement = ResNetStyleGatedTransition(
            source.input_modes,
            source.excitation_modes,
            source.output_modes,
            post_hidden=source.post_hidden,
            carry_gain_initial=carry_gain_initial,
            memory_gain_initial=memory_gain_initial,
        ).to(device=reference.device, dtype=reference.dtype)
    with torch.no_grad():
        replacement.carry_weight.copy_(source.carry_weight)
    for target, current in (
        (replacement.memory_projection, source.memory_projection),
        (replacement.carry_projection, source.carry_projection),
    ):
        if target is None and current is None:
            continue
        if target is None or current is None or type(target) is not type(current):
            raise TypeError("ResNet-style transition changed its projection contract")
        target.load_state_dict(current.state_dict())
    replacement.post_fusion.load_state_dict(source.post_fusion.state_dict())
    replacement.train(source.training)
    return replacement


def normalized_shortcut_transition(
    source: GatedPoleExcitationS2DTransition,
    *,
    memory_scale_initial: float = 0.1,
) -> NormalizedShortcutGatedTransition:
    """Replace one width-changing transition without perturbing global RNG."""
    if source.excitation_modes == source.output_modes:
        raise ValueError("source transition does not change excitation width")
    reference = next(source.parameters())
    with torch.random.fork_rng(devices=[]):
        replacement = NormalizedShortcutGatedTransition(
            source.input_modes,
            source.excitation_modes,
            source.output_modes,
            post_hidden=source.post_hidden,
            memory_scale_initial=memory_scale_initial,
        ).to(device=reference.device, dtype=reference.dtype)
    with torch.no_grad():
        replacement.carry_weight.copy_(source.carry_weight)
    target_carry = replacement.carry_projection
    source_carry = source.carry_projection
    if target_carry is None or source_carry is None:
        raise TypeError("width-changing shortcut lost its carry projection")
    target_carry.load_state_dict(source_carry.state_dict())
    target_memory = replacement.memory_projection
    source_memory = source.memory_projection
    if target_memory is not None or source.input_modes != source.output_modes:
        if (
            target_memory is None
            or source_memory is None
            or type(target_memory) is not type(source_memory)
        ):
            raise TypeError("normalized shortcut changed its memory projection contract")
        target_memory.load_state_dict(source_memory.state_dict())
    replacement.post_fusion.load_state_dict(source.post_fusion.state_dict())
    replacement.train(source.training)
    return replacement


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
    "NormalizedShortcutGatedTransition",
    "RMSMatchedShortcutGatedTransition",
    "ResNetStyleGatedTransition",
    "normalized_shortcut_transition",
    "resized_gated_transition",
    "resnet_style_transition",
    "rms_matched_shortcut_transition",
]
