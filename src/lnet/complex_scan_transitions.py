"""Stage-transition modules used by the complex scan backbone."""

from __future__ import annotations

# Variant constructors intentionally remove or replace inherited modules while
# preserving checkpoint attribute names.
# pyright: reportArgumentType=false, reportIncompatibleMethodOverride=false
# pyright: reportIncompatibleUnannotatedOverride=false, reportUnusedFunction=false
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Self, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_complex_ffn import ComplexFFN, ComplexFFNActivation
from .pac_complex_layers import ComplexLinear, WidelyLinear
from .pac_packed_postcarry_inference import (
    PackedCFFNSpec,
    PackedPostCarrySpec,
    PackedPostCarryWeights,
    PackedPostFusionSpec,
    PackedRMSNormSpec,
    cached_packed_weights,
    can_use_packed_postcarry_inference,
    can_use_packed_postfusion_inference,
    can_use_packed_postfusion_training,
    packed_postcarry_inference,
    packed_postfusion_inference,
    packed_postfusion_training,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .complex_scan_types import ComplexCarryMerge, ComplexField


def _dtype_aligned_rms_norm(inputs: Tensor, norm: nn.Module) -> Tensor:
    """Apply RMSNorm without promoting AMP activations through FP32 weights."""
    if not isinstance(norm, nn.RMSNorm):
        return norm(inputs)
    weight = norm.weight
    if weight is None or weight.dtype == inputs.dtype:
        return norm(inputs)
    return functional.rms_norm(
        inputs,
        norm.normalized_shape,
        weight.to(dtype=inputs.dtype),
        norm.eps,
    )


def _weighted_s2d_carry(
    real: Tensor,
    imag: Tensor,
    weight: Tensor,
    modes: int,
) -> ComplexField:
    """Reduce four S2D positions in FP32 while retaining activation storage."""
    shape = (*real.shape[:-1], 4, modes)
    active_weight = weight.transpose(0, 1).to(dtype=real.dtype)
    return (
        (real.reshape(shape) * active_weight).sum(dim=-2).to(dtype=real.dtype),
        (imag.reshape(shape) * active_weight).sum(dim=-2).to(dtype=imag.dtype),
    )


def complex_rms_unit(
    real: Tensor,
    imag: Tensor,
    epsilon: float = 1.0e-6,
) -> ComplexField:
    """Normalize one complex row without introducing a learnable gain."""
    if real.shape != imag.shape:
        raise ValueError("complex RMS inputs must have matching shapes")
    energy = (real.float().square() + imag.float().square()).mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(energy + epsilon).to(dtype=real.dtype)
    return real * inverse_rms, imag * inverse_rms


class ComplexRMSNorm(nn.Module):
    """Global-phase-equivariant RMS normalization with a direct real weight."""

    def __init__(self, modes: int, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        if modes <= 0:
            message = "complex RMS normalization requires positive modes"
            raise ValueError(message)
        self.modes = modes
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(modes))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "complex RMS normalization inputs have incompatible shapes"
            raise ValueError(message)
        unit_real, unit_imag = complex_rms_unit(real, imag, self.epsilon)
        weight = self.weight.to(dtype=real.dtype)
        return unit_real * weight, unit_imag * weight


class ComplexModulatedTransition(ComplexFFN):
    """Wide phase-equivariant complex FFN between consecutive scan stages."""

    def __init__(
        self,
        input_modes: int,
        hidden_modes: int,
        output_modes: int,
        *,
        expansion: int = 2,
        amplitude_radius: float = 0.4,
        maximum_phase_shift: float = math.pi / 12.0,
        layer_scale_initial: float = 1.0e-2,
    ) -> None:
        super().__init__()
        if input_modes <= 0 or hidden_modes <= 0 or output_modes <= 0:
            message = "complex transition dimensions must be positive"
            raise ValueError(message)
        if expansion <= 0:
            message = "complex transition expansion must be positive"
            raise ValueError(message)
        if not 0.0 < amplitude_radius <= 1.0:
            message = "complex transition amplitude radius must be in (0, 1]"
            raise ValueError(message)
        if not 0.0 <= maximum_phase_shift <= math.pi:
            message = "complex transition phase bound must be in [0, pi]"
            raise ValueError(message)
        if layer_scale_initial <= 0.0:
            message = "complex transition LayerScale initialization must be positive"
            raise ValueError(message)
        expanded_modes = expansion * hidden_modes
        self.input_modes = input_modes
        self.hidden_modes = hidden_modes
        self.output_modes = output_modes
        self.expanded_modes = expanded_modes
        self.amplitude_radius = amplitude_radius
        self.maximum_phase_shift = maximum_phase_shift
        self.input_projection = ComplexLinear(input_modes, hidden_modes)
        self.input_norm = ComplexRMSNorm(hidden_modes)
        self.carrier_projection = ComplexLinear(hidden_modes, expanded_modes)
        self.residual_projection = ComplexLinear(expanded_modes, hidden_modes)
        # These zero-initialized real gates are state controllers, not carriers.
        # Isolate their construction so they do not perturb common complex weights.
        with torch.random.fork_rng(devices=[]):
            self.amplitude_gate = nn.Linear(hidden_modes, expanded_modes)
            self.phase_gate = nn.Linear(hidden_modes, expanded_modes)
        nn.init.zeros_(self.amplitude_gate.weight)
        nn.init.zeros_(self.amplitude_gate.bias)
        nn.init.zeros_(self.phase_gate.weight)
        nn.init.zeros_(self.phase_gate.bias)
        self.layer_scale = nn.Parameter(torch.full((hidden_modes,), layer_scale_initial))
        self.output_norm = ComplexRMSNorm(hidden_modes)
        self.output_projection = ComplexLinear(hidden_modes, output_modes)

    def modulation(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        """Return positive amplitude gain and bounded phase shift."""
        energy = torch.log1p(real.float().square() + imag.float().square())
        amplitude_logits = self.amplitude_gate(energy).float()
        phase_logits = self.phase_gate(energy).float()
        gain = torch.exp(self.amplitude_radius * torch.tanh(amplitude_logits))
        phase = self.maximum_phase_shift * torch.tanh(phase_logits)
        return gain.to(dtype=real.dtype), phase.to(dtype=real.dtype)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "complex transition inputs have incompatible shapes"
            raise ValueError(message)
        hidden_real, hidden_imag = self.input_projection(real, imag)
        unit_real, unit_imag = self.input_norm(hidden_real, hidden_imag)
        carrier_real, carrier_imag = self.carrier_projection(unit_real, unit_imag)
        gain, phase = self.modulation(unit_real, unit_imag)
        cosine = torch.cos(phase)
        sine = torch.sin(phase)
        modulated_real = gain * (cosine * carrier_real - sine * carrier_imag)
        modulated_imag = gain * (cosine * carrier_imag + sine * carrier_real)
        residual_real, residual_imag = self.residual_projection(
            modulated_real,
            modulated_imag,
        )
        layer_scale = self.layer_scale.to(dtype=hidden_real.dtype)
        updated_real = hidden_real + layer_scale * residual_real
        updated_imag = hidden_imag + layer_scale * residual_imag
        normalized_real, normalized_imag = self.output_norm(updated_real, updated_imag)
        return self.output_projection(normalized_real, normalized_imag)


class ComplexInteractionTransition(ComplexFFN):
    """Complex mode composition gated by low-rank relative-phase coherence."""

    def __init__(
        self,
        input_modes: int,
        hidden_modes: int,
        output_modes: int,
        *,
        coherence_rank: int = 8,
        layer_scale_initial: float = 1.0e-2,
    ) -> None:
        super().__init__()
        if input_modes <= 0 or hidden_modes <= 0 or output_modes <= 0:
            message = "complex interaction dimensions must be positive"
            raise ValueError(message)
        if coherence_rank <= 0:
            message = "complex interaction coherence rank must be positive"
            raise ValueError(message)
        if layer_scale_initial <= 0.0:
            message = "complex interaction LayerScale initialization must be positive"
            raise ValueError(message)
        self.input_modes = input_modes
        self.hidden_modes = hidden_modes
        self.output_modes = output_modes
        self.coherence_rank = coherence_rank
        self.input_projection = ComplexLinear(input_modes, hidden_modes)
        self.input_norm = ComplexRMSNorm(hidden_modes)
        self.coherence_left = ComplexLinear(hidden_modes, coherence_rank)
        self.coherence_right = ComplexLinear(hidden_modes, coherence_rank)
        context_modes = hidden_modes + 2 * coherence_rank
        with torch.random.fork_rng(devices=[]):
            self.gate = nn.Linear(context_modes, hidden_modes)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.composition = ComplexLinear(hidden_modes, hidden_modes)
        self.layer_scale = nn.Parameter(torch.full((hidden_modes,), layer_scale_initial))
        self.output_norm = ComplexRMSNorm(hidden_modes)
        self.output_projection = ComplexLinear(hidden_modes, output_modes)

    def coherence_context(self, real: Tensor, imag: Tensor) -> Tensor:
        """Return energy and low-rank relative-phase invariants."""
        energy = torch.log1p(real.float().square() + imag.float().square())
        left_real, left_imag = self.coherence_left(real, imag)
        right_real, right_imag = self.coherence_right(real, imag)
        coherence_real = (
            left_real.float() * right_real.float() + left_imag.float() * right_imag.float()
        )
        coherence_imag = (
            left_imag.float() * right_real.float() - left_real.float() * right_imag.float()
        )
        return torch.cat((energy, coherence_real, coherence_imag), dim=-1)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "complex interaction inputs have incompatible shapes"
            raise ValueError(message)
        hidden_real, hidden_imag = self.input_projection(real, imag)
        unit_real, unit_imag = self.input_norm(hidden_real, hidden_imag)
        context = self.coherence_context(unit_real, unit_imag)
        # Twice-sigmoid keeps the initialized gate exactly one while retaining
        # a bounded positive range, so it cannot induce a pi phase flip.
        gain = (2.0 * torch.sigmoid(self.gate(context))).to(dtype=unit_real.dtype)
        composed_real, composed_imag = self.composition(
            gain * unit_real,
            gain * unit_imag,
        )
        layer_scale = self.layer_scale.to(dtype=hidden_real.dtype)
        updated_real = hidden_real + layer_scale * composed_real
        updated_imag = hidden_imag + layer_scale * composed_imag
        normalized_real, normalized_imag = self.output_norm(updated_real, updated_imag)
        return self.output_projection(normalized_real, normalized_imag)


class AugmentedComplexTransition(ComplexFFN):
    """Widely-linear Cartesian-SiLU residual FFN with optional stage carry."""

    def __init__(
        self,
        input_modes: int,
        hidden_modes: int,
        output_modes: int,
        *,
        expansion: int = 2,
        layer_scale_initial: float = 1.0e-3,
        carry_input_modes: int | None = None,
        carry_merge: ComplexCarryMerge = "pole_main",
        carry_scale_initial: float = 1.0e-2,
        coherence_gated_carry: bool = False,
    ) -> None:
        super().__init__()
        if input_modes <= 0 or hidden_modes <= 0 or output_modes <= 0:
            message = "augmented complex transition dimensions must be positive"
            raise ValueError(message)
        if expansion <= 0 or layer_scale_initial <= 0.0:
            message = "augmented complex expansion and LayerScale must be positive"
            raise ValueError(message)
        if carry_input_modes is not None and carry_input_modes <= 0:
            message = "augmented complex carry input modes must be positive"
            raise ValueError(message)
        if carry_merge not in {"pole_main", "carry_main"}:
            message = f"unsupported augmented complex carry merge: {carry_merge}"
            raise ValueError(message)
        if carry_scale_initial < 0.0:
            message = "augmented complex carry scale cannot be negative"
            raise ValueError(message)
        if coherence_gated_carry and carry_input_modes is None:
            message = "coherence-gated carry requires carry inputs"
            raise ValueError(message)
        expanded_modes = expansion * hidden_modes
        self.input_modes = input_modes
        self.hidden_modes = hidden_modes
        self.output_modes = output_modes
        self.carry_merge: ComplexCarryMerge = carry_merge
        self.coherence_gated_carry = coherence_gated_carry
        self.direction_mixer = WidelyLinear(
            input_modes,
            hidden_modes,
            bias=True,
        )
        self.carry_mixer: WidelyLinear | None
        self.carry_scale: nn.Parameter | None
        # Keep the RNG stream of the established augmented transition unchanged,
        # so pole-main with a zero carry scale is an exact matched control.
        with torch.random.fork_rng(devices=[]):
            self.carry_mixer = (
                WidelyLinear(
                    carry_input_modes,
                    hidden_modes,
                    bias=True,
                )
                if carry_input_modes is not None
                else None
            )
        if self.carry_mixer is None:
            self.register_parameter("carry_scale", None)
        else:
            self.carry_scale = nn.Parameter(torch.full((hidden_modes,), carry_scale_initial))
        if coherence_gated_carry:
            self.carry_gate_pole_energy = nn.Parameter(torch.zeros(hidden_modes))
            self.carry_gate_carry_energy = nn.Parameter(torch.zeros(hidden_modes))
            self.carry_gate_coherence = nn.Parameter(torch.zeros(hidden_modes))
            self.carry_gate_bias = nn.Parameter(torch.zeros(hidden_modes))
        else:
            self.register_parameter("carry_gate_pole_energy", None)
            self.register_parameter("carry_gate_carry_energy", None)
            self.register_parameter("carry_gate_coherence", None)
            self.register_parameter("carry_gate_bias", None)
        self.ffn_norm = ComplexRMSNorm(hidden_modes)
        self.ffn_input = WidelyLinear(
            hidden_modes,
            expanded_modes,
            bias=True,
        )
        self.ffn_output = WidelyLinear(
            expanded_modes,
            hidden_modes,
            bias=True,
        )
        self.layer_scale = nn.Parameter(torch.full((hidden_modes,), layer_scale_initial))
        self.output_norm = ComplexRMSNorm(hidden_modes)
        self.output_projection = WidelyLinear(
            hidden_modes,
            output_modes,
            bias=True,
        )

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "augmented complex transition inputs have incompatible shapes"
            raise ValueError(message)
        hidden_real, hidden_imag = self.direction_mixer(real, imag)
        if self.carry_mixer is None:
            if carry_real is not None or carry_imag is not None:
                message = "carry inputs were passed to a transition without a carry mixer"
                raise ValueError(message)
        else:
            if carry_real is None or carry_imag is None or self.carry_scale is None:
                message = "augmented complex carry transition is missing carry inputs"
                raise ValueError(message)
            carried_real, carried_imag = self.carry_mixer(carry_real, carry_imag)
            if self.coherence_gated_carry:
                pole_weight = self.carry_gate_pole_energy
                carry_weight = self.carry_gate_carry_energy
                coherence_weight = self.carry_gate_coherence
                gate_bias = self.carry_gate_bias
                if (
                    pole_weight is None
                    or carry_weight is None
                    or coherence_weight is None
                    or gate_bias is None
                ):
                    message = "coherence-gated carry is missing gate parameters"
                    raise RuntimeError(message)
                pole_energy = hidden_real.float().square() + hidden_imag.float().square()
                carry_energy = carried_real.float().square() + carried_imag.float().square()
                coherence = (
                    hidden_real.float() * carried_real.float()
                    + hidden_imag.float() * carried_imag.float()
                ) * torch.rsqrt((pole_energy * carry_energy).clamp_min(1.0e-8))
                gate_logits = (
                    pole_weight * torch.log1p(pole_energy)
                    + carry_weight * torch.log1p(carry_energy)
                    + coherence_weight * coherence.clamp(-1.0, 1.0)
                    + gate_bias
                )
                carry_gain = (2.0 * torch.sigmoid(gate_logits)).to(dtype=carried_real.dtype)
                carried_real = carry_gain * carried_real
                carried_imag = carry_gain * carried_imag
            scale = self.carry_scale.to(dtype=hidden_real.dtype)
            if self.carry_merge == "pole_main":
                hidden_real = hidden_real + scale * carried_real
                hidden_imag = hidden_imag + scale * carried_imag
            else:
                hidden_real = carried_real + scale * hidden_real
                hidden_imag = carried_imag + scale * hidden_imag
        unit_real, unit_imag = self.ffn_norm(hidden_real, hidden_imag)
        updated_real, updated_imag = self.run_cffn(
            unit_real,
            unit_imag,
            input_projection=self.ffn_input,
            output_projection=self.ffn_output,
            activation="cartesian_silu",
            residual_scale=self.layer_scale,
            residual_source=(hidden_real, hidden_imag),
        )
        normalized_real, normalized_imag = self.output_norm(updated_real, updated_imag)
        return self.output_projection(normalized_real, normalized_imag)


class S2DPostCFFNCarryMainTransition(ComplexFFN):
    """Add a completed pole CFFN update to a mode-wise S2D carry state."""

    def __init__(
        self,
        modes: int,
        hidden_modes: int,
        *,
        output_modes: int | None = None,
        pole_paths: int = 4,
        expansion: int = 2,
        layer_scale_initial: float = 1.0e-3,
        pole_scale_initial: float = 1.0,
        ffn_activation: ComplexFFNActivation = "cartesian_silu",
    ) -> None:
        super().__init__()
        active_output_modes = output_modes or modes
        if min(modes, hidden_modes, active_output_modes, pole_paths, expansion) <= 0:
            message = "post-CFFN carry-main dimensions must be positive"
            raise ValueError(message)
        if layer_scale_initial <= 0.0 or pole_scale_initial < 0.0:
            message = "post-CFFN carry-main scales are invalid"
            raise ValueError(message)
        expanded_modes = expansion * hidden_modes
        self.modes = modes
        self.pole_paths = pole_paths
        self.input_modes = pole_paths * modes
        self.carry_input_modes = 4 * modes
        self.hidden_modes = hidden_modes
        self.output_modes = active_output_modes
        self.ffn_activation: ComplexFFNActivation = ffn_activation
        self.ffn_activation_scale: Tensor | None
        self.register_optional_persistent_buffer("ffn_activation_scale")
        self.direction_mixer = WidelyLinear(
            self.input_modes,
            hidden_modes,
            bias=True,
        )
        self.ffn_norm = ComplexRMSNorm(hidden_modes)
        self.ffn_input = WidelyLinear(
            hidden_modes,
            expanded_modes,
            bias=True,
        )
        self.ffn_output = WidelyLinear(
            expanded_modes,
            hidden_modes,
            bias=True,
        )
        self.layer_scale = nn.Parameter(torch.full((hidden_modes,), layer_scale_initial))
        self.output_norm = ComplexRMSNorm(hidden_modes)
        self.output_projection = WidelyLinear(
            hidden_modes,
            active_output_modes,
            bias=True,
        )
        # One real coefficient per (mode, 2x2 position).  This path has no
        # bias, normalization, activation, conjugate term, or cross-mode mix.
        if active_output_modes == modes:
            self.carry_weight = nn.Parameter(torch.full((modes, 4), 0.25))
            self.carry_projection = None
        else:
            self.register_parameter("carry_weight", None)
            self.carry_projection = ComplexLinear(
                self.carry_input_modes,
                active_output_modes,
            )
            with torch.no_grad():
                mode_projection = torch.empty(active_output_modes, modes)
                nn.init.orthogonal_(mode_projection)
                self.carry_projection.weight_real.zero_()
                self.carry_projection.weight_imag.zero_()
                for position in range(4):
                    start = position * modes
                    self.carry_projection.weight_real[:, start : start + modes].copy_(
                        0.25 * mode_projection
                    )
        # A stage scalar keeps the first intervention maximally identifiable.
        self.pole_scale = nn.Parameter(torch.tensor(pole_scale_initial))
        # Fused packed blocks for inference.  Deliberately not a buffer: the
        # cache is derived state and must stay out of ``state_dict``.
        self._packed_inference_cache: PackedPostCarryWeights | None = None

    @override
    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> Self:
        """Drop derived storage before parameters move across device or dtype."""
        self._packed_inference_cache = None
        return super()._apply(fn, recurse=recurse)

    @override
    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Invalidate derived blocks before checkpoint parameters are copied."""
        self._packed_inference_cache = None
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _base_packed_postcarry_spec(self) -> PackedPostCarrySpec:
        """Build the base capability after a concrete class opts into it."""
        carry_projection = self.carry_projection
        carry_weight = None if carry_projection is not None else self.carry_weight
        return PackedPostCarrySpec(
            modes=self.modes,
            input_modes=self.input_modes,
            hidden_modes=self.hidden_modes,
            output_modes=self.output_modes,
            training=self.training,
            direction_mixer=self.direction_mixer,
            ffn_norm=PackedRMSNormSpec(
                weight=self.ffn_norm.weight,
                epsilon=self.ffn_norm.epsilon,
            ),
            ffn=PackedCFFNSpec(
                input_projection=self.ffn_input,
                output_projection=self.ffn_output,
                activation=self.ffn_activation,
                activation_scale=self.ffn_activation_scale,
                residual_scale=self.layer_scale,
            ),
            output_norm=PackedRMSNormSpec(
                weight=self.output_norm.weight,
                epsilon=self.output_norm.epsilon,
            ),
            output_projection=self.output_projection,
            pole_scale=self.pole_scale,
            carry_weight=carry_weight,
            carry_projection=carry_projection,
        )

    def packed_postcarry_spec(self) -> PackedPostCarrySpec | None:
        """Opt only the verified concrete base transition into packed inference."""
        if type(self) is not S2DPostCFFNCarryMainTransition:
            return None
        return self._base_packed_postcarry_spec()

    def copy_pole_branch_from(
        self,
        source: AugmentedComplexTransition | S2DPostCFFNCarryMainTransition,
    ) -> None:
        """Copy a compatible A2D pole branch without its carry mixer."""
        if (
            source.input_modes != self.input_modes
            or source.hidden_modes != self.hidden_modes
            or source.output_modes != self.output_modes
        ):
            message = "source augmented transition does not match post-CFFN carry-main"
            raise ValueError(message)
        self.direction_mixer.load_state_dict(source.direction_mixer.state_dict())
        self.ffn_norm.load_state_dict(source.ffn_norm.state_dict())
        self.ffn_input.load_state_dict(source.ffn_input.state_dict())
        self.ffn_output.load_state_dict(source.ffn_output.state_dict())
        self.layer_scale.data.copy_(source.layer_scale.data)
        self.output_norm.load_state_dict(source.output_norm.state_dict())
        self.output_projection.load_state_dict(source.output_projection.state_dict())

    def _modewise_carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.carry_input_modes:
            message = "post-CFFN carry-main S2D inputs have incompatible shapes"
            raise ValueError(message)
        if self.carry_projection is not None:
            return self.carry_projection(real, imag)
        return _weighted_s2d_carry(real, imag, self.carry_weight, self.modes)

    def pole_update(self, real: Tensor, imag: Tensor) -> ComplexField:
        """Complete the pole-only branch through its 48-mode projection."""
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "post-CFFN carry-main pole inputs have incompatible shapes"
            raise ValueError(message)
        hidden_real, hidden_imag = self.direction_mixer(real, imag)
        unit_real, unit_imag = self.ffn_norm(hidden_real, hidden_imag)
        updated_real, updated_imag = self.run_cffn(
            unit_real,
            unit_imag,
            input_projection=self.ffn_input,
            output_projection=self.ffn_output,
            activation=self.ffn_activation,
            activation_scale=self.ffn_activation_scale,
            residual_scale=self.layer_scale,
            residual_source=(hidden_real, hidden_imag),
        )
        normalized_real, normalized_imag = self.output_norm(
            updated_real,
            updated_imag,
        )
        return self.output_projection(normalized_real, normalized_imag)

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if carry_real is None or carry_imag is None:
            message = "post-CFFN carry-main transition requires S2D coordinates"
            raise ValueError(message)
        if can_use_packed_postcarry_inference(self, real, imag, carry_real, carry_imag):
            return packed_postcarry_inference(
                self,
                real,
                imag,
                carry_real,
                carry_imag,
                weights=cached_packed_weights(self, real),
            )
        pole_real, pole_imag = self.pole_update(real, imag)
        carry_state_real, carry_state_imag = self._modewise_carry(
            carry_real,
            carry_imag,
        )
        scale = self.pole_scale.to(dtype=pole_real.dtype)
        return (
            carry_state_real + scale * pole_real,
            carry_state_imag + scale * pole_imag,
        )


class S2DPostFusionCFFNTransition(S2DPostCFFNCarryMainTransition):
    """Apply a pre-normalized mode CFFN after the PostCarry outer residual."""

    def __init__(
        self,
        modes: int,
        hidden_modes: int,
        *,
        output_modes: int | None = None,
        pole_paths: int = 4,
        expansion: int = 2,
        layer_scale_initial: float = 1.0e-3,
        pole_scale_initial: float = 1.0,
        post_hidden_modes: int | None = None,
        post_layer_scale_initial: float = 0.1,
        ffn_activation: ComplexFFNActivation = "cartesian_silu",
        post_ffn_activation: ComplexFFNActivation = "cartesian_silu",
    ) -> None:
        super().__init__(
            modes,
            hidden_modes,
            output_modes=output_modes,
            pole_paths=pole_paths,
            expansion=expansion,
            layer_scale_initial=layer_scale_initial,
            pole_scale_initial=pole_scale_initial,
            ffn_activation=ffn_activation,
        )
        active_output_modes = self.output_modes
        active_post_hidden_modes = post_hidden_modes or 2 * active_output_modes
        if active_post_hidden_modes <= 0 or post_layer_scale_initial < 0.0:
            message = "post-fusion CFFN width and scale are invalid"
            raise ValueError(message)
        self.post_hidden_modes = active_post_hidden_modes
        self.post_ffn_activation: ComplexFFNActivation = post_ffn_activation
        self.post_ffn_activation_scale: Tensor | None
        self.register_optional_persistent_buffer("post_ffn_activation_scale")
        self.post_ffn_norm = ComplexRMSNorm(active_output_modes)
        self.post_ffn_input = WidelyLinear(
            active_output_modes,
            active_post_hidden_modes,
            bias=True,
        )
        self.post_ffn_output = WidelyLinear(
            active_post_hidden_modes,
            active_output_modes,
            bias=True,
        )
        self.post_ffn_scale = nn.Parameter(
            torch.full((active_output_modes,), post_layer_scale_initial)
        )

    def packed_postcarry_spec(self) -> PackedPostCarrySpec | None:
        """Extend the base capability atomically with the post-fusion CFFN."""
        if type(self) is not S2DPostFusionCFFNTransition:
            return None
        base = self._base_packed_postcarry_spec()
        return replace(
            base,
            post_fusion=PackedPostFusionSpec(
                norm=PackedRMSNormSpec(
                    weight=self.post_ffn_norm.weight,
                    epsilon=self.post_ffn_norm.epsilon,
                ),
                cffn=PackedCFFNSpec(
                    input_projection=self.post_ffn_input,
                    output_projection=self.post_ffn_output,
                    activation=self.post_ffn_activation,
                    activation_scale=self.post_ffn_activation_scale,
                    residual_scale=self.post_ffn_scale,
                ),
            ),
        )

    def outer_residual(
        self,
        real: Tensor,
        imag: Tensor,
        carry_real: Tensor,
        carry_imag: Tensor,
    ) -> ComplexField:
        """Return H = C + alpha F before the joint mode CFFN."""
        pole_real, pole_imag = self.pole_update(real, imag)
        carry_state_real, carry_state_imag = self._modewise_carry(
            carry_real,
            carry_imag,
        )
        scale = self.pole_scale.to(dtype=pole_real.dtype)
        return (
            carry_state_real + scale * pole_real,
            carry_state_imag + scale * pole_imag,
        )

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if carry_real is None or carry_imag is None:
            message = "post-fusion CFFN transition requires S2D coordinates"
            raise ValueError(message)
        if can_use_packed_postfusion_training(self, real, imag, carry_real, carry_imag):
            return packed_postfusion_training(
                self,
                real,
                imag,
                carry_real,
                carry_imag,
            )
        if can_use_packed_postfusion_inference(self, real, imag, carry_real, carry_imag):
            return packed_postfusion_inference(
                self,
                real,
                imag,
                carry_real,
                carry_imag,
                weights=cached_packed_weights(self, real),
            )
        outer_real, outer_imag = self.outer_residual(
            real,
            imag,
            carry_real,
            carry_imag,
        )
        unit_real, unit_imag = self.post_ffn_norm(outer_real, outer_imag)
        return self.run_cffn(
            unit_real,
            unit_imag,
            input_projection=self.post_ffn_input,
            output_projection=self.post_ffn_output,
            activation=self.post_ffn_activation,
            activation_scale=self.post_ffn_activation_scale,
            residual_scale=self.post_ffn_scale,
            residual_source=(outer_real, outer_imag),
        )


class ComplexResidualFFN(ComplexFFN):
    """Zero-gated widely-linear residual FFN for an already-combined state."""

    def __init__(self, modes: int, hidden_modes: int) -> None:
        super().__init__()
        if modes <= 0 or hidden_modes <= 0:
            message = "complex residual FFN dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.input = WidelyLinear(
            modes,
            hidden_modes,
            bias=True,
        )
        self.output = WidelyLinear(
            hidden_modes,
            modes,
            bias=True,
        )
        self.layer_scale = nn.Parameter(torch.zeros(modes))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "complex residual FFN inputs have incompatible shapes"
            raise ValueError(message)
        return self.run_cffn(
            real,
            imag,
            input_projection=self.input,
            output_projection=self.output,
            activation="cartesian_silu",
            residual_scale=self.layer_scale,
        )
