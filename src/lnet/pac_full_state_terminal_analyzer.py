from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_headroom_efficient_models import (
    _apply_raw_mask,  # pyright: ignore[reportPrivateUsage]
)
from .pac_raw_efficiency_candidates import TerminalAnalysisRawRepeatedPAC

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACExperimentConfig


FullStateTerminalVariant = Literal[
    "zero_state",
    "full_state",
    "detached_state",
    "partial_025",
    "partial_050",
    "partial_075",
]
FullStateInjectionVariant = FullStateTerminalVariant | Literal["full_late_dense"]
FULL_STATE_TERMINAL_VARIANTS: Final[tuple[FullStateTerminalVariant, ...]] = (
    "zero_state",
    "full_state",
    "detached_state",
    "partial_025",
    "partial_050",
    "partial_075",
)
FULL_STATE_INJECTION_VARIANTS: Final[tuple[FullStateInjectionVariant, ...]] = (
    *FULL_STATE_TERMINAL_VARIANTS,
    "full_late_dense",
)

STATE_GRADIENT_SCALES: Final[dict[FullStateTerminalVariant, float]] = {
    "zero_state": 0.0,
    "full_state": 1.0,
    "detached_state": 0.0,
    "partial_025": 0.25,
    "partial_050": 0.5,
    "partial_075": 0.75,
}


class _LateStateExcitationMixer(nn.Module):
    """Add first-writer modal state directly to the terminal excitation."""

    def __init__(self, modes: int) -> None:
        super().__init__()
        self.adapter = nn.Linear(2 * modes, 2 * modes, bias=False)
        nn.init.zeros_(self.adapter.weight)
        self._state_evidence: Tensor | None = None

    def bind(self, state_evidence: Tensor) -> None:
        if self._state_evidence is not None:
            message = "late state evidence is already bound"
            raise RuntimeError(message)
        self._state_evidence = state_evidence

    def clear(self) -> None:
        self._state_evidence = None

    def forward(
        self,
        excitation_real: Tensor,
        excitation_imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self._state_evidence is None:
            message = "late state evidence must be bound before the terminal scan"
            raise RuntimeError(message)
        state_excitation = self.adapter(self._state_evidence)
        state_real, state_imag = state_excitation.chunk(2, dim=-1)
        return excitation_real + state_real, excitation_imag + state_imag


class FullStateTerminalAnalyzerPAC(TerminalAnalysisRawRepeatedPAC):
    """Inject first-block complex states into the retained Terminal analyzer."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        variant: FullStateTerminalVariant,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        if variant not in FULL_STATE_TERMINAL_VARIANTS:
            message = f"unknown full-state Terminal variant: {variant}"
            raise ValueError(message)
        super().__init__(config, output_dim, objective=objective)
        self.full_state_variant = variant
        projection = nn.Linear(self.model_dim + 2 * self.modes, self.model_dim, bias=False)
        with torch.no_grad():
            projection.weight.zero_()
            projection.weight[:, : self.model_dim].copy_(torch.eye(self.model_dim))
        self.second_projection = projection

    def _state_evidence(self, states_real: Tensor, states_imag: Tensor) -> Tensor:
        states = torch.cat((states_real, states_imag), dim=-1)
        match self.full_state_variant:
            case "zero_state":
                return torch.zeros_like(states)
            case "full_state":
                return states
            case "detached_state":
                return states.detach()
            case "partial_025" | "partial_050" | "partial_075":
                scale = STATE_GRADIENT_SCALES[self.full_state_variant]
                detached = states.detach()
                return detached + scale * (states - detached)
            case _:
                message = f"invalid state-evidence policy: {self.full_state_variant}"
                raise RuntimeError(message)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        first_local = self._mask_features(self.stem(stem_inputs), valid_mask)
        first_stream, first_moments, states_real, states_imag = self.forward_block(
            first_local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
            return_modal_states=True,
        )
        evidence = torch.cat(
            (first_stream, self._state_evidence(states_real, states_imag)),
            dim=-1,
        )
        second_projected = self.second_projection(evidence)
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        encoded = self._mask_features(second_local, valid_mask)
        second_moments = self.backward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
            return_moments_only=True,
        )
        return self._readout(encoded, first_moments, second_moments, valid_mask)


class FullLateDenseTerminalAnalyzerPAC(TerminalAnalysisRawRepeatedPAC):
    """Inject first-writer states after the terminal local lift and modal projection."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.full_state_variant: FullStateInjectionVariant = "full_late_dense"
        self.backward_block.excitation_mixer = _LateStateExcitationMixer(self.modes)

    def _late_mixer(self) -> _LateStateExcitationMixer:
        mixer = self.backward_block.excitation_mixer
        if not isinstance(mixer, _LateStateExcitationMixer):
            message = "Full-Late-Dense excitation mixer is unavailable"
            raise TypeError(message)
        return mixer

    @property
    def state_adapter(self) -> nn.Linear:
        return self._late_mixer().adapter

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        first_local = self._mask_features(self.stem(stem_inputs), valid_mask)
        first_stream, first_moments, states_real, states_imag = self.forward_block(
            first_local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
            return_modal_states=True,
        )
        second_projected = self.second_projection(first_stream)
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        encoded = self._mask_features(second_local, valid_mask)
        state_evidence = self._mask_features(
            torch.cat((states_real, states_imag), dim=-1),
            valid_mask,
        )
        mixer = self._late_mixer()
        mixer.bind(state_evidence)
        try:
            second_moments = self.backward_block(
                encoded,
                time_delta=time_delta,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
                return_moments_only=True,
            )
        finally:
            mixer.clear()
        return self._readout(encoded, first_moments, second_moments, valid_mask)


def build_full_state_terminal_analyzer(
    config: PACExperimentConfig,
    output_dim: int,
    variant: FullStateTerminalVariant,
    *,
    objective: HeadroomObjective = "classification",
) -> FullStateTerminalAnalyzerPAC:
    return FullStateTerminalAnalyzerPAC(config, output_dim, variant, objective=objective)


def build_full_state_injection_analyzer(
    config: PACExperimentConfig,
    output_dim: int,
    variant: FullStateInjectionVariant,
    *,
    objective: HeadroomObjective = "classification",
) -> FullStateTerminalAnalyzerPAC | FullLateDenseTerminalAnalyzerPAC:
    if variant == "full_late_dense":
        return FullLateDenseTerminalAnalyzerPAC(config, output_dim, objective=objective)
    return build_full_state_terminal_analyzer(
        config,
        output_dim,
        variant,
        objective=objective,
    )
