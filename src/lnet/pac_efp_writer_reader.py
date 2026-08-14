from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_headroom_efficient_models import (
    EDGE_FRAME_VARIANT,
    EdgeFramePAC,
    StrideOneConvStem,
    _apply_raw_mask,  # pyright: ignore[reportPrivateUsage]
    _combined_edge_mask,  # pyright: ignore[reportPrivateUsage]
    _degree_normalized_edge_analysis,  # pyright: ignore[reportPrivateUsage]
    _edge_or_singleton_delta,  # pyright: ignore[reportPrivateUsage]
    _edge_or_singleton_mask,  # pyright: ignore[reportPrivateUsage]
)
from .pac_laplace_native_input import EdgeRepeatedTwoForwardPAC
from .pac_raw_efficiency_candidates import (
    _make_terminal_analysis,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACExperimentConfig


EFPWriterReaderVariant = Literal["zero_state", "full_state"]
EFP_WRITER_READER_VARIANTS: Final[tuple[EFPWriterReaderVariant, ...]] = (
    "zero_state",
    "full_state",
)
EFPWriterReaderScreenVariant = Literal[
    "efp16",
    "zero_state",
    "full_state",
    "compact_h_only",
]
EFP_WRITER_READER_SCREEN_VARIANTS: Final[tuple[EFPWriterReaderScreenVariant, ...]] = (
    "efp16",
    *EFP_WRITER_READER_VARIANTS,
    "compact_h_only",
)


class EFPWriterReaderPAC(EdgeRepeatedTwoForwardPAC):
    """Use the structured EFP forward block as a full-state terminal writer.

    The second exact-pole block is read-only: its synthesis and residual update
    are removed, while its modal moments and the retained local reader lift are
    used by the task head.  ``zero_state`` and ``full_state`` have identical
    parameters and initialization; only the first writer's complex trajectory
    supplied to the reader differs.
    """

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        variant: EFPWriterReaderVariant,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        if variant not in EFP_WRITER_READER_VARIANTS:
            message = f"unknown EFP writer-reader variant: {variant}"
            raise ValueError(message)
        super().__init__(
            config,
            output_dim,
            learned_projection=True,
            objective=objective,
        )
        self.writer_reader_variant = variant
        _make_terminal_analysis(self.backward_block)

        projection = nn.Linear(
            self.model_dim + 2 * self.modes,
            self.model_dim,
            bias=False,
        )
        with torch.no_grad():
            projection.weight.zero_()
            projection.weight[:, : self.model_dim].copy_(torch.eye(self.model_dim))
        self.second_projection = projection

    def _state_evidence(self, states_real: Tensor, states_imag: Tensor) -> Tensor:
        states = torch.cat((states_real, states_imag), dim=-1)
        if self.writer_reader_variant == "zero_state":
            return torch.zeros_like(states)
        return states

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        level, detail, active_delta = _degree_normalized_edge_analysis(
            stem_inputs,
            time_delta,
        )
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        edge_mask = _combined_edge_mask(active_observation, active_valid)
        edge_features = torch.cat((level, detail), dim=-1)
        if edge_mask is not None:
            edge_features = edge_features * edge_mask.to(
                device=edge_features.device,
                dtype=edge_features.dtype,
            )

        first_local = self._mask_features(self.stem(edge_features), active_valid)
        first_stream, first_moments, states_real, states_imag = self.forward_block(
            first_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
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
        encoded = self._mask_features(second_local, active_valid)
        second_moments = self.backward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_moments_only=True,
        )
        return self._readout(encoded, first_moments, second_moments, active_valid)


class CompactEFPHOnlyTerminalPAC(EdgeRepeatedTwoForwardPAC):
    """Remove the identically inactive complex-state columns from zero-state B."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            learned_projection=True,
            objective=objective,
        )
        _make_terminal_analysis(self.backward_block)
        projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        nn.init.eye_(projection.weight)
        self.second_projection = projection

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        level, detail, active_delta = _degree_normalized_edge_analysis(
            stem_inputs,
            time_delta,
        )
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        edge_mask = _combined_edge_mask(active_observation, active_valid)
        edge_features = torch.cat((level, detail), dim=-1)
        if edge_mask is not None:
            edge_features = edge_features * edge_mask.to(
                device=edge_features.device,
                dtype=edge_features.dtype,
            )

        first_local = self._mask_features(self.stem(edge_features), active_valid)
        first_stream, first_moments = self.forward_block(
            first_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        second_projected = self.second_projection(first_stream)
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        encoded = self._mask_features(second_local, active_valid)
        second_moments = self.backward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_moments_only=True,
        )
        return self._readout(encoded, first_moments, second_moments, active_valid)


class LearnedTwoTapHOnlyTerminalPAC(CompactEFPHOnlyTerminalPAC):
    """Compact H-only writer/reader with a learned adjacent-pair input map.

    One unconstrained width-two convolution replaces the fixed normalized
    level/detail coordinates and their constrained joint projection.  It still
    emits one token per adjacent raw-input pair, preserving the writer and
    reader's edge-grid contract.
    """

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.stem = StrideOneConvStem(config.raw_input_dim, self.model_dim)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        active_delta = _edge_or_singleton_delta(time_delta)
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)

        # The learned stem performs the adjacent-pair reduction itself.  Raw
        # masks are applied before it and edge masks after it, where lengths
        # agree, including the singleton fallback.
        first_local = self._mask_features(self.stem(stem_inputs), active_valid)
        first_stream, first_moments = self.forward_block(
            first_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        second_projected = self.second_projection(first_stream)
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        encoded = self._mask_features(second_local, active_valid)
        second_moments = self.backward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_moments_only=True,
        )
        return self._readout(encoded, first_moments, second_moments, active_valid)

    def post_optimizer_step(self) -> None:
        # EdgeFramePAC additionally retracts its constrained input projection;
        # this learned stem is intentionally unconstrained.
        self.forward_block.retract_frame()
        self.backward_block.retract_frame()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        self.backward_block.finalize_frame()


@torch.no_grad()
def load_compact_h_only_from_zero_state_(
    target: CompactEFPHOnlyTerminalPAC,
    source: EFPWriterReaderPAC,
) -> CompactEFPHOnlyTerminalPAC:
    """Convert a trained zero-state B model without changing its function."""
    if source.writer_reader_variant != "zero_state":
        message = "compact conversion requires a zero-state writer-reader source"
        raise ValueError(message)
    if (target.model_dim, target.modes) != (source.model_dim, source.modes):
        message = "source and compact target dimensions must match"
        raise ValueError(message)

    source_state = source.state_dict()
    converted: dict[str, Tensor] = {}
    for name, target_value in target.state_dict().items():
        source_value = source_state.get(name)
        if source_value is None:
            message = f"zero-state checkpoint is missing {name}"
            raise KeyError(message)
        if name == "second_projection.weight":
            source_value = source_value[:, : target.model_dim]
        if source_value.shape != target_value.shape:
            message = (
                f"checkpoint tensor {name} has shape {tuple(source_value.shape)}; "
                f"expected {tuple(target_value.shape)}"
            )
            raise ValueError(message)
        converted[name] = source_value.detach().clone()
    target.load_state_dict(converted, strict=True)
    return target


def build_efp_writer_reader_screen_model(
    config: PACExperimentConfig,
    output_dim: int,
    variant: EFPWriterReaderScreenVariant,
    *,
    objective: HeadroomObjective = "classification",
) -> EdgeFramePAC | EFPWriterReaderPAC | CompactEFPHOnlyTerminalPAC:
    if variant == "efp16":
        return EdgeFramePAC(
            config,
            output_dim,
            modes=config.modes,
            semi_orthogonal=True,
            objective=objective,
            model_dim=config.model_dim,
            pac_variant=EDGE_FRAME_VARIANT,
            mode_divisor=2,
        )
    if variant == "compact_h_only":
        return CompactEFPHOnlyTerminalPAC(
            config,
            output_dim,
            objective=objective,
        )
    return EFPWriterReaderPAC(
        config,
        output_dim,
        variant,
        objective=objective,
    )
