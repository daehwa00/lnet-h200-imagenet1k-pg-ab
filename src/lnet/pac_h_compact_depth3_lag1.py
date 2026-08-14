"""Three-stage H-style ALPHABET using only one-step modal correlations."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_headroom_efficient_models import (
    EDGE_FRAME_VARIANT,
    _apply_raw_mask,
    _combined_edge_mask,
    _degree_normalized_edge_analysis,
    _edge_or_singleton_mask,
)
from .pac_headroom_models import _masked_sequence_mean
from .pac_laplace_native_input import EdgeRepeatedTwoForwardPAC
from .pac_raw_efficiency_candidates import _make_terminal_analysis
from .pac_tight_frame_models import _BlockVariant, _TightFrameBlock
from .pac_types import PACExperimentConfig

_LAGS = (1,)


class HCompactDepth3Lag1ALPHABET(EdgeRepeatedTwoForwardPAC):
    """Two full H-style writers followed by one read-only terminal analyzer."""

    def __init__(
        self,
        raw_input_dim: int,
        model_dim: int,
        modes: int,
        output_dim: int,
    ) -> None:
        config = PACExperimentConfig(
            sample_count=1,
            validation_count=1,
            test_count=0,
            sequence_length=2,
            raw_input_dim=raw_input_dim,
            output_dim=output_dim,
            model_dim=model_dim,
            modes=modes,
        )
        super().__init__(
            config,
            output_dim,
            learned_projection=True,
            objective="classification",
        )
        self.forward_block.moment_lags = _LAGS
        self.backward_block.moment_lags = _LAGS

        self.third_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        nn.init.eye_(self.third_projection.weight)
        self.third_local = nn.Conv1d(
            self.model_dim,
            self.model_dim,
            kernel_size=5,
            dilation=4,
            padding=8,
            groups=self.model_dim,
        )
        self.third_block = _TightFrameBlock(
            self.model_dim,
            self.modes,
            _BlockVariant("forward", EDGE_FRAME_VARIANT),
        )
        self.third_block.moment_lags = _LAGS
        _make_terminal_analysis(self.third_block)

        moment_dim = self.modes * 3
        self.head = nn.Linear(self.model_dim + 3 * moment_dim, output_dim)

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

        second_local = functional.silu(
            self.second_local(
                self.second_projection(first_stream).transpose(1, 2)
            ).transpose(1, 2)
        )
        second_local = self._mask_features(second_local, active_valid)
        second_stream, second_moments = self.backward_block(
            second_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        second_stream = self._mask_features(second_stream, active_valid)

        third_local = functional.silu(
            self.third_local(
                self.third_projection(second_stream).transpose(1, 2)
            ).transpose(1, 2)
        )
        encoded = self._mask_features(third_local, active_valid)
        third_moments = self.third_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_moments_only=True,
        )

        if self.final_norm is None:
            message = "three-stage H readout normalization is missing"
            raise RuntimeError(message)
        pooled = _masked_sequence_mean(self.final_norm(encoded), active_valid)
        return self.head(
            torch.cat(
                (pooled, first_moments, second_moments, third_moments),
                dim=-1,
            )
        )

    def post_optimizer_step(self) -> None:
        super().post_optimizer_step()
        self.third_block.retract_frame()

    def finalize_constraints(self) -> None:
        super().finalize_constraints()
        self.third_block.finalize_frame()


__all__ = ["HCompactDepth3Lag1ALPHABET"]
