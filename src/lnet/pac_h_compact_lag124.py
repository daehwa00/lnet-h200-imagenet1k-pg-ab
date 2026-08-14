"""Compact H-only ALPHABET with lag-one, lag-two, and lag-four moments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .pac_efp_writer_reader import CompactEFPHOnlyTerminalPAC
from .pac_tight_frame_models import (
    _InvariantMomentHead,  # pyright: ignore[reportPrivateUsage]
)
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective

_LAGS = (1, 2, 4)


class HCompactLag124PAC(CompactEFPHOnlyTerminalPAC):
    """Config-compatible lag-(1,2,4) H-compact model for campaign runners."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
    ) -> None:
        super().__init__(config, output_dim, objective=objective)

        old_head = self.head
        model_dim = config.model_dim
        modes = config.modes
        old_moment_dim = modes * 5
        new_moment_dim = modes * 7
        rng_state = torch.random.get_rng_state()
        new_head = _InvariantMomentHead(
            model_dim,
            modes,
            output_dim,
            use_modal_moments=True,
            use_backward_moments=True,
            lags=_LAGS,
        )
        torch.random.set_rng_state(rng_state)

        # Preserve the canonical H-compact function at initialization.  The
        # lag-two columns start at zero, while energy, lag-one, and lag-four
        # retain their original classifier weights.
        with torch.no_grad():
            new_head.classifier.weight.zero_()
            new_head.classifier.weight[:, :model_dim].copy_(
                old_head.classifier.weight[:, :model_dim]
            )
            for bank in range(2):
                old_start = model_dim + bank * old_moment_dim
                new_start = model_dim + bank * new_moment_dim
                new_head.classifier.weight[
                    :, new_start : new_start + 3 * modes
                ].copy_(
                    old_head.classifier.weight[
                        :, old_start : old_start + 3 * modes
                    ]
                )
                new_head.classifier.weight[
                    :, new_start + 5 * modes : new_start + 7 * modes
                ].copy_(
                    old_head.classifier.weight[
                        :, old_start + 3 * modes : old_start + 5 * modes
                    ]
                )
            new_head.classifier.bias.copy_(old_head.classifier.bias)

        self.forward_block.moment_lags = _LAGS
        self.backward_block.moment_lags = _LAGS
        for block in (self.forward_block, self.backward_block):
            block.fused_lag124_moments = True
            block.fused_moments_backward_training = True
            block.fused_recurrence_moments_backward_training = True
            block.fused_excitation_recurrence_training = True
        self.head = new_head


class HCompactLag124ALPHABET(HCompactLag124PAC):
    """Convenience classification constructor used by focused experiments."""

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
        super().__init__(config, output_dim, objective="classification")


__all__ = ["HCompactLag124ALPHABET", "HCompactLag124PAC"]
