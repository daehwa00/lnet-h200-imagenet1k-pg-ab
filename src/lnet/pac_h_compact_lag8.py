"""Compact H-only ALPHABET with one additional lag-eight moment."""

from __future__ import annotations

import torch

from .pac_efp_writer_reader import CompactEFPHOnlyTerminalPAC
from .pac_tight_frame_models import _InvariantMomentHead
from .pac_types import PACExperimentConfig

_LAGS = (1, 4, 8)


class HCompactLag8ALPHABET(CompactEFPHOnlyTerminalPAC):
    """Add lag eight to both H-compact moment banks without changing its core."""

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

        old_head = self.head
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

        # Preserve the canonical H-compact function at initialization.  Only
        # the newly appended lag-eight columns start at zero and open through
        # ordinary gradient descent.
        with torch.no_grad():
            new_head.classifier.weight.zero_()
            new_head.classifier.weight[:, :model_dim].copy_(
                old_head.classifier.weight[:, :model_dim]
            )
            new_head.classifier.weight[
                :, model_dim : model_dim + old_moment_dim
            ].copy_(
                old_head.classifier.weight[
                    :, model_dim : model_dim + old_moment_dim
                ]
            )
            old_second_start = model_dim + old_moment_dim
            new_second_start = model_dim + new_moment_dim
            new_head.classifier.weight[
                :, new_second_start : new_second_start + old_moment_dim
            ].copy_(
                old_head.classifier.weight[
                    :, old_second_start : old_second_start + old_moment_dim
                ]
            )
            new_head.classifier.bias.copy_(old_head.classifier.bias)

        self.forward_block.moment_lags = _LAGS
        self.backward_block.moment_lags = _LAGS
        self.head = new_head


__all__ = ["HCompactLag8ALPHABET"]
