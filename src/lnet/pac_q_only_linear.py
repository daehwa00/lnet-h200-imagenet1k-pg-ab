"""Full writer-reader encoder with only pole energies and one linear head."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .alphabet_backbone import AlphabetBackbone

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACExperimentConfig


class _QOnlyLinearHead(nn.Module):
    def __init__(self, modes: int, output_dim: int) -> None:
        super().__init__()
        self.modes = modes
        self.use_modal_moments = True
        self.use_backward_moments = True
        self.classifier = nn.Linear(2 * modes, output_dim)

    def forward(
        self,
        pooled: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
    ) -> Tensor:
        del pooled
        energies = torch.cat(
            (
                forward_moments[..., : self.modes],
                backward_moments[..., : self.modes],
            ),
            dim=-1,
        )
        return self.classifier(energies)


class QOnlyLinearPAC(AlphabetBackbone):
    """Retain both pole banks but classify from their log-energies alone."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.head = _QOnlyLinearHead(self.modes, output_dim)
        self.final_norm = None

        # Optimized exact-split/readout paths assume the canonical D+14M head.
        self.use_efp16_exact_split_training = False
        self.require_external_exact_split_training = False
        self.use_fused_efp16_inference_readout = False
        self.use_fused_rmsnorm_mean_training = False
        self.use_fused_rmsnorm_mean_backward_training = False
        self.use_d32_rmsnorm_backward_training = False

    def _readout(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        del valid_mask
        return self.head(inputs, forward_moments, backward_moments)


__all__ = ["QOnlyLinearPAC"]
