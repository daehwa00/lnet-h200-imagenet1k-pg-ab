"""Lag-(1,2,4) H-compact with a tied writer analysis/synthesis frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .pac_h_compact_lag124 import HCompactLag124PAC
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective


class HCompactLag124TiedPAC(HCompactLag124PAC):
    """Use the writer analysis frame for synthesis as well."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.forward_block.independent_synthesis_frame = None
        self.forward_block.tie_analysis_synthesis = True


class HCompactLag124TiedALPHABET(HCompactLag124TiedPAC):
    """Convenience classification constructor for focused experiments."""

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


__all__ = ["HCompactLag124TiedALPHABET", "HCompactLag124TiedPAC"]
