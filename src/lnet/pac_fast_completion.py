"""Reduced-cost three-stage protocol for the 10-model broad campaign."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .pac_broad_benchmark_completion import audit_campaign, stage_jobs
from .pac_broad_benchmark_completion import select_stage1 as _select_stage1
from .pac_broad_benchmark_completion import select_stage2 as _select_stage2

if TYPE_CHECKING:
    from pathlib import Path

STAGE1_CANDIDATES_PER_CELL = 6
STAGE1_EPOCHS = 15
STAGE2_TOP_K = 2
STAGE2_SEEDS = (11,)
STAGE2_EPOCHS = 30
FINAL_SEEDS = (23, 31, 43)
FINAL_EPOCHS = 60


def select_stage1(root: Path) -> dict[str, object]:
    return _select_stage1(
        root,
        candidates_per_cell=STAGE1_CANDIDATES_PER_CELL,
        top_k=STAGE2_TOP_K,
        confirmation_seeds=STAGE2_SEEDS,
        stage2_epochs=STAGE2_EPOCHS,
    )


def select_stage2(root: Path) -> dict[str, object]:
    return _select_stage2(
        root,
        confirmation_seeds=STAGE2_SEEDS,
        final_seeds=FINAL_SEEDS,
        final_epochs=FINAL_EPOCHS,
    )


__all__ = [
    "FINAL_EPOCHS",
    "FINAL_SEEDS",
    "STAGE1_CANDIDATES_PER_CELL",
    "STAGE1_EPOCHS",
    "STAGE2_EPOCHS",
    "STAGE2_SEEDS",
    "STAGE2_TOP_K",
    "audit_campaign",
    "select_stage1",
    "select_stage2",
    "stage_jobs",
]
