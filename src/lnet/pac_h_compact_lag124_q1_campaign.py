"""Sealed 30-task Q1 campaign for H-compact lag-(1,2,4)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from . import pac_two_tap_q1_campaign as _campaign

DEFAULT_ROOT: Final = Path(".omx/results/pac-h-compact-lag124-q1-final-20260721")
DEFAULT_BASELINE_ROOT: Final = _campaign.DEFAULT_BASELINE_ROOT
CANDIDATE: Final = "h_compact_lag124"

# The campaign implementation deliberately exposes these two globals as the
# adapter boundary.  Every job, selection artifact, and final ledger therefore
# records the lag-(1,2,4) candidate while retaining the sealed Q1 protocol.
_campaign.DEFAULT_ROOT = DEFAULT_ROOT
_campaign.CANDIDATE = CANDIDATE

default_lanes = _campaign.default_lanes
enqueue_stage1 = _campaign.enqueue_stage1
select_stage1 = _campaign.select_stage1
select_stage2 = _campaign.select_stage2
enqueue_final = _campaign.enqueue_final
stage1_jobs = _campaign.stage1_jobs
status = _campaign.status

__all__ = [
    "CANDIDATE",
    "DEFAULT_BASELINE_ROOT",
    "DEFAULT_ROOT",
    "default_lanes",
    "enqueue_final",
    "enqueue_stage1",
    "select_stage1",
    "select_stage2",
    "stage1_jobs",
    "status",
]
