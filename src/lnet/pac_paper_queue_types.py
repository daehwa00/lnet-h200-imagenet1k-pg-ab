from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .pac_types import PACDevice

PaperStage = Literal["sanity", "enqueue", "workers", "report"]
PaperPreset = Literal["smoke", "full"]
JobKind = Literal[
    "param_audit",
    "sampling_rate_ood",
    "irregular_time_ood",
    "damping_counterfactual",
    "expanded_ood",
    "role_ablation",
    "low_data_scaling",
    "strong_baselines_synthetic",
    "strong_baselines_real",
    "speed_correctness",
]
JobStatus = Literal["pending", "running", "done", "failed"]


@dataclass(frozen=True, slots=True)
class PaperQueueConfig:
    output_root: Path = Path(".omx/results/pac-hybrid-prl/paper-queue-20260706")
    preset: PaperPreset = "full"
    seeds: tuple[int, ...] = (7, 11, 19, 23, 31)
    device: PACDevice = "auto"
    workers: int = 4
    total_slots: int = 4


@dataclass(frozen=True, slots=True)
class PaperJob:
    key: str
    kind: JobKind
    seed: int
    model: str
    task: str
    slots: int = 1
    ratio: float | None = None
    value: float | None = None
    dataset: str | None = None


@dataclass(frozen=True, slots=True)
class PaperQueueEvent:
    key: str
    kind: str
    status: JobStatus
    notes: str = ""
