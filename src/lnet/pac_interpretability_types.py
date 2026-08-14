from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .pac_types import PACDevice

InterpretabilityStage = Literal["sanity", "enqueue", "workers", "report"]
InterpretabilityPreset = Literal["smoke", "full"]
InterpretabilityPackage = Literal["synthetic_mechanism", "real_modal"]
InterpretabilityStatus = Literal["running", "done", "failed"]


@dataclass(frozen=True, slots=True)
class InterpretabilityQueueConfig:
    output_root: Path = Path(".omx/results/pac-hybrid-prl/interpretability-evidence-20260709")
    preset: InterpretabilityPreset = "full"
    seeds: tuple[int, ...] = (7, 11, 19, 23, 31)
    device: PACDevice = "auto"
    workers: int = 4
    total_slots: int = 8


@dataclass(frozen=True, slots=True)
class InterpretabilityJob:
    key: str
    package: InterpretabilityPackage
    seed: int
    model: str
    task: str
    slots: int = 1


@dataclass(frozen=True, slots=True)
class InterpretabilityEvent:
    key: str
    package: str
    status: InterpretabilityStatus
    notes: str = ""
