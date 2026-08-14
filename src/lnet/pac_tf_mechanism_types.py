from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .pac_types import PACDevice

MechanismStage = Literal["sanity", "enqueue", "workers", "report"]
MechanismStatus = Literal["running", "done", "failed"]


@dataclass(frozen=True, slots=True)
class MechanismQueueConfig:
    output_root: Path = Path(".omx/results/pac-tf-mechanism-recovery-20260710")
    seeds: tuple[int, ...] = (7, 11, 19, 23, 31)
    device: PACDevice = "auto"
    workers: int = 8
    total_slots: int = 16


@dataclass(frozen=True, slots=True)
class MechanismJob:
    key: str
    seed: int
    model: str
    task: str
    slots: int = 2


@dataclass(frozen=True, slots=True)
class MechanismEvent:
    key: str
    status: MechanismStatus
    notes: str = ""
