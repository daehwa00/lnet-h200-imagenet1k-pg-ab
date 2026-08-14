from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .pac_types import PACDevice

OvernightStage = Literal["sanity", "queue", "report"]
OvernightDevice = PACDevice
OvernightModel = Literal[
    "pac_lite",
    "pac_full",
    "pac_no_damping_control",
    "controlled_tapped_prl_only",
    "fixed_prl",
    "tapped_prl_fixed",
    "gru",
    "lstm",
    "cnn1d",
    "cnn1d_small",
    "tcn",
    "tcn_small",
    "transformer_tiny",
    "fir_classifier",
]
QueueStatus = Literal["pending", "running", "done", "failed"]


@dataclass(frozen=True, slots=True)
class OvernightConfig:
    output_root: Path = Path(".omx/results/pac-hybrid-prl/overnight-20260705")
    seeds: tuple[int, ...] = (7, 11, 19, 23, 31)
    fast_seeds: tuple[int, ...] = (7, 11, 19)
    device: OvernightDevice = "auto"
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4


@dataclass(frozen=True, slots=True)
class QueueEvent:
    stage: str
    item: str
    status: QueueStatus
    notes: str = ""
