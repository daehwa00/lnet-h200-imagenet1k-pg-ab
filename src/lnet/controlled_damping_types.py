from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Tensor

    from .advanced_experiments import SequenceRegressionTask
    from .experiment import SyntheticLaplaceTask

ControlledDampingMode = Literal["smoke", "full"]
ControlledDampingDevice = Literal["auto", "cpu", "cuda"]
ControlledModelKind = Literal["selective", "hybrid", "fir", "gru"]


@dataclass(frozen=True, slots=True)
class ControlledDampingConfig:
    sample_count: int
    validation_count: int
    sequence_length: int
    raw_input_dim: int = 2
    output_dim: int = 2
    model_dim: int = 8
    modes: int = 4
    tap_kernel_size: int = 8
    epochs: int = 30
    learning_rate: float = 4.0e-2
    weight_decay: float = 1.0e-4
    seeds: tuple[int, ...] = (7, 11, 19)
    beta_values: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    device: ControlledDampingDevice = "auto"
    tap_entropy_weight: float = 0.0
    gate_entropy_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class ControlledDampingTask:
    label: str
    task: SequenceRegressionTask | SyntheticLaplaceTask
    train_fast_regime: Tensor | None = None
    validation_fast_regime: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ControlledDampingTaskConfig:
    sample_count: int
    validation_count: int
    sequence_length: int
    raw_input_dim: int
    output_dim: int
    seed: int
    delay_steps: int = 0
    slow_decay: float = 0.9
    fast_decay: float = 0.35


@dataclass(frozen=True, slots=True)
class ControlledTrialSpec:
    kind: ControlledModelKind
    name: str
    variant: str | None = None
    damping_beta: float | None = None
