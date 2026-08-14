from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .advanced_experiments import SequenceRegressionTask
    from .experiment import SyntheticLaplaceTask
    from .hybrid_delay_tasks import TeacherMetadata
    from .tapped_prl_followup_schema import JsonRow, JsonValue

SelectiveMode = Literal["smoke", "full"]
SelectiveSuite = Literal["all", "selectivity", "delay", "parameter"]


@dataclass(frozen=True, slots=True)
class SelectiveExperimentConfig:
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
    seed: int = 7
    device: Literal["auto", "cpu", "cuda"] = "auto"
    tap_entropy_weight: float = 1.0e-4
    gate_entropy_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class LabeledTask:
    label: str
    task: SequenceRegressionTask | SyntheticLaplaceTask
    metadata: TeacherMetadata | None = None


@dataclass(frozen=True, slots=True)
class SelectiveRun:
    mode: SelectiveMode
    suite: SelectiveSuite
    device: str
    sections: dict[str, JsonRow]
    experiment_config: JsonRow
    training_config: JsonRow

    def to_dict(self) -> JsonRow:
        sections: dict[str, JsonValue] = {}
        for name, section in self.sections.items():
            sections[name] = dict(section)
        payload: JsonRow = {
            "schema_version": "selective_tapped_prl.v1",
            "mode": self.mode,
            "suite": self.suite,
            "device": self.device,
            "experiment_config": self.experiment_config,
            "training_config": self.training_config,
            "sections": sections,
        }
        return payload
