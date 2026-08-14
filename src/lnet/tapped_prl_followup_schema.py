from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

FollowupMode = Literal["smoke", "full"]
FollowupSuite = Literal[
    "all",
    "pure_tap_delay",
    "interpretability",
    "gate",
    "gate_normalization",
    "regularization",
    "phase_delay_ambiguity",
    "parameter_efficiency",
]
SectionName = Literal[
    "pure_tap_delay_horizon",
    "interpretability_seed_sweep",
    "gate_specialization",
    "gate_normalization",
    "regularization",
    "phase_delay_ambiguity",
    "parameter_efficiency_verdicts",
]
VerdictStatus = Literal["supports", "mixed", "does_not_support"]
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonRow = dict[str, JsonValue]

SECTION_ORDER: tuple[SectionName, ...] = (
    "pure_tap_delay_horizon",
    "interpretability_seed_sweep",
    "gate_specialization",
    "gate_normalization",
    "regularization",
    "phase_delay_ambiguity",
    "parameter_efficiency_verdicts",
)


@dataclass(frozen=True, slots=True)
class FollowupVerdict:
    status: VerdictStatus
    hypothesis: str
    rationale: str
    row_count: int

    def to_dict(self) -> JsonRow:
        return {
            "status": self.status,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "row_count": self.row_count,
        }


@dataclass(frozen=True, slots=True)
class FollowupSection:
    name: SectionName
    title: str
    verdict: FollowupVerdict
    rows: tuple[JsonRow, ...]

    def to_dict(self) -> JsonRow:
        return {
            "name": self.name,
            "title": self.title,
            "verdict": self.verdict.to_dict(),
            "rows": [dict(row) for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class FollowupRun:
    mode: FollowupMode
    suite: FollowupSuite
    device: str
    sections: dict[SectionName, FollowupSection]
    experiment_config: JsonRow = field(default_factory=dict)
    training_config: JsonRow = field(default_factory=dict)

    def to_dict(self) -> JsonRow:
        payload: JsonRow = {
            "schema_version": "tapped_prl_followup.v1",
            "mode": self.mode,
            "suite": self.suite,
            "device": self.device,
            "experiment_config": dict(self.experiment_config),
            "training_config": dict(self.training_config),
            "sections": {
                section_name: self.sections[section_name].to_dict()
                for section_name in SECTION_ORDER
                if section_name in self.sections
            },
        }
        for section_name in SECTION_ORDER:
            if section_name in self.sections:
                payload[section_name] = self.sections[section_name].to_dict()
        return payload


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not isfinite(value):
        return None
    return float(value)
