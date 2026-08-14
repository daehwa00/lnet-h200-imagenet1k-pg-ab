from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from pathlib import Path

type StageName = Literal["stage1", "stage2", "stage3"]
type HypothesisStatus = Literal["supports", "mixed", "does_not_support"]
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]

REQUIRED_CHECKPOINT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "created_at",
        "config",
        "rows",
        "hypothesis_checks",
        "artifacts",
        "warnings",
        "gate_selection_by_comparison_group",
    }
)
REQUIRED_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "stage",
        "mode",
        "smoke",
        "seeds",
        "split_id",
        "train_size",
        "validation_size",
        "sequence_length",
        "batch_size",
        "optimizer_family",
        "learning_rate",
        "epoch_budget",
        "early_stopping_rule",
        "device",
        "comparison_groups",
    }
)
REQUIRED_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "row_id",
        "stage",
        "comparison_group",
        "gate_selection_scope",
        "model_label",
        "teacher_label",
        "teacher_metadata",
        "comparison_type",
        "isolated_vs_joint",
        "gate_variant",
        "tap_parameterization",
        "prl_tap_kernel_size",
        "fir_kernel_size",
        "mode_count",
        "params",
        "target_params",
        "relative_param_error",
        "matched_param_candidate",
        "context_horizon",
        "target_context_horizon",
        "validation_loss",
        "elapsed",
        "seed",
        "split_id",
        "optimizer_family",
        "learning_rate",
        "batch_size",
        "epoch_budget",
        "early_stopping_rule",
        "availability_status",
        "unavailable_reason",
        "fairness_exception",
        "fairness_metadata",
        "parameter_match_metadata",
        "tap_peak_index",
        "tap_mass_near_true_delay",
        "tap_peak_error",
        "mean_pole_error",
        "max_pole_error",
    }
)
ALLOWED_METADATA_STATUS: Final[frozenset[str]] = frozenset(
    {"full_ground_truth", "delay_only", "pole_only", "proxy_only"}
)
ALLOWED_AVAILABILITY_STATUS: Final[frozenset[str]] = frozenset(
    {"available", "deferred", "unavailable", "mismatched"}
)
ALLOWED_ISOLATED_VS_JOINT: Final[frozenset[str]] = frozenset(
    {"isolated", "joint", "not_applicable"}
)
ALLOWED_HYPOTHESIS_STATUS: Final[frozenset[str]] = frozenset(
    {"supports", "mixed", "does_not_support"}
)
ALLOWED_COMPARISON_TYPE: Final[frozenset[str]] = frozenset(
    {
        "branch_ablation",
        "k_sweep",
        "m_sweep",
        "tap_parameterization",
        "gate_variant",
        "strict_delay",
        "delayed_modal",
        "interpretability",
        "parameter_matched",
    }
)
ALLOWED_WARNING_SEVERITY: Final[frozenset[str]] = frozenset({"info", "warning", "error"})


class CheckpointSchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StageConfig:
    stage: StageName
    mode: str
    smoke: bool
    seeds: tuple[int, ...]
    split_id: str
    train_size: int
    validation_size: int
    sequence_length: int
    batch_size: int
    optimizer_family: str
    learning_rate: float
    epoch_budget: int
    early_stopping_rule: str
    device: str
    comparison_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TeacherMetadata:
    teacher_kind: str
    true_delay: int | None
    discrete_pole_real: float | None
    discrete_pole_imag: float | None
    continuous_pole_real: float | None
    continuous_pole_imag: float | None
    damping_radius: float | None
    angular_frequency: float | None
    target_horizon: int | None
    metadata_status: str


@dataclass(frozen=True, slots=True)
class FairnessMetadata:
    split_id: str
    seed: int
    optimizer_family: str
    learning_rate: float
    batch_size: int
    epoch_budget: int
    early_stopping_rule: str
    shared_seed_group: str
    shared_split_group: str
    fairness_exception: str | None


@dataclass(frozen=True, slots=True)
class ParameterMatchMetadata:
    target_model_label: str | None
    target_params: int | None
    selected_params: int | None
    relative_param_error: float | None
    candidate_grid_id: str | None
    selected_candidate: str | None
    target_context_horizon: int | None
    selected_context_horizon: int | None
    parameter_tolerance: float | None
    parameter_constraint_satisfied: bool | None
    horizon_constraint_satisfied: bool | None
    mismatch_reason: str | None


@dataclass(frozen=True, slots=True)
class ExperimentRow:
    row_id: str
    stage: StageName
    comparison_group: str
    gate_selection_scope: str
    model_label: str
    teacher_label: str
    teacher_metadata: TeacherMetadata
    comparison_type: str
    isolated_vs_joint: str
    gate_variant: str
    tap_parameterization: str
    prl_tap_kernel_size: int | None
    fir_kernel_size: int | None
    mode_count: int | None
    params: int
    target_params: int | None
    relative_param_error: float | None
    matched_param_candidate: str | None
    context_horizon: int | None
    target_context_horizon: int | None
    validation_loss: float
    elapsed: float
    seed: int
    split_id: str
    optimizer_family: str
    learning_rate: float
    batch_size: int
    epoch_budget: int
    early_stopping_rule: str
    availability_status: str
    unavailable_reason: str | None
    fairness_exception: str | None
    fairness_metadata: FairnessMetadata
    parameter_match_metadata: ParameterMatchMetadata
    tap_peak_index: int | None
    tap_mass_near_true_delay: float | None
    tap_peak_error: int | None
    mean_pole_error: float | None
    max_pole_error: float | None


@dataclass(frozen=True, slots=True)
class HypothesisCheck:
    hypothesis_id: str
    hypothesis_status: HypothesisStatus
    evidence_row_ids: tuple[str, ...]
    rationale: str
    comparison_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WarningEntry:
    warning_id: str
    warning_severity: str
    message: str
    row_id: str | None
    comparison_group: str | None


@dataclass(frozen=True, slots=True)
class GateSelectionRecord:
    comparison_group: str
    gate_selection_scope: str
    selected_gate_variant: str
    selection_rule_id: str
    selection_checkpoint_paths: tuple[str, ...]
    selection_source_row_ids: tuple[str, ...]
    selection_metric: str
    tie_breaks: tuple[str, ...]
    override_reason: str | None


@dataclass(frozen=True, slots=True)
class StageCheckpoint:
    schema_version: str
    stage: StageName
    created_at: str
    config: StageConfig
    rows: tuple[ExperimentRow, ...]
    hypothesis_checks: tuple[HypothesisCheck, ...]
    artifacts: tuple[str, ...]
    warnings: tuple[WarningEntry, ...]
    gate_selection_by_comparison_group: tuple[GateSelectionRecord, ...]


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    row_id: str
    comparison_group: str
    gate_selection_scope: str
    comparison_type: str
    gate_variant: str
    validation_loss: float
    relative_param_error: float | None
    elapsed: float
    availability_status: str
    fairness_exception: str | None


def checkpoint_to_json(checkpoint: StageCheckpoint) -> JsonObject:
    return asdict(checkpoint)


def write_checkpoint(path: Path, checkpoint: StageCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(checkpoint_to_json(checkpoint), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validate_checkpoint_file(path)


def validate_checkpoint_file(path: Path) -> None:
    validate_checkpoint_payload(_load_json_object(path))


def load_selection_candidates(path: Path) -> tuple[StageName, tuple[SelectionCandidate, ...]]:
    payload = _load_json_object(path)
    validate_checkpoint_payload(payload)
    stage = _parse_stage(_require_string(payload["stage"], "stage"), "stage")
    rows = tuple(
        _candidate_from_row(_require_object(item, "rows[]"))
        for item in _require_list(payload["rows"], "rows")
    )
    return stage, rows


def validate_checkpoint_payload(payload: JsonObject) -> None:
    _require_exact_keys(payload, REQUIRED_CHECKPOINT_KEYS, "checkpoint")
    _require_exact_keys(
        _require_object(payload["config"], "config"), REQUIRED_CONFIG_KEYS, "config"
    )
    for index, item in enumerate(_require_list(payload["rows"], "rows")):
        row = _require_object(item, f"rows[{index}]")
        _require_exact_keys(row, REQUIRED_ROW_KEYS, f"rows[{index}]")
        _require_exact_keys(
            _require_object(row["teacher_metadata"], f"rows[{index}].teacher_metadata"),
            frozenset(
                asdict(
                    TeacherMetadata("", None, None, None, None, None, None, None, None, "")
                ).keys()
            ),
            f"rows[{index}].teacher_metadata",
        )
        _require_exact_keys(
            _require_object(row["fairness_metadata"], f"rows[{index}].fairness_metadata"),
            frozenset(asdict(FairnessMetadata("", 0, "", 0.0, 0, 0, "", "", "", None)).keys()),
            f"rows[{index}].fairness_metadata",
        )
        _require_exact_keys(
            _require_object(
                row["parameter_match_metadata"], f"rows[{index}].parameter_match_metadata"
            ),
            frozenset(
                asdict(
                    ParameterMatchMetadata(
                        None, None, None, None, None, None, None, None, None, None, None, None
                    )
                ).keys()
            ),
            f"rows[{index}].parameter_match_metadata",
        )
        _require_enum(
            _require_string(row["comparison_type"], f"rows[{index}].comparison_type"),
            ALLOWED_COMPARISON_TYPE,
            f"rows[{index}].comparison_type",
        )
        _require_enum(
            _require_string(row["isolated_vs_joint"], f"rows[{index}].isolated_vs_joint"),
            ALLOWED_ISOLATED_VS_JOINT,
            f"rows[{index}].isolated_vs_joint",
        )
        _require_enum(
            _require_string(row["availability_status"], f"rows[{index}].availability_status"),
            ALLOWED_AVAILABILITY_STATUS,
            f"rows[{index}].availability_status",
        )
        teacher_metadata = _require_object(
            row["teacher_metadata"],
            f"rows[{index}].teacher_metadata",
        )
        _require_enum(
            _require_string(
                teacher_metadata["metadata_status"],
                f"rows[{index}].teacher_metadata.metadata_status",
            ),
            ALLOWED_METADATA_STATUS,
            f"rows[{index}].teacher_metadata.metadata_status",
        )
    for index, item in enumerate(_require_list(payload["hypothesis_checks"], "hypothesis_checks")):
        check = _require_object(item, f"hypothesis_checks[{index}]")
        _require_exact_keys(
            check,
            frozenset(
                asdict(HypothesisCheck("", "supports", (), "", ())).keys(),
            ),
            f"hypothesis_checks[{index}]",
        )
        _require_enum(
            _require_string(
                check["hypothesis_status"],
                f"hypothesis_checks[{index}].hypothesis_status",
            ),
            ALLOWED_HYPOTHESIS_STATUS,
            f"hypothesis_checks[{index}].hypothesis_status",
        )
    for index, item in enumerate(_require_list(payload["warnings"], "warnings")):
        warning = _require_object(item, f"warnings[{index}]")
        _require_exact_keys(
            warning,
            frozenset(asdict(WarningEntry("", "info", "", None, None)).keys()),
            f"warnings[{index}]",
        )
        _require_enum(
            _require_string(warning["warning_severity"], f"warnings[{index}].warning_severity"),
            ALLOWED_WARNING_SEVERITY,
            f"warnings[{index}].warning_severity",
        )
    for index, item in enumerate(
        _require_list(
            payload["gate_selection_by_comparison_group"],
            "gate_selection_by_comparison_group",
        )
    ):
        record = _require_object(item, f"gate_selection_by_comparison_group[{index}]")
        _require_exact_keys(
            record,
            frozenset(asdict(GateSelectionRecord("", "", "", "", (), (), "", (), None)).keys()),
            f"gate_selection_by_comparison_group[{index}]",
        )
        _require_string(
            record["comparison_group"],
            f"gate_selection_by_comparison_group[{index}].comparison_group",
        )
        _require_string(
            record["gate_selection_scope"],
            f"gate_selection_by_comparison_group[{index}].gate_selection_scope",
        )
        _require_string(
            record["selected_gate_variant"],
            f"gate_selection_by_comparison_group[{index}].selected_gate_variant",
        )
        _require_string(
            record["selection_rule_id"],
            f"gate_selection_by_comparison_group[{index}].selection_rule_id",
        )
        _require_string(
            record["selection_metric"],
            f"gate_selection_by_comparison_group[{index}].selection_metric",
        )
        _optional_string(
            record["override_reason"],
            f"gate_selection_by_comparison_group[{index}].override_reason",
        )
        _require_string_list(
            record["selection_checkpoint_paths"],
            f"gate_selection_by_comparison_group[{index}].selection_checkpoint_paths",
        )
        _require_string_list(
            record["selection_source_row_ids"],
            f"gate_selection_by_comparison_group[{index}].selection_source_row_ids",
        )
        _require_string_list(
            record["tie_breaks"],
            f"gate_selection_by_comparison_group[{index}].tie_breaks",
        )


def _candidate_from_row(row: JsonObject) -> SelectionCandidate:
    return SelectionCandidate(
        row_id=_require_string(row["row_id"], "row_id"),
        comparison_group=_require_string(row["comparison_group"], "comparison_group"),
        gate_selection_scope=_require_string(row["gate_selection_scope"], "gate_selection_scope"),
        comparison_type=_require_string(row["comparison_type"], "comparison_type"),
        gate_variant=_require_string(row["gate_variant"], "gate_variant"),
        validation_loss=_require_float(row["validation_loss"], "validation_loss"),
        relative_param_error=_optional_float(row["relative_param_error"], "relative_param_error"),
        elapsed=_require_float(row["elapsed"], "elapsed"),
        availability_status=_require_string(row["availability_status"], "availability_status"),
        fairness_exception=_optional_string(row["fairness_exception"], "fairness_exception"),
    )


def _load_json_object(path: Path) -> JsonObject:
    match json.loads(path.read_text(encoding="utf-8")):
        case dict() as payload:
            return payload
        case _:
            message = f"{path} did not contain a JSON object"
            raise CheckpointSchemaError(message)


def _parse_stage(value: str, path: str) -> StageName:
    match value:
        case "stage1" | "stage2" | "stage3" as stage:
            return stage
        case _:
            message = f"{path} must be a valid stage name"
            raise CheckpointSchemaError(message)


def _require_exact_keys(payload: JsonObject, expected: frozenset[str], path: str) -> None:
    if frozenset(payload.keys()) != expected:
        message = f"{path} keys did not match the required schema"
        raise CheckpointSchemaError(message)


def _require_enum(value: str, allowed: frozenset[str], path: str) -> None:
    if value not in allowed:
        message = f"{path} must be one of {sorted(allowed)}"
        raise CheckpointSchemaError(message)


def _require_object(value: JsonValue, path: str) -> JsonObject:
    match value:
        case dict() as mapping:
            return mapping
        case _:
            message = f"{path} must be an object"
            raise CheckpointSchemaError(message)


def _require_list(value: JsonValue, path: str) -> list[JsonValue]:
    match value:
        case list() as values:
            return values
        case _:
            message = f"{path} must be a list"
            raise CheckpointSchemaError(message)


def _require_string_list(value: JsonValue, path: str) -> None:
    for index, item in enumerate(_require_list(value, path)):
        _require_string(item, f"{path}[{index}]")


def _require_string(value: JsonValue, path: str) -> str:
    match value:
        case str() as text:
            return text
        case _:
            message = f"{path} must be a string"
            raise CheckpointSchemaError(message)


def _optional_string(value: JsonValue, path: str) -> str | None:
    match value:
        case None:
            return None
        case str() as text:
            return text
        case _:
            message = f"{path} must be null or a string"
            raise CheckpointSchemaError(message)


def _require_float(value: JsonValue, path: str) -> float:
    match value:
        case int() as integer:
            return float(integer)
        case float() as number:
            return number
        case _:
            message = f"{path} must be numeric"
            raise CheckpointSchemaError(message)


def _optional_float(value: JsonValue, path: str) -> float | None:
    match value:
        case None:
            return None
        case int() as integer:
            return float(integer)
        case float() as number:
            return number
        case _:
            message = f"{path} must be null or numeric"
            raise CheckpointSchemaError(message)
