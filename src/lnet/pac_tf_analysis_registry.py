from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

REGISTRY_SCHEMA = "pac_tf_confirmatory_analysis_family_registry.v1"
DEFAULT_REGISTRY_PATH = Path(
    ".omx/protocols/pac_tf_confirmatory_analysis_families_20260711.json"
)
DEFAULT_PROTOCOL_PATH = Path(".omx/protocols/pac_tf_confirmatory_20260711.json")


@dataclass(frozen=True, slots=True)
class AnalysisFamily:
    family_id: str
    contrast_generation_rule: str
    inferential_unit: str
    primary_metric: str
    correction: str
    alpha: float
    expected_hypothesis_count: int
    hypotheses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisFamilyRegistry:
    registry_id: str
    protocol_id: str
    protocol_sha256: str
    created_at_utc: str
    families: tuple[AnalysisFamily, ...]

    def family(self, family_id: str) -> AnalysisFamily:
        for family in self.families:
            if family.family_id == family_id:
                return family
        message = f"analysis family is not registered: {family_id}"
        raise KeyError(message)


def load_analysis_family_registry(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    sidecar_path: Path | None = None,
) -> AnalysisFamilyRegistry:
    registry_bytes = registry_path.read_bytes()
    active_sidecar = sidecar_path or registry_path.with_suffix(registry_path.suffix + ".sha256")
    _validate_sidecar(active_sidecar, registry_path, registry_bytes)
    payload = _json_object(registry_bytes, label="analysis-family registry")
    if payload.get("schema_version") != REGISTRY_SCHEMA:
        message = "unsupported PAC-TF analysis-family registry schema"
        raise ValueError(message)

    predeclaration = _object_dict(payload.get("predeclaration"), "predeclaration")
    if predeclaration.get("created_before_selected_evidence_outcomes") is not True:
        message = "analysis-family registry is not marked as pre-outcome"
        raise ValueError(message)
    if predeclaration.get("selected_evidence_outcomes_inspected_during_creation") is not False:
        message = "analysis-family registry lacks the no-outcome-access declaration"
        raise ValueError(message)
    if predeclaration.get("does_not_modify_locked_main_protocol") is not True:
        message = "analysis-family registry must remain separate from the locked main protocol"
        raise ValueError(message)

    protocol_binding = _object_dict(payload.get("main_protocol"), "main_protocol")
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    expected_sha256 = _required_string(protocol_binding, "sha256")
    if protocol_sha256 != expected_sha256:
        message = "analysis-family registry is bound to a different main protocol SHA-256"
        raise ValueError(message)
    protocol = _json_object(protocol_bytes, label="main protocol")
    protocol_id = _required_string(protocol, "protocol_id")
    if protocol_id != _required_string(protocol_binding, "protocol_id"):
        message = "analysis-family registry protocol id does not match the active protocol"
        raise ValueError(message)

    family_payloads = _object_list(payload.get("families"), "families")
    families = tuple(_parse_family(item) for item in family_payloads)
    if not families:
        message = "analysis-family registry must contain at least one family"
        raise ValueError(message)
    family_ids = [family.family_id for family in families]
    if len(set(family_ids)) != len(family_ids):
        message = "analysis-family registry contains duplicate family ids"
        raise ValueError(message)

    return AnalysisFamilyRegistry(
        registry_id=_required_string(payload, "registry_id"),
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha256,
        created_at_utc=_required_string(payload, "created_at_utc"),
        families=families,
    )


def _parse_family(value: object) -> AnalysisFamily:
    payload = _object_dict(value, "analysis family")
    hypotheses = tuple(_string_list(payload.get("hypotheses"), "hypotheses"))
    expected = _required_int(payload, "expected_hypothesis_count")
    if expected < 1 or expected != len(hypotheses):
        message = "analysis-family hypothesis count does not match its explicit inventory"
        raise ValueError(message)
    if len(set(hypotheses)) != len(hypotheses):
        message = "analysis family contains duplicate hypotheses"
        raise ValueError(message)
    correction = _required_string(payload, "correction")
    if correction != "benjamini-hochberg":
        message = "confirmatory analysis families must use Benjamini-Hochberg correction"
        raise ValueError(message)
    alpha = _required_float(payload, "alpha")
    if abs(alpha - 0.05) > 1.0e-12:
        message = "confirmatory analysis-family alpha must be 0.05"
        raise ValueError(message)
    return AnalysisFamily(
        family_id=_required_string(payload, "family_id"),
        contrast_generation_rule=_required_string(payload, "contrast_generation_rule"),
        inferential_unit=_required_string(payload, "inferential_unit"),
        primary_metric=_required_string(payload, "primary_metric"),
        correction=correction,
        alpha=alpha,
        expected_hypothesis_count=expected,
        hypotheses=hypotheses,
    )


def _validate_sidecar(sidecar: Path, registry: Path, content: bytes) -> None:
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != registry.name:
        message = "analysis-family SHA-256 sidecar has an invalid format or filename"
        raise ValueError(message)
    if fields[0] != hashlib.sha256(content).hexdigest():
        message = "analysis-family registry SHA-256 does not match its sidecar"
        raise ValueError(message)


def _json_object(content: bytes, *, label: str) -> dict[str, object]:
    parsed = cast("object", json.loads(content))
    return _object_dict(parsed, label)


def _object_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a JSON object with string keys"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _object_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        message = f"{label} must be a JSON array"
        raise TypeError(message)
    return cast("list[object]", value)


def _string_list(value: object, label: str) -> list[str]:
    values = _object_list(value, label)
    if not all(isinstance(item, str) and item for item in values):
        message = f"{label} must contain only non-empty strings"
        raise TypeError(message)
    return cast("list[str]", values)


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        message = f"analysis-family field must be a non-empty string: {key}"
        raise TypeError(message)
    return value


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"analysis-family field must be an integer: {key}"
        raise TypeError(message)
    return value


def _required_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"analysis-family field must be numeric: {key}"
        raise TypeError(message)
    return float(value)
