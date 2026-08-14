from __future__ import annotations

import json
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .tapped_prl_followup_schema import JsonRow, JsonValue


def write_controlled_damping_artifacts(payload: JsonRow, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "controlled-damping-report.json"
    markdown_path = output_dir / "controlled-damping-report.md"
    safe_payload = _json_safe(payload)
    json_path.write_text(
        json.dumps(safe_payload, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(payload: JsonRow) -> str:
    conclusion = _as_row(payload["conclusion"])
    lines = [
        "# Controlled-Damping Tapped PRL",
        "",
        f"- device: `{payload['device']}`",
        "",
        "## Conclusion",
        "",
        f"- status: `{conclusion['status']}`",
        f"- rationale: {conclusion['rationale']}",
        f"- decision rule: {conclusion['decision_rule']}",
        "",
    ]
    lines.extend(_metadata_lines(_as_row(payload["experiment_config"])))
    sections = _as_row(payload["sections"])
    for section in sections.values():
        lines.extend(_section_lines(_as_row(section)))
    return "\n".join(lines).rstrip() + "\n"


def _metadata_lines(config: JsonRow) -> list[str]:
    lines = ["## Experiment Config", "", "| key | value |", "| --- | --- |"]
    lines.extend(f"| {key} | {_format_cell(value)} |" for key, value in config.items())
    lines.append("")
    return lines


def _section_lines(section: JsonRow) -> list[str]:
    rows = _as_rows(section["rows"])
    lines = [f"## {section['title']}", "", f"- rows: {len(rows)}", ""]
    lines.extend(_table_lines(rows))
    lines.append("")
    return lines


def _table_lines(rows: list[JsonRow]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format_cell(row.get(key)) for key in keys) + " |" for row in rows
    )
    return lines


def _format_cell(value: JsonValue | None) -> str:
    if isinstance(value, float):
        return f"{value:.6f}" if isfinite(value) else "nan"
    return str(value)


def _json_safe(value: JsonValue) -> JsonValue:
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _as_row(value: JsonValue) -> JsonRow:
    if isinstance(value, dict):
        return value
    message = f"expected JSON object, got {type(value).__name__}"
    raise TypeError(message)


def _as_rows(value: JsonValue) -> list[JsonRow]:
    if not isinstance(value, list):
        message = f"expected JSON list, got {type(value).__name__}"
        raise TypeError(message)
    return [_as_row(item) for item in value]
