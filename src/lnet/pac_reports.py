from __future__ import annotations

import csv
import json
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .tapped_prl_followup_schema import JsonRow, JsonValue


def write_pac_artifacts(payload: JsonRow, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pac-hybrid-prl-report.json"
    markdown_path = output_dir / "pac-hybrid-prl-report.md"
    json_path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    _write_csv_tables(payload, output_dir)
    return json_path, markdown_path


def _markdown(payload: JsonRow) -> str:
    conclusion = _as_row(payload["conclusion"])
    sections = payload["sections"]
    lines = [
        "# Pole-Anchored Controlled Hybrid PRL",
        "",
        f"- device: `{payload['device']}`",
        f"- status: `{conclusion['status']}`",
        f"- rationale: {conclusion['rationale']}",
        "",
        "## Sections",
    ]
    if isinstance(sections, dict):
        for name, section in sections.items():
            if isinstance(section, dict):
                rows = section.get("rows", [])
                lines.append(f"- `{name}`: {len(rows) if isinstance(rows, list) else 0} rows")
    return "\n".join(lines) + "\n"


def _write_csv_tables(payload: JsonRow, output_dir: Path) -> None:
    sections = payload["sections"]
    if not isinstance(sections, dict):
        return
    for name, section in sections.items():
        if not isinstance(section, dict):
            continue
        rows = section.get("rows", [])
        if isinstance(rows, list) and rows:
            _write_csv(output_dir / f"{name}.csv", rows)


def _write_csv(path: Path, rows: list[JsonValue]) -> None:
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return
    keys = sorted({key for row in dict_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(dict_rows)


def _safe(value: JsonValue) -> JsonValue:
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    return value


def _as_row(value: JsonValue) -> JsonRow:
    if isinstance(value, dict):
        return value
    message = f"expected JSON row, got {type(value).__name__}"
    raise TypeError(message)
