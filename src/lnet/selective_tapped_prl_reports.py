from __future__ import annotations

import json
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .selective_tapped_prl_types import SelectiveRun
    from .tapped_prl_followup_schema import JsonRow, JsonValue


def write_selective_artifacts(run: SelectiveRun, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "selective-report.json"
    markdown_path = output_dir / "selective-report.md"
    report_payload = run.to_dict()
    report_payload["conclusion"] = _replacement_conclusion(run)
    json_payload = _json_safe(report_payload)
    json_path.write_text(
        json.dumps(json_payload, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(run), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(run: SelectiveRun) -> str:
    conclusion = _replacement_conclusion(run)
    lines = [
        "# Selective Tapped PRL Follow-Up",
        "",
        f"- mode: `{run.mode}`",
        f"- suite: `{run.suite}`",
        f"- device: `{run.device}`",
        "",
        "## Replacement Conclusion",
        "",
        f"- status: `{conclusion['status']}`",
        f"- rationale: {conclusion['rationale']}",
        f"- recommendation: {conclusion['recommendation']}",
        "",
    ]
    lines.extend(_metadata_lines("Experiment Config", run.experiment_config))
    lines.extend(_metadata_lines("Training Config", run.training_config))
    for section in run.sections.values():
        lines.extend(_section_lines(section))
    return "\n".join(lines).rstrip() + "\n"


def _metadata_lines(title: str, metadata: JsonRow) -> list[str]:
    lines = [f"## {title}", "", "| key | value |", "| --- | --- |"]
    lines.extend(f"| {key} | {_format_cell(value)} |" for key, value in metadata.items())
    lines.append("")
    return lines


def _section_lines(section: JsonRow) -> list[str]:
    verdict = section["verdict"]
    if not isinstance(verdict, dict):
        verdict = {"status": "mixed", "rationale": "missing verdict payload"}
    rows = section["rows"]
    row_values = (
        tuple(row for row in rows if isinstance(row, dict)) if isinstance(rows, list) else ()
    )
    lines = [
        f"## {section['title']}",
        "",
        f"- verdict: `{verdict.get('status')}`",
        f"- hypothesis: {section['hypothesis']}",
        f"- rationale: {verdict.get('rationale')}",
        f"- rows: {len(row_values)}",
        "",
    ]
    lines.extend(_table_lines(row_values))
    lines.append("")
    return lines


def _table_lines(rows: tuple[JsonRow, ...]) -> list[str]:
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
        return f"{value:.6f}"
    return str(value)


def _replacement_conclusion(run: SelectiveRun) -> JsonRow:
    statuses = {name: _section_status(section) for name, section in run.sections.items()}
    selectivity = statuses.get("selectivity_ablation")
    parameter = statuses.get("parameter_matched")
    delay = statuses.get("delay_kernel_sweep")
    if selectivity == "supports" and parameter == "supports":
        status = "supports_branch_removal"
        recommendation = "Selective Tapped PRL is a candidate replacement for the hybrid branches."
    else:
        status = "not_safe_to_remove_fir_mlp"
        recommendation = (
            "Keep Direct FIR/MLP branches in the default hybrid model; treat selective taps "
            "as an ablation and delay-localization mechanism until stronger evidence appears."
        )
    rationale = (
        f"selectivity_ablation={selectivity}; delay_kernel_sweep={delay}; "
        f"parameter_matched={parameter}."
    )
    return {
        "status": status,
        "rationale": rationale,
        "recommendation": recommendation,
    }


def _section_status(section: JsonRow) -> str | None:
    verdict = section.get("verdict")
    if not isinstance(verdict, dict):
        return None
    status = verdict.get("status")
    return status if isinstance(status, str) else None


def _json_safe(value: JsonValue) -> JsonValue:
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value
