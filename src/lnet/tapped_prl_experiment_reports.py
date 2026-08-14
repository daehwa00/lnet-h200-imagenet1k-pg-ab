from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

    from .tapped_prl_experiment_schema import (
        ExperimentRow,
        GateSelectionRecord,
        StageCheckpoint,
    )

SECTION_TITLES: Final[tuple[tuple[str, str], ...]] = (
    ("branch_ablation", "Branch / Component Ablation"),
    ("k_sweep", "Tap Length Sweep"),
    ("m_sweep", "Mode Sweep"),
    ("tap_parameterization", "Tap Parameterization Comparison"),
    ("gate_variant", "Gate Variant Comparison"),
    ("strict_delay", "Strict Delay Family"),
    ("delayed_modal", "Delayed Modal Teacher"),
    ("interpretability", "Pole / Tap Interpretability"),
    ("parameter_matched", "Parameter-Matched Baselines"),
)


def write_report_artifacts(
    output_dir: Path, checkpoints: tuple[StageCheckpoint, ...]
) -> tuple[str, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "tapped-prl-extended-report.md"
    csv_path = output_dir / "tapped-prl-extended-report.csv"
    json_path = output_dir / "tapped-prl-extended-report.json"
    interpretation_path = output_dir / "tapped-prl-extended-interpretation.md"
    markdown_path.write_text(render_markdown_report(checkpoints), encoding="utf-8")
    _write_csv_report(csv_path, checkpoints)
    json_path.write_text(
        json.dumps(_json_report(checkpoints), indent=2, sort_keys=True), encoding="utf-8"
    )
    interpretation_path.write_text(render_interpretation_memo(checkpoints), encoding="utf-8")
    return tuple(str(path) for path in (markdown_path, csv_path, json_path, interpretation_path))


def render_markdown_report(checkpoints: tuple[StageCheckpoint, ...]) -> str:
    sections: list[str] = []
    rows = _all_rows(checkpoints)
    warning_by_row_id = _warning_by_row_id(checkpoints)
    for comparison_type, title in SECTION_TITLES:
        sections.append(
            _render_section(
                title,
                tuple(row for row in rows if row.comparison_type == comparison_type),
                warning_by_row_id,
            )
        )
    sections.append(_render_gate_selection_section(_all_gate_selection_records(checkpoints)))
    return "\n\n".join(sections)


def render_interpretation_memo(checkpoints: tuple[StageCheckpoint, ...]) -> str:
    lines = ["# Tapped PRL Interpretation Memo", ""]
    for checkpoint in checkpoints:
        for check in checkpoint.hypothesis_checks:
            lines.append(f"## {check.hypothesis_id}")
            lines.append(f"- Status: {check.hypothesis_status}")
            lines.append(f"- Evidence rows: {', '.join(check.evidence_row_ids)}")
            lines.append(f"- Comparison groups: {', '.join(check.comparison_groups)}")
            lines.append(f"- Rationale: {check.rationale}")
            lines.append("")
    return "\n".join(lines).strip()


def _render_section(
    title: str,
    rows: tuple[ExperimentRow, ...],
    warning_by_row_id: dict[str, str],
) -> str:
    header = (
        "| Comparison Group | Model | Params | Validation Loss | Elapsed | Gate | "
        "Availability | Metadata | Tap Peak | Tap Mass Near Delay | Tap Peak Error | "
        "Mean Pole Error | Max Pole Error | Row Warning |"
    )
    divider = (
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    table_rows = [
        "| "
        + " | ".join(
            (
                row.comparison_group,
                row.model_label,
                str(row.params),
                f"{row.validation_loss:.6f}",
                f"{row.elapsed:.3f}",
                row.gate_variant,
                row.availability_status,
                row.teacher_metadata.metadata_status,
                _optional_value(row.tap_peak_index),
                _optional_value(row.tap_mass_near_true_delay),
                _optional_value(row.tap_peak_error),
                _optional_value(row.mean_pole_error),
                _optional_value(row.max_pole_error),
                warning_by_row_id.get(row.row_id, ""),
            )
        )
        + " |"
        for row in rows
    ]
    return "\n".join((f"## {title}", header, divider, *table_rows))


def _render_gate_selection_section(records: tuple[GateSelectionRecord, ...]) -> str:
    header = (
        "| Comparison Group | Scope | Selected Gate | Rule | Source Rows | "
        "Metric | Tie Breaks | Override Reason | Checkpoints |"
    )
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    table_rows = [
        "| "
        + " | ".join(
            (
                record.comparison_group,
                record.gate_selection_scope,
                record.selected_gate_variant,
                record.selection_rule_id,
                ", ".join(record.selection_source_row_ids),
                record.selection_metric,
                ", ".join(record.tie_breaks),
                record.override_reason or "",
                ", ".join(record.selection_checkpoint_paths),
            )
        )
        + " |"
        for record in records
    ]
    return "\n".join(("## Gate Selection Records", header, divider, *table_rows))


def _write_csv_report(path: Path, checkpoints: tuple[StageCheckpoint, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "stage",
                "comparison_type",
                "comparison_group",
                "model_label",
                "gate_variant",
                "params",
                "validation_loss",
                "elapsed",
                "availability_status",
                "metadata_status",
                "tap_peak_index",
                "tap_mass_near_true_delay",
                "tap_peak_error",
                "mean_pole_error",
                "max_pole_error",
                "row_warning",
            )
        )
        warning_by_row_id = _warning_by_row_id(checkpoints)
        for row in _all_rows(checkpoints):
            writer.writerow(
                (
                    row.stage,
                    row.comparison_type,
                    row.comparison_group,
                    row.model_label,
                    row.gate_variant,
                    row.params,
                    row.validation_loss,
                    row.elapsed,
                    row.availability_status,
                    row.teacher_metadata.metadata_status,
                    row.tap_peak_index,
                    row.tap_mass_near_true_delay,
                    row.tap_peak_error,
                    row.mean_pole_error,
                    row.max_pole_error,
                    warning_by_row_id.get(row.row_id, ""),
                )
            )


def _json_report(checkpoints: tuple[StageCheckpoint, ...]) -> dict[str, object]:
    return {
        "report_sections": [
            {
                "title": title,
                "comparison_type": comparison_type,
                "row_count": sum(
                    1 for row in _all_rows(checkpoints) if row.comparison_type == comparison_type
                ),
            }
            for comparison_type, title in SECTION_TITLES
        ],
        "gate_selection_record_count": len(_all_gate_selection_records(checkpoints)),
        "gate_selection_by_comparison_group": [
            {
                "comparison_group": record.comparison_group,
                "gate_selection_scope": record.gate_selection_scope,
                "selected_gate_variant": record.selected_gate_variant,
                "selection_rule_id": record.selection_rule_id,
                "selection_checkpoint_paths": record.selection_checkpoint_paths,
                "selection_source_row_ids": record.selection_source_row_ids,
                "selection_metric": record.selection_metric,
                "tie_breaks": record.tie_breaks,
                "override_reason": record.override_reason,
            }
            for record in _all_gate_selection_records(checkpoints)
        ],
        "checkpoints": [checkpoint.stage for checkpoint in checkpoints],
    }


def _all_rows(checkpoints: tuple[StageCheckpoint, ...]) -> tuple[ExperimentRow, ...]:
    return tuple(row for checkpoint in checkpoints for row in checkpoint.rows)


def _all_gate_selection_records(
    checkpoints: tuple[StageCheckpoint, ...],
) -> tuple[GateSelectionRecord, ...]:
    return tuple(
        record
        for checkpoint in checkpoints
        for record in checkpoint.gate_selection_by_comparison_group
    )


def _optional_value(value: object | None) -> str:
    return "" if value is None else str(value)


def _warning_by_row_id(checkpoints: tuple[StageCheckpoint, ...]) -> dict[str, str]:
    warnings: dict[str, str] = {}
    for checkpoint in checkpoints:
        for warning in checkpoint.warnings:
            if warning.row_id is not None:
                warnings[warning.row_id] = f"{warning.warning_severity}: {warning.message}"
    return warnings
