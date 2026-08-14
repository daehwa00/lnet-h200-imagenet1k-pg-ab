from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .pac_overnight_io import read_csv

if TYPE_CHECKING:
    from pathlib import Path

REPORTS: tuple[tuple[str, str], ...] = (
    ("sampling_rate_ood.csv", "sampling_irregular_ood.md"),
    ("irregular_time_ood.csv", "sampling_irregular_ood.md"),
    ("damping_counterfactual.csv", "damping_counterfactual.md"),
    ("strong_baselines_synthetic.csv", "baseline_refresh.md"),
    ("strong_baselines_real.csv", "baseline_refresh.md"),
    ("low_data_scaling.csv", "low_data_scaling.md"),
    ("expanded_ood.csv", "sampling_irregular_ood.md"),
    ("role_ablation_knockouts.csv", "role_ablation.md"),
    ("speed_correctness.csv", "speed_correctness.md"),
    ("mamba_ssm_real.csv", "baseline_refresh.md"),
)


def write_paper_queue_reports(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary_lines = ["# PAC-Hybrid PRL Paper Queue Summary", ""]
    report_lines: dict[str, list[str]] = {}
    for csv_name, report_name in REPORTS:
        raw_rows = read_csv(root / "results" / csv_name)
        rows = _valid_rows(raw_rows)
        stale_count = len(raw_rows) - len(rows)
        suffix = f", stale_failed_rows={stale_count}" if stale_count else ""
        summary_lines.append(f"- {csv_name}: rows={len(rows)}{suffix}")
        title = report_name.removesuffix(".md").replace("_", " ").title()
        report_lines.setdefault(report_name, [f"# {title}", ""])
        report_lines[report_name].append(f"- {csv_name}: rows={len(rows)}{suffix}")
    queue_events = (
        (root / "queue_state.jsonl").read_text(encoding="utf-8").splitlines()
        if (root / "queue_state.jsonl").exists()
        else []
    )
    summary_lines.extend(("", f"queue_events: {len(queue_events)}", _overall_status_line(root)))
    (reports / "paper_queue_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    for report_name, lines in report_lines.items():
        (reports / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _valid_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows if row.get("status") != "failed" and row.get("data_ratio") != "failed"
    ]


def _overall_status_line(root: Path) -> str:
    manifest = root / "queue_manifest.jsonl"
    state = root / "queue_state.jsonl"
    if not manifest.exists() or not state.exists():
        return "overall_status: partial_ok"
    latest: dict[str, str] = {}
    for line in state.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("key")
        status = row.get("status")
        if isinstance(key, str) and isinstance(status, str):
            latest[key] = status
    statuses = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        statuses.append(latest.get(str(row["key"]), "pending"))
    if statuses and all(status == "done" for status in statuses):
        return "overall_status: complete"
    if any(status == "failed" for status in statuses):
        return "overall_status: partial_failed"
    return "overall_status: running_or_pending"
