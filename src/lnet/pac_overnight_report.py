from __future__ import annotations

from typing import TYPE_CHECKING

from .pac_overnight_io import read_csv

if TYPE_CHECKING:
    from pathlib import Path

CLAIM_UPDATE = (
    "Use corrected efficiency results only. If CNN/TCN beat PAC, report PAC as "
    "competitive with interpretability rather than state of the art."
)


def write_overnight_summary(output_root: Path) -> None:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    real_status = _real_status(output_root)
    param_status = _status_from_report(reports / "overnight_param_audit.md", "param_count_status")
    efficiency_status = _status_from_report(
        reports / "overnight_efficiency_audit.md", "efficiency_status"
    )
    damping_status = _status_from_report(
        reports / "overnight_damping_diagnostics.md", "damping_diagnostics_status"
    )
    overall = _overall(real_status, param_status, efficiency_status, damping_status)
    lines = [
        "# PAC-Hybrid PRL Overnight Validation Report",
        "",
        "## Executive Summary",
        f"- real_baseline_status: {real_status}",
        f"- param_count_status: {param_status}",
        f"- efficiency_status: {efficiency_status}",
        f"- damping_diagnostics_status: {damping_status}",
        f"- overall_overnight_status: {overall}",
        "",
        "## Completed Rows",
        f"- real_baselines: {_row_count(output_root, 'real_baselines_ecg5000_forda.csv')}",
        f"- param_audit: {len(read_csv(output_root / 'results' / 'param_count_audit.csv'))}",
        f"- efficiency_audit: {_row_count(output_root, 'efficiency_audit.csv')}",
        f"- damping_diagnostics: {_row_count(output_root, 'damping_diagnostics.csv')}",
        f"- knockout: {len(read_csv(output_root / 'results' / 'knockout_damping_off.csv'))}",
        "",
        "## Claim Update",
        CLAIM_UPDATE,
    ]
    (reports / "overnight_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_count(output_root: Path, filename: str) -> int:
    return len(read_csv(output_root / "results" / filename))


def _real_status(output_root: Path) -> str:
    rows = read_csv(output_root / "results" / "real_baselines_ecg5000_forda.csv")
    if not rows:
        return "mixed"
    pac_rows = [row for row in rows if row["model"].startswith("pac")]
    if not pac_rows:
        return "does_not_support"
    return "supports"


def _status_from_report(path: Path, key: str) -> str:
    if not path.exists():
        return "mixed"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return "mixed"


def _overall(real: str, params: str, efficiency: str, damping: str) -> str:
    positive = sum(
        status in {"supports", "verified", "verified_naive_slow"}
        for status in (real, params, efficiency, damping)
    )
    return "supports" if positive >= 3 else "mixed"
