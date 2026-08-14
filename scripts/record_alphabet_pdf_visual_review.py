# ruff: noqa: EM101, EM102, T201, TRY003
# pyright: reportExplicitAny=false
"""Record a completed page-by-page review of the final ALPHABET PDFs.

This command does not perform the visual review.  It records an already
completed review only after rebinding the audit to the current PDF hashes and
checking that every mechanically rendered page and contact sheet still exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _validated_pipeline(
    pipeline_path: Path,
    main_pdf: Path,
    supplement_pdf: Path,
) -> dict[str, Any]:
    pipeline = _read_json(pipeline_path)
    if (
        pipeline.get("schema") != "pac_two_tap_q1_paper_pipeline_complete.v1"
        or pipeline.get("status") != "complete"
        or pipeline.get("chosen_internal_model") != "two_tap_h_only"
        or pipeline.get("q1_rows") != 1050
        or pipeline.get("q2_in_scope") is not False
    ):
        raise RuntimeError("Q1 paper pipeline is not ready for visual completion")
    if (
        pipeline.get("main_pdf_sha256") != file_sha256(main_pdf)
        or pipeline.get("supplement_pdf_sha256") != file_sha256(supplement_pdf)
    ):
        raise RuntimeError("Q1 paper pipeline PDF hashes disagree with the visual audit")
    return pipeline


def _validate_document(
    *,
    repo: Path,
    label: str,
    record: dict[str, Any],
    pdf: Path,
    reviewer: str,
    notes: str,
    reviewed_at_utc: str,
) -> None:
    if record.get("mechanical_status") != "PASS":
        raise RuntimeError(f"{label} mechanical audit has not passed")
    if record.get("banned_text_hits") not in (None, []):
        raise RuntimeError(f"{label} contains banned-text audit hits")
    if not pdf.is_file() or record.get("sha256") != file_sha256(pdf):
        raise RuntimeError(f"{label} PDF hash disagrees with the page audit")

    pages = record.get("pages")
    rendered_pages = record.get("rendered_pages")
    if (
        not isinstance(pages, int)
        or pages < 1
        or not isinstance(rendered_pages, list)
        or len(rendered_pages) != pages
        or len(set(rendered_pages)) != pages
    ):
        raise RuntimeError(f"{label} page audit does not cover every page exactly once")
    missing_pages = [
        value
        for value in rendered_pages
        if not isinstance(value, str) or not _resolve(repo, value).is_file()
    ]
    if missing_pages:
        raise RuntimeError(f"{label} rendered page files are missing: {missing_pages}")

    contact_sheets = record.get("contact_sheets")
    if not isinstance(contact_sheets, list) or not contact_sheets:
        raise RuntimeError(f"{label} contact sheets are absent")
    missing_sheets = [
        value
        for value in contact_sheets
        if not isinstance(value, str) or not _resolve(repo, value).is_file()
    ]
    if missing_sheets:
        raise RuntimeError(f"{label} contact sheets are missing: {missing_sheets}")

    record["visual_review"] = "PASS"
    record["visual_review_record"] = {
        "reviewer": reviewer,
        "reviewed_at_utc": reviewed_at_utc,
        "pages_reviewed": pages,
        "pdf_sha256": record["sha256"],
        "notes": notes,
    }


def record_visual_review(
    *,
    repo: Path,
    report_path: Path,
    main_pdf: Path,
    supplement_pdf: Path,
    reviewer: str,
    main_notes: str,
    supplement_notes: str,
    pipeline_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the audit binding and atomically record both review verdicts."""
    repo = repo.resolve()
    report_path = report_path if report_path.is_absolute() else repo / report_path
    main_pdf = main_pdf if main_pdf.is_absolute() else repo / main_pdf
    supplement_pdf = supplement_pdf if supplement_pdf.is_absolute() else repo / supplement_pdf
    if pipeline_path is not None:
        pipeline_path = pipeline_path if pipeline_path.is_absolute() else repo / pipeline_path
    report = _read_json(report_path)
    if report.get("schema") != "alphabet.pdf_page_audit.v1":
        raise RuntimeError("unexpected final PDF audit schema")
    if report.get("status") != "PASS":
        raise RuntimeError("mechanical PDF audit has not passed")
    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")

    pipeline = (
        _validated_pipeline(pipeline_path, main_pdf, supplement_pdf)
        if pipeline_path is not None
        else None
    )

    reviewed_at_utc = datetime.now(UTC).isoformat()
    for label, pdf, notes in (
        ("main", main_pdf, main_notes),
        ("supplement", supplement_pdf, supplement_notes),
    ):
        record = report.get(label)
        if not isinstance(record, dict):
            raise TypeError(f"final PDF audit is missing the {label} record")
        _validate_document(
            repo=repo,
            label=label,
            record=record,
            pdf=pdf,
            reviewer=reviewer.strip(),
            notes=notes.strip(),
            reviewed_at_utc=reviewed_at_utc,
        )

    report["visual_review"] = "PASS"
    report["visual_review_record"] = {
        "reviewer": reviewer.strip(),
        "reviewed_at_utc": reviewed_at_utc,
        "documents": ["main", "supplement"],
    }
    _atomic_json(report_path, report)
    if pipeline is not None and pipeline_path is not None:
        pipeline["visual_audit_pending"] = False
        pipeline["visual_audit_status"] = "PASS"
        pipeline["visual_audit_sha256"] = file_sha256(report_path)
        pipeline["visual_audit_reviewer"] = reviewer.strip()
        pipeline["visual_audit_reviewed_at_utc"] = reviewed_at_utc
        _atomic_json(pipeline_path, pipeline)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/generated/alphabet_final_pdf_audit.json"),
    )
    parser.add_argument(
        "--main",
        type=Path,
        default=Path("paper/build/final-q1-learned-main/submission.pdf"),
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        default=Path("paper/build/final-q1-learned-supplement/submission.pdf"),
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path(
            ".omx/results/pac-two-tap-q1-final-20260720/paper_pipeline_complete.json"
        ),
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--main-notes", default="All rendered main-paper pages reviewed.")
    parser.add_argument("--supplement-notes", default="All rendered supplement pages reviewed.")
    parser.add_argument("--confirm-main-all-pages-reviewed", action="store_true")
    parser.add_argument("--confirm-supplement-all-pages-reviewed", action="store_true")
    args = parser.parse_args()
    if not args.confirm_main_all_pages_reviewed:
        parser.error("--confirm-main-all-pages-reviewed is required")
    if not args.confirm_supplement_all_pages_reviewed:
        parser.error("--confirm-supplement-all-pages-reviewed is required")
    report = record_visual_review(
        repo=args.repo,
        report_path=args.report,
        main_pdf=args.main,
        supplement_pdf=args.supplement,
        reviewer=args.reviewer,
        main_notes=args.main_notes,
        supplement_notes=args.supplement_notes,
        pipeline_path=args.pipeline,
    )
    main_pages = report["main"]["pages"]
    supplement_pages = report["supplement"]["pages"]
    print(f"PASS: reviewed all {main_pages} main and {supplement_pages} supplement pages")


if __name__ == "__main__":
    main()
