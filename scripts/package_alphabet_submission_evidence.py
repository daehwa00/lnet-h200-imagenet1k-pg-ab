#!/usr/bin/env python3
"""Create an anonymized, checksummed manifest for final ALPHABET evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

SCHEMA = "alphabet.submission_evidence_manifest.v1"
PENDING_SCHEMA = "alphabet.submission_evidence_manifest.pending.v1"
FINAL_SEEDS = [23, 31, 43, 47, 59]
SELECTION_SEEDS = [7, 11, 19]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root: Path, *, repo: Path) -> dict[str, Any]:
    if not root.exists():
        message = f"required evidence path is missing: {root}"
        raise FileNotFoundError(message)
    paths = (
        [root]
        if root.is_file()
        else sorted(path for path in root.rglob("*") if path.is_file())
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_sha = _sha256(path)
        digest.update(f"{relative}\0{size}\0{file_sha}\n".encode())
        total_bytes += size
    return {
        "path": root.relative_to(repo).as_posix(),
        "files": len(paths),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        message = f"expected a JSON object: {path}"
        raise TypeError(message)
    return value


def _optional_status(path: Path) -> str:
    if not path.is_file():
        return "PENDING"
    value = _read_json(path).get("status")
    return value if isinstance(value, str) else "PENDING"


def _git_metadata(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        git = Path("/usr/bin/git")
        if not git.is_file():
            return None
        result = subprocess.run(  # noqa: S603
            [git, *args],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": commit,
        "worktree_dirty": None if status is None else bool(status),
        "tracked_diff_sha256": (
            None
            if (diff := run("diff", "--binary", "HEAD")) is None
            else hashlib.sha256(diff.encode()).hexdigest()
        ),
    }


def _checkpoint_inventory(campaign: Path) -> dict[str, Any]:
    suffixes = {".ckpt", ".pt", ".pth"}
    paths = sorted(
        path for path in campaign.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )
    return {
        "retained_files": len(paths),
        "policy": (
            "TRAIN-derived validation selects the minimum-validation-loss checkpoint; final jobs "
            "refit the frozen configuration on full TRAIN for the frozen epoch budget and retain "
            "the result record rather than a per-fit model checkpoint."
        ),
    }


def _relative(repo: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo.resolve())
    except ValueError as error:
        message = f"evidence path escapes repository: {path}"
        raise ValueError(message) from error


def _complete_marker(marker: dict[str, Any]) -> bool:
    return bool(
        marker.get("schema") == "pac_alphabet_q1_q2_pipeline_complete.v2"
        and marker.get("verified") is True
        and marker.get("chosen_internal_model") == "compact_h_only"
        and marker.get("q1_final_rows") == 1050
        and marker.get("q2_audited_tasks") == 30
        and marker.get("q2_total_model_budget_cells") == 840
        and isinstance(marker.get("q2_realizable_cells"), int)
        and marker["q2_realizable_cells"] > 0
        and marker.get("q2_final_rows") == 5 * marker["q2_realizable_cells"]
    )


def _validate_final_audit_bindings(
    *,
    marker_path: Path,
    marker: dict[str, Any],
    q1_summary_path: Path,
    q2_summary_path: Path,
    independent_audit: dict[str, Any],
) -> None:
    sources = independent_audit.get("sources_sha256")
    expected_sources = {
        "pipeline_complete": _sha256(marker_path),
        "q1_summary": _sha256(q1_summary_path),
        "q2_summary": _sha256(q2_summary_path),
    }
    if sources != expected_sources:
        message = "independent final audit is not bound to the packaged marker and summaries"
        raise RuntimeError(message)

    q2_summary = _read_json(q2_summary_path)
    audited_q1 = independent_audit.get("q1")
    audited_q2 = independent_audit.get("q2")
    if not isinstance(audited_q1, dict) or not isinstance(audited_q2, dict):
        message = "independent final audit is missing Q1/Q2 ledger results"
        raise TypeError(message)
    if audited_q1.get("rows") != marker.get("q1_final_rows"):
        message = "independent Q1 row count disagrees with the completion marker"
        raise RuntimeError(message)
    if audited_q2.get("rows") != marker.get("q2_final_rows"):
        message = "independent Q2 row count disagrees with the completion marker"
        raise RuntimeError(message)
    if audited_q2.get("cells") != marker.get("q2_realizable_cells"):
        message = "independent Q2 cell count disagrees with the completion marker"
        raise RuntimeError(message)

    execution_identity = q2_summary.get("execution_identity")
    required_identity = {
        "code_sha256",
        "manifest_sha256_count",
        "selection_artifact_sha256",
    }
    if (
        not isinstance(execution_identity, dict)
        or set(execution_identity) != required_identity
        or not isinstance(execution_identity.get("code_sha256"), str)
        or not isinstance(execution_identity.get("selection_artifact_sha256"), str)
        or not isinstance(execution_identity.get("manifest_sha256_count"), int)
        or execution_identity["manifest_sha256_count"] < 1
    ):
        message = "packaged Q2 summary has an invalid execution identity"
        raise RuntimeError(message)
    if audited_q2.get("execution_identity") != execution_identity:
        message = "independent Q2 execution identity disagrees with the packaged summary"
        raise RuntimeError(message)


def _validate_pdf_audit(
    *, repo: Path, pdf_audit: dict[str, Any]
) -> str:
    """Bind the page audit to both final PDFs and require per-document review."""
    if pdf_audit.get("schema") != "alphabet.pdf_page_audit.v1":
        message = "final PDF audit has an invalid schema"
        raise RuntimeError(message)
    if pdf_audit.get("status") != "PASS":
        message = "mechanical PDF audit has not passed"
        raise RuntimeError(message)

    pdf_paths = {
        "main": repo / "paper/build/alphabet-q1q2-final-20260719/submission.pdf",
        "supplement": (
            repo
            / "paper/build/alphabet-q1q2-final-supplement-20260719/submission.pdf"
        ),
    }
    document_reviews: list[str] = []
    for label, pdf_path in pdf_paths.items():
        record = pdf_audit.get(label)
        if not isinstance(record, dict):
            message = f"final PDF audit is missing the {label} record"
            raise TypeError(message)
        if record.get("mechanical_status") != "PASS":
            message = f"{label} mechanical PDF audit has not passed"
            raise RuntimeError(message)
        if not pdf_path.is_file() or record.get("sha256") != _sha256(pdf_path):
            message = f"{label} PDF hash disagrees with the final PDF audit"
            raise RuntimeError(message)
        pages = record.get("pages")
        rendered_pages = record.get("rendered_pages")
        if (
            not isinstance(pages, int)
            or pages < 1
            or not isinstance(rendered_pages, list)
            or len(rendered_pages) != pages
        ):
            message = f"{label} PDF audit does not cover every rendered page"
            raise RuntimeError(message)
        document_reviews.append(str(record.get("visual_review", "PENDING")))

    top_level_review = str(pdf_audit.get("visual_review", "PENDING"))
    return (
        "PASS"
        if top_level_review == "PASS" and all(value == "PASS" for value in document_reviews)
        else "PENDING"
    )


def _validate_external_task_provenance(*, campaign: Path) -> None:
    """Require the cross-host full-task copies to match the audited source tensors."""
    provenance = campaign / "provenance"
    preparation = _read_json(provenance / "external_full_preparation.json")
    transfer = _read_json(provenance / "external_full_transfer_verification.json")
    if (
        preparation.get("schema") != "pac_external_full_preparation.v1"
        or preparation.get("selection_identity_verified") is not True
    ):
        message = "external full-task preparation audit is absent or invalid"
        raise RuntimeError(message)
    rows = preparation.get("datasets")
    if not isinstance(rows, list) or len(rows) != 11:
        message = "external full-task preparation audit must contain 11 datasets"
        raise RuntimeError(message)
    expected_hashes: dict[str, str] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("selection_identity_verified") is not True
            or not isinstance(row.get("dataset"), str)
            or not isinstance(row.get("prepared_sha256"), str)
        ):
            message = "external full-task preparation row is invalid"
            raise RuntimeError(message)
        expected_hashes[row["dataset"]] = row["prepared_sha256"]
    if len(expected_hashes) != 11:
        message = "external full-task preparation audit has duplicate datasets"
        raise RuntimeError(message)

    if transfer.get("schema") != "pac_external_full_transfer_verification.v1":
        message = "external full-task transfer audit has an invalid schema"
        raise RuntimeError(message)
    hosts = transfer.get("hosts")
    if not isinstance(hosts, dict) or set(hosts) != {"local_gpu", "secondary_gpu"}:
        message = "external full-task transfer audit must cover local_gpu and KAU"
        raise RuntimeError(message)
    for host, record in hosts.items():
        if (
            not isinstance(record, dict)
            or record.get("verified") is not True
            or record.get("sha256") != expected_hashes
        ):
            message = f"external full-task hashes are not verified on {host}"
            raise RuntimeError(message)


def _validate_architecture_decision(*, campaign: Path, repo: Path) -> None:
    """Require the TEST-free decision that freezes compact_h_only as ALPHABET."""
    decision = _read_json(campaign / "architecture_decision.json")
    if (
        decision.get("schema") != "pac_alphabet_q1_architecture_decision.v1"
        or decision.get("chosen_internal_model") != "compact_h_only"
        or decision.get("public_model") != "ALPHABET"
        or decision.get("test_evidence_used") is not False
        or decision.get("rule")
        != "30-task validation global Top-1 count, then mean rank"
    ):
        message = "final ALPHABET architecture decision is absent or invalid"
        raise RuntimeError(message)
    source_manifest = decision.get("source_manifest")
    if not isinstance(source_manifest, str):
        message = "architecture decision does not identify its source manifest"
        raise TypeError(message)
    source_path = repo / source_manifest
    if not source_path.is_file():
        message = "architecture decision source manifest is missing"
        raise RuntimeError(message)
    source = _read_json(source_path)
    source_hashes = source.get("source_sha256")
    if (
        source.get("schema") != "pac_efp_compact_30_task_source_manifest.v2"
        or not isinstance(source_hashes, dict)
        or len(source_hashes) != 11
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in source_hashes.values()
        )
    ):
        message = "architecture decision source manifest is malformed"
        raise RuntimeError(message)
    source_body = {key: value for key, value in source.items() if key != "sha256"}
    source_digest = hashlib.sha256(
        json.dumps(source_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        source.get("sha256") != source_digest
        or decision.get("source_manifest_sha256") != source_digest
    ):
        message = "architecture decision source-manifest hash does not match"
        raise RuntimeError(message)
    source_comparison = decision.get("source_comparison")
    if not isinstance(source_comparison, str):
        message = "architecture decision does not identify its source comparison"
        raise TypeError(message)
    comparison_path = repo / source_comparison
    if not comparison_path.is_file():
        message = "architecture decision source comparison is missing"
        raise RuntimeError(message)
    comparison = _read_json(comparison_path)
    if (
        comparison.get("schema")
        != "pac_efp_compact_equal_search_30_task_comparison.v1"
        or comparison.get("official_test_accessed") is not False
        or comparison.get("tasks") != 30
        or comparison.get("source_manifest_sha256") != source_digest
        or comparison.get("provisional_champion") != "compact_h_only"
        or comparison.get("selection_rule") != "global top count, then mean rank"
        or comparison.get("global_summary") != decision.get("global_summary")
    ):
        message = "architecture decision disagrees with its TEST-free source comparison"
        raise RuntimeError(message)


def build_manifest(repo: Path, *, pending_ok: bool) -> dict[str, Any]:
    campaign = repo / ".omx/results/pac-alphabet-q1q2-final-20260719"
    marker_path = campaign / "pipeline_complete.json"
    marker = _read_json(marker_path) if marker_path.exists() else {}
    complete = _complete_marker(marker)
    if not complete and not pending_ok:
        message = "final Q1/Q2 completion marker is absent or invalid"
        raise RuntimeError(message)
    if not complete:
        pending: dict[str, Any] = {
            "schema": PENDING_SCHEMA,
            "status": "PENDING",
            "public_model": "ALPHABET",
            "reason": "final Q2 completion marker is not active",
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        q1_summary_path = repo / "paper/generated/q1_final_summary.json"
        q2_summary_path = repo / "paper/generated/q2_final_summary.json"
        if q1_summary_path.is_file() and q2_summary_path.is_file():
            q1_summary = _read_json(q1_summary_path)
            q2_summary = _read_json(q2_summary_path)
            q1_protocol = q1_summary.get("protocol")
            q2_protocol = q2_summary.get("protocol")
            if isinstance(q1_protocol, dict) and isinstance(q2_protocol, dict):
                pending.update(
                    {
                        "campaign_root": campaign.relative_to(repo).as_posix(),
                        "selection_seeds": SELECTION_SEEDS,
                        "final_seeds": FINAL_SEEDS,
                        "progress": {
                            "q1_final_rows": q1_protocol.get("rows"),
                            "q2_completed_rows": q2_protocol.get("final_rows"),
                            "q2_expected_rows": q2_protocol.get("expected_final_rows"),
                            "q2_completed_five_seed_cells": q2_protocol.get(
                                "completed_cells"
                            ),
                            "q2_selected_cells": q2_protocol.get("realizable_cells"),
                            "q2_incomplete_cells": q2_protocol.get(
                                "incomplete_realizable_cells"
                            ),
                            "partial_scores_are_final": False,
                        },
                        "artifact_map": {
                            "q1_results": (
                                ".omx/results/pac-alphabet-q1q2-final-20260719/"
                                "final/completed"
                            ),
                            "q2_calibration_contract": (
                                ".omx/results/pac-alphabet-q1q2-final-20260719/"
                                "q2_calibration/contract.json"
                            ),
                            "q2_selection": (
                                ".omx/results/pac-alphabet-q1q2-final-20260719/"
                                "q2_calibration/selection.json"
                            ),
                            "q2_results": (
                                ".omx/results/pac-alphabet-q1q2-final-20260719/"
                                "q2_final/completed"
                            ),
                            "q2_attempts": (
                                ".omx/results/pac-alphabet-q1q2-final-20260719/"
                                "q2_final/attempts"
                            ),
                            "paper_q1_summary": "paper/generated/q1_final_summary.json",
                            "paper_q2_summary": "paper/generated/q2_final_summary.json",
                        },
                        "verification": {
                            "commands": [
                                "PYTHONPATH=src python -m "
                                "lnet.pac_alphabet_q1_q2_final_cli --stage status "
                                "--output-root .omx/results/"
                                "pac-alphabet-q1q2-final-20260719",
                                "PYTHONPATH=src python "
                                "scripts/audit_pac_alphabet_final_bundle.py --q1-only",
                                "PYTHONPATH=src python "
                                "scripts/audit_pac_alphabet_final_bundle.py --partial-q2",
                                "node paper/verify_evidence.mjs",
                            ],
                            "q1_independent_audit": _optional_status(
                                campaign / "audit/q1_independent_pre_activation.json"
                            ),
                            "q2_partial_integrity_audit": _optional_status(
                                campaign / "audit/q2_partial_progress.json"
                            ),
                            "secondary_evidence_audit": _optional_status(
                                repo
                                / "paper/generated/"
                                "alphabet_secondary_evidence_independent_audit.json"
                            ),
                        },
                    }
                )
        return pending

    independent_audit = _read_json(repo / "paper/generated/alphabet_final_independent_audit.json")
    secondary_audit = _read_json(
        repo / "paper/generated/alphabet_secondary_evidence_independent_audit.json"
    )
    pdf_audit = _read_json(repo / "paper/generated/alphabet_final_pdf_audit.json")
    partition_audit = _read_json(
        campaign / "provenance/q2_final_execution_partition_audit.json"
    )
    if independent_audit.get("status") != "PASS":
        message = "independent final ledger audit has not passed"
        raise RuntimeError(message)
    if secondary_audit.get("status") != "PASS":
        message = "independent secondary-evidence audit has not passed"
        raise RuntimeError(message)
    visual_review = _validate_pdf_audit(repo=repo, pdf_audit=pdf_audit)
    if (
        partition_audit.get("status") != "PASS"
        or partition_audit.get("partition_union_matches_sealed") is not True
        or partition_audit.get("sealed_jobs") != marker.get("q2_final_rows")
    ):
        message = "Q2 execution-partition audit has not passed"
        raise RuntimeError(message)
    _validate_architecture_decision(campaign=campaign, repo=repo)
    _validate_external_task_provenance(campaign=campaign)

    q1_summary_path = repo / "paper/generated/q1_final_summary.json"
    q2_summary_path = repo / "paper/generated/q2_final_summary.json"
    _validate_final_audit_bindings(
        marker_path=marker_path,
        marker=marker,
        q1_summary_path=q1_summary_path,
        q2_summary_path=q2_summary_path,
        independent_audit=independent_audit,
    )

    paper_files = [
        "paper/project.json",
        "paper/main.tex",
        "paper/supplement.tex",
        "paper/references.bib",
        "paper/docs",
        "paper/sections",
        "paper/tables",
        "paper/Figures/alphabet_architecture_exact_pole_v2.pdf",
        "paper/Figures/alphabet_architecture_exact_pole_v2.png",
        "paper/Figures/pac_breadth.pdf",
        "paper/Figures/pac_breadth.png",
        "paper/generated/q1_final_summary.json",
        "paper/generated/q2_final_summary.json",
        "paper/generated/alphabet_final_independent_audit.json",
        "paper/generated/alphabet_secondary_evidence_independent_audit.json",
        "paper/generated/alphabet_final_pdf_audit.json",
        "paper/build/alphabet-q1q2-final-20260719/submission.tex",
        "paper/build/alphabet-q1q2-final-20260719/submission.pdf",
        "paper/build/alphabet-q1q2-final-supplement-20260719/submission.tex",
        "paper/build/alphabet-q1q2-final-supplement-20260719/submission.pdf",
    ]
    model_files = [
        "release/alphabet/__init__.py",
        "release/alphabet/model.py",
        "release/alphabet/README.md",
    ]
    evidence_paths = [
        ".omx/results/pac-alphabet-q1q2-final-20260719/architecture_decision.json",
        ".omx/results/pac-efp-compact-external-equal-search-20260719/reports/"
        "source_manifest.json",
        ".omx/results/pac-efp-compact-external-equal-search-20260719/reports/"
        "combined_30_task_comparison.json",
        ".omx/results/pac-alphabet-q1q2-final-20260719/final/completed",
        ".omx/results/pac-alphabet-q1q2-final-20260719/q2_calibration/contract.json",
        ".omx/results/pac-alphabet-q1q2-final-20260719/q2_calibration/selection.json",
        ".omx/results/pac-alphabet-q1q2-final-20260719/q2_calibration/completed",
        ".omx/results/pac-alphabet-q1q2-final-20260719/q2_final/completed",
        ".omx/results/pac-alphabet-q1q2-final-20260719/q2_final/attempts",
        ".omx/results/pac-alphabet-q1q2-final-20260719/provenance/"
        "q2_final_execution_manifests",
        ".omx/results/pac-alphabet-q1q2-final-20260719/provenance/"
        "q2_final_execution_partition_audit.json",
        ".omx/results/pac-alphabet-q1q2-final-20260719/provenance/"
        "q2_final_recovery_manifests",
        ".omx/results/pac-alphabet-q1q2-final-20260719/provenance/"
        "q2_final_cross_host_recovery.json",
        ".omx/results/pac-alphabet-q1q2-final-20260719/provenance/"
        "external_full_preparation.json",
        ".omx/results/pac-alphabet-q1q2-final-20260719/provenance/"
        "external_full_transfer_verification.json",
        ".omx/results/pac-compact-h-only-ablation-20260719",
        ".omx/results/pac-compact-h-only-boundary-20260719",
        ".omx/results/pac-compact-h-only-synthetic-ood-20260719",
        ".omx/results/pac-compact-h-only-variable-step-20260719",
        ".omx/results/pac-compact-h-only-physionet2012-20260719",
        ".omx/results/pac-compact-h-only-systems-20260719",
    ]
    status = "PASS" if visual_review == "PASS" else "AWAITING_VISUAL_REVIEW"
    return {
        "schema": SCHEMA,
        "status": status,
        "public_model": "ALPHABET",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pipeline_complete": marker,
        "selection_seeds": SELECTION_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "checkpoint_inventory": _checkpoint_inventory(campaign),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "git": _git_metadata(repo),
        "public_model_files": [
            _tree_digest(repo / _relative(repo, repo / relative), repo=repo)
            for relative in model_files
        ],
        "paper_files": [
            _tree_digest(repo / _relative(repo, repo / relative), repo=repo)
            for relative in paper_files
        ],
        "evidence_roots": [
            _tree_digest(repo / _relative(repo, repo / relative), repo=repo)
            for relative in evidence_paths
        ],
        "verification": {
            "independent_ledger_audit": independent_audit.get("status"),
            "independent_secondary_evidence_audit": secondary_audit.get("status"),
            "execution_partition_audit": partition_audit.get("status"),
            "mechanical_pdf_audit": pdf_audit.get("status"),
            "visual_pdf_review": visual_review,
            "commands": [
                "PYTHONPATH=src python paper/generate_pac_final_figures.py",
                "PYTHONPATH=src python scripts/audit_pac_alphabet_final_bundle.py",
                "PYTHONPATH=src python scripts/audit_pac_q2_execution_partition.py",
                "PYTHONPATH=src python scripts/audit_alphabet_secondary_evidence.py",
                "uv run pytest -q tests/test_alphabet_public_release.py "
                "tests/test_paper_submission.py tests/test_paper_active_final_campaign.py "
                "tests/test_audit_pac_alphabet_final_bundle.py "
                "tests/test_audit_pac_q2_execution_partition.py "
                "tests/test_audit_alphabet_secondary_evidence.py "
                "tests/test_package_alphabet_submission_evidence.py",
                "node paper/verify_evidence.mjs",
                "python paper/build_submission.py --source paper/main.tex "
                "--output-dir paper/build/alphabet-q1q2-final-20260719 --strict-overfull",
                "python paper/build_submission.py --source paper/supplement.tex "
                "--output-dir paper/build/alphabet-q1q2-final-supplement-20260719 "
                "--strict-overfull",
                "python scripts/render_alphabet_pdf_audit.py "
                "--main paper/build/alphabet-q1q2-final-20260719/submission.pdf "
                "--supplement "
                "paper/build/alphabet-q1q2-final-supplement-20260719/submission.pdf "
                "--output-dir paper/build/alphabet-q1q2-final-page-audit-20260719 "
                "--report paper/generated/alphabet_final_pdf_audit.json",
                "python scripts/record_alphabet_pdf_visual_review.py "
                "--reviewer <reviewer> --confirm-main-all-pages-reviewed "
                "--confirm-supplement-all-pages-reviewed",
                "python scripts/package_alphabet_submission_evidence.py",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/generated/alphabet_submission_evidence_manifest.json"),
    )
    parser.add_argument("--pending-ok", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    manifest = build_manifest(repo, pending_ok=args.pending_ok)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{manifest['status']}: {output.relative_to(repo)}")  # noqa: T201


if __name__ == "__main__":
    main()
