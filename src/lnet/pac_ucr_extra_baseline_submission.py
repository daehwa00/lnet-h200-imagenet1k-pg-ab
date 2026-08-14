from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .pac_retrained_ablation_campaign import DATASETS, SEEDS

DEFAULT_ROOT: Final = Path(".omx/results/pac-ucr-extra-baselines-submission-20260713")
HISTORICAL_OFFICIAL_ROOT: Final = Path(
    ".omx/results/pac-revised-ucr-official-test-20260712"
)
INCEPTION_DEVELOPMENT: Final = Path(
    ".omx/results/pac-tf-revised-development-baselines-20260712/results/"
    "low_data_recommended_real.csv"
)
INCEPTION_UNSEEN: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/results/"
    "low_data_recommended_real.csv"
)
MINIROCKET_DEVELOPMENT: Final = Path(
    ".omx/results/pac-ucr-s4-minirocket-20260712/results/ucr_s4_minirocket.csv"
)
MINIROCKET_UNSEEN: Final = Path(
    ".omx/results/pac-ucr-s4-minirocket-untouched-validation-20260712/results/"
    "ucr_s4_minirocket.csv"
)
MINIROCKET_SELECTION: Final = Path(
    ".omx/results/pac-ucr-s4-minirocket-20260712/reports/selection.json"
)


@dataclass(frozen=True, slots=True)
class SubmissionSources:
    official_root: Path = HISTORICAL_OFFICIAL_ROOT
    inception_development: Path = INCEPTION_DEVELOPMENT
    inception_unseen: Path = INCEPTION_UNSEEN
    minirocket_development: Path = MINIROCKET_DEVELOPMENT
    minirocket_unseen: Path = MINIROCKET_UNSEEN
    minirocket_selection: Path = MINIROCKET_SELECTION


DEFAULT_SOURCES: Final = SubmissionSources()


def prepare_submission_baselines(
    root: Path = DEFAULT_ROOT, *, sources: SubmissionSources = DEFAULT_SOURCES
) -> dict[str, object]:
    expected = {(dataset, str(seed)) for dataset in DATASETS for seed in SEEDS}
    inception_validation = _inception_validation_cells(sources)
    minirocket_validation = _minirocket_validation_cells(sources)
    minirocket_test = _minirocket_test_cells(sources)
    expected_inception_jobs = _official_inception_jobs(sources.official_root)
    if set(expected_inception_jobs) != {
        f"revised_ucr_test:inception_time:{dataset}:seed{seed}"
        for dataset, seed in expected
    }:
        message = "historical InceptionTime manifest does not cover the locked 18x5 grid"
        raise ValueError(message)
    historical_done = _historical_inception_done(sources.official_root)
    missing_keys = sorted(set(expected_inception_jobs) - historical_done)
    missing_jobs = [expected_inception_jobs[key] for key in missing_keys]

    _require_complete("InceptionTime validation", inception_validation, expected)
    _require_complete("MiniROCKET validation", minirocket_validation, expected)
    _require_complete("MiniROCKET official TEST", minirocket_test, expected)

    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "inception-official-test-missing" / "queue_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in missing_jobs),
        encoding="utf-8",
    )
    coverage = submission_baseline_status(root, sources=sources)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "COVERAGE.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "schema": "pac_ucr_extra_baselines_submission.v1",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "grid_cells_per_family": len(expected),
        "evaluation_protocol": {
            "validation": "same TRAIN-derived validation partitions",
            "official_test": "full official TRAIN refit, one official TEST evaluation",
        },
        "coverage_at_enqueue": coverage,
        "queue": {
            "family": "inception_time",
            "only_missing_cells": True,
            "jobs": len(missing_jobs),
            "manifest": str(manifest.resolve()),
            "restart_safe": True,
        },
        "source_artifacts": {
            field: str(getattr(sources, field).resolve())
            for field in sources.__dataclass_fields__
        },
        "minirocket_requeued": False,
        "reason": "MiniROCKET validation and official TEST are already complete (90/90)",
    }
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return coverage


def submission_baseline_status(
    root: Path = DEFAULT_ROOT, *, sources: SubmissionSources = DEFAULT_SOURCES
) -> dict[str, object]:
    expected = {(dataset, str(seed)) for dataset in DATASETS for seed in SEEDS}
    historical_done = _historical_inception_done(sources.official_root)
    supplement = root / "inception-official-test-missing" / "results" / (
        "low_data_recommended_real.csv"
    )
    supplemental_done = {
        row["job_key"]
        for row in _csv_rows(supplement)
        if row.get("status") == "done" and row.get("baseline_family") == "inception_time"
    }
    expected_keys = set(_official_inception_jobs(sources.official_root))
    combined = expected_keys & (historical_done | supplemental_done)
    return {
        "inception_time": {
            "validation": len(_inception_validation_cells(sources) & expected),
            "official_test_historical": len(expected_keys & historical_done),
            "official_test_supplemental": len(expected_keys & supplemental_done),
            "official_test_combined": len(combined),
            "official_test_remaining": len(expected_keys - combined),
        },
        "minirocket": {
            "validation": len(_minirocket_validation_cells(sources) & expected),
            "official_test": len(_minirocket_test_cells(sources) & expected),
        },
        "expected_per_family_split": len(expected),
    }


def _official_inception_jobs(root: Path) -> dict[str, dict[str, object]]:
    jobs: dict[str, dict[str, object]] = {}
    for manifest in sorted(root.glob("shards/*/queue_manifest.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("baseline_family") == "inception_time":
                jobs[str(row["key"])] = row
    return jobs


def _historical_inception_done(root: Path) -> set[str]:
    return {
        row["job_key"]
        for path in root.glob("shards/*/results/low_data_recommended_real.csv")
        for row in _csv_rows(path)
        if row.get("status") == "done" and row.get("baseline_family") == "inception_time"
    }


def _inception_validation_cells(sources: SubmissionSources) -> set[tuple[str, str]]:
    cells: set[tuple[str, str]] = set()
    for path in (sources.inception_development, sources.inception_unseen):
        for row in _csv_rows(path):
            if (
                row.get("status") == "done"
                and row.get("baseline_family") == "inception_time"
                and row.get("validation_trial") == "6"
                and row.get("evaluation_split") == "validation"
            ):
                cells.add((row["dataset_or_task"], row["seed"]))
    return cells


def _minirocket_validation_cells(sources: SubmissionSources) -> set[tuple[str, str]]:
    selection = json.loads(sources.minirocket_selection.read_text(encoding="utf-8"))
    trial = str(selection["minirocket"]["trial"])
    return {
        (row["dataset"], row["seed"])
        for path in (sources.minirocket_development, sources.minirocket_unseen)
        for row in _csv_rows(path)
        if row.get("status") == "done"
        and row.get("family") == "minirocket"
        and row.get("stage") == "validation"
        and row.get("trial") == trial
    }


def _minirocket_test_cells(sources: SubmissionSources) -> set[tuple[str, str]]:
    return {
        (row["dataset"], row["seed"])
        for row in _csv_rows(sources.minirocket_development)
        if row.get("status") == "done"
        and row.get("family") == "minirocket"
        and row.get("stage") == "test"
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_complete(
    label: str, actual: set[tuple[str, str]], expected: set[tuple[str, str]]
) -> None:
    if actual != expected:
        message = f"{label} coverage is {len(actual & expected)}/{len(expected)}"
        raise ValueError(message)
