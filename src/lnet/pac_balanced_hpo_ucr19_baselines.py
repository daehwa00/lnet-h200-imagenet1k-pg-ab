"""Baseline-only balanced-HPO extension for 19 additional UCR datasets.

The original registry and ALPHABET results remain untouched.  This extension
reuses the frozen baseline architecture/optimizer grid and the same
Stage-1 -> Stage-2 -> final selection protocol, but deliberately schedules no
ALPHABET jobs. MelbournePedestrian is excluded because its official TRAIN
contains missing values and the frozen protocol specifies no imputation rule.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, median
from typing import Final, cast

from .pac_balanced_hpo_campaign import BalancedHPOJob, campaign_status, result_path
from .pac_campaign_utils import write_once
from .pac_balanced_hpo_queue import (
    BASELINES,
    CONFIRMATION_SEEDS,
    FINAL_SEEDS,
    SEARCH_SEED,
    TOP_K,
    UCR_SECONDS,
    JobClass,
    _baseline_jobs,  # pyright: ignore[reportPrivateUsage]
)

DEFAULT_ROOT: Final = Path(".omx/results/baseline-balanced-hpo-ucr19-20260726")
REFERENCE_DATASET: Final = "ArrowHead"

UCR19_DATASETS: Final = (
    "ACSF1",
    "Adiac",
    "BME",
    "CBF",
    "Coffee",
    "Computers",
    "Crop",
    "EOGHorizontalSignal",
    "FaceAll",
    "InsectEPGRegularTrain",
    "InsectWingbeatSound",
    "Meat",
    "PowerCons",
    "Rock",
    "ShapesAll",
    "SmoothSubspace",
    "SwedishLeaf",
    "UWaveGestureLibraryAll",
    "Worms",
)

# Queue-weight priors affect scheduling only, never selection.  They are
# deliberately conservative for long sequences and large training sets.
UCR19_SECONDS: Final = {
    "ACSF1": 120.0,
    "Adiac": 100.0,
    "BME": 12.0,
    "CBF": 10.0,
    "Coffee": 12.0,
    "Computers": 180.0,
    "Crop": 1_500.0,
    "EOGHorizontalSignal": 400.0,
    "FaceAll": 180.0,
    "InsectEPGRegularTrain": 80.0,
    "InsectWingbeatSound": 250.0,
    "Meat": 25.0,
    "PowerCons": 30.0,
    "Rock": 200.0,
    "ShapesAll": 300.0,
    "SmoothSubspace": 20.0,
    "SwedishLeaf": 100.0,
    "UWaveGestureLibraryAll": 900.0,
    "Worms": 180.0,
}


def expected_counts() -> dict[str, int]:
    tasks = len(UCR19_DATASETS)
    models = len(BASELINES)
    return {
        "tasks": tasks,
        "models": models,
        "stage1": tasks * models * 18,
        "stage2": tasks * models * TOP_K * len(CONFIRMATION_SEEDS),
        "final": tasks * models * len(FINAL_SEEDS),
    }


def _job_class(seconds: float) -> str:
    if seconds < 100.0:
        return "short"
    if seconds <= 600.0:
        return "medium"
    return "long"


def _dataset_jobs(dataset: str, model: str) -> list[BalancedHPOJob]:
    """Clone the frozen baseline grid while changing only dataset/runtime metadata."""
    base_seconds = UCR19_SECONDS[dataset]
    reference_seconds = float(UCR_SECONDS[REFERENCE_DATASET])
    jobs: list[BalancedHPOJob] = []
    for template in _baseline_jobs("ucr", REFERENCE_DATASET, model):
        candidate = replace(
            template,
            key=(
                f"balanced-hpo:stage1:ucr:{dataset}:{model}:{template.candidate_id}:"
                f"split{SEARCH_SEED}:seed{SEARCH_SEED}"
            ),
            dataset=dataset,
            job_class=cast("JobClass", _job_class(base_seconds)),
            estimated_seconds=template.estimated_seconds * base_seconds / reference_seconds,
        )
        jobs.append(BalancedHPOJob.from_payload(candidate.payload()))
    return jobs


def stage1_jobs() -> list[BalancedHPOJob]:
    return [
        job
        for dataset in UCR19_DATASETS
        for model in BASELINES
        for job in _dataset_jobs(dataset, model)
    ]


def _write_stage_queue(root: Path, stage: str, jobs: list[BalancedHPOJob]) -> None:
    expected = expected_counts()[stage]
    if len(jobs) != expected:
        message = f"{stage} has {len(jobs)} UCR-19 jobs; expected {expected}"
        raise RuntimeError(message)
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        message = f"{stage} contains duplicate UCR-19 logical keys"
        raise RuntimeError(message)
    if any(job.model == "alphabet" for job in jobs):
        message = "ALPHABET must not appear in the baseline-only extension"
        raise RuntimeError(message)
    ordered = sorted(jobs, key=lambda job: (-job.estimated_seconds, job.key))
    write_once(
        root / stage / "master.jsonl",
        "".join(json.dumps(job.payload(), sort_keys=True) + "\n" for job in ordered),
    )
    for job_class in ("short", "medium", "long"):
        class_jobs = [job for job in ordered if job.job_class == job_class]
        write_once(
            root / stage / "queues" / f"{job_class}.jsonl",
            "".join(json.dumps(job.payload(), sort_keys=True) + "\n" for job in class_jobs),
        )


def enqueue_stage1(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    jobs = stage1_jobs()
    _write_stage_queue(root, "stage1", jobs)
    counts = expected_counts()
    contract: dict[str, object] = {
        "schema": "pac.balanced_hpo_ucr19_baselines.v1",
        "state": "prepared",
        "relationship_to_primary": {
            "primary_registry_unchanged": True,
            "reporting": "separate post-selection UCR extension",
        },
        "datasets": {"ucr": list(UCR19_DATASETS)},
        "excluded_datasets": {
            "MelbournePedestrian": (
                "official TRAIN contains NaN values; frozen protocol has no imputation rule"
            )
        },
        "models": list(BASELINES),
        "excluded_models": ["alphabet"],
        "expected_logical_jobs": counts,
        "selection_policy": {
            "stage1": f"top {TOP_K} per task-model cell using validation only",
            "stage2": "mean validation score over seeds 7, 11, and 19",
            "final": "five frozen seeds; official TEST available only here",
        },
        "seeds": {
            "stage1": [SEARCH_SEED],
            "stage2": list(CONFIRMATION_SEEDS),
            "final": list(FINAL_SEEDS),
        },
        "runtime_estimates": {
            "policy": "scheduling only; never used for model selection",
            "dataset_seconds": UCR19_SECONDS,
        },
    }
    write_once(root / "contract.json", json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def _completed_rows(root: Path, stage: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / stage / "completed").glob("*.json")):
        try:
            row = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("status") == "done":
            rows.append(row)
    return rows


def _expected_jobs(root: Path, stage: str) -> list[BalancedHPOJob]:
    master = root / stage / "master.jsonl"
    if not master.exists():
        return []
    return [
        BalancedHPOJob.from_payload(json.loads(line))
        for line in master.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _require_complete_stage(root: Path, stage: str) -> list[dict[str, object]]:
    expected = _expected_jobs(root, stage)
    expected_keys = {job.key for job in expected}
    rows_by_key = {str(row["job_key"]): row for row in _completed_rows(root, stage)}
    missing = expected_keys - rows_by_key.keys()
    unexpected = rows_by_key.keys() - expected_keys
    if missing or unexpected:
        message = (
            f"{stage} UCR-19 extension is incomplete: expected={len(expected_keys)}, "
            f"complete={len(expected_keys) - len(missing)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
        raise RuntimeError(message)
    return [rows_by_key[job.key] for job in expected]


def _as_int(value: object) -> int:
    return int(cast("str | int | float", value))


def _as_float(value: object) -> float:
    return float(cast("str | int | float", value))


def _rank_row(row: dict[str, object]) -> tuple[float, int, str]:
    score = row.get("selection_score")
    if score is None:
        message = f"selection score is missing for {row.get('job_key')}"
        raise RuntimeError(message)
    return (
        -_as_float(score),
        _as_int(row.get("params_trainable", 0)),
        str(row["config_key"]),
    )


def reuse_completed_stage1_results(source_root: Path, root: Path = DEFAULT_ROOT) -> int:
    """Copy only validated, in-scope Stage-1 results from an earlier campaign."""
    reused = 0
    for job in _expected_jobs(root, "stage1"):
        source = result_path(source_root, job)
        if not source.exists():
            continue
        row = cast("dict[str, object]", json.loads(source.read_text(encoding="utf-8")))
        if (
            row.get("status") != "done"
            or row.get("job_key") != job.key
            or row.get("candidate_valid") is False
            or bool(row.get("official_test_accessed"))
        ):
            message = f"refusing invalid reusable Stage-1 result: {job.key}"
            raise RuntimeError(message)
        write_once(result_path(root, job), source.read_text(encoding="utf-8"))
        reused += 1
    return reused


def _job_key(job: BalancedHPOJob) -> str:
    return (
        f"balanced-hpo:{job.stage}:{job.suite}:{job.dataset}:{job.model}:"
        f"{job.candidate_id}:split{job.split_seed}:seed{job.train_seed}"
    )


def select_stage1(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    rows = _require_complete_stage(root, "stage1")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_key"]), []).append(row)
    selected: dict[str, list[str]] = {}
    jobs: list[BalancedHPOJob] = []
    for cell_key, cell_rows in sorted(grouped.items()):
        if len(cell_rows) != 18:
            message = f"{cell_key} has {len(cell_rows)} rows; expected 18"
            raise RuntimeError(message)
        top = sorted(cell_rows, key=_rank_row)[:TOP_K]
        selected[cell_key] = [str(row["config_key"]) for row in top]
        for row in top:
            base = BalancedHPOJob.from_payload(row)
            for seed in CONFIRMATION_SEEDS:
                candidate = replace(
                    base,
                    key="",
                    stage="stage2",
                    split_seed=seed,
                    train_seed=seed,
                    evaluation_split="validation",
                    official_test_accessed=False,
                )
                jobs.append(replace(candidate, key=_job_key(candidate)))
    _write_stage_queue(root, "stage2", jobs)
    payload: dict[str, object] = {
        "schema": "pac.balanced_hpo_ucr19_baselines_stage1_selection.v1",
        "source_rows": len(rows),
        "cells": len(grouped),
        "top_k": TOP_K,
        "selected": selected,
        "stage2_jobs": len(jobs),
        "official_test_accessed": False,
    }
    write_once(
        root / "stage1" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def select_stage2(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    stage1 = _require_complete_stage(root, "stage1")
    stage2 = _require_complete_stage(root, "stage2")
    stage1_selection = cast(
        "dict[str, object]",
        json.loads((root / "stage1" / "selection.json").read_text(encoding="utf-8")),
    )
    selected_configs = cast("dict[str, list[str]]", stage1_selection["selected"])
    combined: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in (*stage1, *stage2):
        combined.setdefault((str(row["cell_key"]), str(row["config_key"])), []).append(row)

    selected: dict[str, dict[str, object]] = {}
    jobs: list[BalancedHPOJob] = []
    expected_seeds = {SEARCH_SEED, *CONFIRMATION_SEEDS}
    for cell_key, config_keys in sorted(selected_configs.items()):
        candidates: list[tuple[float, str, list[dict[str, object]]]] = []
        for config_key in config_keys:
            config_rows = combined[(cell_key, config_key)]
            seeds = {_as_int(row["train_seed"]) for row in config_rows}
            if len(config_rows) != 3 or seeds != expected_seeds:
                message = (
                    f"{cell_key}/{config_key} has seeds {sorted(seeds)}; "
                    f"expected {sorted(expected_seeds)}"
                )
                raise RuntimeError(message)
            candidates.append(
                (
                    mean(_as_float(row["selection_score"]) for row in config_rows),
                    config_key,
                    config_rows,
                )
            )
        score, config_key, config_rows = min(
            candidates,
            key=lambda item: (-item[0], item[1]),
        )
        base = BalancedHPOJob.from_payload(config_rows[0])
        best_epochs = [
            _as_int(row["best_epoch"]) for row in config_rows if row.get("best_epoch") is not None
        ]
        refit_epochs = max(1, round(median(best_epochs))) if best_epochs else base.epochs
        selected[cell_key] = {
            "config_key": config_key,
            "mean_validation_score": score,
            "selection_seeds": sorted(expected_seeds),
            "width": base.width,
            "modes": base.modes,
            "architecture": base.architecture,
            "architecture_settings": dict(base.architecture_settings),
            "recipe": asdict(base.recipe),
            "final_epochs": refit_epochs,
        }
        for seed in FINAL_SEEDS:
            candidate = replace(
                base,
                key="",
                stage="final",
                split_seed=seed,
                train_seed=seed,
                epochs=refit_epochs,
                evaluation_split="test",
                official_test_accessed=True,
            )
            jobs.append(replace(candidate, key=_job_key(candidate)))
    _write_stage_queue(root, "final", jobs)
    payload: dict[str, object] = {
        "schema": "pac.balanced_hpo_ucr19_baselines_stage2_selection.v1",
        "source_stage1_rows": len(stage1),
        "source_stage2_rows": len(stage2),
        "cells": len(selected),
        "selected": selected,
        "final_jobs": len(jobs),
        "configuration_frozen_before_test": True,
        "official_test_accessed_during_selection": False,
    }
    write_once(
        root / "stage2" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def audit_extension(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    violations: list[str] = []
    seen: set[str] = set()
    for stage in ("stage1", "stage2", "final"):
        for job in _expected_jobs(root, stage):
            if job.key in seen:
                violations.append(f"duplicate logical key across stages: {job.key}")
            seen.add(job.key)
            if job.suite != "ucr" or job.dataset not in UCR19_DATASETS:
                violations.append(f"out-of-scope job: {job.key}")
            if job.model not in BASELINES or job.model == "alphabet":
                violations.append(f"forbidden model: {job.key}")
            should_access_test = stage == "final"
            if job.official_test_accessed != should_access_test:
                violations.append(f"invalid official-test flag: {job.key}")
            if job.evaluation_split != ("test" if should_access_test else "validation"):
                violations.append(f"invalid evaluation split: {job.key}")
        violations.extend(
            f"result test-access violation: {row.get('job_key')}"
            for row in _completed_rows(root, stage)
            if bool(row.get("official_test_accessed")) != (stage == "final")
        )
    return {
        "schema": "pac.balanced_hpo_ucr19_baselines_audit.v1",
        "status": campaign_status(root),
        "datasets": list(UCR19_DATASETS),
        "models": list(BASELINES),
        "logical_keys": len(seen),
        "violations": violations,
        "ok": not violations,
    }


__all__ = [
    "DEFAULT_ROOT",
    "UCR19_DATASETS",
    "UCR19_SECONDS",
    "audit_extension",
    "enqueue_stage1",
    "expected_counts",
    "reuse_completed_stage1_results",
    "select_stage1",
    "select_stage2",
    "stage1_jobs",
]
