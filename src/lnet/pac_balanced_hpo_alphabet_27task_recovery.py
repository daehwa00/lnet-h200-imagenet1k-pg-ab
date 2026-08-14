"""Corrected-path Alphabet-only HPO rerun for all 27 primary tasks."""

# pyright: reportPrivateLocalImportUsage=false

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, median
from typing import Final, cast

from .pac_balanced_hpo_campaign import BalancedHPOJob, campaign_status
from .pac_campaign_utils import write_once
from .pac_balanced_hpo_queue import (
    CONFIRMATION_SEEDS,
    EXTERNAL_DATASETS,
    FINAL_SEEDS,
    SEARCH_SEED,
    TOP_K,
    UCR_DATASETS,
    Suite,
    _alphabet_jobs,  # pyright: ignore[reportPrivateUsage]
)

DEFAULT_ROOT: Final = Path(
    ".omx/results/alphabet-27task-corrected-path-recovery-20260726"
)
MODEL: Final = "alphabet"
SCHEMA_PREFIX = "pac.balanced_hpo_alphabet_27task_recovery"
RELATIONSHIP_TO_PRIMARY: dict[str, object] = {
    "primary_registry_unchanged": True,
    "baseline_results_reused": True,
    "reason": (
        "rerun Alphabet after correcting partial-batch gradient-buffer "
        "lifetime in the external exact-split path"
    ),
    "reporting": "separate corrected-path recovery campaign",
}


def expected_counts() -> dict[str, int]:
    tasks = len(UCR_DATASETS) + len(EXTERNAL_DATASETS)
    return {
        "tasks": tasks,
        "models": 1,
        "stage1": tasks * 18,
        "stage2": tasks * TOP_K * len(CONFIRMATION_SEEDS),
        "final": tasks * len(FINAL_SEEDS),
    }


def stage1_jobs(*, dwconv_dilation: int = 4) -> list[BalancedHPOJob]:
    if dwconv_dilation < 1:
        message = "dwconv_dilation must be positive"
        raise ValueError(message)
    jobs = [
        BalancedHPOJob.from_payload(job.payload())
        for suite, datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS))
        for dataset in datasets
        for job in _alphabet_jobs(cast("Suite", suite), dataset)
    ]
    if dwconv_dilation == 4:
        return jobs
    configured: list[BalancedHPOJob] = []
    for job in jobs:
        candidate_id = f"{job.candidate_id}-dilation{dwconv_dilation}"
        configured.append(
            replace(
                job,
                key=(
                    f"balanced-hpo:stage1:{job.suite}:{job.dataset}:alphabet:"
                    f"{candidate_id}:split{job.split_seed}:seed{job.train_seed}"
                ),
                candidate_id=candidate_id,
                architecture=f"radial-log-r-affine-dwconv-d{dwconv_dilation}",
                architecture_settings=(("dwconv_dilation", dwconv_dilation),),
            )
        )
    return configured


def _write_stage_queue(
    root: Path,
    stage: str,
    jobs: list[BalancedHPOJob],
) -> None:
    expected = expected_counts()[stage]
    if len(jobs) != expected:
        message = f"{stage} has {len(jobs)} recovery jobs; expected {expected}"
        raise RuntimeError(message)
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        message = f"{stage} contains duplicate recovery logical keys"
        raise RuntimeError(message)
    if any(job.model != MODEL for job in jobs):
        message = "the external recovery queue must contain Alphabet only"
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


def enqueue_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    dwconv_dilation: int = 4,
) -> dict[str, object]:
    jobs = stage1_jobs(dwconv_dilation=dwconv_dilation)
    _write_stage_queue(root, "stage1", jobs)
    counts = expected_counts()
    contract: dict[str, object] = {
        "schema": f"{SCHEMA_PREFIX}.v1",
        "state": "prepared",
        "relationship_to_primary": RELATIONSHIP_TO_PRIMARY,
        "datasets": {
            "ucr": list(UCR_DATASETS),
            "external": list(EXTERNAL_DATASETS),
        },
        "models": [MODEL],
        "expected_logical_jobs": counts,
        "selection_policy": {
            "stage1": f"top {TOP_K} per task using validation only",
            "stage2": "mean validation score over seeds 7, 11, and 19",
            "final": "five frozen seeds; official TEST available only here",
        },
        "seeds": {
            "stage1": [SEARCH_SEED],
            "stage2": list(CONFIRMATION_SEEDS),
            "final": list(FINAL_SEEDS),
        },
        "runtime_policy": {
            "external_full_batch": "external exact-split CUDA Graph",
            "external_partial_batch": "eager fallback with persistent gradient buffers",
            "ucr": "existing corrected Alphabet classification path",
            "compile_model_body": False,
            "mode_static_pole_training": False,
        },
        "architecture_control": {
            "dwconv_dilation": dwconv_dilation,
            "all_other_model_and_training_settings": "identical to corrected-path Stage 1",
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
            f"{stage} recovery is incomplete: expected={len(expected_keys)}, "
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
        "schema": f"{SCHEMA_PREFIX}_stage1_selection.v1",
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
        final_epochs = (
            max(1, round(median(best_epochs)))
            if base.suite == "ucr" and best_epochs
            else base.epochs
        )
        selected[cell_key] = {
            "config_key": config_key,
            "mean_validation_score": score,
            "selection_seeds": sorted(expected_seeds),
            "width": base.width,
            "modes": base.modes,
            "architecture": base.architecture,
            "architecture_settings": dict(base.architecture_settings),
            "recipe": asdict(base.recipe),
            "best_epochs": best_epochs,
            "final_epochs": final_epochs,
        }
        for seed in FINAL_SEEDS:
            candidate = replace(
                base,
                key="",
                stage="final",
                split_seed=seed,
                train_seed=seed,
                epochs=final_epochs,
                evaluation_split="test",
                official_test_accessed=True,
            )
            jobs.append(replace(candidate, key=_job_key(candidate)))
    _write_stage_queue(root, "final", jobs)
    payload: dict[str, object] = {
        "schema": f"{SCHEMA_PREFIX}_stage2_selection.v1",
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
            valid_scope = (
                (job.suite == "ucr" and job.dataset in UCR_DATASETS)
                or (job.suite == "external" and job.dataset in EXTERNAL_DATASETS)
            )
            if not valid_scope:
                violations.append(f"out-of-scope job: {job.key}")
            if job.model != MODEL:
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
        "schema": f"{SCHEMA_PREFIX}_audit.v1",
        "status": campaign_status(root),
        "datasets": {
            "ucr": list(UCR_DATASETS),
            "external": list(EXTERNAL_DATASETS),
        },
        "models": [MODEL],
        "logical_keys": len(seen),
        "violations": violations,
        "ok": not violations,
    }


__all__ = [
    "DEFAULT_ROOT",
    "MODEL",
    "audit_extension",
    "enqueue_stage1",
    "expected_counts",
    "select_stage1",
    "select_stage2",
    "stage1_jobs",
]
