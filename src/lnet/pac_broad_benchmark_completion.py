"""Selection and completion audit for the runnable broad-benchmark subset."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from statistics import mean
from typing import TYPE_CHECKING, Literal, cast

from .pac_campaign_utils import write_once
from .pac_broad_benchmark_queue import (
    CONFIRMATION_SEEDS,
    DEFAULT_ROOT,
    FINAL_SEEDS,
    SEARCH_SEED,
    TOP_K,
    BenchmarkJob,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

SelectionStage = Literal["stage1", "stage2"]
CampaignStage = Literal["stage1", "stage2", "final"]


def _read_jobs(paths: Iterable[Path]) -> tuple[BenchmarkJob, ...]:
    jobs = tuple(
        BenchmarkJob.from_payload(cast("dict[str, object]", json.loads(line)))
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        message = "broad completion input repeats a logical job key"
        raise ValueError(message)
    return jobs


def stage_jobs(root: Path, stage: CampaignStage) -> tuple[BenchmarkJob, ...]:
    if stage == "stage1":
        paths = sorted((root / stage / "deployment").glob("*.jsonl"))
    else:
        master = root / stage / "master.jsonl"
        paths = [master] if master.is_file() else []
    if not paths:
        message = f"{stage} has no frozen runnable job manifest under {root}"
        raise FileNotFoundError(message)
    jobs = _read_jobs(paths)
    if any(job.stage != stage for job in jobs):
        message = f"{stage} manifest contains a job from another stage"
        raise ValueError(message)
    return jobs


def _completed_rows(root: Path, stage: CampaignStage) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in sorted((root / stage / "completed").glob("*.json")):
        try:
            row = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
        key = row.get("job_key")
        if row.get("status") == "done" and isinstance(key, str):
            if key in rows:
                message = f"{stage} repeats completed job {key}"
                raise RuntimeError(message)
            rows[key] = row
    return rows


def _require_rows(
    root: Path,
    stage: CampaignStage,
    jobs: tuple[BenchmarkJob, ...],
) -> dict[str, dict[str, object]]:
    rows = _completed_rows(root, stage)
    expected = {job.key for job in jobs}
    missing = expected - rows.keys()
    unexpected = rows.keys() - expected
    if missing or unexpected:
        message = (
            f"{stage} is not transition-ready: expected={len(expected)}, "
            f"complete={len(expected) - len(missing)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
        raise RuntimeError(message)
    return rows


def _score(row: dict[str, object]) -> float:
    value = float(cast("str | int | float", row["selection_score"]))
    if not math.isfinite(value):
        message = f"non-finite selection score for {row.get('job_key')}"
        raise RuntimeError(message)
    return value


def _job_class(seconds: float) -> Literal["short", "medium", "long"]:
    if seconds < 180.0:
        return "short"
    if seconds <= 1_200.0:
        return "medium"
    return "long"


def _transition_job(
    base: BenchmarkJob,
    *,
    stage: Literal["stage2", "final"],
    seed: int,
    candidate_rank: int,
    epochs: int | None = None,
) -> BenchmarkJob:
    final = stage == "final"
    next_epochs = base.epochs if epochs is None else epochs
    estimated_seconds = (
        base.estimated_seconds
        * next_epochs
        / base.epochs
        * (1.15 if final else 1.0)
    )
    return replace(
        base,
        key=(
            f"broad:{stage}:{base.suite}:{base.dataset}:{base.endpoint}:"
            f"{base.model}:{base.candidate_id}:split{base.split_seed}:seed{seed}"
        ),
        stage=stage,
        candidate_rank=candidate_rank,
        train_seed=seed,
        evaluation_split="test" if final else "validation",
        official_test_accessed=final,
        comparison_group=f"{stage}:{base.suite}:{base.dataset}:{base.endpoint}",
        epochs=next_epochs,
        estimated_seconds=estimated_seconds,
        job_class=_job_class(estimated_seconds),
    )


def _write_master(
    root: Path,
    stage: Literal["stage2", "final"],
    jobs: tuple[BenchmarkJob, ...],
) -> Path:
    if not jobs or any(job.stage != stage for job in jobs):
        message = f"{stage} master must contain at least one {stage} job"
        raise ValueError(message)
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        message = f"{stage} master repeats a logical key"
        raise ValueError(message)
    path = root / stage / "master.jsonl"
    write_once(
        path,
        "".join(
            json.dumps(job.payload(), sort_keys=True) + "\n"
            for job in sorted(jobs, key=lambda item: (-item.estimated_seconds, item.key))
        ),
    )
    return path


def select_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    candidates_per_cell: int = 18,
    top_k: int = TOP_K,
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    stage2_epochs: int | None = None,
) -> dict[str, object]:
    jobs = stage_jobs(root, "stage1")
    rows = _require_rows(root, "stage1", jobs)
    grouped: dict[str, list[BenchmarkJob]] = {}
    for job in jobs:
        grouped.setdefault(job.cell_key, []).append(job)
    selected: dict[str, list[str]] = {}
    stage2: list[BenchmarkJob] = []
    for cell, candidates in sorted(grouped.items()):
        if len(candidates) != candidates_per_cell:
            message = (
                f"{cell} has {len(candidates)} Stage-1 candidates; "
                f"expected {candidates_per_cell}"
            )
            raise RuntimeError(message)
        ranked = sorted(
            candidates,
            key=lambda job: (-_score(rows[job.key]), job.candidate_id),
        )
        top = ranked[:top_k]
        selected[cell] = [job.candidate_id for job in top]
        for rank, job in enumerate(top, start=1):
            stage2.extend(
                _transition_job(
                    job,
                    stage="stage2",
                    seed=seed,
                    candidate_rank=rank,
                    epochs=stage2_epochs,
                )
                for seed in confirmation_seeds
            )
    payload: dict[str, object] = {
        "schema": "alphabet.broad_benchmark.runnable_stage1_selection.v1",
        "source_jobs": len(jobs),
        "runnable_cells": len(grouped),
        "candidates_per_cell": candidates_per_cell,
        "top_k": top_k,
        "confirmation_seeds": list(confirmation_seeds),
        "stage2_epochs": stage2_epochs,
        "selection_rule": "descending validation score; lexical candidate-id tie break",
        "official_test_accessed": False,
        "selected": selected,
        "stage2_jobs": len(stage2),
    }
    write_once(
        root / "stage1" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _write_master(root, "stage2", tuple(stage2))
    return payload


def select_stage2(
    root: Path = DEFAULT_ROOT,
    *,
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    final_seeds: tuple[int, ...] = FINAL_SEEDS,
    final_epochs: int | None = None,
) -> dict[str, object]:
    stage1 = stage_jobs(root, "stage1")
    stage2 = stage_jobs(root, "stage2")
    rows1 = _require_rows(root, "stage1", stage1)
    rows2 = _require_rows(root, "stage2", stage2)
    selection = cast(
        "dict[str, list[str]]",
        json.loads((root / "stage1" / "selection.json").read_text(encoding="utf-8"))[
            "selected"
        ],
    )
    jobs_by_key = {job.key: job for job in (*stage1, *stage2)}
    scores: dict[tuple[str, str], list[float]] = {}
    bases: dict[tuple[str, str], BenchmarkJob] = {}
    for key, row in {**rows1, **rows2}.items():
        job = jobs_by_key[key]
        selected = selection.get(job.cell_key, [])
        if job.candidate_id not in selected:
            continue
        identity = job.cell_key, job.candidate_id
        scores.setdefault(identity, []).append(_score(row))
        bases.setdefault(identity, job)
    frozen: dict[str, str] = {}
    evidence: dict[str, object] = {}
    final_jobs: list[BenchmarkJob] = []
    expected_scores = 1 + len(confirmation_seeds)
    for cell, candidates in sorted(selection.items()):
        ranked: list[tuple[float, str]] = []
        for candidate in candidates:
            values = scores.get((cell, candidate), [])
            if len(values) != expected_scores:
                message = f"{cell}/{candidate} has {len(values)} scores; expected {expected_scores}"
                raise RuntimeError(message)
            ranked.append((mean(values), candidate))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        winner = ranked[0][1]
        frozen[cell] = winner
        evidence[cell] = [
            {"candidate_id": candidate, "mean_validation_score": score}
            for score, candidate in ranked
        ]
        base = bases[(cell, winner)]
        final_jobs.extend(
            _transition_job(
                base,
                stage="final",
                seed=seed,
                candidate_rank=1,
                epochs=final_epochs,
            )
            for seed in final_seeds
        )
    payload: dict[str, object] = {
        "schema": "alphabet.broad_benchmark.runnable_stage2_selection.v1",
        "runnable_cells": len(frozen),
        "selection_seeds": [SEARCH_SEED, *confirmation_seeds],
        "final_seeds": list(final_seeds),
        "final_epochs": final_epochs,
        "selection_rule": "mean validation score; lexical candidate-id tie break",
        "official_test_accessed": False,
        "selected": frozen,
        "evidence": evidence,
        "final_jobs": len(final_jobs),
    }
    write_once(
        root / "stage2" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _write_master(root, "final", tuple(final_jobs))
    return payload


def audit_campaign(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    problems: list[str] = []
    stages: dict[str, object] = {}
    for stage in ("stage1", "stage2", "final"):
        jobs = stage_jobs(root, stage)
        rows = _completed_rows(root, stage)
        expected = {job.key for job in jobs}
        missing = expected - rows.keys()
        unexpected = rows.keys() - expected
        final = stage == "final"
        protocol_errors = sum(
            bool(row.get("official_test_accessed")) != final
            or row.get("evaluation_split") != ("test" if final else "validation")
            for row in rows.values()
        )
        if missing:
            problems.append(f"{stage} missing {len(missing)} jobs")
        if unexpected:
            problems.append(f"{stage} has {len(unexpected)} unexpected jobs")
        if protocol_errors:
            problems.append(f"{stage} has {protocol_errors} protocol errors")
        stages[stage] = {
            "expected": len(expected),
            "completed": len(expected) - len(missing),
            "missing": len(missing),
            "unexpected": len(unexpected),
            "protocol_errors": protocol_errors,
        }
    return {
        "schema": "alphabet.broad_benchmark.runnable_completion_audit.v1",
        "ok": not problems,
        "problems": problems,
        "stages": stages,
    }


__all__ = [
    "audit_campaign",
    "select_stage1",
    "select_stage2",
    "stage_jobs",
]
