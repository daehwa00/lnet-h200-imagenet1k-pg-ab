"""Exact 37-dataset reduced-cost Wave-1 campaign contract."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Final

from .pac_broad_benchmark_queue import (
    UEA_30,
    BenchmarkJob,
)
from .pac_broad_benchmark_queue import (
    stage1_jobs as broad_stage1_jobs,
)
from .pac_fast_completion import (
    FINAL_SEEDS,
    STAGE1_CANDIDATES_PER_CELL,
    STAGE1_EPOCHS,
    STAGE2_SEEDS,
    STAGE2_TOP_K,
)

DEFAULT_ROOT: Final = Path(".omx/results/alphabet-wave1-3gpu-20260727")
FORECAST_DATASETS: Final = ("etth1", "etth2", "traffic", "ili", "exchange-rate")
EXTERNAL_DATASETS: Final = (
    *((name, "classification") for name in UEA_30 if name != "PhonemeSpectra"),
    ("human-activity", "activity-classification"),
    ("ushcn-daily", "interpolation-forecasting"),
)
MODELS: Final = (
    "alphabet",
    "cnn1d",
    "tcn",
    "transformer",
    "mamba",
    "s4d",
    "s5",
    "lru",
    "gru",
    "lstm",
)
FAST_ARCHITECTURES: Final = {
    "cnn1d": "d2-k3",
    "tcn": "d3-k3",
    "transformer": "d1-h2",
    "mamba": "s16-c3",
    "s4d": "d1-s16",
    "s5": "d1-s16",
    "lru": "d1-s16",
    "gru": "d1-s16",
    "lstm": "d1-s16",
}
FAST_WIDTHS: Final = frozenset((32, 64))
FAST_ALPHABET_CAPACITIES: Final = frozenset(((32, 8), (64, 16)))


def _fast_candidate(job: BenchmarkJob) -> bool:
    if job.model == "alphabet":
        return (job.width, job.modes) in FAST_ALPHABET_CAPACITIES
    return (
        job.architecture == FAST_ARCHITECTURES[job.model]
        and job.width in FAST_WIDTHS
    )


def _job_class(seconds: float) -> str:
    if seconds < 180.0:
        return "short"
    if seconds <= 1_200.0:
        return "medium"
    return "long"


def _reduced(job: BenchmarkJob) -> BenchmarkJob:
    seconds = job.estimated_seconds * STAGE1_EPOCHS / job.epochs
    return replace(
        job,
        epochs=STAGE1_EPOCHS,
        estimated_seconds=seconds,
        job_class=_job_class(seconds),  # type: ignore[arg-type]
        comparison_group=f"{job.comparison_group}:shard-{job.candidate_rank % 3}",
    )


def _external_from_template(
    template: BenchmarkJob,
    *,
    dataset: str,
    endpoint: str,
) -> BenchmarkJob:
    job = _reduced(template)
    return replace(
        job,
        key=(
            f"broad:stage1:external:{dataset}:{endpoint}:{job.model}:"
            f"{job.candidate_id}:split{job.split_seed}:seed{job.train_seed}"
        ),
        suite="external",
        dataset=dataset,
        endpoint=endpoint,
        comparison_group=(
            f"stage1:external:{dataset}:{endpoint}:shard-{job.candidate_rank % 3}"
        ),
        data_shard=f"external:{dataset}:{endpoint}",
        blockers=(),
    )


@cache
def stage1_jobs() -> tuple[BenchmarkJob, ...]:
    broad = broad_stage1_jobs()
    forecasts = [
        _reduced(job)
        for job in broad
        if (
            job.suite == "forecasting"
            and job.dataset in FORECAST_DATASETS
            and job.model in MODELS
            and _fast_candidate(job)
            and not job.blockers
        )
    ]
    templates = [
        job
        for job in broad
        if (
            job.suite == "forecasting"
            and job.dataset == "etth1"
            and job.model in MODELS
            and _fast_candidate(job)
            and not job.blockers
        )
    ]
    external = [
        _external_from_template(template, dataset=dataset, endpoint=endpoint)
        for dataset, endpoint in EXTERNAL_DATASETS
        for template in templates
    ]
    jobs = tuple(sorted((*forecasts, *external), key=lambda item: item.key))
    expected = expected_counts()["stage1"]
    if len(jobs) != expected or len({job.key for job in jobs}) != expected:
        message = f"Wave-1 Stage 1 expected {expected} unique jobs, found {len(jobs)}"
        raise RuntimeError(message)
    return jobs


def expected_counts() -> dict[str, int]:
    datasets = len(FORECAST_DATASETS) + len(EXTERNAL_DATASETS)
    cells = datasets * len(MODELS)
    return {
        "datasets": datasets,
        "forecasting_datasets": len(FORECAST_DATASETS),
        "uea_datasets": len(UEA_30),
        "irregular_datasets": 2,
        "models": len(MODELS),
        "cells": cells,
        "stage1": cells * STAGE1_CANDIDATES_PER_CELL,
        "stage2": cells * STAGE2_TOP_K * len(STAGE2_SEEDS),
        "final": cells * len(FINAL_SEEDS),
        "total": cells
        * (
            STAGE1_CANDIDATES_PER_CELL
            + STAGE2_TOP_K * len(STAGE2_SEEDS)
            + len(FINAL_SEEDS)
        ),
    }


def audit_stage1() -> dict[str, object]:
    jobs = stage1_jobs()
    counts = expected_counts()
    cells: dict[str, int] = {}
    for job in jobs:
        cells[job.cell_key] = cells.get(job.cell_key, 0) + 1
    problems: list[str] = []
    if set(cells.values()) != {STAGE1_CANDIDATES_PER_CELL}:
        problems.append("a cell does not contain exactly six Stage-1 candidates")
    if any(job.blockers for job in jobs):
        problems.append("Stage 1 contains blocked jobs")
    if any(job.official_test_accessed or job.evaluation_split != "validation" for job in jobs):
        problems.append("Stage 1 accesses official TEST")
    if {job.model for job in jobs} != set(MODELS):
        problems.append("model set differs from the approved ten")
    return {
        "schema": "alphabet.wave1_stage1_audit.v1",
        "ok": not problems,
        "problems": problems,
        "counts": counts,
        "observed_cells": len(cells),
        "observed_jobs": len(jobs),
        "blocked_jobs": sum(bool(job.blockers) for job in jobs),
        "duplicate_jobs": len(jobs) - len({job.key for job in jobs}),
        "official_test_accessed": any(job.official_test_accessed for job in jobs),
    }


__all__ = [
    "DEFAULT_ROOT",
    "EXTERNAL_DATASETS",
    "FAST_ALPHABET_CAPACITIES",
    "FAST_ARCHITECTURES",
    "FAST_WIDTHS",
    "FORECAST_DATASETS",
    "MODELS",
    "audit_stage1",
    "expected_counts",
    "stage1_jobs",
]
