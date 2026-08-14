# ruff: noqa: EM102, TRY003
"""Reduced-cost three-stage campaign for the ten public Wave-2 sensor tasks."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Final

from .pac_broad_benchmark_queue import BenchmarkJob
from .pac_broad_benchmark_queue import stage1_jobs as broad_stage1_jobs
from .pac_fast_completion import (
    FINAL_SEEDS,
    STAGE1_CANDIDATES_PER_CELL,
    STAGE1_EPOCHS,
    STAGE2_SEEDS,
    STAGE2_TOP_K,
)

DEFAULT_ROOT: Final = Path(".omx/results/alphabet-wave2-3gpu-20260727")
DATASETS: Final = (
    ("sleepedf-78", "sleep-stage-5"),
    ("isruc-sleep", "sleep-stage-5"),
    ("chb-mit", "seizure-detection"),
    ("bci-iv-2a", "motor-imagery-4"),
    ("mfpt-bearing", "fault-classification"),
    ("paderborn-kat", "fault-classification"),
    ("xjtu-sy", "fault-classification"),
    ("ims-bearing", "fault-classification"),
    ("chapman-shaoxing", "rhythm-11"),
    ("cpsc-2018", "arrhythmia-9"),
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


def _wave2_job(
    template: BenchmarkJob,
    *,
    dataset: str,
    endpoint: str,
) -> BenchmarkJob:
    seconds = template.estimated_seconds * STAGE1_EPOCHS / template.epochs
    job = replace(
        template,
        epochs=STAGE1_EPOCHS,
        estimated_seconds=seconds,
        job_class=_job_class(seconds),  # type: ignore[arg-type]
    )
    return replace(
        job,
        key=(
            f"wave2:stage1:external:{dataset}:{endpoint}:{job.model}:"
            f"{job.candidate_id}:split{job.split_seed}:seed{job.train_seed}"
        ),
        suite="external",
        dataset=dataset,
        endpoint=endpoint,
        comparison_group=(
            f"wave2:stage1:{dataset}:{endpoint}:shard-{job.candidate_rank % 3}"
        ),
        data_shard=f"external:{dataset}:{endpoint}",
        blockers=(),
    )


@cache
def stage1_jobs() -> tuple[BenchmarkJob, ...]:
    templates = [
        job
        for job in broad_stage1_jobs()
        if (
            job.suite == "forecasting"
            and job.dataset == "etth1"
            and job.model in MODELS
            and _fast_candidate(job)
            and not job.blockers
        )
    ]
    jobs = tuple(
        sorted(
            (
                _wave2_job(template, dataset=dataset, endpoint=endpoint)
                for dataset, endpoint in DATASETS
                for template in templates
            ),
            key=lambda item: item.key,
        )
    )
    expected = expected_counts()["stage1"]
    if len(jobs) != expected or len({job.key for job in jobs}) != expected:
        raise RuntimeError(
            f"Wave-2 Stage 1 expected {expected} unique jobs, found {len(jobs)}"
        )
    return jobs


def expected_counts() -> dict[str, int]:
    cells = len(DATASETS) * len(MODELS)
    return {
        "datasets": len(DATASETS),
        "models": len(MODELS),
        "cells": cells,
        "stage1": cells * STAGE1_CANDIDATES_PER_CELL,
        "stage2": cells * STAGE2_TOP_K * len(STAGE2_SEEDS),
        "final": cells * len(FINAL_SEEDS),
    }


__all__ = ["DATASETS", "DEFAULT_ROOT", "MODELS", "expected_counts", "stage1_jobs"]
