from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

from .pac_headroom_screening import HeadroomScreenJob

DEFAULT_ROOT: Final = Path(".omx/results/pac-sparse-multiscale-validation-20260712")
GPUS: Final = (0, 1, 2, 7)
SEEDS: Final = (7, 11, 19)
UCR_SEEDS: Final = (7, 11, 19, 23, 31)

EXTERNAL_TASKS: Final[dict[str, float]] = {
    "lra-image": 580.0,
    "sequential-cifar": 2_600.0,
    "audioset-balanced": 325.0,
    "sequential-mnist": 1_225.0,
    "cwru": 27.0,
    "ettm2": 125.0,
}


def sparse_multiscale_jobs() -> list[HeadroomScreenJob]:
    jobs = [
        HeadroomScreenJob(
            key=f"external:{dataset}:SMR:seed{seed}",
            task_kind="external",
            dataset=dataset,
            spec="SMR",
            seed=seed,
            epochs=60,
            estimated_seconds=seconds,
            patience=12,
        )
        for dataset, seconds in EXTERNAL_TASKS.items()
        for seed in SEEDS
    ]
    jobs.extend(
        HeadroomScreenJob(
            key=f"ucr_validation:Phoneme:SMR:seed{seed}",
            task_kind="ucr_validation",
            dataset="Phoneme",
            spec="SMR",
            seed=seed,
            epochs=100,
            estimated_seconds=20.0,
            patience=12,
        )
        for seed in UCR_SEEDS
    )
    return jobs


def enqueue_sparse_multiscale_validation(
    root: Path = DEFAULT_ROOT,
    *,
    workers_per_gpu: int = 2,
) -> dict[str, int]:
    if workers_per_gpu < 1:
        message = "workers_per_gpu must be positive"
        raise ValueError(message)
    jobs = sparse_multiscale_jobs()
    completed = _result_keys(root / "completed")
    pending = [job for job in jobs if job.key not in completed]
    worker_names = [
        f"b200-gpu{gpu}-worker{worker}"
        for gpu in GPUS
        for worker in range(workers_per_gpu)
    ]
    shards: list[list[HeadroomScreenJob]] = [[] for _ in worker_names]
    loads = [0.0] * len(worker_names)
    priority = sorted(
        pending,
        key=lambda job: (
            job.dataset not in {"lra-image", "sequential-cifar", "Phoneme"},
            job.seed != 7,
            -job.estimated_seconds,
            job.key,
        ),
    )
    for job in priority:
        index = min(range(len(shards)), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    assignments: list[dict[str, object]] = []
    for name, shard, load in zip(worker_names, shards, loads, strict=True):
        path = manifests / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard),
            encoding="utf-8",
        )
        assignments.append(
            {
                "worker": name,
                "manifest": str(path),
                "jobs": len(shard),
                "estimated_seconds": load,
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": "pac_sparse_multiscale_validation.v1",
        "candidate": "SMR",
        "description": "sparse multiscale low/detail residual mixer plus one PAC core",
        "scales": [1, 2, 4, 8],
        "gpus": list(GPUS),
        "workers_per_gpu": workers_per_gpu,
        "external_seeds": list(SEEDS),
        "ucr_seeds": list(UCR_SEEDS),
        "official_test_accessed": False,
        "total_jobs": len(jobs),
        "pending_jobs": len(pending),
        "assignments": assignments,
    }
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"total": len(jobs), "pending": len(pending), "workers": len(worker_names)}


def sparse_multiscale_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    expected = {job.key for job in sparse_multiscale_jobs()}
    completed = _result_keys(root / "completed")
    failed = _result_keys(root / "failed")
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed - failed),
        "done": expected <= completed and not (expected & failed),
    }


def _result_keys(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        str(json.loads(path.read_text(encoding="utf-8"))["job_key"])
        for path in directory.glob("*.json")
    }
