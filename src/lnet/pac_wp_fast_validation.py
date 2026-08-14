from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Final, cast

from .pac_headroom_screening import HeadroomScreenJob, ScreenTaskKind

DEFAULT_WP_FAST_ROOT: Final = Path(".omx/results/pac-wp-fast-validation-20260712")
SEEDS: Final = (7, 11, 19)

# Existing median/p90 wall times were used to keep this queue bounded. The three
# largest UCR training sets and Speech Commands are deliberately a later tier.
FAST_UCR_SECONDS: Final[dict[str, float]] = {
    "ArrowHead": 8.0,
    "CinCECGTorso": 10.0,
    "CricketX": 55.0,
    "ECG200": 12.0,
    "ECGFiveDays": 9.0,
    "Earthquakes": 24.0,
    "GunPoint": 8.0,
    "ItalyPowerDemand": 8.0,
    "MoteStrain": 8.0,
    "Phoneme": 20.0,
    "Plane": 13.0,
    "StarLightCurves": 80.0,
    "Trace": 17.0,
    "TwoLeadECG": 8.0,
    "Wafer": 60.0,
}
FAST_EXTERNAL_SECONDS: Final[dict[str, float]] = {
    "ettm1": 60.0,
    "electricity": 90.0,
    "weather": 60.0,
    "mit-bih": 75.0,
    "ptb-xl": 150.0,
    "sequential-mnist": 280.0,
}
FORECASTING_DATASETS: Final = frozenset({"ettm1", "electricity", "weather"})


def wp_fast_jobs(seeds: tuple[int, ...] = SEEDS) -> list[HeadroomScreenJob]:
    jobs: list[HeadroomScreenJob] = []
    for dataset, seconds in FAST_UCR_SECONDS.items():
        jobs.extend(_paired_jobs("ucr_validation", dataset, 100, seconds, seeds))
    for dataset, seconds in FAST_EXTERNAL_SECONDS.items():
        jobs.extend(_paired_jobs("external", dataset, 20, seconds, seeds))
    return jobs


def enqueue_wp_fast(
    root: Path = DEFAULT_WP_FAST_ROOT,
    *,
    workers: int = 4,
    seeds: tuple[int, ...] = SEEDS,
) -> int:
    if workers < 1:
        message = "workers must be positive"
        raise ValueError(message)
    jobs = wp_fast_jobs(seeds)
    shards: list[list[HeadroomScreenJob]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(jobs, key=lambda item: item.estimated_seconds, reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards):
        # Short jobs first yield evidence quickly; LPT assignment keeps tails balanced.
        ordered = sorted(shard, key=lambda item: (item.estimated_seconds, item.key))
        (manifests / f"worker-{index}.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in ordered),
            encoding="utf-8",
        )
    contract = {
        "schema": "pac_wp_fast_validation.v1",
        "comparison": ["B", "WP"],
        "baseline": "Revised PAC",
        "candidate": "Revised PAC-WP",
        "datasets": {
            "ucr_validation": list(FAST_UCR_SECONDS),
            "external": list(FAST_EXTERNAL_SECONDS),
        },
        "seeds": list(seeds),
        "jobs": len(jobs),
        "workers": workers,
        "estimated_worker_seconds": loads,
        "evaluation_split": "validation",
        "official_test_accessed": False,
        "excluded_intermediate_or_long": [
            "ECG5000",
            "FordA",
            "FordB",
            "speech-commands",
            "permuted-mnist",
            "sequential-cifar",
            "all LRA tasks",
            "pathfinder",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(jobs)


def status_wp_fast(root: Path = DEFAULT_WP_FAST_ROOT) -> dict[str, object]:
    expected = {job.key for job in wp_fast_jobs()}
    completed = _result_keys(root / "completed")
    failed = _result_keys(root / "failed")
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed - failed),
        "done": expected <= completed and not (expected & failed),
    }


def report_wp_fast(root: Path = DEFAULT_WP_FAST_ROOT) -> Path:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["spec"]))].append(row)
    datasets: list[dict[str, object]] = []
    for dataset in sorted({key[0] for key in grouped}):
        baseline = grouped.get((dataset, "B"), [])
        candidate = grouped.get((dataset, "WP"), [])
        if not baseline or not candidate:
            continue
        metric, lower_is_better = _primary_metric(dataset, baseline[0])
        b_by_seed = {
            int(cast("int | str", row["seed"])): float(cast("float | int | str", row[metric]))
            for row in baseline
        }
        w_by_seed = {
            int(cast("int | str", row["seed"])): float(cast("float | int | str", row[metric]))
            for row in candidate
        }
        shared = sorted(b_by_seed.keys() & w_by_seed.keys())
        if not shared:
            continue
        b_mean = statistics.fmean(b_by_seed[seed] for seed in shared)
        w_mean = statistics.fmean(w_by_seed[seed] for seed in shared)
        signed_improvement = b_mean - w_mean if lower_is_better else w_mean - b_mean
        wins = sum(
            (w_by_seed[seed] < b_by_seed[seed])
            if lower_is_better
            else (w_by_seed[seed] > b_by_seed[seed])
            for seed in shared
        )
        datasets.append(
            {
                "dataset": dataset,
                "metric": metric,
                "lower_is_better": lower_is_better,
                "paired_seeds": shared,
                "baseline_mean": b_mean,
                "wp_mean": w_mean,
                "signed_improvement": signed_improvement,
                "wp_seed_wins": wins,
            }
        )
    dataset_mean_wins = sum(
        float(cast("float | int", row["signed_improvement"])) > 0 for row in datasets
    )
    dataset_mean_ties = sum(
        float(cast("float | int", row["signed_improvement"])) == 0 for row in datasets
    )
    payload = {
        "schema": "pac_wp_fast_validation_report.v1",
        "status": status_wp_fast(root),
        "datasets": datasets,
        "dataset_mean_wins": dataset_mean_wins,
        "dataset_mean_ties": dataset_mean_ties,
        "dataset_count": len(datasets),
        "official_test_accessed": False,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    target = reports / "wp_fast_validation.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _paired_jobs(
    task_kind: ScreenTaskKind,
    dataset: str,
    epochs: int,
    seconds: float,
    seeds: tuple[int, ...],
) -> list[HeadroomScreenJob]:
    return [
        HeadroomScreenJob(
            key=f"{task_kind}:{dataset}:{spec}:seed{seed}",
            task_kind=task_kind,
            dataset=dataset,
            spec=spec,
            seed=seed,
            epochs=epochs,
            estimated_seconds=seconds * (1.1 if spec == "WP" else 1.0),
        )
        for spec in ("B", "WP")
        for seed in seeds
    ]


def _result_keys(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        str(json.loads(path.read_text(encoding="utf-8"))["job_key"])
        for path in directory.glob("*.json")
    }


def _primary_metric(dataset: str, row: dict[str, object]) -> tuple[str, bool]:
    if dataset in FORECASTING_DATASETS:
        return "validation_mse", True
    if str(row["task_kind"]) == "ucr_validation":
        return "validation_balanced_accuracy", False
    if "validation_macro_auprc" in row:
        return "validation_macro_auprc", False
    return "validation_accuracy", False
