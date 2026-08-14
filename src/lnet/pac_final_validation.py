from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Final

from .pac_headroom_screening import HeadroomScreenJob

DEFAULT_FINAL_ROOT: Final = Path(".omx/results/pac-final-validation-20260712")
DEFAULT_REUSE_ROOT: Final = Path(".omx/results/pac-wp-fast-validation-20260712")
SEEDS_UCR: Final = (7, 11, 19, 23, 31)
SEEDS_EXTERNAL: Final = (7, 11, 19)
UCR_DATASETS: Final = (
    "ArrowHead",
    "CinCECGTorso",
    "CricketX",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "Earthquakes",
    "FordA",
    "FordB",
    "GunPoint",
    "ItalyPowerDemand",
    "MoteStrain",
    "Phoneme",
    "Plane",
    "StarLightCurves",
    "Trace",
    "TwoLeadECG",
    "Wafer",
)
# Pathfinder remains excluded by the user's explicit decision. This yields a 34-task
# final table (18 UCR + 16 external) unless that decision is later reversed.
EXTERNAL_DATASETS: Final = (
    "ptb-xl",
    "mit-bih",
    "cwru",
    "speech-commands",
    "ettm1",
    "ettm2",
    "electricity",
    "weather",
    "lra-listops",
    "lra-text",
    "lra-retrieval",
    "lra-image",
    "sequential-mnist",
    "permuted-mnist",
    "sequential-cifar",
    "audioset-balanced",
)
UCR_SECONDS: Final = {
    "ArrowHead": 8.0,
    "CinCECGTorso": 10.0,
    "CricketX": 55.0,
    "ECG200": 12.0,
    "ECG5000": 450.0,
    "ECGFiveDays": 9.0,
    "Earthquakes": 24.0,
    "FordA": 1_200.0,
    "FordB": 1_200.0,
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
EXTERNAL_SECONDS: Final = {
    "ptb-xl": 370.0,
    "mit-bih": 190.0,
    "cwru": 27.0,
    "speech-commands": 1_500.0,
    "ettm1": 145.0,
    "ettm2": 125.0,
    "electricity": 160.0,
    "weather": 115.0,
    "lra-listops": 5_850.0,
    "lra-text": 205.0,
    "lra-retrieval": 15_400.0,
    "lra-image": 580.0,
    "sequential-mnist": 1_225.0,
    "permuted-mnist": 5_750.0,
    "sequential-cifar": 2_600.0,
    "audioset-balanced": 325.0,
}


def final_validation_jobs() -> list[HeadroomScreenJob]:
    jobs = [
        HeadroomScreenJob(
            key=f"ucr_validation:{dataset}:WP:seed{seed}",
            task_kind="ucr_validation",
            dataset=dataset,
            spec="WP",
            seed=seed,
            epochs=100,
            estimated_seconds=UCR_SECONDS[dataset],
        )
        for dataset in UCR_DATASETS
        for seed in SEEDS_UCR
    ]
    jobs.extend(
        HeadroomScreenJob(
            key=f"external:{dataset}:WP:seed{seed}",
            task_kind="external",
            dataset=dataset,
            spec="WP",
            seed=seed,
            epochs=60,
            estimated_seconds=EXTERNAL_SECONDS[dataset],
            patience=12,
        )
        for dataset in EXTERNAL_DATASETS
        for seed in SEEDS_EXTERNAL
    )
    return jobs


def enqueue_final_validation(
    root: Path = DEFAULT_FINAL_ROOT,
    *,
    reuse_root: Path = DEFAULT_REUSE_ROOT,
) -> dict[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    reused = _reuse_completed_ucr(root, reuse_root)
    completed = _result_keys(root / "completed")
    jobs = final_validation_jobs()
    pending = [job for job in jobs if job.key not in completed]
    devices = [("b200", 0, 1.0), ("b200", 1, 1.0), ("b200", 2, 1.0), ("pro6000", 0, 0.75)]
    device_jobs: list[list[HeadroomScreenJob]] = [[] for _ in devices]
    device_loads = [0.0] * len(devices)
    for job in sorted(pending, key=lambda item: item.estimated_seconds, reverse=True):
        # The very long jobs must not share one physical GPU merely because they use
        # different worker processes. Keep them on separate B200 devices.
        eligible = range(3) if job.estimated_seconds >= 3_000.0 else range(len(devices))
        index = min(
            eligible,
            key=lambda item: device_loads[item] / devices[item][2],
        )
        device_jobs[index].append(job)
        device_loads[index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    assignments = []
    assignment_index = 0
    for (platform, gpu, capacity), jobs_for_device in zip(devices, device_jobs, strict=True):
        worker_shards: list[list[HeadroomScreenJob]] = [[] for _ in range(4)]
        worker_loads = [0.0] * 4
        for job in sorted(jobs_for_device, key=lambda item: item.estimated_seconds, reverse=True):
            worker = min(range(4), key=worker_loads.__getitem__)
            worker_shards[worker].append(job)
            worker_loads[worker] += job.estimated_seconds
        for worker, shard in enumerate(worker_shards):
            path = manifests / f"{platform}-gpu{gpu}-worker{worker}.jsonl"
            ordered = sorted(shard, key=lambda item: (item.estimated_seconds, item.key))
            path.write_text(
                "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in ordered),
                encoding="utf-8",
            )
            assignments.append(
                {
                    "index": assignment_index,
                    "platform": platform,
                    "gpu": gpu,
                    "worker": worker,
                    "capacity": capacity,
                    "manifest": str(path),
                    "jobs": len(shard),
                    "estimated_seconds": worker_loads[worker],
                    "device_estimated_seconds": device_loads[len(assignments) // 4],
                }
            )
            assignment_index += 1
    contract = {
        "schema": "pac_final_validation.v1",
        "paper_model": "PAC",
        "internal_spec": "WP",
        "ucr_datasets": list(UCR_DATASETS),
        "external_datasets": list(EXTERNAL_DATASETS),
        "excluded_datasets": ["pathfinder"],
        "seeds_ucr": list(SEEDS_UCR),
        "seeds_external": list(SEEDS_EXTERNAL),
        "total_jobs": len(jobs),
        "reused_jobs": reused,
        "pending_jobs": len(pending),
        "official_test_accessed": False,
        "external_epochs": 60,
        "external_patience": 12,
        "ucr_epochs": 100,
        "assignments": assignments,
    }
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"total": len(jobs), "reused": reused, "pending": len(pending)}


def status_final_validation(root: Path = DEFAULT_FINAL_ROOT) -> dict[str, object]:
    expected = {job.key for job in final_validation_jobs()}
    completed = _result_keys(root / "completed")
    failed = _result_keys(root / "failed")
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed - failed),
        "done": expected <= completed and not (expected & failed),
    }


def _reuse_completed_ucr(root: Path, reuse_root: Path) -> int:
    target = root / "completed"
    target.mkdir(parents=True, exist_ok=True)
    reused = 0
    for source in sorted((reuse_root / "completed").glob("ucr_validation_*_WP_seed*.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("job_key") not in {job.key for job in final_validation_jobs()}:
            continue
        destination = target / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        reused += 1
    return reused


def _result_keys(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        str(json.loads(path.read_text(encoding="utf-8"))["job_key"])
        for path in directory.glob("*.json")
    }
