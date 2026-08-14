from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Final

from .pac_confirmatory_baselines import confirmatory_implementation_metadata
from .pac_recommended_low_data_types import LowDataJob

REVISED_MODEL: Final = "pac_stiefel_revised_fixed_mean_nogate_untied_d64_m16"
SELECTION_PATH: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/"
    "confirmatory_baseline_selection.json"
)
PROTOCOL_PATH: Final = Path(".omx/protocols/pac_tf_confirmatory_20260711.json")
REVISED_VALIDATION_PATH: Final = Path(
    ".omx/results/pac-revised-budget-matched-baselines-20260712/results/"
    "low_data_recommended_real.csv"
)
REVISED_EXTERNAL_PATH: Final = Path(
    ".omx/results/pac-35-dataset-comparison-20260712/input/revised_external.csv"
)
EXTERNAL_JOB_ROOT: Final = Path(
    ".omx/results/pac-selected-d64m16-external-20260711/jobs"
)
UCR_FAMILIES: Final = (
    "tcn",
    "cnn1d",
    "gru",
    "lstm",
    "transformer",
    "mamba",
    "s4d",
    "inception_time",
)
EXTERNAL_MODELS: Final = (
    "pac",
    "tcn",
    "cnn1d",
    "transformer",
    "mamba",
    "gru",
    "lstm",
    "s4d",
    "minirocket",
    "inception_time",
)
EXTERNAL_BATCHES: Final = {
    "ptb-xl": 64,
    "mit-bih": 64,
    "cwru": 64,
    "speech-commands": 64,
    "pathfinder": 32,
    "ettm1": 64,
    "ettm2": 64,
    "electricity": 64,
    "weather": 64,
    "lra-listops": 32,
    "lra-text": 64,
    "lra-retrieval": 16,
    "lra-image": 64,
    "sequential-mnist": 64,
    "permuted-mnist": 64,
    "sequential-cifar": 64,
    "audioset-balanced": 64,
}


def enqueue_final_queues(
    ucr_root: Path,
    external_root: Path,
    *,
    ucr_shards: int = 8,
    b200_external_workers_per_gpu: int = 4,
    pro_external_workers: int = 4,
) -> None:
    enqueue_ucr_test_shards(ucr_root, shard_count=ucr_shards)
    enqueue_external_shards(
        external_root,
        b200_workers_per_gpu=b200_external_workers_per_gpu,
        pro_workers=pro_external_workers,
    )


def enqueue_ucr_test_shards(root: Path, *, shard_count: int = 8) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected_trials = selection["selected_trials"]
    datasets = tuple(protocol["development_datasets"] + protocol["untouched_final_datasets"])
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    refit_epochs, weights = _revised_validation_statistics()
    jobs: list[tuple[LowDataJob, float]] = []
    for seed in seeds:
        for dataset in datasets:
            for family in UCR_FAMILIES:
                selected = selected_trials[family]
                trial = int(selected["trial"])
                job = LowDataJob(
                    key=f"revised_ucr_test:{family}:{dataset}:seed{seed}",
                    seed=seed,
                    model=family,
                    dataset=dataset,
                    ratio=1.0,
                    evaluation_split="test",
                    refit_full_train=True,
                    data_protocol="clean_stratified",
                    restore_best_validation=False,
                    evaluation_collection="revised_budget_official_ucr_test",
                    baseline_family=family,  # type: ignore[arg-type]
                    reference_model=REVISED_MODEL,
                    validation_trial=trial,
                    architecture_metadata_json=json.dumps(
                        confirmatory_implementation_metadata(family, trial),  # type: ignore[arg-type]
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    refit_epochs=refit_epochs[family],
                    learning_rate=float(selected["learning_rate"]),
                    weight_decay=float(selected["weight_decay"]),
                    parameter_match_tolerance=0.065,
                )
                jobs.append((job, weights[(family, dataset)]))
    shards = _balanced_shards(jobs, [1.0] * shard_count)
    for index, shard in enumerate(shards):
        shard_root = root / "shards" / f"shard-{index}"
        shard_root.mkdir(parents=True, exist_ok=True)
        (shard_root / "queue_manifest.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job, _ in shard),
            encoding="utf-8",
        )
    root.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": "pac_revised_budget_official_ucr_test.v1",
        "reference_model": REVISED_MODEL,
        "datasets": list(datasets),
        "families": list(UCR_FAMILIES),
        "seeds": list(seeds),
        "jobs": len(jobs),
        "shards": shard_count,
        "refit_epochs": refit_epochs,
        "parameter_match_tolerance": 0.065,
        "test_policy": "full official TRAIN refit, one official TEST evaluation",
    }
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def enqueue_external_shards(
    root: Path,
    *,
    b200_workers_per_gpu: int = 4,
    pro_workers: int = 4,
) -> None:
    assignments = [
        ("b200", gpu, worker)
        for worker in range(b200_workers_per_gpu)
        for gpu in (0, 1, 2, 4, 5, 6, 7)
    ]
    assignments.extend(("pro6000", 0, worker) for worker in range(pro_workers))
    capacities = [1.0 if platform == "b200" else 0.75 for platform, _, _ in assignments]
    historical = _external_historical_seconds()
    jobs: list[tuple[tuple[str, str, int, int], float]] = []
    for dataset, batch_size in EXTERNAL_BATCHES.items():
        for model in EXTERNAL_MODELS:
            for seed in (7, 11, 19):
                weight = historical.get((dataset, model), historical.get((dataset, "pac"), 60.0))
                jobs.append(((dataset, model, seed, batch_size), max(weight, 1.0)))
    shards = _balanced_shards(jobs, capacities)
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    assignment_rows = []
    for index, (assignment, shard) in enumerate(zip(assignments, shards, strict=True)):
        platform, gpu, worker = assignment
        manifest = manifest_root / f"worker-{index:02d}.tsv"
        manifest.write_text(
            "".join(
                f"{dataset}\t{model}\t{seed}\t{batch_size}\n"
                for (dataset, model, seed, batch_size), _ in shard
            ),
            encoding="utf-8",
        )
        assignment_rows.append(
            {
                "index": index,
                "platform": platform,
                "gpu": gpu,
                "worker": worker,
                "manifest": str(manifest),
                "jobs": len(shard),
                "estimated_seconds": sum(weight for _, weight in shard),
            }
        )
    contract = {
        "schema": "pac_revised_external_final.v1",
        "reference_model": REVISED_MODEL,
        "datasets": list(EXTERNAL_BATCHES),
        "models": list(EXTERNAL_MODELS),
        "seeds": [7, 11, 19],
        "jobs": len(jobs),
        "epochs": 60,
        "patience": 12,
        "max_baseline_width": 8192,
        "parameter_match_tolerance": 0.05,
        "assignments": assignment_rows,
    }
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _revised_validation_statistics() -> tuple[dict[str, int], dict[tuple[str, str], float]]:
    rows = list(csv.DictReader(REVISED_VALIDATION_PATH.open(newline="", encoding="utf-8")))
    epochs: dict[str, list[int]] = {family: [] for family in UCR_FAMILIES}
    elapsed: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.get("status") != "done" or row["model"] not in UCR_FAMILIES:
            continue
        family = row["model"]
        dataset = row["dataset_or_task"]
        epochs[family].append(int(row["best_epoch"]))
        elapsed.setdefault((family, dataset), []).append(float(row["elapsed_total_time"]))
    refit_epochs = {
        family: math.floor(median(values) + 0.5) for family, values in epochs.items()
    }
    weights = {key: median(values) for key, values in elapsed.items()}
    return refit_epochs, weights


def _external_historical_seconds() -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = {}
    paths = sorted(EXTERNAL_JOB_ROOT.glob("*/results/external_comparisons.csv"))
    paths.append(REVISED_EXTERNAL_PATH)
    for path in paths:
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
            if row.get("status") != "done":
                continue
            model = "pac" if row["model"] == "pac" else row["model"]
            values.setdefault((row["dataset"], model), []).append(float(row["train_seconds"]))
    return {key: median(observations) for key, observations in values.items()}


def _balanced_shards[T](
    jobs: list[tuple[T, float]], capacities: list[float]
) -> list[list[tuple[T, float]]]:
    shards: list[list[tuple[T, float]]] = [[] for _ in capacities]
    loads = [0.0] * len(capacities)
    for job in sorted(jobs, key=lambda item: item[1], reverse=True):
        index = min(range(len(shards)), key=lambda item: loads[item] / capacities[item])
        shards[index].append(job)
        loads[index] += job[1]
    return shards
