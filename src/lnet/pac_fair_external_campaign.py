from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final

from .pac_revised_final_queues import _external_historical_seconds

DEFAULT_ROOT: Final = Path(".omx/results/pac-final-alphabet-external16-pro6000-20260713")
PAC_MODEL: Final = "pac_headroom_phase_augmented_ensemble_wp_d64_m16"
DATASETS: Final = (
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
MODELS: Final = (
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
SEEDS: Final = (7, 11, 19, 23, 31)
BATCHES: Final = {
    "lra-listops": 32,
    "lra-retrieval": 16,
}


def enqueue_fair_external(
    root: Path = DEFAULT_ROOT,
    *,
    workers: int = 4,
) -> dict[str, object]:
    if workers < 1:
        message = "workers must be positive"
        raise ValueError(message)
    historical = _external_historical_seconds()
    jobs = [
        (dataset, model, seed, BATCHES.get(dataset, 64))
        for dataset in DATASETS
        for model in MODELS
        for seed in SEEDS
    ]
    weighted = [
        (
            job,
            historical.get(
                (job[0], job[1]),
                historical.get((job[0], "pac"), 120.0),
            ),
        )
        for job in jobs
    ]
    shards: list[list[tuple[str, str, int, int]]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job, weight in sorted(weighted, key=lambda item: item[1], reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += max(weight, 1.0)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards):
        (manifests / f"worker-{index:02d}.tsv").write_text(
            "".join(
                f"{dataset}\t{model}\t{seed}\t{batch_size}\n"
                for dataset, model, seed, batch_size in shard
            ),
            encoding="utf-8",
        )
    contract: dict[str, object] = {
        "schema": "pac_final_alphabet_external16.v1",
        "pac_model": PAC_MODEL,
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "jobs": len(jobs),
        "workers": workers,
        "epochs": 60,
        "patience": 12,
        "parameter_match_tolerance": 0.05,
        "max_baseline_width": 8192,
        "split_policy": "same task-provided train/validation/test split for every model",
        "checkpoint_policy": "minimum validation loss; one final test evaluation",
        "estimated_worker_loads": loads,
    }
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def fair_external_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    datasets = DATASETS
    models = MODELS
    seeds = SEEDS
    contract_path = root / "contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        datasets = tuple(str(value) for value in contract.get("datasets", datasets))
        models = tuple(str(value) for value in contract.get("models", models))
        seeds = tuple(int(value) for value in contract.get("seeds", seeds))
    expected = {
        f"{dataset}-{model}-seed{seed}"
        for dataset in datasets
        for model in models
        for seed in seeds
    }
    done: set[str] = set()
    failed: set[str] = set()
    for run_root in (root / "jobs").glob("*"):
        result = run_root / "results" / "external_comparisons.csv"
        if not result.exists():
            continue
        with result.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if any(row.get("status") == "done" for row in rows):
            done.add(run_root.name)
        elif rows:
            failed.add(run_root.name)
    return {
        "expected": len(expected),
        "completed": len(expected & done),
        "failed": len(expected & failed),
        "remaining": len(expected - done - failed),
        "done": expected <= done,
    }
