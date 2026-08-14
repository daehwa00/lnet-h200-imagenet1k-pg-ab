"""Canonical aeon MiniROCKET on frozen TRAIN-derived UCR-34 validation splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aeon.transformations.collection.convolution_based import MiniRocket  # type: ignore[import-not-found]  # noqa: E402
from sklearn.linear_model import RidgeClassifierCV  # type: ignore[import-not-found]  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # type: ignore[import-not-found]  # noqa: E402
from sklearn.pipeline import make_pipeline  # type: ignore[import-not-found]  # noqa: E402
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]  # noqa: E402

from lnet.pac_eval_sections import clean_validation_classification_task  # noqa: E402
from lnet.pac_real_data import load_ucr_train_only  # noqa: E402

from scripts.run_ucr34_lag_minirocket_controls import (  # noqa: E402
    DATASETS,
    FINAL_SEEDS,
    SELECTION_SEEDS,
    UCR_ROOT,
)

SEEDS = (*SELECTION_SEEDS, *FINAL_SEEDS)
DEFAULT_ROOT = Path(".omx/results/pac-ucr34-canonical-minirocket-20260727")


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp-{random.randrange(1 << 30)}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def result_path(root: Path, dataset: str, seed: int) -> Path:
    return root / "completed" / f"{dataset}__seed{seed}.json"


def run_job(root: Path, dataset: str, seed: int) -> None:
    output = result_path(root, dataset, seed)
    if output.exists() and output.stat().st_size:
        return
    task = clean_validation_classification_task(
        load_ucr_train_only(dataset, UCR_ROOT), seed
    )
    train_x = np.transpose(task.train_inputs.numpy(), (0, 2, 1))
    validation_x = np.transpose(task.validation_inputs.numpy(), (0, 2, 1))
    train_y = task.train_labels.numpy()
    validation_y = task.validation_labels.numpy()

    transform = MiniRocket(n_kernels=10_000, random_state=seed)
    train_features = transform.fit_transform(train_x)
    validation_features = transform.transform(validation_x)
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10),
            class_weight="balanced",
        ),
    )
    classifier.fit(train_features, train_y)
    _write(
        output,
        {
            "dataset": dataset,
            "seed": seed,
            "train_balanced_accuracy": balanced_accuracy_score(
                train_y, classifier.predict(train_features)
            ),
            "validation_balanced_accuracy": balanced_accuracy_score(
                validation_y, classifier.predict(validation_features)
            ),
            "implementation": "aeon MiniRocket",
            "n_kernels_requested": 10_000,
            "classifier": "StandardScaler(with_mean=False) + RidgeClassifierCV",
            "split": "official TRAIN-derived validation",
            "official_test_accessed": False,
        },
    )


def run(root: Path, shard_index: int, shard_count: int) -> None:
    work = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
    for index, (dataset, seed) in enumerate(work):
        if index % shard_count == shard_index:
            run_job(root, dataset, seed)


def _stats(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {"mean": mean(values), "sample_sd": stdev(values)}


def report(root: Path) -> dict[str, object]:
    rows = [
        json.loads(result_path(root, dataset, seed).read_text())
        for dataset in DATASETS
        for seed in SEEDS
    ]
    datasets = {}
    for dataset in DATASETS:
        chosen = [row for row in rows if row["dataset"] == dataset]
        selection = [row for row in chosen if row["seed"] in SELECTION_SEEDS]
        final = [row for row in chosen if row["seed"] in FINAL_SEEDS]
        datasets[dataset] = {
            "selection_validation": _stats(selection, "validation_balanced_accuracy"),
            "selection_train": _stats(selection, "train_balanced_accuracy"),
            "final_validation": _stats(final, "validation_balanced_accuracy"),
            "final_train": _stats(final, "train_balanced_accuracy"),
        }
    payload = {
        "schema": "alphabet.ucr34.canonical_minirocket.summary.v1",
        "implementation": "aeon MiniRocket, 10,000 requested kernels",
        "datasets": datasets,
        "aggregate": {
            "datasets": len(DATASETS),
            "selection_validation_mean": mean(
                value["selection_validation"]["mean"] for value in datasets.values()
            ),
            "selection_train_mean": mean(
                value["selection_train"]["mean"] for value in datasets.values()
            ),
            "final_validation_mean": mean(
                value["final_validation"]["mean"] for value in datasets.values()
            ),
            "final_train_mean": mean(
                value["final_train"]["mean"] for value in datasets.values()
            ),
        },
        "official_test_accessed": False,
    }
    _write(root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.root, args.shard_index, args.shard_count)
    else:
        result = report(args.root)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
