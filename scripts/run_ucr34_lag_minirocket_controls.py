"""TRAIN-only UCR-34 lag-logistic controls."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # type: ignore[import-not-found]  # noqa: E402
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]  # noqa: E402
from sklearn.pipeline import make_pipeline  # type: ignore[import-not-found]  # noqa: E402
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]  # noqa: E402

from lnet.pac_eval_sections import clean_validation_classification_task  # noqa: E402
from lnet.pac_real_data import load_ucr_train_only  # noqa: E402

DATASETS = (
    "ArrowHead", "CinCECGTorso", "CricketX", "ECG200", "ECG5000",
    "ECGFiveDays", "Earthquakes", "FordA", "FordB", "GunPoint",
    "ItalyPowerDemand", "MoteStrain", "Plane", "StarLightCurves", "Trace",
    "TwoLeadECG", "Wafer", "ACSF1", "Adiac", "BME", "CBF", "Coffee", "Crop",
    "EOGHorizontalSignal", "InsectEPGRegularTrain", "InsectWingbeatSound",
    "Meat", "PowerCons", "Rock", "ShapesAll", "SmoothSubspace", "SwedishLeaf",
    "UWaveGestureLibraryAll", "Worms",
)
SELECTION_SEEDS = (7, 11, 19)
FINAL_SEEDS = (23, 31, 43, 47, 59)
SEEDS = (*SELECTION_SEEDS, *FINAL_SEEDS)
LAGS = (1, 2, 4, 8, 16, 32)
CS = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
DEFAULT_ROOT = Path(".omx/results/pac-ucr34-lag-logistic-controls-20260727")
UCR_ROOT = Path(".omx/data/ucr")


def autocovariance_features(values: np.ndarray, max_lag: int) -> np.ndarray:
    x = values[..., 0]
    x = x - x.mean(axis=1, keepdims=True)
    return np.stack(
        [
            np.mean(x[:, lag:] * x[:, : x.shape[1] - lag], axis=1)
            if lag
            else np.mean(x * x, axis=1)
            for lag in range(max_lag + 1)
        ],
        axis=1,
    )


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.tmp-{random.randrange(1 << 30)}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def result_path(root: Path, dataset: str, seed: int) -> Path:
    return root / "completed" / f"{dataset}__seed{seed}.json"


def run_job(root: Path, dataset: str, seed: int) -> None:
    output = result_path(root, dataset, seed)
    if output.exists() and output.stat().st_size:
        return
    task = clean_validation_classification_task(
        load_ucr_train_only(dataset, UCR_ROOT), seed
    )
    train_x = task.train_inputs.numpy()
    train_y = task.train_labels.numpy()
    valid_x = task.validation_inputs.numpy()
    valid_y = task.validation_labels.numpy()
    unique, counts = np.unique(train_y, return_counts=True)
    stratify = train_y if counts.min() >= 2 else None
    inner_train, inner_valid = train_test_split(
        np.arange(len(train_y)),
        test_size=0.2,
        random_state=seed + 10_000,
        stratify=stratify,
    )
    active_lags = sorted({min(lag, train_x.shape[1] - 1) for lag in LAGS})
    lag_features = {
        lag: (
            autocovariance_features(train_x, lag),
            autocovariance_features(valid_x, lag),
        )
        for lag in active_lags
    }
    search = []
    for lag in active_lags:
        features, _ = lag_features[lag]
        for regularization in CS:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=regularization,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            )
            model.fit(features[inner_train], train_y[inner_train])
            score = balanced_accuracy_score(
                train_y[inner_valid], model.predict(features[inner_valid])
            )
            search.append((score, -lag, -regularization, lag, regularization))
    _, _, _, selected_lag, selected_c = max(search)
    train_features, valid_features = lag_features[selected_lag]
    lag_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=selected_c,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    lag_model.fit(train_features, train_y)
    lag_train = balanced_accuracy_score(train_y, lag_model.predict(train_features))
    lag_valid = balanced_accuracy_score(valid_y, lag_model.predict(valid_features))

    _write(
        output,
        {
            "dataset": dataset,
            "seed": seed,
            "lag_selected": selected_lag,
            "lag_c": selected_c,
            "lag_train_balanced_accuracy": lag_train,
            "lag_validation_balanced_accuracy": lag_valid,
            "split": "official TRAIN-derived outer validation; nested lag/C selection",
            "official_test_accessed": False,
        },
    )


def run(root: Path, shard_index: int, shard_count: int) -> None:
    work = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
    for index, (dataset, seed) in enumerate(work):
        if index % shard_count == shard_index:
            run_job(root, dataset, seed)


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
            "lag_selection_mean": mean(
                row["lag_validation_balanced_accuracy"] for row in selection
            ),
            "lag_selection_sample_sd": stdev(
                row["lag_validation_balanced_accuracy"] for row in selection
            ),
            "lag_final_mean": mean(
                row["lag_validation_balanced_accuracy"] for row in final
            ),
            "lag_final_sample_sd": stdev(
                row["lag_validation_balanced_accuracy"] for row in final
            ),
        }
    lag_selection_means = [
        values["lag_selection_mean"] for values in datasets.values()
    ]
    lag_final_means = [values["lag_final_mean"] for values in datasets.values()]
    payload = {
        "schema": "alphabet.ucr34_lag_logistic_controls.summary.v1",
        "datasets": datasets,
        "aggregate": {
            "datasets": len(DATASETS),
            "lag_selection_mean_across_datasets": mean(lag_selection_means),
            "lag_final_mean_across_datasets": mean(lag_final_means),
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
