from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from lnet.astronomy.features import broker_features
from lnet.astronomy.metrics import classification_metrics
from lnet.astronomy.plasticc import (
    PLASTICC_KNOWN_CLASS_WEIGHTS,
    PLASTICC_KNOWN_TARGETS,
    read_light_curves,
    read_phase0_labels,
    stratified_train_validation_split,
)


def _metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    metrics = classification_metrics(
        torch.from_numpy(probability),
        torch.from_numpy(target),
    )
    return {
        "weighted_log_loss": float(metrics["weighted_log_loss_known14"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "expected_calibration_error": float(metrics["ece"]),
    }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.data_dir / "plasticc_train_metadata.csv.gz"
    light_curves = args.data_dir / "plasticc_train_lightcurves.csv.gz"
    labels = read_phase0_labels(
        metadata,
        targets=PLASTICC_KNOWN_TARGETS,
        max_objects_per_class=10_000_000,
        seed=20260729,
    )
    curves = read_light_curves(light_curves, labels)
    train_ids, validation_ids = stratified_train_validation_split(
        labels,
        seed=20260729,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "official_test_accessed": False,
        "evaluation_split": "validation",
        "metric": "weighted_log_loss_known14",
        "metadata_sha256": _digest(metadata),
        "light_curves_sha256": _digest(light_curves),
        "known_targets": PLASTICC_KNOWN_TARGETS,
        "class_weights": PLASTICC_KNOWN_CLASS_WEIGHTS,
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "split_seed": 20260729,
    }
    (args.output_dir / "train-validation-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    results: list[dict[str, int | float]] = []
    for seed in (7, 11, 19, 23, 31):
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
            class_weight=dict(enumerate(PLASTICC_KNOWN_CLASS_WEIGHTS)),
        )
        train_features = np.stack([broker_features(curves[object_id]) for object_id in train_ids])
        train_target = np.asarray([labels[object_id] for object_id in train_ids])
        validation_features = np.stack(
            [broker_features(curves[object_id]) for object_id in validation_ids]
        )
        validation_target = np.asarray([labels[object_id] for object_id in validation_ids])
        model.fit(train_features, train_target)
        probability = model.predict_proba(validation_features)
        result = {"seed": seed, **_metrics(probability, validation_target)}
        results.append(result)
        joblib.dump(model, args.output_dir / f"rf-seed{seed}.joblib")
    (args.output_dir / "validation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
