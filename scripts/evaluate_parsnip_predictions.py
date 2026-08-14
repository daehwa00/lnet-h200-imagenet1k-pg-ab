from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import json
from pathlib import Path

import astropy.table
import numpy as np
import parsnip
import torch

from lnet.astronomy.metrics import classification_metrics

TARGET_NAMES = (
    "muLens-Single",
    "TDE",
    "EB",
    "SNII",
    "SNIax",
    "Mira",
    "SNIbc",
    "KN",
    "M-dwarf",
    "SNIa-91bg",
    "AGN",
    "SNIa",
    "RRL",
    "SLSN-I",
)


def _label_name(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    target = np.asarray([TARGET_NAMES.index(_label_name(value)) for value in labels])
    return classification_metrics(
        torch.from_numpy(probability),
        torch.from_numpy(target),
    )


def _probability(
    classifier: parsnip.Classifier,
    classified: astropy.table.Table,
) -> np.ndarray:
    probability = np.zeros((len(classified), len(TARGET_NAMES)))
    for class_name, column_name in zip(
        classifier.class_names,
        classified.colnames[1:],
        strict=True,
    ):
        probability[:, TARGET_NAMES.index(_label_name(class_name))] = np.asarray(
            classified[column_name]
        )
    return probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = astropy.table.Table.read(args.train_predictions, path="/predictions")
    validation = astropy.table.Table.read(
        args.validation_predictions,
        path="/predictions",
    )
    if "original_object_id" not in train.colnames:
        # ParSNIP 1.4.3 requires this augmentation-group column even for an
        # unaugmented prediction table produced by its own CLI.
        train["original_object_id"] = train["object_id"]
    tuning: list[dict[str, object]] = []
    best_loss = float("inf")
    classifier: parsnip.Classifier | None = None
    for min_child_weight in (1.0, 10.0, 100.0, 1000.0):
        candidate = parsnip.Classifier()
        out_of_fold = candidate.train(
            train,
            num_folds=10,
            reweight=True,
            min_child_weight=min_child_weight,
        )
        metrics = _metrics(_probability(candidate, out_of_fold), np.asarray(train["type"]))
        tuning.append(
            {
                "min_child_weight": min_child_weight,
                "train_out_of_fold_metrics": metrics,
            }
        )
        candidate_loss = float(metrics["weighted_log_loss_known14"])
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            classifier = candidate
    if classifier is None:
        message = "ParSNIP classifier selection produced no candidates"
        raise RuntimeError(message)
    classifier.write(args.classifier)
    classified = classifier.classify(validation)
    probability = _probability(classifier, classified)
    payload = {
        "model": "ParSNIP 1.4.3 representation + official LightGBM classifier",
        "metric_scope": (
            "ParSNIP-accepted supernova-like subset of the shared validation split"
        ),
        "coverage_warning": (
            "ParSNIP's PLAsTiCC loader rejects non-supernova-like classes; "
            "this is not an all-14-class broker comparison"
        ),
        "classifier_selection": (
            "min_child_weight selected by train-only 10-fold out-of-fold "
            "weighted log-loss"
        ),
        "classifier_tuning": tuning,
        "metrics": _metrics(probability, np.asarray(validation["type"])),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
