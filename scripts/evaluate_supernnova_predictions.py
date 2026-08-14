from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lnet.astronomy.metrics import classification_metrics


def _metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    return classification_metrics(
        torch.from_numpy(probability),
        torch.from_numpy(target),
    )


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Files are produced locally by the pinned local_gpu SuperNNova run.
    frame = pd.read_pickle(path).groupby("SNID").median().sort_index()  # noqa: S301
    probability = np.asarray(frame[[f"all_class{index}" for index in range(14)]])
    return (
        np.asarray(frame.index),
        probability,
        np.asarray(frame["target"], dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.predictions) != len(args.models):
        message = "predictions and models must have equal length"
        raise ValueError(message)

    loaded = [_load_prediction(path) for path in args.predictions]
    reference_ids, _, reference_target = loaded[0]
    for object_ids, _, target in loaded[1:]:
        if not np.array_equal(object_ids, reference_ids):
            message = "SuperNNova prediction object IDs differ across seeds"
            raise ValueError(message)
        if not np.array_equal(target, reference_target):
            message = "SuperNNova targets differ across seeds"
            raise ValueError(message)
    probabilities = [probability for _, probability, _ in loaded]
    parameter_counts: list[int] = []
    for path in args.models:
        state = torch.load(path, map_location="cpu", weights_only=True)
        parameter_counts.append(sum(int(value.numel()) for value in state.values()))
    payload = {
        "model": "SuperNNova 3.0.51 vanilla bidirectional LSTM",
        "metric_scope": "shared validation split, known 14 classes",
        "parameter_counts": parameter_counts,
        "per_seed": [
            _metrics(probability, reference_target) for probability in probabilities
        ],
        "ensemble": _metrics(np.mean(probabilities, axis=0), reference_target),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
