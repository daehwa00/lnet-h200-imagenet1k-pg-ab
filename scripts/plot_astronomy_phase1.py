from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import pyplot as plt

MODEL_ORDER = (
    "alphabet_physical",
    "alphabet_token",
    "delta_time_gru",
    "gru_d",
    "statistical_rf",
)
MODEL_LABELS = (
    "ALPHABET physical",
    "ALPHABET token",
    "Δt-GRU",
    "GRU-D-like",
    "RF",
)


def _metric(
    models: dict[str, dict[str, object]],
    model: str,
    metric: str,
    statistic: str,
) -> float:
    metrics = cast("dict[str, dict[str, float]]", models[model]["metrics"])
    return metrics[metric][statistic]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = cast("dict[str, object]", json.loads(args.summary.read_text()))
    models = cast("dict[str, dict[str, object]]", summary["models"])
    throughput = cast("list[dict[str, object]]", summary["throughput"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for axis, metric, title in (
        (axes[0], "weighted_log_loss_known14", "Known-14 weighted log-loss"),
        (axes[1], "balanced_accuracy", "Balanced accuracy"),
    ):
        means = [_metric(models, model, metric, "mean") for model in MODEL_ORDER]
        errors = [
            _metric(models, model, metric, "sample_std") for model in MODEL_ORDER
        ]
        axis.bar(MODEL_LABELS, means, yerr=errors, capsize=4)
        axis.tick_params(axis="x", rotation=30)
        axis.set_title(title)

    gpu_rows = [row for row in throughput if row["device"] == "cuda"]
    axes[2].bar(
        [
            f"{row['model']}:{row['lag_mode']}"
            for row in gpu_rows
        ],
        [float(cast("float", row["objects_per_second"])) for row in gpu_rows],
    )
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylabel("Objects / second")
    axes[2].set_title("GPU inference throughput")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
