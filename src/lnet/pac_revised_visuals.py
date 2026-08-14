from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

DEVELOPMENT_RESULT: Final[Path] = Path(
    ".omx/results/pac-tf-revised-untied-candidate-20260711/results/revised_candidate.csv"
)
UNTOUCHED_RESULT: Final[Path] = Path(
    ".omx/results/pac-tf-revised-untouched-20260712/results/revised_candidate.csv"
)
BASELINE_RESULT: Final[Path] = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/results/low_data_recommended_real.csv"
)
DEVELOPMENT_BASELINE_RESULT: Final[Path] = (
    Path(".omx/results/pac-tf-revised-development-baselines-20260712/results")
    / "low_data_recommended_real.csv"
)
BASELINE_SELECTION: Final[Path] = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    ".omx/results/pac-tf-revised-18-dataset-comparison-20260712"
)


def generate_revised_comparison(output: Path = DEFAULT_OUTPUT) -> None:
    paired = _paired_scores((DEVELOPMENT_RESULT, UNTOUCHED_RESULT))
    baselines = _baseline_scores()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    _write_tables(output / "revised_18_dataset_comparison.md", paired, baselines)
    _plot_paired_heatmap(figures / "revised_18_dataset_heatmap", paired)
    _plot_baseline_heatmap(figures / "baseline_18_dataset_heatmap", baselines)


def _paired_scores(paths: tuple[Path, ...]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for row in csv.DictReader(path.open(newline="")):
            if row["status"] != "done":
                continue
            model = "Canonical PAC" if row["intervention"] == "reference" else "Revised PAC"
            values[row["dataset_or_task"]][model].append(
                float(row["validation_balanced_accuracy"])
            )
    return {
        dataset: {model: mean(scores) for model, scores in models.items()}
        for dataset, models in values.items()
    }


def _baseline_scores() -> dict[str, dict[str, float]]:
    selection = json.loads(BASELINE_SELECTION.read_text(encoding="utf-8"))["selected_trials"]
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(BASELINE_RESULT.open(newline="")):
        family = row["baseline_family"]
        if (
            row["status"] != "done"
            or row["evaluation_collection"] != "unseen_final_validation"
            or not family
            or not row["validation_trial"]
            or int(row["validation_trial"]) != int(selection[family]["trial"])
        ):
            continue
        values[row["dataset_or_task"]][family].append(
            float(row["validation_balanced_accuracy"])
        )
    for row in csv.DictReader(DEVELOPMENT_BASELINE_RESULT.open(newline="")):
        if row["status"] == "done":
            values[row["dataset_or_task"]][row["model"]].append(
                float(row["validation_balanced_accuracy"])
            )
    paired = _paired_scores((DEVELOPMENT_RESULT, UNTOUCHED_RESULT))
    for dataset, scores in paired.items():
        values[dataset]["pac_tf"] = [scores["Canonical PAC"]]
        values[dataset]["revised_pac"] = [scores["Revised PAC"]]
    return {
        dataset: {model: mean(scores) for model, scores in models.items()}
        for dataset, models in values.items()
    }


def baseline_scores() -> dict[str, dict[str, float]]:
    """Return the validated 18-dataset baseline score matrix."""
    return _baseline_scores()


def _write_tables(
    path: Path,
    paired: dict[str, dict[str, float]],
    baselines: dict[str, dict[str, float]],
) -> None:
    rows = [
        "# Revised PAC Comparison",
        "",
        "All values are mean validation balanced accuracy. Official TEST is not read.",
        "",
        "## Eighteen-dataset paired comparison",
        "",
        "| Dataset | Canonical PAC | Revised PAC | Delta |",
        "|---|---:|---:|---:|",
    ]
    for dataset in sorted(paired):
        canonical = paired[dataset]["Canonical PAC"]
        revised = paired[dataset]["Revised PAC"]
        rows.append(
            f"| {dataset} | {canonical:.4f} | **{revised:.4f}** | {revised - canonical:+.4f} |"
        )
    rows.extend(
        [
            "",
            "## Eighteen-dataset baseline comparison",
            "",
            "| Model | Mean balanced accuracy |",
            "|---|---:|",
        ]
    )
    models = sorted({model for scores in baselines.values() for model in scores})
    ranking = sorted(
        ((model, mean(scores[model] for scores in baselines.values())) for model in models),
        key=lambda item: item[1],
        reverse=True,
    )
    for model, score in ranking:
        label = "Revised PAC" if model == "revised_pac" else model.upper()
        rows.append(f"| {label} | {score:.4f} |")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _plot_paired_heatmap(path: Path, scores: dict[str, dict[str, float]]) -> None:
    datasets = sorted(scores)
    models = ("Canonical PAC", "Revised PAC")
    matrix = np.asarray(
        [[scores[dataset][model] for dataset in datasets] for model in models]
    )
    figure, axis = plt.subplots(figsize=(12.5, 2.8))
    image = axis.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0.0, vmax=1.0)
    _label_heatmap(axis, matrix, datasets, ["Canonical PAC", "Revised PAC"])
    axis.set_title("18-dataset TRAIN-derived validation balanced accuracy")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_baseline_heatmap(path: Path, scores: dict[str, dict[str, float]]) -> None:
    datasets = sorted(scores)
    models = sorted({model for values in scores.values() for model in values})
    matrix = np.asarray([[scores[dataset][model] for dataset in datasets] for model in models])
    labels = ["Revised PAC" if model == "revised_pac" else model.upper() for model in models]
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    image = axis.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0.0, vmax=1.0)
    _label_heatmap(axis, matrix, datasets, labels)
    axis.set_title("18-dataset baseline validation balanced accuracy")
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    figure.tight_layout()
    _save_figure(figure, path)


def _label_heatmap(
    axis: Axes, matrix: np.ndarray, datasets: list[str], models: list[str]
) -> None:
    axis.set_xticks(range(len(datasets)), labels=datasets, rotation=45, ha="right")
    axis.set_yticks(range(len(models)), labels=models)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)


def _save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    generate_revised_comparison()
