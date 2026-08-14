from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from .pac_revised_visuals import baseline_scores

DEFAULT_ROOT: Final = Path(".omx/results/pac-35-dataset-comparison-20260712")
UCR_EXTRA_ROOT: Final = Path(".omx/results/pac-ucr-s4-minirocket-20260712")
UCR_EXTRA_UNTOUCHED_ROOT: Final = Path(
    ".omx/results/pac-ucr-s4-minirocket-untouched-validation-20260712"
)
UCR_REVISED_BUDGET_RESULT: Final = Path(
    ".omx/results/pac-revised-budget-matched-baselines-20260712/results/"
    "low_data_recommended_real.csv"
)
LOCAL_EXTERNAL_JOBS: Final = Path(".omx/results/pac-selected-d64m16-external-20260711/jobs")
MODEL_LABELS: Final = {
    "revised_pac": "Revised PAC",
    "pac_tf": "Canonical PAC",
    "pac": "Canonical PAC",
    "cnn1d": "CNN1D",
    "tcn": "TCN",
    "transformer": "Transformer",
    "mamba": "Mamba",
    "gru": "GRU",
    "lstm": "LSTM",
    "s4": "S4",
    "s4d": "S4D-Lin",
    "minirocket": "MiniRocket",
    "inception_time": "InceptionTime",
}
EXTERNAL_BASELINES: Final = frozenset(
    {
        "cnn1d",
        "tcn",
        "transformer",
        "mamba",
        "gru",
        "lstm",
        "s4d",
        "minirocket",
        "inception_time",
    }
)
EXCLUDED_COMPARISON_MODELS: Final = frozenset({"minirocket", "inception_time"})
COMPARISON_MODELS: Final = (EXTERNAL_BASELINES | {"revised_pac"}) - EXCLUDED_COMPARISON_MODELS


def generate_comparison(root: Path = DEFAULT_ROOT) -> None:
    ucr = baseline_scores()
    _merge_ucr_revised_budget_scores(ucr)
    _merge_ucr_extra_scores(ucr)
    external, metrics, directions, runs = _external_scores(root)
    _retain_comparison_models(ucr, external)
    output = root / "reports"
    figures = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    datasets = {**ucr, **external}
    direction = dict.fromkeys(ucr, "max") | directions
    report = _report(ucr, external, metrics, runs, direction)
    (output / "comparison_35_datasets.md").write_text(report, encoding="utf-8")
    _plot_ranks(figures / "comparison_35_dataset_ranks", datasets, direction)


def _retain_comparison_models(*collections: dict[str, dict[str, float]]) -> None:
    """Restrict generated comparisons without deleting collected result artifacts."""
    for collection in collections:
        for scores in collection.values():
            for model in tuple(scores):
                if model not in COMPARISON_MODELS:
                    scores.pop(model)


def _external_scores(
    root: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, str], dict[str, str], dict[tuple[str, str], int]]:
    baseline_paths = [root / "input" / "baseline_external_partial.csv"]
    baseline_paths.extend(sorted(LOCAL_EXTERNAL_JOBS.glob("*/results/external_comparisons.csv")))
    sources = (
        (baseline_paths, None),
        ([root / "input" / "revised_external.csv"], "revised_pac"),
    )
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    metrics: dict[str, str] = {}
    directions: dict[str, str] = {}
    for paths, model_override in sources:
        unique: dict[tuple[str, str, str], dict[str, str]] = {}
        for path in paths:
            for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
                if row.get("status") == "done":
                    unique[(row["dataset"], row["model"], row["seed"])] = row
        for row in unique.values():
            if row.get("status") != "done":
                continue
            dataset = row["dataset"]
            model = model_override or ("pac_tf" if row["model"] == "pac" else row["model"])
            if model_override is None and model not in EXTERNAL_BASELINES:
                continue
            metric, metric_direction = _primary_metric(dataset, row["objective"])
            values[(dataset, model)].append(float(row[metric]))
            metrics[dataset] = metric
            directions[dataset] = metric_direction
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    runs: dict[tuple[str, str], int] = {}
    for (dataset, model), observations in values.items():
        scores[dataset][model] = mean(observations)
        runs[(dataset, model)] = len(observations)
    return dict(scores), metrics, directions, runs


def _merge_ucr_extra_scores(scores: dict[str, dict[str, float]]) -> None:
    selection_path = UCR_EXTRA_ROOT / "reports" / "selection.json"
    if not selection_path.exists():
        return
    import json  # noqa: PLC0415 - optional result integration

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    paths = (
        UCR_EXTRA_ROOT / "results" / "ucr_s4_minirocket.csv",
        UCR_EXTRA_UNTOUCHED_ROOT / "results" / "ucr_s4_minirocket.csv",
    )
    for path in paths:
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
            family = row.get("family", "")
            if (
                row.get("status") != "done"
                or row.get("stage") != "validation"
                or family not in selection
                or int(row["trial"]) != int(selection[family]["trial"])
            ):
                continue
            values[(row["dataset"], family)][int(row["seed"])] = float(row["balanced_accuracy"])
    for (dataset, family), observations in values.items():
        scores[dataset][family] = mean(observations.values())


def _merge_ucr_revised_budget_scores(scores: dict[str, dict[str, float]]) -> None:
    """Replace historical baseline cells with the Revised-budget campaign."""
    values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in csv.DictReader(UCR_REVISED_BUDGET_RESULT.open(newline="", encoding="utf-8")):
        if row.get("status") != "done":
            continue
        values[(row["dataset_or_task"], row["model"])][int(row["seed"])] = float(
            row["validation_balanced_accuracy"]
        )
    for (dataset, model), observations in values.items():
        if len(observations) != 5:
            message = f"incomplete Revised-budget UCR cell: {dataset}/{model}"
            raise ValueError(message)
        scores[dataset][model] = mean(observations.values())


def _primary_metric(dataset: str, objective: str) -> tuple[str, str]:
    if objective == "forecasting":
        return "mse", "min"
    if objective == "multiclass":
        return "accuracy", "max"
    if dataset == "audioset-balanced":
        return "macro_auprc", "max"
    return "macro_auroc", "max"


def _report(
    ucr: dict[str, dict[str, float]],
    external: dict[str, dict[str, float]],
    metrics: dict[str, str],
    runs: dict[tuple[str, str], int],
    directions: dict[str, str],
) -> str:
    all_scores = {**ucr, **external}
    all_directions = dict.fromkeys(ucr, "max") | directions
    models = sorted({model for scores in all_scores.values() for model in scores})
    wins: Counter[str] = Counter()
    rank_values: dict[str, list[float]] = defaultdict(list)
    for dataset, scores in all_scores.items():
        ranks = _average_tie_ranks(scores, all_directions[dataset])
        top = max(scores.values()) if all_directions[dataset] == "max" else min(scores.values())
        for model, score in scores.items():
            rank_values[model].append(ranks[model])
            if np.isclose(score, top):
                wins[model] += 1
    scope_sentence = "UCR and external raw metrics are not averaged together. "
    scope_sentence += (
        "Overall comparisons use per-task average-tie ranks; ties count as wins for every "
        "tied model."
    )
    scope_sentence += " MiniRocket and InceptionTime are excluded from this comparison set."
    rows = [
        "# PAC 35-Task Comparison",
        "",
        scope_sentence,
        "",
        "## Current Overall Ranking",
        "",
        "| Model | Coverage | Top-ranked tasks | Mean rank |",
        "|---|---:|---:|---:|",
    ]
    ranking = sorted(
        models,
        key=lambda model: (mean(rank_values[model]), -wins[model], -len(rank_values[model])),
    )
    rows.extend(
        (
            f"| {MODEL_LABELS.get(model, model)} | {len(rank_values[model])}/35 | "
            f"{wins[model]} | {mean(rank_values[model]):.2f} |"
        )
        for model in ranking
    )
    rows.extend(_ranking_table("UCR 18 ranking", ucr, dict.fromkeys(ucr, "max")))
    rows.extend(_ranking_table("External 17 ranking", external, directions))
    rows.extend(_dataset_table("UCR 18: validation balanced accuracy", ucr, {}, {}))
    rows.extend(
        _dataset_table("External 17: task-specific primary metric", external, metrics, runs)
    )
    revised_runs = sum(runs.get((dataset, "revised_pac"), 0) for dataset in external)
    included_external_baselines = EXTERNAL_BASELINES - EXCLUDED_COMPARISON_MODELS
    expected_baseline_runs = len(external) * len(included_external_baselines) * 3
    baseline_runs = sum(
        count for (_, model), count in runs.items() if model in included_external_baselines
    )
    expected_cells = len(all_scores) * len(COMPARISON_MODELS)
    rows.extend(
        [
            "",
            "## Status",
            "",
            f"- Revised PAC external runs: {revised_runs}/51",
            f"- Baseline external runs: {baseline_runs}/{expected_baseline_runs}",
            f"- Missing baseline seed-runs: {expected_baseline_runs - baseline_runs}",
            f"- All {expected_cells} selected task/model cells are complete; no pending cell "
            "enters the ranks.",
        ]
    )
    return "\n".join(rows) + "\n"


def _ranking_table(
    title: str,
    scores: dict[str, dict[str, float]],
    directions: dict[str, str],
) -> list[str]:
    rank_values: dict[str, list[float]] = defaultdict(list)
    wins: Counter[str] = Counter()
    for dataset, observations in scores.items():
        top = (
            max(observations.values())
            if directions[dataset] == "max"
            else min(observations.values())
        )
        ranks = _average_tie_ranks(observations, directions[dataset])
        for model, score in observations.items():
            rank_values[model].append(ranks[model])
            if np.isclose(score, top):
                wins[model] += 1
    ranking = sorted(rank_values, key=lambda model: (mean(rank_values[model]), -wins[model]))
    lines = [
        "",
        f"## {title}",
        "",
        "| Model | Mean rank | First-place datasets |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {MODEL_LABELS.get(model, model)} | {mean(rank_values[model]):.2f} | {wins[model]} |"
        for model in ranking
    )
    return lines


def _dataset_table(
    title: str,
    scores: dict[str, dict[str, float]],
    metrics: dict[str, str],
    runs: dict[tuple[str, str], int],
) -> list[str]:
    models = sorted({model for values in scores.values() for model in values})
    lines = ["", f"## {title}", ""]
    headers = ["Dataset", "Metric", *(MODEL_LABELS.get(model, model) for model in models)]
    lines.extend(("| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)))
    for dataset in sorted(scores):
        cells = [dataset, metrics.get(dataset, "balanced_accuracy")]
        for model in models:
            if model not in scores[dataset]:
                cells.append("pending")
                continue
            suffix = f" ({runs[(dataset, model)]}/3)" if runs else ""
            cells.append(f"{scores[dataset][model]:.4f}{suffix}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _plot_ranks(
    path: Path, scores: dict[str, dict[str, float]], directions: dict[str, str]
) -> None:
    datasets = sorted(scores)
    models = sorted({model for values in scores.values() for model in values})
    matrix = np.full((len(models), len(datasets)), np.nan)
    for column, dataset in enumerate(datasets):
        values = scores[dataset]
        for row, model in enumerate(models):
            if model not in values:
                continue
            matrix[row, column] = _average_tie_ranks(values, directions[dataset])[model]
    figure, axis = plt.subplots(figsize=(19, 7))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=1)
    axis.set_xticks(range(len(datasets)), labels=datasets, rotation=55, ha="right")
    axis.set_yticks(range(len(models)), labels=[MODEL_LABELS.get(model, model) for model in models])
    axis.set_title("Per-task average-tie rank across 35 benchmark tasks")
    figure.colorbar(image, ax=axis, fraction=0.02, pad=0.01, label="Rank (lower is better)")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _average_tie_ranks(scores: dict[str, float], direction: str) -> dict[str, float]:
    """Assign tied observations the mean of their one-indexed rank positions."""
    ordered = sorted(scores, key=scores.get, reverse=direction == "max")
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and np.isclose(scores[ordered[stop]], scores[ordered[start]]):
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for model in ordered[start:stop]:
            ranks[model] = average_rank
        start = stop
    return ranks


if __name__ == "__main__":
    generate_comparison()
