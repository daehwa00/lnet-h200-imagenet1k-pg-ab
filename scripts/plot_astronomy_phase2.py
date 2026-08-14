from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
from matplotlib import pyplot as plt

JsonRow = dict[str, object]


def _rows(payload: dict[str, object], key: str) -> list[JsonRow]:
    return cast("list[JsonRow]", payload[key])


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"expected numeric result, got {value!r}"
        raise TypeError(message)
    return float(value)


def _mean_std(
    rows: list[JsonRow],
    *,
    group_key: str,
    group_value: str,
    x_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row[group_key] == group_value]
    x_values = np.asarray(sorted({_number(row[x_key]) for row in selected}))
    values = np.asarray(
        [
            [_number(row["balanced_accuracy"]) for row in selected if row[x_key] == x]
            for x in x_values
        ]
    )
    return x_values, values.mean(axis=1), values.std(axis=1)


def _plot_phase2_curves(result_dir: Path, figure_dir: Path) -> None:
    payload = cast(
        "dict[str, object]",
        json.loads((result_dir / "phase2-curves.json").read_text()),
    )
    specifications = (
        ("early_classification", "days", "Days after trigger", "early-classification.png"),
        ("seasonal_gap", "gap_days", "Inserted gap (days)", "seasonal-gap.png"),
    )
    for key, x_key, x_label, filename in specifications:
        figure, axis = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
        for model, color in (("alphabet", "#2b6cb0"), ("gru", "#c05621")):
            x_value, mean, deviation = _mean_std(
                _rows(payload, key),
                group_key="model",
                group_value=model,
                x_key=x_key,
            )
            axis.plot(x_value, mean, marker="o", label=model.upper(), color=color)
            axis.fill_between(
                x_value,
                mean - deviation,
                mean + deviation,
                color=color,
                alpha=0.18,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Balanced accuracy")
        axis.set_ylim(0.25, 1.0)
        axis.legend(frameon=False)
        figure.savefig(figure_dir / filename, dpi=220)
        plt.close(figure)


def _plot_pole_periods(result_dir: Path, figure_dir: Path) -> None:
    attributed: list[float] = []
    lomb_scargle: list[float] = []
    for path in sorted(result_dir.glob("pole-audit-seed*.json")):
        payload = cast("dict[str, object]", json.loads(path.read_text()))
        for row in _rows(payload, "objects"):
            if int(cast("int", row["target"])) != 2:
                continue
            attributed.append(_number(row["attributed_period_days"]))
            lomb_scargle.append(_number(row["lomb_scargle_period_days"]))
    figure, axis = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    axis.scatter(lomb_scargle, attributed, s=14, alpha=0.55, color="#2b6cb0")
    axis.plot([0.2, 10.0], [0.2, 10.0], linestyle="--", color="#444444")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.18, 1.1)
    axis.set_xlabel("Lomb-Scargle period (days)")
    axis.set_ylabel("Top-attributed pole period (days)")
    figure.savefig(figure_dir / "rr-pole-vs-lomb-scargle.png", dpi=220)
    plt.close(figure)


def _plot_interventions(result_dir: Path, figure_dir: Path) -> None:
    paths = sorted(result_dir.glob("pole-interventions-seed*.json"))
    if not paths:
        return
    rows: list[JsonRow] = []
    full_scores: list[float] = []
    for path in paths:
        payload = cast("dict[str, object]", json.loads(path.read_text()))
        full_scores.append(_number(payload["full_test_balanced_accuracy"]))
        rows.extend(_rows(payload, "interventions"))
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    for intervention, axis, metric, ylabel in (
        ("neutralize", axes[0], "balanced_accuracy", "Balanced accuracy"),
        ("retain", axes[1], "prediction_agreement", "Prediction agreement"),
    ):
        for strategy, color in (
            ("top", "#c53030"),
            ("random", "#718096"),
            ("bottom", "#2b6cb0"),
        ):
            selected = [
                row
                for row in rows
                if row["intervention"] == intervention and row["strategy"] == strategy
            ]
            x_values = np.asarray(sorted({int(cast("int", row["k"])) for row in selected}))
            means = [
                np.mean(
                    [
                        _number(row[metric])
                        for row in selected
                        if int(cast("int", row["k"])) == x_value
                    ]
                )
                for x_value in x_values
            ]
            axis.plot(x_values, means, marker="o", label=strategy, color=color)
        if intervention == "neutralize":
            axis.axhline(np.mean(full_scores), linestyle="--", color="#222222")
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Number of poles")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)
    figure.savefig(figure_dir / "pole-removal-retention.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_phase2_curves(args.result_dir, args.figure_dir)
    _plot_pole_periods(args.result_dir, args.figure_dir)
    _plot_interventions(args.result_dir, args.figure_dir)


if __name__ == "__main__":
    main()
