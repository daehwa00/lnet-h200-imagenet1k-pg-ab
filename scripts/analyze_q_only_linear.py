"""Compare Q-only linear results with the completed full-model campaign."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(item.read_text(encoding="utf-8"))
        for item in sorted((path / "final" / "completed").glob("*.json"))
    ]


def _cube(
    q_root: Path,
    reference_root: Path,
) -> tuple[list[str], list[int], np.ndarray, dict[str, list[int]]]:
    values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    parameters: dict[str, list[int]] = defaultdict(list)
    for row in _rows(q_root):
        values[(str(row["dataset"]), "q_only_linear")][int(row["train_seed"])] = float(
            row["balanced_accuracy"]
        )
        parameters["q_only_linear"].append(int(row["params_trainable"]))
    for row in _rows(reference_root):
        variant = str(row["model"]).removeprefix("wr_capacity_")
        if variant not in {"full", "energy_only"}:
            continue
        values[(str(row["dataset"]), variant)][int(row["train_seed"])] = float(
            row["balanced_accuracy"]
        )
        parameters[variant].append(int(row["params_trainable"]))
    tasks = sorted({task for task, _ in values})
    seeds = sorted({seed for active in values.values() for seed in active})
    variants = ("q_only_linear", "full", "energy_only")
    cube = np.empty((len(tasks), len(variants), len(seeds)))
    for task_index, task in enumerate(tasks):
        for variant_index, variant in enumerate(variants):
            cube[task_index, variant_index] = [
                values[(task, variant)][seed] for seed in seeds
            ]
    return tasks, seeds, cube, parameters


def _paired_bootstrap(
    cube: np.ndarray,
    left: int,
    right: int,
    *,
    draws: int,
) -> list[float]:
    rng = np.random.default_rng(20260724)
    task_count, _, seed_count = cube.shape
    result = np.empty(draws)
    for draw in range(draws):
        sampled_tasks = rng.integers(task_count, size=task_count)
        effects = np.empty(task_count)
        for slot, task_index in enumerate(sampled_tasks):
            sampled_seeds = rng.integers(seed_count, size=seed_count)
            means = cube[task_index][:, sampled_seeds].mean(axis=1)
            effects[slot] = means[left] - means[right]
        result[draw] = effects.mean()
    return np.quantile(result, (0.025, 0.975)).tolist()


def analyze(
    q_root: Path,
    reference_root: Path,
    *,
    draws: int = 20_000,
) -> dict[str, Any]:
    tasks, seeds, cube, parameters = _cube(q_root, reference_root)
    task_means = cube.mean(axis=2)
    variants = ("q_only_linear", "full", "energy_only")
    comparisons: dict[str, Any] = {}
    for reference_index, reference in ((1, "full"), (2, "energy_only")):
        effects = task_means[:, 0] - task_means[:, reference_index]
        comparisons[f"q_only_linear_minus_{reference}"] = {
            "mean_balanced_accuracy_gap": float(effects.mean()),
            "hierarchical_paired_bootstrap_95_ci": _paired_bootstrap(
                cube,
                0,
                reference_index,
                draws=draws,
            ),
            "task_wins": int((effects > 0).sum()),
            "task_ties": int((effects == 0).sum()),
            "task_losses": int((effects < 0).sum()),
        }
    payload: dict[str, Any] = {
        "schema": "pac_q_only_linear_analysis.v1",
        "tasks": tasks,
        "seeds": seeds,
        "mean_balanced_accuracy": {
            variant: float(task_means[:, index].mean())
            for index, variant in enumerate(variants)
        },
        "comparisons": comparisons,
        "parameter_count": {
            variant: {
                "minimum": min(active),
                "median": float(np.median(active)),
                "maximum": max(active),
            }
            for variant, active in parameters.items()
        },
    }
    output = q_root / "reports" / "q_only_linear_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("q_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--draws", type=int, default=20_000)
    args = parser.parse_args()
    payload = analyze(args.q_root, args.reference_root, draws=args.draws)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
