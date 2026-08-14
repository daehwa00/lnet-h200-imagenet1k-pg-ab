"""Analyze the completed UCR18 writer/reader ablation with paired bootstrap."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

VARIANTS = (
    "full",
    "one_scan_writer",
    "reader_lift_only",
    "reader_only",
    "pooled_real_only",
    "energy_only",
    "lag_only",
)
PREFIX = "wr_capacity_"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2
        position = end
    return ranks


def _load_cube(root: Path) -> tuple[list[str], list[int], np.ndarray, list[dict[str, Any]]]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "final" / "completed").glob("*.json"))
    ]
    expected = 18 * len(VARIANTS) * 10
    if len(rows) != expected:
        message = f"final campaign incomplete: found {len(rows)}, expected {expected}"
        raise RuntimeError(message)
    tasks = sorted({str(row["dataset"]) for row in rows})
    seeds = sorted({int(row["train_seed"]) for row in rows})
    cube = np.full((len(tasks), len(VARIANTS), len(seeds)), np.nan)
    task_index = {task: index for index, task in enumerate(tasks)}
    variant_index = {variant: index for index, variant in enumerate(VARIANTS)}
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    for row in rows:
        variant = str(row["model"]).removeprefix(PREFIX)
        cube[
            task_index[str(row["dataset"])],
            variant_index[variant],
            seed_index[int(row["train_seed"])],
        ] = float(row["balanced_accuracy"])
    if not np.isfinite(cube).all():
        message = "result cube has missing or non-finite cells"
        raise RuntimeError(message)
    return tasks, seeds, cube, rows


def analyze(root: Path, *, bootstrap_draws: int = 20_000) -> dict[str, Any]:
    tasks, seeds, cube, rows = _load_cube(root)
    task_variant_means = cube.mean(axis=2)
    task_ranks = np.stack([_average_ranks(values) for values in task_variant_means])
    full_index = VARIANTS.index("full")
    control_index = VARIANTS.index("one_scan_writer")
    task_rank_gaps = task_ranks[:, control_index] - task_ranks[:, full_index]
    task_accuracy_gaps = (
        task_variant_means[:, full_index] - task_variant_means[:, control_index]
    )

    rng = np.random.default_rng(20260724)
    rank_bootstrap = np.empty(bootstrap_draws)
    accuracy_bootstrap = np.empty(bootstrap_draws)
    task_count, _, seed_count = cube.shape
    for draw in range(bootstrap_draws):
        sampled_tasks = rng.integers(task_count, size=task_count)
        draw_rank_gaps = np.empty(task_count)
        draw_accuracy_gaps = np.empty(task_count)
        for slot, task_index in enumerate(sampled_tasks):
            sampled_seeds = rng.integers(seed_count, size=seed_count)
            # Index the task first so seed resampling stays on the final axis.
            # Combining both operations in one NumPy expression moves the
            # advanced-indexed seed axis ahead of the variant axis.
            means = cube[task_index][:, sampled_seeds].mean(axis=1)
            ranks = _average_ranks(means)
            draw_rank_gaps[slot] = ranks[control_index] - ranks[full_index]
            draw_accuracy_gaps[slot] = means[full_index] - means[control_index]
        rank_bootstrap[draw] = draw_rank_gaps.mean()
        accuracy_bootstrap[draw] = draw_accuracy_gaps.mean()

    removal_cases: list[dict[str, Any]] = []
    indices = range(task_count)
    for remove_count in (0, 1, 2):
        for removed in itertools.combinations(indices, remove_count):
            retained = [index for index in indices if index not in removed]
            removal_cases.append(
                {
                    "removed": [tasks[index] for index in removed],
                    "mean_rank_gap": float(task_rank_gaps[retained].mean()),
                }
            )
    worst_removal = min(removal_cases, key=lambda item: item["mean_rank_gap"])

    parameter_errors: list[float] = []
    grouped_params: dict[tuple[str, int], dict[str, int]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["train_seed"]))
        variant = str(row["model"]).removeprefix(PREFIX)
        grouped_params.setdefault(key, {})[variant] = int(row["params_trainable"])
    for values in grouped_params.values():
        full_parameters = values["full"]
        parameter_errors.extend(
            (parameters - full_parameters) / full_parameters
            for parameters in values.values()
        )

    rank_ci = np.quantile(rank_bootstrap, (0.025, 0.975))
    accuracy_ci = np.quantile(accuracy_bootstrap, (0.025, 0.975))
    mean_rank_gap = float(task_rank_gaps.mean())
    payload: dict[str, Any] = {
        "schema": "pac_writer_reader_capacity_analysis.v1",
        "tasks": tasks,
        "seeds": seeds,
        "variants": list(VARIANTS),
        "primary": {
            "contrast": "full minus one_scan_writer",
            "mean_rank_gap_control_minus_full": mean_rank_gap,
            "hierarchical_paired_bootstrap_95_ci": rank_ci.tolist(),
            "mean_balanced_accuracy_gap_full_minus_control": float(
                task_accuracy_gaps.mean()
            ),
            "balanced_accuracy_gap_95_ci": accuracy_ci.tolist(),
            "full_better_task_count": int((task_accuracy_gaps > 0).sum()),
            "tie_task_count": int((task_accuracy_gaps == 0).sum()),
            "full_worse_task_count": int((task_accuracy_gaps < 0).sum()),
            "requested_success_criterion": bool(
                mean_rank_gap >= 0.5 and rank_ci[0] > 0
            ),
        },
        "robustness": {
            "worst_leave_up_to_two_tasks_out": worst_removal,
            "not_dependent_on_one_or_two_tasks": bool(
                worst_removal["mean_rank_gap"] > 0
            ),
        },
        "capacity_match": {
            "maximum_absolute_relative_parameter_error": max(
                abs(value) for value in parameter_errors
            ),
            "within_three_percent": bool(
                max(abs(value) for value in parameter_errors) <= 0.03
            ),
        },
        "mean_balanced_accuracy": {
            variant: float(task_variant_means[:, index].mean())
            for index, variant in enumerate(VARIANTS)
        },
        "mean_rank": {
            variant: float(task_ranks[:, index].mean())
            for index, variant in enumerate(VARIANTS)
        },
        "per_task": {
            task: {
                "rank_gap_control_minus_full": float(task_rank_gaps[index]),
                "balanced_accuracy_gap_full_minus_control": float(
                    task_accuracy_gaps[index]
                ),
                "balanced_accuracy": {
                    variant: float(task_variant_means[index, variant_index])
                    for variant_index, variant in enumerate(VARIANTS)
                },
            }
            for index, task in enumerate(tasks)
        },
        "systems_note": (
            "Latency and peak-memory rows were collected under concurrent GPU load; "
            "use an exclusive ABBA/BAAB rerun for inferential systems comparisons."
        ),
    }
    report_path = root / "reports" / "writer_reader_capacity_analysis.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()
    payload = analyze(args.root, bootstrap_draws=args.bootstrap_draws)
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
