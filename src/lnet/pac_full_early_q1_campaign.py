from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Final, Literal, cast

from .pac_baseline_fairness_maximal import (
    CONFIRMATION_SEEDS,
    FINAL_SEEDS,
    SEARCH_SEED,
    TOP_K,
    FairnessJob,
    ResourceLane,
    _job_from_result,  # pyright: ignore[reportPrivateUsage]
    _rank_row,  # pyright: ignore[reportPrivateUsage]
    _require_complete_stage,  # pyright: ignore[reportPrivateUsage]
    _write_manifests,  # pyright: ignore[reportPrivateUsage]
    campaign_status,
)
from .pac_campaign_utils import write_once
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_efp16_final_campaign import EXTERNAL_DATASETS, UCR_DATASETS
from .pac_final_validation import EXTERNAL_SECONDS, UCR_SECONDS

DEFAULT_ROOT: Final = Path(".omx/results/pac-full-early-q1-tuning-20260718")
MODEL_NAME: Final = "full_early"
MODEL_DIMS: Final = (16, 32, 64)
MODE_COUNTS: Final = (8, 16, 32)
OPTIMIZER_TRIALS: Final = (2, 4, 6)
# The semi-orthogonal complex analysis frame requires 2M <= D.  These are all
# feasible pairs from the declared D and M ladders; invalid overcomplete pairs
# are not silently clipped to a smaller effective mode count.
CAPACITY_GRID: Final = tuple(
    (model_dim, modes)
    for model_dim in MODEL_DIMS
    for modes in MODE_COUNTS
    if 2 * modes <= model_dim
)
STAGES: Final = ("stage1", "stage2", "final")


def default_lanes(count: int = 12) -> tuple[ResourceLane, ...]:
    if count < 1:
        message = "lane count must be positive"
        raise ValueError(message)
    return tuple(
        ResourceLane(
            name=f"worker-{index:02d}",
            host="pro6000",
            gpu=0,
            lane=index,
            relative_speed=1.0,
        )
        for index in range(count)
    )


def _estimated_seconds(
    suite: Literal["ucr", "external"],
    dataset: str,
    model_dim: int,
    modes: int,
) -> float:
    base = UCR_SECONDS[dataset] if suite == "ucr" else EXTERNAL_SECONDS[dataset]
    capacity_factor = max(0.25, (model_dim / 32.0) * (modes / 16.0))
    return base * capacity_factor


def _job(
    *,
    stage: Literal["stage1", "stage2", "final"],
    suite: Literal["ucr", "external"],
    dataset: str,
    capacity_index: int,
    model_dim: int,
    modes: int,
    trial: int,
    split_seed: int,
    train_seed: int,
    evaluation_split: Literal["validation", "test"],
    epochs: int | None = None,
) -> FairnessJob:
    spec = confirmatory_trial_spec("pac_tf", trial)
    return FairnessJob(
        stage=stage,
        suite=suite,
        dataset=dataset,
        model=MODEL_NAME,
        width_tier=capacity_index,
        width=model_dim,
        trial=trial,
        split_seed=split_seed,
        train_seed=train_seed,
        epochs=(100 if suite == "ucr" else 60) if epochs is None else epochs,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        evaluation_split=evaluation_split,
        estimated_seconds=_estimated_seconds(suite, dataset, model_dim, modes),
        modes=modes,
    )


def stage1_jobs() -> list[FairnessJob]:
    return [
        _job(
            stage="stage1",
            suite=cast("Literal['ucr', 'external']", suite),
            dataset=dataset,
            capacity_index=capacity_index,
            model_dim=model_dim,
            modes=modes,
            trial=trial,
            split_seed=SEARCH_SEED,
            train_seed=SEARCH_SEED,
            evaluation_split="validation",
        )
        for suite, datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS))
        for dataset in datasets
        for capacity_index, (model_dim, modes) in enumerate(CAPACITY_GRID, start=1)
        for trial in OPTIMIZER_TRIALS
    ]


def enqueue_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    jobs = stage1_jobs()
    active_lanes = default_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "stage1", jobs, active_lanes)
    contract: dict[str, object] = {
        "schema": "pac_full_early_q1_stage1.v1",
        "model": MODEL_NAME,
        "datasets": {"ucr": list(UCR_DATASETS), "external": list(EXTERNAL_DATASETS)},
        "model_dims": list(MODEL_DIMS),
        "mode_counts": list(MODE_COUNTS),
        "optimizer_trials": list(OPTIMIZER_TRIALS),
        "evaluations_per_task": len(CAPACITY_GRID) * len(OPTIMIZER_TRIALS),
        "jobs": len(jobs),
        "search_seed": SEARCH_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "official_test_access": "forbidden until one configuration is frozen per task",
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "stage1" / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def select_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    rows = _require_complete_stage(root, "stage1")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_key"]), []).append(row)
    selected: dict[str, list[str]] = {}
    jobs: list[FairnessJob] = []
    expected = len(CAPACITY_GRID) * len(OPTIMIZER_TRIALS)
    for cell_key, cell_rows in sorted(grouped.items()):
        if len(cell_rows) != expected:
            message = f"{cell_key} has {len(cell_rows)} rows; expected {expected}"
            raise RuntimeError(message)
        top = sorted(cell_rows, key=_rank_row)[:TOP_K]
        selected[cell_key] = [str(row["config_key"]) for row in top]
        for row in top:
            base = _job_from_result(row)
            jobs.extend(
                replace(
                    base,
                    stage="stage2",
                    split_seed=seed,
                    train_seed=seed,
                    evaluation_split="validation",
                )
                for seed in CONFIRMATION_SEEDS
            )
    active_lanes = default_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "stage2", jobs, active_lanes)
    payload: dict[str, object] = {
        "schema": "pac_full_early_q1_stage1_selection.v1",
        "source_rows": len(rows),
        "cells": len(grouped),
        "top_k": TOP_K,
        "selected": selected,
        "stage2_jobs": len(jobs),
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "stage1" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def select_stage2(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    selection_path = root / "stage2" / "selection.json"
    if selection_path.exists():
        return cast("dict[str, object]", json.loads(selection_path.read_text(encoding="utf-8")))
    stage1 = _require_complete_stage(root, "stage1")
    stage2 = _require_complete_stage(root, "stage2")
    combined = stage1 + stage2
    by_cell_config: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in combined:
        key = (str(row["cell_key"]), str(row["config_key"]))
        by_cell_config.setdefault(key, []).append(row)
    stage1_selection = json.loads(
        (root / "stage1" / "selection.json").read_text(encoding="utf-8")
    )
    selected: dict[str, dict[str, object]] = {}
    final_jobs: list[FairnessJob] = []
    expected_seeds = {SEARCH_SEED, *CONFIRMATION_SEEDS}
    for cell_key, config_keys in sorted(stage1_selection["selected"].items()):
        candidates: list[tuple[float, str, list[dict[str, object]]]] = []
        for config_key in config_keys:
            rows = by_cell_config[(cell_key, config_key)]
            seeds = {cast("int", row["train_seed"]) for row in rows}
            if len(rows) != len(expected_seeds) or seeds != expected_seeds:
                message = (
                    f"{cell_key}/{config_key} has seeds {sorted(seeds)}; "
                    f"expected {sorted(expected_seeds)}"
                )
                raise RuntimeError(message)
            candidates.append(
                (
                    mean(cast("float", row["selection_score"]) for row in rows),
                    config_key,
                    rows,
                )
            )
        score, config_key, rows = min(candidates, key=lambda item: (-item[0], item[1]))
        base = _job_from_result(rows[0])
        best_epochs = [cast("int", row["best_epoch"]) for row in rows if row.get("best_epoch")]
        refit_epochs = max(1, round(median(best_epochs))) if best_epochs else base.epochs
        selected[cell_key] = {
            "config_key": config_key,
            "model_dim": base.width,
            "modes": base.modes,
            "trial": base.trial,
            "mean_selection_score": score,
            "selection_seeds": sorted(expected_seeds),
            "ucr_refit_epochs": refit_epochs if base.suite == "ucr" else None,
        }
        final_jobs.extend(
            replace(
                base,
                stage="final",
                split_seed=seed,
                train_seed=seed,
                epochs=refit_epochs if base.suite == "ucr" else base.epochs,
                evaluation_split="test",
            )
            for seed in FINAL_SEEDS
        )
    active_lanes = default_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "final", final_jobs, active_lanes)
    payload: dict[str, object] = {
        "schema": "pac_full_early_q1_stage2_selection.v1",
        "cells": len(selected),
        "selected": selected,
        "final_jobs": len(final_jobs),
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(selection_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    full = campaign_status(root)
    return {"schema": "pac_full_early_q1_status.v1", **{stage: full[stage] for stage in STAGES}}
