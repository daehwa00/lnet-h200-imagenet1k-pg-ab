from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Final, Literal, cast

from .pac_baseline_fairness_maximal import (
    BASELINES,
    CONFIRMATION_SEEDS,
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
from .pac_final_validation import UCR_SECONDS

DEFAULT_ROOT: Final = Path(".omx/results/pac-efp-compact-equal-search-20260719")
DEFAULT_BASELINE_ROOT: Final = Path(".omx/results/pac-baseline-fairness-maximal-20260714")
MODELS: Final = ("efp_tuned", "compact_h_only")
MODEL_DIMS: Final = (16, 32, 64)
MODE_COUNTS: Final = (8, 16, 32)
OPTIMIZER_TRIALS: Final = (2, 4, 6)
CAPACITY_GRID: Final = tuple(
    (model_dim, modes)
    for model_dim in MODEL_DIMS
    for modes in MODE_COUNTS
    if 2 * modes <= model_dim
)
STAGES: Final = ("stage1", "stage2")
COMPARISON_RTOL: Final = 1.0e-5
COMPARISON_ATOL: Final = 1.0e-8
COMPATIBLE_SELECTION_RUNNER_SHA256: Final = frozenset(
    {
        # Equal-search runner before Q2 reconstruction support was added.
        "09f62a8745c84278f513cbc91c63e2cf27a8485f297cd93fe661522751a65389",
        # Same equal-search paths plus Q2-only _public_models/_build_model_from_metadata support.
        "7d1b0f5f278bca808eb910569c20e9ed3a73f12cbe80efce22874698e0bad1af",
    }
)


def _audit_validation_rows(
    rows: list[dict[str, object]],
    *,
    stage: str,
    suite: Literal["ucr", "external"],
) -> list[dict[str, object]]:
    """Reject semantically invalid architecture-selection evidence."""
    expected_datasets = set(UCR_DATASETS if suite == "ucr" else EXTERNAL_DATASETS)
    expected_seeds = {SEARCH_SEED} if stage == "stage1" else set(CONFIRMATION_SEEDS)
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("job_key", ""))
        if not key or key in seen:
            message = f"{stage} contains a missing or duplicate job key: {key!r}"
            raise RuntimeError(message)
        seen.add(key)
        if (
            row.get("status") != "done"
            or row.get("stage") != stage
            or row.get("suite") != suite
            or row.get("dataset") not in expected_datasets
            or row.get("model") not in MODELS
            or row.get("evaluation_split") != "validation"
            or int(cast("int", row.get("split_seed", -1))) not in expected_seeds
            or int(cast("int", row.get("train_seed", -1))) not in expected_seeds
        ):
            message = f"{key} is not a {suite} TRAIN-derived validation result"
            raise RuntimeError(message)
        if (
            row.get("test_evaluated") is not False
            or row.get("official_test_accessed") is not False
        ):
            message = f"{key} accessed or did not audit the official TEST endpoint"
            raise RuntimeError(message)
        if row.get("code_sha256") not in COMPATIBLE_SELECTION_RUNNER_SHA256:
            message = f"{key} was produced by an unsealed selection runner"
            raise RuntimeError(message)
        score = row.get("selection_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            message = f"{key} has a non-finite selection score"
            raise RuntimeError(message)
    return rows


def default_lanes(count: int = 20) -> tuple[ResourceLane, ...]:
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


def _estimated_seconds(dataset: str, model_dim: int, modes: int, model: str) -> float:
    capacity_factor = max(0.25, (model_dim / 32.0) * (modes / 16.0))
    reader_factor = 1.65 if model == "compact_h_only" else 1.0
    return UCR_SECONDS[dataset] * capacity_factor * reader_factor


def _job(
    *,
    stage: Literal["stage1", "stage2"],
    dataset: str,
    model: str,
    capacity_index: int,
    model_dim: int,
    modes: int,
    trial: int,
    seed: int,
) -> FairnessJob:
    spec = confirmatory_trial_spec("pac_tf", trial)
    return FairnessJob(
        stage=stage,
        suite="ucr",
        dataset=dataset,
        model=model,
        width_tier=capacity_index,
        width=model_dim,
        trial=trial,
        split_seed=seed,
        train_seed=seed,
        epochs=100,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        evaluation_split="validation",
        estimated_seconds=_estimated_seconds(dataset, model_dim, modes, model),
        modes=modes,
    )


def stage1_jobs() -> list[FairnessJob]:
    return [
        _job(
            stage="stage1",
            dataset=dataset,
            model=model,
            capacity_index=capacity_index,
            model_dim=model_dim,
            modes=modes,
            trial=trial,
            seed=SEARCH_SEED,
        )
        for dataset in UCR_DATASETS
        for model in MODELS
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
        "schema": "pac_efp_compact_equal_search_stage1.v1",
        "models": list(MODELS),
        "datasets": list(UCR_DATASETS),
        "capacity_grid": [list(pair) for pair in CAPACITY_GRID],
        "optimizer_trials": list(OPTIMIZER_TRIALS),
        "evaluations_per_model_task": len(CAPACITY_GRID) * len(OPTIMIZER_TRIALS),
        "jobs": len(jobs),
        "search_seed": SEARCH_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "official_test_access": "forbidden; architecture selection uses TRAIN-derived validation",
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
    selection_path = root / "stage1" / "selection.json"
    if selection_path.exists():
        return cast("dict[str, object]", json.loads(selection_path.read_text(encoding="utf-8")))
    rows = _audit_validation_rows(
        _require_complete_stage(root, "stage1"),
        stage="stage1",
        suite="ucr",
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_key"]), []).append(row)
    expected_cells = {
        f"ucr:{dataset}:{model}" for dataset in UCR_DATASETS for model in MODELS
    }
    if set(grouped) != expected_cells:
        message = "UCR Stage 1 does not contain the exact candidate-task grid"
        raise RuntimeError(message)
    selected: dict[str, list[str]] = {}
    jobs: list[FairnessJob] = []
    expected = len(CAPACITY_GRID) * len(OPTIMIZER_TRIALS)
    expected_configs = {
        f"d{model_dim}-m{modes}-t{trial}"
        for model_dim, modes in CAPACITY_GRID
        for trial in OPTIMIZER_TRIALS
    }
    for cell_key, cell_rows in sorted(grouped.items()):
        if (
            len(cell_rows) != expected
            or {str(row["config_key"]) for row in cell_rows} != expected_configs
        ):
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
        "schema": "pac_efp_compact_equal_search_stage1_selection.v1",
        "source_rows": len(rows),
        "cells": len(grouped),
        "top_k": TOP_K,
        "selected": selected,
        "stage2_jobs": len(jobs),
        "official_test_access": "forbidden",
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(selection_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _average_tie_ranks(scores: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and math.isclose(
            ordered[stop][1],
            ordered[index][1],
            rel_tol=COMPARISON_RTOL,
            abs_tol=COMPARISON_ATOL,
        ):
            stop += 1
        rank = mean(range(index + 1, stop + 1))
        for model, _ in ordered[index:stop]:
            ranks[model] = rank
        index = stop
    return ranks


def _scores_tie(first: float, second: float) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=COMPARISON_RTOL,
        abs_tol=COMPARISON_ATOL,
    )


def _baseline_scores(baseline_root: Path) -> dict[str, dict[str, float]]:
    path = baseline_root / "stage2" / "selection.json"
    if not path.exists():
        message = f"baseline selection is missing: {path}"
        raise FileNotFoundError(message)
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = cast("dict[str, dict[str, object]]", payload["selected"])
    scores: dict[str, dict[str, float]] = {}
    for dataset in UCR_DATASETS:
        scores[dataset] = {
            model: float(selected[f"ucr:{dataset}:{model}"]["mean_selection_score"])
            for model in BASELINES
        }
    return scores


def _comparison_report(
    selected: dict[str, dict[str, object]],
    baseline_root: Path,
) -> dict[str, object]:
    candidate_scores = {
        model: {
            dataset: float(selected[f"ucr:{dataset}:{model}"]["mean_selection_score"])
            for dataset in UCR_DATASETS
        }
        for model in MODELS
    }
    pairwise = {"compact_wins": 0, "ties": 0, "efp_wins": 0}
    per_dataset: dict[str, dict[str, object]] = {}
    for dataset in UCR_DATASETS:
        efp_score = candidate_scores["efp_tuned"][dataset]
        compact_score = candidate_scores["compact_h_only"][dataset]
        delta = compact_score - efp_score
        if _scores_tie(compact_score, efp_score):
            pairwise["ties"] += 1
            winner = "tie"
        elif delta > 0.0:
            pairwise["compact_wins"] += 1
            winner = "compact_h_only"
        else:
            pairwise["efp_wins"] += 1
            winner = "efp_tuned"
        per_dataset[dataset] = {
            "efp_tuned": efp_score,
            "compact_h_only": compact_score,
            "compact_minus_efp": delta,
            "pairwise_winner": winner,
            "efp_config": selected[f"ucr:{dataset}:efp_tuned"]["config_key"],
            "compact_config": selected[f"ucr:{dataset}:compact_h_only"]["config_key"],
        }

    baselines = _baseline_scores(baseline_root)
    global_summary: dict[str, dict[str, object]] = {}
    for candidate in MODELS:
        rank_values: list[float] = []
        top_count = 0
        for dataset in UCR_DATASETS:
            scores = {candidate: candidate_scores[candidate][dataset], **baselines[dataset]}
            ranks = _average_tie_ranks(scores)
            rank_values.append(ranks[candidate])
            best = max(scores.values())
            top_count += int(_scores_tie(scores[candidate], best))
            per_dataset[dataset][f"{candidate}_global_rank"] = ranks[candidate]
            per_dataset[dataset][f"{candidate}_global_top"] = _scores_tie(scores[candidate], best)
        global_summary[candidate] = {
            "mean_validation_balanced_accuracy": mean(candidate_scores[candidate].values()),
            "mean_rank_vs_six_baselines": mean(rank_values),
            "global_top_count": top_count,
            "mean_parameters": mean(
                float(selected[f"ucr:{dataset}:{candidate}"]["params_trainable"])
                for dataset in UCR_DATASETS
            ),
        }
    champion = min(
        MODELS,
        key=lambda model: (
            -cast("int", global_summary[model]["global_top_count"]),
            cast("float", global_summary[model]["mean_rank_vs_six_baselines"]),
            -cast("float", global_summary[model]["mean_validation_balanced_accuracy"]),
            model,
        ),
    )
    return {
        "schema": "pac_efp_compact_equal_search_comparison.v1",
        "selection_metric": "TRAIN-derived validation balanced accuracy",
        "official_test_accessed": False,
        "pairwise": pairwise,
        "global_summary": global_summary,
        "provisional_champion": champion,
        "selection_rule": "global top count, then mean rank, then mean balanced accuracy",
        "tie_tolerance": {"rtol": COMPARISON_RTOL, "atol": COMPARISON_ATOL},
        "per_dataset": per_dataset,
    }


def comparison_from_frozen_selection(
    root: Path = DEFAULT_ROOT,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
) -> dict[str, object]:
    """Recompute only the validation comparison under the disclosed tie rule."""
    payload = json.loads((root / "stage2/selection.json").read_text(encoding="utf-8"))
    selected = cast("dict[str, dict[str, object]]", payload["selected"])
    return _comparison_report(selected, baseline_root)


def select_stage2(
    root: Path = DEFAULT_ROOT,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
) -> dict[str, object]:
    selection_path = root / "stage2" / "selection.json"
    if selection_path.exists():
        return cast("dict[str, object]", json.loads(selection_path.read_text(encoding="utf-8")))
    stage1 = _audit_validation_rows(
        _require_complete_stage(root, "stage1"),
        stage="stage1",
        suite="ucr",
    )
    stage2 = _audit_validation_rows(
        _require_complete_stage(root, "stage2"),
        stage="stage2",
        suite="ucr",
    )
    by_cell_config: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in stage1 + stage2:
        key = (str(row["cell_key"]), str(row["config_key"]))
        by_cell_config.setdefault(key, []).append(row)
    stage1_selection = json.loads((root / "stage1" / "selection.json").read_text(encoding="utf-8"))
    expected_cells = {
        f"ucr:{dataset}:{model}" for dataset in UCR_DATASETS for model in MODELS
    }
    if set(stage1_selection["selected"]) != expected_cells:
        message = "UCR Stage 1 selection does not contain the exact candidate-task grid"
        raise RuntimeError(message)
    selected: dict[str, dict[str, object]] = {}
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
        selected[cell_key] = {
            "config_key": config_key,
            "model_dim": base.width,
            "modes": base.modes,
            "trial": base.trial,
            "mean_selection_score": score,
            "selection_seeds": sorted(expected_seeds),
            "params_trainable": mean(float(row["params_trainable"]) for row in rows),
        }
    comparison = _comparison_report(selected, baseline_root)
    payload: dict[str, object] = {
        "schema": "pac_efp_compact_equal_search_stage2_selection.v1",
        "cells": len(selected),
        "selected": selected,
        "comparison": comparison,
        "official_test_accessed": False,
        "final_jobs": 0,
    }
    write_once(selection_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_once(
        root / "reports" / "comparison.json",
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    )
    return payload


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    full = campaign_status(root)
    return {"schema": "pac_efp_compact_equal_search_status.v1", **{s: full[s] for s in STAGES}}
