"""Sealed 30-task search and Q1 TEST campaign for learned two-tap ALPHABET."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Final, Literal, cast

from .pac_campaign_utils import canonical_json_sha256, file_sha256, write_once
from .pac_baseline_fairness_maximal import (
    BASELINES,
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
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_efp16_final_campaign import EXTERNAL_DATASETS, UCR_DATASETS
from .pac_final_validation import EXTERNAL_SECONDS, UCR_SECONDS

DEFAULT_ROOT: Final = Path(".omx/results/pac-two-tap-q1-final-20260720")
DEFAULT_BASELINE_ROOT: Final = Path(".omx/results/pac-baseline-fairness-maximal-20260714")
CANDIDATE: Final = "two_tap_h_only"
MODEL_DIMS: Final = (16, 32, 64)
MODE_COUNTS: Final = (8, 16, 32)
OPTIMIZER_TRIALS: Final = (2, 4, 6)
CAPACITY_GRID: Final = tuple(
    (model_dim, modes)
    for model_dim in MODEL_DIMS
    for modes in MODE_COUNTS
    if 2 * modes <= model_dim
)
STAGES: Final = ("stage1", "stage2", "final")
TIE_RTOL: Final = 1.0e-5
TIE_ATOL: Final = 1.0e-8


def _safe_result_name(job_key: str) -> str:
    return job_key.replace(":", "__").replace("/", "_") + ".json"


def default_lanes(count: int = 24) -> tuple[ResourceLane, ...]:
    if count < 1:
        raise ValueError("lane count must be positive")
    return tuple(
        ResourceLane(f"worker-{index:02d}", "pro6000", 0, index, 1.0) for index in range(count)
    )


def _datasets(suite: Literal["ucr", "external"]) -> tuple[str, ...]:
    return UCR_DATASETS if suite == "ucr" else EXTERNAL_DATASETS


def _estimated_seconds(
    suite: Literal["ucr", "external"],
    dataset: str,
    model_dim: int,
    modes: int,
) -> float:
    base = UCR_SECONDS[dataset] if suite == "ucr" else EXTERNAL_SECONDS[dataset]
    capacity = max(0.25, (model_dim / 32.0) * (modes / 16.0))
    return base * capacity * 1.65


def _job(
    *,
    stage: Literal["stage1", "stage2"],
    suite: Literal["ucr", "external"],
    dataset: str,
    capacity_index: int,
    model_dim: int,
    modes: int,
    trial: int,
    seed: int,
) -> FairnessJob:
    recipe = confirmatory_trial_spec("pac_tf", trial)
    return FairnessJob(
        stage=stage,
        suite=suite,
        dataset=dataset,
        model=CANDIDATE,
        width_tier=capacity_index,
        width=model_dim,
        trial=trial,
        split_seed=seed,
        train_seed=seed,
        epochs=100 if suite == "ucr" else 60,
        batch_size=recipe.batch_size,
        learning_rate=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        grad_clip_norm=recipe.grad_clip_norm,
        evaluation_split="validation",
        estimated_seconds=_estimated_seconds(suite, dataset, model_dim, modes),
        modes=modes,
    )


def stage1_jobs() -> list[FairnessJob]:
    return [
        _job(
            stage="stage1",
            suite=suite,
            dataset=dataset,
            capacity_index=capacity_index,
            model_dim=model_dim,
            modes=modes,
            trial=trial,
            seed=SEARCH_SEED,
        )
        for suite in cast("tuple[Literal['ucr', 'external'], ...]", ("ucr", "external"))
        for dataset in _datasets(suite)
        for capacity_index, (model_dim, modes) in enumerate(CAPACITY_GRID, start=1)
        for trial in OPTIMIZER_TRIALS
    ]


def _stage1_config_keys_by_cell() -> dict[str, set[str]]:
    """Return the configuration keys produced by the active candidate jobs.

    ``CANDIDATE`` is intentionally overridden by thin campaign adapters.  Some
    candidates use dimension-based keys while adapter-only candidates use the
    runner's width-tier keys, so reconstructing keys from dimensions here can
    disagree with the completed rows even when the search grid is valid.
    """
    expected: dict[str, set[str]] = {}
    for job in stage1_jobs():
        expected.setdefault(job.cell_key, set()).add(job.config_key)
    return expected


def _source_manifest() -> dict[str, object]:
    root = Path(__file__).resolve().parent
    names = (
        "pac_two_tap_q1_campaign.py",
        "pac_baseline_fairness_maximal.py",
        "pac_efp_writer_reader.py",
        "pac_headroom_efficient_models.py",
        "pac_h_compact_lag124.py",
        "pac_h_compact_lag124_tied.py",
        "pac_h_compact_lag124_tied_q1_campaign.py",
        "pac_compiled_lag124_moments.py",
        "pac_laplace_native_input.py",
        "pac_raw_efficiency_candidates.py",
        "pac_tight_frame_models.py",
        "pac_training.py",
    )
    hashes = {name: file_sha256(root / name) for name in names}
    body: dict[str, object] = {
        "schema": "pac_two_tap_q1_source_manifest.v1",
        "source_sha256": hashes,
        "candidate": CANDIDATE,
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def enqueue_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    jobs = stage1_jobs()
    active = default_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "stage1", jobs, active)
    manifest = _source_manifest()
    write_once(
        root / "reports/source_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    contract: dict[str, object] = {
        "schema": "pac_two_tap_q1_search_contract.v1",
        "candidate": CANDIDATE,
        "tasks": len(UCR_DATASETS) + len(EXTERNAL_DATASETS),
        "ucr_tasks": len(UCR_DATASETS),
        "external_tasks": len(EXTERNAL_DATASETS),
        "capacity_grid": [list(pair) for pair in CAPACITY_GRID],
        "optimizer_trials": list(OPTIMIZER_TRIALS),
        "evaluations_per_task": len(CAPACITY_GRID) * len(OPTIMIZER_TRIALS),
        "jobs": len(jobs),
        "search_seed": SEARCH_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "selection_split": "TRAIN-derived validation",
        "official_test_accessed": False,
        "source_manifest_sha256": manifest["sha256"],
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "stage1/contract.json", json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    return contract


def _audit_validation_rows(
    rows: list[dict[str, object]],
    *,
    stage: Literal["stage1", "stage2"],
) -> None:
    expected_seeds = {SEARCH_SEED} if stage == "stage1" else set(CONFIRMATION_SEEDS)
    seen: set[str] = set()
    code_hashes: set[str] = set()
    for row in rows:
        key = str(row.get("job_key", ""))
        suite = str(row.get("suite", ""))
        valid_dataset = suite in {"ucr", "external"} and str(row.get("dataset")) in _datasets(
            cast("Literal['ucr', 'external']", suite)
        )
        score = row.get("selection_score")
        if (
            not key
            or key in seen
            or row.get("status") != "done"
            or row.get("stage") != stage
            or row.get("model") != CANDIDATE
            or not valid_dataset
            or row.get("evaluation_split") != "validation"
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or int(cast("int", row.get("split_seed", -1))) not in expected_seeds
            or int(cast("int", row.get("train_seed", -1))) not in expected_seeds
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise RuntimeError(f"invalid or TEST-contaminated {stage} row: {key}")
        seen.add(key)
        code_hashes.add(str(row.get("code_sha256", "")))
    if len(code_hashes) != 1 or "" in code_hashes:
        raise RuntimeError(f"{stage} mixes runner implementations: {sorted(code_hashes)}")


def select_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    rows = _require_complete_stage(root, "stage1")
    _audit_validation_rows(rows, stage="stage1")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_key"]), []).append(row)
    expected_cells = {
        f"{suite}:{dataset}:{CANDIDATE}"
        for suite in ("ucr", "external")
        for dataset in _datasets(cast("Literal['ucr', 'external']", suite))
    }
    if set(grouped) != expected_cells:
        raise RuntimeError("Stage 1 does not contain the exact 30-task candidate grid")
    expected_configs_by_cell = _stage1_config_keys_by_cell()
    selected: dict[str, list[str]] = {}
    jobs: list[FairnessJob] = []
    for cell_key, cell_rows in sorted(grouped.items()):
        expected_configs = expected_configs_by_cell[cell_key]
        if (
            len(cell_rows) != len(expected_configs)
            or {str(row["config_key"]) for row in cell_rows} != expected_configs
        ):
            raise RuntimeError(f"invalid Stage 1 configuration grid for {cell_key}")
        top = sorted(cell_rows, key=_rank_row)[:TOP_K]
        selected[cell_key] = [str(row["config_key"]) for row in top]
        for row in top:
            base = _job_from_result(row)
            jobs.extend(
                replace(base, stage="stage2", split_seed=seed, train_seed=seed)
                for seed in CONFIRMATION_SEEDS
            )
    active = default_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "stage2", jobs, active)
    payload: dict[str, object] = {
        "schema": "pac_two_tap_q1_stage1_selection.v1",
        "source_rows": len(rows),
        "cells": len(grouped),
        "top_k": TOP_K,
        "selected": selected,
        "stage2_jobs": len(jobs),
        "official_test_accessed": False,
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "stage1/selection.json", json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _tie(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=TIE_RTOL, abs_tol=TIE_ATOL)


def _average_rank(scores: dict[str, float], candidate: str) -> float:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    index = next(index for index, item in enumerate(ordered) if item[0] == candidate)
    start = index
    stop = index + 1
    while start > 0 and _tie(ordered[start - 1][1], ordered[index][1]):
        start -= 1
    while stop < len(ordered) and _tie(ordered[stop][1], ordered[index][1]):
        stop += 1
    return mean(range(start + 1, stop + 1))


def _validation_comparison(
    selected: dict[str, dict[str, object]],
    baseline_root: Path,
) -> dict[str, object]:
    baseline = json.loads((baseline_root / "stage2/selection.json").read_text(encoding="utf-8"))
    baseline_selected = cast("dict[str, dict[str, object]]", baseline["selected"])
    per_task: dict[str, dict[str, object]] = {}
    ranks: list[float] = []
    scores: list[float] = []
    top_count = 0
    for suite in ("ucr", "external"):
        active_suite = cast("Literal['ucr', 'external']", suite)
        for dataset in _datasets(active_suite):
            task_key = f"{suite}:{dataset}"
            candidate_score = float(selected[f"{task_key}:{CANDIDATE}"]["mean_selection_score"])
            all_scores = {
                CANDIDATE: candidate_score,
                **{
                    model: float(baseline_selected[f"{task_key}:{model}"]["mean_selection_score"])
                    for model in BASELINES
                },
            }
            rank = _average_rank(all_scores, CANDIDATE)
            best = max(all_scores.values())
            is_top = _tie(candidate_score, best)
            ranks.append(rank)
            scores.append(candidate_score)
            top_count += int(is_top)
            per_task[task_key] = {
                "candidate_score": candidate_score,
                "candidate_rank": rank,
                "candidate_top": is_top,
                "selected_config": selected[f"{task_key}:{CANDIDATE}"]["config_key"],
                "scores": all_scores,
            }
    return {
        "schema": "pac_two_tap_q1_30_task_validation_comparison.v1",
        "candidate": CANDIDATE,
        "tasks": 30,
        "global_top_count": top_count,
        "mean_rank_vs_six_baselines": mean(ranks),
        "mean_task_selection_score": mean(scores),
        "selection_rule": "task-specific TRAIN-derived validation score",
        "tie_tolerance": {"rtol": TIE_RTOL, "atol": TIE_ATOL},
        "official_test_accessed": False,
        "per_task": per_task,
    }


def select_stage2(
    root: Path = DEFAULT_ROOT,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
) -> dict[str, object]:
    stage1 = _require_complete_stage(root, "stage1")
    stage2 = _require_complete_stage(root, "stage2")
    _audit_validation_rows(stage1, stage="stage1")
    _audit_validation_rows(stage2, stage="stage2")
    stage1_selection = json.loads((root / "stage1/selection.json").read_text(encoding="utf-8"))
    by_cell_config: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in stage1 + stage2:
        by_cell_config.setdefault((str(row["cell_key"]), str(row["config_key"])), []).append(row)
    selected: dict[str, dict[str, object]] = {}
    expected_seeds = {SEARCH_SEED, *CONFIRMATION_SEEDS}
    for cell_key, config_keys in sorted(stage1_selection["selected"].items()):
        candidates: list[tuple[float, str, list[dict[str, object]]]] = []
        for config_key in config_keys:
            rows = by_cell_config[(cell_key, config_key)]
            if len(rows) != 3 or {int(row["train_seed"]) for row in rows} != expected_seeds:
                raise RuntimeError(f"incomplete Stage 2 confirmation for {cell_key}/{config_key}")
            candidates.append(
                (mean(float(row["selection_score"]) for row in rows), config_key, rows)
            )
        score, config_key, rows = min(candidates, key=lambda item: (-item[0], item[1]))
        base = _job_from_result(rows[0])
        best_epochs = [int(row["best_epoch"]) for row in rows if row.get("best_epoch") is not None]
        refit_epochs = max(1, round(median(best_epochs))) if base.suite == "ucr" else base.epochs
        selected[cell_key] = {
            "config_key": config_key,
            "model_dim": base.width,
            "width": base.width,
            "width_tier": base.width_tier,
            "modes": base.modes,
            "trial": base.trial,
            "mean_selection_score": score,
            "selection_seeds": sorted(expected_seeds),
            "params_trainable": mean(float(row["params_trainable"]) for row in rows),
            "ucr_refit_epochs": refit_epochs if base.suite == "ucr" else None,
        }
    comparison = _validation_comparison(selected, baseline_root)
    payload: dict[str, object] = {
        "schema": "pac_two_tap_q1_stage2_selection.v1",
        "candidate": CANDIDATE,
        "cells": len(selected),
        "selected": selected,
        "comparison": comparison,
        "official_test_accessed": False,
    }
    write_once(
        root / "stage2/selection.json", json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    write_once(
        root / "reports/validation_comparison.json",
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    )
    return payload


def _copy_baseline_results(source_root: Path, target_root: Path) -> list[FairnessJob]:
    jobs: list[FairnessJob] = []
    seen_cells: set[tuple[str, str, str, int]] = set()
    destination = target_root / "final/completed"
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted((source_root / "final/completed").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("model") not in BASELINES:
            continue
        suite = str(row.get("suite"))
        dataset = str(row.get("dataset"))
        seed = int(row.get("train_seed", -1))
        if (
            suite not in {"ucr", "external"}
            or dataset not in _datasets(cast("Literal['ucr', 'external']", suite))
            or seed not in FINAL_SEEDS
            or row.get("status") != "done"
            or row.get("evaluation_split") != "test"
            or row.get("official_test_accessed") is not True
        ):
            raise RuntimeError(f"invalid reusable baseline TEST row: {path}")
        cell = (suite, dataset, str(row["model"]), seed)
        if cell in seen_cells:
            raise RuntimeError(f"duplicate reusable baseline TEST cell: {cell}")
        seen_cells.add(cell)
        job = _job_from_result(row)
        target = destination / _safe_result_name(job.key)
        if not target.exists():
            shutil.copy2(path, target)
        jobs.append(job)
    expected = 30 * len(BASELINES) * len(FINAL_SEEDS)
    if len(jobs) != expected:
        raise RuntimeError(f"baseline TEST reuse is incomplete: {len(jobs)}/{expected}")
    return jobs


def enqueue_final(
    root: Path = DEFAULT_ROOT,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    selection = json.loads((root / "stage2/selection.json").read_text(encoding="utf-8"))
    if (
        selection.get("official_test_accessed") is not False
        or selection.get("candidate") != CANDIDATE
    ):
        raise RuntimeError("candidate selection is not sealed from TEST evidence")
    selected = cast("dict[str, dict[str, object]]", selection["selected"])
    rows = _require_complete_stage(root, "stage1") + _require_complete_stage(root, "stage2")
    by_cell_config: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_cell_config.setdefault((str(row["cell_key"]), str(row["config_key"])), []).append(row)
    candidate_jobs: list[FairnessJob] = []
    for cell_key, metadata in sorted(selected.items()):
        config_key = str(metadata["config_key"])
        source_rows = by_cell_config[(cell_key, config_key)]
        base = _job_from_result(source_rows[0])
        epochs = int(metadata["ucr_refit_epochs"] or base.epochs)
        candidate_jobs.extend(
            replace(
                base,
                stage="final",
                split_seed=seed,
                train_seed=seed,
                epochs=epochs,
                evaluation_split="test",
            )
            for seed in FINAL_SEEDS
        )
    baseline_jobs = _copy_baseline_results(baseline_root, root)
    active = default_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "final", baseline_jobs + candidate_jobs, active)
    source_manifest = json.loads(
        (root / "reports/source_manifest.json").read_text(encoding="utf-8")
    )
    freeze = {
        "schema": "pac_alphabet_q1_final_freeze.v1",
        "chosen_internal_model": CANDIDATE,
        "public_models": [CANDIDATE, *BASELINES],
        "selection_seeds": [SEARCH_SEED, *CONFIRMATION_SEEDS],
        "final_seeds": list(FINAL_SEEDS),
        "selected": selected,
        "test_evidence_used_for_architecture_choice": False,
        "source_comparison": str((root / "reports/validation_comparison.json").resolve()),
        "source_manifest_sha256": source_manifest["sha256"],
    }
    write_once(
        root / "architecture_freeze.json", json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    contract: dict[str, object] = {
        "schema": "pac_two_tap_q1_final_contract.v1",
        "public_model": "ALPHABET",
        "chosen_internal_model": CANDIDATE,
        "tasks": 30,
        "models": 7,
        "final_seeds": list(FINAL_SEEDS),
        "jobs": len(baseline_jobs) + len(candidate_jobs),
        "reused_frozen_baseline_rows": len(baseline_jobs),
        "new_alphabet_jobs": len(candidate_jobs),
        "official_test_access": "allowed only after architecture_freeze.json",
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(root / "final/contract.json", json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    full = campaign_status(root)
    return {"schema": "pac_two_tap_q1_status.v1", **{stage: full[stage] for stage in STAGES}}
