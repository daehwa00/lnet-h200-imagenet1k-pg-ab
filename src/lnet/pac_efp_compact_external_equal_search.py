from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Final, Literal, cast

from .pac_campaign_utils import canonical_json_sha256, file_sha256
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
from .pac_campaign_utils import canonical_json_sha256, file_sha256, write_once
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_efp16_final_campaign import EXTERNAL_DATASETS
from .pac_efp_compact_equal_search import (
    CAPACITY_GRID,
    COMPARISON_ATOL,
    COMPARISON_RTOL,
    COMPATIBLE_SELECTION_RUNNER_SHA256,
    DEFAULT_BASELINE_ROOT,
    MODELS,
    OPTIMIZER_TRIALS,
    _audit_validation_rows,
    _average_tie_ranks,
    _scores_tie,
    comparison_from_frozen_selection,
    default_lanes,
)
from .pac_final_validation import EXTERNAL_SECONDS

DEFAULT_ROOT: Final = Path(".omx/results/pac-efp-compact-external-equal-search-20260719")
DEFAULT_UCR_ROOT: Final = Path(".omx/results/pac-efp-compact-equal-search-20260719")
STAGES: Final = ("stage1", "stage2")
SOURCE_MANIFEST_FILES: Final = (
    "pac_baseline_fairness_maximal.py",
    "pac_efp_compact_equal_search.py",
    "pac_efp_compact_external_equal_search.py",
    "pac_efp_writer_reader.py",
    "pac_external_benchmarks.py",
    "pac_external_tasks.py",
    "pac_headroom_efficient_models.py",
    "pac_laplace_native_input.py",
    "pac_raw_efficiency_candidates.py",
    "pac_tight_frame_models.py",
    "pac_types.py",
)


def _write_source_manifest(root: Path) -> dict[str, object]:
    source_root = Path(__file__).resolve().parent
    hashes = {
        name: file_sha256(source_root / name)
        for name in SOURCE_MANIFEST_FILES
    }
    body: dict[str, object] = {
        "schema": "pac_efp_compact_30_task_source_manifest.v2",
        "source_sha256": hashes,
        "compatible_selection_runner_sha256": sorted(
            COMPATIBLE_SELECTION_RUNNER_SHA256
        ),
        "runner_compatibility_scope": {
            "changed_functions": ["_public_models", "_build_model_from_metadata"],
            "equal_search_stage1_or_stage2_training_path_affected": False,
            "purpose": "enable the frozen winner in subsequent Q2 width reconstruction",
            "pre_extension_hash_reconstructed_from_recorded_patch": True,
        },
    }
    payload = {**body, "sha256": canonical_json_sha256(body)}
    write_once(
        root / "reports/source_manifest.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def _estimated_seconds(dataset: str, model_dim: int, modes: int, model: str) -> float:
    capacity_factor = max(0.25, (model_dim / 32.0) * (modes / 16.0))
    reader_factor = 1.65 if model == "compact_h_only" else 1.0
    return EXTERNAL_SECONDS[dataset] * capacity_factor * reader_factor


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
        suite="external",
        dataset=dataset,
        model=model,
        width_tier=capacity_index,
        width=model_dim,
        trial=trial,
        split_seed=seed,
        train_seed=seed,
        epochs=60,
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
        for dataset in EXTERNAL_DATASETS
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
        "schema": "pac_efp_compact_external_equal_search_stage1.v1",
        "models": list(MODELS),
        "datasets": list(EXTERNAL_DATASETS),
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
        suite="external",
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_key"]), []).append(row)
    expected_cells = {
        f"external:{dataset}:{model}"
        for dataset in EXTERNAL_DATASETS
        for model in MODELS
    }
    if set(grouped) != expected_cells:
        message = "external Stage 1 does not contain the exact candidate-task grid"
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
        "schema": "pac_efp_compact_external_equal_search_stage1_selection.v1",
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


def _baseline_scores(baseline_root: Path) -> dict[str, dict[str, float]]:
    path = baseline_root / "stage2" / "selection.json"
    if not path.exists():
        message = f"baseline selection is missing: {path}"
        raise FileNotFoundError(message)
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = cast("dict[str, dict[str, object]]", payload["selected"])
    return {
        dataset: {
            model: float(selected[f"external:{dataset}:{model}"]["mean_selection_score"])
            for model in BASELINES
        }
        for dataset in EXTERNAL_DATASETS
    }


def _comparison_report(
    selected: dict[str, dict[str, object]],
    baseline_root: Path,
) -> dict[str, object]:
    candidate_scores = {
        model: {
            dataset: float(selected[f"external:{dataset}:{model}"]["mean_selection_score"])
            for dataset in EXTERNAL_DATASETS
        }
        for model in MODELS
    }
    pairwise = {"compact_wins": 0, "ties": 0, "efp_wins": 0}
    baselines = _baseline_scores(baseline_root)
    per_dataset: dict[str, dict[str, object]] = {}
    global_summary = {
        model: {"ranks": [], "global_top_count": 0, "parameters": []} for model in MODELS
    }
    for dataset in EXTERNAL_DATASETS:
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
        row: dict[str, object] = {
            "efp_tuned": efp_score,
            "compact_h_only": compact_score,
            "compact_minus_efp_selection_score": delta,
            "pairwise_winner": winner,
            "efp_config": selected[f"external:{dataset}:efp_tuned"]["config_key"],
            "compact_config": selected[f"external:{dataset}:compact_h_only"]["config_key"],
        }
        for candidate in MODELS:
            scores = {candidate: candidate_scores[candidate][dataset], **baselines[dataset]}
            ranks = _average_tie_ranks(scores)
            best = max(scores.values())
            is_top = _scores_tie(scores[candidate], best)
            cast("list[float]", global_summary[candidate]["ranks"]).append(ranks[candidate])
            global_summary[candidate]["global_top_count"] = cast(
                "int", global_summary[candidate]["global_top_count"]
            ) + int(is_top)
            cast("list[float]", global_summary[candidate]["parameters"]).append(
                float(selected[f"external:{dataset}:{candidate}"]["params_trainable"])
            )
            row[f"{candidate}_global_rank"] = ranks[candidate]
            row[f"{candidate}_global_top"] = is_top
        per_dataset[dataset] = row
    summaries: dict[str, dict[str, object]] = {}
    for model in MODELS:
        summaries[model] = {
            "mean_rank_vs_six_baselines": mean(cast("list[float]", global_summary[model]["ranks"])),
            "global_top_count": global_summary[model]["global_top_count"],
            "mean_parameters": mean(cast("list[float]", global_summary[model]["parameters"])),
        }
    champion = min(
        MODELS,
        key=lambda model: (
            -cast("int", summaries[model]["global_top_count"]),
            cast("float", summaries[model]["mean_rank_vs_six_baselines"]),
            model,
        ),
    )
    return {
        "schema": "pac_efp_compact_external_equal_search_comparison.v1",
        "selection_metric": "task-specific TRAIN-derived validation score; higher is better",
        "official_test_accessed": False,
        "pairwise": pairwise,
        "global_summary": summaries,
        "provisional_champion": champion,
        "selection_rule": "global top count, then mean rank",
        "tie_tolerance": {"rtol": COMPARISON_RTOL, "atol": COMPARISON_ATOL},
        "per_dataset": per_dataset,
    }


def select_stage2(
    root: Path = DEFAULT_ROOT,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    ucr_root: Path = DEFAULT_UCR_ROOT,
) -> dict[str, object]:
    selection_path = root / "stage2" / "selection.json"
    if selection_path.exists():
        return cast("dict[str, object]", json.loads(selection_path.read_text(encoding="utf-8")))
    stage1 = _audit_validation_rows(
        _require_complete_stage(root, "stage1"),
        stage="stage1",
        suite="external",
    )
    stage2 = _audit_validation_rows(
        _require_complete_stage(root, "stage2"),
        stage="stage2",
        suite="external",
    )
    by_cell_config: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in stage1 + stage2:
        key = (str(row["cell_key"]), str(row["config_key"]))
        by_cell_config.setdefault(key, []).append(row)
    stage1_selection = json.loads((root / "stage1" / "selection.json").read_text(encoding="utf-8"))
    expected_cells = {
        f"external:{dataset}:{model}"
        for dataset in EXTERNAL_DATASETS
        for model in MODELS
    }
    if set(stage1_selection["selected"]) != expected_cells:
        message = "external Stage 1 selection does not contain the exact candidate-task grid"
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
    source_manifest = _write_source_manifest(root)
    payload: dict[str, object] = {
        "schema": "pac_efp_compact_external_equal_search_stage2_selection.v1",
        "cells": len(selected),
        "selected": selected,
        "comparison": comparison,
        "official_test_accessed": False,
        "source_manifest_sha256": source_manifest["sha256"],
        "final_jobs": 0,
    }
    write_once(selection_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_once(
        root / "reports" / "comparison.json",
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    )
    ucr_selection_path = ucr_root / "stage2" / "selection.json"
    if ucr_selection_path.exists():
        ucr_comparison = comparison_from_frozen_selection(
            ucr_root,
            baseline_root=baseline_root,
        )
        combined_summary: dict[str, dict[str, object]] = {}
        for model in MODELS:
            ucr_summary = cast(
                "dict[str, object]",
                cast("dict[str, object]", ucr_comparison["global_summary"])[model],
            )
            external_summary = cast(
                "dict[str, object]",
                cast("dict[str, object]", comparison["global_summary"])[model],
            )
            combined_summary[model] = {
                "global_top_count": cast("int", ucr_summary["global_top_count"])
                + cast("int", external_summary["global_top_count"]),
                "mean_rank_vs_six_baselines": (
                    18 * cast("float", ucr_summary["mean_rank_vs_six_baselines"])
                    + 12 * cast("float", external_summary["mean_rank_vs_six_baselines"])
                )
                / 30,
            }
        ucr_pairwise = cast("dict[str, int]", ucr_comparison["pairwise"])
        external_pairwise = cast("dict[str, int]", comparison["pairwise"])
        combined_pairwise = {
            key: ucr_pairwise[key] + external_pairwise[key]
            for key in ("compact_wins", "ties", "efp_wins")
        }
        champion = min(
            MODELS,
            key=lambda model: (
                -cast("int", combined_summary[model]["global_top_count"]),
                cast("float", combined_summary[model]["mean_rank_vs_six_baselines"]),
                model,
            ),
        )
        combined = {
            "schema": "pac_efp_compact_equal_search_30_task_comparison.v1",
            "tasks": 30,
            "ucr_tasks": 18,
            "external_tasks": 12,
            "official_test_accessed": False,
            "source_manifest_sha256": source_manifest["sha256"],
            "global_summary": combined_summary,
            "pairwise": combined_pairwise,
            "provisional_champion": champion,
            "selection_rule": "global top count, then mean rank",
            "tie_tolerance": {"rtol": COMPARISON_RTOL, "atol": COMPARISON_ATOL},
            "ucr": ucr_comparison,
            "external": comparison,
        }
        write_once(
            root / "reports" / "combined_30_task_comparison.json",
            json.dumps(combined, indent=2, sort_keys=True) + "\n",
        )
    return payload


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    full = campaign_status(root)
    return {
        "schema": "pac_efp_compact_external_equal_search_status.v1",
        **{stage: full[stage] for stage in STAGES},
    }
