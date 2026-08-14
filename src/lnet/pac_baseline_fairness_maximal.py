from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from statistics import mean, median
from time import perf_counter, time_ns
from typing import TYPE_CHECKING, Final, Literal, cast
from uuid import uuid4

import torch
from torch import nn

from .pac_campaign_utils import write_once
from .pac_baseline_fullfit_backend_screen import (
    _train_with_cuda_graph,  # pyright: ignore[reportPrivateUsage]
)
from .pac_confirmatory_baselines import (
    build_confirmatory_family,
    confirmatory_trial_spec,
)
from .pac_data_split import stratified_partition_indices
from .pac_device import resolve_device
from .pac_efp16_final_campaign import EXTERNAL_DATASETS, UCR_DATASETS
from .pac_efp_writer_reader import (
    CompactEFPHOnlyTerminalPAC,
    LearnedTwoTapHOnlyTerminalPAC,
)
from .pac_eval_sections import (
    clean_validation_classification_task,
    full_train_classification_task,
)
from .pac_external_benchmarks import (
    ExternalBenchmarkConfig,
    _build_continuous_model,  # pyright: ignore[reportPrivateUsage]
    _loss,  # pyright: ignore[reportPrivateUsage]
    _measure_latency,  # pyright: ignore[reportPrivateUsage]
    _predict,  # pyright: ignore[reportPrivateUsage]
    _release_device,  # pyright: ignore[reportPrivateUsage]
    _seed_everything,  # pyright: ignore[reportPrivateUsage]
    _train_model,  # pyright: ignore[reportPrivateUsage]
    external_metric_bundle,
)
from .pac_external_tasks import (
    ExternalDatasetName,
    ExternalSelectionTask,
    ExternalTask,
    load_external_selection_task,
    load_external_task,
)
from .pac_campaign_utils import canonical_json_sha256, file_sha256
from .pac_final_validation import EXTERNAL_SECONDS, UCR_SECONDS
from .pac_full_state_terminal_analyzer import build_full_state_terminal_analyzer
from .pac_h_compact_lag124 import HCompactLag124PAC
from .pac_h_compact_lag124_tied import HCompactLag124TiedPAC
from .pac_headroom_efficient_models import EdgeFramePAC, build_efficient_headroom_classifier
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_dataset, ensure_ucr_train_only
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACDevice, PACExperimentConfig

UTC = timezone.utc  # noqa: UP017 - B200 campaign runtime remains on Python 3.10.

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .pac_confirmatory_baselines import ConfirmatoryFamily
    from .pac_external_benchmarks import ExternalModelFamily
    from .pac_external_tasks import ExternalTemporalMetadata


DEFAULT_ROOT: Final = Path(".omx/results/pac-baseline-fairness-maximal-20260714")
UCR_DATA_ROOT: Final = Path(".omx/data/ucr")
EXTERNAL_DATA_ROOT: Final = Path("data/external")

MODELS: Final = (
    "efp16",
    "pa2wp",
    "tcn",
    "cnn1d",
    "transformer",
    "mamba",
    "gru",
    "lstm",
)
BASELINES: Final = MODELS[2:]
MODEL_WIDTHS: Final[dict[str, tuple[int, int, int]]] = {
    "efp16": (16, 32, 64),
    "pa2wp": (16, 32, 64),
    "tcn": (32, 64, 128),
    "cnn1d": (32, 64, 128),
    "transformer": (32, 64, 128),
    "mamba": (32, 64, 128),
    "gru": (32, 64, 128),
    "lstm": (32, 64, 128),
}
SEARCH_SEED: Final = 7
CONFIRMATION_SEEDS: Final = (11, 19)
FINAL_SEEDS: Final = (23, 31, 43, 47, 59)
TRIALS: Final = tuple(range(1, 7))
TOP_K: Final = 6
Q2_BUDGET_MULTIPLIERS: Final = (0.5, 1.0, 2.0, 4.0)
Q2_LR_MULTIPLIERS: Final = (0.3, 1.0, 3.0)
Q2_TERMINAL_FAILURE_ATTEMPTS: Final = 2
PARAMETER_TOLERANCE: Final = 0.062
STAGES: Final = (
    "stage1",
    "stage2",
    "final",
    "q2_calibration",
    "q2_final",
    "q2_latency_profile",
    "q3_bridge",
    "q3_latency_profile",
    "profile",
)
REFERENCE_PROFILE_REPEATS: Final = 5
REFERENCE_WARMUP_ITERATIONS: Final = 200
REFERENCE_TIMED_ITERATIONS: Final = 1_000


def _public_models(chosen_internal_model: str) -> tuple[str, ...]:
    if chosen_internal_model not in {
        "efp16",
        "pa2wp",
        "efp_tuned",
        "compact_h_only",
        "two_tap_h_only",
        "h_compact_lag124",
        "h_compact_lag124_tied",
    }:
        message = f"unknown ALPHABET realization: {chosen_internal_model}"
        raise ValueError(message)
    return (chosen_internal_model, *BASELINES)


@dataclass(frozen=True, slots=True)
class ResourceLane:
    name: str
    host: Literal["pro6000", "b200", "local_gpu"]
    gpu: int
    lane: int
    relative_speed: float


def default_resource_lanes() -> tuple[ResourceLane, ...]:
    lanes = [
        ResourceLane(f"pro6000-gpu0-lane{lane:02d}", "pro6000", 0, lane, 1.0) for lane in range(3)
    ]
    lanes.extend(
        ResourceLane(f"b200-gpu{gpu}-lane{lane:02d}", "b200", gpu, lane, 2.5)
        for gpu in (0, 2, 3, 4, 5, 6, 7)
        for lane in range(3)
    )
    return tuple(lanes)


@dataclass(frozen=True, slots=True)
class FairnessJob:
    stage: Literal[
        "stage1",
        "stage2",
        "final",
        "q2_calibration",
        "q2_final",
        "q3_bridge",
    ]
    suite: Literal["ucr", "external"]
    dataset: str
    model: str
    width_tier: int
    width: int
    trial: int
    split_seed: int
    train_seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    evaluation_split: Literal["validation", "test"]
    estimated_seconds: float
    budget_multiplier: float | None = None
    lr_multiplier: float = 1.0
    target_parameters: int | None = None
    relative_parameter_error: float | None = None
    latency_multiplier: float | None = None
    target_latency_ms: float | None = None
    modes: int = 16
    gradient_accumulation_steps: int = 1

    @property
    def config_key(self) -> str:
        if self.stage in {"q2_calibration", "q2_final"}:
            if self.model in {
                "full_early",
                "efp_tuned",
                "compact_h_only",
                "two_tap_h_only",
                "h_compact_lag124",
                "h_compact_lag124_tied",
            }:
                return (
                    f"b{self.budget_multiplier:g}-d{self.width}-m{self.modes}-"
                    f"t{self.trial}-lr{self.lr_multiplier:g}"
                )
            return (
                f"b{self.budget_multiplier:g}-w{self.width}-t{self.trial}-lr{self.lr_multiplier:g}"
            )
        if self.model in {
            "full_early",
            "efp_tuned",
            "compact_h_only",
            "two_tap_h_only",
            "h_compact_lag124",
            "h_compact_lag124_tied",
        }:
            return f"d{self.width}-m{self.modes}-t{self.trial}"
        if self.stage == "q3_bridge":
            return f"lat{self.latency_multiplier:g}-w{self.width}-t{self.trial}"
        return f"w{self.width_tier}-t{self.trial}"

    @property
    def cell_key(self) -> str:
        return f"{self.suite}:{self.dataset}:{self.model}"

    @property
    def key(self) -> str:
        return (
            f"fairness:{self.stage}:{self.cell_key}:{self.config_key}:"
            f"split{self.split_seed}:seed{self.train_seed}"
        )


def stage1_jobs() -> list[FairnessJob]:
    jobs: list[FairnessJob] = []
    for suite, datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS)):
        for dataset in datasets:
            for model in MODELS:
                widths = MODEL_WIDTHS[model]
                for width_tier, width in enumerate(widths, start=1):
                    jobs.extend(
                        _job(
                            stage="stage1",
                            suite=cast("Literal['ucr', 'external']", suite),
                            dataset=dataset,
                            model=model,
                            width_tier=width_tier,
                            width=width,
                            trial=trial,
                            split_seed=SEARCH_SEED,
                            train_seed=SEARCH_SEED,
                            evaluation_split="validation",
                        )
                        for trial in TRIALS
                    )
    return jobs


def enqueue_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    active_lanes = default_resource_lanes() if lanes is None else lanes
    if not active_lanes:
        message = "at least one resource lane is required"
        raise ValueError(message)
    jobs = stage1_jobs()
    loads = _write_manifests(root, "stage1", jobs, active_lanes)
    contract: dict[str, object] = {
        "schema": "pac_baseline_fairness_maximal.v1",
        "public_model": "ALPHABET",
        "candidate_models": ["EFP16", "PA2WP"],
        "question": "native-configuration model-family comparison",
        "datasets": {
            "ucr": list(UCR_DATASETS),
            "external": list(EXTERNAL_DATASETS),
        },
        "models": list(MODELS),
        "width_ladders": {key: list(value) for key, value in MODEL_WIDTHS.items()},
        "search": {
            "stage1": {
                "seed": SEARCH_SEED,
                "trials_per_width": len(TRIALS),
                "widths_per_model": 3,
                "jobs": len(jobs),
            },
            "stage2": {
                "top_k": TOP_K,
                "additional_seeds": list(CONFIRMATION_SEEDS),
                "expected_jobs": len(UCR_DATASETS + EXTERNAL_DATASETS)
                * len(MODELS)
                * TOP_K
                * len(CONFIRMATION_SEEDS),
            },
            "final": {
                "independent_seeds": list(FINAL_SEEDS),
                "expected_jobs": len(UCR_DATASETS + EXTERNAL_DATASETS)
                * (len(BASELINES) + 1)
                * len(FINAL_SEEDS),
                "model_rule": "one validation-frozen ALPHABET realization plus six baselines",
            },
        },
        "selection_metric": {
            "ucr": "TRAIN-derived validation balanced accuracy (higher is better)",
            "external_multiclass": "validation balanced accuracy (higher is better)",
            "external_multilabel": "validation macro AUPRC (higher is better)",
            "external_forecasting": "negative validation MSE (higher is better)",
        },
        "test_policy": {
            "selection_stages": "no test prediction or metric",
            "ucr": "official TEST is loaded only after one configuration per cell is frozen",
            "external": (
                "selection workers require physically separate TRAIN/validation-only artifacts; "
                "full task TEST tensors are loaded only by the offline artifact preparer and "
                "final workers"
            ),
            "final_seed_overlap_with_selection": False,
        },
        "fairness": {
            "same_number_of_configurations_per_model": True,
            "same_selection_and_final_seeds_per_model": True,
            "native_widths_not_parameter_matched": True,
            "dummy_or_adapter_parameters": False,
            "both_alphabet_candidates_receive_full_budget": True,
        },
        "restart_safe": True,
        "resource_lanes": [asdict(lane) for lane in active_lanes],
        "estimated_normalized_lane_seconds": loads,
    }
    root.mkdir(parents=True, exist_ok=True)
    write_once(root / "contract.json", json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def select_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    rows = _require_complete_stage(root, "stage1")
    grouped = _group_rows(rows)
    selected: dict[str, list[str]] = {}
    jobs: list[FairnessJob] = []
    for cell_key, cell_rows in sorted(grouped.items()):
        if len(cell_rows) != 18:
            message = f"{cell_key} has {len(cell_rows)} stage1 rows; expected 18"
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
    active_lanes = default_resource_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "stage2", jobs, active_lanes)
    payload: dict[str, object] = {
        "schema": "pac_baseline_fairness_stage1_selection.v1",
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
        by_cell_config.setdefault((str(row["cell_key"]), str(row["config_key"])), []).append(row)
    stage1_selection = json.loads((root / "stage1" / "selection.json").read_text(encoding="utf-8"))
    paper_cells = set(cast("dict[str, object]", stage1_selection["selected"]))
    relevant_stage2 = [row for row in stage2 if str(row["cell_key"]) in paper_cells]
    expected_stage2_rows = len(paper_cells) * TOP_K * len(CONFIRMATION_SEEDS)
    if len(relevant_stage2) != expected_stage2_rows:
        message = (
            f"paper-facing stage2 has {len(relevant_stage2)} rows; expected {expected_stage2_rows}"
        )
        raise RuntimeError(message)
    selected: dict[str, dict[str, object]] = {}
    frozen_jobs: dict[str, tuple[FairnessJob, int]] = {}
    for cell_key, config_keys in sorted(stage1_selection["selected"].items()):
        candidates: list[tuple[float, str, list[dict[str, object]]]] = []
        for config_key in config_keys:
            rows = by_cell_config[(cell_key, config_key)]
            seeds = {cast("int", row["train_seed"]) for row in rows}
            expected_seeds = {SEARCH_SEED, *CONFIRMATION_SEEDS}
            if len(rows) != 3 or seeds != expected_seeds:
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
            "width_tier": base.width_tier,
            "width": base.width,
            "trial": base.trial,
            "mean_selection_score": score,
            "selection_seeds": [SEARCH_SEED, *CONFIRMATION_SEEDS],
            "ucr_refit_epochs": refit_epochs if base.suite == "ucr" else None,
        }
        frozen_jobs[cell_key] = (base, refit_epochs)

    decision = _alphabet_decision(selected)
    chosen = str(decision["chosen_internal_model"])
    public_models = _public_models(chosen)
    jobs: list[FairnessJob] = []
    for _cell_key, (base, refit_epochs) in sorted(frozen_jobs.items()):
        if base.model not in public_models:
            continue
        jobs.extend(
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
    active_lanes = default_resource_lanes() if lanes is None else lanes
    existing_keys = _manifest_job_keys(root / "final" / "manifests")
    missing_jobs = [job for job in jobs if job.key not in existing_keys]
    loads = (
        _write_manifests(
            root,
            "final",
            missing_jobs,
            active_lanes,
            filename_prefix="stage2-selection-",
        )
        if missing_jobs
        else dict.fromkeys((lane.name for lane in active_lanes), 0.0)
    )
    payload: dict[str, object] = {
        "schema": "pac_baseline_fairness_stage2_selection.v1",
        "source_rows": len(relevant_stage2),
        "raw_source_rows": len(stage2),
        "cells": len(selected),
        "selected": selected,
        "chosen_internal_model": chosen,
        "public_models": list(public_models),
        "final_seeds": list(FINAL_SEEDS),
        "final_jobs": len(jobs),
        "newly_enqueued_final_jobs": len(missing_jobs),
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        selection_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    write_once(
        root / "architecture_decision.json",
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
    )
    return payload


def _alphabet_decision(selected: dict[str, dict[str, object]]) -> dict[str, object]:
    """Choose EFP16 or PA2WP from frozen validation means only."""
    wins = {"efp16": 0, "pa2wp": 0}
    ties = 0
    decisions: list[dict[str, object]] = []
    datasets = UCR_DATASETS + EXTERNAL_DATASETS
    for suite, suite_datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS)):
        for dataset in suite_datasets:
            scores = {
                model: cast("float", selected[f"{suite}:{dataset}:{model}"]["mean_selection_score"])
                for model in ("efp16", "pa2wp")
            }
            if math.isclose(scores["efp16"], scores["pa2wp"], rel_tol=0.0, abs_tol=1.0e-12):
                winner = "tie"
                ties += 1
            else:
                winner = max(scores, key=scores.__getitem__)
                wins[winner] += 1
            decisions.append(
                {
                    "suite": suite,
                    "dataset": dataset,
                    "scores": scores,
                    "winner": winner,
                }
            )
    chosen = min(("efp16", "pa2wp"), key=lambda model: (-wins[model], model))
    payload: dict[str, object] = {
        "schema": "pac_baseline_fairness_alphabet_choice.v1",
        "chosen_internal_model": chosen,
        "public_model": "ALPHABET",
        "rule": (
            "more dataset wins on frozen three-seed TRAIN-derived validation; "
            "exact tie resolves lexicographically to EFP16"
        ),
        "test_evidence_used": False,
        "datasets": len(datasets),
        "wins": wins,
        "ties": ties,
        "per_dataset": decisions,
    }
    return payload


def freeze_alphabet_choice(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    """Freeze EFP16 versus PA2WP from selection-only evidence, never TEST."""
    decision_path = root / "architecture_decision.json"
    if decision_path.exists():
        return cast("dict[str, object]", json.loads(decision_path.read_text(encoding="utf-8")))
    selection_path = root / "stage2" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = cast("dict[str, dict[str, object]]", selection["selected"])
    payload = _alphabet_decision(selected)
    write_once(
        decision_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def enqueue_q2_calibration(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    q1_rows = _require_complete_stage(root, "final")
    decision = freeze_alphabet_choice(root)
    alphabet = str(decision["chosen_internal_model"])
    public_models = _public_models(alphabet)
    representative = _representative_rows(q1_rows)
    jobs: list[FairnessJob] = []
    unavailable: list[dict[str, object]] = []
    for suite, datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS)):
        for dataset in datasets:
            target_row = representative[f"{suite}:{dataset}:{alphabet}"]
            natural_parameters = cast("int", target_row["params_trainable"])
            for model in public_models:
                base_row = representative[f"{suite}:{dataset}:{model}"]
                base = _job_from_result(base_row)
                for multiplier in Q2_BUDGET_MULTIPLIERS:
                    target = max(1, round(natural_parameters * multiplier))
                    match = _match_real_width(base, base_row, target)
                    if match is None:
                        unavailable.append(
                            {
                                "suite": suite,
                                "dataset": dataset,
                                "model": model,
                                "budget_multiplier": multiplier,
                                "target_parameters": target,
                                "status": "not_realizable",
                                "tolerance": PARAMETER_TOLERANCE,
                            }
                        )
                        continue
                    width, _parameters, error = match
                    jobs.extend(
                        replace(
                            base,
                            stage="q2_calibration",
                            width=width,
                            split_seed=SEARCH_SEED,
                            train_seed=SEARCH_SEED,
                            learning_rate=base.learning_rate * lr_multiplier,
                            evaluation_split="validation",
                            budget_multiplier=multiplier,
                            lr_multiplier=lr_multiplier,
                            target_parameters=target,
                            relative_parameter_error=error,
                            estimated_seconds=base.estimated_seconds
                            * max(0.35, width / max(base.width, 1)),
                        )
                        for lr_multiplier in Q2_LR_MULTIPLIERS
                    )
    active_lanes = default_resource_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "q2_calibration", jobs, active_lanes)
    unavailable_root = root / "q2_calibration" / "unavailable"
    for row in unavailable:
        key = f"{row['suite']}__{row['dataset']}__{row['model']}__b{row['budget_multiplier']}"
        _write_json(unavailable_root / f"{key}.json", row)
    payload: dict[str, object] = {
        "schema": "pac_baseline_fairness_q2_calibration.v1",
        "alphabet_internal_model": alphabet,
        "budgets": list(Q2_BUDGET_MULTIPLIERS),
        "lr_multipliers": list(Q2_LR_MULTIPLIERS),
        "parameter_tolerance": PARAMETER_TOLERANCE,
        "matching_policy": "nearest trainable real width; no dummy, adapter, or inert parameters",
        "jobs": len(jobs),
        "maximum_jobs": (
            len(UCR_DATASETS + EXTERNAL_DATASETS)
            * len(public_models)
            * len(Q2_BUDGET_MULTIPLIERS)
            * len(Q2_LR_MULTIPLIERS)
        ),
        "not_realizable": len(unavailable),
        "test_evaluated": False,
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "q2_calibration" / "contract.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def select_q2_calibration(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    rows = _require_complete_stage(root, "q2_calibration")
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["cell_key"]), cast("float", row["budget_multiplier"])), []
        ).append(row)
    terminal_failures = _terminal_failure_rows(root, "q2_calibration")
    failed_grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in terminal_failures:
        job = _job_from_result(row)
        failed_grouped.setdefault((job.cell_key, cast("float", job.budget_multiplier)), []).append(
            row
        )
    selected: dict[str, dict[str, object]] = {}
    jobs: list[FairnessJob] = []
    terminally_unavailable: list[dict[str, object]] = []
    calibration_groups = sorted(set(grouped) | set(failed_grouped))
    for cell_key, multiplier in calibration_groups:
        cell_rows = grouped.get((cell_key, multiplier), [])
        failed_rows = failed_grouped.get((cell_key, multiplier), [])
        candidate_count = len(cell_rows) + len(failed_rows)
        if candidate_count != len(Q2_LR_MULTIPLIERS):
            message = (
                f"{cell_key}/budget{multiplier:g} has {candidate_count} terminal calibration "
                f"outcomes ({len(cell_rows)} successful, {len(failed_rows)} failed); "
                f"expected {len(Q2_LR_MULTIPLIERS)}"
            )
            raise RuntimeError(message)
        if not cell_rows:
            terminally_unavailable.append(
                {
                    "cell_key": cell_key,
                    "budget_multiplier": multiplier,
                    "failed_job_keys": sorted(str(row["job_key"]) for row in failed_rows),
                    "reason": "all learning-rate calibration candidates failed",
                }
            )
            continue
        best = min(
            cell_rows,
            key=lambda row: (
                -cast("float", row["selection_score"]),
                abs(cast("float", row["lr_multiplier"]) - 1.0),
                cast("float", row["lr_multiplier"]),
            ),
        )
        base = _job_from_result(best)
        selection_key = f"{cell_key}:budget{multiplier:g}"
        selected[selection_key] = {
            "config_key": best["config_key"],
            "width": base.width,
            "trial": base.trial,
            "lr_multiplier": base.lr_multiplier,
            "learning_rate": base.learning_rate,
            "params_trainable": best["params_trainable"],
            "target_parameters": base.target_parameters,
            "relative_parameter_error": base.relative_parameter_error,
            "selection_score": best["selection_score"],
            "failed_lr_candidates": sorted(
                cast("float", row["lr_multiplier"]) for row in failed_rows
            ),
        }
        jobs.extend(
            replace(
                base,
                stage="q2_final",
                split_seed=seed,
                train_seed=seed,
                evaluation_split="test",
            )
            for seed in FINAL_SEEDS
        )
    active_lanes = default_resource_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "q2_final", jobs, active_lanes)
    payload: dict[str, object] = {
        "schema": "pac_baseline_fairness_q2_selection.v1",
        "source_rows": len(rows),
        "terminal_failure_rows": len(terminal_failures),
        "terminal_failure_policy": (
            f"a candidate is an audited terminal failure after "
            f"{Q2_TERMINAL_FAILURE_ATTEMPTS} failed attempts; it is never selected"
        ),
        "terminally_unavailable_cells": terminally_unavailable,
        "selected_realizable_cells": len(selected),
        "selected": selected,
        "final_seeds": list(FINAL_SEEDS),
        "final_jobs": len(jobs),
        "maximum_final_jobs": (
            len(UCR_DATASETS + EXTERNAL_DATASETS)
            * len(_public_models(str(freeze_alphabet_choice(root)["chosen_internal_model"])))
            * len(Q2_BUDGET_MULTIPLIERS)
            * len(FINAL_SEEDS)
        ),
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "q2_calibration" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: PACDevice = "cuda",
    ucr_data_root: Path = UCR_DATA_ROOT,
    external_data_root: Path = EXTERNAL_DATA_ROOT,
) -> None:
    jobs = [
        FairnessJob(**json.loads(line)["job"])
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    runtime_device = resolve_device(device)
    provenance_sha256, _ = _record_provenance(root, runtime_device)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for job in jobs:
        completed = _result_path(root, job, failed=False)
        if completed.exists():
            continue
        _close_unfinished_attempts(root, job.stage, job.key)
        attempt_id = f"{time_ns()}-{os.getpid()}-{uuid4().hex}"
        attempt_started = perf_counter()
        _append_attempt_event(
            root,
            job.stage,
            job.key,
            attempt_id,
            "started",
            {
                "immutable_job": asdict(job),
                "manifest_sha256": manifest_sha256,
                "provenance_sha256": provenance_sha256,
                "environment": _environment_metadata(runtime_device),
            },
        )
        try:
            row = run_job(
                job,
                device=device,
                ucr_data_root=ucr_data_root,
                external_data_root=external_data_root,
            )
            row["manifest_sha256"] = manifest_sha256
            row["provenance_sha256"] = provenance_sha256
            selection_path = _selection_artifact(root, job.stage)
            row["selection_artifact_sha256"] = (
                hashlib.sha256(selection_path.read_bytes()).hexdigest()
                if selection_path is not None
                else None
            )
            _require_done(row)
        except Exception as error:  # noqa: BLE001 - queue must preserve every failure
            failure: dict[str, object] = {
                "schema": "pac_baseline_fairness_failure.v1",
                "job_key": job.key,
                **asdict(job),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            failure_path = _result_path(root, job, failed=True)
            _write_json(failure_path, failure)
            _append_attempt_event(
                root,
                job.stage,
                job.key,
                attempt_id,
                "failed",
                {
                    "elapsed_seconds": perf_counter() - attempt_started,
                    "failure_path": str(failure_path),
                    "error": failure["error"],
                    "traceback": failure["traceback"],
                },
            )
        else:
            _write_json(completed, row)
            _append_attempt_event(
                root,
                job.stage,
                job.key,
                attempt_id,
                "succeeded",
                {
                    "elapsed_seconds": perf_counter() - attempt_started,
                    "result_path": str(completed),
                },
            )
            failed = _result_path(root, job, failed=True)
            if failed.exists():
                failed.unlink()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_job(
    job: FairnessJob,
    *,
    device: PACDevice,
    ucr_data_root: Path = UCR_DATA_ROOT,
    external_data_root: Path = EXTERNAL_DATA_ROOT,
    ucr_model_builder: Callable[[FairnessJob, PACExperimentConfig, int], nn.Module] | None = None,
    external_model_builder: Callable[
        [FairnessJob, PACExperimentConfig, ExternalTask | ExternalSelectionTask],
        nn.Module,
    ]
    | None = None,
    use_validated_baseline_cuda_graph: bool = False,
) -> dict[str, object]:
    if job.suite == "ucr":
        return _run_ucr_job(
            job,
            device=device,
            data_root=ucr_data_root,
            model_builder=ucr_model_builder,
            use_validated_baseline_cuda_graph=use_validated_baseline_cuda_graph,
        )
    return _run_external_job(
        job,
        device=device,
        data_root=external_data_root,
        model_builder=external_model_builder,
    )


def campaign_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    payload: dict[str, object] = {"schema": "pac_baseline_fairness_status.v1"}
    for stage in STAGES:
        manifests = sorted((root / stage / "manifests").glob("*.jsonl"))
        manifest_keys = _manifest_job_keys(root / stage / "manifests")
        expected = len(manifest_keys)
        completed_paths = list((root / stage / "completed").glob("*.json"))
        failed_paths = list((root / stage / "failed").glob("*.json"))
        completed = len(completed_paths)
        failed = len(failed_paths)
        completed_keys = _job_keys_from_results(completed_paths)
        failed_keys = _job_keys_from_results(failed_paths)
        expected_completed_keys = completed_keys & manifest_keys
        unexpected_completed_keys = completed_keys - manifest_keys
        terminal_failed_keys = (
            _terminal_failure_job_keys(root, stage, failed_keys)
            if stage == "q2_calibration"
            else set()
        )
        retryable_failed_keys = failed_keys - terminal_failed_keys
        terminal_keys = completed_keys | terminal_failed_keys
        attempts = _attempt_counts(root, stage)
        blocking_unfinished = sum(
            job_key in manifest_keys and job_key not in terminal_keys
            for job_key in _unfinished_attempt_job_keys(root, stage)
        )
        remaining = len(manifest_keys - terminal_keys)
        payload[stage] = {
            "manifests": len(manifests),
            "expected": expected,
            "completed": completed,
            "completed_expected": len(expected_completed_keys),
            "unexpected_completed": len(unexpected_completed_keys),
            "failed": failed,
            "terminal_failed": len(terminal_failed_keys),
            "retryable_failed": len(retryable_failed_keys),
            "attempts": attempts,
            "blocking_unfinished": blocking_unfinished,
            "remaining": remaining,
            "done": (
                bool(manifests)
                and remaining == 0
                and not unexpected_completed_keys
                and not retryable_failed_keys
                and blocking_unfinished == 0
            ),
        }
    return payload


def _job(
    *,
    stage: Literal["stage1", "stage2", "final"],
    suite: Literal["ucr", "external"],
    dataset: str,
    model: str,
    width_tier: int,
    width: int,
    trial: int,
    split_seed: int,
    train_seed: int,
    evaluation_split: Literal["validation", "test"],
) -> FairnessJob:
    recipe_family = "pac_tf" if model in {"efp16", "pa2wp"} else model
    spec = confirmatory_trial_spec(cast("ConfirmatoryFamily", recipe_family), trial)
    seconds = UCR_SECONDS[dataset] if suite == "ucr" else EXTERNAL_SECONDS[dataset]
    base_width = MODEL_WIDTHS[model][1]
    width_factor = max(0.35, min(width / base_width, 4.0))
    model_factor = {
        "efp16": 1.0,
        "pa2wp": 1.8,
        "transformer": 1.5,
        "mamba": 1.3,
        "gru": 1.3,
        "lstm": 1.5,
        "s4d": 1.4,
    }.get(model, 1.0)
    return FairnessJob(
        stage=stage,
        suite=suite,
        dataset=dataset,
        model=model,
        width_tier=width_tier,
        width=width,
        trial=trial,
        split_seed=split_seed,
        train_seed=train_seed,
        epochs=100 if suite == "ucr" else 60,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        evaluation_split=evaluation_split,
        estimated_seconds=seconds * width_factor * model_factor,
    )


def _run_ucr_job(  # noqa: PLR0915 - result provenance stays adjacent to training
    job: FairnessJob,
    *,
    device: PACDevice,
    data_root: Path,
    model_builder: Callable[[FairnessJob, PACExperimentConfig, int], nn.Module] | None = None,
    use_validated_baseline_cuda_graph: bool = False,
) -> dict[str, object]:
    runtime_device = resolve_device(device)
    if job.evaluation_split == "validation":
        dataset = ensure_ucr_train_only(job.dataset, data_root, allow_download=True)
        train_indices, validation_indices = stratified_partition_indices(
            dataset.train_labels,
            0.2,
            job.split_seed,
        )
        split_index_label_sha256 = _full_split_identity_sha256(
            train_indices,
            validation_indices,
            dataset.train_labels,
        )
        task = clean_validation_classification_task(dataset, job.split_seed)
    else:
        dataset = ensure_ucr_dataset(
            job.dataset,
            data_root,
            allow_download=True,
            require_train_label_space=True,
        )
        split_index_label_sha256 = _full_split_identity_sha256(
            torch.arange(dataset.train_inputs.shape[0]),
            torch.arange(dataset.test_inputs.shape[0]),
            dataset.train_labels,
            dataset.test_labels,
        )
        task = full_train_classification_task(dataset)
    split_hash = _split_hash(
        task.train_inputs,
        task.train_labels,
        task.test_inputs if job.evaluation_split == "test" else task.validation_inputs,
        task.test_labels if job.evaluation_split == "test" else task.validation_labels,
    )
    config = _experiment_config(
        train_count=task.train_inputs.shape[0],
        validation_count=task.validation_inputs.shape[0],
        test_count=task.test_inputs.shape[0],
        sequence_length=task.train_inputs.shape[1],
        input_dim=task.train_inputs.shape[-1],
        output_dim=task.class_count,
        job=job,
        device=device,
    )
    _seed_everything(job.train_seed, runtime_device)
    build_model = _build_ucr_model if model_builder is None else model_builder
    model = build_model(job, config, task.class_count).to(device=runtime_device)
    # CinCECGTorso has an odd length (1639) that currently trips an Inductor
    # shape assertion in the fused EFP training kernels.  Preserve the exact
    # reference implementation for this dataset instead of failing the run.
    if not (
        job.model in {"efp16", "efp_tuned", "compact_h_only", "two_tap_h_only"}
        and job.dataset == "CinCECGTorso"
    ):
        _enable_pac_optimized_training(model, job.model)
    # Do not opt the heterogeneous campaign into the shape-static CUDA Graph
    # runtime.  Its exact kernels remain enabled by _enable_pac_optimized_training,
    # while Graph replay is reserved for the separately validated fixed-shape
    # efficiency benchmark.  Reusing it across dataset shapes can replay stale
    # label storage and trigger a device-side cross-entropy assertion.
    if runtime_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    final = job.evaluation_split == "test"
    baseline_graph_full_steps = 0
    baseline_eager_tail_steps = 0
    baseline_graph_fallback_error: str | None = None
    test_loss: float | None = None
    use_baseline_graph = (
        use_validated_baseline_cuda_graph
        and runtime_device == "cuda"
        and job.gradient_accumulation_steps == 1
        and not final
        and job.model in {"s4d", "s5", "lru"}
        and task.train_inputs.shape[0] >= job.batch_size
        and task.train_inputs.shape[1] <= 2_048
    )
    if use_baseline_graph:
        initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        try:
            (
                best_epoch,
                validation_loss,
                baseline_graph_full_steps,
                baseline_eager_tail_steps,
            ) = _train_with_cuda_graph(
                model,
                task,
                config,
                seed=job.train_seed,
                graph_warmups=3,
                device=runtime_device,
            )
        except Exception as error:  # noqa: BLE001 - graph failures vary by CUDA/runtime
            baseline_graph_fallback_error = f"{type(error).__name__}: {error}"
            model.load_state_dict(initial_state, strict=True)
            outcome = train_classifier(
                model,
                task,
                config,
                runtime_device,
                job.train_seed,
                evaluate_test=False,
                restore_best_validation=True,
            )
            best_epoch = outcome.best_epoch
            validation_loss = outcome.validation_loss
        finally:
            del initial_state
    else:
        outcome = train_classifier(
            model,
            task,
            config,
            runtime_device,
            job.train_seed,
            evaluate_test=final,
            restore_best_validation=not final,
        )
        best_epoch = outcome.best_epoch
        validation_loss = outcome.validation_loss if not final else None
        if final:
            test_loss = outcome.test_loss
    metric_inputs = task.test_inputs if final else task.validation_inputs
    metric_labels = task.test_labels if final else task.validation_labels
    metrics = classification_metric_bundle(
        model,
        metric_inputs.to(device=runtime_device),
        metric_labels.to(device=runtime_device),
        batch_size=job.batch_size,
    )
    elapsed = perf_counter() - started
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated() / 1_000_000) if runtime_device == "cuda" else 0.0
    )
    latency_ms = (
        _measure_generic_latency(model, metric_inputs, job.batch_size, runtime_device)
        if final
        else None
    )
    latency_ms_batch1 = (
        _measure_generic_latency(model, metric_inputs, 1, runtime_device) if final else None
    )
    return {
        "schema": "pac_baseline_fairness_result.v1",
        "code_sha256": _code_sha256(),
        "environment": _environment_metadata(runtime_device),
        "job_key": job.key,
        "cell_key": job.cell_key,
        "config_key": job.config_key,
        **asdict(job),
        "status": "done",
        "objective": "multiclass",
        "train_count": task.train_inputs.shape[0],
        "validation_count": task.validation_inputs.shape[0],
        "test_count": task.test_inputs.shape[0],
        "sequence_length": task.train_inputs.shape[1],
        "input_dim": task.train_inputs.shape[-1],
        "output_dim": task.class_count,
        "test_evaluated": final,
        "official_test_accessed": final,
        "split_sha256": split_hash,
        "split_hash_kind": "sampled_tensor_split_fingerprint.v1",
        "split_index_label_sha256": split_index_label_sha256,
        "best_epoch": best_epoch,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "selection_score": metrics.balanced_accuracy if not final else None,
        "train_seconds": elapsed,
        "params_trainable": count_parameters(model),
        "latency_ms": latency_ms,
        "latency_ms_batch1": latency_ms_batch1,
        "peak_memory_mb": peak_memory_mb,
        "training_backend": (
            (
                "validated_baseline_cuda_graph_full_step"
                if use_baseline_graph and baseline_graph_fallback_error is None
                else "validated_baseline_cuda_graph_fallback_eager_fused"
            )
            if use_baseline_graph
            else (
                getattr(
                    model,
                    "efp16_exact_split_runtime_kind",
                    "efp16_exact_split_cuda_graph",
                )
                if getattr(model, "efp16_exact_split_capture_succeeded", False)
                else ("efp16_fused_fallback" if job.model == "efp16" else "default")
            )
        ),
        "baseline_graph_full_steps": baseline_graph_full_steps,
        "baseline_eager_tail_steps": baseline_eager_tail_steps,
        "baseline_graph_fallback_error": baseline_graph_fallback_error,
        "efp16_exact_split_full_steps": getattr(model, "efp16_exact_split_full_steps", None),
        "efp16_exact_split_fallback_steps": getattr(
            model, "efp16_exact_split_fallback_steps", None
        ),
        "efp16_exact_split_capture_error": getattr(model, "efp16_exact_split_capture_error", None),
    }


def _run_external_job(
    job: FairnessJob,
    *,
    device: PACDevice,
    data_root: Path,
    model_builder: Callable[
        [FairnessJob, PACExperimentConfig, ExternalTask | ExternalSelectionTask],
        nn.Module,
    ]
    | None = None,
) -> dict[str, object]:
    runtime_device = resolve_device(device)
    final = job.evaluation_split == "test"
    task: ExternalTask | ExternalSelectionTask
    if final:
        task = load_external_task(cast("ExternalDatasetName", job.dataset), data_root)
        test_count = int(task.test_inputs.shape[0])
        selection_data_sha256 = None
        split_index_label_sha256 = _full_split_identity_sha256(
            task.train_targets,
            task.validation_targets,
            task.test_targets,
        )
    else:
        task = load_external_selection_task(job.dataset, data_root)
        test_count = task.test_count
        selection_path = data_root / "selection-only" / f"{job.dataset}.pt"
        selection_data_sha256 = file_sha256(selection_path)
        split_index_label_sha256 = task.selection_split_sha256
    split_hash = (
        task.selection_split_sha256
        if isinstance(task, ExternalSelectionTask)
        else _split_hash(
            task.train_inputs,
            task.train_targets,
            task.validation_inputs,
            task.validation_targets,
        )
    )
    benchmark = ExternalBenchmarkConfig(
        data_root=data_root,
        output_root=DEFAULT_ROOT,
        datasets=(cast("ExternalDatasetName", job.dataset),),
        models=("pac",),
        model_dim=job.width,
        modes=job.modes,
        max_baseline_width=10_000,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=job.learning_rate,
        weight_decay=job.weight_decay,
        grad_clip_norm=job.grad_clip_norm,
        patience=12,
        seeds=(job.train_seed,),
        device=device,
        latency_warmup=5,
        latency_iterations=20,
        pac_model="EFP16",
        gradient_accumulation_steps=job.gradient_accumulation_steps,
    )
    experiment = _experiment_config(
        train_count=task.train_inputs.shape[0],
        validation_count=task.validation_inputs.shape[0],
        test_count=test_count,
        sequence_length=task.sequence_length,
        input_dim=task.input_dim,
        output_dim=task.output_dim,
        job=job,
        device=device,
    )
    _seed_everything(job.train_seed, runtime_device)
    build_model = _build_external_model if model_builder is None else model_builder
    model = build_model(job, experiment, task).to(device=runtime_device)
    _enable_pac_optimized_training(model, job.model)
    started = perf_counter()
    try:
        best_epoch, validation_loss = _train_model(
            model,
            cast("ExternalTask", task),
            benchmark,
            runtime_device,
            job.train_seed,
        )
        inputs = task.test_inputs if isinstance(task, ExternalTask) else task.validation_inputs
        targets = task.test_targets if isinstance(task, ExternalTask) else task.validation_targets
        metadata = (
            task.test_metadata if isinstance(task, ExternalTask) else task.validation_metadata
        )
        logits, targets = _predict(
            model,
            inputs,
            targets,
            job.batch_size,
            runtime_device,
            metadata=metadata,
        )
        metrics = external_metric_bundle(logits, targets, task.objective)
        evaluated_loss = float(_loss(logits, targets, task.objective).item())
        latency_ms, peak_memory_mb = (
            _measure_latency(
                model,
                task.test_inputs,
                benchmark,
                runtime_device,
                metadata=metadata,
            )
            if isinstance(task, ExternalTask)
            else (None, None)
        )
        latency_ms_batch1 = (
            _measure_generic_latency(
                model,
                task.test_inputs,
                1,
                runtime_device,
                metadata=metadata,
            )
            if isinstance(task, ExternalTask)
            else None
        )
        selection_score = (
            None if final else _external_selection_score(task, metrics, evaluated_loss)
        )
        return {
            "schema": "pac_baseline_fairness_result.v1",
            "code_sha256": _code_sha256(),
            "environment": _environment_metadata(runtime_device),
            "job_key": job.key,
            "cell_key": job.cell_key,
            "config_key": job.config_key,
            **asdict(job),
            "status": "done",
            "objective": task.objective,
            "train_count": task.train_inputs.shape[0],
            "validation_count": task.validation_inputs.shape[0],
            "test_count": test_count,
            "sequence_length": task.sequence_length,
            "input_dim": task.input_dim,
            "output_dim": task.output_dim,
            "test_evaluated": final,
            "official_test_accessed": final,
            "test_container_loaded": final,
            "selection_data_artifact_sha256": selection_data_sha256,
            "split_sha256": split_hash,
            "split_hash_kind": (
                "external_selection_full_content.v1"
                if isinstance(task, ExternalSelectionTask)
                else "sampled_tensor_split_fingerprint.v1"
            ),
            "split_index_label_sha256": split_index_label_sha256,
            "best_epoch": best_epoch,
            "validation_loss": validation_loss,
            "evaluated_split_loss": evaluated_loss,
            "selection_score": selection_score,
            "train_seconds": perf_counter() - started,
            "external_exact_split_full_steps": getattr(model, "external_exact_split_full_steps", 0),
            "external_exact_split_fallback_steps": getattr(
                model, "external_exact_split_fallback_steps", 0
            ),
            "external_exact_split_capture_succeeded": getattr(
                model, "external_exact_split_capture_succeeded", False
            ),
            "external_exact_split_runtime_kind": getattr(
                model, "external_exact_split_runtime_kind", None
            ),
            "external_exact_split_capture_error": getattr(
                model, "external_exact_split_capture_error", None
            ),
            "params_trainable": count_parameters(model),
            "latency_ms": latency_ms,
            "latency_ms_batch1": latency_ms_batch1,
            "peak_memory_mb": peak_memory_mb,
            **metrics,
        }
    finally:
        _release_device(runtime_device)


def _experiment_config(
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    sequence_length: int,
    input_dim: int,
    output_dim: int,
    job: FairnessJob,
    device: PACDevice,
) -> PACExperimentConfig:
    return PACExperimentConfig(
        train_count,
        validation_count,
        test_count,
        sequence_length,
        raw_input_dim=input_dim,
        output_dim=output_dim,
        model_dim=job.width,
        modes=job.modes,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=job.learning_rate,
        weight_decay=job.weight_decay,
        grad_clip_norm=job.grad_clip_norm,
        seeds=(job.train_seed,),
        device=device,
        optimizer_mode="fused" if device == "cuda" else "default",
        gradient_accumulation_steps=job.gradient_accumulation_steps,
    )


def _enable_pac_optimized_training(model: nn.Module, model_name: str) -> None:
    """Enable the verified shape-general PAC training fallback.

    Exact CUDA Graph replay remains restricted to benchmarked shapes. Campaign
    shapes outside that dispatch use fused stems/moments and fused AdamW while
    preserving the same FP32 training algorithm.
    """
    if model_name not in {
        "efp16",
        "pa2wp",
        "efp_tuned",
        "compact_h_only",
        "two_tap_h_only",
        "h_compact_lag124",
        "h_compact_lag124_tied",
    }:
        return
    blocks = [getattr(model, "forward_block", None), getattr(model, "backward_block", None)]
    blocks.extend(getattr(model, "extra_blocks", []))
    for block in blocks:
        if block is not None:
            block.recurrence_backend = "auto"
            block.fused_moments_backward_training = True
    if hasattr(model, "use_fused_pa2wp_stem_training"):
        model.__dict__["use_fused_pa2wp_stem_training"] = True
    if hasattr(model, "use_fused_efp16_stem_training"):
        model.__dict__["use_fused_efp16_stem_training"] = True


def _build_ucr_model(  # noqa: PLR0911 - explicit family dispatch is easier to audit.
    job: FairnessJob,
    config: PACExperimentConfig,
    class_count: int,
) -> nn.Module:
    if job.model == "efp16":
        return EdgeFramePAC(
            config,
            class_count,
            modes=16,
            semi_orthogonal=True,
            objective="classification",
            model_dim=job.width,
        )
    if job.model == "efp_tuned":
        return EdgeFramePAC(
            config,
            class_count,
            modes=job.modes,
            semi_orthogonal=True,
            objective="classification",
            model_dim=job.width,
            mode_divisor=2,
        )
    if job.model in {
        "compact_h_only",
        "two_tap_h_only",
        "h_compact_lag124",
        "h_compact_lag124_tied",
    }:
        if job.model == "h_compact_lag124_tied":
            return HCompactLag124TiedPAC(
                config,
                class_count,
                objective="classification",
            )
        if job.model == "h_compact_lag124":
            return HCompactLag124PAC(
                config,
                class_count,
                objective="classification",
            )
        model_class = (
            LearnedTwoTapHOnlyTerminalPAC
            if job.model == "two_tap_h_only"
            else CompactEFPHOnlyTerminalPAC
        )
        return model_class(
            config,
            class_count,
            objective="classification",
        )
    if job.model == "pa2wp":
        return build_efficient_headroom_classifier(
            "PA2WP", config, class_count, objective="classification"
        )
    if job.model == "full_early":
        return build_full_state_terminal_analyzer(
            config,
            class_count,
            "full_state",
            objective="classification",
        )
    return build_confirmatory_family(
        cast("ConfirmatoryFamily", job.model),
        job.width,
        config,
        class_count,
        validation_trial=job.trial,
    )


def _build_external_model(  # noqa: PLR0911 - explicit family dispatch is easier to audit.
    job: FairnessJob,
    config: PACExperimentConfig,
    task: ExternalTask | ExternalSelectionTask,
) -> nn.Module:
    objective = "regression" if task.objective == "forecasting" else "classification"
    if job.model in {
        "efp16",
        "efp_tuned",
        "compact_h_only",
        "two_tap_h_only",
        "h_compact_lag124",
        "h_compact_lag124_tied",
        "pa2wp",
    }:
        if job.model in {"efp16", "efp_tuned"}:
            return EdgeFramePAC(
                config,
                task.output_dim,
                modes=16 if job.model == "efp16" else job.modes,
                semi_orthogonal=True,
                objective=objective,
                model_dim=job.width,
                mode_divisor=2,
            )
        if job.model == "compact_h_only":
            return CompactEFPHOnlyTerminalPAC(
                config,
                task.output_dim,
                objective=objective,
            )
        if job.model == "two_tap_h_only":
            return LearnedTwoTapHOnlyTerminalPAC(
                config,
                task.output_dim,
                objective=objective,
            )
        if job.model == "h_compact_lag124":
            return HCompactLag124PAC(
                config,
                task.output_dim,
                objective=objective,
            )
        if job.model == "h_compact_lag124_tied":
            return HCompactLag124TiedPAC(
                config,
                task.output_dim,
                objective=objective,
            )
        return build_efficient_headroom_classifier(
            "PA2WP",
            config,
            task.output_dim,
            objective=objective,
        )
    if job.model == "full_early":
        return build_full_state_terminal_analyzer(
            config,
            task.output_dim,
            "full_state",
            objective=objective,
        )
    return _build_continuous_model(
        cast("ExternalModelFamily", job.model),
        job.width,
        task.input_dim,
        task.output_dim,
        config,
        "EFP16",
        objective=objective,
    )


def _external_selection_score(
    task: ExternalTask | ExternalSelectionTask,
    metrics: dict[str, float],
    evaluated_loss: float,
) -> float:
    if task.objective == "multiclass":
        return float(metrics["balanced_accuracy"])
    if task.objective == "multilabel":
        value = float(metrics["macro_auprc"])
        return -math.inf if math.isnan(value) else value
    return -evaluated_loss


def _measure_generic_latency(
    model: nn.Module,
    inputs: torch.Tensor,
    batch_size: int,
    device: str,
    *,
    metadata: ExternalTemporalMetadata | None = None,
) -> float:
    sample_count = min(batch_size, inputs.shape[0])
    batch = inputs[:sample_count].to(device=device)
    batch_metadata = (
        None if metadata is None else metadata.batch_slice(0, sample_count).to(device=device)
    )

    def forward() -> torch.Tensor:
        if batch_metadata is None or batch_metadata.is_empty:
            return model(batch)
        return model(batch, **batch_metadata.model_kwargs())

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(5):
                forward()
            if device == "cuda":
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    forward()
                end.record()
                torch.cuda.synchronize()
                return float(start.elapsed_time(end) / 20.0)
            started = perf_counter()
            for _ in range(20):
                forward()
            return (perf_counter() - started) * 1_000.0 / 20.0
    finally:
        model.train(was_training)


def _timed_forward_iterations(
    model: nn.Module,
    batch: torch.Tensor,
    device: str,
    *,
    iterations: int,
) -> list[float]:
    if device == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        with torch.no_grad():
            for start, end in zip(starts, ends, strict=True):
                start.record()
                model(batch)
                end.record()
        torch.cuda.synchronize()
        return [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
    values: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            started = perf_counter()
            model(batch)
            values.append((perf_counter() - started) * 1_000.0)
    return values


def _timed_training_steps(
    model: nn.Module,
    batch: torch.Tensor,
    device: str,
    *,
    iterations: int,
) -> list[float]:
    model.train()
    for _ in range(10):
        model.zero_grad(set_to_none=True)
        model(batch).to(torch.float32).square().mean().backward()
    if device == "cuda":
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            model.zero_grad(set_to_none=True)
            model(batch).to(torch.float32).square().mean().backward()
            end.record()
        torch.cuda.synchronize()
        return [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
    values: list[float] = []
    for _ in range(iterations):
        started = perf_counter()
        model.zero_grad(set_to_none=True)
        model(batch).to(torch.float32).square().mean().backward()
        values.append((perf_counter() - started) * 1_000.0)
    return values


def _profile_flops(model: nn.Module, batch: torch.Tensor, device: str) -> tuple[float, str]:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with (
            torch.no_grad(),
            torch.profiler.profile(
                activities=activities,
                with_flops=True,
            ) as profile,
        ):
            model(batch)
        flops = float(sum(event.flops or 0 for event in profile.key_averages()))
    except (AssertionError, RuntimeError, ValueError) as error:
        return 0.0, f"unavailable:{type(error).__name__}"
    return flops, "measured" if flops > 0 else "unsupported_ops"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


@cache
def _code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _environment_metadata(device: str) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "execution_mode": "eager",
        "precision": "fp32",
    }


def _record_provenance(root: Path, device: str) -> tuple[str, dict[str, object]]:
    source_names = (
        "pac_alphabet_q1_q2_final_campaign.py",
        "pac_alphabet_q1_q2_final_cli.py",
        "pac_baseline_fairness_maximal.py",
        "pac_confirmatory_baselines.py",
        "pac_data_split.py",
        "pac_efp16_final_campaign.py",
        "pac_efp_compact_equal_search.py",
        "pac_efp_compact_external_equal_search.py",
        "pac_efp_writer_reader.py",
        "pac_external_benchmarks.py",
        "pac_external_reference_baselines.py",
        "pac_external_tasks.py",
        "pac_final_validation.py",
        "pac_full_early_q1_campaign.py",
        "pac_full_state_terminal_analyzer.py",
        "pac_headroom_efficient_models.py",
        "pac_laplace_native_input.py",
        "pac_raw_efficiency_candidates.py",
        "pac_tight_frame_models.py",
        "pac_training.py",
        "pac_types.py",
    )
    source_root = Path(__file__).resolve().parent
    source_hashes = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in source_names
    }
    package_versions: dict[str, str | None] = {}
    for distribution in ("torch", "mamba-ssm", "numpy", "scipy", "scikit-learn"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    payload: dict[str, object] = {
        "schema": "pac_baseline_fairness_provenance.v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "environment": _environment_metadata(device),
        "source_sha256": source_hashes,
        "package_versions": package_versions,
        "git_commit": _command_output(("git", "rev-parse", "HEAD")),
        "git_tracked_dirty": bool(
            _command_output(("git", "status", "--porcelain", "--untracked-files=no"))
        ),
        "nvidia_driver": _command_output(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        ),
        "argv": sys.argv,
    }
    digest = canonical_json_sha256(payload)
    write_once(
        root / "provenance" / f"{digest}.json",
        json.dumps({**payload, "provenance_sha256": digest}, indent=2, sort_keys=True) + "\n",
    )
    return digest, payload


def _command_output(arguments: tuple[str, ...]) -> str | None:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed internal command vectors only
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def _split_hash(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(b"sampled_tensor_split_fingerprint.v1")
    for tensor in tensors:
        values = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(values.shape)).encode())
        digest.update(str(values.dtype).encode())
        flattened = values.reshape(-1)
        if flattened.numel() <= 8_192:
            sample = flattened
        else:
            sample = torch.cat((flattened[:4_096], flattened[-4_096:]))
        digest.update(sample.numpy().tobytes())
        if values.is_floating_point():
            float_sample = sample.to(torch.float64)
            summary = torch.stack(
                (
                    float_sample.sum(),
                    float_sample.square().sum(),
                )
            )
            digest.update(summary.numpy().tobytes())
    return digest.hexdigest()


def _full_split_identity_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256(b"full_split_index_label_identity.v1")
    for tensor in tensors:
        values = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(values.shape)).encode())
        digest.update(str(values.dtype).encode())
        digest.update(memoryview(values.numpy()))
    return digest.hexdigest()


def _selection_artifact(root: Path, stage: str) -> Path | None:
    candidates = {
        "stage2": root / "stage1" / "selection.json",
        "final": root / "stage2" / "selection.json",
        "q2_calibration": root / "q2_calibration" / "contract.json",
        "q2_final": root / "q2_calibration" / "selection.json",
        "q3_bridge": root / "q3_bridge" / "contract.json",
    }
    path = candidates.get(stage)
    return path if path is not None and path.exists() else None


def _representative_rows(
    rows: Iterable[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped = _group_rows(rows)
    return {
        cell_key: min(cell_rows, key=lambda row: cast("int", row["train_seed"]))
        for cell_key, cell_rows in grouped.items()
    }


def _match_real_width(  # noqa: C901 - bounded monotone real-width search
    base: FairnessJob,
    metadata: dict[str, object],
    target_parameters: int,
) -> tuple[int, int, float] | None:
    max_width = 8_192
    candidates: dict[int, int] = {}

    def evaluate(width: int) -> int | None:
        if width in candidates:
            return candidates[width]
        try:
            model = _build_model_from_metadata(base, metadata, width)
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError):
            return None
        parameters = count_parameters(model)
        candidates[width] = parameters
        del model
        return parameters

    lower = 1
    lower_parameters = evaluate(lower)
    while lower_parameters is None and lower < max_width:
        lower *= 2
        lower_parameters = evaluate(lower)
    if lower_parameters is None:
        return None
    if lower_parameters < target_parameters:
        upper = min(2 * lower, max_width)
        while upper > lower:
            upper_parameters = evaluate(upper)
            if upper_parameters is None:
                if upper == max_width:
                    break
                upper = min(2 * upper, max_width)
                continue
            if upper_parameters >= target_parameters:
                while lower + 1 < upper:
                    middle = (lower + upper) // 2
                    middle_parameters = evaluate(middle)
                    if middle_parameters is None or middle_parameters >= target_parameters:
                        upper = middle
                    else:
                        lower = middle
                evaluate(lower)
                evaluate(upper)
                break
            if upper == max_width:
                break
            lower = upper
            upper = min(2 * upper, max_width)
    if not candidates:
        return None
    width, parameters = min(
        candidates.items(),
        key=lambda item: (abs(item[1] - target_parameters), item[0]),
    )
    error = abs(parameters - target_parameters) / max(target_parameters, 1)
    return None if error > PARAMETER_TOLERANCE else (width, parameters, error)


def _build_model_from_metadata(  # noqa: PLR0911 - explicit family dispatch is easier to audit.
    base: FairnessJob,
    metadata: dict[str, object],
    width: int,
) -> nn.Module:
    job = replace(base, width=width)
    config = _experiment_config(
        train_count=cast("int", metadata["train_count"]),
        validation_count=cast("int", metadata["validation_count"]),
        test_count=cast("int", metadata["test_count"]),
        sequence_length=cast("int", metadata["sequence_length"]),
        input_dim=cast("int", metadata["input_dim"]),
        output_dim=cast("int", metadata["output_dim"]),
        job=job,
        device="cpu",
    )
    output_dim = cast("int", metadata["output_dim"])
    if base.suite == "ucr":
        return _build_ucr_model(job, config, output_dim)
    objective = "regression" if metadata["objective"] == "forecasting" else "classification"
    if base.model in {
        "efp16",
        "efp_tuned",
        "compact_h_only",
        "two_tap_h_only",
        "h_compact_lag124",
        "h_compact_lag124_tied",
        "pa2wp",
    }:
        if base.model in {"efp16", "efp_tuned"}:
            return EdgeFramePAC(
                config,
                output_dim,
                modes=16 if base.model == "efp16" else base.modes,
                semi_orthogonal=True,
                objective=objective,
                model_dim=width,
                mode_divisor=2,
            )
        if base.model == "compact_h_only":
            return CompactEFPHOnlyTerminalPAC(
                config,
                output_dim,
                objective=objective,
            )
        if base.model == "two_tap_h_only":
            return LearnedTwoTapHOnlyTerminalPAC(
                config,
                output_dim,
                objective=objective,
            )
        if base.model == "h_compact_lag124":
            return HCompactLag124PAC(
                config,
                output_dim,
                objective=objective,
            )
        if base.model == "h_compact_lag124_tied":
            return HCompactLag124TiedPAC(
                config,
                output_dim,
                objective=objective,
            )
        return build_efficient_headroom_classifier(
            "PA2WP",
            config,
            output_dim,
            objective=objective,
        )
    return _build_continuous_model(
        cast("ExternalModelFamily", base.model),
        width,
        cast("int", metadata["input_dim"]),
        output_dim,
        config,
        "EFP16",
        objective=objective,
    )


def _write_manifests(
    root: Path,
    stage: str,
    jobs: list[FairnessJob],
    lanes: tuple[ResourceLane, ...],
    *,
    filename_prefix: str = "",
) -> dict[str, float]:
    assigned: dict[str, list[FairnessJob]] = {lane.name: [] for lane in lanes}
    raw_loads = dict.fromkeys(assigned, 0.0)
    lane_by_name = {lane.name: lane for lane in lanes}
    for job in sorted(jobs, key=lambda item: (-item.estimated_seconds, item.key)):
        target = min(
            lanes,
            key=lambda lane: (raw_loads[lane.name] / lane.relative_speed, lane.name),
        )
        assigned[target.name].append(job)
        raw_loads[target.name] += max(job.estimated_seconds, 1.0)
    manifest_root = root / stage / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for name, lane_jobs in assigned.items():
        lane = lane_by_name[name]
        payload = "".join(
            json.dumps(
                {"resource": asdict(lane), "job": asdict(job), "key": job.key},
                sort_keys=True,
            )
            + "\n"
            for job in sorted(lane_jobs, key=lambda item: (item.estimated_seconds, item.key))
        )
        write_once(manifest_root / f"{filename_prefix}{name}.jsonl", payload)
    return {lane.name: raw_loads[lane.name] / lane.relative_speed for lane in lanes}


def _manifest_job_keys(manifest_root: Path) -> set[str]:
    keys: set[str] = set()
    for manifest in manifest_root.glob("*.jsonl"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line:
                keys.add(str(json.loads(line)["key"]))
    return keys


def _require_complete_stage(root: Path, stage: str) -> list[dict[str, object]]:
    status = cast("dict[str, object]", campaign_status(root)[stage])
    if not status["done"]:
        message = (
            f"{stage} is incomplete: completed={status['completed']} expected={status['expected']} "
            f"unexpected_completed={status['unexpected_completed']} failed={status['failed']}"
        )
        raise RuntimeError(message)
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / stage / "completed").glob("*.json"))
    ]


def _group_rows(rows: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_key"]), []).append(row)
    return grouped


def _rank_row(row: dict[str, object]) -> tuple[float, str]:
    return (-cast("float", row["selection_score"]), str(row["config_key"]))


def _job_from_result(row: dict[str, object]) -> FairnessJob:
    return FairnessJob(
        stage=cast(
            "Literal['stage1', 'stage2', 'final', 'q2_calibration', 'q2_final', 'q3_bridge']",
            row["stage"],
        ),
        suite=cast("Literal['ucr', 'external']", row["suite"]),
        dataset=cast("str", row["dataset"]),
        model=cast("str", row["model"]),
        width_tier=cast("int", row["width_tier"]),
        width=cast("int", row["width"]),
        trial=cast("int", row["trial"]),
        split_seed=cast("int", row["split_seed"]),
        train_seed=cast("int", row["train_seed"]),
        epochs=cast("int", row["epochs"]),
        batch_size=cast("int", row["batch_size"]),
        learning_rate=cast("float", row["learning_rate"]),
        weight_decay=cast("float", row["weight_decay"]),
        grad_clip_norm=cast("float", row["grad_clip_norm"]),
        evaluation_split=cast("Literal['validation', 'test']", row["evaluation_split"]),
        estimated_seconds=cast("float", row["estimated_seconds"]),
        budget_multiplier=cast("float | None", row.get("budget_multiplier")),
        lr_multiplier=cast("float", row.get("lr_multiplier", 1.0)),
        target_parameters=cast("int | None", row.get("target_parameters")),
        relative_parameter_error=cast("float | None", row.get("relative_parameter_error")),
        latency_multiplier=cast("float | None", row.get("latency_multiplier")),
        target_latency_ms=cast("float | None", row.get("target_latency_ms")),
        modes=cast("int", row.get("modes", 16)),
    )


def _result_path(root: Path, job: FairnessJob, *, failed: bool) -> Path:
    bucket = "failed" if failed else "completed"
    safe = job.key.replace(":", "__").replace("/", "_")
    return root / job.stage / bucket / f"{safe}.json"


def _append_attempt_event(
    root: Path,
    stage: str,
    job_key: str,
    attempt_id: str,
    event: Literal["started", "failed", "succeeded", "abandoned"],
    details: dict[str, object],
) -> Path:
    safe = job_key.replace(":", "__").replace("/", "_")
    path = root / stage / "attempts" / safe / f"{attempt_id}.{event}.json"
    payload: dict[str, object] = {
        "schema": "pac_baseline_fairness_attempt_event.v1",
        "attempt_id": attempt_id,
        "job_key": job_key,
        "stage": stage,
        "event": event,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **details,
    }
    write_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _attempt_counts(root: Path, stage: str) -> dict[str, int]:
    counts = {"started": 0, "failed": 0, "succeeded": 0, "abandoned": 0, "unfinished": 0}
    by_attempt: dict[str, set[str]] = {}
    for path in (root / stage / "attempts").rglob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        event = str(row.get("event"))
        if event in {"started", "failed", "succeeded", "abandoned"}:
            counts[event] += 1
            by_attempt.setdefault(str(row["attempt_id"]), set()).add(event)
    counts["unfinished"] = sum(
        "started" in events and not ({"failed", "succeeded", "abandoned"} & events)
        for events in by_attempt.values()
    )
    return counts


def _job_keys_from_results(paths: Iterable[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if job_key := row.get("job_key"):
            keys.add(str(job_key))
    return keys


def _failed_attempt_counts_by_job(root: Path, stage: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in (root / stage / "attempts").rglob("*.failed.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        job_key = str(row["job_key"])
        counts[job_key] = counts.get(job_key, 0) + 1
    return counts


def _terminal_failure_job_keys(
    root: Path,
    stage: str,
    failed_keys: set[str] | None = None,
) -> set[str]:
    candidates = failed_keys
    if candidates is None:
        candidates = _job_keys_from_results((root / stage / "failed").glob("*.json"))
    attempts = _failed_attempt_counts_by_job(root, stage)
    return {
        job_key
        for job_key in candidates
        if attempts.get(job_key, 0) >= Q2_TERMINAL_FAILURE_ATTEMPTS
    }


def _terminal_failure_rows(root: Path, stage: str) -> list[dict[str, object]]:
    terminal_keys = _terminal_failure_job_keys(root, stage)
    rows: list[dict[str, object]] = []
    for path in sorted((root / stage / "failed").glob("*.json")):
        row = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        if str(row.get("job_key")) in terminal_keys:
            rows.append(row)
    return rows


def _unfinished_attempt_job_keys(root: Path, stage: str) -> list[str]:
    by_attempt: dict[str, tuple[str, set[str]]] = {}
    for path in (root / stage / "attempts").rglob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        attempt_id = str(row["attempt_id"])
        job_key = str(row["job_key"])
        if attempt_id not in by_attempt:
            by_attempt[attempt_id] = (job_key, set())
        by_attempt[attempt_id][1].add(str(row["event"]))
    return [
        job_key
        for job_key, events in by_attempt.values()
        if "started" in events and not ({"failed", "succeeded", "abandoned"} & events)
    ]


def _close_unfinished_attempts(root: Path, stage: str, job_key: str) -> None:
    safe = job_key.replace(":", "__").replace("/", "_")
    by_attempt: dict[str, set[str]] = {}
    for path in (root / stage / "attempts" / safe).glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        by_attempt.setdefault(str(row["attempt_id"]), set()).add(str(row["event"]))
    for attempt_id, events in by_attempt.items():
        if "started" in events and not ({"failed", "succeeded", "abandoned"} & events):
            _append_attempt_event(
                root,
                stage,
                job_key,
                attempt_id,
                "abandoned",
                {
                    "reason": "a later worker invocation found no terminal event",
                    "recovered_by_pid": os.getpid(),
                },
            )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_done(row: dict[str, object]) -> None:
    if row.get("status") != "done":
        message = f"job returned non-done status: {row.get('status')}"
        raise RuntimeError(message)
