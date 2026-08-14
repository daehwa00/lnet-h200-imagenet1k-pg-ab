"""Sealed Stage-1/2 Q1 search for S5, LRU, and DSS baselines."""

# pyright: reportPrivateUsage=false
# ruff: noqa: EM101, EM102, SLF001, TRY003

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Final, Literal, cast

from . import pac_baseline_fairness_maximal as runner
from .pac_baseline_fairness_maximal import FairnessJob, ResourceLane
from .pac_campaign_utils import canonical_json_sha256, file_sha256, write_once
from .pac_confirmatory_baselines import build_confirmatory_family, confirmatory_trial_spec
from .pac_efp16_final_campaign import EXTERNAL_DATASETS, UCR_DATASETS
from .pac_final_validation import EXTERNAL_SECONDS, UCR_SECONDS

if TYPE_CHECKING:
    from torch import nn

    from .pac_confirmatory_baselines import ConfirmatoryFamily
    from .pac_external_tasks import ExternalSelectionTask, ExternalTask
    from .pac_types import PACExperimentConfig


DEFAULT_ROOT: Final = Path(".omx/results/pac-additional-ssm-q1-search-20260722")
MODELS: Final = ("s5", "lru", "dss")
MODEL_WIDTHS: Final = dict.fromkeys(MODELS, (32, 64, 128))
SEARCH_SEED: Final = 7
CONFIRMATION_SEEDS: Final = (11, 19)
TRIALS: Final = tuple(range(1, 7))
TOP_K: Final = 6
STAGES: Final = ("stage1", "stage2")


def default_lanes(count: int = 14) -> tuple[ResourceLane, ...]:
    if count < 1:
        raise ValueError("lane count must be positive")
    return tuple(
        ResourceLane(f"worker-{index:02d}", "pro6000", 0, index, 1.0) for index in range(count)
    )


def _datasets(suite: Literal["ucr", "external"]) -> tuple[str, ...]:
    return UCR_DATASETS if suite == "ucr" else EXTERNAL_DATASETS


def _job(
    *,
    stage: Literal["stage1", "stage2"],
    suite: Literal["ucr", "external"],
    dataset: str,
    model: str,
    width_tier: int,
    width: int,
    trial: int,
    seed: int,
) -> FairnessJob:
    recipe = confirmatory_trial_spec(cast("ConfirmatoryFamily", model), trial)
    seconds = UCR_SECONDS[dataset] if suite == "ucr" else EXTERNAL_SECONDS[dataset]
    model_factor = {"s5": 1.8, "lru": 1.6, "dss": 1.4}[model]
    return FairnessJob(
        stage=stage,
        suite=suite,
        dataset=dataset,
        model=model,
        width_tier=width_tier,
        width=width,
        trial=trial,
        split_seed=seed,
        train_seed=seed,
        epochs=100 if suite == "ucr" else 60,
        batch_size=recipe.batch_size,
        learning_rate=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        grad_clip_norm=recipe.grad_clip_norm,
        evaluation_split="validation",
        estimated_seconds=seconds * (width / 64.0) * model_factor,
    )


def stage1_jobs() -> list[FairnessJob]:
    return [
        _job(
            stage="stage1",
            suite=suite,
            dataset=dataset,
            model=model,
            width_tier=width_tier,
            width=width,
            trial=trial,
            seed=SEARCH_SEED,
        )
        for suite in cast("tuple[Literal['ucr', 'external'], ...]", ("ucr", "external"))
        for dataset in _datasets(suite)
        for model in MODELS
        for width_tier, width in enumerate(MODEL_WIDTHS[model], start=1)
        for trial in TRIALS
    ]


def _source_manifest() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    names = (
        "src/lnet/pac_additional_ssm_baselines.py",
        "src/lnet/pac_additional_ssm_q1_campaign.py",
        "src/lnet/pac_additional_ssm_q1_cli.py",
        "src/lnet/pac_baseline_fairness_maximal.py",
        "src/lnet/pac_confirmatory_baselines.py",
        "src/lnet/pac_training.py",
    )
    hashes = {name: file_sha256(project / name) for name in names}
    body: dict[str, object] = {
        "schema": "pac_additional_ssm_q1_source_manifest.v1",
        "source_sha256": hashes,
        "models": list(MODELS),
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def enqueue_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    jobs = stage1_jobs()
    expected = 30 * len(MODELS) * 18
    if len(jobs) != expected or len({job.key for job in jobs}) != expected:
        raise RuntimeError(f"Stage 1 must contain {expected} unique jobs")
    active_lanes = default_lanes() if lanes is None else lanes
    loads = runner._write_manifests(root, "stage1", jobs, active_lanes)
    source = _source_manifest()
    write_once(
        root / "reports/source_manifest.json",
        json.dumps(source, indent=2, sort_keys=True) + "\n",
    )
    contract: dict[str, object] = {
        "schema": "pac_additional_ssm_q1_search.v1",
        "scope": "TRAIN-derived validation only; Stage 1 and Stage 2",
        "models": list(MODELS),
        "tasks": 30,
        "width_ladders": {model: list(widths) for model, widths in MODEL_WIDTHS.items()},
        "architectures_per_width": len(TRIALS),
        "evaluations_per_model_task": 18,
        "stage1": {"seed": SEARCH_SEED, "jobs": len(jobs)},
        "stage2": {
            "top_k": TOP_K,
            "additional_seeds": list(CONFIRMATION_SEEDS),
            "expected_jobs": 30 * len(MODELS) * TOP_K * len(CONFIRMATION_SEEDS),
        },
        "official_test_accessed": False,
        "source_manifest_sha256": source["sha256"],
        "restart_safe": True,
        "resource_lanes": [lane.name for lane in active_lanes],
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "stage1/contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def select_stage1(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    rows = runner._require_complete_stage(root, "stage1")
    grouped = runner._group_rows(rows)
    expected_cells = 30 * len(MODELS)
    if len(grouped) != expected_cells:
        raise RuntimeError(f"Stage 1 has {len(grouped)} cells; expected {expected_cells}")
    selected: dict[str, list[str]] = {}
    jobs: list[FairnessJob] = []
    for cell_key, cell_rows in sorted(grouped.items()):
        if len(cell_rows) != 18:
            raise RuntimeError(f"{cell_key} has {len(cell_rows)} rows; expected 18")
        top = sorted(cell_rows, key=runner._rank_row)[:TOP_K]
        selected[cell_key] = [str(row["config_key"]) for row in top]
        for row in top:
            base = runner._job_from_result(row)
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
    loads = runner._write_manifests(root, "stage2", jobs, active_lanes)
    payload: dict[str, object] = {
        "schema": "pac_additional_ssm_stage1_selection.v1",
        "source_rows": len(rows),
        "cells": len(grouped),
        "top_k": TOP_K,
        "selected": selected,
        "stage2_jobs": len(jobs),
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "stage1/selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def select_stage2(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    output = root / "stage2/selection.json"
    if output.exists():
        return cast("dict[str, object]", json.loads(output.read_text(encoding="utf-8")))
    stage1 = runner._require_complete_stage(root, "stage1")
    stage2 = runner._require_complete_stage(root, "stage2")
    stage1_selection = json.loads((root / "stage1/selection.json").read_text(encoding="utf-8"))
    by_cell_config: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in stage1 + stage2:
        key = (str(row["cell_key"]), str(row["config_key"]))
        by_cell_config.setdefault(key, []).append(row)
    selected: dict[str, dict[str, object]] = {}
    for cell_key, config_keys in sorted(stage1_selection["selected"].items()):
        candidates: list[tuple[float, str, list[dict[str, object]]]] = []
        for config_key in config_keys:
            rows = by_cell_config[(cell_key, config_key)]
            seeds = {int(cast("int", row["train_seed"])) for row in rows}
            if len(rows) != 3 or seeds != {SEARCH_SEED, *CONFIRMATION_SEEDS}:
                raise RuntimeError(f"{cell_key}/{config_key} has invalid seeds {sorted(seeds)}")
            candidates.append(
                (
                    mean(float(cast("float", row["selection_score"])) for row in rows),
                    config_key,
                    rows,
                )
            )
        score, config_key, rows = min(candidates, key=lambda item: (-item[0], item[1]))
        base = runner._job_from_result(rows[0])
        selected[cell_key] = {
            "model": base.model,
            "config_key": config_key,
            "width_tier": base.width_tier,
            "width": base.width,
            "trial": base.trial,
            "mean_selection_score": score,
            "selection_seeds": [SEARCH_SEED, *CONFIRMATION_SEEDS],
        }
    payload: dict[str, object] = {
        "schema": "pac_additional_ssm_stage2_selection.v1",
        "cells": len(selected),
        "selected": selected,
        "official_test_accessed": False,
        "ready_for_q1_final": True,
    }
    write_once(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def install_runner_hooks() -> None:
    original_ucr = runner._build_ucr_model
    original_external = runner._build_external_model

    def build_ucr(job: FairnessJob, config: PACExperimentConfig, output_dim: int) -> nn.Module:
        if job.model in MODELS:
            return build_confirmatory_family(
                cast("ConfirmatoryFamily", job.model),
                job.width,
                config,
                output_dim,
                validation_trial=job.trial,
                input_dim=config.raw_input_dim,
            )
        return original_ucr(job, config, output_dim)

    def build_external(
        job: FairnessJob,
        config: PACExperimentConfig,
        task: ExternalTask | ExternalSelectionTask,
    ) -> nn.Module:
        if job.model in MODELS:
            return build_confirmatory_family(
                cast("ConfirmatoryFamily", job.model),
                job.width,
                config,
                task.output_dim,
                validation_trial=job.trial,
                input_dim=task.input_dim,
            )
        return original_external(job, config, task)

    runner._build_ucr_model = build_ucr
    runner._build_external_model = build_external


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    full = runner.campaign_status(root)
    return {"schema": "pac_additional_ssm_q1_status.v1", **{stage: full[stage] for stage in STAGES}}
