# ruff: noqa: EM102, TRY003
"""Runnable follow-up breadth queue for protocol-complete missing datasets.

The active broad campaign is immutable and therefore cannot be enlarged while
workers are consuming its manifests.  This module defines a second campaign
containing only new datasets:

* the 83 UCR-2018 datasets absent from the active 45-dataset wave;
* prepared external tasks that do not already have a complete, compatible
  18 -> 6 -> 1 ten-model result in the corrected 27-task campaign.

It retains the same 18 -> 6 -> 1 validation-only selection contract and the
same ten implementation-ready model families as the active UCR comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from statistics import mean
from typing import Final, Literal, cast

from .pac_campaign_utils import write_once
from .pac_broad_benchmark_queue import (
    CONFIRMATION_SEEDS,
    FINAL_SEEDS,
    SEARCH_SEED,
    TOP_K,
    UCR_128,
    BenchmarkJob,
    CampaignCell,
    CandidateSpec,
    ModelSpec,
    candidate_specs,
    model_registry,
)

FOLLOWUP_ROOT: Final = Path(
    ".omx/results/alphabet-broad-new-datasets-3gpu-20260727"
)

ACTIVE_UCR_45: Final = (
    "ACSF1",
    "Adiac",
    "ArrowHead",
    "BME",
    "BeetleFly",
    "BirdChicken",
    "CBF",
    "Car",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "EOGHorizontalSignal",
    "Earthquakes",
    "FaceAll",
    "FordA",
    "FordB",
    "GunPoint",
    "InsectEPGRegularTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "Meat",
    "MelbournePedestrian",
    "MoteStrain",
    "Phoneme",
    "Plane",
    "PowerCons",
    "Rock",
    "ShapesAll",
    "SmoothSubspace",
    "StarLightCurves",
    "SwedishLeaf",
    "Trace",
    "TwoLeadECG",
    "UWaveGestureLibraryAll",
    "Wafer",
    "Worms",
)
NEW_UCR_83: Final = tuple(name for name in UCR_128 if name not in ACTIVE_UCR_45)


@dataclass(frozen=True, slots=True)
class FollowupDatasetSpec:
    key: str
    suite: Literal["regular", "external"]
    family: str
    endpoint: str
    estimated_seconds: float
    max_microbatch: int
    source_artifact: str

    @property
    def data_shard(self) -> str:
        return f"{self.suite}:{self.key}:{self.endpoint}"


PREPARED_EXTERNAL_TASKS: Final = (
    FollowupDatasetSpec(
        "ptb-xl", "external", "ecg", "multilabel-ecg", 1_200.0, 16, "ptb-xl.pt"
    ),
    FollowupDatasetSpec(
        "mit-bih", "external", "ecg", "beat-classification", 900.0, 32, "mit-bih.pt"
    ),
    FollowupDatasetSpec(
        "cwru", "external", "vibration", "bearing-fault-classification", 240.0, 32, "cwru.pt"
    ),
    FollowupDatasetSpec(
        "speech-commands",
        "external",
        "speech",
        "speech-classification",
        2_400.0,
        16,
        "speech-commands.pt",
    ),
    FollowupDatasetSpec(
        "pathfinder",
        "external",
        "long-range",
        "spatial-path-classification",
        3_600.0,
        16,
        "pathfinder.pt",
    ),
    FollowupDatasetSpec(
        "lra-listops",
        "external",
        "long-range",
        "hierarchical-sequence-classification",
        4_800.0,
        8,
        "lra-listops.pt",
    ),
    FollowupDatasetSpec(
        "lra-text",
        "external",
        "long-range",
        "document-classification",
        2_400.0,
        16,
        "lra-text.pt",
    ),
    FollowupDatasetSpec(
        "lra-retrieval",
        "external",
        "long-range",
        "sequence-pair-retrieval",
        7_200.0,
        1,
        "lra-retrieval.pt",
    ),
    FollowupDatasetSpec(
        "lra-image",
        "external",
        "long-range",
        "sequential-image-classification",
        3_000.0,
        16,
        "lra-image.pt",
    ),
    FollowupDatasetSpec(
        "sequential-mnist",
        "external",
        "sequential-image",
        "pixel-sequence-classification",
        1_500.0,
        16,
        "sequential-mnist.pt",
    ),
    FollowupDatasetSpec(
        "permuted-mnist",
        "external",
        "sequential-image",
        "permuted-pixel-classification",
        1_500.0,
        16,
        "permuted-mnist.pt",
    ),
    FollowupDatasetSpec(
        "sequential-cifar",
        "external",
        "sequential-image",
        "row-sequence-classification",
        1_200.0,
        32,
        "sequential-cifar.pt",
    ),
    FollowupDatasetSpec(
        "audioset-balanced",
        "external",
        "audio",
        "multilabel-audio",
        900.0,
        32,
        "audioset-balanced.pt",
    ),
)

REUSED_EXTERNAL_DATASETS: Final = frozenset(
    {
        "audioset-balanced",
        "cwru",
        "mit-bih",
        "ptb-xl",
        "sequential-cifar",
    }
)
EXCLUDED_SLOW_EXTERNAL_DATASETS: Final = frozenset(
    {
        "pathfinder",
        "lra-listops",
        "lra-text",
        "lra-retrieval",
        "lra-image",
        "speech-commands",
        "sequential-mnist",
        "permuted-mnist",
    }
)
REUSE_PROVENANCE: Final = {
    "alphabet": ".omx/results/alphabet-27task-corrected-path-recovery-20260726",
    "baselines": ".omx/results/alphabet-balanced-hpo-27task-20260725",
    "contract": "18 -> 6 -> 1; split seed 7; confirmation seeds 11/19; final seeds 23/31/43/47/59",
}
EXTERNAL_TASKS: Final = tuple(
    task
    for task in PREPARED_EXTERNAL_TASKS
    if task.key not in REUSED_EXTERNAL_DATASETS
    and task.key not in EXCLUDED_SLOW_EXTERNAL_DATASETS
)


@cache
def followup_datasets() -> tuple[FollowupDatasetSpec, ...]:
    ucr = tuple(
        FollowupDatasetSpec(
            name,
            "regular",
            "ucr-128-remainder",
            "classification",
            90.0,
            64,
            f"{name}/{name}_TRAIN.tsv",
        )
        for name in NEW_UCR_83
    )
    return (*ucr, *EXTERNAL_TASKS)


@cache
def followup_models() -> tuple[ModelSpec, ...]:
    return tuple(
        model
        for model in model_registry()["regular"]
        if model.implementation_ready and "balanced_hpo" in model.execution_backends
    )


@cache
def followup_cells() -> tuple[CampaignCell, ...]:
    return tuple(
        CampaignCell(
            suite=dataset.suite,
            dataset=dataset.key,
            endpoint=dataset.endpoint,
            model=model.key,
            runtime_profile=model.runtime_profile,
            blockers=(),
        )
        for dataset in followup_datasets()
        for model in followup_models()
    )


@cache
def _dataset_lookup() -> dict[tuple[str, str], FollowupDatasetSpec]:
    return {(dataset.suite, dataset.key): dataset for dataset in followup_datasets()}


@cache
def _model_lookup() -> dict[str, ModelSpec]:
    return {model.key: model for model in followup_models()}


def _job_class(seconds: float) -> Literal["short", "medium", "long"]:
    if seconds < 180.0:
        return "short"
    if seconds <= 1_200.0:
        return "medium"
    return "long"


def _microbatch(
    dataset: FollowupDatasetSpec,
    model: ModelSpec,
    width: int,
) -> tuple[int, int]:
    profile_limit = 16 if model.runtime_profile == "mamba" else 32 if width >= 128 else 64
    microbatch = min(dataset.max_microbatch, profile_limit)
    if microbatch not in {1, 2, 4, 8, 16, 32, 64}:
        raise ValueError(f"unsupported microbatch {microbatch} for {dataset.key}")
    return microbatch, 64 // microbatch


def _make_job(
    cell: CampaignCell,
    candidate: CandidateSpec,
    *,
    stage: Literal["stage1", "stage2", "final"],
    train_seed: int,
    candidate_rank: int,
) -> BenchmarkJob:
    dataset = _dataset_lookup()[(str(cell.suite), cell.dataset)]
    model = _model_lookup()[cell.model]
    capacity_factor = max(0.35, candidate.width / 64.0)
    stage_factor = {"stage1": 1.0, "stage2": 1.0, "final": 1.15}[stage]
    seconds = dataset.estimated_seconds * model.runtime_factor * capacity_factor * stage_factor
    microbatch, accumulation = _microbatch(dataset, model, candidate.width)
    official_test = stage == "final"
    return BenchmarkJob(
        key=(
            f"broad-new:{stage}:{dataset.suite}:{dataset.key}:{dataset.endpoint}:"
            f"{model.key}:{candidate.candidate_id}:split{train_seed}:seed{train_seed}"
        ),
        stage=stage,
        suite=dataset.suite,
        dataset=dataset.key,
        endpoint=dataset.endpoint,
        model=model.key,
        candidate_id=candidate.candidate_id,
        candidate_rank=candidate_rank,
        recipe=candidate.recipe,
        width=candidate.width,
        modes=candidate.modes,
        architecture=candidate.architecture,
        split_seed=train_seed,
        train_seed=train_seed,
        epochs=100 if dataset.suite == "regular" else 60,
        evaluation_split="test" if official_test else "validation",
        official_test_accessed=official_test,
        runtime_profile=model.runtime_profile,
        comparison_group=f"{stage}:{dataset.suite}:{dataset.key}:{dataset.endpoint}",
        data_shard=dataset.data_shard,
        job_class=_job_class(seconds),
        estimated_seconds=seconds,
        estimated_peak_memory_mb=(
            8_192
            if dataset.key == "lra-retrieval"
            else 4_096 + candidate.width * 32
            if dataset.suite == "external"
            else 2_048 + candidate.width * 32
        ),
        microbatch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        blockers=(),
    )


@cache
def stage1_jobs() -> tuple[BenchmarkJob, ...]:
    models = _model_lookup()
    return tuple(
        _make_job(
            cell,
            candidate,
            stage="stage1",
            train_seed=SEARCH_SEED,
            candidate_rank=candidate.ordinal,
        )
        for cell in followup_cells()
        for candidate in candidate_specs(models[cell.model])
    )


def stage2_jobs(selections: dict[str, tuple[str, ...]]) -> tuple[BenchmarkJob, ...]:
    models = _model_lookup()
    jobs: list[BenchmarkJob] = []
    for cell in followup_cells():
        selected = selections.get(cell.key)
        if selected is None or len(selected) != TOP_K or len(set(selected)) != TOP_K:
            raise ValueError(f"{cell.key} must select exactly {TOP_K} candidates")
        candidates = {
            candidate.candidate_id: candidate
            for candidate in candidate_specs(models[cell.model])
        }
        unknown = set(selected) - set(candidates)
        if unknown:
            raise ValueError(f"{cell.key} selected unknown candidates: {sorted(unknown)}")
        for rank, candidate_id in enumerate(selected, start=1):
            jobs.extend(
                _make_job(
                    cell,
                    candidates[candidate_id],
                    stage="stage2",
                    train_seed=seed,
                    candidate_rank=rank,
                )
                for seed in CONFIRMATION_SEEDS
            )
    return tuple(jobs)


def final_jobs(selections: dict[str, str]) -> tuple[BenchmarkJob, ...]:
    models = _model_lookup()
    jobs: list[BenchmarkJob] = []
    for cell in followup_cells():
        selected = selections.get(cell.key)
        candidates = {
            candidate.candidate_id: candidate
            for candidate in candidate_specs(models[cell.model])
        }
        if selected is None or selected not in candidates:
            raise ValueError(f"{cell.key} must freeze one known candidate")
        jobs.extend(
            _make_job(
                cell,
                candidates[selected],
                stage="final",
                train_seed=seed,
                candidate_rank=1,
            )
            for seed in FINAL_SEEDS
        )
    return tuple(jobs)


def expected_counts() -> dict[str, int]:
    datasets = followup_datasets()
    cells = followup_cells()
    return {
        "datasets": len(datasets),
        "new_ucr_datasets": len(NEW_UCR_83),
        "new_external_tasks": len(EXTERNAL_TASKS),
        "models": len(followup_models()),
        "cells": len(cells),
        "stage1": len(cells) * 18,
        "stage2": len(cells) * TOP_K * len(CONFIRMATION_SEEDS),
        "final": len(cells) * len(FINAL_SEEDS),
        "total_fits": len(cells) * (18 + TOP_K * len(CONFIRMATION_SEEDS) + len(FINAL_SEEDS)),
    }


def audit_registry() -> dict[str, object]:
    datasets = followup_datasets()
    cells = followup_cells()
    jobs = stage1_jobs()
    problems: list[str] = []
    if len(ACTIVE_UCR_45) != 45 or len(NEW_UCR_83) != 83:
        problems.append("active/new UCR partition is not 45/83")
    if set(ACTIVE_UCR_45).intersection(NEW_UCR_83):
        problems.append("follow-up UCR datasets overlap the active wave")
    if len(datasets) != len({(item.suite, item.key) for item in datasets}):
        problems.append("duplicate follow-up dataset")
    pending_external = {item.key for item in EXTERNAL_TASKS}
    if pending_external.intersection(REUSED_EXTERNAL_DATASETS):
        problems.append("follow-up external tasks overlap protocol-complete results")
    prepared_external = {item.key for item in PREPARED_EXTERNAL_TASKS}
    if not REUSED_EXTERNAL_DATASETS.issubset(prepared_external):
        problems.append("reuse ledger names an unknown prepared external task")
    if len(cells) != len({cell.key for cell in cells}):
        problems.append("duplicate follow-up cell")
    if len(jobs) != len({job.key for job in jobs}):
        problems.append("duplicate Stage-1 key")
    if any(job.official_test_accessed or job.evaluation_split != "validation" for job in jobs):
        problems.append("Stage 1 accesses held-out TEST")
    if len(followup_models()) != 10:
        problems.append(f"expected ten runnable model families, got {len(followup_models())}")
    return {
        "schema": "alphabet.broad_new_datasets.registry_audit.v1",
        "ok": not problems,
        "problems": problems,
        "counts": expected_counts(),
    }


def write_campaign_contract(root: Path = FOLLOWUP_ROOT) -> dict[str, object]:
    audit = audit_registry()
    if not audit["ok"]:
        raise RuntimeError(f"follow-up registry audit failed: {audit['problems']}")
    payload: dict[str, object] = {
        "schema": "alphabet.broad_new_datasets.contract.v1",
        "state": "queued_waiting_for_active_stage1",
        "architecture": {
            "name": "ALPHABET",
            "implementation": "lnet.alphabet.Alphabet",
            "descriptor": "writer-reader radial-log R(0,1,2,4)",
            "head": "affine",
        },
        "scope": {
            "policy": (
                "missing protocol-complete dataset/model cells only; exclude compatible "
                "completed results before queue generation"
            ),
            "new_ucr": list(NEW_UCR_83),
            "new_external": [asdict(dataset) for dataset in EXTERNAL_TASKS],
            "reused_completed_external": sorted(REUSED_EXTERNAL_DATASETS),
            "reuse_provenance": REUSE_PROVENANCE,
        },
        "models": [asdict(model) for model in followup_models()],
        "search": {
            "stage1_candidates_per_cell": 18,
            "stage2_top_k": TOP_K,
            "stage1_seed": SEARCH_SEED,
            "stage2_seeds": list(CONFIRMATION_SEEDS),
            "final_seeds": list(FINAL_SEEDS),
            "split_seed_policy": "split_seed equals train_seed at every stage",
            "test_access": "final only after configuration freeze",
        },
        "expected": expected_counts(),
        "audit": audit,
    }
    payload["registry_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "datasets": [asdict(dataset) for dataset in followup_datasets()],
                "models": payload["models"],
                "cells": [asdict(cell) for cell in followup_cells()],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    write_once(root / "contract.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_stage_master(root, "stage1", stage1_jobs())
    return payload


def write_stage_master(
    root: Path,
    stage: Literal["stage1", "stage2", "final"],
    jobs: tuple[BenchmarkJob, ...],
) -> Path:
    if any(job.stage != stage for job in jobs):
        raise ValueError(f"{stage} master received another stage")
    path = root / stage / "master.jsonl"
    write_once(
        path,
        "".join(json.dumps(job.payload(), sort_keys=True) + "\n" for job in jobs),
    )
    return path


def _completed_rows(root: Path, stage: str) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in sorted((root / stage / "completed").glob("*.json")):
        try:
            row = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("status") == "done" and isinstance(row.get("job_key"), str):
            rows[str(row["job_key"])] = row
    return rows


def _score(row: dict[str, object]) -> float:
    value = float(cast("str | int | float", row["selection_score"]))
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite selection score for {row.get('job_key')}")
    return value


def _require_rows(
    root: Path,
    stage: Literal["stage1", "stage2"],
    jobs: tuple[BenchmarkJob, ...],
) -> dict[str, dict[str, object]]:
    rows = _completed_rows(root, stage)
    expected = {job.key for job in jobs}
    missing = expected - rows.keys()
    unexpected = rows.keys() - expected
    if missing or unexpected:
        message = "".join(
            (
                f"{stage} is not selection-ready: expected={len(expected)}, ",
                f"complete={len(expected) - len(missing)}, missing={len(missing)}, ",
                f"unexpected={len(unexpected)}",
            )
        )
        raise RuntimeError(message)
    return rows


def select_stage1(root: Path = FOLLOWUP_ROOT) -> dict[str, object]:
    jobs = stage1_jobs()
    rows = _require_rows(root, "stage1", jobs)
    grouped: dict[str, list[tuple[BenchmarkJob, float]]] = {}
    for job in jobs:
        grouped.setdefault(job.cell_key, []).append((job, _score(rows[job.key])))
    selected = {
        cell: tuple(
            job.candidate_id
            for job, _ in sorted(values, key=lambda item: (-item[1], item[0].candidate_id))[:TOP_K]
        )
        for cell, values in sorted(grouped.items())
    }
    payload: dict[str, object] = {
        "schema": "alphabet.broad_new_datasets.stage1_selection.v1",
        "official_test_accessed": False,
        "selection_rule": "descending validation score; lexical candidate-id tie break",
        "selected": {cell: list(configs) for cell, configs in selected.items()},
    }
    write_once(
        root / "stage1" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    write_stage_master(root, "stage2", stage2_jobs(selected))
    return payload


def select_stage2(root: Path = FOLLOWUP_ROOT) -> dict[str, object]:
    stage1 = stage1_jobs()
    selection = cast(
        "dict[str, list[str]]",
        json.loads((root / "stage1" / "selection.json").read_text(encoding="utf-8"))[
            "selected"
        ],
    )
    selected_stage1 = {cell: tuple(configs) for cell, configs in selection.items()}
    stage2 = stage2_jobs(selected_stage1)
    rows1 = _require_rows(root, "stage1", stage1)
    rows2 = _require_rows(root, "stage2", stage2)
    by_cell_config: dict[tuple[str, str], list[float]] = {}
    for job in (*stage1, *stage2):
        if job.candidate_id not in selected_stage1[job.cell_key]:
            continue
        row = rows1[job.key] if job.stage == "stage1" else rows2[job.key]
        by_cell_config.setdefault((job.cell_key, job.candidate_id), []).append(_score(row))
    frozen: dict[str, str] = {}
    evidence: dict[str, object] = {}
    for cell, configs in sorted(selected_stage1.items()):
        ranked: list[tuple[float, str]] = []
        for config in configs:
            scores = by_cell_config.get((cell, config), [])
            if len(scores) != 1 + len(CONFIRMATION_SEEDS):
                raise RuntimeError(f"{cell}/{config} has {len(scores)} selection scores")
            ranked.append((mean(scores), config))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        frozen[cell] = ranked[0][1]
        evidence[cell] = [
            {"candidate_id": config, "mean_validation_score": score}
            for score, config in ranked
        ]
    payload: dict[str, object] = {
        "schema": "alphabet.broad_new_datasets.stage2_selection.v1",
        "official_test_accessed": False,
        "selection_rule": "mean validation score over seeds 7,11,19; lexical tie break",
        "selected": frozen,
        "evidence": evidence,
    }
    write_once(
        root / "stage2" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    write_stage_master(root, "final", final_jobs(frozen))
    return payload


def audit_campaign(root: Path = FOLLOWUP_ROOT) -> dict[str, object]:
    """Audit exact completion and the validation/test barrier for all stages."""
    jobs_by_stage = {
        "stage1": stage1_jobs(),
        "stage2": tuple(
            BenchmarkJob.from_payload(cast("dict[str, object]", json.loads(line)))
            for line in (root / "stage2" / "master.jsonl").read_text().splitlines()
            if line.strip()
        ),
        "final": tuple(
            BenchmarkJob.from_payload(cast("dict[str, object]", json.loads(line)))
            for line in (root / "final" / "master.jsonl").read_text().splitlines()
            if line.strip()
        ),
    }
    problems: list[str] = []
    stages: dict[str, object] = {}
    for stage, jobs in jobs_by_stage.items():
        rows = _completed_rows(root, stage)
        expected = {job.key for job in jobs}
        missing = expected - rows.keys()
        unexpected = rows.keys() - expected
        invalid_test_access = sum(
            bool(row.get("official_test_accessed")) != (stage == "final")
            or row.get("evaluation_split")
            != ("test" if stage == "final" else "validation")
            for key, row in rows.items()
            if key in expected
        )
        if missing:
            problems.append(f"{stage} is missing {len(missing)} results")
        if unexpected:
            problems.append(f"{stage} has {len(unexpected)} unexpected results")
        if invalid_test_access:
            problems.append(
                f"{stage} has {invalid_test_access} invalid test-access records"
            )
        stages[stage] = {
            "expected": len(expected),
            "completed": len(expected & rows.keys()),
            "missing": len(missing),
            "unexpected": len(unexpected),
            "invalid_test_access": invalid_test_access,
        }
    for path in (
        root / "stage1" / "selection.json",
        root / "stage2" / "selection.json",
    ):
        payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        if payload.get("official_test_accessed") is not False:
            problems.append(f"selection artifact accessed TEST: {path}")
    return {
        "schema": "alphabet.broad_new_datasets.campaign_audit.v1",
        "ok": not problems,
        "problems": problems,
        "stages": stages,
    }


__all__: Final = [
    "ACTIVE_UCR_45",
    "EXCLUDED_SLOW_EXTERNAL_DATASETS",
    "EXTERNAL_TASKS",
    "FOLLOWUP_ROOT",
    "NEW_UCR_83",
    "PREPARED_EXTERNAL_TASKS",
    "REUSED_EXTERNAL_DATASETS",
    "REUSE_PROVENANCE",
    "FollowupDatasetSpec",
    "audit_campaign",
    "audit_registry",
    "expected_counts",
    "final_jobs",
    "followup_cells",
    "followup_datasets",
    "followup_models",
    "select_stage1",
    "select_stage2",
    "stage1_jobs",
    "stage2_jobs",
    "write_campaign_contract",
    "write_stage_master",
]
