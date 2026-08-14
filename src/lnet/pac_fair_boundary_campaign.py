from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final, TypedDict, cast

from .pac_confirmatory_baselines import (
    ConfirmatoryFamily,
    confirmatory_implementation_metadata,
    confirmatory_trial_spec,
)
from .pac_pa2wp_boundary_campaign import LOW_DATASETS, LOW_RATIOS, SEEDS
from .pac_tf_p1p2_types import P1P2Job, RatioOneFitPolicy

DEFAULT_ROOT: Final = Path(".omx/results/pac-fair-boundary-baselines-pro6000-20260713")
REFERENCE_MODEL: Final = "pac_headroom_phase_augmented_ensemble_wp_d64_m16"
SELECTION_PATH: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
MODELS: Final = (
    "tcn",
    "cnn1d",
    "gru",
    "lstm",
    "transformer",
    "mamba",
    "s4d",
    "inception_time",
)


class _FairJobFields(TypedDict):
    selection_trial: int
    architecture_metadata_json: str
    learning_rate: float
    weight_decay: float
    ratio_one_fit_policy: RatioOneFitPolicy
    parameter_match_tolerance: float


def fair_boundary_jobs(selection_path: Path = SELECTION_PATH) -> tuple[P1P2Job, ...]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected_trials"]

    def job_fields(model: str) -> _FairJobFields:
        row = selected[model]
        trial = int(row["trial"])
        family = cast("ConfirmatoryFamily", model)
        spec = confirmatory_trial_spec(family, trial)
        return {
            "selection_trial": trial,
            "architecture_metadata_json": json.dumps(
                confirmatory_implementation_metadata(family, trial),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "ratio_one_fit_policy": "optimization_fold_validation",
            "parameter_match_tolerance": 0.062,
        }

    jobs = [
        P1P2Job(
            key=f"fair_boundary:low_data:{model}:{dataset}:ratio{ratio:g}:seed{seed}",
            package="low_data",
            seed=seed,
            model=model,
            reference_model=REFERENCE_MODEL,
            dataset=dataset,
            ratio=ratio,
            slots=2,
            **job_fields(model),
        )
        for model in MODELS
        for dataset in LOW_DATASETS
        for ratio in LOW_RATIOS
        for seed in SEEDS
    ]
    jobs.extend(
        P1P2Job(
            key=f"fair_boundary:real_diagnostics:{model}:{dataset}:seed{seed}",
            package="real_diagnostics",
            seed=seed,
            model=model,
            reference_model=REFERENCE_MODEL,
            dataset=dataset,
            ratio=1.0,
            slots=2,
            **job_fields(model),
        )
        for model in MODELS
        for dataset in LOW_DATASETS
        for seed in SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"fair_boundary:real_domain_ood:{model}:mit_bih:seed{seed}",
            package="real_domain_ood",
            seed=seed,
            model=model,
            reference_model=REFERENCE_MODEL,
            dataset="mit-bih-ds1-ds2",
            slots=2,
            **job_fields(model),
        )
        for model in MODELS
        for seed in SEEDS
    )
    return tuple(jobs)


def enqueue_fair_boundary(
    root: Path = DEFAULT_ROOT,
    *,
    shard_count: int = 6,
    selection_path: Path = SELECTION_PATH,
) -> dict[str, object]:
    if shard_count < 1:
        message = "shard_count must be positive"
        raise ValueError(message)
    jobs = fair_boundary_jobs(selection_path)
    shards: list[list[P1P2Job]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for job in sorted(jobs, key=_weight, reverse=True):
        index = min(range(shard_count), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += _weight(job)
    for index, shard in enumerate(shards):
        active = root / "shards" / f"shard-{index:02d}"
        active.mkdir(parents=True, exist_ok=True)
        (active / "p1p2_manifest.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard),
            encoding="utf-8",
        )
    contract: dict[str, object] = {
        "schema": "pac_fair_boundary_baselines.v2",
        "reference_model": REFERENCE_MODEL,
        "models": list(MODELS),
        "datasets": list(LOW_DATASETS),
        "ratios": list(LOW_RATIOS),
        "seeds": list(SEEDS),
        "jobs": len(jobs),
        "low_data_jobs": sum(job.package == "low_data" for job in jobs),
        "corruption_jobs": sum(job.package == "real_diagnostics" for job in jobs),
        "patient_ood_jobs": sum(job.package == "real_domain_ood" for job in jobs),
        "parameter_target": "dataset-specific final dual-origin ALPHABET parameter count",
        "parameter_tolerance": 0.062,
        "parameter_tolerance_reason": (
            "nearest frozen GRU width differs by at most 6.18%; no dummy parameters added"
        ),
        "ratio_one_fit_policy": "optimization_fold_validation",
        "ratio_one_fit_policy_reason": (
            "match final PA2WP's TRAIN-derived 80/20 optimization/validation split"
        ),
        "selection_path": str(selection_path),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "shards": shard_count,
        "estimated_shard_loads": loads,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def fair_boundary_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    expected = {job.key for job in fair_boundary_jobs()}
    done: set[str] = set()
    failed: set[str] = set()
    for path in root.glob("shards/*/results/*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = row.get("job_key", "")
                if row.get("status") == "done":
                    done.add(key)
                elif row.get("status") == "failed":
                    failed.add(key)
    failed -= done
    return {
        "expected": len(expected),
        "completed": len(expected & done),
        "failed": len(expected & failed),
        "remaining": len(expected - done - failed),
        "done": expected <= done,
    }


def _weight(job: P1P2Job) -> float:
    if job.package == "real_domain_ood":
        return 12.0
    if job.package == "real_diagnostics":
        return 6.0
    return 1.0 + 8.0 * job.ratio
