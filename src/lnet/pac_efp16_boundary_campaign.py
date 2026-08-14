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
from .pac_efp16_final_campaign import (
    PARAMETER_TOLERANCE as FINAL_PARAMETER_TOLERANCE,
)
from .pac_efp16_final_campaign import (
    UCR_PARAMETER_TOLERANCE_EXCEPTIONS,
    ucr_parameter_tolerance,
)
from .pac_pa2wp_boundary_campaign import LOW_DATASETS, LOW_RATIOS, SEEDS
from .pac_tf_p1p2_types import P1P2Job, RatioOneFitPolicy

DEFAULT_REFERENCE_ROOT: Final = Path(".omx/results/pac-efp16-boundary-reference-20260713")
DEFAULT_BASELINE_ROOT: Final = Path(".omx/results/pac-efp16-boundary-baselines-20260713")
DEFAULT_NATIVE_OPTIMAL_ROOT: Final = Path(".omx/results/pac-efp16-native-optimal-boundary-20260715")
STAGE2_SELECTION_PATH: Final = Path(
    ".omx/results/pac-baseline-fairness-maximal-20260714/stage2/selection.json"
)
FINAL_LOCK_PATH: Final = Path(".omx/results/pac-efp16-final-submission-20260713/contract.json")
ARCHITECTURE_SOURCE: Final = Path("src/lnet/pac_headroom_efficient_models.py")
SELECTION_PATH: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
MODEL: Final = "efp16_pac"
REFERENCE_MODEL: Final = "pac_headroom_edge_frame_parseval_d32_m16"
PUBLIC_MODEL: Final = "ALPHABET"
INTERNAL_SPEC: Final = "EFP16"
PREDECESSOR_INTERNAL_SPEC: Final = "PA2WP"
PARAMETER_TOLERANCE: Final = FINAL_PARAMETER_TOLERANCE
BASELINE_MODELS: Final = (
    "tcn",
    "cnn1d",
    "gru",
    "lstm",
    "transformer",
    "mamba",
    "s4d",
    "inception_time",
)
NATIVE_OPTIMAL_MODELS: Final = (
    "efp16",
    "tcn",
    "cnn1d",
    "gru",
    "lstm",
    "transformer",
    "mamba",
)
NATIVE_FINAL_SEEDS: Final = (23, 31, 43, 47, 59)
NATIVE_CAPACITY_POLICY: Final = (
    "task-specific Stage2-selected native width; parameter counts intentionally "
    "differ and are reported; no parameter matching, dummy parameters, or adapters"
)
BOUNDARY_CLASS_COUNTS: Final = {
    "CinCECGTorso": 4,
    "CricketX": 12,
    "Earthquakes": 2,
    "Phoneme": 39,
    "StarLightCurves": 3,
    "mit-bih-ds1-ds2": 5,
}
PARAMETER_TOLERANCE_EXCEPTION_MAP: Final = {
    f"{family}:class{class_count}": tolerance
    for (family, class_count), tolerance in sorted(UCR_PARAMETER_TOLERANCE_EXCEPTIONS.items())
}
PARAMETER_TOLERANCE_MAX: Final = max(
    PARAMETER_TOLERANCE,
    *UCR_PARAMETER_TOLERANCE_EXCEPTIONS.values(),
)
PARAMETER_MATCH_POLICY: Final = (
    "nearest real architecture width; no dummy parameters and no functional, "
    "temperature, or redundant capacity adapters"
)
PARAMETER_TOLERANCE_EXCEPTIONS_JSON: Final = json.dumps(
    PARAMETER_TOLERANCE_EXCEPTION_MAP,
    sort_keys=True,
    separators=(",", ":"),
)


class _BaselineJobFields(TypedDict):
    selection_trial: int
    architecture_metadata_json: str
    learning_rate: float
    weight_decay: float
    ratio_one_fit_policy: RatioOneFitPolicy
    parameter_match_tolerance: float
    parameter_match_policy: str
    parameter_match_default_tolerance: float
    parameter_match_max_tolerance: float
    parameter_match_exceptions_json: str


class _ReferenceJobFields(TypedDict):
    model: str
    reference_model: str
    slots: int
    learning_rate: float
    weight_decay: float
    ratio_one_fit_policy: RatioOneFitPolicy
    parameter_match_tolerance: float
    parameter_match_policy: str
    parameter_match_default_tolerance: float
    parameter_match_max_tolerance: float
    parameter_match_exceptions_json: str


class _Stage2SelectionRow(TypedDict):
    config_key: str
    trial: int
    width: int


class _NativeJobFields(TypedDict):
    model: str
    reference_model: str
    reference_model_dim: int
    selected_model_width: int
    selection_trial: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    grad_clip_norm: float
    selection_source: str
    selection_config_key: str
    selection_artifact_sha256: str
    ratio_one_fit_policy: RatioOneFitPolicy
    parameter_match_tolerance: None
    parameter_match_policy: str
    capacity_policy: str
    slots: int


def efp16_reference_jobs() -> tuple[P1P2Job, ...]:
    common: _ReferenceJobFields = {
        "model": MODEL,
        "reference_model": REFERENCE_MODEL,
        "slots": 2,
        "learning_rate": 3.0e-3,
        "weight_decay": 1.0e-4,
        "ratio_one_fit_policy": "optimization_fold_validation",
        "parameter_match_tolerance": PARAMETER_TOLERANCE,
        "parameter_match_policy": PARAMETER_MATCH_POLICY,
        "parameter_match_default_tolerance": PARAMETER_TOLERANCE,
        "parameter_match_max_tolerance": PARAMETER_TOLERANCE_MAX,
        "parameter_match_exceptions_json": PARAMETER_TOLERANCE_EXCEPTIONS_JSON,
    }
    jobs = [
        P1P2Job(
            key=f"efp16_boundary:low_data:{dataset}:ratio{ratio:g}:seed{seed}",
            package="low_data",
            seed=seed,
            dataset=dataset,
            ratio=ratio,
            **common,
        )
        for dataset in LOW_DATASETS
        for ratio in LOW_RATIOS
        for seed in SEEDS
    ]
    jobs.extend(
        P1P2Job(
            key=f"efp16_boundary:real_diagnostics:{dataset}:seed{seed}",
            package="real_diagnostics",
            seed=seed,
            dataset=dataset,
            ratio=1.0,
            **common,
        )
        for dataset in LOW_DATASETS
        for seed in SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"efp16_boundary:synthetic_ood:seed{seed}",
            package="synthetic_ood",
            seed=seed,
            synthetic_estimand="endpoint",
            **common,
        )
        for seed in SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"efp16_boundary:real_domain_ood:mit_bih:seed{seed}",
            package="real_domain_ood",
            seed=seed,
            dataset="mit-bih-ds1-ds2",
            **common,
        )
        for seed in SEEDS
    )
    return tuple(jobs)


def efp16_baseline_jobs(
    selection_path: Path = SELECTION_PATH,
) -> tuple[P1P2Job, ...]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected_trials"]

    def job_fields(model: str, class_count: int) -> _BaselineJobFields:
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
            "parameter_match_tolerance": ucr_parameter_tolerance(model, class_count),
            "parameter_match_policy": PARAMETER_MATCH_POLICY,
            "parameter_match_default_tolerance": PARAMETER_TOLERANCE,
            "parameter_match_max_tolerance": PARAMETER_TOLERANCE_MAX,
            "parameter_match_exceptions_json": PARAMETER_TOLERANCE_EXCEPTIONS_JSON,
        }

    jobs = [
        P1P2Job(
            key=(f"efp16_boundary_baseline:low_data:{model}:{dataset}:ratio{ratio:g}:seed{seed}"),
            package="low_data",
            seed=seed,
            model=model,
            reference_model=REFERENCE_MODEL,
            dataset=dataset,
            ratio=ratio,
            slots=2,
            **job_fields(model, BOUNDARY_CLASS_COUNTS[dataset]),
        )
        for model in BASELINE_MODELS
        for dataset in LOW_DATASETS
        for ratio in LOW_RATIOS
        for seed in SEEDS
    ]
    jobs.extend(
        P1P2Job(
            key=f"efp16_boundary_baseline:real_diagnostics:{model}:{dataset}:seed{seed}",
            package="real_diagnostics",
            seed=seed,
            model=model,
            reference_model=REFERENCE_MODEL,
            dataset=dataset,
            ratio=1.0,
            slots=2,
            **job_fields(model, BOUNDARY_CLASS_COUNTS[dataset]),
        )
        for model in BASELINE_MODELS
        for dataset in LOW_DATASETS
        for seed in SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"efp16_boundary_baseline:real_domain_ood:{model}:mit_bih:seed{seed}",
            package="real_domain_ood",
            seed=seed,
            model=model,
            reference_model=REFERENCE_MODEL,
            dataset="mit-bih-ds1-ds2",
            slots=2,
            **job_fields(model, BOUNDARY_CLASS_COUNTS["mit-bih-ds1-ds2"]),
        )
        for model in BASELINE_MODELS
        for seed in SEEDS
    )
    return tuple(jobs)


def efp16_native_optimal_jobs(
    selection_path: Path = STAGE2_SELECTION_PATH,
) -> tuple[P1P2Job, ...]:
    selected, selection_sha = _stage2_selected_rows(selection_path)
    jobs = [
        P1P2Job(
            key=(f"efp16_native_boundary:low_data:{model}:{dataset}:ratio{ratio:g}:seed{seed}"),
            package="low_data",
            seed=seed,
            dataset=dataset,
            ratio=ratio,
            **_native_job_fields(selected, selection_sha, "ucr", dataset, model),
        )
        for dataset in LOW_DATASETS
        for model in NATIVE_OPTIMAL_MODELS
        for ratio in LOW_RATIOS
        for seed in NATIVE_FINAL_SEEDS
    ]
    jobs.extend(
        P1P2Job(
            key=f"efp16_native_boundary:real_diagnostics:{model}:{dataset}:seed{seed}",
            package="real_diagnostics",
            seed=seed,
            dataset=dataset,
            ratio=1.0,
            **_native_job_fields(selected, selection_sha, "ucr", dataset, model),
        )
        for dataset in LOW_DATASETS
        for model in NATIVE_OPTIMAL_MODELS
        for seed in NATIVE_FINAL_SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"efp16_native_boundary:real_domain_ood:{model}:mit_bih:seed{seed}",
            package="real_domain_ood",
            seed=seed,
            dataset="mit-bih-ds1-ds2",
            **_native_job_fields(selected, selection_sha, "external", "mit-bih", model),
        )
        for model in NATIVE_OPTIMAL_MODELS
        for seed in NATIVE_FINAL_SEEDS
    )
    return tuple(jobs)


def _stage2_selected_rows(
    selection_path: Path,
) -> tuple[dict[str, _Stage2SelectionRow], str]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != "pac_baseline_fairness_stage2_selection.v1":
        message = "unexpected Stage2 selection schema"
        raise ValueError(message)
    return cast("dict[str, _Stage2SelectionRow]", selection["selected"]), _sha256(selection_path)


def _native_job_fields(
    selected: dict[str, _Stage2SelectionRow],
    selection_sha: str,
    suite: str,
    dataset: str,
    model: str,
) -> _NativeJobFields:
    selection_key = f"{suite}:{dataset}:{model}"
    reference_key = f"{suite}:{dataset}:efp16"
    row = selected.get(selection_key)
    reference = selected.get(reference_key)
    if not isinstance(row, dict) or not isinstance(reference, dict):
        message = f"missing frozen Stage2 selection: {selection_key}"
        raise TypeError(message)
    trial = row["trial"]
    family = cast("ConfirmatoryFamily", "pac_tf" if model == "efp16" else model)
    spec = confirmatory_trial_spec(family, trial)
    return {
        "model": MODEL if model == "efp16" else model,
        "reference_model": REFERENCE_MODEL,
        "reference_model_dim": reference["width"],
        "selected_model_width": row["width"],
        "selection_trial": trial,
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "batch_size": spec.batch_size,
        "grad_clip_norm": spec.grad_clip_norm,
        "selection_source": f"Stage1/Stage2 ID-only selection:{selection_key}",
        "selection_config_key": row["config_key"],
        "selection_artifact_sha256": selection_sha,
        "ratio_one_fit_policy": "optimization_fold_validation",
        "parameter_match_tolerance": None,
        "parameter_match_policy": "not_applicable_native_optimal_capacity",
        "capacity_policy": NATIVE_CAPACITY_POLICY,
        "slots": 2,
    }


def enqueue_efp16_reference(
    root: Path = DEFAULT_REFERENCE_ROOT,
    *,
    shard_count: int = 6,
    final_lock_path: Path = FINAL_LOCK_PATH,
) -> dict[str, object]:
    jobs = efp16_reference_jobs()
    loads = _write_shards(root, jobs, shard_count)
    provenance = _provenance(final_lock_path)
    contract: dict[str, object] = {
        "schema": "pac_efp16_boundary_reference.v1",
        **_shared_contract(provenance),
        "model": MODEL,
        "reference_model": REFERENCE_MODEL,
        "jobs": len(jobs),
        "low_data_jobs": sum(job.package == "low_data" for job in jobs),
        "real_corruption_ood_jobs": sum(job.package == "real_diagnostics" for job in jobs),
        "synthetic_ood_jobs": sum(job.package == "synthetic_ood" for job in jobs),
        "patient_disjoint_ood_jobs": sum(job.package == "real_domain_ood" for job in jobs),
        "shards": shard_count,
        "estimated_shard_loads": loads,
        "official_test_accessed_at_enqueue": False,
    }
    _write_contract(root, contract)
    return contract


def enqueue_efp16_baselines(
    root: Path = DEFAULT_BASELINE_ROOT,
    *,
    shard_count: int = 12,
    selection_path: Path = SELECTION_PATH,
    final_lock_path: Path = FINAL_LOCK_PATH,
) -> dict[str, object]:
    jobs = efp16_baseline_jobs(selection_path)
    loads = _write_shards(root, jobs, shard_count)
    provenance = _provenance(final_lock_path)
    contract: dict[str, object] = {
        "schema": "pac_efp16_boundary_baselines.v1",
        **_shared_contract(provenance),
        "reference_model": REFERENCE_MODEL,
        "models": list(BASELINE_MODELS),
        "jobs": len(jobs),
        "low_data_jobs": sum(job.package == "low_data" for job in jobs),
        "real_corruption_ood_jobs": sum(job.package == "real_diagnostics" for job in jobs),
        "patient_disjoint_ood_jobs": sum(job.package == "real_domain_ood" for job in jobs),
        "selection_path": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "capacity_policy": (
            "rematch every baseline to each task's EFP16 trainable-parameter target; "
            "do not reuse PA2WP-capacity baseline rows"
        ),
        "integer_width_gap_policy": PARAMETER_MATCH_POLICY,
        "shards": shard_count,
        "estimated_shard_loads": loads,
        "official_test_accessed_at_enqueue": False,
    }
    _write_contract(root, contract)
    return contract


def enqueue_efp16_native_optimal(
    root: Path = DEFAULT_NATIVE_OPTIMAL_ROOT,
    *,
    shard_count: int = 24,
    selection_path: Path = STAGE2_SELECTION_PATH,
    final_lock_path: Path = FINAL_LOCK_PATH,
) -> dict[str, object]:
    jobs = efp16_native_optimal_jobs(selection_path)
    loads = _write_shards(root, jobs, shard_count)
    provenance = _provenance(final_lock_path)
    contract: dict[str, object] = {
        "schema": "pac_efp16_native_optimal_boundary.v1",
        "public_model": PUBLIC_MODEL,
        "final_internal_spec": INTERNAL_SPEC,
        "architecture_family": {
            "analysis": "degree_normalized_full_rate_edge_frame",
            "modes": 16,
            "task_selected_model_dim": True,
            "pairing_boundary": False,
            "dual_origin_inference": False,
        },
        "provenance": provenance,
        "models": list(NATIVE_OPTIMAL_MODELS),
        "jobs": len(jobs),
        "low_data_jobs": sum(job.package == "low_data" for job in jobs),
        "real_corruption_ood_jobs": sum(job.package == "real_diagnostics" for job in jobs),
        "patient_disjoint_ood_jobs": sum(job.package == "real_domain_ood" for job in jobs),
        "seeds": list(NATIVE_FINAL_SEEDS),
        "selection_seeds": [7, 11, 19],
        "selection_final_seed_overlap": False,
        "low_data_datasets": list(LOW_DATASETS),
        "low_data_ratios": list(LOW_RATIOS),
        "selection_policy": (
            "task-specific Stage1/Stage2 native architecture and optimizer selection "
            "using equal search protocol and ID-only validation; OOD and official TEST "
            "are not observed"
        ),
        "capacity_policy": NATIVE_CAPACITY_POLICY,
        "selection_path": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "parameter_matching": False,
        "parameter_reporting": "exact trainable count in every result row",
        "ratio_one_fit_policy": "optimization_fold_validation",
        "evaluation_batching": "locked training batch size; no full-test forward",
        "excluded_models": ["s4d", "inception_time", "minirocket"],
        "shards": shard_count,
        "estimated_shard_loads": loads,
        "official_test_accessed_at_enqueue": False,
        "restart_safe": True,
    }
    _write_contract(root, contract)
    return contract


def efp16_reference_status(root: Path = DEFAULT_REFERENCE_ROOT) -> dict[str, object]:
    return _status(root, efp16_reference_jobs())


def efp16_baseline_status(
    root: Path = DEFAULT_BASELINE_ROOT,
    *,
    selection_path: Path = SELECTION_PATH,
) -> dict[str, object]:
    return _status(root, efp16_baseline_jobs(selection_path))


def efp16_native_optimal_status(
    root: Path = DEFAULT_NATIVE_OPTIMAL_ROOT,
    *,
    selection_path: Path = STAGE2_SELECTION_PATH,
) -> dict[str, object]:
    return _status(root, efp16_native_optimal_jobs(selection_path))


def _shared_contract(provenance: dict[str, object]) -> dict[str, object]:
    return {
        "public_model": PUBLIC_MODEL,
        "final_internal_spec": INTERNAL_SPEC,
        "predecessor_internal_spec": PREDECESSOR_INTERNAL_SPEC,
        "architecture": {
            "model_dim": 32,
            "modes": 16,
            "analysis": "degree_normalized_full_rate_edge_frame",
            "semi_orthogonal_projection": True,
            "pairing_boundary": False,
            "random_pair_origin_training": False,
            "dual_origin_inference": False,
        },
        "provenance": provenance,
        "seeds": list(SEEDS),
        "low_data_datasets": list(LOW_DATASETS),
        "low_data_ratios": list(LOW_RATIOS),
        "ratio_one_fit_policy": "optimization_fold_validation",
        "ratio_one_fit_policy_reason": (
            "same official-TRAIN-derived 80/20 optimization/validation split for "
            "EFP16 and every baseline"
        ),
        "parameter_match_tolerance": PARAMETER_TOLERANCE,
        "parameter_match_tolerance_max": PARAMETER_TOLERANCE_MAX,
        "parameter_match_tolerance_exceptions": PARAMETER_TOLERANCE_EXCEPTION_MAP,
        "parameter_match_policy": PARAMETER_MATCH_POLICY,
        "fairness": {
            "optimization_data": "official TRAIN only",
            "fit_policy": "optimization_fold_validation",
            "capacity_target": "task-specific EFP16 trainable-parameter count",
            "capacity_tolerance": PARAMETER_TOLERANCE,
            "capacity_tolerance_max": PARAMETER_TOLERANCE_MAX,
            "capacity_tolerance_exceptions": PARAMETER_TOLERANCE_EXCEPTION_MAP,
            "capacity_matching_policy": PARAMETER_MATCH_POLICY,
            "baseline_selection": "pre-existing split-aligned five-seed UCR-18 validation",
            "test_driven_tuning": False,
            "pa2wp_capacity_rows_reused": False,
        },
        "restart_safe": True,
        "restart_policy": "successful job keys are skipped; failed keys remain retryable",
    }


def _provenance(final_lock_path: Path) -> dict[str, object]:
    lock = json.loads(final_lock_path.read_text(encoding="utf-8"))
    if lock.get("public_model") != PUBLIC_MODEL:
        message = "final lock does not name public model ALPHABET"
        raise ValueError(message)
    if lock.get("final_internal_spec") != INTERNAL_SPEC:
        message = "final lock does not freeze EFP16"
        raise ValueError(message)
    if lock.get("predecessor_internal_spec") != PREDECESSOR_INTERNAL_SPEC:
        message = "final lock does not preserve PA2WP predecessor provenance"
        raise ValueError(message)
    architecture = cast("dict[str, object]", lock.get("architecture", {}))
    if architecture.get("pairing_boundary") is not False:
        message = "final lock does not remove the pairing boundary"
        raise ValueError(message)
    if architecture.get("dual_origin_inference") is not False:
        message = "final lock unexpectedly enables dual-origin inference"
        raise ValueError(message)
    expected_source_sha = lock.get("architecture_source_sha256")
    source_sha = _sha256(ARCHITECTURE_SOURCE)
    if expected_source_sha != source_sha:
        message = "EFP16 architecture source differs from the final lock"
        raise ValueError(message)
    return {
        "final_lock_path": str(final_lock_path),
        "final_lock_sha256": _sha256(final_lock_path),
        "architecture_source": str(ARCHITECTURE_SOURCE),
        "architecture_source_sha256": source_sha,
    }


def _write_shards(
    root: Path,
    jobs: tuple[P1P2Job, ...],
    shard_count: int,
) -> list[float]:
    if shard_count < 1:
        message = "shard_count must be positive"
        raise ValueError(message)
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
    return loads


def _status(root: Path, jobs: tuple[P1P2Job, ...]) -> dict[str, object]:
    expected = {job.key for job in jobs}
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


def _write_contract(root: Path, contract: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _weight(job: P1P2Job) -> float:
    if job.package == "real_domain_ood":
        return 12.0
    if job.package == "real_diagnostics":
        return 6.0
    if job.package == "synthetic_ood":
        return 8.0
    return 1.0 + 8.0 * job.ratio


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
