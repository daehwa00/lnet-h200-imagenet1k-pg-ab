from __future__ import annotations

import fcntl
import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Final, cast

import torch

from .pac_overnight_io import append_csv_row, prepare_overnight_dirs, read_csv
from .pac_recommended_low_data_eval import run_low_data_job
from .pac_recommended_low_data_jobs import build_low_data_jobs
from .pac_recommended_low_data_report import write_low_data_report
from .pac_recommended_low_data_types import LowDataEvent, LowDataJob, LowDataQueueConfig
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from pathlib import Path

    from .pac_confirmatory_baselines import ConfirmatoryFamily
    from .tapped_prl_followup_schema import JsonRow

RESULT_FILE: Final[str] = "low_data_recommended_real.csv"


def sanity_config(config: LowDataQueueConfig) -> PACExperimentConfig:
    return PACExperimentConfig(
        24,
        12,
        12,
        24,
        raw_input_dim=1,
        output_dim=2,
        model_dim=4,
        modes=2,
        tap_kernel_size=5,
        fir_kernel_size=3,
        epochs=1,
        batch_size=8,
        seeds=(7,),
        device=config.device,
        output_dir=config.output_root,
        compile_mode=config.compile_mode,
        precision=config.precision,
        optimizer_mode=config.optimizer_mode,
    )


def full_config(config: LowDataQueueConfig) -> PACExperimentConfig:
    return PACExperimentConfig(
        2048,
        512,
        512,
        64,
        raw_input_dim=1,
        device=config.device,
        output_dir=config.output_root,
        compile_mode=config.compile_mode,
        precision=config.precision,
        optimizer_mode=config.optimizer_mode,
    )


def run_sanity(config: LowDataQueueConfig) -> None:
    prepare_overnight_dirs(config.output_root)
    enqueue_jobs(replace(config, preset="smoke", seeds=(7,)))
    run_workers(
        replace(config, workers=1, total_slots=2),
        max_jobs=6,
        experiment_config=sanity_config(config),
    )
    _event(config.output_root, LowDataEvent("sanity", "done"))


def enqueue_jobs(config: LowDataQueueConfig) -> None:
    prepare_overnight_dirs(config.output_root)
    jobs = build_low_data_jobs(config)
    manifest = config.output_root / "queue_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    _event(config.output_root, LowDataEvent("enqueue", "done", f"jobs={len(jobs)}"))


def enqueue_selected_test_jobs(config: LowDataQueueConfig) -> None:
    """Lock the completed validation winner and enqueue one untouched TEST pass."""
    selection_path = config.output_root / "reports" / "stiefel_validation_capacity_selection.json"
    if not selection_path.exists():
        message = "run the complete validation-capacity queue and report stage first"
        raise FileNotFoundError(message)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_model = selection.get("selected_model")
    if selection.get("status") != "complete" or not isinstance(selected_model, str):
        message = "capacity selection is incomplete; refusing to observe official TEST"
        raise ValueError(message)
    source_manifest = config.output_root / "queue_manifest.jsonl"
    source_jobs = _read_jobs(source_manifest)
    validation_jobs = [job for job in source_jobs if job.evaluation_split == "validation"]
    if not validation_jobs:
        message = "the active manifest is not a validation-capacity manifest"
        raise ValueError(message)
    jobs = tuple(
        LowDataJob(
            key=f"low_data:locked_test:{job.seed}:{selected_model}:{job.dataset}:{job.ratio}",
            seed=job.seed,
            model=selected_model,
            dataset=job.dataset,
            ratio=job.ratio,
            slots=job.slots,
            evaluation_split="test",
            refit_full_train=True,
            data_protocol="clean_stratified",
            restore_best_validation=False,
            evaluation_collection="development_13_ucr_locked_refit",
        )
        for job in validation_jobs
        if job.model == selected_model
    )
    source_manifest.write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    lock = {
        "schema_version": "pac_locked_capacity_test.v1",
        "selected_model": selected_model,
        "selection_artifact": "<local-path>",
        "jobs": len(jobs),
    }
    (config.output_root / "reports" / "stiefel_locked_test.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _event(
        config.output_root,
        LowDataEvent("enqueue-selected-test", "done", f"jobs={len(jobs)}"),
    )


def enqueue_unseen_final_jobs(
    config: LowDataQueueConfig,
    *,
    selection_root: Path,
    protocol_path: Path,
    datasets: tuple[str, ...] = (),
) -> None:
    """Freeze a validation-selected architecture before creating an unseen UCR collection."""
    from .pac_recommended_low_data_jobs import REAL_DYNAMICAL_DATASETS  # noqa: PLC0415

    protocol_bytes, protocol, locked_datasets = _read_confirmatory_protocol(
        protocol_path,
        config,
    )
    requested = tuple(dict.fromkeys(dataset.strip() for dataset in datasets if dataset.strip()))
    if requested and requested != locked_datasets:
        message = "requested final datasets do not exactly match the locked protocol"
        raise ValueError(message)
    normalized = locked_datasets
    if not normalized:
        message = "locked protocol contains no unseen final datasets"
        raise ValueError(message)
    overlap = sorted(set(normalized) & set(REAL_DYNAMICAL_DATASETS))
    if overlap:
        message = f"unseen final datasets overlap development collection: {overlap}"
        raise ValueError(message)
    selection_path = selection_root / "reports" / "stiefel_validation_capacity_selection.json"
    if not selection_path.exists():
        message = "a completed clean validation-selection artifact is required"
        raise FileNotFoundError(message)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_model = selection.get("selected_model")
    if selection.get("status") != "complete" or not isinstance(selected_model, str):
        message = "selection artifact is incomplete; refusing to create final evaluation jobs"
        raise ValueError(message)
    if selection.get("official_test_observed") is not False:
        message = "selection artifact does not certify an unobserved official TEST"
        raise ValueError(message)

    protocol_families = _protocol_families(protocol)
    tuning_path, selected_trials, tuning_reference_model = _read_baseline_selection(
        config.output_root,
        protocol_families,
        hashlib.sha256(protocol_bytes).hexdigest(),
    )
    if tuning_reference_model != selected_model:
        message = "baseline tuning reference does not match the selected PAC-TF capacity"
        raise ValueError(message)

    prepare_overnight_dirs(config.output_root)
    jobs = tuple(
        LowDataJob(
            key=f"low_data:unseen_final:{seed}:{family}:{dataset}:1.0",
            seed=seed,
            model=family,
            dataset=dataset,
            ratio=1.0,
            evaluation_split="test",
            refit_full_train=True,
            data_protocol="clean_stratified",
            restore_best_validation=False,
            evaluation_collection="unseen_final_ucr",
            baseline_family=family,
            reference_model=selected_model,
            validation_trial=int(selected_trials[family]["trial"]),
            architecture_metadata_json=_confirmatory_architecture_json(
                family, int(selected_trials[family]["trial"])
            ),
            refit_epochs=int(selected_trials[family]["refit_epochs"]),
            learning_rate=float(selected_trials[family]["learning_rate"]),
            weight_decay=float(selected_trials[family]["weight_decay"]),
        )
        for seed in config.seeds
        for dataset in normalized
        for family in protocol_families
    )
    manifest = config.output_root / "queue_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    lock = {
        "schema_version": "pac_unseen_final_collection.v1",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_path": "<local-path>",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "architecture_frozen": True,
        "selected_model": selected_model,
        "selection_artifact": "<local-path>",
        "development_datasets": list(REAL_DYNAMICAL_DATASETS),
        "unseen_datasets": list(normalized),
        "seeds": list(config.seeds),
        "parameter_match_relative_tolerance": 0.05,
        "model_specific_validation_trials": 6,
        "refit": "all_official_train",
        "normalization": "all_official_train_only",
        "refit_epoch_policy": (
            "per-family median best_epoch from the selected validation trial over "
            "dataset-by-seed runs; round half upward"
        ),
        "checkpoint_policy": "frozen full-TRAIN state_dict after selected refit_epochs",
        "checkpoint_directory": "<local-path>",
        "official_test_observed_at_enqueue": False,
        "jobs": len(jobs),
        "baseline_families": list(protocol_families),
        "baseline_selection_artifact": "<local-path>",
    }
    reports = config.output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "unseen_final_collection_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _event(config.output_root, LowDataEvent("enqueue-unseen-final", "done", f"jobs={len(jobs)}"))


def enqueue_unseen_validation_jobs(
    config: LowDataQueueConfig,
    *,
    selection_root: Path,
    protocol_path: Path,
) -> None:
    """Create six TRAIN-derived validation trials for every locked model family."""
    from .pac_confirmatory_baselines import (  # noqa: PLC0415
        CONFIRMATORY_FAMILIES,
        VALIDATION_TRIALS,
    )

    protocol_bytes, protocol, datasets = _read_confirmatory_protocol(protocol_path, config)
    protocol_families = _protocol_families(protocol)
    if protocol_families != CONFIRMATORY_FAMILIES:
        message = "protocol baseline families do not match the implemented nine-family contract"
        raise ValueError(message)
    selection_path = selection_root / "reports" / "stiefel_validation_capacity_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    reference_model = selection.get("selected_model")
    if selection.get("status") != "complete" or not isinstance(reference_model, str):
        message = "a complete PAC-TF capacity selection is required"
        raise ValueError(message)
    prepare_overnight_dirs(config.output_root)
    jobs = tuple(
        LowDataJob(
            key=f"low_data:unseen_validation:{trial}:{seed}:{family}:{dataset}:1.0",
            seed=seed,
            model=family,
            dataset=dataset,
            ratio=1.0,
            evaluation_split="validation",
            refit_full_train=False,
            data_protocol="clean_stratified",
            restore_best_validation=True,
            evaluation_collection="unseen_final_validation",
            baseline_family=family,
            reference_model=reference_model,
            validation_trial=trial,
            architecture_metadata_json=_confirmatory_architecture_json(family, trial),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        for trial, (learning_rate, weight_decay) in enumerate(VALIDATION_TRIALS, start=1)
        for seed in config.seeds
        for dataset in datasets
        for family in CONFIRMATORY_FAMILIES
    )
    manifest_text = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs)
    manifest = config.output_root / "queue_manifest.jsonl"
    manifest.write_text(manifest_text, encoding="utf-8")
    reports = config.output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    # The active manifest is intentionally replaced by the final TEST manifest only
    # after tuning is complete.  Preserve the TRAIN-derived tuning contract so later
    # report refreshes cannot accidentally turn a complete selection back into a
    # partial one after that stage transition.
    snapshot = reports / "unseen_final_validation_manifest.jsonl"
    snapshot.write_text(manifest_text, encoding="utf-8")
    validation_lock = {
        "schema_version": "pac_unseen_validation_collection.v1",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_path": "<local-path>",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "selection_artifact": "<local-path>",
        "evaluation_split": "official_train_stratified_validation",
        "official_test_observed_at_enqueue": False,
        "jobs": len(jobs),
        "manifest_snapshot": "<local-path>",
    }
    (reports / "unseen_final_validation_collection_lock.json").write_text(
        json.dumps(validation_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _event(
        config.output_root,
        LowDataEvent("enqueue-unseen-validation", "done", f"jobs={len(jobs)}"),
    )


def _read_baseline_selection(  # noqa: C901 - fail-closed selection validation
    root: Path,
    protocol_families: tuple[str, ...],
    protocol_sha256: str,
) -> tuple[Path, dict[str, dict[str, float | int]], str]:
    tuning_path = root / "reports" / "confirmatory_baseline_selection.json"
    if not tuning_path.exists():
        message = "complete six-trial baseline validation before final enqueue"
        raise FileNotFoundError(message)
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    if tuning.get("schema_version") != "pac_confirmatory_baseline_selection.v1":
        message = "baseline selection artifact has an unsupported schema"
        raise ValueError(message)
    if tuning.get("status") != "complete":
        message = "baseline validation trials are incomplete"
        raise ValueError(message)
    if tuning.get("protocol_sha256") != protocol_sha256:
        message = "baseline selection is not bound to the active locked protocol"
        raise ValueError(message)
    raw_trials = tuning.get("selected_trials")
    if not isinstance(raw_trials, dict):
        message = "baseline selection artifact has no selected trial map"
        raise TypeError(message)
    if set(raw_trials) != set(protocol_families):
        message = "baseline selection does not cover all locked protocol families"
        raise ValueError(message)
    selected_trials: dict[str, dict[str, float | int]] = {}
    for family, raw_trial in raw_trials.items():
        if not isinstance(family, str) or not isinstance(raw_trial, dict):
            message = "baseline selection trial entries must be objects keyed by family"
            raise TypeError(message)
        selected_trials[family] = {
            "trial": _json_int(raw_trial.get("trial")),
            "refit_epochs": _json_int(raw_trial.get("refit_epochs")),
            "learning_rate": _json_float(raw_trial.get("learning_rate")),
            "weight_decay": _json_float(raw_trial.get("weight_decay")),
        }
        if int(selected_trials[family]["refit_epochs"]) < 1:
            message = "baseline selection refit_epochs must be positive"
            raise ValueError(message)
        from .pac_confirmatory_baselines import confirmatory_trial_spec  # noqa: PLC0415

        spec = confirmatory_trial_spec(
            cast("ConfirmatoryFamily", family), int(selected_trials[family]["trial"])
        )
        if (
            float(selected_trials[family]["learning_rate"]) != spec.learning_rate
            or float(selected_trials[family]["weight_decay"]) != spec.weight_decay
        ):
            message = f"baseline optimizer differs from locked family trial for {family}"
            raise ValueError(message)
        from .pac_confirmatory_baselines import (  # noqa: PLC0415
            confirmatory_implementation_metadata,
        )

        expected_architecture = confirmatory_implementation_metadata(
            cast("ConfirmatoryFamily", family), int(selected_trials[family]["trial"])
        )
        if raw_trial.get("architecture") != expected_architecture:
            message = f"baseline architecture differs from locked family trial for {family}"
            raise ValueError(message)
    reference_model = tuning.get("reference_model")
    if not isinstance(reference_model, str):
        message = "baseline selection artifact has no unique reference model"
        raise TypeError(message)
    return tuning_path, selected_trials, reference_model


def _confirmatory_architecture_json(family: str, trial: int) -> str:
    from .pac_confirmatory_baselines import (  # noqa: PLC0415
        confirmatory_implementation_metadata,
    )

    return json.dumps(
        confirmatory_implementation_metadata(cast("ConfirmatoryFamily", family), trial),
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_confirmatory_protocol(
    protocol_path: Path,
    config: LowDataQueueConfig,
) -> tuple[bytes, dict[str, object], tuple[str, ...]]:
    protocol_bytes = protocol_path.read_bytes()
    protocol: dict[str, object] = json.loads(protocol_bytes)
    if protocol.get("locked_before_final_evaluation") is not True:
        message = "confirmatory protocol was not locked before final evaluation"
        raise ValueError(message)
    locked_datasets = _json_string_tuple(protocol.get("untouched_final_datasets"))
    locked_seeds = _json_int_tuple(protocol.get("seeds"))
    if locked_seeds != config.seeds:
        message = f"queue seeds {config.seeds} do not match locked seeds {locked_seeds}"
        raise ValueError(message)
    training_contract = protocol.get("shared_training_contract", {})
    if not isinstance(training_contract, dict):
        message = "shared_training_contract must be an object"
        raise TypeError(message)
    if _json_int(training_contract.get("model_specific_validation_trials")) != 6:
        message = "confirmatory protocol must lock six validation trials per model"
        raise ValueError(message)
    if _json_float(protocol.get("parameter_match_relative_tolerance")) != 0.05:
        message = "confirmatory protocol must lock a 5% parameter tolerance"
        raise ValueError(message)
    return protocol_bytes, protocol, locked_datasets


def _protocol_families(protocol: dict[str, object]) -> tuple[ConfirmatoryFamily, ...]:
    values = _json_string_tuple(protocol.get("baseline_families"))
    return cast("tuple[ConfirmatoryFamily, ...]", values)


def _json_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        message = "protocol value must be a list of strings"
        raise TypeError(message)
    return tuple(value)


def _json_int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        message = "protocol value must be a list of integers"
        raise TypeError(message)
    return tuple(value)


def _json_int(value: object) -> int:
    if not isinstance(value, int):
        message = "protocol value must be an integer"
        raise TypeError(message)
    return value


def _json_float(value: object) -> float:
    if not isinstance(value, int | float):
        message = "protocol value must be numeric"
        raise TypeError(message)
    return float(value)


def run_workers(
    config: LowDataQueueConfig,
    *,
    max_jobs: int | None = None,
    experiment_config: PACExperimentConfig | None = None,
) -> None:
    prepare_overnight_dirs(config.output_root)
    independent_streams = (
        config.device != "cpu" and torch.cuda.is_available() and config.compile_mode == "none"
    )
    worker_config = {
        "device": config.device,
        "compile_mode": config.compile_mode,
        "precision": config.precision,
        "optimizer_mode": config.optimizer_mode,
        "independent_cuda_streams": independent_streams,
        "max_jobs": max_jobs,
        "preset": config.preset,
        "seeds": config.seeds,
        "total_slots": config.total_slots,
        "workers": config.workers,
    }
    (config.output_root / "worker_config.json").write_text(
        json.dumps(worker_config, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = config.output_root / "queue_manifest.jsonl"
    if not manifest.exists():
        enqueue_jobs(config)
    pending = [job for job in _read_jobs(manifest) if job.key not in _done_keys(config.output_root)]
    run_config = experiment_config or full_config(config)
    if config.compile_mode != "none":
        _run_compiled_jobs_on_main_thread(
            pending,
            run_config,
            config.output_root,
            max_jobs=max_jobs,
        )
        write_low_data_report(config.output_root)
        return
    _prewarm_tight_frame_cuda(run_config, pending)
    active: dict[Future[tuple[LowDataJob, JsonRow | None]], int] = {}
    launched = 0
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        while pending or active:
            available = config.total_slots - sum(active.values())
            while pending and available > 0 and (max_jobs is None or launched < max_jobs):
                index = _next_fit_index(pending, available)
                if index is None:
                    break
                job = pending.pop(index)
                _event(config.output_root, LowDataEvent(job.key, "running"))
                stream = _job_stream(config.device)
                active[pool.submit(_execute_job, run_config, job, config.output_root, stream)] = (
                    job.slots
                )
                launched += 1
                available -= job.slots
            if not active:
                break
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                active.pop(future)
                job, row = future.result()
                _event(config.output_root, LowDataEvent(job.key, "done" if row else "failed"))
            if max_jobs is not None and launched >= max_jobs and not active:
                break
    write_low_data_report(config.output_root)


def _run_compiled_jobs_on_main_thread(
    pending: list[LowDataJob],
    config: PACExperimentConfig,
    root: Path,
    *,
    max_jobs: int | None,
) -> None:
    limit = len(pending) if max_jobs is None else min(max_jobs, len(pending))
    for job in pending[:limit]:
        _event(root, LowDataEvent(job.key, "running"))
        completed_job, row = _execute_compiled_job(config, job, root)
        _event(root, LowDataEvent(completed_job.key, "done" if row else "failed"))


def _execute_compiled_job(
    config: PACExperimentConfig, job: LowDataJob, root: Path
) -> tuple[LowDataJob, JsonRow | None]:
    try:
        return _execute_job(config, job, root, None)
    finally:
        torch.compiler.reset()


def _prewarm_tight_frame_cuda(config: PACExperimentConfig, jobs: list[LowDataJob]) -> None:
    if config.device == "cpu" or not torch.cuda.is_available():
        return
    from .pac_recommended_low_data_models import build_low_data_classifier  # noqa: PLC0415
    from .pac_stiefel_variants import variant_for_model  # noqa: PLC0415
    from .pac_tight_frame_models import TightFrameClassifier  # noqa: PLC0415

    model_names = sorted({job.model for job in jobs if variant_for_model(job.model) is not None})
    if not model_names:
        return
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        inputs = torch.zeros(2, 16, config.raw_input_dim, device="cuda")
        for model_name in model_names:
            model = build_low_data_classifier(model_name, config, class_count=2).cuda()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=config.precision == "bf16",
            ):
                model(inputs).sum().backward()
            if isinstance(model, TightFrameClassifier):
                # QR and matrix-exp first-use initialization is not thread-safe in the
                # CUDA linear-algebra lazy loader. Trigger both paths before workers.
                model.finalize_constraints()
            del model
        torch.cuda.synchronize()
    del inputs


def _execute_job(
    config: PACExperimentConfig,
    job: LowDataJob,
    root: Path,
    stream: torch.cuda.Stream | None,
) -> tuple[LowDataJob, JsonRow | None]:
    try:
        if stream is None:
            row = run_low_data_job(config, job)
        else:
            with torch.cuda.stream(stream):
                row = run_low_data_job(config, job)
            stream.synchronize()
        append_csv_row(root / "results" / RESULT_FILE, row)
    except Exception as error:  # noqa: BLE001 - one failed row must not stop the queue
        append_csv_row(root / "results" / RESULT_FILE, _failed_row(job, error))
        return job, None
    return job, row


def _job_stream(device: str) -> torch.cuda.Stream | None:
    if device == "cpu" or not torch.cuda.is_available():
        return None
    return torch.cuda.Stream()


def _read_jobs(path: Path) -> tuple[LowDataJob, ...]:
    return tuple(
        LowDataJob(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _done_keys(root: Path) -> set[str]:
    path = root / "queue_state.jsonl"
    keys: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "done":
                    keys.add(str(row.get("key")))
    for row in read_csv(root / "results" / RESULT_FILE):
        result_key = _completed_result_key(row)
        if result_key is not None:
            keys.add(result_key)
    return keys


def _completed_result_key(row: dict[str, str]) -> str | None:
    if row.get("status") != "done":
        return None
    job_key = row.get("job_key", "").strip()
    if job_key:
        return job_key
    seed = row.get("seed", "").strip()
    model = row.get("model", "").strip()
    dataset = row.get("dataset_or_task", "").strip()
    ratio = row.get("data_ratio", "").strip()
    if not all((seed, model, dataset, ratio)):
        return None
    try:
        normalized_seed = int(seed)
        normalized_ratio = float(ratio)
    except ValueError:
        return None
    return f"low_data:{normalized_seed}:{model}:{dataset}:{normalized_ratio}"


def _event(root: Path, event: LowDataEvent) -> None:
    path = root / "queue_state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _next_fit_index(jobs: list[LowDataJob], available: int) -> int | None:
    for index, job in enumerate(jobs):
        if job.slots <= available:
            return index
    return None


def _failed_row(job: LowDataJob, error: Exception) -> JsonRow:
    return {
        "job_key": job.key,
        "experiment_group": "recommended_low_data",
        "dataset_or_task": job.dataset,
        "seed": job.seed,
        "model": job.model,
        "data_ratio": job.ratio,
        "evaluation_split": job.evaluation_split,
        "refit_full_train": job.refit_full_train,
        "status": "failed",
        "notes": f"{type(error).__name__}: {error}",
    }
