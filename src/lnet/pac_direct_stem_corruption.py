"""Validation-frozen corruption comparison for the final radial-log ALPHABET."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import math
import os
import platform
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Final, Literal, cast

import torch

from .alphabet import Alphabet
from .pac_campaign_utils import write_once
from .pac_baseline_fairness_maximal import BASELINES
from .pac_classification_diagnostics import (
    classification_diagnostics,
    corruption_diagnostics,
    corruption_suite,
)
from .pac_confirmatory_baselines import (
    ConfirmatoryFamily,
    build_confirmatory_family,
    confirmatory_implementation_metadata,
    confirmatory_trial_spec,
)
from .pac_device import resolve_device
from .pac_metrics import count_parameters
from .pac_pa2wp_boundary_campaign import LOW_DATASETS
from .pac_real_data import ensure_ucr_dataset
from .pac_tf_p1p2_eval import clean_low_data_task
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACDevice, PACExperimentConfig

DEFAULT_ROOT: Final = Path(
    ".omx/results/alphabet-final-radial-log-corruption-20260726"
)
# The low-data campaign consumes an explicit validation-frozen selection record.
DEFAULT_SELECTION_PATH: Final = Path(
    ".omx/results/pac-direct-stem-low-data-selection/validation_frozen_selection.json"
)
DEFAULT_ALPHABET_SELECTION_PATH: Final = (
    Path(".omx/results")
    / "alphabet-27task-corrected-path-recovery-20260726"
    / "stage2"
    / "selection.json"
)
DEFAULT_BASELINE_SELECTION_PATH: Final = Path(
    ".omx/results/pac-alphabet-q1q2-final-20260719/stage2/selection.json"
)
UCR_DATA_ROOT: Final = Path(".omx/data/ucr")
SEEDS: Final = (23, 31, 43, 47, 59)
EPOCHS: Final = 100
ALPHABET_MODEL: Final = "alphabet"
BASELINE_MODELS: Final = tuple(BASELINES)
MODELS: Final = (ALPHABET_MODEL, *BASELINE_MODELS)
CORRUPTION_CONDITIONS: Final = (
    "id",
    "noise_std_0.1",
    "noise_std_0.2",
    "missing_rate_0.1",
    "missing_rate_0.3",
    "amplitude_0.5",
    "amplitude_1.5",
    "resample_half_restore",
)


@dataclass(frozen=True, slots=True)
class DirectStemCorruptionJob:
    key: str
    dataset: str
    model: str
    seed: int
    model_dim: int
    modes: int
    trial: int | None
    config_key: str
    recipe_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    selection_source: str
    selection_policy: str = "train_validation_frozen_capacity_and_recipe_transfer"


def jobs(
    selection_path: Path = DEFAULT_ALPHABET_SELECTION_PATH,
    baseline_selection_path: Path = DEFAULT_BASELINE_SELECTION_PATH,
) -> tuple[DirectStemCorruptionJob, ...]:
    selected = _alphabet_selection(selection_path)
    baseline_selected = _baseline_selection(
        baseline_selection_path
    )
    result: list[DirectStemCorruptionJob] = []
    for dataset in LOW_DATASETS:
        cell_key = f"ucr:{dataset}:alphabet"
        cell = selected[cell_key]
        model_dim = _required_int(cell, "width")
        modes = _required_int(cell, "modes")
        config_key = _required_str(cell, "config_key")
        recipe = _required_mapping(cell, "recipe")
        recipe_name = _required_str(recipe, "name")
        batch_size = _required_int(recipe, "batch_size")
        learning_rate = _required_float(recipe, "learning_rate")
        weight_decay = _required_float(recipe, "weight_decay")
        grad_clip_norm = _required_float(recipe, "grad_clip_norm")
        expected_config_key = (
            f"d{model_dim}-m{modes}-recipe{recipe_name.lower()}"
        )
        if config_key != expected_config_key:
            message = f"malformed final ALPHABET selection row: {cell_key}"
            raise RuntimeError(message)
        result.extend(
            DirectStemCorruptionJob(
                key=f"alphabet_radial_log_corruption:alphabet:{dataset}:seed{seed}",
                dataset=dataset,
                model=ALPHABET_MODEL,
                seed=seed,
                model_dim=model_dim,
                modes=modes,
                trial=None,
                config_key=config_key,
                recipe_name=recipe_name,
                epochs=EPOCHS,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                grad_clip_norm=grad_clip_norm,
                selection_source=f"validation-frozen radial-log selection:{cell_key}",
            )
            for seed in SEEDS
        )
        for model in BASELINE_MODELS:
            baseline_key = f"ucr:{dataset}:{model}"
            baseline_cell = baseline_selected[baseline_key]
            width = _required_int(baseline_cell, "width")
            trial = _required_int(baseline_cell, "trial")
            baseline_config_key = _required_str(baseline_cell, "config_key")
            spec = confirmatory_trial_spec(cast("ConfirmatoryFamily", model), trial)
            result.extend(
                DirectStemCorruptionJob(
                    key=f"alphabet_radial_log_corruption:{model}:{dataset}:seed{seed}",
                    dataset=dataset,
                    model=model,
                    seed=seed,
                    model_dim=width,
                    modes=16,
                    trial=trial,
                    config_key=baseline_config_key,
                    recipe_name=f"trial-{trial}",
                    epochs=EPOCHS,
                    batch_size=spec.batch_size,
                    learning_rate=spec.learning_rate,
                    weight_decay=spec.weight_decay,
                    grad_clip_norm=spec.grad_clip_norm,
                    selection_source=f"validation-frozen baseline selection:{baseline_key}",
                )
                for seed in SEEDS
            )
    expected_jobs = len(LOW_DATASETS) * len(MODELS) * len(SEEDS)
    if len(result) != expected_jobs or len({job.key for job in result}) != expected_jobs:
        message = f"corruption campaign must contain {expected_jobs} unique jobs"
        raise RuntimeError(message)
    return tuple(result)


def enqueue(
    root: Path = DEFAULT_ROOT,
    *,
    shard_count: int = 2,
    selection_path: Path = DEFAULT_ALPHABET_SELECTION_PATH,
    baseline_selection_path: Path = DEFAULT_BASELINE_SELECTION_PATH,
) -> dict[str, object]:
    if shard_count < 1:
        message = "shard_count must be positive"
        raise ValueError(message)
    active = jobs(selection_path, baseline_selection_path)
    shards: list[list[DirectStemCorruptionJob]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for job in sorted(active, key=_job_weight, reverse=True):
        index = min(range(shard_count), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += _job_weight(job)
    for index, shard in enumerate(shards):
        path = root / "shards" / f"shard-{index:02d}" / "manifest.jsonl"
        body = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard)
        write_once(path, body)
    contract: dict[str, object] = {
        "schema": "alphabet_radial_log_corruption.v3",
        "public_model": "ALPHABET",
        "architecture": "radial_log_r_affine",
        "model_class": "lnet.alphabet.Alphabet",
        "models": list(MODELS),
        "baselines": list(BASELINE_MODELS),
        "datasets": list(LOW_DATASETS),
        "seeds": list(SEEDS),
        "jobs": len(active),
        "corruption_conditions": list(CORRUPTION_CONDITIONS),
        "corruption_primary_metric": "balanced_accuracy",
        "missingness_policy": (
            "zero imputation plus observation_mask=0 at missing positions in the "
            "raw stem and writer drive; the latent terminal reader remains active; "
            "the original equal-step grid is retained"
        ),
        "selection_path": str(selection_path),
        "baseline_selection_path": str(baseline_selection_path),
        "selection_uses_official_test_evidence": False,
        "selection_policy": (
            "ALPHABET and every baseline use task-specific configurations frozen "
            "from TRAIN-derived validation before this campaign; both searches "
            "cover 18 capacity/optimizer combinations per task"
        ),
        "claim_status": (
            "paired corruption comparison of the final radial-log ALPHABET and "
            "six validation-frozen baselines; not domain OOD evidence"
        ),
        "optimization_split": "official_train_stratified_80_20",
        "checkpoint_policy": "minimum_official_TRAIN_heldout_validation_loss",
        "evaluation_split": "official_test_and_deterministic_corruptions",
        "normalization_fit": "official_TRAIN_optimization_fold_only",
        "epochs": EPOCHS,
        "shards": shard_count,
        "estimated_shard_loads": loads,
        "restart_safe": True,
    }
    write_once(root / "contract.json", json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def run_manifest(
    shard_root: Path,
    *,
    selection_path: Path = DEFAULT_ALPHABET_SELECTION_PATH,
    baseline_selection_path: Path = DEFAULT_BASELINE_SELECTION_PATH,
    data_root: Path = UCR_DATA_ROOT,
    device: PACDevice = "auto",
) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in (shard_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_jobs = tuple(DirectStemCorruptionJob(**row) for row in rows)
    expected = {
        job.key: job for job in jobs(selection_path, baseline_selection_path)
    }
    if any(expected.get(job.key) != job for job in manifest_jobs):
        message = "manifest differs from the sealed corruption contract"
        raise RuntimeError(message)
    runtime_device = cast("PACDevice", resolve_device(device))
    completed = _local_keys(shard_root, "completed")
    for job in manifest_jobs:
        if job.key in completed:
            continue
        try:
            result = run_job(
                job,
                data_root=data_root,
                output_root=shard_root,
                device=runtime_device,
            )
        except Exception as error:  # noqa: BLE001 - durable queue records failures
            result: dict[str, object] = {
                "schema": "pac_direct_stem_corruption_failure.v1",
                **asdict(job),
                "job_key": job.key,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _write_json(_result_path(shard_root, job.key, "failed"), result, replace=True)
            continue
        _write_json(_result_path(shard_root, job.key, "completed"), result)
        failed = _result_path(shard_root, job.key, "failed")
        if failed.exists():
            failed.unlink()
    return _shard_status(shard_root, manifest_jobs)


def run_job(
    job: DirectStemCorruptionJob,
    *,
    data_root: Path = UCR_DATA_ROOT,
    output_root: Path = DEFAULT_ROOT,
    device: PACDevice = "auto",
) -> dict[str, object]:
    runtime_device = cast("PACDevice", resolve_device(device))
    dataset = ensure_ucr_dataset(
        job.dataset,
        data_root,
        allow_download=True,
        require_train_label_space=True,
    )
    task = clean_low_data_task(dataset, 1.0, job.seed)
    config = PACExperimentConfig(
        sample_count=int(task.train_inputs.shape[0]),
        validation_count=int(task.validation_inputs.shape[0]),
        test_count=int(task.test_inputs.shape[0]),
        sequence_length=int(task.train_inputs.shape[1]),
        raw_input_dim=int(task.train_inputs.shape[-1]),
        output_dim=task.class_count,
        model_dim=job.model_dim,
        modes=job.modes,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=job.learning_rate,
        weight_decay=job.weight_decay,
        grad_clip_norm=job.grad_clip_norm,
        seeds=(job.seed,),
        device=runtime_device,
        output_dir=output_root,
        optimizer_mode=(
            "fused"
            if runtime_device.startswith("cuda") and job.model == ALPHABET_MODEL
            else "default"
        ),
    )
    torch.manual_seed(job.seed)
    if runtime_device.startswith("cuda"):
        torch.cuda.manual_seed_all(job.seed)
    model = _build_model(job, config, task.class_count)
    started = perf_counter()
    outcome = train_classifier(
        model,
        task,
        config,
        runtime_device,
        job.seed,
        evaluate_test=True,
        restore_best_validation=True,
    )
    test_inputs = task.test_inputs.to(runtime_device)
    test_labels = task.test_labels.to(runtime_device)
    metrics = classification_metric_bundle(
        model, test_inputs, test_labels, batch_size=job.batch_size
    )
    diagnostics = corruption_diagnostics(
        model,
        test_inputs,
        test_labels,
        job.seed,
        batch_size=job.batch_size,
    )
    corruption_balanced_accuracy = _corruption_balanced_accuracy(
        model,
        test_inputs,
        test_labels,
        job.seed,
        batch_size=job.batch_size,
    )
    mask_aware_corruption = (
        _mask_aware_corruption_metrics(
            model,
            test_inputs,
            test_labels,
            job.seed,
            batch_size=job.batch_size,
        )
        if job.model == ALPHABET_MODEL
        else None
    )
    return {
        "schema": "alphabet_radial_log_corruption_result.v3",
        **asdict(job),
        "job_key": job.key,
        "status": "done",
        "model": job.model,
        "public_model": "ALPHABET" if job.model == ALPHABET_MODEL else job.model,
        "architecture": (
            "radial_log_r_affine"
            if job.model == ALPHABET_MODEL
            else confirmatory_implementation_metadata(
                cast("ConfirmatoryFamily", job.model),
                cast("int", job.trial),
            )
        ),
        "model_class": (
            "lnet.alphabet.Alphabet" if job.model == ALPHABET_MODEL else None
        ),
        "selection_test_evidence_used": False,
        "claim_status": "validation_frozen_descriptive_corruption_not_domain_ood",
        "optimization_split": "official_train_stratified_80_20",
        "checkpoint_policy": "minimum_official_TRAIN_heldout_validation_loss",
        "normalization_fit": "official_TRAIN_optimization_fold_only",
        "train_count": int(task.train_inputs.shape[0]),
        "validation_count": int(task.validation_inputs.shape[0]),
        "test_count": int(task.test_inputs.shape[0]),
        "params_trainable": count_parameters(model),
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "best_epoch": outcome.best_epoch,
        "train_seconds": outcome.elapsed_time,
        "total_fit_and_evaluation_seconds": perf_counter() - started,
        **classification_diagnostics(
            model, test_inputs, test_labels, batch_size=job.batch_size
        ),
        "real_corruption_ood_json": diagnostics,
        "corruption_balanced_accuracy_json": corruption_balanced_accuracy,
        "mask_aware_corruption_json": mask_aware_corruption,
        "real_ood_scope": "deterministic_corruption_shift_not_domain_shift",
        "environment": _environment_metadata(runtime_device),
    }


def status(
    root: Path = DEFAULT_ROOT,
    *,
    selection_path: Path = DEFAULT_ALPHABET_SELECTION_PATH,
    baseline_selection_path: Path = DEFAULT_BASELINE_SELECTION_PATH,
) -> dict[str, object]:
    expected = {
        job.key for job in jobs(selection_path, baseline_selection_path)
    }
    completed = _all_keys(root, "completed")
    failed = _all_keys(root, "failed") - completed
    return {
        "schema": "pac_direct_stem_corruption_status.v1",
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed_retryable": len(expected & failed),
        "remaining": len(expected - completed),
        "done": expected <= completed,
    }


def report(
    root: Path = DEFAULT_ROOT,
    *,
    selection_path: Path = DEFAULT_ALPHABET_SELECTION_PATH,
    baseline_selection_path: Path = DEFAULT_BASELINE_SELECTION_PATH,
) -> dict[str, object]:
    expected_jobs = {
        job.key: job for job in jobs(selection_path, baseline_selection_path)
    }
    rows_by_key = _validated_completed_rows(root, expected_jobs)
    rows = list(rows_by_key.values())
    conditions: dict[str, dict[str, dict[str, object]]] = {}
    for condition in CORRUPTION_CONDITIONS:
        conditions[condition] = {}
        for model in MODELS:
            scores: list[float] = []
            drops: list[float] = []
            for row in rows:
                if row.get("model") != model:
                    continue
                items = json.loads(str(row["corruption_balanced_accuracy_json"]))
                item = next(value for value in items if value["shift"] == condition)
                scores.append(float(item["balanced_accuracy"]))
                drops.append(float(item["absolute_balanced_accuracy_drop"]))
            conditions[condition][model] = {
                "mean_balanced_accuracy": mean(scores) if scores else None,
                "mean_absolute_balanced_accuracy_drop": mean(drops) if drops else None,
                "rows": len(scores),
            }
    mask_aware_conditions: dict[str, dict[str, object]] = {}
    alphabet_rows = [row for row in rows if row.get("model") == ALPHABET_MODEL]
    for condition in CORRUPTION_CONDITIONS:
        scores = []
        drops = []
        for row in alphabet_rows:
            items = json.loads(str(row["mask_aware_corruption_json"]))
            item = next(value for value in items if value["shift"] == condition)
            scores.append(float(item["balanced_accuracy"]))
            drops.append(float(item["absolute_balanced_accuracy_drop"]))
        mask_aware_conditions[condition] = {
            "mean_balanced_accuracy": mean(scores) if scores else None,
            "mean_absolute_balanced_accuracy_drop": mean(drops) if drops else None,
            "rows": len(scores),
        }
    payload: dict[str, object] = {
        "schema": "alphabet_radial_log_corruption_report.v3",
        "status": status(
            root,
            selection_path=selection_path,
            baseline_selection_path=baseline_selection_path,
        ),
        "scope": "deterministic corruption shift, not domain shift",
        "primary_metric": "balanced_accuracy",
        "models": list(MODELS),
        "missingness_policy": (
            "zero imputation plus writer-only observation_mask=0; the latent reader "
            "remains active and default unit time_delta retains the equal-step grid"
        ),
        "selection_uses_official_test_evidence": False,
        "conditions": conditions,
        "alphabet_mask_aware_conditions": mask_aware_conditions,
        "paired_alphabet_minus_baseline": _paired_comparisons(rows),
        "completed_rows_validated": True,
    }
    _write_json(root / "reports" / "DIRECT_STEM_CORRUPTION.json", payload, replace=True)
    return payload


def _validated_completed_rows(
    root: Path,
    expected: dict[str, DirectStemCorruptionJob],
) -> dict[str, dict[str, object]]:
    """Load only complete, job-bound rows for aggregate reporting."""
    completed_dir = root / "completed"
    rows: dict[str, dict[str, object]] = {}
    expected_paths = {_result_path(root, key, "completed") for key in expected}
    for key, job in expected.items():
        path = _result_path(root, key, "completed")
        if not path.is_file():
            raise RuntimeError(f"missing completed result for {key}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid completed result for {key}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"completed result for {key} is not an object")
        if value.get("status") != "done" or value.get("job_key") != key:
            raise RuntimeError(f"completed result for {key} has invalid identity")
        for field, expected_value in asdict(job).items():
            if value.get(field) != expected_value:
                raise RuntimeError(f"completed result for {key} has wrong {field}")
        if value.get("selection_test_evidence_used") is not False:
            raise RuntimeError(f"completed result for {key} used TEST selection evidence")
        try:
            items = json.loads(str(value["corruption_balanced_accuracy_json"]))
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"completed result for {key} has invalid corruption metrics") from error
        if not isinstance(items, list) or len(items) != len(CORRUPTION_CONDITIONS):
            raise RuntimeError(f"completed result for {key} has incomplete corruption metrics")
        shifts: set[str] = set()
        for item in items:
            if not isinstance(item, dict) or item.get("shift") not in CORRUPTION_CONDITIONS:
                raise RuntimeError(f"completed result for {key} has an unknown corruption condition")
            shift = str(item["shift"])
            if shift in shifts:
                raise RuntimeError(f"completed result for {key} repeats corruption condition {shift}")
            shifts.add(shift)
            for metric in ("balanced_accuracy", "absolute_balanced_accuracy_drop"):
                metric_value = item.get(metric)
                if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                    raise RuntimeError(f"completed result for {key} has invalid {metric}")
                numeric = float(metric_value)
                if not math.isfinite(numeric):
                    raise RuntimeError(f"completed result for {key} has non-finite {metric}")
                if metric == "balanced_accuracy" and not 0.0 <= numeric <= 1.0:
                    raise RuntimeError(f"completed result for {key} has out-of-range balanced_accuracy")
        if shifts != set(CORRUPTION_CONDITIONS):
            raise RuntimeError(f"completed result for {key} has incomplete corruption conditions")
        if job.model == ALPHABET_MODEL:
            try:
                mask_items = json.loads(str(value["mask_aware_corruption_json"]))
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"completed result for {key} has invalid mask-aware metrics") from error
            if not isinstance(mask_items, list) or {
                item.get("shift") for item in mask_items if isinstance(item, dict)
            } != set(CORRUPTION_CONDITIONS):
                raise RuntimeError(f"completed result for {key} has incomplete mask-aware metrics")
        rows[key] = cast("dict[str, object]", value)

    extras = [
        path.name
        for path in completed_dir.glob("*.json")
        if path not in expected_paths
    ]
    if extras:
        raise RuntimeError(f"completed directory contains unexpected result files: {sorted(extras)}")
    return rows


@torch.no_grad()
def _corruption_balanced_accuracy(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    *,
    batch_size: int,
) -> str:
    rows: list[dict[str, float | str]] = []
    was_training = model.training
    model.eval()
    try:
        for shift, shifted in corruption_suite(inputs, seed):
            metrics = classification_metric_bundle(
                model,
                shifted,
                labels,
                batch_size=batch_size,
            )
            rows.append(
                {
                    "shift": shift,
                    "balanced_accuracy": metrics.balanced_accuracy,
                }
            )
    finally:
        model.train(was_training)
    clean = float(rows[0]["balanced_accuracy"])
    for row in rows:
        row["absolute_balanced_accuracy_drop"] = clean - float(
            row["balanced_accuracy"]
        )
    return json.dumps(rows, separators=(",", ":"))


@torch.no_grad()
def _mask_aware_corruption_metrics(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    *,
    batch_size: int,
) -> str:
    suite = corruption_suite(inputs, seed)
    generator = torch.Generator(device=inputs.device).manual_seed(seed + 41_003)
    torch.randn(
        inputs.shape,
        generator=generator,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    missing_draw = torch.rand(
        inputs.shape[:2],
        generator=generator,
        device=inputs.device,
    )
    masks = {
        "missing_rate_0.1": missing_draw.ge(0.1).to(dtype=inputs.dtype),
        "missing_rate_0.3": missing_draw.ge(0.3).to(dtype=inputs.dtype),
    }
    rows: list[dict[str, float | str | bool]] = []
    was_training = model.training
    model.eval()
    try:
        for shift, shifted in suite:
            observation_mask = masks.get(shift)
            logits: list[torch.Tensor] = []
            for start in range(0, shifted.shape[0], batch_size):
                stop = min(start + batch_size, shifted.shape[0])
                batch = shifted[start:stop]
                if observation_mask is None:
                    output = model(batch)
                else:
                    output = model(
                        batch,
                        observation_mask=observation_mask[start:stop],
                    )
                logits.append(output.detach().cpu())
            predictions = torch.cat(logits).argmax(dim=-1)
            labels_cpu = labels.detach().cpu()
            accuracy = float(predictions.eq(labels_cpu).float().mean().item())
            recalls = [
                float(predictions[labels_cpu.eq(class_index)].eq(class_index).float().mean().item())
                for class_index in range(int(labels_cpu.max().item()) + 1)
                if bool(labels_cpu.eq(class_index).any())
            ]
            rows.append(
                {
                    "shift": shift,
                    "accuracy": accuracy,
                    "balanced_accuracy": mean(recalls),
                    "observation_mask_supplied": observation_mask is not None,
                }
            )
    finally:
        model.train(was_training)
    clean_accuracy = float(rows[0]["accuracy"])
    clean_balanced_accuracy = float(rows[0]["balanced_accuracy"])
    for row in rows:
        row["absolute_accuracy_drop"] = clean_accuracy - float(row["accuracy"])
        row["absolute_balanced_accuracy_drop"] = clean_balanced_accuracy - float(
            row["balanced_accuracy"]
        )
    return json.dumps(rows, separators=(",", ":"))


def load_validation_frozen_direct_stem_selection(
    path: Path,
) -> dict[str, dict[str, object]]:
    """Load a direct-stem envelope frozen from TRAIN-derived validation only."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if (
        payload.get("schema") != "pac.direct_stem.validation_frozen_selection.v1"
        or payload.get("configuration_frozen_before_official_test") is not True
        or payload.get("official_test_accessed_during_selection") is not False
        or not isinstance(selected, dict)
    ):
        message = "selection is not a validation-frozen direct-stem summary"
        raise RuntimeError(message)
    required = {f"ucr:{dataset}" for dataset in LOW_DATASETS}
    if not required <= set(selected):
        message = "selection is missing a corruption dataset"
        raise RuntimeError(message)
    return cast("dict[str, dict[str, object]]", selected)


def _alphabet_selection(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if (
        payload.get("schema")
        != "pac.balanced_hpo_alphabet_27task_recovery_stage2_selection.v1"
        or payload.get("configuration_frozen_before_test") is not True
        or payload.get("official_test_accessed_during_selection") is not False
        or not isinstance(selected, dict)
    ):
        message = "selection is not the TEST-free final radial-log ALPHABET freeze"
        raise RuntimeError(message)
    required = {f"ucr:{dataset}:alphabet" for dataset in LOW_DATASETS}
    if not required <= set(selected):
        message = "final ALPHABET selection is missing a corruption dataset"
        raise RuntimeError(message)
    for key in required:
        if selected[key].get("architecture") != "radial-log-r-affine":
            message = f"selection cell is not radial-log-r-affine: {key}"
            raise RuntimeError(message)
    return cast("dict[str, dict[str, object]]", selected)


def _baseline_selection(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if (
        payload.get("schema") != "pac_alphabet_q1_final_freeze.v1"
        or payload.get("selection_seeds") != [7, 11, 19]
        or payload.get("final_seeds") != list(SEEDS)
        or payload.get("test_evidence_used_for_architecture_choice") is not False
        or not isinstance(selected, dict)
    ):
        message = "baseline selection is not the sealed TEST-free final freeze"
        raise RuntimeError(message)
    required = {
        f"ucr:{dataset}:{model}"
        for dataset in LOW_DATASETS
        for model in BASELINE_MODELS
    }
    if not required <= set(selected):
        message = "baseline selection is missing a corruption comparison cell"
        raise RuntimeError(message)
    return cast("dict[str, dict[str, object]]", selected)


def _required_int(row: dict[str, object], name: str) -> int:
    value = row.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"{name} must be an integer"
        raise TypeError(message)
    return value


def _required_float(row: dict[str, object], name: str) -> float:
    value = row.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        message = f"{name} must be numeric"
        raise TypeError(message)
    return float(value)


def _required_str(row: dict[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        message = f"{name} must be a non-empty string"
        raise TypeError(message)
    return value


def _required_mapping(row: dict[str, object], name: str) -> dict[str, object]:
    value = row.get(name)
    if not isinstance(value, dict):
        message = f"{name} must be an object"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _build_model(
    job: DirectStemCorruptionJob,
    config: PACExperimentConfig,
    class_count: int,
) -> torch.nn.Module:
    if job.model == ALPHABET_MODEL:
        if job.trial is not None:
            message = "final ALPHABET job must use an optimizer recipe, not a trial"
            raise ValueError(message)
        return Alphabet(config, class_count, objective="classification")
    if job.model not in BASELINE_MODELS or job.trial is None:
        message = f"unsupported corruption model contract: {job.model}"
        raise ValueError(message)
    return build_confirmatory_family(
        cast("ConfirmatoryFamily", job.model),
        job.model_dim,
        config,
        class_count,
        validation_trial=job.trial,
    )


def _paired_comparisons(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    indexed: dict[tuple[str, str, int], dict[str, object]] = {
        (
            str(row["model"]),
            str(row["dataset"]),
            int(cast("int", row["seed"])),
        ): row
        for row in rows
    }
    output: dict[str, dict[str, dict[str, object]]] = {}
    for condition in CORRUPTION_CONDITIONS:
        output[condition] = {}
        for baseline in BASELINE_MODELS:
            deltas: list[float] = []
            for dataset in LOW_DATASETS:
                for seed in SEEDS:
                    alphabet_row = indexed.get((ALPHABET_MODEL, dataset, seed))
                    baseline_row = indexed.get((baseline, dataset, seed))
                    if alphabet_row is None or baseline_row is None:
                        continue
                    alphabet_items = json.loads(
                        str(alphabet_row["corruption_balanced_accuracy_json"])
                    )
                    baseline_items = json.loads(
                        str(baseline_row["corruption_balanced_accuracy_json"])
                    )
                    alphabet_score = next(
                        float(item["balanced_accuracy"])
                        for item in alphabet_items
                        if item["shift"] == condition
                    )
                    baseline_score = next(
                        float(item["balanced_accuracy"])
                        for item in baseline_items
                        if item["shift"] == condition
                    )
                    deltas.append(alphabet_score - baseline_score)
            tolerance = 1.0e-12
            output[condition][baseline] = {
                "mean_paired_balanced_accuracy_delta": (
                    mean(deltas) if deltas else None
                ),
                "alphabet_wins": sum(delta > tolerance for delta in deltas),
                "ties": sum(abs(delta) <= tolerance for delta in deltas),
                "alphabet_losses": sum(delta < -tolerance for delta in deltas),
                "pairs": len(deltas),
            }
    return output


def _job_weight(job: DirectStemCorruptionJob) -> float:
    dataset_weight = {
        "CinCECGTorso": 4.0,
        "CricketX": 2.0,
        "Earthquakes": 1.0,
        "Phoneme": 5.0,
        "StarLightCurves": 5.0,
    }[job.dataset]
    model_weight = {
        "alphabet": 1.0,
        "cnn1d": 1.2,
        "tcn": 0.7,
        "mamba": 1.3,
        "gru": 1.8,
        "lstm": 3.0,
        "transformer": 2.4,
    }[job.model]
    return (
        dataset_weight
        * model_weight
        * math.sqrt(job.model_dim / 64.0)
        * math.sqrt(job.modes / 16.0)
    )


def _shard_status(
    root: Path, manifest_jobs: tuple[DirectStemCorruptionJob, ...]
) -> dict[str, object]:
    expected = {job.key for job in manifest_jobs}
    completed = _local_keys(root, "completed")
    failed = _local_keys(root, "failed") - completed
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed_retryable": len(expected & failed),
        "remaining": len(expected - completed),
        "done": expected <= completed,
    }


ResultBucket = Literal["completed", "failed"]


def _result_path(root: Path, key: str, bucket: ResultBucket) -> Path:
    return root / bucket / f"{key.replace(':', '__')}.json"


def _local_rows(root: Path, bucket: ResultBucket) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((root / bucket).glob("*.json"))
    ]


def _local_keys(root: Path, bucket: ResultBucket) -> set[str]:
    return {str(row["job_key"]) for row in _local_rows(root, bucket)}


def _all_rows(root: Path, bucket: ResultBucket) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.glob(f"shards/*/{bucket}/*.json"))
    ]


def _all_keys(root: Path, bucket: ResultBucket) -> set[str]:
    return {str(row["job_key"]) for row in _all_rows(root, bucket)}


def _write_json(path: Path, payload: dict[str, object], *, replace: bool = False) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if replace:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    else:
        write_once(path, text)


def _environment_metadata(device: str) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(device).index or 0)
            if device.startswith("cuda")
            else None
        ),
        "precision": "fp32",
    }
