from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .hybrid_experiment_types import resolve_device
from .pac_data_split import stratified_partition_indices
from .pac_eval_sections import (
    classification_task,
    clean_validation_classification_task,
    full_train_classification_task,
)
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_dataset, ensure_ucr_train_only
from .pac_recommended_low_data_models import build_low_data_classifier
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACClassificationTask, PACExperimentConfig

if TYPE_CHECKING:
    from .pac_confirmatory_baselines import ConfirmatoryMatch, ConfirmatoryTrialSpec
    from .pac_recommended_low_data_types import LowDataJob
    from .pac_types import PACTrainOutcome
    from .tapped_prl_followup_schema import JsonRow

_MODEL_BUILD_LOCK = Lock()


def run_low_data_job(config: PACExperimentConfig, job: LowDataJob) -> JsonRow:
    if job.evaluation_collection == "untouched_ucr_confirmatory":
        _validate_untouched_test_access(config, job)
    device = resolve_device(config.device)
    validation_only_clean = (
        job.evaluation_split == "validation" and job.data_protocol == "clean_stratified"
    )
    dataset = (
        ensure_ucr_train_only(job.dataset, Path(".omx/data/ucr"), allow_download=True)
        if validation_only_clean
        else ensure_ucr_dataset(
            job.dataset,
            Path(".omx/data/ucr"),
            allow_download=True,
            require_train_label_space=job.evaluation_collection == "unseen_final_ucr",
        )
    )
    normalization_provenance: dict[str, object] | None = None
    if job.refit_full_train and job.data_protocol == "clean_stratified":
        normalization_provenance = _full_train_normalization_provenance(dataset)
        task = full_train_classification_task(dataset)
    elif job.evaluation_split == "validation" and job.data_protocol == "clean_stratified":
        task = clean_validation_classification_task(dataset, job.seed)
    else:
        task = classification_task(dataset.name, dataset)
        if job.evaluation_split == "validation":
            task = validation_selection_task(task, job.seed)
        elif job.refit_full_train:
            task = full_train_refit_task(task)
    task = low_data_task(task, job.ratio, job.seed)
    trial_spec = _job_trial_spec(job)
    run_config = replace(
        config,
        seeds=(job.seed,),
        sequence_length=task.train_inputs.shape[1],
        raw_input_dim=1,
        output_dim=task.class_count,
        learning_rate=job.learning_rate or config.learning_rate,
        weight_decay=job.weight_decay if job.weight_decay is not None else config.weight_decay,
        epochs=job.refit_epochs if job.refit_epochs is not None else config.epochs,
        batch_size=trial_spec.batch_size if trial_spec is not None else config.batch_size,
        grad_clip_norm=(
            trial_spec.grad_clip_norm if trial_spec is not None else config.grad_clip_norm
        ),
    )
    started = perf_counter()
    model, parameter_match = _build_seeded_job_classifier(job, run_config, task.class_count)
    validation_only = job.evaluation_split == "validation"
    outcome = _train_low_data_model(
        model,
        task,
        run_config,
        device,
        job,
        validation_only=validation_only,
    )
    metric_inputs = task.validation_inputs if validation_only else task.test_inputs
    metric_labels = task.validation_labels if validation_only else task.test_labels
    metrics = classification_metric_bundle(
        model,
        metric_inputs.to(device=device),
        metric_labels.to(device=device),
        batch_size=run_config.batch_size,
    )
    row: JsonRow = {
        "job_key": job.key,
        "experiment_group": "recommended_low_data",
        "dataset_or_task": task.label,
        "seed": job.seed,
        "model": job.model,
        "data_ratio": job.ratio,
        "params_trainable": count_parameters(model),
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "evaluation_split": job.evaluation_split,
        "training_protocol": (
            "official_train_full_refit"
            if job.refit_full_train
            else "official_train_validation_split"
        ),
        "data_protocol": job.data_protocol,
        "official_test_accessed": not validation_only_clean,
        "test_access_policy": (
            "train_only_firewall" if validation_only_clean else "official_test_enabled"
        ),
        "checkpoint_policy": (
            "best_validation_loss" if job.restore_best_validation else "final_fixed_epoch"
        ),
        "best_epoch": outcome.best_epoch,
        "evaluation_collection": job.evaluation_collection,
        "baseline_family": job.baseline_family,
        "reference_model": job.reference_model,
        "validation_trial": job.validation_trial,
        "refit_epochs": job.refit_epochs,
        "learning_rate": run_config.learning_rate,
        "weight_decay": run_config.weight_decay,
        "architecture_metadata_json": (
            _job_architecture_metadata_json(job) if trial_spec is not None else ""
        ),
        "elapsed_train_time": outcome.elapsed_time,
        "elapsed_total_time": perf_counter() - started,
        "status": "done",
    }
    if parameter_match is not None:
        row.update(
            {
                "matched_width": parameter_match.width,
                "target_params": parameter_match.target_params,
                "relative_param_error": parameter_match.relative_error,
            }
        )
    if validation_only:
        row.update(
            {
                "validation_accuracy": metrics.accuracy,
                "validation_macro_f1": metrics.macro_f1,
                "validation_weighted_f1": metrics.weighted_f1,
                "validation_balanced_accuracy": metrics.balanced_accuracy,
            }
        )
    else:
        row.update(
            {
                "test_loss": outcome.test_loss,
                "test_accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "weighted_f1": metrics.weighted_f1,
                "balanced_accuracy": metrics.balanced_accuracy,
            }
        )
    if job.evaluation_collection == "unseen_final_ucr":
        if normalization_provenance is None or job.refit_epochs is None:
            message = "unseen-final refits require frozen normalization and refit_epochs"
            raise ValueError(message)
        checkpoint_path, checkpoint_sha256 = _save_unseen_final_checkpoint(
            model,
            run_config,
            job,
            task,
            normalization_provenance,
            row,
        )
        row["checkpoint_path"] = str(checkpoint_path)
        row["checkpoint_sha256"] = checkpoint_sha256
    return row


def _validate_untouched_test_access(
    config: PACExperimentConfig,
    job: LowDataJob,
) -> None:
    """Fail closed before loading any official TEST member in the untouched suite."""
    from .pac_untouched_ucr_confirmatory import verify_test_access_lock  # noqa: PLC0415

    if job.evaluation_split != "test" or not job.refit_full_train:
        message = "untouched confirmatory jobs must be full-TRAIN refits with TEST evaluation"
        raise ValueError(message)
    verify_test_access_lock(
        config.output_dir,
        expected_contract_sha256=job.official_test_contract_sha256,
        expected_job=job,
    )


def _full_train_normalization_provenance(dataset: object) -> dict[str, object]:
    from .pac_types import UCRDataset  # noqa: PLC0415

    if not isinstance(dataset, UCRDataset):
        message = "full-TRAIN checkpoint normalization requires a UCRDataset"
        raise TypeError(message)
    mean = dataset.train_inputs.mean()
    std = dataset.train_inputs.std(unbiased=False).clamp_min(1.0e-6)
    return {
        "fit_split": "all_official_train",
        "transform": "(x - scalar_mean) / max(scalar_std, 1e-6)",
        "mean": float(mean.item()),
        "std": float(std.item()),
        "official_train_count": int(dataset.train_inputs.shape[0]),
        "sequence_length": int(dataset.train_inputs.shape[1]),
        "raw_input_dim": int(dataset.train_inputs.shape[2]),
    }


def _save_unseen_final_checkpoint(
    model: torch.nn.Module,
    config: PACExperimentConfig,
    job: LowDataJob,
    task: PACClassificationTask,
    normalization: dict[str, object],
    row: JsonRow,
) -> tuple[Path, str]:
    directory = (
        config.output_dir
        / "checkpoints"
        / _safe_checkpoint_component(job.dataset)
        / _safe_checkpoint_component(job.model)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / f"seed{job.seed}.pt").resolve()
    temporary = path.with_suffix(".pt.tmp")
    config_payload = asdict(config)
    config_payload["output_dir"] = "<local-path>"
    config_payload["seeds"] = list(config.seeds)
    metric_names = (
        "train_loss",
        "validation_loss",
        "test_loss",
        "test_accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
    )
    payload = {
        "schema_version": "pac_unseen_final_checkpoint.v1",
        "job_key": job.key,
        "dataset": job.dataset,
        "seed": job.seed,
        "family": job.model,
        "reference_model": job.reference_model,
        "validation_trial": job.validation_trial,
        "architecture_metadata_json": row.get("architecture_metadata_json", ""),
        "refit_epochs": job.refit_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "normalization": normalization,
        "experiment_config": config_payload,
        "class_count": task.class_count,
        "p0_metrics": {name: row[name] for name in metric_names if name in row},
        "state_dict": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def _safe_checkpoint_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )


def _train_low_data_model(
    model: torch.nn.Module,
    task: PACClassificationTask,
    config: PACExperimentConfig,
    device: str,
    job: LowDataJob,
    *,
    validation_only: bool,
) -> PACTrainOutcome:
    kwargs: dict[str, bool] = {}
    if validation_only:
        kwargs["evaluate_test"] = False
    if job.restore_best_validation:
        kwargs["restore_best_validation"] = True
    return train_classifier(model, task, config, device, job.seed, **kwargs)


def _build_seeded_classifier(
    name: str, config: PACExperimentConfig, class_count: int, seed: int
) -> torch.nn.Module:
    # Module constructors use PyTorch's global CPU generator. Serialize only
    # construction, then restore its state so concurrent jobs cannot interfere.
    with _MODEL_BUILD_LOCK, torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return build_low_data_classifier(name, config, class_count)


def _build_seeded_job_classifier(
    job: LowDataJob,
    config: PACExperimentConfig,
    class_count: int,
) -> tuple[torch.nn.Module, ConfirmatoryMatch | None]:
    if job.baseline_family is None:
        return _build_seeded_classifier(job.model, config, class_count, job.seed), None
    if job.reference_model is None:
        message = "confirmatory baseline jobs require a frozen PAC-TF reference model"
        raise ValueError(message)
    from .pac_confirmatory_baselines import (  # noqa: PLC0415
        build_matched_confirmatory_classifier,
    )

    with _MODEL_BUILD_LOCK, torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(job.seed)
        return build_matched_confirmatory_classifier(
            job.baseline_family,
            job.reference_model,
            config,
            class_count,
            tolerance=job.parameter_match_tolerance or 0.05,
            validation_trial=job.validation_trial or 1,
        )


def _job_trial_spec(job: LowDataJob) -> ConfirmatoryTrialSpec | None:
    if job.baseline_family is None:
        return None
    if job.validation_trial is None:
        message = "confirmatory baseline jobs require validation_trial"
        raise ValueError(message)
    from .pac_confirmatory_baselines import confirmatory_trial_spec  # noqa: PLC0415

    spec = confirmatory_trial_spec(job.baseline_family, job.validation_trial)
    if job.learning_rate is not None and job.learning_rate != spec.learning_rate:
        message = "job learning_rate differs from its locked family trial"
        raise ValueError(message)
    if job.weight_decay is not None and job.weight_decay != spec.weight_decay:
        message = "job weight_decay differs from its locked family trial"
        raise ValueError(message)
    return spec


def _job_architecture_metadata_json(job: LowDataJob) -> str:
    if job.baseline_family is None or job.validation_trial is None:
        return ""
    import json  # noqa: PLC0415

    from .pac_confirmatory_baselines import (  # noqa: PLC0415
        confirmatory_implementation_metadata,
    )

    canonical = json.dumps(
        confirmatory_implementation_metadata(job.baseline_family, job.validation_trial),
        sort_keys=True,
        separators=(",", ":"),
    )
    if job.architecture_metadata_json and job.architecture_metadata_json != canonical:
        message = "job architecture metadata differs from its locked family trial"
        raise ValueError(message)
    return canonical


def low_data_task(task: PACClassificationTask, ratio: float, seed: int) -> PACClassificationTask:
    if ratio >= 1.0:
        return task
    train_inputs, train_labels = stratified_subset(
        task.train_inputs, task.train_labels, ratio, seed
    )
    return PACClassificationTask(
        task.label,
        train_inputs,
        train_labels,
        task.validation_inputs,
        task.validation_labels,
        task.test_inputs,
        task.test_labels,
        task.class_count,
    )


def validation_selection_task(
    task: PACClassificationTask,
    seed: int,
    validation_ratio: float = 0.2,
) -> PACClassificationTask:
    """Re-split official TRAIN only; the official TEST tensors remain held out."""
    inputs = torch.cat((task.train_inputs, task.validation_inputs), dim=0)
    labels = torch.cat((task.train_labels, task.validation_labels), dim=0)
    train_indices, validation_indices = stratified_partition_indices(
        labels,
        validation_ratio,
        seed,
    )
    return PACClassificationTask(
        task.label,
        inputs.index_select(0, train_indices.to(device=inputs.device)),
        labels.index_select(0, train_indices.to(device=labels.device)),
        inputs.index_select(0, validation_indices.to(device=inputs.device)),
        labels.index_select(0, validation_indices.to(device=labels.device)),
        task.test_inputs,
        task.test_labels,
        task.class_count,
    )


def full_train_refit_task(task: PACClassificationTask) -> PACClassificationTask:
    """Restore all official TRAIN examples after validation-only model selection."""
    train_inputs = torch.cat((task.train_inputs, task.validation_inputs), dim=0)
    train_labels = torch.cat((task.train_labels, task.validation_labels), dim=0)
    return PACClassificationTask(
        task.label,
        train_inputs,
        train_labels,
        task.validation_inputs,
        task.validation_labels,
        task.test_inputs,
        task.test_labels,
        task.class_count,
    )


def stratified_subset(
    inputs: Tensor, labels: Tensor, ratio: float, seed: int
) -> tuple[Tensor, Tensor]:
    labels_cpu = labels.detach().cpu()
    classes = torch.unique(labels_cpu, sorted=True)
    target = max(int(inputs.shape[0] * ratio), int(classes.numel()))
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    remaining: list[int] = []
    for class_value in classes.tolist():
        indices = torch.nonzero(labels_cpu == int(class_value), as_tuple=False).flatten()
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        selected.append(int(shuffled[0]))
        remaining.extend(int(index) for index in shuffled[1:].tolist())
    if len(selected) < target and remaining:
        order = torch.randperm(len(remaining), generator=generator).tolist()
        selected.extend(remaining[int(index)] for index in order[: target - len(selected)])
    chosen = torch.tensor(selected[:target], dtype=torch.long, device=inputs.device)
    return inputs.index_select(0, chosen), labels.index_select(0, chosen)
