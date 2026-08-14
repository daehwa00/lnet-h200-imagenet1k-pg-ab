# ruff: noqa: EM101, EM102, TRY003
from __future__ import annotations

import hashlib
import json
import platform
from contextlib import suppress
from dataclasses import dataclass, replace
from math import pi
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .hybrid_experiment_types import resolve_device
from .pac_builders import build_regression_model
from .pac_classification_diagnostics import classification_diagnostics, corruption_diagnostics
from .pac_confirmatory_baselines import (
    ConfirmatoryMatch,
    build_confirmatory_family,
    build_matched_confirmatory_classifier,
    confirmatory_implementation_metadata,
    confirmatory_trial_spec,
)
from .pac_data_split import stratified_partition_indices
from .pac_efp16_final_campaign import ucr_parameter_tolerance
from .pac_external_tasks import load_external_task
from .pac_headroom_efficient_models import build_efficient_headroom_classifier
from .pac_metrics import count_parameters, nrmse
from .pac_paper_queue_models import CNN1DRegressor, MambaSSMRegressor, TCNRegressor
from .pac_real_data import ensure_ucr_dataset
from .pac_recommended_low_data_eval import stratified_subset
from .pac_stiefel_variants import capacity_for_model, variant_for_model
from .pac_tf_mechanism_models import S4DRegressor
from .pac_tight_frame_models import (
    TightFrameClassifier,
    TightFrameSequenceRegressor,
)
from .pac_tight_frame_runtime import prepare_tight_frame_inference
from .pac_training import (
    classification_metric_bundle,
    evaluate_regression_loss,
    train_classifier,
    train_regression_model,
)
from .pac_types import PACClassificationTask, PACExperimentConfig, PACRegressionTask, UCRDataset

if TYPE_CHECKING:
    from .pac_confirmatory_baselines import ConfirmatoryFamily
    from .pac_headroom_efficient_models import EfficientHeadroomSpec
    from .pac_tf_p1p2_types import P1P2Config, P1P2Job
    from .pac_types import PACDevice, PACModelName

_MODEL_INIT_LOCK = Lock()
_WP_MODEL = "wp_pac"
_WP_REFERENCE_MODEL = "pac_headroom_wp_d64_m16"
_PA2WP_MODEL = "pa2wp_pac"
_PA2WP_REFERENCE_MODEL = "pac_headroom_phase_augmented_ensemble_wp_d64_m16"
_EFP16_MODEL = "efp16_pac"
_EFP16_REFERENCE_MODEL = "pac_headroom_edge_frame_parseval_d32_m16"
_EFFICIENT_MODEL_SPECS: dict[str, EfficientHeadroomSpec] = {
    _WP_MODEL: "WP",
    _PA2WP_MODEL: "PA2WP",
    _EFP16_MODEL: "EFP16",
}
_EFFICIENT_REFERENCE_SPECS: dict[str, EfficientHeadroomSpec] = {
    _WP_REFERENCE_MODEL: "WP",
    _PA2WP_REFERENCE_MODEL: "PA2WP",
    _EFP16_REFERENCE_MODEL: "EFP16",
}
_EFFICIENT_REFERENCE_CAPACITIES = {
    _WP_REFERENCE_MODEL: (64, 16),
    _PA2WP_REFERENCE_MODEL: (64, 16),
    _EFP16_REFERENCE_MODEL: (32, 16),
}


@dataclass(frozen=True, slots=True)
class _WPMatch:
    params: int
    target_params: int
    relative_error: float = 0.0


class _EfficientHeadroomSyntheticEndpointRegressor(nn.Module):
    def __init__(self, config: PACExperimentConfig, spec: str) -> None:
        super().__init__()
        self.model = build_efficient_headroom_classifier(
            cast("EfficientHeadroomSpec", spec),
            replace(config, raw_input_dim=2),
            config.output_dim,
            objective="regression",
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(
            inputs[..., :2],
            time_delta=inputs[..., 2:3],
            observation_mask=inputs[..., 3:4],
        )

    def post_optimizer_step(self) -> None:
        self.model.post_optimizer_step()

    def finalize_constraints(self) -> None:
        self.model.finalize_constraints()


class _SequenceEndpointRegressor(nn.Module):
    """Expose the final record endpoint from an unchanged sequence regressor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.model(inputs)
        return outputs[:, -1] if outputs.ndim == 3 else outputs

    def post_optimizer_step(self) -> None:
        callback = getattr(self.model, "post_optimizer_step", None)
        if callable(callback):
            callback()

    def finalize_constraints(self) -> None:
        callback = getattr(self.model, "finalize_constraints", None)
        if callable(callback):
            callback()


def run_p1p2_job(config: P1P2Config, job: P1P2Job) -> dict[str, object]:
    if job.package in {"low_data", "real_diagnostics"}:
        return _classification_job(config, job)
    if job.package == "synthetic_ood":
        return _synthetic_ood_job(config, job)
    if job.package == "real_domain_ood":
        return _real_domain_ood_job(config, job)
    return _efficiency_job(config, job)


def _real_domain_ood_job(config: P1P2Config, job: P1P2Job) -> dict[str, object]:
    device = resolve_device(config.device)
    external = load_external_task("mit-bih", Path(".omx/data/external"), mitbih_beat_length=256)
    task = PACClassificationTask(
        external.name,
        external.train_inputs[..., :1],
        external.train_targets,
        external.validation_inputs[..., :1],
        external.validation_targets,
        external.test_inputs[..., :1],
        external.test_targets,
        external.output_dim,
    )
    experiment = _trial_adjusted_experiment(
        job,
        replace(
            _experiment_config(
                config,
                task,
                device,
                job.reference_model,
                reference_model_dim=job.reference_model_dim,
            ),
            learning_rate=job.learning_rate,
            weight_decay=job.weight_decay,
        ),
    )
    model, match = _build_evidence_classifier(job, experiment, task.class_count)
    outcome = train_classifier(
        model,
        task,
        experiment,
        device,
        job.seed,
        restore_best_validation=True,
    )
    validation_inputs = task.validation_inputs.to(device=device)
    validation_labels = task.validation_labels.to(device=device)
    test_inputs = task.test_inputs.to(device=device)
    test_labels = task.test_labels.to(device=device)
    id_support = torch.bincount(validation_labels.detach().cpu(), minlength=task.class_count)
    ood_support = torch.bincount(test_labels.detach().cpu(), minlength=task.class_count)
    common_classes = torch.nonzero((id_support > 0) & (ood_support > 0), as_tuple=False).flatten()
    if common_classes.numel() == 0:
        raise ValueError("MIT-BIH DS1/DS2 have no common-supported classes")
    id_common_mask = torch.isin(
        validation_labels, common_classes.to(device=validation_labels.device)
    )
    ood_common_mask = torch.isin(test_labels, common_classes.to(device=test_labels.device))
    id_metrics = classification_metric_bundle(
        model,
        validation_inputs[id_common_mask],
        validation_labels[id_common_mask],
        batch_size=experiment.batch_size,
    )
    ood_common_metrics = classification_metric_bundle(
        model,
        test_inputs[ood_common_mask],
        test_labels[ood_common_mask],
        batch_size=experiment.batch_size,
    )
    ood_full_metrics = classification_metric_bundle(
        model,
        test_inputs,
        test_labels,
        batch_size=experiment.batch_size,
    )
    class_names = external.class_names or tuple(str(index) for index in range(task.class_count))
    common_names = [class_names[int(index)] for index in common_classes.tolist()]
    return {
        "job_key": job.key,
        "package": "real_domain_ood",
        "result_schema_version": "pac_tf_real_domain_ood.v2_common_support",
        "protocol_id": "pac-tf-confirmatory-20260711-v1",
        "model": job.model,
        "reference_model": job.reference_model,
        "dataset_or_task": "mit-bih-ds1-to-ds2",
        "seed": job.seed,
        "selection_trial": job.selection_trial,
        "learning_rate": job.learning_rate,
        "weight_decay": job.weight_decay,
        "protocol_sha256": job.protocol_sha256,
        "parameter_match_tolerance": _reported_parameter_match_tolerance(job),
        "capacity_policy": job.capacity_policy or job.parameter_match_policy,
        "train_domain": "DS1 patient records excluding held-out DS1 validation records",
        "id_domain": "held-out DS1 patient records",
        "ood_domain": "DS2 patient-disjoint records",
        "signal_channels": "channel_0_MLII_or_primary_lead",
        "train_record_ids_json": json.dumps(list(dict.fromkeys(external.train_groups))),
        "id_record_ids_json": json.dumps(list(dict.fromkeys(external.validation_groups))),
        "ood_record_ids_json": json.dumps(list(dict.fromkeys(external.test_groups))),
        "class_names_json": json.dumps(list(class_names)),
        "id_class_support_json": json.dumps(id_support.tolist()),
        "ood_class_support_json": json.dumps(ood_support.tolist()),
        "common_supported_class_indices_json": json.dumps(common_classes.tolist()),
        "common_supported_class_names_json": json.dumps(common_names),
        "id_ood_comparison_scope": "intersection_of_nonzero_support_classes",
        "id_common_accuracy": id_metrics.accuracy,
        "id_common_macro_f1": id_metrics.macro_f1,
        "ood_common_accuracy": ood_common_metrics.accuracy,
        "ood_common_macro_f1": ood_common_metrics.macro_f1,
        "absolute_common_accuracy_drop": id_metrics.accuracy - ood_common_metrics.accuracy,
        "relative_common_accuracy_drop": (
            (id_metrics.accuracy - ood_common_metrics.accuracy) / id_metrics.accuracy
            if id_metrics.accuracy > 0.0
            else None
        ),
        "id_common_balanced_accuracy": id_metrics.balanced_accuracy,
        "ood_common_balanced_accuracy": ood_common_metrics.balanced_accuracy,
        "absolute_common_balanced_accuracy_drop": (
            id_metrics.balanced_accuracy - ood_common_metrics.balanced_accuracy
        ),
        "relative_common_balanced_accuracy_drop": (
            (id_metrics.balanced_accuracy - ood_common_metrics.balanced_accuracy)
            / id_metrics.balanced_accuracy
            if id_metrics.balanced_accuracy > 0.0
            else None
        ),
        "ood_full_5class_accuracy": ood_full_metrics.accuracy,
        "ood_full_5class_macro_f1": ood_full_metrics.macro_f1,
        "ood_full_5class_balanced_accuracy": ood_full_metrics.balanced_accuracy,
        "validation_loss": outcome.validation_loss,
        "ood_test_loss": outcome.test_loss,
        "best_epoch": outcome.best_epoch,
        **_capacity_result_fields(job, match.params, match.target_params),
        "architecture_metadata_json": _architecture_metadata_json(job),
        **_selection_provenance(job),
        "id_diagnostics_json": json.dumps(
            classification_diagnostics(
                model,
                validation_inputs,
                validation_labels,
                batch_size=experiment.batch_size,
            ),
            separators=(",", ":"),
        ),
        "ood_diagnostics_json": json.dumps(
            classification_diagnostics(
                model,
                test_inputs,
                test_labels,
                batch_size=experiment.batch_size,
            ),
            separators=(",", ":"),
        ),
        "status": "done",
    }


def _synthetic_ood_job(config: P1P2Config, job: P1P2Job) -> dict[str, object]:
    device = resolve_device(config.device)
    model_dim, modes = _selected_capacity(job.reference_model)
    experiment = _trial_adjusted_experiment(
        job,
        PACExperimentConfig(
            2048,
            512,
            512,
            64,
            raw_input_dim=4,
            output_dim=2,
            model_dim=model_dim,
            modes=modes,
            epochs=100,
            batch_size=64,
            learning_rate=job.learning_rate,
            weight_decay=job.weight_decay,
            seeds=(job.seed,),
            device=cast("PACDevice", device),
            output_dir=config.output_root,
        ),
    )
    endpoint_estimand = job.synthetic_estimand == "endpoint" or job.model in _EFFICIENT_MODEL_SPECS
    base = synthetic_ood_training_task(experiment, job.seed)
    if endpoint_estimand:
        base = _endpoint_task(base)
    if job.model in _EFFICIENT_MODEL_SPECS:
        with _MODEL_INIT_LOCK, torch.random.fork_rng(devices=[]):
            torch.manual_seed(job.seed)
            model = _EfficientHeadroomSyntheticEndpointRegressor(
                experiment, _EFFICIENT_MODEL_SPECS[job.model]
            )
        target_parameters = count_parameters(model)
        relative_error = 0.0
    else:
        model, target_parameters, relative_error = _build_synthetic_model(
            job.model,
            job.reference_model,
            experiment,
            job.seed,
            job.selection_trial,
            max_relative_error=0.055 if job.synthetic_estimand == "endpoint" else 0.05,
            explicit_target_params=job.synthetic_target_params,
        )
    if model is None or target_parameters is None or relative_error is None:
        message = f"required synthetic baseline unavailable: {job.model}"
        raise RuntimeError(message)
    if endpoint_estimand and job.model not in _EFFICIENT_MODEL_SPECS:
        model = _SequenceEndpointRegressor(model)
    outcome = train_regression_model(model, base, experiment, device, job.seed)
    rows: list[dict[str, object]] = []
    for condition in synthetic_ood_conditions(experiment, job.seed):
        id_targets = condition.id_targets[:, -1] if endpoint_estimand else condition.id_targets
        ood_targets = condition.ood_targets[:, -1] if endpoint_estimand else condition.ood_targets
        id_loss = evaluate_regression_loss(
            model,
            condition.id_inputs.to(device=device),
            id_targets.to(device=device),
        )
        ood_loss = evaluate_regression_loss(
            model,
            condition.ood_inputs.to(device=device),
            ood_targets.to(device=device),
        )
        id_nrmse = nrmse(id_loss, id_targets)
        ood_nrmse = nrmse(ood_loss, ood_targets)
        rows.append(
            {
                "family": condition.family,
                "level": condition.level,
                "id_test_loss": id_loss,
                "ood_test_loss": ood_loss,
                "id_nrmse": id_nrmse,
                "ood_nrmse": ood_nrmse,
                "absolute_nrmse_increase": ood_nrmse - id_nrmse,
                "relative_nrmse_increase": (
                    (ood_nrmse - id_nrmse) / id_nrmse if id_nrmse > 0.0 else None
                ),
                "paired_base_examples": True,
                "teacher_family": "fixed_damped_oscillator_zoh",
                "observable_delta_channel": True,
                "observable_missing_mask_channel": True,
                "diagnostic_slice": condition.family
                in {"sequence_length", "frequency", "additive_noise"},
            }
        )
    endpoint_note = "; record-level endpoint estimand" if endpoint_estimand else ""
    synthetic_protocol = (
        "paired exogenous examples; one fixed damped-oscillator ZOH teacher; "
        f"constant ID delta/mask channels; TEST-only noise/missingness perturbations{endpoint_note}"
    )
    return {
        "job_key": job.key,
        "package": "synthetic_ood",
        "protocol_id": "pac-tf-confirmatory-20260711-v1",
        "model": job.model,
        "reference_model": job.reference_model,
        "seed": job.seed,
        "selection_trial": job.selection_trial,
        "learning_rate": job.learning_rate,
        "weight_decay": job.weight_decay,
        "protocol_sha256": job.protocol_sha256,
        "synthetic_estimand": "endpoint" if endpoint_estimand else "sequence",
        "synthetic_target_params": job.synthetic_target_params,
        "id_test_loss": outcome.test_loss,
        "id_test_nrmse": nrmse(outcome.test_loss, base.test_targets),
        "ood_sweep_json": json.dumps(rows, separators=(",", ":")),
        "length_frequency_noise_slices_json": json.dumps(
            [
                row
                for row in rows
                if row["family"] in {"sequence_length", "frequency", "additive_noise"}
            ],
            separators=(",", ":"),
        ),
        "ood_family_count": len({str(row["family"]) for row in rows}),
        "ood_condition_count": len(rows),
        "params_trainable": count_parameters(model),
        "target_params": target_parameters,
        "relative_param_error": relative_error,
        "architecture_metadata_json": _architecture_metadata_json(job),
        "synthetic_protocol": synthetic_protocol,
        "status": "done",
    }


@dataclass(frozen=True, slots=True)
class SyntheticOODCondition:
    family: str
    level: str
    id_inputs: Tensor
    id_targets: Tensor
    ood_inputs: Tensor
    ood_targets: Tensor


def synthetic_ood_training_task(config: PACExperimentConfig, seed: int) -> PACRegressionTask:
    train_inputs, train_targets = _paired_oscillator_split(
        config.sample_count, config.sequence_length, seed + 101
    )
    validation_inputs, validation_targets = _paired_oscillator_split(
        config.validation_count, config.sequence_length, seed + 211
    )
    test_inputs, test_targets = _paired_oscillator_split(
        config.test_count, config.sequence_length, seed + 307
    )
    return PACRegressionTask(
        "paired_fixed_damped_oscillator_id",
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        test_inputs,
        test_targets,
        true_delay=4,
        true_frequency=pi / 4,
        true_frequencies=(pi / 4,),
        true_dampings=(0.8,),
        mechanism_expectation="positive",
    )


def _endpoint_task(task: PACRegressionTask) -> PACRegressionTask:
    return PACRegressionTask(
        task.label + "_endpoint",
        task.train_inputs,
        task.train_targets[:, -1],
        task.validation_inputs,
        task.validation_targets[:, -1],
        task.test_inputs,
        task.test_targets[:, -1],
        true_delay=task.true_delay,
        true_frequency=task.true_frequency,
        true_frequencies=task.true_frequencies,
        true_dampings=task.true_dampings,
        mechanism_expectation=task.mechanism_expectation,
    )


def synthetic_ood_conditions(config: PACExperimentConfig, seed: int) -> list[SyntheticOODCondition]:
    specifications: list[tuple[str, str, dict[str, float | int | bool]]] = []
    specifications.extend(
        ("sampling_rate", f"dt_{delta:g}", {"delta": delta})
        for delta in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0)
    )
    specifications.extend(
        (
            "irregular_timestamps_missingness",
            level,
            {"irregular": True, "missing_rate": missing_rate},
        )
        for level, missing_rate in (("moderate", 0.1), ("hard", 0.3))
    )
    specifications.extend(
        (
            ("sequence_length", "128", {"sequence_length": 128}),
            ("sequence_length", "256", {"sequence_length": 256}),
            ("delay", "8", {"delay": 8}),
            ("delay", "12", {"delay": 12}),
            ("additive_noise", "0.05", {"noise": 0.05}),
            ("additive_noise", "0.1", {"noise": 0.1}),
            ("damping", "1.2", {"damping": 1.2}),
            ("damping", "1.6", {"damping": 1.6}),
            ("frequency", "pi_over_8", {"frequency": pi / 8}),
            ("frequency", "pi_over_2", {"frequency": pi / 2}),
        )
    )
    conditions: list[SyntheticOODCondition] = []
    for index, (family, level, shift) in enumerate(specifications):
        ood_length = int(shift.get("sequence_length", config.sequence_length))
        paired_length = max(config.sequence_length, ood_length)
        base_values = _paired_base_values(config.test_count, paired_length, seed + 401)
        id_values = base_values[:, : config.sequence_length]
        id_inputs, id_targets = _oscillator_observation(id_values)
        ood_values = base_values[:, :ood_length]
        ood_inputs, ood_targets = _oscillator_observation(
            ood_values,
            delta=float(shift.get("delta", 1.0)),
            irregular=bool(shift.get("irregular", False)),
            missing_rate=float(shift.get("missing_rate", 0.0)),
            noise=float(shift.get("noise", 0.0)),
            damping=float(shift.get("damping", 0.8)),
            frequency=float(shift.get("frequency", pi / 4)),
            delay=int(shift.get("delay", 4)),
            perturbation_seed=seed + 997 + index,
        )
        conditions.append(
            SyntheticOODCondition(
                family,
                level,
                id_inputs,
                id_targets,
                ood_inputs,
                ood_targets,
            )
        )
    return conditions


def synthetic_ood_tasks(
    config: PACExperimentConfig, seed: int
) -> list[tuple[str, str, PACRegressionTask]]:
    return [
        (
            condition.family,
            condition.level,
            PACRegressionTask(
                f"ood_{condition.family}_{condition.level}",
                condition.ood_inputs[:1],
                condition.ood_targets[:1],
                condition.ood_inputs[:1],
                condition.ood_targets[:1],
                condition.ood_inputs,
                condition.ood_targets,
            ),
        )
        for condition in synthetic_ood_conditions(config, seed)
    ]


def _paired_oscillator_split(count: int, length: int, seed: int) -> tuple[Tensor, Tensor]:
    return _oscillator_observation(_paired_base_values(count, length, seed))


def _paired_base_values(count: int, length: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, length, 2, generator=generator)


def _oscillator_observation(
    base_values: Tensor,
    *,
    delta: float = 1.0,
    irregular: bool = False,
    missing_rate: float = 0.0,
    noise: float = 0.0,
    damping: float = 0.8,
    frequency: float = pi / 4,
    delay: int = 4,
    perturbation_seed: int = 0,
) -> tuple[Tensor, Tensor]:
    samples, length, _ = base_values.shape
    generator = torch.Generator().manual_seed(perturbation_seed)
    if irregular:
        deltas = 0.5 + torch.rand(samples, length, 1, generator=generator)
    else:
        deltas = torch.full((samples, length, 1), delta)
    mask = torch.ones(samples, length, 1)
    if missing_rate > 0.0:
        mask = (torch.rand(samples, length, 1, generator=generator) >= missing_rate).to(
            base_values.dtype
        )
        mask[:, 0] = 1.0
    observed_values = base_values.clone()
    if noise > 0.0:
        observed_values = observed_values + noise * torch.randn(
            observed_values.shape, generator=generator
        )
    observed_values = observed_values * mask
    inputs = torch.cat((observed_values, deltas, mask), dim=-1)
    targets = _damped_oscillator_targets(
        base_values,
        deltas,
        damping=damping,
        frequency=frequency,
        delay=delay,
    )
    return inputs, targets


def _damped_oscillator_targets(
    values: Tensor,
    deltas: Tensor,
    *,
    damping: float,
    frequency: float,
    delay: int,
) -> Tensor:
    state = torch.zeros(values.shape[0], 2, dtype=values.dtype)
    outputs: list[Tensor] = []
    for index in range(values.shape[1]):
        dt = deltas[:, index, 0]
        decay = torch.exp(-damping * dt)
        angle = frequency * dt
        real = decay * (torch.cos(angle) * state[:, 0] - torch.sin(angle) * state[:, 1])
        imag = decay * (torch.sin(angle) * state[:, 0] + torch.cos(angle) * state[:, 1])
        forcing = values[:, max(0, index - delay)] if index >= delay else torch.zeros_like(state)
        state = torch.stack((real, imag), dim=-1) + (1.0 - decay).unsqueeze(-1) * forcing
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def _classification_job(config: P1P2Config, job: P1P2Job) -> dict[str, object]:
    device = resolve_device(config.device)
    dataset = ensure_ucr_dataset(
        job.dataset,
        Path(".omx/data/ucr"),
        allow_download=True,
        require_train_label_space=True,
    )
    if _reuse_frozen_p0(job):
        return _frozen_p0_classification_job(config, job, dataset, device)
    task = clean_low_data_task(dataset, job.ratio, job.seed)
    experiment = _trial_adjusted_experiment(
        job,
        replace(
            _experiment_config(
                config,
                task,
                device,
                job.reference_model,
                reference_model_dim=job.reference_model_dim,
            ),
            learning_rate=job.learning_rate,
            weight_decay=job.weight_decay,
        ),
    )
    model, match = _build_evidence_classifier(job, experiment, task.class_count)
    target_parameters = match.target_params
    model_parameters = match.params
    outcome = train_classifier(
        model,
        task,
        experiment,
        device,
        job.seed,
        restore_best_validation=True,
    )
    test_inputs = task.test_inputs.to(device=device)
    test_labels = task.test_labels.to(device=device)
    metrics = classification_metric_bundle(
        model,
        test_inputs,
        test_labels,
        batch_size=experiment.batch_size,
    )
    class_counts = torch.bincount(task.train_labels, minlength=task.class_count)
    official_train_count = dataset.train_inputs.shape[0]
    optimization_fold_count = official_train_count - task.validation_inputs.shape[0]
    row: dict[str, object] = {
        "job_key": job.key,
        "package": job.package,
        "protocol_id": "pac-tf-confirmatory-20260711-v1",
        "model": job.model,
        "reference_model": job.reference_model,
        "dataset_or_task": job.dataset,
        "seed": job.seed,
        "selection_trial": job.selection_trial,
        "learning_rate": job.learning_rate,
        "weight_decay": job.weight_decay,
        "protocol_sha256": job.protocol_sha256,
        "ratio_one_fit_policy": job.ratio_one_fit_policy,
        "parameter_match_tolerance": _reported_parameter_match_tolerance(job),
        "capacity_policy": job.capacity_policy or job.parameter_match_policy,
        "data_ratio": job.ratio,
        "requested_ratio": job.ratio,
        "realized_count": task.train_inputs.shape[0],
        "optimization_fold_count": optimization_fold_count,
        "requested_optimization_fold_fraction": job.ratio,
        "realized_optimization_fold_fraction": (
            task.train_inputs.shape[0] / optimization_fold_count
        ),
        "realized_official_train_fraction": task.train_inputs.shape[0] / official_train_count,
        "realized_ratio": task.train_inputs.shape[0] / official_train_count,
        "min_class_count": int(class_counts.min().item()),
        "official_train_count": official_train_count,
        **_capacity_result_fields(job, model_parameters, target_parameters),
        "architecture_metadata_json": _architecture_metadata_json(job),
        **_selection_provenance(job),
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "test_accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "best_epoch": outcome.best_epoch,
        "normalization_fit": "sampled_optimization_fold_only",
        "checkpoint_policy": "minimum_validation_loss",
        "test_policy": "single_post_training_evaluation_for_prespecified_estimand",
        "status": "done",
    }
    if job.package == "real_diagnostics":
        row.update(
            classification_diagnostics(
                model,
                test_inputs,
                test_labels,
                batch_size=experiment.batch_size,
            )
        )
        row["real_corruption_ood_json"] = corruption_diagnostics(
            model,
            test_inputs,
            test_labels,
            job.seed,
            batch_size=experiment.batch_size,
        )
        row["real_ood_scope"] = "corruption_shift_not_domain_shift"
    return row


def _frozen_p0_classification_job(
    config: P1P2Config,
    job: P1P2Job,
    dataset: UCRDataset,
    device: str,
) -> dict[str, object]:
    payload = _load_frozen_p0_checkpoint(config, job)
    normalization = payload["normalization"]
    if not isinstance(normalization, dict):
        raise TypeError("P0 checkpoint normalization must be an object")
    mean = _required_float(normalization, "mean")
    std = _required_float(normalization, "std")
    if std <= 0.0:
        raise ValueError("P0 checkpoint normalization std must be positive")
    official_train_count = int(dataset.train_inputs.shape[0])
    if int(normalization.get("official_train_count", -1)) != official_train_count:
        raise ValueError("P0 checkpoint official-TRAIN count does not match loaded dataset")
    normalized_test = (dataset.test_inputs - mean) / std
    task = PACClassificationTask(
        dataset.name,
        (dataset.train_inputs - mean) / std,
        dataset.train_labels,
        normalized_test[:0],
        dataset.test_labels[:0],
        normalized_test,
        dataset.test_labels,
        dataset.class_count,
    )
    experiment = _trial_adjusted_experiment(
        job,
        replace(
            _experiment_config(
                config,
                task,
                device,
                job.reference_model,
                reference_model_dim=job.reference_model_dim,
            ),
            epochs=job.refit_epochs,
            learning_rate=job.learning_rate,
            weight_decay=job.weight_decay,
        ),
    )
    with _MODEL_INIT_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.seed)
        model, match = _build_configured_confirmatory_classifier(job, experiment, task.class_count)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("P0 checkpoint has no state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    test_inputs = task.test_inputs.to(device=device)
    test_labels = task.test_labels.to(device=device)
    reevaluated = classification_metric_bundle(model, test_inputs, test_labels)
    p0_metrics = json.loads(job.p0_metrics_json)
    if not isinstance(p0_metrics, dict):
        raise TypeError("P0 metric provenance must be an object")
    _validate_recomputed_p0_metrics(p0_metrics, reevaluated)
    row: dict[str, object] = {
        "job_key": job.key,
        "package": job.package,
        "protocol_id": "pac-tf-confirmatory-20260711-v1",
        "model": job.model,
        "reference_model": job.reference_model,
        "dataset_or_task": job.dataset,
        "seed": job.seed,
        "selection_trial": job.selection_trial,
        "refit_epochs": job.refit_epochs,
        "learning_rate": job.learning_rate,
        "weight_decay": job.weight_decay,
        "protocol_sha256": job.protocol_sha256,
        "ratio_one_fit_policy": job.ratio_one_fit_policy,
        "parameter_match_tolerance": _reported_parameter_match_tolerance(job),
        "capacity_policy": job.capacity_policy or job.parameter_match_policy,
        "data_ratio": 1.0,
        "requested_ratio": 1.0,
        "requested_optimization_fold_fraction": None,
        "optimization_fold_count": None,
        "realized_count": official_train_count,
        "realized_optimization_fold_fraction": None,
        "realized_official_train_fraction": 1.0,
        "realized_ratio": 1.0,
        "min_class_count": int(
            torch.bincount(dataset.train_labels, minlength=dataset.class_count).min().item()
        ),
        "official_train_count": official_train_count,
        **_capacity_result_fields(job, match.params, match.target_params),
        "architecture_metadata_json": _architecture_metadata_json(job),
        "train_loss": _required_float(p0_metrics, "train_loss"),
        "validation_loss": None,
        "test_loss": _required_float(p0_metrics, "test_loss"),
        "test_accuracy": _required_float(p0_metrics, "test_accuracy"),
        "macro_f1": _required_float(p0_metrics, "macro_f1"),
        "weighted_f1": _required_float(p0_metrics, "weighted_f1"),
        "balanced_accuracy": _required_float(p0_metrics, "balanced_accuracy"),
        "best_epoch": None,
        "normalization_fit": "all_official_train_from_frozen_p0_checkpoint",
        "checkpoint_policy": "reuse_exact_p0_full_train_checkpoint_no_retraining",
        "p0_job_key": job.p0_job_key,
        "p0_checkpoint_path": job.p0_checkpoint_path,
        "p0_checkpoint_sha256": job.p0_checkpoint_sha256,
        "test_policy": "P0 endpoint reused; no second optimization or TEST-driven selection",
        "status": "done",
    }
    if job.package == "real_diagnostics":
        row.update(classification_diagnostics(model, test_inputs, test_labels))
        row["real_corruption_ood_json"] = corruption_diagnostics(
            model, test_inputs, test_labels, job.seed
        )
        row["real_ood_scope"] = "corruption_shift_not_domain_shift"
    return row


def _load_frozen_p0_checkpoint(config: P1P2Config, job: P1P2Job) -> dict[str, object]:
    if not all(
        (job.p0_job_key, job.p0_checkpoint_path, job.p0_checkpoint_sha256, job.p0_metrics_json)
    ):
        raise ValueError("ratio=1 requires complete frozen P0 checkpoint provenance")
    path = Path(job.p0_checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.resolve().is_relative_to((config.unseen_root / "checkpoints").resolve()):
        raise ValueError("P0 checkpoint escapes the configured unseen root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != job.p0_checkpoint_sha256:
        raise ValueError("P0 checkpoint SHA-256 changed after P1 enqueue")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "pac_unseen_final_checkpoint.v1"
    ):
        raise ValueError("unsupported P0 checkpoint schema")
    expected = {
        "job_key": job.p0_job_key,
        "dataset": job.dataset,
        "seed": job.seed,
        "family": job.model,
        "reference_model": job.reference_model,
        "validation_trial": job.selection_trial,
        "refit_epochs": job.refit_epochs,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("P0 checkpoint identity changed after P1 enqueue")
    return cast("dict[str, object]", payload)


def _reuse_frozen_p0(job: P1P2Job) -> bool:
    if job.ratio != 1.0:
        return False
    if job.ratio_one_fit_policy == "frozen_full_train":
        return True
    if job.ratio_one_fit_policy == "optimization_fold_validation":
        return False
    if job.ratio_one_fit_policy == "legacy_model_default":
        return job.model not in _EFFICIENT_MODEL_SPECS
    message = f"unsupported ratio-one fit policy: {job.ratio_one_fit_policy}"
    raise ValueError(message)


def _parameter_match_tolerance(job: P1P2Job) -> float:
    tolerance = job.parameter_match_tolerance
    if tolerance is None:
        return 0.051 if job.reference_model in _EFFICIENT_REFERENCE_SPECS else 0.05
    if tolerance <= 0.0:
        message = "parameter_match_tolerance must be positive"
        raise ValueError(message)
    return tolerance


def _uses_native_selected_width(job: P1P2Job) -> bool:
    return bool(job.selection_source and job.selected_model_width > 0)


def _reported_parameter_match_tolerance(job: P1P2Job) -> float | None:
    if _uses_native_selected_width(job):
        return None
    return _parameter_match_tolerance(job)


def _build_configured_confirmatory_classifier(
    job: P1P2Job,
    config: PACExperimentConfig,
    class_count: int,
) -> tuple[nn.Module, ConfirmatoryMatch]:
    if job.parameter_match_tolerance is None:
        return build_matched_confirmatory_classifier(
            cast("ConfirmatoryFamily", job.model),
            job.reference_model,
            config,
            class_count,
            validation_trial=job.selection_trial,
        )
    return build_matched_confirmatory_classifier(
        cast("ConfirmatoryFamily", job.model),
        job.reference_model,
        config,
        class_count,
        tolerance=_parameter_match_tolerance(job),
        validation_trial=job.selection_trial,
    )


def _required_float(values: dict[str, object], name: str) -> float:
    value = values.get(name)
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _validate_recomputed_p0_metrics(
    stored: dict[str, object], reevaluated: object, tolerance: float = 1.0e-6
) -> None:
    from .pac_types import PACClassificationMetrics  # noqa: PLC0415

    if not isinstance(reevaluated, PACClassificationMetrics):
        raise TypeError("reevaluated P0 metrics have an unexpected type")
    comparisons = {
        "test_accuracy": reevaluated.accuracy,
        "macro_f1": reevaluated.macro_f1,
        "weighted_f1": reevaluated.weighted_f1,
        "balanced_accuracy": reevaluated.balanced_accuracy,
    }
    for name, current in comparisons.items():
        if abs(_required_float(stored, name) - current) > tolerance:
            raise ValueError(f"P0 checkpoint re-evaluation mismatch for {name}")


def clean_low_data_task(
    dataset: UCRDataset,
    ratio: float,
    seed: int,
    validation_ratio: float = 0.2,
) -> PACClassificationTask:
    train_indices, validation_indices = stratified_partition_indices(
        dataset.train_labels, validation_ratio, seed
    )
    raw_train = dataset.train_inputs.index_select(0, train_indices)
    train_labels = dataset.train_labels.index_select(0, train_indices)
    raw_train, train_labels = stratified_subset(raw_train, train_labels, ratio, seed + 17)
    raw_validation = dataset.train_inputs.index_select(0, validation_indices)
    validation_labels = dataset.train_labels.index_select(0, validation_indices)
    mean = raw_train.mean()
    std = raw_train.std(unbiased=False).clamp_min(1.0e-6)
    return PACClassificationTask(
        dataset.name,
        (raw_train - mean) / std,
        train_labels,
        (raw_validation - mean) / std,
        validation_labels,
        (dataset.test_inputs - mean) / std,
        dataset.test_labels,
        dataset.class_count,
    )


def _experiment_config(
    config: P1P2Config,
    task: PACClassificationTask,
    device: str,
    reference_model: str,
    *,
    reference_model_dim: int = 0,
) -> PACExperimentConfig:
    model_dim, modes = _selected_capacity(reference_model)
    if reference_model_dim:
        model_dim = reference_model_dim
    return PACExperimentConfig(
        sample_count=task.train_inputs.shape[0],
        validation_count=task.validation_inputs.shape[0],
        test_count=task.test_inputs.shape[0],
        sequence_length=task.train_inputs.shape[1],
        raw_input_dim=task.train_inputs.shape[-1],
        output_dim=task.class_count,
        model_dim=model_dim,
        modes=modes,
        epochs=100,
        batch_size=64,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        seeds=(7, 11, 19, 23, 31),
        device=cast("PACDevice", device),
        output_dir=config.output_root,
    )


def _trial_adjusted_experiment(
    job: P1P2Job, experiment: PACExperimentConfig
) -> PACExperimentConfig:
    if job.selection_source:
        family = "pac_tf" if job.model in _EFFICIENT_MODEL_SPECS else job.model
        spec = confirmatory_trial_spec(cast("ConfirmatoryFamily", family), job.selection_trial)
        if job.weight_decay != spec.weight_decay:
            raise ValueError("ID-selected P1/P2 weight decay differs from the locked trial")
        batch_size = job.batch_size or spec.batch_size
        grad_clip_norm = job.grad_clip_norm or spec.grad_clip_norm
        return replace(
            experiment,
            batch_size=batch_size,
            grad_clip_norm=grad_clip_norm,
        )
    if job.model in _EFFICIENT_MODEL_SPECS:
        return replace(experiment, batch_size=64, grad_clip_norm=1.0)
    family = cast("ConfirmatoryFamily", job.model)
    spec = confirmatory_trial_spec(family, job.selection_trial)
    if job.learning_rate != spec.learning_rate or job.weight_decay != spec.weight_decay:
        raise ValueError("P1/P2 optimizer values differ from the locked family trial")
    return replace(
        experiment,
        batch_size=spec.batch_size,
        grad_clip_norm=spec.grad_clip_norm,
    )


def _architecture_metadata_json(job: P1P2Job) -> str:
    if job.model in _EFFICIENT_MODEL_SPECS:
        internal_spec = _EFFICIENT_MODEL_SPECS[job.model]
        if internal_spec == "EFP16":
            model_dim = job.reference_model_dim or 32
            canonical = json.dumps(
                {
                    "family": job.model,
                    "paper_model": "ALPHABET",
                    "internal_spec": "EFP16",
                    "model_dim": model_dim,
                    "modes": 16,
                    "analysis": "degree_normalized_full_rate_edge_frame",
                    "semi_orthogonal_projection": True,
                    "pairing_boundary": False,
                    "random_pair_origin_training": False,
                    "dual_phase_ensemble_inference": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if job.architecture_metadata_json and job.architecture_metadata_json != canonical:
                raise ValueError("EFP16 architecture metadata differs from the frozen contract")
            return canonical
        canonical = json.dumps(
            {
                "family": job.model,
                "paper_model": "ALPHABET",
                "internal_spec": internal_spec,
                "model_dim": 64,
                "modes": 16,
                "time_weighted_haar": True,
                "shared_core": True,
                "convex_band_fusion": True,
                "random_pair_origin_training": internal_spec == "PA2WP",
                "dual_phase_ensemble_inference": internal_spec == "PA2WP",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if job.architecture_metadata_json and job.architecture_metadata_json != canonical:
            raise ValueError("WP architecture metadata differs from the frozen contract")
        return canonical
    canonical = json.dumps(
        confirmatory_implementation_metadata(
            cast("ConfirmatoryFamily", job.model), job.selection_trial
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    if job.architecture_metadata_json and job.architecture_metadata_json != canonical:
        raise ValueError("P1/P2 architecture metadata differs from the locked family trial")
    return canonical


def _selection_provenance(job: P1P2Job) -> dict[str, object]:
    return {
        "selection_source": job.selection_source,
        "selection_config_key": job.selection_config_key,
        "selection_artifact_sha256": job.selection_artifact_sha256,
        "selected_model_width": job.selected_model_width or None,
        "reference_model_dim": job.reference_model_dim or None,
        "selected_batch_size": job.batch_size or None,
        "selected_grad_clip_norm": job.grad_clip_norm or None,
    }


def _capacity_result_fields(
    job: P1P2Job, parameters: int, reference_parameters: int
) -> dict[str, object]:
    relative_error = abs(parameters - reference_parameters) / reference_parameters
    return {
        "params_trainable": parameters,
        "reference_params_trainable": reference_parameters,
        "parameter_ratio_to_alphabet": parameters / reference_parameters,
        "target_params": reference_parameters,
        "relative_param_error": None if _uses_native_selected_width(job) else relative_error,
    }


def _efficiency_job(config: P1P2Config, job: P1P2Job) -> dict[str, object]:
    device = resolve_device(config.device)
    model_dim, modes = _selected_capacity(job.reference_model)
    experiment = _trial_adjusted_experiment(
        job,
        PACExperimentConfig(
            64,
            16,
            16,
            job.length,
            raw_input_dim=1,
            output_dim=5,
            model_dim=model_dim,
            modes=modes,
            device=cast("PACDevice", device),
            output_dir=config.output_root,
        ),
    )
    model, match = _build_evidence_classifier(job, experiment, 5)
    estimated_attention_bytes = _dense_attention_lower_bound_bytes(job)
    device_capacity_bytes = _device_capacity_bytes(device)
    if (
        estimated_attention_bytes is not None
        and device_capacity_bytes is not None
        and estimated_attention_bytes > int(0.8 * device_capacity_bytes)
    ):
        reason = (
            "dense_attention_score_lower_bound_exceeds_80pct_of_device_capacity:"
            f"{estimated_attention_bytes}>{int(0.8 * device_capacity_bytes)}"
        )
        return _censored_efficiency_row(
            job,
            match.params,
            match.target_params,
            match.relative_error,
            outcome_status="resource_limit",
            reason=reason,
            estimated_attention_bytes=estimated_attention_bytes,
            device_capacity_bytes=device_capacity_bytes,
            peak_memory_mb=None,
        )
    try:
        model = model.to(device=device)
        inputs = torch.randn(job.batch_size, job.length, 1, device=device)
        labels = torch.randint(0, 5, (job.batch_size,), device=device)
        if job.runtime == "train":
            latency_ms, latency_iqr_ms, latency_samples, peak_memory_mb = _measure_training(
                model,
                inputs,
                labels,
                device,
                learning_rate=job.learning_rate,
                weight_decay=job.weight_decay,
            )
        else:
            model.eval()
            if job.runtime == "compiled":
                if isinstance(model, TightFrameClassifier):
                    model = prepare_tight_frame_inference(model, compile_mode="reduce-overhead")
                else:
                    model = cast("nn.Module", torch.compile(model, mode="reduce-overhead"))
            latency_ms, latency_iqr_ms, latency_samples, peak_memory_mb = _measure_forward(
                model, inputs, device
            )
    except Exception as error:
        outcome_status, reason = _classify_efficiency_resource_error(error, job.runtime)
        if outcome_status is None or reason is None:
            raise
        peak_memory_mb = _safe_peak_memory(device)
        _cleanup_cuda_after_resource_limit(device)
        return _censored_efficiency_row(
            job,
            match.params,
            match.target_params,
            match.relative_error,
            outcome_status=outcome_status,
            reason=reason,
            estimated_attention_bytes=estimated_attention_bytes,
            device_capacity_bytes=device_capacity_bytes,
            peak_memory_mb=peak_memory_mb,
        )
    tokens = job.batch_size * job.length
    return _efficiency_base_row(
        job,
        match.params,
        match.target_params,
        match.relative_error,
    ) | {
        "outcome_status": "measured",
        "resource_limit_reason": "",
        "estimated_attention_bytes": estimated_attention_bytes,
        "device_capacity_bytes": device_capacity_bytes,
        "latency_ms": latency_ms,
        "latency_iqr_ms": latency_iqr_ms,
        "latency_samples_json": json.dumps(latency_samples, separators=(",", ":")),
        "measurement_repetitions": len(latency_samples),
        "tokens_per_second": tokens * 1_000.0 / max(latency_ms, 1.0e-12),
        "peak_memory_mb": peak_memory_mb,
        "runtime_environment_json": json.dumps(
            _runtime_environment(device, job.runtime), sort_keys=True, separators=(",", ":")
        ),
    }


def _efficiency_base_row(
    job: P1P2Job,
    params: int,
    target_params: int,
    relative_param_error: float,
) -> dict[str, object]:
    return {
        "job_key": job.key,
        "package": "efficiency",
        "protocol_id": "pac-tf-confirmatory-20260711-v1",
        "model": job.model,
        "reference_model": job.reference_model,
        "seed": job.seed,
        "sequence_length": job.length,
        "batch_size": job.batch_size,
        "runtime": job.runtime,
        "selection_trial": job.selection_trial,
        "learning_rate": job.learning_rate,
        "weight_decay": job.weight_decay,
        "protocol_sha256": job.protocol_sha256,
        "params_trainable": params,
        "target_params": target_params,
        "relative_param_error": relative_param_error,
        "architecture_metadata_json": _architecture_metadata_json(job),
        "status": "done",
    }


def _censored_efficiency_row(
    job: P1P2Job,
    params: int,
    target_params: int,
    relative_param_error: float,
    *,
    outcome_status: str,
    reason: str,
    estimated_attention_bytes: int | None,
    device_capacity_bytes: int | None,
    peak_memory_mb: float | None,
) -> dict[str, object]:
    return _efficiency_base_row(job, params, target_params, relative_param_error) | {
        "outcome_status": outcome_status,
        "resource_limit_reason": reason,
        "estimated_attention_bytes": estimated_attention_bytes,
        "device_capacity_bytes": device_capacity_bytes,
        "latency_ms": None,
        "tokens_per_second": None,
        "peak_memory_mb": peak_memory_mb,
        "latency_iqr_ms": None,
        "latency_samples_json": "[]",
        "measurement_repetitions": 0,
        "runtime_environment_json": json.dumps(
            _runtime_environment(
                "cuda" if device_capacity_bytes is not None else "cpu", job.runtime
            ),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _dense_attention_lower_bound_bytes(job: P1P2Job) -> int | None:
    if job.model != "transformer":
        return None
    # One dense fp32 attention-score matrix is an unavoidable lower bound for
    # the repository's non-streaming Transformer implementation. Training and
    # backward normally retain additional matrices, so this is conservative.
    return job.batch_size * job.length * job.length * torch.float32.itemsize


def _device_capacity_bytes(device: str) -> int | None:
    if device != "cuda":
        return None
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)


def _classify_efficiency_resource_error(
    error: Exception, runtime: str
) -> tuple[str | None, str | None]:
    name = type(error).__name__
    module = type(error).__module__
    message = str(error).lower()
    if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) or any(
        marker in message
        for marker in ("out of memory", "cannot allocate memory", "can't allocate memory")
    ):
        return "resource_limit", f"{name}: {error}"
    compile_names = {"Unsupported", "InvalidBackend", "BackendCompilerFailed"}
    compile_markers = ("torch.compile", "torch._dynamo", "dynamo unsupported")
    if runtime == "compiled" and (
        name in compile_names
        or module.startswith(("torch._dynamo", "torch._inductor"))
        or any(marker in message for marker in compile_markers)
    ):
        return "compile_unsupported", f"{name}: {error}"
    return None, None


def _safe_peak_memory(device: str) -> float | None:
    try:
        return _peak_memory(device)
    except RuntimeError:
        return None


def _cleanup_cuda_after_resource_limit(device: str) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        return
    with suppress(RuntimeError):
        torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _measure_forward(
    model: torch.nn.Module, inputs: Tensor, device: str
) -> tuple[float, float, tuple[float, ...], float | None]:
    with torch.no_grad():
        for _ in range(5):
            model(inputs)
        _sync(device)
        samples: list[float] = []
        peaks: list[float] = []
        for _ in range(5):
            _reset_peak(device)
            started = perf_counter()
            for _ in range(20):
                model(inputs)
            _sync(device)
            samples.append((perf_counter() - started) * 50.0)
            peak = _peak_memory(device)
            if peak is not None:
                peaks.append(peak)
    latency, iqr = _median_iqr(samples)
    return latency, iqr, tuple(samples), max(peaks) if peaks else None


def _measure_training(
    model: torch.nn.Module,
    inputs: Tensor,
    labels: Tensor,
    device: str,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[float, float, tuple[float, ...], float | None]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(inputs), labels).backward()
        optimizer.step()
    _sync(device)
    samples: list[float] = []
    peaks: list[float] = []
    for _ in range(5):
        _reset_peak(device)
        started = perf_counter()
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(inputs), labels).backward()
            optimizer.step()
        _sync(device)
        samples.append((perf_counter() - started) * 100.0)
        peak = _peak_memory(device)
        if peak is not None:
            peaks.append(peak)
    latency, iqr = _median_iqr(samples)
    return latency, iqr, tuple(samples), max(peaks) if peaks else None


def _median_iqr(samples: list[float]) -> tuple[float, float]:
    values = torch.tensor(samples, dtype=torch.float64)
    return float(values.median().item()), float(
        (torch.quantile(values, 0.75) - torch.quantile(values, 0.25)).item()
    )


def _runtime_environment(device: str, runtime: str) -> dict[str, object]:
    cuda_version = getattr(torch.version, "cuda", None)
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": cuda_version,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device,
        "runtime": runtime,
        "compile_mode": "reduce-overhead" if runtime == "compiled" else "none",
    }
    if device == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        environment.update(
            {
                "device_index": index,
                "device_name": properties.name,
                "device_total_memory_bytes": properties.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return environment


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_memory(device: str) -> float | None:
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device == "cuda" else None


def _build_synthetic_model(
    model: str,
    reference_model: str,
    config: PACExperimentConfig,
    seed: int,
    validation_trial: int,
    *,
    max_relative_error: float = 0.05,
    explicit_target_params: int | None = None,
) -> tuple[torch.nn.Module | None, int | None, float | None]:
    with _MODEL_INIT_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if explicit_target_params is None:
            reference = build_reference_regressor(reference_model, config)
            if reference is None:
                return None, None, None
            target = count_parameters(reference)
        else:
            reference = None
            target = explicit_target_params
        if model == "pac_tf":
            if reference is None:
                return None, target, None
            return reference, target, 0.0
        best: tuple[nn.Module, int, float] | None = None
        max_width = 256 if explicit_target_params is not None else 128
        for width in range(1, max_width + 1):
            try:
                candidate = _synthetic_candidate(model, width, config, validation_trial)
            except (ImportError, ModuleNotFoundError, RuntimeError, ValueError, AssertionError):
                continue
            parameters = count_parameters(candidate)
            error = abs(parameters - target) / target
            if best is None or (error, width) < (best[2], count_parameters(best[0])):
                best = candidate, parameters, error
            if parameters >= target:
                break
        if best is None or best[2] > max_relative_error:
            return None, target, None
        return best[0], target, best[2]


def _synthetic_candidate(
    model: str, width: int, config: PACExperimentConfig, validation_trial: int
) -> nn.Module:
    spec = confirmatory_trial_spec(cast("ConfirmatoryFamily", model), validation_trial)
    active = replace(config, model_dim=width, output_dim=config.output_dim)
    if model == "tcn":
        return TCNRegressor(
            raw_input_dim=config.raw_input_dim,
            channels=width,
            levels=spec.depth,
            output_dim=config.output_dim,
        )
    if model == "cnn1d":
        return CNN1DRegressor(
            raw_input_dim=config.raw_input_dim,
            channels=(width, width, width),
            output_dim=config.output_dim,
        )
    if model in {"gru", "lstm", "transformer"}:
        return build_regression_model(cast("PACModelName", model), active)
    if model == "mamba":
        return MambaSSMRegressor(
            raw_input_dim=config.raw_input_dim,
            model_dim=width,
            output_dim=config.output_dim,
        )
    if model == "s4d":
        return S4DRegressor(
            config,
            model_dim=width,
            modes=max(1, min(spec.state_size, width // 2)),
        )
    if model == "inception_time":
        return _InceptionSequenceRegressor(config.raw_input_dim, width, config.output_dim)
    message = f"unsupported synthetic baseline: {model}"
    raise ValueError(message)


class _InceptionSequenceRegressor(nn.Module):
    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(input_dim, width, kernel_size=kernel, padding="same")
            for kernel in (9, 19, 39)
        )
        self.projection = nn.Linear(3 * width, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        channels = inputs.transpose(1, 2)
        features = torch.cat([branch(channels) for branch in self.branches], dim=1)
        return self.projection(torch.nn.functional.gelu(features).transpose(1, 2))


def _selected_capacity(reference_model: str) -> tuple[int, int]:
    efficient = _EFFICIENT_REFERENCE_CAPACITIES.get(reference_model)
    if efficient is not None:
        return efficient
    capacity = capacity_for_model(reference_model)
    if capacity is None:
        message = f"unsupported selected PAC-TF reference capacity: {reference_model}"
        raise ValueError(message)
    return capacity


def build_reference_regressor(
    reference_model: str, config: PACExperimentConfig
) -> nn.Module | None:
    if reference_model == _EFP16_REFERENCE_MODEL:
        return _EfficientHeadroomSyntheticEndpointRegressor(config, "EFP16")
    capacity = capacity_for_model(reference_model)
    variant = variant_for_model(reference_model)
    if capacity is None or variant is None:
        return None
    active = replace(config, model_dim=capacity[0], modes=capacity[1])
    return TightFrameSequenceRegressor(active, variant)


def _build_evidence_classifier(
    job: P1P2Job,
    config: PACExperimentConfig,
    class_count: int,
) -> tuple[nn.Module, _WPMatch | ConfirmatoryMatch]:
    with _MODEL_INIT_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.seed)
        if job.model in _EFFICIENT_MODEL_SPECS:
            model = build_efficient_headroom_classifier(
                _EFFICIENT_MODEL_SPECS[job.model],
                config,
                class_count,
                objective="classification",
            )
            parameters = count_parameters(model)
            return model, _WPMatch(parameters, parameters)
        if job.reference_model in _EFFICIENT_REFERENCE_SPECS:
            reference_spec = _EFFICIENT_REFERENCE_SPECS[job.reference_model]
            reference = build_efficient_headroom_classifier(
                reference_spec,
                config,
                class_count,
                objective="classification",
            )
            target = count_parameters(reference)
            family = cast("ConfirmatoryFamily", job.model)
            if _uses_native_selected_width(job):
                model = build_confirmatory_family(
                    family,
                    job.selected_model_width,
                    config,
                    class_count,
                    validation_trial=job.selection_trial,
                )
                parameters = count_parameters(model)
                return model, ConfirmatoryMatch(
                    family,
                    job.selected_model_width,
                    parameters,
                    target,
                    abs(parameters - target) / target,
                )
            search_limit = 2048 if family == "inception_time" else 256
            candidates: list[tuple[nn.Module, ConfirmatoryMatch]] = []
            for width in range(1, search_limit + 1):
                candidate = build_confirmatory_family(
                    family,
                    width,
                    config,
                    class_count,
                    validation_trial=job.selection_trial,
                )
                parameters = count_parameters(candidate)
                match = ConfirmatoryMatch(
                    family,
                    width,
                    parameters,
                    target,
                    abs(parameters - target) / target,
                )
                candidates.append((candidate, match))
                if parameters >= target:
                    break
            model, match = min(
                candidates,
                key=lambda item: (item[1].relative_error, item[1].width),
            )
            tolerance = _parameter_match_tolerance(job)
            if reference_spec == "EFP16":
                locked_tolerance = ucr_parameter_tolerance(family, class_count)
                if tolerance != locked_tolerance:
                    message = (
                        f"EFP16 manifest tolerance {tolerance:.4f} differs from "
                        f"locked real-width tolerance {locked_tolerance:.4f}"
                    )
                    raise ValueError(message)
            if match.relative_error > tolerance:
                message = (
                    f"{family} parameter error {match.relative_error:.4f} "
                    f"exceeds final-ALPHABET tolerance {tolerance:.4f}"
                )
                raise ValueError(message)
            return model, match
        return _build_configured_confirmatory_classifier(job, config, class_count)
