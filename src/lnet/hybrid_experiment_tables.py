from __future__ import annotations

from dataclasses import replace
from math import pi
from typing import TYPE_CHECKING

from lnet.advanced_experiments import (
    make_fir_baseline,
    make_gru_baseline,
    make_transformer_baseline,
    train_regression_model,
)
from lnet.hybrid import BRANCH_ORDER
from lnet.hybrid_delay_tasks import (
    DelayedExponentialTeacherConfig,
    DelayedOscillatoryTeacherConfig,
    StrictDelayTaskConfig,
    make_delayed_exponential_teacher_bundle,
    make_delayed_oscillatory_teacher_bundle,
    make_strict_delay_teacher_bundle,
)
from lnet.hybrid_experiment_types import (
    HybridExperimentConfig,
    NamedRegressionTask,
    TrainedHybridModel,
    branch_label,
    fit_hybrid,
    format_float,
)
from lnet.hybrid_metrics import (
    count_trainable_parameters,
    hybrid_gate_diagnostic,
    tapped_prl_pole_diagnostic,
    tapped_prl_tap_diagnostic,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from lnet.experiment import TrainingConfig


def _optional_metric(value: float | None) -> float:
    return value if value is not None else float("nan")


def closest_candidate(
    builder: Callable[[int], nn.Module],
    target_count: int,
    candidates: tuple[int, ...],
) -> nn.Module:
    models = tuple(builder(model_dim) for model_dim in candidates)
    return min(models, key=lambda model: abs(count_trainable_parameters(model) - target_count))


def branch_ablation_rows(
    tasks: tuple[NamedRegressionTask, ...],
    config: HybridExperimentConfig,
    training: TrainingConfig,
) -> tuple[tuple[TrainedHybridModel, ...], tuple[tuple[str, ...], ...]]:
    trained_models: list[TrainedHybridModel] = []
    rows: list[tuple[str, ...]] = []
    for task in tasks:
        for branches in config.branch_sets:
            trained = fit_hybrid(task, config, training, branches)
            trained_models.append(trained)
            rows.append(
                (
                    task.label,
                    trained.branch_label,
                    str(trained.parameter_count),
                    format_float(trained.outcome.validation_loss),
                    format_float(trained.outcome.pole_mae),
                ),
            )
    return tuple(trained_models), tuple(rows)


def full_hybrid_by_task(
    trained_models: tuple[TrainedHybridModel, ...],
    tasks: tuple[NamedRegressionTask, ...],
    config: HybridExperimentConfig,
    training: TrainingConfig,
) -> dict[str, TrainedHybridModel]:
    full_models = {
        model.task_label: model
        for model in trained_models
        if model.branch_label == branch_label(BRANCH_ORDER, config)
    }
    for task in tasks:
        if task.label not in full_models:
            full_models[task.label] = fit_hybrid(task, config, training, BRANCH_ORDER)
    return full_models


def parameter_rows(
    tasks: tuple[NamedRegressionTask, ...],
    config: HybridExperimentConfig,
    training: TrainingConfig,
    full_models: dict[str, TrainedHybridModel],
) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for task in tasks:
        rows.extend(_parameter_rows_for_task(task, config, training, full_models[task.label]))
    return tuple(rows)


def _parameter_rows_for_task(
    task: NamedRegressionTask,
    config: HybridExperimentConfig,
    training: TrainingConfig,
    hybrid: TrainedHybridModel,
) -> tuple[tuple[str, ...], ...]:
    baseline_specs = (
        ("Hybrid Modal PRL", hybrid.model, hybrid.outcome),
        ("Matched FIR", _matched_fir(config, hybrid.parameter_count), None),
        ("Matched GRU", _matched_gru(config, hybrid.parameter_count), None),
        ("Matched Transformer", _matched_transformer(config, hybrid.parameter_count), None),
    )
    rows: list[tuple[str, ...]] = []
    for name, model, cached_outcome in baseline_specs:
        outcome = cached_outcome or train_regression_model(model, task.task, training)
        rows.append(
            (
                task.label,
                name,
                str(count_trainable_parameters(model)),
                format_float(outcome.validation_loss),
            ),
        )
    return tuple(rows)


def _matched_fir(config: HybridExperimentConfig, target_count: int) -> nn.Module:
    return closest_candidate(
        lambda dim: make_fir_baseline(
            raw_input_dim=config.raw_input_dim,
            model_dim=dim,
            output_dim=config.output_dim,
            kernel_size=config.fir_kernel_size,
        ),
        target_count,
        config.baseline_model_dims,
    )


def _matched_gru(config: HybridExperimentConfig, target_count: int) -> nn.Module:
    return closest_candidate(
        lambda dim: make_gru_baseline(
            raw_input_dim=config.raw_input_dim,
            model_dim=dim,
            output_dim=config.output_dim,
        ),
        target_count,
        config.baseline_model_dims,
    )


def _matched_transformer(config: HybridExperimentConfig, target_count: int) -> nn.Module:
    return closest_candidate(
        lambda dim: make_transformer_baseline(
            raw_input_dim=config.raw_input_dim,
            model_dim=dim,
            output_dim=config.output_dim,
            attention_heads=2,
        ),
        target_count,
        config.baseline_model_dims,
    )


def gate_rows(
    tasks: tuple[NamedRegressionTask, ...],
    full_models: dict[str, TrainedHybridModel],
) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for task in tasks:
        diagnostic = hybrid_gate_diagnostic(
            full_models[task.label].model,
            task.task.validation_inputs,
        )
        rows.append(
            (
                task.label,
                format_float(diagnostic.mean_prl_weight),
                format_float(diagnostic.mean_fir_weight),
                format_float(diagnostic.mean_mlp_weight),
                format_float(diagnostic.mean_gate_entropy),
                format_float(diagnostic.prl_contribution_norm),
                format_float(diagnostic.fir_contribution_norm),
                format_float(diagnostic.mlp_contribution_norm),
            ),
        )
    return tuple(rows)


def delay_rows(
    config: HybridExperimentConfig,
    training: TrainingConfig,
) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for delay_steps in config.delay_steps:
        rows.extend(_delay_rows_for_step(config, training, delay_steps))
    return tuple(rows)


def _delay_rows_for_step(
    config: HybridExperimentConfig,
    training: TrainingConfig,
    delay_steps: int,
) -> tuple[tuple[str, ...], ...]:
    bundle = make_strict_delay_teacher_bundle(
        StrictDelayTaskConfig(
            sample_count=config.sample_count,
            validation_count=config.validation_count,
            sequence_length=config.sequence_length,
            raw_input_dim=config.raw_input_dim,
            output_dim=config.output_dim,
            seed=config.seed,
            delay_steps=delay_steps,
        ),
    )
    named_task = NamedRegressionTask(
        label=bundle.task.teacher_label,
        task=bundle.task,
        teacher_metadata=bundle.metadata,
    )
    rows: list[tuple[str, ...]] = []
    for kernel_size in config.delay_kernel_sizes:
        local_config = replace(config, fir_kernel_size=kernel_size)
        trained = fit_hybrid(named_task, local_config, training, BRANCH_ORDER)
        rows.append(
            (
                str(delay_steps),
                str(kernel_size),
                str(trained.parameter_count),
                format_float(trained.outcome.validation_loss),
                format_float(trained.outcome.pole_mae),
                bundle.metadata.metadata_status,
            ),
        )
    return tuple(rows)


def delayed_modal_rows(
    config: HybridExperimentConfig,
    training: TrainingConfig,
) -> tuple[tuple[str, ...], ...]:
    delayed_tasks = (
        make_delayed_exponential_teacher_bundle(
            DelayedExponentialTeacherConfig(
                sample_count=config.sample_count,
                validation_count=config.validation_count,
                sequence_length=config.sequence_length,
                raw_input_dim=config.raw_input_dim,
                output_dim=config.output_dim,
                seed=config.seed,
                delay_steps=4,
                discrete_pole=0.85,
            ),
        ),
        make_delayed_oscillatory_teacher_bundle(
            DelayedOscillatoryTeacherConfig(
                sample_count=config.sample_count,
                validation_count=config.validation_count,
                sequence_length=config.sequence_length,
                raw_input_dim=config.raw_input_dim,
                output_dim=config.output_dim,
                seed=config.seed,
                delay_steps=4,
                damping_radius=0.90,
                angular_frequency=pi / 4.0,
            ),
        ),
    )
    rows: list[tuple[str, ...]] = []
    for bundle in delayed_tasks:
        task = NamedRegressionTask(
            label=bundle.task.teacher_label,
            task=bundle.task,
            teacher_metadata=bundle.metadata,
        )
        trained = fit_hybrid(task, config, training, BRANCH_ORDER)
        tap_diagnostic = tapped_prl_tap_diagnostic(trained.model.temporal_mixer, bundle.metadata)
        pole_diagnostic = tapped_prl_pole_diagnostic(trained.model.temporal_mixer, bundle.metadata)
        rows.append(
            (
                task.label,
                str(bundle.metadata.true_delay),
                str(tap_diagnostic.tap_peak_index),
                format_float(_optional_metric(tap_diagnostic.tap_mass_near_true_delay)),
                (
                    str(tap_diagnostic.tap_peak_error)
                    if tap_diagnostic.tap_peak_error is not None
                    else "n/a"
                ),
                format_float(_optional_metric(pole_diagnostic.mean_pole_error)),
                format_float(_optional_metric(pole_diagnostic.max_pole_error)),
                format_float(trained.outcome.validation_loss),
            ),
        )
    return tuple(rows)
