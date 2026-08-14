from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from .advanced_experiments import SequenceRegressionTask
from .controlled_damping_types import ControlledDampingConfig, ControlledDampingTask
from .controlled_damping_types import ControlledDampingTaskConfig as TaskConfig
from .selective_tapped_prl_tasks import (
    delayed_exponential_task,
    fir_task,
    modal_task,
    strict_delay_task,
    switching_task,
)
from .selective_tapped_prl_types import LabeledTask, SelectiveExperimentConfig


def make_context_damped_exponential_task(config: TaskConfig) -> ControlledDampingTask:
    return _make_context_task(config, "context_damped_exponential")


def make_delayed_context_damped_exponential_task(config: TaskConfig) -> ControlledDampingTask:
    delayed = replace(config, delay_steps=max(config.delay_steps, 4))
    return _make_context_task(delayed, "delayed_context_damped_exponential")


def make_controlled_damping_tasks(
    config: ControlledDampingConfig,
    seed: int,
) -> tuple[ControlledDampingTask, ...]:
    task_config = TaskConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        seed=seed,
    )
    selective_config = SelectiveExperimentConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        model_dim=config.model_dim,
        modes=config.modes,
        tap_kernel_size=config.tap_kernel_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=seed,
        device=config.device,
        tap_entropy_weight=config.tap_entropy_weight,
        gate_entropy_weight=config.gate_entropy_weight,
    )
    control_tasks = (
        _wrap(modal_task(selective_config)),
        _wrap(fir_task(selective_config)),
        _wrap(strict_delay_task(selective_config, 6)),
        _wrap(delayed_exponential_task(selective_config, 4)),
    )
    return (
        make_context_damped_exponential_task(task_config),
        make_delayed_context_damped_exponential_task(task_config),
        _wrap(switching_task(selective_config)),
        *control_tasks,
    )


def _make_context_task(config: TaskConfig, label: str) -> ControlledDampingTask:
    train_inputs, validation_inputs = _random_inputs(config)
    train_targets, train_regime = _targets_and_regime(train_inputs, config)
    validation_targets, validation_regime = _targets_and_regime(validation_inputs, config)
    task = SequenceRegressionTask(
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        teacher_label=label,
    )
    return ControlledDampingTask(label, task, train_regime, validation_regime)


def _random_inputs(config: TaskConfig) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    train_inputs = torch.randn(
        config.sample_count,
        config.sequence_length,
        config.raw_input_dim,
        generator=generator,
        dtype=torch.float32,
    )
    validation_inputs = torch.randn(
        config.validation_count,
        config.sequence_length,
        config.raw_input_dim,
        generator=generator,
        dtype=torch.float32,
    )
    return train_inputs, validation_inputs


def _targets_and_regime(inputs: Tensor, config: TaskConfig) -> tuple[Tensor, Tensor]:
    fast_regime = inputs[..., 0] > 0.0
    decay = torch.where(
        fast_regime,
        torch.full_like(inputs[..., 0], config.fast_decay),
        torch.full_like(inputs[..., 0], config.slow_decay),
    )
    targets = torch.zeros(
        inputs.shape[0],
        inputs.shape[1],
        config.output_dim,
        dtype=inputs.dtype,
    )
    mixing = _mixing_matrix(config.output_dim, config.raw_input_dim, inputs.dtype)
    state = torch.zeros(inputs.shape[0], config.output_dim, dtype=inputs.dtype)
    for time_index in range(inputs.shape[1]):
        source_index = time_index - config.delay_steps
        drive = torch.zeros_like(state)
        if source_index >= 0:
            drive = torch.matmul(inputs[:, source_index, :], mixing.transpose(0, 1))
        state = (decay[:, time_index].unsqueeze(-1) * state) + drive
        targets[:, time_index, :] = state
    return targets, fast_regime


def _mixing_matrix(output_dim: int, raw_input_dim: int, dtype: torch.dtype) -> Tensor:
    output_index = torch.arange(output_dim, dtype=dtype).view(-1, 1)
    input_index = torch.arange(raw_input_dim, dtype=dtype).view(1, -1)
    return 0.6 + (0.1 * (output_index + input_index))


def _wrap(task: LabeledTask) -> ControlledDampingTask:
    return ControlledDampingTask(task.label, task.task)
