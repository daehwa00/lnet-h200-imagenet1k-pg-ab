from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Literal

import torch
from torch import Tensor

from lnet.advanced_experiments import SequenceRegressionTask

MetadataStatus = Literal["full_ground_truth", "delay_only", "pole_only", "proxy_only"]


@dataclass(frozen=True, slots=True)
class TeacherMetadata:
    teacher_kind: str
    true_delay: int | None
    discrete_pole_real: float | None
    discrete_pole_imag: float | None
    continuous_pole_real: float | None
    continuous_pole_imag: float | None
    damping_radius: float | None
    angular_frequency: float | None
    target_horizon: int
    metadata_status: MetadataStatus


@dataclass(frozen=True, slots=True)
class TeacherTaskBundle:
    task: SequenceRegressionTask
    metadata: TeacherMetadata


@dataclass(frozen=True, slots=True)
class StrictDelayTaskConfig:
    sample_count: int
    validation_count: int
    sequence_length: int
    raw_input_dim: int
    output_dim: int
    seed: int
    delay_steps: int
    tail_length: int = 10


@dataclass(frozen=True, slots=True)
class DelayedExponentialTeacherConfig:
    sample_count: int
    validation_count: int
    sequence_length: int
    raw_input_dim: int
    output_dim: int
    seed: int
    delay_steps: int
    discrete_pole: float
    tail_length: int = 10


@dataclass(frozen=True, slots=True)
class DelayedOscillatoryTeacherConfig:
    sample_count: int
    validation_count: int
    sequence_length: int
    raw_input_dim: int
    output_dim: int
    seed: int
    delay_steps: int
    damping_radius: float
    angular_frequency: float
    tail_length: int = 10


def _random_inputs(config: StrictDelayTaskConfig) -> tuple[Tensor, Tensor]:
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


def _mixing_matrix(*, output_dim: int, raw_input_dim: int, dtype: torch.dtype) -> Tensor:
    mixing = torch.empty(output_dim, raw_input_dim, dtype=dtype)
    for output_index in range(output_dim):
        for input_index in range(raw_input_dim):
            mixing[output_index, input_index] = 0.6 + 0.1 * (output_index + input_index)
    return mixing


def _strict_delay_targets(inputs: Tensor, config: StrictDelayTaskConfig) -> Tensor:
    targets = torch.zeros(
        inputs.shape[0],
        inputs.shape[1],
        config.output_dim,
        dtype=inputs.dtype,
    )
    mixing = _mixing_matrix(
        output_dim=config.output_dim,
        raw_input_dim=config.raw_input_dim,
        dtype=inputs.dtype,
    )
    decay = torch.exp(-0.45 * torch.arange(config.tail_length, dtype=inputs.dtype))
    for tail_index, tail_weight in enumerate(decay):
        lag = config.delay_steps + tail_index
        if lag < inputs.shape[1]:
            delayed = torch.matmul(inputs[:, : inputs.shape[1] - lag, :], mixing.transpose(0, 1))
            targets[:, lag:, :] += tail_weight * delayed
    return targets


def _delayed_kernel_targets(
    *,
    inputs: Tensor,
    output_dim: int,
    raw_input_dim: int,
    delay_steps: int,
    kernel_tail: Tensor,
) -> Tensor:
    targets = torch.zeros(
        inputs.shape[0],
        inputs.shape[1],
        output_dim,
        dtype=inputs.dtype,
    )
    mixing = _mixing_matrix(output_dim=output_dim, raw_input_dim=raw_input_dim, dtype=inputs.dtype)
    for tail_index, tail_weight in enumerate(kernel_tail):
        lag = delay_steps + tail_index
        if lag < inputs.shape[1]:
            delayed = torch.matmul(inputs[:, : inputs.shape[1] - lag, :], mixing.transpose(0, 1))
            targets[:, lag:, :] += tail_weight * delayed
    return targets


def make_strict_delay_teacher_bundle(config: StrictDelayTaskConfig) -> TeacherTaskBundle:
    train_inputs, validation_inputs = _random_inputs(config)
    task = SequenceRegressionTask(
        train_inputs=train_inputs,
        train_targets=_strict_delay_targets(train_inputs, config),
        validation_inputs=validation_inputs,
        validation_targets=_strict_delay_targets(validation_inputs, config),
        teacher_label=f"strict_delay_{config.delay_steps}",
    )
    metadata = TeacherMetadata(
        teacher_kind="strict_delay",
        true_delay=config.delay_steps,
        discrete_pole_real=None,
        discrete_pole_imag=None,
        continuous_pole_real=None,
        continuous_pole_imag=None,
        damping_radius=None,
        angular_frequency=None,
        target_horizon=config.sequence_length,
        metadata_status="delay_only",
    )
    return TeacherTaskBundle(task=task, metadata=metadata)


def make_strict_delay_teacher_task(config: StrictDelayTaskConfig) -> SequenceRegressionTask:
    return make_strict_delay_teacher_bundle(config).task


def make_delayed_exponential_teacher_bundle(
    config: DelayedExponentialTeacherConfig,
) -> TeacherTaskBundle:
    _require_unit_interval(config.discrete_pole, "discrete_pole")
    random_inputs_config = StrictDelayTaskConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        seed=config.seed,
        delay_steps=config.delay_steps,
        tail_length=config.tail_length,
    )
    train_inputs, validation_inputs = _random_inputs(random_inputs_config)
    kernel_tail = config.discrete_pole ** torch.arange(config.tail_length, dtype=torch.float32)
    task = SequenceRegressionTask(
        train_inputs=train_inputs,
        train_targets=_delayed_kernel_targets(
            inputs=train_inputs,
            output_dim=config.output_dim,
            raw_input_dim=config.raw_input_dim,
            delay_steps=config.delay_steps,
            kernel_tail=kernel_tail,
        ),
        validation_inputs=validation_inputs,
        validation_targets=_delayed_kernel_targets(
            inputs=validation_inputs,
            output_dim=config.output_dim,
            raw_input_dim=config.raw_input_dim,
            delay_steps=config.delay_steps,
            kernel_tail=kernel_tail,
        ),
        teacher_label=f"delayed_exponential_{config.delay_steps}",
    )
    metadata = TeacherMetadata(
        teacher_kind="delayed_exponential",
        true_delay=config.delay_steps,
        discrete_pole_real=float(config.discrete_pole),
        discrete_pole_imag=0.0,
        continuous_pole_real=log(config.discrete_pole),
        continuous_pole_imag=0.0,
        damping_radius=float(config.discrete_pole),
        angular_frequency=0.0,
        target_horizon=config.sequence_length,
        metadata_status="full_ground_truth",
    )
    return TeacherTaskBundle(task=task, metadata=metadata)


def make_delayed_oscillatory_teacher_bundle(
    config: DelayedOscillatoryTeacherConfig,
) -> TeacherTaskBundle:
    _require_unit_interval(config.damping_radius, "damping_radius")
    random_inputs_config = StrictDelayTaskConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        seed=config.seed,
        delay_steps=config.delay_steps,
        tail_length=config.tail_length,
    )
    train_inputs, validation_inputs = _random_inputs(random_inputs_config)
    tail_indices = torch.arange(config.tail_length, dtype=torch.float32)
    kernel_tail = (config.damping_radius**tail_indices) * torch.cos(
        torch.tensor(config.angular_frequency, dtype=torch.float32) * tail_indices,
    )
    task = SequenceRegressionTask(
        train_inputs=train_inputs,
        train_targets=_delayed_kernel_targets(
            inputs=train_inputs,
            output_dim=config.output_dim,
            raw_input_dim=config.raw_input_dim,
            delay_steps=config.delay_steps,
            kernel_tail=kernel_tail,
        ),
        validation_inputs=validation_inputs,
        validation_targets=_delayed_kernel_targets(
            inputs=validation_inputs,
            output_dim=config.output_dim,
            raw_input_dim=config.raw_input_dim,
            delay_steps=config.delay_steps,
            kernel_tail=kernel_tail,
        ),
        teacher_label=f"delayed_oscillatory_{config.delay_steps}",
    )
    cosine = torch.cos(torch.tensor(config.angular_frequency)).item()
    sine = torch.sin(torch.tensor(config.angular_frequency)).item()
    discrete_pole = config.damping_radius * complex(cosine, sine)
    metadata = TeacherMetadata(
        teacher_kind="delayed_oscillatory",
        true_delay=config.delay_steps,
        discrete_pole_real=discrete_pole.real,
        discrete_pole_imag=discrete_pole.imag,
        continuous_pole_real=log(config.damping_radius),
        continuous_pole_imag=config.angular_frequency,
        damping_radius=config.damping_radius,
        angular_frequency=config.angular_frequency,
        target_horizon=config.sequence_length,
        metadata_status="full_ground_truth",
    )
    return TeacherTaskBundle(task=task, metadata=metadata)


def _require_unit_interval(value: float, name: str) -> None:
    if 0.0 < value < 1.0:
        return
    message = f"{name} must be in the stable open interval (0, 1)"
    raise ValueError(message)
