from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch
from torch import Tensor

from .pac_tasks import make_pac_synthetic_tasks
from .pac_types import PACExperimentConfig, PACRegressionTask


@dataclass(frozen=True, slots=True)
class TimedRegressionTask:
    task: PACRegressionTask
    train_delta: Tensor
    validation_delta: Tensor
    test_delta: Tensor


def append_time_delta(inputs: Tensor, delta: Tensor) -> Tensor:
    return torch.cat((inputs, delta.unsqueeze(-1).to(dtype=inputs.dtype)), dim=-1)


def baseline_timed_task(timed: TimedRegressionTask) -> PACRegressionTask:
    task = timed.task
    return PACRegressionTask(
        label=task.label,
        train_inputs=append_time_delta(task.train_inputs, timed.train_delta),
        train_targets=task.train_targets,
        validation_inputs=append_time_delta(task.validation_inputs, timed.validation_delta),
        validation_targets=task.validation_targets,
        test_inputs=append_time_delta(task.test_inputs, timed.test_delta),
        test_targets=task.test_targets,
        true_delay=task.true_delay,
        true_frequency=task.true_frequency,
        train_teacher_damping=task.train_teacher_damping,
        validation_teacher_damping=task.validation_teacher_damping,
        test_teacher_damping=task.test_teacher_damping,
        validation_regime=task.validation_regime,
    )


def make_sampling_rate_task(
    config: PACExperimentConfig,
    seed: int,
    *,
    test_delta: float,
    irregular: bool,
) -> TimedRegressionTask:
    base = next(
        task for task in make_pac_synthetic_tasks(config, seed) if task.label == "modal_teacher"
    )
    train_delta = torch.ones(base.train_inputs.shape[:2])
    validation_delta = torch.ones(base.validation_inputs.shape[:2])
    generated_test_delta = _test_delta(
        base.test_inputs,
        seed,
        test_delta,
        irregular=irregular,
    )
    targets = _continuous_modal_targets(
        base.test_inputs,
        generated_test_delta,
        config.output_dim,
        alpha=0.2,
        omega=pi / 8,
    )
    label = "irregular_time_ood" if irregular else f"sampling_rate_ood_dt_{test_delta:g}"
    task = PACRegressionTask(
        label=label,
        train_inputs=base.train_inputs,
        train_targets=base.train_targets,
        validation_inputs=base.validation_inputs,
        validation_targets=base.validation_targets,
        test_inputs=base.test_inputs,
        test_targets=targets,
        true_delay=0,
        true_frequency=pi / 8,
    )
    return TimedRegressionTask(task, train_delta, validation_delta, generated_test_delta)


def make_low_data_task(
    config: PACExperimentConfig, seed: int, ratio: float, label: str
) -> PACRegressionTask:
    task = next(task for task in make_pac_synthetic_tasks(config, seed) if task.label == label)
    count = max(1, int(task.train_inputs.shape[0] * ratio))
    return PACRegressionTask(
        label=f"{label}_low_data_{ratio:g}",
        train_inputs=task.train_inputs[:count],
        train_targets=task.train_targets[:count],
        validation_inputs=task.validation_inputs,
        validation_targets=task.validation_targets,
        test_inputs=task.test_inputs,
        test_targets=task.test_targets,
        true_delay=task.true_delay,
        true_frequency=task.true_frequency,
        train_teacher_damping=task.train_teacher_damping[:count]
        if task.train_teacher_damping is not None
        else None,
        validation_teacher_damping=task.validation_teacher_damping,
        test_teacher_damping=task.test_teacher_damping,
        validation_regime=task.validation_regime,
    )


def _test_delta(inputs: Tensor, seed: int, value: float, *, irregular: bool) -> Tensor:
    if not irregular:
        return torch.full(inputs.shape[:2], value, dtype=inputs.dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed + 909)
    return 0.5 + torch.rand(inputs.shape[:2], generator=generator, dtype=inputs.dtype)


def _continuous_modal_targets(
    inputs: Tensor,
    delta: Tensor,
    output_dim: int,
    *,
    alpha: float,
    omega: float,
) -> Tensor:
    mixing = _mixing(output_dim, inputs.shape[-1], inputs.dtype)
    pole = torch.tensor(complex(-alpha, omega), dtype=torch.complex64)
    gamma_ref = torch.expm1(pole) / pole
    state = torch.zeros(inputs.shape[0], output_dim, dtype=torch.complex64)
    targets = torch.zeros(inputs.shape[0], inputs.shape[1], output_dim)
    for time_index in range(inputs.shape[1]):
        step = delta[:, time_index].to(dtype=torch.float32)
        decay = torch.exp(pole * step).unsqueeze(-1)
        gamma = (torch.expm1(pole * step) / pole).unsqueeze(-1)
        drive = inputs[:, time_index, :] @ mixing.T
        state = decay * state + (gamma / gamma_ref) * drive.to(dtype=torch.complex64)
        targets[:, time_index, :] = state.real
    return targets


def _mixing(output_dim: int, input_dim: int, dtype: torch.dtype) -> Tensor:
    output_index = torch.arange(output_dim, dtype=dtype).view(-1, 1)
    input_index = torch.arange(input_dim, dtype=dtype).view(1, -1)
    return 0.5 + 0.1 * torch.cos(output_index + input_index)
