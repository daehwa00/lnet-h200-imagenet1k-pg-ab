from __future__ import annotations

from dataclasses import replace
from math import pi
from typing import Literal

import torch
from torch import Tensor

from .pac_types import PACExperimentConfig, PACRegressionTask


def make_pac_synthetic_tasks(
    config: PACExperimentConfig, seed: int
) -> tuple[PACRegressionTask, ...]:
    base = _random_inputs(config, seed, config.sequence_length)
    return (
        _modal_task(config, base, "modal_teacher", delay=0, alpha=0.2, omega=pi / 8),
        _modal_task(config, base, "delayed_exponential", delay=4, alpha=0.15, omega=0.0),
        _modal_task(config, base, "delayed_oscillatory", delay=4, alpha=0.15, omega=pi / 4),
        _multi_mode_task(config, base),
        _fir_task(config, base),
        _active_damping_task(config, base, delay=4, omega=pi / 4),
    )


def make_pac_tf_mechanism_tasks(
    config: PACExperimentConfig, seed: int
) -> tuple[PACRegressionTask, ...]:
    """Seven causal tasks used by the PAC-TF mechanism-recovery queue."""
    base = _random_inputs(config, seed, config.sequence_length)
    oscillator = _modal_task(
        config, base, "damped_oscillator", delay=0, alpha=0.2, omega=pi / 8
    )
    delayed = _modal_task(
        config, base, "delayed_oscillation", delay=4, alpha=0.15, omega=pi / 4
    )
    multimode = _multi_mode_task(config, base)
    fir = replace(_fir_task(config, base), label="pure_fir_negative_control")
    context = replace(
        _active_damping_task(config, base, delay=0, omega=pi / 4),
        label="context_dependent_damping",
    )
    regime = _damping_regime_task(config, base)
    local = _random_local_pattern_task(config, base)
    tasks = (oscillator, regime, delayed, multimode, fir, context, local)
    return tuple(_attach_diagnostic(config, task) for task in tasks)


def make_ood_task(
    config: PACExperimentConfig,
    seed: int,
    *,
    sequence_length: int,
    delay: int = 4,
    noise: float = 0.0,
    fast_decay: float = 0.8,
    omega: float = pi / 4,
    label: str | None = None,
) -> PACRegressionTask:
    ood_config = replace(config, sequence_length=sequence_length)
    base = _random_inputs(ood_config, seed + sequence_length + delay, sequence_length)
    task = _active_damping_task(ood_config, base, delay=delay, omega=omega, fast_decay=fast_decay)
    task = replace(task, label=label or task.label)
    if noise == 0.0:
        return task
    return PACRegressionTask(
        label=f"{label or task.label}_noise_{noise:g}",
        train_inputs=task.train_inputs,
        train_targets=task.train_targets + noise * torch.randn_like(task.train_targets),
        validation_inputs=task.validation_inputs,
        validation_targets=task.validation_targets,
        test_inputs=task.test_inputs,
        test_targets=task.test_targets,
        true_delay=task.true_delay,
        true_frequency=task.true_frequency,
        train_teacher_damping=task.train_teacher_damping,
        validation_teacher_damping=task.validation_teacher_damping,
        test_teacher_damping=task.test_teacher_damping,
        validation_regime=task.validation_regime,
    )


def _random_inputs(
    config: PACExperimentConfig, seed: int, length: int
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (config.sample_count, length, config.raw_input_dim)
    validation_shape = (config.validation_count, length, config.raw_input_dim)
    test_shape = (config.test_count, length, config.raw_input_dim)
    return (
        torch.randn(shape, generator=generator),
        torch.randn(validation_shape, generator=generator),
        torch.randn(test_shape, generator=generator),
    )


def _modal_task(
    config: PACExperimentConfig,
    inputs: tuple[Tensor, Tensor, Tensor],
    label: str,
    *,
    delay: int,
    alpha: float,
    omega: float,
) -> PACRegressionTask:
    targets = (
        _modal_targets(inputs[0], config.output_dim, delay, alpha, omega),
        _modal_targets(inputs[1], config.output_dim, delay, alpha, omega),
        _modal_targets(inputs[2], config.output_dim, delay, alpha, omega),
    )
    return replace(
        _task(label, inputs, targets, true_delay=delay, true_frequency=omega),
        true_frequencies=(omega,),
        true_dampings=(alpha,),
        mechanism_expectation="positive",
    )


def _multi_mode_task(
    config: PACExperimentConfig,
    inputs: tuple[Tensor, Tensor, Tensor],
) -> PACRegressionTask:
    modes = ((2, 0.08, pi / 8), (5, 0.2, pi / 4), (11, 0.4, pi / 2))
    targets = []
    for split in inputs:
        split_target = torch.zeros(split.shape[0], split.shape[1], config.output_dim)
        for delay, alpha, omega in modes:
            split_target = split_target + _modal_targets(
                split, config.output_dim, delay, alpha, omega
            )
        targets.append(split_target / float(len(modes)))
    return _task(
        "multi_mode_delayed_resonance",
        inputs,
        tuple(targets),
        true_delay=5,
        true_frequency=pi / 4,
        true_frequencies=tuple(mode[2] for mode in modes),
        true_dampings=tuple(mode[1] for mode in modes),
        mechanism_expectation="positive",
    )


def _fir_task(
    config: PACExperimentConfig,
    inputs: tuple[Tensor, Tensor, Tensor],
) -> PACRegressionTask:
    kernel = _fir_kernel(config)
    targets = (
        _fir_targets(inputs[0], kernel),
        _fir_targets(inputs[1], kernel),
        _fir_targets(inputs[2], kernel),
    )
    return replace(
        _task("random_fir_teacher", inputs, targets),
        mechanism_expectation="negative",
    )


def _damping_regime_task(
    config: PACExperimentConfig,
    inputs: tuple[Tensor, Tensor, Tensor],
) -> PACRegressionTask:
    controlled_inputs: list[Tensor] = []
    targets: list[Tensor] = []
    dampings: list[Tensor] = []
    regimes: list[Tensor] = []
    for split in inputs:
        controlled = split.clone()
        channel = 1 if controlled.shape[-1] > 1 else 0
        regime = controlled[:, 0, channel] > 0.0
        controlled[:, :, channel] = torch.where(regime, 1.0, -1.0).unsqueeze(-1)
        target, alpha = _sequence_regime_targets(controlled, config.output_dim, regime)
        controlled_inputs.append(controlled)
        targets.append(target)
        dampings.append(alpha)
        regimes.append(regime.unsqueeze(-1).expand_as(alpha))
    return PACRegressionTask(
        label="known_damping_regime",
        train_inputs=controlled_inputs[0],
        train_targets=targets[0],
        validation_inputs=controlled_inputs[1],
        validation_targets=targets[1],
        test_inputs=controlled_inputs[2],
        test_targets=targets[2],
        true_frequency=pi / 6,
        true_frequencies=(pi / 6,),
        true_dampings=(0.05, 0.8),
        train_teacher_damping=dampings[0],
        validation_teacher_damping=dampings[1],
        test_teacher_damping=dampings[2],
        validation_regime=regimes[1],
        mechanism_expectation="positive",
    )


def _random_local_pattern_task(
    config: PACExperimentConfig,
    inputs: tuple[Tensor, Tensor, Tensor],
) -> PACRegressionTask:
    targets = (
        _local_pattern_targets(inputs[0], config.output_dim),
        _local_pattern_targets(inputs[1], config.output_dim),
        _local_pattern_targets(inputs[2], config.output_dim),
    )
    return replace(
        _task("random_local_pattern", inputs, targets),
        mechanism_expectation="negative",
    )


def _attach_diagnostic(config: PACExperimentConfig, task: PACRegressionTask) -> PACRegressionTask:
    samples = 2 if task.label == "known_damping_regime" else 1
    inputs = torch.zeros(samples, config.sequence_length, config.raw_input_dim)
    inputs[:, 0, 0] = 1.0
    if task.label == "known_damping_regime":
        channel = 1 if config.raw_input_dim > 1 else 0
        inputs[0, :, channel] = -1.0
        inputs[1, :, channel] = 1.0
        targets, _ = _sequence_regime_targets(
            inputs, config.output_dim, torch.tensor((False, True))
        )
    elif task.label == "context_dependent_damping":
        if config.raw_input_dim > 1:
            inputs[:, : config.sequence_length // 2, 1] = -1.0
            inputs[:, config.sequence_length // 2 :, 1] = 1.0
        targets, _, _ = _active_targets(inputs, config.output_dim, 0, pi / 4, 0.8)
    elif task.label == "multi_mode_delayed_resonance":
        targets = sum(
            (
                _modal_targets(inputs, config.output_dim, delay, alpha, omega)
                for delay, alpha, omega in (
                    (2, 0.08, pi / 8),
                    (5, 0.2, pi / 4),
                    (11, 0.4, pi / 2),
                )
            ),
            torch.zeros(samples, config.sequence_length, config.output_dim),
        ) / 3.0
    elif task.label == "pure_fir_negative_control":
        targets = _fir_targets(inputs, _fir_kernel(config))
    elif task.label == "random_local_pattern":
        targets = _local_pattern_targets(inputs, config.output_dim)
    else:
        targets = _modal_targets(
            inputs,
            config.output_dim,
            task.true_delay or 0,
            task.true_dampings[0],
            task.true_frequencies[0],
        )
    return replace(task, diagnostic_inputs=inputs, diagnostic_targets=targets)


def _fir_kernel(config: PACExperimentConfig) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(991)
    return torch.randn(
        9, config.output_dim, config.raw_input_dim, generator=generator
    ) * 0.2


def _sequence_regime_targets(
    inputs: Tensor, output_dim: int, regime: Tensor
) -> tuple[Tensor, Tensor]:
    alpha_by_sample = torch.where(regime, torch.full_like(regime, 0.8, dtype=inputs.dtype), 0.05)
    alpha = alpha_by_sample.unsqueeze(-1).expand(-1, inputs.shape[1])
    mixing = _mixing(output_dim, inputs.shape[-1], inputs.dtype)
    state = torch.zeros(inputs.shape[0], output_dim, dtype=torch.complex64)
    targets = torch.zeros(inputs.shape[0], inputs.shape[1], output_dim)
    for time_index in range(inputs.shape[1]):
        pole = torch.exp(
            torch.complex(-alpha[:, time_index], torch.full_like(alpha[:, time_index], pi / 6))
        )
        state = pole.unsqueeze(-1) * state + (inputs[:, time_index] @ mixing.T).to(
            dtype=torch.complex64
        )
        targets[:, time_index] = state.real
    return targets, alpha


def _local_pattern_targets(inputs: Tensor, output_dim: int) -> Tensor:
    previous = torch.roll(inputs[..., 0], 1, dims=1)
    previous[:, 0] = 0.0
    channel = inputs[..., 1] if inputs.shape[-1] > 1 else inputs[..., 0]
    lag_two = torch.roll(channel, 2, dims=1)
    lag_two[:, :2] = 0.0
    signal = torch.tanh(inputs[..., 0] * previous + 0.5 * lag_two)
    scales = torch.linspace(0.75, 1.25, output_dim, dtype=inputs.dtype)
    return signal.unsqueeze(-1) * scales


def _active_damping_task(
    config: PACExperimentConfig,
    inputs: tuple[Tensor, Tensor, Tensor],
    *,
    delay: int,
    omega: float,
    fast_decay: float = 0.8,
) -> PACRegressionTask:
    train_targets, train_alpha, _ = _active_targets(
        inputs[0], config.output_dim, delay, omega, fast_decay
    )
    val_targets, val_alpha, val_regime = _active_targets(
        inputs[1], config.output_dim, delay, omega, fast_decay
    )
    test_targets, test_alpha, _ = _active_targets(
        inputs[2], config.output_dim, delay, omega, fast_decay
    )
    return PACRegressionTask(
        "active_damping_teacher",
        train_inputs=inputs[0],
        train_targets=train_targets,
        validation_inputs=inputs[1],
        validation_targets=val_targets,
        test_inputs=inputs[2],
        test_targets=test_targets,
        true_delay=delay,
        true_frequency=omega,
        train_teacher_damping=train_alpha,
        validation_teacher_damping=val_alpha,
        test_teacher_damping=test_alpha,
        validation_regime=val_regime,
        true_frequencies=(omega,),
        true_dampings=(0.05, fast_decay),
        mechanism_expectation="positive",
    )


def _modal_targets(
    inputs: Tensor, output_dim: int, delay: int, alpha: float, omega: float
) -> Tensor:
    mixing = _mixing(output_dim, inputs.shape[-1], inputs.dtype)
    pole = torch.exp(torch.tensor(complex(-alpha, omega), dtype=torch.complex64))
    state = torch.zeros(inputs.shape[0], output_dim, dtype=torch.complex64)
    targets = torch.zeros(inputs.shape[0], inputs.shape[1], output_dim)
    for time_index in range(inputs.shape[1]):
        drive = torch.zeros(inputs.shape[0], output_dim)
        if time_index >= delay:
            drive = inputs[:, time_index - delay, :] @ mixing.T
        state = pole * state + drive.to(dtype=torch.complex64)
        targets[:, time_index, :] = state.real
    return targets


def _task(
    label: str,
    inputs: tuple[Tensor, Tensor, Tensor],
    targets: tuple[Tensor, Tensor, Tensor],
    *,
    true_delay: int | None = None,
    true_frequency: float | None = None,
    true_frequencies: tuple[float, ...] = (),
    true_dampings: tuple[float, ...] = (),
    mechanism_expectation: Literal["positive", "negative", "neutral"] = "neutral",
) -> PACRegressionTask:
    return PACRegressionTask(
        label=label,
        train_inputs=inputs[0],
        train_targets=targets[0],
        validation_inputs=inputs[1],
        validation_targets=targets[1],
        test_inputs=inputs[2],
        test_targets=targets[2],
        true_delay=true_delay,
        true_frequency=true_frequency,
        true_frequencies=true_frequencies,
        true_dampings=true_dampings,
        mechanism_expectation=mechanism_expectation,
    )


def _active_targets(
    inputs: Tensor,
    output_dim: int,
    delay: int,
    omega: float,
    fast_decay: float,
) -> tuple[Tensor, Tensor, Tensor]:
    regime = inputs[..., 1] > 0.0 if inputs.shape[-1] > 1 else inputs[..., 0] > 0.0
    alpha = torch.where(
        regime, torch.full_like(inputs[..., 0], fast_decay), torch.full_like(inputs[..., 0], 0.05)
    )
    mixing = _mixing(output_dim, inputs.shape[-1], inputs.dtype)
    state = torch.zeros(inputs.shape[0], output_dim, dtype=torch.complex64)
    targets = torch.zeros(inputs.shape[0], inputs.shape[1], output_dim)
    for time_index in range(inputs.shape[1]):
        pole = torch.exp(
            torch.complex(-alpha[:, time_index], torch.full_like(alpha[:, time_index], omega))
        )
        drive = torch.zeros(inputs.shape[0], output_dim)
        if time_index >= delay:
            drive = inputs[:, time_index - delay, :] @ mixing.T
        state = pole.unsqueeze(-1) * state + drive.to(dtype=torch.complex64)
        targets[:, time_index, :] = state.real
    return targets, alpha, regime


def _fir_targets(inputs: Tensor, kernel: Tensor) -> Tensor:
    targets = torch.zeros(inputs.shape[0], inputs.shape[1], kernel.shape[1])
    for time_index in range(inputs.shape[1]):
        for tap_index in range(min(time_index + 1, kernel.shape[0])):
            targets[:, time_index, :] += inputs[:, time_index - tap_index, :] @ kernel[tap_index].T
    return targets


def _mixing(output_dim: int, input_dim: int, dtype: torch.dtype) -> Tensor:
    output_index = torch.arange(output_dim, dtype=dtype).view(-1, 1)
    input_index = torch.arange(input_dim, dtype=dtype).view(1, -1)
    return 0.5 + 0.1 * torch.cos(output_index + input_index)
