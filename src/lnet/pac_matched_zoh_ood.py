from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .pac_real2d_math import discrete_pole_real2d
from .pac_types import PACExperimentConfig, PACRegressionTask

PHYSICAL_HORIZON = 60.0
SAMPLING_DELTAS = (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0)
_FORCING_FREQUENCIES = (0.07, 0.13, 0.23)


@dataclass(frozen=True, slots=True)
class MatchedZOHCondition:
    family: str
    level: str
    id_inputs: Tensor
    id_targets: Tensor
    ood_inputs: Tensor
    ood_targets: Tensor


@dataclass(frozen=True, slots=True)
class _ForcingParameters:
    cosine: Tensor
    sine: Tensor


def matched_zoh_training_task(
    config: PACExperimentConfig,
    seed: int,
) -> PACRegressionTask:
    train_inputs, train_targets = _regular_split(config.sample_count, seed + 101)
    validation_inputs, validation_targets = _regular_split(
        config.validation_count,
        seed + 211,
    )
    test_inputs, test_targets = _regular_split(config.test_count, seed + 307)
    return PACRegressionTask(
        "matched_exact_zoh_physical_time_id",
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        test_inputs,
        test_targets,
        true_delay=0,
        true_frequency=math.pi / 4,
        true_frequencies=(math.pi / 4,),
        true_dampings=(0.8,),
        mechanism_expectation="positive",
    )


def matched_zoh_conditions(
    config: PACExperimentConfig,
    seed: int,
) -> list[MatchedZOHCondition]:
    parameters = _forcing_parameters(config.test_count, seed + 401)
    id_inputs, id_targets = _build_case(parameters)
    conditions: list[MatchedZOHCondition] = []
    for delta in SAMPLING_DELTAS:
        ood_inputs, ood_targets = _build_case(parameters, delta=delta)
        conditions.append(
            MatchedZOHCondition(
                "sampling_rate",
                f"dt_{delta:g}",
                id_inputs,
                id_targets,
                ood_inputs,
                ood_targets,
            )
        )
    for level, irregularity, missing_rate in (
        ("moderate", 0.25, 0.1),
        ("hard", 0.75, 0.3),
    ):
        ood_inputs, ood_targets = _build_case(
            parameters,
            irregularity=irregularity,
            missing_rate=missing_rate,
            perturbation_seed=seed + 701 + len(conditions),
        )
        conditions.append(
            MatchedZOHCondition(
                "irregular_timestamps_missingness",
                level,
                id_inputs,
                id_targets,
                ood_inputs,
                ood_targets,
            )
        )
    specifications: tuple[tuple[str, str, dict[str, float]], ...] = (
        ("physical_horizon", "120", {"horizon": 120.0}),
        ("physical_horizon", "240", {"horizon": 240.0}),
        ("delay", "4", {"delay_seconds": 4.0}),
        ("delay", "8", {"delay_seconds": 8.0}),
        ("additive_noise", "0.05", {"noise": 0.05}),
        ("additive_noise", "0.1", {"noise": 0.1}),
        ("damping", "1.2", {"damping": 1.2}),
        ("damping", "1.6", {"damping": 1.6}),
        ("frequency", "pi_over_8", {"frequency": math.pi / 8}),
        ("frequency", "pi_over_2", {"frequency": math.pi / 2}),
    )
    for index, (family, level, options) in enumerate(specifications):
        ood_inputs, ood_targets = _build_case(
            parameters,
            horizon=options.get("horizon", PHYSICAL_HORIZON),
            noise=options.get("noise", 0.0),
            damping=options.get("damping", 0.8),
            frequency=options.get("frequency", math.pi / 4),
            delay_seconds=options.get("delay_seconds", 0.0),
            perturbation_seed=seed + 907 + index,
        )
        conditions.append(
            MatchedZOHCondition(
                family,
                level,
                id_inputs,
                id_targets,
                ood_inputs,
                ood_targets,
            )
        )
    return conditions


def _regular_split(count: int, seed: int) -> tuple[Tensor, Tensor]:
    return _build_case(_forcing_parameters(count, seed))


def _forcing_parameters(count: int, seed: int) -> _ForcingParameters:
    generator = torch.Generator().manual_seed(seed)
    modes = len(_FORCING_FREQUENCIES)
    scale = math.sqrt(float(modes))
    return _ForcingParameters(
        torch.randn(count, 2, modes, generator=generator) / scale,
        torch.randn(count, 2, modes, generator=generator) / scale,
    )


def _build_case(
    parameters: _ForcingParameters,
    *,
    horizon: float = PHYSICAL_HORIZON,
    delta: float = 1.0,
    irregularity: float = 0.0,
    missing_rate: float = 0.0,
    noise: float = 0.0,
    damping: float = 0.8,
    frequency: float = math.pi / 4,
    delay_seconds: float = 0.0,
    perturbation_seed: int = 0,
) -> tuple[Tensor, Tensor]:
    count = parameters.cosine.shape[0]
    deltas = _physical_time_steps(
        count,
        horizon,
        delta,
        irregularity,
        perturbation_seed,
    )
    interval_starts = torch.cumsum(deltas, dim=1) - deltas
    values = _evaluate_forcing(parameters, interval_starts[..., 0])
    delayed_times = interval_starts[..., 0] - delay_seconds
    forcing = _evaluate_forcing(parameters, delayed_times.clamp_min(0.0))
    if delay_seconds > 0.0:
        forcing = torch.where(
            (delayed_times >= 0.0).unsqueeze(-1),
            forcing,
            torch.zeros_like(forcing),
        )
    generator = torch.Generator().manual_seed(perturbation_seed)
    observed = values
    if noise > 0.0:
        observed = observed + noise * torch.randn(observed.shape, generator=generator)
    mask = torch.ones(count, deltas.shape[1], 1)
    if missing_rate > 0.0:
        mask = (
            torch.rand(count, deltas.shape[1], 1, generator=generator) >= missing_rate
        ).to(values.dtype)
        mask[:, 0] = 1.0
    inputs = torch.cat((observed * mask, deltas, mask), dim=-1)
    targets = exact_zoh_targets(
        forcing,
        deltas,
        damping=damping,
        frequency=frequency,
    )
    return inputs, targets


def _physical_time_steps(
    count: int,
    horizon: float,
    delta: float,
    irregularity: float,
    seed: int,
) -> Tensor:
    length = round(horizon / delta)
    if length < 1 or not math.isclose(length * delta, horizon, abs_tol=1.0e-6):
        message = "physical horizon must be divisible by the requested sampling interval"
        raise ValueError(message)
    if irregularity == 0.0:
        return torch.full((count, length, 1), delta)
    generator = torch.Generator().manual_seed(seed)
    steps = 1.0 + irregularity * (2.0 * torch.rand(count, length, 1, generator=generator) - 1.0)
    return steps * (horizon / steps.sum(dim=1, keepdim=True))


def _evaluate_forcing(parameters: _ForcingParameters, times: Tensor) -> Tensor:
    frequencies = times.new_tensor(_FORCING_FREQUENCIES)
    phases = times.unsqueeze(-1) * frequencies
    return torch.einsum("bcm,bnm->bnc", parameters.cosine, torch.cos(phases)) + torch.einsum(
        "bcm,bnm->bnc",
        parameters.sine,
        torch.sin(phases),
    )


def exact_zoh_targets(
    forcing: Tensor,
    deltas: Tensor,
    *,
    damping: float,
    frequency: float,
) -> Tensor:
    damping_tensor = torch.full_like(deltas, damping)
    frequency_tensor = torch.full_like(deltas, frequency)
    decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
        damping_tensor,
        frequency_tensor,
        deltas,
    )
    state_real = torch.zeros(forcing.shape[0], dtype=forcing.dtype)
    state_imag = torch.zeros_like(state_real)
    outputs: list[Tensor] = []
    for index in range(forcing.shape[1]):
        active_real = decay_real[:, index, 0] * state_real - decay_imag[:, index, 0] * state_imag
        active_imag = decay_imag[:, index, 0] * state_real + decay_real[:, index, 0] * state_imag
        input_real = (
            gamma_real[:, index, 0] * forcing[:, index, 0]
            - gamma_imag[:, index, 0] * forcing[:, index, 1]
        )
        input_imag = (
            gamma_real[:, index, 0] * forcing[:, index, 1]
            + gamma_imag[:, index, 0] * forcing[:, index, 0]
        )
        state_real = active_real + input_real
        state_imag = active_imag + input_imag
        outputs.append(torch.stack((state_real, state_imag), dim=-1))
    return torch.stack(outputs, dim=1)
