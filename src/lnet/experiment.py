from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.laplace import ProjectedPRLBlock
from lnet.models import LinearRecurrentBaseline, PerStepMLPBaseline


@dataclass(frozen=True, slots=True)
class SyntheticTaskConfig:
    sample_count: int = 128
    validation_count: int = 32
    sequence_length: int = 32
    raw_input_dim: int = 2
    model_dim: int = 6
    output_dim: int = 2
    modes: int = 3
    dt: float = 1.0
    noise_scale: float = 0.0
    seed: int = 7


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 250
    learning_rate: float = 5.0e-2
    weight_decay: float = 1.0e-4
    device: str = "cpu"
    seed: int = 7
    student_model_dim: int | None = None
    student_modes: int | None = None


@dataclass(frozen=True, slots=True)
class SyntheticLaplaceTask:
    train_inputs: Tensor
    train_targets: Tensor
    validation_inputs: Tensor
    validation_targets: Tensor
    teacher: ProjectedPRLBlock
    teacher_poles: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    initial_loss: float
    final_loss: float
    validation_loss: float
    pole_mae: float
    true_poles: tuple[float, ...]
    learned_poles: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrainedLaplaceModel:
    model: torch.nn.Module
    summary: TrainingSummary


def _sorted_float_tuple(values: Tensor) -> tuple[float, ...]:
    sorted_values = torch.sort(values.detach().cpu()).values
    return tuple(float(value) for value in sorted_values.tolist())


def _pole_distance(
    teacher_poles: tuple[float, ...],
    learned_poles: tuple[float, ...],
) -> float:
    aligned_count = min(len(teacher_poles), len(learned_poles))
    aligned_error = sum(
        abs(left - right)
        for left, right in zip(
            teacher_poles[:aligned_count],
            learned_poles[:aligned_count],
            strict=True,
        )
    )
    unmatched_error = sum(abs(value) for value in teacher_poles[aligned_count:]) + sum(
        abs(value) for value in learned_poles[aligned_count:]
    )
    return (aligned_error + unmatched_error) / max(len(teacher_poles), len(learned_poles))


def _configure_teacher(block: ProjectedPRLBlock) -> None:
    input_grid = torch.arange(
        block.model_dim * block.raw_input_dim,
        dtype=torch.float32,
    ).reshape(block.model_dim, block.raw_input_dim)
    output_grid = torch.arange(
        block.output_dim * block.model_dim,
        dtype=torch.float32,
    ).reshape(block.output_dim, block.model_dim)
    with torch.no_grad():
        block.input_projection.weight.copy_(
            0.45 * torch.sin((input_grid + 1.0) / 3.0),
        )
        block.input_projection.bias.zero_()

        poles = -torch.linspace(0.2, 1.1, block.modes, dtype=torch.float64)
        residue_grid = torch.arange(
            block.model_dim * block.modes * block.model_dim,
            dtype=torch.float64,
        ).reshape(block.model_dim, block.modes, block.model_dim)
        residues = 0.08 * torch.cos(residue_grid + 0.5)
        direct_term = 0.03 * torch.eye(block.model_dim, dtype=torch.float64)
        bias = torch.zeros(block.model_dim, dtype=torch.float64)
        block.temporal_mixer.load_transfer_parameters(
            poles=poles,
            residues=residues,
            direct_term=direct_term,
            bias=bias,
        )

        block.output_projection.weight.copy_(
            0.35 * torch.cos((output_grid + 0.5) / 2.5),
        )
        block.output_projection.bias.zero_()


def make_synthetic_task(config: SyntheticTaskConfig) -> SyntheticLaplaceTask:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    teacher = ProjectedPRLBlock(
        raw_input_dim=config.raw_input_dim,
        model_dim=config.model_dim,
        output_dim=config.output_dim,
        modes=config.modes,
        dt=config.dt,
    ).eval()
    _configure_teacher(teacher)

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

    with torch.no_grad():
        train_targets = teacher(train_inputs).to(dtype=torch.float32)
        validation_targets = teacher(validation_inputs).to(dtype=torch.float32)

    if config.noise_scale > 0.0:
        train_targets = train_targets + (config.noise_scale * torch.randn_like(train_targets))
        validation_targets = validation_targets + (
            config.noise_scale * torch.randn_like(validation_targets)
        )

    return SyntheticLaplaceTask(
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        teacher=teacher,
        teacher_poles=_sorted_float_tuple(teacher.temporal_mixer.continuous_poles()),
    )


def _evaluate_loss(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    with torch.no_grad():
        predictions = model(inputs)
        return float(functional.mse_loss(predictions, targets).item())


def train_sequence_regressor(
    model: torch.nn.Module,
    task: SyntheticLaplaceTask,
    config: TrainingConfig,
) -> TrainedLaplaceModel:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_inputs = task.train_inputs.to(device=config.device)
    train_targets = task.train_targets.to(device=config.device)
    validation_inputs = task.validation_inputs.to(device=config.device)
    validation_targets = task.validation_targets.to(device=config.device)

    initial_loss = _evaluate_loss(model, train_inputs, train_targets)
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_inputs)
        loss = functional.mse_loss(predictions, train_targets)
        loss.backward()
        optimizer.step()

    final_loss = _evaluate_loss(model, train_inputs, train_targets)
    validation_loss = _evaluate_loss(model, validation_inputs, validation_targets)
    learned_poles = (
        _sorted_float_tuple(model.temporal_mixer.continuous_poles())
        if isinstance(model, ProjectedPRLBlock)
        else ()
    )
    pole_mae = _pole_distance(task.teacher_poles, learned_poles) if learned_poles else float("nan")
    summary = TrainingSummary(
        initial_loss=initial_loss,
        final_loss=final_loss,
        validation_loss=validation_loss,
        pole_mae=pole_mae,
        true_poles=task.teacher_poles,
        learned_poles=learned_poles,
    )
    return TrainedLaplaceModel(model=model, summary=summary)


def train_synthetic_model(
    task: SyntheticLaplaceTask,
    config: TrainingConfig,
) -> TrainedLaplaceModel:
    torch.manual_seed(config.seed)
    model = ProjectedPRLBlock(
        raw_input_dim=task.train_inputs.shape[-1],
        model_dim=config.student_model_dim or task.teacher.model_dim,
        output_dim=task.train_targets.shape[-1],
        modes=config.student_modes or task.teacher.modes,
        dt=task.teacher.temporal_mixer.dt,
    ).to(device=config.device)
    return train_sequence_regressor(model, task, config)


def train_mlp_baseline(
    task: SyntheticLaplaceTask,
    config: TrainingConfig,
) -> TrainedLaplaceModel:
    torch.manual_seed(config.seed)
    model = PerStepMLPBaseline(
        raw_input_dim=task.train_inputs.shape[-1],
        model_dim=config.student_model_dim or task.teacher.model_dim,
        output_dim=task.train_targets.shape[-1],
    ).to(device=config.device)
    return train_sequence_regressor(model, task, config)


def train_linear_recurrent_baseline(
    task: SyntheticLaplaceTask,
    config: TrainingConfig,
) -> TrainedLaplaceModel:
    torch.manual_seed(config.seed)
    model = LinearRecurrentBaseline(
        raw_input_dim=task.train_inputs.shape[-1],
        model_dim=config.student_model_dim or task.teacher.model_dim,
        state_dim=config.student_modes or task.teacher.modes,
        output_dim=task.train_targets.shape[-1],
    ).to(device=config.device)
    return train_sequence_regressor(model, task, config)
