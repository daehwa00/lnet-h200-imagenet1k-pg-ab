from __future__ import annotations

from dataclasses import dataclass
from math import nan
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.gated_prl import GatedPRLBlock
from lnet.hybrid import BRANCH_ORDER, BranchName, FusionVariant, HybridModalPRLBlock
from lnet.laplace import ProjectedPRLBlock
from lnet.models import (
    FIRSequenceBaseline,
    GRUSequenceBaseline,
    TransformerSequenceBaseline,
)

if TYPE_CHECKING:
    from lnet.experiment import SyntheticLaplaceTask, TrainingConfig
    from lnet.gated_prl import GateVariant
    from lnet.tapped_prl import TapParameterization


@dataclass(frozen=True, slots=True)
class SequenceRegressionTask:
    train_inputs: Tensor
    train_targets: Tensor
    validation_inputs: Tensor
    validation_targets: Tensor
    teacher_label: str


@dataclass(frozen=True, slots=True)
class RegressionOutcome:
    initial_loss: float
    final_loss: float
    validation_loss: float
    pole_mae: float


def _random_inputs(
    *,
    sample_count: int,
    validation_count: int,
    sequence_length: int,
    raw_input_dim: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_inputs = torch.randn(
        sample_count,
        sequence_length,
        raw_input_dim,
        generator=generator,
        dtype=torch.float32,
    )
    validation_inputs = torch.randn(
        validation_count,
        sequence_length,
        raw_input_dim,
        generator=generator,
        dtype=torch.float32,
    )
    return train_inputs, validation_inputs


def _causal_convolution(inputs: Tensor, kernel: Tensor) -> Tensor:
    padded = torch.nn.functional.pad(inputs.transpose(1, 2), (kernel.shape[-1] - 1, 0))
    convolved = torch.nn.functional.conv1d(padded, kernel)
    return convolved.transpose(1, 2)


def make_fir_teacher_task(
    *,
    sample_count: int,
    validation_count: int,
    sequence_length: int,
    raw_input_dim: int,
    output_dim: int,
    seed: int,
) -> SequenceRegressionTask:
    train_inputs, validation_inputs = _random_inputs(
        sample_count=sample_count,
        validation_count=validation_count,
        sequence_length=sequence_length,
        raw_input_dim=raw_input_dim,
        seed=seed,
    )
    kernel_generator = torch.Generator().manual_seed(seed)
    kernel = 0.2 * torch.randn(
        output_dim,
        raw_input_dim,
        9,
        generator=kernel_generator,
    )
    return SequenceRegressionTask(
        train_inputs=train_inputs,
        train_targets=_causal_convolution(train_inputs, kernel),
        validation_inputs=validation_inputs,
        validation_targets=_causal_convolution(validation_inputs, kernel),
        teacher_label="random_fir_teacher",
    )


def make_delay_teacher_task(
    *,
    sample_count: int,
    validation_count: int,
    sequence_length: int,
    raw_input_dim: int,
    output_dim: int,
    seed: int,
    delay_steps: int,
) -> SequenceRegressionTask:
    train_inputs, validation_inputs = _random_inputs(
        sample_count=sample_count,
        validation_count=validation_count,
        sequence_length=sequence_length,
        raw_input_dim=raw_input_dim,
        seed=seed,
    )
    decay = torch.exp(-0.45 * torch.arange(0, 10, dtype=torch.float32))
    kernel = torch.zeros(
        output_dim,
        raw_input_dim,
        delay_steps + decay.numel(),
        dtype=torch.float32,
    )
    for output_index in range(output_dim):
        for input_index in range(raw_input_dim):
            kernel[output_index, input_index, delay_steps:] = decay * (
                0.6 + 0.1 * (output_index + input_index)
            )
    return SequenceRegressionTask(
        train_inputs=train_inputs,
        train_targets=_causal_convolution(train_inputs, kernel),
        validation_inputs=validation_inputs,
        validation_targets=_causal_convolution(validation_inputs, kernel),
        teacher_label=f"delayed_exponential_{delay_steps}",
    )


def make_switching_teacher_task(
    *,
    sample_count: int,
    validation_count: int,
    sequence_length: int,
    raw_input_dim: int,
    output_dim: int,
    seed: int,
) -> SequenceRegressionTask:
    train_inputs, validation_inputs = _random_inputs(
        sample_count=sample_count,
        validation_count=validation_count,
        sequence_length=sequence_length,
        raw_input_dim=raw_input_dim,
        seed=seed,
    )
    kernel_a = torch.tensor(
        [[[0.4, 0.2, 0.1], [0.1, -0.2, 0.3]], [[-0.3, 0.2, 0.4], [0.2, 0.1, -0.1]]],
        dtype=torch.float32,
    )[:output_dim, :raw_input_dim]
    kernel_b = torch.flip(kernel_a, dims=(2,)) * 0.8

    def apply(inputs: Tensor) -> Tensor:
        midpoint = inputs.shape[1] // 2
        first_half = _causal_convolution(inputs[:, :midpoint, :], kernel_a)
        second_half = _causal_convolution(inputs[:, midpoint:, :], kernel_b)
        return torch.cat((first_half, second_half), dim=1)

    return SequenceRegressionTask(
        train_inputs=train_inputs,
        train_targets=apply(train_inputs),
        validation_inputs=validation_inputs,
        validation_targets=apply(validation_inputs),
        teacher_label="switching_teacher",
    )


def _evaluate_loss(
    model: nn.Module,
    task: SequenceRegressionTask | SyntheticLaplaceTask,
    device: str,
) -> float:
    with torch.no_grad():
        predictions = model(task.validation_inputs.to(device=device))
        return float(
            functional.mse_loss(
                predictions,
                task.validation_targets.to(device=device),
            ).item()
        )


def train_regression_model(
    model: nn.Module,
    task: SequenceRegressionTask | SyntheticLaplaceTask,
    config: TrainingConfig,
) -> RegressionOutcome:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.to(device=config.device)
    train_inputs = task.train_inputs.to(device=config.device)
    train_targets = task.train_targets.to(device=config.device)
    initial_loss = float(functional.mse_loss(model(train_inputs), train_targets).item())
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_inputs)
        loss = functional.mse_loss(predictions, train_targets)
        loss.backward()
        optimizer.step()
    final_loss = float(functional.mse_loss(model(train_inputs), train_targets).item())
    validation_loss = _evaluate_loss(model, task, config.device)
    pole_mae = (
        float(torch.mean(torch.abs(model.temporal_mixer.continuous_poles())).item())
        if isinstance(model, (ProjectedPRLBlock, GatedPRLBlock, HybridModalPRLBlock))
        else nan
    )
    return RegressionOutcome(
        initial_loss=initial_loss,
        final_loss=final_loss,
        validation_loss=validation_loss,
        pole_mae=pole_mae,
    )


def make_fixed_prl(
    *,
    raw_input_dim: int,
    model_dim: int,
    output_dim: int,
    modes: int,
) -> ProjectedPRLBlock:
    return ProjectedPRLBlock(
        raw_input_dim=raw_input_dim,
        model_dim=model_dim,
        output_dim=output_dim,
        modes=modes,
    )


def make_gated_prl(
    *,
    raw_input_dim: int,
    model_dim: int,
    output_dim: int,
    modes: int,
    gate_variant: GateVariant,
) -> GatedPRLBlock:
    return GatedPRLBlock(
        raw_input_dim=raw_input_dim,
        model_dim=model_dim,
        output_dim=output_dim,
        modes=modes,
        gate_variant=gate_variant,
    )


def make_hybrid_modal_prl(
    *,
    raw_input_dim: int,
    model_dim: int,
    output_dim: int,
    modes: int,
    fir_kernel_size: int,
    prl_tap_kernel_size: int | None = None,
    tap_parameterization: TapParameterization = "shared_scalar",
    low_rank_rank: int = 2,
    active_branches: tuple[BranchName, ...] = BRANCH_ORDER,
    fusion_variant: FusionVariant = "softmax",
    fusion_temperature: float = 1.0,
) -> HybridModalPRLBlock:
    return HybridModalPRLBlock(
        raw_input_dim=raw_input_dim,
        model_dim=model_dim,
        output_dim=output_dim,
        modes=modes,
        fir_kernel_size=fir_kernel_size,
        prl_tap_kernel_size=prl_tap_kernel_size,
        tap_parameterization=tap_parameterization,
        low_rank_rank=low_rank_rank,
        active_branches=active_branches,
        fusion_variant=fusion_variant,
        fusion_temperature=fusion_temperature,
    )


def make_fir_baseline(
    *,
    raw_input_dim: int,
    model_dim: int,
    output_dim: int,
    kernel_size: int,
) -> FIRSequenceBaseline:
    return FIRSequenceBaseline(
        raw_input_dim=raw_input_dim,
        model_dim=model_dim,
        output_dim=output_dim,
        kernel_size=kernel_size,
    )


def make_gru_baseline(
    *,
    raw_input_dim: int,
    model_dim: int,
    output_dim: int,
) -> GRUSequenceBaseline:
    return GRUSequenceBaseline(
        raw_input_dim=raw_input_dim,
        model_dim=model_dim,
        output_dim=output_dim,
    )


def make_transformer_baseline(
    *,
    raw_input_dim: int,
    model_dim: int,
    output_dim: int,
    attention_heads: int,
) -> TransformerSequenceBaseline:
    return TransformerSequenceBaseline(
        raw_input_dim=raw_input_dim,
        model_dim=model_dim,
        output_dim=output_dim,
        attention_heads=attention_heads,
    )
