from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import TYPE_CHECKING, Literal, assert_never

import torch

from lnet.advanced_experiments import (
    RegressionOutcome,
    SequenceRegressionTask,
    make_fir_teacher_task,
    make_switching_teacher_task,
)
from lnet.experiment import (
    SyntheticLaplaceTask,
    SyntheticTaskConfig,
    TrainingConfig,
    make_synthetic_task,
)
from lnet.hybrid import (
    BRANCH_ORDER,
    BranchName,
    BranchNormalization,
    FusionVariant,
    HybridModalPRLBlock,
)
from lnet.hybrid_delay_tasks import (
    StrictDelayTaskConfig,
    TeacherMetadata,
    make_strict_delay_teacher_bundle,
)
from lnet.hybrid_metrics import count_trainable_parameters
from lnet.hybrid_training import HybridTrainingOptions, train_hybrid_regression_model

if TYPE_CHECKING:
    from lnet.tapped_prl import TapParameterization

TaskKind = Literal["modal", "fir", "switching", "delay"]
DeviceChoice = Literal["auto", "cpu", "cuda"]
DEFAULT_TASKS: tuple[TaskKind, ...] = ("modal", "fir", "switching", "delay")
DEFAULT_BRANCH_SETS: tuple[tuple[BranchName, ...], ...] = (
    ("prl",),
    ("fir",),
    ("mlp",),
    ("prl", "fir"),
    ("prl", "mlp"),
    ("fir", "mlp"),
    BRANCH_ORDER,
)


@dataclass(frozen=True, slots=True)
class HybridExperimentConfig:
    sample_count: int = 128
    validation_count: int = 32
    sequence_length: int = 40
    raw_input_dim: int = 2
    output_dim: int = 2
    model_dim: int = 8
    modes: int = 4
    fir_kernel_size: int = 13
    prl_tap_kernel_size: int | None = None
    tap_parameterization: TapParameterization = "shared_scalar"
    low_rank_rank: int = 2
    fusion_variant: FusionVariant = "softmax"
    fusion_temperature: float = 1.0
    branch_normalization: BranchNormalization = "none"
    branch_dropout_probability: float = 0.0
    gate_entropy_weight: float = 0.0
    tap_entropy_weight: float = 0.0
    epochs: int = 120
    learning_rate: float = 5.0e-2
    weight_decay: float = 1.0e-4
    seed: int = 7
    device: DeviceChoice = "auto"
    task_kinds: tuple[TaskKind, ...] = DEFAULT_TASKS
    branch_sets: tuple[tuple[BranchName, ...], ...] = DEFAULT_BRANCH_SETS
    baseline_model_dims: tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20, 24, 32)
    delay_steps: tuple[int, ...] = (2, 6, 10, 14)
    delay_kernel_sizes: tuple[int, ...] = (5, 9, 13, 17)
    run_parameter_match: bool = True
    run_delay_sweep: bool = True
    run_gate_diagnostics: bool = True


@dataclass(frozen=True, slots=True)
class NamedRegressionTask:
    label: str
    task: SequenceRegressionTask | SyntheticLaplaceTask
    teacher_metadata: TeacherMetadata | None = None


@dataclass(frozen=True, slots=True)
class TrainedHybridModel:
    task_label: str
    branch_label: str
    model: HybridModalPRLBlock
    outcome: RegressionOutcome
    parameter_count: int
    teacher_metadata: TeacherMetadata | None = None


HypothesisStatus = Literal["supports", "mixed", "does_not_support"]


@dataclass(frozen=True, slots=True)
class HypothesisCheck:
    hypothesis_id: str
    hypothesis_status: HypothesisStatus
    evidence_row_ids: tuple[str, ...]
    rationale: str
    comparison_groups: tuple[str, ...]


def format_float(value: float) -> str:
    if isnan(value):
        return "n/a"
    return f"{value:.6f}"


def resolve_device(choice: DeviceChoice) -> str:
    match choice:
        case "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        case "cpu":
            return "cpu"
        case "cuda":
            return "cuda"
        case unreachable:
            assert_never(unreachable)


def training_config(config: HybridExperimentConfig, device: str) -> TrainingConfig:
    return TrainingConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        seed=config.seed,
    )


def make_task(kind: TaskKind, config: HybridExperimentConfig) -> NamedRegressionTask:
    match kind:
        case "modal":
            task = make_synthetic_task(
                SyntheticTaskConfig(
                    sample_count=config.sample_count,
                    validation_count=config.validation_count,
                    sequence_length=config.sequence_length,
                    raw_input_dim=config.raw_input_dim,
                    model_dim=config.model_dim,
                    output_dim=config.output_dim,
                    modes=config.modes,
                    seed=config.seed,
                ),
            )
            return NamedRegressionTask(label="modal_teacher", task=task)
        case "fir":
            task = make_fir_teacher_task(
                sample_count=config.sample_count,
                validation_count=config.validation_count,
                sequence_length=config.sequence_length,
                raw_input_dim=config.raw_input_dim,
                output_dim=config.output_dim,
                seed=config.seed,
            )
            return NamedRegressionTask(label=task.teacher_label, task=task)
        case "switching":
            task = make_switching_teacher_task(
                sample_count=config.sample_count,
                validation_count=config.validation_count,
                sequence_length=config.sequence_length,
                raw_input_dim=config.raw_input_dim,
                output_dim=config.output_dim,
                seed=config.seed,
            )
            return NamedRegressionTask(label=task.teacher_label, task=task)
        case "delay":
            bundle = make_strict_delay_teacher_bundle(
                StrictDelayTaskConfig(
                    sample_count=config.sample_count,
                    validation_count=config.validation_count,
                    sequence_length=config.sequence_length,
                    raw_input_dim=config.raw_input_dim,
                    output_dim=config.output_dim,
                    seed=config.seed,
                    delay_steps=6,
                ),
            )
            return NamedRegressionTask(
                label=bundle.task.teacher_label,
                task=bundle.task,
                teacher_metadata=bundle.metadata,
            )
        case unreachable:
            assert_never(unreachable)


def gate_variant_label(config: HybridExperimentConfig) -> str:
    if config.fusion_variant != "temperature_softmax":
        return config.fusion_variant
    return (
        "temperature_softmax_tau_0_5"
        if config.fusion_temperature == 0.5
        else "temperature_softmax_tau_2_0"
    )


def branch_label(
    branches: tuple[BranchName, ...],
    config: HybridExperimentConfig,
) -> str:
    tap_kernel_size = config.prl_tap_kernel_size or config.fir_kernel_size
    if branches == ("prl",):
        return "instantaneous_prl" if tap_kernel_size == 1 else "tapped_prl"
    branch_names = "_".join(branches)
    return f"hybrid_{branch_names}_{gate_variant_label(config)}"


def make_hybrid(
    task: NamedRegressionTask,
    config: HybridExperimentConfig,
    branches: tuple[BranchName, ...],
) -> HybridModalPRLBlock:
    return HybridModalPRLBlock(
        raw_input_dim=task.task.train_inputs.shape[-1],
        model_dim=config.model_dim,
        output_dim=task.task.train_targets.shape[-1],
        modes=config.modes,
        fir_kernel_size=config.fir_kernel_size,
        prl_tap_kernel_size=config.prl_tap_kernel_size,
        tap_parameterization=config.tap_parameterization,
        low_rank_rank=config.low_rank_rank,
        active_branches=branches,
        fusion_variant=config.fusion_variant,
        fusion_temperature=config.fusion_temperature,
        branch_normalization=config.branch_normalization,
        branch_dropout_probability=config.branch_dropout_probability,
    )


def fit_hybrid(
    task: NamedRegressionTask,
    config: HybridExperimentConfig,
    training: TrainingConfig,
    branches: tuple[BranchName, ...],
) -> TrainedHybridModel:
    torch.manual_seed(config.seed)
    model = make_hybrid(task, config, branches)
    outcome = train_hybrid_regression_model(
        model,
        task.task,
        training,
        HybridTrainingOptions(
            gate_entropy_weight=config.gate_entropy_weight,
            tap_entropy_weight=config.tap_entropy_weight,
        ),
    )
    return TrainedHybridModel(
        task_label=task.label,
        branch_label=branch_label(branches, config),
        model=model,
        outcome=outcome,
        parameter_count=count_trainable_parameters(model),
        teacher_metadata=task.teacher_metadata,
    )
