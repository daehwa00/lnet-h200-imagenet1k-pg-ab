from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING, assert_never

from .advanced_experiments import make_fir_teacher_task, make_switching_teacher_task
from .experiment import SyntheticTaskConfig, make_synthetic_task
from .hybrid_delay_tasks import (
    DelayedExponentialTeacherConfig,
    DelayedOscillatoryTeacherConfig,
    StrictDelayTaskConfig,
    make_delayed_exponential_teacher_bundle,
    make_delayed_oscillatory_teacher_bundle,
    make_strict_delay_teacher_bundle,
)
from .hybrid_experiment_types import NamedRegressionTask

if TYPE_CHECKING:
    from .hybrid_branch_ablation_types import BranchTaskName, HybridBranchAblationConfig


def make_branch_ablation_task(
    task_name: BranchTaskName,
    config: HybridBranchAblationConfig,
    seed: int,
) -> NamedRegressionTask:
    match task_name:
        case "modal_teacher":
            task = make_synthetic_task(
                SyntheticTaskConfig(
                    sample_count=config.sample_count,
                    validation_count=config.validation_count,
                    sequence_length=config.sequence_length,
                    raw_input_dim=config.raw_input_dim,
                    model_dim=config.model_dim,
                    output_dim=config.output_dim,
                    modes=config.modes,
                    seed=seed,
                ),
            )
            return NamedRegressionTask(label="modal_teacher", task=task)
        case "random_fir_teacher":
            task = make_fir_teacher_task(
                sample_count=config.sample_count,
                validation_count=config.validation_count,
                sequence_length=config.sequence_length,
                raw_input_dim=config.raw_input_dim,
                output_dim=config.output_dim,
                seed=seed,
            )
            return NamedRegressionTask(label=task.teacher_label, task=task)
        case "strict_delay_6":
            bundle = make_strict_delay_teacher_bundle(_strict_delay_config(config, seed))
            return NamedRegressionTask(
                label=bundle.task.teacher_label,
                task=bundle.task,
                teacher_metadata=bundle.metadata,
            )
        case "delayed_exponential_4":
            bundle = make_delayed_exponential_teacher_bundle(
                DelayedExponentialTeacherConfig(
                    sample_count=config.sample_count,
                    validation_count=config.validation_count,
                    sequence_length=config.sequence_length,
                    raw_input_dim=config.raw_input_dim,
                    output_dim=config.output_dim,
                    seed=seed,
                    delay_steps=4,
                    discrete_pole=0.85,
                ),
            )
            return NamedRegressionTask(
                label=bundle.task.teacher_label,
                task=bundle.task,
                teacher_metadata=bundle.metadata,
            )
        case "delayed_oscillatory_4":
            bundle = make_delayed_oscillatory_teacher_bundle(
                DelayedOscillatoryTeacherConfig(
                    sample_count=config.sample_count,
                    validation_count=config.validation_count,
                    sequence_length=config.sequence_length,
                    raw_input_dim=config.raw_input_dim,
                    output_dim=config.output_dim,
                    seed=seed,
                    delay_steps=4,
                    damping_radius=0.90,
                    angular_frequency=pi / 4.0,
                ),
            )
            return NamedRegressionTask(
                label=bundle.task.teacher_label,
                task=bundle.task,
                teacher_metadata=bundle.metadata,
            )
        case "switching_teacher":
            task = make_switching_teacher_task(
                sample_count=config.sample_count,
                validation_count=config.validation_count,
                sequence_length=config.sequence_length,
                raw_input_dim=config.raw_input_dim,
                output_dim=config.output_dim,
                seed=seed,
            )
            return NamedRegressionTask(label=task.teacher_label, task=task)
        case unreachable:
            assert_never(unreachable)


def make_seed_tasks(
    config: HybridBranchAblationConfig,
    seed: int,
) -> tuple[NamedRegressionTask, ...]:
    return tuple(
        make_branch_ablation_task(task_name, config, seed) for task_name in config.task_names
    )


def _strict_delay_config(config: HybridBranchAblationConfig, seed: int) -> StrictDelayTaskConfig:
    return StrictDelayTaskConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        seed=seed,
        delay_steps=6,
    )
