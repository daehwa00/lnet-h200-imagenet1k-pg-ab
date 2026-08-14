from __future__ import annotations

from dataclasses import replace
from math import pi

from .advanced_experiments import make_fir_teacher_task, make_switching_teacher_task
from .experiment import SyntheticTaskConfig, make_synthetic_task
from .hybrid import BRANCH_ORDER, HybridModalPRLBlock
from .hybrid_delay_tasks import (
    DelayedExponentialTeacherConfig,
    DelayedOscillatoryTeacherConfig,
    StrictDelayTaskConfig,
    make_delayed_exponential_teacher_bundle,
    make_delayed_oscillatory_teacher_bundle,
    make_strict_delay_teacher_bundle,
)
from .models import FIRSequenceBaseline, GRUSequenceBaseline, TransformerSequenceBaseline
from .selective_tapped_prl import SelectiveTappedPRLBlock, SelectiveVariant
from .selective_tapped_prl_types import LabeledTask, SelectiveExperimentConfig


def ablation_tasks(
    config: SelectiveExperimentConfig,
    mode: str,
) -> tuple[LabeledTask, ...]:
    tasks = [modal_task(config), strict_delay_task(config, 6)]
    if mode == "full":
        tasks.extend(
            [
                fir_task(config),
                switching_task(config),
                delayed_exponential_task(config, 4),
                delayed_oscillatory_task(config, 4),
            ],
        )
    return tuple(tasks)


def parameter_tasks(
    config: SelectiveExperimentConfig,
    mode: str,
) -> tuple[LabeledTask, ...]:
    if mode == "smoke":
        return (modal_task(config), strict_delay_task(config, 6))
    return (
        modal_task(config),
        fir_task(config),
        switching_task(config),
        strict_delay_task(config, 6),
    )


def modal_task(config: SelectiveExperimentConfig) -> LabeledTask:
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
    return LabeledTask("modal_teacher", task)


def fir_task(config: SelectiveExperimentConfig) -> LabeledTask:
    task = make_fir_teacher_task(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        seed=config.seed,
    )
    return LabeledTask(task.teacher_label, task)


def switching_task(config: SelectiveExperimentConfig) -> LabeledTask:
    task = make_switching_teacher_task(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        seed=config.seed,
    )
    return LabeledTask(task.teacher_label, task)


def strict_delay_task(config: SelectiveExperimentConfig, delay: int) -> LabeledTask:
    bundle = make_strict_delay_teacher_bundle(strict_delay_config(config, delay))
    return LabeledTask(bundle.task.teacher_label, bundle.task, bundle.metadata)


def delayed_exponential_task(config: SelectiveExperimentConfig, delay: int) -> LabeledTask:
    bundle = make_delayed_exponential_teacher_bundle(
        DelayedExponentialTeacherConfig(
            sample_count=config.sample_count,
            validation_count=config.validation_count,
            sequence_length=config.sequence_length,
            raw_input_dim=config.raw_input_dim,
            output_dim=config.output_dim,
            seed=config.seed,
            delay_steps=delay,
            discrete_pole=0.85,
        ),
    )
    return LabeledTask(bundle.task.teacher_label, bundle.task, bundle.metadata)


def delayed_oscillatory_task(config: SelectiveExperimentConfig, delay: int) -> LabeledTask:
    bundle = make_delayed_oscillatory_teacher_bundle(
        DelayedOscillatoryTeacherConfig(
            sample_count=config.sample_count,
            validation_count=config.validation_count,
            sequence_length=config.sequence_length,
            raw_input_dim=config.raw_input_dim,
            output_dim=config.output_dim,
            seed=config.seed,
            delay_steps=delay,
            damping_radius=0.90,
            angular_frequency=pi / 4.0,
        ),
    )
    return LabeledTask(bundle.task.teacher_label, bundle.task, bundle.metadata)


def strict_delay_config(config: SelectiveExperimentConfig, delay: int) -> StrictDelayTaskConfig:
    return StrictDelayTaskConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        seed=config.seed,
        delay_steps=delay,
    )


def replace_tap_size(
    config: SelectiveExperimentConfig,
    tap_kernel_size: int,
) -> SelectiveExperimentConfig:
    return replace(config, tap_kernel_size=tap_kernel_size)


def make_selective(
    task: LabeledTask,
    config: SelectiveExperimentConfig,
    variant: SelectiveVariant,
) -> SelectiveTappedPRLBlock:
    return SelectiveTappedPRLBlock(
        raw_input_dim=task.task.train_inputs.shape[-1],
        model_dim=config.model_dim,
        output_dim=task.task.train_targets.shape[-1],
        modes=config.modes,
        tap_kernel_size=config.tap_kernel_size,
        variant=variant,
    )


def make_hybrid(task: LabeledTask, config: SelectiveExperimentConfig) -> HybridModalPRLBlock:
    return HybridModalPRLBlock(
        raw_input_dim=task.task.train_inputs.shape[-1],
        model_dim=config.model_dim,
        output_dim=task.task.train_targets.shape[-1],
        modes=config.modes,
        fir_kernel_size=max(config.tap_kernel_size, 3),
        prl_tap_kernel_size=config.tap_kernel_size,
        active_branches=BRANCH_ORDER,
    )


def make_fir(task: LabeledTask, config: SelectiveExperimentConfig) -> FIRSequenceBaseline:
    return FIRSequenceBaseline(
        raw_input_dim=task.task.train_inputs.shape[-1],
        model_dim=config.model_dim,
        output_dim=task.task.train_targets.shape[-1],
        kernel_size=max(config.tap_kernel_size, 3),
    )


def make_gru(task: LabeledTask, config: SelectiveExperimentConfig) -> GRUSequenceBaseline:
    return GRUSequenceBaseline(
        raw_input_dim=task.task.train_inputs.shape[-1],
        model_dim=config.model_dim,
        output_dim=task.task.train_targets.shape[-1],
    )


def make_transformer(
    task: LabeledTask,
    config: SelectiveExperimentConfig,
) -> TransformerSequenceBaseline:
    return TransformerSequenceBaseline(
        raw_input_dim=task.task.train_inputs.shape[-1],
        model_dim=config.model_dim,
        output_dim=task.task.train_targets.shape[-1],
        attention_heads=1,
    )
