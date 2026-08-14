from __future__ import annotations

from dataclasses import dataclass, replace
from math import log
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from .controlled_damping_tasks import (
    make_context_damped_exponential_task,
    make_delayed_context_damped_exponential_task,
)
from .controlled_damping_types import ControlledDampingTaskConfig
from .hybrid_experiment_types import resolve_device
from .pac_eval_sections import classification_task
from .pac_interpretability_metrics import (
    damping_row,
    impulse_row,
    mode_knockout_rows,
    pole_rows,
    tap_rows,
)
from .pac_interpretability_real import real_modal_rows
from .pac_metrics import count_parameters, nrmse
from .pac_paper_queue_models import build_paper_regressor
from .pac_real_data import ensure_ucr_dataset, load_ucr_dataset, write_tiny_ucr_fixture
from .pac_recommended_low_data_models import build_low_data_classifier
from .pac_tasks import make_pac_synthetic_tasks
from .pac_training import classification_metrics, train_classifier, train_regression_model
from .pac_types import PACClassificationTask, PACExperimentConfig, PACRegressionTask

if TYPE_CHECKING:
    from .pac_interpretability_types import InterpretabilityJob
    from .tapped_prl_followup_schema import JsonRow


@dataclass(frozen=True, slots=True)
class InterpretabilityRows:
    synthetic_performance: tuple[JsonRow, ...] = ()
    pole_recovery: tuple[JsonRow, ...] = ()
    tap_recovery: tuple[JsonRow, ...] = ()
    damping_alignment: tuple[JsonRow, ...] = ()
    mode_knockout: tuple[JsonRow, ...] = ()
    impulse_response_nmse: tuple[JsonRow, ...] = ()
    real_modal_class_stats: tuple[JsonRow, ...] = ()
    hermitian_attribution: tuple[JsonRow, ...] = ()
    real_performance: tuple[JsonRow, ...] = ()


def run_interpretability_job(
    config: PACExperimentConfig, job: InterpretabilityJob
) -> InterpretabilityRows:
    match job.package:
        case "synthetic_mechanism":
            return _run_synthetic(config, job)
        case "real_modal":
            return _run_real(config, job)


def _run_synthetic(config: PACExperimentConfig, job: InterpretabilityJob) -> InterpretabilityRows:
    device = resolve_device(config.device)
    run_config = replace(config, seeds=(job.seed,))
    task = _synthetic_task(run_config, job.task, job.seed)
    model = build_paper_regressor(job.model, run_config)
    started = perf_counter()
    outcome = train_regression_model(model, task, run_config, device, job.seed)
    performance: JsonRow = {
        "package": job.package,
        "queue_key": job.key,
        "task": task.label,
        "seed": job.seed,
        "model": job.model,
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "test_nrmse": nrmse(outcome.test_loss, task.test_targets),
        "params_trainable": count_parameters(model),
        "elapsed_train_time": outcome.elapsed_time,
        "elapsed_total_time": perf_counter() - started,
        "status": "done",
    }
    damping = damping_row(model, task, job.seed, device)
    impulse = impulse_row(model, task, job.seed)
    return InterpretabilityRows(
        synthetic_performance=(performance,),
        pole_recovery=_tag(pole_rows(model, task, job.seed), job),
        tap_recovery=_tag(tap_rows(model, task, job.seed), job),
        damping_alignment=_tag((damping,), job) if damping is not None else (),
        mode_knockout=_tag(
            mode_knockout_rows(model, task, job.seed, outcome.test_loss, device), job
        ),
        impulse_response_nmse=_tag((impulse,), job) if impulse is not None else (),
    )


def _run_real(config: PACExperimentConfig, job: InterpretabilityJob) -> InterpretabilityRows:
    device = resolve_device(config.device)
    run_config = replace(config, seeds=(job.seed,), raw_input_dim=1)
    task = _real_task(config.output_dir, job.task)
    run_config = replace(run_config, output_dim=task.class_count)
    started = perf_counter()
    model = build_low_data_classifier(job.model, run_config, task.class_count)
    outcome = train_classifier(model, task, run_config, device, job.seed)
    accuracy, macro_f1 = classification_metrics(
        model, task.test_inputs.to(device=device), task.test_labels.to(device=device)
    )
    stats, attribution = real_modal_rows(
        model, task.test_inputs, task.test_labels, task.label, job.seed, device
    )
    performance: JsonRow = {
        "package": job.package,
        "queue_key": job.key,
        "dataset_or_task": task.label,
        "seed": job.seed,
        "model": job.model,
        "test_accuracy": accuracy,
        "macro_f1": macro_f1,
        "test_loss": outcome.test_loss,
        "params_trainable": count_parameters(model),
        "elapsed_train_time": outcome.elapsed_time,
        "elapsed_total_time": perf_counter() - started,
        "status": "done",
    }
    return InterpretabilityRows(
        real_performance=(performance,),
        real_modal_class_stats=_tag(stats, job),
        hermitian_attribution=_tag(attribution, job),
    )


def _synthetic_task(config: PACExperimentConfig, label: str, seed: int) -> PACRegressionTask:
    for task in make_pac_synthetic_tasks(config, seed):
        if task.label == label:
            return task
    if label in {"context_damped_exponential", "delayed_context_damped_exponential"}:
        return _context_task(config, label, seed)
    message = f"unsupported interpretability task: {label}"
    raise KeyError(message)


def _context_task(config: PACExperimentConfig, label: str, seed: int) -> PACRegressionTask:
    task_config = ControlledDampingTaskConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count + config.test_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        seed=seed,
        delay_steps=4 if label == "delayed_context_damped_exponential" else 0,
    )
    controlled = (
        make_delayed_context_damped_exponential_task(task_config)
        if label == "delayed_context_damped_exponential"
        else make_context_damped_exponential_task(task_config)
    )
    split = config.validation_count
    regime = controlled.validation_fast_regime
    teacher = _teacher_alpha(regime, task_config.slow_decay, task_config.fast_decay)
    return PACRegressionTask(
        label,
        train_inputs=controlled.task.train_inputs,
        train_targets=controlled.task.train_targets,
        validation_inputs=controlled.task.validation_inputs[:split],
        validation_targets=controlled.task.validation_targets[:split],
        test_inputs=controlled.task.validation_inputs[split:],
        test_targets=controlled.task.validation_targets[split:],
        true_delay=task_config.delay_steps,
        true_frequency=0.0,
        validation_teacher_damping=teacher[:split],
        test_teacher_damping=teacher[split:],
        validation_regime=regime[:split] if regime is not None else None,
    )


def _teacher_alpha(
    regime: torch.Tensor | None, slow_decay: float, fast_decay: float
) -> torch.Tensor:
    if regime is None:
        return torch.empty(0)
    slow = torch.full(regime.shape, -log(slow_decay), dtype=torch.float32)
    fast = torch.full(regime.shape, -log(fast_decay), dtype=torch.float32)
    return torch.where(regime, fast, slow)


def _real_task(output_dir: Path, dataset: str) -> PACClassificationTask:
    root = output_dir / "artifacts" / "ucr" if dataset == "Tiny" else Path(".omx/data/ucr")
    if dataset == "Tiny":
        write_tiny_ucr_fixture(root)
        ucr = load_ucr_dataset(dataset, root)
    else:
        ucr = ensure_ucr_dataset(dataset, root, allow_download=True)
    return classification_task(ucr.name, ucr)


def _tag(rows: tuple[JsonRow | None, ...], job: InterpretabilityJob) -> tuple[JsonRow, ...]:
    tagged: list[JsonRow] = []
    for row in rows:
        if row is not None:
            row["package"] = job.package
            row["queue_key"] = job.key
            row["model"] = job.model
            tagged.append(row)
    return tuple(tagged)
