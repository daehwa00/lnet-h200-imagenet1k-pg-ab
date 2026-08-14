from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

from .hybrid_experiment_types import resolve_device
from .pac_metrics import count_parameters, nrmse
from .pac_optimization import run_optimization
from .pac_paper_queue_interventions import counterfactual_row, role_ablation_row
from .pac_paper_queue_models import build_paper_regressor
from .pac_paper_queue_real import real_baseline_row
from .pac_paper_queue_tasks import baseline_timed_task, make_low_data_task, make_sampling_rate_task
from .pac_paper_queue_timed import timed_pac_model
from .pac_tasks import make_ood_task, make_pac_synthetic_tasks
from .pac_training import evaluate_regression_loss, train_regression_model

if TYPE_CHECKING:
    from pathlib import Path

    from .pac_paper_queue_types import PaperJob
    from .pac_types import PACExperimentConfig, PACRegressionTask, PACTrainOutcome
    from .tapped_prl_followup_schema import JsonRow


def run_job_row(root: Path, config: PACExperimentConfig, job: PaperJob) -> JsonRow:
    device = resolve_device(config.device)
    row: JsonRow
    match job.kind:
        case "sampling_rate_ood":
            row = _sampling_row(config, device, job, irregular=False)
        case "irregular_time_ood":
            row = _sampling_row(config, device, job, irregular=True)
        case "damping_counterfactual":
            row = counterfactual_row(config, device, job)
        case "expanded_ood" | "strong_baselines_synthetic":
            row = _synthetic_row(config, device, job)
        case "role_ablation":
            row = role_ablation_row(config, device, job)
        case "low_data_scaling":
            row = (
                real_baseline_row(config, device, job, ratio=job.ratio or 1.0)
                if job.dataset
                else _low_data_row(config, device, job)
            )
        case "strong_baselines_real":
            row = real_baseline_row(config, device, job, ratio=1.0)
        case "speed_correctness":
            row = _speed_row(root, config, job)
        case _:
            message = f"unsupported paper queue job kind: {job.kind}"
            raise KeyError(message)
    return row


def _sampling_row(
    config: PACExperimentConfig, device: str, job: PaperJob, *, irregular: bool
) -> JsonRow:
    seed = job.seed
    model_name = job.model
    delta = 1.0 if job.value is None else job.value
    timed = make_sampling_rate_task(config, seed, test_delta=delta, irregular=irregular)
    is_pac = model_name.startswith("pac") or "prl" in model_name
    run_config = replace(
        config, seeds=(seed,), raw_input_dim=config.raw_input_dim + (0 if is_pac else 1)
    )
    task = timed.task if is_pac else baseline_timed_task(timed)
    model = build_paper_regressor(model_name, run_config)
    started = perf_counter()
    outcome = train_regression_model(model, task, run_config, device, seed)
    eval_model = timed_pac_model(model, timed.test_delta) if is_pac else model
    test_inputs = timed.task.test_inputs if is_pac else task.test_inputs
    test_loss = evaluate_regression_loss(
        eval_model,
        test_inputs.to(device=device),
        timed.task.test_targets.to(device=device),
    )
    return _base_row("sampling_irregular_ood", model_name, timed.task.label, seed, outcome) | {
        "delta": "irregular" if irregular else delta,
        "test_loss": test_loss,
        "test_nrmse": nrmse(test_loss, timed.task.test_targets),
        "ood_train_loss_ratio": test_loss / max(outcome.train_loss, 1.0e-12),
        "params_trainable": count_parameters(model),
        "elapsed_time": perf_counter() - started,
    }


def _synthetic_row(config: PACExperimentConfig, device: str, job: PaperJob) -> JsonRow:
    run_config = replace(config, seeds=(job.seed,))
    task = _synthetic_task(run_config, job.seed, job.task)
    model = build_paper_regressor(job.model, run_config)
    outcome = train_regression_model(model, task, run_config, device, job.seed)
    return _base_row("synthetic", job.model, task.label, job.seed, outcome) | {
        "params_trainable": count_parameters(model),
        "test_nrmse": nrmse(outcome.test_loss, task.test_targets),
    }


def _low_data_row(config: PACExperimentConfig, device: str, job: PaperJob) -> JsonRow:
    ratio = 1.0 if job.ratio is None else job.ratio
    task = make_low_data_task(config, job.seed, ratio, job.task)
    run_config = replace(config, seeds=(job.seed,), sample_count=task.train_inputs.shape[0])
    model = build_paper_regressor(job.model, run_config)
    outcome = train_regression_model(model, task, run_config, device, job.seed)
    return _base_row("low_data_scaling", job.model, task.label, job.seed, outcome) | {
        "data_ratio": ratio,
        "params_trainable": count_parameters(model),
        "test_nrmse": nrmse(outcome.test_loss, task.test_targets),
    }


def _speed_row(root: Path, config: PACExperimentConfig, job: PaperJob) -> JsonRow:
    output = root / "artifacts" / "speed_correctness"
    run_optimization("smoke", config.device, output, compare_backends=False)
    return {
        "experiment_group": "speed_correctness",
        "seed": job.seed,
        "model": "pac_lite_pac_full",
        "task": "optimization_smoke",
        "artifact_dir": str(output),
        "status": "completed",
    }


def _synthetic_task(config: PACExperimentConfig, seed: int, name: str) -> PACRegressionTask:
    if name.startswith("ood_length_"):
        return make_ood_task(
            config,
            seed,
            sequence_length=int(name.removeprefix("ood_length_")),
            label=name,
        )
    if name.startswith("ood_noise_"):
        return make_ood_task(
            config,
            seed,
            sequence_length=config.sequence_length,
            noise=float(name.removeprefix("ood_noise_")),
            label=name,
        )
    if name.startswith("ood_delay_"):
        return make_ood_task(
            config,
            seed,
            sequence_length=config.sequence_length,
            delay=int(name.removeprefix("ood_delay_")),
            label=name,
        )
    if name.startswith("ood_damping_"):
        return make_ood_task(
            config,
            seed,
            sequence_length=config.sequence_length,
            fast_decay=float(name.removeprefix("ood_damping_")),
            label=name,
        )
    if name.startswith("ood_frequency_"):
        return make_ood_task(
            config,
            seed,
            sequence_length=config.sequence_length,
            omega=float(name.removeprefix("ood_frequency_")),
            label=name,
        )
    return {task.label: task for task in make_pac_synthetic_tasks(config, seed)}[name]


def _base_row(group: str, model: str, task: str, seed: int, outcome: PACTrainOutcome) -> JsonRow:
    return {
        "experiment_group": group,
        "model": model,
        "task": task,
        "seed": seed,
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "grad_norm": outcome.grad_norm,
        "elapsed_train_time": outcome.elapsed_time,
    }
