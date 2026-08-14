from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .pac_eval_sections import classification_task
from .pac_metrics import count_parameters
from .pac_paper_queue_models import build_paper_classifier
from .pac_real_data import ensure_ucr_dataset
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACClassificationTask, PACExperimentConfig

if TYPE_CHECKING:
    from .pac_paper_queue_types import PaperJob
    from .tapped_prl_followup_schema import JsonRow


def real_baseline_row(
    config: PACExperimentConfig, device: str, job: PaperJob, *, ratio: float
) -> JsonRow:
    dataset_name = "ECG5000" if job.dataset is None else job.dataset
    dataset = ensure_ucr_dataset(dataset_name, Path(".omx/data/ucr"), allow_download=True)
    task = classification_task(dataset.name, dataset)
    if ratio < 1.0:
        count = max(1, int(task.train_inputs.shape[0] * ratio))
        task = PACClassificationTask(
            task.label,
            task.train_inputs[:count],
            task.train_labels[:count],
            task.validation_inputs,
            task.validation_labels,
            task.test_inputs,
            task.test_labels,
            task.class_count,
        )
    run_config = replace(config, seeds=(job.seed,), raw_input_dim=1, output_dim=task.class_count)
    model = build_paper_classifier(job.model, run_config, task.class_count)
    outcome = train_classifier(model, task, run_config, device, job.seed)
    metrics = classification_metric_bundle(
        model,
        task.test_inputs.to(device=device),
        task.test_labels.to(device=device),
    )
    return {
        "experiment_group": "strong_baselines_real" if ratio == 1.0 else "low_data_scaling",
        "dataset_or_task": task.label,
        "seed": job.seed,
        "model": job.model,
        "data_ratio": ratio,
        "params_trainable": count_parameters(model),
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "test_accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "elapsed_train_time": outcome.elapsed_time,
    }
