from __future__ import annotations

import csv
import fcntl
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Final, Literal, cast

import torch
import typer

from .hybrid_experiment_types import resolve_device
from .pac_eval_sections import (
    clean_validation_classification_task,
    full_train_classification_task,
)
from .pac_external_reference_baselines import (
    ExternalMiniRocketClassifier,
    ExternalS4Classifier,
)
from .pac_metrics import count_parameters
from .pac_overnight_io import append_csv_row, prepare_overnight_dirs, write_csv_rows
from .pac_real_data import ensure_ucr_dataset, load_ucr_train_only
from .pac_stiefel_variants import REVISED_UNTIED_MODEL
from .pac_tf_evidence_queue import PROTOCOL_PATH, load_protocol
from .pac_tight_frame_models import build_tight_frame_classifier
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACDevice, PACExperimentConfig

if TYPE_CHECKING:
    from .tapped_prl_followup_schema import JsonRow

app = typer.Typer(add_completion=False)
DEFAULT_ROOT: Final = Path(".omx/results/pac-ucr-s4-minirocket-20260712")
FAMILIES: Final = ("s4", "minirocket")
TRIALS: Final = (
    (1.0e-3, 1.0e-5),
    (1.0e-3, 1.0e-4),
    (3.0e-3, 1.0e-5),
    (3.0e-3, 1.0e-4),
    (1.0e-2, 1.0e-5),
    (1.0e-2, 1.0e-4),
)
_BUILD_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class Job:
    key: str
    stage: Literal["validation", "test"]
    family: Literal["s4", "minirocket"]
    dataset: str
    seed: int
    trial: int
    learning_rate: float
    weight_decay: float
    epochs: int = 100
    slots: int = 2


def enqueue(root: Path = DEFAULT_ROOT, protocol_path: Path = PROTOCOL_PATH) -> int:
    protocol = load_protocol(protocol_path)
    jobs = tuple(
        Job(
            key=f"ucr_extra:validation:{family}:trial{trial}:{dataset}:seed{seed}",
            stage="validation",
            family=family,
            dataset=str(dataset),
            seed=int(seed),
            trial=trial,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        for family in FAMILIES
        for trial, (learning_rate, weight_decay) in enumerate(TRIALS, start=1)
        for dataset in protocol["development_datasets"]
        for seed in protocol["seeds"]
    )
    _write_manifest(root, jobs)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "contract.json").write_text(
        json.dumps(
            {
                "schema_version": "pac_ucr_s4_minirocket.v1",
                "selection_split": "official_train_stratified_validation",
                "official_test_read_during_selection": False,
                "families": list(FAMILIES),
                "trials": len(TRIALS),
                "validation_jobs": len(jobs),
                "final_scope": "full-TRAIN refit and official TEST on all 18 datasets",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(jobs)


def select_final(root: Path = DEFAULT_ROOT, protocol_path: Path = PROTOCOL_PATH) -> int:
    protocol = load_protocol(protocol_path)
    validation = {
        row["job_key"]: row
        for row in _read_rows(root / "results" / "ucr_s4_minirocket.csv")
        if row.get("stage") == "validation" and row.get("status") == "done"
    }
    expected = _read_jobs(root / "reports" / "validation_manifest.jsonl")
    missing = [job.key for job in expected if job.key not in validation]
    if missing:
        message = f"validation queue incomplete: missing={len(missing)} first={missing[0]}"
        raise ValueError(message)
    selected: dict[str, dict[str, float | int]] = {}
    for family in FAMILIES:
        scores: dict[int, list[float]] = {trial: [] for trial in range(1, 7)}
        epochs: dict[int, list[int]] = {trial: [] for trial in range(1, 7)}
        for row in validation.values():
            if row["family"] != family:
                continue
            trial = int(row["trial"])
            scores[trial].append(float(row["balanced_accuracy"]))
            epochs[trial].append(int(row["best_epoch"]))
        winner = max(scores, key=lambda trial: (mean(scores[trial]), -trial))
        learning_rate, weight_decay = TRIALS[winner - 1]
        selected[family] = {
            "trial": winner,
            "validation_balanced_accuracy": mean(scores[winner]),
            "refit_epochs": max(1, round(median(epochs[winner]))),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
    datasets = tuple(
        dict.fromkeys(
            list(protocol["development_datasets"]) + list(protocol["untouched_final_datasets"])
        )
    )
    jobs = tuple(
        Job(
            key=f"ucr_extra:test:{family}:{dataset}:seed{seed}",
            stage="test",
            family=family,
            dataset=str(dataset),
            seed=int(seed),
            trial=int(selected[family]["trial"]),
            learning_rate=float(selected[family]["learning_rate"]),
            weight_decay=float(selected[family]["weight_decay"]),
            epochs=int(selected[family]["refit_epochs"]),
        )
        for family in FAMILIES
        for dataset in datasets
        for seed in protocol["seeds"]
    )
    (root / "reports" / "selection.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_manifest(root, jobs, preserve_validation=True)
    return len(jobs)


def run_workers(
    root: Path,
    *,
    device: PACDevice,
    workers: int,
    total_slots: int,
) -> None:
    prepare_overnight_dirs(root)
    jobs = _read_jobs(root / "queue_manifest.jsonl")
    done = _done_keys(root)
    pending = [job for job in jobs if job.key not in done]
    _prewarm_s4_cuda(device, pending)
    active: dict[Future[tuple[Job, dict[str, object] | None]], int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            available = total_slots - sum(active.values())
            while pending and available >= 2:
                job = pending.pop(0)
                _event(root, job.key, "running")
                active[pool.submit(_execute, root, job, device)] = job.slots
                available -= job.slots
            if not active:
                break
            complete, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in complete:
                active.pop(future)
                job, row = future.result()
                _event(root, job.key, "done" if row is not None else "failed")


def _prewarm_s4_cuda(device_name: PACDevice, jobs: list[Job]) -> None:
    if device_name == "cpu" or not torch.cuda.is_available():
        return
    if not any(job.family == "s4" for job in jobs):
        return
    device = resolve_device(device_name)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model = ExternalS4Classifier(1, 1, 2).to(device=device)
        inputs = torch.zeros(2, 8, 1, device=device)
        model(inputs).sum().backward()
        torch.cuda.synchronize(device)
    del model, inputs


def shard_pending_jobs(
    source_root: Path,
    shard_root: Path,
    *,
    shard_index: int,
    shard_count: int,
    shard_weights: tuple[int, ...] | None = None,
) -> int:
    if shard_count < 1 or shard_index not in range(shard_count):
        message = f"invalid shard {shard_index}/{shard_count}"
        raise ValueError(message)
    if shard_weights is not None and (
        len(shard_weights) != shard_count or any(weight < 1 for weight in shard_weights)
    ):
        message = "shard weights must contain one positive integer per shard"
        raise ValueError(message)
    jobs = _read_jobs(source_root / "queue_manifest.jsonl")
    completed = _done_keys(source_root)
    pending = [job for job in jobs if job.key not in completed]
    if shard_weights is None:
        selected = tuple(
            job for index, job in enumerate(pending) if index % shard_count == shard_index
        )
    else:
        cycle = sum(shard_weights)
        lower = sum(shard_weights[:shard_index])
        upper = lower + shard_weights[shard_index]
        selected = tuple(
            job for index, job in enumerate(pending) if lower <= index % cycle < upper
        )
    _write_manifest(shard_root, selected)
    return len(selected)


def merge_result_roots(source_root: Path, shard_roots: tuple[Path, ...]) -> int:
    result = Path("results/ucr_s4_minirocket.csv")
    latest: dict[str, dict[str, str]] = {}
    for root in (source_root, *shard_roots):
        for row in _read_rows(root / result):
            key = row.get("job_key", "")
            if not key:
                continue
            previous = latest.get(key)
            if previous is None or row.get("status") == "done" or previous.get("status") != "done":
                latest[key] = row
    write_csv_rows(source_root / result, cast("list[JsonRow]", list(latest.values())))
    return sum(row.get("status") == "done" for row in latest.values())


def _execute(root: Path, job: Job, device_name: PACDevice) -> tuple[Job, dict[str, object] | None]:
    try:
        device = resolve_device(device_name)
        if job.stage == "validation":
            dataset = load_ucr_train_only(job.dataset, Path(".omx/data/ucr"))
            task = clean_validation_classification_task(dataset, job.seed)
        else:
            dataset = ensure_ucr_dataset(
                job.dataset,
                Path(".omx/data/ucr"),
                allow_download=True,
                require_train_label_space=True,
            )
            task = full_train_classification_task(dataset)
        config = PACExperimentConfig(
            task.train_inputs.shape[0],
            task.validation_inputs.shape[0],
            task.test_inputs.shape[0],
            task.train_inputs.shape[1],
            raw_input_dim=1,
            output_dim=task.class_count,
            model_dim=64,
            modes=16,
            epochs=job.epochs,
            batch_size=64,
            learning_rate=job.learning_rate,
            weight_decay=job.weight_decay,
            grad_clip_norm=1.0,
            seeds=(job.seed,),
            device=device_name,
            output_dir=root,
        )
        with _BUILD_LOCK, torch.random.fork_rng(devices=[]):
            torch.manual_seed(job.seed)
            model, target_params, relative_error, width = match_ucr_extra_model(
                job.family, config, task.class_count
            )
        started = perf_counter()
        outcome = train_classifier(
            model,
            task,
            config,
            device,
            job.seed,
            evaluate_test=job.stage == "test",
            restore_best_validation=job.stage == "validation",
        )
        inputs = task.validation_inputs if job.stage == "validation" else task.test_inputs
        labels = task.validation_labels if job.stage == "validation" else task.test_labels
        metrics = classification_metric_bundle(
            model, inputs.to(device=device), labels.to(device=device), batch_size=64
        )
        row: dict[str, object] = {
            "job_key": job.key,
            "stage": job.stage,
            "family": job.family,
            "dataset": job.dataset,
            "seed": job.seed,
            "trial": job.trial,
            "learning_rate": job.learning_rate,
            "weight_decay": job.weight_decay,
            "epochs": job.epochs,
            "best_epoch": outcome.best_epoch,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "weighted_f1": metrics.weighted_f1,
            "balanced_accuracy": metrics.balanced_accuracy,
            "params_trainable": count_parameters(model),
            "target_params": target_params,
            "relative_param_error": relative_error,
            "matched_width": width,
            "elapsed_time": perf_counter() - started,
            "status": "done",
        }
        append_csv_row(
            root / "results" / "ucr_s4_minirocket.csv", cast("JsonRow", row)
        )
        return job, row  # noqa: TRY300 - success row is finalized inside guarded execution
    except Exception as error:  # noqa: BLE001 - failed jobs remain resumable
        append_csv_row(
            root / "results" / "ucr_s4_minirocket.csv",
            asdict(job) | {"job_key": job.key, "status": "failed", "error": repr(error)},
        )
        return job, None


def match_ucr_extra_model(
    family: str, config: PACExperimentConfig, class_count: int
) -> tuple[torch.nn.Module, int, float, int]:
    reference = build_tight_frame_classifier(REVISED_UNTIED_MODEL, config, class_count)
    if reference is None:
        message = "revised PAC reference unavailable"
        raise RuntimeError(message)
    target = count_parameters(reference)
    if family == "minirocket":
        width = max(1, round((target - class_count) / class_count))
        model = ExternalMiniRocketClassifier(1, width, class_count)
        params = count_parameters(model)
        return model, target, abs(params - target) / target, width
    candidates: list[tuple[float, int, torch.nn.Module, int]] = []
    for width in range(1, 257):
        model = ExternalS4Classifier(1, width, class_count)
        params = count_parameters(model)
        candidates.append((abs(params - target) / target, width, model, params))
        if params >= target:
            break
    relative_error, width, model, _ = min(candidates, key=lambda row: (row[0], row[1]))
    return model, target, relative_error, width


def _write_manifest(
    root: Path, jobs: tuple[Job, ...], *, preserve_validation: bool = False
) -> None:
    prepare_overnight_dirs(root)
    text = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs)
    (root / "queue_manifest.jsonl").write_text(text, encoding="utf-8")
    if not preserve_validation:
        reports = root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "validation_manifest.jsonl").write_text(text, encoding="utf-8")


def _read_jobs(path: Path) -> tuple[Job, ...]:
    return tuple(Job(**json.loads(line)) for line in path.read_text().splitlines() if line)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _done_keys(root: Path) -> set[str]:
    return {
        row["job_key"]
        for row in _read_rows(root / "results" / "ucr_s4_minirocket.csv")
        if row.get("status") == "done"
    }


def _event(root: Path, key: str, status: str) -> None:
    path = root / "queue_state.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps({"key": key, "status": status}, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[
        Literal["enqueue", "workers", "select-final", "shard", "merge"],
        typer.Option("--stage"),
    ] = "enqueue",
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_ROOT,
    protocol: Annotated[Path, typer.Option("--protocol")] = PROTOCOL_PATH,
    device: Annotated[PACDevice, typer.Option("--device")] = "auto",
    workers: Annotated[int, typer.Option("--workers")] = 4,
    total_slots: Annotated[int, typer.Option("--total-slots")] = 8,
    source_root: Annotated[Path | None, typer.Option("--source-root")] = None,
    shard_index: Annotated[int, typer.Option("--shard-index")] = 0,
    shard_count: Annotated[int, typer.Option("--shard-count")] = 1,
    shard_weight: Annotated[list[int] | None, typer.Option("--shard-weight")] = None,
    shard_root: Annotated[list[Path] | None, typer.Option("--shard-root")] = None,
) -> None:
    if stage == "enqueue":
        typer.echo(f"enqueued={enqueue(output_root, protocol)}")
    elif stage == "workers":
        run_workers(
            output_root,
            device=device,
            workers=workers,
            total_slots=total_slots,
        )
    elif stage == "select-final":
        typer.echo(f"enqueued_final={select_final(output_root, protocol)}")
    elif stage == "shard":
        if source_root is None:
            message = "--source-root is required for sharding"
            raise typer.BadParameter(message)
        count = shard_pending_jobs(
            source_root,
            output_root,
            shard_index=shard_index,
            shard_count=shard_count,
            shard_weights=tuple(shard_weight) if shard_weight is not None else None,
        )
        typer.echo(f"sharded={count}")
    else:
        merged = merge_result_roots(output_root, tuple(shard_root or ()))
        typer.echo(f"merged_done={merged}")


if __name__ == "__main__":
    app()
