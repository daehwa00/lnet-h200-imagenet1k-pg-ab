# ruff: noqa: BLE001, EM101, EM102, TRY003
from __future__ import annotations

import csv
import gc
import hashlib
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, cast

import torch
from torch import nn

from .pac_confirmatory_baselines import build_confirmatory_family
from .pac_device import resolve_device
from .pac_eval_sections import clean_validation_classification_task
from .pac_external_benchmarks import (
    ExternalBenchmarkConfig,
    _loss,
    _measure_latency,
    _predict,
    _release_device,
    _run_one,
    _seed_everything,
    _train_model,
    external_metric_bundle,
    match_external_parameter_budget,
)
from .pac_external_reference_baselines import ExternalMiniRocketClassifier
from .pac_external_tasks import ExternalDatasetName, ExternalTask, load_external_task
from .pac_final_validation import UCR_SECONDS
from .pac_headroom_efficient_models import build_efficient_headroom_classifier
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_train_only
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACDevice, PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .pac_confirmatory_baselines import ConfirmatoryFamily
    from .pac_external_benchmarks import ExternalModelFamily

DEFAULT_ROOT: Final = Path(".omx/results/pac-efp16-final-comparison-20260713")
REVISED_EXTERNAL_TIMING: Final = Path(
    ".omx/results/pac-35-dataset-comparison-20260712/input/revised_external.csv"
)
HISTORICAL_EXTERNAL_JOB_ROOT: Final = Path(
    ".omx/results/pac-selected-d64m16-external-20260711/jobs"
)
UCR_DATA_ROOT: Final = Path(".omx/data/ucr")
EXTERNAL_DATA_ROOT: Final = Path("data/external")
BASELINE_SELECTION: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
MINIROCKET_SELECTION: Final = Path(
    ".omx/results/pac-ucr-s4-minirocket-20260712/reports/selection.json"
)
DEFAULT_UCR_IMPORT_ROOTS: Final = (
    Path(".omx/results/pac-edge-frame-secondary_gpu-20260713/completed"),
    Path(".omx/results/pac-edge-frame-extension-secondary_gpu-20260713/completed"),
    Path(".omx/results/pac-edge-frame-cincecg-twolead-secondary_gpu-20260713/completed"),
    Path(".omx/results/pac-edge-frame-ettm1-seqcifar-plane-secondary_gpu-20260713/completed"),
)

UCR_DATASETS: Final = (
    "ArrowHead",
    "CinCECGTorso",
    "CricketX",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "Earthquakes",
    "FordA",
    "FordB",
    "GunPoint",
    "ItalyPowerDemand",
    "MoteStrain",
    "Phoneme",
    "Plane",
    "StarLightCurves",
    "Trace",
    "TwoLeadECG",
    "Wafer",
)


EXTERNAL_DATASETS: Final[tuple[ExternalDatasetName, ...]] = (
    "ptb-xl",
    "mit-bih",
    "cwru",
    "speech-commands",
    "ettm1",
    "ettm2",
    "electricity",
    "weather",
    "sequential-mnist",
    "permuted-mnist",
    "sequential-cifar",
    "audioset-balanced",
)
BASELINES: Final = (
    "tcn",
    "cnn1d",
    "transformer",
    "mamba",
    "gru",
    "lstm",
    "s4d",
    "minirocket",
    "inception_time",
)
MODELS: Final = ("efp16", *BASELINES)
SEEDS: Final = (7, 11, 19, 23, 31)
PARAMETER_TOLERANCE: Final = 0.062
UCR_PARAMETER_TOLERANCE_EXCEPTIONS: Final = {
    ("lstm", 2): 0.064,
    ("lstm", 3): 0.077,
    ("gru", 4): 0.069,
    ("cnn1d", 5): 0.063,
    ("gru", 5): 0.084,
    ("gru", 12): 0.067,
}
EXTERNAL_PARAMETER_TOLERANCE_EXCEPTIONS: Final = {
    ("cnn1d", 1, 10): 0.065,
}
MAX_BASELINE_WIDTH: Final = 8192


def _external_historical_seconds() -> dict[tuple[str, str], float]:
    """Read timing hints without importing the Python-3.12 queue module on B200/Py3.10."""
    values: dict[tuple[str, str], list[float]] = {}
    paths = sorted(HISTORICAL_EXTERNAL_JOB_ROOT.glob("*/results/external_comparisons.csv"))
    paths.append(REVISED_EXTERNAL_TIMING)
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "done":
                    continue
                model = "pac" if row["model"] == "pac" else row["model"]
                values.setdefault((row["dataset"], model), []).append(float(row["train_seconds"]))
    return {key: median(observations) for key, observations in values.items()}


@dataclass(frozen=True, slots=True)
class FinalComparisonJob:
    suite: Literal["ucr", "external"]
    dataset: str
    model: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    validation_trial: int = 0
    estimated_seconds: float = 60.0

    @property
    def key(self) -> str:
        return f"efp16_final:{self.suite}:{self.dataset}:{self.model}:seed{self.seed}"


@dataclass(frozen=True, slots=True)
class MatchedModel:
    model: nn.Module
    width: int
    parameters: int
    target_parameters: int
    relative_error: float


def ucr_jobs(
    *,
    baseline_selection: Path = BASELINE_SELECTION,
    minirocket_selection: Path = MINIROCKET_SELECTION,
) -> list[FinalComparisonJob]:
    recipes = _ucr_recipes(baseline_selection, minirocket_selection)
    jobs: list[FinalComparisonJob] = []
    for dataset in UCR_DATASETS:
        for model in MODELS:
            recipe = recipes[model]
            jobs.extend(
                FinalComparisonJob(
                    suite="ucr",
                    dataset=dataset,
                    model=model,
                    seed=seed,
                    epochs=100,
                    batch_size=int(recipe["batch_size"]),
                    learning_rate=float(recipe["learning_rate"]),
                    weight_decay=float(recipe["weight_decay"]),
                    grad_clip_norm=float(recipe["grad_clip_norm"]),
                    validation_trial=int(recipe["validation_trial"]),
                    estimated_seconds=UCR_SECONDS[dataset] * (2.0 if model == "efp16" else 1.0),
                )
                for seed in SEEDS
            )
    return jobs


def external_jobs() -> list[FinalComparisonJob]:
    historical = _external_historical_seconds()
    return [
        FinalComparisonJob(
            suite="external",
            dataset=dataset,
            model=model,
            seed=seed,
            epochs=60,
            batch_size=64,
            learning_rate=1.0e-3,
            weight_decay=1.0e-4,
            grad_clip_norm=1.0,
            estimated_seconds=historical.get(
                (dataset, "pac" if model == "efp16" else model),
                historical.get((dataset, "pac"), 120.0),
            ),
        )
        for dataset in EXTERNAL_DATASETS
        for model in MODELS
        for seed in SEEDS
    ]


def enqueue(
    root: Path = DEFAULT_ROOT,
    *,
    ucr_shards: int = 16,
    external_shards: int = 16,
    import_roots: tuple[Path, ...] | None = None,
    baseline_selection: Path = BASELINE_SELECTION,
    minirocket_selection: Path = MINIROCKET_SELECTION,
) -> dict[str, object]:
    if ucr_shards < 1 or external_shards < 1:
        raise ValueError("shard counts must be positive")
    active_import_roots = DEFAULT_UCR_IMPORT_ROOTS if import_roots is None else import_roots
    imported = import_existing_ucr_rows(root, active_import_roots)
    ucr = ucr_jobs(
        baseline_selection=baseline_selection,
        minirocket_selection=minirocket_selection,
    )
    external = external_jobs()
    ucr_loads = _write_shards(root, "ucr", ucr, ucr_shards)
    external_loads = _write_shards(root, "external", external, external_shards)
    selection_sha = hashlib.sha256(baseline_selection.read_bytes()).hexdigest()
    minirocket_sha = hashlib.sha256(minirocket_selection.read_bytes()).hexdigest()
    contract: dict[str, object] = {
        "schema": "pac_efp16_final_comparison.v1",
        "public_model": "ALPHABET",
        "internal_model": "EFP16",
        "architecture": (
            "degree-normalized full-rate edge analysis; semi-orthogonal projection; D=32, M=16"
        ),
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "baselines": list(BASELINES),
        "parameter_matching": {
            "target": "dataset/task-specific EFP16 trainable parameters",
            "relative_tolerance": PARAMETER_TOLERANCE,
            "max_width": MAX_BASELINE_WIDTH,
            "matching_policy": "nearest real architecture width; no dummy or adapter parameters",
            "ucr_predeclared_exceptions": [
                {
                    "family": family,
                    "class_count": class_count,
                    "relative_tolerance": tolerance,
                }
                for (family, class_count), tolerance in sorted(
                    UCR_PARAMETER_TOLERANCE_EXCEPTIONS.items()
                )
            ],
            "external_predeclared_exceptions": [
                {
                    "family": family,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "relative_tolerance": tolerance,
                }
                for (family, input_dim, output_dim), tolerance in sorted(
                    EXTERNAL_PARAMETER_TOLERANCE_EXCEPTIONS.items()
                )
            ],
            "exception_rationale": (
                "discrete real widths leave no architecture instance inside the umbrella; "
                "each exception is the next 0.001 ceiling above the audited nearest width"
            ),
        },
        "ucr": {
            "datasets": list(UCR_DATASETS),
            "evaluation_split": "official TRAIN-derived stratified validation only",
            "official_test_accessed": False,
            "jobs": len(ucr),
            "efp16_jobs": len(UCR_DATASETS) * len(SEEDS),
            "baseline_jobs": len(UCR_DATASETS) * len(BASELINES) * len(SEEDS),
            "imported_protocol_compatible_efp16_rows": imported,
            "import_policy": (
                "only EFP16 UCR TRAIN-derived validation rows with the locked 100-epoch, "
                "batch-32 recipe"
            ),
            "shards": ucr_shards,
            "estimated_shard_seconds": ucr_loads,
        },
        "external": {
            "datasets": list(EXTERNAL_DATASETS),
            "excluded_datasets": [
                "lra-listops",
                "lra-text",
                "lra-retrieval",
                "lra-image",
            ],
            "exclusion_rationale": "long-context evaluation is outside the paper claim",
            "jobs": len(external),
            "efp16_jobs": len(EXTERNAL_DATASETS) * len(SEEDS),
            "baseline_jobs": len(EXTERNAL_DATASETS) * len(BASELINES) * len(SEEDS),
            "split_policy": "same task-provided train/validation/test split for every model",
            "checkpoint_policy": "minimum validation loss; one final test evaluation",
            "imported_screen_rows": 0,
            "import_policy": (
                "external EFP16 screening rows are validation-only and are never reused"
            ),
            "shards": external_shards,
            "estimated_shard_seconds": external_loads,
        },
        "baseline_recipe_source": str(baseline_selection),
        "baseline_recipe_source_sha256": selection_sha,
        "minirocket_recipe_source": str(minirocket_selection),
        "minirocket_recipe_source_sha256": minirocket_sha,
        "restart_safe": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_once(root / "contract.json", contract)
    return contract


def import_existing_ucr_rows(root: Path, source_roots: Iterable[Path]) -> int:
    expected = {
        f"efp16_final:ucr:{dataset}:efp16:seed{seed}" for dataset in UCR_DATASETS for seed in SEEDS
    }
    imported: dict[str, tuple[dict[str, object], Path]] = {}
    for source_root in source_roots:
        for path in sorted(source_root.glob("ucr_validation_*_EFP16_seed*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            dataset = str(payload.get("dataset", ""))
            seed = int(payload.get("seed", -1))
            key = f"efp16_final:ucr:{dataset}:efp16:seed{seed}"
            if key not in expected:
                continue
            _validate_import_payload(payload, path)
            previous = imported.get(key)
            if previous is not None and previous[0] != payload:
                raise ValueError(f"conflicting EFP16 import rows for {key}")
            imported[key] = payload, path
    destination = root / "ucr" / "completed"
    destination.mkdir(parents=True, exist_ok=True)
    for key, (payload, source) in imported.items():
        row = {
            **payload,
            "schema": "pac_efp16_final_ucr.v1",
            "job_key": key,
            "source_job_key": payload.get("job_key", payload.get("key")),
            "model": "efp16",
            "imported": True,
            "import_source": str(source),
            "import_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "parameter_target": "self",
            "target_params": int(payload["params_trainable"]),
            "relative_param_error": 0.0,
        }
        _write_result_path(destination / f"{_safe(key)}.json", row, preserve=True)
    return len(imported)


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: PACDevice = "cuda",
    ucr_data_root: Path = UCR_DATA_ROOT,
    external_data_root: Path = EXTERNAL_DATA_ROOT,
) -> None:
    jobs = [
        _job_from_payload(json.loads(line)) for line in manifest.read_text().splitlines() if line
    ]
    for job in jobs:
        completed = _result_path(root, job, failed=False)
        if completed.exists():
            continue
        try:
            if job.suite == "ucr":
                row = run_ucr_job(job, device=device, data_root=ucr_data_root)
            else:
                row = run_external_job(job, device=device, data_root=external_data_root)
            _require_done(row)
        except Exception as error:
            failed_row: dict[str, object] = {
                "schema": "pac_efp16_final_failure.v1",
                "job_key": job.key,
                **asdict(job),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _write_result(root, job, failed_row, failed=True)
        else:
            _write_result(root, job, row, failed=False)
            failed = _result_path(root, job, failed=True)
            if failed.exists():
                failed.unlink()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_ucr_job(
    job: FinalComparisonJob,
    *,
    device: PACDevice,
    data_root: Path = UCR_DATA_ROOT,
) -> dict[str, object]:
    if job.suite != "ucr":
        raise ValueError("run_ucr_job requires a UCR job")
    runtime_device = resolve_device(device)
    dataset = ensure_ucr_train_only(job.dataset, data_root, allow_download=True)
    task = clean_validation_classification_task(dataset, job.seed)
    config = PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        0,
        task.train_inputs.shape[1],
        raw_input_dim=task.train_inputs.shape[-1],
        output_dim=task.class_count,
        model_dim=32,
        modes=16,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=job.learning_rate,
        weight_decay=job.weight_decay,
        grad_clip_norm=job.grad_clip_norm,
        seeds=(job.seed,),
        device=device,
    )
    _seed_everything(job.seed, runtime_device)
    reference = build_efficient_headroom_classifier(
        "EFP16", config, task.class_count, objective="classification"
    )
    target = count_parameters(reference)
    if job.model == "efp16":
        matched = MatchedModel(reference, 32, target, target, 0.0)
    else:
        matched = match_ucr_baseline(
            job.model,
            config,
            task.class_count,
            target_parameters=target,
            validation_trial=job.validation_trial,
            tolerance=ucr_parameter_tolerance(job.model, task.class_count),
        )
    model = matched.model.to(device=runtime_device)
    if job.model == "efp16" and runtime_device == "cuda":
        model.__dict__["use_efp16_exact_split_training"] = True
    if runtime_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    outcome = train_classifier(
        model,
        task,
        config,
        runtime_device,
        job.seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    metrics = classification_metric_bundle(
        model,
        task.validation_inputs.to(device=runtime_device),
        task.validation_labels.to(device=runtime_device),
        batch_size=job.batch_size,
    )
    return {
        "schema": "pac_efp16_final_ucr.v1",
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "evaluation_split": "validation",
        "official_test_accessed": False,
        "checkpoint_policy": "minimum TRAIN-derived validation loss",
        "best_epoch": outcome.best_epoch,
        "validation_loss": outcome.validation_loss,
        "validation_accuracy": metrics.accuracy,
        "validation_macro_f1": metrics.macro_f1,
        "validation_weighted_f1": metrics.weighted_f1,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        "train_seconds": perf_counter() - started,
        "params_trainable": count_parameters(model),
        "target_params": matched.target_parameters,
        "matched_width": matched.width,
        "relative_param_error": matched.relative_error,
        "imported": False,
    }


def match_ucr_baseline(  # noqa: C901 - bounded monotone width search
    family: str,
    config: PACExperimentConfig,
    class_count: int,
    *,
    target_parameters: int,
    validation_trial: int,
    tolerance: float = PARAMETER_TOLERANCE,
    max_width: int = MAX_BASELINE_WIDTH,
) -> MatchedModel:
    def build(width: int) -> nn.Module:
        if family == "minirocket":
            return ExternalMiniRocketClassifier(config.raw_input_dim, width, class_count)
        return build_confirmatory_family(
            cast("ConfirmatoryFamily", family),
            width,
            config,
            class_count,
            validation_trial=validation_trial,
        )

    candidates: list[tuple[float, int, int]] = []

    def evaluate(width: int) -> int:
        parameters = count_parameters(build(width))
        candidates.append(
            (abs(parameters - target_parameters) / max(target_parameters, 1), width, parameters)
        )
        return parameters

    first = evaluate(1)
    if first < target_parameters and max_width > 1:
        lower = 1
        upper = min(2, max_width)
        while upper > lower:
            parameters = evaluate(upper)
            if parameters >= target_parameters:
                while lower + 1 < upper:
                    middle = (lower + upper) // 2
                    middle_parameters = evaluate(middle)
                    if middle_parameters < target_parameters:
                        lower = middle
                    else:
                        upper = middle
                break
            if upper == max_width:
                break
            lower = upper
            upper = min(2 * upper, max_width)
    relative_error, width, parameters = min(candidates)
    model = build(width)
    if relative_error > tolerance:
        raise ValueError(f"{family} parameter error {relative_error:.6f} exceeds {tolerance:.6f}")
    return MatchedModel(
        model,
        width,
        parameters,
        target_parameters,
        relative_error,
    )


def ucr_parameter_tolerance(family: str, class_count: int) -> float:
    """Return the frozen UCR real-width tolerance for a family/output shape."""
    return UCR_PARAMETER_TOLERANCE_EXCEPTIONS.get(
        (family, class_count),
        PARAMETER_TOLERANCE,
    )


def external_parameter_tolerance(family: str, input_dim: int, output_dim: int) -> float:
    """Return the frozen external real-width tolerance for a family/task shape."""
    return EXTERNAL_PARAMETER_TOLERANCE_EXCEPTIONS.get(
        (family, input_dim, output_dim),
        PARAMETER_TOLERANCE,
    )


def run_external_job(
    job: FinalComparisonJob,
    *,
    device: PACDevice,
    data_root: Path = EXTERNAL_DATA_ROOT,
) -> dict[str, object]:
    if job.suite != "external":
        raise ValueError("run_external_job requires an external job")
    runtime_device = resolve_device(device)
    dataset_name = cast("ExternalDatasetName", job.dataset)
    task = load_external_task(dataset_name, data_root)
    if task.input_encoding != "continuous":
        raise ValueError(f"EFP16 final scope requires continuous inputs: {job.dataset}")
    benchmark = _external_benchmark(
        data_root,
        dataset_name,
        model=job.model,
        seed=job.seed,
        device=device,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=job.learning_rate,
        weight_decay=job.weight_decay,
        grad_clip_norm=job.grad_clip_norm,
        parameter_match_tolerance=external_parameter_tolerance(
            job.model,
            task.input_dim,
            task.output_dim,
        ),
    )
    experiment = _external_experiment(task, benchmark)
    _seed_everything(job.seed, runtime_device)
    reference = build_efficient_headroom_classifier(
        "EFP16",
        experiment,
        task.output_dim,
        objective="regression" if task.objective == "forecasting" else "classification",
    )
    target = count_parameters(reference)
    if job.model == "efp16":
        row = _run_external_efp16(task, reference, target, job, benchmark, runtime_device)
    else:
        family = cast("ExternalModelFamily", job.model)
        match = match_external_parameter_budget(family, target, task, benchmark)
        if match.status != "matched" or match.width is None:
            raise RuntimeError(match.reason or f"{job.model} parameter matching failed")
        row = _run_one(task, match, target, job.seed, benchmark, runtime_device)
    return {
        "schema": "pac_efp16_final_external.v1",
        "job_key": job.key,
        **asdict(job),
        **row,
        "dataset": job.dataset,
        "model": job.model,
        "split_policy": "task-provided train/validation/test",
        "checkpoint_policy": "minimum validation loss; one final test evaluation",
        "external_screen_reused": False,
    }


def _run_external_efp16(
    task: ExternalTask,
    model: nn.Module,
    target: int,
    job: FinalComparisonJob,
    benchmark: ExternalBenchmarkConfig,
    device: str,
) -> dict[str, object]:
    _seed_everything(job.seed, device)
    model = model.to(device=device)
    started = perf_counter()
    try:
        best_epoch, validation_loss = _train_model(model, task, benchmark, device, job.seed)
        train_seconds = perf_counter() - started
        logits, targets = _predict(
            model, task.test_inputs, task.test_targets, benchmark.batch_size, device
        )
        metrics = external_metric_bundle(logits, targets, task.objective)
        test_loss = float(_loss(logits, targets, task.objective).item())
        latency_ms, peak_memory_mb = _measure_latency(model, task.test_inputs, benchmark, device)
        return {
            "status": "done",
            "objective": task.objective,
            "best_epoch": best_epoch,
            "validation_loss": validation_loss,
            "test_loss": test_loss,
            "train_seconds": train_seconds,
            "latency_ms": latency_ms,
            "peak_memory_mb": peak_memory_mb,
            "params_trainable": target,
            "target_params": target,
            "matched_width": 32,
            "relative_param_error": 0.0,
            **metrics,
        }
    finally:
        _release_device(device)


def parameter_match_preflight(
    *,
    suites: tuple[Literal["ucr", "external"], ...] = ("ucr", "external"),
    ucr_data_root: Path = UCR_DATA_ROOT,
    external_data_root: Path = EXTERNAL_DATA_ROOT,
    baseline_selection: Path = BASELINE_SELECTION,
    minirocket_selection: Path = MINIROCKET_SELECTION,
    allow_ucr_download: bool = False,
) -> dict[str, object]:
    """Validate real-width parameter matching before any training job is launched."""
    invalid = set(suites) - {"ucr", "external"}
    if invalid:
        raise ValueError(f"unsupported preflight suites: {sorted(invalid)}")
    payload: dict[str, object] = {
        "schema": "pac_efp16_final_parameter_preflight.v1",
        "tolerance": PARAMETER_TOLERANCE,
        "matching_policy": "nearest real architecture width; no dummy or adapter parameters",
        "suites": {},
    }
    suite_payloads = cast("dict[str, object]", payload["suites"])
    if "ucr" in suites:
        suite_payloads["ucr"] = _ucr_parameter_match_preflight(
            ucr_data_root,
            baseline_selection,
            minirocket_selection,
            allow_download=allow_ucr_download,
        )
    if "external" in suites:
        suite_payloads["external"] = _external_parameter_match_preflight(external_data_root)
    violations = sum(
        int(cast("dict[str, object]", value)["violation_count"])
        for value in suite_payloads.values()
    )
    payload["violation_count"] = violations
    payload["passed"] = violations == 0
    return payload


def write_parameter_match_preflight(
    root: Path = DEFAULT_ROOT,
    **kwargs: object,
) -> dict[str, object]:
    payload = parameter_match_preflight(**kwargs)  # type: ignore[arg-type]
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "PARAMETER_MATCH_PREFLIGHT.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (
            previous.get("schema") != payload["schema"]
            or previous.get("tolerance") != payload["tolerance"]
            or previous.get("matching_policy") != payload["matching_policy"]
        ):
            raise ValueError(f"incompatible parameter preflight artifact: {path}")
        merged_suites = {
            **cast("dict[str, object]", previous.get("suites", {})),
            **cast("dict[str, object]", payload["suites"]),
        }
        violations = sum(
            int(cast("dict[str, object]", value)["violation_count"])
            for value in merged_suites.values()
        )
        payload["suites"] = merged_suites
        payload["violation_count"] = violations
        payload["passed"] = violations == 0
    _write_result_path(path, payload)
    return payload


def campaign_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    ucr_expected = {
        f"efp16_final:ucr:{dataset}:{model}:seed{seed}"
        for dataset in UCR_DATASETS
        for model in MODELS
        for seed in SEEDS
    }
    external_expected = {
        f"efp16_final:external:{dataset}:{model}:seed{seed}"
        for dataset in EXTERNAL_DATASETS
        for model in MODELS
        for seed in SEEDS
    }
    return {
        "schema": "pac_efp16_final_status.v1",
        "ucr": _suite_status(root, "ucr", ucr_expected),
        "external": _suite_status(root, "external", external_expected),
    }


def write_report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    status = campaign_status(root)
    ucr_rows = _completed_rows(root / "ucr" / "completed")
    external_rows = _completed_rows(root / "external" / "completed")
    payload: dict[str, object] = {
        "schema": "pac_efp16_final_report.v1",
        "status": status,
        "ucr": _ucr_summary(ucr_rows),
        "external": _external_summary(external_rows),
        "reused": {
            "ucr_efp16_rows": sum(bool(row.get("imported")) for row in ucr_rows),
            "external_screen_rows": 0,
        },
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    _write_result_path(reports / "EFP16_FINAL_COMPARISON.json", payload)
    if bool(cast("dict[str, object]", status["ucr"])["done"]) and bool(
        cast("dict[str, object]", status["external"])["done"]
    ):
        (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return payload


def _ucr_recipes(
    baseline_selection: Path,
    minirocket_selection: Path,
) -> dict[str, dict[str, object]]:
    selection = json.loads(baseline_selection.read_text(encoding="utf-8"))
    if selection.get("status") != "complete":
        raise ValueError("confirmatory baseline selection is incomplete")
    selected = cast("dict[str, dict[str, object]]", selection["selected_trials"])
    recipes: dict[str, dict[str, object]] = {
        "efp16": {
            "learning_rate": 3.0e-3,
            "weight_decay": 1.0e-4,
            "batch_size": 32,
            "grad_clip_norm": 1.0,
            "validation_trial": 0,
        }
    }
    for family in BASELINES:
        if family == "minirocket":
            continue
        row = selected[family]
        architecture = cast("dict[str, object]", row["architecture"])
        recipes[family] = {
            "learning_rate": row["learning_rate"],
            "weight_decay": row["weight_decay"],
            "batch_size": architecture["batch_size"],
            "grad_clip_norm": architecture["grad_clip_norm"],
            "validation_trial": row["trial"],
        }
    mini = json.loads(minirocket_selection.read_text(encoding="utf-8"))["minirocket"]
    recipes["minirocket"] = {
        "learning_rate": mini["learning_rate"],
        "weight_decay": mini["weight_decay"],
        "batch_size": 64,
        "grad_clip_norm": 1.0,
        "validation_trial": mini["trial"],
    }
    return recipes


def _validate_import_payload(payload: dict[str, object], path: Path) -> None:
    required = {
        "status": "done",
        "spec": "EFP16",
        "task_kind": "ucr_validation",
        "evaluation_split": "validation",
        "official_test_accessed": False,
        "epochs": 100,
        "batch_size": 32,
        "patience": 4,
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"incompatible EFP16 import {path}: {mismatches}")
    if "params_trainable" not in payload or "validation_balanced_accuracy" not in payload:
        raise ValueError(f"incomplete EFP16 import {path}")


def _write_shards(
    root: Path,
    suite: str,
    jobs: list[FinalComparisonJob],
    shard_count: int,
) -> list[float]:
    shards: list[list[FinalComparisonJob]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for job in sorted(jobs, key=lambda item: item.estimated_seconds, reverse=True):
        index = min(range(shard_count), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += max(job.estimated_seconds, 1.0)
    shard_root = root / suite / "manifests"
    shard_root.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards):
        payload = "".join(
            json.dumps({**asdict(job), "key": job.key}, sort_keys=True) + "\n"
            for job in sorted(shard, key=lambda item: (item.estimated_seconds, item.key))
        )
        _write_text_once(shard_root / f"shard-{index:02d}.jsonl", payload)
    return loads


def _external_experiment(
    task: ExternalTask,
    benchmark: ExternalBenchmarkConfig,
) -> PACExperimentConfig:
    return PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        task.test_inputs.shape[0],
        task.sequence_length,
        raw_input_dim=task.input_dim,
        output_dim=task.output_dim,
        model_dim=32,
        modes=16,
        epochs=benchmark.epochs,
        batch_size=benchmark.batch_size,
        learning_rate=benchmark.learning_rate,
        weight_decay=benchmark.weight_decay,
        grad_clip_norm=benchmark.grad_clip_norm,
        device=benchmark.device,
    )


def _external_benchmark(
    data_root: Path,
    dataset: ExternalDatasetName,
    *,
    model: str = "efp16",
    seed: int = SEEDS[0],
    device: PACDevice = "cpu",
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    grad_clip_norm: float = 1.0,
    parameter_match_tolerance: float = PARAMETER_TOLERANCE,
) -> ExternalBenchmarkConfig:
    return ExternalBenchmarkConfig(
        data_root=data_root,
        output_root=DEFAULT_ROOT,
        datasets=(dataset,),
        models=(cast("ExternalModelFamily", "pac" if model == "efp16" else model),),
        model_dim=32,
        modes=16,
        max_baseline_width=MAX_BASELINE_WIDTH,
        parameter_match_tolerance=parameter_match_tolerance,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        patience=12,
        seeds=(seed,),
        device=device,
        latency_warmup=5,
        latency_iterations=20,
        pac_model="EFP16",
    )


def _ucr_parameter_match_preflight(
    data_root: Path,
    baseline_selection: Path,
    minirocket_selection: Path,
    *,
    allow_download: bool,
) -> dict[str, object]:
    recipes = _ucr_recipes(baseline_selection, minirocket_selection)
    rows: list[dict[str, object]] = []
    for dataset_name in UCR_DATASETS:
        dataset = ensure_ucr_train_only(
            dataset_name,
            data_root,
            allow_download=allow_download,
        )
        class_count = dataset.class_count
        config = PACExperimentConfig(
            dataset.train_inputs.shape[0],
            1,
            0,
            dataset.train_inputs.shape[1],
            raw_input_dim=dataset.train_inputs.shape[-1],
            output_dim=class_count,
            model_dim=32,
            modes=16,
            epochs=100,
            batch_size=32,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
            grad_clip_norm=1.0,
            seeds=(SEEDS[0],),
            device="cpu",
        )
        reference = build_efficient_headroom_classifier(
            "EFP16", config, class_count, objective="classification"
        )
        target = count_parameters(reference)
        for family in BASELINES:
            match = match_ucr_baseline(
                family,
                config,
                class_count,
                target_parameters=target,
                validation_trial=int(recipes[family]["validation_trial"]),
                tolerance=1.0,
            )
            effective_tolerance = ucr_parameter_tolerance(family, class_count)
            rows.append(
                {
                    "dataset": dataset_name,
                    "class_count": class_count,
                    "family": family,
                    "width": match.width,
                    "parameters": match.parameters,
                    "target_parameters": target,
                    "relative_error": match.relative_error,
                    "umbrella_tolerance": PARAMETER_TOLERANCE,
                    "effective_tolerance": effective_tolerance,
                    "predeclared_exception": effective_tolerance > PARAMETER_TOLERANCE,
                    "within_tolerance": match.relative_error <= effective_tolerance,
                }
            )
        del dataset, reference
        gc.collect()
    return _parameter_preflight_summary(rows)


def _external_parameter_match_preflight(data_root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for dataset_name in EXTERNAL_DATASETS:
        task, task_source = _load_external_parameter_task(dataset_name, data_root)
        if task.input_encoding != "continuous":
            raise ValueError(f"EFP16 final scope requires continuous inputs: {dataset_name}")
        benchmark = _external_benchmark(
            data_root,
            dataset_name,
            parameter_match_tolerance=1.0,
        )
        experiment = _external_experiment(task, benchmark)
        reference = build_efficient_headroom_classifier(
            "EFP16",
            experiment,
            task.output_dim,
            objective="regression" if task.objective == "forecasting" else "classification",
        )
        target = count_parameters(reference)
        for family_name in BASELINES:
            family = cast("ExternalModelFamily", family_name)
            match = match_external_parameter_budget(family, target, task, benchmark)
            error = match.relative_error
            effective_tolerance = external_parameter_tolerance(
                family_name,
                task.input_dim,
                task.output_dim,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "task_source": task_source,
                    "objective": task.objective,
                    "sequence_length": task.sequence_length,
                    "input_dim": task.input_dim,
                    "output_dim": task.output_dim,
                    "family": family_name,
                    "width": match.width,
                    "parameters": match.parameters,
                    "target_parameters": target,
                    "relative_error": error,
                    "match_status": match.status,
                    "reason": match.reason,
                    "umbrella_tolerance": PARAMETER_TOLERANCE,
                    "effective_tolerance": effective_tolerance,
                    "predeclared_exception": effective_tolerance > PARAMETER_TOLERANCE,
                    "within_tolerance": (
                        match.status == "matched"
                        and error is not None
                        and error <= effective_tolerance
                    ),
                }
            )
        del task, reference
        gc.collect()
    return _parameter_preflight_summary(rows)


def _load_external_parameter_task(
    dataset_name: ExternalDatasetName,
    data_root: Path,
) -> tuple[ExternalTask, str]:
    if dataset_name in {"ettm1", "ettm2", "electricity", "weather"}:
        path = data_root / "forecasting" / f"{dataset_name}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        class_names = tuple(column for column in header if column.lower() != "date")
        input_dim = len(class_names)
        if input_dim < 1:
            raise ValueError(f"forecasting CSV has no value columns: {path}")
        inputs = torch.zeros(1, 96, input_dim)
        targets = torch.zeros(1, 96, input_dim)
        task = ExternalTask(
            dataset_name,
            "forecasting",
            inputs,
            targets,
            inputs.clone(),
            targets.clone(),
            inputs.clone(),
            targets.clone(),
            96 * input_dim,
            class_names,
        )
        return task, "actual CSV header plus locked 96/96 loader shape"
    if dataset_name != "speech-commands":
        return load_external_task(dataset_name, data_root), "loaded task"
    root = data_root / "speech-commands"
    class_names = tuple(
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir()
        and not path.name.startswith("_")
        and next(path.glob("*.wav"), None) is not None
    )
    if not class_names:
        raise FileNotFoundError(f"no Speech Commands classes found under {root}")
    inputs = torch.zeros(1, 1000, 16)
    targets = torch.zeros(1, dtype=torch.long)
    task = ExternalTask(
        "speech-commands",
        "multiclass",
        inputs,
        targets,
        inputs.clone(),
        targets.clone(),
        inputs.clone(),
        targets.clone(),
        len(class_names),
        class_names,
        sample_rate_hz=16_000.0,
    )
    return task, "actual class directories plus locked loader shape"


def _parameter_preflight_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    violations = [row for row in rows if not bool(row["within_tolerance"])]
    numeric_rows = [row for row in rows if row["relative_error"] is not None]
    worst = (
        max(numeric_rows, key=lambda row: float(row["relative_error"])) if numeric_rows else None
    )
    return {
        "comparison_count": len(rows),
        "violation_count": len(violations),
        "predeclared_exception_count": sum(bool(row.get("predeclared_exception")) for row in rows),
        "max_relative_error": float(worst["relative_error"]) if worst is not None else None,
        "worst_match": worst,
        "violations": violations,
        "rows": rows,
    }


def _suite_status(
    root: Path,
    suite: str,
    expected: set[str],
) -> dict[str, object]:
    completed = _result_keys(root / suite / "completed")
    failed = _result_keys(root / suite / "failed") - completed
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed - failed),
        "done": expected <= completed and not (expected & failed),
    }


def _ucr_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for dataset in UCR_DATASETS:
        output[dataset] = {
            model: {
                "rows": len(selected),
                "mean_validation_balanced_accuracy": (
                    mean(float(row["validation_balanced_accuracy"]) for row in selected)
                    if selected
                    else None
                ),
            }
            for model in MODELS
            if (
                selected := [
                    row
                    for row in rows
                    if row.get("dataset") == dataset and row.get("model") == model
                ]
            )
        }
    return output


def _external_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for dataset in EXTERNAL_DATASETS:
        models: dict[str, object] = {}
        for model in MODELS:
            selected = [
                row for row in rows if row.get("dataset") == dataset and row.get("model") == model
            ]
            if not selected:
                continue
            metric = _external_primary_metric(str(selected[0]["objective"]), dataset)
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            models[model] = {
                "rows": len(selected),
                "primary_metric": metric,
                "mean_primary": mean(values) if values else None,
            }
        output[dataset] = models
    return output


def _external_primary_metric(objective: str, dataset: str) -> str:
    if objective == "forecasting":
        return "mse"
    if objective == "multilabel":
        return "macro_auprc" if dataset == "audioset-balanced" else "macro_auroc"
    return "accuracy"


def _job_from_payload(payload: dict[str, object]) -> FinalComparisonJob:
    active = dict(payload)
    active.pop("key", None)
    return FinalComparisonJob(**active)  # type: ignore[arg-type]


def _require_done(row: dict[str, object]) -> None:
    if row.get("status") != "done":
        raise RuntimeError(str(row.get("error", "job returned a non-done status")))


def _result_path(root: Path, job: FinalComparisonJob, *, failed: bool) -> Path:
    state = "failed" if failed else "completed"
    return root / job.suite / state / f"{_safe(job.key)}.json"


def _write_result(
    root: Path,
    job: FinalComparisonJob,
    payload: dict[str, object],
    *,
    failed: bool,
) -> None:
    path = _result_path(root, job, failed=failed)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_result_path(path, payload)


def _write_result_path(
    path: Path,
    payload: dict[str, object],
    *,
    preserve: bool = False,
) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if preserve and path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("job_key") != payload.get("job_key"):
            raise ValueError(f"refusing to replace a different result: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_once(path: Path, payload: dict[str, object]) -> None:
    _write_text_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to replace frozen artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _completed_rows(root: Path) -> list[dict[str, object]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def _result_keys(root: Path) -> set[str]:
    return {
        str(row["job_key"])
        for row in _completed_rows(root)
        if row.get("status") in {"done", "failed"}
    }


def _safe(key: str) -> str:
    return key.replace(":", "_").replace("/", "_")
