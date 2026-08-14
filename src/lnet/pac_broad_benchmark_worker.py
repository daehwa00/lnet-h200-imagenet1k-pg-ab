# ruff: noqa: BLE001, EM101, SLF001, TRY003
"""Restart-safe worker for mixed broad-benchmark GPU manifests."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path
from time import perf_counter, time_ns
from typing import TYPE_CHECKING, Final, cast
from uuid import uuid4

import torch
from torch import nn

from . import pac_physionet2012 as p12
from .pac_balanced_hpo_campaign import (
    BalancedHPOJob,
    build_balanced_sequence_model,
)
from .pac_balanced_hpo_campaign import (
    run_job as run_balanced_job,
)
from .pac_balanced_hpo_queue import OptimizerRecipe as BalancedOptimizerRecipe
from .pac_broad_benchmark_queue import BenchmarkJob
from .pac_irregular_campaign import (
    IrregularAlphabet,
    IrregularPackedSequenceBaseline,
    classification_metrics,
    normalization_moments,
)
from .pac_irregular_campaign import (
    fit as fit_irregular,
)
from .pac_irregular_campaign import (
    predict as predict_irregular,
)
from .pac_irregular_campaign import (
    result_payload as irregular_result_payload,
)
from .pac_irregular_campaign import (
    selection_scores as irregular_selection_scores,
)
from .pac_irregular_data import load_raindrop_task
from .pac_metrics import count_parameters
from .pac_types import PACDevice, PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

    from .pac_irregular_data import IrregularDatasetName, IrregularSplit, IrregularTask

MAX_ATTEMPTS: Final = 3
CLAIM_HEARTBEAT_SECONDS: Final = 30.0
CLAIM_STALE_SECONDS: Final = 15 * 60.0
STANDARD_SEQUENCE_MODELS: Final = frozenset(
    {
        "cnn1d",
        "tcn",
        "transformer",
        "mamba",
        "s4d",
        "s5",
        "lru",
        "gru",
        "lstm",
    }
)
RUNTIME_CODE_PATTERNS: Final = (
    "src/**/*",
    "csrc/**/*",
)
RUNTIME_CODE_ENTRYPOINTS: Final = (
    "scripts/run_broad_benchmark_worker.py",
    "pyproject.toml",
)


@dataclass(frozen=True, slots=True)
class BroadDataRoots:
    ucr: Path
    external: Path
    physionet2012: Path | None = None
    raindrop: Path | None = None


@dataclass(frozen=True, slots=True)
class ManifestRunSummary:
    manifest: str
    scheduled: int
    completed_before: int
    succeeded: int
    failed: int
    terminal_failed: int
    claimed_elsewhere: int


def load_manifest(path: Path) -> tuple[BenchmarkJob, ...]:
    jobs = tuple(
        BenchmarkJob.from_payload(cast("dict[str, object]", json.loads(line)))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        message = f"manifest contains duplicate logical keys: {path}"
        raise ValueError(message)
    if any(job.blockers for job in jobs):
        message = f"runnable manifest contains blocked jobs: {path}"
        raise ValueError(message)
    return jobs


def _runtime_code_sha256(project: Path) -> str:
    paths = {
        path
        for pattern in RUNTIME_CODE_PATTERNS
        for path in project.glob(pattern)
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    paths.update(project / relative for relative in RUNTIME_CODE_ENTRYPOINTS)
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(project).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_code_sha256(project: Path) -> str:
    """Hash every deployed runtime source and native artifact under ``project``."""
    return _runtime_code_sha256(project)


@cache
def code_sha256() -> str:
    return runtime_code_sha256(Path(__file__).resolve().parents[2])


def _architecture_settings(job: BenchmarkJob) -> tuple[tuple[str, int], ...]:
    if job.model == "alphabet":
        return ()
    mappings: dict[str, dict[str, tuple[tuple[str, int], ...]]] = {
        "cnn1d": {
            "d2-k3": (("depth", 2), ("kernel_size", 3)),
            "d4-k5": (("depth", 4), ("kernel_size", 5)),
        },
        "tcn": {
            "d3-k3": (("depth", 3), ("kernel_size", 3)),
            "d5-k5": (("depth", 5), ("kernel_size", 5)),
        },
        "transformer": {
            "d1-h2": (("attention_heads", 2), ("depth", 1)),
            "d2-h4": (("attention_heads", 4), ("depth", 2)),
        },
        "mamba": {
            "s16-c3": (("conv_size", 3), ("state_size", 16)),
            "s32-c4": (("conv_size", 4), ("state_size", 32)),
        },
        "s4d": {
            "d1-s16": (("depth", 1), ("state_size", 16)),
            "d3-s16": (("depth", 3), ("state_size", 16)),
        },
        "s5": {
            "d1-s16": (("depth", 1), ("state_size", 16)),
            "d2-s32": (("depth", 2), ("state_size", 32)),
        },
        "lru": {
            "d1-s16": (("depth", 1), ("state_size", 16)),
            "d2-s32": (("depth", 2), ("state_size", 32)),
        },
        "gru": {
            "d1-s16": (("depth", 1), ("state_size", 16)),
            "d2-s32": (("depth", 2), ("state_size", 32)),
        },
        "lstm": {
            "d1-s16": (("depth", 1), ("state_size", 16)),
            "d2-s32": (("depth", 2), ("state_size", 32)),
        },
    }
    try:
        return mappings[job.model][job.architecture]
    except KeyError as error:
        message = f"no balanced runner mapping for {job.model}/{job.architecture}"
        raise ValueError(message) from error


def to_balanced_job(job: BenchmarkJob) -> BalancedHPOJob:
    if job.suite not in {"regular", "forecasting", "external"}:
        message = f"{job.suite} cannot use the balanced runner"
        raise ValueError(message)
    return BalancedHPOJob(
        key=job.key,
        stage=job.stage,
        suite="ucr" if job.suite == "regular" else "external",
        dataset=job.dataset,
        model=job.model,
        candidate_id=job.candidate_id,
        recipe=BalancedOptimizerRecipe(
            job.recipe.name,
            job.recipe.learning_rate,
            job.recipe.weight_decay,
            job.recipe.effective_batch_size,
            job.recipe.grad_clip_norm,
        ),
        width=job.width,
        modes=job.modes,
        architecture=job.architecture,
        architecture_settings=_architecture_settings(job),
        split_seed=job.split_seed,
        train_seed=job.train_seed,
        epochs=job.epochs,
        evaluation_split=job.evaluation_split,
        official_test_accessed=job.official_test_accessed,
        job_class=job.job_class,
        estimated_seconds=job.estimated_seconds,
        microbatch_size=job.microbatch_size,
        gradient_accumulation_steps=job.gradient_accumulation_steps,
    )


def _build_standard_sequence_model(
    job: BenchmarkJob,
    config: PACExperimentConfig,
    *,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    balanced_job = to_balanced_job(replace(job, suite="regular"))
    return build_balanced_sequence_model(
        balanced_job,
        replace(config, raw_input_dim=input_dim, output_dim=output_dim),
        output_dim,
    )


@cache
def _p12_selection_data(
    data_root: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    cohort = p12.load_cohort(Path(data_root), "a")
    train_indices, validation_indices = p12.stratified_split(
        cohort.record_ids,
        cohort.labels,
    )
    packed = p12.pack_cohort(cohort, train_indices)
    return (
        packed[train_indices],
        cohort.labels[train_indices],
        packed[validation_indices],
        cohort.labels[validation_indices],
    )


def _p12_recipe(job: BenchmarkJob) -> p12.Recipe:
    return p12.Recipe(
        trial={"A": 1, "B": 2, "C": 3}[job.recipe.name],
        learning_rate=job.recipe.learning_rate,
        weight_decay=job.recipe.weight_decay,
        batch_size=job.recipe.effective_batch_size,
        grad_clip_norm=job.recipe.grad_clip_norm,
        max_epochs=job.epochs,
        patience=8,
    )


def _p12_config(
    job: BenchmarkJob,
    *,
    sample_count: int,
    validation_count: int,
    sequence_length: int,
) -> PACExperimentConfig:
    recipe = _p12_recipe(job)
    return PACExperimentConfig(
        sample_count=sample_count,
        validation_count=validation_count,
        test_count=4_000,
        sequence_length=sequence_length,
        raw_input_dim=p12.ALPHABET_SIGNAL_DIM,
        output_dim=2,
        model_dim=job.width,
        modes=job.modes or 16,
        epochs=job.epochs,
        batch_size=recipe.batch_size,
        learning_rate=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        grad_clip_norm=recipe.grad_clip_norm,
        device="cuda",
        precision="fp32",
    )


def _build_p12_model(job: BenchmarkJob, config: PACExperimentConfig) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.train_seed)
        if job.model == "alphabet":
            if job.modes is None:
                raise ValueError("P12 Alphabet job is missing modes")
            return p12.ClinicalAlphabet(config)
        if job.model in STANDARD_SEQUENCE_MODELS:
            return p12.ClinicalBaseline(
                _build_standard_sequence_model(
                    job,
                    config,
                    input_dim=p12.PACKED_INPUT_DIM,
                    output_dim=2,
                )
            )
        message = f"unsupported P12 model: {job.model}"
        raise ValueError(message)


def _run_p12_selection(
    job: BenchmarkJob,
    *,
    data_root: Path,
    device: str,
) -> dict[str, object]:
    if job.model not in {
        "alphabet",
        *STANDARD_SEQUENCE_MODELS,
    } or job.stage not in {"stage1", "stage2"}:
        message = f"unsupported P12 job: {job.stage}/{job.model}"
        raise ValueError(message)
    train_inputs, train_labels, validation_inputs, validation_labels = _p12_selection_data(
        str(data_root.resolve())
    )
    recipe = _p12_recipe(job)
    config = _p12_config(
        job,
        sample_count=train_inputs.shape[0],
        validation_count=validation_inputs.shape[0],
        sequence_length=train_inputs.shape[1],
    )
    model = _build_p12_model(job, config)
    fitted = p12._fit(  # pyright: ignore[reportPrivateUsage]
        model,
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        recipe,
        job.train_seed,
        device,
        enable_exact_split=False,
    )
    validation = cast("dict[str, float]", fitted["validation"])
    return {
        "schema": "alphabet.broad_benchmark.result.v1",
        "job_key": job.key,
        "cell_key": job.cell_key,
        "config_key": job.config_key,
        "status": "done",
        "selection_score": validation["auprc"],
        "selection_secondary_score": validation["auroc"],
        "best_epoch": fitted["best_epoch"],
        "train_loss": fitted["train_loss"],
        "validation": validation,
        "training_backend": "pytorch_eager_cross_host",
        "params_trainable": count_parameters(model),
        "data_scope": "physionet_set_a_train_and_fixed_validation_only",
        "official_test_accessed": False,
        "test_evaluated": False,
        **job.payload(),
    }


@cache
def _p12_final_data(data_root: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    train = p12.load_cohort(Path(data_root), "a")
    test = p12.load_cohort(Path(data_root), "c")
    train_indices = torch.arange(len(train.record_ids))
    train_packed = p12.pack_cohort(train, train_indices)
    joined = p12.RawCohort(
        train.record_ids + test.record_ids,
        torch.cat((train.values, test.values)),
        torch.cat((train.observed, test.observed)),
        torch.cat((train.delta_hours, test.delta_hours)),
        torch.cat((train.valid, test.valid)),
        torch.cat((train.labels, test.labels)),
    )
    test_packed = p12.pack_cohort(joined, train_indices)[len(train.record_ids) :]
    return train_packed, train.labels, test_packed, test.labels


def _run_p12_final(
    job: BenchmarkJob,
    *,
    data_root: Path,
    device: str,
) -> dict[str, object]:
    if job.model not in {
        "alphabet",
        *STANDARD_SEQUENCE_MODELS,
    } or job.stage != "final":
        message = f"unsupported P12 final job: {job.stage}/{job.model}"
        raise ValueError(message)
    train_inputs, train_labels, test_inputs, test_labels = _p12_final_data(
        str(data_root.resolve())
    )
    recipe = _p12_recipe(job)
    config = _p12_config(
        job,
        sample_count=train_inputs.shape[0],
        validation_count=0,
        sequence_length=train_inputs.shape[1],
    )
    model = _build_p12_model(job, config)
    fitted = p12._fit(  # pyright: ignore[reportPrivateUsage]
        model,
        train_inputs,
        train_labels,
        None,
        None,
        recipe,
        job.train_seed,
        device,
        enable_exact_split=False,
    )
    probabilities = p12._predict_probabilities(  # pyright: ignore[reportPrivateUsage]
        model,
        test_inputs,
        recipe.batch_size,
        device,
    )
    metrics = p12.binary_metrics(probabilities, test_labels, threshold=0.5)
    return {
        "schema": "alphabet.broad_benchmark.result.v1",
        "job_key": job.key,
        "cell_key": job.cell_key,
        "config_key": job.config_key,
        "status": "done",
        "train_loss": fitted["train_loss"],
        "training_backend": "pytorch_eager_cross_host",
        "official_test": asdict(metrics),
        "params_trainable": count_parameters(model),
        "data_scope": "physionet_set_a_full_train_set_c_official_test",
        "threshold_policy": "preregistered_fixed_0.5",
        "official_test_accessed": True,
        "test_evaluated": True,
        **job.payload(),
    }


def _irregular_config(job: BenchmarkJob, task: IrregularTask) -> PACExperimentConfig:
    train = task.train
    validation = task.validation
    test = task.test
    return PACExperimentConfig(
        sample_count=int(train.labels.numel()),
        validation_count=int(validation.labels.numel()),
        test_count=int(test.labels.numel()),
        sequence_length=int(train.values.shape[1]),
        raw_input_dim=int(train.values.shape[2]),
        output_dim=task.output_dim,
        model_dim=job.width,
        modes=job.modes or 16,
        epochs=job.epochs,
        batch_size=job.microbatch_size,
        learning_rate=job.recipe.learning_rate,
        weight_decay=job.recipe.weight_decay,
        grad_clip_norm=job.recipe.grad_clip_norm,
        device="cuda",
        precision="fp32",
    )


def _build_fixed_irregular_model(
    job: BenchmarkJob,
    task: IrregularTask,
    config: PACExperimentConfig,
    fit_splits: tuple[IrregularSplit, ...],
) -> nn.Module:
    feature_mean, feature_scale, static_mean, static_scale = normalization_moments(
        fit_splits
    )
    if job.model == "alphabet":
        return IrregularAlphabet(
            config,
            task.output_dim,
            feature_mean,
            feature_scale,
            static_mean,
            static_scale,
        )
    if job.model not in STANDARD_SEQUENCE_MODELS:
        message = f"unsupported fixed irregular model: {job.model}"
        raise ValueError(message)
    packed_input_dim = 2 * task.train.values.shape[-1] + 2
    core = _build_standard_sequence_model(
        job,
        config,
        input_dim=packed_input_dim,
        output_dim=task.output_dim,
    )
    return IrregularPackedSequenceBaseline(
        core,
        task.output_dim,
        feature_mean,
        feature_scale,
        static_mean,
        static_scale,
    )


def _run_raindrop_fixed(
    job: BenchmarkJob,
    *,
    data_root: Path,
    device: str,
) -> dict[str, object]:
    if job.dataset not in {"physionet-2019", "pam"}:
        message = f"unsupported fixed Raindrop dataset: {job.dataset}"
        raise ValueError(message)
    if job.model not in {
        "alphabet",
        *STANDARD_SEQUENCE_MODELS,
    }:
        message = f"unsupported fixed Raindrop model: {job.model}"
        raise ValueError(message)
    task = load_raindrop_task(
        cast("IrregularDatasetName", job.dataset),
        data_root,
        split=1,
    )
    config = _irregular_config(job, task)
    fit_splits = (
        (task.train, task.validation) if job.stage == "final" else (task.train,)
    )
    model = _build_fixed_irregular_model(job, task, config, fit_splits)
    validation = None if job.stage == "final" else task.validation
    fitted = fit_irregular(
        model,
        fit_splits,
        validation,
        epochs=job.epochs,
        batch_size=job.microbatch_size,
        learning_rate=job.recipe.learning_rate,
        weight_decay=job.recipe.weight_decay,
        grad_clip_norm=job.recipe.grad_clip_norm,
        seed=job.train_seed,
        device=device,
    )
    base: dict[str, object] = {
        "schema": "alphabet.broad_benchmark.result.v1",
        "job_key": job.key,
        "cell_key": job.cell_key,
        "config_key": job.config_key,
        "status": "done",
        "params_trainable": count_parameters(model),
        "split_sha256": task.split_sha256,
        "source_article": task.source_article,
        "split_protocol": "official_raindrop_fixed_split_1",
        "characteristic_time_scale": task.characteristic_time_scale,
        "static_covariates": (
            "additive_affine_classifier_path" if task.train.static is not None else "none"
        ),
        **irregular_result_payload(fitted),
        **job.payload(),
    }
    if job.stage != "final":
        if fitted.validation is None:
            raise RuntimeError("selection training did not return validation metrics")
        primary, secondary = irregular_selection_scores(
            fitted.validation,
            task.output_dim,
        )
        base.update(
            {
                "selection_score": primary,
                "selection_secondary_score": secondary,
                "data_scope": "official_train_and_validation_only",
                "official_test_accessed": False,
                "test_evaluated": False,
            }
        )
        return base
    probabilities = predict_irregular(
        model,
        task.test,
        batch_size=job.microbatch_size,
        device=device,
    )
    metrics = classification_metrics(probabilities, task.test.labels)
    base.update(
        {
            "official_test": asdict(metrics),
            "data_scope": "official_train_plus_validation_then_fixed_test",
            "official_test_accessed": True,
            "test_evaluated": True,
        }
    )
    return base


def run_job(
    job: BenchmarkJob,
    *,
    device: PACDevice,
    data_roots: BroadDataRoots,
) -> dict[str, object]:
    if job.blockers:
        message = f"refusing to execute blocked job {job.key}: {job.blockers}"
        raise RuntimeError(message)
    if job.stage != "final" and (
        job.evaluation_split != "validation" or job.official_test_accessed
    ):
        message = f"{job.stage} violates the sealed validation-only contract"
        raise RuntimeError(message)
    if job.suite in {"regular", "forecasting", "external"}:
        legacy = to_balanced_job(job)
        row = run_balanced_job(
            legacy,
            device=device,
            ucr_data_root=data_roots.ucr,
            external_data_root=data_roots.external,
        )
        row.update(job.payload())
        row.update(
            {
                "schema": "alphabet.broad_benchmark.result.v1",
                "job_key": job.key,
                "cell_key": job.cell_key,
                "config_key": job.config_key,
                "code_sha256": code_sha256(),
            }
        )
        return row
    if job.suite == "irregular" and job.dataset == "physionet-2012":
        if data_roots.physionet2012 is None:
            raise FileNotFoundError("PhysioNet-2012 data root was not configured")
        runner = _run_p12_final if job.stage == "final" else _run_p12_selection
        row = runner(job, data_root=data_roots.physionet2012, device=str(device))
        row["code_sha256"] = code_sha256()
        return row
    if job.suite == "irregular" and job.dataset in {"physionet-2019", "pam"}:
        if data_roots.raindrop is None:
            raise FileNotFoundError("fixed Raindrop data root was not configured")
        row = _run_raindrop_fixed(job, data_root=data_roots.raindrop, device=str(device))
        row["code_sha256"] = code_sha256()
        return row
    message = f"no broad execution backend for {job.suite}/{job.dataset}/{job.model}"
    raise ValueError(message)


def _key_token(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def result_path(root: Path, job: BenchmarkJob, *, failed: bool = False) -> Path:
    bucket = "failed" if failed else "completed"
    return root / job.stage / bucket / f"{_key_token(job.key)}.json"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class _RenewableClaim:
    def __init__(
        self,
        path: Path,
        job_key: str,
        *,
        stale_seconds: float,
    ) -> None:
        self.path = path
        self.job_key = job_key
        self.stale_seconds = stale_seconds
        # Claim ownership must be unique, but it must not disclose host or
        # process identity in the claim artifact.
        self.token = uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                try:
                    age = max(0.0, time_ns() / 1e9 - self.path.stat().st_mtime)
                except FileNotFoundError:
                    continue
                if age <= self.stale_seconds:
                    return False
                self.path.unlink(missing_ok=True)
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"job_key": self.job_key, "owner": self.token},
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                self._thread = threading.Thread(
                    target=self._heartbeat,
                    daemon=True,
                )
                self._thread.start()
                return True
        return False

    def _heartbeat(self) -> None:
        while not self._stop.wait(CLAIM_HEARTBEAT_SECONDS):
            try:
                os.utime(self.path, None)
            except FileNotFoundError:
                return

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if payload.get("owner") == self.token:
            self.path.unlink(missing_ok=True)


def _attempt_directory(root: Path, job: BenchmarkJob) -> Path:
    return root / job.stage / "attempts" / _key_token(job.key)


def _failed_attempt_count(root: Path, job: BenchmarkJob) -> int:
    directory = _attempt_directory(root, job)
    count = 0
    for path in sorted(directory.glob("*.json")) if directory.exists() else ():
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        count += row.get("status") == "failed"
    return count


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _require_valid_result(row: dict[str, object], job: BenchmarkJob) -> None:
    if row.get("status") != "done" or row.get("job_key") != job.key:
        message = f"runner returned an invalid result for {job.key}"
        raise RuntimeError(message)


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: PACDevice,
    data_roots: BroadDataRoots,
    max_attempts: int = MAX_ATTEMPTS,
    claim_stale_seconds: float = CLAIM_STALE_SECONDS,
    runner: Callable[..., dict[str, object]] = run_job,
) -> ManifestRunSummary:
    jobs = load_manifest(manifest)
    counts = {
        "completed_before": 0,
        "succeeded": 0,
        "failed": 0,
        "terminal_failed": 0,
        "claimed_elsewhere": 0,
    }
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for job in jobs:
        completed = result_path(root, job)
        if completed.exists():
            counts["completed_before"] += 1
            continue
        if _failed_attempt_count(root, job) >= max_attempts:
            counts["terminal_failed"] += 1
            continue
        claim = _RenewableClaim(
            root / job.stage / "claims" / f"{_key_token(job.key)}.lock",
            job.key,
            stale_seconds=claim_stale_seconds,
        )
        if not claim.acquire():
            counts["claimed_elsewhere"] += 1
            continue
        attempt_id = f"{time_ns()}-{uuid4().hex}"
        attempt_path = _attempt_directory(root, job) / f"{attempt_id}.json"
        started = perf_counter()
        base_attempt = {
            "schema": "alphabet.broad_benchmark.attempt.v1",
            "attempt_id": attempt_id,
            "job_key": job.key,
            "manifest_sha256": manifest_sha256,
            "code_sha256": code_sha256(),
            "environment": _environment(),
            "immutable_job": job.payload(),
        }
        _atomic_json(attempt_path, {**base_attempt, "status": "started"})
        try:
            row = runner(job, device=device, data_roots=data_roots)
            _require_valid_result(row, job)
        except Exception as error:
            failure = {
                **base_attempt,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": perf_counter() - started,
            }
            _atomic_json(result_path(root, job, failed=True), failure)
            _atomic_json(attempt_path, failure)
            counts["failed"] += 1
        else:
            row["manifest_sha256"] = manifest_sha256
            row["code_sha256"] = code_sha256()
            _atomic_json(completed, row)
            _atomic_json(
                attempt_path,
                {
                    **base_attempt,
                    "status": "succeeded",
                    "elapsed_seconds": perf_counter() - started,
                    "result_path": str(completed),
                },
            )
            result_path(root, job, failed=True).unlink(missing_ok=True)
            counts["succeeded"] += 1
        finally:
            claim.release()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return ManifestRunSummary(
        manifest=str(manifest),
        scheduled=len(jobs),
        completed_before=counts["completed_before"],
        succeeded=counts["succeeded"],
        failed=counts["failed"],
        terminal_failed=counts["terminal_failed"],
        claimed_elsewhere=counts["claimed_elsewhere"],
    )


__all__: Final = [
    "RUNTIME_CODE_ENTRYPOINTS",
    "RUNTIME_CODE_PATTERNS",
    "BroadDataRoots",
    "ManifestRunSummary",
    "code_sha256",
    "load_manifest",
    "result_path",
    "run_job",
    "run_manifest",
    "runtime_code_sha256",
    "to_balanced_job",
]
