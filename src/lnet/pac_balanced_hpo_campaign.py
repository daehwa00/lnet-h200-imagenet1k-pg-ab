"""Restart-safe execution and stage selection for the balanced 27-task HPO.

The immutable queue contract lives in :mod:`pac_balanced_hpo_queue`.  This
module adds the execution layer without changing the scientific design:

* one logical key owns one atomic result file;
* a renewable claim prevents concurrent duplicate execution on one filesystem;
* failed jobs are retried without replaying completed jobs;
* Stage 2 and final queues are emitted only after their input stage is complete;
* official test data is requested only by final-stage jobs.

The data loading and training loops intentionally reuse the mature fairness
campaign implementation.  Only model construction is injected so the balanced
architecture grid and the final radial-log ALPHABET are honored exactly.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median
from time import perf_counter, time_ns
from typing import TYPE_CHECKING, Final, Literal, cast

from .pac_campaign_utils import write_once
from uuid import uuid4

import torch
from torch import nn

from . import pac_baseline_fairness_maximal as fairness
from .alphabet import (
    Alphabet,
)
from .pac_additional_ssm_baselines import DiagonalSSMClassifier
from .pac_balanced_hpo_queue import (
    CONFIRMATION_SEEDS,
    DEFAULT_ROOT,
    FINAL_SEEDS,
    SEARCH_SEED,
    TOP_K,
    OptimizerRecipe,
    expected_counts,
)
from .pac_confirmatory_baselines import (
    BidirectionalRecurrentClassifier,
    MambaClassifier,
    S4DClassifier,
    TCNClassifier,
    _TrialCNN1DClassifier,  # pyright: ignore[reportPrivateUsage]
    _TrialTransformerClassifier,  # pyright: ignore[reportPrivateUsage]
)
from .pac_external_benchmarks import (
    _NativeTemporalMetadataAdapter,  # pyright: ignore[reportPrivateUsage]
    _PackedTemporalMetadataAdapter,  # pyright: ignore[reportPrivateUsage]
    _TokenEmbeddingClassifier,  # pyright: ignore[reportPrivateUsage]
    _TokenPairClassifier,  # pyright: ignore[reportPrivateUsage]
)
from .pac_types import PACDevice, PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from .pac_external_tasks import ExternalSelectionTask, ExternalTask

Stage = Literal["stage1", "stage2", "final"]
Suite = Literal["ucr", "external"]
JobClass = Literal["short", "medium", "long"]
JSONScalar = str | int | float

UCR_DATA_ROOT: Final = Path(".omx/data/ucr")
EXTERNAL_DATA_ROOT: Final = Path("data/external")
CLAIM_HEARTBEAT_SECONDS: Final = 30.0
CLAIM_STALE_SECONDS: Final = 15 * 60.0
MAX_ATTEMPTS: Final = 3


@dataclass(frozen=True, slots=True)
class BalancedHPOJob:
    key: str
    stage: Stage
    suite: Suite
    dataset: str
    model: str
    candidate_id: str
    recipe: OptimizerRecipe
    width: int
    modes: int | None
    architecture: str
    architecture_settings: tuple[tuple[str, int], ...]
    split_seed: int
    train_seed: int
    epochs: int
    evaluation_split: Literal["validation", "test"]
    official_test_accessed: bool
    job_class: JobClass
    estimated_seconds: float
    microbatch_size: int = 64
    gradient_accumulation_steps: int = 1

    @property
    def cell_key(self) -> str:
        return f"{self.suite}:{self.dataset}:{self.model}"

    @property
    def config_key(self) -> str:
        return self.candidate_id

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["architecture_settings"] = dict(self.architecture_settings)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> BalancedHPOJob:
        recipe_payload = cast("dict[str, object]", payload["recipe"])
        settings_payload = cast("dict[str, object]", payload.get("architecture_settings", {}))
        return cls(
            key=str(payload["key"]),
            stage=cast("Stage", payload["stage"]),
            suite=cast("Suite", payload["suite"]),
            dataset=str(payload["dataset"]),
            model=str(payload["model"]),
            candidate_id=str(payload["candidate_id"]),
            recipe=OptimizerRecipe(
                cast("Literal['A', 'B', 'C']", recipe_payload["name"]),
                _as_float(recipe_payload["learning_rate"]),
                _as_float(recipe_payload["weight_decay"]),
                _as_int(recipe_payload["batch_size"]),
                _as_float(recipe_payload["grad_clip_norm"]),
            ),
            width=_as_int(payload["width"]),
            modes=None if payload.get("modes") is None else _as_int(payload["modes"]),
            architecture=str(payload["architecture"]),
            architecture_settings=tuple(
                sorted((str(key), _as_int(value)) for key, value in settings_payload.items())
            ),
            split_seed=_as_int(payload["split_seed"]),
            train_seed=_as_int(payload["train_seed"]),
            epochs=_as_int(payload["epochs"]),
            evaluation_split=cast(
                "Literal['validation', 'test']",
                payload["evaluation_split"],
            ),
            official_test_accessed=bool(payload["official_test_accessed"]),
            job_class=cast("JobClass", payload["job_class"]),
            estimated_seconds=_as_float(payload["estimated_seconds"]),
            microbatch_size=_as_int(
                payload.get("microbatch_size", recipe_payload["batch_size"])
            ),
            gradient_accumulation_steps=_as_int(
                payload.get("gradient_accumulation_steps", 1)
            ),
        )


def _as_int(value: object) -> int:
    return int(cast("JSONScalar", value))


def _as_float(value: object) -> float:
    return float(cast("JSONScalar", value))


def load_manifest(path: Path) -> list[BalancedHPOJob]:
    jobs = [
        BalancedHPOJob.from_payload(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        message = f"manifest contains duplicate logical keys: {path}"
        raise ValueError(message)
    return jobs


def code_sha256() -> str:
    """Hash execution-critical Python sources for cross-host preflight."""
    project = Path(__file__).resolve().parents[2]
    relative_paths = (
        "src/lnet/pac_balanced_hpo_queue.py",
        "src/lnet/pac_balanced_hpo_campaign.py",
        "src/lnet/pac_baseline_fairness_maximal.py",
        "src/lnet/pac_confirmatory_baselines.py",
        "src/lnet/pac_additional_ssm_baselines.py",
        "src/lnet/alphabet.py",
        "src/lnet/pac_training.py",
        "src/lnet/pac_external_benchmarks.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = project / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _settings(job: BalancedHPOJob) -> dict[str, int]:
    return dict(job.architecture_settings)


def _build_model(  # noqa: PLR0911 - explicit audited family dispatch
    job: BalancedHPOJob,
    config: PACExperimentConfig,
    output_dim: int,
    *,
    objective: Literal["classification", "regression"],
) -> nn.Module:
    settings = _settings(job)
    if job.model == "alphabet":
        return Alphabet(
            config,
            output_dim,
            objective=objective,
            dwconv_dilation=settings.get("dwconv_dilation", 4),
        )

    input_dim = config.raw_input_dim
    if job.model == "cnn1d":
        return _TrialCNN1DClassifier(
            job.width,
            output_dim,
            depth=settings["depth"],
            kernel_size=settings["kernel_size"],
            input_dim=input_dim,
        )
    if job.model == "tcn":
        return TCNClassifier(
            job.width,
            output_dim,
            depth=settings["depth"],
            kernel_size=settings["kernel_size"],
            input_dim=input_dim,
        )
    if job.model == "transformer":
        return _TrialTransformerClassifier(
            job.width,
            output_dim,
            depth=settings["depth"],
            requested_heads=settings["attention_heads"],
            input_dim=input_dim,
        )
    if job.model == "mamba":
        return MambaClassifier(
            job.width,
            output_dim,
            state_size=settings["state_size"],
            conv_kernel=settings["conv_size"],
            input_dim=input_dim,
        )
    if job.model == "s4d":
        return S4DClassifier(
            job.width,
            output_dim,
            depth=settings["depth"],
            state_size=settings["state_size"],
            input_dim=input_dim,
        )
    if job.model in {"s5", "lru"}:
        return DiagonalSSMClassifier(
            job.model,
            job.width,
            output_dim,
            depth=settings["depth"],
            state_size=settings["state_size"],
            input_dim=input_dim,
        )
    if job.model in {"gru", "lstm"}:
        return BidirectionalRecurrentClassifier(
            cast("Literal['gru', 'lstm']", job.model),
            job.width,
            output_dim,
            depth=settings["depth"],
            state_size=settings["state_size"],
            input_dim=input_dim,
        )
    message = f"unsupported balanced HPO model: {job.model}"
    raise ValueError(message)


def build_balanced_sequence_model(
    job: BalancedHPOJob,
    config: PACExperimentConfig,
    output_dim: int,
    *,
    objective: Literal["classification", "regression"] = "classification",
) -> nn.Module:
    """Build one model from the frozen 10-model balanced-HPO architecture contract."""
    return _build_model(job, config, output_dim, objective=objective)


def _build_external_model(
    job: BalancedHPOJob,
    config: PACExperimentConfig,
    task: ExternalTask | ExternalSelectionTask,
) -> nn.Module:
    """Build the same searched core behind the task's sealed input adapter."""
    objective = "regression" if task.objective == "forecasting" else "classification"
    if task.input_encoding == "continuous":
        if not task.has_temporal_metadata:
            return _build_model(job, config, task.output_dim, objective=objective)
        if job.model == "alphabet":
            core = _build_model(job, config, task.output_dim, objective=objective)
            if not (
                getattr(core, "supports_observation_mask", False)
                and getattr(core, "supports_time_delta", False)
            ):
                message = "Alphabet does not expose the required temporal-metadata contract"
                raise RuntimeError(message)
            return _NativeTemporalMetadataAdapter(core)
        packed_config = replace(
            config,
            raw_input_dim=2 * task.input_dim + 2,
        )
        core = _build_model(
            job,
            packed_config,
            task.output_dim,
            objective=objective,
        )
        return _PackedTemporalMetadataAdapter(core, task.input_dim)

    core_output_dim = job.width if task.input_encoding == "token_pair" else task.output_dim
    core_config = replace(
        config,
        raw_input_dim=job.width,
        output_dim=core_output_dim,
    )
    core = _build_model(
        job,
        core_config,
        core_output_dim,
        objective="classification",
    )
    if task.vocab_size is None:
        message = f"{task.name} token task is missing vocab_size"
        raise ValueError(message)
    if task.input_encoding == "tokens":
        return _TokenEmbeddingClassifier(task.vocab_size, job.width, core)
    return _TokenPairClassifier(task.vocab_size, job.width, core, task.output_dim)


def build_model_for_preflight(
    job: BalancedHPOJob,
    *,
    input_dim: int = 1,
    output_dim: int = 5,
) -> nn.Module:
    """Construct one queued model without loading a dataset or touching a GPU."""
    config = PACExperimentConfig(
        8,
        4,
        0,
        32,
        raw_input_dim=input_dim,
        output_dim=output_dim,
        model_dim=job.width,
        modes=job.modes or 16,
        epochs=1,
        batch_size=min(job.recipe.batch_size, 8),
        learning_rate=job.recipe.learning_rate,
        weight_decay=job.recipe.weight_decay,
        grad_clip_norm=job.recipe.grad_clip_norm,
        seeds=(job.train_seed,),
        device="cpu",
    )
    return _build_model(job, config, output_dim, objective="classification")


def preflight_manifest(
    manifest: Path,
    *,
    device: PACDevice,
    ucr_data_root: Path = UCR_DATA_ROOT,
    external_data_root: Path = EXTERNAL_DATA_ROOT,
) -> dict[str, object]:
    """Fail fast on source, environment, model imports, data roots, and CUDA."""
    if not ucr_data_root.exists():
        message = f"UCR data root does not exist: {ucr_data_root}"
        raise FileNotFoundError(message)
    if not external_data_root.exists():
        message = f"external data root does not exist: {external_data_root}"
        raise FileNotFoundError(message)
    if device == "cuda" and not torch.cuda.is_available():
        message = "CUDA was requested but torch.cuda.is_available() is false"
        raise RuntimeError(message)
    jobs = load_manifest(manifest)
    representatives: dict[str, BalancedHPOJob] = {}
    for job in jobs:
        representatives.setdefault(job.model, job)
    built: dict[str, int] = {}
    smoke_device = (
        torch.device("cuda")
        if device == "cuda"
        else torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(0)
    for model_name, job in sorted(representatives.items()):
        model = build_model_for_preflight(job).to(device=smoke_device).train()
        built[model_name] = sum(parameter.numel() for parameter in model.parameters())
        inputs = torch.randn(2, 32, 1, device=smoke_device)
        outputs = model(inputs)
        if outputs.shape != (2, 5) or not torch.isfinite(outputs).all():
            message = f"{model_name} failed preflight forward: shape={tuple(outputs.shape)}"
            raise RuntimeError(message)
        outputs.float().square().mean().backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            message = f"{model_name} produced non-finite preflight gradients"
            raise RuntimeError(message)
        del model
        if smoke_device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "schema": "pac.balanced_hpo_preflight.v1",
        "code_sha256": code_sha256(),
        "manifest": str(manifest),
        "models_constructed": built,
        "forward_backward_smoke_device": str(smoke_device),
        "ucr_data_root": str(ucr_data_root),
        "external_data_root": str(external_data_root),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if device == "cuda" and torch.cuda.is_available()
            else None
        ),
        "environment": _environment(),
        "ok": True,
    }


def _fairness_job(job: BalancedHPOJob) -> fairness.FairnessJob:
    return fairness.FairnessJob(
        stage=job.stage,
        suite=job.suite,
        dataset=job.dataset,
        model=job.model,
        width_tier=job.width,
        width=job.width,
        trial=1,
        split_seed=job.split_seed,
        train_seed=job.train_seed,
        epochs=job.epochs,
        batch_size=job.microbatch_size,
        learning_rate=job.recipe.learning_rate,
        weight_decay=job.recipe.weight_decay,
        grad_clip_norm=job.recipe.grad_clip_norm,
        evaluation_split=job.evaluation_split,
        estimated_seconds=job.estimated_seconds,
        modes=job.modes or 16,
        gradient_accumulation_steps=job.gradient_accumulation_steps,
    )


def run_job(
    job: BalancedHPOJob,
    *,
    device: PACDevice,
    ucr_data_root: Path = UCR_DATA_ROOT,
    external_data_root: Path = EXTERNAL_DATA_ROOT,
) -> dict[str, object]:
    if (
        job.microbatch_size < 1
        or job.gradient_accumulation_steps < 1
        or job.microbatch_size * job.gradient_accumulation_steps
        != job.recipe.batch_size
    ):
        message = (
            f"{job.key} has invalid effective batch: microbatch={job.microbatch_size}, "
            f"accumulation={job.gradient_accumulation_steps}, "
            f"recipe_batch={job.recipe.batch_size}"
        )
        raise ValueError(message)
    if job.stage != "final" and job.evaluation_split != "validation":
        message = f"{job.stage} is forbidden from accessing a non-validation split"
        raise RuntimeError(message)
    if job.official_test_accessed != (job.stage == "final"):
        message = f"inconsistent official-test flag for {job.key}"
        raise RuntimeError(message)

    source_job = _fairness_job(job)

    def build_ucr(
        _source: fairness.FairnessJob,
        config: PACExperimentConfig,
        class_count: int,
    ) -> nn.Module:
        return _build_model(job, config, class_count, objective="classification")

    def build_external(
        _source: fairness.FairnessJob,
        config: PACExperimentConfig,
        task: ExternalTask | ExternalSelectionTask,
    ) -> nn.Module:
        return _build_external_model(job, config, task)

    row = fairness.run_job(
        source_job,
        device=device,
        ucr_data_root=ucr_data_root,
        external_data_root=external_data_root,
        ucr_model_builder=build_ucr,
        external_model_builder=build_external,
        use_validated_baseline_cuda_graph=True,
    )
    row.update(job.payload())
    row.update(
        {
            "schema": "pac.balanced_hpo_result.v1",
            "job_key": job.key,
            "cell_key": job.cell_key,
            "config_key": job.config_key,
            "code_sha256": code_sha256(),
            "status": "done",
            "test_evaluated": job.stage == "final",
            "official_test_accessed": job.stage == "final",
        }
    )
    return row


def _key_token(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def result_path(root: Path, job: BalancedHPOJob, *, failed: bool = False) -> Path:
    bucket = "failed" if failed else "completed"
    return root / job.stage / bucket / f"{_key_token(job.key)}.json"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
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
        stale_seconds: float = CLAIM_STALE_SECONDS,
        heartbeat_seconds: float = CLAIM_HEARTBEAT_SECONDS,
    ) -> None:
        self.path = path
        self.job_key = job_key
        self.stale_seconds = stale_seconds
        self.heartbeat_seconds = heartbeat_seconds
        # Claim ownership must be unique, but it must not disclose host or
        # process identity in the claim artifact.
        self.token = uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "job_key": self.job_key,
                "owner": self.token,
                "started_ns": time_ns(),
            },
            sort_keys=True,
        )
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                try:
                    # stat().st_mtime and time_ns are both wall-clock based.
                    age = max(0.0, time_ns() / 1e9 - self.path.stat().st_mtime)
                except FileNotFoundError:
                    continue
                if age <= self.stale_seconds:
                    return False
                stale = self.path.with_name(f"{self.path.name}.stale.{uuid4().hex}")
                try:
                    self.path.replace(stale)
                    stale.unlink(missing_ok=True)
                except FileNotFoundError:
                    continue
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(payload + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._thread = threading.Thread(
                    target=self._heartbeat,
                    name=f"hpo-claim-{_key_token(self.job_key)[:8]}",
                    daemon=True,
                )
                self._thread.start()
                return True
        return False

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                os.utime(self.path, None)
            except FileNotFoundError:
                return

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_seconds))
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if payload.get("owner") == self.token:
            self.path.unlink(missing_ok=True)


def _attempt_paths(root: Path, job: BalancedHPOJob) -> list[Path]:
    directory = root / job.stage / "attempts" / _key_token(job.key)
    return sorted(directory.glob("*.json")) if directory.exists() else []


def _failed_attempt_count(root: Path, job: BalancedHPOJob) -> int:
    count = 0
    for path in _attempt_paths(root, job):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        count += payload.get("status") == "failed"
    return count


def _attempt_path(root: Path, job: BalancedHPOJob, attempt_id: str) -> Path:
    return root / job.stage / "attempts" / _key_token(job.key) / f"{attempt_id}.json"


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


@dataclass(frozen=True, slots=True)
class ManifestRunSummary:
    manifest: str
    scheduled: int
    completed_before: int
    succeeded: int
    failed: int
    terminal_failed: int
    claimed_elsewhere: int


def _require_valid_result(row: dict[str, object], job: BalancedHPOJob) -> None:
    if row.get("status") != "done" or row.get("job_key") != job.key:
        message = f"runner returned an invalid result for {job.key}"
        raise RuntimeError(message)


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: PACDevice = "cuda",
    ucr_data_root: Path = UCR_DATA_ROOT,
    external_data_root: Path = EXTERNAL_DATA_ROOT,
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
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
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
        attempt_file = _attempt_path(root, job, attempt_id)
        started = perf_counter()
        _atomic_write_json(
            attempt_file,
            {
                "schema": "pac.balanced_hpo_attempt.v1",
                "attempt_id": attempt_id,
                "job_key": job.key,
                "status": "started",
                "manifest_sha256": manifest_hash,
                "code_sha256": code_sha256(),
                "environment": _environment(),
                "immutable_job": job.payload(),
            },
        )
        try:
            row = runner(
                job,
                device=device,
                ucr_data_root=ucr_data_root,
                external_data_root=external_data_root,
            )
            _require_valid_result(row, job)
        except Exception as error:  # noqa: BLE001 - every queue failure must be preserved
            failure = {
                "schema": "pac.balanced_hpo_failure.v1",
                "job_key": job.key,
                **job.payload(),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "manifest_sha256": manifest_hash,
                "code_sha256": code_sha256(),
                "environment": _environment(),
            }
            _atomic_write_json(result_path(root, job, failed=True), failure)
            _atomic_write_json(
                attempt_file,
                {
                    "schema": "pac.balanced_hpo_attempt.v1",
                    "attempt_id": attempt_id,
                    "job_key": job.key,
                    "status": "failed",
                    "elapsed_seconds": perf_counter() - started,
                    "failure_path": str(result_path(root, job, failed=True)),
                    "error": failure["error"],
                    "manifest_sha256": manifest_hash,
                    "code_sha256": code_sha256(),
                    "environment": _environment(),
                    "immutable_job": job.payload(),
                },
            )
            counts["failed"] += 1
        else:
            row["manifest_sha256"] = manifest_hash
            _atomic_write_json(completed, row)
            _atomic_write_json(
                attempt_file,
                {
                    "schema": "pac.balanced_hpo_attempt.v1",
                    "attempt_id": attempt_id,
                    "job_key": job.key,
                    "status": "succeeded",
                    "elapsed_seconds": perf_counter() - started,
                    "result_path": str(completed),
                    "manifest_sha256": manifest_hash,
                    "code_sha256": code_sha256(),
                    "environment": _environment(),
                    "immutable_job": job.payload(),
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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _completed_rows(root: Path, stage: Stage) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / stage / "completed").glob("*.json")):
        try:
            row = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("status") == "done":
            rows.append(row)
    return rows


def _expected_jobs(root: Path, stage: Stage) -> list[BalancedHPOJob]:
    master = root / stage / "master.jsonl"
    if not master.exists():
        return []
    return [BalancedHPOJob.from_payload(row) for row in _read_jsonl(master)]


def _require_complete_stage(root: Path, stage: Stage) -> list[dict[str, object]]:
    expected = _expected_jobs(root, stage)
    expected_keys = {job.key for job in expected}
    rows = _completed_rows(root, stage)
    rows_by_key = {str(row["job_key"]): row for row in rows}
    missing = expected_keys - rows_by_key.keys()
    unexpected = rows_by_key.keys() - expected_keys
    if missing or unexpected:
        message = (
            f"{stage} is incomplete: expected={len(expected_keys)}, "
            f"complete={len(expected_keys) - len(missing)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        raise RuntimeError(message)
    return [rows_by_key[job.key] for job in expected]


def _job_key(job: BalancedHPOJob) -> str:
    return (
        f"balanced-hpo:{job.stage}:{job.suite}:{job.dataset}:{job.model}:"
        f"{job.candidate_id}:split{job.split_seed}:seed{job.train_seed}"
    )


def _job_from_result(row: dict[str, object]) -> BalancedHPOJob:
    return BalancedHPOJob.from_payload(row)


def _write_stage_queue(root: Path, stage: Stage, jobs: list[BalancedHPOJob]) -> None:
    expected = expected_counts()[stage]
    if len(jobs) != expected:
        message = f"{stage} has {len(jobs)} jobs; expected {expected}"
        raise RuntimeError(message)
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        message = f"{stage} contains duplicate logical keys"
        raise RuntimeError(message)
    ordered = sorted(jobs, key=lambda job: (-job.estimated_seconds, job.key))
    write_once(
        root / stage / "master.jsonl",
        "".join(json.dumps(job.payload(), sort_keys=True) + "\n" for job in ordered),
    )
    for job_class in ("short", "medium", "long"):
        class_jobs = [job for job in ordered if job.job_class == job_class]
        write_once(
            root / stage / "queues" / f"{job_class}.jsonl",
            "".join(json.dumps(job.payload(), sort_keys=True) + "\n" for job in class_jobs),
        )


def _rank_row(row: dict[str, object]) -> tuple[float, int, str]:
    score = row.get("selection_score")
    if score is None:
        message = f"selection score is missing for {row.get('job_key')}"
        raise RuntimeError(message)
    return (
        -_as_float(score),
        _as_int(row.get("params_trainable", 0)),
        str(row["config_key"]),
    )


def select_stage1(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    rows = _require_complete_stage(root, "stage1")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_key"]), []).append(row)
    selected: dict[str, list[str]] = {}
    jobs: list[BalancedHPOJob] = []
    for cell_key, cell_rows in sorted(grouped.items()):
        if len(cell_rows) != 18:
            message = f"{cell_key} has {len(cell_rows)} rows; expected 18"
            raise RuntimeError(message)
        top = sorted(cell_rows, key=_rank_row)[:TOP_K]
        selected[cell_key] = [str(row["config_key"]) for row in top]
        for row in top:
            base = _job_from_result(row)
            for seed in CONFIRMATION_SEEDS:
                candidate = replace(
                    base,
                    key="",
                    stage="stage2",
                    split_seed=seed,
                    train_seed=seed,
                    evaluation_split="validation",
                    official_test_accessed=False,
                )
                jobs.append(replace(candidate, key=_job_key(candidate)))
    _write_stage_queue(root, "stage2", jobs)
    payload: dict[str, object] = {
        "schema": "pac.balanced_hpo_stage1_selection.v1",
        "source_rows": len(rows),
        "cells": len(grouped),
        "top_k": TOP_K,
        "ranking": "validation score descending, parameters ascending, config key ascending",
        "selected": selected,
        "stage2_jobs": len(jobs),
        "official_test_accessed": False,
    }
    write_once(
        root / "stage1" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def select_stage2(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    stage1 = _require_complete_stage(root, "stage1")
    stage2 = _require_complete_stage(root, "stage2")
    stage1_selection = cast(
        "dict[str, object]",
        json.loads((root / "stage1" / "selection.json").read_text(encoding="utf-8")),
    )
    selected_configs = cast("dict[str, list[str]]", stage1_selection["selected"])
    combined: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in (*stage1, *stage2):
        combined.setdefault(
            (str(row["cell_key"]), str(row["config_key"])),
            [],
        ).append(row)

    selected: dict[str, dict[str, object]] = {}
    jobs: list[BalancedHPOJob] = []
    expected_selection_seeds = {SEARCH_SEED, *CONFIRMATION_SEEDS}
    for cell_key, config_keys in sorted(selected_configs.items()):
        candidates: list[tuple[float, str, list[dict[str, object]]]] = []
        for config_key in config_keys:
            config_rows = combined[(cell_key, config_key)]
            seeds = {_as_int(row["train_seed"]) for row in config_rows}
            if len(config_rows) != 3 or seeds != expected_selection_seeds:
                message = (
                    f"{cell_key}/{config_key} has seeds {sorted(seeds)}; "
                    f"expected {sorted(expected_selection_seeds)}"
                )
                raise RuntimeError(message)
            candidates.append(
                (
                    mean(_as_float(row["selection_score"]) for row in config_rows),
                    config_key,
                    config_rows,
                )
            )
        score, config_key, config_rows = min(
            candidates,
            key=lambda item: (-item[0], item[1]),
        )
        base = _job_from_result(config_rows[0])
        best_epochs = [
            _as_int(row["best_epoch"]) for row in config_rows if row.get("best_epoch") is not None
        ]
        refit_epochs = (
            max(1, round(median(best_epochs)))
            if base.suite == "ucr" and best_epochs
            else base.epochs
        )
        selected[cell_key] = {
            "config_key": config_key,
            "mean_validation_score": score,
            "selection_seeds": sorted(expected_selection_seeds),
            "width": base.width,
            "modes": base.modes,
            "architecture": base.architecture,
            "architecture_settings": dict(base.architecture_settings),
            "recipe": asdict(base.recipe),
            "final_epochs": refit_epochs,
        }
        for seed in FINAL_SEEDS:
            candidate = replace(
                base,
                key="",
                stage="final",
                split_seed=seed,
                train_seed=seed,
                epochs=refit_epochs,
                evaluation_split="test",
                official_test_accessed=True,
            )
            jobs.append(replace(candidate, key=_job_key(candidate)))
    _write_stage_queue(root, "final", jobs)
    payload: dict[str, object] = {
        "schema": "pac.balanced_hpo_stage2_selection.v1",
        "source_stage1_rows": len(stage1),
        "source_stage2_rows": len(stage2),
        "cells": len(selected),
        "selected": selected,
        "final_jobs": len(jobs),
        "configuration_frozen_before_test": True,
        "official_test_accessed_during_selection": False,
    }
    write_once(
        root / "stage2" / "selection.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def campaign_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "pac.balanced_hpo_status.v1",
        "code_sha256": code_sha256(),
    }
    for stage in cast("tuple[Stage, ...]", ("stage1", "stage2", "final")):
        jobs = _expected_jobs(root, stage)
        expected_keys = {job.key for job in jobs}
        completed_rows = _completed_rows(root, stage)
        completed_keys = {str(row["job_key"]) for row in completed_rows}
        terminal_failed = {
            job.key
            for job in jobs
            if not result_path(root, job).exists()
            and _failed_attempt_count(root, job) >= MAX_ATTEMPTS
        }
        retryable_failed = {
            job.key
            for job in jobs
            if result_path(root, job, failed=True).exists() and job.key not in terminal_failed
        }
        unexpected = completed_keys - expected_keys
        remaining = expected_keys - completed_keys - terminal_failed
        payload[stage] = {
            "expected": len(expected_keys),
            "completed": len(completed_keys & expected_keys),
            "remaining": len(remaining),
            "retryable_failed": len(retryable_failed),
            "terminal_failed": len(terminal_failed),
            "unexpected_completed": len(unexpected),
            "done": bool(jobs) and not remaining and not unexpected,
        }
    return payload


def audit_campaign(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    status = campaign_status(root)
    violations: list[str] = []
    seen: set[str] = set()
    for stage in cast("tuple[Stage, ...]", ("stage1", "stage2", "final")):
        for job in _expected_jobs(root, stage):
            if job.key in seen:
                violations.append(f"duplicate logical key across stages: {job.key}")
            seen.add(job.key)
            should_access_test = stage == "final"
            if job.official_test_accessed != should_access_test:
                violations.append(f"invalid official-test flag: {job.key}")
            if job.evaluation_split != ("test" if should_access_test else "validation"):
                violations.append(f"invalid evaluation split: {job.key}")
        violations.extend(
            f"result test-access violation: {row.get('job_key')}"
            for row in _completed_rows(root, stage)
            if bool(row.get("official_test_accessed")) != (stage == "final")
        )
    return {
        "schema": "pac.balanced_hpo_audit.v1",
        "status": status,
        "logical_keys": len(seen),
        "violations": violations,
        "ok": not violations,
    }


__all__ = [
    "BalancedHPOJob",
    "ManifestRunSummary",
    "audit_campaign",
    "build_model_for_preflight",
    "campaign_status",
    "code_sha256",
    "load_manifest",
    "preflight_manifest",
    "result_path",
    "run_job",
    "run_manifest",
    "select_stage1",
    "select_stage2",
]
