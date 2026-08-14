#!/usr/bin/env python3
"""Failure-isolated smoke gate and sequential trainer for one GPU lane."""

# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import run_a2d_deep4_p96_full_state_overnight_imagenet100 as runner
import torch

OOM_EXIT_CODE = 42
INVALID_SMOKE_EVIDENCE_EXIT_CODE = 65
CAPACITY_ATTEMPTS = (128, 64, 32)
ACTIVE_DUPLICATE_STATUSES = {"DONE", "RUNNING", "QUEUED"}


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        message = f"cannot safely read campaign state: {path}"
        raise RuntimeError(message) from error


def _known_signatures(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = _load_json(path)
    if not isinstance(payload, list):
        message = "signature registry must be a JSON list"
        raise TypeError(message)
    signatures: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            message = "signature registry rows must be objects"
            raise TypeError(message)
        status = row.get("status")
        signature = row.get("signature_sha256")
        if status in ACTIVE_DUPLICATE_STATUSES:
            if not isinstance(signature, str) or len(signature) != 64:
                message = "active signature registry row has no canonical SHA256"
                raise ValueError(message)
            signatures.add(signature)
    return signatures


def filter_lane_variants(
    variants: tuple[str, ...],
    known_signatures: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Filter exact architecture duplicates and reject duplicate lane entries."""
    selected = []
    skipped = []
    observed = set(known_signatures)
    for variant in variants:
        spec = runner.SPECS_BY_VARIANT[variant]
        signature = spec.signature_hash()
        target = skipped if signature in observed else selected
        target.append(variant)
        observed.add(signature)
    return tuple(selected), tuple(skipped)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--variants", choices=runner.VARIANTS, nargs="+", required=True)
    parser.add_argument("--signature-registry", type=Path)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compile-mode", default=runner.COMPILE_MODE)
    parser.add_argument("--wandb-project", default="alphabet2d-imagenet100")
    parser.add_argument("--wandb-entity", default="daehwa")
    parser.add_argument("--wandb-group", required=True)
    return parser.parse_args()


def _status_path(root: Path) -> Path:
    return root / "queue-status.json"


def _initial_status(
    args: argparse.Namespace,
    selected: tuple[str, ...],
    skipped: tuple[str, ...],
) -> dict[str, object]:
    existing = _status_path(args.root)
    if existing.exists():
        payload = _load_json(existing)
        if not isinstance(payload, dict):
            message = "queue status must be a JSON object"
            raise TypeError(message)
        existing_jobs = payload.get("jobs")
        if not isinstance(existing_jobs, dict):
            message = "queue status has no jobs object"
            raise TypeError(message)
        for variant in selected:
            job = existing_jobs.get(variant)
            expected = runner.SPECS_BY_VARIANT[variant].signature_hash()
            if not isinstance(job, dict) or job.get("signature_sha256") != expected:
                message = f"existing queue state does not match requested variant {variant}"
                raise RuntimeError(message)
        return payload
    jobs: dict[str, dict[str, object]] = {}
    for variant in selected:
        spec = runner.SPECS_BY_VARIANT[variant]
        jobs[variant] = {
            "status": "QUEUED",
            "signature": spec.signature(),
            "signature_sha256": spec.signature_hash(),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoint": None,
            "log_path": None,
            "exit_code": None,
            "exception": None,
            "start_time": None,
            "end_time": None,
            "attempts": [],
            "git_commit": os.environ.get("LNET_SOURCE_COMMIT", "unknown"),
        }
    for variant in skipped:
        spec = runner.SPECS_BY_VARIANT[variant]
        jobs[variant] = {
            "status": "DUPLICATE_SKIPPED",
            "signature": spec.signature(),
            "signature_sha256": spec.signature_hash(),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoint": None,
            "log_path": None,
            "exit_code": 0,
            "exception": None,
            "start_time": None,
            "end_time": _now(),
            "git_commit": os.environ.get("LNET_SOURCE_COMMIT", "unknown"),
        }
    return {
        "schema": "lnet.full_state_overnight.queue.v2",
        "lane": args.wandb_group,
        "created_at": _now(),
        "updated_at": _now(),
        "jobs": jobs,
    }


def _write_status(args: argparse.Namespace, status: dict[str, object]) -> None:
    status["updated_at"] = _now()
    _atomic_json(_status_path(args.root), status)


def _job(status: dict[str, object], variant: str) -> dict[str, object]:
    jobs = status.get("jobs")
    if not isinstance(jobs, dict) or not isinstance(jobs.get(variant), dict):
        message = f"queue status has no job entry for {variant}"
        raise TypeError(message)
    return jobs[variant]


def _environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        message = "overnight queue requires exactly one CUDA_VISIBLE_DEVICES entry"
        raise RuntimeError(message)
    for name in ("LNET_SOURCE_COMMIT", "LNET_SOURCE_FINGERPRINT", "LNET_DEVICE_IDENTITY"):
        if environment.get(name) in {None, "", "unknown"}:
            message = f"overnight deployment is missing immutable {name}"
            raise RuntimeError(message)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{args.repo / 'src'}:{args.repo / 'scripts'}"
            + (f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""),
            "LNET_COMPILE_MODE": args.compile_mode,
            "LNET_PERSISTENT_WORKERS": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": environment.get(
                "TORCHINDUCTOR_COMPILE_THREADS",
                "1",
            ),
            "TORCHINDUCTOR_CACHE_DIR": str(args.root / "cache" / "torchinductor"),
            "LNET_PHASE_GATED_BACKEND_CACHE": str(args.root / "cache" / "phase.json"),
            "LNET_LAUNCH_CACHE": str(args.root / "cache" / "launch"),
            "WANDB_MODE": environment.get("WANDB_MODE", "online"),
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_GROUP": args.wandb_group,
            "MALLOC_ARENA_MAX": "2",
        }
    )
    return environment


def _run_logged(
    command: list[str],
    *,
    log: Path,
    cwd: Path,
    environment: dict[str, str],
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(json.dumps({"event": "launch", "command": command, "time": _now()}) + "\n")
        stream.flush()
        result = subprocess.run(  # noqa: S603 - command is assembled from typed campaign inputs
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(
            json.dumps({"event": "exit", "exit_code": result.returncode, "time": _now()}) + "\n"
        )
    return result.returncode


def _smoke(
    args: argparse.Namespace,
    variant: str,
    batch_size: int,
    environment: dict[str, str],
) -> tuple[int, Path, Path]:
    root = args.root / "smoke" / variant / f"bs{batch_size}"
    evidence = root / "evidence.json"
    log = root / "smoke.log"
    if evidence.exists() and _valid_smoke_evidence_file(
        evidence,
        args=args,
        variant=variant,
        batch_size=batch_size,
    ):
        return 0, evidence, log
    command = [
        str(args.python),
        "-u",
        str(args.repo / "scripts" / "smoke_a2d_full_state_overnight.py"),
        "--variant",
        variant,
        "--data-root",
        str(args.data_root),
        "--output",
        str(evidence),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(args.seed),
        "--compile-mode",
        args.compile_mode,
    ]
    exit_code = _run_logged(command, log=log, cwd=args.repo, environment=environment)
    if exit_code == 0 and not _valid_smoke_evidence_file(
        evidence,
        args=args,
        variant=variant,
        batch_size=batch_size,
    ):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "smoke_evidence_validation_failed",
                        "evidence": str(evidence),
                        "time": _now(),
                    }
                )
                + "\n"
            )
        exit_code = INVALID_SMOKE_EVIDENCE_EXIT_CODE
    return exit_code, evidence, log


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_hash_matches(path: Path | None, expected: object) -> bool:
    if path is None or not path.is_file() or not isinstance(expected, str):
        return False
    try:
        return _sha256(path) == expected
    except OSError:
        return False


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_sequence(value: object, *, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_finite_number(item) for item in value)
    )


def _valid_smoke_evidence(
    payload: dict[object, object],
    *,
    args: argparse.Namespace,
    variant: str,
    batch_size: int,
) -> bool:
    checkpoint_value = payload.get("checkpoint")
    checkpoint = Path(checkpoint_value) if isinstance(checkpoint_value, str) else None
    expected_checkpoint = (
        args.root / "smoke" / variant / f"bs{batch_size}" / f"{variant}__smoke.pt"
    ).resolve()
    prepare_pid = payload.get("prepare_pid")
    resume_pid = payload.get("resume_pid")
    eager_steps = payload.get("eager_steps")
    compiled_steps = payload.get("compiled_steps")
    eager_losses = payload.get("eager_losses")
    compiled_losses = payload.get("compiled_losses")
    return bool(
        payload.get("schema") == "lnet.full_state_overnight.smoke.v3"
        and payload.get("status") == "PASS"
        and payload.get("variant") == variant
        and payload.get("signature_sha256") == runner.SPECS_BY_VARIANT[variant].signature_hash()
        and payload.get("compile_mode") == args.compile_mode
        and payload.get("dtype") == "bfloat16"
        and payload.get("torch_version") == torch.__version__
        and payload.get("seed") == args.seed
        and payload.get("data_root") == str(args.data_root.resolve())
        and payload.get("capacity_batch_size") == batch_size
        and payload.get("microbatch_size") == 2
        and payload.get("gradient_accumulation_steps") == 128 // batch_size
        and payload.get("source_commit") == os.environ.get("LNET_SOURCE_COMMIT")
        and payload.get("source_fingerprint") == os.environ.get("LNET_SOURCE_FINGERPRINT")
        and payload.get("device_identity") == os.environ.get("LNET_DEVICE_IDENTITY")
        and payload.get("resume_verified") is True
        and payload.get("exact_state_verified") is True
        and _positive_integer(prepare_pid)
        and _positive_integer(resume_pid)
        and prepare_pid != resume_pid
        and eager_steps == 2
        and compiled_steps == 2
        and payload.get("capacity_optimizer_steps") == 2
        and payload.get("capacity_microbatches") == 2 * (128 // batch_size)
        and _finite_sequence(eager_losses, length=2)
        and _finite_sequence(compiled_losses, length=2)
        and _finite_number(payload.get("capacity_loss"))
        and isinstance(compiled_losses, list)
        and payload.get("capacity_loss") == compiled_losses[-1]
        and payload.get("restored_epoch") == 1
        and payload.get("restored_global_step") == 2
        and payload.get("final_global_step") == 4
        and _finite_number(payload.get("peak_allocated_gib"))
        and _finite_number(payload.get("peak_reserved_gib"))
        and isinstance(payload.get("cudagraphs_active"), bool)
        and checkpoint is not None
        and checkpoint.resolve() == expected_checkpoint
        and _checkpoint_hash_matches(checkpoint, payload.get("checkpoint_sha256"))
    )


def _valid_smoke_evidence_file(
    evidence: Path,
    *,
    args: argparse.Namespace,
    variant: str,
    batch_size: int,
) -> bool:
    try:
        payload = _load_json(evidence)
    except RuntimeError:
        return False
    return isinstance(payload, dict) and _valid_smoke_evidence(
        payload,
        args=args,
        variant=variant,
        batch_size=batch_size,
    )


def _failure_excerpt(*, evidence: Path | None = None, log: Path | None = None) -> str:
    if evidence is not None and evidence.exists():
        try:
            payload = _load_json(evidence)
        except RuntimeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("exception"), str):
            return payload["exception"]
    if log is not None and log.exists():
        return log.read_text(errors="replace")[-4096:]
    return "subprocess failed without structured exception evidence"


def _record_attempt(
    job: dict[str, object],
    *,
    phase: str,
    batch_size: int,
    exit_code: int | None,
    log: Path | None,
    evidence: Path | None = None,
) -> None:
    attempts = job.setdefault("attempts", [])
    if not isinstance(attempts, list):
        message = "queue job attempts field is corrupt"
        raise TypeError(message)
    attempts.append(
        {
            "phase": phase,
            "batch_size": batch_size,
            "gradient_accumulation_steps": 128 // batch_size,
            "exit_code": exit_code,
            "log_path": str(log) if log is not None else None,
            "evidence": str(evidence) if evidence is not None else None,
            "time": _now(),
        }
    )


def _preflight(
    args: argparse.Namespace,
    status: dict[str, object],
    selected: tuple[str, ...],
    environment: dict[str, str],
) -> dict[str, int]:
    capacities: dict[str, int] = {}
    for variant in selected:
        job = _job(status, variant)
        if job.get("status") == "DONE":
            continue
        job["start_time"] = job.get("start_time") or _now()
        passed = False
        for batch_size in CAPACITY_ATTEMPTS:
            expected_root = args.root / "smoke" / variant / f"bs{batch_size}"
            evidence = expected_root / "evidence.json"
            log = expected_root / "smoke.log"
            try:
                exit_code, evidence, log = _smoke(args, variant, batch_size, environment)
            except Exception as error:  # noqa: BLE001 - isolate one variant from queue faults
                _record_attempt(
                    job,
                    phase="smoke",
                    batch_size=batch_size,
                    exit_code=None,
                    log=log,
                    evidence=evidence,
                )
                job.update(
                    {
                        "status": "SMOKE_FAILED",
                        "exit_code": None,
                        "exception": repr(error),
                        "log_path": str(log),
                        "checkpoint": str(expected_root / f"{variant}__smoke.pt"),
                        "end_time": _now(),
                    }
                )
                break
            _record_attempt(
                job,
                phase="smoke",
                batch_size=batch_size,
                exit_code=exit_code,
                log=log,
                evidence=evidence,
            )
            job.update(
                {
                    "smoke_evidence": str(evidence),
                    "smoke_log": str(log),
                    "smoke_exit_code": exit_code,
                    "last_capacity_batch": batch_size,
                    "log_path": str(log),
                    "checkpoint": str(expected_root / f"{variant}__smoke.pt"),
                }
            )
            if exit_code == 0:
                job["status"] = "QUEUED"
                job["resume_verified"] = True
                job["batch_size"] = batch_size
                job["gradient_accumulation_steps"] = 128 // batch_size
                capacities[variant] = batch_size
                passed = True
                break
            if exit_code != OOM_EXIT_CODE:
                job["status"] = "SMOKE_FAILED"
                job["exit_code"] = exit_code
                job["exception"] = _failure_excerpt(evidence=evidence, log=log)
                job["end_time"] = _now()
                break
            job["status"] = "OOM_FAILED"
            job["exit_code"] = exit_code
            job["exception"] = _failure_excerpt(evidence=evidence, log=log)
            _write_status(args, status)
        if not passed and job.get("status") == "OOM_FAILED":
            job["end_time"] = _now()
        _write_status(args, status)
    return capacities


def _is_oom_log(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        tail = path.read_text(errors="replace")[-200_000:].lower()
    except OSError:
        return False
    latest_launch = tail.rfind('"event": "launch"')
    current_attempt = tail[latest_launch:] if latest_launch >= 0 else tail
    return "out of memory" in current_attempt or "cuda error: out of memory" in current_attempt


def _training_result_problem(  # noqa: C901, PLR0911, PLR0912
    args: argparse.Namespace,
    variant: str,
    batch_size: int,
) -> str | None:
    run_root = args.root / "runs" / variant / f"bs{batch_size}"
    result_path = run_root / "results" / f"{variant}__seed{args.seed}.json"
    contract_path = run_root / "contract.json"
    try:
        result = _load_json(result_path)
        contract = _load_json(contract_path)
    except RuntimeError as error:
        return str(error)
    if not isinstance(result, dict) or not isinstance(contract, dict):
        return "training result or immutable contract is not a JSON object"
    if result.get("variant") != variant or result.get("seed") != args.seed:
        return "training result identity does not match the queue job"
    history = result.get("history")
    if not isinstance(history, list) or len(history) != args.epochs:
        return "training result does not contain every requested epoch"
    history_metrics = (
        "learning_rate",
        "train_loss",
        "train_mixed_accuracy",
        "validation_accuracy",
        "validation_cross_entropy",
    )
    for epoch, row in enumerate(history, start=1):
        if not isinstance(row, dict) or row.get("epoch") != epoch:
            return "training history epoch sequence is incomplete"
        if any(not _finite_number(row.get(name)) for name in history_metrics):
            return f"training history epoch {epoch} contains a non-finite metric"
    recipe = contract.get("recipe")
    data = contract.get("data")
    if not isinstance(recipe, dict) or not isinstance(data, dict):
        return "training contract has no recipe or data manifest"
    accumulation_steps = 128 // batch_size
    worker_override = os.environ.get("LNET_DATALOADER_WORKERS")
    active_workers = args.workers if worker_override is None else int(worker_override)
    expected_recipe = {
        "epochs": args.epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": 128,
        "precision": "bfloat16",
        "compile_mode": args.compile_mode,
        "loader_workers": active_workers,
        "loader_persistent_workers": active_workers > 0,
    }
    for name, expected_value in expected_recipe.items():
        if recipe.get(name) != expected_value:
            return f"training contract recipe changed {name}"
    train_images = data.get("train_images")
    if (
        not isinstance(train_images, int)
        or isinstance(train_images, bool)
        or train_images <= 0
    ):
        return "training contract has no positive training-image count"
    train_batches = train_images // batch_size
    expected_steps_per_epoch = (train_batches + accumulation_steps - 1) // accumulation_steps
    expected_global_step = args.epochs * expected_steps_per_epoch
    global_step = result.get("global_step")
    if global_step != expected_global_step:
        return "training result global optimizer step does not match the immutable recipe"
    for name in (
        "parameters",
        "training_seconds",
        "complete_training_examples_per_second",
        "best_validation_accuracy_diagnostic",
    ):
        if not _finite_number(result.get(name)):
            return f"training result has no finite {name}"
    validation = result.get("final_validation")
    if not isinstance(validation, dict):
        return "training result has no final validation metrics"
    for name in ("accuracy", "cross_entropy"):
        value = validation.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return f"training result has a non-finite final {name}"
    source = contract.get("deployment_source")
    if not isinstance(source, dict):
        return "training contract has no immutable deployment source"
    if source.get("git_commit") != os.environ.get("LNET_SOURCE_COMMIT"):
        return "training contract git commit differs from the deployed queue"
    if source.get("snapshot_sha256") != os.environ.get("LNET_SOURCE_FINGERPRINT"):
        return "training contract source snapshot differs from the deployed queue"
    configs = contract.get("variant_configs")
    if not isinstance(configs, dict) or not isinstance(configs.get(variant), dict):
        return "training contract has no matching variant configuration"
    transition = configs[variant].get("backbone", {}).get("stage_transition", {})
    if not isinstance(transition, dict):
        return "training contract has no stage-transition signature"
    expected = runner.SPECS_BY_VARIANT[variant].signature_hash()
    if transition.get("architecture_signature_sha256") != expected:
        return "training contract architecture signature changed"
    return None


def _reconcile_completed_jobs(
    args: argparse.Namespace,
    status: dict[str, object],
    selected: tuple[str, ...],
) -> None:
    for variant in selected:
        job = _job(status, variant)
        if job.get("status") != "DONE":
            continue
        batch_size = job.get("batch_size")
        if not isinstance(batch_size, int):
            message = f"completed queue job has no batch size: {variant}"
            raise TypeError(message)
        problem = _training_result_problem(args, variant, batch_size)
        if problem is not None:
            message = f"completed queue evidence is stale for {variant}: {problem}"
            raise RuntimeError(message)


def _train(  # noqa: C901, PLR0915
    args: argparse.Namespace,
    status: dict[str, object],
    selected: tuple[str, ...],
    capacities: dict[str, int],
    environment: dict[str, str],
) -> None:
    for variant in selected:
        job = _job(status, variant)
        if job.get("status") in {"DONE", "SMOKE_FAILED", "DUPLICATE_SKIPPED"}:
            continue
        batch_size = capacities.get(variant)
        if batch_size is None:
            continue
        attempt_index = CAPACITY_ATTEMPTS.index(batch_size)
        while attempt_index < len(CAPACITY_ATTEMPTS):
            batch_size = CAPACITY_ATTEMPTS[attempt_index]
            if batch_size != capacities.get(variant):
                smoke_root = args.root / "smoke" / variant / f"bs{batch_size}"
                evidence = smoke_root / "evidence.json"
                smoke_log = smoke_root / "smoke.log"
                try:
                    smoke_exit, evidence, smoke_log = _smoke(
                        args,
                        variant,
                        batch_size,
                        environment,
                    )
                except Exception as error:  # noqa: BLE001 - isolate one variant from queue faults
                    _record_attempt(
                        job,
                        phase="fallback_smoke",
                        batch_size=batch_size,
                        exit_code=None,
                        log=smoke_log,
                        evidence=evidence,
                    )
                    job.update(
                        {
                            "status": "SMOKE_FAILED",
                            "exit_code": None,
                            "exception": repr(error),
                            "log_path": str(smoke_log),
                            "end_time": _now(),
                        }
                    )
                    _write_status(args, status)
                    break
                _record_attempt(
                    job,
                    phase="fallback_smoke",
                    batch_size=batch_size,
                    exit_code=smoke_exit,
                    log=smoke_log,
                    evidence=evidence,
                )
                job.update(
                    {
                        "smoke_evidence": str(evidence),
                        "smoke_log": str(smoke_log),
                        "smoke_exit_code": smoke_exit,
                    }
                )
                if smoke_exit == OOM_EXIT_CODE:
                    attempt_index += 1
                    continue
                if smoke_exit != 0:
                    job["status"] = "SMOKE_FAILED"
                    job["exit_code"] = smoke_exit
                    job["exception"] = _failure_excerpt(
                        evidence=evidence,
                        log=smoke_log,
                    )
                    job["end_time"] = _now()
                    _write_status(args, status)
                    break
            run_root = args.root / "runs" / variant / f"bs{batch_size}"
            log = args.root / "logs" / f"{variant}__bs{batch_size}.log"
            checkpoint = run_root / "checkpoints" / f"{variant}__seed{args.seed}.pt"
            job.update(
                {
                    "status": "RUNNING",
                    "start_time": job.get("start_time") or _now(),
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": 128 // batch_size,
                    "log_path": str(log),
                    "checkpoint": str(checkpoint),
                }
            )
            _write_status(args, status)
            command = [
                str(args.python),
                "-u",
                str(
                    args.repo / "scripts" / "run_a2d_deep4_p96_full_state_overnight_imagenet100.py"
                ),
                "--root",
                str(run_root),
                "--data-root",
                str(args.data_root),
                "--variants",
                variant,
                "--run-seeds",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(batch_size),
                "--gradient-accumulation-steps",
                str(128 // batch_size),
                "--workers",
                str(args.workers),
                "--precision",
                "bfloat16",
            ]
            try:
                exit_code = _run_logged(command, log=log, cwd=args.repo, environment=environment)
            except Exception as error:  # noqa: BLE001 - isolate one variant from queue faults
                _record_attempt(
                    job,
                    phase="training",
                    batch_size=batch_size,
                    exit_code=None,
                    log=log,
                )
                job.update(
                    {
                        "status": "FAILED",
                        "exit_code": None,
                        "exception": repr(error),
                        "end_time": _now(),
                    }
                )
                _write_status(args, status)
                break
            _record_attempt(
                job,
                phase="training",
                batch_size=batch_size,
                exit_code=exit_code,
                log=log,
            )
            job["exit_code"] = exit_code
            if exit_code == 0:
                problem = _training_result_problem(args, variant, batch_size)
                job["status"] = "DONE" if problem is None else "FAILED"
                job["exception"] = problem
                job["end_time"] = _now()
                _write_status(args, status)
                break
            if _is_oom_log(log) and attempt_index + 1 < len(CAPACITY_ATTEMPTS):
                job["status"] = "OOM_FAILED"
                _write_status(args, status)
                attempt_index += 1
                continue
            job["status"] = "OOM_FAILED" if _is_oom_log(log) else "FAILED"
            job["exception"] = _failure_excerpt(log=log)
            job["end_time"] = _now()
            _write_status(args, status)
            break


def main() -> None:
    args = _arguments()
    args.root.mkdir(parents=True, exist_ok=True)
    known = _known_signatures(args.signature_registry)
    selected, skipped = filter_lane_variants(tuple(args.variants), known)
    status = _initial_status(args, selected, skipped)
    _write_status(args, status)
    environment = _environment(args)
    _reconcile_completed_jobs(args, status, selected)
    capacities = _preflight(args, status, selected, environment)
    _train(args, status, selected, capacities, environment)


if __name__ == "__main__":
    main()
