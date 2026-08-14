#!/usr/bin/env python3
"""Restart-safe smoke-gated sequential queue for one physical GPU lane."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import run_a2d_pole_backbone_overnight_imagenet100 as runner


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shared-smoke-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=runner.VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--wandb-project", default="alphabet2d-imagenet100")
    parser.add_argument("--wandb-entity", default="daehwa")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-token")
    return parser.parse_args()


def _read_json(path: Path) -> object:
    return json.loads(path.read_text())


def _new_status(args: argparse.Namespace) -> dict[str, object]:
    jobs = {
        variant: {
            "status": "QUEUED",
            "index": runner.SPECS_BY_VARIANT[variant].index,
            "seed": runner.SPECS_BY_VARIANT[variant].seed,
            "signature_sha256": runner.SPECS_BY_VARIANT[variant].signature_hash(),
            "attempts": 0,
            "smoke_evidence": None,
            "run_root": None,
            "log": None,
            "wandb_url": None,
            "error": None,
            "started_at": None,
            "ended_at": None,
        }
        for variant in args.variants
    }
    return {
        "schema": "lnet.pole_backbone_overnight.queue.v1",
        "created_at": _now(),
        "updated_at": _now(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": os.environ.get("LNET_CPU_AFFINITY_ACTIVE"),
        "source_fingerprint": os.environ.get("LNET_SOURCE_FINGERPRINT"),
        "wandb_group": args.wandb_group,
        "jobs": jobs,
    }


def _load_status(args: argparse.Namespace) -> dict[str, object]:
    path = args.root / "queue-status.json"
    if not path.exists():
        return _new_status(args)
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), dict):
        message = "existing overnight queue state is invalid"
        raise TypeError(message)
    if payload.get("source_fingerprint") != os.environ.get("LNET_SOURCE_FINGERPRINT"):
        message = "existing queue state belongs to a different source snapshot"
        raise RuntimeError(message)
    jobs = cast("dict[str, object]", payload["jobs"])
    if tuple(jobs) != tuple(args.variants):
        message = "existing queue variants differ from the requested ordered lane"
        raise RuntimeError(message)
    for variant in args.variants:
        job = jobs[variant]
        if not isinstance(job, dict):
            message = f"queue job is invalid: {variant}"
            raise TypeError(message)
        if job.get("signature_sha256") != runner.SPECS_BY_VARIANT[variant].signature_hash():
            message = f"queue architecture signature changed: {variant}"
            raise RuntimeError(message)
        if job.get("status") == "RUNNING":
            job["status"] = "QUEUED"
    return cast("dict[str, object]", payload)


def _write_status(args: argparse.Namespace, status: dict[str, object]) -> None:
    status["updated_at"] = _now()
    _atomic_json(args.root / "queue-status.json", status)


def _job(status: dict[str, object], variant: str) -> dict[str, object]:
    jobs = status["jobs"]
    if not isinstance(jobs, dict) or not isinstance(jobs.get(variant), dict):
        message = f"queue state has no job: {variant}"
        raise TypeError(message)
    return jobs[variant]


def _pid_matches(pid: int, token: str | None) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return token is None or token in command


def _wait_for_predecessor(args: argparse.Namespace, status: dict[str, object]) -> None:
    if args.wait_pid is None:
        return
    status["predecessor"] = {
        "pid": args.wait_pid,
        "token": args.wait_token,
        "status": "WAITING" if _pid_matches(args.wait_pid, args.wait_token) else "DONE",
    }
    _write_status(args, status)
    while _pid_matches(args.wait_pid, args.wait_token):
        time.sleep(30)
        _write_status(args, status)
    cast("dict[str, object]", status["predecessor"])["status"] = "DONE"
    cast("dict[str, object]", status["predecessor"])["ended_at"] = _now()
    _write_status(args, status)


def _environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        message = "one queue process must see exactly one GPU"
        raise RuntimeError(message)
    source = environment.get("LNET_SOURCE_FINGERPRINT")
    if not source or source == "unknown":
        message = "queue requires an immutable source fingerprint"
        raise RuntimeError(message)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{args.repo / 'src'}:{args.repo / 'scripts'}",
            "LNET_COMPILE_MODE": runner.COMPILE_MODE,
            "LNET_PERSISTENT_WORKERS": "1",
            "LNET_DATALOADER_WORKERS": str(args.workers),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(args.root / "cache" / "torchinductor"),
            "LNET_PHASE_GATED_BACKEND_CACHE": str(args.root / "cache" / "phase.json"),
            "MALLOC_ARENA_MAX": "2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "WANDB_MODE": environment.get("WANDB_MODE", "online"),
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_GROUP": args.wandb_group,
        }
    )
    return environment


def _run_logged(
    command: list[str],
    *,
    log: Path,
    repo: Path,
    environment: dict[str, str],
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(json.dumps({"event": "launch", "time": _now(), "command": command}) + "\n")
        stream.flush()
        result = subprocess.run(  # noqa: S603 - typed, internal campaign command
            command,
            cwd=repo,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(
            json.dumps({"event": "exit", "time": _now(), "code": result.returncode}) + "\n"
        )
    return result.returncode


def _valid_smoke(path: Path, signature: str, args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("status") == "PASS"
        and payload.get("signature_sha256") == signature
        and payload.get("batch_size") == args.batch_size
        and payload.get("compile_mode") == runner.COMPILE_MODE
        and payload.get("source_fingerprint") == os.environ.get("LNET_SOURCE_FINGERPRINT")
        and payload.get("resume_verified") is True
        and isinstance(payload.get("prepare_loss"), (float, int))
        and math.isfinite(float(payload["prepare_loss"]))
        and isinstance(payload.get("resume_loss"), (float, int))
        and math.isfinite(float(payload["resume_loss"]))
    )


def _smoke(
    args: argparse.Namespace,
    variant: str,
    environment: dict[str, str],
) -> tuple[int, Path, Path]:
    signature = runner.SPECS_BY_VARIANT[variant].signature_hash()
    root = args.shared_smoke_root / signature
    evidence = root / "evidence.json"
    log = root / "smoke.log"
    lock_path = args.shared_smoke_root / f"{signature}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _valid_smoke(evidence, signature, args):
            return 0, evidence, log
        command = [
            str(args.python),
            "-u",
            str(args.repo / "scripts" / "smoke_a2d_pole_backbone_overnight.py"),
            "--variant",
            variant,
            "--data-root",
            str(args.data_root),
            "--output",
            str(evidence),
            "--batch-size",
            str(args.batch_size),
            "--compile-mode",
            runner.COMPILE_MODE,
        ]
        code = _run_logged(command, log=log, repo=args.repo, environment=environment)
        if code == 0 and not _valid_smoke(evidence, signature, args):
            return 65, evidence, log
        return code, evidence, log


def _preflight(
    args: argparse.Namespace,
    status: dict[str, object],
    environment: dict[str, str],
) -> None:
    for variant in args.variants:
        job = _job(status, variant)
        if job["status"] == "DONE":
            continue
        code, evidence, log = _smoke(args, variant, environment)
        job["smoke_evidence"] = str(evidence)
        job["smoke_log"] = str(log)
        job["smoke_exit_code"] = code
        if code != 0:
            job["status"] = "SMOKE_FAILED"
            job["error"] = log.read_text(errors="replace")[-4096:] if log.exists() else None
            job["ended_at"] = _now()
        _write_status(args, status)


def _result_problem(args: argparse.Namespace, variant: str, run_root: Path) -> str | None:
    spec = runner.SPECS_BY_VARIANT[variant]
    result_path = run_root / "results" / f"{variant}__seed{spec.seed}.json"
    if not result_path.exists():
        return f"missing result: {result_path}"
    payload = _read_json(result_path)
    if not isinstance(payload, dict):
        return "result is not a JSON object"
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != args.epochs:
        return "result does not contain every requested epoch"
    if payload.get("variant") != variant or payload.get("seed") != spec.seed:
        return "result identity differs from the queue job"
    return None


def _train(
    args: argparse.Namespace,
    status: dict[str, object],
    environment: dict[str, str],
) -> None:
    for variant in args.variants:
        job = _job(status, variant)
        if job["status"] in {"DONE", "SMOKE_FAILED"}:
            continue
        spec = runner.SPECS_BY_VARIANT[variant]
        run_root = args.root / "runs" / f"{spec.index:02d}-{variant}"
        log = args.root / "logs" / f"{spec.index:02d}-{variant}.log"
        command = [
            str(args.python),
            "-u",
            str(args.repo / "scripts" / "run_a2d_pole_backbone_overnight_imagenet100.py"),
            "--root",
            str(run_root),
            "--data-root",
            str(args.data_root),
            "--variants",
            variant,
            "--run-seeds",
            str(spec.seed),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--gradient-accumulation-steps",
            "1",
            "--workers",
            str(args.workers),
            "--precision",
            "bfloat16",
        ]
        while True:
            attempts = job.get("attempts")
            if not isinstance(attempts, int):
                message = f"queue attempt count is invalid: {variant}"
                raise TypeError(message)
            job.update(
                {
                    "status": "RUNNING",
                    "attempts": attempts + 1,
                    "run_root": str(run_root),
                    "log": str(log),
                    "started_at": job["started_at"] or _now(),
                    "error": None,
                }
            )
            _write_status(args, status)
            code = _run_logged(command, log=log, repo=args.repo, environment=environment)
            problem = (
                _result_problem(args, variant, run_root)
                if code == 0
                else f"exit code {code}"
            )
            job["exit_code"] = code
            if problem is None:
                job["status"] = "DONE"
                job["ended_at"] = _now()
                _write_status(args, status)
                break
            tail = log.read_text(errors="replace")[-200_000:].lower()
            retryable = (
                code != 0
                and attempts + 1 < args.max_attempts
                and "out of memory" not in tail
            )
            job["error"] = problem
            if retryable:
                job["status"] = "RETRYING"
                _write_status(args, status)
                time.sleep(30)
                continue
            job["status"] = "FAILED"
            job["ended_at"] = _now()
            _write_status(args, status)
            break


def main() -> None:
    args = _arguments()
    if args.max_attempts < 1:
        message = "max-attempts must be positive"
        raise ValueError(message)
    args.root.mkdir(parents=True, exist_ok=True)
    status = _load_status(args)
    _write_status(args, status)
    _wait_for_predecessor(args, status)
    environment = _environment(args)
    _preflight(args, status, environment)
    _train(args, status, environment)


if __name__ == "__main__":
    main()
