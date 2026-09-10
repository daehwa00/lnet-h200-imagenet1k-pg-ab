#!/usr/bin/env python3
"""Safely hand the qlab Vision-Mamba queue from seed 501 to seed 509.

The old queue manager is stopped while its seed-501 worker continues to run.
Once that worker has written a complete result, the manager is allowed to exit
without starting another job and a fresh one-job queue is started from the
immutable runtime checkout.  This module intentionally imports only Python's
standard library: it is safe to run next to CUDA workloads.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

SCHEMA = "lnet.qlab.vim_handoff.v1"
MANIFEST_SCHEMA = "lnet.qlab.input_queue.v1"
MODEL_KEY = "vision_mamba_tiny"
OLD_SEED = 501
NEXT_SEED = 509
OLD_JOB_ID = "vision-mamba-tiny-s501"
NEXT_JOB_ID = "vision-mamba-tiny-s509"
WANDB_ID = "6cf3cad9e8cb7332"
WANDB_GROUP = "I1K-SharedInput-100ep-v1"
EXPECTED_RUNTIME_REVISION = "198093a"
RESULT_NAME = "result.json"
STOP_NAME = "STOP"
STATE_NAME = "handoff-state.json"
RECEIPT_NAME = "handoff-receipt.json"
LOCK_NAME = ".handoff.lock"


class HandoffError(RuntimeError):
    """Raised when continuing would risk duplicating or interrupting a run."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    """Write a JSON document durably and replace the destination atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = _canonical_json(value) + b"\n"
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HandoffError(f"cannot read JSON document {path}: {error}") from error
    if not isinstance(value, dict):
        raise HandoffError(f"JSON document is not an object: {path}")
    return value


def _read_proc_cmdline(pid: int) -> str | None:
    """Return a process command line, or ``None`` if the pid is gone."""

    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if not raw:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _read_proc_state(pid: int) -> str | None:
    """Read the Linux process state character from ``/proc/<pid>/stat``."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    # The comm field can contain ')', therefore split after its final ')'.
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    return fields[0] if fields else None


def _process_gone(pid: int) -> bool:
    """Treat a zombie as gone; the handoff process is not its parent."""

    state = _read_proc_state(pid)
    if state is not None:
        return state in {"Z", "z"}
    return _read_proc_cmdline(pid) is None


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _now() -> float:
    return time.time()


def _resolve(path: Path) -> str:
    return str(path.expanduser().resolve())


def _args_receipt(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "runtime_revision": EXPECTED_RUNTIME_REVISION,
        "old_root": _resolve(args.old_root),
        "old_queue_pid": args.old_queue_pid,
        "old_worker_pid": args.old_worker_pid,
        "runtime_root": _resolve(args.runtime_root),
        "new_root": _resolve(args.new_root),
        "data_root": _resolve(args.data_root),
        "source_root": _resolve(args.source_root),
        "python": _resolve(args.python_bin),
        "old_job_id": OLD_JOB_ID,
        "next_job_id": NEXT_JOB_ID,
    }


def _receipt_hash(receipt: dict[str, Any]) -> str:
    """Hash only the immutable receipt fields (not its informational timestamp)."""

    immutable = {
        key: receipt[key]
        for key in (
            "schema",
            "runtime_revision",
            "old_root",
            "old_queue_pid",
            "old_worker_pid",
            "runtime_root",
            "new_root",
            "data_root",
            "source_root",
            "python",
            "old_job_id",
            "next_job_id",
        )
        if key in receipt
    }
    return _sha256_payload(immutable)


def _validate_receipt(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise HandoffError(
                f"handoff receipt mismatch for {key}: "
                f"expected {expected_value!r}, got {actual.get(key)!r}"
            )
    if actual.get("schema") != SCHEMA:
        raise HandoffError("handoff receipt has an unsupported schema")


def _validate_paths(args: argparse.Namespace) -> None:
    if args.old_queue_pid <= 0 or args.old_worker_pid <= 0:
        raise HandoffError("old queue and worker pids must be positive")
    if args.old_queue_pid == args.old_worker_pid:
        raise HandoffError("old queue manager and worker must be different processes")
    if args.poll_interval < 0:
        raise HandoffError("poll interval must be non-negative")
    if args.manager_timeout <= 0:
        raise HandoffError("manager timeout must be positive")

    if not args.old_root.is_dir():
        raise HandoffError(f"old queue root does not exist: {args.old_root}")
    if not args.runtime_root.is_dir():
        raise HandoffError(f"runtime root does not exist: {args.runtime_root}")
    queue_script = args.runtime_root / "scripts" / "run_qlab_input_queue.py"
    if not queue_script.is_file():
        raise HandoffError(f"immutable runtime is missing {queue_script}")
    if not args.data_root.is_dir():
        raise HandoffError(f"data root does not exist: {args.data_root}")
    if not args.source_root.is_dir():
        raise HandoffError(f"source root does not exist: {args.source_root}")
    args.new_root.mkdir(parents=True, exist_ok=True)
    if args.new_root.is_symlink():
        raise HandoffError(f"new root must not be a symlink: {args.new_root}")


def _require_cmdline(pid: int, label: str, required: tuple[str, ...]) -> str:
    if _process_gone(pid):
        raise HandoffError(f"{label} process {pid} is not alive")
    command_line = _read_proc_cmdline(pid)
    if command_line is None:
        raise HandoffError(f"cannot inspect {label} process {pid}")
    missing = [part for part in required if part not in command_line]
    if missing:
        raise HandoffError(f"{label} process {pid} command line is not expected; missing {missing}")
    return command_line


def _validate_old_processes(args: argparse.Namespace) -> None:
    old_root = _resolve(args.old_root)
    _require_cmdline(
        args.old_queue_pid,
        "old queue manager",
        ("run_qlab_input_queue.py", old_root),
    )
    _require_cmdline(
        args.old_worker_pid,
        "seed-501 worker",
        ("run_qlab_input_single.py", old_root, MODEL_KEY, str(OLD_SEED)),
    )


def _queue_job(state: dict[str, Any], job_id: str) -> dict[str, Any]:
    jobs = state.get("jobs")
    if not isinstance(jobs, dict):
        raise HandoffError("old queue state has no jobs object")
    job = jobs.get(job_id)
    if not isinstance(job, dict):
        raise HandoffError(f"old queue state has no {job_id} entry")
    return job


def _validate_old_state(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.old_root / "queue-state.json"
    state = _load_json(state_path)
    job = _queue_job(state, OLD_JOB_ID)
    if job.get("status") != "running":
        raise HandoffError(f"seed-501 is not running in old queue: {job.get('status')!r}")
    if job.get("pid") != args.old_worker_pid:
        raise HandoffError(
            f"old queue state worker pid mismatch: expected {args.old_worker_pid}, got {job.get('pid')!r}"
        )
    if job.get("seed") not in (None, OLD_SEED):
        raise HandoffError(f"old queue state seed mismatch: {job.get('seed')!r}")
    models = job.get("models", [MODEL_KEY])
    if not isinstance(models, list) or MODEL_KEY not in models:
        raise HandoffError("old queue state does not identify vision_mamba_tiny")
    return state


def _result_value(result: dict[str, Any], key: str) -> Any:
    if key in result:
        return result[key]
    task = result.get("task")
    if isinstance(task, dict) and key in task:
        return task[key]
    return None


def _valid_result(path: Path, *, seed: int) -> tuple[bool, str]:
    if not path.is_file():
        return False, "result file is missing"
    try:
        result = _load_json(path)
    except HandoffError as error:
        return False, str(error)
    expected = {
        "model_key": MODEL_KEY,
        "seed": seed,
        "phase": "full",
        "status": "completed",
        "completed_epochs": 100,
        "requested_epochs": 100,
        "stopped_at_max_steps": False,
    }
    for key, expected_value in expected.items():
        if _result_value(result, key) != expected_value:
            return False, f"result {key} is not {expected_value!r}"
    return True, "ok"


def _write_stop_marker(old_root: Path) -> Path:
    stop_path = old_root / STOP_NAME
    if stop_path.is_symlink():
        raise HandoffError(f"refusing to use symlink stop marker: {stop_path}")
    if not stop_path.exists():
        temporary = old_root / f".{STOP_NAME}.{os.getpid()}.tmp"
        with temporary.open("wb") as stream:
            stream.write(b"qlab-vim-handoff: manager stop requested\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(stop_path)
        _fsync_directory(old_root)
    return stop_path


def _new_manifest() -> dict[str, Any]:
    """Return exactly one fresh seed-509 job; H200 seed 521 is never included."""

    return {
        "schema": MANIFEST_SCHEMA,
        "intent": "Continue the original Vision Mamba matched ImageNet-1K run on qlab",
        "recipe": "ImageNet-1K 100 epochs, batch256, BF16, AdamW LR0.003",
        "wandb_group": WANDB_GROUP,
        "jobs": [
            {
                "id": NEXT_JOB_ID,
                "models": [MODEL_KEY],
                "seed": NEXT_SEED,
                "compile_models": [],
                "display_name": "qlab-VisionMamba-Tiny-s509",
                "wandb_id": WANDB_ID,
                "wandb_group": WANDB_GROUP,
            }
        ],
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise HandoffError("new qlab manifest must contain exactly one job")
    job = jobs[0]
    if not isinstance(job, dict):
        raise HandoffError("new qlab manifest job is not an object")
    expected = {
        "id": NEXT_JOB_ID,
        "seed": NEXT_SEED,
        "display_name": "qlab-VisionMamba-Tiny-s509",
        "wandb_id": WANDB_ID,
    }
    for key, value in expected.items():
        if job.get(key) != value:
            raise HandoffError(f"new manifest {key} mismatch")
    if job.get("models") != [MODEL_KEY]:
        raise HandoffError("new manifest model mismatch")
    if job.get("wandb_group") != WANDB_GROUP or manifest.get("wandb_group") != WANDB_GROUP:
        raise HandoffError("new manifest W&B group mismatch")
    if job.get("resume_source") is not None or job.get("resume") is True:
        raise HandoffError("seed-509 must start without a resume source")
    if 521 in [item.get("seed") for item in jobs if isinstance(item, dict)]:
        raise HandoffError("H200 seed-521 must not be included in the qlab handoff")


def _ensure_manifest(new_root: Path) -> Path:
    manifest_path = new_root / "manifest.json"
    expected = _new_manifest()
    if manifest_path.exists():
        actual = _load_json(manifest_path)
        _validate_manifest(actual)
        if _sha256_payload(actual) != _sha256_payload(expected):
            raise HandoffError("existing new manifest differs from the immutable seed-509 manifest")
    else:
        _atomic_json(manifest_path, expected)
    target_root = new_root / NEXT_JOB_ID
    if target_root.exists():
        # A pre-existing checkpoint would make the generic queue add --resume.
        # Seed 509 must be a fresh run; a restart of this handoff is represented
        # by handoff-state.json and never by silently reusing this directory.
        artifacts = tuple(target_root.glob("*"))
        if artifacts:
            raise HandoffError(
                f"seed-509 output directory already contains artifacts: {target_root}"
            )
    return manifest_path


def _queue_command(args: argparse.Namespace, manifest_path: Path) -> list[str]:
    return [
        str(args.python_bin),
        "-u",
        str(args.runtime_root / "scripts" / "run_qlab_input_queue.py"),
        "--manifest",
        str(manifest_path),
        "--root",
        str(args.new_root),
        "--data-root",
        str(args.data_root),
        "--source-root",
        str(args.source_root),
        "--gpus",
        "1",
    ]


def _state_base(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "receipt_sha256": _receipt_hash(receipt),
        "status": "validated",
        "created_at": _now(),
        "updated_at": _now(),
        "old_stop_path": str(Path(receipt["old_root"]) / STOP_NAME),
        "worker_stop_path": str(Path(receipt["old_root"]) / OLD_JOB_ID / STOP_NAME),
        "sigstop_sent": False,
        "manager_term_sent": False,
        "manager_cont_sent": False,
    }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    _atomic_json(path, state)


def _validate_state(state: dict[str, Any], receipt: dict[str, Any]) -> None:
    if state.get("schema") != SCHEMA:
        raise HandoffError("handoff state has an unsupported schema")
    if state.get("receipt_sha256") != _receipt_hash(receipt):
        raise HandoffError("handoff state does not match the handoff receipt")
    if state.get("worker_stop_path") != str(Path(receipt["old_root"]) / OLD_JOB_ID / STOP_NAME):
        raise HandoffError("handoff state worker stop path is unsafe")
    if state.get("status") == "failed":
        raise HandoffError(f"handoff is already failed: {state.get('error', 'unknown error')}")


def _send(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError as error:
        raise HandoffError(f"process {pid} disappeared before {sig.name}") from error
    except PermissionError as error:
        raise HandoffError(f"permission denied sending {sig.name} to process {pid}") from error


def _manager_gone_or_wrong(manager_pid: int, old_root: str) -> bool:
    if _process_gone(manager_pid):
        return True
    command_line = _read_proc_cmdline(manager_pid)
    if command_line is None:
        return True
    if "run_qlab_input_queue.py" not in command_line or old_root not in command_line:
        raise HandoffError(f"old queue manager pid {manager_pid} changed ownership")
    return False


def _wait_for_result(
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
) -> Path:
    result_path = args.old_root / OLD_JOB_ID / RESULT_NAME
    while True:
        valid, reason = _valid_result(result_path, seed=OLD_SEED)
        if valid:
            # The queue manager's STOP path may signal its active child when
            # it resumes.  Wait for the worker to finish its own finalization
            # so the manager can never indirectly interrupt it.
            if _process_gone(args.old_worker_pid):
                state["status"] = "old_result_complete"
                state["old_result_path"] = str(result_path)
                _write_state(state_path, state)
                return result_path
            _sleep(args.poll_interval)
            continue
        if _process_gone(args.old_worker_pid):
            raise HandoffError(f"seed-501 worker exited before a complete result ({reason})")
        _sleep(args.poll_interval)


def _wait_for_manager_exit(
    args: argparse.Namespace, state_path: Path, state: dict[str, Any]
) -> None:
    deadline = time.monotonic() + args.manager_timeout
    old_root = str(args.old_root.resolve())
    while not _manager_gone_or_wrong(args.old_queue_pid, old_root):
        if time.monotonic() >= deadline:
            raise HandoffError("old queue manager did not exit within the bounded timeout")
        _sleep(args.poll_interval)
    state["status"] = "manager_exited"
    _write_state(state_path, state)


def _queue_result_complete(new_root: Path) -> bool:
    result_path = new_root / NEXT_JOB_ID / RESULT_NAME
    valid, _ = _valid_result(result_path, seed=NEXT_SEED)
    return valid


def _wait_for_new_queue(
    args: argparse.Namespace,
    state_path: Path,
    state: dict[str, Any],
    process: subprocess.Popen[bytes] | None,
    log: TextIO | None,
) -> int:
    try:
        if process is not None:
            return_code = process.wait()
        else:
            pid = state.get("new_queue_pid")
            if not isinstance(pid, int) or pid <= 0:
                raise HandoffError("handoff state has no new queue pid")
            while not _process_gone(pid):
                _sleep(args.poll_interval)
            return_code = int(state.get("new_queue_returncode", 0))
        state["new_queue_returncode"] = return_code
        if return_code != 0:
            raise HandoffError(f"new qlab queue exited with status {return_code}")
        if not _queue_result_complete(args.new_root):
            raise HandoffError("new qlab queue exited without a complete seed-509 result")
        state["status"] = "completed"
        _write_state(state_path, state)
        return return_code
    finally:
        if log is not None:
            log.close()


def _launch_new_queue(
    args: argparse.Namespace,
    manifest_path: Path,
    state_path: Path,
    state: dict[str, Any],
) -> int:
    log_path = args.new_root / "handoff-queue.log"
    log = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "WANDB_ENTITY": "daehwa",
            "WANDB_PROJECT": "alphabet2d-imagenet1k-h200-baselines",
            "WANDB_GROUP": WANDB_GROUP,
            "H200_ALLOW_NOASSERTION_SOURCES": "research-only",
        }
    )
    command = _queue_command(args, manifest_path)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(args.runtime_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log.close()
        raise
    if process.pid <= 0:
        log.close()
        raise HandoffError(f"new qlab queue returned an invalid pid: {process.pid}")
    state.update(
        {
            "status": "new_queue_running",
            "new_queue_pid": process.pid,
            "new_manifest_path": str(manifest_path),
            "new_log_path": str(log_path),
            "new_command": command,
        }
    )
    _write_state(state_path, state)
    return _wait_for_new_queue(args, state_path, state, process, log)


def _acquire_lock(new_root: Path) -> TextIO:
    lock_path = new_root / LOCK_NAME
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        lock.close()
        if error.errno in (errno.EACCES, errno.EAGAIN):
            raise HandoffError(f"another Vim handoff already owns {lock_path}") from error
        raise
    return lock


def run(args: argparse.Namespace) -> int:
    """Run or resume a handoff, returning the new queue's exit code."""

    _validate_paths(args)
    lock = _acquire_lock(args.new_root)
    try:
        receipt = _args_receipt(args)
        receipt_path = args.new_root / RECEIPT_NAME
        state_path = args.new_root / STATE_NAME
        if receipt_path.exists():
            existing_receipt = _load_json(receipt_path)
            _validate_receipt(existing_receipt, receipt)
            receipt = existing_receipt
        elif state_path.exists():
            raise HandoffError("handoff state exists without a valid receipt")
        else:
            _atomic_json(receipt_path, {**receipt, "created_at": _now()})

        if state_path.exists():
            state = _load_json(state_path)
            _validate_state(state, receipt)
        else:
            # These checks deliberately precede the first SIGSTOP.
            _validate_old_processes(args)
            _validate_old_state(args)
            state = _state_base(receipt)
            _write_state(state_path, state)

        status = state.get("status")
        if status == "completed":
            return 0
        if status == "new_queue_running":
            return _wait_for_new_queue(args, state_path, state, None, None)

        # A restart after a crash between SIGSTOP and the state write must not
        # signal the manager twice.  sigstop_sent is durable before the signal.
        if status in {"validated", "freezing"}:
            _validate_old_processes(args)
            _validate_old_state(args)
            if not state.get("sigstop_sent"):
                state["status"] = "freezing"
                _write_state(state_path, state)
                _send(args.old_queue_pid, signal.SIGSTOP)
                state["sigstop_sent"] = True
            stop_path = _write_stop_marker(args.old_root)
            state["old_stop_path"] = str(stop_path)
            state["status"] = "waiting_old_result"
            _write_state(state_path, state)
        elif status == "waiting_old_result":
            # The durable sigstop flag is the authority on restart; never
            # inspect or signal the worker as part of this recovery path.
            if not state.get("sigstop_sent"):
                raise HandoffError("waiting state has no durable SIGSTOP receipt")
            _write_stop_marker(args.old_root)

        if state.get("status") == "waiting_old_result":
            _wait_for_result(args, state, state_path)

        if state.get("status") == "old_result_complete":
            if not state.get("manager_term_sent"):
                _send(args.old_queue_pid, signal.SIGTERM)
                state["manager_term_sent"] = True
                _write_state(state_path, state)
            if not state.get("manager_cont_sent"):
                _send(args.old_queue_pid, signal.SIGCONT)
                state["manager_cont_sent"] = True
                state["status"] = "manager_stopping"
                _write_state(state_path, state)
        elif state.get("status") == "manager_stopping":
            if not state.get("manager_term_sent") or not state.get("manager_cont_sent"):
                raise HandoffError("manager-stopping state is missing signal receipts")

        if state.get("status") == "manager_stopping":
            _wait_for_manager_exit(args, state_path, state)

        if state.get("status") == "manager_exited":
            # Protect against a stale state file or PID reuse before starting
            # the fresh queue.  The old manager must truly be gone.
            if not _manager_gone_or_wrong(args.old_queue_pid, str(args.old_root.resolve())):
                _wait_for_manager_exit(args, state_path, state)
            manifest_path = _ensure_manifest(args.new_root)
            return _launch_new_queue(args, manifest_path, state_path, state)
        raise HandoffError(f"cannot continue from handoff state {state.get('status')!r}")
    except HandoffError as error:
        with suppress(Exception):
            state_path = args.new_root / STATE_NAME
            if state_path.exists():
                state = _load_json(state_path)
                state["status"] = "failed"
                state["error"] = str(error)
                _write_state(state_path, state)
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--old-queue-pid", type=int, required=True)
    parser.add_argument("--old-worker-pid", type=int, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", dest="python_bin", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--manager-timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
