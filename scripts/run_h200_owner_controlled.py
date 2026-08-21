"""Supervise an H200 command with an owner-controlled Git kill switch."""

from __future__ import annotations

# ruff: noqa: S603, T201
# pyright: reportExplicitAny=false
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from remote_run_control import GitHubRunControl, StopRequestedError
except ModuleNotFoundError:  # Imported as scripts.run_h200_owner_controlled in tests.
    from scripts.remote_run_control import GitHubRunControl, StopRequestedError


STOP_SCHEMA = "lnet.h200.owner_stop.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--target-commit", required=True)
    parser.add_argument("--stop-marker", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--grace-seconds", type=float, default=120.0)
    parser.add_argument("--term-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a supervised command is required after --")
    if min(args.grace_seconds, args.term_seconds) < 0:
        parser.error("termination windows must be nonnegative")
    return args


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_stopped_generation(
    path: Path,
    *,
    campaign_id: str,
    target_commit: str,
) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    generation = payload.get("generation")
    if (
        payload.get("schema") != STOP_SCHEMA
        or payload.get("campaign_id") != campaign_id
        or payload.get("target_commit") != target_commit
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        return None
    return generation


def _stop_payload(
    record: dict[str, Any],
    *,
    phase: str,
    forced: bool,
) -> dict[str, Any]:
    return {
        "schema": STOP_SCHEMA,
        "campaign_id": record["campaign_id"],
        "target_commit": record["target_commit"],
        "generation": record["generation"],
        "reason": record["reason"],
        "control_updated_at": record["updated_at"],
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        "forced": forced,
        "checkpoint_policy": "last completed epoch remains authoritative",
        "partial_epoch_discarded": True,
    }


def _archive_cleared_marker(path: Path, generation: int) -> None:
    if not path.exists():
        return
    archived = path.with_name(f"stopped-before-generation-{generation}.json")
    path.replace(archived)


def _wait(process: subprocess.Popen[bytes], seconds: float) -> bool:
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    term_seconds: float,
) -> bool:
    if process.poll() is not None:
        return False
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    if _wait(process, term_seconds):
        return True
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    process.wait()
    return True


def _control(args: argparse.Namespace, stopped_generation: int | None) -> GitHubRunControl:
    return GitHubRunControl(
        repo_url=args.repo_url,
        ref=args.ref,
        control_path=args.control_path,
        campaign_id=args.campaign_id,
        target_commit=args.target_commit,
        repo_root=args.repo_root,
        poll_seconds=args.poll_seconds,
        stopped_generation=stopped_generation,
    )


def _record_stop(
    args: argparse.Namespace,
    error: StopRequestedError,
    *,
    phase: str,
    forced: bool,
) -> None:
    _atomic_json(
        args.stop_marker,
        _stop_payload(error.record, phase=phase, forced=forced),
    )
    print(
        "H200_OWNER_STOPPED="
        + json.dumps(
            {
                "generation": error.record["generation"],
                "marker": str(args.stop_marker),
                "reason": error.record["reason"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    args = _parse_args()
    stopped_generation = _read_stopped_generation(
        args.stop_marker,
        campaign_id=args.campaign_id,
        target_commit=args.target_commit,
    )
    control = _control(args, stopped_generation)
    try:
        start_record = control.require_start()
    except StopRequestedError as error:
        _record_stop(args, error, phase="before_launch", forced=False)
        return 0
    if stopped_generation is not None:
        _archive_cleared_marker(args.stop_marker, int(start_record["generation"]))

    environment = os.environ.copy()
    environment["H200_CONTROL_STOP_MARKER"] = str(args.stop_marker)
    environment["H200_CONTROL_START_GENERATION"] = str(start_record["generation"])
    process = subprocess.Popen(
        args.command,
        cwd=args.repo_root,
        env=environment,
        start_new_session=True,
    )
    while True:
        try:
            control.poll(force=True)
        except StopRequestedError as error:
            _record_stop(args, error, phase="cooperative_stop_requested", forced=False)
            if _wait(process, args.grace_seconds):
                return 0
            forced = _terminate_process_group(process, term_seconds=args.term_seconds)
            _record_stop(args, error, phase="process_group_terminated", forced=forced)
            return 0
        return_code = process.poll()
        if return_code is not None:
            return return_code
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
