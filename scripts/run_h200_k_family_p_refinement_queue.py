#!/usr/bin/env python3
"""Restart-safe sequential H200 queue for five P-refinement cells."""

from __future__ import annotations

# ruff: noqa: BLE001
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import run_a2d_r2k3_k_family_p_refinement_imagenet100 as experiment

SCHEMA = "lnet.h200.imagenet100.k_family_p_refinement.queue.v1"
RUNTIME_SCHEMA = "lnet.h200.imagenet100.k_family_p_refinement.runtime.v1"
RUNTIME_ENV_VAR = "H200_K_FAMILY_P_REFINEMENT_WANDB_RUNTIME"
WORKER_SCRIPT = "scripts/run_h200_imagenet100_k_family_p_refinement.py"
STATUS_FILENAME = "k-family-p-refinement-queue.json"
VARIANTS = experiment.VARIANTS
PARAMETER_COUNTS = {
    "M-K48-P80-80-80-80": 857_124,
    "L-K64-P80-96-80-80": 1_290_308,
    "L-K64-P80-96-96-80": 1_339_652,
    "S-K32-P48-48-64-48": 430_404,
    "XL-K96-P128-144-128-128": 2_814_116,
}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _runtime() -> dict[str, Any]:
    path = os.environ.get(RUNTIME_ENV_VAR)
    if not path:
        raise RuntimeError(f"{RUNTIME_ENV_VAR} is required")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    training = payload.get("training", {})
    if (
        payload.get("schema") != RUNTIME_SCHEMA
        or tuple(training.get("variants", ())) != VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or payload.get("parameter_counts") != PARAMETER_COUNTS
    ):
        raise RuntimeError("H200 P-refinement runtime changed")
    return payload


def _complete_result(root: Path, variant: str) -> bool:
    path = root / "results" / f"{variant}__seed501.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    history = payload.get("history")
    return bool(
        payload.get("variant") == variant
        and payload.get("seed") == 501
        and payload.get("parameters") == PARAMETER_COUNTS[variant]
        and isinstance(payload.get("contract_sha256"), str)
        and len(payload["contract_sha256"]) == 64
        and isinstance(history, list)
        and len(history) == 100
        and history[-1].get("epoch") == 100
    )


def _initial_status(runtime: dict[str, Any], target_commit: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "campaign_id": runtime["campaign_id"],
        "target_commit": target_commit,
        "ordered_variants": list(VARIANTS),
        "jobs": {variant: {"attempts": 0, "status": "PENDING"} for variant in VARIANTS},
        "status": "RUNNING",
        "updated_at_unix": int(time.time()),
    }


def _load_status(path: Path, runtime: dict[str, Any], target_commit: str) -> dict[str, Any]:
    if not path.is_file():
        return _initial_status(runtime, target_commit)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": SCHEMA,
        "campaign_id": runtime["campaign_id"],
        "target_commit": target_commit,
        "ordered_variants": list(VARIANTS),
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RuntimeError("persistent H200 P-refinement queue identity changed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    runtime = _runtime()
    target_commit = os.environ.get("H200_EXPECTED_COMMIT", "")
    if len(target_commit) != 40 or args.workers != 8:
        raise RuntimeError("H200 P-refinement queue requires a frozen commit and 8 workers")
    args.root.mkdir(parents=True, exist_ok=True)
    status_path = args.root / STATUS_FILENAME
    status = _load_status(status_path, runtime, target_commit)
    failures = 0
    for variant in VARIANTS:
        job = status["jobs"][variant]
        if _complete_result(args.root, variant):
            job.update(status="COMPLETED", returncode=0)
            _atomic_json(status_path, status)
            continue
        job.update(attempts=int(job.get("attempts", 0)) + 1, status="RUNNING")
        _atomic_json(status_path, status)
        command = [
            sys.executable,
            WORKER_SCRIPT,
            "--root", str(args.root), "--data-root", str(args.data_root),
            "--variants", variant, "--run-seeds", "501", "--epochs", "100",
            "--batch-size", "128", "--gradient-accumulation-steps", "1",
            "--workers", "8", "--precision", "bfloat16",
        ]
        try:
            returncode = subprocess.run(command, check=False).returncode
        except Exception as error:
            print(f"H200_P_REFINEMENT_LAUNCH_ERROR={variant}:{type(error).__name__}")
            returncode = 125
        if Path(os.environ["H200_CONTROL_STOP_MARKER"]).is_file():
            job.update(status="STOPPED", returncode=returncode)
            status["status"] = "STOPPED"
            _atomic_json(status_path, status)
            return 0
        complete = _complete_result(args.root, variant)
        job.update(status="COMPLETED" if complete else "FAILED", returncode=returncode)
        failures += 0 if complete else 1
        _atomic_json(status_path, status)
    status["status"] = "COMPLETE" if failures == 0 else "COMPLETE_WITH_FAILURES"
    status["updated_at_unix"] = int(time.time())
    _atomic_json(status_path, status)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
