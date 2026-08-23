#!/usr/bin/env python3
"""Restart-safe persistent H200 lane for the two canonical K64 P-depth-interaction cells."""

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

import run_a2d_r2k3_k64_p_depth_interaction_imagenet100 as experiment

SCHEMA = "lnet.h200.imagenet100.k64_p_depth_interaction.queue.v1"
VARIANTS = experiment.H200_VARIANTS


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _runtime() -> dict[str, Any]:
    path = os.environ.get("H200_K64_P_DEPTH_INTERACTION_WANDB_RUNTIME")
    if not path:
        raise RuntimeError("H200 K64 P-depth-interaction runtime is missing")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "lnet.h200.imagenet100.k64_p_depth_interaction.runtime.v2"
        or tuple(payload.get("training", {}).get("variants", ())) != VARIANTS
    ):
        raise RuntimeError("H200 K64 P-depth-interaction runtime changed")
    return payload


def _complete_result(root: Path, variant: str, seed: int, epochs: int) -> bool:
    path = root / "results" / f"{variant}__seed{seed}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    history = payload.get("history")
    return bool(
        payload.get("variant") == variant
        and payload.get("seed") == seed
        and isinstance(history, list)
        and len(history) == epochs
        and history[-1].get("epoch") == epochs
        and isinstance(payload.get("contract_sha256"), str)
        and len(payload["contract_sha256"]) == 64
    )


def _initial_status(runtime: dict[str, Any], target_commit: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "campaign_id": runtime["campaign_id"],
        "target_commit": target_commit,
        "ordered_variants": list(VARIANTS),
        "jobs": {
            variant: {
                "attempts": 0,
                "returncode": None,
                "status": "PENDING",
                "updated_at_unix": int(time.time()),
            }
            for variant in VARIANTS
        },
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
        raise RuntimeError("persistent H200 queue identity changed")
    if set(payload.get("jobs", {})) != set(VARIANTS):
        raise RuntimeError("persistent H200 queue membership changed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    runtime = _runtime()
    training = runtime["training"]
    target_commit = os.environ.get("H200_EXPECTED_COMMIT", "")
    if len(target_commit) != 40 or args.workers != 8:
        raise RuntimeError("H200 queue requires the frozen commit and exactly 8 workers")
    args.root.mkdir(parents=True, exist_ok=True)
    status_path = args.root / "k64-p-depth-interaction-queue.json"
    status = _load_status(status_path, runtime, target_commit)
    failures = 0

    for variant in VARIANTS:
        job = status["jobs"][variant]
        if _complete_result(args.root, variant, training["seed"], training["epochs"]):
            job.update(status="COMPLETED", returncode=0, updated_at_unix=int(time.time()))
            _atomic_json(status_path, status)
            print(f"H200_K64_P_DEPTH_INTERACTION_QUEUE_SKIP={variant}", flush=True)
            continue

        job.update(
            attempts=int(job.get("attempts", 0)) + 1,
            returncode=None,
            status="RUNNING",
            updated_at_unix=int(time.time()),
        )
        status.update(status="RUNNING", updated_at_unix=int(time.time()))
        _atomic_json(status_path, status)
        command = [
            sys.executable,
            "scripts/run_h200_imagenet100_k64_p_depth_interaction.py",
            "--root",
            str(args.root),
            "--data-root",
            str(args.data_root),
            "--variants",
            variant,
            "--run-seeds",
            str(training["seed"]),
            "--epochs",
            str(training["epochs"]),
            "--batch-size",
            str(training["batch_size"]),
            "--gradient-accumulation-steps",
            "1",
            "--workers",
            str(args.workers),
            "--precision",
            training["precision"],
        ]
        print(f"H200_K64_P_DEPTH_INTERACTION_QUEUE_MODEL={variant}", flush=True)
        try:
            completed = subprocess.run(command, check=False)
            returncode = completed.returncode
        except Exception as error:
            print(f"H200_QUEUE_LAUNCH_ERROR={variant}:{type(error).__name__}", flush=True)
            returncode = 125

        if Path(os.environ["H200_CONTROL_STOP_MARKER"]).is_file():
            job.update(status="STOPPED", returncode=returncode, updated_at_unix=int(time.time()))
            status.update(status="STOPPED", updated_at_unix=int(time.time()))
            _atomic_json(status_path, status)
            print(f"H200_EXPERIMENT_STOPPED={os.environ['H200_CONTROL_STOP_MARKER']}", flush=True)
            return 0

        complete = _complete_result(
            args.root,
            variant,
            training["seed"],
            training["epochs"],
        )
        job.update(
            status="COMPLETED" if complete else "FAILED",
            returncode=returncode,
            updated_at_unix=int(time.time()),
        )
        if not complete:
            failures += 1
            print(f"H200_K64_P_DEPTH_INTERACTION_QUEUE_FAILURE={variant}:{returncode}", flush=True)
        _atomic_json(status_path, status)

    status.update(
        status="COMPLETE" if failures == 0 else "COMPLETE_WITH_FAILURES",
        updated_at_unix=int(time.time()),
    )
    _atomic_json(status_path, status)
    print("H200_K64_P_DEPTH_INTERACTION_QUEUE_STATUS=" + json.dumps(status, sort_keys=True), flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
