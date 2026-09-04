#!/usr/bin/env python3
"""Run K128 seed509 and seed521 sequentially on one H200."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL_KEY = "lnet_k128_p160_160_160_128_d2262_h200_v1"
SEEDS = (509, 521)
SCHEMA = "lnet.imagenet1k.k128_h200_confirmation_queue.v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.output_root.resolve()
    status_path = root / "queue-status.json"
    status: dict[str, object] = {"schema": SCHEMA, "order": list(SEEDS), "jobs": {}}
    if status_path.is_file():
        loaded = json.loads(status_path.read_text())
        if loaded.get("schema") != SCHEMA or loaded.get("order") != list(SEEDS):
            raise RuntimeError("existing K128 H200 queue belongs to another contract")
        status = loaded
    jobs = status.get("jobs")
    if not isinstance(jobs, dict):
        raise TypeError("K128 H200 queue has no jobs object")
    failures = 0
    for seed in SEEDS:
        output = root / MODEL_KEY / f"seed_{seed}"
        result = output / "result.json"
        job = jobs.setdefault(str(seed), {"attempts": 0, "status": "QUEUED"})
        if not isinstance(job, dict):
            raise TypeError(f"invalid K128 H200 seed status: {seed}")
        if result.is_file():
            job["status"] = "COMPLETED"
        else:
            attempts = job.get("attempts", 0)
            if isinstance(attempts, bool) or not isinstance(attempts, int):
                raise TypeError(f"invalid K128 H200 attempt count: {seed}")
            job["attempts"] = attempts + 1
            job["status"] = "RUNNING"
            _write(status_path, status)
            command = [
                str(args.python), "-u", str(args.runner),
                "--data-root", str(args.data_root),
                "--output-root", str(root),
                "--seed", str(seed),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
                "--wandb-mode", "online",
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode == 0 and result.is_file():
                job["status"] = "COMPLETED"
            else:
                job["status"] = "FAILED"
                job["returncode"] = completed.returncode
                failures += 1
        _write(status_path, status)
    status["complete"] = failures == 0 and all(
        isinstance(jobs.get(str(seed)), dict) and jobs[str(seed)].get("status") == "COMPLETED"
        for seed in SEEDS
    )
    _write(status_path, status)
    return 0 if status["complete"] else 1


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
