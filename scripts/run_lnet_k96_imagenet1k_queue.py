#!/usr/bin/env python3
"""Run the three LNet K96 ImageNet-1K seeds sequentially and restart safely."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SEEDS = (501, 509, 521)
SCHEMA = "lnet.imagenet1k.lnet_k96_3seed_queue.v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wandb-mode", choices=("disabled", "online"), default="online")
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    root = args.output_root.resolve()
    status_path = root / "queue-status.json"
    status: dict[str, object] = {
        "schema": SCHEMA,
        "order": list(SEEDS),
        "jobs": {},
    }
    if status_path.is_file():
        loaded = json.loads(status_path.read_text())
        if loaded.get("schema") != SCHEMA or loaded.get("order") != list(SEEDS):
            raise RuntimeError("existing LNet K96 queue status belongs to another contract")
        status = loaded
    jobs = status["jobs"]
    if not isinstance(jobs, dict):
        raise TypeError("LNet K96 queue status has no jobs object")
    failures = 0
    for seed in SEEDS:
        result = root / "lnet_k96_p128x4_d2262" / f"seed_{seed}" / "result.json"
        job = jobs.setdefault(str(seed), {"attempts": 0, "status": "QUEUED"})
        if not isinstance(job, dict):
            raise TypeError(f"invalid LNet K96 seed status: {seed}")
        if result.is_file():
            job["status"] = "COMPLETED"
            _atomic_json(status_path, status)
            continue
        job["attempts"] = int(job.get("attempts", 0)) + 1
        job["status"] = "RUNNING"
        _atomic_json(status_path, status)
        command = [
            str(args.python),
            "-u",
            str(args.runner),
            "--data-root",
            str(args.data_root),
            "--output-root",
            str(root),
            "--seed",
            str(seed),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--wandb-mode",
            str(args.wandb_mode),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0 and result.is_file():
            job["status"] = "COMPLETED"
        else:
            job["status"] = "FAILED"
            job["returncode"] = completed.returncode
            failures += 1
        _atomic_json(status_path, status)
    status["complete"] = failures == 0 and all(
        isinstance(jobs.get(str(seed)), dict)
        and jobs[str(seed)].get("status") == "COMPLETED"
        for seed in SEEDS
    )
    _atomic_json(status_path, status)
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
