#!/usr/bin/env python3
"""Run the three missing comparison models sequentially on one H200 MIG1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = ("tinyvim_s", "efficientvim_m1", "mambaout_femto")
SEEDS = (501, 509, 521)
LEARNING_RATE = 3.0e-3
SCHEMA = "lnet.imagenet1k.mig1_missing_models_queue.v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    runtime_path = Path(os.environ["H200_BASELINE_WANDB_RUNTIME"])
    wandb_runtime = json.loads(runtime_path.read_text())
    root = args.output_root.resolve()
    status_path = root / "queue-status.json"
    order = [[model, seed] for seed in SEEDS for model in MODELS]
    status: dict[str, object] = {"schema": SCHEMA, "order": order, "jobs": {}}
    if status_path.is_file():
        loaded = json.loads(status_path.read_text())
        if loaded.get("schema") != SCHEMA or loaded.get("order") != order:
            raise RuntimeError("existing MIG1 status belongs to another contract")
        status = loaded
    jobs = status.get("jobs")
    if not isinstance(jobs, dict):
        raise TypeError("MIG1 status has no jobs object")
    failures = 0
    for seed in SEEDS:
        for model in MODELS:
            task_id = f"{model}:seed{seed}"
            output = root / model / f"seed_{seed}"
            result = output / "result.json"
            checkpoint = output / "checkpoint.pt"
            job = jobs.setdefault(task_id, {"attempts": 0, "status": "QUEUED"})
            if not isinstance(job, dict):
                raise TypeError(f"invalid MIG1 job status: {task_id}")
            if result.is_file():
                job["status"] = "COMPLETED"
                _atomic_json(status_path, status)
                continue
            attempts = job.get("attempts", 0)
            if isinstance(attempts, bool) or not isinstance(attempts, int):
                raise TypeError(f"invalid MIG1 attempt count: {task_id}")
            job["attempts"] = attempts + 1
            job["status"] = "RUNNING"
            _atomic_json(status_path, status)
            command = [
                str(args.python),
                "-u",
                str(args.worker),
                "--phase",
                "full",
                "--model-key",
                model,
                "--seed",
                str(seed),
                "--learning-rate",
                str(LEARNING_RATE),
                "--epochs",
                "100",
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(output),
                "--result-path",
                str(result),
                "--checkpoint-path",
                str(checkpoint),
                "--source-root",
                str(args.source_root),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--wandb-mode",
                "online",
            ]
            if checkpoint.is_file():
                command.append("--resume")
            log = root / "logs" / f"{model}__seed{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            run = wandb_runtime["runs"][model]["seeds"][str(seed)]
            environment = os.environ.copy()
            environment.update(
                {
                    "H200_BASELINE_RUN_ID": str(run["id"]),
                    "H200_BASELINE_DISPLAY_NAME": str(run["display_name"]),
                    "H200_BASELINE_TAGS_JSON": json.dumps(run["tags"], separators=(",", ":")),
                }
            )
            with log.open("ab") as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    check=False,
                )
            if completed.returncode == 0 and result.is_file():
                job["status"] = "COMPLETED"
            else:
                job["status"] = "FAILED"
                job["returncode"] = completed.returncode
                failures += 1
            _atomic_json(status_path, status)
    status["complete"] = failures == 0 and all(
        isinstance(jobs.get(f"{model}:seed{seed}"), dict)
        and jobs[f"{model}:seed{seed}"].get("status") == "COMPLETED"
        for seed in SEEDS
        for model in MODELS
    )
    _atomic_json(status_path, status)
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
