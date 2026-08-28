#!/usr/bin/env python3
"""Run the two slowest remaining 100-epoch ImageNet-1K baselines on H200."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import argparse
import json
import signal
from pathlib import Path

import run_h200_baseline_queue as queue

MODEL_KEYS = (
    "moganet_xt",
    "emov2_1m",
)
SEED = 501
LEARNING_RATE = 3.0e-3
GPU_MEMORY_FRACTION = 0.9


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, choices=(1,), default=1)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = _args()
    campaign = queue.load_campaign(args.manifest.resolve())
    if not set(MODEL_KEYS).issubset(queue.MODEL_KEYS):
        raise RuntimeError("H200 backlog contains an unknown baseline model")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status = queue._load_or_create_status(
        campaign,
        queue._manifest_sha256(args.manifest.resolve()),
        root,
    )
    status["h200_backlog"] = {
        "models": list(MODEL_KEYS),
        "seeds": [SEED],
        "learning_rate": LEARNING_RATE,
        "gpu_memory_fraction": GPU_MEMORY_FRACTION,
        "calibration_source": "completed RTX4090 fixed-grid screen",
    }
    session = queue.start_mps(root, "off")

    def stop(_signal_number: int, _frame: object) -> None:
        queue._terminate_active_processes()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        max_parallel = args.max_parallel
        selected: dict[str, float] = dict.fromkeys(MODEL_KEYS, LEARNING_RATE)
        tasks = [
            task
            for task in queue.full_tasks(campaign, root, selected)
            if task.seed == SEED and task.model_key in MODEL_KEYS
        ]
        if tuple(task.model_key for task in tasks) != MODEL_KEYS:
            raise RuntimeError("H200 backlog task order changed")
        queue.reconcile_tasks(tasks, status)
        status["selected_learning_rates"] = selected
        status["backlog_status"] = "RUNNING"
        status["effective_max_parallel"] = max_parallel
        queue._write_status(root, status)
        queue.run_task_pool(
            tasks,
            status=status,
            root=root,
            repo=args.repo.resolve(),
            python=args.python.absolute(),
            worker=args.worker.resolve(),
            data_root=args.data_root.resolve(),
            max_parallel=max_parallel,
            max_attempts=args.max_attempts,
            mps_active=session.active,
            mps_percentage=min(100, 200 // max_parallel),
            batch_size=campaign.batch_size,
            dataloader_workers=campaign.dataloader_workers,
            gpu_memory_fraction=GPU_MEMORY_FRACTION,
            poll_seconds=args.poll_seconds,
        )
        jobs = queue._jobs(status)
        completed = all(jobs[task.task_id].get("status") == "COMPLETED" for task in tasks)
        if not completed:
            for task in tasks:
                if jobs[task.task_id].get("status") == "COMPLETED":
                    continue
                log_path = task.output_dir / "worker.log"
                print(f"H200_BACKLOG_FAILURE_LOG={log_path}")
                if log_path.is_file():
                    print(log_path.read_text(encoding="utf-8")[-20_000:])
        status["backlog_status"] = "COMPLETE" if completed else "COMPLETE_WITH_FAILURES"
        queue._write_status(root, status)
        print(json.dumps(queue._status_summary(status), indent=2, sort_keys=True))
        return 0 if completed else 1
    finally:
        queue.stop_mps(session)


if __name__ == "__main__":
    raise SystemExit(main())
