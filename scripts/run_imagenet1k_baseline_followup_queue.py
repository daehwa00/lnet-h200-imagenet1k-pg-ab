#!/usr/bin/env python3
"""Run the balanced remaining-seed ImageNet-1K baseline lanes."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import argparse
import json
import signal
from pathlib import Path

import run_h200_baseline_queue as queue

LEARNING_RATE = 3.0e-3
FOLLOWUP_TASKS = {
    "qlab0": (
        ("parc_net_xs", 509),
        ("parc_net_s", 509),
        ("convnextv2_atto", 509),
        ("moganet_xt", 501),
    ),
    "qlab1": (
        ("parc_net_xs", 521),
        ("parc_net_s", 521),
        ("emov2_1m", 509),
        ("emov2_1m", 521),
    ),
    "h200": (
        ("convnextv2_atto", 521),
        ("tinynext_t", 509),
        ("tinynext_t", 521),
        ("moganet_xt", 509),
        ("moganet_xt", 521),
    ),
}
MOGANET_STABILITY_TASK_ID = (
    "preflight:stability:moganet_xt:seed501:bf16_train_fp32_validation"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=tuple(FOLLOWUP_TASKS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def _moganet_stability_task(root: Path) -> queue.Task:
    output = root / "stability" / "moganet_xt" / "seed_501"
    return queue.Task(
        task_id=MOGANET_STABILITY_TASK_ID,
        phase="preflight",
        model_key="moganet_xt",
        seed=501,
        learning_rate=LEARNING_RATE,
        epochs=1,
        wandb_mode="disabled",
        output_dir=output,
        result_path=output / "result.json",
        checkpoint_path=output / "checkpoint.pt",
        max_steps=None,
    )


def _selected_tasks(campaign: queue.Campaign, root: Path, lane: str) -> list[queue.Task]:
    selected = {model: LEARNING_RATE for model, _seed in FOLLOWUP_TASKS[lane]}
    available = {
        (task.model_key, task.seed): task
        for task in queue.full_tasks(campaign, root, selected)
    }
    tasks = [available[key] for key in FOLLOWUP_TASKS[lane]]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RuntimeError("follow-up lane contains duplicate tasks")
    return tasks


def _run_pool(
    tasks: list[queue.Task],
    *,
    args: argparse.Namespace,
    status: dict[str, object],
    root: Path,
    attempts: int,
    memory_fraction: float,
) -> None:
    queue.run_task_pool(
        tasks,
        status=status,
        root=root,
        repo=args.repo.resolve(),
        python=args.python.absolute(),
        worker=args.worker.resolve(),
        data_root=args.data_root.resolve(),
        max_parallel=1,
        max_attempts=attempts,
        mps_active=False,
        mps_percentage=100,
        batch_size=args.batch_size,
        dataloader_workers=args.workers,
        gpu_memory_fraction=memory_fraction,
        poll_seconds=args.poll_seconds,
    )


def main() -> int:
    args = _arguments()
    campaign = queue.load_campaign(args.manifest.resolve())
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status = queue._load_or_create_status(
        campaign,
        queue._manifest_sha256(args.manifest.resolve()),
        root,
    )
    lane = str(args.lane)
    tasks = _selected_tasks(campaign, root, lane)
    memory_fraction = 0.9 if lane == "h200" else 1.0
    status["remaining_seed_followup"] = {
        "lane": lane,
        "tasks": [task.task_id for task in tasks],
        "learning_rate": LEARNING_RATE,
        "max_parallel": 1,
        "gpu_memory_fraction": memory_fraction,
    }

    def stop(_signal_number: int, _frame: object) -> None:
        queue._terminate_active_processes()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if lane == "h200":
            stability = _moganet_stability_task(root)
            queue._jobs(status).setdefault(stability.task_id, queue._new_job(stability))
            _run_pool(
                [stability],
                args=args,
                status=status,
                root=root,
                attempts=1,
                memory_fraction=memory_fraction,
            )
            if queue._result_payload(stability) is None:
                tasks = [task for task in tasks if task.model_key != "moganet_xt"]
                status["moganet_stability"] = "FAILED_SKIP_FULL_RUNS"
            else:
                status["moganet_stability"] = "PASSED"
        _run_pool(
            tasks,
            args=args,
            status=status,
            root=root,
            attempts=args.max_attempts,
            memory_fraction=memory_fraction,
        )
    finally:
        queue._write_status(root, status)

    jobs = queue._jobs(status)
    incomplete = [task for task in tasks if jobs[task.task_id].get("status") != "COMPLETED"]
    for task in incomplete:
        log_path = task.output_dir / "worker.log"
        print(f"BASELINE_FOLLOWUP_FAILURE_LOG={log_path}")
        if log_path.is_file():
            print(log_path.read_text(encoding="utf-8")[-20_000:])
    status["remaining_seed_followup_status"] = (
        "COMPLETE" if not incomplete else "COMPLETE_WITH_FAILURES"
    )
    queue._write_status(root, status)
    print(json.dumps(queue._status_summary(status), indent=2, sort_keys=True))
    return 0 if not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
