#!/usr/bin/env python3
"""Keep both qlab GPUs occupied through the A2D Q-head halving campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


SCREEN_LANES = (
    ("E0-Current", "E2-ProtoK2", "E4-StageProtoK2-LRQ", "E6-Current-ProtoAux"),
    ("E1-ProtoK1", "E3-StageProtoK2", "E5-StageProtoK2-FusionLRQ"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wandb-project", default="alphabet2d-imagenet100")
    parser.add_argument("--wandb-entity", default="daehwa")
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _gpu_processes(index: int) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]


def _wait_for_gpu(index: int, poll_seconds: int, log_path: Path) -> None:
    while True:
        processes = _gpu_processes(index)
        if not processes:
            return
        with log_path.open("a") as sink:
            sink.write(
                json.dumps(
                    {
                        "event": "gpu_wait",
                        "gpu": index,
                        "processes": processes,
                        "time": time.time(),
                    }
                )
                + "\n"
            )
        time.sleep(poll_seconds)


def _run_job(
    args: argparse.Namespace,
    *,
    gpu: int,
    root: Path,
    variant: str,
    seed: int,
    epochs: int,
    group: str,
) -> None:
    result_path = root / "results" / f"{variant}__seed{seed}.json"
    if result_path.exists():
        return
    log_path = args.output_root / "queue-logs" / f"{group}__{variant}__s{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while not result_path.exists():
        attempt += 1
        _wait_for_gpu(gpu, args.poll_seconds, log_path)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": f"{args.repo / 'src'}:{args.repo / 'scripts'}",
                "WANDB_PROJECT": args.wandb_project,
                "WANDB_ENTITY": args.wandb_entity,
                "WANDB_GROUP": group,
                "WANDB_NAME": variant,
                "PYTHONUNBUFFERED": "1",
                "TORCHINDUCTOR_CACHE_DIR": str(
                    args.output_root / "torchinductor" / f"gpu{gpu}"
                ),
            }
        )
        command = [
            str(args.python),
            "-u",
            str(args.repo / "scripts/run_a2d_qhead_e2e_imagenet100.py"),
            "--root",
            str(root),
            "--data-root",
            str(args.data_root),
            "--variants",
            variant,
            "--run-seeds",
            str(seed),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(args.batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--workers",
            str(args.workers),
            "--precision",
            "float32",
        ]
        with log_path.open("a") as sink:
            sink.write(
                json.dumps(
                    {
                        "event": "launch",
                        "attempt": attempt,
                        "gpu": gpu,
                        "variant": variant,
                        "seed": seed,
                        "epochs": epochs,
                        "time": time.time(),
                    }
                )
                + "\n"
            )
            sink.flush()
            completed = subprocess.run(
                command,
                cwd=args.repo,
                env=environment,
                stdout=sink,
                stderr=subprocess.STDOUT,
                check=False,
            )
            sink.write(
                json.dumps(
                    {
                        "event": "exit",
                        "attempt": attempt,
                        "returncode": completed.returncode,
                        "result_exists": result_path.exists(),
                        "time": time.time(),
                    }
                )
                + "\n"
            )
        if not result_path.exists():
            # The runner checkpoints every epoch, so retrying resumes exactly.
            time.sleep(min(300, args.poll_seconds * max(1, attempt)))


def _parallel_jobs(
    args: argparse.Namespace,
    lanes: tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]],
    *,
    root: Path,
    epochs: int,
    group: str,
) -> None:
    def lane(gpu: int) -> None:
        for variant, seed in lanes[gpu]:
            _run_job(
                args,
                gpu=gpu,
                root=root,
                variant=variant,
                seed=seed,
                epochs=epochs,
                group=group,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lane, gpu) for gpu in (0, 1)]
        for future in futures:
            future.result()


def _rank(root: Path, variants: list[str]) -> list[tuple[str, float]]:
    values = []
    for variant in variants:
        path = root / "results" / f"{variant}__seed501.json"
        payload = json.loads(path.read_text())
        values.append((variant, float(payload["final_validation"]["accuracy"])))
    return sorted(values, key=lambda value: value[1], reverse=True)


def main() -> None:
    args = _parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    screen_root = args.output_root / "screen30"
    screen_lanes = tuple(
        tuple((variant, 501) for variant in lane) for lane in SCREEN_LANES
    )
    _parallel_jobs(
        args,
        screen_lanes,  # type: ignore[arg-type]
        root=screen_root,
        epochs=30,
        group="A2D-QHead-Screen30",
    )
    ranked_screen = _rank(screen_root, [variant for lane in SCREEN_LANES for variant in lane])
    selected = ["E0-Current"]
    selected.extend(
        variant for variant, _ in ranked_screen if variant != "E0-Current"
    )
    selected = selected[:4]
    confirm_root = args.output_root / "confirm100"
    confirm_lanes = (
        tuple((variant, 501) for variant in selected[0::2]),
        tuple((variant, 501) for variant in selected[1::2]),
    )
    _parallel_jobs(
        args,
        confirm_lanes,
        root=confirm_root,
        epochs=100,
        group="A2D-QHead-Confirm100",
    )
    ranked_confirm = _rank(confirm_root, selected)
    finalists = [variant for variant, _ in ranked_confirm[:2]]
    final_tasks = [(variant, seed) for variant in finalists for seed in (509, 521)]
    final_lanes = (tuple(final_tasks[0::2]), tuple(final_tasks[1::2]))
    _parallel_jobs(
        args,
        final_lanes,
        root=confirm_root,
        epochs=100,
        group="A2D-QHead-Final3Seed",
    )
    _atomic_json(
        args.output_root / "campaign-complete.json",
        {
            "schema": "lnet.a2d.qhead_campaign.v1",
            "screen_ranking": ranked_screen,
            "selected_for_100_epochs": selected,
            "confirm_ranking": ranked_confirm,
            "finalists_three_seed": finalists,
            "completed_at": time.time(),
        },
    )


if __name__ == "__main__":
    main()
