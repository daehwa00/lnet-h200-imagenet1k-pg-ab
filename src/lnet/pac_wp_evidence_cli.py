from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from .pac_overnight_io import append_csv_row
from .pac_recommended_low_data_runner import full_config, run_workers
from .pac_recommended_low_data_types import LowDataQueueConfig
from .pac_tf_evidence_eval import run_evidence_job
from .pac_tf_evidence_queue import EvidenceJob, full_evidence_config, mechanism_checkpoint_config
from .pac_tf_p1p2_eval import run_p1p2_job
from .pac_tf_p1p2_runner import RESULT_FILES
from .pac_tf_p1p2_types import P1P2Config, P1P2Job
from .pac_wp_evidence_campaign import (
    enqueue_evidence_shards,
    enqueue_low_data_shards,
    enqueue_p1p2_shards,
)

if TYPE_CHECKING:
    from .pac_types import PACDevice, PACExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "enqueue-low-data",
            "low-data-worker",
            "enqueue-p1p2",
            "p1p2-worker",
            "enqueue-evidence",
            "evidence-worker",
        ),
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".omx/results/pac-wp-evidence-20260712"),
    )
    parser.add_argument("--shard-root", type=Path)
    parser.add_argument("--shards", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--total-slots", type=int, default=8)
    parser.add_argument("--phase", choices=("training", "interpretability", "sensitivity"))
    args = parser.parse_args()
    if args.stage == "enqueue-low-data":
        sys.stdout.write(
            json.dumps(enqueue_low_data_shards(args.output_root, shard_count=args.shards)) + "\n"
        )
        return
    if args.stage == "enqueue-p1p2":
        sys.stdout.write(json.dumps(enqueue_p1p2_shards(args.output_root)) + "\n")
        return
    if args.stage == "enqueue-evidence":
        sys.stdout.write(json.dumps(enqueue_evidence_shards(args.output_root)) + "\n")
        return
    if args.shard_root is None:
        parser.error("--shard-root is required for worker stages")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.stage == "low-data-worker":
        config = LowDataQueueConfig(
            output_root=args.shard_root,
            preset="full",
            device=args.device,
            workers=args.workers,
            total_slots=args.total_slots,
            optimizer_mode="fused",
        )
        run_workers(config, experiment_config=wp_low_data_experiment(config))
        return
    if args.stage == "p1p2-worker":
        run_p1p2_manifest(args.shard_root, args.device, args.workers)
        return
    if args.phase is None:
        parser.error("--phase is required for evidence-worker")
    _run_evidence_manifest(args.shard_root, args.device, args.workers, args.phase)


def run_p1p2_manifest(root: Path, device: str, workers: int) -> None:
    jobs = tuple(
        P1P2Job(**json.loads(line))
        for line in (root / "p1p2_manifest.jsonl").read_text().splitlines()
        if line.strip()
    )
    config = P1P2Config(
        output_root=root,
        device=cast("PACDevice", device),
        workers=workers,
        total_slots=max(2, workers * 2),
        models=tuple(dict.fromkeys(job.model for job in jobs)),
    )
    if any(job.package == "efficiency" for job in jobs):
        jobs = tuple(sorted(jobs, key=lambda job: job.runtime == "compiled"))
    completed = _p1p2_done_keys(root)

    def execute(job: P1P2Job) -> None:
        if job.key in completed:
            return
        try:
            row = run_p1p2_job(config, job)
        except Exception as error:  # noqa: BLE001 - durable queue continues
            if job.package == "efficiency" and job.runtime == "compiled":
                row = {
                    **asdict(job),
                    "job_key": job.key,
                    "status": "done",
                    "outcome_status": "compile_unsupported",
                    "resource_limit_reason": f"{type(error).__name__}: {error}",
                }
            else:
                row = {
                    **asdict(job),
                    "job_key": job.key,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
        append_csv_row(root / "results" / RESULT_FILES[job.package], row)
        if job.package == "efficiency":
            torch.compiler.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    active_workers = 1 if any(job.package == "efficiency" for job in jobs) else workers
    with ThreadPoolExecutor(max_workers=active_workers) as pool:
        tuple(pool.map(execute, jobs))


def wp_low_data_experiment(config: LowDataQueueConfig) -> PACExperimentConfig:
    return replace(full_config(config), model_dim=64, modes=16)


def _p1p2_done_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for path in (root / "results").glob("*.csv") if (root / "results").exists() else ():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "done" and row.get("job_key"):
                    keys.add(row["job_key"])
    return keys


def _run_evidence_manifest(root: Path, device: str, workers: int, phase: str) -> None:
    jobs = tuple(
        EvidenceJob(**json.loads(line))
        for line in (root / f"{phase}_manifest.jsonl").read_text().splitlines()
        if line.strip()
    )
    completed = _evidence_done_keys(root)

    def execute(job: EvidenceJob) -> None:
        if job.key in completed:
            return
        factory = (
            mechanism_checkpoint_config
            if job.kind == "mechanism_checkpoint"
            else full_evidence_config
        )
        config = factory(root, model_dim=64, modes=16, device=cast("PACDevice", device))
        try:
            row = run_evidence_job(root, config, job)
        except Exception as error:  # noqa: BLE001 - durable queue continues
            row = {
                **asdict(job),
                "queue_key": job.key,
                "experiment_group": job.kind,
                "status": "failed",
                "notes": f"{type(error).__name__}: {error}",
            }
        append_csv_row(root / "results" / f"{phase}.csv", row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        tuple(pool.map(execute, jobs))


def _evidence_done_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for path in (root / "results").glob("*.csv") if (root / "results").exists() else ():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "done" and row.get("queue_key"):
                    keys.add(row["queue_key"])
    return keys


if __name__ == "__main__":
    main()
