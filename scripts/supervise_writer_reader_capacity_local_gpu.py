"""Run the two-stage writer/reader campaign on local_gpu's two GPUs."""

# ruff: noqa: S603

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from lnet.pac_writer_reader_capacity_campaign import (
    enqueue_final,
    freeze_selection,
    status,
)

ROOT = Path(".omx/results/pac-writer-reader-capacity-ablation-ucr18-local_gpu-20260724")


def _gpu_from_manifest(path: Path) -> str:
    return path.stem.split("-gpu", maxsplit=1)[1].split("-", maxsplit=1)[0]


def _run_manifests(stage: str) -> None:
    processes: list[subprocess.Popen[bytes]] = []
    for manifest in sorted((ROOT / stage / "manifests").glob("*.jsonl")):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = _gpu_from_manifest(manifest)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lnet.pac_writer_reader_capacity_cli",
                    "--stage",
                    "worker",
                    "--device",
                    "cuda",
                    "--output-root",
                    str(ROOT),
                    "--manifest",
                    str(manifest),
                ],
                env=environment,
            )
        )
    return_codes = [process.wait() for process in processes]
    if any(return_codes):
        message = f"{stage} workers exited nonzero: {return_codes}"
        raise RuntimeError(message)


def _complete_stage(stage: str, *, attempts: int = 3) -> None:
    for _ in range(attempts):
        _run_manifests(stage)
        if status(ROOT)[stage]["done"]:
            return
    message = f"{stage} did not complete after {attempts} attempts"
    raise RuntimeError(message)


def main() -> None:
    _complete_stage("stage1")
    freeze_selection(ROOT)
    enqueue_final(ROOT, lane_count=4)
    _complete_stage("final")
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_writer_reader_capacity_ablation.py",
            str(ROOT),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
