#!/usr/bin/env python3
"""Run a clean TinyNeXt-T seed521 on one H200 MIG1 partition."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL = "tinynext_t_mig1_clean"
SEED = 521
LEARNING_RATE = 3.0e-3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output_root.resolve() / MODEL / f"seed_{SEED}"
    result = output / "result.json"
    checkpoint = output / "checkpoint.pt"
    if result.is_file():
        return 0
    runtime_path = Path(os.environ["H200_BASELINE_WANDB_RUNTIME"])
    runtime = json.loads(runtime_path.read_text())
    run = runtime["runs"][MODEL]["seeds"][str(SEED)]
    environment = os.environ.copy()
    environment.update(
        {
            "H200_BASELINE_RUN_ID": str(run["id"]),
            "H200_BASELINE_DISPLAY_NAME": str(run["display_name"]),
            "H200_BASELINE_TAGS_JSON": json.dumps(run["tags"], separators=(",", ":")),
        }
    )
    command = [
        str(args.python),
        "-u",
        str(args.worker),
        "--phase",
        "full",
        "--model-key",
        MODEL,
        "--seed",
        str(SEED),
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
    output.mkdir(parents=True, exist_ok=True)
    log = output / "worker.log"
    with log.open("ab") as stream:
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    return 0 if completed.returncode == 0 and result.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
