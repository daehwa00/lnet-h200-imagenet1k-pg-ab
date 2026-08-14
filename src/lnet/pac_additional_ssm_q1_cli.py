"""CLI for the restart-safe S5/LRU/DSS Q1 Stage-1/2 campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pac_additional_ssm_q1_campaign import (
    DEFAULT_ROOT,
    default_lanes,
    enqueue_stage1,
    install_runner_hooks,
    select_stage1,
    select_stage2,
    status,
)
from .pac_baseline_fairness_maximal import run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="S5/LRU/DSS 30-task Q1 Stage-1/2 search")
    parser.add_argument(
        "--stage",
        choices=("enqueue-stage1", "select-stage1", "select-stage2", "worker", "status"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ucr-data-root", type=Path, default=Path(".omx/data/ucr"))
    parser.add_argument("--external-data-root", type=Path, default=Path("data/external"))
    parser.add_argument("--lanes", type=int, default=14)
    args = parser.parse_args()
    lanes = default_lanes(args.lanes)

    if args.stage == "enqueue-stage1":
        payload = enqueue_stage1(args.output_root, lanes=lanes)
    elif args.stage == "select-stage1":
        payload = select_stage1(args.output_root, lanes=lanes)
    elif args.stage == "select-stage2":
        payload = select_stage2(args.output_root)
    elif args.stage == "worker":
        if args.manifest is None:
            parser.error("--manifest is required for worker")
        install_runner_hooks()
        run_manifest(
            args.output_root,
            args.manifest,
            device=args.device,
            ucr_data_root=args.ucr_data_root,
            external_data_root=args.external_data_root,
        )
        payload = status(args.output_root)
    else:
        payload = status(args.output_root)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
