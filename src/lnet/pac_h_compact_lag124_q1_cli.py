"""CLI for the H-compact lag-(1,2,4) Q1 campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pac_baseline_fairness_maximal import run_manifest
from .pac_h_compact_lag124_q1_campaign import (
    DEFAULT_BASELINE_ROOT,
    DEFAULT_ROOT,
    default_lanes,
    enqueue_final,
    enqueue_stage1,
    select_stage1,
    select_stage2,
    status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="H-compact lag-(1,2,4) 30-task Q1")
    parser.add_argument(
        "--stage",
        choices=(
            "enqueue-stage1",
            "select-stage1",
            "select-stage2",
            "enqueue-final",
            "worker",
            "status",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ucr-data-root", type=Path, default=Path(".omx/data/ucr"))
    parser.add_argument("--external-data-root", type=Path, default=Path("data/external"))
    parser.add_argument("--lanes", type=int, default=24)
    args = parser.parse_args()
    lanes = default_lanes(args.lanes)

    if args.stage == "enqueue-stage1":
        payload = enqueue_stage1(args.output_root, lanes=lanes)
    elif args.stage == "select-stage1":
        payload = select_stage1(args.output_root, lanes=lanes)
    elif args.stage == "select-stage2":
        payload = select_stage2(args.output_root, baseline_root=args.baseline_root)
    elif args.stage == "enqueue-final":
        payload = enqueue_final(args.output_root, baseline_root=args.baseline_root, lanes=lanes)
    elif args.stage == "worker":
        if args.manifest is None:
            parser.error("--manifest is required for worker")
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
