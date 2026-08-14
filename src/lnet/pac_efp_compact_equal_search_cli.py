from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pac_baseline_fairness_maximal import run_manifest
from .pac_efp_compact_equal_search import (
    DEFAULT_BASELINE_ROOT,
    DEFAULT_ROOT,
    default_lanes,
    enqueue_stage1,
    select_stage1,
    select_stage2,
    status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="EFP vs compact H-only equal-search campaign")
    parser.add_argument(
        "--stage",
        choices=("enqueue", "worker", "select-stage1", "select-stage2", "status"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ucr-data-root", type=Path, default=Path(".omx/data/ucr"))
    parser.add_argument("--lanes", type=int, default=20)
    args = parser.parse_args()
    lanes = default_lanes(args.lanes)

    if args.stage == "enqueue":
        payload = enqueue_stage1(args.output_root, lanes=lanes)
    elif args.stage == "worker":
        if args.manifest is None:
            parser.error("--manifest is required for worker stage")
        run_manifest(
            args.output_root,
            args.manifest,
            device=args.device,
            ucr_data_root=args.ucr_data_root,
        )
        payload = status(args.output_root)
    elif args.stage == "select-stage1":
        payload = select_stage1(args.output_root, lanes=lanes)
    elif args.stage == "select-stage2":
        payload = select_stage2(args.output_root, baseline_root=args.baseline_root)
    else:
        payload = status(args.output_root)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
