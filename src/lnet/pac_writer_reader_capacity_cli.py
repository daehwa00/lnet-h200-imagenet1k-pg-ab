"""CLI for the prospective writer/reader capacity ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pac_writer_reader_capacity_campaign import (
    DEFAULT_ROOT,
    enqueue_final,
    enqueue_selection,
    freeze_selection,
    run_manifest,
    status,
    synthetic_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight",
            "enqueue-selection",
            "worker",
            "freeze-selection",
            "enqueue-final",
            "status",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lane-count", type=int, default=4)
    parser.add_argument("--ucr-data-root", type=Path, default=Path(".omx/data/ucr"))
    args = parser.parse_args()

    if args.stage == "preflight":
        payload = synthetic_preflight(args.device)
    elif args.stage == "enqueue-selection":
        payload = enqueue_selection(args.output_root, lane_count=args.lane_count)
    elif args.stage == "freeze-selection":
        payload = freeze_selection(args.output_root)
    elif args.stage == "enqueue-final":
        payload = enqueue_final(args.output_root, lane_count=args.lane_count)
    elif args.stage == "worker":
        if args.manifest is None:
            parser.error("--manifest is required for worker")
        run_manifest(
            args.output_root,
            args.manifest,
            device=args.device,
            ucr_data_root=args.ucr_data_root,
        )
        payload = status(args.output_root)
    else:
        payload = status(args.output_root)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
