"""CLI for the Q-only single-linear-head follow-up."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pac_q_only_linear_campaign import (
    DEFAULT_ROOT,
    enqueue,
    run_manifest,
    status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("enqueue", "worker", "status"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ucr-data-root", type=Path, default=Path(".omx/data/ucr"))
    args = parser.parse_args()

    if args.stage == "enqueue":
        if args.selection is None:
            parser.error("--selection is required for enqueue")
        payload = enqueue(args.selection, args.output_root)
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
