"""CLI for the exploratory pointwise Identity ALPHABET Q1-final sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pac_pointwise_identity_capacity_campaign import (
    DEFAULT_ROOT,
    enqueue,
    report,
    run_manifest,
    status,
    synthetic_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "enqueue", "worker", "status", "report"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ucr-data-root", type=Path, default=Path(".omx/data/ucr"))
    parser.add_argument("--external-data-root", type=Path, default=Path("data/external"))
    args = parser.parse_args()
    if args.stage == "preflight":
        payload = synthetic_preflight(args.device)
    elif args.stage == "enqueue":
        payload = enqueue(args.output_root)
    elif args.stage == "worker":
        if args.manifest is None:
            parser.error("--manifest is required for worker stage")
        run_manifest(
            args.output_root,
            args.manifest,
            device=args.device,
            ucr_data_root=args.ucr_data_root,
            external_data_root=args.external_data_root,
        )
        payload = status(args.output_root)
    elif args.stage == "report":
        payload = report(args.output_root)
    else:
        payload = status(args.output_root)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
