from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .pac_fair_boundary_campaign import (
    DEFAULT_ROOT,
    enqueue_fair_boundary,
    fair_boundary_status,
)
from .pac_wp_evidence_cli import run_p1p2_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("enqueue", "worker", "status"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shard-root", type=Path)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "enqueue":
        payload = enqueue_fair_boundary(args.output_root, shard_count=args.shards)
    elif args.stage == "worker":
        if args.shard_root is None:
            parser.error("--shard-root is required for worker stage")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        run_p1p2_manifest(args.shard_root, args.device, args.workers)
        payload = fair_boundary_status(args.output_root)
    else:
        payload = fair_boundary_status(args.output_root)
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
