from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .pac_retrained_ablation_campaign import (
    DEFAULT_ROOT,
    enqueue_retrained_ablation,
    retrained_ablation_status,
    run_manifest,
    write_retrained_ablation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("enqueue", "worker", "status", "report"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "enqueue":
        payload = enqueue_retrained_ablation(args.output_root, workers=args.workers)
    elif args.stage == "worker":
        if args.manifest is None:
            parser.error("--manifest is required for worker stage")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        run_manifest(args.output_root, args.manifest, device=args.device)
        payload = retrained_ablation_status(args.output_root)
    elif args.stage == "report":
        payload = write_retrained_ablation_report(args.output_root)
    else:
        payload = retrained_ablation_status(args.output_root)
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
