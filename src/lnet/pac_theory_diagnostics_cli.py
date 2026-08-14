from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .pac_theory_diagnostics import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_ROOT,
    prepare_theory_diagnostics,
    run_theory_diagnostics,
    theory_diagnostic_status,
    write_theory_diagnostic_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "worker", "status", "report"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "prepare":
        payload = prepare_theory_diagnostics(
            args.output_root, checkpoint_root=args.checkpoint_root
        )
    elif args.stage == "worker":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        run_theory_diagnostics(args.output_root, device=args.device)
        payload = theory_diagnostic_status(
            args.output_root, checkpoint_root=args.checkpoint_root
        )
    elif args.stage == "report":
        payload = write_theory_diagnostic_report(args.output_root)
    else:
        payload = theory_diagnostic_status(
            args.output_root, checkpoint_root=args.checkpoint_root
        )
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
