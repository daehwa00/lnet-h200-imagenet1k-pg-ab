from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pac_ucr_extra_baseline_submission import (
    DEFAULT_ROOT,
    prepare_submission_baselines,
    submission_baseline_status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "status"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    payload = (
        prepare_submission_baselines(args.output_root)
        if args.stage == "prepare"
        else submission_baseline_status(args.output_root)
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
