from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pac_fair_external_campaign import (
    DEFAULT_ROOT,
    enqueue_fair_external,
    fair_external_status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("enqueue", "status"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "enqueue":
        payload = enqueue_fair_external(args.output_root, workers=args.workers)
    else:
        payload = fair_external_status(args.output_root)
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
