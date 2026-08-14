from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pac_alphabet_q1_q2_final_campaign import (
    DEFAULT_BASELINE_ROOT,
    DEFAULT_EXTERNAL_SEARCH_ROOT,
    DEFAULT_ROOT,
    DEFAULT_UCR_SEARCH_ROOT,
    enqueue_q1_final,
    enqueue_q2,
    finalize_pipeline,
    select_q2,
    status,
)
from .pac_baseline_fairness_maximal import run_manifest
from .pac_efp_compact_equal_search import default_lanes


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen ALPHABET Q1-Q2 final campaign")
    parser.add_argument(
        "--stage",
        choices=(
            "enqueue-final",
            "worker",
            "enqueue-q2",
            "select-q2",
            "finalize",
            "status",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ucr-search-root", type=Path, default=DEFAULT_UCR_SEARCH_ROOT)
    parser.add_argument("--external-search-root", type=Path, default=DEFAULT_EXTERNAL_SEARCH_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ucr-data-root", type=Path, default=Path(".omx/data/ucr"))
    parser.add_argument("--external-data-root", type=Path, default=Path("data/external"))
    parser.add_argument("--lanes", type=int, default=20)
    args = parser.parse_args()
    lanes = default_lanes(args.lanes)

    if args.stage == "enqueue-final":
        payload = enqueue_q1_final(
            args.output_root,
            ucr_search_root=args.ucr_search_root,
            external_search_root=args.external_search_root,
            baseline_root=args.baseline_root,
            lanes=lanes,
        )
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
    elif args.stage == "enqueue-q2":
        payload = enqueue_q2(args.output_root, lanes=lanes)
    elif args.stage == "select-q2":
        payload = select_q2(args.output_root, lanes=lanes)
    elif args.stage == "finalize":
        payload = finalize_pipeline(args.output_root)
    else:
        payload = status(args.output_root)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
