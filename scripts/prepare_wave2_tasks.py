"""Build common Wave-2 tensors from dataset-specific extraction manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from lnet.pac_wave2_tasks import POLICIES, prepare_manifest_task, write_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(POLICIES), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260727)
    args = parser.parse_args()
    task = prepare_manifest_task(
        args.manifest,
        args.output_root,
        policy=POLICIES[args.dataset],
        split_seed=args.split_seed,
    )
    write_summary(task, args.output_root / "audit" / f"{args.dataset}.json")


if __name__ == "__main__":
    main()
