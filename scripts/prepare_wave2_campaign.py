# ruff: noqa: EM102, TRY003
"""Freeze the ten-dataset, ten-model Wave-2 Stage-1 queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lnet.pac_wave2_campaign import DEFAULT_ROOT, expected_counts, stage1_jobs


def _write_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to overwrite frozen Wave-2 artifact: {path}")
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    jobs = stage1_jobs()
    manifest = "".join(
        json.dumps(job.payload(), sort_keys=True) + "\n" for job in jobs
    )
    _write_once(args.root / "stage1/master.jsonl", manifest)
    contract = {
        "schema": "alphabet.wave2.contract.v1",
        "state": "stage1_frozen",
        "counts": expected_counts(),
        "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "selection_policy": "6 candidates -> top 2 seed 11 -> winner seeds 23/31/43",
        "split_policy": "source-group-disjoint 70/15/15; TRAIN-only normalization",
        "models": sorted({job.model for job in jobs}),
        "datasets": sorted({job.dataset for job in jobs}),
    }
    _write_once(
        args.root / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
