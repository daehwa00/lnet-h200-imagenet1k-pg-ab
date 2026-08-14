"""Validate and shard the sealed pointwise Q1-final campaign for 8/4/2 workers."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from scripts.prepare_identity_capacity_resume import LANES, prepare

EXPECTED_JOBS = 2_700
EXPECTED_TRIALS = [2, 4, 6]
EXPECTED_CAPACITIES = {
    (64, 4),
    (64, 8),
    (64, 16),
    (64, 32),
    (128, 16),
    (128, 32),
}


def _validate_contract(root: Path) -> None:
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    if contract.get("schema") != "pac_pointwise_identity_capacity_q1_final_contract.v1":
        raise RuntimeError("not a sealed pointwise Identity Q1-final campaign")
    if contract.get("jobs") != EXPECTED_JOBS:
        raise RuntimeError(f"campaign must contain exactly {EXPECTED_JOBS} jobs")
    if contract.get("optimizer_trials") != EXPECTED_TRIALS:
        raise RuntimeError(f"optimizer trials must be {EXPECTED_TRIALS}")
    variants = cast("dict[str, dict[str, int]]", contract.get("variants"))
    capacities = {(spec["model_dim"], spec["modes"]) for spec in variants.values()}
    if capacities != EXPECTED_CAPACITIES:
        raise RuntimeError(f"capacity grid mismatch: {sorted(capacities)}")
    lane_counts = {
        host: sum(lane_host == host for _name, lane_host, _gpu, _speed in LANES)
        for host in {lane_host for _name, lane_host, _gpu, _speed in LANES}
    }
    if lane_counts != {"pro6000": 8, "local_gpu": 4, "secondary_gpu": 2}:
        raise RuntimeError(f"physical worker policy mismatch: {lane_counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    repository = args.repository.resolve()
    _validate_contract(root)
    payload = prepare(root, repository)
    if payload["original_jobs"] != EXPECTED_JOBS:
        raise RuntimeError("manifest job count changed after contract sealing")
    payload["schema"] = "pointwise_identity_capacity_aggressive_resume.v1"
    report = root / "reports" / "aggressive_resume_provenance.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
