# ruff: noqa: T201
"""Freeze and audit the broad Stage-1 queue without launching training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lnet.pac_broad_benchmark_distributed import (
    audit_deployment,
    load_deployment_config,
    plan_jobs,
)
from lnet.pac_broad_benchmark_queue import (
    DEFAULT_ROOT,
    stage1_jobs,
    write_campaign_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("optimization/hosts/broad_benchmark_3gpu.local.json"),
    )
    arguments = parser.parse_args()

    contract = write_campaign_contract(arguments.root)
    deployment = plan_jobs(
        arguments.root,
        stage="stage1",
        jobs=stage1_jobs(),
        config=load_deployment_config(arguments.config),
    )
    audit = audit_deployment(arguments.root, stage="stage1")
    if not audit["ok"]:
        message = f"Stage-1 deployment audit failed: {audit}"
        raise RuntimeError(message)
    print(
        json.dumps(
            {
                "root": str(arguments.root),
                "contract_state": contract["state"],
                "expected": contract["expected"],
                "deployment": {
                    "assigned_jobs": deployment["assigned_jobs"],
                    "blocked_jobs": deployment["blocked_jobs"],
                    "units": len(deployment["units"]),
                    "estimated_wall_hours": (
                        float(deployment["max_estimated_wall_seconds"]) / 3_600.0
                    ),
                },
                "audit": audit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
