#!/usr/bin/env python3
# ruff: noqa: E402, EM101, EM102, T201, TRY003
"""Run corrected-path ALPHABET-only UCR-19 HPO on a frozen snapshot."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

PROJECT = Path(__file__).resolve().parents[1]
for python_root in (PROJECT, PROJECT / "src"):
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

from lnet.pac_balanced_hpo_alphabet_ucr19 import (
    DEFAULT_ROOT,
    audit_extension,
    enqueue_stage1,
    select_stage1,
    select_stage2,
)
from lnet.pac_balanced_hpo_distributed import (
    DeploymentConfig,
    audit_deployment,
    load_deployment_config,
    plan_stage,
)


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def frozen_stage(
    root: Path, stage: str, config_path: Path, snapshot: Path
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{snapshot / 'src'}:{snapshot}"
    environment["PYTHONSAFEPATH"] = "1"
    command = [
        sys.executable,
        str(PROJECT / "scripts/run_frozen_balanced_hpo_supervisor.py"),
        "--action", "run-stage",
        "--config", str(config_path.resolve()),
        "--output-root", str(root),
        "--snapshot", str(snapshot.resolve()),
        "--stage", stage,
        "--wait-for-idle",
    ]
    log(f"starting frozen executor stage={stage}")
    subprocess.run(command, check=True, cwd=PROJECT, env=environment)  # noqa: S603


def prepare(root: Path, config: DeploymentConfig) -> dict[str, object]:
    contract = enqueue_stage1(root)
    deployment = plan_stage(root, stage="stage1", config=config)
    audit = audit_deployment(root, stage="stage1")
    if not audit["ok"]:
        raise RuntimeError(f"UCR-19 ALPHABET Stage-1 deployment audit failed: {audit}")
    return {"contract": contract, "deployment": deployment, "audit": audit}


def run_campaign(
    root: Path, config_path: Path, config: DeploymentConfig, snapshot: Path
) -> None:
    prepare(root, config)
    frozen_stage(root, "stage1", config_path, snapshot)
    select_stage1(root)
    plan_stage(root, stage="stage2", config=config)
    frozen_stage(root, "stage2", config_path, snapshot)
    select_stage2(root)
    plan_stage(root, stage="final", config=config)
    frozen_stage(root, "final", config_path, snapshot)
    audit = audit_extension(root)
    if not audit["ok"]:
        raise RuntimeError(f"UCR-19 ALPHABET final audit failed: {audit}")
    log("ALPHABET-only UCR-19 extension completed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("prepare", "run-campaign", "status"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    root = cast("Path", args.output_root)
    if root.is_absolute() or ".." in root.parts:
        raise SystemExit("--output-root must be repository-relative")
    config_path = cast("Path", args.config)
    config = load_deployment_config(config_path)
    if args.action == "status":
        print(json.dumps(audit_extension(root), indent=2, sort_keys=True))
        return
    lock = root / f"{args.action}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        if args.action == "prepare":
            print(json.dumps(prepare(root, config), indent=2, sort_keys=True))
            return
        snapshot = cast("Path | None", args.snapshot)
        if snapshot is None:
            raise SystemExit("--snapshot is required for run-campaign")
        run_campaign(root, config_path, config, snapshot.resolve())


if __name__ == "__main__":
    main()
