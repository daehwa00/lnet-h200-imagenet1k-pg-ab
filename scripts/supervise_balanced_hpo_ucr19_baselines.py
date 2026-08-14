#!/usr/bin/env python3
# ruff: noqa: E402, T201
"""Run the baseline-only UCR-19 extension through the frozen HPO executor."""

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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _python_root in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_python_root) not in sys.path:
        sys.path.insert(0, str(_python_root))

from lnet.pac_balanced_hpo_distributed import (
    DeploymentConfig,
    audit_deployment,
    load_deployment_config,
    plan_stage,
)
from lnet.pac_balanced_hpo_ucr19_baselines import (
    DEFAULT_ROOT,
    audit_extension,
    enqueue_stage1,
    reuse_completed_stage1_results,
    select_stage1,
    select_stage2,
)


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def _frozen_stage(
    *,
    root: Path,
    stage: str,
    config_path: Path,
    snapshot: Path,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{snapshot / 'src'}:{snapshot}"
    environment["PYTHONSAFEPATH"] = "1"
    command = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "run_frozen_balanced_hpo_supervisor.py"),
        "--action",
        "run-stage",
        "--config",
        str(config_path.resolve()),
        "--output-root",
        str(root),
        "--snapshot",
        str(snapshot.resolve()),
        "--stage",
        stage,
        "--wait-for-idle",
    ]
    _log(f"starting frozen executor stage={stage}")
    subprocess.run(  # noqa: S603 - all paths come from the frozen campaign configuration
        command,
        check=True,
        cwd=_PROJECT_ROOT,
        env=environment,
    )


def prepare(root: Path, config: DeploymentConfig) -> dict[str, object]:
    contract = enqueue_stage1(root)
    deployment = plan_stage(root, stage="stage1", config=config)
    audit = audit_deployment(root, stage="stage1")
    if not audit["ok"]:
        message = f"UCR-19 Stage-1 deployment audit failed: {audit}"
        raise RuntimeError(message)
    return {"contract": contract, "deployment": deployment, "audit": audit}


def run_campaign(
    root: Path,
    *,
    config_path: Path,
    config: DeploymentConfig,
    snapshot: Path,
    reuse_root: Path | None,
) -> None:
    prepare(root, config)
    if reuse_root is not None:
        reused = reuse_completed_stage1_results(reuse_root, root)
        _log(f"reused completed Stage-1 results={reused}")
    _frozen_stage(root=root, stage="stage1", config_path=config_path, snapshot=snapshot)
    select_stage1(root)
    plan_stage(root, stage="stage2", config=config)
    _frozen_stage(root=root, stage="stage2", config_path=config_path, snapshot=snapshot)
    select_stage2(root)
    plan_stage(root, stage="final", config=config)
    _frozen_stage(root=root, stage="final", config_path=config_path, snapshot=snapshot)
    audit = audit_extension(root)
    if not audit["ok"]:
        message = f"UCR-19 final audit failed: {audit}"
        raise RuntimeError(message)
    _log("baseline-only UCR-19 extension completed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("prepare", "run-campaign", "status"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--reuse-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = cast("Path", args.output_root)
    if root.is_absolute() or ".." in root.parts:
        message = "--output-root must be a safe repository-relative path"
        raise SystemExit(message)
    config_path = cast("Path", args.config)
    config = load_deployment_config(config_path)
    if args.action == "status":
        print(json.dumps(audit_extension(root), indent=2, sort_keys=True))
        return
    lock = root / f"{args.action}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _log(f"another process already owns {lock}")
            return
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        if args.action == "prepare":
            print(json.dumps(prepare(root, config), indent=2, sort_keys=True))
            return
        snapshot = cast("Path | None", args.snapshot)
        if snapshot is None:
            message = "--snapshot is required for run-campaign"
            raise SystemExit(message)
        run_campaign(
            root,
            config_path=config_path,
            config=config,
            snapshot=snapshot.resolve(),
            reuse_root=cast("Path | None", args.reuse_root),
        )


if __name__ == "__main__":
    main()
