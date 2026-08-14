#!/usr/bin/env python3
# ruff: noqa: E402, T201
"""Prepare data and run the deferred three-task HPO after the primary final."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _python_root in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_python_root) not in sys.path:
        sys.path.insert(0, str(_python_root))

from lnet.pac_balanced_hpo_campaign import audit_campaign, campaign_status
from lnet.pac_balanced_hpo_distributed import (
    DeploymentConfig,
    HostSpec,
    audit_deployment,
    load_deployment_config,
    plan_stage,
)
from lnet.pac_balanced_hpo_heavy_extension import (
    DEFAULT_ROOT,
    HEAVY_DATASETS,
    audit_extension,
    enqueue_stage1,
    select_stage1,
    select_stage2,
)
from scripts.supervise_balanced_hpo_27task import run_stage

PRIMARY_ROOT = Path(".omx/results/alphabet-balanced-hpo-27task-20260725")
DATA_FILES = tuple(
    Path(prefix) / f"{dataset}.pt"
    for dataset in HEAVY_DATASETS
    for prefix in ("selection-only", ".")
)


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def _ssh_prefix(host: HostSpec) -> list[str]:
    if host.ssh_host is None or host.ssh_key is None:
        message = f"SSH host {host.name} is missing connection settings"
        raise ValueError(message)
    return [
        "ssh",
        "-i",
        host.ssh_key,
        "-p",
        str(host.ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        host.ssh_host,
    ]


def _remote(
    host: HostSpec,
    command: str,
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - frozen host config constructs the command
        [*_ssh_prefix(host), command],
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def _external_root(host: HostSpec) -> Path:
    path = Path(host.external_data_root)
    return path if path.is_absolute() else Path(host.repo) / path


def _metadata(host: HostSpec, path: Path) -> tuple[int, str] | None:
    command = (
        f"if [ -f {shlex.quote(str(path))} ]; then "
        f"stat -c '%s' {shlex.quote(str(path))}; "
        f"sha256sum {shlex.quote(str(path))} | cut -d' ' -f1; "
        "fi"
    )
    result = _remote(host, command, capture=True)
    lines = result.stdout.splitlines()
    if not lines:
        return None
    if len(lines) != 2:
        message = f"invalid metadata response from {host.name} for {path}: {lines}"
        raise RuntimeError(message)
    return int(lines[0]), lines[1]


def _available_bytes(host: HostSpec, root: Path) -> int:
    result = _remote(
        host,
        f"df --output=avail -B1 {shlex.quote(str(root))} | tail -n 1",
        capture=True,
    )
    return int(result.stdout.strip())


def _copy_remote_file(
    source: HostSpec,
    destination: HostSpec,
    relative: Path,
    *,
    expected_size: int,
    bytes_per_second: int,
) -> None:
    source_path = _external_root(source) / relative
    destination_path = _external_root(destination) / relative
    temporary = destination_path.with_name(
        f".{destination_path.name}.partial-{os.getpid()}-{int(time.time())}"
    )
    source_process = subprocess.Popen(  # noqa: S603 - fixed SSH transport
        [*_ssh_prefix(source), f"cat -- {shlex.quote(str(source_path))}"],
        stdout=subprocess.PIPE,
    )
    destination_command = (
        "set -e; "
        f"mkdir -p {shlex.quote(str(destination_path.parent))}; "
        f"cat > {shlex.quote(str(temporary))}; "
        f"test \"$(stat -c '%s' {shlex.quote(str(temporary))})\" = "
        f"{shlex.quote(str(expected_size))}; "
        f"mv -f {shlex.quote(str(temporary))} {shlex.quote(str(destination_path))}"
    )
    destination_process = subprocess.Popen(  # noqa: S603 - fixed SSH transport
        [*_ssh_prefix(destination), destination_command],
        stdin=subprocess.PIPE,
    )
    if source_process.stdout is None or destination_process.stdin is None:
        message = "failed to create the remote data transfer pipes"
        raise RuntimeError(message)
    started = time.monotonic()
    transferred = 0
    try:
        while chunk := source_process.stdout.read(4 * 1024 * 1024):
            destination_process.stdin.write(chunk)
            transferred += len(chunk)
            target_elapsed = transferred / bytes_per_second
            delay = target_elapsed - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
        destination_process.stdin.close()
        source_return = source_process.wait()
        destination_return = destination_process.wait()
    except BaseException:
        source_process.terminate()
        destination_process.terminate()
        _remote(
            destination,
            f"rm -f {shlex.quote(str(temporary))}",
            check=False,
        )
        raise
    if source_return != 0 or destination_return != 0 or transferred != expected_size:
        _remote(
            destination,
            f"rm -f {shlex.quote(str(temporary))}",
            check=False,
        )
        message = (
            f"data transfer failed for {relative}: source={source_return}, "
            f"destination={destination_return}, bytes={transferred}/{expected_size}"
        )
        raise RuntimeError(message)


def sync_data(  # noqa: C901 - transfer verification remains one transactional operation
    config: DeploymentConfig,
    root: Path,
    *,
    bytes_per_second: int,
) -> dict[str, object]:
    hosts = {host.name: host for host in config.hosts if host.enabled}
    try:
        source = hosts["local_gpu"]
    except KeyError as error:
        message = "the heavy extension requires local_gpu as the authoritative data source"
        raise RuntimeError(message) from error
    source_manifest: dict[str, dict[str, object]] = {}
    for relative in DATA_FILES:
        metadata = _metadata(source, _external_root(source) / relative)
        if metadata is None:
            message = f"authoritative heavy-task artifact is missing: {relative}"
            raise FileNotFoundError(message)
        size, digest = metadata
        source_manifest[str(relative)] = {"bytes": size, "sha256": digest}

    host_rows: dict[str, dict[str, object]] = {}
    for host in hosts.values():
        missing_bytes = 0
        pending: list[tuple[Path, int, str]] = []
        for relative in DATA_FILES:
            expected = source_manifest[str(relative)]
            expected_size = int(cast("int", expected["bytes"]))
            expected_digest = str(expected["sha256"])
            current = _metadata(host, _external_root(host) / relative)
            if current != (expected_size, expected_digest):
                pending.append((relative, expected_size, expected_digest))
                missing_bytes += expected_size
        if host.name != source.name and pending:
            available = _available_bytes(host, _external_root(host))
            if available < missing_bytes + 5_000_000_000:
                message = (
                    f"{host.name} lacks space for heavy data: "
                    f"need={missing_bytes}, available={available}"
                )
                raise RuntimeError(message)
            for index, (relative, expected_size, expected_digest) in enumerate(pending, start=1):
                _log(
                    f"sync host={host.name} file={relative} "
                    f"item={index}/{len(pending)} bytes={expected_size}"
                )
                _copy_remote_file(
                    source,
                    host,
                    relative,
                    expected_size=expected_size,
                    bytes_per_second=bytes_per_second,
                )
                actual = _metadata(host, _external_root(host) / relative)
                if actual != (expected_size, expected_digest):
                    message = f"post-copy digest mismatch on {host.name}: {relative}"
                    raise RuntimeError(message)
        host_rows[host.name] = {
            "verified_files": len(DATA_FILES),
            "copied_files": 0 if host.name == source.name else len(pending),
            "ok": True,
        }
    payload: dict[str, object] = {
        "schema": "pac.balanced_hpo_heavy_data_sync.v1",
        "source_host": source.name,
        "files": source_manifest,
        "hosts": host_rows,
        "ok": True,
    }
    destination = root / "data-sync.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _require_primary_complete(primary_root: Path) -> None:
    status = campaign_status(primary_root)
    final = cast("dict[str, object]", status["final"])
    audit = audit_campaign(primary_root)
    if not final.get("done") or not audit.get("ok"):
        message = (
            "primary campaign gate is not satisfied: "
            f"final_done={final.get('done')}, audit_ok={audit.get('ok')}"
        )
        raise RuntimeError(message)


def prepare(root: Path, config: DeploymentConfig) -> dict[str, object]:
    contract = enqueue_stage1(root)
    deployment = plan_stage(root, stage="stage1", config=config)
    audit = audit_deployment(root, stage="stage1")
    if not audit["ok"]:
        message = f"heavy-extension stage1 deployment audit failed: {audit}"
        raise RuntimeError(message)
    return {"contract": contract, "deployment": deployment, "audit": audit}


def run_campaign(
    root: Path,
    primary_root: Path,
    *,
    config: DeploymentConfig,
    snapshot: Path,
    bytes_per_second: int,
) -> None:
    _require_primary_complete(primary_root)
    sync_data(config, root, bytes_per_second=bytes_per_second)
    prepare(root, config)
    run_stage(root, "stage1", config=config, snapshot=snapshot, wait_for_idle=True)
    select_stage1(root)
    plan_stage(root, stage="stage2", config=config)
    run_stage(root, "stage2", config=config, snapshot=snapshot, wait_for_idle=True)
    select_stage2(root)
    plan_stage(root, stage="final", config=config)
    run_stage(root, "final", config=config, snapshot=snapshot, wait_for_idle=True)
    audit = audit_extension(root)
    if not audit["ok"]:
        message = f"heavy-extension final audit failed: {audit}"
        raise RuntimeError(message)
    _log("heavy three-task extension completed Stage 1, Stage 2, and final")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("prepare", "sync-data", "run-campaign", "status"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--primary-root", type=Path, default=PRIMARY_ROOT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--sync-mib-per-second", type=int, default=32)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = cast("Path", args.output_root)
    if root.is_absolute() or ".." in root.parts:
        message = "--output-root must be a safe repository-relative path"
        raise SystemExit(message)
    config = load_deployment_config(args.config)
    rate = int(args.sync_mib_per_second) * 1024 * 1024
    if rate <= 0:
        message = "--sync-mib-per-second must be positive"
        raise SystemExit(message)
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
        if args.action == "sync-data":
            print(json.dumps(sync_data(config, root, bytes_per_second=rate), indent=2))
            return
        if args.snapshot is None:
            message = "--snapshot is required for run-campaign"
            raise SystemExit(message)
        run_campaign(
            root,
            cast("Path", args.primary_root),
            config=config,
            snapshot=cast("Path", args.snapshot).resolve(),
            bytes_per_second=rate,
        )


if __name__ == "__main__":
    main()
