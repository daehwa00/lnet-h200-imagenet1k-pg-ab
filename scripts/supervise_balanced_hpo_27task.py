#!/usr/bin/env python3
# ruff: noqa: E402, T201
# pyright: reportImplicitStringConcatenation=false
"""Resumable local_gpu/kau/3080Ti supervisor for the balanced 27-task campaign.

The supervisor never uses DDP.  It launches independent fits in long-lived
workers, with class-specific process counts from the frozen contract.  Remote
workers run from one immutable source snapshot; results are continuously
collected into the coordinator root and completed logical keys are never
replayed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Literal, cast

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _python_root in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_python_root) not in sys.path:
        sys.path.insert(0, str(_python_root))

from lnet.pac_balanced_hpo_campaign import (
    BalancedHPOJob,
    _failed_attempt_count,  # pyright: ignore[reportPrivateUsage]
    campaign_status,
    code_sha256,
    load_manifest,
    result_path,
    select_stage1,
    select_stage2,
)
from lnet.pac_balanced_hpo_distributed import (
    DeploymentConfig,
    HostSpec,
    audit_deployment,
    load_deployment_config,
    plan_stage,
)
from lnet.pac_balanced_hpo_queue import DEFAULT_ROOT, enqueue_stage1

CLI_MODULE = "lnet.pac_balanced_hpo_campaign_cli"
Stage = Literal["stage1", "stage2", "final"]


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def _run(
    command: list[str],
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - commands are constructed from frozen host config
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


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


def _rsync_transport(host: HostSpec) -> str:
    if host.ssh_key is None:
        message = f"SSH host {host.name} is missing ssh_key"
        raise ValueError(message)
    return (
        f"ssh -i {shlex.quote(host.ssh_key)} -p {host.ssh_port} "
        "-o BatchMode=yes -o ConnectTimeout=10"
    )


def _remote(host: HostSpec, command: str, *, capture: bool = False) -> str:
    result = _run([*_ssh_prefix(host), command], capture=capture)
    return result.stdout.strip() if result.stdout is not None else ""


def _snapshot_hash(snapshot: Path) -> str:
    command = [
        sys.executable,
        "-c",
        "from lnet.pac_balanced_hpo_campaign import code_sha256; print(code_sha256())",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{snapshot / 'src'}:{snapshot}"
    environment["PYTHONSAFEPATH"] = "1"
    result = subprocess.run(  # noqa: S603 - fixed interpreter and command
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        env=environment,
        cwd=snapshot,
    )
    return result.stdout.strip()


def _snapshot_name(digest: str) -> str:
    return f"balanced-hpo-{digest[:16]}"


def _sync_snapshot(host: HostSpec, snapshot: Path, digest: str) -> Path:
    if host.transport == "local":
        return snapshot
    remote_snapshot = Path(host.repo) / ".omx" / "source-snapshots" / _snapshot_name(digest)
    profile = host.profiles[0]
    hash_command = (
        f"PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 "
        f"PYTHONPATH={shlex.quote(str(remote_snapshot / 'src'))}:"
        f"{shlex.quote(str(remote_snapshot))} "
        f"{shlex.quote(profile.python)} -c "
        + shlex.quote(
            "from lnet.pac_balanced_hpo_campaign import code_sha256; print(code_sha256())"
        )
    )
    existing = ""
    with suppress(subprocess.CalledProcessError):
        existing = _remote(host, hash_command, capture=True)
    if existing == digest:
        return remote_snapshot
    if existing:
        message = f"remote immutable snapshot hash mismatch on {host.name}: {existing} != {digest}"
        raise RuntimeError(message)

    staging = remote_snapshot.with_name(
        f".{remote_snapshot.name}.staging-{os.getpid()}-{int(time.time())}"
    )
    _remote(
        host,
        f"mkdir -p {shlex.quote(str(staging))} {shlex.quote(str(remote_snapshot.parent))}",
    )
    source = f"{snapshot.resolve()}/"
    destination = f"{host.ssh_host}:{staging}/"
    _run(
        [
            "rsync",
            "-az",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "-e",
            _rsync_transport(host),
            source,
            destination,
        ]
    )
    _remote(
        host,
        "set -e; "
        f"if [ -e {shlex.quote(str(remote_snapshot))} ]; then "
        f"rm -rf {shlex.quote(str(staging))}; "
        "else "
        f"mv {shlex.quote(str(staging))} {shlex.quote(str(remote_snapshot))}; "
        f"chmod -R a-w {shlex.quote(str(remote_snapshot))}; "
        "fi",
    )
    actual = _remote(host, hash_command, capture=True)
    if actual != digest:
        message = f"source verification failed on {host.name}: {actual} != {digest}"
        raise RuntimeError(message)
    return remote_snapshot


def _require_relative_root(root: Path) -> None:
    if root.is_absolute() or ".." in root.parts:
        message = "distributed output root must be a safe repository-relative path"
        raise ValueError(message)


def _sync_stage_to_host(host: HostSpec, root: Path, stage: str) -> None:
    if host.transport == "local":
        return
    remote_root = Path(host.repo) / root
    _remote(
        host,
        f"mkdir -p {shlex.quote(str(remote_root / stage / 'completed'))} "
        f"{shlex.quote(str(remote_root / stage / 'failed'))} "
        f"{shlex.quote(str(remote_root / stage / 'attempts'))} "
        f"{shlex.quote(str(remote_root / stage / 'logs'))} "
        f"{shlex.quote(str(remote_root / 'provenance'))}",
    )
    _run(
        [
            "rsync",
            "-az",
            "-e",
            _rsync_transport(host),
            f"{root / stage}/",
            f"{host.ssh_host}:{remote_root / stage}/",
        ]
    )
    contract = root / "contract.json"
    if contract.exists():
        _run(
            [
                "rsync",
                "-az",
                "-e",
                _rsync_transport(host),
                str(contract),
                f"{host.ssh_host}:{remote_root / 'contract.json'}",
            ]
        )


def _collect_host(host: HostSpec, root: Path, stage: str) -> None:
    if host.transport == "local":
        return
    remote_root = Path(host.repo) / root / stage
    for bucket in ("completed", "failed", "attempts"):
        local = root / stage / bucket
        local.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "rsync",
                "-az",
                "-e",
                _rsync_transport(host),
                f"{host.ssh_host}:{remote_root / bucket}/",
                f"{local}/",
            ],
            check=False,
        )


def _session_name(root: Path, stage: str, lane_name: str) -> str:
    token = hashlib.sha256(str(root).encode()).hexdigest()[:8]
    return f"alpha-hpo-{token}-{stage}-{lane_name}"[:120]


def _as_int(value: object) -> int:
    return int(cast("str | int | float", value))


def _unit_jobs(unit: dict[str, object]) -> list[BalancedHPOJob]:
    return load_manifest(Path(str(unit["manifest"])))


def _unit_pending(
    root: Path,
    unit: dict[str, object],
    max_attempts: int,
) -> list[BalancedHPOJob]:
    return [
        job
        for job in _unit_jobs(unit)
        if not result_path(root, job).exists() and _failed_attempt_count(root, job) < max_attempts
    ]


def _unit_terminal(
    root: Path,
    unit: dict[str, object],
    max_attempts: int,
) -> list[BalancedHPOJob]:
    return [
        job
        for job in _unit_jobs(unit)
        if not result_path(root, job).exists() and _failed_attempt_count(root, job) >= max_attempts
    ]


def _tmux_has_session(host: HostSpec, session: str) -> bool:
    command = f"tmux has-session -t {shlex.quote(session)}"
    if host.transport == "local":
        return _run(["bash", "-lc", command], check=False).returncode == 0
    try:
        _remote(host, command)
    except subprocess.CalledProcessError:
        return False
    return True


def _gpu_idle(host: HostSpec, gpu: int) -> bool:
    query = (
        f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i {gpu}"
    )
    try:
        output = (
            _run(["bash", "-lc", query], capture=True).stdout.strip()
            if host.transport == "local"
            else _remote(host, query, capture=True)
        )
        utilization, memory = [int(value.strip()) for value in output.split(",")[:2]]
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False
    return utilization <= 10 and memory <= 2048


def _start_unit(
    host: HostSpec,
    *,
    root: Path,
    stage: str,
    unit: dict[str, object],
    snapshot: Path,
    max_attempts: int,
) -> None:
    lane = cast("dict[str, object]", unit["lane"])
    profile_name = str(lane["profile"])
    profile = host.profile(profile_name)
    if profile is None:
        message = f"host {host.name} does not provide profile {profile_name}"
        raise RuntimeError(message)
    session = _session_name(root, stage, str(lane["name"]))
    if _tmux_has_session(host, session):
        return
    manifest = Path(str(unit["manifest"]))
    logfile = root / stage / "logs" / f"{lane['name']}.log"
    cache_root = Path(host.repo) / ".omx" / "cache" / "balanced-hpo" / profile_name
    inductor_cache = cache_root / "inductor"
    triton_cache = cache_root / "triton"
    worker = (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(str(logfile.parent))} "
        f"{shlex.quote(str(cache_root))}; "
        f"for retry in $(seq 1 {max_attempts}); do "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 "
        f"TORCHINDUCTOR_CACHE_DIR={shlex.quote(str(inductor_cache))} "
        f"TRITON_CACHE_DIR={shlex.quote(str(triton_cache))} "
        f"PYTHONPATH={shlex.quote(str(snapshot / 'src'))}:{shlex.quote(str(snapshot))} "
        f"CUDA_VISIBLE_DEVICES={_as_int(lane['gpu'])} "
        f"{shlex.quote(profile.python)} -m {CLI_MODULE} "
        f"--action worker --output-root {shlex.quote(str(root))} "
        f"--manifest {shlex.quote(str(manifest))} --device cuda "
        f"--ucr-data-root {shlex.quote(host.ucr_data_root)} "
        f"--external-data-root {shlex.quote(host.external_data_root)} "
        f"--max-attempts {max_attempts} "
        f">>{shlex.quote(str(logfile))} 2>&1; "
        "done"
    )
    launch = (
        f"cd {shlex.quote(host.repo)} && "
        f"tmux new-session -d -s {shlex.quote(session)} "
        f"{shlex.quote(worker)}"
    )
    if host.transport == "local":
        _run(["bash", "-lc", launch])
    else:
        _remote(host, launch)


def _profile_preflight_manifest(
    root: Path,
    stage: str,
    host_name: str,
    profile: str,
    units: list[dict[str, object]],
) -> Path:
    representatives: dict[str, BalancedHPOJob] = {}
    for unit in units:
        for job in _unit_jobs(unit):
            representatives.setdefault(job.model, job)
    path = root / stage / "preflight" / f"{host_name}-{profile}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(job.payload(), sort_keys=True) + "\n" for job in representatives.values()
    )
    if path.exists() and path.read_text(encoding="utf-8") != content:
        message = f"preflight manifest changed after creation: {path}"
        raise FileExistsError(message)
    path.write_text(content, encoding="utf-8")
    return path


def _preflight_host_profile(
    host: HostSpec,
    *,
    root: Path,
    stage: str,
    profile_name: str,
    units: list[dict[str, object]],
    snapshot: Path,
    digest: str,
) -> None:
    profile = host.profile(profile_name)
    if profile is None:
        return
    manifest = _profile_preflight_manifest(
        root,
        stage,
        host.name,
        profile_name,
        units,
    )
    if host.transport == "ssh":
        _sync_stage_to_host(host, root, stage)
    command = (
        f"cd {shlex.quote(host.repo)} && "
        "PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 "
        f"PYTHONPATH={shlex.quote(str(snapshot / 'src'))}:{shlex.quote(str(snapshot))} "
        f"CUDA_VISIBLE_DEVICES={host.gpus[0]} "
        f"{shlex.quote(profile.python)} -m {CLI_MODULE} "
        f"--action preflight --output-root {shlex.quote(str(root))} "
        f"--manifest {shlex.quote(str(manifest))} --device cuda "
        f"--ucr-data-root {shlex.quote(host.ucr_data_root)} "
        f"--external-data-root {shlex.quote(host.external_data_root)}"
    )
    output = (
        _run(["bash", "-lc", command], capture=True).stdout
        if host.transport == "local"
        else _remote(host, command, capture=True)
    )
    payload = cast("dict[str, object]", json.loads(output))
    if payload.get("code_sha256") != digest or not payload.get("ok"):
        message = f"preflight failed on {host.name}/{profile_name}"
        raise RuntimeError(message)
    target = root / "preflight" / f"{host.name}-{profile_name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _deployment(root: Path, stage: str) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        json.loads((root / stage / "deployment.json").read_text(encoding="utf-8")),
    )


def _host_by_name(config: DeploymentConfig) -> dict[str, HostSpec]:
    return {host.name: host for host in config.hosts if host.enabled}


def _wait_wave(  # noqa: C901 - recovery loop is intentionally centralized
    *,
    root: Path,
    stage: str,
    wave: dict[str, object],
    config: DeploymentConfig,
    snapshots: dict[str, Path],
    wait_for_idle: bool,
) -> None:
    hosts = _host_by_name(config)
    units = cast("list[dict[str, object]]", wave["units"])
    launched_gpus: set[tuple[str, int]] = set()
    unreachable: defaultdict[str, int] = defaultdict(int)
    while True:
        for host in hosts.values():
            try:
                _collect_host(host, root, stage)
            except subprocess.CalledProcessError as error:
                unreachable[host.name] += 1
                _log(f"collect retry host={host.name}: {error}")
        terminal = [
            job.key for unit in units for job in _unit_terminal(root, unit, config.max_attempts)
        ]
        if terminal:
            preview = ", ".join(terminal[:5])
            message = f"{len(terminal)} jobs exhausted retries in {stage}: {preview}"
            raise RuntimeError(message)
        pending_by_unit = {
            str(cast("dict[str, object]", unit["lane"])["name"]): _unit_pending(
                root,
                unit,
                config.max_attempts,
            )
            for unit in units
        }
        if not any(pending_by_unit.values()):
            _log(
                f"wave complete stage={stage} class={wave['job_class']} "
                f"profile={wave['profile']} jobs={wave['jobs']}"
            )
            return

        groups: defaultdict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
        for unit in units:
            lane = cast("dict[str, object]", unit["lane"])
            if pending_by_unit[str(lane["name"])]:
                groups[(str(lane["host"]), _as_int(lane["gpu"]))].append(unit)
        for (host_name, gpu), group in groups.items():
            host = hosts[host_name]
            sessions_active = any(
                _tmux_has_session(
                    host,
                    _session_name(
                        root,
                        stage,
                        str(cast("dict[str, object]", unit["lane"])["name"]),
                    ),
                )
                for unit in group
            )
            gpu_key = (host_name, gpu)
            if (
                wait_for_idle
                and gpu_key not in launched_gpus
                and not sessions_active
                and not _gpu_idle(host, gpu)
            ):
                _log(f"waiting for idle GPU host={host_name} gpu={gpu}")
                continue
            launched_gpus.add(gpu_key)
            for unit in group:
                try:
                    _start_unit(
                        host,
                        root=root,
                        stage=stage,
                        unit=unit,
                        snapshot=snapshots[host_name],
                        max_attempts=config.max_attempts,
                    )
                    unreachable[host.name] = 0
                except subprocess.CalledProcessError as error:
                    unreachable[host.name] += 1
                    _log(f"launch retry host={host.name}: {error}")
        if any(count >= 20 for count in unreachable.values()):
            failed_hosts = [host for host, count in unreachable.items() if count >= 20]
            message = f"hosts remained unreachable for 20 polls: {failed_hosts}"
            raise RuntimeError(message)
        completed = sum(
            result_path(root, job).exists() for unit in units for job in _unit_jobs(unit)
        )
        _log(
            f"wave progress stage={stage} class={wave['job_class']} "
            f"profile={wave['profile']} completed={completed}/{wave['jobs']}"
        )
        time.sleep(config.poll_seconds)


def run_stage(
    root: Path,
    stage: str,
    *,
    config: DeploymentConfig,
    snapshot: Path,
    wait_for_idle: bool,
) -> None:
    digest = _snapshot_hash(snapshot)
    if digest != code_sha256():
        message = (
            "the supplied snapshot does not match the coordinator execution code: "
            f"{digest} != {code_sha256()}"
        )
        raise RuntimeError(message)
    hosts = _host_by_name(config)
    snapshots = {host.name: _sync_snapshot(host, snapshot, digest) for host in hosts.values()}
    for host in hosts.values():
        _sync_stage_to_host(host, root, stage)
    deployment = _deployment(root, stage)
    waves = cast("list[dict[str, object]]", deployment["waves"])
    preflight_groups: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for wave in waves:
        for unit in cast("list[dict[str, object]]", wave["units"]):
            lane = cast("dict[str, object]", unit["lane"])
            preflight_groups[(str(lane["host"]), str(lane["profile"]))].append(unit)
    for (host_name, profile), units in preflight_groups.items():
        _preflight_host_profile(
            hosts[host_name],
            root=root,
            stage=stage,
            profile_name=profile,
            units=units,
            snapshot=snapshots[host_name],
            digest=digest,
        )
    for wave in waves:
        _wait_wave(
            root=root,
            stage=stage,
            wave=wave,
            config=config,
            snapshots=snapshots,
            wait_for_idle=wait_for_idle,
        )
    status = cast("dict[str, object]", campaign_status(root)[stage])
    if not status["done"]:
        message = f"{stage} waves ended but stage barrier is not satisfied: {status}"
        raise RuntimeError(message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("plan", "preflight", "run-stage", "run-campaign", "status", "audit"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--stage", choices=("stage1", "stage2", "final"))
    parser.add_argument(
        "--wait-for-idle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:  # noqa: C901, PLR0912, PLR0915 - CLI action dispatch
    args = _parser().parse_args()
    root = cast("Path", args.output_root)
    _require_relative_root(root)
    config = load_deployment_config(args.config)
    lock_handle = None
    if args.action in {"preflight", "run-stage", "run-campaign"}:
        lock_path = root / "supervisor.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _log(f"another supervisor already owns {lock_path}")
            return
        lock_handle.write(f"pid={os.getpid()}\n")
        lock_handle.flush()
    if args.action == "status":
        print(json.dumps(campaign_status(root), indent=2, sort_keys=True))
        return
    if args.action == "audit":
        stages = [args.stage] if args.stage else ["stage1", "stage2", "final"]
        payload = {
            stage: audit_deployment(root, stage=cast("Stage", stage))
            for stage in stages
            if (root / stage / "deployment.json").exists()
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.action == "plan":
        if args.stage is None:
            message = "--stage is required for plan"
            raise SystemExit(message)
        payload = plan_stage(root, stage=args.stage, config=config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.snapshot is None:
        message = "--snapshot is required for distributed preflight/execution"
        raise SystemExit(message)
    snapshot = args.snapshot.resolve()
    if args.action == "preflight":
        if args.stage is None:
            message = "--stage is required for preflight"
            raise SystemExit(message)
        # run_stage owns the same source/data/model preflight and is intentionally
        # not duplicated as a second divergent path.  A preflight-only invocation
        # plans and verifies deployment/source artifacts without starting workers.
        digest = _snapshot_hash(snapshot)
        if digest != code_sha256():
            message = (
                "the supplied snapshot does not match coordinator code: "
                f"{digest} != {code_sha256()}"
            )
            raise RuntimeError(message)
        hosts = _host_by_name(config)
        snapshots = {host.name: _sync_snapshot(host, snapshot, digest) for host in hosts.values()}
        deployment = _deployment(root, args.stage)
        groups: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for wave in cast("list[dict[str, object]]", deployment["waves"]):
            for unit in cast("list[dict[str, object]]", wave["units"]):
                lane = cast("dict[str, object]", unit["lane"])
                groups[(str(lane["host"]), str(lane["profile"]))].append(unit)
        for (host_name, profile), units in groups.items():
            _sync_stage_to_host(hosts[host_name], root, args.stage)
            _preflight_host_profile(
                hosts[host_name],
                root=root,
                stage=args.stage,
                profile_name=profile,
                units=units,
                snapshot=snapshots[host_name],
                digest=digest,
            )
        print(json.dumps({"ok": True, "code_sha256": digest}, indent=2))
        return
    if args.action == "run-stage":
        if args.stage is None:
            message = "--stage is required for run-stage"
            raise SystemExit(message)
        run_stage(
            root,
            args.stage,
            config=config,
            snapshot=snapshot,
            wait_for_idle=args.wait_for_idle,
        )
        return

    enqueue_stage1(root)
    plan_stage(root, stage="stage1", config=config)
    run_stage(
        root,
        "stage1",
        config=config,
        snapshot=snapshot,
        wait_for_idle=args.wait_for_idle,
    )
    select_stage1(root)
    plan_stage(root, stage="stage2", config=config)
    run_stage(
        root,
        "stage2",
        config=config,
        snapshot=snapshot,
        wait_for_idle=args.wait_for_idle,
    )
    select_stage2(root)
    plan_stage(root, stage="final", config=config)
    run_stage(
        root,
        "final",
        config=config,
        snapshot=snapshot,
        wait_for_idle=args.wait_for_idle,
    )
    _log("balanced HPO campaign completed all Stage 1, Stage 2, and final barriers")


if __name__ == "__main__":
    main()
