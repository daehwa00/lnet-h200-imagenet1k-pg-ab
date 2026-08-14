# ruff: noqa: S603
"""Finish the runnable broad benchmark through Stage 2 and final on three GPUs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from lnet.pac_broad_benchmark_completion import (
    audit_campaign,
    select_stage1,
    select_stage2,
    stage_jobs,
)
from lnet.pac_broad_benchmark_distributed import (
    BroadDeploymentConfig,
    audit_deployment,
    load_deployment_config,
    plan_jobs,
)
from lnet.pac_broad_benchmark_queue import BenchmarkJob
from lnet.pac_broad_benchmark_worker import code_sha256

if TYPE_CHECKING:
    from collections.abc import Mapping

ROOT = Path(".omx/results/alphabet-broad-benchmark-3gpu-20260727")
CONFIG = Path("optimization/hosts/broad_benchmark_3gpu.local.json")
STATE = ROOT / "completion-supervisor-state.json"
Stage = Literal["stage2", "final"]
STAGE2_WORKERS_PER_HOST = {
    "secondary_gpu": 3,
    "rtx3080ti-1": 2,
    "rtx3080ti-2": 2,
}
BOOSTED_DEPLOYMENT_SCHEMA = "alphabet.broad_deployment.stage2_multiworker.v1"


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def _ssh(host: str, command: str, *, capture: bool = False) -> str:
    completed = _run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        capture=capture,
    )
    return completed.stdout.strip() if capture else ""


def _state(payload: Mapping[str, object]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def _runtime_repo(host: dict[str, object]) -> str:
    if str(host["name"]) == "secondary_gpu":
        return "<remote-home>/lnet-broad-completion-runtime-20260727"
    return "<remote-home>/lnet-broad-completion-runtime-20260727"


def _hosts(config_path: Path) -> list[dict[str, object]]:
    payload = cast(
        "dict[str, object]",
        json.loads(config_path.read_text(encoding="utf-8")),
    )
    hosts = cast("list[dict[str, object]]", payload["hosts"])
    enabled = [dict(host) for host in hosts if bool(host.get("enabled", True))]
    for host in enabled:
        host["completion_repo"] = _runtime_repo(host)
    return enabled


def _completion_deployment_config(config_path: Path) -> BroadDeploymentConfig:
    config = load_deployment_config(config_path)
    hosts = tuple(
        replace(
            host,
            data_shards=tuple(
                shard
                for shard in host.data_shards
                if host.name != "secondary_gpu" or not shard.startswith("forecasting:")
            ),
        )
        for host in config.hosts
    )
    return replace(
        config,
        schema="alphabet.broad_completion_deployment_config.cuda_capability.v1",
        hosts=hosts,
    )


def _sync_metadata(host: dict[str, object]) -> None:
    ssh_host = str(host["ssh_host"])
    repo = str(host["completion_repo"])
    remote_root = f"{repo}/{ROOT}"
    _ssh(ssh_host, f"mkdir -p {shlex.quote(remote_root)}")
    command = [
        "rsync",
        "-a",
        "--exclude=*/completed/***",
        "--exclude=*/failed/***",
        "--exclude=*/attempts/***",
        "--exclude=*/claims/***",
        f"{ROOT}/",
        f"{ssh_host}:{remote_root}/",
    ]
    _run(command)


def _stage_host(host: dict[str, object]) -> dict[str, object]:
    ssh_host = str(host["ssh_host"])
    repo = str(host["completion_repo"])
    _state(
        {
            "schema": "alphabet.broad_benchmark.completion_state.v1",
            "state": "staging_runtime",
            "host": host["name"],
        }
    )
    _ssh(
        ssh_host,
        (
            f"mkdir -p {shlex.quote(repo)}/src {shlex.quote(repo)}/scripts "
            f"{shlex.quote(repo)}/.torch_extensions"
        ),
    )
    _run(["rsync", "-a", "--checksum", "src/", f"{ssh_host}:{repo}/src/"])
    _run(
        [
            "rsync",
            "-a",
            "--checksum",
            "scripts/run_broad_benchmark_worker.py",
            f"{ssh_host}:{repo}/scripts/",
        ]
    )
    _run(["rsync", "-a", "--checksum", "csrc/", f"{ssh_host}:{repo}/csrc/"])
    _run(["rsync", "-a", "--checksum", "pyproject.toml", f"{ssh_host}:{repo}/"])
    _sync_metadata(host)
    python = str(cast("list[dict[str, object]]", host["profiles"])[0]["python"])
    hash_command = (
        "from lnet.pac_broad_benchmark_worker import code_sha256; print(code_sha256())"
    )
    remote_hash = _ssh(
        ssh_host,
        (
            f"cd {shlex.quote(repo)} && PYTHONPATH=src {shlex.quote(python)} -c "
            + shlex.quote(hash_command)
        ),
        capture=True,
    )
    expected_hash = code_sha256()
    if remote_hash != expected_hash:
        message = f"code hash mismatch on {host['name']}: {remote_hash} != {expected_hash}"
        raise RuntimeError(message)
    return {
        "host": host["name"],
        "ssh_host": ssh_host,
        "runtime_repo": repo,
        "code_sha256": remote_hash,
    }


def _collect_stage(host: dict[str, object], stage: Stage) -> None:
    ssh_host = str(host["ssh_host"])
    repo = str(host["completion_repo"])
    for bucket in ("completed", "failed", "attempts"):
        remote = f"{repo}/{ROOT}/{stage}/{bucket}"
        local = ROOT / stage / bucket
        local.mkdir(parents=True, exist_ok=True)
        _ssh(ssh_host, f"mkdir -p {shlex.quote(remote)}")
        _run(["rsync", "-a", f"{ssh_host}:{remote}/", f"{local}/"])


def _completed_keys(stage: Stage) -> set[str]:
    completed: set[str] = set()
    for path in sorted((ROOT / stage / "completed").glob("*.json")):
        try:
            row = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("status") == "done" and isinstance(row.get("job_key"), str):
            completed.add(str(row["job_key"]))
    return completed


def _worker_alive(host: str, pid: int, stage: Stage, lane: str) -> bool:
    command = (
        f"ps -p {pid} -o args= 2>/dev/null | "
        f"grep -F -- {shlex.quote(str(ROOT))} | "
        f"grep -F -- {shlex.quote(stage)} | "
        f"grep -F -- {shlex.quote(lane)} >/dev/null && echo yes || echo no"
    )
    return _ssh(host, command, capture=True) == "yes"


def _pid_file(host: dict[str, object], stage: Stage, lane: str) -> str:
    return f"{host['completion_repo']}/{ROOT}/{stage}/{lane}.pid"


def _existing_pid(
    host: dict[str, object],
    stage: Stage,
    lane: str,
) -> int | None:
    ssh_host = str(host["ssh_host"])
    raw = _ssh(
        ssh_host,
        f"cat {shlex.quote(_pid_file(host, stage, lane))} 2>/dev/null || true",
        capture=True,
    )
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if _worker_alive(ssh_host, pid, stage, lane) else None


def _launch_worker(
    host: dict[str, object],
    stage: Stage,
    lane: str,
) -> int:
    ssh_host = str(host["ssh_host"])
    repo = str(host["completion_repo"])
    python = str(cast("list[dict[str, object]]", host["profiles"])[0]["python"])
    roots = cast("dict[str, object]", host["data_roots"])
    gpu = int(cast("list[int]", host["gpus"])[0])
    stage_root = f"{repo}/{ROOT}/{stage}"
    manifest = ROOT / stage / "deployment" / f"{lane}.jsonl"
    log_dir = f"{repo}/.omx/logs/alphabet-broad-benchmark-completion-20260727"
    log = f"{log_dir}/{stage}-{lane}.log"
    extension_cache = (
        ""
        if str(host["name"]) == "secondary_gpu"
        else f" TORCH_EXTENSIONS_DIR={shlex.quote(f'{repo}/.torch_extensions')}"
    )
    command = [
        f"cd {shlex.quote(repo)}",
        "&&",
        f"echo $$ > {shlex.quote(_pid_file(host, stage, lane))}",
        "&&",
        "exec",
        (
            "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
            f"{extension_cache}"
        ),
        f"PYTHONPATH=src CUDA_VISIBLE_DEVICES={gpu}",
        shlex.quote(python),
        "scripts/run_broad_benchmark_worker.py",
        f"--root {shlex.quote(str(ROOT))}",
        f"--manifest {shlex.quote(str(manifest))}",
        "--device cuda",
        f"--ucr-data-root {shlex.quote(str(roots['ucr']))}",
        f"--external-data-root {shlex.quote(str(roots['external']))}",
    ]
    if roots.get("physionet2012"):
        command.append(
            f"--physionet2012-data-root {shlex.quote(str(roots['physionet2012']))}"
        )
    command.extend(("--max-attempts 3", f">>{shlex.quote(log)} 2>&1"))
    directories = (
        log_dir,
        f"{stage_root}/completed",
        f"{stage_root}/failed",
        f"{stage_root}/attempts",
        f"{stage_root}/claims",
    )
    _ssh(
        ssh_host,
        (
            "mkdir -p "
            + " ".join(shlex.quote(path) for path in directories)
            + " && setsid -f sh -lc "
            + shlex.quote(" ".join(command))
            + " </dev/null >/dev/null 2>&1"
        ),
    )
    for _ in range(50):
        pid = _existing_pid(host, stage, lane)
        if pid is not None:
            return pid
        time.sleep(0.1)
    message = f"{stage}/{lane} failed to publish a live worker PID"
    raise RuntimeError(message)


def _remote_counts(
    host: dict[str, object],
    stage: Stage,
) -> dict[str, int]:
    ssh_host = str(host["ssh_host"])
    stage_root = f"{host['completion_repo']}/{ROOT}/{stage}"
    output = _ssh(
        ssh_host,
        (
            "for bucket in completed failed claims; do "
            f"printf '%s=' \"$bucket\"; find {shlex.quote(stage_root)}/$bucket "
            "-maxdepth 1 -type f 2>/dev/null | wc -l; done"
        ),
        capture=True,
    )
    return {
        key: int(value)
        for line in output.splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }


def _deployment_units(stage: Stage) -> list[dict[str, object]]:
    payload = cast(
        "dict[str, object]",
        json.loads((ROOT / stage / "deployment.json").read_text(encoding="utf-8")),
    )
    return cast("list[dict[str, object]]", payload["units"])


def _run_stage(  # noqa: C901, PLR0912 - retries and barriers stay in one state machine
    stage: Stage,
    hosts: list[dict[str, object]],
    *,
    poll_seconds: int,
) -> None:
    expected_jobs = stage_jobs(ROOT, stage)
    expected = {job.key for job in expected_jobs}
    units = _deployment_units(stage)
    unit_jobs: dict[str, tuple[BenchmarkJob, ...]] = {
        str(cast("dict[str, object]", unit["lane"])["name"]): stage_jobs_from_manifest(
            Path(str(unit["manifest"]))
        )
        for unit in units
    }
    host_by_name = {str(host["name"]): host for host in hosts}
    for retry_pass in range(1, 4):
        for host in hosts:
            _collect_stage(host, stage)
        completed = _completed_keys(stage)
        unexpected = completed - expected
        if unexpected:
            message = f"{stage} collected {len(unexpected)} unexpected results"
            raise RuntimeError(message)
        if completed == expected:
            return
        for host in hosts:
            _sync_metadata(host)
        launched: list[dict[str, object]] = []
        for unit in units:
            lane_info = cast("dict[str, object]", unit["lane"])
            lane = str(lane_info["name"])
            pending = {job.key for job in unit_jobs[lane]} - completed
            if not pending:
                continue
            host = host_by_name[str(lane_info["host"])]
            pid = _existing_pid(host, stage, lane)
            resumed = pid is not None
            if pid is None:
                pid = _launch_worker(host, stage, lane)
            launched.append(
                {
                    "host": host["name"],
                    "ssh_host": host["ssh_host"],
                    "lane": lane,
                    "pid": pid,
                    "pending_jobs": len(pending),
                    "resumed_existing_worker": resumed,
                }
            )
        if not launched:
            message = f"{stage} has {len(expected - completed)} missing jobs and no live lane"
            raise RuntimeError(message)
        while True:
            alive = [
                item
                for item in launched
                if _worker_alive(
                    str(item["ssh_host"]),
                    int(cast("int", item["pid"])),
                    stage,
                    str(item["lane"]),
                )
            ]
            counts = {
                str(host["name"]): _remote_counts(host, stage)
                for host in hosts
            }
            _state(
                {
                    "schema": "alphabet.broad_benchmark.completion_state.v1",
                    "state": f"{stage}_running",
                    "retry_pass": retry_pass,
                    "expected": len(expected),
                    "completed_before_launch": len(completed),
                    "active_lanes": [item["lane"] for item in alive],
                    "remote_counts": counts,
                }
            )
            if not alive:
                break
            time.sleep(poll_seconds)
    for host in hosts:
        _collect_stage(host, stage)
    missing = expected - _completed_keys(stage)
    if missing:
        message = f"{stage} exhausted retries with {len(missing)} missing jobs"
        raise RuntimeError(message)


def stage_jobs_from_manifest(path: Path) -> tuple[BenchmarkJob, ...]:
    return tuple(
        BenchmarkJob.from_payload(cast("dict[str, object]", json.loads(line)))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _boost_stage2_deployment(deployment: dict[str, object]) -> dict[str, object]:
    if deployment.get("schema") == BOOSTED_DEPLOYMENT_SCHEMA:
        return deployment
    boosted_units: list[dict[str, object]] = []
    for unit in cast("list[dict[str, object]]", deployment["units"]):
        lane = cast("dict[str, object]", unit["lane"])
        host = str(lane["host"])
        worker_count = STAGE2_WORKERS_PER_HOST.get(host, 1)
        jobs = stage_jobs_from_manifest(Path(str(unit["manifest"])))
        grouped: dict[str, list[BenchmarkJob]] = {}
        for job in jobs:
            grouped.setdefault(job.comparison_group, []).append(job)
        groups = sorted(
            grouped.values(),
            key=lambda items: (
                -sum(job.estimated_seconds for job in items),
                items[0].comparison_group,
            ),
        )
        buckets: list[list[BenchmarkJob]] = [[] for _ in range(worker_count)]
        loads = [0.0] * worker_count
        for group in groups:
            index = min(range(worker_count), key=lambda item: (loads[item], item))
            buckets[index].extend(group)
            loads[index] += sum(job.estimated_seconds for job in group)
        raw_total = sum(loads)
        wall_scale = (
            float(cast("str | int | float", unit["estimated_wall_seconds"])) / raw_total
            if raw_total
            else 1.0
        )
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            lane_name = f"{lane['name']}-w{index}"
            manifest = ROOT / "stage2" / "deployment" / f"{lane_name}.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps(job.payload(), sort_keys=True) + "\n"
                    for job in sorted(
                        bucket,
                        key=lambda item: (-item.estimated_seconds, item.key),
                    )
                ),
                encoding="utf-8",
            )
            boosted_units.append(
                {
                    **unit,
                    "comparison_groups": len(
                        {job.comparison_group for job in bucket}
                    ),
                    "estimated_wall_seconds": loads[index] * wall_scale,
                    "jobs": len(bucket),
                    "lane": {**lane, "name": lane_name},
                    "manifest": str(manifest),
                }
            )
    boosted = {
        **deployment,
        "schema": BOOSTED_DEPLOYMENT_SCHEMA,
        "strategy": (
            f"{deployment['strategy']}; comparison-group LPT multiworker expansion"
        ),
        "units": boosted_units,
        "max_estimated_wall_seconds": max(
            (
                float(cast("str | int | float", unit["estimated_wall_seconds"]))
                for unit in boosted_units
            ),
            default=0.0,
        ),
    }
    path = ROOT / "stage2" / "deployment.json"
    temporary = path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(boosted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return boosted


def _plan(stage: Stage, config_path: Path) -> dict[str, object]:
    deployment_path = ROOT / stage / "deployment.json"
    if stage == "stage2" and deployment_path.is_file():
        frozen = cast(
            "dict[str, object]",
            json.loads(deployment_path.read_text(encoding="utf-8")),
        )
        if frozen.get("schema") == BOOSTED_DEPLOYMENT_SCHEMA:
            audit = audit_deployment(ROOT, stage=stage)
            if not audit["ok"]:
                message = f"{stage} boosted deployment audit failed: {audit}"
                raise RuntimeError(message)
            return frozen
    deployment = plan_jobs(
        ROOT,
        stage=stage,
        jobs=stage_jobs(ROOT, stage),
        config=_completion_deployment_config(config_path),
    )
    if stage == "stage2":
        deployment = _boost_stage2_deployment(deployment)
    audit = audit_deployment(ROOT, stage=stage)
    if not audit["ok"]:
        message = f"{stage} deployment audit failed: {audit}"
        raise RuntimeError(message)
    return deployment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--poll-seconds", type=int, default=30)
    arguments = parser.parse_args()
    if arguments.poll_seconds < 5:
        message = "poll interval must be at least five seconds"
        raise ValueError(message)
    lock_path = ROOT / "completion-supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        hosts = _hosts(arguments.config)
        selection1 = select_stage1(ROOT)
        deployment2 = _plan("stage2", arguments.config)
        staged = [_stage_host(host) for host in hosts]
        _state(
            {
                "schema": "alphabet.broad_benchmark.completion_state.v1",
                "state": "stage2_ready",
                "stage1_selection": selection1,
                "stage2_deployment": deployment2,
                "hosts": staged,
            }
        )
        _run_stage("stage2", hosts, poll_seconds=arguments.poll_seconds)
        selection2 = select_stage2(ROOT)
        deployment_final = _plan("final", arguments.config)
        for host in hosts:
            _sync_metadata(host)
        _state(
            {
                "schema": "alphabet.broad_benchmark.completion_state.v1",
                "state": "final_ready",
                "stage2_selection": selection2,
                "final_deployment": deployment_final,
            }
        )
        _run_stage("final", hosts, poll_seconds=arguments.poll_seconds)
        audit = audit_campaign(ROOT)
        if not audit["ok"]:
            message = f"broad completion audit failed: {audit}"
            raise RuntimeError(message)
        _state(
            {
                "schema": "alphabet.broad_benchmark.completion_state.v1",
                "state": "complete",
                "audit": audit,
            }
        )


if __name__ == "__main__":
    main()
