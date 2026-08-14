# ruff: noqa: S603
"""Run the complete new-dataset campaign after the active GPU wave exits."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from lnet.pac_broad_benchmark_distributed import (
    audit_deployment,
    load_deployment_config,
    plan_jobs,
)
from lnet.pac_broad_benchmark_worker import code_sha256, load_manifest
from lnet.pac_broad_followup_queue import (
    EXTERNAL_TASKS,
    FOLLOWUP_ROOT,
    NEW_UCR_83,
    audit_campaign,
    select_stage1,
    select_stage2,
)

ACTIVE_ROOT = Path(".omx/results/alphabet-broad-benchmark-3gpu-20260727")
ACTIVE_NONFINITE_REPAIR_ROOT = ACTIVE_ROOT / "stage1" / "nonfinite-repair"
DEFAULT_CONFIG = Path("optimization/hosts/broad_followup_3gpu.local.json")
Stage = Literal["stage1", "stage2", "final"]
MULTIWORKER_SCHEMA = "alphabet.broad_new_datasets.multiworker_deployment.v1"
SEARCH_WORKERS_PER_HOST = {
    "secondary_gpu": 3,
    "rtx3080ti-1": 2,
    "rtx3080ti-2": 2,
}

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lnet.pac_broad_benchmark_queue import BenchmarkJob


def _run(
    command: list[str],
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )


def _ssh(host: str, command: str, *, capture: bool = False) -> str:
    completed = _run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        capture=capture,
    )
    return completed.stdout.strip() if capture else ""


def _state(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _worker_alive(host: str, pid: int, root: str, lane: str) -> bool:
    command = (
        f"ps -p {pid} -o args= 2>/dev/null | "
        f"grep -F -- {shlex.quote(root)} | "
        f"grep -F -- {shlex.quote(lane)} >/dev/null && echo yes || echo no"
    )
    return _ssh(host, command, capture=True) == "yes"


def _remote_count(host: str, directory: str) -> int:
    output = _ssh(
        host,
        f"find {shlex.quote(directory)} -maxdepth 1 -type f 2>/dev/null | wc -l",
        capture=True,
    )
    return int(output or 0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_active_nonfinite_repair(
    workers: list[dict[str, object]],
) -> None:
    """Deploy the audited UCR repair only after the currently loaded workers exit."""
    normalization_source = Path("src/lnet/pac_eval_sections.py")
    worker_source = ACTIVE_NONFINITE_REPAIR_ROOT / "pac_broad_benchmark_worker.py"
    expected = {
        "pac_eval_sections.py": _sha256(normalization_source),
        "pac_broad_benchmark_worker.py": _sha256(worker_source),
    }
    deployed: list[dict[str, object]] = []
    for worker in workers:
        host = str(worker["host"])
        repo = str(worker["active_repo"])
        destinations = {
            normalization_source: f"{repo}/src/lnet/pac_eval_sections.py",
            worker_source: f"{repo}/src/lnet/pac_broad_benchmark_worker.py",
        }
        for source, destination in destinations.items():
            _run(["rsync", "-a", "--checksum", str(source), f"{host}:{destination}"])
        remote_hashes = _ssh(
            host,
            "sha256sum "
            + " ".join(shlex.quote(destination) for destination in destinations.values()),
            capture=True,
        )
        observed = {line.split()[0] for line in remote_hashes.splitlines()}
        if observed != set(expected.values()):
            message = f"active nonfinite repair hash mismatch on {host}: {remote_hashes}"
            raise RuntimeError(message)
        deployed.append(
            {
                "host": host,
                "lane": worker["lane"],
                "active_repo": repo,
                "verified_sha256": sorted(observed),
            }
        )
    _state(
        ACTIVE_NONFINITE_REPAIR_ROOT / "deployment.json",
        {
            "schema": "alphabet.broad_benchmark.nonfinite_repair.v1",
            "deployed_at": datetime.now().astimezone().isoformat(),
            "reason": (
                "UCR NaN/Inf values made validation losses and checkpoint selection "
                "non-finite"
            ),
            "policy": (
                "fit scalar mean/std on finite optimization-TRAIN observations only; "
                "map non-finite inputs to normalized zero for every model"
            ),
            "finite_input_compatibility": "bitwise identical normalization path",
            "source_sha256": expected,
            "deployments": deployed,
        },
    )


def _active_worker_command(
    host: dict[str, object],
    lane: str,
) -> str:
    repo = str(host["active_repo"])
    python = str(host["python"])
    roots = cast("dict[str, object]", host["data_roots"])
    parts = [
        f"cd {shlex.quote(repo)} &&",
        "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
        f"PYTHONPATH=src CUDA_VISIBLE_DEVICES={int(cast('int', host['gpu']))}",
        shlex.quote(python),
        "scripts/run_broad_benchmark_worker.py",
        f"--root {shlex.quote(str(ACTIVE_ROOT))}",
        (
            "--manifest "
            + shlex.quote(str(ACTIVE_ROOT / "stage1" / "deployment" / f"{lane}.jsonl"))
        ),
        "--device cuda",
        f"--ucr-data-root {shlex.quote(str(roots['ucr']))}",
        f"--external-data-root {shlex.quote(str(roots['external']))}",
    ]
    if roots.get("physionet2012"):
        parts.append(
            f"--physionet2012-data-root {shlex.quote(str(roots['physionet2012']))}"
        )
    parts.append("--max-attempts 3")
    log = f".omx/logs/alphabet-broad-benchmark-3gpu-20260727/{lane}-retry.log"
    return " ".join(parts) + f" >>{shlex.quote(log)} 2>&1"


def _wait_active(
    workers: list[dict[str, object]],
    *,
    poll_seconds: int,
    state_path: Path,
) -> None:
    for retry_pass in range(1, 4):
        while True:
            alive = [
                str(worker["lane"])
                for worker in workers
                if _worker_alive(
                    str(worker["host"]),
                    int(cast("int", worker["pid"])),
                    str(ACTIVE_ROOT),
                    str(worker["lane"]),
                )
            ]
            _state(
                state_path,
                {
                    "schema": "alphabet.broad_new_datasets.watcher_state.v1",
                    "state": "waiting_for_active_stage1",
                    "active_lanes": alive,
                    "active_retry_pass": retry_pass,
                },
            )
            if not alive:
                break
            time.sleep(poll_seconds)

        incomplete: list[dict[str, object]] = []
        for worker in workers:
            host = str(worker["host"])
            repo = str(worker["active_repo"])
            expected = int(cast("int", worker["jobs"]))
            completed = _remote_count(
                host,
                f"{repo}/{ACTIVE_ROOT}/stage1/completed",
            )
            if completed < expected:
                incomplete.append({**worker, "completed": completed})
        if not incomplete:
            return
        if retry_pass == 3:
            message = f"active Stage 1 has terminal incomplete lanes: {incomplete}"
            raise RuntimeError(message)

        if retry_pass == 1:
            _stage_active_nonfinite_repair(incomplete)

        for worker in incomplete:
            host = str(worker["host"])
            lane = str(worker["lane"])
            command = _active_worker_command(worker, lane)
            pid = int(
                _ssh(
                    host,
                    (
                        f"cd {shlex.quote(str(worker['active_repo']))} && "
                        f"nohup sh -lc {shlex.quote(command)} </dev/null >/dev/null 2>&1 "
                        "& echo $!"
                    ),
                    capture=True,
                )
            )
            worker["pid"] = pid


def _rsync_files(
    source_root: Path,
    relative_files: list[str],
    host: str,
    destination: str,
    list_path: Path,
) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(relative_files) + "\n", encoding="utf-8")
    _run(
        [
            "rsync",
            "-a",
            "--checksum",
            "--partial",
            "--files-from",
            str(list_path),
            f"{source_root}/",
            f"{host}:{destination}/",
        ]
    )


def _stage_followup_host(
    host: dict[str, object],
    *,
    config_path: Path,
    state_dir: Path,
) -> None:
    name = str(host["name"])
    ssh_host = str(host["ssh_host"])
    repo = str(host["repo"])
    roots = cast("dict[str, object]", host["data_roots"])
    _ssh(
        ssh_host,
        "mkdir -p "
        + " ".join(
            shlex.quote(path)
            for path in (
                repo,
                f"{repo}/src",
                f"{repo}/scripts",
                f"{repo}/csrc",
                f"{repo}/.torch_extensions",
                f"{repo}/{FOLLOWUP_ROOT}",
                str(roots["ucr"]),
                f"{roots['external']}/selection-only",
            )
        ),
    )
    _run(["rsync", "-a", "src/", f"{ssh_host}:{repo}/src/"])
    _run(["rsync", "-a", "csrc/", f"{ssh_host}:{repo}/csrc/"])
    _run(
        [
            "rsync",
            "-a",
            "scripts/run_broad_benchmark_worker.py",
            f"{ssh_host}:{repo}/scripts/",
        ]
    )
    _run(["rsync", "-a", "pyproject.toml", f"{ssh_host}:{repo}/"])
    python = str(cast("list[dict[str, object]]", host["profiles"])[0]["python"])
    expected_hash = code_sha256()
    observed_hash = _ssh(
        ssh_host,
        (
            f"cd {shlex.quote(repo)} && PYTHONPATH=src {shlex.quote(python)} "
            "-c 'from lnet.pac_broad_benchmark_worker import code_sha256; "
            "print(code_sha256())'"
        ),
        capture=True,
    )
    if observed_hash != expected_hash:
        message = (
            f"follow-up execution hash mismatch on {name}: "
            f"{observed_hash} != {expected_hash}"
        )
        raise RuntimeError(message)
    _run(
        [
            "rsync",
            "-a",
            f"{FOLLOWUP_ROOT}/",
            f"{ssh_host}:{repo}/{FOLLOWUP_ROOT}/",
        ]
    )

    ucr_files = [
        relative
        for dataset in NEW_UCR_83
        for relative in (
            f"{dataset}/{dataset}_TRAIN.tsv",
            f"{dataset}/{dataset}_TEST.tsv",
        )
    ]
    _rsync_files(
        Path(".omx/data/ucr"),
        ucr_files,
        ssh_host,
        str(roots["ucr"]),
        state_dir / f"{name}-ucr-files.txt",
    )

    allowed_shards = set(cast("list[str]", host["data_shards"]))
    external_files = [
        f"{spec.key}.pt"
        for spec in EXTERNAL_TASKS
        if spec.data_shard in allowed_shards
    ]
    _rsync_files(
        Path("data/external/selection-only"),
        external_files,
        ssh_host,
        f"{roots['external']}/selection-only",
        state_dir / f"{name}-external-selection-files.txt",
    )
    full_external_files = [
        spec.source_artifact
        for spec in EXTERNAL_TASKS
        if spec.data_shard in allowed_shards
    ]
    _rsync_files(
        Path("data/external"),
        full_external_files,
        ssh_host,
        str(roots["external"]),
        state_dir / f"{name}-external-full-files.txt",
    )
    _run(["rsync", "-a", str(config_path), f"{ssh_host}:{repo}/"])


def _sync_followup_artifacts(host: dict[str, object]) -> None:
    ssh_host = str(host["ssh_host"])
    repo = str(host["repo"])
    _ssh(ssh_host, f"mkdir -p {shlex.quote(f'{repo}/{FOLLOWUP_ROOT}')}")
    _run(
        [
            "rsync",
            "-a",
            f"{FOLLOWUP_ROOT}/",
            f"{ssh_host}:{repo}/{FOLLOWUP_ROOT}/",
        ]
    )


def _collect_stage(host: dict[str, object], stage: Stage) -> None:
    ssh_host = str(host["ssh_host"])
    repo = str(host["repo"])
    for bucket in ("completed", "failed", "attempts"):
        remote = f"{repo}/{FOLLOWUP_ROOT}/{stage}/{bucket}"
        local = FOLLOWUP_ROOT / stage / bucket
        local.mkdir(parents=True, exist_ok=True)
        _ssh(ssh_host, f"mkdir -p {shlex.quote(remote)}")
        _run(["rsync", "-a", f"{ssh_host}:{remote}/", f"{local}/"])


def _completed_keys(stage: Stage) -> set[str]:
    completed: set[str] = set()
    for path in sorted((FOLLOWUP_ROOT / stage / "completed").glob("*.json")):
        try:
            row = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("status") == "done" and isinstance(row.get("job_key"), str):
            completed.add(str(row["job_key"]))
    return completed


def _launch_followup(
    host: dict[str, object],
    lane: str,
    stage: Stage,
) -> int:
    ssh_host = str(host["ssh_host"])
    repo = str(host["repo"])
    roots = cast("dict[str, object]", host["data_roots"])
    python = str(cast("list[dict[str, object]]", host["profiles"])[0]["python"])
    gpu = int(cast("list[int]", host["gpus"])[0])
    manifest = FOLLOWUP_ROOT / stage / "deployment" / f"{lane}.jsonl"
    log_dir = f"{repo}/.omx/logs/alphabet-broad-new-datasets-3gpu-20260727"
    log = f"{log_dir}/{stage}-{lane}.log"
    stage_root = f"{repo}/{FOLLOWUP_ROOT}/{stage}"
    pid_file = f"{stage_root}/{lane}.pid"
    extension_cache = (
        ""
        if str(host["name"]) == "secondary_gpu"
        else f" TORCH_EXTENSIONS_DIR={shlex.quote(f'{repo}/.torch_extensions')}"
    )
    command = " ".join(
        (
            f"cd {shlex.quote(repo)}",
            "&&",
            f"echo $$ > {shlex.quote(pid_file)}",
            "&&",
            "exec",
            (
                "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
                f"{extension_cache}"
            ),
            f"PYTHONPATH=src CUDA_VISIBLE_DEVICES={gpu}",
            shlex.quote(python),
            "scripts/run_broad_benchmark_worker.py",
            f"--root {shlex.quote(str(FOLLOWUP_ROOT))}",
            f"--manifest {shlex.quote(str(manifest))}",
            "--device cuda",
            f"--ucr-data-root {shlex.quote(str(roots['ucr']))}",
            f"--external-data-root {shlex.quote(str(roots['external']))}",
            "--max-attempts 3",
            f">>{shlex.quote(log)} 2>&1",
        )
    )
    _ssh(
        ssh_host,
        (
            "mkdir -p "
            + " ".join(
                shlex.quote(path)
                for path in (
                    log_dir,
                    f"{stage_root}/completed",
                    f"{stage_root}/failed",
                    f"{stage_root}/attempts",
                    f"{stage_root}/claims",
                )
            )
            + " && "
            + f"setsid -f sh -lc {shlex.quote(command)} </dev/null >/dev/null 2>&1"
        )
    )
    for _ in range(50):
        pid = _existing_followup_pid(host, lane, stage)
        if pid is not None:
            return pid
        time.sleep(0.1)
    message = f"{stage}/{lane} did not publish a live worker PID on {ssh_host}"
    raise RuntimeError(message)


def _existing_followup_pid(
    host: dict[str, object],
    lane: str,
    stage: Stage,
) -> int | None:
    ssh_host = str(host["ssh_host"])
    repo = str(host["repo"])
    pid_file = f"{repo}/{FOLLOWUP_ROOT}/{stage}/{lane}.pid"
    raw_pid = _ssh(
        ssh_host,
        f"cat {shlex.quote(pid_file)} 2>/dev/null || true",
        capture=True,
    )
    try:
        pid = int(raw_pid)
    except ValueError:
        return None
    if _worker_alive(ssh_host, pid, str(FOLLOWUP_ROOT), lane):
        return pid
    return None


def _deployment_units(stage: Stage) -> list[dict[str, object]]:
    payload = cast(
        "dict[str, object]",
        json.loads(
            (FOLLOWUP_ROOT / stage / "deployment.json").read_text(encoding="utf-8")
        ),
    )
    return cast("list[dict[str, object]]", payload["units"])


def _plan_stage(stage: Stage, config_path: Path) -> None:
    deployment_path = FOLLOWUP_ROOT / stage / "deployment.json"
    if stage != "final" and deployment_path.is_file():
        frozen = cast(
            "dict[str, object]",
            json.loads(deployment_path.read_text(encoding="utf-8")),
        )
        if frozen.get("schema") == MULTIWORKER_SCHEMA:
            audit = audit_deployment(FOLLOWUP_ROOT, stage=stage)
            if not audit["ok"]:
                message = f"{stage} frozen deployment audit failed: {audit}"
                raise RuntimeError(message)
            return
    jobs = load_manifest(FOLLOWUP_ROOT / stage / "master.jsonl")
    deployment = plan_jobs(
        FOLLOWUP_ROOT,
        stage=stage,
        jobs=jobs,
        config=load_deployment_config(config_path),
    )
    if stage != "final":
        _expand_search_deployment(stage, deployment)
    audit = audit_deployment(FOLLOWUP_ROOT, stage=stage)
    if not audit["ok"]:
        message = f"{stage} deployment audit failed: {audit}"
        raise RuntimeError(message)


def _expand_search_deployment(
    stage: Literal["stage1", "stage2"],
    deployment: dict[str, object],
) -> None:
    units: list[dict[str, object]] = []
    for unit in cast("list[dict[str, object]]", deployment["units"]):
        lane = cast("dict[str, object]", unit["lane"])
        worker_count = SEARCH_WORKERS_PER_HOST.get(str(lane["host"]), 1)
        jobs = load_manifest(Path(str(unit["manifest"])))
        grouped: dict[str, list[BenchmarkJob]] = {}
        for job in jobs:
            grouped.setdefault(job.comparison_group, []).append(job)
        groups = sorted(
            grouped.values(),
            key=lambda items: (
                -sum(
                    job.estimated_seconds for job in items
                ),
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
        scale = (
            float(cast("str | int | float", unit["estimated_wall_seconds"]))
            / raw_total
            if raw_total
            else 1.0
        )
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            lane_name = f"{lane['name']}-w{index}"
            manifest = FOLLOWUP_ROOT / stage / "deployment" / f"{lane_name}.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps(job.payload(), sort_keys=True) + "\n"
                    for job in sorted(
                        bucket,
                        key=lambda item: (
                            -item.estimated_seconds,
                            item.key,
                        ),
                    )
                ),
                encoding="utf-8",
            )
            units.append(
                {
                    **unit,
                    "comparison_groups": len(
                        {job.comparison_group for job in bucket}
                    ),
                    "estimated_wall_seconds": loads[index] * scale,
                    "jobs": len(bucket),
                    "lane": {**lane, "name": lane_name},
                    "manifest": str(manifest),
                }
            )
    expanded = {
        **deployment,
        "schema": MULTIWORKER_SCHEMA,
        "strategy": f"{deployment['strategy']}; comparison-group LPT multiworker",
        "units": units,
        "max_estimated_wall_seconds": max(
            (
                float(cast("str | int | float", unit["estimated_wall_seconds"]))
                for unit in units
            ),
            default=0.0,
        ),
    }
    path = FOLLOWUP_ROOT / stage / "deployment.json"
    temporary = path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(expanded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_followup_stage(  # noqa: C901, PLR0912 - stage barrier and retry state stay atomic
    stage: Stage,
    hosts: list[dict[str, object]],
    *,
    poll_seconds: int,
    state_path: Path,
) -> None:
    host_by_name = {str(host["name"]): host for host in hosts}
    expected = {
        job.key for job in load_manifest(FOLLOWUP_ROOT / stage / "master.jsonl")
    }
    units = _deployment_units(stage)
    unit_keys = {
        str(cast("dict[str, object]", unit["lane"])["name"]): {
            job.key for job in load_manifest(Path(str(unit["manifest"])))
        }
        for unit in units
    }
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
            _sync_followup_artifacts(host)
        launched: list[dict[str, object]] = []
        for unit in units:
            lane_info = cast("dict[str, object]", unit["lane"])
            lane = str(lane_info["name"])
            pending = unit_keys[lane] - completed
            if not pending:
                continue
            host = host_by_name[str(lane_info["host"])]
            pid = _existing_followup_pid(host, lane, stage)
            resumed = pid is not None
            if pid is None:
                pid = _launch_followup(host, lane, stage)
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
            message = f"{stage} has {len(expected - completed)} missing jobs but no runnable lane"
            raise RuntimeError(message)
        launch_payload: dict[str, object] = {
            "schema": "alphabet.broad_new_datasets.stage_launch.v1",
            "stage": stage,
            "retry_pass": retry_pass,
            "expected": len(expected),
            "completed_before_launch": len(completed),
            "launched": launched,
        }
        _state(FOLLOWUP_ROOT / stage / "launch.json", launch_payload)
        while True:
            alive = [
                item
                for item in launched
                if _worker_alive(
                    str(item["ssh_host"]),
                    int(cast("int", item["pid"])),
                    str(FOLLOWUP_ROOT),
                    str(item["lane"]),
                )
            ]
            _state(
                state_path,
                {
                    "schema": "alphabet.broad_new_datasets.watcher_state.v1",
                    "state": f"followup_{stage}_running",
                    "retry_pass": retry_pass,
                    "completed_before_launch": len(completed),
                    "expected": len(expected),
                    "active_lanes": [item["lane"] for item in alive],
                },
            )
            if not alive:
                break
            time.sleep(poll_seconds)

    for host in hosts:
        _collect_stage(host, stage)
    completed = _completed_keys(stage)
    missing = expected - completed
    if missing:
        message = f"{stage} exhausted retries with {len(missing)} incomplete jobs"
        raise RuntimeError(message)


def _require_final_audit() -> dict[str, object]:
    audit = audit_campaign(FOLLOWUP_ROOT)
    if not audit["ok"]:
        message = f"follow-up final audit failed: {audit}"
        raise RuntimeError(message)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--state",
        type=Path,
        default=FOLLOWUP_ROOT / "watcher-state.json",
    )
    arguments = parser.parse_args()
    if arguments.poll_seconds < 5:
        message = "poll interval must be at least five seconds"
        raise ValueError(message)

    active_release = cast(
        "dict[str, object]",
        json.loads((ACTIVE_ROOT / "stage1" / "release.json").read_text(encoding="utf-8")),
    )
    followup_config = cast(
        "dict[str, object]",
        json.loads(arguments.config.read_text(encoding="utf-8")),
    )
    configured_hosts = {
        str(host["name"]): host
        for host in cast("list[dict[str, object]]", followup_config["hosts"])
    }
    configured_transports = {
        str(host["ssh_host"]): host
        for host in cast("list[dict[str, object]]", followup_config["hosts"])
    }
    workers: list[dict[str, object]] = []
    for released in cast("list[dict[str, object]]", active_release["workers"]):
        name = str(released["host"])
        configured = configured_hosts.get(name) or configured_transports[name]
        profiles = cast("list[dict[str, object]]", configured["profiles"])
        workers.append(
            {
                "host": str(configured["ssh_host"]),
                "lane": str(released["lane"]),
                "pid": int(cast("int", released["pid_at_release"])),
                "jobs": int(cast("int", released["jobs"])),
                "active_repo": (
                    "<remote-home>/lnet-broad-runtime-20260727"
                    if str(configured["name"]) == "secondary_gpu"
                    else "<remote-home>/lnet-broad-runtime-20260727"
                ),
                "python": str(profiles[0]["python"]),
                "gpu": int(cast("list[int]", configured["gpus"])[0]),
                "data_roots": configured["data_roots"],
            }
        )

    lock_path = FOLLOWUP_ROOT / "watcher.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = f"another follow-up watcher owns {lock_path}"
            raise RuntimeError(message) from error
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        try:
            _wait_active(
                workers,
                poll_seconds=arguments.poll_seconds,
                state_path=arguments.state,
            )
            hosts = cast("list[dict[str, object]]", followup_config["hosts"])
            for host in hosts:
                _state(
                    arguments.state,
                    {
                        "schema": "alphabet.broad_new_datasets.watcher_state.v1",
                        "state": "staging_followup",
                        "host": host["name"],
                    },
                )
                _stage_followup_host(
                    host,
                    config_path=arguments.config,
                    state_dir=arguments.state.parent / "transfer-manifests",
                )

            _plan_stage("stage1", arguments.config)
            _run_followup_stage(
                "stage1",
                hosts,
                poll_seconds=arguments.poll_seconds,
                state_path=arguments.state,
            )
            select_stage1(FOLLOWUP_ROOT)
            _plan_stage("stage2", arguments.config)
            _run_followup_stage(
                "stage2",
                hosts,
                poll_seconds=arguments.poll_seconds,
                state_path=arguments.state,
            )
            select_stage2(FOLLOWUP_ROOT)
            _plan_stage("final", arguments.config)
            _run_followup_stage(
                "final",
                hosts,
                poll_seconds=arguments.poll_seconds,
                state_path=arguments.state,
            )
            audit = _require_final_audit()
            payload: dict[str, object] = {
                "schema": "alphabet.broad_new_datasets.watcher_state.v1",
                "state": "complete",
                "audit": audit,
            }
            _state(arguments.state, payload)
            _state(FOLLOWUP_ROOT / "release.json", payload)
        except Exception as error:
            _state(
                arguments.state,
                {
                    "schema": "alphabet.broad_new_datasets.watcher_state.v1",
                    "state": "blocked",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


if __name__ == "__main__":
    main()
