# ruff: noqa: S603
"""Drive a frozen broad campaign through Stage 1, Stage 2, and final."""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

from lnet.pac_broad_benchmark_completion import (
    audit_campaign,
    select_stage1,
    select_stage2,
    stage_jobs,
)
from lnet.pac_broad_benchmark_distributed import (
    audit_deployment,
    load_deployment_config,
    plan_jobs,
)
from lnet.pac_broad_benchmark_queue import BenchmarkJob

ROOT = Path(
    os.environ.get(
        "ALPHABET_PRIORITY_ROOT",
        ".omx/results/alphabet-new3-10model-5gpu-20260727",
    )
)
CONFIG = Path(
    os.environ.get(
        "ALPHABET_PRIORITY_CONFIG",
        "optimization/hosts/new3_10model_5gpu.local.json",
    )
)
STATE = ROOT / "supervisor-state.json"
Stage = Literal["stage1", "stage2", "final"]
WORKERS_PER_HOST = {"secondary_gpu": 3, "rtx3080ti-1": 2, "rtx3080ti-2": 2}


def _run(command: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=capture)
    return completed.stdout.strip() if capture else ""


def _ssh(host: str, command: str, *, capture: bool = False) -> str:
    return _run(["ssh", "-o", "BatchMode=yes", host, command], capture=capture)


def _write_state(payload: dict[str, object]) -> None:
    temporary = STATE.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE)


def _hosts() -> list[dict[str, object]]:
    payload = cast("dict[str, object]", json.loads(CONFIG.read_text(encoding="utf-8")))
    return [
        host
        for host in cast("list[dict[str, object]]", payload["hosts"])
        if bool(host.get("enabled", True))
    ]


def _stage_host(host: dict[str, object]) -> None:
    ssh_host, repo = str(host["ssh_host"]), str(host["repo"])
    remote_root = f"{repo}/{ROOT}"
    frozen_runtime = ROOT / "frozen-runtime"
    source_root = frozen_runtime if frozen_runtime.is_dir() else Path()
    _ssh(
        ssh_host,
        (
            f"mkdir -p {shlex.quote(remote_root)} {shlex.quote(repo + '/src')} "
            f"{shlex.quote(repo + '/scripts')} {shlex.quote(repo + '/csrc')}"
        ),
    )
    _run(
        [
            "rsync",
            "-a",
            "--checksum",
            "--delete",
            f"{source_root}/src/",
            f"{ssh_host}:{repo}/src/",
        ]
    )
    _run(
        [
            "rsync",
            "-a",
            "--checksum",
            "--delete",
            f"{source_root}/csrc/",
            f"{ssh_host}:{repo}/csrc/",
        ]
    )
    _run(
        [
            "rsync",
            "-a",
            "--checksum",
            str(source_root / "scripts/run_broad_benchmark_worker.py"),
            f"{ssh_host}:{repo}/scripts/",
        ]
    )
    _run(
        [
            "rsync",
            "-a",
            "--checksum",
            str(source_root / "pyproject.toml"),
            f"{ssh_host}:{repo}/",
        ]
    )
    _run(
        [
            "rsync",
            "-a",
            "--checksum",
            "--exclude",
            "frozen-runtime/",
            f"{ROOT}/",
            f"{ssh_host}:{remote_root}/",
        ]
    )
    python = str(cast("list[dict[str, object]]", host["profiles"])[0]["python"])
    hash_command = "from lnet.pac_broad_benchmark_worker import code_sha256;print(code_sha256())"
    observed = _ssh(
        ssh_host,
        (
            f"cd {shlex.quote(repo)} && PYTHONPATH=src {shlex.quote(python)} -c "
            + shlex.quote(hash_command)
        ),
        capture=True,
    )
    contract = cast(
        "dict[str, object]",
        json.loads((ROOT / "contract.json").read_text(encoding="utf-8")),
    )
    expected = str(contract["code_sha256"])
    if observed != expected:
        message = f"frozen campaign code hash mismatch on {host['name']}: {observed} != {expected}"
        raise RuntimeError(message)


def _sync_manifest(host: dict[str, object], manifest: Path) -> None:
    ssh_host, repo = str(host["ssh_host"]), str(host["repo"])
    remote = f"{repo}/{manifest}"
    _ssh(ssh_host, f"mkdir -p {shlex.quote(str(Path(remote).parent))}")
    _run(["rsync", "-a", "--checksum", str(manifest), f"{ssh_host}:{remote}"])


def _collect(host: dict[str, object], stage: Stage) -> None:
    ssh_host, repo = str(host["ssh_host"]), str(host["repo"])
    for bucket in ("completed", "failed", "attempts"):
        remote = f"{repo}/{ROOT}/{stage}/{bucket}"
        local = ROOT / stage / bucket
        local.mkdir(parents=True, exist_ok=True)
        _ssh(ssh_host, f"mkdir -p {shlex.quote(remote)}")
        _run(["rsync", "-a", f"{ssh_host}:{remote}/", f"{local}/"])


def _completed(stage: Stage) -> set[str]:
    keys: set[str] = set()
    for path in (ROOT / stage / "completed").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "done":
            keys.add(str(row["job_key"]))
    return keys


def _stage_complete(stage: Literal["stage2", "final"]) -> bool:
    return {job.key for job in stage_jobs(ROOT, stage)} <= _completed(stage)


def _launch(
    host: dict[str, object],
    stage: Stage,
    lane: str,
    *,
    gpu: int | None = None,
) -> int:
    ssh_host, repo = str(host["ssh_host"]), str(host["repo"])
    roots = cast("dict[str, object]", host["data_roots"])
    python = str(cast("list[dict[str, object]]", host["profiles"])[0]["python"])
    selected_gpu = int(cast("list[int]", host["gpus"])[0]) if gpu is None else gpu
    stage_root = f"{repo}/{ROOT}/{stage}"
    pid_file = f"{stage_root}/{lane}.pid"
    log = f"{repo}/.omx/logs/broad-campaign-{stage}-{lane}.log"
    command_parts = [
            f"cd {shlex.quote(repo)} && echo $$ > {shlex.quote(pid_file)} && exec",
            "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
            f"PYTHONPATH=src CUDA_VISIBLE_DEVICES={selected_gpu}",
            shlex.quote(python),
            "scripts/run_broad_benchmark_worker.py",
            f"--root {shlex.quote(str(ROOT))}",
            f"--manifest {shlex.quote(str(ROOT / stage / 'deployment' / f'{lane}.jsonl'))}",
            "--device cuda",
            f"--ucr-data-root {shlex.quote(str(roots['ucr']))}",
            f"--external-data-root {shlex.quote(str(roots['external']))}",
    ]
    if roots.get("physionet2012"):
        command_parts.append(
            f"--physionet2012-data-root {shlex.quote(str(roots['physionet2012']))}"
        )
    if roots.get("raindrop"):
        command_parts.append(
            f"--raindrop-data-root {shlex.quote(str(roots['raindrop']))}"
        )
    command_parts.extend(("--max-attempts 3", f">>{shlex.quote(log)} 2>&1"))
    command = " ".join(command_parts)
    prepare_launch = " ".join(
        (
            f"rm -f {shlex.quote(pid_file)} && mkdir -p",
            shlex.quote(stage_root),
            shlex.quote(str(Path(log).parent)),
            "&&",
        )
    )
    launch_command = " ".join(
        (
            prepare_launch,
            f"setsid -f sh -lc {shlex.quote(command)} </dev/null >/dev/null 2>&1",
        )
    )
    _ssh(
        ssh_host,
        launch_command,
    )
    for _ in range(50):
        raw = _ssh(ssh_host, f"cat {shlex.quote(pid_file)} 2>/dev/null || true", capture=True)
        if raw.isdigit():
            return int(raw)
        time.sleep(0.1)
    message = f"{stage}/{lane} did not publish a PID"
    raise RuntimeError(message)


def _alive(host: dict[str, object], pid: int) -> bool:
    command = "".join(
        (
            f"ps -p {pid} -o args= 2>/dev/null | grep -F ",
            f"{shlex.quote(str(ROOT))} >/dev/null && echo yes || echo no",
        )
    )
    return (
        _ssh(
            str(host["ssh_host"]),
            command,
            capture=True,
        )
        == "yes"
    )


def _wait_stage(
    stage: Stage,
    hosts: list[dict[str, object]],
    workers: list[tuple[dict[str, object], int]],
) -> None:
    deployment = cast(
        "dict[str, object]",
        json.loads((ROOT / stage / "deployment.json").read_text(encoding="utf-8")),
    )
    units = cast("list[dict[str, object]]", deployment["units"])
    if stage == "stage1":
        expected = {
            job.key
            for unit in units
            for job in _manifest_jobs(Path(str(unit["manifest"])))
        }
    else:
        expected = {job.key for job in stage_jobs(ROOT, stage)}
    host_by_name = {str(host["name"]): host for host in hosts}
    for retry_pass in range(1, 4):
        while any(_alive(host, pid) for host, pid in workers):
            for host in hosts:
                _collect(host, stage)
            _write_state(
                {
                    "state": f"{stage}_running",
                    "retry_pass": retry_pass,
                    "expected": len(expected),
                    "completed": len(_completed(stage) & expected),
                    "updated_at_unix": time.time(),
                }
            )
            time.sleep(15)
        for host in hosts:
            _collect(host, stage)
        missing = expected - _completed(stage)
        if not missing:
            return
        workers = []
        for unit in units:
            unit_jobs = _manifest_jobs(Path(str(unit["manifest"])))
            if not ({job.key for job in unit_jobs} & missing):
                continue
            lane = cast("dict[str, object]", unit["lane"])
            host = host_by_name[str(lane["host"])]
            worker_lane = f"{lane['name']}-retry{retry_pass}"
            manifest = Path(str(unit["manifest"])).with_name(f"{worker_lane}.jsonl")
            manifest.write_bytes(Path(str(unit["manifest"])).read_bytes())
            _stage_host(host)
            workers.append(
                (
                    host,
                    _launch(
                        host,
                        stage,
                        worker_lane,
                        gpu=int(cast("str | int", lane["gpu"])),
                    ),
                )
            )
    missing = expected - _completed(stage)
    message = f"{stage} exhausted retries with {len(missing)} missing results"
    raise RuntimeError(message)


def _plan_and_launch(
    stage: Literal["stage2", "final"],
    hosts: list[dict[str, object]],
) -> list[tuple[dict[str, object], int]]:
    jobs = tuple(
        replace(job, comparison_group=f"{job.comparison_group}:shard-{index % 3}")
        for index, job in enumerate(sorted(stage_jobs(ROOT, stage), key=lambda item: item.key))
    )
    deployment = plan_jobs(
        ROOT,
        stage=stage,
        jobs=jobs,
        config=load_deployment_config(CONFIG),
    )
    audit = audit_deployment(ROOT, stage=stage)
    if not audit["ok"] or deployment["blocked_jobs"]:
        message = f"{stage} deployment audit failed: {audit}"
        raise RuntimeError(message)
    by_name = {str(host["name"]): host for host in hosts}
    workers: list[tuple[dict[str, object], int]] = []
    staged_hosts: set[str] = set()
    for unit in cast("list[dict[str, object]]", deployment["units"]):
        lane = cast("dict[str, object]", unit["lane"])
        host = by_name[str(lane["host"])]
        host_name = str(host["name"])
        if host_name not in staged_hosts:
            _stage_host(host)
            staged_hosts.add(host_name)
        source = Path(str(unit["manifest"]))
        for index in range(WORKERS_PER_HOST[str(host["name"])]):
            worker_lane = f"{lane['name']}-w{index}"
            manifest = source.with_name(f"{worker_lane}.jsonl")
            manifest.write_bytes(source.read_bytes())
            _sync_manifest(host, manifest)
            workers.append(
                (
                    host,
                    _launch(
                        host,
                        stage,
                        worker_lane,
                        gpu=int(cast("str | int", lane["gpu"])),
                    ),
                )
            )
    return workers


def _manifest_jobs(path: Path) -> tuple[BenchmarkJob, ...]:
    return tuple(
        BenchmarkJob.from_payload(cast("dict[str, object]", json.loads(line)))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _archive_parallel_stage1_manifests() -> None:
    deployment = cast(
        "dict[str, object]",
        json.loads((ROOT / "stage1/deployment.json").read_text(encoding="utf-8")),
    )
    canonical = {
        Path(str(unit["manifest"])).resolve()
        for unit in cast("list[dict[str, object]]", deployment["units"])
    }
    archive = ROOT / "stage1/parallel_manifests"
    archive.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "stage1/deployment").glob("*.jsonl"):
        if path.resolve() not in canonical:
            path.replace(archive / path.name)


def main() -> None:
    lock_path = ROOT / "supervisor.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        hosts = _hosts()
        launch = json.loads((ROOT / "stage1/launch.json").read_text(encoding="utf-8"))
        by_name = {str(host["name"]): host for host in hosts}
        stage1_workers = [
            (by_name[str(item["host"])], int(item["pid"]))
            for item in launch["launched"]
            if str(item["host"]) in by_name
        ]
        _wait_stage("stage1", hosts, stage1_workers)
        _archive_parallel_stage1_manifests()
        select_stage1(ROOT)
        stage2_workers = [] if _stage_complete("stage2") else _plan_and_launch("stage2", hosts)
        _wait_stage("stage2", hosts, stage2_workers)
        select_stage2(ROOT)
        final_workers = [] if _stage_complete("final") else _plan_and_launch("final", hosts)
        _wait_stage("final", hosts, final_workers)
        audit = audit_campaign(ROOT)
        if not audit["ok"]:
            message = f"campaign audit failed: {audit}"
            raise RuntimeError(message)
        _write_state({"state": "complete", "audit": audit})


if __name__ == "__main__":
    main()
