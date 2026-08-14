"""Resume the frozen broad follow-up campaign on all five available GPUs."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path

import supervise_broad_campaign as engine

from lnet.pac_broad_benchmark_completion import (
    audit_campaign,
    select_stage2,
)


ROOT = Path(".omx/results/alphabet-broad-new-datasets-3gpu-20260727")
BASE_CONFIG = Path("optimization/hosts/broad_followup_3gpu.local.json")
RUNTIME_CONFIG = ROOT / "five-gpu-runtime-config.json"


def _write_runtime_config() -> None:
    payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    payload["hosts"].append(
        {
            "name": "local_gpu",
            "enabled": True,
            "transport": "ssh",
            "ssh_host": "local_gpu",
            "repo": "<local-home>/lnet-broad-new-runtime-20260727",
            "gpu_type": "rtx4090",
            "gpus": [0, 1],
            "memory_mb": 24564,
            "profiles": [
                {
                    "name": "core",
                    "python": "<local-home>/miniconda3/envs/brelu/bin/python",
                },
                {
                    "name": "mamba",
                    "python": "<local-home>/miniconda3/envs/brelu/bin/python",
                },
            ],
            "profile_relative_speeds": {"core": 1.0, "mamba": 1.0},
            "data_shards": payload["hosts"][0]["data_shards"],
            "data_roots": {
                "ucr": "<local-home>/lnet-data/ucr",
                "external": "<local-home>/lnet-data/ucr",
            },
        }
    )
    RUNTIME_CONFIG.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_frozen_host(host: dict[str, object]) -> None:
    """Preserve the Stage-1 runtime and verify its worker hash."""
    repo = str(host["repo"])
    python = str(host["profiles"][0]["python"])
    sample = next((ROOT / "stage1/completed").glob("*.json"))
    expected = str(json.loads(sample.read_text(encoding="utf-8"))["code_sha256"])
    command = "from lnet.pac_broad_benchmark_worker import code_sha256;print(code_sha256())"
    observed = engine._ssh(
        str(host["ssh_host"]),
        f"cd {shlex.quote(repo)} && PYTHONPATH=src {shlex.quote(python)} -c "
        f"{shlex.quote(command)}",
        capture=True,
    )
    if observed != expected:
        raise RuntimeError(
            f"frozen campaign code hash mismatch on {host['name']}: "
            f"{observed} != {expected}"
        )


def _resilient_alive(host: dict[str, object], pid: int) -> bool:
    """Tolerate transient SSH failures without killing the supervisor."""
    for attempt in range(5):
        try:
            return engine._alive_original(host, pid)
        except subprocess.CalledProcessError:
            if attempt == 4:
                return True
            time.sleep(2)
    return True


def _wait_until_complete(stage: str, hosts: list[dict[str, object]]) -> None:
    """Collect indefinitely; remote workers are independently restart-safe."""
    expected = {job.key for job in engine.stage_jobs(ROOT, stage)}
    while True:
        for host in hosts:
            try:
                engine._collect(host, stage)
            except subprocess.CalledProcessError:
                continue
        completed = engine._completed(stage)
        engine._write_state(
            {
                "state": f"{stage}_running",
                "expected": len(expected),
                "completed": len(expected & completed),
                "updated_at_unix": time.time(),
            }
        )
        if expected <= completed:
            return
        time.sleep(15)


def main() -> None:
    _write_runtime_config()
    engine.ROOT = ROOT
    engine.CONFIG = RUNTIME_CONFIG
    engine.STATE = ROOT / "five-gpu-supervisor-state.json"
    engine.WORKERS_PER_HOST["local_gpu"] = 1
    engine._stage_host = _verify_frozen_host
    engine._alive_original = engine._alive
    engine._alive = _resilient_alive
    hosts = engine._hosts()
    stage2_workers = (
        []
        if (ROOT / "stage2/deployment.json").is_file()
        else engine._plan_and_launch("stage2", hosts)
    )
    del stage2_workers
    _wait_until_complete("stage2", hosts)
    select_stage2(ROOT)
    final_workers = engine._plan_and_launch("final", hosts)
    del final_workers
    _wait_until_complete("final", hosts)
    result = audit_campaign(ROOT)
    engine._write_state({"state": "complete", "audit": result})
    if not result["ok"]:
        raise RuntimeError(f"campaign audit failed: {result}")


if __name__ == "__main__":
    main()
