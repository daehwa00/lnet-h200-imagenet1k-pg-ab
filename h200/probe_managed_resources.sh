#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
cd "${PROJECT_ROOT}"

if [[ ! "${H200_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: H200_EXPECTED_COMMIT must be the exact 40-character probe commit" >&2
  exit 2
fi
ACTUAL_COMMIT="$(git rev-parse --verify HEAD)"
readonly ACTUAL_COMMIT
if [[ "${ACTUAL_COMMIT}" != "${H200_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: resource probe commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "ERROR: resource probe checkout is not clean" >&2
  exit 2
fi

python3 - <<'PY'
from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
from pathlib import Path
from typing import Any


def read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def integer(value: str | None) -> int | str | None:
    if value is None or value == "max":
        return value
    try:
        return int(value)
    except ValueError:
        return value


proc_cgroup = read("/proc/self/cgroup") or ""
cgroup_root = Path("/sys/fs/cgroup")
cgroup_bases: list[Path] = []
for line in proc_cgroup.splitlines():
    fields = line.split(":", maxsplit=2)
    if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
        current = cgroup_root / fields[2].lstrip("/")
        while current != cgroup_root:
            cgroup_bases.append(current)
            current = current.parent
        break
cgroup_bases.append(cgroup_root)
cgroup_bases = list(dict.fromkeys(cgroup_bases))


def read_cgroup(name: str) -> str | None:
    for base in cgroup_bases:
        value = read(str(base / name))
        if value is not None:
            return value
    return None


cpu_max = read_cgroup("cpu.max")
quota_cores: float | str | None = None
if cpu_max:
    quota, period = cpu_max.split()
    quota_cores = "unlimited" if quota == "max" else int(quota) / int(period)
else:
    quota = read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota is not None and period is not None:
        quota_cores = "unlimited" if int(quota) < 0 else int(quota) / int(period)

cpuinfo = read("/proc/cpuinfo") or ""
model_names = sorted(
    {
        line.split(":", maxsplit=1)[1].strip()
        for line in cpuinfo.splitlines()
        if line.startswith("model name") and ":" in line
    }
)
status = read("/proc/self/status") or ""
status_fields = {}
for line in status.splitlines():
    if line.startswith(("Cpus_allowed_list", "Mems_allowed_list", "Threads")):
        name, value = line.split(":", maxsplit=1)
        status_fields[name] = value.strip()

payload: dict[str, Any] = {
    "schema": "lnet.h200.managed_resource_probe.v1",
    "hostname": platform.node(),
    "kernel": platform.release(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "cpu": {
        "model_names": model_names,
        "os_cpu_count": os.cpu_count(),
        "sched_affinity": sorted(os.sched_getaffinity(0)),
        "sched_affinity_count": len(os.sched_getaffinity(0)),
        "nproc": integer(command("nproc")),
        "nproc_all": integer(command("nproc", "--all")),
        "quota_cores": quota_cores,
        "cpu_max": cpu_max,
        "cpu_weight": integer(read_cgroup("cpu.weight")),
        "cpuset_cpus": read_cgroup("cpuset.cpus"),
        "cpuset_cpus_effective": read_cgroup("cpuset.cpus.effective"),
        "cpu_stat": read_cgroup("cpu.stat"),
        "proc_status": status_fields,
    },
    "memory": {
        "max": integer(read_cgroup("memory.max")),
        "current": integer(read_cgroup("memory.current")),
        "swap_max": integer(read_cgroup("memory.swap.max")),
    },
    "process": {
        "cgroup": proc_cgroup,
        "cgroup_paths_checked": [str(path) for path in cgroup_bases],
        "pids_max": integer(read_cgroup("pids.max")),
        "rlimit_nproc": list(resource.getrlimit(resource.RLIMIT_NPROC)),
        "rlimit_nofile": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
    },
    "gpu": {
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": command(
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ),
    },
}
print("H200_RESOURCE_PROBE=" + json.dumps(payload, sort_keys=True), flush=True)
PY

echo "H200_RESOURCE_PROBE_COMPLETE=1"
