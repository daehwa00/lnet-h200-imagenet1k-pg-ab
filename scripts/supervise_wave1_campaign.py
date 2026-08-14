# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
# ruff: noqa: SLF001
"""Wait for the active campaign, then execute Wave 1 on the three approved hosts."""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from lnet import pac_fast_completion as completion
from lnet.pac_wave1_campaign import DEFAULT_ROOT, expected_counts
from scripts import supervise_broad_campaign as engine
from scripts.prepare_wave1_campaign import orchestration_sha256

ACTIVE_ROOT = Path(".omx/results/alphabet-new3-10model-5gpu-20260727")
APPROVED_HOSTS = ("secondary_gpu", "rtx3080ti-1", "rtx3080ti-2")
ENVIRONMENT_COMPATIBILITY_KEYS = (
    "torch_version",
    "torch_cuda_version",
    "cudnn_version",
    "mamba_ssm_version",
    "mamba_class_module",
    "mamba_class_relative_path",
    "mamba_source_sha256",
)
engine.ROOT = DEFAULT_ROOT
engine.CONFIG = Path("optimization/hosts/wave1_3gpu.local.json")
engine.STATE = engine.ROOT / "supervisor-state.json"
engine.audit_campaign = completion.audit_campaign
engine.select_stage1 = completion.select_stage1
engine.stage_jobs = completion.stage_jobs
engine.WORKERS_PER_HOST = {
    **engine.WORKERS_PER_HOST,
    "secondary_gpu": 2,
    "rtx3080ti-1": 2,
    "rtx3080ti-2": 2,
}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _active_complete() -> bool:
    state = ACTIVE_ROOT / "supervisor-state.json"
    if not state.is_file():
        return False
    payload = json.loads(state.read_text(encoding="utf-8"))
    return payload.get("state") == "complete" and bool(
        cast("dict[str, object]", payload.get("audit", {})).get("ok")
    )


def _verify_orchestration_freeze() -> None:
    contract_path = engine.ROOT / "contract.json"
    try:
        contract = cast(
            "dict[str, object]",
            json.loads(contract_path.read_text(encoding="utf-8")),
        )
    except (json.JSONDecodeError, OSError) as error:
        message = f"cannot read frozen Wave-1 contract: {contract_path}"
        raise RuntimeError(message) from error
    expected = contract.get("orchestration_sha256")
    observed = orchestration_sha256()
    if not isinstance(expected, str) or len(expected) != 64 or observed != expected:
        _atomic_json(
            engine.ROOT / "orchestration-revalidation-failure.json",
            {
                "schema": "alphabet.wave1_orchestration_revalidation_failure.v1",
                "checked_at_utc": datetime.now(UTC).isoformat(),
                "expected_orchestration_sha256": expected,
                "observed_orchestration_sha256": observed,
            },
        )
        message = "Wave-1 orchestration code changed after the campaign freeze"
        raise RuntimeError(message)


def _host_environment_fingerprint(host: dict[str, object]) -> dict[str, object]:
    profiles = cast("list[dict[str, object]]", host["profiles"])
    configured_pythons = {str(profile["python"]) for profile in profiles}
    if len(configured_pythons) != 1:
        message = (
            f"Wave-1 host {host['name']} has multiple configured Python runtimes: "
            f"{sorted(configured_pythons)}"
        )
        raise RuntimeError(message)
    python = next(iter(configured_pythons))
    probe = """
import hashlib
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path

import mamba_ssm
import torch

mamba_class = mamba_ssm.Mamba
mamba_source = inspect.getsourcefile(mamba_class)
if mamba_source is None:
    raise RuntimeError("cannot resolve mamba_ssm.Mamba source")
package_root = Path(mamba_ssm.__file__).resolve().parent
mamba_relative_path = str(Path(mamba_source).resolve().relative_to(package_root))
with open(mamba_source, "rb") as handle:
    mamba_source_sha256 = hashlib.sha256(handle.read()).hexdigest()
payload = {
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "mamba_ssm_version": importlib.metadata.version("mamba-ssm"),
    "mamba_package_file": mamba_ssm.__file__,
    "mamba_class_module": mamba_class.__module__,
    "mamba_class_file": mamba_source,
    "mamba_class_relative_path": mamba_relative_path,
    "mamba_source_sha256": mamba_source_sha256,
}
print(json.dumps(payload, sort_keys=True))
""".strip()
    output = engine._ssh(
        str(host["ssh_host"]),
        f"{shlex.quote(python)} -c {shlex.quote(probe)}",
        capture=True,
    )
    try:
        fingerprint = cast("dict[str, object]", json.loads(output))
    except json.JSONDecodeError as error:
        message = f"invalid Wave-1 environment fingerprint from {host['name']}"
        raise RuntimeError(message) from error
    required = {
        *ENVIRONMENT_COMPATIBILITY_KEYS,
        "python_executable",
        "mamba_package_file",
        "mamba_class_file",
        "cuda_available",
        "gpu_name",
    }
    if not required <= fingerprint.keys() or fingerprint["cuda_available"] is not True:
        message = f"incomplete or non-CUDA Wave-1 environment on {host['name']}"
        raise RuntimeError(message)
    return {
        "host": host["name"],
        "ssh_host": host["ssh_host"],
        "configured_python": python,
        "configured_profiles": [str(profile["name"]) for profile in profiles],
        **fingerprint,
    }


def _environment_mismatches(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    mismatches: dict[str, dict[str, object]] = {}
    for key in ENVIRONMENT_COMPATIBILITY_KEYS:
        values = {str(row["host"]): row.get(key) for row in rows}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            mismatches[key] = values
    return mismatches


def _compatibility_reference(rows: list[dict[str, object]]) -> dict[str, object]:
    return {key: rows[0][key] for key in ENVIRONMENT_COMPATIBILITY_KEYS}


def _frozen_environment_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = payload.get("hosts")
    if (
        payload.get("schema") != "alphabet.wave1_environment_audit.v1"
        or payload.get("compatible") is not True
        or payload.get("local_gpu_excluded") is not True
        or not isinstance(rows, list)
    ):
        message = "existing Wave-1 environment audit is not a valid successful freeze"
        raise RuntimeError(message)
    return cast("list[dict[str, object]]", rows)


def _freeze_environment() -> dict[str, object]:
    hosts = engine._hosts()
    names = tuple(str(host["name"]) for host in hosts)
    if len(hosts) != len(APPROVED_HOSTS) or set(names) != set(APPROVED_HOSTS):
        message = (
            "Wave-1 environment freeze requires exactly the three approved hosts "
            f"{APPROVED_HOSTS}; observed {names}"
        )
        raise RuntimeError(message)
    rows = [_host_environment_fingerprint(host) for host in hosts]
    mismatches = _environment_mismatches(rows)
    reference = _compatibility_reference(rows)
    path = engine.ROOT / "environment-audit.json"
    if path.is_file():
        existing = cast(
            "dict[str, object]",
            json.loads(path.read_text(encoding="utf-8")),
        )
        if existing.get("compatible") is True:
            frozen_rows = _frozen_environment_rows(existing)
            frozen_reference = existing.get("compatibility_reference")
            if mismatches or reference != frozen_reference:
                failure = {
                    "schema": "alphabet.wave1_environment_revalidation_failure.v1",
                    "checked_at_utc": datetime.now(UTC).isoformat(),
                    "local_gpu_excluded": True,
                    "mismatches": mismatches,
                    "frozen_compatibility_reference": frozen_reference,
                    "observed_compatibility_reference": reference,
                    "frozen_hosts": frozen_rows,
                    "observed_hosts": rows,
                }
                _atomic_json(engine.ROOT / "environment-revalidation-failure.json", failure)
                message = "Wave-1 remote environment changed after the successful freeze"
                raise RuntimeError(message)
            _atomic_json(
                engine.ROOT / "environment-revalidation.json",
                {
                    "schema": "alphabet.wave1_environment_revalidation.v1",
                    "checked_at_utc": datetime.now(UTC).isoformat(),
                    "compatible": True,
                    "local_gpu_excluded": True,
                    "compatibility_reference": reference,
                    "hosts": rows,
                },
            )
            return existing
        if existing.get("schema") != "alphabet.wave1_environment_audit.v1":
            message = "existing Wave-1 environment audit has an unknown schema"
            raise RuntimeError(message)
        _atomic_json(engine.ROOT / "environment-previous-incompatible-audit.json", existing)
    payload: dict[str, object] = {
        "schema": "alphabet.wave1_environment_audit.v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "compatible": not mismatches,
        "local_gpu_excluded": True,
        "approved_hosts": list(APPROVED_HOSTS),
        "compatibility_keys": list(ENVIRONMENT_COMPATIBILITY_KEYS),
        "provenance_only_keys": ["python_executable", "python_version", "gpu_name"],
        "compatibility_reference": reference,
        "gpu_model_may_differ": True,
        "hosts": rows,
        "mismatches": mismatches,
    }
    _atomic_json(path, payload)
    if mismatches:
        message = f"incompatible Wave-1 remote environments: {sorted(mismatches)}"
        raise RuntimeError(message)
    return payload


def _final_selection_frozen() -> bool:
    path = engine.ROOT / "stage2/selection.json"
    if not path.is_file():
        return False
    try:
        payload = cast(
            "dict[str, object]",
            json.loads(path.read_text(encoding="utf-8")),
        )
    except (json.JSONDecodeError, OSError) as error:
        message = f"invalid frozen Stage-2 selection: {path}"
        raise RuntimeError(message) from error
    selected = payload.get("selected")
    counts = expected_counts()
    valid = (
        payload.get("schema") == "alphabet.broad_benchmark.runnable_stage2_selection.v1"
        and payload.get("official_test_accessed") is False
        and payload.get("runnable_cells") == counts["cells"]
        and payload.get("final_jobs") == counts["final"]
        and isinstance(selected, dict)
        and len(selected) == counts["cells"]
        and all(isinstance(key, str) and isinstance(value, str) for key, value in selected.items())
    )
    if not valid:
        message = f"Stage-2 selection is not a complete frozen winner set: {path}"
        raise RuntimeError(message)
    return True


def _wait_for_active() -> None:
    while not _active_complete():
        completed = len(list((ACTIVE_ROOT / "final/completed").glob("*.json")))
        failed = len(list((ACTIVE_ROOT / "final/failed").glob("*.json")))
        _atomic_json(
            engine.ROOT / "queue-state.json",
            {
                "schema": "alphabet.wave1_queue_state.v1",
                "state": "waiting_for_active_campaign",
                "active_final_completed": completed,
                "active_final_expected": 90,
                "active_final_failed": failed,
                "local_gpu_excluded": True,
            },
        )
        time.sleep(30)


def _data_rows() -> list[dict[str, object]]:
    payload = cast(
        "dict[str, object]",
        json.loads((engine.ROOT / "data-audit.json").read_text(encoding="utf-8")),
    )
    return cast("list[dict[str, object]]", payload["datasets"])


def _expected_data(*, full: bool) -> tuple[list[Path], dict[str, str]]:
    source_key = "full_path" if full else "selection_path"
    sha_key = "full_sha256" if full else "selection_sha256"
    sources = [Path(str(row[source_key])) for row in _data_rows()]
    expected = {
        Path(str(row[source_key])).name: str(row[sha_key])
        for row in _data_rows()
    }
    return sources, expected


def _assert_selection_root_sealed(
    host: dict[str, object],
    remote_root: str,
    expected_names: set[str],
) -> None:
    top_level = set(
        engine._ssh(
            str(host["ssh_host"]),
            (
                f"find {shlex.quote(remote_root)} -maxdepth 1 -type f "
                "-name '*.pt' -printf '%f\\n'"
            ),
            capture=True,
        ).splitlines()
    )
    leaked = sorted(top_level & expected_names)
    if leaked:
        message = (
            f"Wave-1 selection root on {host['name']} already exposes "
            f"full TEST artifacts: {leaked}"
        )
        raise RuntimeError(message)


def _verify_data_host(
    host: dict[str, object],
    *,
    full: bool,
    allow_full_top_level: bool = False,
) -> dict[str, object]:
    ssh_host = str(host["ssh_host"])
    remote_root = str(cast("dict[str, object]", host["data_roots"])["external"])
    destination = remote_root if full else f"{remote_root}/selection-only"
    sources, expected = _expected_data(full=full)
    expected_names = set(expected)
    if not full and not allow_full_top_level:
        _assert_selection_root_sealed(host, remote_root, expected_names)
    remote_paths = [f"{destination}/{source.name}" for source in sources]
    output = engine._ssh(
        ssh_host,
        "sha256sum " + " ".join(shlex.quote(path) for path in remote_paths),
        capture=True,
    )
    observed = {
        Path(line.split(maxsplit=1)[1]).name: line.split(maxsplit=1)[0]
        for line in output.splitlines()
    }
    if observed != expected:
        message = f"Wave-1 {'full' if full else 'selection'} data hash mismatch on {host['name']}"
        raise RuntimeError(message)
    remote_files = set(
        engine._ssh(
            ssh_host,
            (
                f"find {shlex.quote(destination)} -maxdepth 1 -type f "
                "-name '*.pt' -printf '%f\\n'"
            ),
            capture=True,
        ).splitlines()
    )
    if remote_files != expected_names:
        message = (
            f"Wave-1 {'full' if full else 'selection'} root on {host['name']} "
            "does not contain exactly the frozen 37 artifacts"
        )
        raise RuntimeError(message)
    return {
        "host": host["name"],
        "ssh_host": ssh_host,
        "remote_root": destination,
        "artifacts": len(observed),
        "sha256_verified": True,
    }


def _sync_data_host(host: dict[str, object], *, full: bool) -> dict[str, object]:
    ssh_host = str(host["ssh_host"])
    remote_root = str(cast("dict[str, object]", host["data_roots"])["external"])
    destination = remote_root if full else f"{remote_root}/selection-only"
    engine._ssh(ssh_host, f"mkdir -p {shlex.quote(destination)}")
    sources, expected = _expected_data(full=full)
    if not full:
        _assert_selection_root_sealed(host, remote_root, set(expected))
    engine._run(
        [
            "rsync",
            "-a",
            "--checksum",
            "--partial",
            *map(str, sources),
            f"{ssh_host}:{destination}/",
        ]
    )
    return _verify_data_host(host, full=full)


def _write_data_verification(
    rows: list[dict[str, object]],
    *,
    full: bool,
) -> None:
    _atomic_json(
        engine.ROOT / f"data-sync-{'full' if full else 'selection'}.json",
        {
            "schema": "alphabet.wave1_data_sync.v1",
            "kind": "full" if full else "selection-only",
            "hosts": rows,
            "official_test_available_to_workers": full,
        },
    )


def _sync_all_data(*, full: bool) -> list[dict[str, object]]:
    rows = [_sync_data_host(host, full=full) for host in engine._hosts()]
    _write_data_verification(rows, full=full)
    return rows


def _reverify_all_data(*, full: bool) -> list[dict[str, object]]:
    rows = [_verify_data_host(host, full=full) for host in engine._hosts()]
    _write_data_verification(rows, full=full)
    return rows


_base_select_stage2 = completion.select_stage2


def _select_stage2_and_stage_full(root: Path) -> dict[str, object]:
    payload = _base_select_stage2(root)
    _sync_all_data(full=True)
    return payload


engine.select_stage2 = _select_stage2_and_stage_full


def _write_launch(path: Path, launched: list[dict[str, object]]) -> None:
    _atomic_json(
        path,
        {
            "schema": "alphabet.wave1_stage1_launch.v1",
            "launched": launched,
            "workers_per_host": engine.WORKERS_PER_HOST,
        },
    )


def _launch_stage1() -> None:
    launch_path = engine.ROOT / "stage1/launch.json"
    hosts = engine._hosts()
    hosts_by_name = {str(host["name"]): host for host in hosts}
    deployment = cast(
        "dict[str, object]",
        json.loads((engine.ROOT / "stage1/deployment.json").read_text(encoding="utf-8")),
    )
    launched = (
        cast(
            "list[dict[str, object]]",
            json.loads(launch_path.read_text(encoding="utf-8"))["launched"],
        )
        if launch_path.is_file()
        else []
    )
    existing = {(str(row["host"]), str(row["lane"])) for row in launched}
    staged_hosts: set[str] = set()
    for unit in cast("list[dict[str, object]]", deployment["units"]):
        lane = cast("dict[str, object]", unit["lane"])
        host = hosts_by_name[str(lane["host"])]
        host_name = str(host["name"])
        if host_name not in staged_hosts:
            engine._stage_host(host)
            staged_hosts.add(host_name)
        source = Path(str(unit["manifest"]))
        for index in range(engine.WORKERS_PER_HOST[host_name]):
            worker_lane = f"{lane['name']}-w{index}"
            if (host_name, worker_lane) in existing:
                continue
            manifest = source.with_name(f"{worker_lane}.jsonl")
            manifest.write_bytes(source.read_bytes())
            engine._sync_manifest(host, manifest)
            remote_pid_path = (
                f"{host['repo']}/{engine.ROOT}/stage1/{worker_lane}.pid"
            )
            raw_pid = engine._ssh(
                str(host["ssh_host"]),
                f"cat {shlex.quote(remote_pid_path)} 2>/dev/null || true",
                capture=True,
            )
            recovered = raw_pid.isdigit() and engine._alive(host, int(raw_pid))
            pid = (
                int(raw_pid)
                if recovered
                else engine._launch(
                    host,
                    "stage1",
                    worker_lane,
                    gpu=int(cast("str | int", lane["gpu"])),
                )
            )
            launched.append(
                {
                    "host": host_name,
                    "ssh_host": host["ssh_host"],
                    "lane": worker_lane,
                    "manifest_lane": lane["name"],
                    "pid": pid,
                    "recovered_existing_process": recovered,
                }
            )
            existing.add((host_name, worker_lane))
            _write_launch(launch_path, launched)


def main() -> None:
    engine.ROOT.mkdir(parents=True, exist_ok=True)
    bootstrap_lock = engine.ROOT / "wave1-supervisor.lock"
    with bootstrap_lock.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        _verify_orchestration_freeze()
        _wait_for_active()
        _verify_orchestration_freeze()
        _freeze_environment()
        final_frozen = _final_selection_frozen()
        if final_frozen:
            for host in engine._hosts():
                _verify_data_host(
                    host,
                    full=False,
                    allow_full_top_level=True,
                )
            if (engine.ROOT / "data-sync-full.json").is_file():
                _reverify_all_data(full=True)
            else:
                _sync_all_data(full=True)
        elif not (engine.ROOT / "data-sync-selection.json").is_file():
            _atomic_json(
                engine.ROOT / "queue-state.json",
                {
                    "schema": "alphabet.wave1_queue_state.v1",
                    "state": "syncing_selection_only_data",
                    "local_gpu_excluded": True,
                },
            )
            _sync_all_data(full=False)
        else:
            _reverify_all_data(full=False)
        _launch_stage1()
        engine.main()


if __name__ == "__main__":
    main()
