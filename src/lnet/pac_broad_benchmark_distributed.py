# ruff: noqa: C901, EM101, EM102, PLR0912, TRY003
"""Work-conserving heterogeneous-GPU planning for the broad benchmark.

Unlike the legacy wave planner, this module emits one mixed-profile manifest
per physical GPU lane.  Every dataset-stage comparison group remains on one
GPU, while profile-specific speed calibration drives deterministic LPT
assignment.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from .pac_campaign_utils import write_once

if TYPE_CHECKING:
    from .pac_broad_benchmark_queue import BenchmarkJob, Stage

Transport = Literal["local", "ssh"]
GPUType = Literal["rtx4090", "rtx3080ti"]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    python: str


@dataclass(frozen=True, slots=True)
class BroadHostSpec:
    name: str
    transport: Transport
    repo: str
    gpu_type: GPUType
    gpus: tuple[int, ...]
    memory_mb: int
    profiles: tuple[RuntimeProfile, ...]
    profile_relative_speeds: tuple[tuple[str, float], ...]
    data_shards: tuple[str, ...]
    ssh_host: str | None = None
    enabled: bool = True

    def profile(self, name: str) -> RuntimeProfile | None:
        return next((profile for profile in self.profiles if profile.name == name), None)

    def relative_speed(self, profile: str) -> float:
        try:
            return dict(self.profile_relative_speeds)[profile]
        except KeyError as error:
            raise ValueError(f"{self.name} has no calibrated speed for {profile}") from error

    def has_data(self, shard: str) -> bool:
        return shard in self.data_shards


@dataclass(frozen=True, slots=True)
class BroadDeploymentConfig:
    schema: str
    hosts: tuple[BroadHostSpec, ...]
    poll_seconds: int = 30
    max_attempts: int = 3


@dataclass(frozen=True, slots=True)
class WorkerLane:
    name: str
    host: str
    gpu: int
    gpu_type: GPUType
    memory_mb: int


@dataclass(frozen=True, slots=True)
class DeploymentUnit:
    lane: WorkerLane
    manifest: str
    jobs: int
    comparison_groups: int
    estimated_wall_seconds: float


def _identifier(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    if not result:
        raise ValueError(f"invalid empty identifier from {value!r}")
    return result


def _runtime_profile(payload: dict[str, object]) -> RuntimeProfile:
    return RuntimeProfile(
        name=_identifier(str(payload["name"])),
        python=str(payload["python"]),
    )


def _host(payload: dict[str, object]) -> BroadHostSpec:
    transport = cast("Transport", payload["transport"])
    if transport not in {"local", "ssh"}:
        raise ValueError(f"unsupported transport: {transport}")
    gpu_type = cast("GPUType", payload["gpu_type"])
    if gpu_type not in {"rtx4090", "rtx3080ti"}:
        raise ValueError(f"unsupported GPU type: {gpu_type}")
    profiles = tuple(
        _runtime_profile(profile)
        for profile in cast("list[dict[str, object]]", payload["profiles"])
    )
    speeds_payload = cast("dict[str, object]", payload["profile_relative_speeds"])
    speeds = tuple(
        sorted(
            (str(name), float(cast("str | int | float", value)))
            for name, value in speeds_payload.items()
        )
    )
    host = BroadHostSpec(
        name=_identifier(str(payload["name"])),
        transport=transport,
        repo=str(payload["repo"]),
        gpu_type=gpu_type,
        gpus=tuple(
            int(cast("str | int", value)) for value in cast("list[object]", payload["gpus"])
        ),
        memory_mb=int(cast("str | int", payload["memory_mb"])),
        profiles=profiles,
        profile_relative_speeds=speeds,
        data_shards=tuple(
            sorted(str(value) for value in cast("list[object]", payload.get("data_shards", [])))
        ),
        ssh_host=None if payload.get("ssh_host") is None else str(payload["ssh_host"]),
        enabled=bool(payload.get("enabled", True)),
    )
    profile_names = {profile.name for profile in profiles}
    if not host.gpus:
        raise ValueError(f"{host.name} has no GPU")
    if host.memory_mb < 1:
        raise ValueError(f"{host.name} memory_mb must be positive")
    if set(dict(speeds)) != profile_names:
        raise ValueError(f"{host.name} must calibrate every declared runtime profile")
    if any(speed <= 0 for _, speed in speeds):
        raise ValueError(f"{host.name} profile speeds must be positive")
    if transport == "ssh" and host.ssh_host is None:
        raise ValueError(f"SSH host {host.name} requires ssh_host")
    return host


def load_deployment_config(path: Path) -> BroadDeploymentConfig:
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    shard_sets = cast(
        "dict[str, list[object]]",
        payload.get("data_shard_sets", {}),
    )
    host_payloads: list[dict[str, object]] = []
    for raw_host in cast("list[dict[str, object]]", payload["hosts"]):
        host = dict(raw_host)
        shard_set = host.pop("data_shard_set", None)
        if shard_set is not None:
            if "data_shards" in host:
                raise ValueError(f"{host['name']} cannot declare data_shards and data_shard_set")
            try:
                host["data_shards"] = shard_sets[str(shard_set)]
            except KeyError as error:
                raise ValueError(
                    f"{host['name']} references unknown data_shard_set {shard_set}"
                ) from error
        host_payloads.append(host)
    config = BroadDeploymentConfig(
        schema=str(payload.get("schema", "alphabet.broad_deployment_config.v1")),
        hosts=tuple(_host(host) for host in host_payloads),
        poll_seconds=int(cast("str | int", payload.get("poll_seconds", 30))),
        max_attempts=int(cast("str | int", payload.get("max_attempts", 3))),
    )
    enabled = tuple(host for host in config.hosts if host.enabled)
    if not enabled:
        raise ValueError("deployment config has no enabled host")
    if len({host.name for host in config.hosts}) != len(config.hosts):
        raise ValueError("deployment config repeats a host name")
    return config


def _lanes(config: BroadDeploymentConfig) -> tuple[WorkerLane, ...]:
    return tuple(
        WorkerLane(
            name=f"{host.name}-gpu{gpu}",
            host=host.name,
            gpu=gpu,
            gpu_type=host.gpu_type,
            memory_mb=host.memory_mb,
        )
        for host in config.hosts
        if host.enabled
        for gpu in host.gpus
    )


def _host_for_lane(
    lane: WorkerLane,
    config: BroadDeploymentConfig,
) -> BroadHostSpec:
    return next(host for host in config.hosts if host.name == lane.host)


def _eligible_lanes(
    jobs: tuple[BenchmarkJob, ...],
    lanes: tuple[WorkerLane, ...],
    config: BroadDeploymentConfig,
) -> tuple[WorkerLane, ...]:
    eligible: list[WorkerLane] = []
    for lane in lanes:
        host = _host_for_lane(lane, config)
        if all(
            host.profile(job.runtime_profile) is not None
            and job.estimated_peak_memory_mb <= host.memory_mb
            and host.has_data(job.data_shard)
            for job in jobs
        ):
            eligible.append(lane)
    return tuple(eligible)


def _group_seconds(
    jobs: tuple[BenchmarkJob, ...],
    lane: WorkerLane,
    config: BroadDeploymentConfig,
) -> float:
    host = _host_for_lane(lane, config)
    return sum(job.estimated_seconds / host.relative_speed(job.runtime_profile) for job in jobs)


def plan_jobs(  # noqa: PLR0915 - explicit audited scheduling and freeze pipeline
    root: Path,
    *,
    stage: Stage,
    jobs: tuple[BenchmarkJob, ...],
    config: BroadDeploymentConfig,
) -> dict[str, object]:
    if any(job.stage != stage for job in jobs):
        raise ValueError(f"{stage} planner received another stage")
    if len({job.key for job in jobs}) != len(jobs):
        raise ValueError("planner received duplicate logical keys")
    lanes = _lanes(config)
    groups: dict[str, list[BenchmarkJob]] = {}
    blocked: list[BenchmarkJob] = []
    for job in jobs:
        if job.blockers:
            blocked.append(job)
        else:
            groups.setdefault(job.comparison_group, []).append(job)

    runnable_groups: list[tuple[str, tuple[BenchmarkJob, ...], tuple[WorkerLane, ...]]] = []
    for group, group_jobs_list in sorted(groups.items()):
        group_jobs = tuple(group_jobs_list)
        eligible = _eligible_lanes(group_jobs, lanes, config)
        if not eligible:
            blocked.extend(
                replace(
                    job,
                    blockers=(*job.blockers, "blocked_host_data_profile_or_memory"),
                )
                for job in group_jobs
            )
            continue
        runnable_groups.append((group, group_jobs, eligible))

    assignments: dict[str, list[BenchmarkJob]] = {lane.name: [] for lane in lanes}
    group_assignments: dict[str, list[str]] = {lane.name: [] for lane in lanes}
    loads = {lane.name: 0.0 for lane in lanes}
    lane_lookup = {lane.name: lane for lane in lanes}
    runnable_groups.sort(
        key=lambda item: (
            -max(_group_seconds(item[1], lane, config) for lane in item[2]),
            item[0],
        )
    )
    for group, group_jobs, eligible in runnable_groups:
        lane = min(
            eligible,
            key=lambda item: (
                loads[item.name] + _group_seconds(group_jobs, item, config),
                loads[item.name],
                item.name,
            ),
        )
        assignments[lane.name].extend(group_jobs)
        group_assignments[lane.name].append(group)
        loads[lane.name] += _group_seconds(group_jobs, lane, config)

    units: list[DeploymentUnit] = []
    assigned_keys: list[str] = []
    for lane_name, lane_jobs in sorted(assignments.items()):
        if not lane_jobs:
            continue
        lane_jobs.sort(
            key=lambda job: (
                job.runtime_profile,
                job.model,
                job.suite,
                job.dataset,
                job.width,
                job.architecture,
                job.key,
            )
        )
        manifest = root / stage / "deployment" / f"{lane_name}.jsonl"
        write_once(
            manifest,
            "".join(json.dumps(job.payload(), sort_keys=True) + "\n" for job in lane_jobs),
        )
        lane = lane_lookup[lane_name]
        units.append(
            DeploymentUnit(
                lane=lane,
                manifest=str(manifest),
                jobs=len(lane_jobs),
                comparison_groups=len(group_assignments[lane_name]),
                estimated_wall_seconds=loads[lane_name],
            )
        )
        assigned_keys.extend(job.key for job in lane_jobs)

    blocked_path = root / stage / "blocked.jsonl"
    write_once(
        blocked_path,
        "".join(
            json.dumps(job.payload(), sort_keys=True) + "\n"
            for job in sorted(blocked, key=lambda item: item.key)
        ),
    )
    blocked_keys = [job.key for job in blocked]
    if len(blocked_keys) != len(set(blocked_keys)):
        raise RuntimeError("a blocked job was recorded more than once")
    if set(assigned_keys).intersection(blocked_keys):
        raise RuntimeError("a job is both assigned and blocked")
    expected = {job.key for job in jobs}
    resolved = set(assigned_keys) | set(blocked_keys)
    if resolved != expected:
        missing_count = len(expected - resolved)
        extra_count = len(resolved - expected)
        message = (
            f"deployment resolution mismatch: missing={missing_count}, extra={extra_count}"
        )
        raise RuntimeError(message)
    payload: dict[str, object] = {
        "schema": "alphabet.broad_deployment.v1",
        "strategy": "comparison-group constrained profile-weighted LPT",
        "work_conserving": True,
        "stage": stage,
        "input_jobs": len(jobs),
        "assigned_jobs": len(assigned_keys),
        "blocked_jobs": len(blocked_keys),
        "comparison_groups": len(runnable_groups),
        "units": [asdict(unit) for unit in units],
        "max_estimated_wall_seconds": max(
            (unit.estimated_wall_seconds for unit in units),
            default=0.0,
        ),
        "blocked_manifest": str(blocked_path),
        "hosts": [
            {
                **asdict(host),
                "data_shard_count": len(host.data_shards),
            }
            for host in config.hosts
        ],
    }
    write_once(
        root / stage / "deployment.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def audit_deployment(root: Path, *, stage: Stage) -> dict[str, object]:
    path = root / stage / "deployment.json"
    payload = cast(
        "dict[str, object]",
        json.loads(path.read_text(encoding="utf-8")),
    )
    assigned: list[str] = []
    group_owner: dict[str, str] = {}
    for unit in cast("list[dict[str, object]]", payload["units"]):
        lane = cast("dict[str, object]", unit["lane"])
        lane_name = str(lane["name"])
        manifest = Path(str(unit["manifest"]))
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = cast("dict[str, object]", json.loads(line))
            key = str(row["key"])
            group = str(row["comparison_group"])
            assigned.append(key)
            prior = group_owner.setdefault(group, lane_name)
            if prior != lane_name:
                raise RuntimeError(f"comparison group {group} spans {prior} and {lane_name}")
    blocked = [
        cast("dict[str, object]", json.loads(line))
        for line in Path(str(payload["blocked_manifest"])).read_text(encoding="utf-8").splitlines()
    ]
    blocked_keys = [str(row["key"]) for row in blocked]
    duplicate_assignments = len(assigned) - len(set(assigned))
    duplicate_blocked = len(blocked_keys) - len(set(blocked_keys))
    overlap = set(assigned).intersection(blocked_keys)
    ok = (
        duplicate_assignments == 0
        and duplicate_blocked == 0
        and not overlap
        and len(assigned) == int(cast("str | int", payload["assigned_jobs"]))
        and len(blocked_keys) == int(cast("str | int", payload["blocked_jobs"]))
    )
    return {
        "schema": "alphabet.broad_deployment_audit.v1",
        "stage": stage,
        "ok": ok,
        "assigned_jobs": len(assigned),
        "blocked_jobs": len(blocked_keys),
        "duplicate_assignments": duplicate_assignments,
        "duplicate_blocked": duplicate_blocked,
        "assigned_blocked_overlap": len(overlap),
        "comparison_groups": len(group_owner),
    }


__all__: Final = [
    "BroadDeploymentConfig",
    "BroadHostSpec",
    "DeploymentUnit",
    "RuntimeProfile",
    "WorkerLane",
    "audit_deployment",
    "load_deployment_config",
    "plan_jobs",
]
