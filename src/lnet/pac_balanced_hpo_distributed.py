"""Deterministic heterogeneous-GPU deployment planning for balanced HPO.

The planner creates immutable, non-overlapping manifests.  Jobs are grouped by
duration class and runtime profile, then assigned with constrained LPT
scheduling.  This keeps every GPU busy while preserving one owner per logical
job and lets one long-lived Python worker amortize imports and kernel caches
over many fits.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, cast

from .pac_balanced_hpo_campaign import BalancedHPOJob, load_manifest
from .pac_balanced_hpo_queue import DEFAULT_ROOT, MODELS
from .pac_campaign_utils import write_once

JobClass = Literal["short", "medium", "long"]
Transport = Literal["local", "ssh"]
GPUType = Literal["rtx4090", "rtx3080ti"]
JSONScalar = str | int | float

DEFAULT_CONCURRENCY: Final[dict[GPUType, dict[JobClass, int]]] = {
    "rtx4090": {"short": 3, "medium": 2, "long": 1},
    "rtx3080ti": {"short": 1, "medium": 1, "long": 1},
}


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    python: str


@dataclass(frozen=True, slots=True)
class HostSpec:
    name: str
    transport: Transport
    repo: str
    gpu_type: GPUType
    gpus: tuple[int, ...]
    relative_speed: float
    profiles: tuple[RuntimeProfile, ...]
    ucr_data_root: str
    external_data_root: str
    ssh_host: str | None = None
    ssh_port: int = 22
    ssh_key: str | None = None
    concurrency: tuple[tuple[JobClass, int], ...] = ()
    enabled: bool = True

    def profile(self, name: str) -> RuntimeProfile | None:
        return next((profile for profile in self.profiles if profile.name == name), None)

    def process_count(self, job_class: JobClass) -> int:
        overrides = dict(self.concurrency)
        return overrides.get(job_class, DEFAULT_CONCURRENCY[self.gpu_type][job_class])


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    schema: str
    model_profiles: tuple[tuple[str, str], ...]
    hosts: tuple[HostSpec, ...]
    class_order: tuple[JobClass, ...] = ("long", "medium", "short")
    poll_seconds: int = 30
    max_attempts: int = 3

    def profile_for_model(self, model: str) -> str:
        profiles = dict(self.model_profiles)
        try:
            return profiles[model]
        except KeyError as error:
            message = f"no runtime profile is assigned to model {model}"
            raise ValueError(message) from error


@dataclass(frozen=True, slots=True)
class WorkerLane:
    name: str
    host: str
    gpu: int
    process_index: int
    job_class: JobClass
    profile: str
    relative_speed: float


@dataclass(frozen=True, slots=True)
class DeploymentUnit:
    lane: WorkerLane
    manifest: str
    jobs: int
    estimated_seconds: float
    normalized_seconds: float


def _identifier(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    if not sanitized:
        message = f"invalid empty deployment identifier from {value!r}"
        raise ValueError(message)
    return sanitized


def _profile(payload: dict[str, object]) -> RuntimeProfile:
    return RuntimeProfile(
        name=_identifier(str(payload["name"])),
        python=str(payload["python"]),
    )


def _as_int(value: object) -> int:
    return int(cast("JSONScalar", value))


def _as_float(value: object) -> float:
    return float(cast("JSONScalar", value))


def _host(payload: dict[str, object]) -> HostSpec:
    transport = cast("Transport", payload["transport"])
    if transport not in {"local", "ssh"}:
        message = f"unsupported transport: {transport}"
        raise ValueError(message)
    gpu_type = cast("GPUType", payload["gpu_type"])
    if gpu_type not in DEFAULT_CONCURRENCY:
        message = f"unsupported GPU type: {gpu_type}"
        raise ValueError(message)
    concurrency_payload = cast("dict[str, object]", payload.get("concurrency", {}))
    concurrency: list[tuple[JobClass, int]] = []
    for job_class in cast("tuple[JobClass, ...]", ("short", "medium", "long")):
        if job_class not in concurrency_payload:
            continue
        count = _as_int(concurrency_payload[job_class])
        if count < 1:
            message = f"{payload['name']} concurrency for {job_class} must be positive"
            raise ValueError(message)
        concurrency.append((job_class, count))
    profiles_payload = cast("list[dict[str, object]]", payload["profiles"])
    profiles = tuple(_profile(profile) for profile in profiles_payload)
    if len({profile.name for profile in profiles}) != len(profiles):
        message = f"host {payload['name']} repeats a runtime profile"
        raise ValueError(message)
    host = HostSpec(
        name=_identifier(str(payload["name"])),
        transport=transport,
        repo=str(payload["repo"]),
        gpu_type=gpu_type,
        gpus=tuple(_as_int(gpu) for gpu in cast("list[object]", payload["gpus"])),
        relative_speed=_as_float(payload.get("relative_speed", 1.0)),
        profiles=profiles,
        ucr_data_root=str(payload["ucr_data_root"]),
        external_data_root=str(payload["external_data_root"]),
        ssh_host=None if payload.get("ssh_host") is None else str(payload["ssh_host"]),
        ssh_port=_as_int(payload.get("ssh_port", 22)),
        ssh_key=None if payload.get("ssh_key") is None else str(payload["ssh_key"]),
        concurrency=tuple(concurrency),
        enabled=bool(payload.get("enabled", True)),
    )
    if not host.gpus:
        message = f"host {host.name} has no GPUs"
        raise ValueError(message)
    if host.relative_speed <= 0:
        message = f"host {host.name} relative_speed must be positive"
        raise ValueError(message)
    if transport == "ssh" and (host.ssh_host is None or host.ssh_key is None):
        message = f"SSH host {host.name} requires ssh_host and ssh_key"
        raise ValueError(message)
    return host


def load_deployment_config(path: Path) -> DeploymentConfig:
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    model_profiles_payload = cast("dict[str, object]", payload["model_profiles"])
    model_profiles = tuple(
        sorted(
            (str(model), _identifier(str(profile)))
            for model, profile in model_profiles_payload.items()
        )
    )
    if set(dict(model_profiles)) != set(MODELS):
        missing = set(MODELS) - set(dict(model_profiles))
        extra = set(dict(model_profiles)) - set(MODELS)
        message = f"model_profiles mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        raise ValueError(message)
    class_order = tuple(
        cast("JobClass", value)
        for value in cast("list[object]", payload.get("class_order", ["long", "medium", "short"]))
    )
    if set(class_order) != {"short", "medium", "long"} or len(class_order) != 3:
        message = "class_order must contain short, medium, and long exactly once"
        raise ValueError(message)
    config = DeploymentConfig(
        schema=str(payload.get("schema", "pac.balanced_hpo_deployment_config.v1")),
        model_profiles=model_profiles,
        hosts=tuple(_host(host) for host in cast("list[dict[str, object]]", payload["hosts"])),
        class_order=cast("tuple[JobClass, ...]", class_order),
        poll_seconds=_as_int(payload.get("poll_seconds", 30)),
        max_attempts=_as_int(payload.get("max_attempts", 3)),
    )
    if not any(host.enabled for host in config.hosts):
        message = "deployment config has no enabled hosts"
        raise ValueError(message)
    required_profiles = set(dict(config.model_profiles).values())
    available_profiles = {
        profile.name for host in config.hosts if host.enabled for profile in host.profiles
    }
    missing_profiles = required_profiles - available_profiles
    if missing_profiles:
        message = f"no enabled host provides profiles: {sorted(missing_profiles)}"
        raise ValueError(message)
    return config


def _lanes(
    config: DeploymentConfig,
    job_class: JobClass,
    profile: str,
) -> list[WorkerLane]:
    lanes: list[WorkerLane] = []
    for host in config.hosts:
        if not host.enabled or host.profile(profile) is None:
            continue
        lanes.extend(
            WorkerLane(
                name=(f"{host.name}-gpu{gpu}-{job_class}-{profile}-p{process_index:02d}"),
                host=host.name,
                gpu=gpu,
                process_index=process_index,
                job_class=job_class,
                profile=profile,
                relative_speed=host.relative_speed,
            )
            for gpu in host.gpus
            for process_index in range(host.process_count(job_class))
        )
    return lanes


def _assign_lpt(
    jobs: list[BalancedHPOJob],
    lanes: list[WorkerLane],
) -> dict[str, list[BalancedHPOJob]]:
    if jobs and not lanes:
        message = f"no worker lane can run {jobs[0].job_class}/{jobs[0].model}"
        raise RuntimeError(message)
    assignments = {lane.name: [] for lane in lanes}
    loads = {lane.name: 0.0 for lane in lanes}
    lane_by_name = {lane.name: lane for lane in lanes}
    for job in sorted(jobs, key=lambda item: (-item.estimated_seconds, item.key)):
        lane_name = min(
            loads,
            key=lambda name: (
                loads[name] / lane_by_name[name].relative_speed,
                name,
            ),
        )
        assignments[lane_name].append(job)
        loads[lane_name] += job.estimated_seconds
    return assignments


def plan_stage(  # noqa: C901 - validation and deterministic scheduling stay together
    root: Path = DEFAULT_ROOT,
    *,
    stage: Literal["stage1", "stage2", "final"],
    config: DeploymentConfig,
) -> dict[str, object]:
    master = root / stage / "master.jsonl"
    jobs = load_manifest(master)
    units: list[DeploymentUnit] = []
    assigned_keys: list[str] = []
    profiles = sorted(set(dict(config.model_profiles).values()))
    for job_class in config.class_order:
        for profile in profiles:
            profile_jobs = [
                job
                for job in jobs
                if job.job_class == job_class and config.profile_for_model(job.model) == profile
            ]
            if not profile_jobs:
                continue
            lanes = _lanes(config, job_class, profile)
            assignments = _assign_lpt(profile_jobs, lanes)
            lane_by_name = {lane.name: lane for lane in lanes}
            for lane_name, lane_jobs in sorted(assignments.items()):
                if not lane_jobs:
                    continue
                # Keep compile-equivalent fits adjacent inside the persistent
                # worker.  LPT decides ownership; this ordering only amortizes
                # Python imports, Triton compilation, and allocator warm-up.
                lane_jobs.sort(
                    key=lambda job: (
                        job.model,
                        job.suite,
                        job.dataset,
                        job.width,
                        job.modes or 0,
                        job.architecture,
                        job.recipe.batch_size,
                        job.key,
                    )
                )
                manifest = root / stage / "deployment" / job_class / profile / f"{lane_name}.jsonl"
                content = "".join(
                    json.dumps(job.payload(), sort_keys=True) + "\n" for job in lane_jobs
                )
                write_once(manifest, content)
                seconds = sum(job.estimated_seconds for job in lane_jobs)
                units.append(
                    DeploymentUnit(
                        lane=lane_by_name[lane_name],
                        manifest=str(manifest),
                        jobs=len(lane_jobs),
                        estimated_seconds=seconds,
                        normalized_seconds=seconds / lane_by_name[lane_name].relative_speed,
                    )
                )
                assigned_keys.extend(job.key for job in lane_jobs)
    master_keys = [job.key for job in jobs]
    if len(assigned_keys) != len(set(assigned_keys)):
        message = f"{stage} deployment assigns at least one logical key twice"
        raise RuntimeError(message)
    if set(assigned_keys) != set(master_keys):
        missing = set(master_keys) - set(assigned_keys)
        extra = set(assigned_keys) - set(master_keys)
        message = f"{stage} deployment mismatch: missing={len(missing)}, extra={len(extra)}"
        raise RuntimeError(message)

    waves: list[dict[str, object]] = []
    for job_class in config.class_order:
        for profile in profiles:
            wave_units = [
                unit
                for unit in units
                if unit.lane.job_class == job_class and unit.lane.profile == profile
            ]
            if wave_units:
                waves.append(
                    {
                        "job_class": job_class,
                        "profile": profile,
                        "units": [asdict(unit) for unit in wave_units],
                        "jobs": sum(unit.jobs for unit in wave_units),
                        "estimated_wall_seconds": max(
                            unit.normalized_seconds for unit in wave_units
                        ),
                    }
                )
    payload: dict[str, object] = {
        "schema": "pac.balanced_hpo_deployment.v1",
        "stage": stage,
        "jobs": len(jobs),
        "units": len(units),
        "class_order": list(config.class_order),
        "model_profiles": dict(config.model_profiles),
        "hosts": [
            {
                "name": host.name,
                "transport": host.transport,
                "repo": host.repo,
                "gpu_type": host.gpu_type,
                "gpus": list(host.gpus),
                "relative_speed": host.relative_speed,
                "profiles": [asdict(profile) for profile in host.profiles],
                "concurrency": {
                    job_class: host.process_count(job_class)
                    for job_class in cast(
                        "tuple[JobClass, ...]",
                        ("short", "medium", "long"),
                    )
                },
                "enabled": host.enabled,
            }
            for host in config.hosts
        ],
        "waves": waves,
    }
    write_once(
        root / stage / "deployment.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def audit_deployment(
    root: Path,
    *,
    stage: Literal["stage1", "stage2", "final"],
) -> dict[str, object]:
    deployment_path = root / stage / "deployment.json"
    deployment = cast(
        "dict[str, object]",
        json.loads(deployment_path.read_text(encoding="utf-8")),
    )
    assigned: list[str] = []
    manifests: list[str] = []
    for wave in cast("list[dict[str, object]]", deployment["waves"]):
        for unit in cast("list[dict[str, object]]", wave["units"]):
            manifest = str(unit["manifest"])
            manifests.append(manifest)
            assigned.extend(job.key for job in load_manifest(Path(manifest)))
    master = load_manifest(root / stage / "master.jsonl")
    master_keys = {job.key for job in master}
    duplicate_count = len(assigned) - len(set(assigned))
    missing = master_keys - set(assigned)
    extra = set(assigned) - master_keys
    return {
        "schema": "pac.balanced_hpo_deployment_audit.v1",
        "stage": stage,
        "master_jobs": len(master),
        "manifests": len(manifests),
        "assigned_jobs": len(assigned),
        "duplicate_assignments": duplicate_count,
        "missing_assignments": len(missing),
        "extra_assignments": len(extra),
        "ok": duplicate_count == 0 and not missing and not extra,
    }


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DeploymentConfig",
    "DeploymentUnit",
    "HostSpec",
    "RuntimeProfile",
    "WorkerLane",
    "audit_deployment",
    "load_deployment_config",
    "plan_stage",
]
