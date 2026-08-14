# ruff: noqa: T201
"""Freeze and audit the exact 37-dataset Wave-1 Stage-1 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

from lnet.pac_broad_benchmark_distributed import (
    audit_deployment,
    load_deployment_config,
    plan_jobs,
)
from lnet.pac_broad_benchmark_queue import UEA_30, BenchmarkJob
from lnet.pac_broad_benchmark_worker import runtime_code_sha256
from lnet.pac_wave1_campaign import (
    DEFAULT_ROOT,
    EXTERNAL_DATASETS,
    FORECAST_DATASETS,
    MODELS,
    audit_stage1,
    expected_counts,
    stage1_jobs,
)

BASE_CONFIG = Path("optimization/hosts/new3_10model_5gpu.local.json")
DEFAULT_CONFIG = Path("optimization/hosts/wave1_3gpu.local.json")
LOCAL_DATA_ROOT = Path("data/external")
MAX_MICROBATCH_TEMPORAL_TOKENS = 512 * 1024
MAX_MICROBATCH_RAW_VALUES = 16 * 1024 * 1024
MAX_TRANSFORMER_ATTENTION_PAIRS = 384 * 1024 * 1024
ORCHESTRATION_CODE_PATHS = (
    "scripts/prepare_wave1_campaign.py",
    "scripts/supervise_wave1_campaign.py",
    "scripts/supervise_broad_campaign.py",
    "scripts/run_broad_benchmark_worker.py",
    "scripts/prepare_wave1_forecasting_data.py",
    "scripts/prepare_uea_30.py",
    "scripts/prepare_wave1_irregular_tasks.py",
    "src/lnet/pac_wave1_campaign.py",
    "src/lnet/pac_fast_completion.py",
    "src/lnet/pac_broad_benchmark_completion.py",
    "src/lnet/pac_broad_benchmark_distributed.py",
    "src/lnet/pac_broad_benchmark_queue.py",
    "src/lnet/pac_uea_tasks.py",
    "src/lnet/pac_wave1_irregular_tasks.py",
)


def _write_once(path: Path, payload: object) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        message = f"refusing to overwrite a different frozen artifact: {path}"
        raise FileExistsError(message)
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def orchestration_sha256(project: Path | None = None) -> str:
    """Bind the local queue, transition, supervision, and preparation closure."""
    root = Path(__file__).resolve().parents[1] if project is None else project
    digest = hashlib.sha256()
    for relative in ORCHESTRATION_CODE_PATHS:
        path = root / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
    }


def freeze_runtime_snapshot(destination: Path, project: Path) -> dict[str, object]:
    """Create the immutable source tree deployed by every Wave-1 stage."""
    if destination.exists():
        overlapping_drift = [
            relative
            for relative in ORCHESTRATION_CODE_PATHS
            if (destination / relative).is_file()
            and (
                not (project / relative).is_file()
                or (destination / relative).read_bytes()
                != (project / relative).read_bytes()
            )
        ]
        if overlapping_drift:
            message = (
                "live orchestration code disagrees with the frozen runtime snapshot: "
                + ", ".join(overlapping_drift)
            )
            raise RuntimeError(message)
        return {
            "path": str(destination),
            "code_sha256": runtime_code_sha256(destination),
            "files": sum(path.is_file() for path in destination.rglob("*")),
        }
    live_before = runtime_code_sha256(project)
    orchestration_before = orchestration_sha256(project)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        shutil.copytree(
            project / "src",
            temporary / "src",
            ignore=_runtime_ignore,
        )
        shutil.copytree(
            project / "csrc",
            temporary / "csrc",
            ignore=_runtime_ignore,
        )
        (temporary / "scripts").mkdir()
        shutil.copy2(
            project / "scripts/run_broad_benchmark_worker.py",
            temporary / "scripts/run_broad_benchmark_worker.py",
        )
        shutil.copy2(project / "pyproject.toml", temporary / "pyproject.toml")
        digest = runtime_code_sha256(temporary)
        live_after = runtime_code_sha256(project)
        orchestration_after = orchestration_sha256(project)
        if (
            live_before != digest
            or digest != live_after
            or orchestration_before != orchestration_after
        ):
            message = "Wave-1 source changed while freezing the runtime snapshot"
            raise RuntimeError(message)
        files = sum(path.is_file() for path in temporary.rglob("*"))
        temporary.replace(destination)
        return {
            "path": str(destination),
            "code_sha256": digest,
            "files": files,
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def dataset_keys() -> tuple[str, ...]:
    return (
        *FORECAST_DATASETS,
        *(dataset for dataset, _ in EXTERNAL_DATASETS),
    )


def data_shards() -> tuple[str, ...]:
    return (
        *(f"forecasting:{dataset}:forecasting" for dataset in FORECAST_DATASETS),
        *(
            f"external:{dataset}:{endpoint}"
            for dataset, endpoint in EXTERNAL_DATASETS
        ),
    )


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _preparation_record(data_root: Path, dataset: str) -> dict[str, object]:
    if dataset in FORECAST_DATASETS:
        audit = _load_json(data_root / "wave1-forecasting-audit.json")
        if audit.get("all_source_checksums_verified") is not True:
            return {}
        return next(
            (
                cast("dict[str, object]", row)
                for row in cast("list[object]", audit.get("datasets", []))
                if cast("dict[str, object]", row).get("dataset") == dataset
            ),
            {},
        )
    if dataset in {"human-activity", "ushcn-daily"}:
        audit = _load_json(data_root / "wave1-irregular-preparation-audit.json")
        sources = cast("list[dict[str, object]]", audit.get("sources", []))
        if (
            len(sources) != 2
            or any(row.get("sha256") != row.get("expected_sha256") for row in sources)
        ):
            return {}
        return next(
            (
                cast("dict[str, object]", row)
                for row in cast("list[object]", audit.get("tasks", []))
                if cast("dict[str, object]", row).get("dataset") == dataset
            ),
            {},
        )
    if dataset in UEA_30:
        record = _load_json(data_root / "provenance" / f"{dataset}.json")
        revision = cast("dict[str, object]", record.get("source_revision", {}))
        return record if revision.get("verified_download_manifest") is True else {}
    return {}


def _record_hash(record: dict[str, object], *, full: bool) -> str | None:
    keys = (
        ("full_sha256", "prepared_sha256", "full_artifact_sha256")
        if full
        else ("selection_sha256", "selection_artifact_sha256")
    )
    return next((str(record[key]) for key in keys if key in record), None)


def _identity_verified(dataset: str, record: dict[str, object]) -> bool:
    if dataset in FORECAST_DATASETS:
        identity = cast("dict[str, object]", record.get("selection_full_identity", {}))
        return (
            record.get("selection_contains_test_tensors") is False
            and record.get("raw_full_identity_verified") is True
            and identity.get("verified") is True
            and len(str(identity.get("selection_split_sha256", ""))) == 64
        )
    identity = cast("dict[str, object]", record.get("selection_full_identity", {}))
    return (
        identity.get("verified") is True
        and len(str(identity.get("selection_split_sha256", ""))) == 64
        and record.get("selection_contains_test_tensors", False) is False
    )


def _dataset_shape(data_root: Path, dataset: str) -> tuple[int, int, int]:
    record = _preparation_record(data_root, dataset)
    if dataset in FORECAST_DATASETS:
        shape = cast("list[int]", record["train_shape"])
        return int(shape[0]), int(shape[1]), int(shape[2])
    if dataset in {"human-activity", "ushcn-daily"}:
        irregular = cast("dict[str, int]", record)
        return (
            int(irregular["train_count"]),
            int(irregular["sequence_length"]),
            int(irregular["input_dim"]),
        )
    counts = cast("dict[str, int]", record["counts"])
    shape = cast("dict[str, int]", record["shape"])
    return int(counts["train"]), int(shape["train_steps"]), int(shape["channels"])


def safe_microbatch(
    effective_batch_size: int,
    *,
    sequence_length: int,
    input_dim: int,
    model: str | None = None,
) -> int:
    caps = [
        effective_batch_size,
        max(1, MAX_MICROBATCH_TEMPORAL_TOKENS // sequence_length),
        max(1, MAX_MICROBATCH_RAW_VALUES // (sequence_length * input_dim)),
    ]
    if model == "transformer":
        caps.append(
            max(1, MAX_TRANSFORMER_ATTENTION_PAIRS // (sequence_length * sequence_length))
        )
    cap = min(caps)
    return max(
        candidate
        for candidate in range(1, effective_batch_size + 1)
        if effective_batch_size % candidate == 0 and candidate <= cap
    )


def resource_tuned_jobs(
    jobs: tuple[BenchmarkJob, ...],
    data_root: Path,
) -> tuple[BenchmarkJob, ...]:
    shapes = {dataset: _dataset_shape(data_root, dataset) for dataset in dataset_keys()}
    tuned: list[BenchmarkJob] = []
    for job in jobs:
        train_count, sequence_length, input_dim = shapes[job.dataset]
        effective_batch = job.recipe.effective_batch_size
        microbatch = safe_microbatch(
            effective_batch,
            sequence_length=sequence_length,
            input_dim=input_dim,
            model=job.model,
        )
        steps = math.ceil(train_count / microbatch) * job.epochs
        temporal_activation_mb = (
            microbatch * sequence_length * job.width * 4 / (1024 * 1024)
        )
        raw_input_mb = microbatch * sequence_length * input_dim * 4 / (1024 * 1024)
        activation_multiplier = 24.0 if job.model == "transformer" else 8.0
        peak_memory_mb = math.ceil(
            max(
                job.estimated_peak_memory_mb,
                temporal_activation_mb * activation_multiplier + raw_input_mb * 3.0,
            )
        )
        tuned.append(
            replace(
                job,
                microbatch_size=microbatch,
                gradient_accumulation_steps=effective_batch // microbatch,
                estimated_peak_memory_mb=peak_memory_mb,
                estimated_seconds=max(
                    job.estimated_seconds,
                    steps * (0.002 + sequence_length / 200_000),
                ),
            )
        )
    return tuple(tuned)


def write_config(source: Path, destination: Path) -> dict[str, object]:
    payload = cast("dict[str, object]", json.loads(source.read_text(encoding="utf-8")))
    active_hosts: list[dict[str, object]] = []
    for original in cast("list[dict[str, object]]", payload["hosts"]):
        if str(original["name"]) not in {"secondary_gpu", "rtx3080ti-1", "rtx3080ti-2"}:
            continue
        host = dict(original)
        name = str(host["name"])
        host["enabled"] = True
        host.pop("data_shard_set", None)
        host["data_shards"] = list(data_shards())
        host["repo"] = (
            "<remote-home>/lnet-wave1-runtime-20260727"
            if name == "secondary_gpu"
            else "<remote-home>/lnet-wave1-runtime-20260727"
        )
        roots = dict(cast("dict[str, object]", host["data_roots"]))
        roots["external"] = (
            "<remote-home>/lnet-wave1-data-20260727"
            if name == "secondary_gpu"
            else "<remote-home>/lnet-wave1-data-20260727"
        )
        host["data_roots"] = roots
        active_hosts.append(host)
    if len(active_hosts) != 3:
        message = f"Wave 1 requires exactly three non-local_gpu hosts, found {len(active_hosts)}"
        raise RuntimeError(message)
    payload["hosts"] = active_hosts
    payload.pop("data_shard_sets", None)
    payload["schema"] = "alphabet.wave1.deployment_config.v1"
    payload["campaign"] = {
        "datasets": list(dataset_keys()),
        "models": list(MODELS),
        "protocol": "6 candidates -> top 2 seed 11 -> winner at seeds 23/31/43",
        "local_gpu_excluded": True,
    }
    _write_once(destination, payload)
    return payload


def audit_data(data_root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    provenance_failures: list[str] = []
    for dataset in dataset_keys():
        full = data_root / f"{dataset}.pt"
        selection = data_root / "selection-only" / f"{dataset}.pt"
        if not full.is_file() or not selection.is_file():
            missing.append(dataset)
            continue
        full_sha256 = _sha256(full)
        selection_sha256 = _sha256(selection)
        record = _preparation_record(data_root, dataset)
        provenance_ok = (
            bool(record)
            and _record_hash(record, full=True) == full_sha256
            and _record_hash(record, full=False) == selection_sha256
            and _identity_verified(dataset, record)
        )
        if not provenance_ok:
            provenance_failures.append(dataset)
        rows.append(
            {
                "dataset": dataset,
                "full_path": str(full),
                "full_bytes": full.stat().st_size,
                "full_sha256": full_sha256,
                "selection_path": str(selection),
                "selection_bytes": selection.stat().st_size,
                "selection_sha256": selection_sha256,
                "preparation_provenance_verified": provenance_ok,
                "selection_full_identity_verified": (
                    bool(record) and _identity_verified(dataset, record)
                ),
            }
        )
    return {
        "schema": "alphabet.wave1_data_presence_audit.v1",
        "ok": (
            not missing
            and not provenance_failures
            and len(rows) == len(dataset_keys())
        ),
        "datasets": rows,
        "missing": missing,
        "provenance_failures": provenance_failures,
        "full_artifacts": len(rows),
        "selection_only_artifacts": len(rows),
        "selection_workers_receive_full_artifacts": False,
    }


def prepare(
    root: Path,
    config_path: Path,
    base_config: Path,
    data_root: Path,
) -> dict[str, object]:
    runtime_snapshot = freeze_runtime_snapshot(
        root / "frozen-runtime",
        Path.cwd(),
    )
    config = write_config(base_config, config_path)
    queue_audit = audit_stage1()
    if not queue_audit["ok"]:
        message = f"Wave-1 queue audit failed: {queue_audit}"
        raise RuntimeError(message)
    data_audit = audit_data(data_root)
    if not data_audit["ok"]:
        message = f"Wave-1 data audit failed: {data_audit['missing']}"
        raise RuntimeError(message)

    tuned_jobs = resource_tuned_jobs(stage1_jobs(), data_root)
    jobs = tuple(
        replace(
            job,
            comparison_group=f"{job.comparison_group}:lane-{index % 3}",
        )
        for index, job in enumerate(tuned_jobs)
    )
    counts = expected_counts()
    maximum_peak_memory_mb = max(job.estimated_peak_memory_mb for job in jobs)
    for host in cast("list[dict[str, object]]", config["hosts"]):
        if 2 * maximum_peak_memory_mb > int(cast("str | int", host["memory_mb"])):
            message = (
                f"{host['name']} cannot safely co-reside two Wave-1 workers: "
                f"2*{maximum_peak_memory_mb}MB > {host['memory_mb']}MB"
            )
            raise RuntimeError(message)
    _write_once(
        root / "contract.json",
        {
            "schema": "alphabet.wave1.contract.v1",
            "state": "stage1_frozen",
            "datasets": list(dataset_keys()),
            "models": list(MODELS),
            "counts": counts,
            "stage1_epochs": 15,
            "stage2_epochs": 30,
            "final_epochs": 60,
            "selection_seed": 7,
            "confirmation_seeds": [11],
            "final_seeds": [23, 31, 43],
            "code_sha256": runtime_snapshot["code_sha256"],
            "runtime_snapshot": runtime_snapshot,
            "orchestration_sha256": orchestration_sha256(),
            "resource_policy": {
                "shape_source": "checksum-verified preparation provenance",
                "maximum_temporal_tokens_per_microbatch": (
                    MAX_MICROBATCH_TEMPORAL_TOKENS
                ),
                "maximum_raw_values_per_microbatch": MAX_MICROBATCH_RAW_VALUES,
                "maximum_transformer_attention_pairs_per_microbatch": (
                    MAX_TRANSFORMER_ATTENTION_PAIRS
                ),
                "effective_batch_size_preserved_by_gradient_accumulation": True,
                "workers_per_gpu": 2,
                "co_resident_peak_memory_gate": "2 * max job estimate <= host memory",
            },
            "test_policy": (
                "Stage 1/2 hosts receive selection-only artifacts; full TEST "
                "artifacts are copied only after the Stage-2 winner is frozen"
            ),
            "hosts": [
                str(host["name"])
                for host in cast("list[dict[str, object]]", config["hosts"])
            ],
            "local_gpu_excluded": True,
        },
    )
    _write_once(root / "data-audit.json", data_audit)
    deployment = plan_jobs(
        root,
        stage="stage1",
        jobs=jobs,
        config=load_deployment_config(config_path),
    )
    deployment_audit = audit_deployment(root, stage="stage1")
    if not deployment_audit["ok"] or deployment["blocked_jobs"]:
        message = f"Wave-1 deployment audit failed: {deployment_audit}"
        raise RuntimeError(message)
    payload: dict[str, object] = {
        "root": str(root),
        "config": str(config_path),
        "queue_audit": queue_audit,
        "data_audit": {
            "ok": data_audit["ok"],
            "full_artifacts": data_audit["full_artifacts"],
            "selection_only_artifacts": data_audit["selection_only_artifacts"],
        },
        "deployment": deployment,
        "deployment_audit": deployment_audit,
        "resource_audit": {
            "minimum_microbatch": min(job.microbatch_size for job in jobs),
            "maximum_microbatch": max(job.microbatch_size for job in jobs),
            "adjusted_jobs": sum(
                job.microbatch_size != job.recipe.effective_batch_size for job in jobs
            ),
            "maximum_estimated_peak_memory_mb": max(
                job.estimated_peak_memory_mb for job in jobs
            ),
            "two_worker_memory_gate_passed": True,
        },
    }
    _write_once(root / "preparation-audit.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=LOCAL_DATA_ROOT)
    arguments = parser.parse_args()
    print(
        json.dumps(
            prepare(
                arguments.root,
                arguments.config,
                arguments.base_config,
                arguments.data_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
