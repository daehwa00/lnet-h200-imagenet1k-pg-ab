"""Build fixed-shape resume shards for the identity ALPHABET Q1-final sweep.

The original manifests intentionally balance estimated work across long-lived
workers. That is a poor fit for the fixed-shape CUDA Graph runtime because one
process encounters many sequence lengths and classifier widths. This helper
keeps every immutable job unchanged, skips completed keys, and writes one
process-sized shard per (task, capacity, recipe) group. A worker therefore sees
at most the five fixed-shape seed runs before exiting and releasing compile
caches.
"""

# pyright: reportExplicitAny=false

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SOURCE_PATHS = (
    "csrc/pac_cuda_conditional_switch.cu",
    "csrc/pac_cuda_fused_optimizer.cu",
    "csrc/pac_cuda_outer_graph.cu",
    "src/lnet/alphabet_backbone.py",
    "src/lnet/pac_training.py",
    "src/lnet/pac_baseline_fairness_maximal.py",
    "src/lnet/pac_external_benchmarks.py",
    "src/lnet/pac_external_tasks.py",
    "src/lnet/pac_efp16_exact_split_training.py",
    "src/lnet/pac_cuda_conditional_matrix_exp.py",
    "src/lnet/pac_cuda_outer_graph.py",
    "src/lnet/pac_native_matrix_exp_vjp.py",
    "src/lnet/pac_pointwise_identity_capacity_campaign.py",
    "src/lnet/pac_pointwise_identity_capacity_cli.py",
    "src/lnet/pac_recurrence.py",
    "src/lnet/pac_tight_frame_models.py",
    "src/lnet/pac_triton_direct_stem_training.py",
    "src/lnet/pac_triton_parallel_static_recurrence.py",
    "src/lnet/pac_triton_recurrence_lag124.py",
    "src/lnet/pac_triton_recurrence_lag124_training.py",
    "src/lnet/pac_triton_recurrence_moments.py",
    "src/lnet/pac_triton_recurrence_moments_training.py",
    "src/lnet/pac_triton_recurrence_op.py",
    "src/lnet/pac_triton_small_qr.py",
    "src/lnet/pac_triton_terminal_reader_local_training.py",
    "src/lnet/pac_triton_terminal_reader_scan_training.py",
    "src/lnet/pac_triton_writer_reader_local_training.py",
)

# These are per-worker effective throughputs, not raw GPU peak rates.  Eight
# PRO workers share one device while each 4090 serves only two workers.  Live
# campaign measurements showed the original 1.35:1 weighting repeatedly
# drained every remote queue while hundreds of PRO-assigned jobs remained.
LANES = (
    *((f"pro-{index:02d}", "pro6000", 0, 1.0) for index in range(8)),
    *((f"local_gpu-g0-{index}", "local_gpu", 0, 2.0) for index in range(2)),
    *((f"local_gpu-g1-{index}", "local_gpu", 1, 2.0) for index in range(2)),
    *((f"kau-{index}", "secondary_gpu", 0, 2.0) for index in range(2)),
)

# The secondary_gpu cache contains every sealed Q1-final dataset except the
# 6.7 GiB full Speech Commands TEST artifact.  Keep those fixed-shape groups
# on hosts with a verified complete cache instead of copying a large dataset
# for a single assigned group.
HOST_DATASET_DENY: dict[str, frozenset[str]] = {
    "secondary_gpu": frozenset({"speech-commands"}),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace_label(path: Path, repository: Path) -> str:
    try:
        return str(path.relative_to(repository))
    except ValueError:
        parts = path.parts
        if ".omx" in parts:
            return str(Path(*parts[parts.index(".omx") :]))
        return str(path)


def _read_payloads(manifest_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(manifest_dir.glob("worker-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            payload = json.loads(line)
            key = str(payload["key"])
            previous = payloads.setdefault(key, payload)
            if previous["job"] != payload["job"]:
                message = f"immutable job collision for {key}"
                raise RuntimeError(message)
    return payloads


def _completed_rows(completed_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(completed_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[str(row["job_key"])] = row
    return rows


def _group_key(payload: dict[str, Any]) -> tuple[str, str, str, int]:
    job = payload["job"]
    return (str(job["suite"]), str(job["dataset"]), str(job["model"]), int(job["trial"]))


def prepare(root: Path, repository: Path) -> dict[str, Any]:
    final = root / "final"
    payloads = _read_payloads(final / "manifests")
    completed = _completed_rows(final / "completed")
    unknown = set(completed) - set(payloads)
    if unknown:
        message = f"completed directory contains {len(unknown)} unknown jobs"
        raise RuntimeError(message)

    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for key, payload in payloads.items():
        if key not in completed:
            groups[_group_key(payload)].append(payload)

    shard_dir = final / "aggressive-manifests"
    queue_dir = final / "aggressive-queues"
    shard_dir.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)
    for path in (*shard_dir.glob("group-*.jsonl"), *queue_dir.glob("*.txt")):
        path.unlink()

    group_records: list[dict[str, Any]] = []
    for index, (key, members) in enumerate(sorted(groups.items())):
        members.sort(key=lambda payload: int(payload["job"]["train_seed"]))
        path = shard_dir / f"group-{index:04d}.jsonl"
        path.write_text(
            "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in members),
            encoding="utf-8",
        )
        group_records.append(
            {
                "group": key,
                "path": _workspace_label(path, repository),
                "jobs": len(members),
                "estimated_seconds": sum(
                    float(item["job"]["estimated_seconds"]) for item in members
                ),
            }
        )

    lane_loads = {name: 0.0 for name, _host, _gpu, _speed in LANES}
    lane_groups: dict[str, list[dict[str, Any]]] = {name: [] for name in lane_loads}
    lane_specs = {name: (host, gpu, speed) for name, host, gpu, speed in LANES}
    ordered_groups = sorted(
        group_records,
        key=lambda item: float(item["estimated_seconds"]),
        reverse=True,
    )
    for group in ordered_groups:
        dataset = str(group["group"][1])
        eligible_lanes = tuple(
            name
            for name in lane_loads
            if dataset not in HOST_DATASET_DENY.get(lane_specs[name][0], frozenset())
        )
        if not eligible_lanes:
            message = f"no eligible lane has the required dataset: {dataset}"
            raise RuntimeError(message)
        lane = min(
            eligible_lanes,
            key=lambda name: lane_loads[name] / lane_specs[name][2],
        )
        lane_groups[lane].append(group)
        lane_loads[lane] += float(group["estimated_seconds"])

    queues: list[dict[str, Any]] = []
    for name, host, gpu, speed in LANES:
        path = queue_dir / f"{name}.txt"
        path.write_text(
            "".join(str(group["path"]) + "\n" for group in lane_groups[name]),
            encoding="utf-8",
        )
        queues.append(
            {
                "name": name,
                "host": host,
                "gpu": gpu,
                "relative_speed": speed,
                "groups": len(lane_groups[name]),
                "jobs": sum(int(group["jobs"]) for group in lane_groups[name]),
                "estimated_seconds": lane_loads[name],
                "queue": _workspace_label(path, repository),
            }
        )

    completed_digest = hashlib.sha256(
        "\n".join(
            f"{key}:{row.get('code_sha256')}:{row.get('provenance_sha256')}"
            for key, row in sorted(completed.items())
        ).encode()
    ).hexdigest()
    source_hashes = {
        path: _sha256(repository / path)
        for path in SOURCE_PATHS
        if (repository / path).is_file()
    }
    root_label = _workspace_label(root, repository)
    payload = {
        "schema": "identity_capacity_aggressive_resume.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "root": root_label,
        "original_jobs": len(payloads),
        "preserved_completed_jobs": len(completed),
        "preserved_completed_digest": completed_digest,
        "pending_jobs": sum(int(group["jobs"]) for group in group_records),
        "fixed_shape_groups": len(group_records),
        "source_hashes": source_hashes,
        "queues": queues,
    }
    report_path = root / "reports" / "aggressive_resume_provenance.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repository = args.repository.resolve()
    payload = prepare(args.root.resolve(), repository)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
