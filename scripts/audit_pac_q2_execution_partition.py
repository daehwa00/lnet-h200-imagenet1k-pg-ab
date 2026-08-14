#!/usr/bin/env python3
# ruff: noqa: EM101, EM102, T201, TRY003
"""Audit the immutable Q2-final job partition used by distributed workers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key_digest(keys: set[str]) -> str:
    payload = "".join(f"{key}\n" for key in sorted(keys)).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_recovery_manifests(
    recovery_dir: Path, sealed_keys: set[str]
) -> dict[str, Any]:
    paths = sorted(recovery_dir.glob("*.jsonl")) if recovery_dir.is_dir() else []
    rows = [row for path in paths for row in _read_jsonl(path)]
    keys = {str(row["key"]) for row in rows}
    extra = sorted(keys - sealed_keys)
    if extra:
        raise RuntimeError(f"recovery manifests contain unsealed keys: {extra[:1]}")
    manifests: dict[str, Any] = {}
    for path in paths:
        path_keys = [str(row["key"]) for row in _read_jsonl(path)]
        manifests[path.name] = {
            "jobs": len(path_keys),
            "unique_jobs": len(set(path_keys)),
            "key_digest_sha256": _key_digest(set(path_keys)),
            "manifest_sha256": _file_sha256(path),
        }
    return {
        "recovery_manifest_rows": len(rows),
        "recovery_unique_jobs": len(keys),
        "recovery_keys_subset_sealed": not extra,
        "recovery_key_digest_sha256": _key_digest(keys),
        "recovery_manifests": manifests,
    }


def audit_partition(
    campaign_root: Path,
    execution_manifest_dir: Path,
    recovery_manifest_dir: Path | None = None,
) -> dict[str, Any]:
    sealed_paths = sorted((campaign_root / "q2_final/manifests").glob("*.jsonl"))
    worker_paths = sorted(execution_manifest_dir.glob("worker-*.jsonl"))
    if not sealed_paths:
        raise RuntimeError("no sealed Q2-final manifests found")
    if not worker_paths:
        raise RuntimeError("no execution worker manifests found")

    sealed_rows = [row for path in sealed_paths for row in _read_jsonl(path)]
    worker_rows = [row for path in worker_paths for row in _read_jsonl(path)]
    sealed_counts = Counter(str(row["key"]) for row in sealed_rows)
    worker_counts = Counter(str(row["key"]) for row in worker_rows)
    sealed_duplicates = sorted(key for key, count in sealed_counts.items() if count != 1)
    worker_duplicates = sorted(key for key, count in worker_counts.items() if count != 1)
    if sealed_duplicates:
        raise RuntimeError(f"sealed manifests contain duplicate keys: {sealed_duplicates[:1]}")
    if worker_duplicates:
        raise RuntimeError(f"worker manifests contain duplicate keys: {worker_duplicates[:1]}")

    sealed_keys = set(sealed_counts)
    worker_keys = set(worker_counts)
    extra = sorted(worker_keys - sealed_keys)
    if extra:
        raise RuntimeError(f"worker manifests contain unsealed keys: {extra[:1]}")
    excluded = sorted(sealed_keys - worker_keys)
    completed_by_key: dict[str, Path] = {}
    for path in sorted((campaign_root / "q2_final/completed").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        completed_by_key[str(row["job_key"])] = path
    not_completed = [key for key in excluded if key not in completed_by_key]
    if not_completed:
        raise RuntimeError(
            "sealed keys omitted from worker manifests are not completed: "
            f"{not_completed[:1]}"
        )

    workers: dict[str, Any] = {}
    for path in worker_paths:
        keys = {str(row["key"]) for row in _read_jsonl(path)}
        workers[path.name] = {
            "jobs": len(keys),
            "key_digest_sha256": _key_digest(keys),
            "manifest_sha256": _file_sha256(path),
        }
    recovery_dir = recovery_manifest_dir or (
        campaign_root / "provenance/q2_final_recovery_manifests"
    )
    recovery = _audit_recovery_manifests(recovery_dir, sealed_keys)
    return {
        "schema": "pac_alphabet_q2_execution_partition_audit.v1",
        "status": "PASS",
        "sealed_jobs": len(sealed_keys),
        "execution_manifest_jobs": len(worker_keys),
        "completed_before_split_jobs": len(excluded),
        "partition_union_jobs": len(worker_keys) + len(excluded),
        "partition_union_matches_sealed": worker_keys | set(excluded) == sealed_keys,
        "sealed_key_digest_sha256": _key_digest(sealed_keys),
        "execution_key_digest_sha256": _key_digest(worker_keys),
        "completed_before_split": [
            {
                "job_key": key,
                "result_sha256": _file_sha256(completed_by_key[key]),
            }
            for key in excluded
        ],
        "workers": workers,
        **recovery,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path(".omx/results/pac-alphabet-q1q2-final-20260719"),
    )
    parser.add_argument(
        "--execution-manifest-dir",
        type=Path,
        default=Path(
            ".omx/results/pac-alphabet-q1q2-final-20260719/"
            "provenance/q2_final_execution_manifests"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".omx/results/pac-alphabet-q1q2-final-20260719/"
            "provenance/q2_final_execution_partition_audit.json"
        ),
    )
    parser.add_argument(
        "--recovery-manifest-dir",
        type=Path,
        default=Path(
            ".omx/results/pac-alphabet-q1q2-final-20260719/"
            "provenance/q2_final_recovery_manifests"
        ),
    )
    args = parser.parse_args()
    audit = audit_partition(
        args.campaign_root,
        args.execution_manifest_dir,
        args.recovery_manifest_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "PASS: Q2 execution partition covers "
        f"{audit['partition_union_jobs']}/{audit['sealed_jobs']} sealed jobs across "
        f"{len(audit['workers'])} worker manifests"
    )


if __name__ == "__main__":
    main()
