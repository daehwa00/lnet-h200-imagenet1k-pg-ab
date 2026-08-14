#!/usr/bin/env python3
# ruff: noqa: EM101, T201, TRY003
"""Split the Q2 local_gpu-GPU0 lane without changing any scientific job.

All Speech Commands jobs remain on one speech-capable worker.  Remaining
non-Speech jobs are assigned to that worker or a second worker by estimated
remaining load.  Completed keys contribute zero load and are retained in the
partition so the execution-manifest union remains identical to the original
immutable local_gpu0 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _completed_keys(completed_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in completed_dir.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "done":
            keys.add(str(row.get("job_key") or row.get("key")))
    return keys


def partition_rows(
    rows: list[dict[str, Any]], completed: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return disjoint speech-capable and non-Speech balanced lanes."""
    keys = [str(row["key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("source local_gpu0 manifest contains duplicate keys")
    for row in rows:
        if not isinstance(row.get("job"), dict):
            raise TypeError("manifest row is missing its immutable job payload")

    index = {str(row["key"]): position for position, row in enumerate(rows)}
    speech = [row for row in rows if row["job"]["dataset"] == "speech-commands"]
    non_speech = [row for row in rows if row["job"]["dataset"] != "speech-commands"]
    if not speech or not non_speech:
        raise RuntimeError("local_gpu0 split requires both Speech Commands and non-Speech jobs")

    speech_lane = list(speech)
    nonspeech_lane: list[dict[str, Any]] = []
    speech_load = sum(
        0.0 if str(row["key"]) in completed else float(row["job"].get("estimated_seconds", 0.0))
        for row in speech
    )
    nonspeech_load = 0.0

    pending_non_speech = [row for row in non_speech if str(row["key"]) not in completed]
    completed_non_speech = [row for row in non_speech if str(row["key"]) in completed]
    pending_non_speech.sort(
        key=lambda row: float(row["job"].get("estimated_seconds", 0.0)),
        reverse=True,
    )
    for row in pending_non_speech:
        seconds = float(row["job"].get("estimated_seconds", 0.0))
        if speech_load <= nonspeech_load:
            speech_lane.append(row)
            speech_load += seconds
        else:
            nonspeech_lane.append(row)
            nonspeech_load += seconds
    for row in completed_non_speech:
        target = speech_lane if len(speech_lane) <= len(nonspeech_lane) else nonspeech_lane
        target.append(row)

    speech_lane.sort(key=lambda row: index[str(row["key"])])
    nonspeech_lane.sort(key=lambda row: index[str(row["key"])])
    speech_keys = {str(row["key"]) for row in speech_lane}
    nonspeech_keys = {str(row["key"]) for row in nonspeech_lane}
    if speech_keys & nonspeech_keys or speech_keys | nonspeech_keys != set(keys):
        raise RuntimeError("split lanes are not an exact disjoint partition")
    if any(row["job"]["dataset"] == "speech-commands" for row in nonspeech_lane):
        raise RuntimeError("non-Speech lane received a Speech Commands job")
    return speech_lane, nonspeech_lane


def _remaining_load(rows: list[dict[str, Any]], completed: set[str]) -> float:
    return sum(
        0.0 if str(row["key"]) in completed else float(row["job"].get("estimated_seconds", 0.0))
        for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path(".omx/results/pac-alphabet-q1q2-final-20260719"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".omx/tmp/q2-local_gpu0-split-20260720"),
    )
    args = parser.parse_args()
    execution_dir = args.campaign_root / "provenance/q2_final_execution_manifests"
    archive_dir = (
        args.campaign_root
        / "provenance/q2_final_superseded_execution_manifests/local_gpu0-single-worker-20260720"
    )
    source = execution_dir / "worker-local_gpu0.jsonl"
    archived_source = archive_dir / source.name
    if not source.is_file() and archived_source.is_file():
        source = archived_source
    if not source.is_file():
        raise FileNotFoundError("authoritative local_gpu0 execution manifest is missing")

    rows = _read_jsonl(source)
    completed = _completed_keys(args.campaign_root / "q2_final/completed")
    speech_lane, nonspeech_lane = partition_rows(rows, completed)
    names = {
        "speech_capable": "worker-local_gpu0-speech-capable.jsonl",
        "nonspeech": "worker-local_gpu0-nonspeech.jsonl",
    }
    payloads = {
        "speech_capable": _payload(speech_lane),
        "nonspeech": _payload(nonspeech_lane),
    }

    archive_dir.mkdir(parents=True, exist_ok=True)
    if not archived_source.exists():
        _atomic_write(archived_source, _payload(rows))
    for label, name in names.items():
        _atomic_write(args.output_dir / name, payloads[label])
        _atomic_write(execution_dir / name, payloads[label])
    original = execution_dir / "worker-local_gpu0.jsonl"
    if original.exists():
        original.unlink()

    audit = {
        "schema": "pac_alphabet_q2_local_gpu0_split.v1",
        "status": "PASS",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reason": "use otherwise-idle local_gpu GPU0 capacity without concurrent Speech jobs",
        "source_manifest": source.name,
        "source_manifest_sha256": _sha256(archived_source),
        "source_jobs": len(rows),
        "completed_at_split": sum(str(row["key"]) in completed for row in rows),
        "lanes": {
            "speech_capable": {
                "manifest": names["speech_capable"],
                "jobs": len(speech_lane),
                "remaining_jobs": sum(str(row["key"]) not in completed for row in speech_lane),
                "remaining_estimated_seconds": _remaining_load(speech_lane, completed),
                "speech_jobs": sum(
                    row["job"]["dataset"] == "speech-commands" for row in speech_lane
                ),
            },
            "nonspeech": {
                "manifest": names["nonspeech"],
                "jobs": len(nonspeech_lane),
                "remaining_jobs": sum(str(row["key"]) not in completed for row in nonspeech_lane),
                "remaining_estimated_seconds": _remaining_load(nonspeech_lane, completed),
                "speech_jobs": 0,
            },
        },
    }
    _atomic_write(archive_dir / "audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
