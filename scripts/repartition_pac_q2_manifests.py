"""Repartition Q2 manifests around the Speech Commands host-RAM bound.

This is an execution-only transformation: immutable jobs and job keys are
unchanged.  Speech Commands jobs are moved onto three local serial lanes and
one lane per local_gpu GPU.  Q2 calibration reads the sealed selection tensor, and
Q2 final reads an identity-checked full prepared tensor, avoiding the much
larger raw-WAV construction peak seen during Q1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import cast

SCHEMA = "pac_alphabet_q2_memory_repartition.v1"
TARGET_LANES = (0, 1, 2, 10, 13)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


JsonRow = dict[str, object]


def _job_set_digest(rows: list[JsonRow]) -> str:
    payload = "\n".join(sorted(str(row["key"]) for row in rows)).encode()
    return _digest(payload)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"expected a numeric scheduling estimate, got {value!r}"
        raise TypeError(message)
    return float(value)


def _load_manifests(
    paths: list[Path], stage: str
) -> tuple[dict[int, list[JsonRow]], dict[int, JsonRow], list[JsonRow], set[str]]:
    buckets: dict[int, list[JsonRow]] = {index: [] for index in range(20)}
    resources: dict[int, JsonRow] = {}
    rows: list[JsonRow] = []
    seen: set[str] = set()
    for path in paths:
        index = int(path.stem.rsplit("-", 1)[1])
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = cast("JsonRow", json.loads(line))
            key = str(row["key"])
            if key in seen:
                message = f"{stage}: duplicate job key before repartition: {key}"
                raise RuntimeError(message)
            seen.add(key)
            rows.append(row)
            resources.setdefault(index, deepcopy(cast("JsonRow", row["resource"])))
            job = cast("JsonRow", row["job"])
            if job["dataset"] != "speech-commands":
                buckets[index].append(row)
    return buckets, resources, rows, seen


def _assign_speech(
    buckets: dict[int, list[JsonRow]],
    resources: dict[int, JsonRow],
    rows: list[JsonRow],
) -> tuple[dict[int, float], dict[int, int], list[JsonRow]]:
    speech = sorted(
        (row for row in rows if cast("JsonRow", row["job"])["dataset"] == "speech-commands"),
        key=lambda row: (
            -_as_float(cast("JsonRow", row["job"])["estimated_seconds"]),
            str(row["key"]),
        ),
    )
    speech_load: dict[int, float] = dict.fromkeys(TARGET_LANES, 0.0)
    speech_counts: dict[int, int] = dict.fromkeys(TARGET_LANES, 0)
    for source in speech:
        target = min(TARGET_LANES, key=lambda lane: (speech_load[lane], lane))
        row = deepcopy(source)
        row["resource"] = deepcopy(resources[target])
        buckets[target].append(row)
        job = cast("JsonRow", row["job"])
        speech_load[target] += _as_float(job["estimated_seconds"])
        speech_counts[target] += 1
    return speech_load, speech_counts, speech


def repartition(output_root: Path, stage: str) -> JsonRow:
    manifest_root = output_root / stage / "manifests"
    paths = sorted(manifest_root.glob("worker-[0-9][0-9].jsonl"))
    if len(paths) != 20:
        message = f"{stage}: expected 20 canonical worker manifests, found {len(paths)}"
        raise RuntimeError(message)

    buckets, resources, rows, seen = _load_manifests(paths, stage)

    if set(resources) != set(range(20)):
        missing = sorted(set(range(20)) - set(resources))
        message = f"{stage}: no resource template for worker lanes {missing}"
        raise RuntimeError(message)

    speech_load, speech_counts, speech = _assign_speech(buckets, resources, rows)

    before_digest = _job_set_digest(rows)
    written_rows: list[JsonRow] = []
    post_hashes: dict[str, str] = {}
    for path in paths:
        index = int(path.stem.rsplit("-", 1)[1])
        payload = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in buckets[index]
        ).encode()
        path.write_bytes(payload)
        post_hashes[path.name] = _digest(payload)
        written_rows.extend(buckets[index])

    written_keys = [str(row["key"]) for row in written_rows]
    if len(written_keys) != len(seen) or set(written_keys) != seen:
        message = f"{stage}: repartition changed the immutable job-key set"
        raise RuntimeError(message)
    after_digest = _job_set_digest(written_rows)
    if after_digest != before_digest:
        message = f"{stage}: job-set digest changed during repartition"
        raise RuntimeError(message)

    report: JsonRow = {
        "schema": SCHEMA,
        "stage": stage,
        "jobs": len(rows),
        "speech_commands_jobs": len(speech),
        "job_set_sha256": after_digest,
        "target_lanes": list(TARGET_LANES),
        "speech_jobs_per_lane": {str(k): v for k, v in speech_counts.items()},
        "estimated_speech_seconds_per_lane": {
            str(k): v for k, v in speech_load.items()
        },
        "post_manifest_sha256": post_hashes,
        "scientific_contract": "job payloads and keys unchanged; scheduling resource only",
    }
    report_path = output_root / stage / "memory_repartition.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("q2_calibration", "q2_final"), required=True)
    args = parser.parse_args()
    sys.stdout.write(
        json.dumps(repartition(args.output_root, args.stage), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
