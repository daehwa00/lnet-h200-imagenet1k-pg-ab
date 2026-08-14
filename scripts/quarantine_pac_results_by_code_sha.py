#!/usr/bin/env python3
"""Move completed PAC rows from a rejected runner into an auditable quarantine.

The command is intentionally conservative: it matches only an exact
``code_sha256`` value, refuses to overwrite an existing quarantine, and moves
the matching completed row together with its attempt directory and any stale
failure record.  No evidence is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--label", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    stage_root = args.output_root / args.stage
    completed_root = stage_root / "completed"
    rows: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(completed_root.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("code_sha256") == args.code_sha256:
            rows.append((path, row))

    quarantine = args.output_root / "invalidated" / args.label
    if quarantine.exists():
        raise FileExistsError(f"quarantine already exists: {quarantine}")
    for name in ("completed", "attempts", "failed"):
        (quarantine / name).mkdir(parents=True, exist_ok=False)

    records: list[dict[str, object]] = []
    for path, row in rows:
        stem = path.stem
        record: dict[str, object] = {
            "job_key": row.get("job_key"),
            "completed_file": path.name,
            "completed_sha256": _sha256(path),
            "manifest_sha256": row.get("manifest_sha256"),
            "provenance_sha256": row.get("provenance_sha256"),
        }
        shutil.move(str(path), quarantine / "completed" / path.name)

        attempt_dir = stage_root / "attempts" / stem
        if attempt_dir.exists():
            record["attempt_files"] = sorted(item.name for item in attempt_dir.iterdir())
            shutil.move(str(attempt_dir), quarantine / "attempts" / stem)
        else:
            record["attempt_files"] = []

        failed_path = stage_root / "failed" / path.name
        if failed_path.exists():
            record["failed_sha256"] = _sha256(failed_path)
            shutil.move(str(failed_path), quarantine / "failed" / path.name)
        records.append(record)

    audit = {
        "schema": "pac_result_quarantine.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Results were produced by a stale runner whose compact-model job "
            "key omitted the sealed budget and learning-rate fields.  All rows "
            "from that exact runner hash are quarantined fail-closed, including "
            "rows whose keys happened to match the manifest."
        ),
        "stage": args.stage,
        "rejected_code_sha256": args.code_sha256,
        "row_count": len(records),
        "rows": records,
    }
    (quarantine / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"quarantine": str(quarantine), "row_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
