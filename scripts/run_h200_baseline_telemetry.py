#!/usr/bin/env python3
"""Tail one durable baseline telemetry spool and mirror it to W&B."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# ruff: noqa: BLE001, EM101, PLC0415, T201, TRY003, TRY301
import argparse
import json
import os
from pathlib import Path
from typing import Any


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--cursor", type=Path, required=True)
    parser.add_argument("--complete-marker", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def _synced_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("synced_ids") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError("telemetry cursor is invalid")
    return set(values)


def main() -> int:
    args = _arguments()
    records = _records(args.spool)
    synced = _synced_ids(args.cursor)
    pending = [record for record in records if str(record["id"]) not in synced]
    stopping = args.stop.is_file()
    if not pending and not stopping:
        return 0
    run_id = os.environ["H200_BASELINE_RUN_ID"]
    display_name = os.environ["H200_BASELINE_DISPLAY_NAME"]
    tags = json.loads(os.environ["H200_BASELINE_TAGS_JSON"])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise TypeError("H200_BASELINE_TAGS_JSON must contain strings")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))

    import wandb

    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        group=os.environ["WANDB_GROUP"],
        name=display_name,
        id=run_id,
        resume="allow",
        mode="online",
        anonymous="never",
        force=True,
        tags=tuple(tags),
        config=contract,
        settings=wandb.Settings(
            console="off",
            disable_code=True,
            disable_git=True,
            disable_job_creation=True,
            init_timeout=10,
            save_code=False,
            x_disable_meta=True,
            x_disable_stats=True,
            x_disable_viewer=True,
            x_extra_http_headers={
                "User-Agent": "Mozilla/5.0 lnet-h200-baseline-sidecar/1"
            },
            x_save_requirements=False,
        ),
    )
    if run is None:
        raise RuntimeError("wandb.init returned None")
    for record in pending:
        record_id = str(record["id"])
        run.log(record["metrics"], step=int(record["step"]))
        synced.add(record_id)
    if stopping and args.result.is_file():
        result = json.loads(args.result.read_text(encoding="utf-8"))
        run.summary["final_validation_accuracy"] = result["final_validation"]["accuracy"]
        run.summary["final_validation_top5_accuracy"] = result["final_validation"][
            "top5_accuracy"
        ]
    run.finish()
    _atomic_json(
        args.cursor,
        {"run_id": run_id, "synced_ids": sorted(synced), "synced_records": len(synced)},
    )
    if stopping:
        _atomic_json(
            args.complete_marker,
            {"run_id": run_id, "status": "completed", "synced_records": len(synced)},
        )
    print(
        "H200_BASELINE_WANDB_FLUSH_JSON="
        + json.dumps(
            {
                "pending_records": len(pending),
                "run_id": run_id,
                "stopping": stopping,
                "synced_records": len(synced),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            "H200_BASELINE_WANDB_SIDECAR_DEGRADED_JSON="
            + json.dumps({"error_type": type(error).__name__}, sort_keys=True),
            flush=True,
        )
        raise SystemExit(1) from None
