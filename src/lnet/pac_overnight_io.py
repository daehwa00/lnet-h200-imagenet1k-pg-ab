from __future__ import annotations

import csv
import fcntl
import json
import os
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pac_overnight_types import QueueEvent
    from .tapped_prl_followup_schema import JsonValue


def prepare_overnight_dirs(root: Path) -> None:
    for name in ("results", "reports", "figures"):
        (root / name).mkdir(parents=True, exist_ok=True)


def append_csv_row(path: Path, row: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing: list[dict[str, str]] = []
        existing_fields: list[str] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                existing = list(reader)
                existing_fields = list(reader.fieldnames or ())
        fieldnames = [*existing_fields, *(key for key in row if key not in existing_fields)]
        temporary_path: str | None = None
        try:
            with NamedTemporaryFile(
                "w",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                newline="",
                encoding="utf-8",
                delete=False,
            ) as target:
                temporary_path = target.name
                writer = csv.DictWriter(target, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing)
                writer.writerow(row)
                target.flush()
                os.fsync(target.fileno())
            Path(temporary_path).replace(path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                with suppress(FileNotFoundError):
                    Path(temporary_path).unlink()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def write_csv_rows(path: Path, rows: list[dict[str, JsonValue]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_queue_event(root: Path, event: QueueEvent) -> None:
    payload = json.dumps(asdict(event), sort_keys=True)
    with (root / "queue_state.jsonl").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(payload + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
