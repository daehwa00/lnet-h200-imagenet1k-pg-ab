#!/usr/bin/env python3
"""Read the preserved H200 baseline queue without restarting training."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "lnet.h200.imagenet1k.baseline-backlog-diagnostic.v1"
MODEL_KEYS = ("moganet_xt", "emov2_1m")
DEFAULT_ROOT = Path("/app/output/daehwa00") / (
    "lnet-h200-imagenet1k-baselines-v1-974305aba84d-c3b275ca11d0"
)
DEFAULT_CONTROL_ROOT = (
    Path("/app/output/daehwa00/run-control")
    / "h200-imagenet1k-moga-emo-100ep-s501-v1"
    / "c3b275ca11d0d99aa9436345b9f8ebb01ca4836d"
)
MAX_LOG_BYTES = 8 * 1024
SECRET_VALUE = re.compile(
    r"(?i)(authorization|api[_-]?key|password|secret|token)(\s*[:=]\s*)([^\s,}\]]+)"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    return parser.parse_args()


def _safe_root(path: Path, allowed: Path) -> Path:
    resolved = path.resolve(strict=True)
    allowed = allowed.resolve(strict=True)
    if allowed != resolved and allowed not in resolved.parents:
        raise ValueError(f"diagnostic root is outside {allowed}: {resolved}")
    return resolved


def _stat(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    value = path.stat()
    return {
        "exists": True,
        "modified_utc": datetime.fromtimestamp(value.st_mtime, timezone.utc).isoformat(),
        "path": str(path),
        "size_bytes": value.st_size,
    }


def _redact(text: str) -> str:
    return SECRET_VALUE.sub(r"\1\2<redacted>", text)


def _tail(path: Path) -> dict[str, Any]:
    metadata = _stat(path)
    if not metadata["exists"]:
        return metadata
    with path.open("rb") as stream:
        stream.seek(max(0, int(metadata["size_bytes"]) - MAX_LOG_BYTES))
        content = stream.read().decode("utf-8", errors="replace")
    metadata["tail"] = _redact(content)
    return metadata


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {"value_type": type(value).__name__}


def _checkpoint(path: Path) -> dict[str, Any]:
    metadata = _stat(path)
    if not metadata["exists"]:
        return metadata
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("checkpoint payload is not an object")
        history = payload.get("history")
        last_history = history[-1] if isinstance(history, list) and history else None
        metadata.update(
            {
                "completed_epochs": payload.get("completed_epochs"),
                "contract_sha256": payload.get("contract_sha256"),
                "global_step": payload.get("global_step"),
                "history_rows": len(history) if isinstance(history, list) else None,
                "last_history": last_history,
                "parameters": payload.get("parameters"),
                "task_sha256": payload.get("task_sha256"),
                "training_seconds": payload.get("training_seconds"),
            }
        )
    except Exception as error:  # The remaining diagnostics must still be reported.
        metadata["read_error_type"] = type(error).__name__
    return metadata


def _telemetry(path: Path) -> dict[str, Any]:
    metadata = _stat(path)
    if not metadata["exists"]:
        return metadata
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    metadata["records"] = len(records)
    metadata["last_record"] = records[-1] if records else None
    return metadata


def _task_directory(root: Path, model_key: str) -> Path | None:
    candidates = sorted(root.glob(f"run-data*/full/{model_key}/seed_501"))
    if len(candidates) > 1:
        raise RuntimeError(f"ambiguous preserved task directories for {model_key}")
    return candidates[0] if candidates else None


def diagnose(
    root: Path,
    control_root: Path,
    *,
    allowed_root: Path | None = None,
) -> dict[str, Any]:
    allowed_root = allowed_root or Path("/app/output/daehwa00")
    root = _safe_root(root, allowed_root)
    control_root = _safe_root(control_root, allowed_root)
    queue_paths = sorted(root.glob("run-data*/queue-status.json"))
    models: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        task_root = _task_directory(root, model_key)
        if task_root is None:
            models[model_key] = {"task_directory_exists": False}
            continue
        task_name = f"{model_key}__full__seed501__lr0p003"
        telemetry_root = task_root / "telemetry"
        models[model_key] = {
            "task_directory": str(task_root),
            "checkpoint": _checkpoint(task_root / "checkpoint.pt"),
            "result": _json(task_root / "result.json"),
            "worker_log": _tail(task_root / "worker.log"),
            "wandb_sidecar_log": _tail(telemetry_root / "wandb-sidecar.log"),
            "telemetry": _telemetry(telemetry_root / f"{task_name}.jsonl"),
            "telemetry_complete": _json(
                telemetry_root / f"{task_name}.mirror-complete.json"
            ),
            "telemetry_stop": _json(telemetry_root / f"{task_name}.stop.json"),
        }
    return {
        "schema": SCHEMA,
        "diagnosed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_source_commit": "c3b275ca11d0d99aa9436345b9f8ebb01ca4836d",
        "models": models,
        "output_root": str(root),
        "owner_stop": _json(control_root / "stopped.json"),
        "queue_status": [
            {"path": str(path), "payload": _json(path)} for path in queue_paths
        ],
    }


def main() -> None:
    args = _arguments()
    payload = diagnose(args.root, args.control_root)
    print(
        "H200_BASELINE_DIAGNOSTIC_JSON="
        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
