"""Shared utilities for campaign orchestration modules."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def write_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            message = f"refusing to overwrite frozen artifact: {path}"
            raise FileExistsError(message)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def canonical_json_sha256(obj: object) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def source_file_hashes(
    names: Sequence[str],
    *,
    project_root: Path,
    missing: Literal["raise", "placeholder"] = "raise",
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = project_root / name
        if path.is_file():
            result[name] = file_sha256(path)
        elif missing == "raise":
            message = f"required source file is missing: {name}"
            raise FileNotFoundError(message)
        else:
            result[name] = "not-yet-created"
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
