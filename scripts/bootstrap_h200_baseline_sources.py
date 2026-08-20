#!/usr/bin/env python3
"""Create immutable, persistent checkouts for H200 external baselines."""

from __future__ import annotations

# ruff: noqa: BLE001
import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "h200" / "baselines" / "sources.json"
DEFAULT_SOURCE_ROOT = Path("/app/scratch/input/lnet-h200-baseline-sources")


def _load_sources(path: Path = MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "lnet.h200.imagenet1k.external_sources.v1":
        msg = f"unsupported external-source manifest schema: {path}"
        raise RuntimeError(msg)
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        msg = f"external-source manifest has no sources: {path}"
        raise RuntimeError(msg)
    for name, source in sources.items():
        if not isinstance(name, str) or not isinstance(source, dict):
            msg = f"invalid source entry in {path}"
            raise TypeError(msg)
        repository = source.get("repository")
        commit = source.get("commit")
        if not isinstance(repository, str) or not repository.startswith("https://github.com/"):
            msg = f"source {name!r} does not use an official HTTPS GitHub URL"
            raise RuntimeError(msg)
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            msg = f"source {name!r} is not pinned to a full Git commit"
            raise RuntimeError(msg)
        if source.get("license") == "NOASSERTION" and source.get("redistribution_allowed"):
            msg = f"source {name!r} cannot permit redistribution without a license"
            raise RuntimeError(msg)
    return sources


def _run_git(*args: str, cwd: Path | None = None) -> str:
    command = ["git", *args]
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return completed.stdout.strip()


def _verify_checkout(path: Path, source: dict[str, Any]) -> None:
    if not (path / ".git").is_dir():
        msg = f"external source is not a Git checkout: {path}"
        raise RuntimeError(msg)
    expected_commit = str(source["commit"])
    actual_commit = _run_git("rev-parse", "--verify", "HEAD", cwd=path)
    if actual_commit != expected_commit:
        msg = f"external source commit mismatch at {path}: {actual_commit} != {expected_commit}"
        raise RuntimeError(msg)
    actual_remote = _run_git("config", "--get", "remote.origin.url", cwd=path)
    if actual_remote != source["repository"]:
        msg = f"external source remote mismatch at {path}: {actual_remote!r}"
        raise RuntimeError(msg)
    if _run_git("status", "--porcelain", "--untracked-files=all", cwd=path):
        msg = f"external source checkout is dirty: {path}"
        raise RuntimeError(msg)
    license_file = source.get("license_file")
    if license_file is not None and not (path / str(license_file)).is_file():
        msg = f"declared license file is missing from external source: {path / str(license_file)}"
        raise RuntimeError(msg)


def _checkout_source(source_root: Path, name: str, source: dict[str, Any]) -> Path:
    destination = source_root / name
    if destination.exists():
        _verify_checkout(destination, source)
        return destination

    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.incomplete-", dir=source_root))
    try:
        _run_git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            str(source["repository"]),
            str(temporary),
        )
        _run_git("checkout", "--detach", str(source["commit"]), cwd=temporary)
        _verify_checkout(temporary, source)
        try:
            temporary.rename(destination)
        except FileExistsError:
            _verify_checkout(destination, source)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def bootstrap_sources(
    source_root: Path,
    *,
    selected: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    sources = _load_sources()
    unknown = set() if selected is None else selected.difference(sources)
    if unknown:
        msg = f"unknown external sources: {sorted(unknown)}"
        raise ValueError(msg)
    selected_names = set(sources) if selected is None else selected
    noassertion = [
        name for name in selected_names if sources[name].get("license") == "NOASSERTION"
    ]
    if noassertion and os.environ.get("H200_ALLOW_NOASSERTION_SOURCES") != "research-only":
        msg = f"NOASSERTION sources require research-only opt-in: {sorted(noassertion)}"
        raise RuntimeError(msg)
    source_root.mkdir(parents=True, exist_ok=True)
    lock_path = source_root / ".bootstrap.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        names = sources if selected is None else [name for name in sources if name in selected]
        checkouts: dict[str, str] = {}
        failures: dict[str, str] = {}
        for name in names:
            try:
                checkouts[name] = str(_checkout_source(source_root, name, sources[name]))
            except Exception as error:  # Each later source must still be attempted.
                failures[name] = type(error).__name__
        return {"checkouts": checkouts, "failures": failures}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("H200_BASELINE_SOURCE_ROOT", DEFAULT_SOURCE_ROOT)),
    )
    parser.add_argument("--source", action="append", dest="sources")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = None if args.sources is None else set(args.sources)
    result = bootstrap_sources(args.source_root.expanduser().resolve(), selected=selected)
    sys.stdout.write(f"{json.dumps(result, sort_keys=True)}\n")
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
