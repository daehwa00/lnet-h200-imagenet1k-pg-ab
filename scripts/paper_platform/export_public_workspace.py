#!/usr/bin/env python3
"""Export only the reusable authoring platform into a clean public repository."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_PATHS = (
    Path("apps/paper_workspace/static"),
    Path("apps/paper_workspace/compiler"),
    Path("apps/paper_workspace/backup"),
    Path("apps/paper_workspace/collaboration"),
    Path("apps/paper_workspace/codex_bridge"),
    Path("apps/paper_workspace/password_gate"),
    Path("infra/paper-workspace"),
    Path("examples/paper-workspace-project"),
    Path("examples/project-library"),
    Path("docs/demo"),
    Path("docs/paper-platform"),
    Path("tests/paper_platform"),
    Path("scripts/paper_platform"),
)
ROOT_FILES = {
    Path("apps/paper_workspace/README.md"): Path("README.md"),
    Path("apps/paper_workspace/README.ko.md"): Path("README.ko.md"),
    Path("apps/paper_workspace/LICENSE"): Path("LICENSE"),
    Path("apps/paper_workspace/THIRD_PARTY_NOTICES.md"): Path("THIRD_PARTY_NOTICES.md"),
    Path("apps/paper_workspace/SECURITY.md"): Path("SECURITY.md"),
    Path("apps/paper_workspace/github/workflows/security.yml"): Path(".github/workflows/security.yml"),
    Path("apps/paper_workspace/github/workflows/platform.yml"): Path(".github/workflows/platform.yml"),
    Path("apps/paper_workspace/public.gitignore"): Path(".gitignore"),
    Path("apps/paper_workspace/public.dockerignore"): Path(".dockerignore"),
}
IGNORED_NAMES = {
    ".env",
    ".env.auth",
    ".env.password",
    ".auth",
    "auth.json",
    "allowed-emails",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "test-results",
    "playwright-report",
    ".DS_Store",
}
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,255}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,255}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,255}"),
    re.compile(rb"A[KS]IA[0-9A-Z]{16}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,255}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
)
SECRET_SCAN_CHUNK_BYTES = 64 * 1024
SECRET_SCAN_OVERLAP_BYTES = 512


def allowed_file(path: Path) -> bool:
    return not any(part in IGNORED_NAMES for part in path.parts) and path.suffix not in {".pyc", ".pyo"}


def normalize_public_mode(path: Path) -> None:
    """Make exported files readable by the unprivileged runtime user."""
    executable = bool(path.stat().st_mode & 0o111)
    path.chmod(0o755 if executable else 0o644)


def normalize_public_tree(destination: Path) -> None:
    destination.chmod(0o755)
    for path in destination.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            normalize_public_mode(path)


def copy_public_path(relative: Path, destination: Path) -> None:
    source = ROOT / relative
    for item in source.rglob("*"):
        item_relative = item.relative_to(source)
        if any(part in IGNORED_NAMES for part in item_relative.parts):
            continue
        if item.is_symlink():
            raise RuntimeError(f"Refusing to export symlink: {item.relative_to(ROOT)}")
        if not item.is_file() or not allowed_file(item_relative):
            continue
        target = destination / relative / item_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        normalize_public_mode(target)


def contains_secret(path: Path) -> bool:
    """Scan every exported byte without loading arbitrarily large files."""
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(SECRET_SCAN_CHUNK_BYTES):
            window = overlap + chunk
            if any(pattern.search(window) for pattern in SECRET_PATTERNS):
                return True
            overlap = window[-SECRET_SCAN_OVERLAP_BYTES:]
    return False


def verify_export(destination: Path) -> None:
    forbidden_parts = {"private", "datasets", "results", "checkpoints", ".codex"}
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Export contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(destination)
        if forbidden_parts.intersection(relative.parts) or path.name in IGNORED_NAMES:
            raise RuntimeError(f"Export contains a private path: {relative}")
        if contains_secret(path):
            raise RuntimeError(f"Possible secret found in {relative}")


def export(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative in PUBLIC_PATHS:
        copy_public_path(relative, destination)
    for source, target in ROOT_FILES.items():
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        source_path = ROOT / source
        if not source_path.is_file():
            # A clean public clone already stores these files at their exported
            # locations. Supporting that layout keeps the exporter reproducible
            # without widening the public allowlist.
            source_path = ROOT / target
        if not source_path.is_file():
            raise RuntimeError(f"Missing public source file: {source}")
        shutil.copy2(source_path, output)
        normalize_public_mode(output)
    normalize_public_tree(destination)
    verify_export(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new or empty output directory")
    args = parser.parse_args()
    export(args.destination.resolve())
    print(f"Public workspace exported to {args.destination.resolve()}")


if __name__ == "__main__":
    main()
