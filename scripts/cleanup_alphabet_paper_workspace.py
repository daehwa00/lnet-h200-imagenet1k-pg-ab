#!/usr/bin/env python3
"""Archive paper artifacts that are not used by the active figure package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ARCHIVE_NAME = "20260727-unused"
SCHEMA = "alphabet.paper_cleanup.v1"
RETIRED_FILES = ("generate_finite_bank_margin_figure.py",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(path: Path, *, paper: Path) -> dict[str, object]:
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        size = item.stat().st_size
        file_digest = _sha256(item)
        digest.update(f"{relative}\0{size}\0{file_digest}\n".encode())
        total += size
    return {
        "path": path.relative_to(paper).as_posix(),
        "kind": "file" if path.is_file() else "directory",
        "files": len(files),
        "bytes": total,
        "tree_sha256": digest.hexdigest(),
    }


def _active_figure_paths(paper: Path) -> set[Path]:
    manifest = json.loads(
        (paper / "figure_reproduction.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "alphabet.figure_reproduction.v1":
        message = "unexpected figure reproduction schema"
        raise RuntimeError(message)
    active: set[Path] = set()
    for figure in manifest.get("figures", []):
        for value in figure.get("outputs", []):
            relative = Path(value)
            if relative.parts[:1] != ("paper",):
                message = f"figure output is not rooted under paper/: {value}"
                raise ValueError(message)
            active.add(Path(*relative.parts[1:]))
    return active


def _candidates(paper: Path, *, keep_build: str) -> list[Path]:
    active = _active_figure_paths(paper)
    figures = paper / "Figures"
    candidates = [
        path
        for path in sorted(figures.iterdir())
        if path.is_file() and path.relative_to(paper) not in active
    ]
    build = paper / "build"
    if build.is_dir():
        candidates.extend(
            path
            for path in sorted(build.iterdir())
            if path.name != keep_build
        )
    legacy_supplement = paper / "build-supplement"
    if legacy_supplement.exists():
        candidates.append(legacy_supplement)
    candidates.extend(
        path
        for name in RETIRED_FILES
        if (path := paper / name).is_file()
    )
    return candidates


def cleanup(repo: Path, *, apply: bool, keep_build: str) -> dict[str, Any]:
    repo = repo.resolve()
    paper = repo / "paper"
    archive = paper / "archive" / ARCHIVE_NAME
    candidates = _candidates(paper, keep_build=keep_build)
    records = [_inventory(path, paper=paper) for path in candidates]
    tree = hashlib.sha256()
    for record in records:
        tree.update(
            (
                f"{record['path']}\0{record['files']}\0{record['bytes']}\0"
                f"{record['tree_sha256']}\n"
            ).encode()
        )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "APPLIED" if apply else "DRY_RUN",
        "archive": archive.relative_to(paper).as_posix(),
        "kept_build": f"build/{keep_build}",
        "items": len(records),
        "files": sum(int(record["files"]) for record in records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "tree_sha256": tree.hexdigest(),
        "records": records,
    }
    if not apply:
        return report
    if archive.exists():
        message = f"cleanup archive already exists: {archive}"
        raise FileExistsError(message)
    for source in candidates:
        destination = archive / source.relative_to(paper)
        if destination.exists():
            message = f"cleanup destination already exists: {destination}"
            raise FileExistsError(message)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    manifest_path = archive / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--keep-build",
        default="figure-package-verification-20260727",
    )
    arguments = parser.parse_args()
    report = cleanup(
        arguments.repo,
        apply=arguments.apply,
        keep_build=arguments.keep_build,
    )
    print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
