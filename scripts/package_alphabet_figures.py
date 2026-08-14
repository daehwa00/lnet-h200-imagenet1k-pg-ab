#!/usr/bin/env python3
"""Build a compact, checksummed reproduction bundle for active paper figures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

FIGURE_SCHEMA = "alphabet.figure_reproduction.v1"
PACKAGE_SCHEMA = "alphabet.figure_package.v1"
PACKAGE_ROOT = "alphabet-figures"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != FIGURE_SCHEMA:
        message = f"unexpected figure manifest schema: {path}"
        raise RuntimeError(message)
    figures = value.get("figures")
    if not isinstance(figures, list) or not figures:
        message = "figure manifest must contain a non-empty figures list"
        raise RuntimeError(message)
    return value


def _checked_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        message = "package paths must be non-empty strings"
        raise TypeError(message)
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        message = f"package path must be repository-relative: {value}"
        raise ValueError(message)
    return Path(*pure.parts)


def _root_files(repo: Path, values: object) -> list[str]:
    if not isinstance(values, list):
        message = "shared_roots must be a list"
        raise TypeError(message)
    files: list[str] = []
    for value in values:
        root = _checked_relative(value)
        source_root = repo / root
        if not source_root.is_dir() or source_root.is_symlink():
            message = f"required package directory is missing: {root.as_posix()}"
            raise FileNotFoundError(message)
        files.extend(
            path.relative_to(repo).as_posix()
            for path in sorted(source_root.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    return files


def _figure_files(figures: object) -> list[object]:
    if not isinstance(figures, list):
        message = "figures must be a list"
        raise TypeError(message)
    values: list[object] = []
    for figure in figures:
        if not isinstance(figure, dict):
            message = "each figure record must be an object"
            raise TypeError(message)
        for key in ("outputs", "generators", "package_inputs"):
            paths = figure.get(key)
            if not isinstance(paths, list) or not paths:
                message = f"{figure.get('id', '<unknown>')}: {key} is empty"
                raise RuntimeError(message)
            values.extend(paths)
    return values


def package_paths(repo: Path, manifest: dict[str, Any]) -> tuple[Path, ...]:
    shared = manifest.get("shared_files", [])
    if not isinstance(shared, list):
        message = "shared_files must be a list"
        raise TypeError(message)
    values: list[object] = ["paper/figure_reproduction.json", *shared]
    shared_roots = manifest.get("shared_roots", [])
    values.extend(_root_files(repo, shared_roots))
    values.extend(_figure_files(manifest["figures"]))

    paths = tuple(sorted({_checked_relative(value) for value in values}))
    for relative in paths:
        source = repo / relative
        if not source.is_file():
            message = f"required package file is missing: {relative.as_posix()}"
            raise FileNotFoundError(message)
        if source.is_symlink():
            message = f"package files must not be symlinks: {relative.as_posix()}"
            raise RuntimeError(message)
    return paths


def _inventory(repo: Path, paths: tuple[Path, ...]) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    tree = hashlib.sha256()
    for relative in paths:
        source = repo / relative
        size = source.stat().st_size
        digest = _sha256_file(source)
        name = relative.as_posix()
        rows.append({"path": name, "bytes": size, "sha256": digest})
        tree.update(f"{name}\0{size}\0{digest}\n".encode())
    return rows, tree.hexdigest()


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build_archive(repo: Path, manifest_path: Path, archive: Path) -> dict[str, Any]:
    repo = repo.resolve()
    manifest_path = manifest_path.resolve()
    expected_manifest = repo / "paper/figure_reproduction.json"
    if manifest_path != expected_manifest:
        message = f"manifest must be {expected_manifest}"
        raise ValueError(message)
    manifest = load_manifest(manifest_path)
    paths = package_paths(repo, manifest)
    inventory, tree_digest = _inventory(repo, paths)
    package_manifest: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "source_schema": FIGURE_SCHEMA,
        "files": len(inventory),
        "bytes": sum(int(row["bytes"]) for row in inventory),
        "tree_sha256": tree_digest,
        "inventory": inventory,
        "reproduction": {
            str(figure["id"]): str(figure["reproduce"])
            for figure in manifest["figures"]
        },
    }
    manifest_bytes = (
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()

    archive.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as bundle,
    ):
        bundle.addfile(
            _tar_info(f"{PACKAGE_ROOT}/PACKAGE_MANIFEST.json", len(manifest_bytes)),
            io.BytesIO(manifest_bytes),
        )
        for relative in paths:
            source = repo / relative
            data = source.read_bytes()
            bundle.addfile(
                _tar_info(f"{PACKAGE_ROOT}/{relative.as_posix()}", len(data)),
                io.BytesIO(data),
            )
    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper/figure_reproduction.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/alphabet-figures.tar.gz"),
    )
    arguments = parser.parse_args()
    repo = arguments.repo.resolve()
    manifest = arguments.manifest
    if not manifest.is_absolute():
        manifest = repo / manifest
    output = arguments.output
    if not output.is_absolute():
        output = repo / output
    result = build_archive(repo, manifest, output)
    print(  # noqa: T201
        f"PASS: {output.relative_to(repo)} "
        f"({result['files']} files, {result['bytes']} bytes)"
    )


if __name__ == "__main__":
    main()
