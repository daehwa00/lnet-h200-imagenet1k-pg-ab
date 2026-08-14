#!/usr/bin/env python3
"""Publish only manifest-listed project files into the web runtime volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath


SLUG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_RUNTIME_FILES = 240
MAX_RUNTIME_FILE_BYTES = 64 * 1024 * 1024
MAX_RUNTIME_PROJECT_BYTES = 512 * 1024 * 1024
AUTO_ASSET_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")
AUTO_TEXT_EXTENSIONS = {".bib", ".bst", ".cls", ".csv", ".dat", ".json", ".sty", ".tex", ".txt"}
AUTO_EXTENSIONS = set(AUTO_ASSET_EXTENSIONS) | AUTO_TEXT_EXTENSIONS
INPUT_PATTERN = re.compile(r"\\(?:input|include)\{([^{}]+)\}")
GRAPHICS_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}")


def safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value or len(value) > 240 or "\\" in value:
        raise ValueError("invalid project manifest path")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) > 12 or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("invalid project manifest path")
    if any(part.startswith(".") for part in path.parts):
        raise ValueError("hidden project paths are not publishable")
    return Path(*path.parts)


def read_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError(f"invalid JSON manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def checked_source(root: Path, relative: Path) -> Path:
    candidate = root / relative
    current = root
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"invalid project root: {root}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"project symlinks are not publishable: {relative.as_posix()}")
    if not candidate.is_file():
        raise ValueError(f"manifest file is missing: {relative.as_posix()}")
    size = candidate.stat().st_size
    if size > MAX_RUNTIME_FILE_BYTES:
        raise ValueError(f"manifest file is too large: {relative.as_posix()}")
    return candidate


def declared_source_exists(root: Path, relative: Path) -> bool:
    try:
        (root / relative).lstat()
    except FileNotFoundError:
        return False
    checked_source(root, relative)
    return True


def automatic_dependency_entries(
    project_root: Path,
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    configured_roots = manifest.get("auto_include_roots", [])
    if not isinstance(configured_roots, list) or len(configured_roots) > 8:
        raise ValueError("auto include roots must be a bounded list")
    roots = tuple(safe_relative(value) for value in configured_roots)
    if not roots:
        return []

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("project manifest files must be a list")
    published = {
        safe_relative(entry.get("path"))
        for entry in entries
        if isinstance(entry, dict)
    }
    pending = [
        relative
        for entry in entries
        if isinstance(entry, dict)
        and (relative := safe_relative(entry.get("source") or entry.get("path"))).suffix == ".tex"
        and declared_source_exists(project_root, relative)
    ]
    discovered: list[dict[str, object]] = []
    scanned: set[Path] = set()

    def opted_in(candidate: Path) -> bool:
        return any(candidate == root or root in candidate.parents for root in roots)

    while pending:
        source = pending.pop()
        if source in scanned:
            continue
        scanned.add(source)
        text = checked_source(project_root, source).read_text(encoding="utf-8")
        text = re.sub(r"(?<!\\)%.*", "", text)
        references: list[tuple[str, bool]] = [
            *((match.group(1).strip(), False) for match in INPUT_PATTERN.finditer(text)),
            *((match.group(1).strip(), True) for match in GRAPHICS_PATTERN.finditer(text)),
        ]
        for value, asset in references:
            try:
                relative = safe_relative(value)
            except ValueError:
                continue
            candidates = [relative]
            if asset and not relative.suffix:
                candidates = [Path(f"{relative.as_posix()}{extension}") for extension in AUTO_ASSET_EXTENSIONS]
            elif not asset and not relative.suffix:
                candidates = [Path(f"{relative.as_posix()}.tex")]
            for candidate in candidates:
                if not opted_in(candidate) or candidate.suffix.lower() not in AUTO_EXTENSIONS:
                    continue
                try:
                    (project_root / candidate).lstat()
                except FileNotFoundError:
                    continue
                checked_source(project_root, candidate)
                if candidate in published:
                    break
                entry: dict[str, object] = {
                    "path": candidate.as_posix(),
                    "managed": True,
                }
                if candidate.suffix.lower() in AUTO_ASSET_EXTENSIONS:
                    entry["type"] = "asset"
                discovered.append(entry)
                published.add(candidate)
                if candidate.suffix.lower() == ".tex":
                    pending.append(candidate)
                if len(entries) + len(discovered) > MAX_RUNTIME_FILES:
                    raise ValueError("project runtime discovered too many files")
                break
    return discovered


def project_spec(project_root: Path) -> tuple[list[Path], list[dict[str, object]], set[Path]]:
    manifest_path = checked_source(project_root, Path("project.json"))
    manifest = read_object(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) > MAX_RUNTIME_FILES:
        raise ValueError("project manifest files must be a bounded list")
    automatic_entries = automatic_dependency_entries(project_root, manifest)
    paths = {Path("project.json")}
    missing: set[Path] = set()
    for entry in [*entries, *automatic_entries]:
        if not isinstance(entry, dict):
            raise ValueError("invalid project manifest file entry")
        relative = safe_relative(entry.get("source") or entry.get("path"))
        if declared_source_exists(project_root, relative):
            paths.add(relative)
        else:
            missing.add(relative)
    for field in ("preview_pdf", "preview_synctex"):
        if manifest.get(field):
            paths.add(safe_relative(manifest[field]))
    if len(paths) > MAX_RUNTIME_FILES:
        raise ValueError("project manifest contains too many runtime files")
    total = sum(checked_source(project_root, path).stat().st_size for path in paths)
    if total > MAX_RUNTIME_PROJECT_BYTES:
        raise ValueError("project runtime exceeds its size limit")
    return sorted(paths), automatic_entries, missing


def project_files(project_root: Path) -> list[Path]:
    return project_spec(project_root)[0]


def project_revision(project_root: Path, paths: list[Path]) -> tuple[str, dict[str, str]]:
    revisions: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative in paths:
        source = checked_source(project_root, relative)
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            while chunk := handle.read(64 * 1024):
                digest.update(chunk)
        revision = digest.hexdigest()
        relative_name = relative.as_posix()
        revisions[relative_name] = revision
        combined.update(relative_name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(revision.encode("ascii"))
        combined.update(b"\0")
    return combined.hexdigest(), revisions


def copy_project(project_root: Path, destination: Path) -> None:
    paths, automatic_entries, missing = project_spec(project_root)
    for relative in paths:
        source = checked_source(project_root, relative)
        target = destination / relative
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o644)
    runtime_manifest_path = destination / "project.json"
    runtime_manifest = read_object(runtime_manifest_path)
    runtime_manifest["files"] = [
        entry
        for entry in runtime_manifest["files"]
        if safe_relative(entry.get("source") or entry.get("path")) not in missing
    ]
    runtime_manifest["files"].extend(automatic_entries)
    if missing:
        runtime_manifest["runtime_warnings"] = [
            f"manifest file is missing: {relative.as_posix()}"
            for relative in sorted(missing)
        ]
    else:
        runtime_manifest.pop("runtime_warnings", None)
    runtime_manifest_path.write_text(
        json.dumps(runtime_manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    # Hash exactly the staged bytes that the workspace will serve. Hashing the
    # source first and reopening it for copy lets an in-place server edit create
    # a descriptor/content mismatch between those two reads.
    runtime_revision, file_revisions = project_revision(destination, paths)
    # The served manifest contains these revision fields, so it cannot contain
    # a non-recursive hash of its own final bytes. Source and preview entries are
    # the cacheable artifacts consumers need to verify independently.
    file_revisions.pop("project.json", None)
    runtime_manifest = read_object(runtime_manifest_path)
    runtime_manifest["runtime_revision"] = runtime_revision
    runtime_manifest["runtime_file_revisions"] = file_revisions
    runtime_manifest_path.write_text(
        json.dumps(runtime_manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    runtime_manifest_path.chmod(0o644)


def catalog_projects(catalog_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    catalog = read_object(catalog_path)
    projects = catalog.get("projects")
    if not isinstance(projects, list) or len(projects) > MAX_RUNTIME_FILES:
        raise ValueError("project catalog must contain a bounded projects list")
    valid: list[dict[str, object]] = []
    for project in projects:
        if not isinstance(project, dict) or not isinstance(project.get("slug"), str):
            raise ValueError("invalid project catalog entry")
        if not SLUG_PATTERN.fullmatch(project["slug"]):
            raise ValueError("invalid project slug")
        valid.append(project)
    return catalog, valid


def replace_tree(staged: Path, destination: Path) -> None:
    previous = destination.with_name(f".{destination.name}.previous")
    shutil.rmtree(previous, ignore_errors=True)
    if destination.exists():
        os.replace(destination, previous)
    try:
        os.replace(staged, destination)
    except Exception:
        if previous.exists() and not destination.exists():
            os.replace(previous, destination)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def sync_runtime(default_project: Path, projects_root: Path, output_root: Path) -> str:
    catalog_path = checked_source(projects_root, Path("index.json"))
    _, projects = catalog_projects(catalog_path)
    output_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-", dir=output_root) as temporary:
        staging = Path(temporary)
        staged_default = staging / "project"
        staged_projects = staging / "projects"
        staged_default.mkdir()
        staged_projects.mkdir()
        copy_project(default_project, staged_default)
        shutil.copyfile(catalog_path, staged_projects / "index.json")
        (staged_projects / "index.json").chmod(0o644)
        for project in projects:
            slug = str(project["slug"])
            if project.get("source") == "default":
                continue
            copy_project(projects_root / slug, staged_projects / slug)
        digest = hashlib.sha256()
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(staging).as_posix().encode())
                with path.open("rb") as handle:
                    while chunk := handle.read(64 * 1024):
                        digest.update(chunk)
        fingerprint = digest.hexdigest()
        fingerprint_path = output_root / ".fingerprint"
        previous_fingerprint = (
            fingerprint_path.read_text(encoding="utf-8").strip()
            if fingerprint_path.is_file()
            else ""
        )
        if previous_fingerprint == fingerprint:
            return fingerprint
        replace_tree(staged_default, output_root / "project")
        replace_tree(staged_projects, output_root / "projects")
        temporary_fingerprint = output_root / ".fingerprint.tmp"
        temporary_fingerprint.write_text(f"{fingerprint}\n", encoding="utf-8")
        os.replace(temporary_fingerprint, fingerprint_path)
        return fingerprint


def write_ready(path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{time.time_ns()}\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-project", type=Path, required=True)
    parser.add_argument("--projects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch-seconds", type=float, default=10)
    args = parser.parse_args()
    args.output.mkdir(mode=0o755, parents=True, exist_ok=True)
    while True:
        try:
            sync_runtime(args.default_project, args.projects, args.output)
            write_ready(args.output / ".ready")
        except Exception as error:
            print(f"project runtime sync failed: {error}", flush=True)
        if args.watch_seconds <= 0:
            break
        time.sleep(max(args.watch_seconds, 1))


if __name__ == "__main__":
    main()
