# Copyright (c) 2026 QLab contributors.
# pyright: reportExplicitAny=false, reportImplicitStringConcatenation=false

"""Verify and, when explicitly requested, stage dense-transfer assets.

The default operation is read-only verification.  In particular, an existing
COCO/ADE tree or checkpoint is never replaced by this module.  Downloads are
limited to the fixed upstream URLs below and use a sibling ``.part`` file so
an interrupted transfer can be resumed.  Archive extraction is deliberately a
separate operation (``--extract``) and performs a complete safety preflight
before creating any file.

This file intentionally uses only the Python standard library.  Loading a
checkpoint with torch is left to the training engine, which can validate the
backbone at epoch one without making asset preparation depend on torch.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA = "dense-transfer.assets.v1"
FREE_SPACE_RESERVE_BYTES = 15 * 1024**3

# The files are the official 2017 COCO detection downloads.  No upstream
# checksums are asserted: the official download pages do not publish stable
# SHA-256 values for these archives.  Reports therefore distinguish an
# observed local digest from an upstream-verified digest (which is absent).
COCO_TRAIN_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip"
COCO_VAL_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
ADE_URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

COCO_LICENSE_URL = "https://cocodataset.org/#termsofuse"
COCO_TERMS_URL = "https://cocodataset.org/#termsofuse"
COCO_SOURCE_PAGE_URL = "https://cocodataset.org/dataset/detection-2017.htm"
ADE_LICENSE_URL = "https://ade20k.csail.mit.edu/terms/"
ADE_TERMS_URL = "https://ade20k.csail.mit.edu/terms/"
ADE_SOURCE_PAGE_URL = "https://ade20k.csail.mit.edu/"

COCO_EXPECTED_COUNTS: dict[str, int] = {"train": 118_287, "val": 5_000}
ADE_EXPECTED_COUNTS: dict[str, int] = {"training": 20_210, "validation": 2_000}
CHECKPOINT_NAMES: tuple[str, ...] = (
    "va_k128_seed501_ep100.pt",
    "convnextv2_atto_seed501_ep100.pt",
    "tinyvim_s_seed501_ep100.pt",
)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_DYNAMIC_MANIFEST_KEYS = frozenset(
    {
        "generated_at_utc",
        "observed_at_utc",
        "timestamp",
        "timestamps",
        "free_space",
        "free_space_bytes",
        "elapsed_seconds",
        "mtime",
        "mtime_ns",
        "ctime",
        "atime",
        "last_checked",
        "operations",
    }
)


class AssetError(RuntimeError):
    """Base error for preparation failures."""


class UnsafeArchiveError(AssetError):
    """Raised when a ZIP archive fails the extraction safety preflight."""


class DownloadError(AssetError):
    """Raised when an official archive cannot be downloaded safely."""


class ArchiveCollisionError(UnsafeArchiveError):
    """Raised when extraction would replace an existing user file."""


class ArchiveSpec:
    """Immutable metadata for one supported upstream archive."""

    __slots__ = ("kind", "license_url", "name", "terms_url", "url")

    def __init__(
        self,
        name: str,
        url: str,
        kind: str,
        license_url: str,
        terms_url: str,
    ) -> None:
        self.name = name
        self.url = url
        self.kind = kind
        self.license_url = license_url
        self.terms_url = terms_url

    def as_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "kind": self.kind,
            "license_url": self.license_url,
            "terms_url": self.terms_url,
            "source_page_url": (
                ADE_SOURCE_PAGE_URL if self.kind == "ade20k" else COCO_SOURCE_PAGE_URL
            ),
            "upstream_sha256": None,
            "sha256_verification": "observed_only_no_published_upstream_hash",
            "terms_acceptance_automated": False,
        }


DownloadSpec = ArchiveSpec


ARCHIVE_SPECS: tuple[ArchiveSpec, ...] = (
    ArchiveSpec(
        "train2017.zip",
        COCO_TRAIN_URL,
        "coco_train_images",
        COCO_LICENSE_URL,
        COCO_TERMS_URL,
    ),
    ArchiveSpec(
        "val2017.zip",
        COCO_VAL_URL,
        "coco_val_images",
        COCO_LICENSE_URL,
        COCO_TERMS_URL,
    ),
    ArchiveSpec(
        "annotations_trainval2017.zip",
        COCO_ANNOTATIONS_URL,
        "coco_annotations",
        COCO_LICENSE_URL,
        COCO_TERMS_URL,
    ),
    ArchiveSpec(
        "ADEChallengeData2016.zip",
        ADE_URL,
        "ade20k",
        ADE_LICENSE_URL,
        ADE_TERMS_URL,
    ),
)
# A short alias is useful to callers building a preparation UI and keeps the
# supported source list discoverable without parsing the CLI.
OFFICIAL_ARCHIVES = ARCHIVE_SPECS
OFFICIAL_SOURCE_URLS = frozenset(spec.url for spec in ARCHIVE_SPECS)


def canonical_json(value: object) -> bytes:
    """Encode JSON deterministically for reports and manifest hashes."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _without_dynamic_values(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_dynamic_values(item)
            for key, item in value.items()
            if str(key) not in _DYNAMIC_MANIFEST_KEYS
        }
    if isinstance(value, list):
        return [_without_dynamic_values(item) for item in value]
    if isinstance(value, tuple):
        return [_without_dynamic_values(item) for item in value]
    return value


def canonical_manifest(value: Mapping[str, object] | object) -> object:
    """Return the stable portion of a report suitable for hashing.

    Passing a full report uses its ``manifest`` member.  Passing a manifest
    directly also works.  Timestamps, free-space observations, and operation
    logs are excluded recursively so a second verify of unchanged assets has
    the same digest.
    """

    if isinstance(value, Mapping) and isinstance(value.get("manifest"), Mapping):
        value = value["manifest"]
    return _without_dynamic_values(value)


def manifest_sha256(value: Mapping[str, object] | object) -> str:
    return hashlib.sha256(canonical_json(canonical_manifest(value))).hexdigest()


# Friendly compatibility alias used by small integration scripts.
manifest_hash = manifest_sha256


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    """Write JSON atomically without leaving a partially written report."""

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_json(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        temporary = None
        try:
            directory_fd = os.open(parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


atomic_json = _atomic_json
write_report = _atomic_json


def _safe_resolve(path: Path) -> Path:
    return path.expanduser().absolute().resolve(strict=False)


def _scan_regular_files(directory: Path, *, recursive: bool = True) -> set[str]:
    """Return relative regular-file names with one scandir traversal.

    ``DirEntry`` is used for the type check and names are collected into a
    set.  Verification consequently does not perform a ``Path.exists`` call
    for every annotation reference.
    """

    if not directory.is_dir() or directory.is_symlink():
        return set()
    found: set[str] = set()
    pending: list[tuple[Path, str]] = [(directory, "")]
    while pending:
        current, prefix = pending.pop()
        try:
            entries = os.scandir(current)
        except OSError:
            continue
        with entries:
            for entry in entries:
                relative = f"{prefix}{entry.name}"
                try:
                    if entry.is_file(follow_symlinks=False):
                        found.add(relative)
                    elif recursive and entry.is_dir(follow_symlinks=False):
                        pending.append((Path(entry.path), f"{relative}/"))
                except OSError:
                    # An unreadable/racing entry is simply absent from the
                    # observed set and therefore cannot produce readiness.
                    continue
    return found


def _scan_directory_entries(directory: Path) -> dict[str, str]:
    """Map direct regular-file names to a type marker without per-file stat."""

    result: dict[str, str] = {}
    if not directory.is_dir() or directory.is_symlink():
        return result
    try:
        entries = os.scandir(directory)
    except OSError:
        return result
    with entries:
        for entry in entries:
            try:
                if entry.is_file(follow_symlinks=False):
                    result[entry.name] = "file"
            except OSError:
                continue
    return result


def _normalise_reference(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    raw = value.replace("\\", "/")
    if raw.startswith("/") or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        return None
    parts = [part for part in PurePosixPath(raw).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def _load_json(path: Path) -> tuple[object | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON: {exc}"


def _normalise_expected_counts(
    expected: Mapping[str, int] | None,
    defaults: Mapping[str, int],
    aliases: Mapping[str, str],
) -> dict[str, int]:
    result = dict(defaults)
    if expected is None:
        return result
    for key, value in expected.items():
        canonical = aliases.get(key, key)
        if canonical not in result:
            raise ValueError(f"unknown expected split {key!r}")
        if type(value) is not int or value < 0:
            raise ValueError(f"expected count for {key!r} must be a non-negative integer")
        result[canonical] = value
    return result


def _segmentation_info(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {
            "annotation_count": 0,
            "segmentation_field_count": 0,
            "segmentation_nonempty_count": 0,
            "segmentation_field_coverage": 0.0,
            "segmentation_coverage": 0.0,
            "missing_segmentation": True,
        }
    rows = payload.get("annotations")
    if not isinstance(rows, list):
        rows = []
    fields = 0
    nonempty = 0
    for row in rows:
        if not isinstance(row, Mapping) or "segmentation" not in row:
            continue
        fields += 1
        value = row.get("segmentation")
        if value not in (None, "", [], {}):
            nonempty += 1
    total = len(rows)
    field_coverage = fields / total if total else 0.0
    coverage = nonempty / total if total else 0.0
    return {
        "annotation_count": total,
        "segmentation_field_count": fields,
        "segmentation_nonempty_count": nonempty,
        "segmentation_field_coverage": field_coverage,
        "segmentation_coverage": coverage,
        "missing_segmentation": fields != total,
    }


def _verify_coco_split(
    root: Path,
    split: str,
    expected_count: int,
) -> dict[str, object]:
    split_dir = root / f"{split}2017"
    annotation_path = root / "annotations" / f"instances_{split}2017.json"
    files = _scan_regular_files(split_dir)
    image_files = {name for name in files if Path(name).suffix.lower() in IMAGE_SUFFIXES}
    payload, error = _load_json(annotation_path)

    missing_reference: list[str] = []
    invalid_image_rows = 0
    categories_count: int | None = None
    annotation_rows_ok = False
    segmentation = _segmentation_info(payload)
    if isinstance(payload, Mapping):
        images = payload.get("images")
        annotation_rows_ok = isinstance(payload.get("annotations"), list)
        categories = payload.get("categories")
        categories_count = len(categories) if isinstance(categories, list) else None
        if isinstance(images, list):
            references: set[str] = set()
            for row in images:
                if not isinstance(row, Mapping):
                    invalid_image_rows += 1
                    continue
                name = _normalise_reference(row.get("file_name"))
                if name is None:
                    invalid_image_rows += 1
                    continue
                references.add(name)
            missing_reference = sorted(references.difference(files))
        else:
            invalid_image_rows += 1
    checks = {
        "image_directory": str(split_dir),
        "annotation_file": str(annotation_path),
        "expected_image_count": expected_count,
        "expected_count": expected_count,
        "observed_image_count": len(image_files),
        "image_count": len(image_files),
        "image_count_ok": len(image_files) == expected_count,
        "annotation_json_ok": error is None,
        "annotation_rows_ok": annotation_rows_ok,
        "categories_count": categories_count,
        "categories_ok": categories_count == 80,
        "invalid_image_rows": invalid_image_rows,
        "missing_referenced_files": missing_reference,
        "all_referenced_files_present": not missing_reference and invalid_image_rows == 0,
        "masks_field_exists": not bool(segmentation["missing_segmentation"]),
        "mask_field_exists": not bool(segmentation["missing_segmentation"]),
        **segmentation,
    }
    checks["ready"] = bool(
        checks["image_count_ok"]
        and checks["annotation_json_ok"]
        and checks["annotation_rows_ok"]
        and checks["categories_ok"]
        and checks["all_referenced_files_present"]
        and not checks["missing_segmentation"]
        and checks["segmentation_coverage"] == 1.0
    )
    checks["state"] = "ready" if checks["ready"] else "incomplete"
    if error is not None:
        checks["annotation_error"] = error
    return checks


def verify_coco(
    root: Path | str,
    expected_counts: Mapping[str, int] | None = None,
    *,
    expected_image_counts: Mapping[str, int] | None = None,
    expected_train_count: int | None = None,
    expected_val_count: int | None = None,
) -> dict[str, object]:
    """Verify a COCO 2017 tree.

    ``expected_counts`` is intentionally injectable for tiny unit fixtures;
    production defaults remain the canonical 118,287/5,000 counts.  Keys
    ``train``, ``val``, ``train2017`` and ``val2017`` are accepted.
    """

    if expected_counts is not None and expected_image_counts is not None:
        raise ValueError("pass only one of expected_counts and expected_image_counts")
    named_counts = {
        key: value
        for key, value in (("train", expected_train_count), ("val", expected_val_count))
        if value is not None
    }
    if named_counts and (expected_counts is not None or expected_image_counts is not None):
        raise ValueError("pass either a count mapping or named expected split counts")
    counts = _normalise_expected_counts(
        (
            expected_counts
            if expected_counts is not None
            else expected_image_counts
            if expected_image_counts is not None
            else named_counts or None
        ),
        COCO_EXPECTED_COUNTS,
        {"train2017": "train", "val2017": "val", "training": "train", "validation": "val"},
    )
    resolved = _safe_resolve(Path(root))
    splits = {split: _verify_coco_split(resolved, split, count) for split, count in counts.items()}
    ready = all(bool(item["ready"]) for item in splits.values())
    return {
        "name": "coco2017",
        "root": str(resolved),
        "expected_counts": counts,
        "splits": splits,
        "ready": ready,
        "state": "ready" if ready else "incomplete",
    }


def _find_ade_root(root: Path) -> Path:
    resolved = _safe_resolve(root)
    direct = resolved / "images" / "training"
    nested = resolved / "ADEChallengeData2016" / "images" / "training"
    if direct.is_dir() or not nested.is_dir():
        return resolved
    return resolved / "ADEChallengeData2016"


def _verify_ade_split(root: Path, split: str, expected_count: int) -> dict[str, object]:
    image_dir = root / "images" / split
    label_dir = root / "annotations" / split
    image_files = {
        name
        for name in _scan_regular_files(image_dir)
        if Path(name).suffix.lower() in {".jpg", ".jpeg"}
    }
    label_files = {
        name
        for name in _scan_regular_files(label_dir)
        if Path(name).suffix.lower() == ".png"
    }
    image_stems = {str(Path(name).with_suffix("")) for name in image_files}
    label_stems = {str(Path(name).with_suffix("")) for name in label_files}
    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    checks: dict[str, object] = {
        "image_directory": str(image_dir),
        "label_directory": str(label_dir),
        "expected_count": expected_count,
        "observed_image_count": len(image_files),
        "observed_label_count": len(label_files),
        "image_count": len(image_files),
        "label_count": len(label_files),
        "image_count_ok": len(image_files) == expected_count,
        "label_count_ok": len(label_files) == expected_count,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "matched_labels": not missing_labels and not orphan_labels,
        # ADE's canonical zero label is preserved.  The dataloader, not this
        # asset checker, owns any ignore-index mapping.
        "ignore_index": 0,
        "ignore_mapping": "handled_by_dataloader",
    }
    checks["ready"] = bool(
        checks["image_count_ok"]
        and checks["label_count_ok"]
        and checks["matched_labels"]
    )
    checks["state"] = "ready" if checks["ready"] else "incomplete"
    return checks


def verify_ade(
    root: Path | str,
    expected_counts: Mapping[str, int] | None = None,
    *,
    expected_image_counts: Mapping[str, int] | None = None,
    expected_training_count: int | None = None,
    expected_validation_count: int | None = None,
) -> dict[str, object]:
    """Verify ADEChallengeData2016 image/PNG label pairs."""

    if expected_counts is not None and expected_image_counts is not None:
        raise ValueError("pass only one of expected_counts and expected_image_counts")
    named_counts = {
        key: value
        for key, value in (
            ("training", expected_training_count),
            ("validation", expected_validation_count),
        )
        if value is not None
    }
    if named_counts and (expected_counts is not None or expected_image_counts is not None):
        raise ValueError("pass either a count mapping or named expected split counts")
    counts = _normalise_expected_counts(
        (
            expected_counts
            if expected_counts is not None
            else expected_image_counts
            if expected_image_counts is not None
            else named_counts or None
        ),
        ADE_EXPECTED_COUNTS,
        {"train": "training", "val": "validation", "train2016": "training", "val2016": "validation"},
    )
    resolved = _find_ade_root(Path(root))
    splits = {
        split: _verify_ade_split(resolved, split, count) for split, count in counts.items()
    }
    ready = all(bool(item["ready"]) for item in splits.values())
    return {
        "name": "ade20k",
        "root": str(resolved),
        "expected_counts": counts,
        "splits": splits,
        "ready": ready,
        "state": "ready" if ready else "incomplete",
        "ignore_index": 0,
        "ignore_mapping": "handled_by_dataloader",
    }


# Descriptive aliases keep the helpers discoverable for callers that use the
# archive/dataset names rather than the short task names.
verify_coco2017 = verify_coco
verify_ade20k = verify_ade
validate_coco = verify_coco
validate_ade = verify_ade


def verify_checkpoints(root: Path | str) -> dict[str, object]:
    """Report required checkpoint presence and observed SHA-256 values."""

    resolved = _safe_resolve(Path(root))
    entries = _scan_directory_entries(resolved)
    files: list[dict[str, object]] = []
    for name in CHECKPOINT_NAMES:
        present = entries.get(name) == "file"
        digest: str | None = None
        if present:
            try:
                digest = sha256_file(resolved / name)
            except OSError:
                present = False
        files.append(
            {
                "name": name,
                "present": present,
                "sha256": digest,
                "sha256_observed": digest,
                "upstream_sha256": None,
                "sha256_verified_upstream": False,
                "sha256_verification": "observed_only",
            }
        )
    ready = all(bool(item["present"]) for item in files)
    return {
        "root": str(resolved),
        "files": files,
        "ready": ready,
        "state": "ready" if ready else "incomplete",
    }


validate_checkpoints = verify_checkpoints


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    if candidate.exists() and candidate.is_file():
        candidate = candidate.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.is_dir() else Path.cwd()


def _free_bytes(path: Path) -> int:
    try:
        return int(shutil.disk_usage(_nearest_existing_directory(path)).free)
    except OSError:
        return 0


def _free_space_observation(
    path: Path,
    *,
    reserve_bytes: int = FREE_SPACE_RESERVE_BYTES,
) -> dict[str, object]:
    location = _nearest_existing_directory(path)
    return {
        "path": str(location),
        "free_bytes": _free_bytes(location),
        "reserve_bytes": reserve_bytes,
    }


def _archive_observation(spec: ArchiveSpec, source_root: Path) -> dict[str, object]:
    archive = source_root / spec.name
    part = archive.with_name(archive.name + ".part")
    present = archive.is_file() and not archive.is_symlink()
    observed: str | None = None
    size: int | None = None
    if present:
        try:
            observed = sha256_file(archive)
            size = archive.stat().st_size
        except OSError:
            present = False
    part_size: int | None = None
    if part.is_file() and not part.is_symlink():
        try:
            part_size = part.stat().st_size
        except OSError:
            part_size = None
    return {
        **spec.as_manifest(),
        "source_root": str(_safe_resolve(source_root)),
        "present": present,
        "size_bytes": size,
        "sha256": observed,
        "sha256_observed": observed,
        "sha256_verified_upstream": False,
        "partial_present": part_size is not None,
        "partial_size_bytes": part_size,
    }


def verify_archives(source_root: Path | str) -> dict[str, object]:
    resolved = _safe_resolve(Path(source_root))
    archives = [_archive_observation(spec, resolved) for spec in ARCHIVE_SPECS]
    return {
        "root": str(resolved),
        "archives": archives,
        "ready": all(bool(item["present"]) for item in archives),
        "state": "ready" if all(bool(item["present"]) for item in archives) else "incomplete",
    }


def _response_header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except AttributeError:
        return None
    return None if value is None else str(value)


def _response_status(response: Any) -> int:
    value = getattr(response, "status", None)
    if value is None:
        value = getattr(response, "code", 200)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 200


def _validate_official_url(url: str) -> None:
    if url not in OFFICIAL_SOURCE_URLS:
        raise DownloadError(f"refusing non-official source URL: {url}")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DownloadError(f"unsupported source URL: {url}")


def _response_url_is_official(response: Any) -> bool:
    final_url = getattr(response, "url", None)
    if not final_url:
        return True
    parsed = urllib.parse.urlparse(str(final_url))
    allowed_hosts = {urllib.parse.urlparse(url).hostname for url in OFFICIAL_SOURCE_URLS}
    return parsed.hostname in allowed_hosts


def _guard_free_space(path: Path, required_bytes: int, *, reserve_bytes: int) -> None:
    available = _free_bytes(path)
    if available < reserve_bytes + max(0, required_bytes):
        raise DownloadError(
            f"insufficient free space at {_nearest_existing_directory(path)}: "
            f"need {reserve_bytes + max(0, required_bytes)} bytes, have {available}"
        )


def download_archive(
    spec: ArchiveSpec,
    destination: Path | str,
    *,
    reserve_bytes: int = FREE_SPACE_RESERVE_BYTES,
    chunk_size: int = 1024 * 1024,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    """Download one official archive into ``destination`` using ``.part``.

    An existing final file is never touched.  If a partial file exists, the
    server must honour a byte-range request; otherwise the partial file is
    retained and an error is raised rather than silently duplicating bytes.
    """

    _validate_official_url(spec.url)
    final = Path(destination).expanduser().absolute()
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.is_symlink() or final.is_dir():
        raise DownloadError(f"refusing to replace existing destination: {final}")
    if final.exists():
        digest = sha256_file(final)
        return {
            "name": spec.name,
            "status": "existing",
            "path": str(final),
            "bytes": final.stat().st_size,
            "sha256_observed": digest,
            "upstream_sha256": None,
            "sha256_verified_upstream": False,
        }

    part = final.with_name(final.name + ".part")
    if part.is_symlink() or part.is_dir():
        raise DownloadError(f"refusing unsafe partial path: {part}")
    resumed_bytes = part.stat().st_size if part.exists() else 0
    _guard_free_space(final.parent, 0, reserve_bytes=reserve_bytes)

    headers: dict[str, str] = {}
    if resumed_bytes:
        headers["Range"] = f"bytes={resumed_bytes}-"
    request = urllib.request.Request(spec.url, headers=headers, method="GET")  # noqa: S310
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)  # noqa: S310
    except urllib.error.URLError as exc:
        raise DownloadError(f"download failed for {spec.name}: {exc}") from exc
    close = getattr(response, "close", None)
    try:
        status = _response_status(response)
        if not _response_url_is_official(response):
            raise DownloadError(f"refusing redirect outside official source host for {spec.name}")
        if resumed_bytes and status == 416:
            # A server can report that the existing partial is already the
            # complete object.  Without an upstream digest this is still only
            # an observed byte stream, so finalize it atomically.
            content_range = _response_header(response, "Content-Range") or ""
            if content_range.endswith(f"/{resumed_bytes}"):
                os.link(part, final)
                part.unlink()
                return {
                    "name": spec.name,
                    "status": "resumed_complete",
                    "path": str(final),
                    "bytes": resumed_bytes,
                    "sha256_observed": sha256_file(final),
                    "upstream_sha256": None,
                    "sha256_verified_upstream": False,
                }
            raise DownloadError(f"range is not satisfiable for existing partial {part}")
        if status < 200 or status >= 300:
            raise DownloadError(f"unexpected HTTP status {status} for {spec.name}")
        if resumed_bytes and status != 206:
            raise DownloadError(
                f"server did not honour resume range for {spec.name} (status {status}); partial retained"
            )
        content_length_text = _response_header(response, "Content-Length")
        try:
            content_length = int(content_length_text) if content_length_text is not None else None
        except ValueError:
            content_length = None
        total_expected = (
            resumed_bytes + content_length if content_length is not None else None
        )
        if total_expected is not None:
            _guard_free_space(
                final.parent,
                max(0, total_expected - resumed_bytes),
                reserve_bytes=reserve_bytes,
            )
        mode = "ab" if resumed_bytes else "xb"
        written = resumed_bytes
        with part.open(mode) as stream:
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                remaining = max(0, (total_expected - written) if total_expected is not None else 0)
                _guard_free_space(final.parent, remaining, reserve_bytes=reserve_bytes)
                stream.write(block)
                written += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if total_expected is not None and written != total_expected:
            raise DownloadError(
                f"short download for {spec.name}: expected {total_expected}, received {written}; partial retained"
            )
        if final.exists() or final.is_symlink():
            raise DownloadError(f"destination appeared during download; partial retained: {final}")
        # A hard link gives a no-replace commit on local filesystems.  The
        # partial is removed only after the final name is durably linked.
        os.link(part, final)
        part.unlink()
        return {
            "name": spec.name,
            "status": "resumed" if resumed_bytes else "downloaded",
            "path": str(final),
            "bytes": written,
            "sha256_observed": sha256_file(final),
            "upstream_sha256": None,
            "sha256_verified_upstream": False,
        }
    finally:
        if callable(close):
            close()


download_resumable = download_archive


def _normalise_member_name(name: str) -> str:
    if not name or "\x00" in name:
        raise UnsafeArchiveError("ZIP member has an empty or NUL-containing name")
    raw = name.replace("\\", "/")
    if raw.startswith("/") or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise UnsafeArchiveError(f"absolute ZIP member path: {name!r}")
    parts = [part for part in PurePosixPath(raw).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise UnsafeArchiveError(f"path traversal ZIP member: {name!r}")
    return "/".join(parts)


def _check_target_ancestors(path: Path) -> None:
    current = path
    ancestors: list[Path] = []
    while True:
        ancestors.append(current)
        if current.parent == current:
            break
        current = current.parent
    for candidate in reversed(ancestors):
        if candidate.is_symlink():
            raise UnsafeArchiveError(f"extraction path contains symlink: {candidate}")
        if candidate.exists() and not candidate.is_dir():
            raise UnsafeArchiveError(f"extraction path is not a directory: {candidate}")


def _zip_preflight(
    archive: Path,
    target: Path,
    *,
    max_member_bytes: int,
    max_total_bytes: int,
    max_compression_ratio: int,
    reserve_bytes: int,
) -> tuple[list[tuple[zipfile.ZipInfo, str]], int]:
    target = target.expanduser().absolute()
    if target == Path(target.anchor or target):
        raise UnsafeArchiveError(f"refusing archive extraction at filesystem root: {target}")
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise UnsafeArchiveError(f"invalid extraction target: {target}")
    _check_target_ancestors(target.parent)
    members: list[tuple[zipfile.ZipInfo, str]] = []
    names: set[str] = set()
    file_names: set[str] = set()
    total = 0
    try:
        zf = zipfile.ZipFile(archive, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsafeArchiveError(f"cannot open ZIP archive {archive}: {exc}") from exc
    with zf:
        for info in zf.infolist():
            normal = _normalise_member_name(info.filename)
            folded = normal.casefold()
            if folded in {item.casefold() for item in names}:
                raise UnsafeArchiveError(f"duplicate ZIP member: {normal}")
            names.add(normal)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise UnsafeArchiveError(f"symbolic-link ZIP member: {normal}")
            is_directory = info.is_dir() or normal.endswith("/")
            # ZIP creators commonly store only permission bits (e.g. 0o600)
            # rather than an explicit ``S_IFREG`` type.  A zero type field is
            # therefore a normal file; explicit non-regular types remain
            # rejected.
            mode_type = stat.S_IFMT(mode)
            if mode_type not in (0, stat.S_IFREG) and not is_directory:
                raise UnsafeArchiveError(f"unsupported ZIP member type: {normal}")
            if not is_directory:
                if info.file_size > max_member_bytes:
                    raise UnsafeArchiveError(f"ZIP member exceeds size limit: {normal}")
                if info.file_size and (
                    info.compress_size <= 0 or info.file_size / info.compress_size > max_compression_ratio
                ):
                    raise UnsafeArchiveError(f"ZIP member compression ratio is unsafe: {normal}")
                total += info.file_size
                if total > max_total_bytes:
                    raise UnsafeArchiveError("ZIP archive exceeds total uncompressed-size limit")
                file_names.add(normal.rstrip("/"))
            members.append((info, normal.rstrip("/")))
        for file_name in file_names:
            prefix = file_name + "/"
            if any(other != file_name and other.startswith(prefix) for other in names):
                raise UnsafeArchiveError(f"ZIP file/directory path collision: {file_name}")
    _guard_free_space(target.parent, total, reserve_bytes=reserve_bytes)
    for _info, normal in members:
        destination = target / normal
        _check_target_ancestors(destination.parent)
        if (destination.exists() or destination.is_symlink()) and (
            destination.is_symlink() or not destination.is_dir()
        ):
            raise ArchiveCollisionError(f"refusing to replace existing path: {destination}")
            # Existing directories are safe to reuse; material files are not.
    return members, total


def safe_extract_zip(
    archive: Path | str,
    target: Path | str,
    *,
    reserve_bytes: int = FREE_SPACE_RESERVE_BYTES,
    max_member_bytes: int = 8 * 1024**3,
    max_total_bytes: int = 128 * 1024**3,
    max_compression_ratio: int = 1_000,
) -> dict[str, object]:
    """Safely extract a ZIP without traversal, symlink, bomb, or overwrite.

    All members are preflighted before the target directory is created.  An
    existing directory may be reused, but an existing material file causes a
    collision error; no user file is deleted or replaced.
    """

    archive_path = Path(archive).expanduser().absolute()
    target_path = Path(target).expanduser().absolute()
    members, total = _zip_preflight(
        archive_path,
        target_path,
        max_member_bytes=max_member_bytes,
        max_total_bytes=max_total_bytes,
        max_compression_ratio=max_compression_ratio,
        reserve_bytes=reserve_bytes,
    )
    target_path.mkdir(parents=True, exist_ok=True)
    written = 0
    files = 0
    directories = 0
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info, normal in members:
                destination = target_path / normal
                if info.is_dir() or info.filename.endswith("/"):
                    if destination.exists():
                        directories += 1
                        continue
                    destination.mkdir(parents=True, exist_ok=False)
                    directories += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                file_written = 0
                with zf.open(info, "r") as source, destination.open("xb") as stream:
                    while block := source.read(1024 * 1024):
                        _guard_free_space(
                            target_path.parent,
                            max(0, total - written),
                            reserve_bytes=reserve_bytes,
                        )
                        stream.write(block)
                        file_written += len(block)
                        written += len(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                if file_written != info.file_size:
                    raise UnsafeArchiveError(f"unexpected extraction size for {normal}")
                files += 1
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, UnsafeArchiveError):
            raise
        raise UnsafeArchiveError(f"failed extracting {archive_path}: {exc}") from exc
    return {
        "archive": str(archive_path),
        "target": str(target_path),
        "files_extracted": files,
        "directories_seen": directories,
        "bytes_extracted": written,
        "state": "complete" if written == total else "incomplete",
    }


# Alternate spelling retained for callers that prefer an explicit verb.
extract_zip_safely = safe_extract_zip
extract_archive = safe_extract_zip
extract_archive_safely = safe_extract_zip


def _extract_target_for_archive(spec: ArchiveSpec, coco_root: Path, ade_root: Path) -> Path:
    if spec.kind == "ade20k":
        # The official archive normally contains a top-level
        # ADEChallengeData2016/ directory.  Extracting next to the requested
        # root avoids accidentally creating root/ADEChallengeData2016 twice.
        return ade_root.parent
    return coco_root


def _ade_extract_target(archive: Path, ade_root: Path) -> Path:
    """Choose the parent/root target without trusting archive member paths."""

    try:
        with zipfile.ZipFile(archive, "r") as stream:
            top_levels = {
                raw_name.replace("\\", "/").split("/", 1)[0]
                for raw_name in stream.namelist()
                if raw_name
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsafeArchiveError(f"cannot inspect ADE archive {archive}: {exc}") from exc
    if ade_root.name in top_levels:
        return ade_root.parent
    return ade_root


def extract_downloaded_archives(
    source_root: Path | str,
    coco_root: Path | str,
    ade_root: Path | str,
    *,
    reserve_bytes: int = FREE_SPACE_RESERVE_BYTES,
) -> list[dict[str, object]]:
    """Extract present official archives into explicit dataset targets."""

    source = _safe_resolve(Path(source_root))
    coco = _safe_resolve(Path(coco_root))
    ade = _safe_resolve(Path(ade_root))
    results: list[dict[str, object]] = []
    for spec in ARCHIVE_SPECS:
        archive = source / spec.name
        if not archive.is_file() or archive.is_symlink():
            continue
        target = _extract_target_for_archive(spec, coco, ade)
        if spec.kind == "ade20k":
            target = _ade_extract_target(archive, ade)
        results.append(safe_extract_zip(archive, target, reserve_bytes=reserve_bytes))
    return results


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.UTC).isoformat().replace("+00:00", "Z")


def build_report(
    *,
    coco_root: Path | str,
    ade_root: Path | str,
    checkpoint_root: Path | str,
    source_root: Path | str,
    operations: Sequence[Mapping[str, object]] = (),
    expected_coco_counts: Mapping[str, int] | None = None,
    expected_ade_counts: Mapping[str, int] | None = None,
    reserve_bytes: int = FREE_SPACE_RESERVE_BYTES,
) -> dict[str, object]:
    coco = verify_coco(coco_root, expected_counts=expected_coco_counts)
    ade = verify_ade(ade_root, expected_counts=expected_ade_counts)
    checkpoints = verify_checkpoints(checkpoint_root)
    archives = verify_archives(source_root)
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "sources": archives["archives"],
        "datasets": {"coco": coco, "ade20k": ade},
        "checkpoints": checkpoints,
        "requirements": {
            "coco_categories": 80,
            "coco_segmentation_required": True,
            "ade_ignore_index": 0,
            "ade_ignore_mapping": "handled_by_dataloader",
        },
    }
    ready = bool(coco["ready"] and ade["ready"] and checkpoints["ready"])
    manifest_digest = manifest_sha256(manifest)
    return {
        "schema": SCHEMA,
        "status": "ready" if ready else "incomplete",
        "state": "ready" if ready else "incomplete",
        "ready": ready,
        "datasets": {"coco": coco, "ade20k": ade},
        "checkpoints": checkpoints,
        "sources": archives,
        "manifest": manifest,
        "manifest_sha256": manifest_digest,
        "manifest_hash": manifest_digest,
        "generated_at_utc": _utc_now(),
        "free_space": {
            "coco": _free_space_observation(Path(coco_root), reserve_bytes=reserve_bytes),
            "ade": _free_space_observation(Path(ade_root), reserve_bytes=reserve_bytes),
            "source": _free_space_observation(Path(source_root), reserve_bytes=reserve_bytes),
        },
        "operations": [dict(item) for item in operations],
    }


def prepare(
    *,
    coco_root: Path | str,
    ade_root: Path | str,
    checkpoint_root: Path | str,
    source_root: Path | str,
    report: Path | str,
    download: bool = False,
    extract: bool = False,
    reserve_bytes: int = FREE_SPACE_RESERVE_BYTES,
) -> dict[str, object]:
    """Run the requested read-only/download/extract stages and write a report."""

    source = Path(source_root).expanduser().absolute()
    operations: list[dict[str, object]] = []
    if download:
        source.mkdir(parents=True, exist_ok=True)
        for spec in ARCHIVE_SPECS:
            destination = source / spec.name
            if destination.exists() and not destination.is_symlink():
                operations.append({"kind": "download", "name": spec.name, "status": "skipped_existing"})
                continue
            result = download_archive(spec, destination, reserve_bytes=reserve_bytes)
            operations.append({"kind": "download", **result})
    if extract:
        results = extract_downloaded_archives(
            source,
            Path(coco_root),
            Path(ade_root),
            reserve_bytes=reserve_bytes,
        )
        operations.extend({"kind": "extract", **result} for result in results)
    result = build_report(
        coco_root=coco_root,
        ade_root=ade_root,
        checkpoint_root=checkpoint_root,
        source_root=source_root,
        operations=operations,
        reserve_bytes=reserve_bytes,
    )
    _atomic_json(Path(report).expanduser().absolute(), result)
    return result


def _default_paths() -> dict[str, Path]:
    experiment_root = Path(__file__).resolve().parents[2]
    return {
        "coco_root": Path("/data/datasets/coco"),
        "ade_root": Path("/data/datasets/ADEChallengeData2016"),
        "checkpoint_root": experiment_root / "assets" / "checkpoints",
        "source_root": experiment_root / "assets" / "sources",
        "report": experiment_root / "reports" / "dense_transfer_assets.json",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=defaults["coco_root"])
    parser.add_argument("--ade-root", type=Path, default=defaults["ade_root"])
    parser.add_argument("--checkpoint-root", type=Path, default=defaults["checkpoint_root"])
    parser.add_argument("--source-root", type=Path, default=defaults["source_root"])
    parser.add_argument("--report", type=Path, default=defaults["report"])
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing official archives into source-root using resumable .part files",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="safely extract present archives after an optional download stage",
    )
    parser.add_argument(
        "--free-space-reserve-gib",
        type=float,
        default=15.0,
        help="free-space reserve for download/extraction guards (default: 15 GiB)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.free_space_reserve_gib < 0:
        parser.error("--free-space-reserve-gib must be non-negative")
    reserve_bytes = int(arguments.free_space_reserve_gib * 1024**3)
    try:
        result = prepare(
            coco_root=arguments.coco_root,
            ade_root=arguments.ade_root,
            checkpoint_root=arguments.checkpoint_root,
            source_root=arguments.source_root,
            report=arguments.report,
            download=arguments.download,
            extract=arguments.extract,
            reserve_bytes=reserve_bytes,
        )
    except (AssetError, OSError, ValueError) as exc:
        print(f"dense-transfer asset preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "manifest_sha256": result["manifest_sha256"]}))
    # Incomplete is a valid, explicit verifier result.  A caller that wants a
    # hard gate can use the process status; no incomplete tree is ever called
    # ready in the report.
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
