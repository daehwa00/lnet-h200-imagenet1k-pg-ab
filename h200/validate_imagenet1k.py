"""Validate and freeze the ImageNet-1K dataset identity used by H200 runs."""

from __future__ import annotations

# ruff: noqa: EM101, EM102, T201, TRY003
import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from torchvision.datasets.folder import IMG_EXTENSIONS

EXPECTED_TRAIN_IMAGES = 1_281_167
EXPECTED_VAL_IMAGES = 50_000
EXPECTED_CLASSES = 1_000
MANIFEST_SCHEMA_VERSION = 1
FILE_CHUNK_BYTES = 1024 * 1024


def _split_identity(split_root: Path, class_names: list[str]) -> tuple[int, str]:
    """Hash a split in lexical order without retaining 1.28M path tuples."""
    digest = hashlib.sha256()
    count = 0
    for class_name in class_names:
        class_root = split_root / class_name
        for path in sorted(class_root.iterdir(), key=lambda item: item.name):
            if path.is_dir():
                raise RuntimeError(
                    f"canonical ImageNet class directories must be flat: {path}"
                )
            if path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            if path.is_symlink():
                raise RuntimeError(f"dataset image symlinks are not allowed: {path}")
            relative = path.relative_to(split_root).as_posix()
            if "\n" in relative or "\0" in relative:
                raise RuntimeError(f"dataset path contains an unsupported control byte: {path}")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                while chunk := stream.read(FILE_CHUNK_BYTES):
                    digest.update(chunk)
            digest.update(b"\n")
            count += 1
    return count, digest.hexdigest()


def _class_names(split_root: Path) -> list[str]:
    return sorted(path.name for path in split_root.iterdir() if path.is_dir())


def build_manifest(
    dataset_root: Path,
    *,
    expected_train: int = EXPECTED_TRAIN_IMAGES,
    expected_val: int = EXPECTED_VAL_IMAGES,
    expected_classes: int = EXPECTED_CLASSES,
) -> dict[str, Any]:
    root = dataset_root.resolve(strict=True)
    train_root = root / "train"
    val_root = root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise RuntimeError(f"ImageNet root must contain train/ and val/: {root}")

    train_classes = _class_names(train_root)
    val_classes = _class_names(val_root)
    if len(train_classes) != expected_classes:
        raise RuntimeError(
            f"ImageNet class count mismatch: expected={expected_classes}, "
            f"train={len(train_classes)}"
        )
    if train_classes != val_classes:
        raise RuntimeError("ImageNet train/val class directory sets differ")

    train_count, train_digest = _split_identity(train_root, train_classes)
    val_count, val_digest = _split_identity(val_root, val_classes)
    if train_count != expected_train or val_count != expected_val:
        raise RuntimeError(
            "ImageNet image count mismatch: "
            f"expected=train:{expected_train},val:{expected_val}; "
            f"actual=train:{train_count},val:{val_count}"
        )

    class_digest = hashlib.sha256("\0".join(train_classes).encode("utf-8")).hexdigest()
    identity_payload = {
        "classes": {"count": len(train_classes), "sha256": class_digest},
        "splits": {
            "train": {
                "count": train_count,
                "relpath_size_content_sha256": train_digest,
            },
            "val": {
                "count": val_count,
                "relpath_size_content_sha256": val_digest,
            },
        },
    }
    identity_bytes = json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "identity_method": (
            "sha256(sorted(split-relative-path + NUL + decimal-size + NUL + "
            "file-bytes + LF))"
        ),
        "identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "dataset_root": str(root),
        "created_utc": datetime.now(UTC).isoformat(),
        **identity_payload,
    }


def persist_manifest(manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != manifest["identity_sha256"]:
            raise RuntimeError(
                "persisted dataset identity differs from the current ImageNet tree: "
                f"existing={existing.get('identity_sha256')}, "
                f"current={manifest['identity_sha256']}"
            )
        return existing

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-train", type=int, default=EXPECTED_TRAIN_IMAGES)
    parser.add_argument("--expected-val", type=int, default=EXPECTED_VAL_IMAGES)
    parser.add_argument("--expected-classes", type=int, default=EXPECTED_CLASSES)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_manifest(
        args.root,
        expected_train=args.expected_train,
        expected_val=args.expected_val,
        expected_classes=args.expected_classes,
    )
    persisted = persist_manifest(manifest, args.output)
    print(
        "IMAGENET1K_DATASET="
        f"train:{persisted['splits']['train']['count']},"
        f"val:{persisted['splits']['val']['count']},"
        f"classes:{persisted['classes']['count']},"
        f"identity:{persisted['identity_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
