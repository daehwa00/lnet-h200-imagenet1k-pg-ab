"""Validate and freeze the ImageNet-1K dataset identity used by H200 runs."""

from __future__ import annotations

# ruff: noqa: EM101, EM102, T201, TRY003
import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from torchvision.datasets.folder import IMG_EXTENSIONS

EXPECTED_TRAIN_IMAGES = 1_281_167
EXPECTED_VAL_IMAGES = 50_000
EXPECTED_CLASSES = 1_000
MANIFEST_SCHEMA_VERSION = 1
MANAGED_RECEIPT_SCHEMA_VERSION = 2
FILE_CHUNK_BYTES = 1024 * 1024
CANONICAL_IMAGENET1K_IDENTITY_SHA256 = (
    "992f76a6fb0949826e1217d624fb8307292c04077d385665464cbbb7d917eceb"
)
IDENTITY_METHOD = "sha256(sorted(split-relative-path + NUL + decimal-size + NUL + file-bytes + LF))"
MANAGED_RECEIPT_METHOD = "trusted managed ImageNet mount pinned to a prior full-content SHA-256"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _split_identity(split_root: Path, class_names: list[str]) -> tuple[int, str]:
    """Hash a split in lexical order without retaining 1.28M path tuples."""
    digest = hashlib.sha256()
    count = 0
    for class_name in class_names:
        class_root = split_root / class_name
        for path in sorted(class_root.iterdir(), key=lambda item: item.name):
            if path.is_dir():
                raise RuntimeError(f"canonical ImageNet class directories must be flat: {path}")
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
        "identity_method": IDENTITY_METHOD,
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


def managed_manifest_receipt(
    dataset_root: Path,
    *,
    trusted_receipt_path: Path,
    expected_identity_sha256: str,
    expected_train: int = EXPECTED_TRAIN_IMAGES,
    expected_val: int = EXPECTED_VAL_IMAGES,
    expected_classes: int = EXPECTED_CLASSES,
) -> dict[str, Any]:
    """Pin a previously verified managed mount without re-reading image bytes."""
    try:
        receipt_bytes = trusted_receipt_path.read_bytes()
        trusted = json.loads(receipt_bytes)
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError("trusted managed ImageNet receipt is unreadable") from error
    if not HEX_64.fullmatch(expected_identity_sha256):
        raise ValueError("expected ImageNet identity must be a lowercase SHA-256")
    root = dataset_root.resolve(strict=True)
    if (
        trusted.get("schema") != "lnet.h200.imagenet1k.canonical_receipt.v1"
        or trusted.get("dataset_root") != str(root)
        or trusted.get("identity_sha256") != expected_identity_sha256
        or trusted.get("classes") != expected_classes
        or trusted.get("train_images") != expected_train
        or trusted.get("validation_images") != expected_val
        or trusted.get("identity_method") != "full path-size-content SHA-256"
    ):
        raise RuntimeError("trusted managed ImageNet receipt changed its pinned contract")
    train_root = root / "train"
    val_root = root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise RuntimeError("managed ImageNet root must contain train/ and val/")
    train_classes = _class_names(train_root)
    val_classes = _class_names(val_root)
    if len(train_classes) != expected_classes or train_classes != val_classes:
        raise RuntimeError("managed ImageNet class directory contract changed")
    class_digest = hashlib.sha256("\0".join(train_classes).encode("utf-8")).hexdigest()
    validation = f"pinned-prior-full-content:{expected_identity_sha256}"
    return {
        "schema_version": MANAGED_RECEIPT_SCHEMA_VERSION,
        "identity_method": MANAGED_RECEIPT_METHOD,
        "identity_sha256": expected_identity_sha256,
        "dataset_root": str(root),
        "created_utc": datetime.now(UTC).isoformat(),
        "trusted_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "classes": {"count": len(train_classes), "sha256": class_digest},
        "splits": {
            "train": {"count": expected_train, "content_validation": validation},
            "val": {"count": expected_val, "content_validation": validation},
        },
    }


def reusable_manifest(  # noqa: C901, PLR0912
    dataset_root: Path,
    output: Path,
    *,
    expected_identity_sha256: str,
    expected_train: int = EXPECTED_TRAIN_IMAGES,
    expected_val: int = EXPECTED_VAL_IMAGES,
    expected_classes: int = EXPECTED_CLASSES,
    managed_receipt_path: Path | None = None,
) -> dict[str, Any] | None:
    """Return a pinned durable manifest without re-reading ImageNet file bytes."""
    if not output.is_file():
        return None
    if not HEX_64.fullmatch(expected_identity_sha256):
        raise ValueError("expected ImageNet identity must be a lowercase SHA-256")
    root = dataset_root.resolve(strict=True)
    if not (root / "train").is_dir() or not (root / "val").is_dir():
        raise RuntimeError("cached ImageNet root no longer contains train/ and val/")
    try:
        manifest = json.loads(output.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError("persisted ImageNet manifest is unreadable") from error
    classes = manifest.get("classes")
    splits = manifest.get("splits")
    schema_version = manifest.get("schema_version")
    if (
        schema_version not in (MANIFEST_SCHEMA_VERSION, MANAGED_RECEIPT_SCHEMA_VERSION)
        or manifest.get("identity_sha256") != expected_identity_sha256
        or manifest.get("dataset_root") != str(root)
        or not isinstance(classes, dict)
        or classes.get("count") != expected_classes
        or not HEX_64.fullmatch(str(classes.get("sha256", "")))
        or not isinstance(splits, dict)
    ):
        raise RuntimeError("persisted ImageNet manifest changed its immutable identity")
    if (
        schema_version == MANIFEST_SCHEMA_VERSION
        and manifest.get("identity_method") != IDENTITY_METHOD
    ):
        raise RuntimeError("persisted ImageNet full-content identity method changed")
    if (
        schema_version == MANAGED_RECEIPT_SCHEMA_VERSION
        and manifest.get("identity_method") != MANAGED_RECEIPT_METHOD
    ):
        raise RuntimeError("persisted managed ImageNet receipt method changed")
    if schema_version == MANAGED_RECEIPT_SCHEMA_VERSION:
        if managed_receipt_path is None:
            raise RuntimeError("managed ImageNet receipt reuse requires explicit trust input")
        try:
            receipt_bytes = managed_receipt_path.read_bytes()
        except OSError as error:
            raise RuntimeError("trusted managed ImageNet receipt is unreadable") from error
        if manifest.get("trusted_receipt_sha256") != hashlib.sha256(receipt_bytes).hexdigest():
            raise RuntimeError("managed ImageNet receipt no longer matches its trusted source")
        current_train_classes = _class_names(root / "train")
        current_val_classes = _class_names(root / "val")
        current_class_digest = hashlib.sha256(
            "\0".join(current_train_classes).encode("utf-8")
        ).hexdigest()
        if (
            current_train_classes != current_val_classes
            or len(current_train_classes) != expected_classes
            or classes.get("sha256") != current_class_digest
        ):
            raise RuntimeError("managed ImageNet class directories changed after receipt creation")
    for name, expected_count in (("train", expected_train), ("val", expected_val)):
        split = splits.get(name)
        if not isinstance(split, dict) or split.get("count") != expected_count:
            raise RuntimeError(f"persisted ImageNet {name} split contract changed")
        if schema_version == MANIFEST_SCHEMA_VERSION and not HEX_64.fullmatch(
            str(split.get("relpath_size_content_sha256", ""))
        ):
            raise RuntimeError(f"persisted ImageNet {name} full-content digest changed")
        expected_validation = f"pinned-prior-full-content:{expected_identity_sha256}"
        if (
            schema_version == MANAGED_RECEIPT_SCHEMA_VERSION
            and split.get("content_validation") != expected_validation
        ):
            raise RuntimeError(f"persisted ImageNet {name} managed receipt changed")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-train", type=int, default=EXPECTED_TRAIN_IMAGES)
    parser.add_argument("--expected-val", type=int, default=EXPECTED_VAL_IMAGES)
    parser.add_argument("--expected-classes", type=int, default=EXPECTED_CLASSES)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--managed-canonical-receipt", type=Path)
    parser.add_argument(
        "--expected-identity-sha256",
        default=CANONICAL_IMAGENET1K_IDENTITY_SHA256,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    persisted = (
        reusable_manifest(
            args.root,
            args.output,
            expected_identity_sha256=args.expected_identity_sha256,
            expected_train=args.expected_train,
            expected_val=args.expected_val,
            expected_classes=args.expected_classes,
            managed_receipt_path=args.managed_canonical_receipt,
        )
        if args.reuse_existing
        else None
    )
    cache_hit = persisted is not None
    if persisted is None:
        if args.managed_canonical_receipt is not None:
            manifest = managed_manifest_receipt(
                args.root,
                trusted_receipt_path=args.managed_canonical_receipt,
                expected_identity_sha256=args.expected_identity_sha256,
                expected_train=args.expected_train,
                expected_val=args.expected_val,
                expected_classes=args.expected_classes,
            )
        else:
            manifest = build_manifest(
                args.root,
                expected_train=args.expected_train,
                expected_val=args.expected_val,
                expected_classes=args.expected_classes,
            )
            if manifest["identity_sha256"] != args.expected_identity_sha256:
                raise RuntimeError(
                    "ImageNet-1K content identity differs from the pinned canonical dataset"
                )
        persisted = persist_manifest(manifest, args.output)
    print(f"IMAGENET1K_DATASET_CACHE_HIT={int(cache_hit)}", flush=True)
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
