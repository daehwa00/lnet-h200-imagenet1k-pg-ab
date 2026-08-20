#!/usr/bin/env python3
# ruff: noqa: T201
# pyright: reportImplicitStringConcatenation=false
"""Create a durable, zero-copy first-100-synset ImageNet view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _classes(source: Path) -> tuple[str, ...]:
    train = source / "train"
    validation = source / "val"
    names = tuple(sorted(path.name for path in train.iterdir() if path.is_dir())[:100])
    if len(names) != 100 or names[0] != "n01440764" or names[-1] != "n02077923":
        raise RuntimeError("ImageNet-1K first-100 synset selection changed")
    missing = [name for name in names if not (validation / name).is_dir()]
    if missing:
        raise RuntimeError(f"ImageNet validation is missing selected classes: {missing[:4]}")
    return names


def _link_view(source: Path, output: Path, classes: tuple[str, ...]) -> None:
    for split in ("train", "val"):
        target_split = output / split
        target_split.mkdir(parents=True, exist_ok=True)
        for name in classes:
            source_class = (source / split / name).resolve()
            target_class = target_split / name
            if target_class.is_symlink():
                if target_class.resolve() != source_class:
                    raise RuntimeError(f"existing ImageNet-100 link changed: {target_class}")
                continue
            if target_class.exists():
                raise RuntimeError(f"ImageNet-100 view contains a non-link: {target_class}")
            target_class.symlink_to(source_class, target_is_directory=True)


def _count(source: Path, split: str, classes: tuple[str, ...]) -> int:
    return sum(
        1
        for name in classes
        for path in (source / split / name).iterdir()
        if path.is_file()
    )


def prepare(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    classes = _classes(source)
    train_images = _count(source, "train", classes)
    validation_images = _count(source, "val", classes)
    if (train_images, validation_images) != (130000, 5000):
        raise RuntimeError(
            "ImageNet-100 image counts changed: "
            f"train={train_images}, validation={validation_images}"
        )
    _link_view(source, output, classes)
    identity = hashlib.sha256("\n".join(classes).encode()).hexdigest()
    payload: dict[str, object] = {
        "schema": "lnet.h200.imagenet100.first100_view.v1",
        "source": str(source),
        "view": str(output),
        "selection": "lexicographically first 100 synset directories",
        "classes": list(classes),
        "class_list_sha256": identity,
        "train_images": train_images,
        "validation_images": validation_images,
    }
    _atomic_json(output / "manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
