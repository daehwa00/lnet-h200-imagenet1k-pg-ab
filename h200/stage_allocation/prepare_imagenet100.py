#!/usr/bin/env python3
# ruff: noqa: EM101, EM102, T201, TRY003
# pyright: reportImplicitStringConcatenation=false
"""Create a durable, zero-copy first-100-synset ImageNet view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

SYNSET_PATH = Path(__file__).with_name("imagenet100_synsets.txt")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _desired_classes() -> tuple[str, ...]:
    names = tuple(line.strip() for line in SYNSET_PATH.read_text().splitlines() if line.strip())
    if (
        len(names) != 100
        or len(set(names)) != 100
        or names[0] != "n01440764"
        or names[-1] != "n02077923"
    ):
        raise RuntimeError("frozen ImageNet-100 synset list changed")
    return names


def _directory_names(root: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))


def _numeric_index(name: str) -> int | None:
    match = re.fullmatch(r"(?:class[_-]?)?0*(\d+)(?:[_-].*)?", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _resolve_source_classes(source: Path) -> dict[str, str]:
    desired = _desired_classes()
    train_names = _directory_names(source / "train")
    validation_names = set(_directory_names(source / "val"))
    if len(train_names) != 1000 or set(train_names) != validation_names:
        raise RuntimeError("ImageNet-1K train/validation class directories changed")

    direct: dict[str, str] = {}
    for synset in desired:
        matches = [name for name in train_names if synset in name]
        if len(matches) == 1:
            direct[synset] = matches[0]
    if len(direct) == len(desired):
        return direct

    numeric: dict[int, str] = {}
    for name in train_names:
        index = _numeric_index(name)
        if index is not None and index not in numeric:
            numeric[index] = name
    if all(index in numeric for index in range(100)):
        return {synset: numeric[index] for index, synset in enumerate(desired)}

    sample = list(train_names[:5])
    raise RuntimeError(
        f"cannot map H200 ImageNet class directories; count={len(train_names)}, sample={sample}"
    )


def _link_view(source: Path, output: Path, classes: dict[str, str]) -> None:
    for split in ("train", "val"):
        target_split = output / split
        target_split.mkdir(parents=True, exist_ok=True)
        for synset, source_name in classes.items():
            source_class = (source / split / source_name).resolve()
            target_class = target_split / synset
            if target_class.is_symlink():
                if target_class.resolve() != source_class:
                    raise RuntimeError(f"existing ImageNet-100 link changed: {target_class}")
                continue
            if target_class.exists():
                raise RuntimeError(f"ImageNet-100 view contains a non-link: {target_class}")
            target_class.symlink_to(source_class, target_is_directory=True)


def _count(source: Path, split: str, classes: dict[str, str]) -> int:
    return sum(
        1
        for source_name in classes.values()
        for path in (source / split / source_name).iterdir()
        if path.is_file()
    )


def prepare(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    source_classes = _resolve_source_classes(source)
    classes = _desired_classes()
    train_images = _count(source, "train", source_classes)
    validation_images = _count(source, "val", source_classes)
    if (train_images, validation_images) != (130000, 5000):
        raise RuntimeError(
            "ImageNet-100 image counts changed: "
            f"train={train_images}, validation={validation_images}"
        )
    _link_view(source, output, source_classes)
    identity = hashlib.sha256("\n".join(classes).encode()).hexdigest()
    payload: dict[str, object] = {
        "schema": "lnet.h200.imagenet100.first100_view.v1",
        "source": str(source),
        "view": str(output),
        "selection": "lexicographically first 100 synset directories",
        "classes": list(classes),
        "source_class_by_synset": source_classes,
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
