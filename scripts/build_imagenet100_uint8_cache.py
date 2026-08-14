#!/usr/bin/env python3
"""Build a deterministic resized uint8 cache for ImageNet100 training."""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import InterpolationMode

if TYPE_CHECKING:
    from torch import Tensor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--decoder",
        choices=("pil", "torchvision"),
        default="pil",
    )
    return parser


def _decode(path: str) -> Tensor:
    return decode_image(path, mode=ImageReadMode.RGB)


def _identity(dataset: datasets.ImageFolder, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    for filename, target in dataset.samples:
        path = Path(filename)
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(str(target).encode())
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\n")
    return {
        "root": str(root),
        "samples": len(dataset),
        "classes": len(dataset.classes),
        "class_to_idx": dataset.class_to_idx,
        "path_target_size_sha256": digest.hexdigest(),
    }


def main() -> None:
    args = _parser().parse_args()
    training_root = args.data_root.resolve() / "train"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_images = output_dir / "train-images.npy"
    final_labels = output_dir / "train-labels.npy"
    final_metadata = output_dir / "metadata.json"
    if final_images.exists() or final_labels.exists() or final_metadata.exists():
        message = "refusing to overwrite an existing cache"
        raise FileExistsError(message)

    transform_steps: list[Any] = [
        transforms.Resize(
            (args.image_size, args.image_size),
            interpolation=InterpolationMode.BICUBIC,
        )
    ]
    if args.decoder == "pil":
        transform_steps.append(transforms.PILToTensor())
    loader_function = (
        _decode if args.decoder == "torchvision" else datasets.folder.default_loader
    )
    dataset = datasets.ImageFolder(
        training_root,
        transform=transforms.Compose(transform_steps),
        loader=loader_function,
    )
    if len(dataset.classes) != 100:
        message = "cache source must contain exactly 100 classes"
        raise ValueError(message)
    identity = _identity(dataset, training_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        multiprocessing_context="spawn" if args.workers > 0 else None,
    )

    temporary_images = output_dir / "train-images.tmp.npy"
    temporary_labels = output_dir / "train-labels.tmp.npy"
    images = np.lib.format.open_memmap(
        temporary_images,
        mode="w+",
        dtype=np.uint8,
        shape=(len(dataset), 3, args.image_size, args.image_size),
    )
    labels = np.lib.format.open_memmap(
        temporary_labels,
        mode="w+",
        dtype=np.int64,
        shape=(len(dataset),),
    )
    started = time.perf_counter()
    offset = 0
    for batch_index, (batch_images, batch_labels) in enumerate(loader):
        count = batch_images.shape[0]
        images[offset : offset + count] = batch_images.numpy()
        labels[offset : offset + count] = batch_labels.numpy()
        offset += count
        if (batch_index + 1) % 100 == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "samples": offset,
                        "images_per_second": offset / elapsed,
                    }
                ),
                flush=True,
            )
    images.flush()
    labels.flush()
    del images, labels
    temporary_images.replace(final_images)
    temporary_labels.replace(final_labels)
    metadata = {
        "schema": "lnet.imagenet100.uint8-cache.v1",
        "image_size": args.image_size,
        "images": final_images.name,
        "labels": final_labels.name,
        "decoder": args.decoder,
        "training_identity": identity,
    }
    temporary_metadata = final_metadata.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(final_metadata)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "event": "cache_complete",
                "samples": offset,
                "seconds": elapsed,
                "images_per_second": offset / elapsed,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
