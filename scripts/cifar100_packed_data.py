"""Packed CIFAR-100 dataset and deterministic train/validation loaders."""

# pyright: reportImplicitRelativeImport=false

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from PIL.Image import Image as PILImageType


class PackedCifar100Payload(TypedDict):
    train_images: list[bytes]
    train_targets: list[int]
    test_images: list[bytes]
    test_targets: list[int]


def build_cifar100_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return the shared training and evaluation transforms."""
    mean = (0.5071, 0.4867, 0.4408)
    standard_deviation = (0.2675, 0.2565, 0.2761)
    training = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
            transforms.RandomErasing(p=0.25),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    return training, evaluation


def stratified_cifar100_indices(targets: list[int]) -> tuple[list[int], list[int]]:
    """Return the deterministic 450/50 train/validation split per class."""
    generator = np.random.default_rng(20260730)
    training: list[int] = []
    validation: list[int] = []
    values = np.asarray(targets)
    for class_id in range(100):
        indices = np.flatnonzero(values == class_id)
        generator.shuffle(indices)
        validation.extend(indices[:50].tolist())
        training.extend(indices[50:].tolist())
    return sorted(training), sorted(validation)


@lru_cache(maxsize=1)
def _load_packed(path: Path) -> PackedCifar100Payload:
    return cast(
        "PackedCifar100Payload",
        torch.load(path, map_location="cpu", weights_only=True),
    )


class PackedCifar100(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        path: Path,
        split: str,
        transform: Callable[[PILImageType], torch.Tensor],
    ) -> None:
        payload = _load_packed(path)
        if split == "train":
            self.images = payload["train_images"]
            self.targets = payload["train_targets"]
        elif split == "test":
            self.images = payload["test_images"]
            self.targets = payload["test_targets"]
        else:
            message = f"unsupported packed CIFAR-100 split: {split}"
            raise ValueError(message)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(BytesIO(self.images[index])) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.targets[index]


def build_loaders(
    data_root: Path,
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
]:
    train_transform, evaluation_transform = build_cifar100_transforms()
    path = data_root / "cifar100_packed.pt"
    training = PackedCifar100(path, "train", train_transform)
    evaluation_train = PackedCifar100(path, "train", evaluation_transform)
    test = PackedCifar100(path, "test", evaluation_transform)
    train_indices, validation_indices = stratified_cifar100_indices(training.targets)
    train_loader = DataLoader(
        Subset(training, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        Subset(evaluation_train, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return train_loader, validation_loader, test_loader


__all__ = [
    "PackedCifar100",
    "build_cifar100_transforms",
    "build_loaders",
    "stratified_cifar100_indices",
]
