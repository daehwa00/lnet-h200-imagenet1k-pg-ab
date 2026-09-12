# Copyright (c) 2026

# ruff: noqa: S311, FBT001, FBT002, PLR0917, NPY002

"""PyTorch data pipelines for the dense-transfer preparation experiments.

The two datasets in this module deliberately have a small, framework-neutral
interface.  ``build_dataset`` returns a regular :class:`torch.utils.data.Dataset`;
ADE items are ``(image, mask)`` and COCO items are ``(image, target)``.  The
public collate functions keep that interface when used with a DataLoader:
``ade_collate`` returns padded tensors and ``coco_collate`` returns lists.

There is no implicit download or conversion step here.  In particular, COCO
annotations are read by the official ``pycocotools`` package and masks are
decoded only for the image being indexed.  This is important for a full COCO
checkout, where decoding all masks while constructing the dataset would use a
large amount of memory and make preparation unexpectedly expensive.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF  # noqa: N812

type Task = Literal["ade20k", "coco"]
type Split = Literal["train", "val"]

_ADE_IGNORE_INDEX = 255
_ADE_NUM_CLASSES = 150
_ADE_CROP_SIZE = 512
_ADE_SCALE = (2048, 512)  # width, height, as used by the standard ADE pipeline
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# COCO's category ids intentionally have holes.  This is the canonical order
# used to map the 80 categories to labels 1..80.  Tiny fixtures often contain
# only a few categories; those use the sorted ids present in the fixture (see
# ``_category_mapping``), while a complete annotation file always gets this
# canonical mapping.
COCO_CATEGORY_IDS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)


@dataclass(frozen=True)
class ADEMetadata:
    """Size and identity information retained for ADE validation."""

    image_id: str
    image_path: Path
    mask_path: Path
    original_size: tuple[int, int]  # (height, width)
    resized_size: tuple[int, int]  # (height, width)
    original_mask: Tensor | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "image_id": self.image_id,
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path),
            "original_size": self.original_size,
            "resized_size": self.resized_size,
            "size": self.resized_size,
        }
        if self.original_mask is not None:
            result["original_mask"] = self.original_mask
        return result


@dataclass(frozen=True)
class DatasetValidation:
    """Read-only filename/count validation result."""

    task: str
    split: str
    expected: int
    present: int
    missing: tuple[str, ...]
    extra: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing and self.expected == self.present


@dataclass(frozen=True)
class LoaderBundle:
    """Named train/validation loader pair accepted by the dense CLI."""

    train: DataLoader[Any]
    validation: DataLoader[Any]

    @property
    def val(self) -> DataLoader[Any]:
        return self.validation

    def __iter__(self) -> Iterator[DataLoader[Any]]:
        # Retain the convenient ``train_loader, val_loader = build_loaders(...)``
        # idiom for small scripts and tests.
        yield self.train
        yield self.validation


def _as_path(root: str | Path) -> Path:
    return Path(root).expanduser()


def _resolve_ade_root(root: str | Path) -> Path:
    path = _as_path(root)
    if (path / "images").is_dir() and (path / "annotations").is_dir():
        return path
    child = path / "ADEChallengeData2016"
    if (child / "images").is_dir() and (child / "annotations").is_dir():
        return child
    # Return the most useful path in the error message.  The constructor will
    # produce a detailed missing-directory error below.
    return child if child.exists() else path


def _resolve_coco_root(root: str | Path) -> Path:
    path = _as_path(root)
    if (path / "annotations").is_dir() and (
        (path / "train2017").is_dir() or (path / "val2017").is_dir()
    ):
        return path
    child = path / "coco"
    if (child / "annotations").is_dir():
        return child
    return path


def _require_directory(path: Path, what: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{what} directory does not exist: {path}")


def _list_image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def _map_ade_labels(mask: Image.Image | np.ndarray) -> Tensor:
    """Map ADE's 1..150 labels to 0..149 and everything else to 255."""

    values = np.asarray(mask, dtype=np.int64)
    mapped = np.full(values.shape, _ADE_IGNORE_INDEX, dtype=np.int64)
    valid = (values >= 1) & (values <= _ADE_NUM_CLASSES)
    mapped[valid] = values[valid] - 1
    return torch.from_numpy(mapped)


def map_ade_labels(mask: Image.Image | np.ndarray | Tensor) -> Tensor:
    """Public ADE label mapping helper used by tests and validation code."""

    if isinstance(mask, Tensor):
        mask = mask.detach().cpu().numpy()
    return _map_ade_labels(mask)


def _resize_pair(
    image: Image.Image,
    mask: Image.Image,
    size: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    width, height = size
    return (
        image.resize((width, height), Image.Resampling.BILINEAR),
        mask.resize((width, height), Image.Resampling.NEAREST),
    )


def _random_resize_pair(
    image: Image.Image,
    mask: Image.Image,
    *,
    ratio_range: tuple[float, float] = (0.5, 2.0),
) -> tuple[Image.Image, Image.Image]:
    """Randomly resize around 2048x512 while retaining the source aspect ratio."""

    width, height = image.size
    factor = random.uniform(*ratio_range)
    base_scale = min(_ADE_SCALE[0] / max(width, 1), _ADE_SCALE[1] / max(height, 1))
    scale = base_scale * factor
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return _resize_pair(image, mask, (new_width, new_height))


def _validation_resize_pair(
    image: Image.Image,
    mask: Image.Image,
    *,
    short_side: int = 512,
    max_long_side: int = 2048,
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    short = min(width, height)
    long = max(width, height)
    scale = min(short_side / max(short, 1), max_long_side / max(long, 1))
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return _resize_pair(image, mask, (new_width, new_height))


def _photo_metric_distortion(image: Image.Image) -> Image.Image:
    """A compact PIL implementation of the usual PhotoMetricDistortion op."""

    image = image.convert("RGB")
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.5))
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    if random.random() < 0.5:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5))
    if random.random() < 0.5:
        # PIL stores HSV hue on [0, 255].  A +/-18 degree perturbation is the
        # same range commonly used by segmentation PhotoMetricDistortion.
        hue_delta = random.uniform(-18.0, 18.0) / 360.0 * 255.0
        hsv = np.asarray(image.convert("HSV"), dtype=np.int16)
        hsv[..., 0] = (hsv[..., 0] + round(hue_delta)) % 256
        image = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
    return image


def _crop_fraction(mask: Tensor) -> float:
    valid = mask != _ADE_IGNORE_INDEX
    count = int(valid.sum())
    if count == 0:
        return 0.0
    values = mask[valid]
    counts = torch.bincount(values, minlength=_ADE_NUM_CLASSES)
    return float(counts.max().item()) / float(count)


def _random_crop_pair(
    image: Image.Image,
    mask: Image.Image,
    *,
    crop_size: int = _ADE_CROP_SIZE,
    max_category_fraction: float = 0.75,
    max_attempts: int = 10,
) -> tuple[Image.Image, Image.Image]:
    """Crop a square, rejecting nearly single-class crops when possible."""

    width, height = image.size
    max_left = max(width - crop_size, 0)
    max_top = max(height - crop_size, 0)
    candidates: list[tuple[int, int, float, Image.Image, Image.Image]] = []
    attempts = max(1, max_attempts)
    for _ in range(attempts):
        left = random.randint(0, max_left) if max_left else 0
        top = random.randint(0, max_top) if max_top else 0
        right = min(left + crop_size, width)
        bottom = min(top + crop_size, height)
        candidate_image = image.crop((left, top, right, bottom))
        candidate_mask = mask.crop((left, top, right, bottom))
        fraction = _crop_fraction(_map_ade_labels(candidate_mask))
        candidates.append((left, top, fraction, candidate_image, candidate_mask))
        if fraction <= max_category_fraction:
            return candidate_image, candidate_mask
    # If every candidate is dominated by one class, preserving a real crop is
    # preferable to an unbounded retry loop.  The best candidate is useful for
    # highly imbalanced images and makes the bound explicit and deterministic.
    _left, _top, _fraction, candidate_image, candidate_mask = min(candidates, key=lambda item: item[2])
    return candidate_image, candidate_mask


def _normalize_and_pad(
    image: Image.Image,
    mask: Tensor,
    *,
    size: int | None = None,
) -> tuple[Tensor, Tensor]:
    image_tensor = TF.to_tensor(image.convert("RGB"))
    image_tensor = TF.normalize(image_tensor, _IMAGENET_MEAN, _IMAGENET_STD)
    if size is None:
        return image_tensor, mask.to(dtype=torch.int64)
    height, width = image_tensor.shape[-2:]
    if height > size or width > size:
        image_tensor = image_tensor[:, :size, :size]
        mask = mask[:size, :size]
        height, width = image_tensor.shape[-2:]
    output_image = torch.zeros((3, size, size), dtype=image_tensor.dtype)
    output_mask = torch.full((size, size), _ADE_IGNORE_INDEX, dtype=torch.int64)
    output_image[:, :height, :width] = image_tensor
    output_mask[:height, :width] = mask.to(dtype=torch.int64)
    return output_image, output_mask


class ADE20KDataset(Dataset[tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Any]]]):
    """ADE20K semantic segmentation dataset with native PyTorch outputs."""

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        *,
        train: bool | None = None,
        crop_size: int = _ADE_CROP_SIZE,
        max_category_fraction: float = 0.75,
        max_crop_attempts: int = 10,
        return_metadata: bool = False,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"ADE split must be 'train' or 'val', got {split!r}")
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        self.root = _resolve_ade_root(root)
        self.split: Split = split
        self.train = split == "train" if train is None else bool(train)
        self.crop_size = int(crop_size)
        self.max_category_fraction = float(max_category_fraction)
        self.max_crop_attempts = max(1, int(max_crop_attempts))
        self.return_metadata = bool(return_metadata)

        image_subdir = "training" if split == "train" else "validation"
        image_dir = self.root / "images" / image_subdir
        mask_dir = self.root / "annotations" / image_subdir
        _require_directory(image_dir, "ADE image")
        _require_directory(mask_dir, "ADE annotation")
        self.image_paths = _list_image_files(image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"no ADE images found in {image_dir}")
        pairs: list[tuple[Path, Path]] = []
        missing: list[str] = []
        for image_path in self.image_paths:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                missing.append(mask_path.name)
            else:
                pairs.append((image_path, mask_path))
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"ADE {split} annotations missing for {len(missing)} image(s) in {mask_dir}: {preview}"
            )
        self.samples = pairs
        self.metadata: list[ADEMetadata | None] = [None] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, index: int) -> tuple[Image.Image, Image.Image, ADEMetadata]:
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            original_size = (image.height, image.width)
        with Image.open(mask_path) as source_mask:
            # ADE labels are stored as 8-bit values, including 255/void.
            mask = source_mask.convert("L")
        original_mask = _map_ade_labels(mask) if self.return_metadata else None
        metadata = ADEMetadata(
            image_id=image_path.stem,
            image_path=image_path,
            mask_path=mask_path,
            original_size=original_size,
            resized_size=original_size,
            original_mask=original_mask,
        )
        return image, mask, metadata

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Any]]:
        image, raw_mask, metadata = self._load_pair(index)
        mapped_mask = _map_ade_labels(raw_mask)
        if self.train:
            image, raw_mask = _random_resize_pair(image, raw_mask)
            mapped_mask = _map_ade_labels(raw_mask)
            image, raw_mask = _random_crop_pair(
                image,
                raw_mask,
                crop_size=self.crop_size,
                max_category_fraction=self.max_category_fraction,
                max_attempts=self.max_crop_attempts,
            )
            mapped_mask = _map_ade_labels(raw_mask)
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                mapped_mask = torch.flip(mapped_mask, dims=(1,))
            image = _photo_metric_distortion(image)
            image_tensor, mask_tensor = _normalize_and_pad(image, mapped_mask, size=self.crop_size)
            metadata = ADEMetadata(
                image_id=metadata.image_id,
                image_path=metadata.image_path,
                mask_path=metadata.mask_path,
                original_size=metadata.original_size,
                resized_size=(self.crop_size, self.crop_size),
                original_mask=metadata.original_mask,
            )
        else:
            image, raw_mask = _validation_resize_pair(image, raw_mask)
            mapped_mask = _map_ade_labels(raw_mask)
            image_tensor, mask_tensor = _normalize_and_pad(image, mapped_mask)
            metadata = ADEMetadata(
                image_id=metadata.image_id,
                image_path=metadata.image_path,
                mask_path=metadata.mask_path,
                original_size=metadata.original_size,
                resized_size=(image_tensor.shape[-2], image_tensor.shape[-1]),
                original_mask=metadata.original_mask,
            )
        # Keep the returned original GT available to the evaluator, but do not
        # retain every validation mask in a long-lived dataset object when a
        # single-process DataLoader is used.
        self.metadata[index] = replace(metadata, original_mask=None)
        if self.return_metadata:
            return image_tensor, mask_tensor, metadata.as_dict()
        return image_tensor, mask_tensor

    def get_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata without requiring callers to decode the sample twice."""

        cached = self.metadata[index]
        if cached is None:
            self[index]
            cached = self.metadata[index]
        if cached is None:  # pragma: no cover - guarded by the load above
            raise RuntimeError("ADE metadata cache was not populated")
        return cached.as_dict()

    def original_mask(self, index: int) -> Tensor:
        """Load an original-resolution mapped GT mask without resizing it."""

        _image_path, mask_path = self.samples[index]
        with Image.open(mask_path) as source_mask:
            return _map_ade_labels(source_mask.convert("L"))


# A short alias is convenient for callers that use the common dataset name.
ADEDataset = ADE20KDataset


def _load_coco_class() -> Any:
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "COCODataset requires the official pycocotools package; install "
            "'pycocotools' before opening a COCO dataset"
        ) from exc
    return COCO


def _category_mapping(category_ids: Sequence[int]) -> tuple[dict[int, int], dict[int, int]]:
    ids = tuple(sorted({int(value) for value in category_ids}))
    ordered = COCO_CATEGORY_IDS if set(ids) == set(COCO_CATEGORY_IDS) else ids
    to_label = {category_id: label for label, category_id in enumerate(ordered, start=1)}
    to_category = {label: category_id for category_id, label in to_label.items()}
    return to_label, to_category


def _coerce_image_id(value: Any) -> int:
    if isinstance(value, Tensor):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() == 0:
            raise ValueError("image_id tensor is empty")
        return int(flat[0].item())
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if flat.size == 0:
            raise ValueError("image_id array is empty")
        return int(flat[0])
    return int(value)


def _clip_box(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    if len(box) < 4:
        return None
    x, y, box_width, box_height = (float(box[i]) for i in range(4))
    x0 = max(0.0, min(x, float(width)))
    y0 = max(0.0, min(y, float(height)))
    x1 = max(0.0, min(x + max(box_width, 0.0), float(width)))
    y1 = max(0.0, min(y + max(box_height, 0.0), float(height)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


class COCODataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """COCO detection/instance-segmentation dataset.

    Images are returned in ``[0, 1]`` CHW format.  The target follows the
    torchvision detection convention and additionally carries ``category_ids``
    (the original COCO ids), ``orig_size``, and ``size``.  Labels are always
    contiguous and start at one; label zero remains reserved for background.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        *,
        train: bool | None = None,
        validate_files: bool = True,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"COCO split must be 'train' or 'val', got {split!r}")
        self.root = _resolve_coco_root(root)
        self.split: Split = split
        self.train = split == "train" if train is None else bool(train)
        image_dir = self.root / ("train2017" if split == "train" else "val2017")
        annotation_path = self.root / "annotations" / f"instances_{'train2017' if split == 'train' else 'val2017'}.json"
        _require_directory(image_dir, "COCO image")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotation file does not exist: {annotation_path}")
        coco_type = _load_coco_class()
        self.coco = coco_type(str(annotation_path))
        self.ids = sorted(int(image_id) for image_id in self.coco.getImgIds())
        if not self.ids:
            raise ValueError(f"COCO annotation file contains no images: {annotation_path}")
        self.image_dir = image_dir
        self.image_paths: dict[int, Path] = {}
        missing: list[str] = []
        for image_id in self.ids:
            info = self.coco.loadImgs([image_id])[0]
            file_name = Path(str(info["file_name"]))
            candidate = image_dir / file_name
            if not candidate.is_file():
                # Some private fixtures store the split directory in
                # ``file_name`` already.  Try the root once, without walking or
                # modifying the checkout.
                alternate = self.root / file_name
                candidate = alternate if alternate.is_file() else candidate
            if not candidate.is_file():
                missing.append(str(file_name))
            self.image_paths[image_id] = candidate
        if validate_files and missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"COCO {split} image files missing for {len(missing)} annotation(s): {preview}"
            )
        category_ids = sorted(int(category_id) for category_id in self.coco.getCatIds())
        self.category_id_to_label, self.label_to_category_id = _category_mapping(category_ids)
        self.category_ids = tuple(category_ids)

    def __len__(self) -> int:
        return len(self.ids)

    def _annotations_for(self, image_id: int, height: int, width: int) -> dict[str, Tensor]:
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(annotation_ids)
        boxes: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        category_ids: list[int] = []
        areas: list[float] = []
        crowds: list[int] = []
        masks: list[np.ndarray] = []
        for annotation in annotations:
            category_id = int(annotation.get("category_id", -1))
            label = self.category_id_to_label.get(category_id)
            if label is None:
                # An annotation referring to a category omitted from the JSON's
                # category table cannot be evaluated and is safest to ignore.
                continue
            box = _clip_box(annotation.get("bbox", ()), width, height)
            if box is None:
                continue
            try:
                decoded = np.asarray(self.coco.annToMask(annotation), dtype=np.uint8)
            except (KeyError, TypeError, ValueError, IndexError):
                # Malformed optional segmentation should not make a valid box
                # unusable.  A zero mask is also what COCO uses for an empty
                # polygon after rasterisation.
                decoded = np.zeros((height, width), dtype=np.uint8)
            if decoded.shape != (height, width):
                decoded = np.zeros((height, width), dtype=np.uint8)
            boxes.append(box)
            labels.append(label)
            category_ids.append(category_id)
            area = float(annotation.get("area", (box[2] - box[0]) * (box[3] - box[1])))
            areas.append(max(area, 0.0))
            crowds.append(int(annotation.get("iscrowd", 0)))
            masks.append(decoded)
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.int64)
            category_tensor = torch.tensor(category_ids, dtype=torch.int64)
            area_tensor = torch.tensor(areas, dtype=torch.float32)
            crowd_tensor = torch.tensor(crowds, dtype=torch.int64)
            mask_tensor = torch.from_numpy(np.stack(masks, axis=0).astype(np.uint8, copy=False))
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.int64)
            category_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            crowd_tensor = torch.zeros((0,), dtype=torch.int64)
            mask_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
        return {
            "boxes": box_tensor,
            "labels": label_tensor,
            "category_ids": category_tensor,
            "masks": mask_tensor,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": crowd_tensor,
            "orig_size": torch.tensor((height, width), dtype=torch.int64),
            "size": torch.tensor((height, width), dtype=torch.int64),
        }

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_id = self.ids[index]
        image_path = self.image_paths[image_id]
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image does not exist: {image_path}")
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        height, width = image.height, image.width
        image_tensor = TF.to_tensor(image)
        target = self._annotations_for(image_id, height, width)
        if self.train and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=(2,))
            masks = target["masks"]
            target["masks"] = torch.flip(masks, dims=(2,))
            boxes = target["boxes"]
            if boxes.numel():
                old_left = boxes[:, 0].clone()
                old_right = boxes[:, 2].clone()
                boxes[:, 0] = float(width) - old_right
                boxes[:, 2] = float(width) - old_left
                target["boxes"] = boxes
        return image_tensor, target

    def category_id(self, label: int) -> int:
        """Translate a contiguous model label back to an official COCO id."""

        if int(label) not in self.label_to_category_id:
            raise KeyError(f"unknown contiguous COCO label: {label}")
        return self.label_to_category_id[int(label)]


# Common spelling used by torchvision-style callers.
COCODetectionDataset = COCODataset


def _pad_ade_batch(images: Sequence[Tensor], masks: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    if not images:
        raise ValueError("cannot collate an empty ADE batch")
    # VA's dense path requires both spatial dimensions to be divisible by its
    # patch stride.  Validation preserves aspect ratio, so even a one-item
    # batch (for example 512x683) must be rounded up here rather than only to
    # the largest item in the batch.  Training crops are already 512x512 and
    # therefore remain byte-for-byte the same shape.
    stride = 32
    max_height = max(int(image.shape[-2]) for image in images)
    max_width = max(int(image.shape[-1]) for image in images)
    max_height = ((max_height + stride - 1) // stride) * stride
    max_width = ((max_width + stride - 1) // stride) * stride
    image_batch = torch.zeros((len(images), 3, max_height, max_width), dtype=images[0].dtype)
    mask_batch = torch.full(
        (len(masks), max_height, max_width), _ADE_IGNORE_INDEX, dtype=torch.int64
    )
    for index, (image, mask) in enumerate(zip(images, masks, strict=True)):
        height, width = image.shape[-2:]
        image_batch[index, :, :height, :width] = image
        mask_batch[index, :height, :width] = mask.to(dtype=torch.int64)
    return image_batch, mask_batch


def ade_collate(
    batch: Sequence[
        tuple[Tensor, Tensor]
        | tuple[Tensor, Tensor, Mapping[str, Any]]
    ],
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, list[Mapping[str, Any]]]:
    """Collate ADE samples and pad variable-size validation images with ignore."""

    if not batch:
        raise ValueError("cannot collate an empty ADE batch")
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    image_batch, mask_batch = _pad_ade_batch(images, masks)
    if any(len(item) >= 3 for item in batch):
        metadata = [item[2] if len(item) >= 3 else {} for item in batch]
        return image_batch, mask_batch, metadata
    return image_batch, mask_batch


def coco_collate(
    batch: Sequence[tuple[Tensor, Mapping[str, Tensor]]],
) -> tuple[list[Tensor], list[Mapping[str, Tensor]]]:
    """Collate variable-size COCO images without introducing implicit padding."""

    if not batch:
        raise ValueError("cannot collate an empty COCO batch")
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def validate_dataset_files(
    dataset: Dataset[Any] | str | Path,
    task: Task | None = None,
    split: Split | None = None,
) -> DatasetValidation:
    """Validate image/annotation filename counts without changing the checkout.

    A dataset instance is preferred because its resolved paths and JSON ids are
    already available.  Passing ``root`` plus ``task``/``split`` constructs the
    corresponding dataset (and therefore performs the same validation).
    """

    if isinstance(dataset, ADE20KDataset):
        expected = len(dataset.image_paths)
        present = len(dataset.samples)
        return DatasetValidation(
            "ade20k",
            dataset.split,
            expected,
            present,
            tuple(
                str(path.name)
                for path in dataset.image_paths
                if not (
                    dataset.root
                    / "annotations"
                    / ("training" if dataset.split == "train" else "validation")
                    / f"{path.stem}.png"
                ).is_file()
            ),
            (),
        )
    if isinstance(dataset, COCODataset):
        missing = tuple(
            str(dataset.coco.loadImgs([image_id])[0]["file_name"])
            for image_id, path in dataset.image_paths.items()
            if not path.is_file()
        )
        return DatasetValidation(
            "coco", dataset.split, len(dataset.ids), len(dataset.ids) - len(missing), missing, ()
        )
    if task is None or split is None:
        raise TypeError("task and split are required when validating from a root path")
    built = build_dataset(task, Path(dataset), split, train=split == "train")
    return validate_dataset_files(built)


def validate_filenames(
    task: Task,
    root: str | Path,
    split: Split,
) -> DatasetValidation:
    """Alias with an explicit root-first signature for preparation scripts."""

    return validate_dataset_files(root, task=task, split=split)


@overload
def build_dataset(
    task: Task,
    root: str | Path,
    split: Split,
    train: bool,
    **kwargs: Any,
) -> ADE20KDataset: ...


@overload
def build_dataset(
    task: Literal["coco"],
    root: str | Path,
    split: Split,
    train: bool,
    **kwargs: Any,
) -> COCODataset: ...


def build_dataset(
    task: Task,
    root: str | Path,
    split: Split,
    train: bool,
    **kwargs: Any,
) -> ADE20KDataset | COCODataset:
    """Build one of the preparation datasets without downloading assets."""

    if task == "ade20k":
        return ADE20KDataset(root, split, train=train, **kwargs)
    if task == "coco":
        return COCODataset(root, split, train=train, **kwargs)
    raise ValueError(f"unknown dense-transfer task: {task!r}")


def _seed_worker(worker_id: int) -> None:
    # DataLoader derives each worker's torch seed from its generator.  Mirror
    # that seed into Python/NumPy because ADE's geometric and photometric
    # augmentations intentionally use those two RNGs.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def build_loaders(
    task: Task,
    data_root: str | Path,
    physical_batch_size: int = 2,
    workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int = 501,
    **dataset_kwargs: Any,
) -> LoaderBundle:
    """Build deterministic train/validation loaders for the dense engine.

    ``physical_batch_size`` and ``workers`` are positional-compatible with the
    CLI's preparation API.  Any remaining keyword options are forwarded to
    both datasets; task-specific options can be supplied with
    ``train_dataset_kwargs``/``val_dataset_kwargs`` mappings.
    """

    batch_size_alias = dataset_kwargs.pop("batch_size", None)
    if batch_size_alias is not None:
        if physical_batch_size != 2 and int(batch_size_alias) != physical_batch_size:
            raise ValueError("physical_batch_size and batch_size disagree")
        physical_batch_size = int(batch_size_alias)
    num_workers_alias = dataset_kwargs.pop("num_workers", None)
    if num_workers_alias is not None:
        workers = int(num_workers_alias)
    if physical_batch_size <= 0:
        raise ValueError("physical_batch_size must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative")
    train_options = dict(dataset_kwargs.pop("train_dataset_kwargs", {}) or {})
    val_options = dict(dataset_kwargs.pop("val_dataset_kwargs", {}) or {})
    # ``validate_files`` is useful for a one-time preparation audit but can be
    # disabled by a caller that already performed that read-only check.
    train_options.update(dataset_kwargs)
    val_options.update(dataset_kwargs)
    if task == "ade20k":
        # Original-resolution GT is needed for the standard ADE single-scale
        # score.  The third collate item is metadata plus this lazy-loaded GT.
        val_options.setdefault("return_metadata", True)
    train_dataset = build_dataset(task, data_root, "train", train=True, **train_options)
    val_dataset = build_dataset(task, data_root, "val", train=False, **val_options)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    use_persistent = bool(persistent_workers and workers > 0)
    common: dict[str, Any] = {
        "batch_size": int(physical_batch_size),
        "num_workers": int(workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": use_persistent,
        "worker_init_fn": _seed_worker,
    }
    if workers > 0:
        # The parent initializes CUDA before creating these loaders. Do not
        # inherit its CUDA/native-library state through fork.
        common["multiprocessing_context"] = "spawn"
        common["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        collate_fn=ade_collate if task == "ade20k" else coco_collate,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=ade_collate if task == "ade20k" else coco_collate,
        **common,
    )
    # Expose the generator for checkpointing/resume code without changing the
    # stable two-loader return value.
    train_loader.generator = generator  # type: ignore[attr-defined]
    train_loader.dense_transfer_generator = generator  # type: ignore[attr-defined]
    return LoaderBundle(train_loader, val_loader)


def _move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_nested(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_nested(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested(item, device) for item in value)
    return value


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device | str = "cpu",
    task: Task = "ade20k",
    bf16: bool = False,
) -> dict[str, Any]:
    """Run a read-only validation pass using the public metric APIs.

    This adapter is intentionally small so the CLI can use a data-owned
    evaluator without embedding task-specific assumptions.  Training and
    benchmarking remain in ``dense_transfer.engine``.
    """

    target_device = torch.device(device)
    model_was_training = model.training
    model.eval()
    try:
        if task == "ade20k":
            from .metrics import SegmentationMetric

            metric = SegmentationMetric(num_classes=_ADE_NUM_CLASSES, ignore_index=_ADE_IGNORE_INDEX)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if bf16 and target_device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                for batch in loader:
                    images, masks = batch[0], batch[1]
                    outputs = model(images.to(target_device))
                    logits = outputs.get("logits", outputs) if isinstance(outputs, Mapping) else outputs
                    if not isinstance(logits, Tensor):
                        raise TypeError("ADE model output must be logits or a mapping containing logits")
                    metadata = batch[2] if len(batch) >= 3 else None
                    if metadata is None:
                        if logits.shape[-2:] != masks.shape[-2:]:
                            logits = torch.nn.functional.interpolate(
                                logits,
                                size=masks.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        metric.update(logits.detach().float().cpu(), masks)
                        continue
                    canvas_height, canvas_width = images.shape[-2:]
                    if logits.shape[-2:] != (canvas_height, canvas_width):
                        # Restore the model output to the actual padded canvas
                        # first.  Cropping after this interpolation is crucial:
                        # resizing a smaller image directly from the batch
                        # canvas would squeeze another sample's right/bottom
                        # padding into its content.
                        logits = torch.nn.functional.interpolate(
                            logits,
                            size=(canvas_height, canvas_width),
                            mode="bilinear",
                            align_corners=False,
                        )
                    for index, item in enumerate(metadata):
                        resized_height, resized_width = item["resized_size"]
                        original_height, original_width = item["original_size"]
                        sample_logits = logits[
                            index : index + 1,
                            :,
                            : int(resized_height),
                            : int(resized_width),
                        ]
                        sample_logits = torch.nn.functional.interpolate(
                            sample_logits,
                            size=(int(original_height), int(original_width)),
                            mode="bilinear",
                            align_corners=False,
                        )
                        original_mask = item.get("original_mask")
                        if original_mask is None:
                            with Image.open(item["mask_path"]) as source_mask:
                                original_mask = map_ade_labels(source_mask.convert("L"))
                        metric.update(sample_logits.detach().float().cpu(), original_mask.unsqueeze(0))
            return metric.compute()
        if task == "coco":
            from .metrics import COCOEvaluator

            evaluator = COCOEvaluator(getattr(loader, "dataset", None))
            with torch.inference_mode():
                for images, targets in loader:
                    device_images = [_move_nested(image, target_device) for image in images]
                    outputs = model(device_images)
                    if not isinstance(outputs, (list, tuple)):
                        raise TypeError("COCO model output must be a list of prediction mappings")
                    image_ids = [_coerce_image_id(target["image_id"]) for target in targets]
                    evaluator.update([_move_nested(output, torch.device("cpu")) for output in outputs], image_ids=image_ids)
            return evaluator.compute()
        raise ValueError(f"unknown dense-transfer task: {task!r}")
    finally:
        model.train(model_was_training)


__all__ = [
    "COCO_CATEGORY_IDS",
    "ADE20KDataset",
    "ADEDataset",
    "ADEMetadata",
    "COCODataset",
    "COCODetectionDataset",
    "DatasetValidation",
    "LoaderBundle",
    "Split",
    "Task",
    "ade_collate",
    "build_dataset",
    "build_loaders",
    "coco_collate",
    "evaluate",
    "map_ade_labels",
    "validate_dataset_files",
    "validate_filenames",
]
