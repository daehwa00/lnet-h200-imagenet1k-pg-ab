#!/usr/bin/env python3
"""Train the preregistered ALPHABET-2D-T0 classifier on ImageNet100."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, S311, T201, TRY003
# pyright: reportExplicitAny=false, reportAny=false

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import InterpolationMode

from lnet.alphabet2d import Alphabet2D, Alphabet2DConfig
from lnet.alphabet2d_t1 import (
    Alphabet2DT1Compact,
    Alphabet2DT1Config,
)
from lnet.pac_capture_safe_orthogonal import prepare_capture_safe_orthogonal_

AmpDtype = Literal["bfloat16", "float16", "none"]
Architecture = Literal["t0", "t1_product", "t1_pole_free"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    data_root: str
    output_dir: str
    architecture: Architecture = "t1_product"
    image_size: int = 224
    patch_size: int = 16
    model_dim: int = 192
    modes: int = 16
    depth: int = 8
    mlp_ratio: float = 2.0
    epochs: int = 100
    warmup_epochs: int = 5
    batch_size: int = 256
    workers: int = 12
    learning_rate: float = 5.0e-4
    minimum_learning_rate: float = 1.0e-6
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    random_erasing_probability: float = 0.25
    augmentation: bool = True
    gradient_clip_norm: float = 1.0
    finite_check_interval: int = 1
    fused_optimizer: bool = False
    validation_interval: int = 1
    seed: int = 20260730
    amp_dtype: AmpDtype = "bfloat16"
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    log_interval: int = 100


class UInt8ImageCache(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        root: Path,
        *,
        expected_identity: dict[str, Any],
        image_size: int,
    ) -> None:
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing uint8 cache metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != "lnet.imagenet100.uint8-cache.v1":
            raise ValueError("unsupported uint8 image cache schema")
        if metadata.get("training_identity") != expected_identity:
            raise ValueError("uint8 image cache dataset identity mismatch")
        if metadata.get("image_size") != image_size:
            raise ValueError("uint8 image cache size mismatch")
        self.root = root
        self.image_path = root / str(metadata["images"])
        self.label_path = root / str(metadata["labels"])
        self.sample_count = int(expected_identity["samples"])
        self.image_size = image_size
        self._images: Any | None = None
        self._labels: Any | None = None

    def __len__(self) -> int:
        return self.sample_count

    def _open(self) -> None:
        if self._images is None:
            self._images = np.load(self.image_path, mmap_mode="c")
            self._labels = np.load(self.label_path, mmap_mode="c")
            expected_shape = (
                self.sample_count,
                3,
                self.image_size,
                self.image_size,
            )
            if self._images.shape != expected_shape:
                raise ValueError("uint8 image cache tensor shape mismatch")
            if self._images.dtype != np.uint8:
                raise ValueError("uint8 image cache tensor dtype mismatch")
            if self._labels.shape != (self.sample_count,):
                raise ValueError("uint8 image cache label shape mismatch")

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        self._open()
        return torch.from_numpy(self._images[index]), int(self._labels[index])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_images"] = None
        state["_labels"] = None
        return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("t0", "t1_product", "t1_pole_free"),
        default="t1_product",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--random-erasing-probability", type=float, default=0.25)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help=(
            "Use deterministic resize/center-crop preprocessing and disable "
            "Mixup, CutMix, RandAugment, random crop/flip, and random erasing."
        ),
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--finite-check-interval", type=int, default=1)
    parser.add_argument("--fused-optimizer", action="store_true")
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16", "none"),
        default="bfloat16",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def _validate_config(config: TrainingConfig) -> None:
    positive_integer_fields = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "workers": config.workers,
        "log_interval": config.log_interval,
        "finite_check_interval": config.finite_check_interval,
        "validation_interval": config.validation_interval,
    }
    for name, value in positive_integer_fields.items():
        if value < 1 and not (name == "workers" and value == 0):
            raise ValueError(f"{name} must be positive")
    if config.warmup_epochs < 0 or config.warmup_epochs >= config.epochs:
        raise ValueError("warmup_epochs must be nonnegative and smaller than epochs")
    for name, value in (
        ("learning_rate", config.learning_rate),
        ("minimum_learning_rate", config.minimum_learning_rate),
        ("weight_decay", config.weight_decay),
        ("label_smoothing", config.label_smoothing),
        ("mixup_alpha", config.mixup_alpha),
        ("cutmix_alpha", config.cutmix_alpha),
        ("random_erasing_probability", config.random_erasing_probability),
        ("gradient_clip_norm", config.gradient_clip_norm),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if config.minimum_learning_rate > config.learning_rate:
        raise ValueError("minimum_learning_rate cannot exceed learning_rate")
    if config.label_smoothing >= 1.0:
        raise ValueError("label_smoothing must be smaller than one")
    if config.random_erasing_probability > 1.0:
        raise ValueError("random_erasing_probability cannot exceed one")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _json_normalized(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the exact list/scalar representation persisted by JSON."""
    return cast("dict[str, Any]", json.loads(json.dumps(payload)))


def _pin_data_worker(worker_id: int, environment_name: str) -> None:
    specification = os.environ.get(environment_name, "").strip()
    if not specification:
        return
    cpus = tuple(int(value) for value in specification.split(","))
    if not cpus or any(cpu < 0 for cpu in cpus):
        raise ValueError(f"{environment_name} must list nonnegative CPU ids")
    os.sched_setaffinity(0, {cpus[worker_id % len(cpus)]})


def _pin_training_worker(worker_id: int) -> None:
    _pin_data_worker(worker_id, "LNET_TRAINING_DATALOADER_CPU_AFFINITY")


def _pin_validation_worker(worker_id: int) -> None:
    _pin_data_worker(worker_id, "LNET_VALIDATION_DATALOADER_CPU_AFFINITY")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_tensor_image(path: str) -> Tensor:
    return decode_image(path, mode=ImageReadMode.RGB)


def _dataset_identity(
    dataset: datasets.ImageFolder,
    root: Path,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for filename, target in dataset.samples:
        path = Path(filename)
        relative = path.relative_to(root)
        digest.update(str(relative).encode())
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


def _transforms(
    image_size: int,
    random_erasing_probability: float,
    *,
    augmentation: bool,
    tensor_input: bool = False,
) -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    convert = (
        transforms.ConvertImageDtype(torch.float32)
        if tensor_input
        else transforms.ToTensor()
    )
    validation = transforms.Compose(
        [
            transforms.Resize(
                round(image_size / 0.875),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            convert,
            transforms.Normalize(mean, std),
        ]
    )
    if not augmentation:
        training = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                convert,
                transforms.Normalize(mean, std),
            ]
        )
        return training, validation
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(
                num_ops=2,
                magnitude=9,
                interpolation=InterpolationMode.BICUBIC,
            ),
            convert,
            transforms.Normalize(mean, std),
            transforms.RandomErasing(
                p=random_erasing_probability,
                value=cast("Any", "random"),
            ),
        ]
    )
    return training, validation


def build_datasets(
    config: TrainingConfig,
) -> tuple[Dataset[Any], datasets.ImageFolder, dict[str, Any]]:
    decoder = os.environ.get("LNET_IMAGE_DECODER", "pil").strip().lower()
    if decoder not in {"pil", "torchvision"}:
        raise ValueError("LNET_IMAGE_DECODER must be pil or torchvision")
    tensor_input = decoder == "torchvision"
    training_transform, validation_transform = _transforms(
        config.image_size,
        config.random_erasing_probability,
        augmentation=config.augmentation,
        tensor_input=tensor_input,
    )
    root = Path(config.data_root)
    training_root = root / "train"
    validation_root = root / "val"
    if not training_root.is_dir() or not validation_root.is_dir():
        raise FileNotFoundError("ImageNet100 requires train/ and val/ directories")
    loader = _decode_tensor_image if tensor_input else datasets.folder.default_loader
    training = datasets.ImageFolder(
        training_root,
        transform=training_transform,
        loader=loader,
    )
    validation = datasets.ImageFolder(
        validation_root,
        transform=validation_transform,
        loader=loader,
    )
    if training.class_to_idx != validation.class_to_idx:
        raise ValueError("training and validation class mappings differ")
    if len(training.classes) != 100:
        raise ValueError(f"ImageNet100 requires 100 classes, found {len(training.classes)}")
    identity = {
        "training": _dataset_identity(training, training_root),
        "validation": _dataset_identity(validation, validation_root),
    }
    cache_root = os.environ.get("LNET_TRAINING_UINT8_CACHE", "").strip()
    training_dataset: Dataset[Any] = training
    if cache_root:
        training_dataset = UInt8ImageCache(
            Path(cache_root),
            expected_identity=identity["training"],
            image_size=config.image_size,
        )
    return training_dataset, validation, identity


def _learning_rate(config: TrainingConfig, epoch: int) -> float:
    if epoch < config.warmup_epochs:
        return config.learning_rate * float(epoch + 1) / max(1, config.warmup_epochs)
    progress = (epoch - config.warmup_epochs) / max(
        1,
        config.epochs - config.warmup_epochs - 1,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def _set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _cutmix_box(
    height: int,
    width: int,
    fraction: float,
) -> tuple[int, int, int, int]:
    cut_ratio = math.sqrt(max(0.0, 1.0 - fraction))
    cut_height = round(height * cut_ratio)
    cut_width = round(width * cut_ratio)
    center_y = random.randrange(height)
    center_x = random.randrange(width)
    y0 = max(0, center_y - cut_height // 2)
    y1 = min(height, center_y + cut_height // 2)
    x0 = max(0, center_x - cut_width // 2)
    x1 = min(width, center_x + cut_width // 2)
    return y0, y1, x0, x1


def mix_batch(
    images: Tensor,
    targets: Tensor,
    *,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> tuple[Tensor, Tensor, Tensor, float, str]:
    if images.shape[0] < 2 or (mixup_alpha <= 0.0 and cutmix_alpha <= 0.0):
        return images, targets, targets, 1.0, "none"
    permutation = torch.randperm(images.shape[0], device=images.device)
    use_cutmix = cutmix_alpha > 0.0 and (
        mixup_alpha <= 0.0 or random.random() < 0.5
    )
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    fraction = random.betavariate(alpha, alpha)
    if not use_cutmix:
        mixed = images.mul(fraction).add(
            images[permutation],
            alpha=1.0 - fraction,
        )
        return mixed, targets, targets[permutation], fraction, "mixup"
    y0, y1, x0, x1 = _cutmix_box(images.shape[-2], images.shape[-1], fraction)
    mixed = images.clone()
    mixed[:, :, y0:y1, x0:x1] = images[permutation, :, y0:y1, x0:x1]
    retained = 1.0 - ((y1 - y0) * (x1 - x0)) / (
        images.shape[-2] * images.shape[-1]
    )
    return mixed, targets, targets[permutation], retained, "cutmix"


def _mixed_loss(
    logits: Tensor,
    first_targets: Tensor,
    second_targets: Tensor,
    fraction: float,
    label_smoothing: float,
) -> Tensor:
    first = functional.cross_entropy(
        logits,
        first_targets,
        label_smoothing=label_smoothing,
    )
    if fraction >= 1.0:
        return first
    second = functional.cross_entropy(
        logits,
        second_targets,
        label_smoothing=label_smoothing,
    )
    return fraction * first + (1.0 - fraction) * second


def _autocast_dtype(config: TrainingConfig) -> torch.dtype | None:
    if config.amp_dtype == "bfloat16":
        return torch.bfloat16
    if config.amp_dtype == "float16":
        return torch.float16
    return None


def _prepare_runtime_model(model: nn.Module) -> nn.Module:
    compile_mode = os.environ.get("LNET_TORCH_COMPILE_MODE", "").strip()
    if not compile_mode:
        return model
    return cast(
        "nn.Module",
        torch.compile(
            model,
            mode=compile_mode,
            fullgraph=False,
            dynamic=False,
        ),
    )


def _device_images(cpu_images: Tensor, device: torch.device) -> Tensor:
    images = cpu_images.to(
        device,
        non_blocking=True,
        memory_format=torch.channels_last,
    )
    if images.dtype != torch.uint8:
        return images
    mean = images.new_tensor(
        (0.485, 0.456, 0.406),
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    std = images.new_tensor(
        (0.229, 0.224, 0.225),
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    return images.to(dtype=torch.float32).mul_(1.0 / 255.0).sub_(mean).div_(std)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    config: TrainingConfig,
    *,
    epoch: int,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    amp_dtype = _autocast_dtype(config)
    total_loss = torch.zeros((), device=device)
    total_samples = 0
    mix_counts = {"mixup": 0, "cutmix": 0, "none": 0}
    started = time.perf_counter()
    step_index = 0
    stop_training = False
    for cpu_images, cpu_targets in loader:
        device_images = _device_images(cpu_images, device)
        targets = cpu_targets.to(device, non_blocking=True)
        for start in range(0, device_images.shape[0], config.batch_size):
            if (
                config.max_train_batches is not None
                and step_index >= config.max_train_batches
            ):
                stop_training = True
                break
            stop = start + config.batch_size
            if stop > device_images.shape[0]:
                break
            images, first_targets, second_targets, fraction, mix_kind = mix_batch(
                device_images[start:stop],
                targets[start:stop],
                mixup_alpha=config.mixup_alpha,
                cutmix_alpha=config.cutmix_alpha,
            )
            mix_counts[mix_kind] += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                logits = model(images)
                loss = _mixed_loss(
                    logits,
                    first_targets,
                    second_targets,
                    fraction,
                    config.label_smoothing,
                )
            check_finite = step_index % config.finite_check_interval == 0
            if check_finite and not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite training loss at batch {step_index}"
                )
            scaler.scale(loss).backward()
            if config.gradient_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )
                if check_finite and not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError(
                        f"non-finite gradient norm at batch {step_index}"
                    )
            scaler.step(optimizer)
            scaler.update()
            batch_samples = images.shape[0]
            total_samples += batch_samples
            total_loss.add_(loss.detach(), alpha=batch_samples)
            step_index += 1
            if step_index % config.log_interval == 0:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": epoch,
                            "batch": step_index,
                            "loss": float(total_loss / total_samples),
                            "images_per_second": total_samples / elapsed,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if stop_training:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if total_samples == 0:
        raise RuntimeError("training loader produced no samples")
    return {
        "loss": float(total_loss / total_samples),
        "samples": total_samples,
        "seconds": elapsed,
        "images_per_second": total_samples / elapsed,
        "mix_counts": mix_counts,
    }


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    config: TrainingConfig,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    amp_dtype = _autocast_dtype(config)
    total_loss = 0.0
    total_samples = 0
    top1 = 0
    top5 = 0
    started = time.perf_counter()
    for batch_index, (cpu_images, cpu_targets) in enumerate(loader):
        if (
            config.max_validation_batches is not None
            and batch_index >= config.max_validation_batches
        ):
            break
        images = _device_images(cpu_images, device)
        targets = cpu_targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = model(images)
            loss = functional.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite validation loss at batch {batch_index}")
        predictions = logits.topk(5, dim=1).indices
        batch_samples = images.shape[0]
        total_samples += batch_samples
        total_loss += float(loss) * batch_samples
        top1 += int(predictions[:, 0].eq(targets).sum())
        top5 += int(predictions.eq(targets[:, None]).any(dim=1).sum())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if total_samples == 0:
        raise RuntimeError("validation loader produced no samples")
    return {
        "loss": total_loss / total_samples,
        "top1": top1 / total_samples,
        "top5": top5 / total_samples,
        "samples": total_samples,
        "seconds": elapsed,
        "images_per_second": total_samples / elapsed,
    }


def _environment() -> dict[str, Any]:
    device = torch.cuda.current_device()
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
    }


def _model_config(
    architecture: Architecture,
) -> Alphabet2DConfig | Alphabet2DT1Config:
    if architecture == "t0":
        return Alphabet2DConfig(
            input_channels=3,
            output_dim=100,
            image_size=224,
            patch_size=16,
            model_dim=192,
            modes=16,
            depth=8,
            mlp_ratio=2.0,
            windows="global_2x2",
            fixed_direct_atlas=True,
            recurrence_backend="auto",
        )
    return Alphabet2DT1Config(
        transport="product" if architecture == "t1_product" else "pole_free",
    )


def _build_model(
    architecture: Architecture,
    model_config: Alphabet2DConfig | Alphabet2DT1Config,
) -> Alphabet2D | Alphabet2DT1Compact:
    if architecture == "t0":
        return Alphabet2D(cast("Alphabet2DConfig", model_config))
    return Alphabet2DT1Compact(cast("Alphabet2DT1Config", model_config))


def main() -> None:
    args = _parser().parse_args()
    config = TrainingConfig(
        data_root=str(args.data_root.resolve()),
        output_dir=str(args.output_dir.resolve()),
        architecture=cast("Architecture", args.architecture),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=0.0 if args.no_augmentation else args.mixup_alpha,
        cutmix_alpha=0.0 if args.no_augmentation else args.cutmix_alpha,
        random_erasing_probability=(
            0.0 if args.no_augmentation else args.random_erasing_probability
        ),
        augmentation=not args.no_augmentation,
        gradient_clip_norm=args.gradient_clip_norm,
        finite_check_interval=args.finite_check_interval,
        fused_optimizer=args.fused_optimizer,
        validation_interval=args.validation_interval,
        seed=args.seed,
        amp_dtype=cast("AmpDtype", args.amp_dtype),
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        log_interval=args.log_interval,
    )
    _validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet100 training requires CUDA")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dataset, validation_dataset, dataset_identity = build_datasets(config)
    script_path = Path(__file__).resolve()
    contract_source_path = Path(
        os.environ.get("LNET_TRAINING_CONTRACT_SOURCE", script_path)
    ).resolve()
    model_paths = (
        script_path.parents[1] / "src/lnet/alphabet2d.py",
        script_path.parents[1] / "src/lnet/alphabet2d_t1.py",
    )
    model_config = _model_config(config.architecture)
    contract = _json_normalized(
        {
            "schema": "lnet.alphabet2d.imagenet100.v2",
            "training_config": asdict(config),
            "model_config": asdict(model_config),
            "dataset": dataset_identity,
            "source_sha256": {
                "training_script": _sha256(contract_source_path),
                **{path.name: _sha256(path) for path in model_paths},
            },
            "environment": _environment(),
        }
    )
    contract_path = output_dir / "contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        for key in ("training_config", "model_config", "dataset", "source_sha256"):
            if existing[key] != contract[key]:
                raise RuntimeError(f"existing contract differs for {key}")
        contract = existing
    else:
        _atomic_json(contract_path, contract)

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    training_workers = int(
        os.environ.get("LNET_TRAINING_DATALOADER_WORKERS", config.workers)
    )
    validation_workers = int(
        os.environ.get("LNET_VALIDATION_DATALOADER_WORKERS", config.workers)
    )
    if training_workers < 0 or validation_workers < 0:
        raise ValueError("runtime DataLoader worker counts must be nonnegative")
    training_batch_multiplier = int(
        os.environ.get("LNET_TRAINING_LOADER_BATCH_MULTIPLIER", "1")
    )
    if training_batch_multiplier < 1:
        raise ValueError("LNET_TRAINING_LOADER_BATCH_MULTIPLIER must be positive")
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size * training_batch_multiplier,
        shuffle=True,
        num_workers=training_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=training_workers > 0,
        multiprocessing_context="spawn" if training_workers > 0 else None,
        worker_init_fn=_pin_training_worker if training_workers > 0 else None,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=validation_workers,
        pin_memory=True,
        persistent_workers=False,
        multiprocessing_context="spawn" if validation_workers > 0 else None,
        worker_init_fn=_pin_validation_worker if validation_workers > 0 else None,
    )
    device = torch.device("cuda")
    model = _build_model(config.architecture, model_config)
    if os.environ.get("LNET_CAPTURE_SAFE_ORTHOGONAL", "0") == "1":
        replaced = prepare_capture_safe_orthogonal_(model, compute_dtype=torch.float32)
        print(
            json.dumps(
                {
                    "event": "capture_safe_orthogonal_prepared",
                    "replaced": list(replaced),
                }
            ),
            flush=True,
        )
    model = model.to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=config.fused_optimizer,
    )
    scaler = torch.GradScaler(
        device.type,
        enabled=config.amp_dtype == "float16",
    )
    start_epoch = 0
    best_top1 = -math.inf
    history: list[dict[str, Any]] = []
    latest_path = output_dir / "latest.pt"
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_top1 = float(checkpoint["best_top1"])
        history = list(checkpoint["history"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        generator.set_state(checkpoint["data_loader_generator_state"])
        print(
            json.dumps(
                {"event": "resumed", "start_epoch": start_epoch, "best_top1": best_top1}
            ),
            flush=True,
        )
    runtime_model = _prepare_runtime_model(model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        json.dumps(
            {
                "event": "run_started",
                "parameters": parameters,
                "trainable_parameters": trainable_parameters,
                "train_samples": len(training_dataset),
                "validation_samples": len(validation_dataset),
                "start_epoch": start_epoch,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(start_epoch, config.epochs):
        learning_rate = _learning_rate(config, epoch)
        _set_learning_rate(optimizer, learning_rate)
        training_metrics = train_epoch(
            runtime_model,
            training_loader,
            optimizer,
            scaler,
            config,
            epoch=epoch,
            device=device,
        )
        should_validate = (
            epoch == 0
            or (epoch + 1) % config.validation_interval == 0
            or epoch + 1 == config.epochs
        )
        validation_metrics = (
            validate(
                runtime_model,
                validation_loader,
                config,
                device=device,
            )
            if should_validate
            else None
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "training": training_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        if (
            validation_metrics is not None
            and float(validation_metrics["top1"]) > best_top1
        ):
            improved = True
            best_top1 = float(validation_metrics["top1"])
        else:
            improved = False
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top1": best_top1,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
            "data_loader_generator_state": generator.get_state(),
        }
        _atomic_torch_save(latest_path, checkpoint)
        if improved:
            _atomic_torch_save(output_dir / "best.pt", checkpoint)
        validated_history = [
            item for item in history if item["validation"] is not None
        ]
        summary = {
            "status": "running" if epoch + 1 < config.epochs else "complete",
            "completed_epochs": epoch + 1,
            "best_top1": best_top1,
            "best_epoch": max(
                validated_history,
                key=lambda item: float(item["validation"]["top1"]),
            )["epoch"],
            "parameters": parameters,
            "trainable_parameters": trainable_parameters,
            "history": history,
        }
        _atomic_json(output_dir / "summary.json", summary)
        print(
            json.dumps({"event": "epoch_complete", **record, "best_top1": best_top1}),
            flush=True,
        )

    best_checkpoint = torch.load(
        output_dir / "best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint["model"])
    pole_audit = {
        bank: {
            coordinate: value.detach().cpu().tolist()
            for coordinate, value in coordinates.items()
        }
        for bank, coordinates in model.pole_audit().items()
    }
    _atomic_json(output_dir / "pole-audit.json", pole_audit)
    print(
        json.dumps(
            {
                "event": "run_complete",
                "best_top1": best_top1,
                "best_epoch": best_checkpoint["epoch"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
