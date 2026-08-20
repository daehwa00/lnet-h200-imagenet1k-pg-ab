"""Data loading, CPU placement, and device prefetch for ImageNet-100."""

from __future__ import annotations

# DataLoader batches and torchvision transforms expose dynamic tuple types.
# pyright: reportArgumentType=false, reportExplicitAny=false
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

if TYPE_CHECKING:
    from collections.abc import Iterator


CUDA_GRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})
PREFETCH_FACTOR = 2
WANDB_MODEL_ALIASES = {"product_four": "Product4", "pole_free": "PoleFree"}


def begin_cudagraph_step(device: torch.device) -> None:
    """Declare one model invocation boundary for compiled CUDA Graph runtimes."""
    if (
        device.type == "cuda"
        and os.environ.get("LNET_COMPILE_MODE") in CUDA_GRAPH_COMPILE_MODES
        and os.environ.get("LNET_CUDAGRAPHS_ACTIVE", "1") == "1"
    ):
        torch.compiler.cudagraph_mark_step_begin()


def active_loader_workers(requested: int) -> int:
    workers = int(os.environ.get("LNET_DATALOADER_WORKERS", requested))
    if workers < 0:
        message = "DataLoader workers must be nonnegative"
        raise ValueError(message)
    return workers


def persistent_loader_workers(active_workers: int) -> bool:
    return active_workers > 0 and os.environ.get("LNET_PERSISTENT_WORKERS", "1") == "1"


def parse_cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for field in value.split(","):
        bounds = field.strip().split("-", maxsplit=1)
        if not bounds[0]:
            message = f"invalid CPU affinity: {value}"
            raise ValueError(message)
        start = int(bounds[0])
        stop = int(bounds[-1])
        if min(start, stop) < 0 or stop < start:
            message = f"invalid CPU affinity: {value}"
            raise ValueError(message)
        cpus.update(range(start, stop + 1))
    if not cpus:
        message = "CPU affinity cannot be empty"
        raise ValueError(message)
    return cpus


def format_cpu_set(cpus: set[int]) -> str:
    ordered = sorted(cpus)
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def cpu_topology(cpus: set[int]) -> dict[tuple[int, int], set[int]]:
    topology: dict[tuple[int, int], set[int]] = defaultdict(set)
    root = Path("/sys/devices/system/cpu")
    for cpu in sorted(cpus):
        topology_root = root / f"cpu{cpu}" / "topology"
        package = int((topology_root / "physical_package_id").read_text().strip())
        core = int((topology_root / "core_id").read_text().strip())
        topology[(package, core)].add(cpu)
    return dict(topology)


def partition_physical_cores(
    topology: dict[tuple[int, int], set[int]],
    *,
    partition: int,
    partitions: int,
) -> set[int]:
    if partitions < 1 or partition not in range(partitions):
        message = "CPU partition index is outside the available GPU partitions"
        raise ValueError(message)
    cores = [siblings for _, siblings in sorted(topology.items())]
    quotient, remainder = divmod(len(cores), partitions)
    start = partition * quotient + min(partition, remainder)
    count = quotient + (partition < remainder)
    return set().union(*cores[start : start + count]) if count else set()


def configure_cpu_affinity() -> set[int]:
    """Give a single-GPU run a stable, disjoint physical-core partition."""
    allowed = set(os.sched_getaffinity(0))
    explicit = os.environ.get("LNET_CPU_AFFINITY")
    if explicit:
        selected = parse_cpu_set(explicit)
        if not selected <= allowed:
            message = "requested CPU affinity falls outside the launcher affinity"
            raise RuntimeError(message)
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_paths = sorted(Path("/proc/driver/nvidia/gpus").glob("*"))
        if not visible.isdecimal() or len(gpu_paths) < 2:
            selected = allowed
        else:
            online = set(range(os.cpu_count() or 1))
            if allowed != online:
                selected = allowed
            else:
                selected = partition_physical_cores(
                    cpu_topology(allowed),
                    partition=int(visible),
                    partitions=len(gpu_paths),
                )
    if not selected:
        message = "automatic GPU CPU partition is empty"
        raise RuntimeError(message)
    os.sched_setaffinity(0, selected)
    os.environ["LNET_CPU_AFFINITY_ACTIVE"] = format_cpu_set(selected)
    return selected


def compact_wandb_name(variant: str) -> str:
    """Return a short model-only label; provenance stays in W&B config."""
    if variant in WANDB_MODEL_ALIASES:
        return WANDB_MODEL_ALIASES[variant]
    name = variant
    for prefix in ("capacity_dual_", "alphabet2d_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    replacements = (
        ("double_precomplex_fc", "DoublePreFC"),
        ("precomplex_fc", "PreFC"),
        ("first_order", "FO"),
        ("cccn_shortcut", "CCCN"),
    )
    for source, target in replacements:
        name = name.replace(source, target)
    return name.replace("_", "-")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(
        path for split in ("train", "val") for path in (root / split).glob("*/*") if path.is_file()
    )
    train_count = 0
    for path in files:
        relative = path.relative_to(root)
        if relative.parts[0] == "train":
            train_count += 1
        digest.update(str(relative).encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest(), train_count, len(files) - train_count


def transforms_for_imagenet100() -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    standard_deviation = (0.229, 0.224, 0.225)
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
            transforms.RandomErasing(p=0.25),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    return train, evaluation


def loaders(
    data_root: Path,
    *,
    batch_size: int,
    workers: int,
    training_generator: torch.Generator,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_transform, evaluation_transform = transforms_for_imagenet100()
    train_dataset = datasets.ImageFolder(data_root / "train", train_transform)
    validation_dataset = datasets.ImageFolder(data_root / "val", evaluation_transform)
    if train_dataset.classes != validation_dataset.classes:
        message = "ImageNet-100 train and validation class sets differ"
        raise RuntimeError(message)
    active_workers = active_loader_workers(workers)
    persistent_workers = persistent_loader_workers(active_workers)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": active_workers,
        "pin_memory": True,
    }
    if active_workers > 0:
        common["prefetch_factor"] = PREFETCH_FACTOR
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=training_generator,
        drop_last=True,
        persistent_workers=persistent_workers,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        persistent_workers=False,
        **common,
    )
    return train_loader, validation_loader


def move_batch(
    batch_inputs: Tensor,
    batch_targets: Tensor,
    device: torch.device,
    *,
    channels_last: bool,
) -> tuple[Tensor, Tensor]:
    memory_format = torch.channels_last if channels_last else torch.preserve_format
    inputs = batch_inputs.to(
        device=device,
        non_blocking=True,
        memory_format=memory_format,
    )
    targets = batch_targets.to(device=device, non_blocking=True)
    return inputs, targets


def device_batches(
    loader: DataLoader[Any],
    device: torch.device,
    *,
    channels_last: bool,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Transfer batches in order while overlapping the next H2D copy on CUDA."""
    if device.type != "cuda":
        for batch_inputs, batch_targets in loader:
            yield move_batch(
                batch_inputs,
                batch_targets,
                device,
                channels_last=channels_last,
            )
        return

    prefetch_stream = torch.cuda.Stream(device=device)
    iterator = iter(loader)
    try:
        batch_inputs, batch_targets = next(iterator)
    except StopIteration:
        return
    with torch.cuda.stream(prefetch_stream):
        next_inputs, next_targets = move_batch(
            batch_inputs,
            batch_targets,
            device,
            channels_last=False,
        )

    while True:
        current_stream = torch.cuda.current_stream(device=device)
        current_stream.wait_stream(prefetch_stream)
        inputs, targets = next_inputs, next_targets
        inputs.record_stream(current_stream)
        targets.record_stream(current_stream)

        try:
            batch_inputs, batch_targets = next(iterator)
        except StopIteration:
            has_next = False
        else:
            has_next = True
            with torch.cuda.stream(prefetch_stream):
                next_inputs, next_targets = move_batch(
                    batch_inputs,
                    batch_targets,
                    device,
                    channels_last=False,
                )

        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        yield inputs, targets
        if not has_next:
            break


__all__ = [
    "PREFETCH_FACTOR",
    "active_loader_workers",
    "begin_cudagraph_step",
    "compact_wandb_name",
    "configure_cpu_affinity",
    "dataset_digest",
    "device_batches",
    "digest",
    "loaders",
    "persistent_loader_workers",
    "transforms_for_imagenet100",
]
