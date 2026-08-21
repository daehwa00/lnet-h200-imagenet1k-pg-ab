# pyright: reportExplicitAny=false, reportMissingImports=false
"""Train matched four-scan and pole-free Nano models on ImageNet-100."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from lnet.alphabet2d_cifar import ImageNetNanoConfig, build_imagenet_nano

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sized

    from wandb.sdk.wandb_run import Run as WandbRun

VARIANTS = ("product_four", "pole_free")
SEEDS = (501, 509, 521)
_CUDA_GRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})


@dataclass(frozen=True, slots=True)
class RunnerBindings:
    """Explicit experiment callbacks consumed by the shared training harness."""

    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    model_config: Callable[..., Any]
    build_model: Callable[..., nn.Module]
    contract: Callable[[argparse.Namespace], dict[str, Any]]
    build_optimizer: Callable[..., torch.optim.Optimizer]
    prepare_model: Callable[[nn.Module, dict[str, Any]], nn.Module]
    train_epoch: Callable[..., dict[str, float]]
    evaluate: Callable[..., dict[str, float]]
    wandb_model_metrics: Callable[[nn.Module], dict[str, float]]
    summarize: Callable[[Path, dict[str, Any]], dict[str, Any] | None]


def _begin_cudagraph_step(device: torch.device) -> None:
    """Declare one model invocation boundary for compiled CUDA Graph runtimes."""
    if (
        device.type == "cuda"
        and os.environ.get("LNET_COMPILE_MODE") in _CUDA_GRAPH_COMPILE_MODES
        and os.environ.get("LNET_CUDAGRAPHS_ACTIVE", "1") == "1"
    ):
        torch.compiler.cudagraph_mark_step_begin()


PREFETCH_FACTOR = 2
WANDB_MODEL_ALIASES = {
    "product_four": "Product4",
    "pole_free": "PoleFree",
}


def _active_loader_workers(requested: int) -> int:
    workers = int(os.environ.get("LNET_DATALOADER_WORKERS", requested))
    if workers < 0:
        message = "DataLoader workers must be nonnegative"
        raise ValueError(message)
    return workers


def _active_loader_prefetch_factor() -> int:
    factor = int(os.environ.get("LNET_DATALOADER_PREFETCH_FACTOR", PREFETCH_FACTOR))
    if not 1 <= factor <= 8:
        message = "DataLoader prefetch factor must be between 1 and 8"
        raise ValueError(message)
    return factor


def _persistent_loader_workers(active_workers: int) -> bool:
    """Keep worker lifetimes at epoch boundaries so loader RNG can be restored.

    A persistent worker owns Python and Torch RNG states that are not represented
    in an epoch checkpoint.  Reusing that worker after an uninterrupted epoch and
    recreating it after a restart therefore produce different augmentations.  The
    confirmatory harness deliberately rejects that configuration instead of
    claiming an exact resume it cannot provide.
    """
    requested = os.environ.get("LNET_PERSISTENT_WORKERS", "0") == "1"
    if active_workers > 0 and requested:
        message = (
            "persistent DataLoader workers are incompatible with RNG-continuous "
            "epoch-boundary resume; set LNET_PERSISTENT_WORKERS=0"
        )
        raise RuntimeError(message)
    return False


def _parse_cpu_set(value: str) -> set[int]:
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


def _format_cpu_set(cpus: set[int]) -> str:
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


def _cpu_topology(cpus: set[int]) -> dict[tuple[int, int], set[int]]:
    topology: dict[tuple[int, int], set[int]] = defaultdict(set)
    root = Path("/sys/devices/system/cpu")
    for cpu in sorted(cpus):
        topology_root = root / f"cpu{cpu}" / "topology"
        package = int((topology_root / "physical_package_id").read_text().strip())
        core = int((topology_root / "core_id").read_text().strip())
        topology[(package, core)].add(cpu)
    return dict(topology)


def _partition_physical_cores(
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


def _configure_cpu_affinity() -> set[int]:
    """Give a single-GPU run a stable, disjoint physical-core partition."""
    allowed = set(os.sched_getaffinity(0))
    explicit = os.environ.get("LNET_CPU_AFFINITY")
    if explicit:
        selected = _parse_cpu_set(explicit)
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
                selected = _partition_physical_cores(
                    _cpu_topology(allowed),
                    partition=int(visible),
                    partitions=len(gpu_paths),
                )
    if not selected:
        message = "automatic GPU CPU partition is empty"
        raise RuntimeError(message)
    os.sched_setaffinity(0, selected)
    os.environ["LNET_CPU_AFFINITY_ACTIVE"] = _format_cpu_set(selected)
    return selected


def _compact_wandb_name(variant: str) -> str:
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


def _parse_args(
    *,
    variants: tuple[str, ...] = VARIANTS,
    seeds: tuple[int, ...] = SEEDS,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--variants", choices=variants, nargs="+", default=list(variants))
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(seeds))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_digest(root: Path) -> tuple[str, int, int]:
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


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _sync_directory(path: Path) -> None:
    """Persist a preceding rename on filesystems that support directory fsync."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _contract_sha256(contract: dict[str, Any]) -> str:
    canonical = json.dumps(contract, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.gradient_accumulation_steps < 1:
        message = "gradient_accumulation_steps must be positive"
        raise ValueError(message)
    model = ImageNetNanoConfig()
    loader_workers = _active_loader_workers(args.workers)
    loader_persistent_workers = _persistent_loader_workers(loader_workers)
    data_digest, train_count, validation_count = _dataset_digest(args.data_root)
    return {
        "schema": "lnet.alphabet2d.imagenet100_nano.v2",
        "evidence_status": "confirmatory G4 internal control",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(model),
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "optimizer": "AdamW",
            "learning_rate": 3.0e-3,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "precision": args.precision,
            "loader_workers": loader_workers,
            "loader_persistent_workers": loader_persistent_workers,
            "validation_loader_persistent_workers": False,
            "loader_prefetch_factor": _active_loader_prefetch_factor(),
            "cpu_affinity": os.environ.get("LNET_CPU_AFFINITY_ACTIVE"),
            "device_prefetch_stream": True,
            "device_prefetch_scope": "copy_only",
            "fused_h2d_channels_last": True,
            "augmentation": ("RandomResizedCrop(224,bicubic)+HFlip+RandAugment(2,9)+RandomErasing"),
            "selection": "fixed final epoch; validation is not used for selection",
            "resume": (
                "epoch-boundary continuity for sampler, augmentation-worker, mixup, "
                "process, and CUDA RNG; persistent workers forbidden; bitwise CUDA "
                "kernel determinism is not claimed"
            ),
        },
        "data": {
            "manifest_sha256": data_digest,
            "train_images": train_count,
            "validation_images": validation_count,
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "source_sha256": {
            "runner": _digest(Path(__file__)),
            "models": _digest(Path("src/lnet/alphabet2d_cifar.py")),
            "alphabet2d": _digest(Path("src/lnet/alphabet2d.py")),
        },
    }


def _initialize(root: Path, contract: dict[str, Any]) -> None:
    path = root / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            message = "existing ImageNet Nano root has a different immutable contract"
            raise RuntimeError(message)
    else:
        artifact_roots = ("checkpoints", "results", "telemetry")
        stale_artifacts = [
            candidate
            for name in artifact_roots
            for candidate in (root / name).glob("*")
            if candidate.is_file()
        ]
        if stale_artifacts or (root / "summary.json").exists():
            message = "experiment artifacts exist without their immutable contract"
            raise RuntimeError(message)
        _atomic_json(path, contract)


def _transforms() -> tuple[transforms.Compose, transforms.Compose]:
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


def _loaders(
    data_root: Path,
    *,
    batch_size: int,
    workers: int,
    training_generator: torch.Generator,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_transform, evaluation_transform = _transforms()
    train_dataset = datasets.ImageFolder(data_root / "train", train_transform)
    validation_dataset = datasets.ImageFolder(data_root / "val", evaluation_transform)
    if train_dataset.classes != validation_dataset.classes:
        message = "ImageNet-100 train and validation class sets differ"
        raise RuntimeError(message)
    active_workers = _active_loader_workers(workers)
    persistent_workers = _persistent_loader_workers(active_workers)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": active_workers,
        "pin_memory": True,
    }
    if active_workers > 0:
        common["prefetch_factor"] = _active_loader_prefetch_factor()
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


def _move_batch(
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


def _device_batches(
    loader: DataLoader[Any],
    device: torch.device,
    *,
    channels_last: bool,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Transfer batches in order while overlapping the next H2D copy on CUDA."""
    if device.type != "cuda":
        for batch_inputs, batch_targets in loader:
            yield _move_batch(
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
        next_inputs, next_targets = _move_batch(
            batch_inputs,
            batch_targets,
            device,
            channels_last=channels_last,
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
                next_inputs, next_targets = _move_batch(
                    batch_inputs,
                    batch_targets,
                    device,
                    channels_last=channels_last,
                )

        yield inputs, targets
        if not has_next:
            break


def _evaluate(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    model.eval()
    runtime.eval()
    correct_terms: list[Tensor] = []
    count = 0
    loss_terms: list[Tensor] = []
    with torch.inference_mode():
        for inputs, targets in _device_batches(
            loader,
            device,
            channels_last=channels_last,
        ):
            _begin_cudagraph_step(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bfloat16",
            ):
                logits = runtime(inputs)
            loss_terms.append(functional.cross_entropy(logits, targets, reduction="sum"))
            correct_terms.append((logits.argmax(dim=-1) == targets).sum())
            count += targets.numel()
    loss_sum = float(torch.stack(loss_terms).double().sum())
    correct = int(torch.stack(correct_terms).sum())
    return {"accuracy": correct / count, "cross_entropy": loss_sum / count}


def _learning_rate_factor(epoch: int, epochs: int) -> float:
    if epoch < 5:
        return (epoch + 1) / 5.0
    progress = (epoch - 5) / max(1, epochs - 5)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _train_epoch(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mixup_generator: np.random.Generator,
    mixup_alpha: float,
    precision: str,
    gradient_accumulation_steps: int = 1,
    channels_last: bool = False,
) -> dict[str, float]:
    if gradient_accumulation_steps < 1:
        message = "gradient accumulation steps must be positive"
        raise ValueError(message)
    model.train()
    runtime.train()
    correct_terms: list[Tensor] = []
    count = 0
    loss_terms: list[Tensor] = []
    batch_count = len(loader)
    batches = _device_batches(loader, device, channels_last=channels_last)
    for batch_index, (inputs, targets) in enumerate(batches):
        group_offset = batch_index % gradient_accumulation_steps
        if group_offset == 0:
            optimizer.zero_grad(set_to_none=True)
        group_size = min(
            gradient_accumulation_steps,
            batch_count - (batch_index - group_offset),
        )
        permutation = torch.randperm(targets.numel(), device=device)
        mixing = float(mixup_generator.beta(mixup_alpha, mixup_alpha))
        mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
        _begin_cudagraph_step(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            logits = runtime(mixed_inputs)
            loss = mixing * functional.cross_entropy(
                logits,
                targets,
                label_smoothing=0.1,
            ) + (1.0 - mixing) * functional.cross_entropy(
                logits,
                targets[permutation],
                label_smoothing=0.1,
            )
        (loss / group_size).backward()
        group_complete = group_offset + 1 == group_size or batch_index + 1 == batch_count
        if group_complete:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        loss_terms.append(loss.detach() * targets.numel())
        correct_terms.append((logits.argmax(dim=-1) == targets).sum())
        count += targets.numel()
    loss_sum = float(torch.stack(loss_terms).double().sum())
    correct = int(torch.stack(correct_terms).sum())
    return {"loss": loss_sum / count, "mixed_accuracy": correct / count}


def _restore_checkpoint(
    path: Path,
    *,
    variant: str,
    seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_generator: torch.Generator,
    mixup_generator: np.random.Generator,
    contract_sha256: str | None = None,
    progress: dict[str, int] | None = None,
    optimizer_steps_per_epoch: int | None = None,
) -> tuple[int, list[dict[str, float]], float]:
    if not path.exists():
        if progress is not None:
            progress["global_step"] = 0
        return 0, [], 0.0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["variant"] != variant or payload["seed"] != seed:
        message = "checkpoint identity does not match requested ImageNet Nano job"
        raise RuntimeError(message)
    if contract_sha256 is not None and payload.get("contract_sha256") != contract_sha256:
        message = "checkpoint is not bound to the active immutable contract"
        raise RuntimeError(message)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    required_rng = {
        "training_generator_state",
        "torch_rng_state",
        "cuda_rng_state",
        "python_rng_state",
        "mixup_rng_state",
    }
    missing_rng = required_rng.difference(payload)
    if missing_rng:
        message = (
            "checkpoint predates exact-resume RNG capture; restart this "
            f"confirmatory job from epoch zero (missing {sorted(missing_rng)})"
        )
        raise RuntimeError(message)
    training_generator.set_state(payload["training_generator_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    random.setstate(payload["python_rng_state"])
    mixup_generator.bit_generator.state = payload["mixup_rng_state"]
    epoch = _checkpoint_nonnegative_integer(payload["epoch"], name="epoch")
    if progress is not None:
        stored_global_step = payload.get("global_step")
        if stored_global_step is None:
            steps_per_epoch = _checkpoint_positive_integer(
                optimizer_steps_per_epoch,
                name="optimizer_steps_per_epoch",
            )
            global_step = epoch * steps_per_epoch
        else:
            global_step = _checkpoint_nonnegative_integer(
                stored_global_step,
                name="global_step",
            )
        progress["global_step"] = global_step
    return epoch, payload["history"], float(payload["training_seconds"])


def _checkpoint_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = f"checkpoint {name} must be a nonnegative integer"
        raise RuntimeError(message)
    return value


def _checkpoint_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"checkpoint {name} must be a positive integer"
        raise RuntimeError(message)
    return value


def _optimizer_steps_per_epoch(
    loader: DataLoader[Any],
    gradient_accumulation_steps: int,
) -> int:
    if gradient_accumulation_steps < 1:
        message = "gradient accumulation steps must be positive"
        raise ValueError(message)
    batch_count = len(loader)
    if batch_count < 1:
        message = "training loader must contain at least one batch"
        raise RuntimeError(message)
    return math.ceil(batch_count / gradient_accumulation_steps)


def _train_epoch_with_step_count(
    train_epoch: Callable[..., dict[str, float]],
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mixup_generator: np.random.Generator,
    mixup_alpha: float,
    precision: str,
    gradient_accumulation_steps: int,
    channels_last: bool,
) -> tuple[dict[str, float], int]:
    """Run one epoch and count successful optimizer updates exactly."""
    original_step = optimizer.step
    optimizer_steps = 0

    def step_and_count(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        nonlocal optimizer_steps
        result = original_step(*args, **kwargs)
        optimizer_steps += 1
        return result

    optimizer.step = step_and_count  # type: ignore[method-assign]
    try:
        metrics = train_epoch(
            model,
            runtime,
            loader,
            optimizer,
            device=device,
            mixup_generator=mixup_generator,
            mixup_alpha=mixup_alpha,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            channels_last=channels_last,
        )
    finally:
        optimizer.step = original_step  # type: ignore[method-assign]
    return metrics, optimizer_steps


def _build_optimizer(
    model: nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
    )


def _restore_optimizer_runtime_options(
    optimizer: torch.optim.Optimizer,
    recipe: dict[str, Any],
) -> None:
    """Keep implementation-only optimizer flags out of checkpoint semantics."""
    if not bool(recipe.get("fused_optimizer", False)):
        return
    # ``Optimizer.load_state_dict`` preserves the archived tensor strides.
    # When a resumed model is converted to channels-last before loading the
    # checkpoint, convolution parameters and their Adam moments can therefore
    # have different memory formats.  Fused AdamW requires matching layouts.
    for parameter, state in optimizer.state.items():
        for name, value in tuple(state.items()):
            if not isinstance(value, Tensor) or value.shape != parameter.shape:
                continue
            if (
                value.dtype == parameter.dtype
                and value.device == parameter.device
                and value.layout == parameter.layout
                and value.stride() == parameter.stride()
            ):
                continue
            aligned = torch.empty_like(parameter, memory_format=torch.preserve_format)
            aligned.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            state[name] = aligned
    for group in optimizer.param_groups:
        group["fused"] = True
        group["foreach"] = None


def _prepare_model(model: nn.Module, recipe: dict[str, Any]) -> nn.Module:
    if bool(recipe.get("channels_last", False)):
        model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    return model


def _exclude_capture_hostile_submodules(
    model: nn.Module,
    *,
    exclude_classifier: bool = False,
) -> list[str]:
    """Keep parametrized submodules out of the compiled region.

    ``torch.nn.utils.parametrizations.orthogonal`` rebuilds its weight through
    ``matrix_exp``/``linalg.solve``/Householder on every access.  Those are
    cuSOLVER calls that synchronize with the host and allocate workspace, which
    CUDA stream capture forbids -- ``reduce-overhead`` aborts with
    ``operation not permitted when stream is capturing``.  Running them eagerly
    costs nothing measurable and lets everything else be captured.
    """
    excluded: list[str] = []
    for name, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        module.forward = torch.compiler.disable(module.forward)
        excluded.append(name)
    classifier = getattr(model, "classifier", None)
    if exclude_classifier and isinstance(classifier, nn.Module):
        # A graph break immediately before the head can otherwise create a
        # separately captured classifier graph whose output storage is reused
        # before autograd consumes it.  The head is tiny relative to the pole
        # backbone, so keep it eager while retaining backbone graph replay.
        classifier.forward = torch.compiler.disable(classifier.forward)
        excluded.append("classifier")
    return excluded


def _has_replayed_cudagraph_stages(model: nn.Module) -> bool:
    if bool(getattr(model, "_lnet_requires_classic_cudagraph", False)):
        return True
    stage_count = sum(
        callable(getattr(module, "_damping_fields", None)) for module in model.modules()
    )
    return stage_count > 1


def _replayed_stage_compile_options(compile_mode: str) -> dict[str, Any]:
    """Preserve mode tuning while using classic CUDA Graph replay."""
    options = {
        "triton.cudagraphs": True,
        "triton.cudagraph_trees": False,
    }
    if compile_mode == "max-autotune":
        options.update(
            {
                "coordinate_descent_tuning": True,
                "max_autotune": True,
            }
        )
    return options


def _build_runtime(model: nn.Module, recipe: dict[str, Any]) -> nn.Module:
    # The env override lets a measurement select a compile mode without editing
    # the recipe contract.  Unset, behaviour is exactly the recipe's.
    compile_mode = os.environ.get("LNET_COMPILE_MODE") or recipe.get("compile_mode")
    if compile_mode is None:
        return model
    requested_cudagraphs = str(compile_mode) in _CUDA_GRAPH_COMPILE_MODES
    repeated_stages = requested_cudagraphs and _has_replayed_cudagraph_stages(model)
    uses_cudagraphs = requested_cudagraphs
    os.environ["LNET_CUDAGRAPHS_ACTIVE"] = "1" if uses_cudagraphs else "0"
    _exclude_capture_hostile_submodules(
        model,
        exclude_classifier=uses_cudagraphs,
    )
    if repeated_stages:
        # CUDA Graph Trees split the large pole graph around opaque custom-op
        # launches and then reuse one memory pool for live saved tensors. A
        # classic graph keeps the same Inductor/Triton kernels and fixed-shape
        # replay without cross-segment Tree storage reuse.
        return cast(
            "nn.Module",
            torch.compile(
                model,
                options=_replayed_stage_compile_options(str(compile_mode)),
                fullgraph=False,
                dynamic=False,
            ),
        )
    return cast(
        "nn.Module",
        torch.compile(
            model,
            mode=str(compile_mode),
            fullgraph=False,
            dynamic=False,
        ),
    )


def _configure_compile_runtime(_root: Path, recipe: dict[str, Any]) -> str | None:
    """Publish the actual compile contract for launch-kernel tuning."""
    compile_mode = os.environ.get("LNET_COMPILE_MODE") or recipe.get("compile_mode")
    if compile_mode is None:
        return None
    profile = json.dumps(
        {
            "dynamic": False,
            "fullgraph": False,
            "mode": str(compile_mode),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    os.environ["LNET_COMPILE_PROFILE"] = profile
    # Launch records are already guarded by device, dtype, shape, compiler,
    # and kernel revision.  Keep them machine-local rather than run-local so a
    # measured winner is reused by later experiment roots automatically.
    os.environ.setdefault("LNET_LAUNCH_CACHE", str(Path.home() / ".cache" / "lnet"))
    return profile


def _initialize_wandb_run(
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    parameters: int,
) -> WandbRun | None:
    """Create a deterministic, resume-safe W&B run when tracking is enabled."""
    project = os.environ.get("WANDB_PROJECT")
    if not project or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb  # noqa: PLC0415
    except ModuleNotFoundError as error:
        message = "WANDB_PROJECT is set but the wandb package is not installed"
        raise RuntimeError(message) from error
    run_key = f"{root.resolve()}::{variant}::seed{seed}"
    run_id = os.environ.get("WANDB_RUN_ID") or hashlib.sha256(run_key.encode()).hexdigest()[:16]
    tracking_root = root / "wandb"
    tracking_root.mkdir(parents=True, exist_ok=True)
    variant_config = contract.get("variant_configs", {}).get(variant)
    return wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        group=os.environ.get("WANDB_GROUP", root.name),
        name=os.environ.get("WANDB_NAME", _compact_wandb_name(variant)),
        id=run_id,
        resume="allow",
        dir=str(tracking_root),
        settings=wandb.Settings(
            init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "300")),
        ),
        config={
            "variant": variant,
            "seed": seed,
            "parameters": parameters,
            # The builder applies the variant transform to the model template.
            # Log the resulting architecture as `model` so W&B columns describe
            # the model that is actually trained, not its constructor template.
            "model": variant_config or contract["model"],
            "model_template": contract["model"],
            "variant_config": variant_config,
            "recipe": contract["recipe"],
            "schema": contract["schema"],
        },
    )


def _wandb_model_metrics(_model: nn.Module) -> dict[str, float]:
    """Return optional architecture-specific scalar diagnostics."""
    return {}


def _telemetry_spool_path(root: Path, *, variant: str, seed: int) -> Path:
    return root / "telemetry" / f"{variant}__seed{seed}.jsonl"


def _report_telemetry_degraded(operation: str, error: BaseException) -> None:
    payload = {
        "error_type": type(error).__name__,
        "operation": operation,
        "training_continues": True,
    }
    print(  # noqa: T201
        "H200_TELEMETRY_DEGRADED_JSON="
        + json.dumps(payload, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def _read_telemetry_spool(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        _report_telemetry_degraded("read", error)
        return []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            _report_telemetry_degraded(f"repair-line-{line_number}", error)
            break
        if not isinstance(record, dict) or record.get("kind") not in {"epoch", "final"}:
            error = RuntimeError(f"invalid telemetry record at line {line_number}")
            _report_telemetry_degraded(f"repair-line-{line_number}", error)
            break
        step = record.get("step")
        if record["kind"] == "epoch" and (
            isinstance(step, bool) or not isinstance(step, int) or step < 1
        ):
            error = RuntimeError(f"invalid epoch telemetry step at line {line_number}")
            _report_telemetry_degraded(f"repair-line-{line_number}", error)
            break
        records.append(record)
    return records


def _telemetry_record_key(record: dict[str, Any]) -> tuple[str, int | None]:
    kind = str(record["kind"])
    if kind == "epoch":
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            message = "epoch telemetry step must be a positive integer"
            raise RuntimeError(message)
        return kind, step
    return kind, None


def _append_telemetry_record(path: Path, record: dict[str, Any]) -> bool:
    """Durably append one idempotent telemetry record.

    Checkpoints are the source of truth.  The append-only spool is retained even
    after a successful upload so a later invocation can reconcile a W&B run whose
    asynchronous client failed after accepting ``log`` locally.
    """
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        key = _telemetry_record_key(record)
        records = _read_telemetry_spool(path)
        for existing in records:
            if _telemetry_record_key(existing) != key:
                continue
            if existing != record:
                message = f"telemetry record {key} conflicts with durable local history"
                _report_telemetry_degraded("conflict", RuntimeError(message))
                return False
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w") as stream:
            for existing in (*records, record):
                stream.write(json.dumps(existing, separators=(",", ":"), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _sync_directory(path.parent)
    except Exception as error:  # noqa: BLE001  # Telemetry cannot stop training.
        _report_telemetry_degraded("append", error)
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        return False
    return True


def _report_wandb_degraded(operation: str, error: BaseException) -> None:
    """Report a safe failure class without echoing URLs, headers, or credentials."""
    payload = {
        "error_type": type(error).__name__,
        "operation": operation,
        "training_continues": True,
    }
    print(  # noqa: T201
        "H200_WANDB_DEGRADED_JSON=" + json.dumps(payload, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def _best_effort_initialize_wandb(
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    parameters: int,
) -> WandbRun | None:
    try:
        return _initialize_wandb_run(
            root,
            contract,
            variant=variant,
            seed=seed,
            parameters=parameters,
        )
    except Exception as error:  # noqa: BLE001  # W&B is outside the critical path.
        _report_wandb_degraded("init", error)
        return None


def _sync_telemetry_spool(
    wandb_run: WandbRun,
    path: Path,
    delivered: set[tuple[str, int | None]],
) -> bool:
    """Replay records once per client session; keep the durable spool forever."""
    for record in _read_telemetry_spool(path):
        key = _telemetry_record_key(record)
        if key in delivered:
            continue
        try:
            if record["kind"] == "epoch":
                wandb_run.log(record["metrics"], step=record["step"])
            else:
                for name, value in record["summary"].items():
                    wandb_run.summary[name] = value
        except Exception as error:  # noqa: BLE001  # W&B is outside the critical path.
            _report_wandb_degraded(f"sync-{record['kind']}", error)
            return False
        delivered.add(key)
    return True


def _best_effort_finish_wandb(wandb_run: WandbRun) -> None:
    try:
        wandb_run.finish()
    except Exception as error:  # noqa: BLE001  # W&B is outside the critical path.
        _report_wandb_degraded("finish", error)


def _epoch_telemetry_metrics(row: dict[str, float]) -> dict[str, float]:
    metrics = {
        "epoch": row["epoch"],
        "learning_rate": row["learning_rate"],
        "train/loss": row["train_loss"],
        "train/mixed_accuracy": row["train_mixed_accuracy"],
        "validation/accuracy": row["validation_accuracy"],
        "validation/cross_entropy": row["validation_cross_entropy"],
    }
    optional = {
        "time/training_seconds": "training_seconds",
        "time/host_input_wait_seconds": "host_input_wait_seconds",
        "global_step": "global_step",
        "optimizer_steps/epoch": "optimizer_steps",
    }
    for metric_name, row_name in optional.items():
        if row_name in row:
            metrics[metric_name] = row[row_name]
    return metrics


def _final_telemetry_summary(result: dict[str, Any]) -> dict[str, float]:
    history = result["history"]
    return {
        "final_validation_accuracy": float(result["final_validation"]["accuracy"]),
        "final_validation_cross_entropy": float(result["final_validation"]["cross_entropy"]),
        "best_validation_accuracy": max(float(row["validation_accuracy"]) for row in history),
        "training_seconds": float(result["training_seconds"]),
    }


def _ensure_completed_telemetry_spool(path: Path, result: dict[str, Any]) -> None:
    """Backfill only missing keys without replacing richer retained telemetry."""
    existing = {_telemetry_record_key(record) for record in _read_telemetry_spool(path)}
    for row in result["history"]:
        step = int(row["epoch"])
        if ("epoch", step) in existing:
            continue
        _append_telemetry_record(
            path,
            {
                "kind": "epoch",
                "metrics": _epoch_telemetry_metrics(row),
                "step": step,
            },
        )
        existing.add(("epoch", step))
    if ("final", None) not in existing:
        _append_telemetry_record(
            path,
            {"kind": "final", "summary": _final_telemetry_summary(result)},
        )


def _backfill_checkpoint_telemetry(path: Path, history: list[dict[str, float]]) -> None:
    """Repair a crash window where checkpoint rename preceded spool append."""
    existing = {_telemetry_record_key(record) for record in _read_telemetry_spool(path)}
    for row in history:
        step = int(row["epoch"])
        if ("epoch", step) in existing:
            continue
        _append_telemetry_record(
            path,
            {
                "kind": "epoch",
                "metrics": _epoch_telemetry_metrics(row),
                "step": step,
            },
        )
        existing.add(("epoch", step))


def _wandb_retry_due(epoch: int) -> bool:
    """Bound a failed relay's cost to epoch 1 and ten-epoch intervals."""
    return epoch == 1 or epoch % 10 == 0


def _sync_or_abandon_wandb(
    wandb_run: WandbRun,
    path: Path,
    delivered: set[tuple[str, int | None]],
) -> WandbRun | None:
    if _sync_telemetry_spool(wandb_run, path, delivered):
        return wandb_run
    _best_effort_finish_wandb(wandb_run)
    delivered.clear()
    return None


def _reconcile_completed_result(
    root: Path,
    contract: dict[str, Any],
    result: dict[str, Any],
    *,
    variant: str,
    seed: int,
) -> None:
    spool_path = _telemetry_spool_path(root, variant=variant, seed=seed)
    _ensure_completed_telemetry_spool(spool_path, result)
    wandb_run = _best_effort_initialize_wandb(
        root,
        contract,
        variant=variant,
        seed=seed,
        parameters=int(result["parameters"]),
    )
    if wandb_run is None:
        return
    _sync_telemetry_spool(wandb_run, spool_path, set())
    _best_effort_finish_wandb(wandb_run)


def _run_job(  # noqa: C901, PLR0915
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    data_root: Path,
    workers: int,
    device: torch.device,
    bindings: RunnerBindings,
) -> None:
    contract_sha256 = _contract_sha256(contract)
    result_path = root / "results" / f"{variant}__seed{seed}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (
            result.get("variant") != variant
            or result.get("seed") != seed
            or result.get("contract_sha256") != contract_sha256
        ):
            message = "completed result is not bound to the active immutable contract"
            raise RuntimeError(message)
        _reconcile_completed_result(
            root,
            contract,
            result,
            variant=variant,
            seed=seed,
        )
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = bindings.build_model(
        variant,
        bindings.model_config(**contract["model"]),
    ).to(device)
    recipe = contract["recipe"]
    model = bindings.prepare_model(model, recipe)
    training_generator = torch.Generator().manual_seed(seed)
    mixup_generator = np.random.default_rng(seed)
    train_loader, validation_loader = _loaders(
        data_root,
        batch_size=recipe["batch_size"],
        workers=workers,
        training_generator=training_generator,
    )
    optimizer = bindings.build_optimizer(model, recipe)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _learning_rate_factor(epoch, recipe["epochs"]),
    )
    gradient_accumulation_steps = int(recipe.get("gradient_accumulation_steps", 1))
    steps_per_epoch = _optimizer_steps_per_epoch(
        train_loader,
        gradient_accumulation_steps,
    )
    checkpoint_path = root / "checkpoints" / f"{variant}__seed{seed}.pt"
    progress = {"global_step": 0}
    start_epoch, history, training_seconds = _restore_checkpoint(
        checkpoint_path,
        variant=variant,
        seed=seed,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_generator=training_generator,
        mixup_generator=mixup_generator,
        contract_sha256=contract_sha256,
        progress=progress,
        optimizer_steps_per_epoch=steps_per_epoch,
    )
    global_step = progress["global_step"]
    _restore_optimizer_runtime_options(optimizer, recipe)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    spool_path = _telemetry_spool_path(root, variant=variant, seed=seed)
    wandb_run = None
    delivered_telemetry: set[tuple[str, int | None]] = set()
    # A fresh job performs no W&B network work until epoch 1 is durably local.
    # A resumed job already has an authoritative checkpoint, so it can replay
    # its retained telemetry while the runtime is being rebuilt.
    if start_epoch > 0:
        _backfill_checkpoint_telemetry(spool_path, history)
        wandb_run = _best_effort_initialize_wandb(
            root,
            contract,
            variant=variant,
            seed=seed,
            parameters=parameters,
        )
    if wandb_run is not None:
        wandb_run = _sync_or_abandon_wandb(
            wandb_run,
            spool_path,
            delivered_telemetry,
        )
    runtime = _build_runtime(model, recipe)
    channels_last = bool(recipe.get("channels_last", False))
    for epoch in range(start_epoch, recipe["epochs"]):
        started = time.perf_counter()
        train, optimizer_steps = _train_epoch_with_step_count(
            bindings.train_epoch,
            model,
            runtime,
            train_loader,
            optimizer,
            device=device,
            mixup_generator=mixup_generator,
            mixup_alpha=recipe["mixup_alpha"],
            precision=recipe["precision"],
            gradient_accumulation_steps=gradient_accumulation_steps,
            channels_last=channels_last,
        )
        global_step += optimizer_steps
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - started
        validation = bindings.evaluate(
            model,
            runtime,
            validation_loader,
            device,
            precision=recipe["precision"],
            channels_last=channels_last,
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train["loss"],
            "train_mixed_accuracy": train["mixed_accuracy"],
            "validation_accuracy": validation["accuracy"],
            "validation_cross_entropy": validation["cross_entropy"],
            "training_seconds": training_seconds,
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
            "host_input_wait_seconds": float(train.get("host_input_wait_seconds", 0.0)),
        }
        history.append(row)
        metrics = _epoch_telemetry_metrics(row)
        try:
            metrics.update(bindings.wandb_model_metrics(model))
        except Exception as error:  # noqa: BLE001  # Optional diagnostics are telemetry.
            _report_wandb_degraded("model-metrics", error)
        scheduler.step()
        _atomic_torch(
            checkpoint_path,
            {
                "variant": variant,
                "seed": seed,
                "contract_sha256": contract_sha256,
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "training_seconds": training_seconds,
                "training_generator_state": training_generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "python_rng_state": random.getstate(),
                "mixup_rng_state": mixup_generator.bit_generator.state,
            },
        )
        _append_telemetry_record(
            spool_path,
            {"kind": "epoch", "metrics": metrics, "step": epoch + 1},
        )
        progress_payload = {
            "checkpoint": str(checkpoint_path),
            "contract_sha256": contract_sha256,
            "epoch": epoch + 1,
            "global_step": global_step,
            "seed": seed,
            "telemetry_spool": str(spool_path),
            "training_seconds": training_seconds,
            "variant": variant,
        }
        print(  # noqa: T201
            "H200_PROGRESS_JSON="
            + json.dumps(progress_payload, separators=(",", ":"), sort_keys=True),
            flush=True,
        )
        if wandb_run is None and _wandb_retry_due(epoch + 1):
            wandb_run = _best_effort_initialize_wandb(
                root,
                contract,
                variant=variant,
                seed=seed,
                parameters=parameters,
            )
            delivered_telemetry = set()
        if wandb_run is not None:
            wandb_run = _sync_or_abandon_wandb(
                wandb_run,
                spool_path,
                delivered_telemetry,
            )
    final_validation = bindings.evaluate(
        model,
        runtime,
        validation_loader,
        device,
        precision=recipe["precision"],
        channels_last=channels_last,
    )
    result = {
        "variant": variant,
        "seed": seed,
        "contract_sha256": contract_sha256,
        "parameters": parameters,
        "global_step": global_step,
        "final_validation": final_validation,
        "best_validation_accuracy_diagnostic": max(row["validation_accuracy"] for row in history),
        "training_seconds": training_seconds,
        "complete_training_examples_per_second": (
            recipe["epochs"]
            * len(cast("Sized", cast("object", train_loader.dataset)))
            / training_seconds
        ),
        "history": history,
    }
    _atomic_json(result_path, result)
    _append_telemetry_record(
        spool_path,
        {"kind": "final", "summary": _final_telemetry_summary(result)},
    )
    if wandb_run is None:
        wandb_run = _best_effort_initialize_wandb(
            root,
            contract,
            variant=variant,
            seed=seed,
            parameters=parameters,
        )
        delivered_telemetry = set()
    if wandb_run is not None:
        _sync_telemetry_spool(wandb_run, spool_path, delivered_telemetry)
        _best_effort_finish_wandb(wandb_run)


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [
        root / "results" / f"{variant}__seed{seed}.json" for variant in VARIANTS for seed in SEEDS
    ]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    means = {
        variant: sum(
            row["final_validation"]["accuracy"] for row in rows if row["variant"] == variant
        )
        / len(SEEDS)
        for variant in VARIANTS
    }
    paired = [
        next(
            row["final_validation"]["accuracy"]
            for row in rows
            if row["variant"] == "product_four" and row["seed"] == seed
        )
        - next(
            row["final_validation"]["accuracy"]
            for row in rows
            if row["variant"] == "pole_free" and row["seed"] == seed
        )
        for seed in SEEDS
    ]
    payload = {
        "schema": contract["schema"],
        "mean_final_validation_accuracy": means,
        "paired_product_minus_pole_free": paired,
        "mean_product_minus_pole_free_pp": 100.0 * sum(paired) / len(paired),
        "G4_product_beats_pole_free_1pp": sum(paired) / len(paired) >= 0.01,
        "decision": (
            "continue ALPHABET-2D"
            if sum(paired) / len(paired) >= 0.01
            else "kill: gain belongs to shared backbone"
        ),
        "parameter_counts": {
            variant: sorted({row["parameters"] for row in rows if row["variant"] == variant})
            for variant in VARIANTS
        },
    }
    _atomic_json(root / "summary.json", payload)
    return payload


def runner_bindings(
    *,
    variants: tuple[str, ...] = VARIANTS,
    seeds: tuple[int, ...] = SEEDS,
    model_config: Callable[..., Any] = ImageNetNanoConfig,
    build_model: Callable[..., nn.Module] = build_imagenet_nano,
    contract: Callable[[argparse.Namespace], dict[str, Any]] = _contract,
    build_optimizer: Callable[..., torch.optim.Optimizer] = _build_optimizer,
    prepare_model: Callable[[nn.Module, dict[str, Any]], nn.Module] = _prepare_model,
    train_epoch: Callable[..., dict[str, float]] = _train_epoch,
    evaluate: Callable[..., dict[str, float]] = _evaluate,
    wandb_model_metrics: Callable[[nn.Module], dict[str, float]] = _wandb_model_metrics,
    summarize: Callable[[Path, dict[str, Any]], dict[str, Any] | None] = _summarize,
) -> RunnerBindings:
    """Build immutable per-run callbacks without mutating harness globals."""
    return RunnerBindings(
        variants=variants,
        seeds=seeds,
        model_config=model_config,
        build_model=build_model,
        contract=contract,
        build_optimizer=build_optimizer,
        prepare_model=prepare_model,
        train_epoch=train_epoch,
        evaluate=evaluate,
        wandb_model_metrics=wandb_model_metrics,
        summarize=summarize,
    )


def main(bindings: RunnerBindings | None = None) -> None:
    active = bindings or runner_bindings()
    args = _parse_args(variants=active.variants, seeds=active.seeds)
    if not torch.cuda.is_available():
        message = "ImageNet Nano runner requires CUDA"
        raise RuntimeError(message)
    _configure_cpu_affinity()
    if not set(args.run_seeds) <= set(active.seeds):
        message = "run seeds fall outside the ImageNet Nano contract"
        raise ValueError(message)
    contract = active.contract(args)
    _configure_compile_runtime(args.root, contract["recipe"])
    args.root.mkdir(parents=True, exist_ok=True)
    _initialize(args.root, contract)
    if args.initialize_only:
        return
    device = torch.device("cuda")
    for variant in args.variants:
        for seed in args.run_seeds:
            _run_job(
                args.root,
                contract,
                variant=variant,
                seed=seed,
                data_root=args.data_root,
                workers=args.workers,
                device=device,
                bindings=active,
            )
    summary = active.summarize(args.root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
