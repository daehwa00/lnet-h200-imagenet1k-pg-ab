#!/usr/bin/env python3
"""Run one durable model/seed task from the H200 ImageNet-1K baseline queue."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# ruff: noqa: ANN401, BLE001, C901, NPY002, PLC0415, PLR0912, PLR0915, PLW2901, S603, T201
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import h200_baseline_registry as registry
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


NUM_CLASSES = 1000
IMAGE_SIZE = 224
EFFECTIVE_BATCH_SIZE = 256
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.8
RANDOM_ERASING_PROBABILITY = 0.25
SCHEMA = "lnet.h200.imagenet1k.matched_baseline.worker.v1"
PROGRESS_PREFIX = "H200_BASELINE_PROGRESS_JSON="
RESULT_PREFIX = "H200_BASELINE_RESULT_JSON="
WANDB_DEGRADED_PREFIX = "H200_BASELINE_WANDB_DEGRADED_JSON="


class _Mixup(Protocol):
    def __call__(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True, slots=True)
class BaselineTask:
    """One isolated queue item, including all result-affecting inputs."""

    phase: Literal["calibration", "full", "preflight"]
    model_key: str
    seed: int
    learning_rate: float
    epochs: int
    data_root: Path
    output_dir: Path
    result_path: Path
    checkpoint_path: Path
    source_root: Path | None = None
    batch_size: int = EFFECTIVE_BATCH_SIZE
    workers: int = 2
    wandb_mode: Literal["disabled", "online"] = "disabled"
    resume: bool = False
    max_steps: int | None = None

    @property
    def gradient_accumulation_steps(self) -> int:
        return EFFECTIVE_BATCH_SIZE // self.batch_size

    @property
    def task_name(self) -> str:
        lr_label = f"{self.learning_rate:.8g}".replace(".", "p")
        return f"{self.model_key}__{self.phase}__seed{self.seed}__lr{lr_label}"


@dataclass(slots=True)
class LoaderBundle:
    train: DataLoader[Any]
    validation: DataLoader[Any]
    train_generator: torch.Generator
    validation_generator: torch.Generator


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = _canonical_json(payload) + b"\n"
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _task_payload(task: BaselineTask) -> dict[str, Any]:
    payload = asdict(task)
    for key in ("data_root", "output_dir", "result_path", "checkpoint_path"):
        payload[key] = str(cast("Path", payload[key]).resolve())
    if task.source_root is not None:
        payload["source_root"] = str(task.source_root.resolve())
    payload.pop("resume")
    return payload


def _source_digests(model_key: str) -> dict[str, str]:
    paths = {
        "worker": Path(__file__).resolve(),
        "registry": Path(cast("str", registry.__file__)).resolve(),
    }
    if registry.model_spec(model_key).backend == "external":
        external = importlib.util.find_spec("h200_external_models")
        if external is None or external.origin is None:
            message = f"cannot bind {model_key} to missing h200_external_models.py"
            raise RuntimeError(message)
        paths["external_models"] = Path(external.origin).resolve()
    return {name: _sha256_file(path) for name, path in paths.items()}


def _dataset_identity() -> dict[str, str | None]:
    manifest_value = os.environ.get("LNET_DATASET_MANIFEST_PATH")
    if not manifest_value:
        return {
            "identity_sha256": os.environ.get("LNET_DATASET_IDENTITY_SHA256"),
            "manifest_sha256": None,
        }
    manifest_path = Path(manifest_value).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = cast("str | None", manifest.get("identity_sha256"))
    expected = os.environ.get("LNET_DATASET_IDENTITY_SHA256")
    if expected is not None and identity != expected:
        message = "dataset manifest identity does not match LNET_DATASET_IDENTITY_SHA256"
        raise RuntimeError(message)
    return {
        "identity_sha256": identity,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _contract(task: BaselineTask) -> dict[str, Any]:
    source_sha256 = _source_digests(task.model_key)
    effective_warmup_epochs = 1 if task.phase in {"calibration", "preflight"} else WARMUP_EPOCHS
    external_provenance: dict[str, object] | None = None
    if registry.model_spec(task.model_key).backend == "external":
        from h200_external_models import external_model_provenance

        external_provenance = external_model_provenance(task.model_key, task.source_root)
    native_extension = None
    if task.model_key == "uniconvnet_a":
        wheel_sha = os.environ.get("H200_DCNV3_WHEEL_SHA256")
        patch_sha = os.environ.get("H200_DCNV3_PATCH_SHA256")
        if not wheel_sha or len(wheel_sha) != 64 or not patch_sha or len(patch_sha) != 64:
            message = "UniConvNet-A requires verified DCNv3 wheel and patch digests"
            raise RuntimeError(message)
        native_extension = {
            "name": "DCNv3",
            "wheel_sha256": wheel_sha,
            "compatibility_patch_sha256": patch_sha,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "driver_version": os.environ.get("H200_GPU_DRIVER_VERSION"),
        }
    payload = {
        "schema": SCHEMA,
        "task": _task_payload(task),
        "task_sha256": _sha256_payload(_task_payload(task)),
        "model": {
            "key": task.model_key,
            "display_name": registry.model_spec(task.model_key).display_name,
            "num_classes": NUM_CLASSES,
            "pretrained": False,
            "distillation": False,
            "timm_version": (
                registry.TIMM_VERSION
                if registry.model_spec(task.model_key).backend == "timm"
                else None
            ),
        },
        "recipe": {
            "image_size": IMAGE_SIZE,
            "optimizer": "AdamW",
            "learning_rate": task.learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "configured_full_run_warmup_epochs": WARMUP_EPOCHS,
            "effective_warmup_epochs": effective_warmup_epochs,
            "schedule": "linear warmup plus cosine decay per optimizer step",
            "label_smoothing": LABEL_SMOOTHING,
            "mixup_alpha": MIXUP_ALPHA,
            "cutmix_alpha": 0.0,
            "randaugment": "rand-m9-mstd0.5-inc1 (N=2, M=9)",
            "random_erasing_probability": RANDOM_ERASING_PROBABILITY,
            "precision": registry.model_spec(task.model_key).precision,
            "batch_size": task.batch_size,
            "gradient_accumulation_steps": task.gradient_accumulation_steps,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "persistent_workers": False,
            "validation": "streaming full validation set",
            "resume": (
                "epoch-boundary RNG continuity; bitwise CUDA kernel determinism "
                "is not claimed"
            ),
        },
        "dataset": {
            "name": "ImageNet-1K",
            "root": str(task.data_root.resolve()),
            **_dataset_identity(),
        },
        "source_sha256": source_sha256,
        "source_digest_sha256": _sha256_payload(source_sha256),
        "external_source_provenance": external_provenance,
        "native_extension": native_extension,
        "runtime": {
            "gpu_memory_fraction": os.environ.get("H200_GPU_MEMORY_FRACTION"),
            "mps_active": os.environ.get("H200_BASELINE_MPS_ACTIVE"),
            "mps_active_thread_percentage": os.environ.get(
                "H200_BASELINE_MPS_ACTIVE_THREAD_PERCENTAGE"
            ),
            "cuda_device": cuda_device,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }
    return json.loads(json.dumps(payload))


def _validate_task(task: BaselineTask) -> None:
    registry.model_spec(task.model_key)
    if task.seed < 0:
        message = f"seed must be non-negative, got {task.seed}"
        raise ValueError(message)
    if not math.isfinite(task.learning_rate) or task.learning_rate <= 0:
        message = f"learning rate must be positive and finite, got {task.learning_rate}"
        raise ValueError(message)
    if task.epochs <= 0:
        message = f"epochs must be positive, got {task.epochs}"
        raise ValueError(message)
    if task.batch_size <= 0 or EFFECTIVE_BATCH_SIZE % task.batch_size:
        message = (
            f"batch size must be a positive divisor of {EFFECTIVE_BATCH_SIZE}, "
            f"got {task.batch_size}"
        )
        raise ValueError(message)
    if task.workers < 0:
        message = f"workers must be non-negative, got {task.workers}"
        raise ValueError(message)
    if task.max_steps is not None and (task.phase != "preflight" or task.max_steps <= 0):
        message = "max_steps is positive and supported only for preflight tasks"
        raise ValueError(message)
    if task.phase != "full" and task.wandb_mode != "disabled":
        message = "W&B is forbidden for calibration and preflight tasks"
        raise ValueError(message)


def _load_task_json(value: str) -> dict[str, Any]:
    candidate = Path(value)
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        message = "task JSON must contain an object"
        raise TypeError(message)
    if isinstance(payload.get("task"), dict):
        payload = cast("dict[str, Any]", payload["task"])
    return {str(key).replace("-", "_"): item for key, item in payload.items()}


def _parse_task(argv: Sequence[str] | None = None) -> BaselineTask:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json")
    parser.add_argument("--phase", choices=("calibration", "full", "preflight"))
    parser.add_argument("--model-key")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", "--root", dest="output_dir", type=Path)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--wandb-mode", choices=("disabled", "online"))
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--max-steps", type=int)
    arguments = vars(parser.parse_args(argv))
    task_json = arguments.pop("task_json")
    merged = _load_task_json(task_json) if task_json else {}
    merged.update({key: value for key, value in arguments.items() if value is not None})

    aliases = {
        "model": "model_key",
        "lr": "learning_rate",
        "root": "output_dir",
        "batch": "batch_size",
    }
    for old, new in aliases.items():
        if old in merged and new not in merged:
            merged[new] = merged.pop(old)

    required = ("phase", "model_key", "seed", "learning_rate", "epochs", "data_root")
    missing = [key for key in required if key not in merged]
    if "output_dir" not in merged and "result_path" in merged:
        merged["output_dir"] = str(Path(merged["result_path"]).parent.parent)
    if "output_dir" not in merged:
        missing.append("output_dir")
    if missing:
        parser.error(f"missing task fields: {', '.join(missing)}")

    output_dir = Path(merged["output_dir"])
    model_key = str(merged["model_key"])
    phase = str(merged["phase"])
    seed = int(merged["seed"])
    learning_rate = float(merged["learning_rate"])
    lr_label = f"{learning_rate:.8g}".replace(".", "p")
    stem = f"{model_key}__{phase}__seed{seed}__lr{lr_label}"
    task = BaselineTask(
        phase=cast("Literal['calibration', 'full', 'preflight']", phase),
        model_key=model_key,
        seed=seed,
        learning_rate=learning_rate,
        epochs=int(merged["epochs"]),
        data_root=Path(merged["data_root"]),
        output_dir=output_dir,
        result_path=Path(merged.get("result_path", output_dir / "results" / f"{stem}.json")),
        checkpoint_path=Path(
            merged.get("checkpoint_path", output_dir / "checkpoints" / f"{stem}.pt")
        ),
        source_root=(Path(merged["source_root"]) if merged.get("source_root") else None),
        batch_size=int(merged.get("batch_size", EFFECTIVE_BATCH_SIZE)),
        workers=int(merged.get("workers", 2)),
        wandb_mode=cast("Literal['disabled', 'online']", merged.get("wandb_mode", "disabled")),
        resume=bool(merged.get("resume", False)),
        max_steps=(
            int(merged["max_steps"])
            if merged.get("max_steps") is not None
            else (100 if phase == "preflight" else None)
        ),
    )
    _validate_task(task)
    return task


def _build_loaders(task: BaselineTask, device: torch.device) -> LoaderBundle:
    from timm.data import create_transform
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import (
        InterpolationMode,
        v2,
    )

    train_root = task.data_root / "train"
    validation_root = task.data_root / "val"
    if not train_root.is_dir() or not validation_root.is_dir():
        message = f"ImageNet-1K requires train/ and val/ below {task.data_root}"
        raise FileNotFoundError(message)
    train_transform = create_transform(
        input_size=(3, IMAGE_SIZE, IMAGE_SIZE),
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=RANDOM_ERASING_PROBABILITY,
        re_mode="pixel",
        re_count=1,
    )
    validation_transform = v2.Compose(
        [
            v2.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = ImageFolder(train_root, transform=train_transform)
    validation_dataset = ImageFolder(validation_root, transform=validation_transform)
    if len(train_dataset.classes) != NUM_CLASSES:
        message = f"training split has {len(train_dataset.classes)} classes, expected {NUM_CLASSES}"
        raise RuntimeError(message)
    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        message = "training and validation class mappings differ"
        raise RuntimeError(message)

    train_generator = torch.Generator().manual_seed(task.seed)
    validation_generator = torch.Generator().manual_seed(task.seed + 1)
    common = {
        "batch_size": task.batch_size,
        "num_workers": task.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    if task.workers:
        common["prefetch_factor"] = 1
    train = DataLoader(
        train_dataset,
        shuffle=True,
        # timm Mixup requires even batches; ImageNet-1K leaves an odd final
        # partial batch at effective batch 256.
        drop_last=True,
        generator=train_generator,
        **common,
    )
    validation = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        generator=validation_generator,
        **common,
    )
    return LoaderBundle(train, validation, train_generator, validation_generator)


def _make_mixup() -> _Mixup:
    from timm.data import Mixup

    return Mixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=0.0,
        prob=1.0,
        switch_prob=0.0,
        mode="batch",
        label_smoothing=LABEL_SMOOTHING,
        num_classes=NUM_CLASSES,
    )


def _extract_logits(output: object) -> Tensor:
    """Normalize common public-repository classifier output conventions."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "out", "pred", "prediction", "output"):
            if key in output:
                return _extract_logits(output[key])
        tensors = [value for value in output.values() if isinstance(value, Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        message = f"ambiguous dict model output keys: {sorted(map(str, output))}"
        raise TypeError(message)
    if isinstance(output, (tuple, list)) and output:
        tensors = [value for value in output if isinstance(value, Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        message = "model returned ambiguous auxiliary logits"
        raise TypeError(message)
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    message = f"unsupported model output type: {type(output).__name__}"
    raise TypeError(message)


def _classification_logits(output: object, batch_size: int) -> Tensor:
    logits = _extract_logits(output)
    if logits.shape != (batch_size, NUM_CLASSES):
        message = (
            f"classifier logits must have shape {(batch_size, NUM_CLASSES)}, "
            f"got {tuple(logits.shape)}"
        )
        raise RuntimeError(message)
    return logits


def _build_optimizer(model: nn.Module, learning_rate: float, device: torch.device) -> Any:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {"lr": learning_rate, "weight_decay": WEIGHT_DECAY}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def _learning_rate(
    base_learning_rate: float,
    optimizer_step: int,
    optimizer_steps_per_epoch: int,
    total_epochs: int,
    warmup_epochs: int,
) -> float:
    warmup_steps = warmup_epochs * optimizer_steps_per_epoch
    total_steps = total_epochs * optimizer_steps_per_epoch
    completed = optimizer_step + 1
    if completed <= warmup_steps:
        return base_learning_rate * completed / warmup_steps
    cosine_steps = max(total_steps - warmup_steps, 1)
    progress = min((completed - warmup_steps) / cosine_steps, 1.0)
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _soft_target_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return torch.sum(-targets * torch.nn.functional.log_softmax(logits, dim=-1), dim=-1).mean()


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: Any,
    mixup: _Mixup,
    device: torch.device,
    task: BaselineTask,
    *,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, float], int, bool]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = task.gradient_accumulation_steps
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation)
    loss_terms: list[Tensor] = []
    correct_terms: list[Tensor] = []
    total = 0
    optimizer_steps = 0
    stopped = False
    window_start = 0
    use_bfloat16 = registry.model_spec(task.model_key).precision == "bfloat16"

    for batch_index, (inputs, hard_targets) in enumerate(loader):
        if batch_index == window_start:
            window_size = min(accumulation, len(loader) - window_start)
        inputs, soft_targets = mixup(inputs, hard_targets)
        inputs = inputs.to(device, non_blocking=True)
        hard_targets = hard_targets.to(device, non_blocking=True)
        soft_targets = soft_targets.to(device, non_blocking=True)
        if inputs.ndim == 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and use_bfloat16,
        ):
            logits = _classification_logits(model(inputs), hard_targets.numel())
            loss = _soft_target_cross_entropy(logits.float(), soft_targets.float())
        (loss / window_size).backward()
        batch_size = hard_targets.numel()
        loss_terms.append(loss.detach() * batch_size)
        correct_terms.append(logits.detach().argmax(dim=-1).eq(hard_targets).sum())
        total += batch_size

        end_of_window = batch_index + 1 == window_start + window_size
        if not end_of_window:
            continue
        learning_rate = _learning_rate(
            task.learning_rate,
            global_step,
            optimizer_steps_per_epoch,
            task.epochs,
            1 if task.phase in {"calibration", "preflight"} else WARMUP_EPOCHS,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        optimizer_steps += 1
        window_start = batch_index + 1
        if task.max_steps is not None and global_step >= task.max_steps:
            stopped = True
            break

    if total == 0:
        message = "training loader produced no examples"
        raise RuntimeError(message)
    loss_sum = float(torch.stack(loss_terms).double().sum())
    correct = int(torch.stack(correct_terms).sum())
    if not math.isfinite(loss_sum):
        message = f"non-finite training loss for {task.model_key} at epoch {epoch}"
        raise FloatingPointError(message)
    return (
        {
            "epoch": float(epoch),
            "loss": loss_sum / total,
            "mixed_accuracy": correct / total,
            "examples": float(total),
            "optimizer_steps": float(optimizer_steps),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        },
        global_step,
        stopped,
    )


def _evaluate_streaming(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    use_bfloat16: bool,
) -> dict[str, float]:
    model.eval()
    total = 0
    loss_terms: list[Tensor] = []
    top1_terms: list[Tensor] = []
    top5_terms: list[Tensor] = []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if inputs.ndim == 4:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and use_bfloat16,
            ):
                logits = _classification_logits(model(inputs), targets.numel())
            float_logits = logits.float()
            loss_terms.append(
                torch.nn.functional.cross_entropy(float_logits, targets, reduction="sum")
            )
            batch_size = targets.numel()
            total += batch_size
            predictions = float_logits.topk(min(5, float_logits.shape[-1]), dim=-1).indices
            matches = predictions.eq(targets[:, None])
            top1_terms.append(matches[:, :1].any(dim=-1).sum())
            top5_terms.append(matches.any(dim=-1).sum())
    if total == 0:
        message = "validation loader produced no examples"
        raise RuntimeError(message)
    loss_sum = float(torch.stack(loss_terms).double().sum())
    top1 = int(torch.stack(top1_terms).sum())
    top5 = int(torch.stack(top5_terms).sum())
    if not math.isfinite(loss_sum):
        message = "validation produced non-finite cross entropy"
        raise FloatingPointError(message)
    return {
        "accuracy": top1 / total,
        "top5_accuracy": top5 / total,
        "cross_entropy": loss_sum / total,
        "examples": float(total),
    }


def _capture_rng(bundle: LoaderBundle) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": bundle.train_generator.get_state(),
        "validation_generator": bundle.validation_generator.get_state(),
    }


def _restore_rng(payload: dict[str, Any], bundle: LoaderBundle) -> None:
    random.setstate(payload["python"])
    numpy_state = payload["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload["cuda"]:
        torch.cuda.set_rng_state_all(payload["cuda"])
    bundle.train_generator.set_state(payload["train_generator"])
    bundle.validation_generator.set_state(payload["validation_generator"])


def _validate_binding(payload: dict[str, Any], contract: dict[str, Any]) -> None:
    expected = {
        "contract_sha256": _sha256_payload(contract),
        "task_sha256": contract["task_sha256"],
        "source_digest_sha256": contract["source_digest_sha256"],
    }
    mismatches = {
        name: {"expected": value, "actual": payload.get(name)}
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        message = f"artifact binding mismatch: {json.dumps(mismatches, sort_keys=True)}"
        raise RuntimeError(message)


def _progress_path(task: BaselineTask) -> Path:
    return task.output_dir / "progress" / f"{task.task_name}.json"


def _emit_progress(task: BaselineTask, payload: dict[str, Any]) -> None:
    _atomic_json(_progress_path(task), payload)
    print(f"{PROGRESS_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)


def _telemetry_spool_path(task: BaselineTask) -> Path:
    return task.output_dir / "telemetry" / f"{task.task_name}.jsonl"


def _append_telemetry(task: BaselineTask, record: dict[str, Any]) -> bool:
    path = _telemetry_spool_path(task)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        records = _read_telemetry(task)
        record_id = str(record["id"])
        for existing in records:
            if str(existing.get("id")) != record_id:
                continue
            if existing != record:
                degraded = {
                    "error_type": "RuntimeError",
                    "message_class": "telemetry_conflict",
                    "model_key": task.model_key,
                    "seed": task.seed,
                }
                print(
                    f"{WANDB_DEGRADED_PREFIX}{json.dumps(degraded, sort_keys=True)}",
                    flush=True,
                )
                return False
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            for existing in (*records, record):
                stream.write(_canonical_json(existing) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except Exception as error:  # W&B telemetry cannot stop training.
        degraded = {
            "error_type": type(error).__name__,
            "model_key": task.model_key,
            "seed": task.seed,
            "stage": "local_spool",
        }
        print(f"{WANDB_DEGRADED_PREFIX}{json.dumps(degraded, sort_keys=True)}", flush=True)
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        return False
    return True


def _read_telemetry(task: BaselineTask) -> list[dict[str, Any]]:
    path = _telemetry_spool_path(task)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _epoch_telemetry_record(row: dict[str, Any]) -> dict[str, Any]:
    epoch = int(row["epoch"])
    train = row["train"]
    validation = row["validation"]
    return {
        "id": f"epoch:{epoch}",
        "kind": "epoch",
        "step": int(row["global_step"]),
        "metrics": {
            "epoch": epoch,
            "train/loss": train["loss"],
            "train/mixed_accuracy": train["mixed_accuracy"],
            "train/images_per_second": train["images_per_second"],
            "validation/accuracy": validation["accuracy"],
            "validation/top5_accuracy": validation["top5_accuracy"],
            "validation/cross_entropy": validation["cross_entropy"],
            "optimizer/learning_rate": train["learning_rate"],
        },
    }


def _backfill_telemetry(task: BaselineTask, history: list[dict[str, Any]]) -> None:
    for row in history:
        _append_telemetry(task, _epoch_telemetry_record(row))


def _final_telemetry_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "final",
        "kind": "final",
        "step": int(result["global_step"]),
        "metrics": {
            "final/validation_accuracy": result["final_validation"]["accuracy"],
            "final/validation_top5_accuracy": result["final_validation"]["top5_accuracy"],
            "final/training_seconds": result["training_seconds"],
        },
    }


def _wandb_run_id(task: BaselineTask) -> str:
    direct = os.environ.get("H200_BASELINE_RUN_ID")
    if direct:
        return direct
    mapping_value = os.environ.get("H200_BASELINE_RUN_IDS_JSON")
    if mapping_value:
        mapping_path = Path(mapping_value)
        mapping = json.loads(
            mapping_path.read_text(encoding="utf-8") if mapping_path.is_file() else mapping_value
        )
        for key in (task.task_name, f"{task.model_key}__seed{task.seed}"):
            if isinstance(mapping, dict) and isinstance(mapping.get(key), str):
                return cast("str", mapping[key])
    message = "online W&B requires a campaign-issued H200_BASELINE_RUN_ID(S_JSON)"
    raise RuntimeError(message)


class _WandbMirror:
    """Own a killable W&B sidecar; training performs no W&B network calls."""

    def __init__(self, task: BaselineTask, contract: dict[str, Any]) -> None:
        self.task = task
        self.contract = contract
        self.process: subprocess.Popen[str] | None = None
        self.log_stream: Any | None = None
        self.next_retry_record = 0
        self.stop_path = _telemetry_spool_path(task).with_suffix(".stop.json")
        self.complete_path = _telemetry_spool_path(task).with_suffix(".mirror-complete.json")
        if not task.result_path.exists() and self.stop_path.exists():
            self.stop_path.replace(self.stop_path.with_suffix(f".stale-{os.getpid()}.json"))

    @property
    def enabled(self) -> bool:
        return self.task.phase == "full" and self.task.wandb_mode == "online"

    def _close_process(self) -> None:
        if self.log_stream is not None and not self.log_stream.closed:
            self.log_stream.close()
        self.log_stream = None
        self.process = None

    def _start(self) -> None:
        _wandb_run_id(self.task)
        for name in (
            "H200_BASELINE_DISPLAY_NAME",
            "H200_BASELINE_TAGS_JSON",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_GROUP",
        ):
            if not os.environ.get(name):
                message = f"online W&B sidecar requires {name}"
                raise RuntimeError(message)
        contract_path = self.task.output_dir / "contracts" / f"{self.task.task_name}.json"
        telemetry_script = Path(__file__).with_name("run_h200_baseline_telemetry.py")
        if not telemetry_script.is_file():
            message = f"baseline telemetry sidecar is missing: {telemetry_script}"
            raise FileNotFoundError(message)
        log_path = self.task.output_dir / "telemetry" / "wandb-sidecar.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = log_path.open("a")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(telemetry_script),
                "--spool",
                str(_telemetry_spool_path(self.task)),
                "--result",
                str(self.task.result_path),
                "--stop",
                str(self.stop_path),
                "--complete-marker",
                str(self.complete_path),
                "--contract",
                str(contract_path),
                "--model-key",
                self.task.model_key,
                "--seed",
                str(self.task.seed),
                "--parent-pid",
                str(os.getpid()),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=os.environ.copy(),
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def sync(self) -> None:
        if not self.enabled or self.complete_path.is_file():
            return
        records = _read_telemetry(self.task)
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None:
            return_code = self.process.poll()
            self._close_process()
            if return_code not in (None, 0):
                self.next_retry_record = len(records) + 10
        if len(records) < self.next_retry_record:
            return
        try:
            self._start()
        except Exception as error:  # W&B is non-authoritative; never expose error text.
            degraded = {
                "error_type": type(error).__name__,
                "model_key": self.task.model_key,
                "seed": self.task.seed,
            }
            print(f"{WANDB_DEGRADED_PREFIX}{json.dumps(degraded, sort_keys=True)}", flush=True)
            self.next_retry_record = len(records) + 10

    def finish(self, result: dict[str, Any]) -> None:
        del result
        if not self.enabled or self.complete_path.is_file():
            return
        _atomic_json(self.stop_path, {"stop": True})
        self.next_retry_record = 0
        self.sync()
        if self.process is None:
            self.next_retry_record = 0
            self.sync()
        if self.process is None:
            return
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3)
            degraded = {
                "error_type": "TimeoutExpired",
                "model_key": self.task.model_key,
                "seed": self.task.seed,
            }
            print(f"{WANDB_DEGRADED_PREFIX}{json.dumps(degraded, sort_keys=True)}", flush=True)
        finally:
            self._close_process()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_cuda_memory_limit(device: torch.device) -> float | None:
    value = os.environ.get("H200_GPU_MEMORY_FRACTION")
    if device.type != "cuda" or value is None:
        return None
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        message = f"H200_GPU_MEMORY_FRACTION must be in (0, 1], got {value!r}"
        raise ValueError(message)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.cuda.reset_peak_memory_stats(device)
    return fraction


def run_task(
    task: BaselineTask,
    *,
    device: torch.device | None = None,
    model_builder: Callable[[str, str | Path | None, int], nn.Module] = registry.build_model,
    loader_builder: Callable[[BaselineTask, torch.device], LoaderBundle] = _build_loaders,
) -> dict[str, Any]:
    """Execute or RNG-continuously resume one task; local artifacts are authoritative."""
    _validate_task(task)
    contract = _contract(task)
    contract_sha256 = _sha256_payload(contract)
    contract_path = task.output_dir / "contracts" / f"{task.task_name}.json"
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if _sha256_payload(existing_contract) != contract_sha256:
            message = f"existing contract changed for {task.task_name}"
            raise RuntimeError(message)
    else:
        _atomic_json(contract_path, contract)
    mirror = _WandbMirror(task, contract)

    if task.result_path.exists():
        result = json.loads(task.result_path.read_text(encoding="utf-8"))
        _validate_binding(result, contract)
        _backfill_telemetry(task, cast("list[dict[str, Any]]", result["history"]))
        _append_telemetry(task, _final_telemetry_record(result))
        print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)
        mirror.finish(result)
        return result
    if task.checkpoint_path.exists() and not task.resume:
        message = f"checkpoint exists; pass --resume: {task.checkpoint_path}"
        raise RuntimeError(message)
    if task.resume and not task.checkpoint_path.exists():
        message = f"--resume requested but checkpoint is missing: {task.checkpoint_path}"
        raise FileNotFoundError(message)

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    memory_fraction = _configure_cuda_memory_limit(resolved_device)
    _seed_everything(task.seed)
    model = model_builder(task.model_key, task.source_root, NUM_CLASSES)
    model = model.to(resolved_device)
    model = model.to(memory_format=torch.channels_last)
    optimizer = _build_optimizer(model, task.learning_rate, resolved_device)
    loaders = loader_builder(task, resolved_device)
    mixup = _make_mixup()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    history: list[dict[str, Any]] = []
    completed_epochs = 0
    global_step = 0
    elapsed_before_resume = 0.0

    if task.resume:
        checkpoint = torch.load(
            task.checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, dict):
            message = "checkpoint payload is not an object"
            raise RuntimeError(message)
        _validate_binding(checkpoint, contract)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = cast("list[dict[str, Any]]", checkpoint["history"])
        completed_epochs = int(checkpoint["completed_epochs"])
        global_step = int(checkpoint["global_step"])
        elapsed_before_resume = float(checkpoint["training_seconds"])
        _restore_rng(checkpoint["rng"], loaders)
        _backfill_telemetry(task, history)

    started = time.monotonic()
    stopped_at_max_steps = False
    for epoch in range(completed_epochs + 1, task.epochs + 1):
        train_started = time.monotonic()
        train_metrics, global_step, stopped = _train_one_epoch(
            model,
            loaders.train,
            optimizer,
            mixup,
            resolved_device,
            task,
            epoch=epoch,
            global_step=global_step,
        )
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        train_seconds = time.monotonic() - train_started
        train_metrics["images_per_second"] = train_metrics["examples"] / max(train_seconds, 1.0e-12)
        validation_metrics = _evaluate_streaming(
            model,
            loaders.validation,
            resolved_device,
            use_bfloat16=registry.model_spec(task.model_key).precision == "bfloat16",
        )
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        training_seconds = elapsed_before_resume + time.monotonic() - started
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "training_seconds": training_seconds,
        }
        history.append(row)
        completed_epochs = epoch
        checkpoint_payload = {
            "schema": SCHEMA,
            "contract_sha256": contract_sha256,
            "task_sha256": contract["task_sha256"],
            "source_digest_sha256": contract["source_digest_sha256"],
            "source_sha256": contract["source_sha256"],
            "completed_epochs": completed_epochs,
            "global_step": global_step,
            "parameters": parameters,
            "history": history,
            "training_seconds": training_seconds,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _capture_rng(loaders),
        }
        # The checkpoint is authoritative and must be durable before progress
        # output or any optional network telemetry observes this epoch.
        _atomic_torch(task.checkpoint_path, checkpoint_payload)
        _append_telemetry(task, _epoch_telemetry_record(row))
        mirror.sync()
        progress = {
            "schema": SCHEMA,
            "status": "running",
            "phase": task.phase,
            "model_key": task.model_key,
            "seed": task.seed,
            "epoch": epoch,
            "epochs": task.epochs,
            "global_step": global_step,
            "validation_accuracy": validation_metrics["accuracy"],
            "checkpoint_path": str(task.checkpoint_path.resolve()),
            "contract_sha256": contract_sha256,
        }
        _emit_progress(task, progress)
        if stopped:
            stopped_at_max_steps = True
            break

    if not history:
        message = "checkpoint has no completed epoch history"
        raise RuntimeError(message)
    final_training_seconds = elapsed_before_resume + time.monotonic() - started
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "phase": task.phase,
        "model_key": task.model_key,
        "display_name": registry.model_spec(task.model_key).display_name,
        "seed": task.seed,
        "learning_rate": task.learning_rate,
        "parameters": parameters,
        "completed_epochs": completed_epochs,
        "requested_epochs": task.epochs,
        "global_step": global_step,
        "stopped_at_max_steps": stopped_at_max_steps,
        "training_seconds": final_training_seconds,
        "final_validation": history[-1]["validation"],
        "metrics": {
            "validation_top1": history[-1]["validation"]["accuracy"],
            "validation_top5": history[-1]["validation"]["top5_accuracy"],
            "validation_cross_entropy": history[-1]["validation"]["cross_entropy"],
            "images_per_second": history[-1]["train"]["images_per_second"],
            "peak_gpu_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(resolved_device)
                if resolved_device.type == "cuda"
                else 0
            ),
            "peak_gpu_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved(resolved_device)
                if resolved_device.type == "cuda"
                else 0
            ),
        },
        "gpu_memory_fraction": memory_fraction,
        "history": history,
        "contract_sha256": contract_sha256,
        "task_sha256": contract["task_sha256"],
        "source_digest_sha256": contract["source_digest_sha256"],
        "source_sha256": contract["source_sha256"],
        "external_source_provenance": contract["external_source_provenance"],
        "native_extension": contract["native_extension"],
        "checkpoint_path": str(task.checkpoint_path.resolve()),
        "result_path": str(task.result_path.resolve()),
    }
    _atomic_json(task.result_path, result)
    _append_telemetry(task, _final_telemetry_record(result))
    final_progress = {
        "schema": SCHEMA,
        "status": "completed",
        "phase": task.phase,
        "model_key": task.model_key,
        "seed": task.seed,
        "epoch": completed_epochs,
        "epochs": task.epochs,
        "global_step": global_step,
        "validation_accuracy": result["final_validation"]["accuracy"],
        "result_path": str(task.result_path.resolve()),
        "contract_sha256": contract_sha256,
    }
    _emit_progress(task, final_progress)
    print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)
    mirror.finish(result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    task = _parse_task(argv)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = task.output_dir / ".worker.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = f"another worker owns task output: {task.output_dir}"
            raise RuntimeError(message) from error
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        run_task(task)


if __name__ == "__main__":
    main()
