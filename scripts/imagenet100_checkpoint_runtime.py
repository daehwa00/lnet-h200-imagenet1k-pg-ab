"""Checkpoint and optimizer runtime for ImageNet-100 training."""

from __future__ import annotations

# Optimizer callback signatures and checkpoint payloads are intentionally dynamic.
# pyright: reportArgumentType=false, reportExplicitAny=false
import json
import math
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from torch.utils.data import DataLoader


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(
    path: Path,
    *,
    variant: str,
    seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_generator: torch.Generator,
    mixup_generator: np.random.Generator,
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
    epoch = checkpoint_nonnegative_integer(payload["epoch"], name="epoch")
    if progress is not None:
        stored_global_step = payload.get("global_step")
        if stored_global_step is None:
            steps_per_epoch = checkpoint_positive_integer(
                optimizer_steps_per_epoch,
                name="optimizer_steps_per_epoch",
            )
            global_step = epoch * steps_per_epoch
        else:
            global_step = checkpoint_nonnegative_integer(
                stored_global_step,
                name="global_step",
            )
        progress["global_step"] = global_step
    return epoch, payload["history"], float(payload["training_seconds"])


def checkpoint_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = f"checkpoint {name} must be a nonnegative integer"
        raise RuntimeError(message)
    return value


def checkpoint_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"checkpoint {name} must be a positive integer"
        raise RuntimeError(message)
    return value


def optimizer_steps_per_epoch(
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


def train_epoch_with_step_count(
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

    def step_and_count(*args: Any, **kwargs: Any) -> Any:
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


def build_optimizer(
    model: nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
    )


def restore_optimizer_runtime_options(
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


__all__ = [
    "atomic_json",
    "atomic_torch",
    "build_optimizer",
    "checkpoint_nonnegative_integer",
    "checkpoint_positive_integer",
    "optimizer_steps_per_epoch",
    "restore_checkpoint",
    "restore_optimizer_runtime_options",
    "train_epoch_with_step_count",
]
