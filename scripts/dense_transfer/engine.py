"""Small, restart-safe primitives used by the dense-transfer command line.

The module deliberately contains no dataset/model recipe.  Those live in
``dense_transfer.data`` and ``dense_transfer.models`` so this code can be
tested with CPU-sized fixtures and cannot accidentally start a large run.
"""
from __future__ import annotations

import hashlib
import inspect
import importlib.metadata
import json
import math
import os
import random
import sys
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor, nn


SCHEMA = "dense-transfer.engine.v1"
VALID_TASKS = ("ade20k", "coco")
VALID_MODELS = ("va_k128", "convnextv2_atto", "tinyvim_s")


@dataclass(frozen=True, slots=True)
class OptimizerPlan:
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_updates: int
    polynomial_power: float = 1.0
    drop_epochs: tuple[int, ...] = ()
    drop_factor: float = 0.1


def default_optimizer_plan(task: str) -> OptimizerPlan:
    if task == "ade20k":
        # 80k means optimizer updates, not dataloader iterations.
        return OptimizerPlan(epochs=0, learning_rate=6e-5, weight_decay=0.01, warmup_updates=1500)
    if task == "coco":
        return OptimizerPlan(
            epochs=12,
            learning_rate=1e-4,
            weight_decay=0.05,
            warmup_updates=1000,
            drop_epochs=(8, 11),
        )
    raise ValueError(f"unsupported task {task!r}; choose one of {VALID_TASKS}")


@dataclass(frozen=True, slots=True)
class RunSettings:
    task: str
    model_key: str
    checkpoint: Path
    source_root: Path | None
    data_root: Path
    output_root: Path
    device: str = "cuda"
    physical_batch_size: int = 2
    effective_batch_size: int = 16
    workers: int = 2
    bf16: bool = True
    compiled_backbone: bool = False
    channels_last: bool = False
    fused_adamw: bool = False
    checkpoint_backbone_blocks: bool = False
    seed: int = 501

    @property
    def accumulation_steps(self) -> int:
        if self.physical_batch_size <= 0 or self.effective_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.effective_batch_size % self.physical_batch_size:
            raise ValueError("effective batch size must be divisible by physical batch size")
        return self.effective_batch_size // self.physical_batch_size


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(data_root: Path) -> dict[str, object]:
    """Stable dataset identity from annotation content or an ADE file manifest."""
    root = data_root.resolve()
    coco_annotations = sorted((root / "annotations").glob("instances_*.json"))
    if coco_annotations:
        return {
            "kind": "coco_annotations",
            "annotations": {path.name: sha256_file(path) for path in coco_annotations},
        }
    ade_root = root / "ADEChallengeData2016" if (root / "ADEChallengeData2016").is_dir() else root
    annotation_root = ade_root / "annotations"
    if annotation_root.is_dir():
        entries = [
            (path.relative_to(annotation_root).as_posix(), path.stat().st_size)
            for path in sorted(annotation_root.rglob("*.png"))
        ]
        if entries:
            return {
                "kind": "ade_annotation_manifest", "files": len(entries),
                "sha256": hashlib.sha256(canonical_json(entries)).hexdigest(),
            }
    # An unsupported layout is still stable for a given explicitly supplied
    # root, but is never confused with a validated ADE/COCO installation.
    return {"kind": "unrecognized_root", "path_name": root.name}


def prevalidate_backbone_checkpoint(checkpoint: Path, model_key: str) -> dict[str, object]:
    """Verify the supplied 100-epoch ImageNet checkpoint before transfer.

    Checkpoint A is identified by its immutable filename as well as its stored
    completion count.  Its legacy run has 500500 updates, unlike the 500400
    updates of the other two sources, so update count is intentionally not used
    as a validity criterion.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"backbone checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or int(payload.get("completed_epochs", -1)) != 100:
        raise ValueError("backbone checkpoint must contain completed_epochs == 100")
    if not isinstance(payload.get("model"), Mapping):
        raise ValueError("backbone checkpoint has no model state mapping")
    if model_key == "va_k128":
        name = checkpoint.name.lower()
        if "seed501" not in name or "ep100" not in name:
            raise ValueError("va_k128 requires prevalidated checkpoint A (seed501, ep100)")
    return {
        "completed_epochs": int(payload["completed_epochs"]),
        "global_step": int(payload.get("global_step", -1)),
        "contract_sha256": str(payload.get("contract_sha256", "")),
        "file_sha256": sha256_file(checkpoint),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def runtime_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "torch": torch.__version__}
    for package in ("torchvision", "timm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def runtime_contract(
    settings: RunSettings, plan: OptimizerPlan, backbone: nn.Module | None = None,
    runtime_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    if not settings.checkpoint.is_file():
        raise FileNotFoundError(f"backbone checkpoint does not exist: {settings.checkpoint}")
    if not settings.data_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {settings.data_root}")
    config = asdict(settings)
    for key in ("checkpoint", "source_root", "data_root", "output_root"):
        value = config[key]
        config[key] = None if value is None else str(Path(value).resolve())
    validation = prevalidate_backbone_checkpoint(settings.checkpoint, settings.model_key)
    backbone_identity: dict[str, object] | None = None
    if backbone is not None:
        backbone_identity = {
            "feature_channels": list(getattr(backbone, "feature_channels", ())),
            "classifier_parameters": getattr(backbone, "classifier_parameters", None),
            "pretrained_provenance": _json_safe(getattr(backbone, "pretrained_provenance", {})),
        }
    return {
        "schema": SCHEMA,
        "checkpoint_sha256": validation["file_sha256"],
        "checkpoint_prevalidation": validation,
        "dataset": dataset_identity(settings.data_root),
        "settings": config,
        "optimizer": asdict(plan),
        "backbone": backbone_identity,
        "runtime_source_sha256": ({name: sha256_file(path) for name, path in runtime_sources.items()}
                                  if runtime_sources is not None else {"engine": sha256_file(Path(__file__))}),
        "runtime_versions": runtime_versions(),
    }


def contract_hash(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(contract)).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_json(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


class UpdateScheduler:
    """Scheduler indexed exclusively by successful optimizer updates."""

    def __init__(self, optimizer: torch.optim.Optimizer, plan: OptimizerPlan, total_updates: int):
        if total_updates <= 0:
            raise ValueError("total_updates must be positive")
        self.optimizer, self.plan, self.total_updates, self.update_count = optimizer, plan, total_updates, 0
        self.epoch = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._apply()

    def _factor(self, update: int) -> float:
        if update < self.plan.warmup_updates:
            return (update + 1) / max(1, self.plan.warmup_updates)
        if self.plan.drop_epochs:
            return 1.0
        remaining = max(1, self.total_updates - self.plan.warmup_updates)
        progress = min(1.0, (update - self.plan.warmup_updates) / remaining)
        return (1.0 - progress) ** self.plan.polynomial_power

    def _apply(self) -> None:
        drops = sum(self.epoch >= value for value in self.plan.drop_epochs)
        factor = self._factor(self.update_count) * self.plan.drop_factor**drops
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        self.update_count += 1
        self._apply()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._apply()

    def state_dict(self) -> dict[str, object]:
        return {"total_updates": self.total_updates, "update_count": self.update_count, "epoch": self.epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["total_updates"]) != self.total_updates:
            raise ValueError("checkpoint has a different total optimizer-update count")
        self.update_count = int(state["update_count"])
        self.epoch = int(state.get("epoch", 0))
        self.base_lrs = list(state["base_lrs"])  # type: ignore[arg-type]
        self._apply()


def build_optimizer(model: nn.Module, plan: OptimizerPlan, device: torch.device, fused: bool = False) -> torch.optim.AdamW:
    kwargs: dict[str, object] = {"lr": plan.learning_rate, "weight_decay": plan.weight_decay}
    if fused and device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def validate_device(device_name: str) -> torch.device:
    if device_name not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("device must be cpu, cuda, or cuda:0; arbitrary CUDA indices are disallowed")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def cuda_is_idle(device: torch.device) -> bool:
    """Conservative selected-GPU guard; unknown telemetry is treated as busy."""
    if device.type != "cuda":
        return True
    try:
        inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        rows = [tuple(piece.strip() for piece in line.split(",", 1)) for line in inventory.stdout.splitlines() if line.strip()]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        logical_index = torch.cuda.current_device() if device.index is None else device.index
        if visible is not None and visible.strip() not in {"", "-1"}:
            tokens = [item.strip() for item in visible.split(",")]
            if logical_index >= len(tokens):
                return False
            selected = tokens[logical_index]
            matches = [uuid for index, uuid in rows if selected == index or uuid.startswith(selected)]
        else:
            matches = [uuid for index, uuid in rows if index == str(logical_index)]
        if len(matches) != 1:
            return False
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    own_pid, selected_uuid = str(os.getpid()), matches[0]
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2 and parts[0] != own_pid and parts[1] == selected_uuid:
            return False
    return True


def require_idle_cuda(device: torch.device, allow_busy: bool) -> None:
    if device.type == "cuda" and not allow_busy and not cuda_is_idle(device):
        raise RuntimeError("CUDA is occupied; pass --allow-busy only for an approved minimal smoke/benchmark")


def _move(value: Any, device: torch.device, channels_last: bool = False) -> Any:
    if isinstance(value, Tensor):
        result = value.to(device, non_blocking=True)
        if channels_last and result.ndim == 4:
            result = result.contiguous(memory_format=torch.channels_last)
        return result
    if isinstance(value, Mapping):
        return {key: _move(item, device, channels_last) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device, channels_last) for item in value)
    if isinstance(value, list):
        return [_move(item, device, channels_last) for item in value]
    return value


def split_batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, Mapping):
        target = batch.get("target", batch.get("mask", batch.get("labels")))
        inputs = batch.get("image", batch.get("images", batch.get("inputs")))
        if inputs is None or target is None:
            raise ValueError("mapping batches need image(s)/inputs and target/mask/labels")
        return inputs, target
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("batches must be (inputs, targets) or a supported mapping")


def batch_size(batch: Any) -> int:
    inputs, _ = split_batch(batch)
    if isinstance(inputs, Tensor):
        return int(inputs.shape[0])
    if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], Tensor):
        return len(inputs)
    if isinstance(inputs, Mapping):
        first = next(iter(inputs.values()))
        if isinstance(first, Tensor):
            return int(first.shape[0])
    raise ValueError("could not infer batch size")


def default_loss(outputs: Any, targets: Any) -> Tensor:
    # torchvision detection models return their already-reduced loss mapping
    # when called with targets in training mode.
    if isinstance(outputs, Mapping) and outputs and all(str(key).startswith("loss") for key in outputs):
        values = list(outputs.values())
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError("detection loss mapping must contain tensors")
        return sum((value.float() for value in values), torch.zeros((), device=values[0].device))
    logits = outputs["logits"] if isinstance(outputs, Mapping) and "logits" in outputs else outputs
    if not isinstance(logits, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("default loss requires tensor logits and targets")
    if logits.ndim == 4 and targets.ndim == 3 and logits.shape[-2:] != targets.shape[-2:]:
        logits = torch.nn.functional.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    if logits.ndim >= 3 and targets.ndim == logits.ndim - 1:
        # PyTorch's mean-reduced cross entropy is NaN when every label is its
        # ignore index.  A fully void crop has a well-defined contribution of
        # zero, including zero gradient for both the primary and auxiliary
        # segmentation heads.  Do not special-case non-finite logits: their
        # NaN/Inf must remain visible to the training finite-loss guard.
        all_ignored = not bool(torch.any(targets != 255))
        if all_ignored:
            loss = logits.float().sum() * 0.0
        else:
            loss = torch.nn.functional.cross_entropy(logits.float(), targets.long(), ignore_index=255)
        if isinstance(outputs, Mapping) and isinstance(outputs.get("aux_logits"), Tensor):
            auxiliary = outputs["aux_logits"]
            if all_ignored:
                loss = loss + 0.4 * (auxiliary.float().sum() * 0.0)
            else:
                if auxiliary.shape[-2:] != targets.shape[-2:]:
                    auxiliary = torch.nn.functional.interpolate(auxiliary, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss + 0.4 * torch.nn.functional.cross_entropy(auxiliary.float(), targets.long(), ignore_index=255)
        return loss
    return torch.nn.functional.cross_entropy(logits.float(), targets.long())


def model_outputs(model: nn.Module, inputs: Any, targets: Any) -> Any:
    """Call torchvision detection models with targets, all other heads normally."""
    if isinstance(targets, (list, tuple)) and targets and isinstance(targets[0], Mapping):
        return model(list(inputs), list(targets))
    return model(inputs)


@dataclass(slots=True)
class TrainProgress:
    epoch: int = 0
    batch_in_epoch: int = 0
    micro_steps: int = 0
    optimizer_updates: int = 0


def _require_finite_gradients(model: nn.Module, micro_step: int) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient for {name} at micro step {micro_step}")


def _loss_diagnostics(outputs: Any, targets: Any) -> str:
    """Return small, non-image diagnostics for a failed dense loss check."""

    if isinstance(targets, Tensor):
        valid_pixels = int((targets != 255).sum().item())
        target_pixels = int(targets.numel())
    else:
        valid_pixels = None
        target_pixels = None
    logits = outputs.get("logits") if isinstance(outputs, Mapping) else outputs
    logits_finite = bool(torch.isfinite(logits).all().item()) if isinstance(logits, Tensor) else None
    return (
        f"valid_pixels={valid_pixels} target_pixels={target_pixels} "
        f"logits_finite={logits_finite}"
    )


def checkpoint_payload(
    model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: UpdateScheduler,
    progress: TrainProgress, contract: Mapping[str, object], loader_generator: torch.Generator | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SCHEMA, "contract_hash": contract_hash(contract), "contract": dict(contract),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "progress": asdict(progress), "rng": rng_state(),
    }
    if loader_generator is not None:
        payload["loader_generator_state"] = loader_generator.get_state()
    return payload


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: UpdateScheduler,
    contract: Mapping[str, object], loader_generator: torch.Generator | None = None,
) -> TrainProgress:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or payload.get("contract_hash") != contract_hash(contract):
        raise ValueError("checkpoint runtime contract does not match this run")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    if loader_generator is not None and "loader_generator_state" in payload:
        loader_generator.set_state(payload["loader_generator_state"])
    return TrainProgress(**payload["progress"])


def train_batches(
    model: nn.Module, batches: Iterable[Any], optimizer: torch.optim.Optimizer, scheduler: UpdateScheduler,
    device: torch.device, accumulation_steps: int, *, progress: TrainProgress | None = None,
    max_updates: int | None = None, bf16: bool = True, channels_last: bool = False,
    physical_batch_size: int | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_update: Callable[[TrainProgress], None] | None = None,
    loss_fn: Callable[[Any, Any], Tensor] = default_loss,
) -> dict[str, float | int | bool]:
    """Train an iterable, with an exact correction for a short final window.

    Each loss is weighted by its actual image count.  Immediately before an
    update, gradients are normalized by the actual number of images in the
    window; this handles both a short final window and a short final batch.
    """
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    progress = progress or TrainProgress()
    optimizer.zero_grad(set_to_none=True)
    if progress.optimizer_updates >= scheduler.total_updates or (max_updates is not None and progress.optimizer_updates >= max_updates):
        return {"loss": 0.0, "updates": progress.optimizer_updates, "micro_steps": progress.micro_steps, "interrupted": False}
    window = 0
    window_examples = 0
    losses: list[float] = []
    interrupted = False
    iterator = iter(batches)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            batch = None
        if batch is None:
            if window:
                assert physical_batch_size is not None
                correction = accumulation_steps * physical_batch_size / window_examples
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
                _require_finite_gradients(model, progress.micro_steps)
                optimizer.step()
                scheduler.step()
                progress.optimizer_updates += 1
                optimizer.zero_grad(set_to_none=True)
                if on_update is not None:
                    on_update(progress)
                interrupted = stop_requested is not None and stop_requested()
            break
        inputs, targets = split_batch(_move(batch, device, channels_last))
        examples = batch_size(batch)
        if physical_batch_size is None:
            physical_batch_size = examples
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 and device.type == "cuda" else nullcontext()
        with context:
            outputs = model_outputs(model, inputs, targets)
            # Reductions are explicitly FP32 in default_loss and required of custom loss functions.
            loss = loss_fn(outputs, targets).float()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at micro step {progress.micro_steps}; "
                f"{_loss_diagnostics(outputs, targets)}"
            )
        (loss * examples / (physical_batch_size * accumulation_steps)).backward()
        losses.append(float(loss.detach().cpu()))
        progress.micro_steps += 1
        progress.batch_in_epoch += 1
        window += 1
        window_examples += examples
        if window == accumulation_steps:
            correction = accumulation_steps * physical_batch_size / window_examples
            if correction != 1.0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            _require_finite_gradients(model, progress.micro_steps)
            optimizer.step()
            scheduler.step()
            progress.optimizer_updates += 1
            optimizer.zero_grad(set_to_none=True)
            if on_update is not None:
                on_update(progress)
            window = 0
            window_examples = 0
            interrupted = stop_requested is not None and stop_requested()
            if interrupted or progress.optimizer_updates >= scheduler.total_updates or (max_updates is not None and progress.optimizer_updates >= max_updates):
                break
    return {"loss": float(sum(losses) / max(1, len(losses))), "updates": progress.optimizer_updates, "micro_steps": progress.micro_steps, "interrupted": interrupted}


def benchmark_batches(
    model: nn.Module, batches: Iterable[Any], optimizer: torch.optim.Optimizer, scheduler: UpdateScheduler,
    device: torch.device, accumulation_steps: int, *, steps: int, bf16: bool = True,
    channels_last: bool = False, physical_batch_size: int | None = None,
    loss_fn: Callable[[Any, Any], Tensor] = default_loss,
) -> dict[str, float | int | bool]:
    """Time real loader batches.  The caller must use disposable model weights."""
    if steps <= 0:
        raise ValueError("benchmark steps must be positive")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    iterator = iter(batches)
    timings: list[float] = []
    loss_finite = True
    # Cold is the first true update; warm excludes it.
    for index in range(steps):
        start = time.perf_counter()
        try:
            outcome = train_batches(model, [next(iterator) for _ in range(accumulation_steps)], optimizer, scheduler, device, accumulation_steps, max_updates=index + 1, bf16=bf16, channels_last=channels_last, physical_batch_size=physical_batch_size, loss_fn=loss_fn)
        except StopIteration as exc:
            raise RuntimeError("loader was shorter than benchmark request") from exc
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        loss_finite = loss_finite and math.isfinite(float(outcome["loss"]))
    warm = timings[1:] or timings
    sample_count = steps * accumulation_steps
    result: dict[str, float | int | bool] = {
        "updates": steps, "cold_sec_per_update": timings[0], "warm_sec_per_update": sum(warm) / len(warm),
        "sec_per_update": sum(timings) / steps, "sec_per_microbatch": sum(timings) / sample_count,
        "throughput_microbatches_per_sec": sample_count / sum(timings),
        "loss_finite": loss_finite,
        "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
    }
    if device.type == "cuda":
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return result
