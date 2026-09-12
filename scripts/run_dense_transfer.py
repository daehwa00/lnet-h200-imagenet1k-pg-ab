#!/usr/bin/env python3
"""Safe CLI for dense transfer experiments.

``train`` is intentionally difficult to invoke: it requires a ready queue
contract and an explicit confirmation token.  Smoke and benchmark runs never
write a production-resume checkpoint.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import signal
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from dense_transfer.engine import (
    VALID_MODELS, VALID_TASKS, OptimizerPlan, RunSettings, TrainProgress, UpdateScheduler,
    atomic_json, atomic_torch, benchmark_batches, build_optimizer, checkpoint_payload, contract_hash,
    default_loss, default_optimizer_plan, load_checkpoint, require_idle_cuda, runtime_contract,
    set_seed, train_batches, validate_device,
)


TRAIN_CONFIRMATION = "DENSE_TRANSFER_TRAIN"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--task", choices=VALID_TASKS, required=True)
    command.add_argument("--model", dest="model_key", choices=VALID_MODELS, required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--source-root", type=Path)
    command.add_argument("--data-root", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--mode", choices=("inspect", "smoke", "benchmark", "train", "evaluate"), default="inspect")
    command.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cuda")
    command.add_argument("--physical-batch-size", type=int, default=2)
    command.add_argument("--effective-batch-size", type=int, default=16)
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--no-bf16", action="store_true")
    command.add_argument("--compile-backbone", action="store_true", help="opt in; eager is the default")
    command.add_argument("--checkpoint-backbone-blocks", action="store_true", help="opt in for VA/ConvNeXt activation checkpointing; TinyViM rejects it")
    command.add_argument("--channels-last", action="store_true")
    command.add_argument("--fused-adamw", action="store_true")
    command.add_argument("--allow-busy", action="store_true", help="approved minimal smoke/benchmark only")
    command.add_argument("--steps", type=int, default=3, help="capped optimizer updates for smoke/benchmark")
    command.add_argument("--epochs", type=int)
    command.add_argument("--max-updates", type=int)
    command.add_argument("--checkpoint-interval-updates", type=int, default=200)
    command.add_argument("--resume", type=Path)
    command.add_argument("--confirm-training", metavar="TOKEN")
    return command


def settings_from_args(args: argparse.Namespace) -> RunSettings:
    return RunSettings(
        task=args.task, model_key=args.model_key, checkpoint=args.checkpoint, source_root=args.source_root,
        data_root=args.data_root, output_root=args.output_root, device=args.device,
        physical_batch_size=args.physical_batch_size, effective_batch_size=args.effective_batch_size,
        workers=args.workers, bf16=not args.no_bf16, compiled_backbone=args.compile_backbone,
        channels_last=args.channels_last, fused_adamw=args.fused_adamw,
        checkpoint_backbone_blocks=args.checkpoint_backbone_blocks,
    )


def plan_from_args(args: argparse.Namespace) -> OptimizerPlan:
    plan = default_optimizer_plan(args.task)
    if args.epochs is None:
        return plan
    return OptimizerPlan(
        epochs=args.epochs, learning_rate=plan.learning_rate, weight_decay=plan.weight_decay,
        warmup_updates=plan.warmup_updates, polynomial_power=plan.polynomial_power,
        drop_epochs=plan.drop_epochs, drop_factor=plan.drop_factor,
    )


def _call_supported(function: Callable[..., Any], **kwargs: Any) -> Any:
    signature = __import__("inspect").signature(function)
    if any(item.kind is item.VAR_KEYWORD for item in signature.parameters.values()):
        return function(**kwargs)
    return function(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def build_loaders(data: Any, settings: RunSettings, device: torch.device) -> tuple[Any, Any]:
    """Use a data-owned loader factory when present, else its documented datasets."""
    options = {
        "task": settings.task, "data_root": settings.data_root,
        "physical_batch_size": settings.physical_batch_size, "batch_size": settings.physical_batch_size,
        "workers": settings.workers, "pin_memory": device.type == "cuda",
        "persistent_workers": settings.workers > 0, "seed": settings.seed,
    }
    if hasattr(data, "build_loaders"):
        loaders = _call_supported(data.build_loaders, **options)
        if isinstance(loaders, dict):
            return loaders["train"], loaders.get("validation", loaders.get("val"))
        return loaders.train, getattr(loaders, "validation", getattr(loaders, "val", None))
    generator = torch.Generator().manual_seed(settings.seed)
    collate = data.ade_collate if settings.task == "ade20k" else data.coco_collate
    train = data.build_dataset(settings.task, settings.data_root, "train", train=True)
    validation = data.build_dataset(settings.task, settings.data_root, "val", train=False)
    common = {"num_workers": settings.workers, "pin_memory": device.type == "cuda",
              "persistent_workers": settings.workers > 0, "collate_fn": collate}
    train_loader = DataLoader(train, batch_size=settings.physical_batch_size, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation, batch_size=settings.physical_batch_size, shuffle=False, **common)
    # Kept with the loader so a future data factory can expose the same optional
    # state without widening the CLI contract.
    setattr(train_loader, "dense_transfer_generator", generator)
    return train_loader, validation_loader


def build_runtime(settings: RunSettings, plan: OptimizerPlan) -> tuple[torch.nn.Module, Any, Any, torch.optim.AdamW, UpdateScheduler, dict[str, object], torch.device]:
    device = validate_device(settings.device)
    data = importlib.import_module("dense_transfer.data")
    backbones = importlib.import_module("dense_transfer.backbones")
    models = importlib.import_module("dense_transfer.models")
    backbone = _call_supported(
        backbones.build_backbone, model_key=settings.model_key, checkpoint=settings.checkpoint,
        source_root=settings.source_root, compiled=settings.compiled_backbone,
        checkpoint_blocks=settings.checkpoint_backbone_blocks,
    )
    model = _call_supported(models.build_task_model, task=settings.task, backbone=backbone)
    if settings.compiled_backbone:
        # Compile only the transferred feature extractor, never untrusted task glue.
        model.backbone = torch.compile(model.backbone)  # type: ignore[attr-defined]
    if settings.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    train_loader, val_loader = build_loaders(data, settings, device)
    if train_loader is None:
        raise RuntimeError("dense_transfer.data.build_loaders did not provide a train loader")
    updates_per_epoch = (len(train_loader) + settings.accumulation_steps - 1) // settings.accumulation_steps
    total_updates = 80_000 if settings.task == "ade20k" and plan.epochs == 0 else updates_per_epoch * plan.epochs
    optimizer = build_optimizer(model, plan, device, settings.fused_adamw)
    scheduler = UpdateScheduler(optimizer, plan, total_updates)
    source_modules = {"engine": Path(__file__).with_name("dense_transfer") / "engine.py"}
    metrics_module = importlib.import_module("dense_transfer.metrics")
    for name, module in (("backbones", backbones), ("models", models), ("data", data), ("metrics", metrics_module)):
        if module.__file__ is not None:
            source_modules[name] = Path(module.__file__)
    return model, train_loader, val_loader, optimizer, scheduler, runtime_contract(settings, plan, backbone, source_modules), device


def readiness_path(settings: RunSettings) -> Path:
    return settings.output_root / "queue" / "readiness.json"


def require_train_permission(settings: RunSettings, contract: dict[str, object], token: str | None) -> None:
    if token != TRAIN_CONFIRMATION:
        raise RuntimeError(f"train requires --confirm-training {TRAIN_CONFIRMATION}")
    path = readiness_path(settings)
    try:
        readiness = json.loads(path.read_text())
    except OSError as exc:
        raise RuntimeError(f"train requires completed queue readiness: {path}") from exc
    if readiness.get("status") != "ready" or readiness.get("contract_hash") != contract_hash(contract):
        raise RuntimeError("queue readiness is incomplete or does not match this runtime contract")


def evaluate(model: torch.nn.Module, loader: Any, device: torch.device, data_module: Any, settings: RunSettings) -> dict[str, object]:
    if loader is None:
        raise RuntimeError("evaluation requested but no validation loader was supplied")
    if hasattr(data_module, "evaluate"):
        result = _call_supported(data_module.evaluate, model=model, loader=loader, device=device, task=settings.task, bf16=settings.bf16)
        return dict(result) if isinstance(result, dict) else {"result": result}
    metrics = importlib.import_module("dense_transfer.metrics")
    model.eval()
    with torch.inference_mode():
        if settings.task == "ade20k":
            meter = metrics.SegmentationMetric(device="cpu")
            for images, targets, *_metadata in loader:
                outputs = model(images.to(device, non_blocking=True))
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                logits = torch.nn.functional.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                meter.update(logits.cpu(), targets.cpu())
            return dict(meter.compute())
        evaluator = metrics.COCOEvaluator(loader.dataset)
        for images, targets in loader:
            predictions = model([image.to(device, non_blocking=True) for image in images])
            cpu_predictions = [{key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in item.items()} for item in predictions]
            evaluator.update(cpu_predictions, image_ids=[int(target["image_id"].item()) for target in targets])
    return dict(evaluator.compute())


def _install_stop_handlers() -> tuple[Callable[[], bool], Callable[[], None]]:
    """Turn SIGINT/SIGTERM into a cooperative stop at the next update boundary."""
    requested = [False]
    previous = {kind: signal.getsignal(kind) for kind in (signal.SIGINT, signal.SIGTERM)}

    def request_stop(_signum: int, _frame: Any) -> None:
        requested[0] = True

    for kind in previous:
        signal.signal(kind, request_stop)

    def restore() -> None:
        for kind, handler in previous.items():
            signal.signal(kind, handler)

    return lambda: requested[0], restore


def run(args: argparse.Namespace) -> dict[str, object]:
    settings, plan = settings_from_args(args), plan_from_args(args)
    if args.mode == "inspect":
        # Inspect is read-only and deliberately does not import external model code.
        contract = runtime_contract(settings, plan)
        return {"mode": "inspect", "contract_hash": contract_hash(contract), "optimizer": contract["optimizer"]}
    if args.mode in {"smoke", "benchmark"} and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.mode == "evaluate" and args.resume is None:
        raise RuntimeError("evaluate requires --resume pointing to a trained task checkpoint")
    # This must precede model construction: a CUDA model allocation itself is
    # already interference with an occupied device.
    requested_device = validate_device(settings.device)
    require_idle_cuda(requested_device, args.allow_busy if args.mode in {"smoke", "benchmark"} else False)
    set_seed(settings.seed)
    model, train_loader, val_loader, optimizer, scheduler, contract, device = build_runtime(settings, plan)
    if args.mode == "train":
        require_train_permission(settings, contract, args.confirm_training)
    if args.mode == "evaluate":
        load_checkpoint(args.resume, model, optimizer, scheduler, contract, getattr(train_loader, "dense_transfer_generator", None))
        return {"mode": "evaluate", **evaluate(model, val_loader, device, importlib.import_module("dense_transfer.data"), settings)}
    if args.mode == "benchmark":
        # This object is new and unresumed.  Benchmark output cannot be used as a training checkpoint.
        metrics = benchmark_batches(model, train_loader, optimizer, scheduler, device, settings.accumulation_steps, steps=args.steps, bf16=settings.bf16, channels_last=settings.channels_last, physical_batch_size=settings.physical_batch_size)
        target = settings.output_root / "benchmarks" / f"{settings.task}-{settings.model_key}.json"
        atomic_json(target, {"contract": contract, "metrics": metrics})
        return {"mode": "benchmark", "result_path": str(target), **metrics}
    if args.mode == "smoke":
        metrics = train_batches(model, itertools.islice(train_loader, args.steps * settings.accumulation_steps), optimizer, scheduler, device, settings.accumulation_steps, max_updates=args.steps, bf16=settings.bf16, channels_last=settings.channels_last, physical_batch_size=settings.physical_batch_size)
        target = settings.output_root / "smoke" / f"{settings.task}-{settings.model_key}.json"
        atomic_json(target, {"contract": contract, "metrics": metrics})
        return {"mode": "smoke", "result_path": str(target), **metrics}
    progress = TrainProgress()
    if args.resume is not None:
        progress = load_checkpoint(
            args.resume, model, optimizer, scheduler, contract,
            getattr(train_loader, "dense_transfer_generator", None),
        )
    epochs = plan.epochs if plan.epochs else 10**9
    update_cap = scheduler.total_updates if args.max_updates is None else min(args.max_updates, scheduler.total_updates)
    metrics: dict[str, float | int] = {}
    checkpoint_target = settings.output_root / "checkpoints" / "last.pt"
    progress_status = settings.output_root / "status" / "progress.json"
    interval = getattr(args, "checkpoint_interval_updates", 200)
    if interval <= 0:
        raise ValueError("--checkpoint-interval-updates must be positive")

    def save_checkpoint(state: str) -> None:
        atomic_torch(checkpoint_target, checkpoint_payload(model, optimizer, scheduler, progress, contract, getattr(train_loader, "dense_transfer_generator", None)))
        atomic_json(progress_status, {"state": state, "progress": {"optimizer_updates": progress.optimizer_updates, "epoch": progress.epoch, "batch_in_epoch": progress.batch_in_epoch}, "checkpoint": str(checkpoint_target)})

    stop_requested, restore_handlers = _install_stop_handlers()
    paused = False
    try:
        for epoch in range(progress.epoch, epochs):
            if progress.optimizer_updates >= scheduler.total_updates or stop_requested():
                paused = stop_requested()
                break
            scheduler.set_epoch(epoch)
            generator = getattr(train_loader, "dense_transfer_generator", None)
            if generator is not None:
                generator.manual_seed(settings.seed + epoch)
            iterator = itertools.islice(train_loader, progress.batch_in_epoch, None) if epoch == progress.epoch else train_loader
            if epoch != progress.epoch:
                progress.batch_in_epoch = 0

            def periodic(current: TrainProgress) -> None:
                if current.optimizer_updates % interval == 0:
                    save_checkpoint("running")

            outcome = train_batches(model, iterator, optimizer, scheduler, device, settings.accumulation_steps, progress=progress, max_updates=update_cap, bf16=settings.bf16, channels_last=settings.channels_last, physical_batch_size=settings.physical_batch_size, stop_requested=stop_requested, on_update=periodic)
            metrics = {key: value for key, value in outcome.items() if key != "interrupted"}
            epoch_finished = progress.batch_in_epoch >= len(train_loader)
            paused = bool(outcome["interrupted"])
            capped = progress.optimizer_updates >= update_cap
            if epoch_finished:
                progress.epoch = epoch + 1
                progress.batch_in_epoch = 0
            else:
                progress.epoch = epoch
            save_checkpoint("paused" if paused or capped and progress.optimizer_updates < scheduler.total_updates else "running")
            if paused or capped:
                break
    finally:
        restore_handlers()

    full_schedule = progress.optimizer_updates >= scheduler.total_updates or (plan.epochs > 0 and progress.epoch >= plan.epochs and not paused)
    if not full_schedule:
        state = "paused" if paused else "stopped"
        atomic_json(settings.output_root / "status" / "final.json", {"state": state, "contract": contract, "checkpoint": str(checkpoint_target), "progress": {"optimizer_updates": progress.optimizer_updates, "epoch": progress.epoch}, "train": metrics})
        return {"mode": "train", "state": state, "checkpoint": str(checkpoint_target), **metrics}

    endpoint = evaluate(model, val_loader, device, importlib.import_module("dense_transfer.data"), settings)
    status_path = settings.output_root / "status" / "final.json"
    checkpoint_path = checkpoint_target if checkpoint_target.is_file() else args.resume
    atomic_json(status_path, {
        "state": "completed", "contract": contract, "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "progress": {"optimizer_updates": progress.optimizer_updates, "epoch": progress.epoch},
        "train": metrics, "evaluation": endpoint,
    })
    return {"mode": "train", "checkpoint": None if checkpoint_path is None else str(checkpoint_path), "status_path": str(status_path), **metrics, **{f"eval_{key}": value for key, value in endpoint.items()}}


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(run(args), sort_keys=True, default=str))


if __name__ == "__main__":
    main()
