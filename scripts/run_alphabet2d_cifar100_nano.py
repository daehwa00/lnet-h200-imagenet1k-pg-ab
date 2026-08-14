# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportMissingImports=false, reportMissingTypeArgument=false
"""Train four-scan ALPHABET-2D-Nano and pole-free control on CIFAR-100."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from cifar100_packed_data import (
    build_cifar100_transforms,
    stratified_cifar100_indices,
)
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from lnet.alphabet2d_cifar import CifarNanoConfig, build_cifar_nano

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandbRun

VARIANTS = ("product_four", "pole_free")
SEEDS = (401, 409, 419)
WANDB_MODEL_ALIASES = {
    "product_four": "Product4",
    "pole_free": "PoleFree",
    "resnet20": "ResNet20",
    "resnet32": "ResNet32",
    "mobilenetv2_015x": "MobileNetV2-0.15x",
    "mobilenetv2_025x": "MobileNetV2-0.25x",
}


def _compact_wandb_name(variant: str) -> str:
    """Return a short model label while keeping full provenance in config."""
    return WANDB_MODEL_ALIASES.get(variant, variant.replace("_", "-"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="seal the official test split for validation-only architecture screening",
    )
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    model = CifarNanoConfig()
    return {
        "schema": "lnet.alphabet2d.cifar100_nano.v1",
        "evidence_status": "exploratory Phase-1 internal G4 control",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(model),
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": 3.0e-3,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "augmentation": "RandomCrop(32,pad4)+HFlip+RandAugment(2,9)+RandomErasing",
            "validation": "fixed stratified 5k from CIFAR-100 train",
            "test_selection": False,
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "data_sha256": {
            name: _digest(args.data_root / "cifar-100-python" / name)
            for name in ("train", "test", "meta")
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
            message = "existing CIFAR Nano root has a different immutable contract"
            raise RuntimeError(message)
    else:
        _atomic_json(path, contract)


def _loaders(
    data_root: Path,
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, evaluation_transform = build_cifar100_transforms()
    raw = datasets.CIFAR100(data_root, train=True, download=False)
    train_indices, validation_indices = stratified_cifar100_indices(raw.targets)
    train_dataset = datasets.CIFAR100(
        data_root,
        train=True,
        transform=train_transform,
        download=False,
    )
    evaluation_train = datasets.CIFAR100(
        data_root,
        train=True,
        transform=evaluation_transform,
        download=False,
    )
    test_dataset = datasets.CIFAR100(
        data_root,
        train=False,
        transform=evaluation_transform,
        download=False,
    )
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
        **common,
    )
    validation_loader = DataLoader(
        Subset(evaluation_train, validation_indices),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, validation_loader, test_loader


def _mixed_loss(
    logits: Tensor,
    targets: Tensor,
    permutation: Tensor,
    mixing: float,
) -> Tensor:
    return mixing * functional.cross_entropy(
        logits,
        targets,
        label_smoothing=0.1,
    ) + (1.0 - mixing) * functional.cross_entropy(
        logits,
        targets[permutation],
        label_smoothing=0.1,
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    count = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for batch_inputs, batch_targets in loader:
            inputs = batch_inputs.to(device, non_blocking=True)
            targets = batch_targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss_sum += float(functional.cross_entropy(logits, targets, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == targets).sum())
            count += targets.numel()
    return {"accuracy": correct / count, "cross_entropy": loss_sum / count}


def _learning_rate_factor(epoch: int, epochs: int) -> float:
    if epoch < 5:
        return (epoch + 1) / 5.0
    progress = (epoch - 5) / max(1, epochs - 5)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _restore_checkpoint(
    path: Path,
    *,
    variant: str,
    seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, float, int, dict[str, Tensor] | None, list[dict[str, float]], float]:
    if not path.exists():
        return 0, -1.0, 0, None, [], 0.0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["variant"] != variant or payload["seed"] != seed:
        message = "checkpoint identity does not match requested CIFAR Nano job"
        raise RuntimeError(message)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return (
        int(payload["epoch"]),
        float(payload["best_accuracy"]),
        int(payload["best_epoch"]),
        payload["best_state"],
        payload["history"],
        float(payload["training_seconds"]),
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mixup_generator: np.random.Generator,
    mixup_alpha: float,
) -> dict[str, float]:
    model.train()
    correct = 0
    count = 0
    loss_sum = 0.0
    for batch_inputs, batch_targets in loader:
        inputs = batch_inputs.to(device, non_blocking=True)
        targets = batch_targets.to(device, non_blocking=True)
        permutation = torch.randperm(targets.numel(), device=device)
        mixing = float(mixup_generator.beta(mixup_alpha, mixup_alpha))
        mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
        optimizer.zero_grad(set_to_none=True)
        logits = model(mixed_inputs)
        loss = _mixed_loss(logits, targets, permutation, mixing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(loss.detach()) * targets.numel()
        correct += int((logits.argmax(dim=-1) == targets).sum())
        count += targets.numel()
    return {"loss": loss_sum / count, "mixed_accuracy": correct / count}


def _build_optimizer(
    model: nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
    )


def _runtime_model(model: nn.Module, *, compile_model: bool) -> nn.Module:
    if not compile_model:
        return model
    return cast(
        "nn.Module",
        torch.compile(model, fullgraph=False, dynamic=False),
    )


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
    run_id = hashlib.sha256(run_key.encode()).hexdigest()[:16]
    tracking_root = root / "wandb"
    tracking_root.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        group=os.environ.get("WANDB_GROUP", root.name),
        name=os.environ.get("WANDB_NAME", _compact_wandb_name(variant)),
        id=run_id,
        resume="allow",
        dir=str(tracking_root),
        config={
            "variant": variant,
            "seed": seed,
            "parameters": parameters,
            "model": contract["model"],
            "recipe": contract["recipe"],
            "schema": contract["schema"],
        },
    )


def _run_job(
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    data_root: Path,
    workers: int,
    device: torch.device,
    compile_model: bool,
    evaluate_test: bool,
) -> None:
    result_path = root / "results" / f"{variant}__seed{seed}.json"
    if result_path.exists():
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mixup_generator = np.random.default_rng(seed)
    model = build_cifar_nano(
        variant,
        CifarNanoConfig(**contract["model"]),
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    recipe = contract["recipe"]
    train_loader, validation_loader, test_loader = _loaders(
        data_root,
        batch_size=recipe["batch_size"],
        workers=workers,
        seed=seed,
    )
    optimizer = _build_optimizer(model, recipe)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _learning_rate_factor(epoch, recipe["epochs"]),
    )
    checkpoint_path = root / "checkpoints" / f"{variant}__seed{seed}.pt"
    (
        start_epoch,
        best_accuracy,
        best_epoch,
        best_state,
        history,
        training_seconds,
    ) = _restore_checkpoint(
        checkpoint_path,
        variant=variant,
        seed=seed,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    wandb_run = _initialize_wandb_run(
        root,
        contract,
        variant=variant,
        seed=seed,
        parameters=parameters,
    )
    runtime_model = _runtime_model(model, compile_model=compile_model)
    mixup_generator = np.random.default_rng(seed)
    if start_epoch:
        mixup_generator.beta(
            recipe["mixup_alpha"],
            recipe["mixup_alpha"],
            size=start_epoch * len(train_loader),
        )
    for epoch in range(start_epoch, recipe["epochs"]):
        started = time.perf_counter()
        train = _train_epoch(
            runtime_model,
            train_loader,
            optimizer,
            device=device,
            mixup_generator=mixup_generator,
            mixup_alpha=recipe["mixup_alpha"],
        )
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - started
        validation = _evaluate(runtime_model, validation_loader, device)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train["loss"],
            "train_mixed_accuracy": train["mixed_accuracy"],
            "validation_accuracy": validation["accuracy"],
            "validation_cross_entropy": validation["cross_entropy"],
        }
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": row["epoch"],
                    "learning_rate": row["learning_rate"],
                    "train/loss": row["train_loss"],
                    "train/mixed_accuracy": row["train_mixed_accuracy"],
                    "validation/accuracy": row["validation_accuracy"],
                    "validation/cross_entropy": row["validation_cross_entropy"],
                    "time/training_seconds": training_seconds,
                },
                step=epoch + 1,
            )
        print(  # noqa: T201
            json.dumps(
                {
                    "variant": variant,
                    "epoch": epoch + 1,
                    "train_loss": round(train["loss"], 6),
                    "validation_accuracy": round(validation["accuracy"], 6),
                    "validation_cross_entropy": round(validation["cross_entropy"], 6),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        scheduler.step()
        _atomic_torch(
            checkpoint_path,
            {
                "variant": variant,
                "seed": seed,
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "history": history,
                "training_seconds": training_seconds,
            },
        )
    if best_state is None:
        message = "CIFAR Nano training produced no validation checkpoint"
        raise RuntimeError(message)
    model.load_state_dict(best_state)
    test = _evaluate(runtime_model, test_loader, device) if evaluate_test else None
    result = {
        "variant": variant,
        "seed": seed,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "test": test,
        "training_seconds": training_seconds,
        "complete_training_examples_per_second": (
            recipe["epochs"] * len(train_loader.dataset) / training_seconds
        ),
        "history": history,
    }
    _atomic_json(result_path, result)
    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_validation_accuracy"] = best_accuracy
        wandb_run.summary["training_seconds"] = training_seconds
        if test is not None:
            wandb_run.summary["test_accuracy"] = test["accuracy"]
            wandb_run.summary["test_cross_entropy"] = test["cross_entropy"]
        wandb_run.finish()


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [
        root / "results" / f"{variant}__seed{seed}.json" for variant in VARIANTS for seed in SEEDS
    ]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    means = {
        variant: sum(row["test"]["accuracy"] for row in rows if row["variant"] == variant)
        / len(SEEDS)
        for variant in VARIANTS
    }
    paired = [
        next(
            row["test"]["accuracy"]
            for row in rows
            if row["variant"] == "product_four" and row["seed"] == seed
        )
        - next(
            row["test"]["accuracy"]
            for row in rows
            if row["variant"] == "pole_free" and row["seed"] == seed
        )
        for seed in SEEDS
    ]
    payload = {
        "schema": contract["schema"],
        "mean_test_accuracy": means,
        "paired_product_minus_pole_free": paired,
        "mean_product_minus_pole_free_pp": 100.0 * sum(paired) / len(paired),
        "G4_product_beats_pole_free_1pp": sum(paired) / len(paired) >= 0.01,
        "parameter_counts": {
            variant: sorted({row["parameters"] for row in rows if row["variant"] == variant})
            for variant in VARIANTS
        },
    }
    _atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        message = "CIFAR Nano runner requires CUDA"
        raise RuntimeError(message)
    contract = _contract(args)
    if args.compile_model:
        contract["runtime_compile"] = {
            "backend": "torch.compile",
            "fullgraph": False,
            "dynamic": False,
            "mode": "default",
        }
    args.root.mkdir(parents=True, exist_ok=True)
    _initialize(args.root, contract)
    if args.initialize_only:
        return
    if not set(args.run_seeds) <= set(SEEDS):
        message = "run seeds fall outside the CIFAR Nano contract"
        raise ValueError(message)
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
                compile_model=args.compile_model,
                evaluate_test=not args.skip_test,
            )
    summary = _summarize(args.root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
