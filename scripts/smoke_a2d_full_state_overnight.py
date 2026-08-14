#!/usr/bin/env python3
"""Real-ImageNet eager, compile, and exact-resume smoke for one full-state variant."""

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import run_a2d_deep4_p96_full_state_overnight_imagenet100 as runner
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from lnet.pac_full_state_operators import GroupedPhaseGatedComplexFFN
from lnet.pac_full_state_overnight import StructuredFullStateTransition
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN

if TYPE_CHECKING:
    from collections.abc import Mapping

OOM_EXIT_CODE = 42


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=runner.VARIANTS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--compile-mode", default=runner.COMPILE_MODE)
    parser.add_argument(
        "--phase",
        choices=("orchestrate", "prepare", "resume"),
        default="orchestrate",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _recipe(batch_size: int, compile_mode: str) -> dict[str, Any]:
    return {
        "epochs": 100,
        "batch_size": batch_size,
        "gradient_accumulation_steps": 128 // batch_size,
        "effective_batch_size": 128,
        "optimizer": "AdamW (fused, pole-aware parameter groups)",
        "fused_optimizer": True,
        "learning_rate": 3.0e-3,
        "modal_learning_rate_multiplier": 1.0 / 3.0,
        "pole_geometry_learning_rate_multiplier": 0.1,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "schedule": "warmup plus cosine",
        "label_smoothing": 0.1,
        "mixup_alpha": 0.8,
        "precision": "bfloat16",
        "compile_mode": compile_mode,
        "channels_last": True,
    }


def _model(variant: str, recipe: dict[str, Any], device: torch.device) -> nn.Module:
    config = runner.base.control.stemres.uniform.base.PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = runner._build(variant, config).to(device)
    ramp = runner.base.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    return source._prepare_model(model, recipe)


def _loader(data_root: Path, count: int) -> DataLoader[Any]:
    harness = runner.base.control.stemres.uniform.base.heads.harness
    train_transform, _ = harness._transforms()
    dataset = ImageFolder(data_root / "train", transform=train_transform)
    return DataLoader(dataset, batch_size=count, shuffle=False, num_workers=0, drop_last=True)


def _to_device(batch: object, device: torch.device) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        message = "ImageNet smoke loader returned an invalid batch"
        raise TypeError(message)
    inputs, targets = batch
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        message = "ImageNet smoke loader batch does not contain tensors"
        raise TypeError(message)
    return (
        inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last),
        targets.to(device, non_blocking=True),
    )


def _batch(data_root: Path, count: int, device: torch.device) -> tuple[Tensor, Tensor]:
    return _to_device(next(iter(_loader(data_root, count))), device)


def _mixed_loss(output: object, targets: Tensor) -> Tensor:
    if not isinstance(output, tuple) or len(output) != 5:
        message = "full-state smoke requires the established five-output classifier"
        raise RuntimeError(message)
    permutation = torch.arange(targets.numel() - 1, -1, -1, device=targets.device)
    mixing = 0.7

    def mixed_cross_entropy(logits: Tensor) -> Tensor:
        return mixing * functional.cross_entropy(
            logits.float(),
            targets,
            label_smoothing=0.1,
        ) + (1.0 - mixing) * functional.cross_entropy(
            logits.float(),
            targets[permutation],
            label_smoothing=0.1,
        )

    return mixed_cross_entropy(output[0]) + 0.5 * mixed_cross_entropy(output[1])


def _assert_finite_output(output: object) -> None:
    if (
        not isinstance(output, tuple)
        or len(output) != 5
        or not all(isinstance(value, Tensor) for value in output)
    ):
        message = "full-state smoke output must contain exactly five tensors"
        raise TypeError(message)
    if any(not torch.isfinite(value).all() for value in output):
        message = "full-state smoke produced a non-finite output"
        raise FloatingPointError(message)


def _assert_finite(
    model: nn.Module,
    output: object,
    optimizer: torch.optim.Optimizer,
) -> None:
    _assert_finite_output(output)
    active_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not active_gradients or any(not torch.isfinite(value).all() for value in active_gradients):
        message = "full-state smoke produced a missing or non-finite active gradient"
        raise FloatingPointError(message)
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        message = f"full-state smoke has inactive trainable parameters: {missing_gradients[:8]}"
        raise RuntimeError(message)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            message = f"non-finite model parameter: {name}"
            raise FloatingPointError(message)
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and not torch.isfinite(value).all():
                message = "full-state smoke produced non-finite optimizer state"
                raise FloatingPointError(message)


def _assert_pg_diagnostics(model: nn.Module) -> None:
    observed = 0
    for module in model.modules():
        if isinstance(module, (PhaseGatedComplexFFN, GroupedPhaseGatedComplexFFN)):
            metrics = module.diagnostic_metrics()
            gate_metrics = {
                name: value for name, value in metrics.items() if name.startswith("gate_")
            }
            if not gate_metrics or any(not math.isfinite(value) for value in gate_metrics.values()):
                message = f"non-finite Phase-Gated diagnostics in {type(module).__name__}"
                raise FloatingPointError(message)
            if float(gate_metrics["gate_mean"]) <= 0.0:
                message = f"Phase-Gated block did not execute: {type(module).__name__}"
                raise RuntimeError(message)
            observed += 1
    if observed == 0:
        message = "full-state smoke did not find a Phase-Gated block"
        raise RuntimeError(message)


def _step(
    model: nn.Module,
    runtime: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
) -> float:
    model.train()
    runtime.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime(inputs)
        _assert_finite_output(output)
        loss = _mixed_loss(output, targets)
    if not torch.isfinite(loss):
        message = "full-state smoke produced a non-finite loss"
        raise FloatingPointError(message)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    _assert_finite(model, output, optimizer)
    return float(loss.detach())


def _accumulated_step(
    model: nn.Module,
    runtime: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: DataLoader[Any],
    device: torch.device,
    *,
    group_size: int,
) -> float:
    model.train()
    runtime.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[Tensor] = []
    output: object | None = None
    iterator = iter(batches)
    harness = runner.base.control.stemres.uniform.base.heads.harness
    for _ in range(group_size):
        inputs, targets = _to_device(next(iterator), device)
        harness._begin_cudagraph_step(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = runtime(inputs)
            _assert_finite_output(output)
            loss = _mixed_loss(output, targets)
        if not torch.isfinite(loss):
            message = "capacity accumulation produced a non-finite loss"
            raise FloatingPointError(message)
        (loss / group_size).backward()
        losses.append(loss.detach())
        del inputs, targets
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if output is None:
        message = "capacity accumulation received no batches"
        raise RuntimeError(message)
    _assert_finite(model, output, optimizer)
    return float(torch.stack(losses).mean())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_nested_equal(  # noqa: C901
    expected: object,
    actual: object,
    *,
    name: str,
) -> None:
    if isinstance(expected, Tensor):
        if not isinstance(actual, Tensor) or not torch.equal(expected.cpu(), actual.detach().cpu()):
            message = f"checkpoint restore mismatch at {name}"
            raise RuntimeError(message)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            message = f"checkpoint mapping mismatch at {name}"
            raise RuntimeError(message)
        for key, value in expected.items():
            _assert_nested_equal(value, actual[key], name=f"{name}.{key}")
        return
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            message = f"checkpoint sequence mismatch at {name}"
            raise RuntimeError(message)
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_nested_equal(left, right, name=f"{name}[{index}]")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(expected, actual):
            message = f"checkpoint array mismatch at {name}"
            raise RuntimeError(message)
        return
    if expected != actual:
        message = f"checkpoint scalar mismatch at {name}"
        raise RuntimeError(message)


def _checkpoint(
    path: Path,
    *,
    variant: str,
    seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_generator: torch.Generator,
    mixup_generator: np.random.Generator,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "seed": seed,
            "epoch": 1,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": [{"epoch": 1.0}],
            "training_seconds": 0.0,
            "training_generator_state": training_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
            "mixup_rng_state": mixup_generator.bit_generator.state,
        },
        path,
    )


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return args.output.parent / f"{args.variant}__smoke.pt"


def _prepare_evidence_path(args: argparse.Namespace) -> Path:
    return args.output.parent / "prepare-evidence.json"


def _configure_runtime(args: argparse.Namespace) -> torch.device:
    if args.batch_size not in {32, 64, 128} or 128 % args.batch_size:
        message = "capacity batch must be one of 128, 64, or 32"
        raise ValueError(message)
    if not 0 < args.microbatch_size <= args.batch_size:
        message = "microbatch size must be positive and no larger than capacity batch"
        raise ValueError(message)
    os.environ["LNET_COMPILE_MODE"] = args.compile_mode
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "full-state smoke requires exactly one visible CUDA device"
        raise RuntimeError(message)
    return torch.device("cuda")


def _scheduler(
    optimizer: torch.optim.Optimizer,
    recipe: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler:
    harness = runner.base.control.stemres.uniform.base.heads.harness
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: harness._learning_rate_factor(epoch, int(recipe["epochs"])),
    )


def _enable_grouped_diagnostics(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, StructuredFullStateTransition):
            module.set_diagnostics_enabled(enabled=True)


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    device = _configure_runtime(args)
    recipe = _recipe(args.batch_size, args.compile_mode)
    inputs, targets = _batch(args.data_root, args.microbatch_size, device)
    model = _model(args.variant, recipe, device)
    _enable_grouped_diagnostics(model)
    optimizer = runner._build_optimizer(model, recipe)
    scheduler = _scheduler(optimizer, recipe)
    eager_losses = [_step(model, model, optimizer, inputs, targets) for _ in range(2)]
    _assert_pg_diagnostics(model)
    scheduler.step()
    training_generator = torch.Generator().manual_seed(args.seed)
    mixup_generator = np.random.default_rng(args.seed)
    checkpoint = _checkpoint_path(args)
    _checkpoint(
        checkpoint,
        variant=args.variant,
        seed=args.seed,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_generator=training_generator,
        mixup_generator=mixup_generator,
        global_step=2,
    )
    result: dict[str, object] = {
        "status": "PREPARED",
        "variant": args.variant,
        "signature_sha256": runner.SPECS_BY_VARIANT[args.variant].signature_hash(),
        "prepare_pid": os.getpid(),
        "eager_steps": 2,
        "eager_losses": eager_losses,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    _atomic_json(_prepare_evidence_path(args), result)
    return result


def _load_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        message = f"smoke evidence is not a JSON object: {path}"
        raise TypeError(message)
    return payload


def _verify_restored_state(
    checkpoint: dict[str, object],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_generator: torch.Generator,
    mixup_generator: np.random.Generator,
) -> None:
    _assert_nested_equal(checkpoint["model"], model.state_dict(), name="model")
    _assert_nested_equal(checkpoint["optimizer"], optimizer.state_dict(), name="optimizer")
    _assert_nested_equal(checkpoint["scheduler"], scheduler.state_dict(), name="scheduler")
    _assert_nested_equal(
        checkpoint["training_generator_state"],
        training_generator.get_state(),
        name="training_generator_state",
    )
    _assert_nested_equal(
        checkpoint["torch_rng_state"], torch.get_rng_state(), name="torch_rng_state"
    )
    _assert_nested_equal(
        checkpoint["cuda_rng_state"],
        torch.cuda.get_rng_state_all(),
        name="cuda_rng_state",
    )
    _assert_nested_equal(checkpoint["python_rng_state"], random.getstate(), name="python_rng_state")
    _assert_nested_equal(
        checkpoint["mixup_rng_state"],
        mixup_generator.bit_generator.state,
        name="mixup_rng_state",
    )


def _resume(args: argparse.Namespace) -> dict[str, object]:
    device = _configure_runtime(args)
    prepared = _load_mapping(_prepare_evidence_path(args))
    prepare_pid = prepared.get("prepare_pid")
    if not isinstance(prepare_pid, int) or prepare_pid == os.getpid():
        message = "resume smoke must run in a fresh operating-system process"
        raise RuntimeError(message)
    checkpoint_path = _checkpoint_path(args)
    if prepared.get("checkpoint_sha256") != _sha256(checkpoint_path):
        message = "smoke checkpoint changed between prepare and resume processes"
        raise RuntimeError(message)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        message = "smoke checkpoint payload is invalid"
        raise TypeError(message)
    recipe = _recipe(args.batch_size, args.compile_mode)
    resumed = _model(args.variant, recipe, device)
    resumed_optimizer = runner._build_optimizer(resumed, recipe)
    resumed_scheduler = _scheduler(resumed_optimizer, recipe)
    training_generator = torch.Generator()
    mixup_generator = np.random.default_rng()
    progress: dict[str, int] = {}
    harness = runner.base.control.stemres.uniform.base.heads.harness
    restored = harness._restore_checkpoint(
        checkpoint_path,
        variant=args.variant,
        seed=args.seed,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        training_generator=training_generator,
        mixup_generator=mixup_generator,
        progress=progress,
    )
    if restored[0] != 1 or progress.get("global_step") != 2:
        message = "production checkpoint loader did not restore epoch and global step"
        raise RuntimeError(message)
    _verify_restored_state(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        training_generator=training_generator,
        mixup_generator=mixup_generator,
    )
    harness._restore_optimizer_runtime_options(resumed_optimizer, recipe)
    runtime = harness._build_runtime(resumed, recipe)
    accumulation_steps = 128 // args.batch_size
    compiled_losses = [
        _accumulated_step(
            resumed,
            runtime,
            resumed_optimizer,
            _loader(args.data_root, args.batch_size),
            device,
            group_size=accumulation_steps,
        )
        for _ in range(2)
    ]
    progress["global_step"] += 2
    torch.cuda.synchronize()
    return {
        "schema": "lnet.full_state_overnight.smoke.v3",
        "status": "PASS",
        "variant": args.variant,
        "signature_sha256": runner.SPECS_BY_VARIANT[args.variant].signature_hash(),
        "seed": args.seed,
        "data_root": str(args.data_root.resolve()),
        "source_commit": os.environ.get("LNET_SOURCE_COMMIT", "unknown"),
        "source_fingerprint": os.environ.get("LNET_SOURCE_FINGERPRINT", "unknown"),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "device_identity": os.environ.get("LNET_DEVICE_IDENTITY", "unknown"),
        "dtype": "bfloat16",
        "compile_mode": args.compile_mode,
        "microbatch_size": args.microbatch_size,
        "capacity_batch_size": args.batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "prepare_pid": prepare_pid,
        "resume_pid": os.getpid(),
        "eager_steps": prepared["eager_steps"],
        "compiled_steps": 2,
        "capacity_optimizer_steps": 2,
        "capacity_microbatches": 2 * accumulation_steps,
        "eager_losses": prepared["eager_losses"],
        "compiled_losses": compiled_losses,
        "capacity_loss": compiled_losses[-1],
        "resume_verified": True,
        "exact_state_verified": True,
        "restored_epoch": restored[0],
        "restored_global_step": 2,
        "final_global_step": progress["global_step"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "cudagraphs_active": os.environ.get("LNET_CUDAGRAPHS_ACTIVE") == "1",
    }


def _phase_command(args: argparse.Namespace, phase: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--variant",
        args.variant,
        "--data-root",
        str(args.data_root),
        "--output",
        str(args.output),
        "--batch-size",
        str(args.batch_size),
        "--microbatch-size",
        str(args.microbatch_size),
        "--seed",
        str(args.seed),
        "--compile-mode",
        args.compile_mode,
        "--phase",
        phase,
    ]


def _orchestrate(args: argparse.Namespace) -> dict[str, object]:
    for path in (args.output, _prepare_evidence_path(args), _checkpoint_path(args)):
        path.unlink(missing_ok=True)
    for phase in ("prepare", "resume"):
        result = subprocess.run(_phase_command(args, phase), check=False)  # noqa: S603
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    result = _load_mapping(args.output)
    if result.get("status") != "PASS":
        message = "fresh-process smoke did not produce PASS evidence"
        raise RuntimeError(message)
    return result


def _looks_like_oom(error: BaseException) -> bool:
    current: BaseException | None = error
    for _ in range(8):
        if current is None:
            return False
        if isinstance(current, (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError)):
            return True
        message = repr(current).lower()
        if "out of memory" in message and ("cuda" in message or "cublas" in message):
            return True
        current = current.__cause__ or current.__context__
    return False


def main() -> None:
    args = _arguments()
    try:
        if args.phase == "prepare":
            result = _prepare(args)
        elif args.phase == "resume":
            result = _resume(args)
        else:
            result = _orchestrate(args)
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as error:
        result = {
            "status": "OOM_FAILED",
            "variant": args.variant,
            "capacity_batch_size": args.batch_size,
            "exception": repr(error),
        }
        _atomic_json(args.output, result)
        raise SystemExit(OOM_EXIT_CODE) from error
    except Exception as error:
        if _looks_like_oom(error):
            result = {
                "status": "OOM_FAILED",
                "variant": args.variant,
                "capacity_batch_size": args.batch_size,
                "exception": repr(error),
            }
            _atomic_json(args.output, result)
            raise SystemExit(OOM_EXIT_CODE) from error
        result = {
            "status": "SMOKE_FAILED",
            "variant": args.variant,
            "capacity_batch_size": args.batch_size,
            "exception": repr(error),
        }
        _atomic_json(args.output, result)
        raise
    if args.phase != "prepare":
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
