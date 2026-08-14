#!/usr/bin/env python3
"""Compiled BF16, optimizer, and fresh-process resume gate for one queue arm."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001, T201
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_pole_backbone_overnight_imagenet100 as runner
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN
from lnet.pac_pole_backbone_ablation import PoleBackboneAblationStage

if TYPE_CHECKING:
    from collections.abc import Mapping

OOM_EXIT_CODE = 42


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=runner.VARIANTS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--compile-mode", default=runner.COMPILE_MODE)
    parser.add_argument(
        "--phase",
        choices=("orchestrate", "prepare", "resume"),
        default="orchestrate",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recipe(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "epochs": 100,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": args.batch_size,
        "fused_optimizer": True,
        "learning_rate": 3.0e-3,
        "modal_learning_rate_multiplier": 1.0 / 3.0,
        "pole_geometry_learning_rate_multiplier": 0.1,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.8,
        "precision": "bfloat16",
        "compile_mode": args.compile_mode,
        "channels_last": True,
    }


def _runtime_parts(
    args: argparse.Namespace,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    spec = runner.SPECS_BY_VARIANT[args.variant]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "overnight smoke requires exactly one visible CUDA device"
        raise RuntimeError(message)
    os.environ["LNET_COMPILE_MODE"] = args.compile_mode
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    runner.control.base.joint.control.base.base._configure_ramp()
    ramp = runner.control.base.joint.control.base.base.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    recipe = _recipe(args)
    model = runner._build(args.variant, config).cuda()
    model = source._prepare_model(model, recipe)
    optimizer = runner._build_optimizer(model, recipe)
    harness = ramp.heads.harness
    harness._configure_compile_runtime(args.output.parent, recipe)
    runtime = harness._build_runtime(model, recipe)
    return model, runtime, optimizer, recipe


def _batch(args: argparse.Namespace) -> tuple[Tensor, Tensor]:
    ramp = runner.control.base.joint.control.base.base.control.control.stemres.uniform.base
    harness = ramp.heads.harness
    train_transform, _ = harness._transforms()
    dataset = ImageFolder(args.data_root / "train", transform=train_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    inputs, targets = next(iter(loader))
    return (
        inputs.cuda().contiguous(memory_format=torch.channels_last),
        targets.cuda(),
    )


def _step(
    model: nn.Module,
    runtime: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ramp = runner.control.base.joint.control.base.base.control.control.stemres.uniform.base
    harness = ramp.heads.harness
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    harness._begin_cudagraph_step(torch.device("cuda"))
    permutation = torch.arange(targets.numel() - 1, -1, -1, device=targets.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime(inputs)
        _, loss, _ = source.heads._training_objective(
            model,
            output,
            targets,
            targets[permutation],
            0.7,
        )
    if not torch.isfinite(loss):
        message = "overnight smoke produced a non-finite loss"
        raise FloatingPointError(message)
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        message = f"overnight smoke has inactive parameters: {missing[:8]}"
        raise RuntimeError(message)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        message = "overnight smoke produced non-finite gradients"
        raise FloatingPointError(message)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def _assert_execution(model: nn.Module) -> None:
    stages = [module for module in model.modules() if isinstance(module, PoleBackboneAblationStage)]
    if len(stages) != 4 or any(stage.diagnostic_updates.item() < 1 for stage in stages):
        message = "overnight smoke did not execute all four pole-memory stages"
        raise RuntimeError(message)
    for stage in stages:
        metrics = stage.diagnostic_metrics()
        if not metrics or any(not math.isfinite(value) for value in metrics.values()):
            message = "overnight smoke produced invalid stage diagnostics"
            raise FloatingPointError(message)
    for block in (module for module in model.modules() if isinstance(module, PhaseGatedComplexFFN)):
        metrics = block.diagnostic_metrics()
        if not metrics or any(not math.isfinite(value) for value in metrics.values()):
            message = "overnight smoke produced invalid Phase-Gated diagnostics"
            raise FloatingPointError(message)


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return args.output.parent / f"{args.variant}__smoke.pt"


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    model, runtime, optimizer, recipe = _runtime_parts(args)
    inputs, targets = _batch(args)
    torch.cuda.reset_peak_memory_stats()
    loss = _step(model, runtime, optimizer, inputs, targets)
    torch.cuda.synchronize()
    _assert_execution(model)
    checkpoint = _checkpoint_path(args)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": args.variant,
            "seed": runner.SPECS_BY_VARIANT[args.variant].seed,
            "epoch": 1,
            "global_step": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "recipe": recipe,
        },
        checkpoint,
    )
    return {
        "status": "PREPARED",
        "prepare_pid": os.getpid(),
        "loss": loss,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def _resume(args: argparse.Namespace) -> dict[str, object]:
    prepared_path = args.output.parent / "prepare.json"
    prepared = json.loads(prepared_path.read_text())
    checkpoint_path = _checkpoint_path(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, runtime, optimizer, _ = _runtime_parts(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if any(
        not torch.equal(expected.cpu(), actual.detach().cpu())
        for expected, actual in zip(
            cast("dict[str, Tensor]", checkpoint["model"]).values(),
            model.state_dict().values(),
            strict=True,
        )
    ):
        message = "fresh-process resume did not restore the exact model state"
        raise RuntimeError(message)
    inputs, targets = _batch(args)
    loss = _step(model, runtime, optimizer, inputs, targets)
    torch.cuda.synchronize()
    _assert_execution(model)
    return {
        "schema": "lnet.pole_backbone_overnight.smoke.v1",
        "status": "PASS",
        "variant": args.variant,
        "signature_sha256": runner.SPECS_BY_VARIANT[args.variant].signature_hash(),
        "seed": runner.SPECS_BY_VARIANT[args.variant].seed,
        "batch_size": args.batch_size,
        "dtype": "bfloat16",
        "compile_mode": args.compile_mode,
        "device": torch.cuda.get_device_name(),
        "source_fingerprint": os.environ.get("LNET_SOURCE_FINGERPRINT", "unknown"),
        "prepare_pid": prepared["prepare_pid"],
        "resume_pid": os.getpid(),
        "prepare_loss": prepared["loss"],
        "resume_loss": loss,
        "resume_verified": True,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "peak_allocated_gib": max(
            float(prepared["peak_allocated_gib"]),
            torch.cuda.max_memory_allocated() / 2**30,
        ),
        "peak_reserved_gib": max(
            float(prepared["peak_reserved_gib"]),
            torch.cuda.max_memory_reserved() / 2**30,
        ),
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
        "--compile-mode",
        args.compile_mode,
        "--phase",
        phase,
    ]


def _orchestrate(args: argparse.Namespace) -> dict[str, object]:
    args.output.unlink(missing_ok=True)
    prepare_path = args.output.parent / "prepare.json"
    prepare_path.unlink(missing_ok=True)
    _checkpoint_path(args).unlink(missing_ok=True)
    for phase in ("prepare", "resume"):
        result = subprocess.run(_phase_command(args, phase), check=False)  # noqa: S603
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    payload = json.loads(args.output.read_text())
    if payload.get("status") != "PASS":
        message = "overnight fresh-process smoke did not pass"
        raise RuntimeError(message)
    return cast("dict[str, object]", payload)


def _looks_like_oom(error: BaseException) -> bool:
    return "out of memory" in repr(error).lower()


def main() -> None:
    args = _arguments()
    try:
        if args.phase == "prepare":
            result = _prepare(args)
            _atomic_json(args.output.parent / "prepare.json", result)
        elif args.phase == "resume":
            result = _resume(args)
            _atomic_json(args.output, result)
        else:
            result = _orchestrate(args)
    except Exception as error:
        status = "OOM_FAILED" if _looks_like_oom(error) else "SMOKE_FAILED"
        _atomic_json(args.output, {"status": status, "exception": repr(error)})
        if status == "OOM_FAILED":
            raise SystemExit(OOM_EXIT_CODE) from error
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
