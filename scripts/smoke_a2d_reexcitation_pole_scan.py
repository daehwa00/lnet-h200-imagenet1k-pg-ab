#!/usr/bin/env python3
"""Compile, train, restore, and benchmark the re-excitation pole model."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: PLR0915, SLF001, T201
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import run_a2d_reexcitation_pole_scan_imagenet100 as runner
import smoke_a2d_wide_memory_joint_vector_readout as control_smoke
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from lnet.pac_reexcitation_pole_memory import ReexcitationPoleMemoryStage

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lnet.pac_product_scan_pipeline import ScanMemoryPolicy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument(
        "--scan-memory-policy",
        choices=("retain", "recompute"),
        default="retain",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--benchmark-steps", type=int, default=3)
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "re-excitation pole smoke requires exactly one visible CUDA device"
        raise RuntimeError(message)
    if args.warmup_steps < 1 or args.benchmark_steps < 1:
        message = "smoke benchmark requires positive warmup and measurement counts"
        raise ValueError(message)
    memory_policy: ScanMemoryPolicy = args.scan_memory_policy
    os.environ["LNET_COMPILE_MODE"] = args.compile_mode
    torch.manual_seed(953)
    torch.cuda.manual_seed_all(953)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    runner.base.joint.control.base.base._configure_ramp()
    recipe = control_smoke._recipe(args.batch_size, args.compile_mode)
    ramp = runner.base.joint.control.base.base.control.control.stemres.uniform.base
    harness = ramp.heads.harness
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = runner._build_with_policy(runner.VARIANT, config, memory_policy).cuda()
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    model = source._prepare_model(model, recipe)

    train_transform, _ = harness._transforms()
    dataset = ImageFolder(args.data_root / "train", transform=train_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    inputs, targets = next(iter(loader))
    inputs = inputs.cuda().contiguous(memory_format=torch.channels_last)
    targets = targets.cuda()
    optimizer = ramp.backbone.a2d_base.residuals.optimizer_source._build_optimizer(model, recipe)
    harness._configure_compile_runtime(args.root, recipe)
    runtime = harness._build_runtime(model, recipe)

    def training_step() -> torch.Tensor:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        harness._begin_cudagraph_step(torch.device("cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = runtime(inputs)
            loss = control_smoke._mixed_loss(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.detach()

    loss = torch.zeros((), device="cuda")
    for _ in range(args.warmup_steps):
        loss = training_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(args.benchmark_steps):
        started = time.perf_counter()
        loss = training_step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    stages = [
        module for module in model.modules() if isinstance(module, ReexcitationPoleMemoryStage)
    ]
    pg_blocks = [
        block
        for stage in stages
        for block in (stage.memory_pg, stage.reexcitation_pg)
        if block is not None
    ]
    if (
        missing
        or len(stages) != 4
        or any(stage.diagnostic_updates.item() < 1 for stage in stages)
        or len(pg_blocks) != 6
        or any(block.diagnostic_updates.item() < 1 for block in pg_blocks)
    ):
        message = f"re-excitation pole smoke has inactive parameters or blocks: {missing[:8]}"
        raise RuntimeError(message)

    checkpoint = args.root / "reexcitation-pole-smoke.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    restored = runner._build_with_policy(runner.VARIANT, config, memory_policy)
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    median_seconds = statistics.median(durations)
    payload = {
        "status": "PASS",
        "variant": runner.VARIANT,
        "excitation_schedule": runner.EXCITATION_SCHEDULE,
        "pole_schedule": runner.POLE_SCHEDULE,
        "reexcitation_hidden": runner.REEXCITATION_HIDDEN,
        "descriptor_dim": runner.DESCRIPTOR_DIM,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": args.batch_size,
        "compile_mode": args.compile_mode,
        "scan_memory_policy": memory_policy,
        "device": torch.cuda.get_device_name(),
        "loss": float(loss),
        "phase_gated_blocks": len(pg_blocks),
        "reexcitation_pole_stages": len(stages),
        "median_step_ms": 1.0e3 * median_seconds,
        "median_images_per_second": args.batch_size / median_seconds,
        "step_ms": [1.0e3 * duration for duration in durations],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "checkpoint_restore": "strict",
    }
    _write_json(args.root / "smoke.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
