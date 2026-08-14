#!/usr/bin/env python3
"""Run one real-ImageNet compiled BF16 joint-vector-readout training step."""

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: PLR0915, SLF001, T201

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_wide_memory_joint_vector_readout_imagenet100 as runner
import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from lnet.pac_wide_memory_joint_vector_readout import WideMemoryJointVectorReadoutStage

if TYPE_CHECKING:
    from collections.abc import Mapping


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--compile-mode", default="default")
    return parser.parse_args()


def _recipe(batch_size: int, compile_mode: str) -> dict[str, Any]:
    return {
        "epochs": 100,
        "batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": batch_size,
        "fused_optimizer": True,
        "learning_rate": 3.0e-3,
        "modal_learning_rate_multiplier": 1.0 / 3.0,
        "pole_geometry_learning_rate_multiplier": 0.1,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.8,
        "precision": "bfloat16",
        "compile_mode": compile_mode,
        "channels_last": True,
    }


def _mixed_loss(output: object, targets: Tensor) -> Tensor:
    if not isinstance(output, tuple) or len(output) != 5:
        message = "joint-readout smoke requires the established five-output classifier"
        raise RuntimeError(message)
    permutation = torch.arange(targets.numel() - 1, -1, -1, device=targets.device)

    def cross_entropy(logits: Tensor) -> Tensor:
        return 0.7 * functional.cross_entropy(
            logits.float(),
            targets,
            label_smoothing=0.1,
        ) + 0.3 * functional.cross_entropy(
            logits.float(),
            targets[permutation],
            label_smoothing=0.1,
        )

    return cross_entropy(output[0]) + 0.5 * cross_entropy(output[1])


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "joint-readout smoke requires exactly one visible CUDA device"
        raise RuntimeError(message)
    os.environ["LNET_COMPILE_MODE"] = args.compile_mode
    torch.manual_seed(827)
    torch.cuda.manual_seed_all(827)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()

    runner.control.base.base._configure_ramp()
    recipe = _recipe(args.batch_size, args.compile_mode)
    ramp = runner.control.base.base.control.control.stemres.uniform.base
    harness = ramp.heads.harness
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = runner._build(runner.VARIANT, config).cuda()
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

    model.train()
    optimizer.zero_grad(set_to_none=True)
    harness._begin_cudagraph_step(torch.device("cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime(inputs)
        loss = _mixed_loss(output, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    stages = [
        module
        for module in model.modules()
        if isinstance(module, WideMemoryJointVectorReadoutStage)
    ]
    pg_blocks = [
        block
        for stage in stages
        for block in (stage.pole_pg, stage.path_pg, stage.post_pg)
        if block is not None
    ]
    if (
        missing
        or len(stages) != 4
        or any(stage.diagnostic_updates.item() < 1 for stage in stages)
        or len(pg_blocks) != 11
        or any(block.diagnostic_updates.item() < 1 for block in pg_blocks)
    ):
        message = f"joint-readout smoke has inactive parameters or blocks: {missing[:8]}"
        raise RuntimeError(message)

    checkpoint = args.root / "joint-vector-readout-smoke.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    restored = runner._build(runner.VARIANT, config)
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    payload = {
        "status": "PASS",
        "variant": runner.VARIANT,
        "excitation_schedule": runner.EXCITATION_SCHEDULE,
        "pole_schedule": runner.POLE_SCHEDULE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": args.batch_size,
        "compile_mode": args.compile_mode,
        "device": torch.cuda.get_device_name(),
        "loss": float(loss.detach()),
        "phase_gated_blocks": len(pg_blocks),
        "joint_memory_stages": len(stages),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "checkpoint_restore": "strict",
    }
    _write_json(args.root / "smoke.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
