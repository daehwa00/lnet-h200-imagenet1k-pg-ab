#!/usr/bin/env python3
"""Compile one real ImageNet-1K BF16 train step for one H200 A/B variant."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# ruff: noqa: C901, PLR0915, SLF001, T201
import argparse
import json
import os
from pathlib import Path
from typing import Any

import run_h200_imagenet1k_pg_ab as runner
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from lnet.pac_complex_scan_reader import PackedComplexConv2dReader
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=runner.VARIANTS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
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


def _tensor_leaves(value: object) -> tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(tensor for item in value for tensor in _tensor_leaves(item))
    if isinstance(value, dict):
        return tuple(tensor for item in value.values() for tensor in _tensor_leaves(item))
    return ()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "H200 smoke requires exactly one visible CUDA device"
        raise RuntimeError(message)
    if args.batch_size < 2:
        message = "the MixUp smoke requires at least two examples"
        raise ValueError(message)

    runner._configure()
    os.environ["LNET_COMPILE_MODE"] = args.compile_mode
    torch.manual_seed(runner.SEEDS[0])
    torch.cuda.manual_seed_all(runner.SEEDS[0])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()

    ramp = runner.base.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    harness = source.heads.harness
    recipe = _recipe(args.batch_size, args.compile_mode)
    config = ramp.PoleModelConfig(output_dim=runner.NUM_CLASSES, stem_strides=(2, 2))
    model = runner._build(args.variant, config).cuda()
    model = source._prepare_model(model, recipe)
    optimizer = ramp.backbone.a2d_base.residuals.optimizer_source._build_optimizer(
        model,
        recipe,
    )

    train_transform, _ = harness._transforms()
    dataset = ImageFolder(args.data_root / "train", transform=train_transform)
    if len(dataset.classes) != runner.NUM_CLASSES:
        message = f"expected 1000 ImageNet classes, found {len(dataset.classes)}"
        raise RuntimeError(message)
    loader_options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "shuffle": False,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = harness.PREFETCH_FACTOR
        loader_options["persistent_workers"] = False
    loader = DataLoader(dataset, **loader_options)
    inputs, targets = next(iter(loader))
    inputs = inputs.cuda().contiguous(memory_format=torch.channels_last)
    targets = targets.cuda()
    permutation = torch.arange(targets.numel() - 1, -1, -1, device=targets.device)

    harness._configure_compile_runtime(args.root, recipe)
    runtime = harness._build_runtime(model, recipe)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    harness._begin_cudagraph_step(torch.device("cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime(inputs)
        leaves = _tensor_leaves(output)
        if not leaves or any(not torch.isfinite(tensor).all() for tensor in leaves):
            message = "compiled H200 forward produced missing or non-finite output"
            raise FloatingPointError(message)
        _, loss, _ = source.heads._training_objective(
            model,
            output,
            targets,
            targets[permutation],
            0.7,
        )
    if not torch.isfinite(loss):
        message = "compiled H200 step produced a non-finite loss"
        raise FloatingPointError(message)
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        message = f"compiled H200 step has inactive parameters: {missing[:12]}"
        raise RuntimeError(message)
    if any(
        not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ):
        message = "compiled H200 step produced non-finite gradients"
        raise FloatingPointError(message)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        message = "compiled H200 step produced a non-finite gradient norm"
        raise FloatingPointError(message)
    optimizer.step()
    torch.cuda.synchronize()

    pg_blocks = sum(isinstance(module, PhaseGatedComplexFFN) for module in model.modules())
    readers = sum(isinstance(module, PackedComplexConv2dReader) for module in model.modules())
    expected_pg = 0 if args.variant == runner.NO_PG_VARIANT else 3
    if pg_blocks != expected_pg or readers != 4:
        message = (
            f"unexpected topology: PG blocks={pg_blocks} (expected {expected_pg}), "
            f"K3 readers={readers} (expected 4)"
        )
        raise RuntimeError(message)

    checkpoint = args.root / f"{args.variant}-smoke.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    restored = runner._build(args.variant, config)
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    payload = {
        "status": "PASS",
        "variant": args.variant,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "compile_mode": args.compile_mode,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "phase_gated_blocks": pg_blocks,
        "k3_readers": readers,
        "all_trainable_gradients_connected": True,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "checkpoint_restore": "strict",
    }
    _atomic_json(args.root / f"smoke-{args.variant}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
