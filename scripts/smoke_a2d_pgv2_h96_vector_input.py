#!/usr/bin/env python3
"""Real-ImageNet BF16 compiled step for PGv2-H96 scan-reader variants."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: PLR0915, SLF001, T201
import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_imagenet100 as control
import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_local_reader_imagenet100 as local_reader
import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_vector_input_imagenet100 as vector
import run_a2d_deep4_pgv2_h96_scan_reader_followup_imagenet100 as followup
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from lnet.pac_complex_layers import PackedComplexLinear
from lnet.pac_complex_scan_reader import (
    PackedComplexConv2dReader,
    ResidualGatedComplexConv2dReader,
)
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN

if TYPE_CHECKING:
    from collections.abc import Mapping


RUNNERS = {
    "control": control,
    "followup": followup,
    "local-reader": local_reader,
    "vector": vector,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(RUNNERS), required=True)
    parser.add_argument("--model-variant")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--compile-mode", default="max-autotune")
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


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "PGv2-H96 smoke requires exactly one visible CUDA device"
        raise RuntimeError(message)
    runner = RUNNERS[args.variant]
    model_variant = args.model_variant or runner.VARIANT
    supported_variants = getattr(runner, "VARIANTS", (runner.VARIANT,))
    if model_variant not in supported_variants:
        message = f"unsupported {args.variant} model variant: {model_variant}"
        raise ValueError(message)
    os.environ["LNET_COMPILE_MODE"] = args.compile_mode
    torch.manual_seed(501)
    torch.cuda.manual_seed_all(501)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()

    recipe = _recipe(args.batch_size, args.compile_mode)
    ramp = control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    harness = source.heads.harness
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = runner._build(model_variant, config).cuda()
    model = source._prepare_model(model, recipe)
    optimizer = ramp.backbone.a2d_base.residuals.optimizer_source._build_optimizer(
        model,
        recipe,
    )
    train_transform, _ = harness._transforms()
    dataset = ImageFolder(args.data_root / "train", transform=train_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
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
        _, loss, _ = source.heads._training_objective(
            model,
            output,
            targets,
            targets[permutation],
            0.7,
        )
    if not torch.isfinite(loss):
        message = "PGv2-H96 smoke produced a non-finite loss"
        raise FloatingPointError(message)
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        message = f"PGv2-H96 smoke has inactive parameters: {missing[:8]}"
        raise RuntimeError(message)
    if any(
        not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ):
        message = "PGv2-H96 smoke produced non-finite gradients"
        raise FloatingPointError(message)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()

    pg_blocks = [module for module in model.modules() if isinstance(module, PhaseGatedComplexFFN)]
    projections = [
        getattr(model, name).pole_input_projection
        for name in ("stage1", "stage2", "stage3", "terminal")
        if isinstance(
            getattr(model, name).pole_input_projection,
            (
                PackedComplexConv2dReader,
                PackedComplexLinear,
                ResidualGatedComplexConv2dReader,
            ),
        )
    ]
    expected_projections = 0 if args.variant == "control" else 4
    if not pg_blocks or len(projections) != expected_projections:
        message = "PGv2-H96 smoke observed an unexpected module topology"
        raise RuntimeError(message)
    checkpoint = args.root / f"{model_variant}-smoke.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    restored = runner._build(model_variant, config)
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    payload = {
        "status": "PASS",
        "variant": model_variant,
        "batch_size": args.batch_size,
        "compile_mode": args.compile_mode,
        "device": torch.cuda.get_device_name(),
        "loss": float(loss.detach()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "phase_gated_blocks": len(pg_blocks),
        "pole_input_projections": len(projections),
        "all_trainable_gradients_connected": True,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "checkpoint_restore": "strict",
    }
    _atomic_json(args.root / "smoke.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
