#!/usr/bin/env python3
"""Run one compiled BF16 training step for the PGv2 vector-pole model."""

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

import run_a2d_pgv2_vector_pole_scan_imagenet100 as runner
import torch
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from lnet.complex_scan_transitions import FixedComplexRMSNorm
from lnet.pac_pgv2_vector_pole_stage import PGv2VectorPoleStage

if TYPE_CHECKING:
    from collections.abc import Mapping


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def _validate_transition(name: str, stage: PGv2VectorPoleStage) -> None:
    if stage.terminal:
        return
    if type(stage.post_norm) is not FixedComplexRMSNorm or stage.carry_logits is None:
        message = f"{name} lost its stabilized transition"
        raise RuntimeError(message)
    carry = torch.softmax(stage.carry_logits.detach().float(), dim=-1)
    if bool((carry <= 0.0).any()) or float((carry.sum(-1) - 1.0).abs().max()) > 1.0e-6:
        message = f"{name} S2D carry left its simplex"
        raise RuntimeError(message)


def _validate_stages(model: torch.nn.Module) -> dict[str, list[float]]:
    stages = [getattr(model, name) for name in runner.STAGE_NAMES]
    if not all(isinstance(stage, PGv2VectorPoleStage) for stage in stages):
        message = "PGv2 vector-pole smoke observed the wrong stage implementation"
        raise RuntimeError(message)
    observed = [(stage.content_modes, stage.poles) for stage in stages]
    expected = list(zip(runner.EXCITATION_SCHEDULE, runner.POLE_SCHEDULE, strict=True))
    if observed != expected:
        message = "PGv2 vector-pole smoke observed the wrong K/P schedule"
        raise RuntimeError(message)

    singular_ranges: dict[str, list[float]] = {}
    for name, stage in zip(runner.STAGE_NAMES, stages, strict=True):
        if hasattr(stage, "input_norm"):
            message = f"{name} unexpectedly normalized excitation before the scan"
            raise RuntimeError(message)
        if not parametrize.is_parametrized(stage.pole_input, "weight"):
            message = f"{name} pole-input FC lost its semi-orthogonal parametrization"
            raise RuntimeError(message)
        singular = torch.linalg.svdvals(stage.pole_input.weight.detach().float())
        maximum_deviation = float((singular - 1.0).abs().max())
        if not torch.isfinite(singular).all() or maximum_deviation > 1.0e-3:
            message = (
                f"{name} pole-input FC left its semi-orthogonal manifold: "
                f"range=[{float(singular.min()):.8f}, {float(singular.max()):.8f}], "
                f"max_deviation={maximum_deviation:.3e}"
            )
            raise RuntimeError(message)
        if hasattr(stage, "descriptor_projection"):
            message = f"{name} retained the learned pole-to-Q projection"
            raise RuntimeError(message)
        singular_ranges[name] = [float(singular.min()), float(singular.max())]
        _validate_transition(name, stage)
    return singular_ranges


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "pole-scaled PGv2 smoke requires exactly one visible CUDA device"
        raise RuntimeError(message)
    os.environ["LNET_COMPILE_MODE"] = args.compile_mode
    torch.manual_seed(501)
    torch.cuda.manual_seed_all(501)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()

    recipe = _recipe(args.batch_size, args.compile_mode)
    ramp = runner.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    harness = source.heads.harness
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = runner._build(runner.VARIANT, config).cuda()
    model = source._prepare_model(model, recipe)
    optimizer = runner._build_optimizer(model, recipe)
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
        message = "PGv2 vector-pole smoke produced a non-finite loss"
        raise FloatingPointError(message)
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        message = f"PGv2 vector-pole smoke has inactive parameters: {missing[:8]}"
        raise RuntimeError(message)
    if any(
        not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ):
        message = "PGv2 vector-pole smoke produced non-finite gradients"
        raise FloatingPointError(message)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()

    singular_ranges = _validate_stages(model)
    standardizers = runner._head_standardizers(model)
    if any(module.affine for module in standardizers):
        message = "PGv2 vector-pole smoke changed the affine-free Q standardizer"
        raise RuntimeError(message)

    checkpoint = args.root / "smoke.pt"
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
        "batch_size": args.batch_size,
        "compile_mode": args.compile_mode,
        "device": torch.cuda.get_device_name(),
        "loss": float(loss.detach()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "excitation_schedule": list(runner.EXCITATION_SCHEDULE),
        "pole_schedule": list(runner.POLE_SCHEDULE),
        "descriptor_dim": model.descriptor_dim,
        "descriptor": "full_grid_raw_directional_pole_energy",
        "head_standardizer_epsilon": standardizers[0].eps,
        "pole_input_norm": "none",
        "pole_input_constraint": "trainable_semi_orthogonal",
        "s2d_carry": "learned_positive_unit_sum_simplex",
        "pole_input_singular_ranges": singular_ranges,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "checkpoint_restore": "strict",
    }
    _atomic_json(args.root / "smoke.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
