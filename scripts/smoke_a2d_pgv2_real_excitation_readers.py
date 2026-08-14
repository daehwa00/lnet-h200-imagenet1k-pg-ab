#!/usr/bin/env python3
"""One compiled BF16 ImageNet step for a real-excitation reader candidate."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: PLR0915, SLF001, T201
import argparse
import gc
import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_imagenet100 as control
import run_a2d_deep4_pgv2_real_excitation_readers_imagenet100 as runner
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

VARIANTS = (
    "R0_JIT_COMPLEX_K3",
    "R1_REAL_U",
    "R2_DUAL_FULL_K3",
    "R3_CONTENT_DWQ",
    "R4_FIXED_CONTRAST_Q",
    "R5_CONTENT_PWQ",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
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


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _tensor_leaves(value: object) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _tensor_leaves(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)


def _assert_finite_output(value: object, *, label: str) -> tuple[tuple[int, ...], ...]:
    leaves = tuple(_tensor_leaves(value))
    if not leaves:
        message = f"{label} produced no tensor outputs"
        raise RuntimeError(message)
    if any(not torch.isfinite(tensor).all() for tensor in leaves):
        message = f"{label} produced a non-finite tensor"
        raise FloatingPointError(message)
    return tuple(tuple(tensor.shape) for tensor in leaves)


def _diagnostics(model: torch.nn.Module) -> dict[str, float]:
    probe = getattr(runner, "_wandb_model_metrics", None)
    if not callable(probe):
        probe = getattr(runner, "_diagnostic_metrics", None)
    if not callable(probe):
        message = "real-excitation runner must expose its diagnostics probe"
        raise TypeError(message)
    observed = probe(model)
    if not isinstance(observed, Mapping):
        message = "reader diagnostics probe did not return a mapping"
        raise TypeError(message)
    metrics: dict[str, float] = {}
    for name, value in observed.items():
        if not isinstance(value, (int, float)):
            message = f"reader diagnostic {name!r} is not a scalar"
            raise TypeError(message)
        scalar = float(value)
        if not math.isfinite(scalar):
            message = f"reader diagnostic {name!r} is non-finite"
            raise FloatingPointError(message)
        metrics[str(name)] = scalar
    if not metrics:
        message = "reader diagnostics probe returned no metrics"
        raise RuntimeError(message)
    return metrics


def _strict_topology(model: torch.nn.Module, variant: str) -> None:
    assert_model = getattr(runner, "_assert_model", None)
    if not callable(assert_model):
        message = "real-excitation runner must expose _assert_model"
        raise TypeError(message)
    assert_model(model, variant)


def main() -> None:  # noqa: C901, PLR0912
    args = _arguments()
    if tuple(runner.VARIANTS) != VARIANTS:
        message = "smoke and runner disagree on the exact six-reader campaign"
        raise RuntimeError(message)
    if args.batch_size <= 0:
        message = "batch size must be positive"
        raise ValueError(message)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "real-excitation smoke requires exactly one visible CUDA device"
        raise RuntimeError(message)

    smoke_path = args.root / "smoke.json"
    smoke_path.unlink(missing_ok=True)
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
    model = runner._build(args.variant, config).cuda()
    _strict_topology(model, args.variant)
    model = source._prepare_model(model, recipe)
    optimizer = ramp.backbone.a2d_base.residuals.optimizer_source._build_optimizer(
        model,
        recipe,
    )

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
    if inputs.shape[0] != args.batch_size:
        message = "smoke did not load the requested real ImageNet batch"
        raise RuntimeError(message)
    inputs = inputs.cuda(non_blocking=False).contiguous(memory_format=torch.channels_last)
    targets = targets.cuda(non_blocking=False)
    permutation = torch.arange(targets.numel() - 1, -1, -1, device=targets.device)

    harness._configure_compile_runtime(args.root, recipe)
    runtime = harness._build_runtime(model, recipe)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    harness._begin_cudagraph_step(torch.device("cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime(inputs)
        output_shapes = _assert_finite_output(output, label="compiled training forward")
        _, loss, _ = source.heads._training_objective(
            model,
            output,
            targets,
            targets[permutation],
            0.7,
        )
    if not torch.isfinite(loss):
        message = "real-excitation smoke produced a non-finite loss"
        raise FloatingPointError(message)
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        message = f"real-excitation smoke has inactive parameters: {missing[:12]}"
        raise RuntimeError(message)
    if any(
        not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ):
        message = "real-excitation smoke produced non-finite gradients"
        raise FloatingPointError(message)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        message = "real-excitation smoke produced a non-finite gradient norm"
        raise FloatingPointError(message)
    optimizer.step()
    torch.cuda.synchronize()

    loss_value = float(loss.detach())
    validation_probe = getattr(runner, "_diagnostic_validation_batch", None)
    if callable(validation_probe):
        model.eval()
        validation_probe(
            model,
            loader,
            torch.device("cuda"),
            precision="bfloat16",
            channels_last=True,
        )
        model.train()
    diagnostics = _diagnostics(model)
    for suffix in (
        "reader_real_rms",
        "reader_imag_rms",
        "reader_imag_real_rms_ratio",
        "reader_phase_circular_variance",
    ):
        matching = [value for name, value in diagnostics.items() if name.endswith(suffix)]
        if len(matching) != 4:
            message = f"diagnostic probe did not report four stages of {suffix}"
            raise RuntimeError(message)
    updates = [value for name, value in diagnostics.items() if name.endswith("reader_updates")]
    if len(updates) != 4 or any(value <= 0.0 for value in updates):
        message = "reader diagnostic validation probe did not update every stage"
        raise RuntimeError(message)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    checkpoint = args.root / f"{args.variant}-smoke.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(
        {
            "variant": args.variant,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(checkpoint)

    del runtime, optimizer, output, loss, model
    gc.collect()
    torch.cuda.empty_cache()

    restored = runner._build(args.variant, config).cuda()
    _strict_topology(restored, args.variant)
    restored = source._prepare_model(restored, recipe)
    restored_optimizer = ramp.backbone.a2d_base.residuals.optimizer_source._build_optimizer(
        restored,
        recipe,
    )
    payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    if payload.get("variant") != args.variant:
        message = "smoke checkpoint identity does not match the requested variant"
        raise RuntimeError(message)
    restored.load_state_dict(payload["model"], strict=True)
    restored_optimizer.load_state_dict(payload["optimizer"])
    restored.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        restored_output = restored(inputs[:1])
    restored_shapes = _assert_finite_output(restored_output, label="restored forward")
    if restored_shapes != output_shapes:
        # Only the leading batch dimension may differ in the one-sample replay.
        canonical = tuple((args.batch_size, *shape[1:]) for shape in restored_shapes)
        if canonical != output_shapes:
            message = "restored forward changed the model output topology"
            raise RuntimeError(message)
    torch.cuda.synchronize()

    result: dict[str, object] = {
        "status": "PASS",
        "variant": args.variant,
        "batch_size": args.batch_size,
        "compile_mode": args.compile_mode,
        "device": torch.cuda.get_device_name(),
        "parameters": parameters,
        "loss": loss_value,
        "gradient_norm_before_clip": float(gradient_norm),
        "all_trainable_gradients_connected": True,
        "all_gradients_finite": True,
        "compiled_forward": True,
        "checkpoint_restore": "strict-model-and-optimizer",
        "restored_forward": True,
        "diagnostic_metric_count": len(diagnostics),
        "diagnostics": diagnostics,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    # Keep the status marker atomic and absent on every failed path.
    _atomic_json(smoke_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
