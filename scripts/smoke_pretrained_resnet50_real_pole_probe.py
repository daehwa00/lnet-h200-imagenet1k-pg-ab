#!/usr/bin/env python3
"""Forward/backward/compile/checkpoint smoke for the frozen external-CNN probes."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# ruff: noqa: SLF001, T201
import argparse
import io

import run_pretrained_resnet50_real_pole_probe_imagenet100 as probe
import torch
from torch import nn


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--variants", nargs="+", choices=probe.VARIANTS, default=probe.VARIANTS)
    return parser.parse_args()


def _step(
    model: probe.FrozenResNet50FeatureProbe,
    runtime: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    use_bfloat16: bool,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=inputs.device.type,
        dtype=torch.bfloat16,
        enabled=use_bfloat16,
    ):
        logits = runtime(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        message = "frozen-CNN probe smoke produced a non-finite loss"
        raise RuntimeError(message)
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        message = "frozen pretrained backbone received a gradient"
        raise RuntimeError(message)
    return logits


def main() -> None:
    args = _args()
    if args.device == "cuda" and not torch.cuda.is_available():
        message = "CUDA smoke requested without a CUDA device"
        raise RuntimeError(message)
    device = torch.device(args.device)
    weights = probe.PRETRAINED_WEIGHTS if args.pretrained else None
    config = probe.FrozenRealFeatureProbeConfig()
    recipe = {
        "learning_rate": 3.0e-3,
        "pole_geometry_learning_rate_multiplier": 1.0,
        "weight_decay": 0.05,
        "fused_optimizer": device.type == "cuda",
    }
    torch.set_float32_matmul_precision("high")
    for variant in args.variants:
        torch.manual_seed(501)
        model = probe._build_with_weights(variant, config, weights=weights).to(device)
        model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
        probe._assert_model(model, variant)
        optimizer = probe._build_optimizer(model, recipe)
        runtime: nn.Module = model
        if device.type == "cuda":
            runtime = torch.compile(  # pyright: ignore[reportAssignmentType]
                model,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
        inputs = torch.randn(
            args.batch_size,
            3,
            224,
            224,
            device=device,
        ).contiguous(memory_format=torch.channels_last)
        targets = torch.arange(args.batch_size, device=device) % config.output_dim
        first = _step(
            model,
            runtime,
            optimizer,
            inputs,
            targets,
            use_bfloat16=device.type == "cuda",
        )
        second = _step(
            model,
            runtime,
            optimizer,
            inputs,
            targets,
            use_bfloat16=device.type == "cuda",
        )
        if first.shape != (args.batch_size, config.output_dim) or second.shape != first.shape:
            message = f"{variant} smoke returned an incompatible classifier shape"
            raise RuntimeError(message)

        archive = io.BytesIO()
        torch.save(model.state_dict(), archive)
        archive.seek(0)
        restored = probe._build_with_weights(variant, config, weights=weights)
        restored.load_state_dict(torch.load(archive, map_location="cpu", weights_only=True))
        probe._assert_model(restored, variant)
        final_loss = nn.functional.cross_entropy(second, targets).detach()
        print(f"{variant}: ok loss={float(final_loss):.6f}")


if __name__ == "__main__":
    main()
