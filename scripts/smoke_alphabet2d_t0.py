#!/usr/bin/env python3
"""Persist a reproducible full-shape ALPHABET-2D-T0 CUDA training smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional

from lnet.alphabet2d import Alphabet2D, Alphabet2DConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser


def _synchronize() -> None:
    torch.cuda.synchronize()


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size < 1 or args.steps < 1:
        message = "batch size and steps must be positive"
        raise ValueError(message)
    if not torch.cuda.is_available():
        message = "ALPHABET-2D-T0 smoke requires CUDA"
        raise RuntimeError(message)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = Alphabet2DConfig(
        input_channels=3,
        output_dim=100,
        image_size=224,
        patch_size=16,
        model_dim=192,
        modes=16,
        depth=8,
        mlp_ratio=2.0,
        windows="global_2x2",
        fixed_direct_atlas=True,
        recurrence_backend="auto",
    )
    device = torch.device("cuda")
    model = Alphabet2D(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.05)
    inputs = torch.randn(args.batch_size, 3, 224, 224, device=device)
    targets = torch.randint(100, (args.batch_size,), device=device)
    torch.cuda.reset_peak_memory_stats()
    step_seconds: list[float] = []
    losses: list[float] = []
    finite_gradients = True
    logits = torch.empty(0, device=device)
    for _ in range(args.steps):
        _synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = functional.cross_entropy(logits, targets)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        finite_gradients = finite_gradients and bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        optimizer.step()
        _synchronize()
        step_seconds.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))
    source = Path(__file__).parents[1] / "src/lnet/alphabet2d.py"
    warm_steps = step_seconds[1:] if len(step_seconds) > 1 else step_seconds
    sorted_warm = sorted(warm_steps)
    median_step = sorted_warm[len(sorted_warm) // 2]
    direct_damping_trainable = any(
        name.endswith("field.raw_damping_x") and parameter.requires_grad
        for name, parameter in model.blocks.named_parameters()
    )
    payload = {
        "status": "done",
        "config": asdict(config),
        "seed": args.seed,
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
            "cudnn": torch.backends.cudnn.version(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "descriptor_dim": model.descriptor_dim,
            "direct_damping_trainable": direct_damping_trainable,
            "reader_damping_trainable": model.reader.raw_damping_x.requires_grad,
        },
        "training": {
            "batch_size": args.batch_size,
            "steps": args.steps,
            "losses": losses,
            "step_seconds": step_seconds,
            "warm_median_step_seconds": median_step,
            "warm_examples_per_second": args.batch_size / median_step,
            "finite_logits": bool(torch.isfinite(logits).all()),
            "finite_gradients": finite_gradients,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
