#!/usr/bin/env python3
"""Forward/backward smoke test for every A2D head-design finalist."""

# ruff: noqa: SLF001, T201

from __future__ import annotations

import argparse
import gc

import run_a2d_head_design_e2e_imagenet100 as experiment
import torch
from torch.nn import functional

from lnet.complex_scan import ComplexScanConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--variants", nargs="*", default=list(experiment.VARIANTS))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        message = "A2D end-to-end smoke requires CUDA"
        raise RuntimeError(message)
    device = torch.device("cuda:0")
    # Initialize the context before resetting allocator counters.  On hosts
    # with a broken NVML userspace library, reset-before-first-allocation is
    # rejected even though CUDA execution itself is healthy.
    torch.empty(1, device=device)
    unknown = set(args.variants) - set(experiment.VARIANTS)
    if unknown:
        message = f"unknown smoke variants: {sorted(unknown)}"
        raise ValueError(message)
    for variant in args.variants:
        torch.cuda.reset_peak_memory_stats(0)
        torch.manual_seed(501)
        config = ComplexScanConfig(
            output_dim=100,
            stem_strides=(2, 2),
        )
        model = experiment._build(variant, config).to(device).train()
        inputs = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
        targets = torch.arange(args.batch_size, device=device) % 100
        logits = model(inputs)
        loss = functional.cross_entropy(logits, targets)
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            message = f"{variant} produced non-finite gradients"
            raise RuntimeError(message)
        print(
            variant,
            "parameters=",
            sum(parameter.numel() for parameter in model.parameters()),
            "loss=",
            float(loss.detach()),
            "peak_allocated_gib=",
            torch.cuda.max_memory_allocated(0) / 1024**3,
            flush=True,
        )
        del model, inputs, targets, logits, loss, gradients
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
