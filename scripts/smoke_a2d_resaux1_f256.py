#!/usr/bin/env python3
"""Forward/backward smoke test for A2D-ResAux1-F256."""

# ruff: noqa: SLF001, T201

from __future__ import annotations

import argparse

import run_a2d_affine_qhead_imagenet100 as heads
import run_a2d_resaux1_f256_imagenet100 as candidate
import torch

from lnet.complex_scan import ComplexScanConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        message = "CUDA smoke requested without an available CUDA device"
        raise RuntimeError(message)
    device = torch.device(args.device)
    torch.manual_seed(501)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = candidate._build(candidate.VARIANT, config).to(device).train()
    inputs = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    targets = torch.arange(args.batch_size, device=device) % 100
    output = model(inputs)
    logits, loss, diagnostics = heads._training_objective(
        model,
        output,
        targets,
        targets.flip(0),
        0.7,
    )
    loss.backward()
    gradients = [
        gradient
        for parameter in model.parameters()
        if (gradient := parameter.grad) is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    ):
        message = "A2D-ResAux1-F256 produced invalid gradients"
        raise RuntimeError(message)
    if not bool(torch.isfinite(logits).all() and torch.isfinite(loss)):
        message = "A2D-ResAux1-F256 produced non-finite outputs"
        raise RuntimeError(message)
    print(
        candidate.VARIANT,
        "parameters=",
        sum(parameter.numel() for parameter in model.parameters()),
        "fusion_width=",
        model.classifier.fusion.hidden_dim,
        "loss=",
        float(loss.detach()),
        "diagnostics=",
        sorted(diagnostics),
        flush=True,
    )


if __name__ == "__main__":
    main()
