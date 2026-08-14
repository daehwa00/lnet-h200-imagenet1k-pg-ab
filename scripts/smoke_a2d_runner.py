#!/usr/bin/env python3
"""Generic forward/backward smoke test for an A2D experiment runner."""

# ruff: noqa: SLF001, T201

from __future__ import annotations

import argparse
import importlib

import torch

from lnet.complex_scan import ComplexScanConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument(
        "--variant",
        default=None,
        help="optional variant override for runners that expose multiple candidates",
    )
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    candidate = importlib.import_module(args.runner)
    variant = args.variant or candidate.VARIANT
    if args.device == "cuda" and not torch.cuda.is_available():
        message = "CUDA smoke requested without an available CUDA device"
        raise RuntimeError(message)
    device = torch.device(args.device)
    torch.manual_seed(501)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = candidate._build(variant, config).to(device).train()
    inputs = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    targets = torch.arange(args.batch_size, device=device) % 100
    output = model(inputs)
    logits, loss, diagnostics = candidate.heads._training_objective(
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
        message = f"{variant} produced invalid gradients"
        raise RuntimeError(message)
    if not bool(torch.isfinite(logits).all() and torch.isfinite(loss)):
        message = f"{variant} produced non-finite outputs"
        raise RuntimeError(message)
    print(
        variant,
        "parameters=",
        sum(parameter.numel() for parameter in model.parameters()),
        "aux_weight=",
        model.classifier.affine_auxiliary_weight,
        "loss=",
        float(loss.detach()),
        "diagnostics=",
        sorted(diagnostics),
        flush=True,
    )


if __name__ == "__main__":
    main()
