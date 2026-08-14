#!/usr/bin/env python3
"""CUDA forward/backward smoke for every A2D affine Q-head candidate."""

# ruff: noqa: EM101, EM102, SLF001, T201, TRY003

from __future__ import annotations

import argparse

import run_a2d_affine_qhead_imagenet100 as candidate
import run_double_prefc_imagenet100 as a2d_base
import torch

from lnet.complex_scan import ComplexScanConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=candidate.VARIANTS)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A2D affine Q-head smoke requires CUDA")
    variants = tuple(args.variants) if args.variants else candidate.VARIANTS
    device = torch.device("cuda")
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    optimizer_builder = a2d_base.residuals.optimizer_source
    recipe = {
        "learning_rate": 3.0e-3,
        "weight_decay": 0.05,
        "modal_learning_rate_multiplier": 1.0 / 3.0,
        "pole_geometry_learning_rate_multiplier": 0.1,
        "fused_optimizer": True,
    }
    for variant in variants:
        torch.manual_seed(501)
        model = candidate._build(variant, config).to(device).train()
        optimizer = optimizer_builder._build_optimizer(model, recipe)
        inputs = torch.randn(2, 3, args.size, args.size, device=device)
        targets = torch.tensor([3, 71], device=device)
        output = model(inputs)
        logits, loss, diagnostics = candidate._training_objective(
            model,
            output,
            targets,
            targets.flip(0),
            0.7,
        )
        loss.backward()
        if not bool(torch.isfinite(logits).all() and torch.isfinite(loss)):
            raise RuntimeError(f"{variant} produced non-finite output")
        gradients = [
            parameter.grad
            for parameter in model.classifier.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError(f"{variant} produced invalid classifier gradients")
        optimizer.step()
        print(
            variant,
            "parameters=",
            sum(parameter.numel() for parameter in model.parameters()),
            "loss=",
            float(loss.detach()),
            "diagnostics=",
            sorted(diagnostics),
            flush=True,
        )
        del model, optimizer, inputs, targets, output, logits, loss, gradients
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
