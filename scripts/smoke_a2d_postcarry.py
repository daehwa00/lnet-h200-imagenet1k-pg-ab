#!/usr/bin/env python3
"""CUDA training-step smoke for the A2D post-CFFN carry-main transition."""

# ruff: noqa: EM101, SLF001, T201, TRY003

from __future__ import annotations

import run_a2d_d4_postcarry_imagenet100 as candidate
import run_double_prefc_imagenet100 as a2d_base
import torch

from lnet.complex_scan import (
    ComplexScanConfig,
    S2DPostCFFNCarryMainTransition,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A2D PostCarry smoke requires CUDA")
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    torch.manual_seed(501)
    model = candidate._build(candidate.VARIANT, config).cuda().train()
    recipe = {
        "learning_rate": 3.0e-3,
        "weight_decay": 0.05,
        "modal_learning_rate_multiplier": 1.0 / 3.0,
        "pole_geometry_learning_rate_multiplier": 0.1,
        "fused_optimizer": True,
    }
    optimizer_source = a2d_base.residuals.optimizer_source
    optimizer = optimizer_source._build_optimizer(model, recipe)
    inputs = torch.randn(2, 3, 64, 64, device="cuda")
    targets = torch.tensor([3, 71], device="cuda")
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    decay = {
        id(parameter): float(group["weight_decay"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for stage in (model.stage1, model.stage2):
        transition = stage.augmented
        if not isinstance(transition, S2DPostCFFNCarryMainTransition):
            raise TypeError("smoke model lost its PostCarry transition")
        required = (
            transition.pole_scale,
            transition.carry_weight,
            transition.direction_mixer.weight_real,
            transition.ffn_input.weight_real,
            transition.ffn_output.weight_real,
            transition.output_projection.weight_real,
        )
        if any(value.grad is None for value in required):
            raise RuntimeError("PostCarry smoke found a disconnected branch")
        if not all(bool(torch.isfinite(value.grad).all()) for value in required):
            raise RuntimeError("PostCarry smoke produced non-finite gradients")
        if decay[id(transition.pole_scale)] != 0.0:
            raise RuntimeError("PostCarry alpha entered a weight-decay group")
    optimizer.step()
    if not bool(torch.isfinite(logits).all() and torch.isfinite(loss)):
        raise RuntimeError("PostCarry smoke produced non-finite outputs")
    print(
        "A2D-PostCarry",
        "parameters=",
        sum(parameter.numel() for parameter in model.parameters()),
        "loss=",
        float(loss.detach()),
        flush=True,
    )


if __name__ == "__main__":
    main()
