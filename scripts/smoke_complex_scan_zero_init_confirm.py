# ruff: noqa: T201
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
"""CUDA compile smoke for the locked S2D pole-main zero-init model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_complex_scan_stage_carry_cifar100 import _build

from lnet.complex_scan import ComplexScanConfig


def main() -> None:
    torch.manual_seed(20260802)
    model = _build("s2d_pole_main", ComplexScanConfig()).cuda().train()
    if any(
        parameter is not None
        for stage in (model.stage1, model.stage2, model.terminal)
        for parameter in (
            stage.periodic_gate_x,
            stage.static_initial_real_x,
            stage.line_mean_gate_x,
        )
    ):
        message = "zero-init smoke found an enabled initial-state parameter"
        raise RuntimeError(message)
    compiled = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3)
    loss = torch.tensor(float("nan"), device="cuda")
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(16, 3, 32, 32, device="cuda")
        targets = torch.randint(100, (16,), device="cuda")
        logits = compiled(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            message = "zero-init CUDA smoke produced a non-finite loss"
            raise RuntimeError(message)
    print(
        "PASS",
        "s2d_pole_main_zero_init",
        sum(parameter.numel() for parameter in model.parameters()),
        float(loss.detach()),
        flush=True,
    )


if __name__ == "__main__":
    main()
