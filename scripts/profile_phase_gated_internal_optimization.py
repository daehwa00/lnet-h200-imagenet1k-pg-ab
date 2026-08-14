#!/usr/bin/env python3
"""Profile one production-shaped Phase-Gated training step."""

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile

from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN


class TrainingBlock(nn.Module):
    def __init__(self, modes: int, hidden: int) -> None:
        super().__init__()
        self.block = PhaseGatedComplexFFN(modes, hidden)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        output_real, output_imag, _, _, _ = self.block._optimized_forward(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            real,
            imag,
        )
        return output_real, output_imag


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--modes", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--inner-rows", type=int)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _coordinate(rows: int, modes: int, inner_rows: int | None) -> Tensor:
    if inner_rows is None:
        return torch.randn(rows, modes, device="cuda", requires_grad=True)
    outer_rows, remainder = divmod(rows, inner_rows)
    if remainder:
        message = "rows must be divisible by inner rows"
        raise ValueError(message)
    storage = torch.randn(outer_rows, modes, inner_rows, device="cuda")
    return storage.transpose(-2, -1).requires_grad_()


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "profiler requires exactly one visible CUDA device"
        raise RuntimeError(message)
    torch.manual_seed(2107)
    module = TrainingBlock(args.modes, args.hidden).cuda()
    active = cast(
        "nn.Module",
        torch.compile(module, mode="max-autotune", fullgraph=False, dynamic=False),
    )
    real = _coordinate(args.rows, args.modes, args.inner_rows)
    imag = _coordinate(args.rows, args.modes, args.inner_rows)
    grad_real = torch.randn_like(real)
    grad_imag = torch.randn_like(imag)

    def step() -> None:
        active.zero_grad(set_to_none=True)
        real.grad = None
        imag.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_real, output_imag = active(real, imag)
        torch.autograd.backward((output_real, output_imag), (grad_real, grad_imag))

    for _ in range(args.warmups):
        step()
    torch.cuda.synchronize()
    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=True,
        profile_memory=True,
    ) as result:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()

    table = result.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total",
        row_limit=80,
    )
    payload = {
        "device": torch.cuda.get_device_name(),
        "hidden": args.hidden,
        "inner_rows": args.inner_rows,
        "modes": args.modes,
        "rows": args.rows,
        "steps": args.steps,
        "table": table,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(table)


if __name__ == "__main__":
    main()
