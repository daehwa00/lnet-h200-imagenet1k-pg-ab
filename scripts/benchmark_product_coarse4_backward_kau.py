#!/usr/bin/env python3
"""Measure the fused four-product scan/coarsening forward and backward."""

from __future__ import annotations

# ruff: noqa: EM101, T201, TC003, TRY003
import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch
from torch import Tensor

from lnet.pac_real2d_math import discrete_pole_real2d
from lnet.pac_triton_product_scan_coarse4 import pac_triton_product_scan_coarse4


def _inputs(
    batch: int,
    height: int,
    modes: int,
    storage_dtype: torch.dtype,
) -> tuple[tuple[Tensor, ...], ...]:
    generator = torch.Generator(device="cuda").manual_seed(1701 + batch + height)
    damping = torch.logspace(
        math.log10(0.04),
        math.log10(0.35),
        modes,
        device="cuda",
    ).view(1, 1, 1, modes)
    poles = []
    for pole_index, phase_scale in enumerate((0.75, 0.70)):
        phase = torch.linspace(
            0.0,
            phase_scale * math.pi,
            modes,
            device="cuda",
        ).view(1, 1, 1, modes)
        poles.append(
            tuple(
                value.detach().requires_grad_(pole_index == 1)
                for value in discrete_pole_real2d(damping, phase, 1.0)
            )
        )
    sources = tuple(
        torch.randn(
            (batch, height, height, modes),
            generator=generator,
            device="cuda",
        ).to(dtype=storage_dtype).requires_grad_()
        for _ in range(4)
    )
    return poles[0], poles[1], sources


def _measure(function: Callable[[], object], *, groups: int, repeats: int) -> float:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(groups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--compile-mode")
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    storage_dtype = getattr(torch, arguments.storage_dtype)
    pole_x, pole_y, sources = _inputs(
        arguments.batch,
        arguments.height,
        arguments.modes,
        storage_dtype,
    )
    source_a = sources[:2]
    source_b = sources[2:]
    leaves = (*pole_y, *sources)

    def forward() -> tuple[Tensor, Tensor, Tensor]:
        return pac_triton_product_scan_coarse4(
            pole_x,
            pole_y,
            source_a,
            source_b,
        )

    runtime = (
        torch.compile(forward, mode=arguments.compile_mode, dynamic=False)
        if arguments.compile_mode is not None
        else forward
    )
    outputs = runtime()
    generator = torch.Generator(device="cuda").manual_seed(1901)
    grad_outputs = tuple(
        torch.randn(value.shape, generator=generator, device="cuda", dtype=value.dtype)
        for value in outputs
    )

    def step() -> tuple[Tensor, ...]:
        return torch.autograd.grad(runtime(), leaves, grad_outputs)

    torch.cuda.reset_peak_memory_stats()
    elapsed_ms = _measure(step, groups=arguments.groups, repeats=arguments.repeats)
    gradients = step()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "batch": arguments.batch,
                "height": arguments.height,
                "modes": arguments.modes,
                "storage_dtype": arguments.storage_dtype,
                "execution": "compiled" if arguments.compile_mode is not None else "eager",
                "median_forward_backward_ms": elapsed_ms,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "finite_gradients": all(bool(torch.isfinite(value).all()) for value in gradients),
            }
        )
    )


if __name__ == "__main__":
    main()
