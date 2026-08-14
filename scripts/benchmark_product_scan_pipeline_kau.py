#!/usr/bin/env python3
"""Measure retained versus scan-only-recomputed D4 product pipelines."""

from __future__ import annotations

# ruff: noqa: EM101, T201, TC003, TRY003
import argparse
import json
import math
import statistics
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor

from lnet.pac_product_scan_pipeline import run_product_scan_pipeline
from lnet.pac_real2d_math import discrete_pole_real2d


def _inputs(
    batch: int,
    height: int,
    modes: int,
    storage_dtype: torch.dtype,
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor],
]:
    generator = torch.Generator(device="cuda").manual_seed(2701 + batch + height)
    damping = torch.logspace(
        math.log10(0.04),
        math.log10(0.35),
        modes,
        device="cuda",
    ).view(1, 1, 1, modes)
    poles: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for phase_scale in (0.75, 0.70):
        phase = torch.linspace(
            0.0,
            phase_scale * math.pi,
            modes,
            device="cuda",
        ).view(1, 1, 1, modes)
        poles.append(
            cast(
                "tuple[Tensor, Tensor, Tensor, Tensor]",
                tuple(
                    value.detach().requires_grad_()
                    for value in discrete_pole_real2d(damping, phase, 1.0)
                ),
            )
        )
    source = cast(
        "tuple[Tensor, Tensor]",
        tuple(
            torch.randn(
                (batch, height, height, modes),
                generator=generator,
                device="cuda",
            )
            .to(dtype=storage_dtype)
            .requires_grad_()
            for _ in range(2)
        ),
    )
    return poles[0], poles[1], source


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
    parser.add_argument("--memory-policy", choices=("retain", "recompute"), required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--compile-mode")
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    storage_dtype = getattr(torch, arguments.storage_dtype)
    pole_x, pole_y, source = _inputs(
        arguments.batch,
        arguments.height,
        arguments.modes,
        storage_dtype,
    )
    scan_inputs = (*pole_x, *pole_y, *source)
    leaves = scan_inputs

    def retained() -> tuple[Tensor, Tensor, Tensor]:
        output = run_product_scan_pipeline(
            pole_x,
            pole_y,
            source,
            epilogue="coarse",
            gain_normalization="global",
            memory_policy=arguments.memory_policy,
        )
        if not isinstance(output, tuple):
            raise TypeError("coarse product scan did not return its three outputs")
        return output

    forward = retained
    runtime = (
        torch.compile(forward, mode=arguments.compile_mode, dynamic=False)
        if arguments.compile_mode is not None
        else forward
    )
    outputs = runtime()
    generator = torch.Generator(device="cuda").manual_seed(2901)
    grad_outputs = tuple(
        torch.randn(value.shape, generator=generator, device="cuda", dtype=value.dtype)
        for value in outputs
    )

    def step() -> tuple[Tensor, ...]:
        return torch.autograd.grad(runtime(), leaves, grad_outputs)

    elapsed_ms = _measure(step, groups=arguments.groups, repeats=arguments.repeats)
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    measured_outputs = runtime()
    torch.cuda.synchronize()
    forward_live_bytes = torch.cuda.memory_allocated() - baseline_bytes
    forward_peak_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
    gradients = torch.autograd.grad(measured_outputs, leaves, grad_outputs)
    torch.cuda.synchronize()
    step_peak_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
    print(
        json.dumps(
            {
                "memory_policy": arguments.memory_policy,
                "batch": arguments.batch,
                "height": arguments.height,
                "modes": arguments.modes,
                "storage_dtype": arguments.storage_dtype,
                "execution": "compiled" if arguments.compile_mode is not None else "eager",
                "median_forward_backward_ms": elapsed_ms,
                "forward_live_bytes": forward_live_bytes,
                "forward_peak_bytes": forward_peak_bytes,
                "step_peak_bytes": step_peak_bytes,
                "finite_gradients": all(bool(torch.isfinite(value).all()) for value in gradients),
            }
        )
    )


if __name__ == "__main__":
    main()
