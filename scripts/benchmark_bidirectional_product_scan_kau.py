#!/usr/bin/env python3
"""Measure one complete horizontal bidirectional scan forward and backward."""

from __future__ import annotations

# ruff: noqa: EM101, T201, TC003, TRY003
import argparse
import json
import statistics
from collections.abc import Callable

import torch
from torch import Tensor

from lnet.pac_triton_bidirectional_product_scan import pac_triton_bidirectional_product_scan


def _measure(function: Callable[[], object], *, groups: int, repeats: int) -> list[float]:
    for _ in range(20):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(groups):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        for _ in range(repeats):
            function()
        finished.record()
        finished.synchronize()
        samples.append(started.elapsed_time(finished) / repeats)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    storage_dtype = getattr(torch, arguments.storage_dtype)
    generator = torch.Generator(device="cuda").manual_seed(
        511 + arguments.batch + arguments.height + arguments.modes
    )
    pole = tuple(
        (
            0.03
            * torch.randn(
                (1, 1, 1, arguments.modes),
                generator=generator,
                device="cuda",
            )
        ).requires_grad_()
        for _ in range(4)
    )
    with torch.no_grad():
        pole[0].fill_(0.82)
        pole[2].fill_(0.18)
    source = tuple(
        torch.randn(
            (arguments.batch, arguments.height, arguments.height, arguments.modes),
            generator=generator,
            device="cuda",
        )
        .to(dtype=storage_dtype)
        .requires_grad_()
        for _ in range(2)
    )
    leaves = (*pole, *source)

    def forward() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return pac_triton_bidirectional_product_scan(pole, source)

    runtime = torch.compile(forward, mode=arguments.compile_mode, dynamic=False)
    outputs = runtime()
    grad_outputs = tuple(torch.randn_like(value) for value in outputs)

    def step() -> tuple[Tensor, ...]:
        return torch.autograd.grad(runtime(), leaves, grad_outputs)

    torch.cuda.reset_peak_memory_stats()
    samples = _measure(step, groups=arguments.groups, repeats=arguments.repeats)
    gradients = step()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "label": arguments.label,
                "batch": arguments.batch,
                "height": arguments.height,
                "modes": arguments.modes,
                "storage_dtype": arguments.storage_dtype,
                "median_forward_backward_ms": statistics.median(samples),
                "samples_ms": samples,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "finite_gradients": all(bool(torch.isfinite(value).all()) for value in gradients),
            }
        )
    )


if __name__ == "__main__":
    main()
