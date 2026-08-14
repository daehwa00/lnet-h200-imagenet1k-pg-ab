#!/usr/bin/env python3
"""Check local-reader structure and optionally measure CUDA BF16 cost."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch import Tensor, nn

from lnet.pac_complex_layers import PackedComplexLinear
from lnet.pac_complex_scan_reader import PackedComplexConv2dReader


def _parameter_report(modes: int, kernel_size: int) -> dict[str, int | float]:
    point = PackedComplexLinear(modes, modes)
    local = PackedComplexConv2dReader(modes, modes, kernel_size=kernel_size)
    point_parameters = sum(parameter.numel() for parameter in point.parameters())
    local_parameters = sum(parameter.numel() for parameter in local.parameters())
    full_spatial_parameters = 2 * modes * modes * kernel_size**2
    return {
        "point_parameters": point_parameters,
        "local_parameters": local_parameters,
        "full_spatial_parameters": full_spatial_parameters,
        "local_vs_full_spatial": local_parameters / full_spatial_parameters,
    }


def _cuda_step(
    module: nn.Module,
    real: Tensor,
    imag: Tensor,
) -> None:
    module.zero_grad(set_to_none=True)
    output_real, output_imag = module(real, imag)
    (output_real.square().mean() + output_imag.square().mean()).backward()


def _measure_cuda(
    module: nn.Module,
    real: Tensor,
    imag: Tensor,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        _cuda_step(module, real, imag)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        _cuda_step(module, real, imag)
        torch.cuda.synchronize()
        samples.append(1.0e3 * (time.perf_counter() - start))
    return {
        "median_forward_backward_ms": statistics.median(samples),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def _cuda_report(
    modes: int,
    kernel_size: int,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    point = PackedComplexLinear(modes, modes).to(device=device, dtype=dtype)
    local = PackedComplexConv2dReader(
        modes,
        modes,
        kernel_size=kernel_size,
    ).to(device=device, dtype=dtype)
    shape = (batch_size, 56, 56, modes)
    real = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "point": _measure_cuda(point, real, imag, warmup=warmup, repeats=repeats),
        "local": _measure_cuda(local, real, imag, warmup=warmup, repeats=repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=int, default=96)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    parameters = _parameter_report(args.modes, args.kernel_size)
    report: dict[str, object] = {
        "parameters": parameters,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report["cuda"] = _cuda_report(
            args.modes,
            args.kernel_size,
            args.batch_size,
            args.warmup,
            args.repeats,
        )
    print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201

    if (
        args.check
        and parameters["local_parameters"] != parameters["full_spatial_parameters"]
    ):
        message = "reader is not a full strict-complex spatial convolution"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
