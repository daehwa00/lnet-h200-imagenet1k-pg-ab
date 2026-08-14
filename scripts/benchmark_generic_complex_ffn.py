#!/usr/bin/env python3
"""Compare explicit projection and packed CFFN graphs.

Both graphs are written directly in this benchmark so production dispatch
cannot collapse the comparison into packed-versus-packed during compilation.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lnet.pac_complex_layers import (
    WidelyLinear,
    packed_widely_linear_bias,
    packed_widely_linear_weight,
)


def _packed_weight(projection: WidelyLinear) -> Tensor:
    return packed_widely_linear_weight(
        projection.weight_real,
        projection.weight_imag,
        projection.conjugate_real,
        projection.conjugate_imag,
    )


def _packed_bias(projection: WidelyLinear) -> Tensor:
    bias = packed_widely_linear_bias(projection.bias_real, projection.bias_imag)
    if bias is None:
        message = "benchmark requires affine biases"
        raise RuntimeError(message)
    return bias


class ResidualCFFN(nn.Module):
    def __init__(
        self,
        modes: int,
        hidden_modes: int,
        *,
        layout: Literal["projection", "packed"],
    ) -> None:
        super().__init__()
        self.modes = modes
        self.layout = layout
        self.input = WidelyLinear(modes, hidden_modes)
        self.output = WidelyLinear(hidden_modes, modes)
        self.layer_scale = nn.Parameter(torch.full((modes,), 0.01))

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        if self.layout == "projection":
            hidden_real, hidden_imag = self.input(real, imag)
            update_real, update_imag = self.output(
                functional.silu(hidden_real),
                functional.silu(hidden_imag),
            )
            scale = self.layer_scale.to(real.dtype)
            return real + scale * update_real, imag + scale * update_imag
        source = torch.cat((real, imag), dim=-1)
        hidden = functional.silu(
            functional.linear(
                source,
                _packed_weight(self.input),
                _packed_bias(self.input),
            )
        )
        update = functional.linear(
            hidden,
            _packed_weight(self.output),
            _packed_bias(self.output),
        )
        scale = torch.cat((self.layer_scale, self.layer_scale)).to(source.dtype)
        output_real, output_imag = (source + scale * update).split(
            self.modes,
            dim=-1,
        )
        return output_real, output_imag


def _compiled(module: nn.Module, mode: str) -> nn.Module:
    return cast(
        "nn.Module",
        torch.compile(module, mode=mode, fullgraph=True, dynamic=False),
    )


def _autocast(precision: str) -> torch.autocast:
    return torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=precision == "bfloat16",
    )


def _step(module: nn.Module, real: Tensor, imag: Tensor, precision: str) -> None:
    module.zero_grad(set_to_none=True)
    real.grad = None
    imag.grad = None
    with _autocast(precision):
        output_real, output_imag = module(real, imag)
        loss = output_real.float().square().mean() + output_imag.float().square().mean()
    loss.backward()


def _parity(
    reference: nn.Module,
    candidate: nn.Module,
    real: Tensor,
    imag: Tensor,
    precision: str,
) -> dict[str, float]:
    reference_real = real.detach().clone().requires_grad_()
    reference_imag = imag.detach().clone().requires_grad_()
    candidate_real = real.detach().clone().requires_grad_()
    candidate_imag = imag.detach().clone().requires_grad_()
    with _autocast(precision):
        expected = reference(reference_real, reference_imag)
        actual = candidate(candidate_real, candidate_imag)
    cotangent = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(
        expected,
        (reference_real, reference_imag, *reference.parameters()),
        cotangent,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        (candidate_real, candidate_imag, *candidate.parameters()),
        cotangent,
    )

    def errors(value: Tensor, target: Tensor) -> tuple[float, float]:
        difference = value.detach().float() - target.detach().float()
        denominator = target.detach().float().square().sum().sqrt().clamp_min(1.0e-12)
        return (
            float(difference.abs().max()),
            float(difference.square().sum().sqrt() / denominator),
        )

    output_errors = [errors(value, target) for value, target in zip(actual, expected, strict=True)]
    gradient_errors = [
        errors(value, target)
        for value, target in zip(actual_gradients, expected_gradients, strict=True)
    ]
    return {
        "output_max_abs": max(error[0] for error in output_errors),
        "output_max_relative_l2": max(error[1] for error in output_errors),
        "gradient_max_abs": max(error[0] for error in gradient_errors),
        "gradient_max_relative_l2": max(error[1] for error in gradient_errors),
    }


def _paired_order(round_index: int) -> tuple[Literal["projection", "packed"], ...]:
    if round_index % 2 == 0:
        return "projection", "packed"
    return "packed", "projection"


def _summarize(samples: list[float], peak_allocated_bytes: int) -> dict[str, float | int]:
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "mad_ms": statistics.median(abs(value - median) for value in samples),
        "peak_allocated_bytes": peak_allocated_bytes,
    }


def _measure_pair(
    projection: nn.Module,
    packed: nn.Module,
    real: Tensor,
    imag: Tensor,
    *,
    precision: str,
    warmups: int,
    iterations: int,
) -> dict[str, dict[str, float | int]]:
    runtimes = {"projection": projection, "packed": packed}
    for round_index in range(warmups):
        for role in _paired_order(round_index):
            _step(runtimes[role], real, imag, precision)
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {"projection": [], "packed": []}
    peak_allocated_bytes = {"projection": 0, "packed": 0}
    for round_index in range(iterations):
        for role in _paired_order(round_index):
            torch.cuda.reset_peak_memory_stats()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            _step(runtimes[role], real, imag, precision)
            end.record()
            end.synchronize()
            samples[role].append(begin.elapsed_time(end))
            peak_allocated_bytes[role] = max(
                peak_allocated_bytes[role],
                torch.cuda.max_memory_allocated(),
            )
    return {
        role: _summarize(samples[role], peak_allocated_bytes[role])
        for role in ("projection", "packed")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--spatial", type=int, nargs="+", default=[28, 14, 7])
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--hidden-modes", type=int, default=128)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        message = "CFFN benchmark requires CUDA"
        raise RuntimeError(message)

    torch.manual_seed(501)
    torch.set_float32_matmul_precision("high")
    results: dict[str, object] = {}
    for spatial in args.spatial:
        projection = ResidualCFFN(
            args.modes,
            args.hidden_modes,
            layout="projection",
        ).cuda().train()
        packed = copy.deepcopy(projection)
        packed.layout = "packed"
        real = torch.randn(
            args.batch_size,
            spatial,
            spatial,
            args.modes,
            device="cuda",
            requires_grad=True,
        )
        imag = torch.randn_like(real, requires_grad=True)
        parity = _parity(projection, packed, real, imag, args.precision)
        projection_runtime = _compiled(projection, args.compile_mode)
        packed_runtime = _compiled(packed, args.compile_mode)
        measured = _measure_pair(
            projection_runtime,
            packed_runtime,
            real,
            imag,
            precision=args.precision,
            warmups=args.warmups,
            iterations=args.iterations,
        )
        projection_result = measured["projection"]
        packed_result = measured["packed"]
        speedup = float(projection_result["median_ms"]) / float(
            packed_result["median_ms"]
        )
        results[str(spatial)] = {
            "projection_reference": projection_result,
            "packed_candidate": packed_result,
            "packed_speedup": speedup,
            "parity": parity,
        }
        torch.compiler.reset()
        torch.cuda.empty_cache()

    report = {
        "schema": "lnet.cffn.benchmark.v2",
        "device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "modes": args.modes,
        "hidden_modes": args.hidden_modes,
        "precision": args.precision,
        "compile_mode": args.compile_mode,
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)  # noqa: T201
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
