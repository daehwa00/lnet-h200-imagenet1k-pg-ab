#!/usr/bin/env python3
"""Measure one Phase-Gated training shape in a fresh CUDA process."""

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN

if TYPE_CHECKING:
    from collections.abc import Callable


class TrainingBlock(nn.Module):
    """Expose only the compute body used by production Phase-Gated modules."""

    def __init__(self, modes: int, hidden: int) -> None:
        super().__init__()
        self.block = PhaseGatedComplexFFN(modes, hidden)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        output_real, output_imag, _, _, _ = self.block._optimized_forward(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            real,
            imag,
        )
        return output_real, output_imag


@dataclass(frozen=True, slots=True)
class Measurement:
    rows: int
    modes: int
    hidden: int
    inner_rows: int | None
    cudagraphs: bool
    median_ms: float
    fixed_workload_bytes: int
    persistent_execution_bytes: int
    replay_peak_bytes: int
    total_execution_peak_bytes: int


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--modes", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--inner-rows", type=int)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="max-autotune",
    )
    parser.add_argument("--disable-cudagraphs", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    parser.add_argument("--maximum-peak-increase-bytes", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _coordinate(
    rows: int,
    modes: int,
    inner_rows: int | None,
    dtype: torch.dtype,
) -> Tensor:
    if inner_rows is None:
        return torch.randn(
            rows,
            modes,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
    outer_rows, remainder = divmod(rows, inner_rows)
    if remainder:
        message = "rows must be divisible by inner rows"
        raise ValueError(message)
    storage = torch.randn(
        outer_rows,
        modes,
        inner_rows,
        device="cuda",
        dtype=dtype,
    )
    return storage.transpose(-2, -1).requires_grad_()


def _comparison(
    baseline: dict[str, object],
    measurement: Measurement,
    *,
    dtype: str,
    minimum_speedup: float,
    maximum_peak_increase_bytes: int,
) -> dict[str, object]:
    baseline_measurement = baseline.get("measurement")
    if not isinstance(baseline_measurement, dict):
        message = "isolated Phase-Gated baseline has no measurement"
        raise TypeError(message)
    signature = (
        measurement.rows,
        measurement.modes,
        measurement.hidden,
        measurement.inner_rows,
        dtype,
    )
    baseline_signature = (
        int(baseline_measurement["rows"]),
        int(baseline_measurement["modes"]),
        int(baseline_measurement["hidden"]),
        baseline_measurement.get("inner_rows"),
        baseline.get("dtype", "float32"),
    )
    if signature != baseline_signature:
        message = "isolated Phase-Gated baseline shape or dtype differs"
        raise RuntimeError(message)
    baseline_latency = float(baseline_measurement["median_ms"])
    baseline_peak = int(baseline_measurement["replay_peak_bytes"])
    speedup = baseline_latency / measurement.median_ms
    peak_increase = measurement.replay_peak_bytes - baseline_peak
    return {
        "baseline_median_ms": baseline_latency,
        "baseline_replay_peak_bytes": baseline_peak,
        "maximum_peak_increase_bytes": maximum_peak_increase_bytes,
        "minimum_speedup": minimum_speedup,
        "passed": (
            speedup >= minimum_speedup
            and peak_increase <= maximum_peak_increase_bytes
        ),
        "peak_increase_bytes": peak_increase,
        "speedup": speedup,
    }


def _step(
    module: nn.Module,
    real: Tensor,
    imag: Tensor,
    grad_real: Tensor,
    grad_imag: Tensor,
) -> None:
    module.zero_grad(set_to_none=True)
    real.grad = None
    imag.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = module(real, imag)
    torch.autograd.backward((output_real, output_imag), (grad_real, grad_imag))


def _median_event_ms(step: Callable[[], None], iterations: int) -> float:
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "isolated measurement requires exactly one visible CUDA device"
        raise RuntimeError(message)
    if min(args.rows, args.modes, args.hidden, args.warmups, args.iterations) <= 0:
        message = "shape and repetition arguments must be positive"
        raise ValueError(message)

    torch.manual_seed(1701)
    torch.set_float32_matmul_precision("high")
    module = TrainingBlock(args.modes, args.hidden).cuda()
    if args.disable_cudagraphs:
        active = cast(
            "nn.Module",
            torch.compile(
                module,
                fullgraph=False,
                dynamic=False,
                options={
                    "max_autotune": args.compile_mode == "max-autotune",
                    "triton.cudagraphs": False,
                },
            ),
        )
    else:
        active = cast(
            "nn.Module",
            torch.compile(
                module,
                mode=args.compile_mode,
                fullgraph=False,
                dynamic=False,
            ),
        )
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    real = _coordinate(args.rows, args.modes, args.inner_rows, dtype)
    imag = _coordinate(args.rows, args.modes, args.inner_rows, dtype)
    grad_real = torch.randn_like(real)
    grad_imag = torch.randn_like(imag)
    fixed_workload_bytes = torch.cuda.memory_allocated()

    def step() -> None:
        _step(active, real, imag, grad_real, grad_imag)

    for _ in range(args.warmups):
        step()
    torch.cuda.synchronize()
    module.zero_grad(set_to_none=True)
    real.grad = None
    imag.grad = None
    torch.cuda.empty_cache()
    persistent_allocated = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    replay_peak = torch.cuda.max_memory_allocated()
    median_ms = _median_event_ms(step, args.iterations)

    measurement = Measurement(
        rows=args.rows,
        modes=args.modes,
        hidden=args.hidden,
        inner_rows=args.inner_rows,
        cudagraphs=not args.disable_cudagraphs,
        median_ms=median_ms,
        fixed_workload_bytes=fixed_workload_bytes,
        persistent_execution_bytes=max(0, persistent_allocated - fixed_workload_bytes),
        replay_peak_bytes=max(0, replay_peak - persistent_allocated),
        total_execution_peak_bytes=max(0, replay_peak - fixed_workload_bytes),
    )
    payload = {
        "compile_mode": args.compile_mode,
        "device": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "measurement": asdict(measurement),
        "torch_version": torch.__version__,
    }
    comparison = None
    if args.baseline is not None:
        comparison = _comparison(
            json.loads(args.baseline.read_text()),
            measurement,
            dtype=args.dtype,
            minimum_speedup=args.minimum_speedup,
            maximum_peak_increase_bytes=args.maximum_peak_increase_bytes,
        )
        payload["comparison"] = comparison
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.check and (comparison is None or not comparison.get("passed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
