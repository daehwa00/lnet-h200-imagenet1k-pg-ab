#!/usr/bin/env python3
"""Measure one checkpointed factorized wide-pole stage in its training path."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: C901, PLR0915, T201
import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import run_a2d_factorized_wide_pole_memory_imagenet100 as runner
import torch

from lnet.pac_factorized_wide_pole_memory import FactorizedWidePoleMemoryStage

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poles", type=int, default=256)
    parser.add_argument("--stage-index", type=int, default=3)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--compile-mode", choices=("none", "default"), default="default")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _stage(args: argparse.Namespace) -> FactorizedWidePoleMemoryStage:
    return (
        FactorizedWidePoleMemoryStage(
            runner.CONTENT_MODES,
            args.poles,
            stage_index=args.stage_index,
            maximum_phase=runner.canonical8.MAXIMUM_PHASES[args.stage_index],
            frequency_scale=runner.calibrated.FREQUENCY_SCALES[args.stage_index],
            damping_scale=runner.calibrated.DAMPING_SCALES[args.stage_index],
            terminal=args.stage_index == 3,
            scan_memory_policy="recompute",
        )
        .cuda()
        .train()
    )


def _loss(output: tuple[tuple[Tensor, Tensor] | None, Tensor]) -> Tensor:
    state, descriptor = output
    result = descriptor.float().square().mean()
    if state is not None:
        result = result + state[0].float().square().mean() + state[1].float().square().mean()
    return result


def _diagnostic_counts(stage: FactorizedWidePoleMemoryStage) -> dict[str, int]:
    blocks = {
        "content_pg": stage.content_pg,
        "pole_pg": stage.pole_pg,
        "path_pg": stage.path_pg,
        "post_pg": stage.post_pg,
    }
    result = {"stage": int(stage.diagnostic_updates.item())}
    result.update(
        {
            name: int(block.diagnostic_updates.item())
            for name, block in blocks.items()
            if block is not None
        }
    )
    return result


def _timed_step(
    step: Callable[[], Tensor],
    *,
    counts: Callable[[], dict[str, int]],
) -> tuple[float, dict[str, int], float]:
    before = counts()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = step()
    end.record()
    torch.cuda.synchronize()
    after = counts()
    return (
        start.elapsed_time(end),
        {name: after[name] - before[name] for name in before},
        float(loss),
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "production wide-pole evaluation requires exactly one visible CUDA device"
        raise RuntimeError(message)
    torch.manual_seed(841)
    torch.cuda.manual_seed_all(841)
    torch.set_float32_matmul_precision("high")
    stage = _stage(args)
    runtime: Callable[[Tensor, Tensor], tuple[tuple[Tensor, Tensor] | None, Tensor]] = stage
    if args.compile_mode != "none":
        runtime = torch.compile(stage, mode=args.compile_mode)
    real = torch.randn(
        args.batch_size,
        args.size,
        args.size,
        runner.CONTENT_MODES,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    imag = torch.randn_like(real, requires_grad=True)

    def step() -> Tensor:
        stage.zero_grad(set_to_none=True)
        real.grad = None
        imag.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = runtime(real, imag)
            loss = _loss(output)
        loss.backward()
        return loss.detach()

    compile_started = perf_counter()
    for _ in range(args.warmups):
        step()
    torch.cuda.synchronize()
    compile_seconds = perf_counter() - compile_started
    torch.cuda.reset_peak_memory_stats()
    measurements = [
        _timed_step(step, counts=lambda: _diagnostic_counts(stage)) for _ in range(args.iterations)
    ]
    latency_ms = [measurement[0] for measurement in measurements]
    diagnostic_deltas = [measurement[1] for measurement in measurements]
    missing_gradients = [
        name
        for name, parameter in stage.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()))
    ]
    payload: dict[str, object] = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "shape": {
            "batch": args.batch_size,
            "size": args.size,
            "content_modes": runner.CONTENT_MODES,
            "poles": args.poles,
            "stage_index": args.stage_index,
        },
        "compile_mode": args.compile_mode,
        "compile_seconds": compile_seconds,
        "latency_ms": {
            "median": statistics.median(latency_ms),
            "mean": statistics.fmean(latency_ms),
            "samples": latency_ms,
        },
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "diagnostic_deltas": diagnostic_deltas,
        "missing_or_nonfinite_gradients": missing_gradients,
        "loss": measurements[-1][2],
    }
    failures: list[str] = []
    if missing_gradients:
        failures.append(f"missing or non-finite gradients: {missing_gradients[:8]}")
    if args.check and any(
        any(value != 1 for value in delta.values()) for delta in diagnostic_deltas
    ):
        failures.append(f"diagnostics did not update exactly once: {diagnostic_deltas}")
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        baseline_latency = float(baseline["latency_ms"]["median"])
        baseline_peak = float(baseline["peak_allocated_gib"])
        payload["baseline"] = {
            "latency_ms": baseline_latency,
            "peak_allocated_gib": baseline_peak,
            "speedup": baseline_latency / statistics.median(latency_ms),
        }
        if args.check and statistics.median(latency_ms) > baseline_latency * 1.03:
            failures.append("median latency regressed by more than three percent")
        if args.check and torch.cuda.max_memory_allocated() / 2**30 > baseline_peak:
            failures.append("peak allocated memory exceeded the baseline")
    if failures:
        payload["status"] = "FAIL"
        payload["failures"] = failures
    if args.output is not None:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
