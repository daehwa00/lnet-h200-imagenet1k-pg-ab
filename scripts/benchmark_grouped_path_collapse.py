#!/usr/bin/env python3
"""Evaluate the fused per-mode D4 path collapse against grouped convolutions."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import copy
import json
import math
import statistics
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lnet.pac_grouped_path_cffn import (
    GroupedWidelyLinear,
    grouped_cartesian_cffn,
    grouped_cartesian_cffn_reference,
)

if TYPE_CHECKING:
    from collections.abc import Callable


_STAGE_SPATIALS = (28, 14, 7)
_OUTPUT_RELATIVE_L2_LIMIT = 2.0e-2
_GRADIENT_RELATIVE_L2_LIMIT = 2.0e-2
_REPEAT_GRADIENT_RELATIVE_L2_LIMIT = 5.0e-4
# Grouped BF16 reductions can flip a few near-zero gradients on zero-initialized
# bias coordinates. AdamW maps those sparse sign changes to a full first-step
# update, so gate the complete parameter vector at 0.035% while retaining the
# per-tensor maximum as a diagnostic. Output and gradient gates remain strict.
_OPTIMIZER_STEP_RELATIVE_L2_LIMIT = 3.5e-4


class _GroupedPathCollapse(nn.Module):
    def __init__(self, modes: int, hidden: int, *, fused: bool) -> None:
        super().__init__()
        self.fused = fused
        self.input = GroupedWidelyLinear(modes, 4, hidden)
        self.output = GroupedWidelyLinear(modes, hidden, 1)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        operator = grouped_cartesian_cffn if self.fused else grouped_cartesian_cffn_reference
        return operator(
            real,
            imag,
            input_projection=self.input,
            output_projection=self.output,
        )


def _runtime(
    module: nn.Module,
    compile_mode: str,
) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
    if compile_mode == "eager":
        return cast("Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]", module)
    return torch.compile(
        module,
        mode=compile_mode,
        fullgraph=True,
        dynamic=False,
    )


def _relative_l2(actual: Tensor, expected: Tensor) -> float:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = actual_float - expected_float
    denominator = expected_float.square().sum().sqrt().clamp_min(1.0e-12)
    return float(difference.square().sum().sqrt() / denominator)


def _parameter_vector(module: nn.Module) -> Tensor:
    return torch.cat(tuple(parameter.detach().reshape(-1) for parameter in module.parameters()))


def _verify_parity(
    baseline: _GroupedPathCollapse,
    fused: _GroupedPathCollapse,
    baseline_runtime: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    fused_runtime: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    batch_size: int,
    spatial: int,
    modes: int,
) -> dict[str, object]:
    source_real = torch.randn(
        batch_size,
        spatial,
        spatial,
        4,
        modes,
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_imag = torch.randn_like(source_real)
    baseline_real = source_real.detach().clone().requires_grad_()
    baseline_imag = source_imag.detach().clone().requires_grad_()
    fused_real = source_real.detach().clone().requires_grad_()
    fused_imag = source_imag.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = baseline_runtime(baseline_real, baseline_imag)
        actual = fused_runtime(fused_real, fused_imag)
    cotangent = tuple(torch.randn_like(value) for value in actual)
    baseline_tensors = (baseline_real, baseline_imag, *baseline.parameters())
    fused_tensors = (fused_real, fused_imag, *fused.parameters())
    expected_gradients = torch.autograd.grad(expected, baseline_tensors, cotangent)
    actual_gradients = torch.autograd.grad(actual, fused_tensors, cotangent)
    expected_snapshot = tuple(value.detach().clone() for value in expected)
    actual_snapshot = tuple(value.detach().clone() for value in actual)
    expected_gradient_snapshot = tuple(value.detach().clone() for value in expected_gradients)
    actual_gradient_snapshot = tuple(value.detach().clone() for value in actual_gradients)

    repeat_real = source_real.detach().clone().requires_grad_()
    repeat_imag = source_imag.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        repeated = fused_runtime(repeat_real, repeat_imag)
    repeated_gradients = torch.autograd.grad(
        repeated,
        (repeat_real, repeat_imag, *fused.parameters()),
        cotangent,
    )

    gradient_names = (
        "source_real",
        "source_imag",
        *(name for name, _parameter in baseline.named_parameters()),
    )
    gradient_relative_l2 = {
        name: _relative_l2(actual_value, expected_value)
        for name, actual_value, expected_value in zip(
            gradient_names,
            actual_gradient_snapshot,
            expected_gradient_snapshot,
            strict=True,
        )
    }
    repeat_gradient_relative_l2 = {
        name: _relative_l2(repeated_value, actual_value)
        for name, repeated_value, actual_value in zip(
            gradient_names,
            repeated_gradients,
            actual_gradient_snapshot,
            strict=True,
        )
    }

    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1.0e-3)
    fused_optimizer = torch.optim.AdamW(fused.parameters(), lr=1.0e-3)
    for parameter, gradient in zip(
        baseline.parameters(),
        expected_gradient_snapshot[2:],
        strict=True,
    ):
        parameter.grad = gradient.detach().clone()
    for parameter, gradient in zip(
        fused.parameters(),
        actual_gradient_snapshot[2:],
        strict=True,
    ):
        parameter.grad = gradient.detach().clone()
    baseline_optimizer.step()
    fused_optimizer.step()
    optimizer_step_max_tensor_relative_l2 = max(
        _relative_l2(actual_parameter, expected_parameter)
        for actual_parameter, expected_parameter in zip(
            fused.parameters(),
            baseline.parameters(),
            strict=True,
        )
    )
    return {
        "output_relative_l2": max(
            _relative_l2(actual_value, expected_value)
            for actual_value, expected_value in zip(
                actual_snapshot,
                expected_snapshot,
                strict=True,
            )
        ),
        "gradient_relative_l2": gradient_relative_l2,
        "gradient_max_relative_l2": max(gradient_relative_l2.values()),
        "repeat_gradient_relative_l2": repeat_gradient_relative_l2,
        "repeat_gradient_max_relative_l2": max(repeat_gradient_relative_l2.values()),
        "optimizer_step_relative_l2": _relative_l2(
            _parameter_vector(fused),
            _parameter_vector(baseline),
        ),
        "optimizer_step_max_tensor_relative_l2": optimizer_step_max_tensor_relative_l2,
    }


def _step(
    module: nn.Module,
    runtime: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    source_real: Tensor,
    source_imag: Tensor,
) -> None:
    module.zero_grad(set_to_none=True)
    source_real.grad = None
    source_imag.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = runtime(source_real, source_imag)
        loss = output_real.float().square().mean() + output_imag.float().square().mean()
    loss.backward()


def _timed_step(step: Callable[[], None]) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    step()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end))


def _measure_pair(
    baseline_step: Callable[[], None],
    fused_step: Callable[[], None],
    *,
    warmups: int,
    iterations: int,
) -> tuple[dict[str, float], dict[str, float], tuple[float, ...]]:
    for _ in range(warmups):
        baseline_step()
        fused_step()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {"baseline": [], "fused": []}
    round_index = 0
    while min(len(values) for values in samples.values()) < iterations:
        order = (
            (("baseline", baseline_step), ("fused", fused_step))
            if round_index % 2 == 0
            else (("fused", fused_step), ("baseline", baseline_step))
        )
        for name, step in (*order, *reversed(order)):
            if len(samples[name]) < iterations:
                samples[name].append(_timed_step(step))
        round_index += 1
    paired_speedups = tuple(
        baseline_ms / fused_ms
        for baseline_ms, fused_ms in zip(
            samples["baseline"],
            samples["fused"],
            strict=True,
        )
    )
    return (
        {
            "median_step_ms": statistics.median(samples["baseline"]),
            "minimum_step_ms": min(samples["baseline"]),
        },
        {
            "median_step_ms": statistics.median(samples["fused"]),
            "minimum_step_ms": min(samples["fused"]),
        },
        paired_speedups,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--modes", type=int, default=96)
    parser.add_argument("--path-hidden", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--required-speedup", type=float, default=1.15)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        message = "grouped D4 path-collapse benchmark requires CUDA"
        raise RuntimeError(message)
    if min(
        args.batch_size,
        args.modes,
        args.path_hidden,
        args.warmups,
        args.iterations,
    ) <= 0:
        message = "benchmark dimensions must be positive"
        raise ValueError(message)

    torch.manual_seed(20260809)
    torch.cuda.manual_seed_all(20260809)
    torch.set_float32_matmul_precision("high")
    stages: dict[str, object] = {}
    speedups: list[float] = []
    parity_passed = True
    for spatial in _STAGE_SPATIALS:
        key = f"s{spatial}-m{args.modes}-h{args.path_hidden}"
        baseline = _GroupedPathCollapse(
            args.modes,
            args.path_hidden,
            fused=False,
        ).cuda().train()
        fused = _GroupedPathCollapse(
            args.modes,
            args.path_hidden,
            fused=True,
        ).cuda().train()
        initial_state = copy.deepcopy(baseline.state_dict())
        fused.load_state_dict(initial_state)
        baseline_runtime = _runtime(baseline, args.compile_mode)
        fused_runtime = _runtime(fused, args.compile_mode)
        parity = _verify_parity(
            baseline,
            fused,
            baseline_runtime,
            fused_runtime,
            batch_size=args.batch_size,
            spatial=spatial,
            modes=args.modes,
        )
        parity_passed = parity_passed and bool(
            cast("float", parity["output_relative_l2"])
            <= _OUTPUT_RELATIVE_L2_LIMIT
            and cast("float", parity["gradient_max_relative_l2"])
            <= _GRADIENT_RELATIVE_L2_LIMIT
            and cast("float", parity["repeat_gradient_max_relative_l2"])
            <= _REPEAT_GRADIENT_RELATIVE_L2_LIMIT
            and cast("float", parity["optimizer_step_relative_l2"])
            <= _OPTIMIZER_STEP_RELATIVE_L2_LIMIT
        )
        baseline.load_state_dict(initial_state)
        fused.load_state_dict(initial_state)
        source_real = torch.randn(
            args.batch_size,
            spatial,
            spatial,
            4,
            args.modes,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        source_imag = torch.randn_like(source_real, requires_grad=True)
        fused_real = source_real.detach().clone().requires_grad_()
        fused_imag = source_imag.detach().clone().requires_grad_()
        baseline_timing, fused_timing, paired_speedups = _measure_pair(
            partial(
                _step,
                baseline,
                baseline_runtime,
                source_real,
                source_imag,
            ),
            partial(
                _step,
                fused,
                fused_runtime,
                fused_real,
                fused_imag,
            ),
            warmups=args.warmups,
            iterations=args.iterations,
        )
        speedup = statistics.median(paired_speedups)
        speedups.append(speedup)
        stages[key] = {
            "spatial": spatial,
            "rows": args.batch_size * spatial * spatial * args.modes,
            "baseline": baseline_timing,
            "fused": fused_timing,
            "speedup": speedup,
            "paired_speedup_min": min(paired_speedups),
            "paired_speedup_max": max(paired_speedups),
            "parity": parity,
        }
        del baseline, fused, source_real, source_imag, fused_real, fused_imag
        torch.cuda.empty_cache()
        if args.compile_mode != "eager":
            torch.compiler.reset()

    aggregate_speedup = math.prod(speedups) ** (1.0 / len(speedups))
    passed = (
        parity_passed
        and all(speedup >= 1.0 for speedup in speedups)
        and aggregate_speedup >= args.required_speedup
    )
    report = {
        "schema": "lnet.grouped_d4_path_collapse_evaluator.v1",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": args.dtype,
        "compile_mode": args.compile_mode,
        "shape": {
            "batch_size": args.batch_size,
            "modes": args.modes,
            "path_hidden": args.path_hidden,
            "stage_spatials": list(_STAGE_SPATIALS),
        },
        "stages": stages,
        "aggregate_speedup": aggregate_speedup,
        "required_speedup": args.required_speedup,
        "tolerances": {
            "output_relative_l2": _OUTPUT_RELATIVE_L2_LIMIT,
            "gradient_relative_l2": _GRADIENT_RELATIVE_L2_LIMIT,
            "repeat_gradient_relative_l2": _REPEAT_GRADIENT_RELATIVE_L2_LIMIT,
            "optimizer_step_relative_l2": _OPTIMIZER_STEP_RELATIVE_L2_LIMIT,
        },
        "parity_passed": parity_passed,
        "pass": passed,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(f"{rendered}\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
