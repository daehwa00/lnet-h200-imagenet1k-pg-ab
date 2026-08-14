#!/usr/bin/env python3
"""Evaluate generic Phase-Gated training latency and incremental peak memory."""

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# ruff: noqa: T201

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.pac_complex_layers import packed_complex_linear_weight
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN
from lnet.pac_triton_complex_rmsnorm import packed_complex_rms_norm
from lnet.pac_triton_phase_gate_linear import phase_gate_output_linear

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    rows: int
    modes: int
    hidden: int
    inner_rows: int | None = None
    focus: bool = True
    memory_gate: bool = False


@dataclass(frozen=True, slots=True)
class Measurement:
    name: str
    rows: int
    modes: int
    hidden: int
    layout: str
    median_ms: float
    incremental_peak_bytes: int


class CopiedInputPhaseGatedComplexFFN(PhaseGatedComplexFFN):
    """Pre-optimization control that copies coordinates before normalization."""

    def _optimized_forward(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        active_real = real.contiguous()
        active_imag = imag.contiguous()
        normalized = packed_complex_rms_norm(
            active_real,
            active_imag,
            self.norm.weight,
            self.norm.epsilon,
        )
        input_weight = packed_complex_linear_weight(
            self.input_projection.weight_real,
            self.input_projection.weight_imag,
        ).to(dtype=normalized.dtype)
        projected = functional.linear(normalized, input_weight)
        output_weight = packed_complex_linear_weight(
            self.output_projection.weight_real,
            self.output_projection.weight_imag,
        ).to(dtype=projected.dtype)
        packed_update = phase_gate_output_linear(
            projected,
            self.alpha,
            output_weight,
            redistribution=self.gate_redistribution,
            self_gated=self.self_gated,
        )
        packed_update = packed_update * self.gamma.to(dtype=packed_update.dtype)
        update_real, update_imag = packed_update.split(self.modes, dim=-1)
        return (
            real + update_real,
            imag + update_imag,
            projected,
            update_real,
            update_imag,
        )


class PackedTrainingBlock(nn.Module):
    """Measure the packed compute body without cache or telemetry noise."""

    def __init__(self, block: PhaseGatedComplexFFN) -> None:
        super().__init__()
        self.block = block

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        output_real, output_imag, _, _, _ = self.block._optimized_forward(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            real,
            imag,
        )
        return output_real, output_imag


PROFILES = (
    Profile("mode_h96_stage1", 401_408, 96, 96, memory_gate=True),
    Profile("mode_h96_stage2", 100_352, 96, 96),
    Profile("mode_h96_stage3", 25_088, 96, 96),
    Profile("path_h8_stage1", 9_633_792, 4, 8, inner_rows=96, memory_gate=True),
    Profile("path_h8_stage2", 2_408_448, 4, 8, inner_rows=96),
    Profile("path_h8_stage3", 602_112, 4, 8, inner_rows=96),
    Profile("mode_h192_stage1", 401_408, 96, 192, focus=False),
    Profile("mode_h192_stage2", 100_352, 96, 192, focus=False),
    Profile("mode_h192_stage3", 25_088, 96, 192, focus=False),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        choices=("copied", "direct"),
        default="direct",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="max-autotune",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _compile(module: nn.Module, mode: str) -> nn.Module:
    if mode == "none":
        return module
    return cast("nn.Module", torch.compile(module, mode=mode, fullgraph=False, dynamic=False))


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


def _event_time_ms(step: Callable[[], None], iterations: int) -> float:
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


def _coordinate(profile: Profile) -> Tensor:
    if profile.inner_rows is None:
        return torch.randn(
            profile.rows,
            profile.modes,
            device="cuda",
            requires_grad=True,
        )
    outer_rows, remainder = divmod(profile.rows, profile.inner_rows)
    if remainder:
        message = "transposed profile rows must be divisible by inner rows"
        raise ValueError(message)
    storage = torch.randn(
        outer_rows,
        profile.modes,
        profile.inner_rows,
        device="cuda",
    )
    return storage.transpose(-2, -1).requires_grad_()


def _measure(profile: Profile, args: argparse.Namespace, index: int) -> Measurement:
    torch.compiler.reset()
    torch.manual_seed(1701 + index)
    module_type = (
        CopiedInputPhaseGatedComplexFFN if args.implementation == "copied" else PhaseGatedComplexFFN
    )
    module = PackedTrainingBlock(module_type(profile.modes, profile.hidden)).cuda()
    active = _compile(module, args.compile_mode)
    real = _coordinate(profile)
    imag = _coordinate(profile)
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
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    incremental_peak = torch.cuda.max_memory_allocated() - fixed_workload_bytes
    median_ms = _event_time_ms(step, args.iterations)
    return Measurement(
        name=profile.name,
        rows=profile.rows,
        modes=profile.modes,
        hidden=profile.hidden,
        layout="contiguous" if profile.inner_rows is None else "last_two_transposed",
        median_ms=median_ms,
        incremental_peak_bytes=incremental_peak,
    )


def _source_revision() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1] / "src" / "lnet"
    for name in (
        "pac_phase_gated_cffn.py",
        "pac_reduction_tiling.py",
        "pac_triton_complex_rmsnorm.py",
        "pac_triton_hardware.py",
        "pac_triton_phase_gate.py",
        "pac_triton_phase_gate_linear.py",
        "pac_triton_phase_gate_linear_fused.py",
        "pac_triton_phase_gated_cffn_fused.py",
        "pac_triton_phase_gate_residual_fused.py",
        "pac_triton_rmsnorm_linear_fused.py",
    ):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:16]


def _comparison(
    baseline: dict[str, object],
    measurements: tuple[Measurement, ...],
) -> dict[str, object]:
    baseline_measurements = baseline.get("measurements")
    if not isinstance(baseline_measurements, list):
        message = "baseline measurements must be a list"
        raise TypeError(message)
    baseline_rows = {
        str(row["name"]): row
        for row in baseline_measurements
        if isinstance(row, dict) and "name" in row
    }
    profiles = {profile.name: profile for profile in PROFILES}
    current = {measurement.name: measurement for measurement in measurements}
    if set(baseline_rows) != set(current):
        message = "baseline and candidate profile sets differ"
        raise RuntimeError(message)

    focus_names = [profile.name for profile in PROFILES if profile.focus]
    baseline_total = sum(float(baseline_rows[name]["median_ms"]) for name in focus_names)
    candidate_total = sum(current[name].median_ms for name in focus_names)
    speed_ratios = {
        name: current[name].median_ms / float(baseline_rows[name]["median_ms"]) for name in current
    }
    memory_reductions = {
        name: 1.0
        - current[name].incremental_peak_bytes / int(baseline_rows[name]["incremental_peak_bytes"])
        for name, profile in profiles.items()
        if profile.memory_gate
    }
    aggregate_speedup = baseline_total / candidate_total
    worst_regression = max(speed_ratios.values())
    passed = (
        aggregate_speedup >= 1.03
        and worst_regression <= 1.03
        and all(reduction >= 0.05 for reduction in memory_reductions.values())
    )
    return {
        "aggregate_speedup": aggregate_speedup,
        "baseline_focus_ms": baseline_total,
        "candidate_focus_ms": candidate_total,
        "memory_reductions": memory_reductions,
        "passed": passed,
        "speed_ratios": speed_ratios,
        "worst_regression": worst_regression,
    }


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        message = "Phase-Gated evaluator requires exactly one visible CUDA device"
        raise RuntimeError(message)
    if min(args.warmups, args.iterations) <= 0:
        message = "warmups and iterations must be positive"
        raise ValueError(message)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    measurements = tuple(_measure(profile, args, index) for index, profile in enumerate(PROFILES))
    payload: dict[str, object] = {
        "compile_mode": args.compile_mode,
        "device": torch.cuda.get_device_name(),
        "implementation": args.implementation,
        "measurements": [asdict(measurement) for measurement in measurements],
        "source_revision": _source_revision(),
        "torch_version": torch.__version__,
    }
    comparison = None
    if args.baseline is not None:
        comparison = _comparison(json.loads(args.baseline.read_text()), measurements)
        payload["comparison"] = comparison
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.check and (comparison is None or not comparison.get("passed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
