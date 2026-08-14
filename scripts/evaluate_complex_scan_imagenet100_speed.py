#!/usr/bin/env python3
# ruff: noqa: C901, PERF401, PLR0912
# pyright: reportAny=false, reportExplicitAny=false
"""Evaluate a complex scan speed candidate against a frozen baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--minimum-speedup", type=float, default=1.15)
    parser.add_argument("--maximum-memory-ratio", type=float, default=1.0)
    parser.add_argument("--maximum-logits-error", type=float, default=2.0e-2)
    parser.add_argument("--maximum-loss-error", type=float, default=5.0e-3)
    parser.add_argument("--maximum-gradient-error", type=float, default=5.0e-2)
    parser.add_argument("--maximum-gradient-relative-rmse", type=float, default=2.0e-2)
    return parser.parse_args()


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"{key} must be numeric"
        raise TypeError(message)
    result = float(value)
    if not math.isfinite(result):
        message = f"{key} must be finite"
        raise ValueError(message)
    return result


def main() -> int:
    args = _arguments()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    failures: list[str] = []

    for key in ("schema", "device", "torch", "batch_size", "parameters"):
        if candidate.get(key) != baseline.get(key):
            failures.append(
                f"{key} mismatch: {candidate.get(key)!r} != {baseline.get(key)!r}"
            )
    baseline_speed = _number(baseline, "images_per_second")
    candidate_speed = _number(candidate, "images_per_second")
    speedup = candidate_speed / baseline_speed
    if speedup < args.minimum_speedup:
        failures.append(f"speedup {speedup:.4f}x < {args.minimum_speedup:.4f}x")
    baseline_memory = _number(baseline, "peak_allocated_bytes")
    candidate_memory = _number(candidate, "peak_allocated_bytes")
    memory_ratio = candidate_memory / baseline_memory
    if memory_ratio > args.maximum_memory_ratio:
        measured = f"{memory_ratio:.4f}"
        limit = f"{args.maximum_memory_ratio:.4f}"
        failures.append(f"peak allocated memory ratio {measured} > {limit}")
    if int(candidate.get("parameters", -1)) != 247_700:
        failures.append("candidate parameter count is not 247700")
    if candidate.get("scan_pipeline") != "associative_product":
        failures.append("candidate did not use the associative product scan pipeline")
    if not bool(candidate.get("regression_tests_passed")):
        failures.append("candidate regression tests did not pass")
    parity = candidate.get("parity")
    if not isinstance(parity, dict):
        failures.append("candidate parity payload is missing")
    else:
        limits = {
            "logits_max_abs": args.maximum_logits_error,
            "loss_abs": args.maximum_loss_error,
            "gradient_max_abs": args.maximum_gradient_error,
            "gradient_relative_rmse": args.maximum_gradient_relative_rmse,
        }
        for key, limit in limits.items():
            value = _number(parity, key)
            if value > limit:
                failures.append(f"parity {key} {value:.6g} > {limit:.6g}")
    for key in ("training_loss", "gradient_norm"):
        if _number(candidate, key) < 0.0:
            failures.append(f"candidate {key} is negative")

    payload = {
        "status": "pass" if not failures else "fail",
        "baseline_images_per_second": baseline_speed,
        "candidate_images_per_second": candidate_speed,
        "speedup": speedup,
        "baseline_peak_allocated_bytes": baseline_memory,
        "candidate_peak_allocated_bytes": candidate_memory,
        "memory_ratio": memory_ratio,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
