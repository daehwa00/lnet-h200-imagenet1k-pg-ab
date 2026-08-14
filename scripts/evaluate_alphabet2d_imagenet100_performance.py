#!/usr/bin/env python3
"""Evaluate the secondary_gpu ImageNet100 training performance artifact."""

# ruff: noqa: C901, EM102, T201, TRY003, TRY004

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--minimum-images-per-second", type=float, default=1000.0)
    parser.add_argument("--minimum-speedup", type=float, default=2.5)
    parser.add_argument("--maximum-cpu-temperature-c", type=float, default=95.0)
    return parser


def _finite_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def evaluate(
    payload: dict[str, Any],
    *,
    minimum_images_per_second: float,
    minimum_speedup: float,
    maximum_cpu_temperature_c: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    baseline = _finite_number(payload, "baseline_images_per_second")
    throughput = _finite_number(payload, "images_per_second")
    temperature = _finite_number(payload, "maximum_cpu_temperature_c")
    throttle_delta = _finite_number(payload, "thermal_throttle_delta")
    training_loss = _finite_number(payload, "training_loss")
    gradient_norm = _finite_number(payload, "gradient_norm")
    speedup = throughput / baseline if baseline > 0.0 else 0.0

    if throughput < minimum_images_per_second:
        failures.append(
            f"throughput {throughput:.2f} < {minimum_images_per_second:.2f} img/s"
        )
    if speedup < minimum_speedup:
        failures.append(f"speedup {speedup:.2f}x < {minimum_speedup:.2f}x")
    if temperature >= maximum_cpu_temperature_c:
        failures.append(
            f"CPU temperature {temperature:.1f} >= "
            f"{maximum_cpu_temperature_c:.1f} C"
        )
    if throttle_delta != 0.0:
        failures.append(f"thermal throttle delta is {throttle_delta:g}, expected 0")
    if not bool(payload.get("cpu23_offline")):
        failures.append("CPU23 is not offline")
    if payload.get("recurrence_backend") != "triton_fused":
        failures.append("optimized triton_fused recurrence backend was not verified")
    if not bool(payload.get("training_semantics_unchanged")):
        failures.append("training/model/dataset semantics were not verified unchanged")
    if not bool(payload.get("regression_tests_passed")):
        failures.append("regression tests did not pass")
    if training_loss < 0.0:
        failures.append("training loss is negative")
    if gradient_norm < 0.0:
        failures.append("gradient norm is negative")
    return not failures, failures


def main() -> int:
    args = _parser().parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    passed, failures = evaluate(
        payload,
        minimum_images_per_second=args.minimum_images_per_second,
        minimum_speedup=args.minimum_speedup,
        maximum_cpu_temperature_c=args.maximum_cpu_temperature_c,
    )
    result = {
        "status": "pass" if passed else "fail",
        "artifact": str(args.artifact),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
