#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false
"""Benchmark one fixed-shape ImageNet-100 complex scan training step."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
from lnet.pac_capture_safe_orthogonal import prepare_capture_safe_orthogonal_

Candidate = str


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=(
            "eager",
            "compile-default",
            "compile-reduce-overhead",
            "extreme",
            "extreme-default",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--regression-tests-passed", action="store_true")
    return parser.parse_args()


def _config() -> ComplexScanConfig:
    base = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    return replace(
        base,
        augmented_widths=(48, 64),
        carry_bases=("s2d", "s2d"),
        carry_merge="pole_main",
        carry_scale_initial=1.0e-2,
    )


def _model(
    state: dict[str, Tensor] | None = None,
    *,
    optimized: bool = False,
) -> ComplexScanBackbone:
    model = ComplexScanBackbone(_config()).cuda().train()
    if state is not None:
        model.load_state_dict(state)
    if optimized:
        prepare_capture_safe_orthogonal_(model)
        model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    return model


def _runtime(model: ComplexScanBackbone, candidate: Candidate) -> nn.Module:
    if candidate == "eager":
        return model
    if candidate in {"compile-default", "extreme-default"}:
        mode = "default"
    elif candidate == "compile-reduce-overhead":
        mode = "reduce-overhead"
    else:
        mode = "max-autotune-no-cudagraphs"
    return cast(
        "nn.Module",
        torch.compile(model, mode=mode, fullgraph=False, dynamic=False),
    )


def _optimizer(model: nn.Module, *, fused: bool = False) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    modal: list[nn.Parameter] = []
    geometry: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if "damping_logits" in name or "phase_" in name:
            geometry.append(parameter)
        elif (
            "analysis" in name
            or "augmented.direction_mixer" in name
            or "augmented.output_projection" in name
        ):
            modal.append(parameter)
        elif parameter.ndim < 2 or "norm" in name or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    learning_rate = 3.0e-3
    return torch.optim.AdamW(
        [
            {"params": decay, "lr": learning_rate, "weight_decay": 0.05},
            {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0},
            {
                "params": modal,
                "lr": learning_rate / 3.0,
                "weight_decay": 0.0,
            },
            {
                "params": geometry,
                "lr": learning_rate * 0.1,
                "weight_decay": 0.0,
            },
        ],
        fused=fused,
    )


def _forward_backward(
    runtime: nn.Module,
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = runtime(inputs)
        loss = functional.cross_entropy(logits, targets, label_smoothing=0.1)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return logits.detach().float().cpu().clone(), loss.detach().float().cpu(), gradients


def _gradient_error(
    reference: dict[str, Tensor],
    candidate: dict[str, Tensor],
) -> tuple[float, float]:
    if reference.keys() != candidate.keys():
        message = "candidate and reference gradient keys differ"
        raise RuntimeError(message)
    squared_error = 0.0
    squared_reference = 0.0
    maximum = 0.0
    for name, reference_gradient in reference.items():
        difference = candidate[name] - reference_gradient
        maximum = max(maximum, float(difference.abs().max()))
        squared_error += float(difference.double().square().sum())
        squared_reference += float(reference_gradient.double().square().sum())
    relative_rmse = math.sqrt(squared_error / max(squared_reference, 1.0e-24))
    return maximum, relative_rmse


def _parity(
    state: dict[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    candidate_name: Candidate,
) -> tuple[ComplexScanBackbone, nn.Module, dict[str, float]]:
    optimized = candidate_name.startswith("extreme")
    if optimized:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    reference_model = _model(state, optimized=optimized)
    reference_logits, reference_loss, reference_gradients = _forward_backward(
        reference_model,
        reference_model,
        inputs,
        targets,
    )
    del reference_model
    torch.cuda.empty_cache()

    candidate_model = _model(state, optimized=optimized)
    runtime = _runtime(candidate_model, candidate_name)
    candidate_logits, candidate_loss, candidate_gradients = _forward_backward(
        runtime,
        candidate_model,
        inputs,
        targets,
    )
    gradient_maximum, gradient_relative_rmse = _gradient_error(
        reference_gradients,
        candidate_gradients,
    )
    return candidate_model, runtime, {
        "logits_max_abs": float((candidate_logits - reference_logits).abs().max()),
        "loss_abs": float((candidate_loss - reference_loss).abs()),
        "gradient_max_abs": gradient_maximum,
        "gradient_relative_rmse": gradient_relative_rmse,
    }


def _step(
    runtime: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = runtime(inputs)
        loss = functional.cross_entropy(logits, targets, label_smoothing=0.1)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach()), float(gradient_norm.detach())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        message = "the complex scan backbone speed benchmark requires CUDA"
        raise RuntimeError(message)
    if min(args.batch_size, args.warmups, args.iterations, args.rounds) <= 0:
        message = "benchmark dimensions and repetition counts must be positive"
        raise ValueError(message)
    emulate_precision_casts = os.environ.get("TORCHINDUCTOR_EMULATE_PRECISION_CASTS") == "1"
    optimized = args.candidate.startswith("extreme")
    if optimized and not emulate_precision_casts:
        message = "extreme benchmark requires TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1"
        raise RuntimeError(message)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    initial = ComplexScanBackbone(_config())
    state = {name: value.detach().cpu().clone() for name, value in initial.state_dict().items()}
    del initial
    inputs = torch.randn(args.batch_size, 3, 224, 224, device="cuda")
    targets = torch.randint(100, (args.batch_size,), device="cuda")
    model, runtime, parity = _parity(state, inputs, targets, args.candidate)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if optimized:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    optimizer = _optimizer(model, fused=optimized)

    last_loss = math.nan
    last_gradient_norm = math.nan
    for _ in range(args.warmups):
        last_loss, last_gradient_norm = _step(
            runtime,
            model,
            optimizer,
            inputs,
            targets,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    round_seconds: list[float] = []
    for _ in range(args.rounds):
        started = time.perf_counter()
        for _ in range(args.iterations):
            last_loss, last_gradient_norm = _step(
                runtime,
                model,
                optimizer,
                inputs,
                targets,
            )
        torch.cuda.synchronize()
        round_seconds.append(time.perf_counter() - started)

    seconds_per_step = [seconds / args.iterations for seconds in round_seconds]
    median_step = statistics.median(seconds_per_step)
    payload: dict[str, Any] = {
        "schema": "lnet.complex_scan.imagenet100_speed.v2",
        "candidate": args.candidate,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "seconds_per_step": seconds_per_step,
        "median_milliseconds_per_step": 1000.0 * median_step,
        "images_per_second": args.batch_size / median_step,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "parameters": parameters,
        "optimization_bundle": {
            "capture_safe_orthogonal": optimized,
            "channels_last": optimized,
            "compile_mode": (
                "max-autotune-no-cudagraphs"
                if args.candidate == "extreme"
                else "default"
                if args.candidate == "extreme-default"
                else args.candidate
            ),
            "fused_adamw": optimized,
            "emulate_precision_casts": emulate_precision_casts,
        },
        "scan_pipeline": "associative_product",
        "parity": parity,
        "training_loss": last_loss,
        "gradient_norm": last_gradient_norm,
        "regression_tests_passed": args.regression_tests_passed,
        "source_sha256": {
            "model": _digest(Path("src/lnet/complex_scan.py")),
            "product_scan_pipeline": _digest(
                Path("src/lnet/pac_product_scan_pipeline.py")
            ),
            "benchmark": _digest(Path(__file__)),
            "capture_safe_orthogonal": _digest(
                Path("src/lnet/pac_capture_safe_orthogonal.py")
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
