#!/usr/bin/env python3
"""Compare foreach/default and fused AdamW on the compiled FP32 A2D step."""

from __future__ import annotations

# ruff: noqa: SLF001, T201
import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import functional

from lnet.pac_capture_safe_orthogonal import prepare_capture_safe_orthogonal_
from lnet.complex_scan import ComplexScanConfig

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_double_prefc_imagenet100 as a2d_runner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--blocks", type=int, default=8)
    return parser


def _model(state: dict[str, torch.Tensor] | None = None) -> nn.Module:
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = a2d_runner._build(a2d_runner.A2D, config)
    prepare_capture_safe_orthogonal_(model)
    if state is not None:
        model.load_state_dict(state)
    return model.cuda().train().to(memory_format=torch.channels_last)


def _optimizer(model: nn.Module, *, fused: bool) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=1.0e-3, fused=fused)


def _step(
    runtime: Callable[[torch.Tensor], torch.Tensor],
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = functional.cross_entropy(runtime(inputs), targets)
    loss.backward()
    optimizer.step()


def _block(step: Callable[[], None], *, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _parameter_error(reference: nn.Module, candidate: nn.Module) -> dict[str, float]:
    maximum = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    for expected, actual in zip(
        reference.parameters(), candidate.parameters(), strict=True
    ):
        difference = actual.detach().double() - expected.detach().double()
        maximum = max(maximum, float(difference.abs().max()))
        squared_error += float(difference.square().sum())
        squared_reference += float(expected.detach().double().square().sum())
    return {
        "maximum_absolute": maximum,
        "relative_rmse": (squared_error / max(squared_reference, 1.0e-30)) ** 0.5,
    }


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        message = "CUDA is required"
        raise RuntimeError(message)
    torch.manual_seed(509)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    template = _model().cpu()
    initial = copy.deepcopy(template.state_dict())
    del template
    inputs = torch.randn(
        args.batch_size,
        3,
        224,
        224,
        device="cuda",
        dtype=torch.float32,
    ).contiguous(memory_format=torch.channels_last)
    targets = torch.randint(100, (args.batch_size,), device="cuda")

    models = {
        "reference": _model(initial),
        "candidate": _model(initial),
    }
    runtimes = {
        name: cast("Callable[[torch.Tensor], torch.Tensor]", torch.compile(model))
        for name, model in models.items()
    }
    optimizers = {
        "reference": _optimizer(models["reference"], fused=False),
        "candidate": _optimizer(models["candidate"], fused=True),
    }
    steps = {
        name: (
            lambda active=name: _step(
                runtimes[active], optimizers[active], inputs, targets
            )
        )
        for name in models
    }

    # A single matched update checks that the optimizer swap preserves the FP32
    # training trajectory up to ordinary CUDA implementation rounding.
    steps["reference"]()
    steps["candidate"]()
    parity = _parameter_error(models["reference"], models["candidate"])
    for _ in range(args.warmups - 1):
        steps["reference"]()
        steps["candidate"]()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {"reference": [], "candidate": []}
    for block in range(args.blocks):
        order = (
            ("reference", "candidate")
            if block % 2 == 0
            else ("candidate", "reference")
        )
        for name in order:
            samples[name].append(_block(steps[name], iterations=args.iterations))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "precision": "float32",
                "tf32": False,
                "batch_size": args.batch_size,
                "reference": "AdamW foreach/default",
                "candidate": "AdamW fused",
                "parameter_parity_after_one_step": parity,
                "samples_ms": samples,
                "medians_ms": medians,
                "speedup": medians["reference"] / medians["candidate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
