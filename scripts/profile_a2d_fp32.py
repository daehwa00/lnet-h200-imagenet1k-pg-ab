#!/usr/bin/env python3
"""Profile one compiled FP32 A2D ImageNet-size training step."""

from __future__ import annotations

# ruff: noqa: SLF001, T201
import argparse
import sys
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional
from torch.profiler import ProfilerActivity, profile

from lnet.pac_capture_safe_orthogonal import prepare_capture_safe_orthogonal_
from lnet.complex_scan import ComplexScanConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_double_prefc_imagenet100 as a2d_runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--row-limit", type=int, default=80)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        message = "CUDA is required"
        raise RuntimeError(message)
    torch.manual_seed(521)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = a2d_runner._build(a2d_runner.A2D, config)
    prepare_capture_safe_orthogonal_(model)
    model = model.cuda().train().to(memory_format=torch.channels_last)
    runtime = cast("nn.Module", torch.compile(model, dynamic=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, fused=True)
    inputs = torch.randn(
        args.batch_size, 3, 224, 224, device="cuda", dtype=torch.float32
    ).contiguous(memory_format=torch.channels_last)
    targets = torch.randint(100, (args.batch_size,), device="cuda")

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(runtime(inputs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        optimizer.step()

    for _ in range(args.warmups):
        step()
    torch.cuda.synchronize()
    with profile(activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA)) as result:
        step()
        torch.cuda.synchronize()
    print(
        result.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=args.row_limit,
        )
    )


if __name__ == "__main__":
    main()
