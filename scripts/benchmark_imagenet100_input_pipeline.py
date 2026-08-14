"""Benchmark the shared ImageNet-100 decode/augment/H2D pipeline."""

from __future__ import annotations

# ruff: noqa: EM101, PLC0415, SLF001, T201, TRY003
import argparse
import gc
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import run_alphabet2d_imagenet100_nano as nano
import torch
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Iterator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--baseline-workers", type=int, default=8)
    parser.add_argument("--candidate-workers", type=int, default=16)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--measure-batches", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--require-speedup", type=float, default=1.0)
    parser.add_argument(
        "--p4-smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


@contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _measure(
    data_root: Path,
    *,
    batch_size: int,
    workers: int,
    warmup_batches: int,
    measure_batches: int,
) -> tuple[float, dict[str, object], tuple[torch.Tensor, torch.Tensor]]:
    settings = {"LNET_DATALOADER_WORKERS": str(workers)}
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(501)
    with _environment(settings):
        loader, _ = nano._loaders(
            data_root,
            batch_size=batch_size,
            workers=workers,
            training_generator=generator,
        )
        batches = nano._device_batches(loader, device, channels_last=True)
        checksum = torch.zeros((), device=device)
        for _ in range(warmup_batches):
            inputs, targets = next(batches)
            checksum = checksum + inputs[:, :, :2, :2].sum() + targets[:1].sum()
        torch.cuda.synchronize()
        started = time.perf_counter()
        sample: tuple[torch.Tensor, torch.Tensor] | None = None
        for _ in range(measure_batches):
            inputs, targets = next(batches)
            checksum = checksum + inputs[:, :, :2, :2].sum() + targets[:1].sum()
            if sample is None:
                sample = inputs[:2].clone(), targets[:2].clone()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if sample is None:
            raise RuntimeError("input-pipeline benchmark measured no batches")
        sample_inputs, sample_targets = sample
        metadata = {
            "workers": workers,
            "prefetch_factor": nano.PREFETCH_FACTOR,
            "batch_shape": list(inputs.shape),
            "dtype": str(inputs.dtype),
            "channels_last": inputs.is_contiguous(memory_format=torch.channels_last),
            "finite": bool(torch.isfinite(inputs).all()),
            "checksum": float(checksum),
        }
        del batches, loader
    gc.collect()
    return batch_size * measure_batches / elapsed, metadata, (sample_inputs, sample_targets)


def _p4_smoke(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    import run_a2d_p4_imagenet100 as p4

    from lnet.complex_scan import ComplexScanConfig

    model = p4._build(
        p4.VARIANT,
        ComplexScanConfig(
            output_dim=100,
            stem_strides=(2, 2),
        ),
    ).cuda().train()
    model.to(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(logits, dict):
            logits = logits["logits"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("P4 smoke model did not return tensor logits")
        loss = functional.cross_entropy(logits, targets)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("P4 data-pipeline smoke produced a non-finite loss")
    return float(loss.detach())


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet-100 input-pipeline benchmark requires CUDA")
    if min(args.batch_size, args.measure_batches, args.trials) <= 0:
        raise ValueError("batch size, measured batches, and trials must be positive")
    torch.manual_seed(501)
    # Prime filesystem caches without including first-use penalties.
    for workers in (args.baseline_workers, args.candidate_workers):
        _measure(
            args.data_root,
            batch_size=args.batch_size,
            workers=workers,
            warmup_batches=2,
            measure_batches=2,
        )

    measurements: dict[str, list[float]] = {"baseline": [], "candidate": []}
    metadata: dict[str, dict[str, object]] = {}
    candidate_sample: tuple[torch.Tensor, torch.Tensor] | None = None
    for trial in range(args.trials):
        order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
        for name in order:
            candidate = name == "candidate"
            throughput, current, sample = _measure(
                args.data_root,
                batch_size=args.batch_size,
                workers=(args.candidate_workers if candidate else args.baseline_workers),
                warmup_batches=args.warmup_batches,
                measure_batches=args.measure_batches,
            )
            measurements[name].append(throughput)
            metadata[name] = current
            if candidate:
                candidate_sample = sample

    baseline = statistics.median(measurements["baseline"])
    candidate = statistics.median(measurements["candidate"])
    speedup = candidate / baseline
    smoke_loss = None
    if args.p4_smoke:
        if candidate_sample is None:
            raise RuntimeError("candidate sample is missing")
        smoke_loss = _p4_smoke(*candidate_sample)
    result = {
        "device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "baseline_images_per_second": measurements["baseline"],
        "candidate_images_per_second": measurements["candidate"],
        "baseline_median_images_per_second": baseline,
        "candidate_median_images_per_second": candidate,
        "speedup": speedup,
        "required_speedup": args.require_speedup,
        "metadata": metadata,
        "p4_smoke_loss": smoke_loss,
        "passed": speedup >= args.require_speedup,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if speedup < args.require_speedup:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
