#!/usr/bin/env python3
"""Smoke one R2K3 campaign candidate on CPU or CUDA.

The runner is selected with ``--runner`` instead of by one shim file per
campaign.  Single-variant runners emit the flat evidence shape and
multi-variant runners emit the per-variant shape their queues consume.
"""

from __future__ import annotations

# Reuse the established factorial smoke algebra and compiler checks.
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import r2k3_runner_registry as registry
import smoke_a2d_r2k3_same_resolution_factorial as shared
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    registry.add_runner_argument(parser)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--full-batch", action="store_true")
    parser.add_argument("--full-batch-size", type=int, default=128)
    parser.add_argument("--repeat-steps", type=int, default=3)
    return parser.parse_args(argv)


def _device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested without an available CUDA device")
    if (args.compile or args.full_batch) and device.type != "cuda":
        raise RuntimeError("compiled and full-batch smokes require CUDA")
    return device


def _single_variant_payload(
    active_runner: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "variant": active_runner.VARIANT,
        "candidate": shared._run_candidate(
            active_runner.VARIANT,
            device,
            args.root,
            compile_model=args.compile and not args.full_batch,
        ),
        "full_batch": (
            shared._full_batch_cuda_step(
                active_runner.VARIANT,
                args.full_batch_size,
                args.repeat_steps,
            )
            if args.full_batch
            else None
        ),
    }


def _multi_variant_payload(
    active_runner: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    full_batch: dict[str, Any] = {}
    for variant in active_runner.VARIANTS:
        variants[variant] = shared._run_candidate(
            variant,
            device,
            args.root,
            compile_model=args.compile and not args.full_batch,
        )
        if args.full_batch:
            full_batch[variant] = shared._full_batch_cuda_step(
                variant,
                args.full_batch_size,
                args.repeat_steps,
            )
    return {
        "candidate_count": len(active_runner.VARIANTS),
        "variants": variants,
        "full_batch": full_batch,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    active_runner = args.runner
    args.root.mkdir(parents=True, exist_ok=True)
    device = _device(args)

    # The shared factorial smoke reads its runner from a module global.
    shared.runner = active_runner  # type: ignore[reportConstantRedefinition]
    torch.set_float32_matmul_precision("high")

    payload: dict[str, Any] = {
        "device": str(device),
        "cell_reconstruction_max_abs": shared._cell_reconstruction_check(device),
        "full_state_semantic_max_abs": shared._full_state_semantic_check(device),
    }
    if getattr(active_runner, "VARIANTS", None):
        payload.update(_multi_variant_payload(active_runner, args, device))
    else:
        payload.update(_single_variant_payload(active_runner, args, device))
    payload["status"] = "passed"

    (args.root / "smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
