#!/usr/bin/env python3
"""Tune this machine's Triton launch geometries against a real model step.

Run once per GPU.  The winners are written to the per-device store that every
later run -- eager or compiled -- reads and pins.

    python scripts/tune_launch_geometry.py --runner run_a2d_resaux1_imagenet100
"""

# ruff: noqa: C901, PLR0912, PLR0915, SLF001, T201

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from lnet.complex_scan import ComplexScanConfig
from lnet.pac_kernel_launch_config import device_key, stored_geometry
from lnet.pac_launch_tuning import TuningResult, tune_all

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


@contextmanager
def _compiled_cache_rotation(*, enabled: bool) -> Generator[Callable[[], None]]:
    """Give each compiled candidate a fresh cache and remove it afterwards."""
    previous = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    active: tempfile.TemporaryDirectory[str] | None = None

    def rotate() -> None:
        nonlocal active
        if not enabled:
            return
        torch.compiler.reset()
        if active is not None:
            active.cleanup()
        active = tempfile.TemporaryDirectory(prefix="lnet_tune_")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = active.name

    try:
        yield rotate
    finally:
        if active is not None:
            active.cleanup()
        if previous is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default="run_a2d_resaux1_imagenet100")
    parser.add_argument(
        "--variant",
        help=(
            "runner variant to calibrate; required when the runner exposes "
            "multiple VARIANTS without one default VARIANT"
        ),
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--step-iterations",
        type=int,
        default=20,
        help="whole forward/backward steps used for the before/after timing",
    )
    parser.add_argument(
        "--step-repeats",
        type=int,
        default=1,
        help="independent whole-step samples; the reported result is their median",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="measure and report without writing the store",
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help=(
            "compile and execute the real step so Triton selects and caches "
            "registered candidates without recompiling the whole model per candidate"
        ),
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help=(
            "tune only these registered kernels.  Compiled tuning recompiles "
            "once per candidate, so narrowing is usually necessary."
        ),
    )
    parser.add_argument(
        "--regime",
        choices=("compiled", "eager"),
        default="compiled",
        help=(
            "measure in the regime you will train in.  Eager winners were "
            "measured not to transfer to compiled training on an RTX 4090, so "
            "eager is a prefilter only."
        ),
    )
    return parser.parse_args()


def _runner_variant(runner: object, requested: str | None) -> str:
    """Resolve one explicit model graph without silently choosing an ablation."""
    default: object = getattr(runner, "VARIANT", None)
    declared: object = getattr(runner, "VARIANTS", ())
    if default is not None and not isinstance(default, str):
        message = "runner VARIANT must be a string"
        raise TypeError(message)
    if isinstance(declared, str):
        message = "runner VARIANTS must be a sequence of strings"
        raise TypeError(message)
    if not isinstance(declared, (list, tuple)):
        message = "runner VARIANTS must be a sequence of strings"
        raise TypeError(message)
    untyped_variants: tuple[object, ...] = tuple(declared)
    if any(type(value) is not str for value in untyped_variants):
        message = "runner VARIANTS must contain only strings"
        raise TypeError(message)
    variants = cast("tuple[str, ...]", untyped_variants)
    if default is not None and default not in variants:
        variants = (default, *variants)
    if requested is not None:
        if requested not in variants:
            choices = ", ".join(variants) or "<none>"
            message = f"unsupported runner variant {requested!r}; choose one of: {choices}"
            raise ValueError(message)
        return requested
    if default is not None:
        return default
    if len(variants) == 1:
        return variants[0]
    if not variants:
        message = "runner exposes neither VARIANT nor VARIANTS"
        raise ValueError(message)
    choices = ", ".join(variants)
    message = f"--variant is required for this multi-variant runner: {choices}"
    raise ValueError(message)


def _print_stored_result(result: TuningResult) -> None:
    if not result.scopes:
        print(f"  {result.name} -> {stored_geometry(result.name)}")
        return
    for scope in result.scopes:
        shape = ",".join(f"{key}={value}" for key, value in scope.shape_items)
        stored = stored_geometry(result.name, scope=scope)
        print(f"  {result.name} [{scope.execution_regime}/{scope.dtype}; {shape}] -> {stored}")


def _require_requested_results(
    requested: Sequence[str] | None,
    results: Sequence[TuningResult],
) -> None:
    """Fail when an explicitly requested kernel was not observed and tuned."""
    if requested is None:
        return
    completed = {result.name for result in results}
    missing = sorted(set(requested) - completed)
    if missing:
        message = f"requested kernels were not exercised or measured: {', '.join(missing)}"
        raise RuntimeError(message)


def _synthetic_inputs(
    batch: int,
    resolution: int,
    *,
    device: torch.device | str,
    channels_last: bool,
) -> torch.Tensor:
    """Build inputs with the same memory-format contract as training."""
    inputs = torch.randn(batch, 3, resolution, resolution, device=device)
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    return inputs


def _step_time_ms(step: Callable[[], None], *, iterations: int) -> float:
    """Measure the complete GPU step around an already-compiled callable."""
    if iterations < 1:
        message = "step timing iterations must be positive"
        raise ValueError(message)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _step_time_samples(
    step: Callable[[], None],
    *,
    iterations: int,
    repeats: int,
) -> tuple[float, tuple[float, ...]]:
    """Measure repeated whole-step samples and return their robust center."""
    if repeats < 1:
        message = "step timing repeats must be positive"
        raise ValueError(message)
    samples = tuple(_step_time_ms(step, iterations=iterations) for _ in range(repeats))
    return float(statistics.median(samples)), samples


def _print_step_samples(label: str, median_ms: float, samples: Sequence[float]) -> None:
    print(f"{label}: {median_ms:.3f} ms (median of {len(samples)})")
    if len(samples) > 1:
        values = ", ".join(f"{sample:.3f}" for sample in samples)
        print(f"  samples: [{values}] ms; range={min(samples):.3f}..{max(samples):.3f}")


def _require_finite_step(loss: torch.Tensor | None, model: torch.nn.Module) -> int:
    """Reject a fast launch selection that corrupted the training step."""
    if loss is None or not bool(torch.isfinite(loss)):
        message = "calibrated step produced a non-finite loss"
        raise RuntimeError(message)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        message = "calibrated step produced no parameter gradients"
        raise RuntimeError(message)
    finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()
    if not bool(finite):
        message = "calibrated step produced non-finite parameter gradients"
        raise RuntimeError(message)
    return len(gradients)


def main() -> None:
    arguments = _parse()
    if not torch.cuda.is_available():
        message = "launch-geometry tuning requires CUDA"
        raise RuntimeError(message)
    torch.set_float32_matmul_precision("high")

    runner = importlib.import_module(arguments.runner)
    variant = _runner_variant(runner, arguments.variant)
    config = ComplexScanConfig(
        output_dim=arguments.classes,
        stem_strides=(2, 2),
    )
    torch.manual_seed(501)
    model = cast("torch.nn.Module", runner._build(variant, config)).cuda()
    if arguments.regime == "compiled":
        prepare = getattr(model, "prepare_for_compiled_training_", None)
        if callable(prepare):
            model = cast("torch.nn.Module", prepare())
        model = model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    model = model.train()
    if arguments.regime == "compiled":
        compile_profile = json.dumps(
            {
                "dynamic": False,
                "fullgraph": False,
                "mode": arguments.compile_mode,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        os.environ["LNET_COMPILE_PROFILE"] = compile_profile
        stepped = torch.compile(
            model,
            mode=arguments.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
    else:
        stepped = model
    inputs = _synthetic_inputs(
        arguments.batch,
        arguments.resolution,
        device="cuda",
        channels_last=arguments.regime == "compiled",
    )
    targets = torch.randint(0, arguments.classes, (arguments.batch,), device="cuda")
    last_loss: torch.Tensor | None = None

    def step() -> None:
        nonlocal last_loss
        model.zero_grad(set_to_none=True)
        precision_context = (
            nullcontext()
            if arguments.precision == "float32"
            else torch.autocast(
                "cuda",
                dtype={
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                }[arguments.precision],
            )
        )
        with precision_context:
            output = stepped(inputs)
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, dict):
                output = output["logits"]
            loss = torch.nn.functional.cross_entropy(output, targets)
        last_loss = loss.detach()
        loss.backward()

    print(f"device : {device_key()}")
    print(f"runner : {arguments.runner} ({variant})")
    print(f"shape  : batch={arguments.batch} resolution={arguments.resolution}")
    print(f"regime : {arguments.regime}")
    print(f"compute: {arguments.precision}")
    if arguments.regime == "compiled":
        note = (
            "note   : give each candidate a fresh TORCHINDUCTOR_CACHE_DIR, or Inductor "
            "serves the first candidate's artifact to all of them\n"
        )
        print(note)
    else:
        print("note   : eager winners were measured not to transfer to compiled\n")

    if arguments.calibrate_only:
        if arguments.regime != "compiled":
            message = "compiler-native calibration requires --regime compiled"
            raise ValueError(message)
        selected_step_ms, selected_samples = _step_time_samples(
            step,
            iterations=arguments.step_iterations,
            repeats=arguments.step_repeats,
        )
        gradient_count = _require_finite_step(last_loss, model)
        _print_step_samples("compiler-autotuned whole step", selected_step_ms, selected_samples)
        print(f"finite: loss and {gradient_count} parameter gradients")
        return

    with _compiled_cache_rotation(enabled=arguments.regime == "compiled") as rotate_cache:
        rotate_cache()
        baseline_step_ms, _ = _step_time_samples(
            step,
            iterations=arguments.step_iterations,
            repeats=arguments.step_repeats,
        )
        results = tune_all(
            step,
            names=arguments.names,
            iterations=arguments.iterations,
            store=not arguments.dry_run,
            on_geometry_change=rotate_cache,
        )
        rotate_cache()
        selected_step_ms, _ = _step_time_samples(
            step,
            iterations=arguments.step_iterations,
            repeats=arguments.step_repeats,
        )
    _require_requested_results(arguments.names, results)
    if not results:
        print("no registered kernel was exercised by this model; nothing to tune")
        return

    print(f"{'kernel':<38} {'before ms':>10} {'after ms':>9} {'gain':>7}  geometry")
    total_before = total_after = 0.0
    for result in sorted(results, key=lambda item: -(item.baseline_ms - item.winner_ms)):
        total_before += result.baseline_ms
        total_after += result.winner_ms
        blocks = ",".join(f"{key}={value}" for key, value in sorted(result.winner.blocks.items()))
        row = (
            f"{result.name:<38} {result.baseline_ms:10.3f} {result.winner_ms:9.3f} "
            f"{result.speedup:6.2f}x  warps={result.winner.num_warps} {blocks}"
        )
        print(row)
    saved = total_before - total_after
    scope_summary = f"\nsummed scope observations {total_before:.3f} -> {total_after:.3f} ms"
    scope_summary += f" ({saved:+.3f} ms)"
    print(scope_summary)
    step_saved = baseline_step_ms - selected_step_ms
    step_speedup = baseline_step_ms / selected_step_ms
    step_summary = f"whole step GPU time   {baseline_step_ms:.3f} -> {selected_step_ms:.3f} ms"
    step_summary += f" ({step_saved:+.3f} ms, {step_speedup:.3f}x)"
    print(step_summary)
    if arguments.dry_run:
        print("dry run: store not written")
    else:
        print("stored:")
        for result in results:
            _print_stored_result(result)


if __name__ == "__main__":
    main()
