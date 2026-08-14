#!/usr/bin/env python3
"""Profile a production D4-Cal-U96-Stem32 BF16 training step."""

from __future__ import annotations

# pyright: reportExplicitAny=false
# ruff: noqa: C901, PLR0915, SLF001, T201
import argparse
import importlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

if TYPE_CHECKING:
    from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))

_RUNNERS = {
    "phase-gated": "run_a2d_deep4_calibrated_uniform_p96_phase_gated_imagenet100",
    "stemres": "run_a2d_deep4_calibrated_uniform_p96_stemres_imagenet100",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--runner", choices=tuple(_RUNNERS), default="phase-gated")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--row-limit", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    return parser


def _build(
    candidate: ModuleType,
    runner: str,
    contract: dict[str, Any],
    checkpoint: Path | None,
    compile_mode: str,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer, dict[str, Any], Any, Any]:
    candidate._configure_ramp()
    ramp = (
        candidate.control.stemres.uniform.base
        if runner == "phase-gated"
        else candidate.uniform.base
    )
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = ramp.backbone.a2d_base.residuals
    harness = source.heads.harness
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    recipe = cast("dict[str, Any]", contract["recipe"])
    model = candidate._build(
        candidate.VARIANT,
        ramp.PoleModelConfig(
            output_dim=100,
            stem_strides=(2, 2),
        ),
    ).cuda()
    expected_parameters = cast("dict[str, int]", contract["parameter_counts"])[
        candidate.VARIANT
    ]
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != expected_parameters:
        message = (
            f"profile model {candidate.VARIANT} does not match its contract: "
            f"{actual_parameters} != {expected_parameters}"
        )
        raise RuntimeError(message)
    model = source._prepare_model(model, recipe)
    optimizer = residuals.optimizer_source._build_optimizer(model, recipe)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        harness._restore_optimizer_runtime_options(optimizer, recipe)
    os.environ["LNET_COMPILE_MODE"] = compile_mode
    runtime = harness._build_runtime(model, recipe)
    return model, runtime, optimizer, recipe, harness, source.structured


def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _elapsed(pair: tuple[torch.cuda.Event, torch.cuda.Event]) -> float:
    return float(pair[0].elapsed_time(pair[1]))


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        message = "CUDA is required"
        raise RuntimeError(message)
    if args.batch_size <= 0 or args.warmups < 1 or args.steps < 2:
        message = "profiling requires a positive batch and at least one warmup/two steps"
        raise ValueError(message)
    if args.gradient_accumulation_steps <= 0:
        message = "gradient accumulation steps must be positive"
        raise ValueError(message)

    torch.manual_seed(20260809)
    torch.cuda.manual_seed_all(20260809)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    candidate = importlib.import_module(_RUNNERS[args.runner])
    contract = json.loads(args.contract.read_text())
    model, runtime, optimizer, recipe, harness, structured = _build(
        candidate,
        args.runner,
        contract,
        args.checkpoint,
        args.compile_mode,
    )
    device = torch.device("cuda")
    channels_last = bool(recipe.get("channels_last", False))
    inputs = torch.randn(args.batch_size, 3, 224, 224, device=device)
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    targets = torch.randint(100, (args.batch_size,), device=device)
    permutation = torch.randperm(args.batch_size, device=device)
    mixing = 0.4
    accumulation_steps = args.gradient_accumulation_steps

    def training_microbatch(index: int, *, timed: bool) -> dict[str, float]:
        group_offset = index % accumulation_steps
        if group_offset == 0:
            optimizer.zero_grad(set_to_none=True)
        phase_names = ("mix", "forward", "loss", "backward", "clip", "optimizer")
        events = {name: _event_pair() for name in phase_names}

        events["mix"][0].record()
        with record_function("phase/mix"):
            mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
        events["mix"][1].record()

        harness._begin_cudagraph_step(device)
        events["forward"][0].record()
        with (
            record_function("phase/forward"),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ),
        ):
            output = runtime(mixed_inputs)
        events["forward"][1].record()

        events["loss"][0].record()
        with record_function("phase/loss"):
            _logits, loss, _diagnostics = structured._training_objective(
                model,
                output,
                targets,
                targets[permutation],
                mixing,
            )
        events["loss"][1].record()

        events["backward"][0].record()
        with record_function("phase/backward"):
            (loss / accumulation_steps).backward()
        events["backward"][1].record()

        if group_offset == accumulation_steps - 1:
            events["clip"][0].record()
            with record_function("phase/clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            events["clip"][1].record()
            events["optimizer"][0].record()
            with record_function("phase/optimizer"):
                optimizer.step()
            events["optimizer"][1].record()
        else:
            for name in ("clip", "optimizer"):
                events[name][0].record()
                events[name][1].record()

        if not timed:
            return {}
        torch.cuda.synchronize()
        return {name: _elapsed(events[name]) for name in phase_names}

    model.train()
    runtime.train()
    warmup_microbatches = args.warmups * accumulation_steps
    for index in range(warmup_microbatches):
        training_microbatch(index, timed=False)
    torch.cuda.synchronize()

    samples = [training_microbatch(index, timed=True) for index in range(args.steps)]
    phase_means = {
        name: statistics.fmean(sample[name] for sample in samples) for name in samples[0]
    }
    totals = [sum(sample.values()) for sample in samples]
    phase_means["total"] = statistics.fmean(totals)
    phase_means["images_per_second"] = args.batch_size * 1000.0 / phase_means["total"]
    total_median = statistics.median(totals)

    optimizer.zero_grad(set_to_none=True)
    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=True,
        profile_memory=False,
    ) as result:
        training_microbatch(1, timed=False)
        torch.cuda.synchronize()
    table = result.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total",
        row_limit=args.row_limit,
    )
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        result.export_chrome_trace(str(args.trace))

    payload = {
        "schema": "lnet.a2d_u96_stem32_bf16_profile.v1",
        "device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "runner": args.runner,
        "variant": candidate.VARIANT,
        "compile_mode": args.compile_mode,
        "precision": "bfloat16",
        "gradient_accumulation_steps": accumulation_steps,
        "warmup_microbatches": warmup_microbatches,
        "measured_microbatches": args.steps,
        "phase_mean_ms_per_microbatch": phase_means,
        "median_total_ms_per_microbatch": total_median,
        "median_images_per_second": args.batch_size * 1000.0 / total_median,
        "phase_samples_ms": samples,
        "profiler_table": table,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "profiler_table"}))
    print(table)


if __name__ == "__main__":
    main()
