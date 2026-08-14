# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Protocol, cast

import torch

from .pac_training_cuda_graph_benchmark import (
    CURRENT_BACKENDS,
    _build_graph_context,
    _release_cuda,
    _set_seed,
)
from .pac_training_exact_split_benchmark import DEFAULT_CONFIG, _graph_config
from .pac_training_speed_comparison import TrainingModelName, build_training_model
from .pac_training_ultimate_benchmark import _build_context


class _ProfilerEvent(Protocol):
    key: str
    count: int


_PROFILE_CELLS: tuple[tuple[TrainingModelName, int, int], ...] = (
    ("efp16", 128, 64),
    ("efp16", 512, 64),
    ("pa2wp", 128, 64),
    ("efp16", 2048, 1),
    ("pa2wp", 2048, 64),
)
_FULL_GRAPH_PROFILE_OPTIONS: dict[tuple[TrainingModelName, int, int], tuple[str, str | None]] = {
    ("efp16", 2048, 1): ("float32", None),
    ("pa2wp", 2048, 64): ("float32", "auto"),
}


def _event_time(event: object, *, device: bool, self_time: bool) -> float:
    prefix = "self_" if self_time else ""
    names = (
        f"{prefix}{'device' if device else 'cpu'}_time_total",
        f"{prefix}{'cuda' if device else 'cpu'}_time_total",
    )
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(cast("float | int", value))
    return 0.0


def profile_ultimate_cell(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    *,
    warmups: int,
    steps: int,
    trace_output: Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        message = "ultimate training profiling requires CUDA"
        raise RuntimeError(message)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _set_seed(DEFAULT_CONFIG.seed)
    base_model, _ = build_training_model(model_name, length, batch_size)
    state_dict = copy.deepcopy(base_model.state_dict())
    del base_model
    generator = torch.Generator().manual_seed(DEFAULT_CONFIG.seed + length + batch_size)
    cpu_inputs = torch.randn(batch_size, length, 1, generator=generator)
    cpu_labels = torch.randint(0, 5, (batch_size,), generator=generator)
    backend = CURRENT_BACKENDS[(model_name, length, batch_size)]
    full_graph_options = _FULL_GRAPH_PROFILE_OPTIONS.get((model_name, length, batch_size))
    if full_graph_options is None:
        context = _build_context(
            model_name,
            length,
            batch_size,
            backend,
            state_dict,
            cpu_inputs,
            cpu_labels,
            ultimate=True,
        )
    else:
        graph_compute_dtype, recurrence_override = full_graph_options
        context = _build_graph_context(
            model_name,
            length,
            batch_size,
            backend,
            graph_compute_dtype=graph_compute_dtype,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=_graph_config(DEFAULT_CONFIG),
            device="cuda",
            fused_recurrence_moments_backward_training=True,
            recurrence_backend_override=recurrence_override,
        )
    loss = context.step()
    for _ in range(warmups - 1):
        loss = context.step()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
    ) as profiler:
        for _ in range(steps):
            with torch.profiler.record_function(
                f"{model_name}/N{length}/B{batch_size}/ultimate_step"
            ):
                loss = context.step()
    torch.cuda.synchronize()
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_output))
    events = [
        event
        for event in cast("list[_ProfilerEvent]", list(profiler.key_averages()))
        if not event.key.endswith("/ultimate_step")
    ]
    device_events = sorted(
        events,
        key=lambda event: _event_time(event, device=True, self_time=True),
        reverse=True,
    )
    cpu_events = sorted(
        events,
        key=lambda event: _event_time(event, device=False, self_time=True),
        reverse=True,
    )

    def serialize(event: _ProfilerEvent) -> dict[str, object]:
        return {
            "name": event.key,
            "calls": event.count,
            "self_device_time_total_us": _event_time(event, device=True, self_time=True),
            "device_time_total_us": _event_time(event, device=True, self_time=False),
            "self_cpu_time_total_us": _event_time(event, device=False, self_time=True),
            "cpu_time_total_us": _event_time(event, device=False, self_time=False),
        }

    result: dict[str, object] = {
        "model": model_name,
        "length": length,
        "batch_size": batch_size,
        "backend": backend,
        "warmups": warmups,
        "profiled_steps": steps,
        "last_loss": float(loss.detach().item()),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "trace": str(trace_output),
        "dominant_device_operations": [serialize(event) for event in device_events[:30]],
        "dominant_cpu_operations": [serialize(event) for event in cpu_events[:20]],
    }
    del context
    _release_cuda()
    return result


def profile_ultimate_training(*, output_dir: Path, warmups: int, steps: int) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for model_name, length, batch_size in _PROFILE_CELLS:
        trace = output_dir / f"{model_name}-N{length}-B{batch_size}-trace.json"
        cells.append(
            profile_ultimate_cell(
                model_name,
                length,
                batch_size,
                warmups=warmups,
                steps=steps,
                trace_output=trace,
            )
        )
    dominant_kernels: list[dict[str, object]] = []
    for cell in cells:
        dominant_kernels.extend(
            {
                "cell": [cell["model"], cell["length"], cell["batch_size"]],
                "name": event["name"],
                "calls": event["calls"],
                "total_us": event["self_device_time_total_us"],
                "per_step_us": float(cast("float", event["self_device_time_total_us"])) / steps,
            }
            for event in cast("list[dict[str, object]]", cell["dominant_device_operations"])[:8]
        )
    dominant_kernels.sort(
        key=lambda event: float(cast("float", event["per_step_us"])),
        reverse=True,
    )
    return {
        "schema": "pac_training_ultimate_profile.v1",
        "profiled_cells": [list(cell) for cell in _PROFILE_CELLS],
        "warmups": warmups,
        "profiled_steps": steps,
        "cells": cells,
        "dominant_kernels": dominant_kernels[:24],
        "remaining_host_boundaries": [
            "one two-frame norm device-to-host branch selection before direct matrix_exp",
            "one two-frame norm device-to-host branch selection before matrix_exp VJP",
            "PA2WP scalar phase schedule host refill once per 64 steps",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.steps < 1:
        message = "warmups and steps must be positive"
        raise ValueError(message)
    result = profile_ultimate_training(
        output_dir=arguments.output_dir,
        warmups=arguments.warmups,
        steps=arguments.steps,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
