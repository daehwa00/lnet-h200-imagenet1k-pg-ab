from __future__ import annotations

import argparse
import copy
import gc
import json
import platform
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Final, cast

import torch
from torch import Tensor, nn

from .pac_efp16_efficiency import EFP16EfficiencyJob, build_model
from .pac_efp16_fused_readout import prepare_efp16_fused_readout_candidate
from .pac_headroom_models import HeadroomPACClassifier
from .pac_tight_frame_runtime import prepare_efp16_ceiling_inference

_LENGTHS: Final = (128, 512, 2048)
_BATCHES: Final = (1, 64)


@dataclass(frozen=True, slots=True)
class FusedReadoutScreenConfig:
    warmups: int = 20
    groups: int = 9
    iterations_per_group: int = 100
    gpu_clock_precondition_cycles: int = 0
    seed: int = 7


DEFAULT_FUSED_READOUT_SCREEN_CONFIG = FusedReadoutScreenConfig()


def screen_efp16_fused_readout(
    *,
    lengths: tuple[int, ...] = _LENGTHS,
    batches: tuple[int, ...] = _BATCHES,
    config: FusedReadoutScreenConfig = DEFAULT_FUSED_READOUT_SCREEN_CONFIG,
) -> dict[str, object]:
    """Measure the opt-in one-kernel EFP16 readout against the current ceiling."""
    _validate_screen_inputs(lengths, batches, config)
    if not torch.cuda.is_available():
        message = "the EFP16 fused readout screen requires CUDA"
        raise RuntimeError(message)
    previous_precision = torch.get_float32_matmul_precision()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        cells = [
            _screen_cell(length, batch_size, config=config)
            for length in lengths
            for batch_size in batches
        ]
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema": "pac_efp16_fused_readout_screen.v1",
        "environment": {
            "host": platform.node(),
            "device": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": asdict(config),
        "protocol": {
            "scope": "canonical EFP16 D32/M16 invariant five-class inference",
            "candidate": (
                "one B-row Triton kernel for final RMSNorm, temporal mean, "
                "forward/backward moment contributions, classifier, and bias"
            ),
            "reference": "same-state eager FP32 logits and current final ceiling runtime",
            "dtype": "float32",
            "tf32": False,
            "autocast": False,
            "compile_and_capture_excluded": True,
            "timing": "same-run alternating raw synchronized wall and CUDA-event groups",
            "maximum_absolute_error": 2.0e-5,
            "minimum_prediction_agreement": 1.0,
            "production_dispatch_changed": False,
            "selection": "evidence-only screen; never selects by default",
        },
        "lengths": list(lengths),
        "batches": list(batches),
        "cells": cells,
        "selected": False,
    }


def _screen_cell(
    length: int,
    batch_size: int,
    *,
    config: FusedReadoutScreenConfig,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    cpu_model, _ = build_model(EFP16EfficiencyJob("efp16", length, batch_size))
    state_dict = copy.deepcopy(cpu_model.state_dict())
    del cpu_model
    generator = torch.Generator().manual_seed(config.seed + length + batch_size)
    cpu_inputs = torch.randn(batch_size, length, 1, generator=generator)

    eager_model = _load_model(length, batch_size, state_dict)
    eager_inputs = cpu_inputs.cuda()
    with torch.inference_mode():
        eager_output = eager_model(eager_inputs).detach().clone()
    del eager_model, eager_inputs
    _collect_cuda()

    baseline_model = _load_model(length, batch_size, state_dict)
    baseline_inputs = cpu_inputs.cuda()
    baseline_build_start = perf_counter()
    baseline_runtime = prepare_efp16_ceiling_inference(
        baseline_model,
        sequence_length=length,
        batch_size=batch_size,
    )
    baseline_output = _prime_runtime(baseline_runtime, baseline_inputs, config.warmups)
    baseline_build_seconds = perf_counter() - baseline_build_start

    candidate_model = _load_model(length, batch_size, state_dict)
    candidate_inputs = cpu_inputs.cuda()
    candidate_build_start = perf_counter()
    candidate_runtime = prepare_efp16_fused_readout_candidate(
        candidate_model,
        sequence_length=length,
        batch_size=batch_size,
    )
    candidate_output = _prime_runtime(candidate_runtime, candidate_inputs, config.warmups)
    candidate_build_seconds = perf_counter() - candidate_build_start

    raw = _measure_paired(
        baseline_runtime,
        baseline_inputs,
        candidate_runtime,
        candidate_inputs,
        config=config,
    )
    baseline_timing = _timing(raw["baseline"])
    candidate_timing = _timing(raw["candidate"])
    baseline_accuracy = _accuracy(baseline_output, eager_output)
    candidate_accuracy = _accuracy(candidate_output, eager_output)
    cross_error = float((candidate_output - baseline_output).abs().max().item())
    qualifies = (
        candidate_accuracy["max_abs_error"] <= 2.0e-5
        and candidate_accuracy["prediction_agreement"] == 1.0
        and cross_error <= 2.0e-5
    )
    cell: dict[str, object] = {
        "length": length,
        "batch_size": batch_size,
        "baseline": {
            "runtime": "current_ceiling_fp32",
            "timing": baseline_timing,
            "accuracy": baseline_accuracy,
            "compile_capture_seconds": baseline_build_seconds,
        },
        "candidate": {
            "runtime": "fused_readout_candidate_fp32",
            "timing": candidate_timing,
            "accuracy": candidate_accuracy,
            "compile_capture_seconds": candidate_build_seconds,
        },
        "event_speedup": cast("float", baseline_timing["cuda_event_ms"])
        / cast("float", candidate_timing["cuda_event_ms"]),
        "wall_speedup": cast("float", baseline_timing["wall_ms"])
        / cast("float", candidate_timing["wall_ms"]),
        "candidate_vs_current_max_abs_error": cross_error,
        "candidate_qualifies_accuracy": qualifies,
        "selected": False,
    }
    del (
        baseline_runtime,
        baseline_model,
        baseline_inputs,
        baseline_output,
        candidate_runtime,
        candidate_model,
        candidate_inputs,
        candidate_output,
        eager_output,
    )
    _release_cuda()
    return cell


def _measure_paired(
    baseline_runtime: nn.Module,
    baseline_inputs: Tensor,
    candidate_runtime: nn.Module,
    candidate_inputs: Tensor,
    *,
    config: FusedReadoutScreenConfig,
) -> dict[str, dict[str, list[float]]]:
    raw = {
        "baseline": {"event": [], "wall": []},
        "candidate": {"event": [], "wall": []},
    }
    runtimes = {
        "baseline": (baseline_runtime, baseline_inputs),
        "candidate": (candidate_runtime, candidate_inputs),
    }
    names = tuple(runtimes)
    for group in range(config.groups):
        order = names if group % 2 == 0 else tuple(reversed(names))
        for name in order:
            runtime, inputs = runtimes[name]
            event_ms, wall_ms = _measure_group(runtime, inputs, config=config)
            raw[name]["event"].append(event_ms)
            raw[name]["wall"].append(wall_ms)
    return raw


def _measure_group(
    runtime: nn.Module,
    inputs: Tensor,
    *,
    config: FusedReadoutScreenConfig,
) -> tuple[float, float]:
    if config.gpu_clock_precondition_cycles:
        sleeper = getattr(torch.cuda, "_sleep", None)
        if not callable(sleeper):
            message = "CUDA clock preconditioning is unavailable"
            raise TypeError(message)
        sleeper(config.gpu_clock_precondition_cycles)
        torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = perf_counter()
    start_event.record()
    with torch.inference_mode():
        for _ in range(config.iterations_per_group):
            runtime(inputs)
    end_event.record()
    end_event.synchronize()
    return (
        start_event.elapsed_time(end_event) / config.iterations_per_group,
        (perf_counter() - start_wall) * 1000.0 / config.iterations_per_group,
    )


def _timing(raw: dict[str, list[float]]) -> dict[str, object]:
    return {
        "cuda_event_ms": statistics.median(raw["event"]),
        "cuda_event_samples_ms": raw["event"],
        "wall_ms": statistics.median(raw["wall"]),
        "wall_samples_ms": raw["wall"],
        "normalized": False,
    }


def _accuracy(output: Tensor, reference: Tensor) -> dict[str, float]:
    absolute = (output - reference).abs()
    relative = absolute / reference.abs().clamp_min(1.0e-6)
    return {
        "max_abs_error": float(absolute.max().item()),
        "max_rel_error": float(relative.max().item()),
        "prediction_agreement": float(
            (output.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean().item()
        ),
    }


def _load_model(
    length: int,
    batch_size: int,
    state_dict: dict[str, Tensor],
) -> HeadroomPACClassifier:
    model, _ = build_model(EFP16EfficiencyJob("efp16", length, batch_size))
    if not isinstance(model, HeadroomPACClassifier):
        message = "EFP16 builder returned an incompatible model"
        raise TypeError(message)
    model.load_state_dict(state_dict, strict=True)
    return model.cuda().eval()


def _prime_runtime(runtime: nn.Module, inputs: Tensor, warmups: int) -> Tensor:
    output: Tensor | None = None
    with torch.inference_mode():
        for _ in range(warmups + 1):
            output = runtime(inputs)
    torch.cuda.synchronize()
    if output is None:
        message = "runtime priming did not produce logits"
        raise RuntimeError(message)
    return output.detach().clone()


def _validate_screen_inputs(
    lengths: tuple[int, ...],
    batches: tuple[int, ...],
    config: FusedReadoutScreenConfig,
) -> None:
    if not lengths or min(lengths) < 2 or not batches or min(batches) < 1:
        message = "screen lengths and batches must be non-empty and valid"
        raise ValueError(message)
    if (
        min(config.groups, config.iterations_per_group) < 1
        or config.warmups < 0
        or config.gpu_clock_precondition_cycles < 0
    ):
        message = "screen iteration counts must be valid"
        raise ValueError(message)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(",") if item)


def _collect_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _release_cuda() -> None:
    _collect_cuda()
    torch.compiler.reset()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen the canonical EFP16 one-kernel exact-FP32 readout",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default=",".join(map(str, _LENGTHS)))
    parser.add_argument("--batches", default=",".join(map(str, _BATCHES)))
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--groups", type=int, default=9)
    parser.add_argument("--iterations-per-group", type=int, default=100)
    parser.add_argument("--gpu-clock-precondition-cycles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    result = screen_efp16_fused_readout(
        lengths=_parse_int_tuple(arguments.lengths),
        batches=_parse_int_tuple(arguments.batches),
        config=FusedReadoutScreenConfig(
            warmups=arguments.warmups,
            groups=arguments.groups,
            iterations_per_group=arguments.iterations_per_group,
            gpu_clock_precondition_cycles=arguments.gpu_clock_precondition_cycles,
            seed=arguments.seed,
        ),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(  # noqa: T201
        json.dumps(
            {
                "cells": len(cast("list[object]", result["cells"])),
                "selected": result["selected"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
