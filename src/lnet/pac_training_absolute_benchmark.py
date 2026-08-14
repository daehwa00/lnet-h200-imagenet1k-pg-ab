# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from .pac_training_cuda_graph_benchmark import (
    CURRENT_BACKENDS,
    _build_graph_context,
    _release_cuda,
)
from .pac_training_exact_split_benchmark import DEFAULT_CONFIG, _as_float, _as_int, _graph_config
from .pac_training_speed_comparison import TrainingModelName, build_training_model
from .pac_training_ultimate_benchmark import (
    DEFAULT_ULTIMATE_CONFIG,
    UltimateBenchmarkConfig,
    _accuracy_passes,
    _build_context,
    _FullGraphTrainingContext,
    _measure_paired,
    _measure_separate_context_parity,
    _TrainingContext,
)

if TYPE_CHECKING:
    from collections.abc import Generator


_MINIMUM_PAIRED_SPEEDUP = 1.005
_EXACT_SPLIT_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset(
    {
        ("efp16", 128, 64),
        ("efp16", 512, 64),
        ("pa2wp", 128, 64),
    }
)
_CUDA_SWITCH_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset({("efp16", 128, 64)})
_CUDA_OUTER_GRAPH_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset(
    {("efp16", 128, 64)}
)
_CUDA_COMPUTE_OUTER_GRAPH_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset(
    {}
)
_CUDA_CAPTURABLE_OUTER_GRAPH_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset(
    {("efp16", 512, 64)}
)
_FUSED_OPTIMIZER_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset(
    {("efp16", 128, 1)}
)
_POST_OPTIMIZER_GRAPH_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset(
    {("efp16", 512, 64)}
)
_FUSED_RMSNORM_MEAN_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset()
_FUSED_RMSNORM_MEAN_BACKWARD_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset(
    {("pa2wp", 128, 64)}
)
_MODE_STATIC_POLE_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset(
    {
        (model, length, 1)
        for model in cast("tuple[TrainingModelName, ...]", ("efp16", "pa2wp"))
        for length in (128, 512, 2048)
    }
)
_IDENTITY_ELISION_CELLS: frozenset[tuple[TrainingModelName, int, int]] = frozenset(
    {
        (model, length, batch_size)
        for model in cast("tuple[TrainingModelName, ...]", ("efp16", "pa2wp"))
        for length in (128, 512, 2048)
        for batch_size in (1, 64)
        if (model, length, batch_size) != ("pa2wp", 2048, 64)
    }
)
_EFP_STEM_STRATEGY_DISPATCH: dict[tuple[TrainingModelName, int, int], str] = {
    ("efp16", 128, 64): "serial",
}
_RECURRENCE_MODE_DISPATCH: dict[tuple[TrainingModelName, int, int], int] = {
    ("efp16", 128, 1): 16,
    ("efp16", 128, 64): 8,
    ("efp16", 512, 1): 16,
    ("efp16", 512, 64): 1,
    ("efp16", 2048, 1): 8,
    ("efp16", 2048, 64): 2,
    ("pa2wp", 128, 1): 1,
    ("pa2wp", 128, 64): 8,
    ("pa2wp", 512, 1): 1,
    ("pa2wp", 512, 64): 8,
    ("pa2wp", 2048, 1): 1,
    ("pa2wp", 2048, 64): 2,
}
_SPLIT_BACKWARD_DISPATCH: dict[tuple[TrainingModelName, int, int], tuple[int, int]] = {
    ("efp16", 2048, 1): (16, 16),
}
_SELECTIVE_SPLIT_BACKWARD_DISPATCH: dict[
    tuple[TrainingModelName, int, int], tuple[tuple[int, ...], int, int]
] = {
    ("efp16", 2048, 64): ((1,), 4, 1),
}


def benchmark_training_absolute(
    frozen_baseline: dict[str, object],
    *,
    config: UltimateBenchmarkConfig = DEFAULT_ULTIMATE_CONFIG,
) -> dict[str, object]:
    """Screen the complete exact-FP32 ceiling bundle against the frozen result."""
    if not torch.cuda.is_available():
        message = "absolute training benchmark requires CUDA"
        raise RuntimeError(message)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen_rows = cast("list[dict[str, object]]", frozen_baseline["rows"])
    frozen_index: dict[tuple[TrainingModelName, int, int], dict[str, object]] = {
        (
            cast("TrainingModelName", str(row["model"])),
            _as_int(row["length"]),
            _as_int(row["batch_size"]),
        ): row
        for row in frozen_rows
    }
    screens: dict[tuple[TrainingModelName, int, int], dict[str, object]] = {}
    for model_name in cast("tuple[TrainingModelName, ...]", ("efp16", "pa2wp")):
        for length in (128, 512, 2048):
            for batch_size in (1, 64):
                cell = (model_name, length, batch_size)
                screens[cell] = _benchmark_cell(cell, frozen_index[cell], config=config)

    rows: list[dict[str, object]] = []
    for frozen_row in frozen_rows:
        cell = (
            cast("TrainingModelName", str(frozen_row["model"])),
            _as_int(frozen_row["length"]),
            _as_int(frozen_row["batch_size"]),
        )
        screen = screens[cell]
        row = copy.deepcopy(frozen_row)
        row["absolute_screen"] = copy.deepcopy(screen)
        if bool(screen["selected"]):
            frozen_ms = _as_float(frozen_row["selected_wall_ms"])
            speedup = _as_float(screen["paired_speedup"])
            selected_ms = frozen_ms / speedup
            row.update(
                {
                    "frozen_absolute_baseline_wall_ms": frozen_ms,
                    "selected_runtime": screen["runtime_name"],
                    "selected_wall_ms": selected_ms,
                    "selected_sequences_per_second": cell[2] * 1000.0 / selected_ms,
                    "selected_accuracy": copy.deepcopy(screen["accuracy"]),
                    "selected_peak_memory_mb": screen["peak_memory_mb"],
                    "speedup_vs_campaign_eager": _as_float(
                        cast("dict[str, object]", row["campaign_eager"])["wall_ms"]
                    )
                    / selected_ms,
                    "speedup_vs_current_optimized": _as_float(
                        cast("dict[str, object]", row["current_optimized"])["wall_ms"]
                    )
                    / selected_ms,
                }
            )
        else:
            row["frozen_absolute_baseline_wall_ms"] = _as_float(row["selected_wall_ms"])
        rows.append(row)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = copy.deepcopy(frozen_baseline)
    payload.update(
        {
            "schema": "pac_training_absolute_benchmark.v1",
            "environment": {
                "host": platform.node(),
                "device": properties.name,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "config": {
                "warmups": config.warmups,
                "groups": config.groups,
                "iterations_per_group": config.iterations_per_group,
                "parity_steps": config.parity_steps,
                "seed": config.seed,
                "gpu_clock_ramp_cycles": config.gpu_clock_ramp_cycles,
                "gpu_clock_precondition_cycles": config.gpu_clock_precondition_cycles,
            },
            "rows": rows,
            "absolute_candidates": {_cell_label(cell): value for cell, value in screens.items()},
            "summary": _summarize(rows, frozen_index),
        }
    )
    protocol = cast("dict[str, object]", payload["protocol"])
    protocol.update(
        {
            "absolute_ceiling_timing": (
                "same-run alternating paired CUDA-event and synchronized wall median; "
                "compile, extension build, autotune, and CUDA Graph capture excluded"
            ),
            "absolute_ceiling_accuracy": (
                "75 consecutive exact-FP32 updates; loss, final gradient, and final "
                "parameter max absolute errors <=2e-5; PA2WP phase sequence exact"
            ),
            "matrix_exp_dispatch": (
                "shape-dispatched CUDA 13 ten-way conditional SWITCH for EFP16 N128/B64; "
                "device-side native-equivalent Taylor branch selection with host dispatch "
                "retained where SWITCH did not improve the full step"
            ),
            "tf32": False,
            "autocast": False,
        }
    )
    return payload


def _benchmark_cell(
    cell: tuple[TrainingModelName, int, int],
    frozen_row: dict[str, object],
    *,
    config: UltimateBenchmarkConfig,
) -> dict[str, object]:
    model_name, length, batch_size = cell
    backend = CURRENT_BACKENDS[cell]
    torch.manual_seed(config.seed)
    base_model, _ = build_training_model(model_name, length, batch_size)
    state_dict = copy.deepcopy(base_model.state_dict())
    del base_model
    generator = torch.Generator().manual_seed(config.seed + length + batch_size)
    cpu_inputs = torch.randn(batch_size, length, 1, generator=generator)
    cpu_labels = torch.randint(0, 5, (batch_size,), generator=generator)

    def build(
        *, candidate: bool, deterministic_accuracy_reference: bool = False
    ) -> _TrainingContext | _FullGraphTrainingContext:
        return _build_absolute_context(
            cell,
            frozen_row,
            state_dict,
            cpu_inputs,
            cpu_labels,
            candidate=candidate,
            deterministic_accuracy_reference=deterministic_accuracy_reference,
        )

    accuracy_reference = build(candidate=False, deterministic_accuracy_reference=True)
    accuracy_candidate = build(candidate=True)
    accuracy = _measure_separate_context_parity(
        accuracy_reference,
        accuracy_candidate,
        model_name=model_name,
        config=config,
        seed=config.seed + 200_000 + length + batch_size,
        reference_runtime="deterministic_exact_fp32_reference_same_run",
    )
    del accuracy_reference, accuracy_candidate
    _release_cuda()
    reference = build(candidate=False)
    candidate_context = build(candidate=True)
    timing = _measure_paired(
        {"frozen": reference, "absolute": candidate_context},
        config=config,
        seed=config.seed + 210_000 + length + batch_size,
    )
    paired_speedup = _as_float(timing["frozen"]["wall_ms"]) / _as_float(
        timing["absolute"]["wall_ms"]
    )
    del reference, candidate_context
    _release_cuda()
    memory_context = build(candidate=True)
    memory_context.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_context.step()
    torch.cuda.synchronize()
    peak_memory_mb = torch.cuda.max_memory_allocated() / 2**20
    del memory_context
    _release_cuda()
    accuracy_passed = _accuracy_passes(accuracy)
    performance_passed = paired_speedup > _MINIMUM_PAIRED_SPEEDUP
    return {
        "runtime_name": (
            "absolute_exact_split_cuda_outer_graph"
            if cell in _CUDA_OUTER_GRAPH_CELLS
            else (
                "efp16_post_ceiling_capturable_outer_graph"
                if cell in _CUDA_CAPTURABLE_OUTER_GRAPH_CELLS
                else (
                    "efp16_post_ceiling_split_backward"
                    if cell in _SPLIT_BACKWARD_DISPATCH
                    else (
                        "efp16_post_ceiling_selective_split_backward"
                        if cell in _SELECTIVE_SPLIT_BACKWARD_DISPATCH
                        else (
                            "absolute_exact_split_cuda_switch"
                            if cell in _CUDA_SWITCH_CELLS
                            else (
                                "absolute_exact_split_host"
                                if cell in _EXACT_SPLIT_CELLS
                                else "absolute_full_graph"
                            )
                        )
                    )
                )
            )
        ),
        "model": model_name,
        "length": length,
        "batch_size": batch_size,
        "backend": backend,
        "accuracy": accuracy,
        "accuracy_passed": accuracy_passed,
        "performance_passed": performance_passed,
        "selected": accuracy_passed and performance_passed,
        "paired_speedup": paired_speedup,
        "paired_frozen": timing["frozen"],
        "paired_absolute": timing["absolute"],
        "raw_absolute_wall_ms": timing["absolute"]["wall_ms"],
        "peak_memory_mb": peak_memory_mb,
        "feature_dispatch": {
            "recurrence_block_modes": _RECURRENCE_MODE_DISPATCH[cell],
            "split_backward_block_modes": _SPLIT_BACKWARD_DISPATCH.get(cell),
            "selective_split_backward": _SELECTIVE_SPLIT_BACKWARD_DISPATCH.get(cell),
            "canonical_identity_elision": cell in _IDENTITY_ELISION_CELLS,
            "mode_static_pole_training": cell in _MODE_STATIC_POLE_CELLS,
            "packed_recurrence_moments_training": "auto",
            "two_pass_reverse_recurrence_moments_training": "auto",
            "efp16_stem_parameter_gradient_strategy": _EFP_STEM_STRATEGY_DISPATCH.get(cell, "auto"),
            "pa2wp_phase_schedule_capacity": 64,
            "fused_optimizer_tail": cell in _FUSED_OPTIMIZER_CELLS,
            "post_optimizer_cuda_graph": cell in _POST_OPTIMIZER_GRAPH_CELLS,
            "cuda_outer_graph": cell
            in _CUDA_OUTER_GRAPH_CELLS | _CUDA_CAPTURABLE_OUTER_GRAPH_CELLS,
            "cuda_compute_outer_graph": cell in _CUDA_COMPUTE_OUTER_GRAPH_CELLS,
            "capturable_optimizer": cell in _CUDA_CAPTURABLE_OUTER_GRAPH_CELLS,
            "fused_rmsnorm_mean_training": cell in _FUSED_RMSNORM_MEAN_CELLS,
            "fused_rmsnorm_mean_backward_training": (cell in _FUSED_RMSNORM_MEAN_BACKWARD_CELLS),
            "matrix_exp_dispatch": (
                "cuda_switch"
                if cell in _CUDA_SWITCH_CELLS
                else ("host" if cell in _EXACT_SPLIT_CELLS else "captured")
            ),
        },
    }


def _build_absolute_context(
    cell: tuple[TrainingModelName, int, int],
    frozen_row: dict[str, object],
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    *,
    candidate: bool,
    deterministic_accuracy_reference: bool = False,
    post_ceiling: bool = True,
) -> _TrainingContext | _FullGraphTrainingContext:
    model_name, length, batch_size = cell
    backend = CURRENT_BACKENDS[cell]
    block_modes = _RECURRENCE_MODE_DISPATCH[cell] if candidate else 1
    split_modes = _split_backward_modes(cell) if candidate and post_ceiling else None
    with _block_modes(block_modes), _split_backward(split_modes):
        if cell in _EXACT_SPLIT_CELLS:
            return _build_context(
                model_name,
                length,
                batch_size,
                backend,
                state_dict,
                cpu_inputs,
                cpu_labels,
                ultimate=True,
                canonical_identity_elision=candidate and cell in _IDENTITY_ELISION_CELLS,
                mode_static_pole_training=candidate and cell in _MODE_STATIC_POLE_CELLS,
                packed_recurrence_moments_training=None if candidate else False,
                two_pass_reverse_recurrence_moments_training=None if candidate else False,
                efp16_stem_parameter_gradient_strategy=(
                    _EFP_STEM_STRATEGY_DISPATCH.get(cell, "auto")
                    if candidate
                    else _reference_efp_stem_strategy(
                        cell,
                        deterministic_accuracy_reference=deterministic_accuracy_reference,
                    )
                ),
                matrix_exp_dispatch=(
                    "cuda_switch"
                    if candidate
                    and cell
                    in _CUDA_SWITCH_CELLS
                    | _CUDA_COMPUTE_OUTER_GRAPH_CELLS
                    | (
                        _CUDA_CAPTURABLE_OUTER_GRAPH_CELLS
                        if post_ceiling
                        else frozenset()
                    )
                    else "host"
                ),
                pa2wp_phase_schedule_capacity=64 if candidate else None,
                fused_optimizer_tail=candidate and cell in _FUSED_OPTIMIZER_CELLS,
                capture_efp_post_optimizer_step=(
                    True if candidate and cell in _POST_OPTIMIZER_GRAPH_CELLS else None
                ),
                fused_rmsnorm_mean_training=(candidate and cell in _FUSED_RMSNORM_MEAN_CELLS),
                fused_rmsnorm_mean_backward_training=(
                    candidate and cell in _FUSED_RMSNORM_MEAN_BACKWARD_CELLS
                ),
                efp16_outer_graph=(
                    candidate
                    and cell
                    in _CUDA_OUTER_GRAPH_CELLS
                    | (
                        _CUDA_CAPTURABLE_OUTER_GRAPH_CELLS
                        if post_ceiling
                        else frozenset()
                    )
                ),
                efp16_compute_outer_graph=(
                    candidate and post_ceiling and cell in _CUDA_COMPUTE_OUTER_GRAPH_CELLS
                ),
                efp16_capturable_optimizer=(
                    True
                    if candidate and post_ceiling and cell in _CUDA_CAPTURABLE_OUTER_GRAPH_CELLS
                    else None
                ),
            )
        recurrence_override = "auto" if model_name == "pa2wp" and length == 2048 else None
        prior_fused_adjoint = not (model_name == "efp16" and length == 2048 and batch_size == 64)
        graph = _build_graph_context(
            model_name,
            length,
            batch_size,
            backend,
            graph_compute_dtype=str(frozen_row["graph_matrix_exp_compute_dtype"]),
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=_graph_config(DEFAULT_CONFIG),
            device="cuda",
            fused_recurrence_moments_backward_training=prior_fused_adjoint,
            fused_recurrence_moments_backward_blocks=(
                _SELECTIVE_SPLIT_BACKWARD_DISPATCH[cell][0]
                if candidate and post_ceiling and cell in _SELECTIVE_SPLIT_BACKWARD_DISPATCH
                else None
            ),
            recurrence_backend_override=recurrence_override,
            pa2wp_phase_schedule_capacity=64 if candidate else None,
            pa2wp_large_fused_stem_training=False,
            canonical_identity_elision=candidate and cell in _IDENTITY_ELISION_CELLS,
            mode_static_pole_training=candidate and cell in _MODE_STATIC_POLE_CELLS,
            packed_recurrence_moments_training=None if candidate else False,
            two_pass_reverse_recurrence_moments_training=None if candidate else False,
            efp16_stem_parameter_gradient_strategy=(
                _EFP_STEM_STRATEGY_DISPATCH.get(cell, "auto")
                if candidate
                else _reference_efp_stem_strategy(
                    cell,
                    deterministic_accuracy_reference=deterministic_accuracy_reference,
                )
            ),
            fused_optimizer_tail=candidate and cell in _FUSED_OPTIMIZER_CELLS,
            fused_rmsnorm_mean_training=(candidate and cell in _FUSED_RMSNORM_MEAN_CELLS),
            fused_rmsnorm_mean_backward_training=(
                candidate and cell in _FUSED_RMSNORM_MEAN_BACKWARD_CELLS
            ),
        )
        return _FullGraphTrainingContext(graph)


@contextmanager
def _block_modes(value: int) -> Generator[None]:
    previous = os.environ.get("LNET_PAC_BLOCK_MODES")
    os.environ["LNET_PAC_BLOCK_MODES"] = str(value)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LNET_PAC_BLOCK_MODES", None)
        else:
            os.environ["LNET_PAC_BLOCK_MODES"] = previous


def _split_backward_modes(
    cell: tuple[TrainingModelName, int, int],
) -> tuple[int, int] | None:
    full = _SPLIT_BACKWARD_DISPATCH.get(cell)
    if full is not None:
        return full
    selective = _SELECTIVE_SPLIT_BACKWARD_DISPATCH.get(cell)
    return (selective[1], selective[2]) if selective is not None else None


@contextmanager
def _split_backward(value: tuple[int, int] | None) -> Generator[None]:
    names = (
        "LNET_PAC_SPLIT_BACKWARD",
        "LNET_PAC_SPLIT_STATS_BLOCK_MODES",
        "LNET_PAC_SPLIT_ADJOINT_BLOCK_MODES",
    )
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    if value is not None:
        os.environ["LNET_PAC_SPLIT_BACKWARD"] = "1"
        os.environ["LNET_PAC_SPLIT_STATS_BLOCK_MODES"] = str(value[0])
        os.environ["LNET_PAC_SPLIT_ADJOINT_BLOCK_MODES"] = str(value[1])
    try:
        yield
    finally:
        for name, prior_value in previous.items():
            if prior_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior_value


def _previous_efp_stem_strategy(cell: tuple[TrainingModelName, int, int]) -> str:
    model_name, length, batch_size = cell
    if model_name != "efp16":
        return "auto"
    return "atomic" if length * batch_size >= 16_384 else "serial"


def _reference_efp_stem_strategy(
    cell: tuple[TrainingModelName, int, int], *, deterministic_accuracy_reference: bool
) -> str:
    if deterministic_accuracy_reference and cell in {
        ("efp16", 512, 64),
        ("efp16", 2048, 64),
    }:
        return "split_k"
    return _previous_efp_stem_strategy(cell)


def _cell_label(cell: tuple[TrainingModelName, int, int]) -> str:
    return f"{cell[0]}/N{cell[1]}/B{cell[2]}"


def _summarize(
    rows: list[dict[str, object]],
    frozen_index: dict[tuple[TrainingModelName, int, int], dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for model_name in cast("tuple[TrainingModelName, ...]", ("efp16", "pa2wp")):
        model_rows = [row for row in rows if row["model"] == model_name]
        ratios = [
            _as_float(
                frozen_index[(model_name, _as_int(row["length"]), _as_int(row["batch_size"]))][
                    "selected_wall_ms"
                ]
            )
            / _as_float(row["selected_wall_ms"])
            for row in model_rows
        ]
        result[model_name] = {
            "shape_count": len(model_rows),
            "selected_count": sum(
                str(row["selected_runtime"]).startswith("absolute_") for row in model_rows
            ),
            "geometric_mean_speedup_vs_frozen_baseline": math.exp(
                sum(math.log(value) for value in ratios) / len(ratios)
            ),
            "maximum_selected_wall_ms": max(
                _as_float(row["selected_wall_ms"]) for row in model_rows
            ),
            "maximum_selected_peak_memory_mb": max(
                _as_float(row["selected_peak_memory_mb"]) for row in model_rows
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    baseline = cast("dict[str, object]", json.loads(arguments.baseline.read_text()))
    result = benchmark_training_absolute(baseline)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    gc.collect()


if __name__ == "__main__":
    main()
