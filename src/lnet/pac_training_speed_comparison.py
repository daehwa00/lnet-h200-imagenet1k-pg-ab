# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import platform
import statistics
import typing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Final, Literal, Protocol, cast

try:
    from typing import assert_never as _assert_never
except ImportError:
    from typing_extensions import assert_never as _assert_never  # noqa: UP035

typing.assert_never = _assert_never  # type: ignore[attr-defined]

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_headroom_efficient_models import build_efficient_headroom_classifier
from .pac_metrics import count_parameters
from .pac_types import PACExperimentConfig

TrainingModelName = Literal["efp16", "pa2wp"]
TrainingPhase = Literal["forward", "forward_backward", "full"]
TrainingBackend = Literal[
    "campaign_auto_default_adamw",
    "block_scan_default_adamw",
    "campaign_auto_fused_adamw",
    "block_scan_fused_adamw",
]

MODELS: Final[tuple[TrainingModelName, ...]] = ("efp16", "pa2wp")
DISPLAY_NAMES: Final[dict[TrainingModelName, str]] = {
    "efp16": "EFP16",
    "pa2wp": "PA2WP",
}
EXPECTED_PARAMETER_COUNTS: Final[dict[TrainingModelName, int]] = {
    "efp16": 5_989,
    "pa2wp": 11_239,
}
LENGTHS: Final = (128, 512, 2048)
BATCHES: Final = (1, 64)
EAGER_BACKEND: Final[TrainingBackend] = "campaign_auto_default_adamw"
OPTIMIZATION_CANDIDATES: Final[tuple[TrainingBackend, ...]] = (
    "block_scan_default_adamw",
    "campaign_auto_fused_adamw",
    "block_scan_fused_adamw",
)
EAGER_RUNTIME: Final = "campaign_eager_fp32_training"
OPTIMIZED_RUNTIME: Final = "optimized_fp32_training"
MAXIMUM_PARITY_ERROR: Final = 2.0e-5


class _PA2WPTrainingStemControl(Protocol):
    use_fused_pa2wp_stem_training: bool


class _EFP16TrainingStemControl(Protocol):
    use_fused_efp16_stem_training: bool


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    warmups: int = 5
    groups: int = 7
    iterations_per_group: int = 10
    screening_warmups: int = 2
    screening_groups: int = 3
    screening_iterations_per_group: int = 5
    parity_steps: int = 1
    seed: int = 7
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    gpu_clock_ramp_cycles: int = 2_000_000_000
    gpu_clock_precondition_cycles: int = 20_000_000
    backend_prime_steps: int = 1


DEFAULT_BENCHMARK_CONFIG: Final = BenchmarkConfig()


def benchmark(
    *,
    models: tuple[TrainingModelName, ...] = MODELS,
    lengths: tuple[int, ...] = LENGTHS,
    batches: tuple[int, ...] = BATCHES,
    config: BenchmarkConfig = DEFAULT_BENCHMARK_CONFIG,
    device: str = "cuda",
) -> dict[str, object]:
    _validate_benchmark_arguments(models, lengths, batches, config, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    architectures: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for model_name in models:
        for length in lengths:
            for batch_size in batches:
                cell_rows, architecture = _benchmark_cell(
                    model_name,
                    length,
                    batch_size,
                    config=config,
                    device=device,
                )
                previous = architectures.setdefault(model_name, architecture)
                if _architecture_identity(previous) != _architecture_identity(architecture):
                    message = f"{model_name} training architecture changed across shapes"
                    raise RuntimeError(message)
                rows.extend(cell_rows)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema": "pac_training_speed_comparison.v2",
        "environment": {
            "host": platform.node(),
            "device": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allow_tf32": False,
        },
        "protocol": _protocol(config),
        "config": asdict(config),
        "models": list(models),
        "lengths": list(lengths),
        "batches": list(batches),
        "architectures": architectures,
        "autograd_support": autograd_support_matrix(),
        "training_optimization_evidence": training_optimization_evidence(),
        "rows": rows,
        "summary": summarize(rows),
    }


def _protocol(config: BenchmarkConfig) -> dict[str, object]:
    return {
        "scope": "actual campaign FP32 training and training-safe optimized candidates",
        "forward_column": "training-mode model forward plus cross-entropy loss",
        "forward_backward_column": (
            "zero_grad(set_to_none=True), training-mode forward, cross-entropy, backward"
        ),
        "full_train_step_column": (
            "forward+backward, clip_grad_norm_, optimizer.step, model.post_optimizer_step"
        ),
        "forward_only": False,
        "dtype": "float32",
        "autocast": False,
        "tf32": False,
        "eager_backend": (
            "campaign recurrence_backend='auto' (autograd Triton on CUDA) plus default AdamW"
        ),
        "optimized_search": list(OPTIMIZATION_CANDIDATES),
        "optimized_selection": (
            "all candidates use the backward-safe fused stems and shape-dispatched online-"
            "moments backward; select minimum screening full-step wall latency across scan and "
            "optimizer choices after loss, gradient, and parameter-update parity"
        ),
        "shared_training_optimizations": (
            "custom-autograd EFP16/PA2WP stems, streaming Triton online-moments backward for "
            "small batch-time products, selective Inductor online-moments backward otherwise"
        ),
        "backend_priming": (
            f"{config.backend_prime_steps} untimed full steps for eager and every candidate "
            "before screening, to initialize CUDA libraries and kernels without order bias"
        ),
        "compile_mode": "none",
        "selective_kernel_compile_mode": "online_moments_backward/fullgraph/static/no-cudagraphs",
        "compile_cost_included": False,
        "cuda_graph_capture": False,
        "wall_timing": (
            "perf_counter around each group with CUDA synchronization immediately before start "
            "and after end; includes Python launch and host optimizer overhead"
        ),
        "gpu_timing": "CUDA events around the same group",
        "warmups": config.warmups,
        "groups": config.groups,
        "iterations_per_group": config.iterations_per_group,
        "aggregation": "median of per-group means; inclusive IQR",
        "gpu_clock_precondition": (
            f"{config.gpu_clock_ramp_cycles} CUDA busy cycles once before each phase and "
            f"{config.gpu_clock_precondition_cycles} cycles before every timed group; excluded "
            "from timing"
        ),
        "peak_memory": "torch.cuda.max_memory_allocated during each measured phase",
        "pa2wp_training_phase_policy": (
            "actual stochastic single-phase augmentation; random original/shifted origin"
        ),
        "parity_reference": "actual campaign eager FP32 training trajectory",
        "parity_candidate": "selected training-safe optimized FP32 training trajectory",
        "parity_steps": config.parity_steps,
        "maximum_loss_gradient_update_absolute_error": MAXIMUM_PARITY_ERROR,
    }


def training_optimization_evidence() -> dict[str, object]:
    return {
        "selected": [
            "shape-dispatched streaming Triton/Inductor online-moments backward",
            "custom-autograd PA2WP selected-phase Haar+stem fusion",
            "custom-autograd EFP16 edge-analysis+stem fusion",
            "per-cell associative-scan and fused-AdamW screening",
            "fixed-work capture-safe orthogonal matrix exponential",
            "full EFP16 FP32 training-step CUDA Graph",
            "stochastic-policy-preserving PA2WP original/shifted training CUDA Graphs",
        ],
        "rejected_candidates": [
            {
                "candidate": "full-model torch.compile",
                "reason": (
                    "the stochastic PA2WP Python branch graph-breaks and whole-model compilation "
                    "was less stable than compiling only the launch-bound moments backward"
                ),
            },
            {
                "candidate": "manual streaming moments backward at every shape",
                "reason": (
                    "the sequential per-mode kernel loses to Inductor vector fusion at large "
                    "batch-time products, so it is retained only below the measured threshold"
                ),
            },
            {
                "candidate": "static pole/gamma folding or dual-phase removal",
                "reason": (
                    "these inference transformations would remove required parameter gradients "
                    "or change the stochastic PA2WP campaign semantics"
                ),
            },
        ],
    }


def autograd_support_matrix() -> dict[str, dict[str, object]]:
    return {
        "campaign_triton_recurrence": {
            "used_by_eager_training": True,
            "eligible_for_optimized_training": True,
            "backward_supported": True,
            "evidence": (
                "pac_triton_recurrence_op.py registers torch.library autograd for "
                "lnet::pac_real2d_recurrence and supplies a Triton backward kernel"
            ),
        },
        "block_associative_scan": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": True,
            "backward_supported": True,
            "evidence": (
                "_TritonScanBlocks is a torch.autograd.Function with an explicit backward; "
                "the benchmark enables it without inference preparation"
            ),
        },
        "fused_adamw": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": True,
            "backward_supported": True,
            "evidence": "PyTorch fused AdamW updates the same trainable parameter set on CUDA",
        },
        "static_pole_gamma_folding": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": False,
            "backward_supported": False,
            "evidence": (
                "prepare_for_inference_ runs under no_grad and forward requires "
                "not torch.is_grad_enabled() before consuming folded drive/pole buffers"
            ),
        },
        "fused_recurrence_moments": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": False,
            "backward_supported": False,
            "evidence": (
                "forward gates it on not torch.is_grad_enabled(); the fused moment Triton ops "
                "do not register an autograd formula"
            ),
        },
        "fused_online_moments_backward": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": True,
            "backward_supported": True,
            "evidence": (
                "pac_triton_online_moments.py registers a training-only custom op whose "
                "streaming Triton backward fuses both lag correlations and normalization; "
                "long high-batch cells dispatch to the faster exact PyTorch backward"
            ),
        },
        "fused_efp16_stem": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": False,
            "backward_supported": False,
            "evidence": (
                "EdgeFramePAC gates it on not torch.is_grad_enabled(); the stem Triton op has "
                "no registered autograd formula"
            ),
        },
        "fused_efp16_training_stem": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": True,
            "backward_supported": True,
            "evidence": (
                "pac_triton_edge_frame_stem_training.py fuses degree-normalized edge analysis, "
                "2-to-D projection, dilated depthwise K5 convolution, and SiLU with explicit "
                "FP32 gradients for raw input and all stem parameters"
            ),
        },
        "fused_pa2wp_stem_and_phase_batching": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": False,
            "backward_supported": False,
            "evidence": (
                "dual-phase batching is guarded by not self.training and the fused stem has no "
                "autograd formula; campaign training samples one phase"
            ),
        },
        "fused_pa2wp_training_stem": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": True,
            "backward_supported": True,
            "evidence": (
                "pac_triton_pa2wp_training_stem.py fuses the already-selected stochastic "
                "phase's Haar analysis, low/detail packing, causal K9/S2 convolution, and "
                "SiLU with custom FP32 gradients for input, weight, and bias"
            ),
        },
        "manual_cuda_graph": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": False,
            "backward_supported": False,
            "evidence": (
                "BorrowedInputCudaGraphInference.forward is torch.no_grad and captures forward "
                "only; no backward or optimizer graph is implemented"
            ),
        },
        "capture_safe_matrix_exp": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": True,
            "backward_supported": True,
            "evidence": (
                "pac_capture_safe_orthogonal.py uses a fixed degree-12 Taylor polynomial at "
                "one-eighth scale followed by three squarings; CUDA Graph capture and the "
                "75-step FP32 drift contract are tested"
            ),
        },
        "full_step_training_cuda_graph": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": True,
            "backward_supported": True,
            "evidence": (
                "pac_efp16_training_cuda_graph.py captures loss, backward, foreach clipping, "
                "capturable fused AdamW, and the post-optimizer projection"
            ),
        },
        "pa2wp_dual_training_cuda_graph": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": True,
            "backward_supported": True,
            "evidence": (
                "pac_pa2wp_training_cuda_graph.py keeps the Bernoulli draw outside two "
                "deterministic phase graphs sharing one parameter and optimizer state"
            ),
        },
        "borrowed_output": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": False,
            "backward_supported": False,
            "evidence": (
                "the static inference output is overwritten on replay and cannot retain a "
                "training autograd graph"
            ),
        },
        "torch_compile_training": {
            "used_by_eager_training": False,
            "eligible_for_optimized_training": False,
            "backward_supported": True,
            "evidence": (
                "the generic trainer supports compile modes, but the actual campaigns use none; "
                "PA2WP's Python random-phase branch prevents one static fullgraph contract"
            ),
        },
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for model_name in MODELS:
        eager = _runtime_rows(rows, model_name, EAGER_RUNTIME)
        optimized = _runtime_rows(rows, model_name, OPTIMIZED_RUNTIME)
        if not eager or not optimized:
            continue
        eager_full = [_as_float(row["full_train_step_wall_ms"]) for row in eager]
        optimized_full = [_as_float(row["full_train_step_wall_ms"]) for row in optimized]
        result[model_name] = {
            "display_name": DISPLAY_NAMES[model_name],
            "eager_geometric_mean_forward_wall_ms": _geometric_mean(
                [_as_float(row["forward_wall_ms"]) for row in eager]
            ),
            "optimized_geometric_mean_forward_wall_ms": _geometric_mean(
                [_as_float(row["forward_wall_ms"]) for row in optimized]
            ),
            "eager_geometric_mean_forward_backward_wall_ms": _geometric_mean(
                [_as_float(row["forward_backward_wall_ms"]) for row in eager]
            ),
            "optimized_geometric_mean_forward_backward_wall_ms": _geometric_mean(
                [_as_float(row["forward_backward_wall_ms"]) for row in optimized]
            ),
            "eager_geometric_mean_full_train_step_wall_ms": _geometric_mean(eager_full),
            "optimized_geometric_mean_full_train_step_wall_ms": _geometric_mean(optimized_full),
            "geometric_mean_full_step_speedup": _geometric_mean(
                [
                    eager_latency / optimized_latency
                    for eager_latency, optimized_latency in zip(
                        eager_full, optimized_full, strict=True
                    )
                ]
            ),
            "eager_maximum_peak_memory_mb": max(_as_float(row["peak_memory_mb"]) for row in eager),
            "optimized_maximum_peak_memory_mb": max(
                _as_float(row["peak_memory_mb"]) for row in optimized
            ),
            "maximum_loss_abs_error": max(_as_float(row["loss_abs_error"]) for row in optimized),
            "maximum_gradient_abs_error": max(
                _as_float(row["gradient_max_abs_error"]) for row in optimized
            ),
            "maximum_parameter_update_abs_error": max(
                _as_float(row["parameter_update_max_abs_error"]) for row in optimized
            ),
        }
    return result


def evaluate_result(payload: dict[str, object]) -> dict[str, object]:  # noqa: C901, PLR0912
    failures: list[str] = []
    environment = cast("dict[str, object]", payload.get("environment", {}))
    if "4090" not in str(environment.get("device", "")):
        failures.append("benchmark device is not an RTX 4090")
    protocol = cast("dict[str, object]", payload.get("protocol", {}))
    for key, expected in (
        ("forward_only", False),
        ("dtype", "float32"),
        ("autocast", False),
        ("tf32", False),
        ("compile_mode", "none"),
        ("compile_cost_included", False),
        ("cuda_graph_capture", False),
    ):
        if protocol.get(key) != expected:
            failures.append(f"invalid protocol {key}: {protocol.get(key)!r}")

    architectures = cast("dict[str, dict[str, object]]", payload.get("architectures", {}))
    rows = cast("list[dict[str, object]]", payload.get("rows", []))
    indexed = {
        (
            str(row.get("model")),
            _as_int(row.get("length", 0)),
            _as_int(row.get("batch_size", 0)),
            str(row.get("runtime")),
        ): row
        for row in rows
    }
    for model_name in MODELS:
        architecture = architectures.get(model_name)
        if architecture is None:
            failures.append(f"missing architecture provenance for {model_name}")
        elif not architecture.get("actual_campaign_training_path", False):
            failures.append(f"{model_name} is not marked as the actual campaign path")
        elif architecture.get("trainable_parameters") != EXPECTED_PARAMETER_COUNTS[model_name]:
            failures.append(f"unexpected trainable parameter count for {model_name}")
        for length in LENGTHS:
            for batch_size in BATCHES:
                cell = f"{model_name}/N{length}/B{batch_size}"
                eager = indexed.get((model_name, length, batch_size, EAGER_RUNTIME))
                optimized = indexed.get((model_name, length, batch_size, OPTIMIZED_RUNTIME))
                for runtime, row in ((EAGER_RUNTIME, eager), (OPTIMIZED_RUNTIME, optimized)):
                    if row is None or row.get("status") != "measured":
                        failures.append(f"missing measured {runtime} cell {cell}")
                        continue
                    for metric in (
                        "forward_wall_ms",
                        "forward_backward_wall_ms",
                        "full_train_step_wall_ms",
                        "full_train_step_gpu_ms",
                        "sequences_per_second",
                        "tokens_per_second",
                        "peak_memory_mb",
                    ):
                        value = _as_float(row.get(metric, 0.0))
                        if not math.isfinite(value) or value <= 0.0:
                            failures.append(f"invalid {metric} for {cell}/{runtime}")
                if eager is None or optimized is None:
                    continue
                if optimized.get("selected_backend") not in OPTIMIZATION_CANDIDATES:
                    failures.append(f"invalid selected optimized backend for {cell}")
                for metric in (
                    "loss_abs_error",
                    "gradient_max_abs_error",
                    "parameter_update_max_abs_error",
                ):
                    value = _as_float(optimized.get(metric, math.inf))
                    if not math.isfinite(value) or value > MAXIMUM_PARITY_ERROR:
                        failures.append(f"{metric} exceeded for {cell}: {value:.6g}")
                if optimized.get("gradient_key_agreement") is not True:
                    failures.append(f"gradient key disagreement for {cell}")
                expected_speedup = _as_float(eager["full_train_step_wall_ms"]) / _as_float(
                    optimized["full_train_step_wall_ms"]
                )
                if not math.isclose(
                    _as_float(optimized["speedup_vs_eager"]),
                    expected_speedup,
                    rel_tol=1.0e-9,
                ):
                    failures.append(f"full-step speedup mismatch for {cell}")

    support = cast("dict[str, dict[str, object]]", payload.get("autograd_support", {}))
    expected_support = {
        "campaign_triton_recurrence": True,
        "block_associative_scan": True,
        "fused_adamw": True,
        "fused_online_moments_backward": True,
        "static_pole_gamma_folding": False,
        "fused_recurrence_moments": False,
        "fused_efp16_stem": False,
        "fused_efp16_training_stem": True,
        "fused_pa2wp_stem_and_phase_batching": False,
        "fused_pa2wp_training_stem": True,
        "manual_cuda_graph": False,
        "borrowed_output": False,
        "torch_compile_training": True,
    }
    for optimization, backward_supported in expected_support.items():
        support_row = support.get(optimization)
        if support_row is None:
            failures.append(f"missing autograd support entry: {optimization}")
        elif support_row.get("backward_supported") is not backward_supported:
            failures.append(f"wrong autograd support status: {optimization}")
    return {
        "schema": "pac_training_speed_comparison_evaluation.v2",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_models": len(MODELS),
        "checked_shapes_per_model": len(LENGTHS) * len(BATCHES),
    }


def evaluate_max_result(  # noqa: C901, PLR0912, PLR0915
    payload: dict[str, object], baseline: dict[str, object]
) -> dict[str, object]:
    """Evaluate the training-specific candidate against the frozen July 14 baseline."""
    base_evaluation = evaluate_result(payload)
    failures = list(cast("list[str]", base_evaluation["failures"]))
    protocol = cast("dict[str, object]", payload.get("protocol", {}))
    baseline_protocol = cast("dict[str, object]", baseline.get("protocol", {}))
    phase_policy = "actual stochastic single-phase augmentation; random original/shifted origin"
    if protocol.get("pa2wp_training_phase_policy") != phase_policy:
        failures.append("PA2WP stochastic single-phase training policy changed")
    if baseline_protocol.get("pa2wp_training_phase_policy") != phase_policy:
        failures.append("baseline PA2WP phase policy is not the frozen campaign policy")
    if not protocol.get("selective_kernel_compile_mode"):
        failures.append("selective training-kernel compile mode is not documented")

    evidence = cast("dict[str, object]", payload.get("training_optimization_evidence", {}))
    rejected = cast("list[dict[str, object]]", evidence.get("rejected_candidates", []))
    if not rejected or any(not row.get("candidate") or not row.get("reason") for row in rejected):
        failures.append("rejected training-specific candidates are not fully documented")

    architectures = cast("dict[str, dict[str, object]]", payload.get("architectures", {}))
    baseline_architectures = cast("dict[str, dict[str, object]]", baseline.get("architectures", {}))
    semantic_keys = (
        "family",
        "internal_spec",
        "effective_model_dim",
        "effective_modes",
        "trainable_parameters",
        "objective",
        "output_dim",
        "pa2wp_phase_policy",
    )
    for model_name in MODELS:
        candidate_architecture = architectures.get(model_name, {})
        baseline_architecture = baseline_architectures.get(model_name, {})
        failures.extend(
            f"model semantics changed for {model_name}: {key}"
            for key in semantic_keys
            if candidate_architecture.get(key) != baseline_architecture.get(key)
        )

    candidate_rows = cast("list[dict[str, object]]", payload.get("rows", []))
    baseline_rows = cast("list[dict[str, object]]", baseline.get("rows", []))
    candidate_index = {
        (
            str(row.get("model")),
            _as_int(row.get("length", 0)),
            _as_int(row.get("batch_size", 0)),
            str(row.get("runtime")),
        ): row
        for row in candidate_rows
    }
    baseline_index = {
        (
            str(row.get("model")),
            _as_int(row.get("length", 0)),
            _as_int(row.get("batch_size", 0)),
            str(row.get("runtime")),
        ): row
        for row in baseline_rows
    }
    speedups: dict[str, float] = {}
    maximum_regressions: dict[str, float] = {}
    absolute_speedups: dict[str, float] = {}
    for model_name in MODELS:
        model_speedups: list[float] = []
        model_regressions: list[float] = []
        model_absolute_speedups: list[float] = []
        for length in LENGTHS:
            for batch_size in BATCHES:
                optimized_key = (model_name, length, batch_size, OPTIMIZED_RUNTIME)
                eager_key = (model_name, length, batch_size, EAGER_RUNTIME)
                candidate = candidate_index.get(optimized_key)
                candidate_eager = candidate_index.get(eager_key)
                baseline_row = baseline_index.get(optimized_key)
                baseline_eager = baseline_index.get(eager_key)
                cell = f"{model_name}/N{length}/B{batch_size}"
                if (
                    candidate is None
                    or candidate_eager is None
                    or baseline_row is None
                    or baseline_eager is None
                ):
                    failures.append(f"missing candidate or baseline paired cell {cell}")
                    continue
                candidate_latency = _as_float(candidate.get("full_train_step_wall_ms", math.inf))
                candidate_eager_latency = _as_float(
                    candidate_eager.get("full_train_step_wall_ms", math.inf)
                )
                baseline_latency = _as_float(baseline_row.get("full_train_step_wall_ms", math.inf))
                baseline_eager_latency = _as_float(
                    baseline_eager.get("full_train_step_wall_ms", math.inf)
                )
                latencies = (
                    candidate_latency,
                    candidate_eager_latency,
                    baseline_latency,
                    baseline_eager_latency,
                )
                if any(not math.isfinite(value) or value <= 0.0 for value in latencies):
                    failures.append(f"non-finite candidate or baseline latency {cell}")
                    continue
                candidate_normalized = candidate_latency / candidate_eager_latency
                baseline_normalized = baseline_latency / baseline_eager_latency
                speedup = baseline_normalized / candidate_normalized
                regression = candidate_normalized / baseline_normalized
                model_speedups.append(speedup)
                model_regressions.append(regression)
                model_absolute_speedups.append(baseline_latency / candidate_latency)
                if regression > 1.05:
                    failures.append(
                        f"paired cell regressed >5% vs baseline {cell}: {regression:.6f}x"
                    )
        if len(model_speedups) == len(LENGTHS) * len(BATCHES):
            speedups[model_name] = _geometric_mean(model_speedups)
            maximum_regressions[model_name] = max(model_regressions)
            absolute_speedups[model_name] = _geometric_mean(model_absolute_speedups)
            if speedups[model_name] <= 1.0:
                failures.append(
                    f"paired optimized geometric-mean latency did not improve for {model_name}"
                )
    return {
        "schema": "pac_training_speed_comparison_max_evaluation.v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_models": len(MODELS),
        "checked_shapes_per_model": len(LENGTHS) * len(BATCHES),
        "paired_normalized_optimized_geometric_mean_speedup_vs_baseline": speedups,
        "maximum_cell_paired_normalized_latency_ratio_vs_baseline": maximum_regressions,
        "diagnostic_absolute_geometric_mean_speedup_vs_baseline": absolute_speedups,
        "comparison_note": (
            "gates compare optimized/eager paired ratios within each run because low-load RTX "
            "4090 cells switch between discrete latency regimes; absolute cross-run ratios are "
            "reported only as diagnostics"
        ),
        "baseline_schema": baseline.get("schema"),
    }


def merge_payloads(payloads: list[dict[str, object]]) -> dict[str, object]:
    if not payloads:
        message = "at least one training benchmark payload is required"
        raise ValueError(message)
    first = payloads[0]
    shared_keys = ("environment", "protocol", "config", "lengths", "batches")
    architectures: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    present_models: set[str] = set()
    for payload in payloads:
        for key in shared_keys:
            if payload.get(key) != first.get(key):
                message = f"cannot merge training payloads with different {key}"
                raise ValueError(message)
        payload_models = cast("list[str]", payload.get("models", []))
        duplicate = present_models.intersection(payload_models)
        if duplicate:
            message = f"duplicate training model payloads: {sorted(duplicate)}"
            raise ValueError(message)
        present_models.update(payload_models)
        architectures.update(cast("dict[str, dict[str, object]]", payload.get("architectures", {})))
        rows.extend(cast("list[dict[str, object]]", payload.get("rows", [])))
    ordered_models = [model_name for model_name in MODELS if model_name in present_models]
    return {
        "schema": "pac_training_speed_comparison.v2",
        **{key: first.get(key) for key in shared_keys},
        "models": ordered_models,
        "architectures": architectures,
        "autograd_support": autograd_support_matrix(),
        "rows": rows,
        "summary": summarize(rows),
    }


def replace_cells(
    base: dict[str, object], replacements: list[dict[str, object]]
) -> dict[str, object]:
    result = copy.deepcopy(base)
    base_rows = cast("list[dict[str, object]]", result.get("rows", []))
    indexed = {
        (
            str(row.get("model")),
            _as_int(row.get("length", 0)),
            _as_int(row.get("batch_size", 0)),
            str(row.get("runtime")),
        ): index
        for index, row in enumerate(base_rows)
    }
    replaced: set[tuple[str, int, int, str]] = set()
    for payload in replacements:
        for key in ("environment", "protocol", "config"):
            if payload.get(key) != base.get(key):
                message = f"cannot replace training cells with different {key}"
                raise ValueError(message)
        for row in cast("list[dict[str, object]]", payload.get("rows", [])):
            cell = (
                str(row.get("model")),
                _as_int(row.get("length", 0)),
                _as_int(row.get("batch_size", 0)),
                str(row.get("runtime")),
            )
            index = indexed.get(cell)
            if index is None:
                message = f"replacement training cell is absent from base: {cell}"
                raise ValueError(message)
            base_rows[index] = copy.deepcopy(row)
            replaced.add(cell)
    if not replaced:
        message = "at least one replacement training cell is required"
        raise ValueError(message)
    result["rows"] = base_rows
    result["summary"] = summarize(base_rows)
    result["replacement_cells"] = [list(cell) for cell in sorted(replaced)]
    return result


def build_training_model(
    model_name: TrainingModelName, length: int, batch_size: int
) -> tuple[nn.Module, dict[str, object]]:
    model_dim = 32 if model_name == "efp16" else 64
    internal_spec = "EFP16" if model_name == "efp16" else "PA2WP"
    config = PACExperimentConfig(
        sample_count=max(batch_size, 64),
        validation_count=16,
        test_count=16,
        sequence_length=length,
        raw_input_dim=1,
        output_dim=5,
        model_dim=model_dim,
        modes=16,
        epochs=1,
        batch_size=batch_size,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        device="cpu",
        compile_mode="none",
        precision="fp32",
        optimizer_mode="default",
    )
    model = build_efficient_headroom_classifier(
        internal_spec, config, 5, objective="classification"
    )
    return model, {
        "display_name": DISPLAY_NAMES[model_name],
        "family": model_name,
        "internal_spec": internal_spec,
        "requested_model_dim": model_dim,
        "effective_model_dim": int(model.model_dim),
        "requested_modes": 16,
        "effective_modes": int(model.modes),
        "trainable_parameters": count_parameters(model),
        "objective": "classification",
        "output_dim": 5,
        "actual_campaign_training_path": True,
        "compile_mode": "none",
        "precision": "fp32",
        "optimizer_mode": "default AdamW",
        "pa2wp_phase_policy": (
            "random original-or-shifted single phase per training call"
            if model_name == "pa2wp"
            else "not applicable"
        ),
        "state_dict_sha256": _state_dict_digest(model),
    }


def _benchmark_cell(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    *,
    config: BenchmarkConfig,
    device: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(config.seed)
    base_model, architecture = build_training_model(model_name, length, batch_size)
    state_dict = copy.deepcopy(base_model.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(config.seed + length + batch_size)
    cpu_inputs = torch.randn(batch_size, length, 1, generator=generator)
    cpu_labels = torch.randint(0, 5, (batch_size,), generator=generator)
    del base_model

    for backend in (EAGER_BACKEND, *OPTIMIZATION_CANDIDATES):
        _prime_training_backend(
            model_name,
            length,
            batch_size,
            backend,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=config,
            device=device,
        )

    screening: list[dict[str, object]] = []
    for backend in OPTIMIZATION_CANDIDATES:
        try:
            parity = _measure_training_parity(
                model_name,
                length,
                batch_size,
                backend,
                state_dict=state_dict,
                cpu_inputs=cpu_inputs,
                cpu_labels=cpu_labels,
                config=config,
                device=device,
            )
            quick_config = replace(
                config,
                warmups=config.screening_warmups,
                groups=config.screening_groups,
                iterations_per_group=config.screening_iterations_per_group,
            )
            quick = _measure_phase(
                model_name,
                length,
                batch_size,
                backend,
                phase="full",
                state_dict=state_dict,
                cpu_inputs=cpu_inputs,
                cpu_labels=cpu_labels,
                config=quick_config,
                device=device,
            )
            exact = _parity_is_exact(parity)
            screening.append(
                {
                    "backend": backend,
                    "status": "measured",
                    "exact": exact,
                    "screening_full_train_step_wall_ms": quick["wall_ms"],
                    **parity,
                }
            )
        except Exception as error:  # noqa: BLE001 - preserve candidate evidence
            screening.append(
                {
                    "backend": backend,
                    "status": "failed",
                    "exact": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _release_cuda()
    exact_candidates = [
        row for row in screening if row.get("status") == "measured" and row.get("exact") is True
    ]
    if not exact_candidates:
        message = (
            f"no exact training optimization candidate for {model_name}/N{length}/B{batch_size}"
        )
        raise RuntimeError(message)
    selected_screen = min(
        exact_candidates,
        key=lambda row: _as_float(row["screening_full_train_step_wall_ms"]),
    )
    selected_backend = cast("TrainingBackend", selected_screen["backend"])

    eager_measurement, optimized_measurement = _measure_paired_training_runtimes(
        model_name,
        length,
        batch_size,
        selected_backend,
        state_dict=state_dict,
        cpu_inputs=cpu_inputs,
        cpu_labels=cpu_labels,
        config=config,
        device=device,
    )
    parity = {
        key: value
        for key, value in selected_screen.items()
        if key
        in {
            "reference_loss",
            "candidate_loss",
            "loss_abs_error",
            "gradient_key_agreement",
            "gradient_tensor_count",
            "gradient_max_abs_error",
            "gradient_max_rel_error",
            "parameter_update_max_abs_error",
            "parameter_update_max_rel_error",
            "parameter_value_max_abs_error",
            "parameter_value_max_rel_error",
        }
    }
    common = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "length": length,
        "batch_size": batch_size,
        "status": "measured",
    }
    eager_row = {
        **common,
        "runtime": EAGER_RUNTIME,
        "selected_backend": EAGER_BACKEND,
        **eager_measurement,
        "speedup_vs_eager": 1.0,
        "forward_speedup_vs_eager": 1.0,
        "forward_backward_speedup_vs_eager": 1.0,
        "loss_abs_error": 0.0,
        "gradient_key_agreement": True,
        "gradient_tensor_count": parity["gradient_tensor_count"],
        "gradient_max_abs_error": 0.0,
        "gradient_max_rel_error": 0.0,
        "parameter_update_max_abs_error": 0.0,
        "parameter_update_max_rel_error": 0.0,
        "parameter_value_max_abs_error": 0.0,
        "parameter_value_max_rel_error": 0.0,
    }
    optimized_row = {
        **common,
        "runtime": OPTIMIZED_RUNTIME,
        "selected_backend": selected_backend,
        "candidate_screening": screening,
        **optimized_measurement,
        **parity,
        "speedup_vs_eager": _as_float(eager_measurement["full_train_step_wall_ms"])
        / _as_float(optimized_measurement["full_train_step_wall_ms"]),
        "forward_speedup_vs_eager": _as_float(eager_measurement["forward_wall_ms"])
        / _as_float(optimized_measurement["forward_wall_ms"]),
        "forward_backward_speedup_vs_eager": _as_float(
            eager_measurement["forward_backward_wall_ms"]
        )
        / _as_float(optimized_measurement["forward_backward_wall_ms"]),
    }
    return [eager_row, optimized_row], architecture


def _measure_paired_training_runtimes(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    optimized_backend: TrainingBackend,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> tuple[dict[str, object], dict[str, object]]:
    paired_phases = {
        phase: _measure_paired_phase(
            model_name,
            length,
            batch_size,
            optimized_backend,
            phase=phase,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=config,
            device=device,
        )
        for phase in ("forward", "forward_backward", "full")
    }
    eager_peak = _measure_full_peak_memory(
        model_name,
        length,
        batch_size,
        EAGER_BACKEND,
        state_dict=state_dict,
        cpu_inputs=cpu_inputs,
        cpu_labels=cpu_labels,
        config=config,
        device=device,
    )
    optimized_peak = _measure_full_peak_memory(
        model_name,
        length,
        batch_size,
        optimized_backend,
        state_dict=state_dict,
        cpu_inputs=cpu_inputs,
        cpu_labels=cpu_labels,
        config=config,
        device=device,
    )
    eager_phases = {
        phase: measurements[EAGER_BACKEND] for phase, measurements in paired_phases.items()
    }
    optimized_phases = {
        phase: measurements[optimized_backend] for phase, measurements in paired_phases.items()
    }
    return (
        _assemble_runtime_measurement(eager_phases, eager_peak, length, batch_size),
        _assemble_runtime_measurement(optimized_phases, optimized_peak, length, batch_size),
    )


def _assemble_runtime_measurement(
    phases: dict[str, dict[str, object]],
    peak_memory_mb: float,
    length: int,
    batch_size: int,
) -> dict[str, object]:
    forward = phases["forward"]
    forward_backward = phases["forward_backward"]
    full = phases["full"]
    full_wall = _as_float(full["wall_ms"])
    return {
        "forward_wall_ms": forward["wall_ms"],
        "forward_wall_iqr_ms": forward["wall_iqr_ms"],
        "forward_gpu_ms": forward["gpu_ms"],
        "forward_backward_wall_ms": forward_backward["wall_ms"],
        "forward_backward_wall_iqr_ms": forward_backward["wall_iqr_ms"],
        "forward_backward_gpu_ms": forward_backward["gpu_ms"],
        "full_train_step_wall_ms": full_wall,
        "full_train_step_wall_iqr_ms": full["wall_iqr_ms"],
        "full_train_step_wall_samples_ms": full["wall_samples_ms"],
        "full_train_step_gpu_ms": full["gpu_ms"],
        "full_train_step_gpu_iqr_ms": full["gpu_iqr_ms"],
        "full_train_step_gpu_samples_ms": full["gpu_samples_ms"],
        "peak_memory_mb": peak_memory_mb,
        "sequences_per_second": batch_size * 1000.0 / full_wall,
        "tokens_per_second": batch_size * length * 1000.0 / full_wall,
        "last_timed_loss": full["last_loss"],
        "compile_seconds": 0.0,
        "compile_cost_included": False,
        "forward_only": False,
    }


def _measure_paired_phase(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    optimized_backend: TrainingBackend,
    *,
    phase: TrainingPhase,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> dict[TrainingBackend, dict[str, object]]:
    backends = (EAGER_BACKEND, optimized_backend)
    contexts = {
        backend: _build_training_context(
            model_name,
            length,
            batch_size,
            backend,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=config,
            device=device,
        )
        for backend in backends
    }
    phase_offset = {"forward": 1, "forward_backward": 2, "full": 3}[phase]
    step_seed = config.seed + 300_000 + length + batch_size + phase_offset
    _precondition_gpu_clock(config.gpu_clock_ramp_cycles)
    losses = {backend: torch.zeros((), device=contexts[backend][2].device) for backend in backends}
    for warmup_index in range(config.warmups):
        order = backends if warmup_index % 2 == 0 else tuple(reversed(backends))
        for backend in order:
            _set_step_seed(step_seed + warmup_index)
            losses[backend] = _phase_step(phase, contexts[backend], config.grad_clip_norm)
    torch.cuda.synchronize()
    wall_samples = {backend: [] for backend in backends}
    gpu_samples = {backend: [] for backend in backends}
    for group_index in range(config.groups):
        order = backends if group_index % 2 == 0 else tuple(reversed(backends))
        for backend in order:
            _set_step_seed(step_seed + config.warmups + group_index * config.iterations_per_group)
            _precondition_gpu_clock(config.gpu_clock_precondition_cycles)
            torch.cuda.synchronize()
            start_wall = perf_counter()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(config.iterations_per_group):
                losses[backend] = _phase_step(phase, contexts[backend], config.grad_clip_norm)
            end_event.record()
            end_event.synchronize()
            wall_samples[backend].append(
                (perf_counter() - start_wall) * 1000.0 / config.iterations_per_group
            )
            gpu_samples[backend].append(
                start_event.elapsed_time(end_event) / config.iterations_per_group
            )
    result: dict[TrainingBackend, dict[str, object]] = {}
    for backend in backends:
        wall_quartiles = statistics.quantiles(wall_samples[backend], n=4, method="inclusive")
        gpu_quartiles = statistics.quantiles(gpu_samples[backend], n=4, method="inclusive")
        result[backend] = {
            "wall_ms": statistics.median(wall_samples[backend]),
            "wall_iqr_ms": wall_quartiles[2] - wall_quartiles[0],
            "wall_samples_ms": wall_samples[backend],
            "gpu_ms": statistics.median(gpu_samples[backend]),
            "gpu_iqr_ms": gpu_quartiles[2] - gpu_quartiles[0],
            "gpu_samples_ms": gpu_samples[backend],
            "last_loss": float(losses[backend].detach().item()),
            "measurement_order": "ABBA alternating by group",
        }
    contexts.clear()
    losses.clear()
    _release_cuda()
    return result


def _measure_full_peak_memory(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    backend: TrainingBackend,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> float:
    context = _build_training_context(
        model_name,
        length,
        batch_size,
        backend,
        state_dict=state_dict,
        cpu_inputs=cpu_inputs,
        cpu_labels=cpu_labels,
        config=config,
        device=device,
    )
    _set_step_seed(config.seed + 400_000 + length + batch_size)
    _phase_step("full", context, config.grad_clip_norm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _phase_step("full", context, config.grad_clip_norm)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    del context
    _release_cuda()
    return peak


def _build_training_context(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    backend: TrainingBackend,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> tuple[nn.Module, torch.optim.AdamW, Tensor, Tensor]:
    model, _ = build_training_model(model_name, length, batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).train()
    _configure_backend(model, backend)
    optimizer = _make_optimizer(model, backend, config)
    return (
        model,
        optimizer,
        cpu_inputs.to(device=device, dtype=torch.float32),
        cpu_labels.to(device=device),
    )


def _phase_step(
    phase: TrainingPhase,
    context: tuple[nn.Module, torch.optim.AdamW, Tensor, Tensor],
    grad_clip_norm: float,
) -> Tensor:
    model, optimizer, inputs, labels = context
    if phase == "forward":
        return functional.cross_entropy(model(inputs), labels)
    if phase == "forward_backward":
        model.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(model(inputs), labels)
        loss.backward()
        return loss
    return _campaign_training_step(model, optimizer, inputs, labels, grad_clip_norm)


def _set_step_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prime_training_backend(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    backend: TrainingBackend,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> None:
    model, _ = build_training_model(model_name, length, batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).train()
    _configure_backend(model, backend)
    optimizer = _make_optimizer(model, backend, config)
    inputs = cpu_inputs.to(device=device, dtype=torch.float32)
    labels = cpu_labels.to(device=device)
    step_seed = config.seed + 50_000 + length + batch_size
    torch.manual_seed(step_seed)
    torch.cuda.manual_seed_all(step_seed)
    for _ in range(config.backend_prime_steps):
        _campaign_training_step(model, optimizer, inputs, labels, config.grad_clip_norm)
    torch.cuda.synchronize()
    del model, optimizer, inputs, labels
    _release_cuda()


def _measure_phase(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    backend: TrainingBackend,
    *,
    phase: TrainingPhase,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> dict[str, object]:
    model, _ = build_training_model(model_name, length, batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).train()
    _configure_backend(model, backend)
    optimizer = _make_optimizer(model, backend, config)
    inputs = cpu_inputs.to(device=device, dtype=torch.float32)
    labels = cpu_labels.to(device=device)
    phase_offset = {"forward": 1, "forward_backward": 2, "full": 3}[phase]
    step_seed = config.seed + 100_000 + length + batch_size + phase_offset
    torch.manual_seed(step_seed)
    torch.cuda.manual_seed_all(step_seed)
    _precondition_gpu_clock(config.gpu_clock_ramp_cycles)

    def step() -> Tensor:
        if phase == "forward":
            return functional.cross_entropy(model(inputs), labels)
        if phase == "forward_backward":
            model.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(inputs), labels)
            loss.backward()
            return loss
        return _campaign_training_step(model, optimizer, inputs, labels, config.grad_clip_norm)

    loss = torch.zeros((), device=inputs.device)
    for _ in range(config.warmups):
        loss = step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_samples: list[float] = []
    gpu_samples: list[float] = []
    for _ in range(config.groups):
        _precondition_gpu_clock(config.gpu_clock_precondition_cycles)
        torch.cuda.synchronize()
        start_wall = perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(config.iterations_per_group):
            loss = step()
        end_event.record()
        end_event.synchronize()
        wall_samples.append((perf_counter() - start_wall) * 1000.0 / config.iterations_per_group)
        gpu_samples.append(start_event.elapsed_time(end_event) / config.iterations_per_group)
    wall_quartiles = statistics.quantiles(wall_samples, n=4, method="inclusive")
    gpu_quartiles = statistics.quantiles(gpu_samples, n=4, method="inclusive")
    result: dict[str, object] = {
        "wall_ms": statistics.median(wall_samples),
        "wall_iqr_ms": wall_quartiles[2] - wall_quartiles[0],
        "wall_samples_ms": wall_samples,
        "gpu_ms": statistics.median(gpu_samples),
        "gpu_iqr_ms": gpu_quartiles[2] - gpu_quartiles[0],
        "gpu_samples_ms": gpu_samples,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "last_loss": float(loss.detach().item()),
    }
    del loss
    _release_cuda()
    return result


def _measure_training_parity(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    candidate_backend: TrainingBackend,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> dict[str, object]:
    initial_parameters: dict[str, Tensor] | None = None
    outcomes: dict[TrainingBackend, dict[str, object]] = {}
    step_seed = config.seed + 200_000 + length + batch_size
    for backend in (EAGER_BACKEND, candidate_backend):
        model, _ = build_training_model(model_name, length, batch_size)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device=device, dtype=torch.float32).train()
        _configure_backend(model, backend)
        optimizer = _make_optimizer(model, backend, config)
        inputs = cpu_inputs.to(device=device, dtype=torch.float32)
        labels = cpu_labels.to(device=device)
        if initial_parameters is None:
            initial_parameters = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
            }
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        for _ in range(config.parity_steps - 1):
            _campaign_training_step(model, optimizer, inputs, labels, config.grad_clip_norm)
        loss, gradients = _campaign_training_step_with_outputs(
            model, optimizer, inputs, labels, config.grad_clip_norm
        )
        outcomes[backend] = {
            "loss": loss,
            "gradients": gradients,
            "parameters": {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
            },
        }
        del model, optimizer, inputs, labels
        _release_cuda()
    if initial_parameters is None:
        message = "training parity did not capture initial parameters"
        raise RuntimeError(message)
    reference = outcomes[EAGER_BACKEND]
    candidate = outcomes[candidate_backend]
    reference_gradients = cast("dict[str, Tensor]", reference["gradients"])
    candidate_gradients = cast("dict[str, Tensor]", candidate["gradients"])
    reference_parameters = cast("dict[str, Tensor]", reference["parameters"])
    candidate_parameters = cast("dict[str, Tensor]", candidate["parameters"])
    reference_updates = {
        name: reference_parameters[name] - initial for name, initial in initial_parameters.items()
    }
    candidate_updates = {
        name: candidate_parameters[name] - initial for name, initial in initial_parameters.items()
    }
    gradient_abs, gradient_rel = _mapping_errors(candidate_gradients, reference_gradients)
    update_abs, update_rel = _mapping_errors(candidate_updates, reference_updates)
    parameter_abs, parameter_rel = _mapping_errors(candidate_parameters, reference_parameters)
    return {
        "reference_loss": _as_float(reference["loss"]),
        "candidate_loss": _as_float(candidate["loss"]),
        "loss_abs_error": abs(_as_float(candidate["loss"]) - _as_float(reference["loss"])),
        "gradient_key_agreement": set(candidate_gradients) == set(reference_gradients),
        "gradient_tensor_count": len(reference_gradients),
        "gradient_max_abs_error": gradient_abs,
        "gradient_max_rel_error": gradient_rel,
        "parameter_update_max_abs_error": update_abs,
        "parameter_update_max_rel_error": update_rel,
        "parameter_value_max_abs_error": parameter_abs,
        "parameter_value_max_rel_error": parameter_rel,
    }


def _parity_is_exact(parity: dict[str, object]) -> bool:
    return (
        parity.get("gradient_key_agreement") is True
        and _as_float(parity["loss_abs_error"]) <= MAXIMUM_PARITY_ERROR
        and _as_float(parity["gradient_max_abs_error"]) <= MAXIMUM_PARITY_ERROR
        and _as_float(parity["parameter_update_max_abs_error"]) <= MAXIMUM_PARITY_ERROR
    )


def _campaign_training_step(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    inputs: Tensor,
    labels: Tensor,
    grad_clip_norm: float,
) -> Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = functional.cross_entropy(model(inputs), labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    _post_optimizer_step(model)
    return loss


def _campaign_training_step_with_outputs(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    inputs: Tensor,
    labels: Tensor,
    grad_clip_norm: float,
) -> tuple[float, dict[str, Tensor]]:
    loss = _campaign_training_step(model, optimizer, inputs, labels, grad_clip_norm)
    torch.cuda.synchronize()
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return float(loss.detach().item()), gradients


def _configure_backend(model: nn.Module, backend: TrainingBackend) -> None:
    recurrence_backend = "triton_scan_blocks" if backend.startswith("block_scan") else "auto"
    blocks = [getattr(model, "forward_block", None), getattr(model, "backward_block", None)]
    blocks.extend(getattr(model, "extra_blocks", []))
    for block in blocks:
        if block is not None:
            block.recurrence_backend = recurrence_backend
            block.fused_moments_backward_training = backend != EAGER_BACKEND
    if hasattr(model, "use_fused_pa2wp_stem_training"):
        cast("_PA2WPTrainingStemControl", cast("object", model)).use_fused_pa2wp_stem_training = (
            backend != EAGER_BACKEND
        )
    if hasattr(model, "use_fused_efp16_stem_training"):
        cast("_EFP16TrainingStemControl", cast("object", model)).use_fused_efp16_stem_training = (
            backend != EAGER_BACKEND
        )


def _make_optimizer(
    model: nn.Module, backend: TrainingBackend, config: BenchmarkConfig
) -> torch.optim.AdamW:
    if backend.endswith("fused_adamw"):
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _post_optimizer_step(model: nn.Module) -> None:
    post_step = getattr(model, "post_optimizer_step", None)
    if callable(post_step):
        post_step()


def _mapping_errors(
    candidate: dict[str, Tensor], reference: dict[str, Tensor]
) -> tuple[float, float]:
    if set(candidate) != set(reference):
        return math.inf, math.inf
    maximum_abs = 0.0
    maximum_rel = 0.0
    for name, reference_value in reference.items():
        absolute = (candidate[name] - reference_value).abs()
        relative = absolute / reference_value.abs().clamp_min(1.0e-6)
        maximum_abs = max(maximum_abs, float(absolute.max().item()))
        maximum_rel = max(maximum_rel, float(relative.max().item()))
    return maximum_abs, maximum_rel


def _runtime_rows(
    rows: list[dict[str, object]], model_name: TrainingModelName, runtime: str
) -> list[dict[str, object]]:
    return sorted(
        [
            row
            for row in rows
            if row.get("model") == model_name
            and row.get("runtime") == runtime
            and row.get("status") == "measured"
        ],
        key=lambda row: (_as_int(row["length"]), _as_int(row["batch_size"])),
    )


def _state_dict_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _architecture_identity(architecture: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in architecture.items() if key != "state_dict_sha256"}


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _precondition_gpu_clock(cycles: int) -> None:
    if cycles <= 0:
        return
    sleeper = getattr(torch.cuda, "_sleep", None)
    if not callable(sleeper):
        message = "this CUDA runtime does not expose clock preconditioning"
        raise TypeError(message)
    sleeper(cycles)
    torch.cuda.synchronize()


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        message = f"expected integer-compatible value, got {value!r}"
        raise TypeError(message)
    return int(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        message = f"expected float-compatible value, got {value!r}"
        raise TypeError(message)
    return float(value)


def _validate_benchmark_arguments(
    models: tuple[TrainingModelName, ...],
    lengths: tuple[int, ...],
    batches: tuple[int, ...],
    config: BenchmarkConfig,
    device: str,
) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        message = "the training speed comparison requires CUDA"
        raise RuntimeError(message)
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        message = f"unknown training models: {unknown}"
        raise ValueError(message)
    if not models or not lengths or not batches:
        message = "models, lengths, and batches must be non-empty"
        raise ValueError(message)
    if min(lengths) < 2 or min(batches) < 1:
        message = "lengths must be >=2 and batches must be positive"
        raise ValueError(message)
    repeats = (
        config.warmups,
        config.groups,
        config.iterations_per_group,
        config.screening_warmups,
        config.screening_groups,
        config.screening_iterations_per_group,
        config.backend_prime_steps,
        config.parity_steps,
    )
    if min(repeats) < 1 or config.groups < 2 or config.screening_groups < 2:
        message = "warmups/iterations must be positive and group counts must be at least two"
        raise ValueError(message)
    if config.gpu_clock_precondition_cycles < 0:
        message = "GPU clock precondition cycles cannot be negative"
        raise ValueError(message)
    if config.gpu_clock_ramp_cycles < 0:
        message = "GPU clock ramp cycles cannot be negative"
        raise ValueError(message)


def _parse_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(",") if item)


def _parse_models(raw: str) -> tuple[TrainingModelName, ...]:
    return cast(
        "tuple[TrainingModelName, ...]",
        tuple(item for item in raw.split(",") if item),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Benchmark actual FP32 PAC training steps")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--models", default=",".join(MODELS))
    benchmark_parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    benchmark_parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    benchmark_parser.add_argument("--warmups", type=int, default=5)
    benchmark_parser.add_argument("--groups", type=int, default=7)
    benchmark_parser.add_argument("--iterations-per-group", type=int, default=10)
    benchmark_parser.add_argument("--screening-warmups", type=int, default=2)
    benchmark_parser.add_argument("--screening-groups", type=int, default=3)
    benchmark_parser.add_argument("--screening-iterations-per-group", type=int, default=5)
    benchmark_parser.add_argument("--parity-steps", type=int, default=1)
    benchmark_parser.add_argument("--seed", type=int, default=7)
    benchmark_parser.add_argument("--gpu-clock-ramp-cycles", type=int, default=2_000_000_000)
    benchmark_parser.add_argument("--gpu-clock-precondition-cycles", type=int, default=20_000_000)
    benchmark_parser.add_argument("--backend-prime-steps", type=int, default=1)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)

    replace_parser = subparsers.add_parser("replace-cells")
    replace_parser.add_argument("--base", type=Path, required=True)
    replace_parser.add_argument("--replacements", type=Path, nargs="+", required=True)
    replace_parser.add_argument("--output", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_max_parser = subparsers.add_parser("evaluate-max")
    evaluate_max_parser.add_argument("--input", type=Path, required=True)
    evaluate_max_parser.add_argument("--baseline", type=Path, required=True)
    evaluate_max_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        payload = benchmark(
            models=_parse_models(args.models),
            lengths=_parse_tuple(args.lengths),
            batches=_parse_tuple(args.batches),
            config=BenchmarkConfig(
                warmups=args.warmups,
                groups=args.groups,
                iterations_per_group=args.iterations_per_group,
                screening_warmups=args.screening_warmups,
                screening_groups=args.screening_groups,
                screening_iterations_per_group=args.screening_iterations_per_group,
                parity_steps=args.parity_steps,
                seed=args.seed,
                gpu_clock_ramp_cycles=args.gpu_clock_ramp_cycles,
                gpu_clock_precondition_cycles=args.gpu_clock_precondition_cycles,
                backend_prime_steps=args.backend_prime_steps,
            ),
        )
        _write_json(args.output, payload)
        return
    if args.command == "merge":
        payloads = [
            cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
            for path in args.inputs
        ]
        _write_json(args.output, merge_payloads(payloads))
        return
    if args.command == "replace-cells":
        base = cast("dict[str, object]", json.loads(args.base.read_text(encoding="utf-8")))
        replacements = [
            cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
            for path in args.replacements
        ]
        _write_json(args.output, replace_cells(base, replacements))
        return
    payload = cast("dict[str, object]", json.loads(args.input.read_text(encoding="utf-8")))
    if args.command == "evaluate-max":
        baseline = cast("dict[str, object]", json.loads(args.baseline.read_text(encoding="utf-8")))
        evaluation = evaluate_max_result(payload, baseline)
    else:
        evaluation = evaluate_result(payload)
    _write_json(args.output, evaluation)
    if evaluation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
