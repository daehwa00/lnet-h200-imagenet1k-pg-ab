# ruff: noqa: T201
# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import platform
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Final, cast

import torch
from torch import Tensor, nn

from .pac_efp16_training_cuda_graph import (
    EFP16TrainingCudaGraph,
    make_capturable_adamw,
    prepare_efp16_training_cuda_graph,
)
from .pac_pa2wp_training_cuda_graph import (
    PA2WPTrainingCudaGraph,
    prepare_pa2wp_training_cuda_graph,
)
from .pac_training_speed_comparison import (
    EAGER_BACKEND,
    TrainingBackend,
    TrainingModelName,
    _campaign_training_step,
    _configure_backend,
    _make_optimizer,
    build_training_model,
)
from .pac_training_speed_comparison import (
    BenchmarkConfig as CampaignBenchmarkConfig,
)

LENGTHS: Final = (128, 512, 2048)
BATCHES: Final = (1, 64)
MODELS: Final[tuple[TrainingModelName, ...]] = ("efp16", "pa2wp")
MAXIMUM_ERROR: Final = 2.0e-5
GRAPH_BACKENDS: Final[tuple[TrainingBackend, ...]] = (
    "campaign_auto_fused_adamw",
    "block_scan_fused_adamw",
)
GRAPH_COMPUTE_DTYPES: Final = ("float32", "float64")
CURRENT_BACKENDS: Final[dict[tuple[TrainingModelName, int, int], TrainingBackend]] = {
    ("efp16", 128, 1): "campaign_auto_fused_adamw",
    ("efp16", 128, 64): "block_scan_default_adamw",
    ("efp16", 512, 1): "campaign_auto_fused_adamw",
    ("efp16", 512, 64): "campaign_auto_fused_adamw",
    ("efp16", 2048, 1): "campaign_auto_fused_adamw",
    ("efp16", 2048, 64): "campaign_auto_fused_adamw",
    ("pa2wp", 128, 1): "campaign_auto_fused_adamw",
    ("pa2wp", 128, 64): "campaign_auto_fused_adamw",
    ("pa2wp", 512, 1): "campaign_auto_fused_adamw",
    ("pa2wp", 512, 64): "campaign_auto_fused_adamw",
    ("pa2wp", 2048, 1): "block_scan_fused_adamw",
    ("pa2wp", 2048, 64): "block_scan_default_adamw",
}


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    warmups: int = 5
    groups: int = 7
    iterations_per_group: int = 10
    screening_warmups: int = 2
    screening_groups: int = 3
    screening_iterations_per_group: int = 5
    parity_steps: int = 75
    seed: int = 7
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    maximum_graph_regression: float = 0.0
    taylor_degree: int = 12
    scaling_steps: int = 3
    matrix_exp_compute_dtypes: tuple[str, ...] = GRAPH_COMPUTE_DTYPES
    gpu_clock_ramp_cycles: int = 2_000_000_000
    gpu_clock_precondition_cycles: int = 20_000_000


DEFAULT_CONFIG: Final = BenchmarkConfig()


@dataclass
class _EagerContext:
    model: nn.Module
    optimizer: torch.optim.AdamW
    inputs: Tensor
    labels: Tensor
    grad_clip_norm: float

    def step(self) -> Tensor:
        return _campaign_training_step(
            self.model,
            self.optimizer,
            self.inputs,
            self.labels,
            self.grad_clip_norm,
        )


@dataclass
class _GraphContext:
    model: nn.Module
    runtime: EFP16TrainingCudaGraph | PA2WPTrainingCudaGraph
    inputs: Tensor
    labels: Tensor

    def step(self) -> Tensor:
        if isinstance(self.runtime, EFP16TrainingCudaGraph):
            return self.runtime.step(self.inputs, self.labels)
        return self.runtime(self.inputs, self.labels).loss

    @property
    def phase(self) -> str | None:
        if isinstance(self.runtime, PA2WPTrainingCudaGraph):
            return self.runtime.last_phase
        return None


def benchmark(
    *,
    models: tuple[TrainingModelName, ...] = MODELS,
    lengths: tuple[int, ...] = LENGTHS,
    batches: tuple[int, ...] = BATCHES,
    config: BenchmarkConfig = DEFAULT_CONFIG,
    device: str = "cuda",
) -> dict[str, object]:
    if device != "cuda" or not torch.cuda.is_available():
        message = "training CUDA Graph benchmark requires CUDA"
        raise RuntimeError(message)
    if config.parity_steps < 1 or config.groups < 2 or config.iterations_per_group < 1:
        message = "invalid benchmark repetition configuration"
        raise ValueError(message)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows: list[dict[str, object]] = []
    architectures: dict[str, dict[str, object]] = {}
    for model_name in models:
        for length in lengths:
            for batch_size in batches:
                row, architecture = _benchmark_cell(
                    model_name,
                    length,
                    batch_size,
                    config=config,
                    device=device,
                )
                architectures.setdefault(model_name, architecture)
                rows.append(row)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload: dict[str, object] = {
        "schema": "pac_training_cuda_graph_benchmark.v1",
        "environment": {
            "host": platform.node(),
            "device": properties.name,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "models": list(models),
        "lengths": list(lengths),
        "batches": list(batches),
        "config": asdict(config),
        "protocol": {
            "dtype": "float32 parameters, activations, gradients, and AdamW state",
            "orthogonal_compute_dtype_candidates": list(config.matrix_exp_compute_dtypes),
            "tf32": False,
            "autocast": False,
            "full_train_step": (
                "cross_entropy + backward + foreach gradient clip + AdamW + post step"
            ),
            "input_copy_included": True,
            "compile_and_capture_cost_included": False,
            "pa2wp_phase_policy": "one GPU Bernoulli per step; original/shifted probability 0.5",
            "timing": "CUDA event and synchronized wall time; alternating runtime order",
            "selection": (
                "screen auto/block recurrence and FP32/FP64 orthogonal compute; select graph "
                "only when 75-step exact; dispatch the fastest of campaign eager, current "
                "optimized, and the exact graph"
            ),
        },
        "architectures": architectures,
        "rows": rows,
    }
    payload["summary"] = summarize(rows)
    payload["evaluation"] = evaluate(payload)
    return payload


def _benchmark_cell(  # noqa: PLR0915 - benchmark lifecycle is intentionally linear
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    *,
    config: BenchmarkConfig,
    device: str,
) -> tuple[dict[str, object], dict[str, object]]:
    torch.manual_seed(config.seed)
    base_model, architecture = build_training_model(model_name, length, batch_size)
    state_dict = copy.deepcopy(base_model.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(config.seed + length + batch_size)
    cpu_inputs = torch.randn(batch_size, length, 1, generator=generator)
    cpu_labels = torch.randint(0, 5, (batch_size,), generator=generator)
    del base_model
    current_backend = CURRENT_BACKENDS[(model_name, length, batch_size)]

    screening: list[dict[str, object]] = []
    for graph_compute_dtype in config.matrix_exp_compute_dtypes:
        for graph_backend in GRAPH_BACKENDS:
            try:
                parity = _measure_parity(
                    model_name,
                    length,
                    batch_size,
                    current_backend,
                    graph_backend,
                    graph_compute_dtype=graph_compute_dtype,
                    steps=1,
                    state_dict=state_dict,
                    cpu_inputs=cpu_inputs,
                    cpu_labels=cpu_labels,
                    config=config,
                    device=device,
                )
                quick = _measure_one(
                    _build_graph_context(
                        model_name,
                        length,
                        batch_size,
                        graph_backend,
                        graph_compute_dtype=graph_compute_dtype,
                        state_dict=state_dict,
                        cpu_inputs=cpu_inputs,
                        cpu_labels=cpu_labels,
                        config=config,
                        device=device,
                    ),
                    warmups=config.screening_warmups,
                    groups=config.screening_groups,
                    iterations=config.screening_iterations_per_group,
                    seed=config.seed + 10_000 + length + batch_size,
                    gpu_clock_ramp_cycles=config.gpu_clock_ramp_cycles,
                    gpu_clock_precondition_cycles=config.gpu_clock_precondition_cycles,
                )
                screening.append(
                    {
                        "backend": graph_backend,
                        "matrix_exp_compute_dtype": graph_compute_dtype,
                        "status": "measured",
                        "exact": _parity_passes(parity),
                        "screening_wall_ms": quick["wall_ms"],
                        **parity,
                    }
                )
            except Exception as error:  # noqa: BLE001 - preserve candidate evidence
                screening.append(
                    {
                        "backend": graph_backend,
                        "matrix_exp_compute_dtype": graph_compute_dtype,
                        "status": "failed",
                        "exact": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            _release_cuda()
    exact = [row for row in screening if row.get("exact") is True]
    if not exact:
        message = f"no exact CUDA Graph candidate for {model_name}/N{length}/B{batch_size}"
        raise RuntimeError(message)
    ranked = sorted(exact, key=lambda row: _as_float(row["screening_wall_ms"]))
    selected_screen: dict[str, object] | None = None
    parity: dict[str, object] | None = None
    for candidate in ranked:
        candidate_parity = _measure_parity(
            model_name,
            length,
            batch_size,
            current_backend,
            cast("TrainingBackend", candidate["backend"]),
            graph_compute_dtype=str(candidate["matrix_exp_compute_dtype"]),
            steps=config.parity_steps,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=config,
            device=device,
        )
        candidate["long_parity"] = candidate_parity
        if _parity_passes(candidate_parity):
            selected_screen = candidate
            parity = candidate_parity
            break
    if selected_screen is None or parity is None:
        selected_screen = ranked[0]
        stored_parity = selected_screen.get("long_parity")
        if not isinstance(stored_parity, dict):
            stored_parity = _measure_parity(
                model_name,
                length,
                batch_size,
                current_backend,
                cast("TrainingBackend", selected_screen["backend"]),
                graph_compute_dtype=str(selected_screen["matrix_exp_compute_dtype"]),
                steps=config.parity_steps,
                state_dict=state_dict,
                cpu_inputs=cpu_inputs,
                cpu_labels=cpu_labels,
                config=config,
                device=device,
            )
            selected_screen["long_parity"] = stored_parity
        parity = cast("dict[str, object]", stored_parity)
    graph_backend = cast("TrainingBackend", selected_screen["backend"])
    graph_compute_dtype = str(selected_screen["matrix_exp_compute_dtype"])
    contexts: dict[str, _EagerContext | _GraphContext] = {
        "campaign_eager": _build_eager_context(
            model_name,
            length,
            batch_size,
            EAGER_BACKEND,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=config,
            device=device,
        ),
        "current_optimized": _build_eager_context(
            model_name,
            length,
            batch_size,
            current_backend,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=config,
            device=device,
        ),
        "cuda_graph": _build_graph_context(
            model_name,
            length,
            batch_size,
            graph_backend,
            graph_compute_dtype=graph_compute_dtype,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=config,
            device=device,
        ),
    }
    measurements = _measure_paired(
        contexts,
        warmups=config.warmups,
        groups=config.groups,
        iterations=config.iterations_per_group,
        seed=config.seed + 30_000 + length + batch_size,
        gpu_clock_ramp_cycles=config.gpu_clock_ramp_cycles,
        gpu_clock_precondition_cycles=config.gpu_clock_precondition_cycles,
    )
    contexts.clear()
    _release_cuda()
    measurements["campaign_eager"]["peak_memory_mb"] = _measure_peak_memory(
        _build_eager_context(
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
    )
    measurements["current_optimized"]["peak_memory_mb"] = _measure_peak_memory(
        _build_eager_context(
            model_name,
            length,
            batch_size,
            current_backend,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=config,
            device=device,
        )
    )
    measurements["cuda_graph"]["peak_memory_mb"] = _measure_peak_memory(
        _build_graph_context(
            model_name,
            length,
            batch_size,
            graph_backend,
            graph_compute_dtype=graph_compute_dtype,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            config=config,
            device=device,
        )
    )
    eager_ms = _as_float(measurements["campaign_eager"]["wall_ms"])
    current_ms = _as_float(measurements["current_optimized"]["wall_ms"])
    graph_ms = _as_float(measurements["cuda_graph"]["wall_ms"])
    baseline_runtime, baseline_ms = min(
        (("campaign_eager", eager_ms), ("current_optimized", current_ms)),
        key=lambda item: item[1],
    )
    graph_eligible = _parity_passes(parity) and graph_ms < baseline_ms * (
        1.0 + config.maximum_graph_regression
    )
    selected_runtime = "cuda_graph" if graph_eligible else baseline_runtime
    selected_ms = graph_ms if graph_eligible else baseline_ms
    row = {
        "model": model_name,
        "length": length,
        "batch_size": batch_size,
        "status": "measured",
        "current_backend": current_backend,
        "graph_backend": graph_backend,
        "graph_matrix_exp_compute_dtype": graph_compute_dtype,
        "graph_screening": screening,
        "campaign_eager": measurements["campaign_eager"],
        "current_optimized": measurements["current_optimized"],
        "cuda_graph": measurements["cuda_graph"],
        "selected_runtime": selected_runtime,
        "selected_wall_ms": selected_ms,
        "selected_sequences_per_second": batch_size * 1000.0 / selected_ms,
        "speedup_vs_campaign_eager": eager_ms / selected_ms,
        "speedup_vs_current_optimized": current_ms / selected_ms,
        "raw_graph_speedup_vs_current": current_ms / graph_ms,
        **parity,
    }
    _release_cuda()
    return row, architecture


def _build_eager_context(
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
) -> _EagerContext:
    model, _ = build_training_model(model_name, length, batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).train()
    _configure_backend(model, backend)
    optimizer = _make_optimizer(model, backend, _campaign_config(config))
    return _EagerContext(
        model,
        optimizer,
        cpu_inputs.to(device=device, dtype=torch.float32),
        cpu_labels.to(device=device),
        config.grad_clip_norm,
    )


def _build_graph_context(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    backend: TrainingBackend,
    *,
    graph_compute_dtype: str,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    config: BenchmarkConfig,
    device: str,
    fused_recurrence_moments_backward_training: bool = False,
    recurrence_backend_override: str | None = None,
    pa2wp_phase_schedule_capacity: int | None = 64,
    pa2wp_large_fused_stem_training: bool = False,
    canonical_identity_elision: bool = True,
    mode_static_pole_training: bool = False,
    packed_recurrence_moments_training: bool | None = None,
    two_pass_reverse_recurrence_moments_training: bool | None = None,
    efp16_stem_parameter_gradient_strategy: str = "auto",
    fused_optimizer_tail: bool = False,
    fused_rmsnorm_mean_training: bool = False,
    fused_rmsnorm_mean_backward_training: bool = False,
    fused_recurrence_moments_backward_blocks: tuple[int, ...] | None = None,
) -> _GraphContext:
    model, _ = build_training_model(model_name, length, batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).train()
    _configure_backend(model, backend)
    model_blocks = (
        getattr(model, "forward_block", None),
        getattr(model, "backward_block", None),
        *getattr(model, "extra_blocks", []),
    )
    for block_index, block in enumerate(model_blocks):
        if block is not None:
            block.canonical_identity_elision = canonical_identity_elision
            block.mode_static_pole_training = mode_static_pole_training
            block.packed_recurrence_moments_training = packed_recurrence_moments_training
            block.two_pass_reverse_recurrence_moments_training = (
                two_pass_reverse_recurrence_moments_training
            )
            _configure_selective_fused_recurrence(
                block,
                block_index=block_index,
                selected_blocks=fused_recurrence_moments_backward_blocks,
            )
    if hasattr(model, "efp16_stem_parameter_gradient_strategy"):
        model.__dict__["efp16_stem_parameter_gradient_strategy"] = (
            efp16_stem_parameter_gradient_strategy
        )
    if hasattr(model, "use_fused_rmsnorm_mean_training"):
        model.__dict__["use_fused_rmsnorm_mean_training"] = fused_rmsnorm_mean_training
    if hasattr(model, "use_fused_rmsnorm_mean_backward_training"):
        model.__dict__["use_fused_rmsnorm_mean_backward_training"] = (
            fused_rmsnorm_mean_backward_training
        )
    if recurrence_backend_override is not None:
        blocks = [getattr(model, "forward_block", None), getattr(model, "backward_block", None)]
        blocks.extend(getattr(model, "extra_blocks", []))
        for block in blocks:
            if block is not None:
                block.__dict__["recurrence_backend"] = recurrence_backend_override
    inputs = cpu_inputs.to(device=device, dtype=torch.float32)
    labels = cpu_labels.to(device=device)
    if model_name == "efp16":
        optimizer = make_capturable_adamw(
            model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        runtime: EFP16TrainingCudaGraph | PA2WPTrainingCudaGraph = (
            prepare_efp16_training_cuda_graph(
                model,
                optimizer,
                inputs,
                labels,
                grad_clip_norm=config.grad_clip_norm,
                taylor_degree=config.taylor_degree,
                scaling_steps=config.scaling_steps,
                matrix_exp_compute_dtype=_matrix_exp_dtype(graph_compute_dtype),
                fused_recurrence_moments_backward_training=(
                    fused_recurrence_moments_backward_training
                    and fused_recurrence_moments_backward_blocks is None
                ),
                fused_optimizer_tail=fused_optimizer_tail,
            )
        )
    else:
        runtime = prepare_pa2wp_training_cuda_graph(
            model,
            batch_size=batch_size,
            sequence_length=length,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            grad_clip_norm=config.grad_clip_norm,
            taylor_degree=config.taylor_degree,
            scaling_steps=config.scaling_steps,
            matrix_exp_compute_dtype=_matrix_exp_dtype(graph_compute_dtype),
            fused_recurrence_moments_backward_training=(fused_recurrence_moments_backward_training),
            phase_schedule_capacity=pa2wp_phase_schedule_capacity,
            large_fused_stem_training=pa2wp_large_fused_stem_training,
            fused_optimizer_tail=fused_optimizer_tail,
        )
    return _GraphContext(model, runtime, inputs, labels)


def _configure_selective_fused_recurrence(
    block: nn.Module,
    *,
    block_index: int,
    selected_blocks: tuple[int, ...] | None,
) -> None:
    if selected_blocks is None:
        return
    block.__dict__["fused_recurrence_moments_backward_training"] = (
        block_index in selected_blocks
    )


def _campaign_config(config: BenchmarkConfig) -> CampaignBenchmarkConfig:
    return CampaignBenchmarkConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )


def _matrix_exp_dtype(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float64":
        return torch.float64
    message = f"unsupported matrix-exp compute dtype: {value}"
    raise ValueError(message)


def _measure_one(
    context: _EagerContext | _GraphContext,
    *,
    warmups: int,
    groups: int,
    iterations: int,
    seed: int,
    gpu_clock_ramp_cycles: int,
    gpu_clock_precondition_cycles: int,
) -> dict[str, object]:
    return _measure_paired(
        {"runtime": context},
        warmups=warmups,
        groups=groups,
        iterations=iterations,
        seed=seed,
        gpu_clock_ramp_cycles=gpu_clock_ramp_cycles,
        gpu_clock_precondition_cycles=gpu_clock_precondition_cycles,
    )["runtime"]


def _measure_peak_memory(context: _EagerContext | _GraphContext) -> float:
    context.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    context.step()
    torch.cuda.synchronize()
    peak_memory_mb = torch.cuda.max_memory_allocated() / 2**20
    del context
    _release_cuda()
    return peak_memory_mb


def _measure_paired(
    contexts: dict[str, _EagerContext | _GraphContext],
    *,
    warmups: int,
    groups: int,
    iterations: int,
    seed: int,
    gpu_clock_ramp_cycles: int,
    gpu_clock_precondition_cycles: int,
) -> dict[str, dict[str, object]]:
    names = tuple(contexts)
    last_loss = {name: torch.zeros((), device="cuda") for name in names}
    _precondition_gpu_clock(gpu_clock_ramp_cycles)
    for warmup in range(warmups):
        order = names if warmup % 2 == 0 else tuple(reversed(names))
        for name in order:
            _set_seed(seed + warmup)
            last_loss[name] = contexts[name].step()
    torch.cuda.synchronize()
    wall = {name: [] for name in names}
    gpu = {name: [] for name in names}
    for group in range(groups):
        rotation = group % len(names)
        rotated = names[rotation:] + names[:rotation]
        order = rotated if group % 2 == 0 else tuple(reversed(rotated))
        for name in order:
            _set_seed(seed + warmups + group * iterations)
            _precondition_gpu_clock(gpu_clock_precondition_cycles)
            torch.cuda.synchronize()
            start_wall = perf_counter()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                last_loss[name] = contexts[name].step()
            end.record()
            end.synchronize()
            wall[name].append((perf_counter() - start_wall) * 1000.0 / iterations)
            gpu[name].append(start.elapsed_time(end) / iterations)
    result: dict[str, dict[str, object]] = {}
    for name in names:
        wall_q = statistics.quantiles(wall[name], n=4, method="inclusive")
        gpu_q = statistics.quantiles(gpu[name], n=4, method="inclusive")
        result[name] = {
            "wall_ms": statistics.median(wall[name]),
            "wall_iqr_ms": wall_q[2] - wall_q[0],
            "wall_samples_ms": wall[name],
            "gpu_ms": statistics.median(gpu[name]),
            "gpu_iqr_ms": gpu_q[2] - gpu_q[0],
            "gpu_samples_ms": gpu[name],
            "last_loss": float(last_loss[name].detach().item()),
        }
    return result


def _measure_parity(
    model_name: TrainingModelName,
    length: int,
    batch_size: int,
    current_backend: TrainingBackend,
    graph_backend: TrainingBackend,
    *,
    graph_compute_dtype: str,
    steps: int,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> dict[str, object]:
    reference = _build_eager_context(
        model_name,
        length,
        batch_size,
        current_backend,
        state_dict=state_dict,
        cpu_inputs=cpu_inputs,
        cpu_labels=cpu_labels,
        config=config,
        device=device,
    )
    candidate = _build_graph_context(
        model_name,
        length,
        batch_size,
        graph_backend,
        graph_compute_dtype=graph_compute_dtype,
        state_dict=state_dict,
        cpu_inputs=cpu_inputs,
        cpu_labels=cpu_labels,
        config=config,
        device=device,
    )
    step_seed = config.seed + 20_000 + length + batch_size
    expected_phases: list[str] = []
    if model_name == "pa2wp":
        _set_seed(step_seed)
        expected_phases = [
            "shifted" if bool(torch.rand((), device=device) < 0.5) else "original"
            for _ in range(steps)
        ]
    _set_seed(step_seed)
    reference_losses = [float(reference.step().detach().item()) for _ in range(steps)]
    candidate_losses: list[float] = []
    actual_phases: list[str] = []
    _set_seed(step_seed)
    for _ in range(steps):
        candidate_losses.append(float(candidate.step().detach().item()))
        if candidate.phase is not None:
            actual_phases.append(candidate.phase)
    torch.cuda.synchronize()
    loss_error = max(
        abs(candidate_loss - reference_loss)
        for candidate_loss, reference_loss in zip(candidate_losses, reference_losses, strict=True)
    )
    gradient_error = _named_tensor_error(
        {
            name: parameter.grad
            for name, parameter in candidate.model.named_parameters()
            if parameter.grad is not None
        },
        {
            name: parameter.grad
            for name, parameter in reference.model.named_parameters()
            if parameter.grad is not None
        },
    )
    parameter_error = _named_tensor_error(
        dict(candidate.model.named_parameters()),
        dict(reference.model.named_parameters()),
    )
    result: dict[str, object] = {
        "parity_steps": steps,
        "loss_trajectory_max_abs_error": loss_error,
        "final_gradient_max_abs_error": gradient_error,
        "final_parameter_max_abs_error": parameter_error,
        "pa2wp_phase_sequence_agreement": (
            actual_phases == expected_phases if model_name == "pa2wp" else True
        ),
        "pa2wp_original_steps": actual_phases.count("original"),
        "pa2wp_shifted_steps": actual_phases.count("shifted"),
    }
    del reference, candidate
    _release_cuda()
    return result


def _named_tensor_error(
    candidate: dict[str, Tensor | None], reference: dict[str, Tensor | None]
) -> float:
    if set(candidate) != set(reference):
        return math.inf
    maximum = 0.0
    for name, reference_value in reference.items():
        candidate_value = candidate[name]
        if candidate_value is None or reference_value is None:
            if candidate_value is not reference_value:
                return math.inf
            continue
        maximum = max(
            maximum,
            float((candidate_value.detach() - reference_value.detach()).abs().max().item()),
        )
    return maximum


def _parity_passes(parity: dict[str, object]) -> bool:
    return (
        bool(parity["pa2wp_phase_sequence_agreement"])
        and _as_float(parity["loss_trajectory_max_abs_error"]) <= MAXIMUM_ERROR
        and _as_float(parity["final_gradient_max_abs_error"]) <= MAXIMUM_ERROR
        and _as_float(parity["final_parameter_max_abs_error"]) <= MAXIMUM_ERROR
    )


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for model_name in MODELS:
        model_rows = [row for row in rows if row.get("model") == model_name]
        if not model_rows:
            continue
        summary[model_name] = {
            "shape_count": len(model_rows),
            "graph_selected_count": sum(
                row.get("selected_runtime") == "cuda_graph" for row in model_rows
            ),
            "geometric_mean_speedup_vs_campaign_eager": _geometric_mean(
                [_as_float(row["speedup_vs_campaign_eager"]) for row in model_rows]
            ),
            "geometric_mean_speedup_vs_current_optimized": _geometric_mean(
                [_as_float(row["speedup_vs_current_optimized"]) for row in model_rows]
            ),
            "median_selected_wall_ms": statistics.median(
                [_as_float(row["selected_wall_ms"]) for row in model_rows]
            ),
            "maximum_selected_loss_trajectory_abs_error": max(
                _selected_error(row, "loss_trajectory_max_abs_error") for row in model_rows
            ),
            "maximum_selected_final_gradient_abs_error": max(
                _selected_error(row, "final_gradient_max_abs_error") for row in model_rows
            ),
            "maximum_selected_final_parameter_abs_error": max(
                _selected_error(row, "final_parameter_max_abs_error") for row in model_rows
            ),
            "maximum_screened_graph_parameter_abs_error": max(
                _as_float(row["final_parameter_max_abs_error"]) for row in model_rows
            ),
        }
    return summary


def evaluate(payload: dict[str, object]) -> dict[str, object]:
    failures: list[str] = []
    environment = cast("dict[str, object]", payload.get("environment", {}))
    if "4090" not in str(environment.get("device", "")):
        failures.append("benchmark device is not an RTX 4090")
    rows = cast("list[dict[str, object]]", payload.get("rows", []))
    expected = (
        len(cast("list[object]", payload.get("models", [])))
        * len(cast("list[object]", payload.get("lengths", [])))
        * len(cast("list[object]", payload.get("batches", [])))
    )
    if len(rows) != expected:
        failures.append(f"expected {expected} measured cells, got {len(rows)}")
    for row in rows:
        cell = f"{row.get('model')}/N{row.get('length')}/B{row.get('batch_size')}"
        if row.get("status") != "measured":
            failures.append(f"unmeasured cell {cell}")
            continue
        if row.get("selected_runtime") == "cuda_graph":
            for key in (
                "loss_trajectory_max_abs_error",
                "final_gradient_max_abs_error",
                "final_parameter_max_abs_error",
            ):
                value = _as_float(row.get(key, math.inf))
                if not math.isfinite(value) or value > MAXIMUM_ERROR:
                    failures.append(f"{key} exceeded for selected graph {cell}: {value:.6g}")
        if row.get("pa2wp_phase_sequence_agreement") is not True:
            failures.append(f"phase sequence disagreement for {cell}")
        current = _as_float(cast("dict[str, object]", row["current_optimized"])["wall_ms"])
        eager = _as_float(cast("dict[str, object]", row["campaign_eager"])["wall_ms"])
        selected = _as_float(row["selected_wall_ms"])
        if selected > min(eager, current) * 1.000001:
            failures.append(f"dispatch regressed an exact baseline runtime for {cell}")
    return {
        "schema": "pac_training_cuda_graph_benchmark_evaluation.v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_cells": len(rows),
    }


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _selected_error(row: dict[str, object], key: str) -> float:
    return _as_float(row[key]) if row.get("selected_runtime") == "cuda_graph" else 0.0


def _as_float(value: object) -> float:
    return float(cast("float | int", value))


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _precondition_gpu_clock(cycles: int) -> None:
    if cycles <= 0:
        return
    sleeper = getattr(torch.cuda, "_sleep", None)
    if not callable(sleeper):
        message = "this CUDA runtime does not expose clock preconditioning"
        raise TypeError(message)
    sleeper(cycles)


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _parse_models(value: str) -> tuple[TrainingModelName, ...]:
    parsed = tuple(item for item in value.split(",") if item)
    if any(item not in MODELS for item in parsed):
        message = f"models must be selected from {MODELS}"
        raise ValueError(message)
    return cast("tuple[TrainingModelName, ...]", parsed)


def _parse_compute_dtypes(value: str) -> tuple[str, ...]:
    parsed = tuple(item for item in value.split(",") if item)
    if not parsed or any(item not in GRAPH_COMPUTE_DTYPES for item in parsed):
        message = f"matrix-exp compute dtypes must be selected from {GRAPH_COMPUTE_DTYPES}"
        raise ValueError(message)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    parser.add_argument("--warmups", type=int, default=DEFAULT_CONFIG.warmups)
    parser.add_argument("--groups", type=int, default=DEFAULT_CONFIG.groups)
    parser.add_argument(
        "--iterations-per-group", type=int, default=DEFAULT_CONFIG.iterations_per_group
    )
    parser.add_argument("--parity-steps", type=int, default=DEFAULT_CONFIG.parity_steps)
    parser.add_argument("--taylor-degree", type=int, default=DEFAULT_CONFIG.taylor_degree)
    parser.add_argument("--scaling-steps", type=int, default=DEFAULT_CONFIG.scaling_steps)
    parser.add_argument(
        "--matrix-exp-compute-dtypes",
        default=",".join(DEFAULT_CONFIG.matrix_exp_compute_dtypes),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = BenchmarkConfig(
        warmups=arguments.warmups,
        groups=arguments.groups,
        iterations_per_group=arguments.iterations_per_group,
        parity_steps=arguments.parity_steps,
        taylor_degree=arguments.taylor_degree,
        scaling_steps=arguments.scaling_steps,
        matrix_exp_compute_dtypes=_parse_compute_dtypes(arguments.matrix_exp_compute_dtypes),
    )
    payload = benchmark(
        models=_parse_models(arguments.models),
        lengths=_parse_csv_ints(arguments.lengths),
        batches=_parse_csv_ints(arguments.batches),
        config=config,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps(payload["evaluation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
