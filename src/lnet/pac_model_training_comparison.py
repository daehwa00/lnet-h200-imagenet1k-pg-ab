"""Uniform exact-FP32 training-speed comparison for the ten PAC paper models.

The benchmark deliberately keeps input tensors resident on the GPU.  Every
runtime copies those caller tensors into runtime-owned static buffers before a
step, so eager and CUDA Graph measurements include the same device-to-device
input and label copies.  Model construction, CUDA Graph capture, extension
builds, and other setup are excluded from latency and reported separately.

The module is safe to import on CPU-only hosts.  CUDA is required only by the
``benchmark`` and ``memory`` commands; merge/evaluate and schema tests remain
CPU-only.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_efp16_training_cuda_graph import (
    make_capturable_adamw,
    prepare_efp16_training_cuda_graph,
)
from .pac_metrics import count_parameters
from .pac_model_speed_comparison import (
    BATCHES,
    DISPLAY_NAMES,
    LENGTHS,
    MODELS,
    ModelName,
    build_comparison_model,
)
from .pac_pa2wp_training_cuda_graph import prepare_pa2wp_training_cuda_graph

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

TrainingPhase = Literal["forward", "forward_backward", "full"]
BackendName = Literal["eager_fp32_training", "cuda_graph_fp32_training"]
RuntimeName = Literal[
    "eager_fp32_training",
    "cuda_graph_fp32_training",
    "fastest_exact_fp32_training",
]

EAGER_RUNTIME: Final[BackendName] = "eager_fp32_training"
GRAPH_RUNTIME: Final[BackendName] = "cuda_graph_fp32_training"
FASTEST_RUNTIME: Final[RuntimeName] = "fastest_exact_fp32_training"
PHASES: Final[tuple[TrainingPhase, ...]] = ("forward", "forward_backward", "full")
MAXIMUM_PARITY_ERROR: Final = 2.0e-5
SCHEMA: Final = "pac_model_training_comparison.v1"
EVALUATION_SCHEMA: Final = "pac_model_training_comparison_evaluation.v1"
MEMORY_SCHEMA: Final = "pac_model_training_comparison_memory.v1"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    warmups: int = 5
    groups: int = 7
    iterations_per_group: int = 10
    graph_warmups: int = 3
    parity_steps: int = 1
    seed: int = 7
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    gpu_clock_ramp_cycles: int = 2_000_000_000
    gpu_clock_precondition_cycles: int = 20_000_000


DEFAULT_CONFIG: Final = BenchmarkConfig()


class _TrainingRuntime(Protocol):
    model: nn.Module
    backend: str
    setup_seconds: float

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor: ...

    def reset_rng_schedule(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainingSnapshot:
    loss: Tensor
    gradients: dict[str, Tensor]
    parameters: dict[str, Tensor]


class _DeviceResidentEagerRuntime:
    """Eager phase runner with the same D2D input-copy contract as graph runners."""

    backend = EAGER_RUNTIME

    def __init__(
        self,
        model: nn.Module,
        phase: TrainingPhase,
        example_inputs: Tensor,
        example_labels: Tensor,
        *,
        config: BenchmarkConfig,
    ) -> None:
        started = perf_counter()
        self.model = model.train()
        self.phase = phase
        self.static_inputs = example_inputs.clone()
        self.static_labels = example_labels.clone()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.grad_clip_norm = config.grad_clip_norm
        self.setup_seconds = perf_counter() - started

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        self.static_inputs.copy_(inputs, non_blocking=True)
        self.static_labels.copy_(labels, non_blocking=True)
        if self.phase != "forward":
            self.optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(self.model(self.static_inputs), self.static_labels)
        if self.phase == "forward":
            return loss
        loss.backward()
        if self.phase == "full":
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm,
            )
            self.optimizer.step()
            _post_optimizer_step(self.model)
        return loss

    def reset_rng_schedule(self) -> None:
        return


class _GenericCudaGraphRuntime:
    """One captured training phase for graph-compatible baseline models."""

    backend = GRAPH_RUNTIME

    def __init__(
        self,
        model: nn.Module,
        phase: TrainingPhase,
        example_inputs: Tensor,
        example_labels: Tensor,
        *,
        config: BenchmarkConfig,
    ) -> None:
        started = perf_counter()
        self.model = model.train()
        self.phase = phase
        self.static_inputs = example_inputs.clone()
        self.static_labels = example_labels.clone()
        self.grad_clip_norm = config.grad_clip_norm
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
            capturable=True,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.loss: Tensor | None = None
        if phase != "forward":
            for parameter in self.model.parameters():
                parameter.grad = torch.zeros_like(parameter)
        if phase == "full":
            _materialize_capturable_optimizer_state(self.model, self.optimizer)
        pristine = _capture_mutable_state(self.model, self.optimizer)
        capture_stream = torch.cuda.Stream(device=example_inputs.device)
        capture_stream.wait_stream(torch.cuda.current_stream(example_inputs.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(config.graph_warmups):
                self._static_step()
        torch.cuda.current_stream(example_inputs.device).wait_stream(capture_stream)
        torch.cuda.synchronize(example_inputs.device)
        _restore_mutable_state(self.model, self.optimizer, pristine)
        capture_stream.wait_stream(torch.cuda.current_stream(example_inputs.device))
        with (
            torch.cuda.stream(capture_stream),
            torch.cuda.graph(self.graph, stream=capture_stream),
        ):
            self.loss = self._static_step()
        torch.cuda.current_stream(example_inputs.device).wait_stream(capture_stream)
        torch.cuda.synchronize(example_inputs.device)
        _restore_mutable_state(self.model, self.optimizer, pristine)
        self.setup_seconds = perf_counter() - started

    def _static_step(self) -> Tensor:
        if self.phase != "forward":
            self.optimizer.zero_grad(set_to_none=False)
        loss = functional.cross_entropy(self.model(self.static_inputs), self.static_labels)
        if self.phase == "forward":
            return loss
        loss.backward()
        if self.phase == "full":
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm,
                foreach=True,
            )
            self.optimizer.step()
            _post_optimizer_step(self.model)
        return loss

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        self.static_inputs.copy_(inputs, non_blocking=True)
        self.static_labels.copy_(labels, non_blocking=True)
        self.graph.replay()
        if self.loss is None:
            message = "captured training phase did not retain a loss tensor"
            raise RuntimeError(message)
        return self.loss.detach()

    def reset_rng_schedule(self) -> None:
        return


# Reusable public name for fixed-shape baseline optimization screens.
GenericCudaGraphRuntime = _GenericCudaGraphRuntime


class _EFP16FullGraphRuntime:
    backend = GRAPH_RUNTIME

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Tensor,
        example_labels: Tensor,
        *,
        config: BenchmarkConfig,
    ) -> None:
        started = perf_counter()
        self.model = model.train()
        optimizer = make_capturable_adamw(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.runtime = prepare_efp16_training_cuda_graph(
            self.model,
            optimizer,
            example_inputs,
            example_labels,
            grad_clip_norm=config.grad_clip_norm,
            warmup_steps=config.graph_warmups,
            copy_inputs=True,
            copy_loss=False,
            prepare_model=True,
        )
        self.setup_seconds = perf_counter() - started

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        return self.runtime.step(inputs, labels)

    def reset_rng_schedule(self) -> None:
        return


class _PA2WPFullGraphRuntime:
    backend = GRAPH_RUNTIME

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Tensor,
        _example_labels: Tensor,
        *,
        config: BenchmarkConfig,
    ) -> None:
        started = perf_counter()
        self.model = model.train()
        self.runtime = prepare_pa2wp_training_cuda_graph(
            self.model,
            batch_size=example_inputs.shape[0],
            sequence_length=example_inputs.shape[1],
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            grad_clip_norm=config.grad_clip_norm,
            warmup_steps_per_phase=config.graph_warmups,
            phase_schedule_capacity=64,
        )
        self.setup_seconds = perf_counter() - started

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        return self.runtime(inputs, labels).loss

    def reset_rng_schedule(self) -> None:
        self.runtime.reset_phase_schedule()


def benchmark(
    *,
    models: tuple[ModelName, ...] = MODELS,
    lengths: tuple[int, ...] = LENGTHS,
    batches: tuple[int, ...] = BATCHES,
    config: BenchmarkConfig = DEFAULT_CONFIG,
    device: str = "cuda",
    memory_dir: Path | None = None,
    isolated_memory: bool = True,
) -> dict[str, object]:
    """Measure eager, CUDA Graph candidate, and fastest-exact rows."""
    _validate_benchmark_arguments(models, lengths, batches, config, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _precondition_gpu_clock(config.gpu_clock_ramp_cycles)

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
                    memory_dir=memory_dir,
                    isolated_memory=isolated_memory,
                )
                previous = architectures.setdefault(model_name, architecture)
                if _architecture_identity(previous) != _architecture_identity(architecture):
                    message = f"{model_name} architecture changed across shapes"
                    raise RuntimeError(message)
                rows.extend(cell_rows)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema": SCHEMA,
        "environment": {
            "device": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allow_tf32": False,
        },
        "protocol": _protocol(isolated_memory=isolated_memory),
        "config": asdict(config),
        "models": list(models),
        "lengths": list(lengths),
        "batches": list(batches),
        "architectures": architectures,
        "rows": rows,
        "summary": summarize(rows),
    }


def _benchmark_cell(
    model_name: ModelName,
    length: int,
    batch_size: int,
    *,
    config: BenchmarkConfig,
    device: str,
    memory_dir: Path | None,
    isolated_memory: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(config.seed)
    base_model, architecture = build_comparison_model(model_name, length, batch_size)
    state_dict = copy.deepcopy(base_model.state_dict())
    architecture = {
        **architecture,
        "trainable_parameters": count_parameters(base_model),
        "state_dict_sha256": _state_dict_digest(base_model),
    }
    parameter_count = _as_int(architecture["trainable_parameters"])
    del base_model
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + 1009 * length + 9176 * batch_size
    )
    inputs = torch.randn(batch_size, length, 1, generator=generator).to(
        device=device, dtype=torch.float32
    )
    labels = torch.randint(0, 5, (batch_size,), generator=generator).to(
        device=device, dtype=torch.long
    )

    eager_measurements, eager_setup, eager_backends = _measure_backend_phases(
        model_name,
        length,
        batch_size,
        EAGER_RUNTIME,
        state_dict=state_dict,
        inputs=inputs,
        labels=labels,
        config=config,
        device=device,
    )
    eager_memory = _memory_for_backend(
        model_name,
        length,
        batch_size,
        EAGER_RUNTIME,
        config=config,
        memory_dir=memory_dir,
        isolated_memory=isolated_memory,
    )
    eager_row = _assemble_row(
        model_name,
        length,
        batch_size,
        EAGER_RUNTIME,
        measurements=eager_measurements,
        setup_seconds=eager_setup,
        phase_backends=eager_backends,
        parameters=parameter_count,
        memory=eager_memory,
        parity=_zero_parity(config.parity_steps),
        status="measured",
    )

    graph_row: dict[str, object]
    try:
        parity = _measure_parity(
            model_name,
            length,
            batch_size,
            state_dict=state_dict,
            inputs=inputs,
            labels=labels,
            config=config,
            device=device,
        )
        graph_measurements, graph_setup, graph_backends = _measure_backend_phases(
            model_name,
            length,
            batch_size,
            GRAPH_RUNTIME,
            state_dict=state_dict,
            inputs=inputs,
            labels=labels,
            config=config,
            device=device,
        )
        exact = _parity_passes(parity)
        graph_memory = (
            _memory_for_backend(
                model_name,
                length,
                batch_size,
                GRAPH_RUNTIME,
                config=config,
                memory_dir=memory_dir,
                isolated_memory=isolated_memory,
            )
            if exact
            else _missing_memory("candidate failed accuracy; isolated memory skipped")
        )
        graph_row = _assemble_row(
            model_name,
            length,
            batch_size,
            GRAPH_RUNTIME,
            measurements=graph_measurements,
            setup_seconds=graph_setup,
            phase_backends=graph_backends,
            parameters=parameter_count,
            memory=graph_memory,
            parity=parity,
            status="measured" if exact else "rejected",
        )
        if not exact:
            graph_row["rejection_reason"] = "loss/gradient/parameter update parity exceeded 2e-5"
    except Exception as error:  # noqa: BLE001 - preserve unsupported-candidate evidence
        graph_row = {
            "model": model_name,
            "display_name": DISPLAY_NAMES[model_name],
            "length": length,
            "batch_size": batch_size,
            "runtime": GRAPH_RUNTIME,
            "status": "unsupported",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _release_cuda()

    exact_graph = graph_row.get("status") == "measured"
    graph_is_faster = exact_graph and _as_float(
        graph_row.get("full_train_step_wall_ms"), math.inf
    ) < _as_float(eager_row["full_train_step_wall_ms"])
    selected_source = graph_row if graph_is_faster else eager_row
    fastest_row = _selected_row(selected_source, eager=eager_row, graph=graph_row)
    del inputs, labels
    _release_cuda()
    return [eager_row, graph_row, fastest_row], architecture


def _measure_backend_phases(
    model_name: ModelName,
    length: int,
    batch_size: int,
    backend: BackendName,
    *,
    state_dict: dict[str, Tensor],
    inputs: Tensor,
    labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> tuple[dict[TrainingPhase, dict[str, object]], dict[str, float], dict[str, str]]:
    measurements: dict[TrainingPhase, dict[str, object]] = {}
    setup: dict[str, float] = {}
    phase_backends: dict[str, str] = {}
    for phase in PHASES:
        runtime_backend = backend
        try:
            runtime = _build_runtime(
                model_name,
                length,
                batch_size,
                backend,
                phase,
                state_dict=state_dict,
                inputs=inputs,
                labels=labels,
                config=config,
                device=device,
            )
        except Exception:
            if backend != GRAPH_RUNTIME or phase == "full":
                raise
            _release_cuda()
            runtime_backend = EAGER_RUNTIME
            runtime = _build_runtime(
                model_name,
                length,
                batch_size,
                EAGER_RUNTIME,
                phase,
                state_dict=state_dict,
                inputs=inputs,
                labels=labels,
                config=config,
                device=device,
            )
        measurements[phase] = _measure_phase(runtime, inputs, labels, config=config)
        setup[phase] = runtime.setup_seconds
        phase_backends[phase] = runtime_backend
        del runtime
        _release_cuda()
    return measurements, setup, phase_backends


def _build_runtime(
    model_name: ModelName,
    length: int,
    batch_size: int,
    backend: BackendName,
    phase: TrainingPhase,
    *,
    state_dict: dict[str, Tensor],
    inputs: Tensor,
    labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> _TrainingRuntime:
    started = perf_counter()
    model, _ = build_comparison_model(model_name, length, batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).train()
    if backend == EAGER_RUNTIME:
        runtime: _TrainingRuntime = _DeviceResidentEagerRuntime(
            model, phase, inputs, labels, config=config
        )
    elif phase != "full":
        if model_name in {"efp16", "pa2wp"}:
            message = f"{model_name} uses eager fallback for standalone {phase} phase"
            raise RuntimeError(message)
        runtime = _GenericCudaGraphRuntime(model, phase, inputs, labels, config=config)
    elif model_name == "efp16":
        runtime = _EFP16FullGraphRuntime(model, inputs, labels, config=config)
    elif model_name == "pa2wp":
        runtime = _PA2WPFullGraphRuntime(model, inputs, labels, config=config)
    else:
        runtime = _GenericCudaGraphRuntime(model, phase, inputs, labels, config=config)
    runtime.setup_seconds = perf_counter() - started
    return runtime


def _measure_phase(
    runtime: _TrainingRuntime,
    inputs: Tensor,
    labels: Tensor,
    *,
    config: BenchmarkConfig,
) -> dict[str, object]:
    loss = torch.zeros((), device=inputs.device)
    for _ in range(config.warmups):
        loss = runtime.step(inputs, labels)
    torch.cuda.synchronize(inputs.device)
    wall_samples: list[float] = []
    gpu_samples: list[float] = []
    for _ in range(config.groups):
        _precondition_gpu_clock(config.gpu_clock_precondition_cycles)
        torch.cuda.synchronize(inputs.device)
        start_wall = perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(config.iterations_per_group):
            loss = runtime.step(inputs, labels)
        end_event.record()
        end_event.synchronize()
        wall_samples.append((perf_counter() - start_wall) * 1000.0 / config.iterations_per_group)
        gpu_samples.append(start_event.elapsed_time(end_event) / config.iterations_per_group)
    wall_quartiles = statistics.quantiles(wall_samples, n=4, method="inclusive")
    gpu_quartiles = statistics.quantiles(gpu_samples, n=4, method="inclusive")
    return {
        "wall_ms": statistics.median(wall_samples),
        "wall_iqr_ms": wall_quartiles[2] - wall_quartiles[0],
        "wall_samples_ms": wall_samples,
        "gpu_ms": statistics.median(gpu_samples),
        "gpu_iqr_ms": gpu_quartiles[2] - gpu_quartiles[0],
        "gpu_samples_ms": gpu_samples,
        "last_loss": float(loss.detach().item()),
    }


def _measure_parity(
    model_name: ModelName,
    length: int,
    batch_size: int,
    *,
    state_dict: dict[str, Tensor],
    inputs: Tensor,
    labels: Tensor,
    config: BenchmarkConfig,
    device: str,
) -> dict[str, object]:
    eager = _build_runtime(
        model_name,
        length,
        batch_size,
        EAGER_RUNTIME,
        "full",
        state_dict=state_dict,
        inputs=inputs,
        labels=labels,
        config=config,
        device=device,
    )
    candidate = _build_runtime(
        model_name,
        length,
        batch_size,
        GRAPH_RUNTIME,
        "full",
        state_dict=state_dict,
        inputs=inputs,
        labels=labels,
        config=config,
        device=device,
    )
    eager_losses: list[Tensor] = []
    candidate_losses: list[Tensor] = []
    eager_snapshot: TrainingSnapshot | None = None
    candidate_snapshot: TrainingSnapshot | None = None
    for step in range(config.parity_steps):
        parity_seed = config.seed + 1_000_003 + step
        torch.cuda.manual_seed_all(parity_seed)
        eager.reset_rng_schedule()
        eager_loss = eager.step(inputs, labels)
        torch.cuda.synchronize(inputs.device)
        eager_losses.append(eager_loss.detach().clone())
        eager_snapshot = _snapshot_runtime(eager, eager_loss)

        torch.cuda.manual_seed_all(parity_seed)
        candidate.reset_rng_schedule()
        candidate_loss = candidate.step(inputs, labels)
        torch.cuda.synchronize(inputs.device)
        candidate_losses.append(candidate_loss.detach().clone())
        candidate_snapshot = _snapshot_runtime(candidate, candidate_loss)
    if eager_snapshot is None or candidate_snapshot is None:
        message = "parity requires at least one update"
        raise RuntimeError(message)
    result = compare_training_snapshots(eager_snapshot, candidate_snapshot)
    result.update(_final_prediction_metrics(eager, candidate, inputs))
    loss_trajectory_error = max(
        float((reference - actual).abs().item())
        for reference, actual in zip(eager_losses, candidate_losses, strict=True)
    )
    result["loss_trajectory_max_abs_error"] = loss_trajectory_error
    result["parity_steps"] = config.parity_steps
    del eager, candidate
    _release_cuda()
    return result


def compare_training_snapshots(
    reference: TrainingSnapshot,
    candidate: TrainingSnapshot,
) -> dict[str, object]:
    """Pure tensor comparison used by runtime parity and CPU unit tests."""
    gradient_keys = set(reference.gradients)
    candidate_gradient_keys = set(candidate.gradients)
    parameter_keys = set(reference.parameters)
    candidate_parameter_keys = set(candidate.parameters)
    gradient_agreement = gradient_keys == candidate_gradient_keys
    parameter_agreement = parameter_keys == candidate_parameter_keys
    gradient_abs, gradient_rel = _mapping_error(
        reference.gradients,
        candidate.gradients,
        gradient_keys.intersection(candidate_gradient_keys),
    )
    parameter_abs, parameter_rel = _mapping_error(
        reference.parameters,
        candidate.parameters,
        parameter_keys.intersection(candidate_parameter_keys),
    )
    return {
        "loss_abs_error": float((reference.loss - candidate.loss).abs().max().item()),
        "gradient_key_agreement": gradient_agreement,
        "gradient_tensor_count": len(gradient_keys),
        "gradient_max_abs_error": gradient_abs,
        "gradient_max_rel_error": gradient_rel,
        "parameter_key_agreement": parameter_agreement,
        "parameter_tensor_count": len(parameter_keys),
        "parameter_update_max_abs_error": parameter_abs,
        "parameter_update_max_rel_error": parameter_rel,
        "final_parameter_max_abs_error": parameter_abs,
    }


def _snapshot_runtime(runtime: _TrainingRuntime, loss: Tensor) -> TrainingSnapshot:
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in runtime.model.named_parameters()
        if parameter.grad is not None
    }
    parameters = {
        name: parameter.detach().clone() for name, parameter in runtime.model.named_parameters()
    }
    return TrainingSnapshot(loss.detach().clone(), gradients, parameters)


@torch.no_grad()
def _final_prediction_metrics(
    reference: _TrainingRuntime,
    candidate: _TrainingRuntime,
    inputs: Tensor,
) -> dict[str, object]:
    """Compare deterministic evaluation logits after the full parity trajectory."""
    reference.model.eval()
    candidate.model.eval()
    try:
        reference_logits = reference.model(inputs).detach()
        candidate_logits = candidate.model(inputs).detach()
        maximum_error = float((reference_logits - candidate_logits).abs().max().item())
        agreement = float(
            (reference_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1))
            .float()
            .mean()
            .item()
        )
    finally:
        reference.model.train()
        candidate.model.train()
    return {
        "final_logit_max_abs_error": maximum_error,
        "prediction_agreement": agreement,
    }


def _assemble_row(
    model_name: ModelName,
    length: int,
    batch_size: int,
    runtime: BackendName,
    *,
    measurements: dict[TrainingPhase, dict[str, object]],
    setup_seconds: dict[str, float],
    phase_backends: dict[str, str],
    parameters: int,
    memory: dict[str, object],
    parity: dict[str, object],
    status: str,
) -> dict[str, object]:
    full_wall = _as_float(measurements["full"]["wall_ms"])
    row: dict[str, object] = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "length": length,
        "batch_size": batch_size,
        "runtime": runtime,
        "status": status,
        "parameters": parameters,
        "phase_backends": phase_backends,
        "setup_seconds_by_phase": setup_seconds,
        "setup_seconds": sum(setup_seconds.values()),
        "compile_and_capture_cost_included": False,
        "forward_only": False,
        "sequences_per_second": batch_size * 1000.0 / full_wall,
        "tokens_per_second": batch_size * length * 1000.0 / full_wall,
        **parity,
        **memory,
    }
    for phase in PHASES:
        prefix = "full_train_step" if phase == "full" else phase
        measurement = measurements[phase]
        for key in (
            "wall_ms",
            "wall_iqr_ms",
            "wall_samples_ms",
            "gpu_ms",
            "gpu_iqr_ms",
            "gpu_samples_ms",
        ):
            row[f"{prefix}_{key}"] = measurement[key]
    row["last_timed_loss"] = measurements["full"]["last_loss"]
    return row


def _selected_row(
    source: dict[str, object],
    *,
    eager: dict[str, object],
    graph: dict[str, object],
) -> dict[str, object]:
    selected = copy.deepcopy(source)
    selected["runtime"] = FASTEST_RUNTIME
    selected["status"] = "measured"
    selected["selected_backend"] = source["runtime"]
    selected["candidate_status"] = graph.get("status")
    if graph.get("status") in {"unsupported", "rejected"}:
        selected["candidate_error"] = graph.get("error") or graph.get("rejection_reason")
    for phase in PHASES:
        prefix = "full_train_step" if phase == "full" else phase
        eager_latency = _as_float(eager[f"{prefix}_wall_ms"])
        selected_latency = _as_float(selected[f"{prefix}_wall_ms"])
        selected[f"{prefix}_speedup_vs_eager"] = eager_latency / selected_latency
    selected["speedup_vs_eager"] = selected["full_train_step_speedup_vs_eager"]
    return selected


def measure_isolated_peak_memory(
    model_name: ModelName,
    length: int,
    batch_size: int,
    backend: BackendName,
    *,
    config: BenchmarkConfig = DEFAULT_CONFIG,
    device: str = "cuda",
) -> dict[str, object]:
    """Measure one backend in a process intended to contain exactly one CUDA cell."""
    _validate_benchmark_arguments((model_name,), (length,), (batch_size,), config, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config.seed)
    base_model, _ = build_comparison_model(model_name, length, batch_size)
    state_dict = copy.deepcopy(base_model.state_dict())
    del base_model
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + 1009 * length + 9176 * batch_size
    )
    inputs = torch.randn(batch_size, length, 1, generator=generator).to(
        device=device, dtype=torch.float32
    )
    labels = torch.randint(0, 5, (batch_size,), generator=generator).to(
        device=device, dtype=torch.long
    )
    runtime = _build_runtime(
        model_name,
        length,
        batch_size,
        backend,
        "full",
        state_dict=state_dict,
        inputs=inputs,
        labels=labels,
        config=config,
        device=device,
    )
    runtime.step(inputs, labels)
    torch.cuda.synchronize(inputs.device)
    torch.cuda.reset_peak_memory_stats(inputs.device)
    runtime.step(inputs, labels)
    torch.cuda.synchronize(inputs.device)
    return {
        "schema": MEMORY_SCHEMA,
        "model": model_name,
        "length": length,
        "batch_size": batch_size,
        "backend": backend,
        "peak_memory_mb": torch.cuda.max_memory_allocated(inputs.device) / 2**20,
        "isolated_process": True,
        "config": asdict(config),
        "protocol": _protocol(isolated_memory=True),
        "environment": {
            "device": torch.cuda.get_device_name(inputs.device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }


def _memory_for_backend(
    model_name: ModelName,
    length: int,
    batch_size: int,
    backend: BackendName,
    *,
    config: BenchmarkConfig,
    memory_dir: Path | None,
    isolated_memory: bool,
) -> dict[str, object]:
    if not isolated_memory:
        return _missing_memory("isolated memory disabled for this smoke benchmark")
    target_dir = memory_dir
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if target_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="pac-model-training-memory-")
        target_dir = Path(temporary.name)
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / f"{model_name}-N{length}-B{batch_size}-{backend}.json"
    command = [
        sys.executable,
        "-m",
        "lnet.pac_model_training_comparison",
        "memory",
        "--model",
        model_name,
        "--length",
        str(length),
        "--batch-size",
        str(batch_size),
        "--backend",
        backend,
        "--warmups",
        str(config.warmups),
        "--groups",
        str(config.groups),
        "--iterations-per-group",
        str(config.iterations_per_group),
        "--graph-warmups",
        str(config.graph_warmups),
        "--parity-steps",
        str(config.parity_steps),
        "--seed",
        str(config.seed),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--grad-clip-norm",
        str(config.grad_clip_norm),
        "--gpu-clock-ramp-cycles",
        str(config.gpu_clock_ramp_cycles),
        "--gpu-clock-precondition-cycles",
        str(config.gpu_clock_precondition_cycles),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
    if completed.returncode != 0:
        if temporary is not None:
            temporary.cleanup()
        detail = completed.stderr.strip() or completed.stdout.strip()
        message = f"isolated memory subprocess failed: {detail}"
        raise RuntimeError(message)
    payload = cast("dict[str, object]", json.loads(output.read_text()))
    result = {
        "peak_memory_mb": payload["peak_memory_mb"],
        "memory_isolated_process": payload.get("isolated_process") is True,
        "memory_artifact": str(output) if temporary is None else None,
    }
    if temporary is not None:
        temporary.cleanup()
    return result


def evaluate_result(  # noqa: C901, PLR0912, PLR0915
    payload: dict[str, object],
) -> dict[str, object]:
    """Evaluate completeness, fairness, formulas, parity, and exact selection."""
    failures: list[str] = []
    if payload.get("schema") != SCHEMA:
        failures.append(f"unexpected schema: {payload.get('schema')}")
    environment = cast("dict[str, object]", payload.get("environment", {}))
    if "4090" not in str(environment.get("device", "")):
        failures.append("benchmark device is not an RTX 4090")
    if environment.get("allow_tf32") is not False:
        failures.append("environment must record TF32 disabled")
    protocol = cast("dict[str, object]", payload.get("protocol", {}))
    if protocol.get("dtype") != "float32":
        failures.append("protocol dtype must be float32")
    failures.extend(
        f"protocol {key} must be false"
        for key in ("tf32", "autocast")
        if protocol.get(key) is not False
    )
    if protocol.get("isolated_peak_memory") is not True:
        failures.append("final payload must use isolated peak-memory subprocesses")
    if "device-resident" not in str(protocol.get("input_policy", "")):
        failures.append("uniform device-resident input policy is missing")
    if protocol.get("compile_and_capture_cost_included") is not False:
        failures.append("setup/capture must be excluded from latency")
    if payload.get("models") != list(MODELS):
        failures.append("payload must contain the canonical ten-model order")
    if payload.get("lengths") != list(LENGTHS):
        failures.append("payload must contain the canonical three sequence lengths")
    if payload.get("batches") != list(BATCHES):
        failures.append("payload must contain the canonical two batch sizes")

    config = cast("dict[str, object]", payload.get("config", {}))
    groups = _as_int(config.get("groups"))
    architectures = cast("dict[str, dict[str, object]]", payload.get("architectures", {}))
    rows = cast("list[dict[str, object]]", payload.get("rows", []))
    indexed: dict[tuple[str, int, int, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("model")),
            _as_int(row.get("length")),
            _as_int(row.get("batch_size")),
            str(row.get("runtime")),
        )
        if key in indexed:
            failures.append(f"duplicate row: {key}")
        indexed[key] = row

    for model_name in MODELS:
        architecture = architectures.get(model_name)
        if architecture is None:
            failures.append(f"missing architecture: {model_name}")
            continue
        parameters = _as_int(architecture.get("trainable_parameters"))
        if parameters <= 0 or not str(architecture.get("state_dict_sha256", "")):
            failures.append(f"invalid architecture provenance: {model_name}")
        for length in LENGTHS:
            for batch_size in BATCHES:
                label = f"{model_name}/N{length}/B{batch_size}"
                eager = indexed.get((model_name, length, batch_size, EAGER_RUNTIME))
                graph = indexed.get((model_name, length, batch_size, GRAPH_RUNTIME))
                fastest = indexed.get((model_name, length, batch_size, FASTEST_RUNTIME))
                if eager is None or eager.get("status") != "measured":
                    failures.append(f"missing measured eager row: {label}")
                if graph is None:
                    failures.append(f"missing CUDA Graph evidence row: {label}")
                if fastest is None or fastest.get("status") != "measured":
                    failures.append(f"missing measured fastest row: {label}")
                if eager is None or fastest is None:
                    continue
                for runtime_row in (eager, fastest):
                    failures.extend(
                        _validate_measured_row(
                            runtime_row,
                            label=label,
                            parameters=parameters,
                            groups=groups,
                            length=length,
                            batch_size=batch_size,
                        )
                    )
                exact_graph = graph is not None and graph.get("status") == "measured"
                if graph is not None and graph.get("status") == "measured":
                    failures.extend(
                        _validate_measured_row(
                            graph,
                            label=f"{label}/graph",
                            parameters=parameters,
                            groups=groups,
                            length=length,
                            batch_size=batch_size,
                        )
                    )
                expected_backend = EAGER_RUNTIME
                graph_latency = _as_float(
                    graph.get("full_train_step_wall_ms") if graph is not None else None,
                    math.inf,
                )
                eager_latency = _as_float(eager.get("full_train_step_wall_ms"), math.inf)
                if exact_graph and graph_latency < eager_latency:
                    expected_backend = GRAPH_RUNTIME
                if fastest.get("selected_backend") != expected_backend:
                    failures.append(f"fastest selection mismatch: {label}")
                expected_source = graph if expected_backend == GRAPH_RUNTIME else eager
                if expected_source is None:
                    failures.append(f"missing selected source: {label}")
                    continue
                expected_latency = _as_float(expected_source.get("full_train_step_wall_ms"))
                if not math.isclose(
                    _as_float(fastest.get("full_train_step_wall_ms")),
                    expected_latency,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-12,
                ):
                    failures.append(f"fastest latency mismatch: {label}")

    return {
        "schema": EVALUATION_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_models": len(MODELS),
        "checked_shapes_per_model": len(LENGTHS) * len(BATCHES),
        "checked_required_rows": len(MODELS) * len(LENGTHS) * len(BATCHES) * 2,
    }


def _validate_measured_row(  # noqa: C901, PLR0912
    row: dict[str, object],
    *,
    label: str,
    parameters: int,
    groups: int,
    length: int,
    batch_size: int,
) -> list[str]:
    failures: list[str] = []
    if _as_int(row.get("parameters")) != parameters:
        failures.append(f"parameter count mismatch: {label}")
    if row.get("compile_and_capture_cost_included") is not False:
        failures.append(f"setup inclusion flag mismatch: {label}")
    if _as_float(row.get("setup_seconds"), -1.0) < 0.0:
        failures.append(f"invalid setup time: {label}")
    for phase in PHASES:
        prefix = "full_train_step" if phase == "full" else phase
        for clock in ("wall", "gpu"):
            latency = _as_float(row.get(f"{prefix}_{clock}_ms"), -1.0)
            samples = cast("list[object]", row.get(f"{prefix}_{clock}_samples_ms", []))
            if latency <= 0.0:
                failures.append(f"invalid {prefix} {clock} latency: {label}")
            if len(samples) != groups or any(_as_float(value, -1.0) <= 0.0 for value in samples):
                failures.append(f"invalid {prefix} {clock} samples: {label}")
    wall = _as_float(row.get("full_train_step_wall_ms"), math.inf)
    expected_sequences = batch_size * 1000.0 / wall
    expected_tokens = batch_size * length * 1000.0 / wall
    if not math.isclose(
        _as_float(row.get("sequences_per_second")), expected_sequences, rel_tol=1.0e-9
    ):
        failures.append(f"sequence throughput mismatch: {label}")
    if not math.isclose(_as_float(row.get("tokens_per_second")), expected_tokens, rel_tol=1.0e-9):
        failures.append(f"token throughput mismatch: {label}")
    if _as_float(row.get("peak_memory_mb"), -1.0) <= 0.0:
        failures.append(f"missing peak memory: {label}")
    if row.get("memory_isolated_process") is not True:
        failures.append(f"peak memory is not isolated: {label}")
    failures.extend(
        f"{metric} exceeded: {label}"
        for metric in (
            "loss_abs_error",
            "loss_trajectory_max_abs_error",
            "gradient_max_abs_error",
            "parameter_update_max_abs_error",
            "final_parameter_max_abs_error",
            "final_logit_max_abs_error",
        )
        if _as_float(row.get(metric), math.inf) > MAXIMUM_PARITY_ERROR
    )
    if row.get("gradient_key_agreement") is not True:
        failures.append(f"gradient keys disagree: {label}")
    if row.get("parameter_key_agreement") is not True:
        failures.append(f"parameter keys disagree: {label}")
    if _as_float(row.get("prediction_agreement"), -1.0) != 1.0:
        failures.append(f"predictions disagree: {label}")
    return failures


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for model_name in MODELS:
        selected = [
            row
            for row in rows
            if row.get("model") == model_name
            and row.get("runtime") == FASTEST_RUNTIME
            and row.get("status") == "measured"
        ]
        if not selected:
            continue
        summary[model_name] = {
            "display_name": DISPLAY_NAMES[model_name],
            "geometric_mean_full_step_wall_ms": _geometric_mean(
                [_as_float(row["full_train_step_wall_ms"]) for row in selected]
            ),
            "geometric_mean_speedup_vs_eager": _geometric_mean(
                [_as_float(row["speedup_vs_eager"]) for row in selected]
            ),
            "maximum_peak_memory_mb": max(
                _as_float(row.get("peak_memory_mb", 0.0)) for row in selected
            ),
            "selected_backend_counts": dict(
                Counter(str(row.get("selected_backend")) for row in selected)
            ),
        }
    return summary


def merge_payloads(payloads: Sequence[dict[str, object]]) -> dict[str, object]:
    if not payloads:
        message = "at least one benchmark payload is required"
        raise ValueError(message)
    first = payloads[0]
    shared_keys = ("schema", "environment", "protocol", "config", "lengths", "batches")
    present: set[str] = set()
    architectures: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for payload in payloads:
        for key in shared_keys:
            if payload.get(key) != first.get(key):
                message = f"cannot merge payloads with different {key}"
                raise ValueError(message)
        payload_models = {str(model) for model in cast("list[object]", payload.get("models", []))}
        duplicates = present.intersection(payload_models)
        if duplicates:
            message = f"duplicate model payloads: {sorted(duplicates)}"
            raise ValueError(message)
        present.update(payload_models)
        architectures.update(cast("dict[str, dict[str, object]]", payload.get("architectures", {})))
        rows.extend(cast("list[dict[str, object]]", payload.get("rows", [])))
    ordered_models = [model for model in MODELS if model in present]
    return {
        "schema": SCHEMA,
        "environment": first["environment"],
        "protocol": first["protocol"],
        "config": first["config"],
        "models": ordered_models,
        "lengths": first["lengths"],
        "batches": first["batches"],
        "architectures": architectures,
        "rows": rows,
        "summary": summarize(rows),
    }


def _protocol(*, isolated_memory: bool) -> dict[str, object]:
    return {
        "dtype": "float32",
        "tf32": False,
        "autocast": False,
        "compile_and_capture_cost_included": False,
        "input_policy": (
            "uniform device-resident caller inputs; every runtime includes one D2D copy into "
            "runtime-owned static input and label buffers per phase step"
        ),
        "forward": "input copy + training-mode model forward + cross-entropy",
        "forward_backward": (
            "input copy + zero_grad + training-mode forward + cross-entropy + backward"
        ),
        "full_train_step": (
            "input copy + zero_grad + forward + cross-entropy + backward + clip_grad_norm_ + "
            "AdamW + model post_optimizer_step"
        ),
        "timing": "CUDA events and synchronized wall time; median of per-group means",
        "setup_reporting": "model construction and CUDA Graph setup excluded and reported",
        "candidate_selection": (
            "minimum full-step wall latency between eager and a CUDA Graph candidate passing "
            "loss, gradient, and parameter-update absolute error <=2e-5"
        ),
        "maximum_absolute_error": MAXIMUM_PARITY_ERROR,
        "isolated_peak_memory": isolated_memory,
        "memory": "fresh Python/CUDA subprocess; one warm step then one measured full step",
        "pa2wp_phase_policy": "actual stochastic original-or-shifted single phase",
    }


def _capture_mutable_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Tensor], list[tuple[Tensor, Tensor]]]:
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer_tensors = [
        (value, value.detach().clone())
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    ]
    return parameters, optimizer_tensors


@torch.no_grad()
def _restore_mutable_state(
    model: nn.Module,
    _optimizer: torch.optim.Optimizer,
    snapshot: tuple[dict[str, Tensor], list[tuple[Tensor, Tensor]]],
) -> None:
    parameters, optimizer_tensors = snapshot
    for name, parameter in model.named_parameters():
        parameter.copy_(parameters[name])
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        else:
            parameter.grad.zero_()
    for destination, value in optimizer_tensors:
        destination.copy_(value)


@torch.no_grad()
def _materialize_capturable_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    parameters = tuple(model.parameters())
    values = tuple(parameter.detach().clone() for parameter in parameters)
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for parameter, value in zip(parameters, values, strict=True):
        parameter.copy_(value)
    for state in optimizer.state.values():
        for value in state.values():
            if not isinstance(value, Tensor):
                message = "capturable optimizer state must contain tensors only"
                raise TypeError(message)
            value.zero_()
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            message = "optimizer initialization lost a persistent gradient buffer"
            raise RuntimeError(message)
        gradient.zero_()


def _post_optimizer_step(model: nn.Module) -> None:
    post_step = getattr(model, "post_optimizer_step", None)
    if callable(post_step):
        post_step()


def _mapping_error(
    reference: dict[str, Tensor],
    candidate: dict[str, Tensor],
    keys: Iterable[str],
) -> tuple[float, float]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for key in keys:
        absolute = (reference[key] - candidate[key]).abs()
        relative = absolute / reference[key].abs().clamp_min(1.0e-6)
        maximum_absolute = max(maximum_absolute, float(absolute.max().item()))
        maximum_relative = max(maximum_relative, float(relative.max().item()))
    return maximum_absolute, maximum_relative


def _zero_parity(parity_steps: int) -> dict[str, object]:
    return {
        "loss_abs_error": 0.0,
        "loss_trajectory_max_abs_error": 0.0,
        "gradient_key_agreement": True,
        "gradient_tensor_count": 0,
        "gradient_max_abs_error": 0.0,
        "gradient_max_rel_error": 0.0,
        "parameter_key_agreement": True,
        "parameter_tensor_count": 0,
        "parameter_update_max_abs_error": 0.0,
        "parameter_update_max_rel_error": 0.0,
        "final_parameter_max_abs_error": 0.0,
        "final_logit_max_abs_error": 0.0,
        "prediction_agreement": 1.0,
        "parity_steps": parity_steps,
    }


def _parity_passes(parity: dict[str, object]) -> bool:
    return (
        parity.get("gradient_key_agreement") is True
        and parity.get("parameter_key_agreement") is True
        and _as_float(parity.get("prediction_agreement"), -1.0) == 1.0
        and all(
            _as_float(parity.get(metric), math.inf) <= MAXIMUM_PARITY_ERROR
            for metric in (
                "loss_abs_error",
                "loss_trajectory_max_abs_error",
                "gradient_max_abs_error",
                "parameter_update_max_abs_error",
                "final_logit_max_abs_error",
            )
        )
    )


def _missing_memory(reason: str) -> dict[str, object]:
    return {
        "peak_memory_mb": None,
        "memory_isolated_process": False,
        "memory_artifact": None,
        "memory_error": reason,
    }


def _architecture_identity(architecture: dict[str, object]) -> str:
    stable = {key: value for key, value in architecture.items() if key != "state_dict_sha256"}
    return json.dumps(stable, sort_keys=True)


def _state_dict_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_benchmark_arguments(
    models: tuple[ModelName, ...],
    lengths: tuple[int, ...],
    batches: tuple[int, ...],
    config: BenchmarkConfig,
    device: str,
) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        message = "10-model training comparison requires CUDA"
        raise RuntimeError(message)
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        message = f"unknown models: {unknown}"
        raise ValueError(message)
    if not models or not lengths or not batches:
        message = "models, lengths, and batches must be non-empty"
        raise ValueError(message)
    if min(lengths) < 2 or min(batches) < 1:
        message = "lengths must be >=2 and batches must be positive"
        raise ValueError(message)
    if (
        min(
            config.warmups,
            config.groups,
            config.iterations_per_group,
            config.graph_warmups,
            config.parity_steps,
        )
        < 1
    ):
        message = "warmups, groups, iterations, graph warmups, and parity steps must be positive"
        raise ValueError(message)


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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _geometric_mean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        converted = float(cast("float | int | str", value))
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _as_int(value: object) -> int:
    try:
        return int(cast("int | str", value))
    except (TypeError, ValueError):
        return 0


def _parse_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _parse_models(raw: str) -> tuple[ModelName, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    unknown = sorted(set(values) - set(MODELS))
    if unknown:
        message = f"unknown models: {unknown}"
        raise ValueError(message)
    return cast("tuple[ModelName, ...]", values)


def _config_from_args(arguments: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        warmups=arguments.warmups,
        groups=arguments.groups,
        iterations_per_group=arguments.iterations_per_group,
        graph_warmups=arguments.graph_warmups,
        parity_steps=arguments.parity_steps,
        seed=arguments.seed,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        grad_clip_norm=arguments.grad_clip_norm,
        gpu_clock_ramp_cycles=arguments.gpu_clock_ramp_cycles,
        gpu_clock_precondition_cycles=arguments.gpu_clock_precondition_cycles,
    )


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmups", type=int, default=DEFAULT_CONFIG.warmups)
    parser.add_argument("--groups", type=int, default=DEFAULT_CONFIG.groups)
    parser.add_argument(
        "--iterations-per-group", type=int, default=DEFAULT_CONFIG.iterations_per_group
    )
    parser.add_argument("--graph-warmups", type=int, default=DEFAULT_CONFIG.graph_warmups)
    parser.add_argument("--parity-steps", type=int, default=DEFAULT_CONFIG.parity_steps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=DEFAULT_CONFIG.grad_clip_norm)
    parser.add_argument(
        "--gpu-clock-ramp-cycles", type=int, default=DEFAULT_CONFIG.gpu_clock_ramp_cycles
    )
    parser.add_argument(
        "--gpu-clock-precondition-cycles",
        type=int,
        default=DEFAULT_CONFIG.gpu_clock_precondition_cycles,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--memory-dir", type=Path)
    benchmark_parser.add_argument("--no-isolated-memory", action="store_true")
    benchmark_parser.add_argument("--models", default=",".join(MODELS))
    benchmark_parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    benchmark_parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    _add_config_arguments(benchmark_parser)

    memory_parser = subparsers.add_parser("memory")
    memory_parser.add_argument("--output", type=Path, required=True)
    memory_parser.add_argument("--model", choices=MODELS, required=True)
    memory_parser.add_argument("--length", type=int, required=True)
    memory_parser.add_argument("--batch-size", type=int, required=True)
    memory_parser.add_argument("--backend", choices=(EAGER_RUNTIME, GRAPH_RUNTIME), required=True)
    _add_config_arguments(memory_parser)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    result: dict[str, object]
    if arguments.command == "benchmark":
        result = benchmark(
            models=_parse_models(arguments.models),
            lengths=_parse_tuple(arguments.lengths),
            batches=_parse_tuple(arguments.batches),
            config=_config_from_args(arguments),
            memory_dir=arguments.memory_dir,
            isolated_memory=not arguments.no_isolated_memory,
        )
        output = cast("Path", arguments.output)
    elif arguments.command == "memory":
        result = measure_isolated_peak_memory(
            cast("ModelName", arguments.model),
            arguments.length,
            arguments.batch_size,
            cast("BackendName", arguments.backend),
            config=_config_from_args(arguments),
        )
        output = cast("Path", arguments.output)
    elif arguments.command == "merge":
        payloads = [
            cast("dict[str, object]", json.loads(path.read_text())) for path in arguments.inputs
        ]
        result = merge_payloads(payloads)
        output = cast("Path", arguments.output)
    else:
        payload = cast("dict[str, object]", json.loads(arguments.input.read_text()))
        result = evaluate_result(payload)
        output = cast("Path", arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if arguments.command == "evaluate" and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
