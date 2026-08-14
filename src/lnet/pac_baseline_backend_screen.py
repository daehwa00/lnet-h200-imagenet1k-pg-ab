"""Exact-FP32 training-backend screen for the expensive PAC baselines.

This is an optimization diagnostic, not a paper-result runner.  It compares the
current fused-AdamW eager step with fixed-shape ``torch.compile`` and a captured
full training step while holding model initialization, input batches, optimizer
hyperparameters, and gradient clipping fixed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_compact_h_only_systems import StaticTransformerInference
from .pac_confirmatory_baselines import build_confirmatory_family
from .pac_metrics import count_parameters
from .pac_model_training_comparison import (
    BenchmarkConfig as GraphBenchmarkConfig,
)
from .pac_model_training_comparison import (
    GenericCudaGraphRuntime,
)
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

Family = Literal[
    "cnn1d",
    "tcn",
    "gru",
    "lstm",
    "mamba",
    "s4d",
    "lru",
    "s5",
    "transformer",
]
Backend = Literal["eager_fused", "compile_reduce_overhead", "cuda_graph_full_step"]

FAMILIES: Final[tuple[Family, ...]] = (
    "cnn1d",
    "tcn",
    "gru",
    "lstm",
    "mamba",
    "s4d",
    "lru",
    "s5",
    "transformer",
)
BACKENDS: Final[tuple[Backend, ...]] = (
    "eager_fused",
    "compile_reduce_overhead",
    "cuda_graph_full_step",
)
SELECTED_TRIAL: Final[dict[Family, int]] = {
    "cnn1d": 3,
    "tcn": 6,
    "gru": 6,
    "lstm": 5,
    "mamba": 6,
    "s4d": 5,
    "lru": 6,
    "s5": 6,
    "transformer": 6,
}
MAXIMUM_PARITY_ERROR: Final = 2.0e-5
AMORTIZATION_STEPS: Final = (100, 1_000, 10_000)


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    width: int = 64
    class_count: int = 10
    input_dim: int = 1
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    seed: int = 7
    parity_steps: int = 3
    warmups: int = 3
    groups: int = 5
    iterations_per_group: int = 10
    graph_warmups: int = 3


@dataclass(frozen=True, slots=True)
class _Snapshot:
    loss: Tensor
    gradients: dict[str, Tensor]
    parameters: dict[str, Tensor]


class _Runtime(Protocol):
    model: nn.Module
    backend: str
    setup_seconds: float

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor: ...


class _FusedEagerRuntime:
    backend = "eager_fused"

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Tensor,
        example_labels: Tensor,
        *,
        config: ScreenConfig,
    ) -> None:
        started = perf_counter()
        self.model = model.train()
        self.static_inputs = example_inputs.clone()
        self.static_labels = example_labels.clone()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
        )
        self.grad_clip_norm = config.grad_clip_norm
        self.setup_seconds = perf_counter() - started

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        self.static_inputs.copy_(inputs, non_blocking=True)
        self.static_labels.copy_(labels, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(self.model(self.static_inputs), self.static_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        return loss


class _CompiledRuntime:
    backend = "compile_reduce_overhead"

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Tensor,
        example_labels: Tensor,
        *,
        config: ScreenConfig,
    ) -> None:
        started = perf_counter()
        self.model = model.train()
        self.static_inputs = example_inputs.clone()
        self.static_labels = example_labels.clone()
        self.compiled_model = cast(
            "nn.Module",
            torch.compile(
                self.model,
                fullgraph=True,
                dynamic=False,
                mode="reduce-overhead",
            ),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
        )
        self.grad_clip_norm = config.grad_clip_norm
        self.setup_seconds = perf_counter() - started

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        self.static_inputs.copy_(inputs, non_blocking=True)
        self.static_labels.copy_(labels, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(
            self.compiled_model(self.static_inputs),
            self.static_labels,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        return loss


class _GraphRuntime:
    backend = "cuda_graph_full_step"

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Tensor,
        example_labels: Tensor,
        *,
        config: ScreenConfig,
    ) -> None:
        graph_config = _graph_config(config)
        runtime = GenericCudaGraphRuntime(
            model,
            "full",
            example_inputs,
            example_labels,
            config=graph_config,
        )
        self._runtime = runtime
        self.model = runtime.model
        self.setup_seconds = runtime.setup_seconds

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        return self._runtime.step(inputs, labels)


def build_screen_model(
    family: Family,
    *,
    length: int,
    batch_size: int,
    config: ScreenConfig,
    static_transformer_positions: bool,
) -> nn.Module:
    experiment = PACExperimentConfig(
        sample_count=batch_size,
        validation_count=batch_size,
        test_count=batch_size,
        sequence_length=length,
        raw_input_dim=config.input_dim,
        output_dim=config.class_count,
        model_dim=config.width,
        modes=16,
        epochs=1,
        batch_size=batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
        seeds=(config.seed,),
        device="cuda",
        optimizer_mode="fused",
    )
    model = build_confirmatory_family(
        family,
        config.width,
        experiment,
        config.class_count,
        validation_trial=SELECTED_TRIAL[family],
        input_dim=config.input_dim,
    )
    if family == "transformer" and static_transformer_positions:
        model = StaticTransformerInference(model, length=length, device=torch.device("cpu"))
    return model


def benchmark_cell(
    family: Family,
    *,
    length: int,
    batch_size: int,
    config: ScreenConfig,
    backends: tuple[Backend, ...] = BACKENDS,
    device: str = "cuda",
) -> list[dict[str, object]]:
    if "eager_fused" not in backends:
        message = "backend screen requires eager_fused as the reference"
        raise ValueError(message)
    _validate_cuda(device)
    torch.manual_seed(config.seed)
    base_model = build_screen_model(
        family,
        length=length,
        batch_size=batch_size,
        config=config,
        static_transformer_positions=False,
    )
    state_dict = copy.deepcopy(base_model.state_dict())
    parameters = count_parameters(base_model)
    del base_model

    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + 1009 * length + 9176 * batch_size
    )
    inputs = torch.randn(
        batch_size,
        length,
        config.input_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    labels = torch.randint(
        0,
        config.class_count,
        (batch_size,),
        generator=generator,
    ).to(device)

    rows: list[dict[str, object]] = []
    eager_parity, _, _ = _parity_snapshot(
        family,
        "eager_fused",
        state_dict,
        inputs,
        labels,
        length=length,
        batch_size=batch_size,
        config=config,
        device=device,
    )
    for backend in backends:
        try:
            parity = (
                _zero_parity(config.parity_steps)
                if backend == "eager_fused"
                else _compare_candidate(
                    family,
                    backend,
                    state_dict,
                    eager_parity,
                    inputs,
                    labels,
                    length=length,
                    batch_size=batch_size,
                    config=config,
                    device=device,
                )
            )
            exact = _parity_passes(parity)
            if backend != "eager_fused" and not exact:
                rows.append(
                    _base_row(
                        family,
                        backend,
                        length,
                        batch_size,
                        parameters,
                        status="rejected",
                        parity=parity,
                    )
                )
                continue
            timing = _measure_runtime(
                family,
                backend,
                state_dict,
                inputs,
                labels,
                length=length,
                batch_size=batch_size,
                config=config,
                device=device,
            )
            rows.append(
                {
                    **_base_row(
                        family,
                        backend,
                        length,
                        batch_size,
                        parameters,
                        status="measured",
                        parity=parity,
                    ),
                    **timing,
                }
            )
        except Exception as error:  # noqa: BLE001 - failed candidates are evidence
            rows.append(
                {
                    **_base_row(
                        family,
                        backend,
                        length,
                        batch_size,
                        parameters,
                        status="unsupported",
                        parity=None,
                    ),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        finally:
            _release_cuda()

    eager = next(row for row in rows if row["backend"] == "eager_fused")
    eager_ms = _as_float(eager.get("wall_ms", math.inf))
    for row in rows:
        latency = _as_float(row.get("wall_ms", math.inf))
        row["speedup_vs_eager"] = eager_ms / latency if math.isfinite(latency) else None
        _add_amortization(row, eager)
    del inputs, labels
    _release_cuda()
    return rows


def run_screen(
    *,
    families: tuple[Family, ...],
    lengths: tuple[int, ...],
    batch_size: int,
    config: ScreenConfig,
    output: Path,
    backends: tuple[Backend, ...] = BACKENDS,
    device: str = "cuda",
) -> dict[str, object]:
    _validate_cuda(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows: list[dict[str, object]] = []
    payload = _payload(families, lengths, batch_size, config, backends, rows)
    _write_json(output, payload)
    for family in families:
        for length in lengths:
            rows.extend(
                benchmark_cell(
                    family,
                    length=length,
                    batch_size=batch_size,
                    config=config,
                    backends=backends,
                    device=device,
                )
            )
            payload = _payload(families, lengths, batch_size, config, backends, rows)
            _write_json(output, payload)
    return payload


def _parity_snapshot(
    family: Family,
    backend: Backend,
    state_dict: dict[str, Tensor],
    inputs: Tensor,
    labels: Tensor,
    *,
    length: int,
    batch_size: int,
    config: ScreenConfig,
    device: str,
) -> tuple[list[_Snapshot], float, float]:
    runtime = _build_runtime(
        family,
        backend,
        state_dict,
        inputs,
        labels,
        length=length,
        batch_size=batch_size,
        config=config,
        device=device,
    )
    snapshots: list[_Snapshot] = []
    first_step_seconds = math.nan
    for step in range(config.parity_steps):
        torch.cuda.manual_seed_all(config.seed + 1_000_003 + step)
        started = perf_counter()
        loss = runtime.step(inputs, labels)
        torch.cuda.synchronize(inputs.device)
        if step == 0:
            first_step_seconds = perf_counter() - started
        snapshots.append(_snapshot(runtime, loss))
    setup_seconds = runtime.setup_seconds
    del runtime
    _release_cuda()
    return snapshots, setup_seconds, first_step_seconds


def _compare_candidate(
    family: Family,
    backend: Backend,
    state_dict: dict[str, Tensor],
    reference: list[_Snapshot],
    inputs: Tensor,
    labels: Tensor,
    *,
    length: int,
    batch_size: int,
    config: ScreenConfig,
    device: str,
) -> dict[str, object]:
    candidate, cold_setup_seconds, cold_first_step_seconds = _parity_snapshot(
        family,
        backend,
        state_dict,
        inputs,
        labels,
        length=length,
        batch_size=batch_size,
        config=config,
        device=device,
    )
    comparisons = [
        _compare_snapshots(expected, actual)
        for expected, actual in zip(reference, candidate, strict=True)
    ]
    return {
        "parity_steps": config.parity_steps,
        "cold_setup_seconds": cold_setup_seconds,
        "cold_first_step_seconds": cold_first_step_seconds,
        "loss_trajectory_max_abs_error": max(
            _as_float(item["loss_abs_error"]) for item in comparisons
        ),
        "gradient_key_agreement": all(
            item["gradient_key_agreement"] is True for item in comparisons
        ),
        "gradient_max_abs_error": max(
            _as_float(item["gradient_max_abs_error"]) for item in comparisons
        ),
        "parameter_key_agreement": all(
            item["parameter_key_agreement"] is True for item in comparisons
        ),
        "parameter_update_max_abs_error": max(
            _as_float(item["parameter_update_max_abs_error"]) for item in comparisons
        ),
    }


def _measure_runtime(
    family: Family,
    backend: Backend,
    state_dict: dict[str, Tensor],
    inputs: Tensor,
    labels: Tensor,
    *,
    length: int,
    batch_size: int,
    config: ScreenConfig,
    device: str,
) -> dict[str, object]:
    runtime = _build_runtime(
        family,
        backend,
        state_dict,
        inputs,
        labels,
        length=length,
        batch_size=batch_size,
        config=config,
        device=device,
    )
    first_started = perf_counter()
    loss = runtime.step(inputs, labels)
    torch.cuda.synchronize(inputs.device)
    first_step_seconds = perf_counter() - first_started
    for _ in range(max(0, config.warmups - 1)):
        loss = runtime.step(inputs, labels)
    torch.cuda.synchronize(inputs.device)
    torch.cuda.reset_peak_memory_stats(inputs.device)
    wall_samples: list[float] = []
    gpu_samples: list[float] = []
    for _ in range(config.groups):
        torch.cuda.synchronize(inputs.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = perf_counter()
        start_event.record()
        for _ in range(config.iterations_per_group):
            loss = runtime.step(inputs, labels)
        end_event.record()
        end_event.synchronize()
        wall_samples.append(
            (perf_counter() - started) * 1_000.0 / config.iterations_per_group
        )
        gpu_samples.append(
            start_event.elapsed_time(end_event) / config.iterations_per_group
        )
    result: dict[str, object] = {
        "setup_seconds": runtime.setup_seconds,
        "first_step_seconds": first_step_seconds,
        "wall_ms": statistics.median(wall_samples),
        "wall_iqr_ms": _iqr(wall_samples),
        "wall_samples_ms": wall_samples,
        "gpu_ms": statistics.median(gpu_samples),
        "gpu_iqr_ms": _iqr(gpu_samples),
        "gpu_samples_ms": gpu_samples,
        "peak_memory_mb": torch.cuda.max_memory_allocated(inputs.device) / 2**20,
        "last_loss": float(loss.detach().item()),
    }
    del runtime
    _release_cuda()
    return result


def _build_runtime(
    family: Family,
    backend: Backend,
    state_dict: dict[str, Tensor],
    inputs: Tensor,
    labels: Tensor,
    *,
    length: int,
    batch_size: int,
    config: ScreenConfig,
    device: str,
) -> _Runtime:
    model = build_screen_model(
        family,
        length=length,
        batch_size=batch_size,
        config=config,
        static_transformer_positions=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).train()
    if family == "transformer" and backend != "eager_fused":
        model = StaticTransformerInference(
            model,
            length=length,
            device=next(model.parameters()).device,
        ).train()
    if backend == "eager_fused":
        return _FusedEagerRuntime(model, inputs, labels, config=config)
    if backend == "compile_reduce_overhead":
        return _CompiledRuntime(model, inputs, labels, config=config)
    if backend == "cuda_graph_full_step":
        return _GraphRuntime(model, inputs, labels, config=config)
    raise AssertionError(backend)


def _snapshot(runtime: _Runtime, loss: Tensor) -> _Snapshot:
    return _Snapshot(
        loss.detach().clone(),
        {
            name: parameter.grad.detach().clone()
            for name, parameter in runtime.model.named_parameters()
            if parameter.grad is not None
        },
        {
            name: parameter.detach().clone()
            for name, parameter in runtime.model.named_parameters()
        },
    )


def _compare_snapshots(reference: _Snapshot, candidate: _Snapshot) -> dict[str, object]:
    gradient_keys = set(reference.gradients)
    parameter_keys = set(reference.parameters)
    candidate_gradient_keys = set(candidate.gradients)
    candidate_parameter_keys = set(candidate.parameters)
    return {
        "loss_abs_error": float((reference.loss - candidate.loss).abs().max().item()),
        "gradient_key_agreement": gradient_keys == candidate_gradient_keys,
        "gradient_max_abs_error": _mapping_error(
            reference.gradients,
            candidate.gradients,
            gradient_keys.intersection(candidate_gradient_keys),
        ),
        "parameter_key_agreement": parameter_keys == candidate_parameter_keys,
        "parameter_update_max_abs_error": _mapping_error(
            reference.parameters,
            candidate.parameters,
            parameter_keys.intersection(candidate_parameter_keys),
        ),
    }


def _mapping_error(
    reference: dict[str, Tensor],
    candidate: dict[str, Tensor],
    keys: set[str],
) -> float:
    return max(
        (
            float((reference[key] - candidate[key]).abs().max().item())
            for key in keys
        ),
        default=math.inf,
    )


def _zero_parity(parity_steps: int) -> dict[str, object]:
    return {
        "parity_steps": parity_steps,
        "loss_trajectory_max_abs_error": 0.0,
        "gradient_key_agreement": True,
        "gradient_max_abs_error": 0.0,
        "parameter_key_agreement": True,
        "parameter_update_max_abs_error": 0.0,
    }


def _parity_passes(parity: dict[str, object]) -> bool:
    return (
        parity.get("gradient_key_agreement") is True
        and parity.get("parameter_key_agreement") is True
        and all(
            _as_float(parity.get(metric, math.inf)) <= MAXIMUM_PARITY_ERROR
            for metric in (
                "loss_trajectory_max_abs_error",
                "gradient_max_abs_error",
                "parameter_update_max_abs_error",
            )
        )
    )


def _base_row(
    family: Family,
    backend: Backend,
    length: int,
    batch_size: int,
    parameters: int,
    *,
    status: str,
    parity: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "family": family,
        "backend": backend,
        "length": length,
        "batch_size": batch_size,
        "parameters": parameters,
        "status": status,
        **({} if parity is None else parity),
    }


def _payload(
    families: tuple[Family, ...],
    lengths: tuple[int, ...],
    batch_size: int,
    config: ScreenConfig,
    backends: tuple[Backend, ...],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    measured = [row for row in rows if row.get("status") == "measured"]
    winners: dict[str, dict[str, object]] = {}
    amortized_winners: dict[str, dict[str, dict[str, object]]] = {
        str(steps): {} for steps in AMORTIZATION_STEPS
    }
    for family in families:
        for length in lengths:
            cell = [
                row
                for row in measured
                if row["family"] == family and row["length"] == length
            ]
            if cell:
                winner = min(cell, key=lambda row: _as_float(row["wall_ms"]))
                winners[f"{family}:T{length}"] = {
                    "backend": winner["backend"],
                    "wall_ms": winner["wall_ms"],
                    "speedup_vs_eager": winner["speedup_vs_eager"],
                }
                for steps in AMORTIZATION_STEPS:
                    total_key = f"estimated_total_seconds_{steps}_steps"
                    amortized = min(cell, key=lambda row: _as_float(row[total_key]))
                    amortized_winners[str(steps)][f"{family}:T{length}"] = {
                        "backend": amortized["backend"],
                        "estimated_total_seconds": amortized[total_key],
                    }
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "schema": "pac.baseline_backend_screen.v1",
        "environment": {
            "host": platform.node(),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_index": device_index,
            "device": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allow_tf32": False,
        },
        "protocol": {
            "purpose": "optimization screening only; not paper accuracy evidence",
            "precision": "fp32",
            "optimizer": "fused AdamW",
            "compile_mode": "reduce-overhead/fullgraph/static",
            "cuda_graph_scope": "input copy plus forward, loss, backward, clipping, optimizer",
            "parity_threshold": MAXIMUM_PARITY_ERROR,
            "static_transformer_positions_on_candidates": True,
            "compile_and_capture_excluded_from_steady_state_timing": True,
            "amortization_steps": list(AMORTIZATION_STEPS),
            "cold_candidate_cost_measured_before_runtime_cache_reuse": True,
        },
        "families": list(families),
        "backends": list(backends),
        "lengths": list(lengths),
        "batch_size": batch_size,
        "config": asdict(config),
        "rows": rows,
        "winners": winners,
        "amortized_winners": amortized_winners,
    }


def _graph_config(config: ScreenConfig) -> GraphBenchmarkConfig:
    return GraphBenchmarkConfig(
        graph_warmups=config.graph_warmups,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )


def _validate_cuda(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        message = "baseline backend screen requires an available CUDA device"
        raise ValueError(message)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return math.inf


def _add_amortization(row: dict[str, object], eager: dict[str, object]) -> None:
    if row.get("status") != "measured":
        row["break_even_steps_vs_eager"] = None
        return
    latency_seconds = _as_float(row["wall_ms"]) / 1_000.0
    eager_latency_seconds = _as_float(eager["wall_ms"]) / 1_000.0
    cold_setup = _as_float(row.get("cold_setup_seconds", row.get("setup_seconds")))
    cold_first = _as_float(
        row.get("cold_first_step_seconds", row.get("first_step_seconds"))
    )
    eager_cold_setup = _as_float(eager.get("setup_seconds"))
    eager_cold_first = _as_float(eager.get("first_step_seconds"))
    for steps in AMORTIZATION_STEPS:
        row[f"estimated_total_seconds_{steps}_steps"] = (
            cold_setup + cold_first + (steps - 1) * latency_seconds
        )
    saving = eager_latency_seconds - latency_seconds
    if row is eager or saving <= 0.0:
        row["break_even_steps_vs_eager"] = 1 if row is eager else None
        return
    extra_cold_seconds = (cold_setup + cold_first) - (
        eager_cold_setup + eager_cold_first
    )
    row["break_even_steps_vs_eager"] = max(
        1,
        1 + math.ceil(max(0.0, extra_cold_seconds) / saving),
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _csv_tuple(value: str, *, allowed: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        message = f"expected subset of {allowed}, got {value!r}"
        raise argparse.ArgumentTypeError(message)
    return values


def _positive_int_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        message = "expected comma-separated integers"
        raise argparse.ArgumentTypeError(message) from error
    if not values or any(item <= 0 for item in values):
        message = "all values must be positive"
        raise argparse.ArgumentTypeError(message)
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--lengths", default="128")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--class-count", type=int, default=10)
    parser.add_argument("--input-dim", type=int, default=1)
    parser.add_argument("--parity-steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--iterations-per-group", type=int, default=10)
    parser.add_argument("--graph-warmups", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    families = cast(
        "tuple[Family, ...]",
        _csv_tuple(args.families, allowed=cast("tuple[str, ...]", FAMILIES)),
    )
    lengths = _positive_int_csv(args.lengths)
    backends = cast(
        "tuple[Backend, ...]",
        _csv_tuple(args.backends, allowed=cast("tuple[str, ...]", BACKENDS)),
    )
    config = ScreenConfig(
        width=args.width,
        class_count=args.class_count,
        input_dim=args.input_dim,
        parity_steps=args.parity_steps,
        warmups=args.warmups,
        groups=args.groups,
        iterations_per_group=args.iterations_per_group,
        graph_warmups=args.graph_warmups,
    )
    payload = run_screen(
        families=families,
        lengths=lengths,
        batch_size=args.batch_size,
        config=config,
        output=args.output,
        backends=backends,
    )
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
