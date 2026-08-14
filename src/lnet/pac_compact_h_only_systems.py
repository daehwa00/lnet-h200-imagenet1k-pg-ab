from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import platform
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_confirmatory_baselines import (
    ConfirmatoryFamily,
    build_confirmatory_family,
    confirmatory_implementation_metadata,
)
from .pac_efp_writer_reader import CompactEFPHOnlyTerminalPAC
from .pac_metrics import count_parameters
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


ModelName = Literal[
    "compact_h_only",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
]
InferenceRuntime = Literal[
    "eager_fp32",
    "manual_cuda_graph_fp32",
    "torch_compile_fp32",
    "best_exact_fp32",
]
TrainingRuntime = Literal[
    "eager_default_adamw_fp32",
    "eager_fused_adamw_fp32",
    "best_exact_train_step_fp32",
]

MODELS: Final[tuple[ModelName, ...]] = (
    "compact_h_only",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
DISPLAY_NAMES: Final[dict[ModelName, str]] = {
    "compact_h_only": "ALPHABET",
    "cnn1d": "CNN1D",
    "tcn": "TCN",
    "mamba": "Mamba",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
}
SELECTED_TRIALS: Final[dict[ModelName, int]] = {
    "compact_h_only": 0,
    "cnn1d": 3,
    "tcn": 6,
    "mamba": 6,
    "gru": 6,
    "lstm": 5,
    "transformer": 6,
}
REFERENCE_PARAMETERS: Final = 5_733
REFERENCE_DIMENSION: Final = 32
REFERENCE_MODES: Final = 16
EXPECTED_WIDTHS: Final[dict[ModelName, int | None]] = {
    "compact_h_only": None,
    "cnn1d": 13,
    "tcn": 17,
    "mamba": 17,
    "gru": 10,
    "lstm": 11,
    "transformer": 18,
}
EXPECTED_PARAMETER_COUNTS: Final[dict[ModelName, int]] = {
    "compact_h_only": 5_733,
    "cnn1d": 5_517,
    "tcn": 6_040,
    "mamba": 5_530,
    "gru": 5_421,
    "lstm": 6_076,
    "transformer": 5_711,
}
LENGTHS: Final = (128, 512, 2048)
BATCHES: Final = (1, 64)
INFERENCE_CANDIDATES: Final[tuple[InferenceRuntime, ...]] = (
    "manual_cuda_graph_fp32",
    "torch_compile_fp32",
)
MAX_INFERENCE_ABSOLUTE_ERROR: Final = 2.0e-5
MAX_TRAINING_ABSOLUTE_ERROR: Final = 2.0e-5


@dataclass(frozen=True, slots=True)
class SystemsConfig:
    inference_warmups: int = 20
    inference_groups: int = 9
    inference_iterations_per_group: int = 100
    training_warmups: int = 3
    training_groups: int = 5
    training_iterations_per_group: int = 10
    parity_steps: int = 1
    seed: int = 7
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0


DEFAULT_CONFIG: Final = SystemsConfig()


class _BorrowedCudaGraphInference(nn.Module):
    """Replay a static eager graph using a caller-owned input allocation."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()
        self.graph: torch.cuda.CUDAGraph | None = None
        self.input_data_ptr: int | None = None
        self.output: Tensor | None = None

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        if not inputs.is_cuda:
            message = "manual CUDA Graph inference requires CUDA inputs"
            raise ValueError(message)
        if self.graph is None:
            self.model(inputs)
            torch.cuda.synchronize(inputs.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.model(inputs)
            self.graph = graph
            self.input_data_ptr = inputs.data_ptr()
            self.output = output
        elif inputs.data_ptr() != self.input_data_ptr:
            message = "manual CUDA Graph inference requires the captured input allocation"
            raise ValueError(message)
        self.graph.replay()
        if self.output is None:
            message = "manual CUDA Graph did not capture an output buffer"
            raise RuntimeError(message)
        return self.output


class _StaticTransformerInference(nn.Module):
    """Hoist fixed-shape sinusoidal positions before compile or graph capture."""

    def __init__(self, model: nn.Module, *, length: int, device: torch.device) -> None:
        super().__init__()
        self.input_projection = cast("nn.Linear", model.get_submodule("input_projection"))
        self.encoder = model.get_submodule("encoder")
        self.classifier = cast("nn.Linear", model.get_submodule("classifier"))
        width = self.input_projection.out_features
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0, device=device)) / width)
        )
        encoding = torch.zeros(length, width, device=device)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if width > 1:
            encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("positions", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.input_projection(inputs)
        return self.classifier(self.encoder(projected + self.positions).mean(dim=1))


# Reusable public name for fixed-shape training/inference diagnostics.
StaticTransformerInference = _StaticTransformerInference


def build_systems_model(
    model_name: ModelName,
    *,
    length: int,
    batch_size: int,
) -> tuple[nn.Module, dict[str, object]]:
    """Build one real Q1-family architecture at the compact reference capacity."""
    config = _base_config(length, batch_size)
    if model_name == "compact_h_only":
        model = CompactEFPHOnlyTerminalPAC(config, 5, objective="classification")
        parameters = count_parameters(model)
        if parameters != REFERENCE_PARAMETERS:
            message = (
                "compact D32/M16 parameter drift: "
                f"expected {REFERENCE_PARAMETERS}, got {parameters}"
            )
            raise RuntimeError(message)
        return model, {
            "display_name": DISPLAY_NAMES[model_name],
            "family": model_name,
            "implementation": "repository-native compact H-only writer/read-only reader",
            "model_dim": REFERENCE_DIMENSION,
            "modes_per_scan": REFERENCE_MODES,
            "writer_scans": 1,
            "read_only_reader_scans": 1,
            "trainable_parameters": parameters,
            "target_parameters": REFERENCE_PARAMETERS,
            "relative_parameter_difference": 0.0,
            "capacity_contract": "D32/M16 scalar-input five-output reference",
            "state_dict_sha256": _state_dict_digest(model),
        }

    trial = SELECTED_TRIALS[model_name]
    model, width, parameters = _nearest_natural_width_baseline(
        cast("ConfirmatoryFamily", model_name),
        config,
        target_parameters=REFERENCE_PARAMETERS,
        validation_trial=trial,
    )
    metadata = confirmatory_implementation_metadata(cast("ConfirmatoryFamily", model_name), trial)
    return model, {
        "display_name": DISPLAY_NAMES[model_name],
        **metadata,
        "family": model_name,
        "selected_validation_trial": trial,
        "matched_width": width,
        "trainable_parameters": parameters,
        "target_parameters": REFERENCE_PARAMETERS,
        "relative_parameter_difference": abs(parameters - REFERENCE_PARAMETERS)
        / REFERENCE_PARAMETERS,
        "capacity_contract": (
            "real architecture at the nearest natural integer width; no inert parameter padding"
        ),
        "state_dict_sha256": _state_dict_digest(model),
    }


def _nearest_natural_width_baseline(
    family: ConfirmatoryFamily,
    config: PACExperimentConfig,
    *,
    target_parameters: int,
    validation_trial: int,
    max_width: int = 256,
) -> tuple[nn.Module, int, int]:
    """Find the nearest real integer width without an artificial budget adapter."""
    candidates: dict[int, int] = {}

    def build(width: int) -> nn.Module:
        return build_confirmatory_family(
            family,
            width,
            config,
            5,
            validation_trial=validation_trial,
        )

    def evaluate(width: int) -> int:
        parameters = count_parameters(build(width))
        candidates[width] = parameters
        return parameters

    lower = 1
    lower_parameters = evaluate(lower)
    if lower_parameters < target_parameters:
        upper = min(2, max_width)
        while upper > lower:
            upper_parameters = evaluate(upper)
            if upper_parameters >= target_parameters:
                while lower + 1 < upper:
                    middle = (lower + upper) // 2
                    middle_parameters = evaluate(middle)
                    if middle_parameters < target_parameters:
                        lower = middle
                    else:
                        upper = middle
                break
            if upper == max_width:
                break
            lower = upper
            upper = min(2 * upper, max_width)
    width, parameters = min(
        candidates.items(),
        key=lambda item: (abs(item[1] - target_parameters), item[0]),
    )
    return build(width), width, parameters


def benchmark(
    *,
    models: tuple[ModelName, ...] = MODELS,
    lengths: tuple[int, ...] = LENGTHS,
    batches: tuple[int, ...] = BATCHES,
    config: SystemsConfig = DEFAULT_CONFIG,
    device: str = "cuda",
    include_training: bool = True,
) -> dict[str, object]:
    _validate_arguments(models, lengths, batches, config, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    architectures: dict[str, dict[str, object]] = {}
    inference_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    for model_name in models:
        for length in lengths:
            for batch_size in batches:
                cell_inference, architecture, state_dict, cpu_inputs = _benchmark_inference_cell(
                    model_name,
                    length,
                    batch_size,
                    config=config,
                    device=device,
                )
                previous = architectures.setdefault(model_name, architecture)
                if _architecture_identity(previous) != _architecture_identity(architecture):
                    message = f"{model_name} architecture changed across static shapes"
                    raise RuntimeError(message)
                inference_rows.extend(cell_inference)
                if include_training:
                    training_rows.extend(
                        _benchmark_training_cell(
                            model_name,
                            length,
                            batch_size,
                            state_dict=state_dict,
                            cpu_inputs=cpu_inputs,
                            config=config,
                            device=device,
                        )
                    )

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema": "pac.compact_h_only.systems.v1",
        "environment": {
            "device": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allow_tf32": False,
        },
        "protocol": _protocol(include_training=include_training),
        "config": asdict(config),
        "models": list(models),
        "lengths": list(lengths),
        "batches": list(batches),
        "architectures": architectures,
        "inference_rows": inference_rows,
        "training_rows": training_rows,
        "summary": summarize(inference_rows, training_rows),
    }


def _protocol(*, include_training: bool) -> dict[str, object]:
    return {
        "scope": "seven final public Q1 families only",
        "excluded_families": ["EFP", "PA2WP", "MiniRocket", "S4D", "InceptionTime"],
        "static_shapes": {"lengths": list(LENGTHS), "batches": list(BATCHES)},
        "dtype": "float32",
        "autocast": False,
        "tf32": False,
        "compile_and_capture_excluded_from_timing": True,
        "inference_eager": "repository-native eval forward under torch.inference_mode",
        "inference_candidates": list(INFERENCE_CANDIDATES),
        "inference_selection": (
            "minimum CUDA-event latency among eager and candidates passing absolute-logit and "
            "prediction parity against eager"
        ),
        "compact_optimization_restriction": (
            "generic exact-FP32 graph/compile candidates only; no legacy EFP stem, fused "
            "synthesis, or recurrence-readout kernel is attributed to the read-only reader"
        ),
        "training_measured": include_training,
        "training_step": (
            "zero_grad, forward, cross_entropy, backward, clip_grad_norm_, AdamW.step, and "
            "post_optimizer_step when supplied by the model"
        ),
        "training_candidates": ["default AdamW", "CUDA fused AdamW"],
        "training_selection": (
            "minimum full-step wall latency among candidates passing loss, gradient-key, "
            "gradient-value, and parameter-update parity"
        ),
        "maximum_inference_absolute_error": MAX_INFERENCE_ABSOLUTE_ERROR,
        "maximum_training_absolute_error": MAX_TRAINING_ABSOLUTE_ERROR,
        "input_ownership": "one fixed caller-owned allocation per static cell",
        "capacity_contract": (
            "compact D32/M16 scalar-input five-output model (5733 parameters); each baseline "
            "uses the nearest natural integer width without inert padding"
        ),
    }


def _benchmark_inference_cell(
    model_name: ModelName,
    length: int,
    batch_size: int,
    *,
    config: SystemsConfig,
    device: str,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, Tensor], Tensor]:
    torch.manual_seed(config.seed)
    base_model, architecture = build_systems_model(model_name, length=length, batch_size=batch_size)
    state_dict = copy.deepcopy(base_model.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(config.seed + length + batch_size)
    cpu_inputs = torch.randn(batch_size, length, 1, generator=generator)
    del base_model

    eager, reference = _measure_inference_runtime(
        model_name,
        length,
        batch_size,
        "eager_fp32",
        state_dict=state_dict,
        cpu_inputs=cpu_inputs,
        reference=None,
        config=config,
        device=device,
    )
    rows = [eager]
    for runtime in INFERENCE_CANDIDATES:
        try:
            row, _ = _measure_inference_runtime(
                model_name,
                length,
                batch_size,
                runtime,
                state_dict=state_dict,
                cpu_inputs=cpu_inputs,
                reference=reference,
                config=config,
                device=device,
            )
        except Exception as error:  # noqa: BLE001 - candidate failures are audit evidence
            row = _failure_row(model_name, length, batch_size, runtime, error)
            _release_cuda()
        rows.append(row)
    best = select_best_exact_inference(rows)
    rows.append(_selected_row(best, "best_exact_fp32", eager))
    return rows, architecture, state_dict, cpu_inputs


def _measure_inference_runtime(
    model_name: ModelName,
    length: int,
    batch_size: int,
    runtime: InferenceRuntime,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    reference: Tensor | None,
    config: SystemsConfig,
    device: str,
) -> tuple[dict[str, object], Tensor]:
    model, _ = build_systems_model(model_name, length=length, batch_size=batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    inputs = cpu_inputs.to(device=device, dtype=torch.float32)
    compile_seconds = 0.0
    if runtime != "eager_fp32":
        started = perf_counter()
        model = _prepare_static_model(model_name, model, length=length)
        if runtime == "manual_cuda_graph_fp32":
            model = _BorrowedCudaGraphInference(model)
        elif runtime == "torch_compile_fp32":
            model = torch.compile(
                model,
                fullgraph=True,
                mode="max-autotune-no-cudagraphs",
            )
        else:
            message = f"unsupported inference runtime: {runtime}"
            raise ValueError(message)
        with torch.inference_mode():
            model(inputs)
        torch.cuda.synchronize()
        compile_seconds = perf_counter() - started

    with torch.inference_mode():
        output = model(inputs)
        for _ in range(config.inference_warmups):
            output = model(inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(config.inference_groups):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = perf_counter()
        start_event.record()
        with torch.inference_mode():
            for _ in range(config.inference_iterations_per_group):
                output = model(inputs)
        end_event.record()
        end_event.synchronize()
        wall_samples.append(
            (perf_counter() - started) * 1_000.0 / config.inference_iterations_per_group
        )
        gpu_samples.append(
            start_event.elapsed_time(end_event) / config.inference_iterations_per_group
        )
    output_cpu = output.detach().float().cpu()
    max_abs_error, max_rel_error, agreement = _output_parity(output_cpu, reference)
    latency = statistics.median(gpu_samples)
    row: dict[str, object] = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "length": length,
        "batch_size": batch_size,
        "runtime": runtime,
        "status": "measured",
        "latency_ms": latency,
        "latency_iqr_ms": _iqr(gpu_samples),
        "latency_samples_ms": gpu_samples,
        "wall_latency_ms": statistics.median(wall_samples),
        "compile_seconds": compile_seconds,
        "examples_per_second": batch_size * 1_000.0 / latency,
        "tokens_per_second": batch_size * length * 1_000.0 / latency,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "prediction_agreement": agreement,
    }
    del model, inputs
    _release_cuda()
    return row, output_cpu


def select_best_exact_inference(rows: list[dict[str, object]]) -> dict[str, object]:
    exact = [
        row
        for row in rows
        if row.get("status") == "measured"
        and _as_float(row.get("max_abs_error", math.inf)) <= MAX_INFERENCE_ABSOLUTE_ERROR
        and _as_float(row.get("prediction_agreement", 0.0)) == 1.0
    ]
    if not exact:
        message = "no exact-FP32 inference runtime survived parity filtering"
        raise RuntimeError(message)
    return min(exact, key=lambda row: _as_float(row["latency_ms"]))


def _benchmark_training_cell(
    model_name: ModelName,
    length: int,
    batch_size: int,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    config: SystemsConfig,
    device: str,
) -> list[dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 2 * length + batch_size)
    cpu_labels = torch.randint(0, 5, (batch_size,), generator=generator)
    reference = _training_parity_snapshot(
        model_name,
        length,
        batch_size,
        state_dict=state_dict,
        cpu_inputs=cpu_inputs,
        cpu_labels=cpu_labels,
        fused=False,
        config=config,
        device=device,
    )
    eager = _measure_training_runtime(
        model_name,
        length,
        batch_size,
        "eager_default_adamw_fp32",
        state_dict=state_dict,
        cpu_inputs=cpu_inputs,
        cpu_labels=cpu_labels,
        fused=False,
        parity_reference=reference,
        config=config,
        device=device,
    )
    rows = [eager]
    try:
        fused_reference = _training_parity_snapshot(
            model_name,
            length,
            batch_size,
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            fused=True,
            config=config,
            device=device,
        )
        parity = _compare_training_snapshots(reference, fused_reference)
        fused = _measure_training_runtime(
            model_name,
            length,
            batch_size,
            "eager_fused_adamw_fp32",
            state_dict=state_dict,
            cpu_inputs=cpu_inputs,
            cpu_labels=cpu_labels,
            fused=True,
            parity_reference=reference,
            config=config,
            device=device,
        )
        fused.update(parity)
    except Exception as error:  # noqa: BLE001 - preserve unsupported optimizer evidence
        fused = _failure_row(
            model_name,
            length,
            batch_size,
            "eager_fused_adamw_fp32",
            error,
        )
        _release_cuda()
    rows.append(fused)
    candidates = [eager]
    if _training_row_is_exact(fused):
        candidates.append(fused)
    best = min(candidates, key=lambda row: _as_float(row["full_train_step_wall_ms"]))
    rows.append(_selected_training_row(best, eager))
    return rows


def _training_parity_snapshot(
    model_name: ModelName,
    length: int,
    batch_size: int,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    fused: bool,
    config: SystemsConfig,
    device: str,
) -> dict[str, object]:
    model, _ = build_systems_model(model_name, length=length, batch_size=batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).train()
    inputs = cpu_inputs.to(device=device, dtype=torch.float32)
    labels = cpu_labels.to(device=device)
    optimizer = _optimizer(model, fused=fused, config=config)
    last_loss = torch.tensor(0.0, device=device)
    gradients: dict[str, Tensor] = {}
    for _ in range(config.parity_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        last_loss = functional.cross_entropy(logits, labels)
        last_loss.backward()
        gradients = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        _post_optimizer_step(model)
    snapshot: dict[str, object] = {
        "loss": float(last_loss.detach().item()),
        "gradients": gradients,
        "parameters": {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
        },
    }
    del model, optimizer, inputs, labels
    _release_cuda()
    return snapshot


def _compare_training_snapshots(
    reference: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    reference_gradients = cast("dict[str, Tensor]", reference["gradients"])
    candidate_gradients = cast("dict[str, Tensor]", candidate["gradients"])
    reference_parameters = cast("dict[str, Tensor]", reference["parameters"])
    candidate_parameters = cast("dict[str, Tensor]", candidate["parameters"])
    gradient_keys = set(reference_gradients) == set(candidate_gradients)
    parameter_keys = set(reference_parameters) == set(candidate_parameters)
    gradient_error = (
        max(
            (reference_gradients[key] - candidate_gradients[key]).abs().max().item()
            for key in reference_gradients
        )
        if gradient_keys and reference_gradients
        else math.inf
    )
    update_error = (
        max(
            (reference_parameters[key] - candidate_parameters[key]).abs().max().item()
            for key in reference_parameters
        )
        if parameter_keys and reference_parameters
        else math.inf
    )
    return {
        "loss_abs_error": abs(_as_float(reference["loss"]) - _as_float(candidate["loss"])),
        "gradient_key_agreement": gradient_keys,
        "gradient_max_abs_error": float(gradient_error),
        "parameter_key_agreement": parameter_keys,
        "parameter_update_max_abs_error": float(update_error),
    }


def _measure_training_runtime(
    model_name: ModelName,
    length: int,
    batch_size: int,
    runtime: TrainingRuntime,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    fused: bool,
    parity_reference: dict[str, object],
    config: SystemsConfig,
    device: str,
) -> dict[str, object]:
    model, _ = build_systems_model(model_name, length=length, batch_size=batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).train()
    inputs = cpu_inputs.to(device=device, dtype=torch.float32)
    labels = cpu_labels.to(device=device)
    optimizer = _optimizer(model, fused=fused, config=config)

    def step() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = functional.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        _post_optimizer_step(model)
        return loss

    for _ in range(config.training_warmups):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    loss = torch.tensor(0.0, device=device)
    for _ in range(config.training_groups):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = perf_counter()
        start_event.record()
        for _ in range(config.training_iterations_per_group):
            loss = step()
        end_event.record()
        end_event.synchronize()
        wall_samples.append(
            (perf_counter() - started) * 1_000.0 / config.training_iterations_per_group
        )
        gpu_samples.append(
            start_event.elapsed_time(end_event) / config.training_iterations_per_group
        )
    parity = (
        {
            "loss_abs_error": 0.0,
            "gradient_key_agreement": True,
            "gradient_max_abs_error": 0.0,
            "parameter_key_agreement": True,
            "parameter_update_max_abs_error": 0.0,
        }
        if not fused
        else _compare_training_snapshots(
            parity_reference,
            _training_parity_snapshot(
                model_name,
                length,
                batch_size,
                state_dict=state_dict,
                cpu_inputs=cpu_inputs,
                cpu_labels=cpu_labels,
                fused=True,
                config=config,
                device=device,
            ),
        )
    )
    wall_latency = statistics.median(wall_samples)
    row: dict[str, object] = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "length": length,
        "batch_size": batch_size,
        "runtime": runtime,
        "status": "measured",
        "full_train_step_wall_ms": wall_latency,
        "full_train_step_wall_iqr_ms": _iqr(wall_samples),
        "full_train_step_gpu_ms": statistics.median(gpu_samples),
        "full_train_step_gpu_iqr_ms": _iqr(gpu_samples),
        "steps_per_second": 1_000.0 / wall_latency,
        "examples_per_second": batch_size * 1_000.0 / wall_latency,
        "tokens_per_second": batch_size * length * 1_000.0 / wall_latency,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "last_measured_loss": float(loss.detach().item()),
        **parity,
    }
    _release_cuda()
    return row


def _optimizer(model: nn.Module, *, fused: bool, config: SystemsConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=fused,
    )


def _post_optimizer_step(model: nn.Module) -> None:
    callback = getattr(model, "post_optimizer_step", None)
    if callable(callback):
        cast("Callable[[], object]", callback)()


def _training_row_is_exact(row: dict[str, object]) -> bool:
    return (
        row.get("status") == "measured"
        and row.get("gradient_key_agreement") is True
        and row.get("parameter_key_agreement") is True
        and all(
            _as_float(row.get(metric, math.inf)) <= MAX_TRAINING_ABSOLUTE_ERROR
            for metric in (
                "loss_abs_error",
                "gradient_max_abs_error",
                "parameter_update_max_abs_error",
            )
        )
    )


def _selected_training_row(
    source: dict[str, object], eager: dict[str, object]
) -> dict[str, object]:
    row = copy.deepcopy(source)
    row["selected_backend"] = source["runtime"]
    row["runtime"] = "best_exact_train_step_fp32"
    row["speedup_vs_eager"] = _as_float(eager["full_train_step_wall_ms"]) / _as_float(
        source["full_train_step_wall_ms"]
    )
    return row


def _prepare_static_model(model_name: ModelName, model: nn.Module, *, length: int) -> nn.Module:
    if model_name == "transformer":
        return _StaticTransformerInference(
            model,
            length=length,
            device=next(model.parameters()).device,
        )
    return model


def _selected_row(
    source: dict[str, object], runtime: InferenceRuntime, eager: dict[str, object]
) -> dict[str, object]:
    row = copy.deepcopy(source)
    row["selected_backend"] = source["runtime"]
    row["runtime"] = runtime
    row["speedup_vs_eager"] = _as_float(eager["latency_ms"]) / _as_float(source["latency_ms"])
    return row


def summarize(
    inference_rows: list[dict[str, object]], training_rows: list[dict[str, object]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for model_name in MODELS:
        selected_inference = [
            row
            for row in inference_rows
            if row.get("model") == model_name
            and row.get("runtime") == "best_exact_fp32"
            and row.get("status") == "measured"
        ]
        selected_training = [
            row
            for row in training_rows
            if row.get("model") == model_name
            and row.get("runtime") == "best_exact_train_step_fp32"
            and row.get("status") == "measured"
        ]
        if not selected_inference:
            continue
        entry: dict[str, object] = {
            "display_name": DISPLAY_NAMES[model_name],
            "inference_geometric_mean_speedup_vs_eager": _geometric_mean(
                [_as_float(row["speedup_vs_eager"]) for row in selected_inference]
            ),
            "inference_selected_backend_counts": _counts(
                str(row["selected_backend"]) for row in selected_inference
            ),
        }
        if selected_training:
            entry.update(
                {
                    "training_geometric_mean_speedup_vs_eager": _geometric_mean(
                        [_as_float(row["speedup_vs_eager"]) for row in selected_training]
                    ),
                    "training_selected_backend_counts": _counts(
                        str(row["selected_backend"]) for row in selected_training
                    ),
                }
            )
        result[model_name] = entry
    return result


def evaluate_result(  # noqa: C901, PLR0912, PLR0915 - explicit audit checks stay local
    payload: dict[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    environment = cast("dict[str, object]", payload.get("environment", {}))
    if "4090" not in str(environment.get("device", "")):
        failures.append("benchmark device is not an RTX 4090")
    if payload.get("models") != list(MODELS):
        failures.append("public model family list is incomplete or reordered")
    if payload.get("lengths") != list(LENGTHS) or payload.get("batches") != list(BATCHES):
        failures.append("static shape grid is incomplete or reordered")
    protocol = cast("dict[str, object]", payload.get("protocol", {}))
    if protocol.get("dtype") != "float32" or protocol.get("tf32") is not False:
        failures.append("benchmark is not strict FP32 with TF32 disabled")
    architectures = cast("dict[str, dict[str, object]]", payload.get("architectures", {}))
    inference_rows = cast("list[dict[str, object]]", payload.get("inference_rows", []))
    training_rows = cast("list[dict[str, object]]", payload.get("training_rows", []))
    inference_index = _index_rows(inference_rows)
    training_index = _index_rows(training_rows)
    if len(inference_index) != len(inference_rows):
        failures.append("duplicate inference cells are present")
    if len(training_index) != len(training_rows):
        failures.append("duplicate training cells are present")
    for model_name in MODELS:
        architecture = architectures.get(model_name)
        if architecture is None:
            failures.append(f"missing architecture provenance for {model_name}")
        else:
            if architecture.get("trainable_parameters") != EXPECTED_PARAMETER_COUNTS[model_name]:
                failures.append(f"unexpected parameter count for {model_name}")
            expected_width = EXPECTED_WIDTHS[model_name]
            if expected_width is not None and architecture.get("matched_width") != expected_width:
                failures.append(f"unexpected nearest natural width for {model_name}")
        for length in LENGTHS:
            for batch_size in BATCHES:
                cell = f"{model_name}/N{length}/B{batch_size}"
                eager = inference_index.get((model_name, length, batch_size, "eager_fp32"))
                best = inference_index.get((model_name, length, batch_size, "best_exact_fp32"))
                inference_candidates = [
                    inference_index.get((model_name, length, batch_size, runtime))
                    for runtime in INFERENCE_CANDIDATES
                ]
                if any(row is None for row in inference_candidates):
                    failures.append(f"missing inference candidate evidence for {cell}")
                if eager is None or best is None:
                    failures.append(f"missing eager/best inference cell {cell}")
                elif not _inference_row_is_exact(best):
                    failures.append(f"non-exact selected inference cell {cell}")
                else:
                    measured_exact = [
                        row
                        for row in (eager, *inference_candidates)
                        if row is not None and _inference_row_is_exact(row)
                    ]
                    selected_backend = str(best.get("selected_backend", ""))
                    selected = inference_index.get(
                        (model_name, length, batch_size, selected_backend)
                    )
                    if selected is None or not _inference_row_is_exact(selected):
                        failures.append(
                            f"selected inference backend lacks exact evidence for {cell}"
                        )
                    elif not math.isclose(
                        _as_float(best["latency_ms"]),
                        min(_as_float(row["latency_ms"]) for row in measured_exact),
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-12,
                    ):
                        failures.append(
                            f"best inference selection is not minimum latency for {cell}"
                        )
                eager_train = training_index.get(
                    (model_name, length, batch_size, "eager_default_adamw_fp32")
                )
                fused_train = training_index.get(
                    (model_name, length, batch_size, "eager_fused_adamw_fp32")
                )
                best_train = training_index.get(
                    (model_name, length, batch_size, "best_exact_train_step_fp32")
                )
                if protocol.get("training_measured") is True:
                    if eager_train is None or fused_train is None or best_train is None:
                        failures.append(f"missing eager/best training cell {cell}")
                    elif not _training_row_is_exact(best_train):
                        failures.append(f"non-exact selected training cell {cell}")
                    else:
                        exact_training = [
                            row for row in (eager_train, fused_train) if _training_row_is_exact(row)
                        ]
                        selected_backend = str(best_train.get("selected_backend", ""))
                        selected = training_index.get(
                            (model_name, length, batch_size, selected_backend)
                        )
                        if selected is None or not _training_row_is_exact(selected):
                            failures.append(
                                f"selected training backend lacks exact evidence for {cell}"
                            )
                        elif not math.isclose(
                            _as_float(best_train["full_train_step_wall_ms"]),
                            min(
                                _as_float(row["full_train_step_wall_ms"]) for row in exact_training
                            ),
                            rel_tol=1.0e-9,
                            abs_tol=1.0e-12,
                        ):
                            failures.append(
                                f"best training selection is not minimum latency for {cell}"
                            )
    return {
        "schema": "pac.compact_h_only.systems.evaluation.v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_models": len(MODELS),
        "checked_shapes_per_model": len(LENGTHS) * len(BATCHES),
    }


def merge_payloads(payloads: list[dict[str, object]]) -> dict[str, object]:
    """Merge disjoint model shards measured under one identical RTX4090 contract."""
    if not payloads:
        message = "at least one systems benchmark payload is required"
        raise ValueError(message)
    first = payloads[0]
    invariant_keys = ("environment", "protocol", "config", "lengths", "batches")
    architectures: dict[str, dict[str, object]] = {}
    inference_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    present: set[str] = set()
    for payload in payloads:
        for key in invariant_keys:
            if payload.get(key) != first.get(key):
                message = f"cannot merge systems shards with different {key}"
                raise ValueError(message)
        shard_models = cast("list[str]", payload.get("models", []))
        duplicates = present.intersection(shard_models)
        if duplicates:
            message = f"duplicate systems model shards: {sorted(duplicates)}"
            raise ValueError(message)
        present.update(shard_models)
        architectures.update(cast("dict[str, dict[str, object]]", payload.get("architectures", {})))
        inference_rows.extend(cast("list[dict[str, object]]", payload.get("inference_rows", [])))
        training_rows.extend(cast("list[dict[str, object]]", payload.get("training_rows", [])))
    ordered_models = [model for model in MODELS if model in present]
    return {
        "schema": "pac.compact_h_only.systems.v1",
        **{key: copy.deepcopy(first.get(key)) for key in invariant_keys},
        "models": ordered_models,
        "architectures": architectures,
        "inference_rows": inference_rows,
        "training_rows": training_rows,
        "summary": summarize(inference_rows, training_rows),
    }


def smoke(
    *,
    models: tuple[ModelName, ...] = ("compact_h_only", "cnn1d"),
    device: str = "cpu",
    length: int = 16,
    batch_size: int = 2,
) -> dict[str, object]:
    """Cheap model-build, forward, backward, and parameter-contract smoke path."""
    if device == "cpu" and "mamba" in models:
        message = "the official Mamba implementation is CUDA-only; omit it from CPU smoke"
        raise ValueError(message)
    rows: list[dict[str, object]] = []
    for model_name in models:
        torch.manual_seed(7)
        model, architecture = build_systems_model(model_name, length=length, batch_size=batch_size)
        model = model.to(device=device, dtype=torch.float32).train()
        inputs = torch.randn(batch_size, length, 1, device=device)
        labels = torch.arange(batch_size, device=device) % 5
        logits = model(inputs)
        loss = functional.cross_entropy(logits, labels)
        loss.backward()
        finite = bool(
            torch.isfinite(logits).all()
            and torch.isfinite(loss)
            and all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )
        rows.append(
            {
                "model": model_name,
                "logit_shape": list(logits.shape),
                "trainable_parameters": architecture["trainable_parameters"],
                "finite_forward_backward": finite,
            }
        )
    return {
        "schema": "pac.compact_h_only.systems.smoke.v1",
        "device": device,
        "length": length,
        "batch_size": batch_size,
        "rows": rows,
        "status": "PASS" if all(row["finite_forward_backward"] for row in rows) else "FAIL",
    }


def _inference_row_is_exact(row: dict[str, object]) -> bool:
    return (
        row.get("status") == "measured"
        and _as_float(row.get("max_abs_error", math.inf)) <= MAX_INFERENCE_ABSOLUTE_ERROR
        and _as_float(row.get("prediction_agreement", 0.0)) == 1.0
    )


def _output_parity(output: Tensor, reference: Tensor | None) -> tuple[float, float, float]:
    if reference is None:
        return 0.0, 0.0, 1.0
    absolute = (output - reference).abs()
    relative = absolute / reference.abs().clamp_min(1.0e-6)
    agreement = (output.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean()
    return float(absolute.max()), float(relative.max()), float(agreement)


def _failure_row(
    model_name: ModelName,
    length: int,
    batch_size: int,
    runtime: str,
    error: Exception,
) -> dict[str, object]:
    return {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "length": length,
        "batch_size": batch_size,
        "runtime": runtime,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _base_config(length: int, batch_size: int) -> PACExperimentConfig:
    return PACExperimentConfig(
        sample_count=64,
        validation_count=16,
        test_count=16,
        sequence_length=length,
        raw_input_dim=1,
        output_dim=5,
        model_dim=REFERENCE_DIMENSION,
        modes=REFERENCE_MODES,
        epochs=1,
        batch_size=batch_size,
        device="cpu",
    )


def _state_dict_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _architecture_identity(architecture: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in architecture.items() if key != "state_dict_sha256"}


def _index_rows(
    rows: list[dict[str, object]],
) -> dict[tuple[str, int, int, str], dict[str, object]]:
    return {
        (
            str(row.get("model")),
            _as_int(row.get("length", 0)),
            _as_int(row.get("batch_size", 0)),
            str(row.get("runtime")),
        ): row
        for row in rows
    }


def _validate_arguments(
    models: tuple[ModelName, ...],
    lengths: tuple[int, ...],
    batches: tuple[int, ...],
    config: SystemsConfig,
    device: str,
) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        message = "the full systems benchmark requires CUDA"
        raise RuntimeError(message)
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        message = f"unknown public model families: {unknown}"
        raise ValueError(message)
    if not models or not lengths or not batches:
        message = "models, lengths, and batches must be non-empty"
        raise ValueError(message)
    if min(lengths) < 2 or min(batches) < 1:
        message = "lengths must be >=2 and batches must be positive"
        raise ValueError(message)
    repeats = (
        config.inference_warmups,
        config.inference_groups,
        config.inference_iterations_per_group,
        config.training_warmups,
        config.training_groups,
        config.training_iterations_per_group,
        config.parity_steps,
    )
    if min(repeats) < 1 or config.inference_groups < 2 or config.training_groups < 2:
        message = "warmups/iterations must be positive and group counts at least two"
        raise ValueError(message)


def _selected_models(raw: str) -> tuple[ModelName, ...]:
    return cast("tuple[ModelName, ...]", tuple(item for item in raw.split(",") if item))


def _integer_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(",") if item)


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


def _iqr(values: list[float]) -> float:
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _release_cuda() -> None:
    """Release dead CUDA allocations without invalidating a completed timing cell.

    Inductor's ``max-autotune-no-cudagraphs`` mode can briefly retain an
    internal allocator capture marker after its autotuning work has completed.
    Calling ``empty_cache`` in that state raises the PyTorch internal assertion
    ``captures_underway.empty()`` even though the measured forward has already
    finished and synchronized.  Garbage collection is still safe, and the
    allocator releases the cache at the next safe point.  Suppress only this
    exact cleanup-only assertion; every other CUDA error remains fail-closed.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as error:
            if "captures_underway.empty()" not in str(error):
                raise


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit compact ALPHABET and the six public Q1 baselines on static shapes"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--models", default=",".join(MODELS))
    benchmark_parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    benchmark_parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    benchmark_parser.add_argument("--inference-warmups", type=int, default=20)
    benchmark_parser.add_argument("--inference-groups", type=int, default=9)
    benchmark_parser.add_argument("--inference-iterations", type=int, default=100)
    benchmark_parser.add_argument("--training-warmups", type=int, default=3)
    benchmark_parser.add_argument("--training-groups", type=int, default=5)
    benchmark_parser.add_argument("--training-iterations", type=int, default=10)
    benchmark_parser.add_argument("--parity-steps", type=int, default=1)
    benchmark_parser.add_argument("--seed", type=int, default=7)
    benchmark_parser.add_argument("--skip-training", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--output", type=Path, required=True)
    smoke_parser.add_argument("--models", default="compact_h_only,cnn1d")
    smoke_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke_parser.add_argument("--length", type=int, default=16)
    smoke_parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        payload = benchmark(
            models=_selected_models(args.models),
            lengths=_integer_tuple(args.lengths),
            batches=_integer_tuple(args.batches),
            config=SystemsConfig(
                inference_warmups=args.inference_warmups,
                inference_groups=args.inference_groups,
                inference_iterations_per_group=args.inference_iterations,
                training_warmups=args.training_warmups,
                training_groups=args.training_groups,
                training_iterations_per_group=args.training_iterations,
                parity_steps=args.parity_steps,
                seed=args.seed,
            ),
            include_training=not args.skip_training,
        )
    elif args.command == "smoke":
        payload = smoke(
            models=_selected_models(args.models),
            device=args.device,
            length=args.length,
            batch_size=args.batch_size,
        )
    elif args.command == "merge":
        payload = merge_payloads(
            [
                cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
                for path in args.inputs
            ]
        )
    else:
        source = cast("dict[str, object]", json.loads(args.input.read_text(encoding="utf-8")))
        payload = evaluate_result(source)
    _write_json(args.output, payload)
    if payload.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
