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
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, cast

try:
    from typing import assert_never as _assert_never
except ImportError:
    from typing_extensions import assert_never as _assert_never  # noqa: UP035

typing.assert_never = _assert_never  # type: ignore[attr-defined]

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_confirmatory_baselines import (
    ConfirmatoryFamily,
    confirmatory_implementation_metadata,
)
from .pac_efp16_final_campaign import match_ucr_baseline
from .pac_external_reference_baselines import ExternalMiniRocketClassifier
from .pac_headroom_efficient_models import build_efficient_headroom_classifier
from .pac_metrics import count_parameters
from .pac_pa2wp_runtime import prepare_pa2wp_persistent_core_inference
from .pac_tight_frame_runtime import (
    BorrowedInputCudaGraphInference,
    prepare_efp16_ceiling_inference,
)
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .pac_headroom_models import HeadroomPACClassifier

ModelName = Literal[
    "efp16",
    "gru",
    "minirocket",
    "cnn1d",
    "lstm",
    "mamba",
    "transformer",
    "tcn",
    "pa2wp",
    "s4d",
]
RuntimeName = Literal[
    "eager_fp32",
    "manual_graph_fp32",
    "compiled_graph_fp32",
    "specialized_fp32",
    "optimized_fp32",
    "fastest_exact_fp32",
]

MODELS: Final[tuple[ModelName, ...]] = (
    "efp16",
    "gru",
    "minirocket",
    "cnn1d",
    "lstm",
    "mamba",
    "transformer",
    "tcn",
    "pa2wp",
    "s4d",
)
LENGTHS: Final = (128, 512, 2048)
BATCHES: Final = (1, 64)
CANDIDATE_RUNTIMES: Final[tuple[RuntimeName, ...]] = (
    "manual_graph_fp32",
    "compiled_graph_fp32",
    "specialized_fp32",
)
SELECTED_RUNTIMES: Final[tuple[RuntimeName, ...]] = (
    "eager_fp32",
    "optimized_fp32",
    "fastest_exact_fp32",
)
DISPLAY_NAMES: Final[dict[ModelName, str]] = {
    "efp16": "EFP16",
    "gru": "GRU",
    "minirocket": "MiniRocket",
    "cnn1d": "CNN1D",
    "lstm": "LSTM",
    "mamba": "Mamba",
    "transformer": "Transformer",
    "tcn": "TCN",
    "pa2wp": "PA2WP",
    "s4d": "S4D",
}
SELECTED_TRIALS: Final[dict[ModelName, int]] = {
    "gru": 6,
    "minirocket": 5,
    "cnn1d": 3,
    "lstm": 5,
    "mamba": 6,
    "transformer": 6,
    "tcn": 6,
    "s4d": 5,
    "efp16": 0,
    "pa2wp": 0,
}
BASELINE_SELECTION: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
MINIROCKET_SELECTION: Final = Path(
    ".omx/results/pac-ucr-s4-minirocket-20260712/reports/selection.json"
)
MAX_ABSOLUTE_ERROR: Final = 2.0e-5


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    warmups: int = 20
    groups: int = 9
    iterations_per_group: int = 100
    seed: int = 7


DEFAULT_BENCHMARK_CONFIG: Final = BenchmarkConfig()


class BorrowedEagerCudaGraphInference(nn.Module):
    """Capture eager inference against a caller-owned input allocation."""

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
            graph.replay()
            return output
        if inputs.data_ptr() != self.input_data_ptr:
            message = "manual CUDA Graph inference requires the captured input allocation"
            raise ValueError(message)
        self.graph.replay()
        if self.output is None:
            message = "manual CUDA Graph output buffer was not captured"
            raise RuntimeError(message)
        return self.output


class StaticMiniRocketInference(nn.Module):
    """Hoist deterministic dilation groups so MiniRocket becomes graph-safe."""

    _DILATIONS: Final = (1, 2, 4, 8)
    _KERNEL_SIZE: Final = 9

    def __init__(self, model: ExternalMiniRocketClassifier) -> None:
        super().__init__()
        self.register_buffer("kernels", model.get_buffer("kernels"), persistent=False)
        self.register_buffer("bias", model.get_buffer("bias"), persistent=False)
        dilations = model.get_buffer("dilations")
        for dilation in self._DILATIONS:
            self.register_buffer(
                f"indices_{dilation}",
                torch.nonzero(dilations == dilation, as_tuple=False).squeeze(-1),
                persistent=False,
            )
        self.head = model.head

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs.transpose(1, 2)
        kernels = self.get_buffer("kernels")
        bias = self.get_buffer("bias")
        features = torch.empty(
            inputs.shape[0],
            kernels.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for dilation in self._DILATIONS:
            indices = self.get_buffer(f"indices_{dilation}")
            padding = dilation * (self._KERNEL_SIZE - 1) // 2
            response = functional.conv1d(
                values,
                kernels[indices].to(dtype=values.dtype),
                dilation=dilation,
                padding=padding,
            )
            threshold = bias[indices].to(dtype=response.dtype).view(1, -1, 1)
            features[:, indices] = (response > threshold).to(response.dtype).mean(dim=-1)
        return self.head(features)


class StaticTransformerInference(nn.Module):
    """Hoist fixed-shape sinusoidal positions before CUDA Graph capture."""

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


def benchmark(
    *,
    models: tuple[ModelName, ...] = MODELS,
    lengths: tuple[int, ...] = LENGTHS,
    batches: tuple[int, ...] = BATCHES,
    config: BenchmarkConfig = DEFAULT_BENCHMARK_CONFIG,
    device: str = "cuda",
) -> dict[str, object]:
    _validate_benchmark_arguments(models, lengths, batches, config, device)
    _validate_selection_provenance(models)
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
                    message = f"{model_name} architecture changed across shapes"
                    raise RuntimeError(message)
                rows.extend(cell_rows)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema": "pac_model_speed_comparison.v1",
        "environment": {
            "device": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allow_tf32": False,
        },
        "protocol": {
            "dtype": "float32",
            "timing": "CUDA events; median of per-group means",
            "compile_and_capture_excluded": True,
            "input_ownership": "one fixed caller-owned allocation per model/shape/runtime",
            "output_ownership": "borrowed until next replay for graph runtimes",
            "candidate_selection": "minimum measured exact-FP32 latency per cell",
            "maximum_absolute_error": MAX_ABSOLUTE_ERROR,
            "minimum_prediction_agreement": 1.0,
            "capacity_contract": (
                "accuracy-selected baseline architecture with width matched to EFP16 D32/M16; "
                "final PA2WP D64/M16 retained as the optimized paper model"
            ),
        },
        "config": asdict(config),
        "models": list(models),
        "lengths": list(lengths),
        "batches": list(batches),
        "architectures": architectures,
        "rows": rows,
        "summary": summarize(payload_rows=rows),
    }


def summarize(*, payload_rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for model_name in MODELS:
        selected = [
            row
            for row in payload_rows
            if row.get("model") == model_name
            and row.get("runtime") == "fastest_exact_fp32"
            and row.get("status") == "measured"
        ]
        if not selected:
            continue
        speedups = [float(cast("float", row["speedup_vs_eager"])) for row in selected]
        summary[model_name] = {
            "display_name": DISPLAY_NAMES[model_name],
            "geometric_mean_speedup_vs_eager": _geometric_mean(speedups),
            "selected_backend_counts": _counts(str(row["selected_backend"]) for row in selected),
        }
    return summary


def evaluate_result(  # noqa: C901, PLR0912 - explicit contract checks stay auditable
    payload: dict[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    environment = cast("dict[str, object]", payload.get("environment", {}))
    if "4090" not in str(environment.get("device", "")):
        failures.append("benchmark device is not an RTX 4090")
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
        elif _as_int(architecture.get("trainable_parameters", 0)) <= 0:
            failures.append(f"invalid parameter count for {model_name}")
        for length in LENGTHS:
            for batch_size in BATCHES:
                eager = indexed.get((model_name, length, batch_size, "eager_fp32"))
                optimized = indexed.get((model_name, length, batch_size, "optimized_fp32"))
                fastest = indexed.get((model_name, length, batch_size, "fastest_exact_fp32"))
                for runtime, row in (
                    ("eager_fp32", eager),
                    ("optimized_fp32", optimized),
                    ("fastest_exact_fp32", fastest),
                ):
                    if row is None or row.get("status") != "measured":
                        failures.append(
                            f"missing measured {model_name}/N{length}/B{batch_size}/{runtime}"
                        )
                if eager is None or optimized is None or fastest is None:
                    continue
                selected_backend = str(optimized.get("selected_backend", ""))
                selected = indexed.get((model_name, length, batch_size, selected_backend))
                if selected is None or selected.get("status") != "measured":
                    failures.append(
                        f"unmeasured selected backend for {model_name}/N{length}/B{batch_size}"
                    )
                expected = min(
                    float(cast("float", eager["latency_ms"])),
                    float(cast("float", optimized["latency_ms"])),
                )
                actual = float(cast("float", fastest["latency_ms"]))
                if not math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
                    failures.append(
                        f"fastest selection mismatch for {model_name}/N{length}/B{batch_size}"
                    )
                for row in (optimized, fastest):
                    if _as_float(row.get("max_abs_error", math.inf)) > MAX_ABSOLUTE_ERROR:
                        failures.append(
                            f"FP32 error exceeded for {model_name}/N{length}/B{batch_size}"
                        )
                    if _as_float(row.get("prediction_agreement", 0.0)) < 1.0:
                        failures.append(
                            f"prediction disagreement for {model_name}/N{length}/B{batch_size}"
                        )
    return {
        "schema": "pac_model_speed_comparison_evaluation.v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_models": len(MODELS),
        "checked_shapes_per_model": len(LENGTHS) * len(BATCHES),
    }


def merge_payloads(payloads: list[dict[str, object]]) -> dict[str, object]:
    if not payloads:
        message = "at least one benchmark payload is required"
        raise ValueError(message)
    first = payloads[0]
    environment = first.get("environment")
    protocol = first.get("protocol")
    config = first.get("config")
    lengths = first.get("lengths")
    batches = first.get("batches")
    architectures: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    present_models: set[str] = set()
    for payload in payloads:
        for key, expected in (
            ("environment", environment),
            ("protocol", protocol),
            ("config", config),
            ("lengths", lengths),
            ("batches", batches),
        ):
            if payload.get(key) != expected:
                message = f"cannot merge payloads with different {key}"
                raise ValueError(message)
        payload_models = cast("list[str]", payload.get("models", []))
        duplicates = present_models.intersection(payload_models)
        if duplicates:
            message = f"duplicate model payloads: {sorted(duplicates)}"
            raise ValueError(message)
        present_models.update(payload_models)
        architectures.update(cast("dict[str, dict[str, object]]", payload.get("architectures", {})))
        rows.extend(cast("list[dict[str, object]]", payload.get("rows", [])))
    ordered_models = [model_name for model_name in MODELS if model_name in present_models]
    return {
        "schema": "pac_model_speed_comparison.v1",
        "environment": environment,
        "protocol": protocol,
        "config": config,
        "models": ordered_models,
        "lengths": lengths,
        "batches": batches,
        "architectures": architectures,
        "rows": rows,
        "summary": summarize(payload_rows=rows),
    }


def subset_payload(
    payload: dict[str, object], models: tuple[ModelName, ...]
) -> dict[str, object]:
    """Select complete model shards without changing their raw measurements."""
    present = tuple(cast("list[ModelName]", payload.get("models", [])))
    missing = sorted(set(models) - set(present))
    if not models or missing:
        message = f"subset models must be present and non-empty: missing={missing}"
        raise ValueError(message)
    selected = set(models)
    architectures = cast("dict[str, object]", payload.get("architectures", {}))
    rows = cast("list[dict[str, object]]", payload.get("rows", []))
    result = copy.deepcopy(payload)
    result["models"] = [model for model in MODELS if model in selected]
    result["architectures"] = {
        model: architectures[model] for model in MODELS if model in selected
    }
    selected_rows = [row for row in rows if row.get("model") in selected]
    result["rows"] = selected_rows
    result["summary"] = summarize(payload_rows=selected_rows)
    return result


def _benchmark_cell(
    model_name: ModelName,
    length: int,
    batch_size: int,
    *,
    config: BenchmarkConfig,
    device: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(config.seed)
    base_model, architecture = build_comparison_model(model_name, length, batch_size)
    state_dict = copy.deepcopy(base_model.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(config.seed + length + batch_size)
    cpu_inputs = torch.randn(batch_size, length, 1, generator=generator)
    del base_model

    eager_row, reference = _measure_runtime(
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
    rows = [eager_row]
    for runtime in CANDIDATE_RUNTIMES:
        if runtime == "specialized_fp32" and model_name not in {"efp16", "pa2wp"}:
            continue
        if runtime in {"manual_graph_fp32", "compiled_graph_fp32"} and model_name in {
            "efp16",
            "pa2wp",
        }:
            # Both PAC stems create device constants in eager forward on Torch 2.6.
            # Their specialized paths hoist constants and poles before graph capture.
            continue
        row: dict[str, object]
        try:
            row, _ = _measure_runtime(
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
        except Exception as error:  # noqa: BLE001 - preserve candidate failure evidence
            row = {
                "model": model_name,
                "display_name": DISPLAY_NAMES[model_name],
                "length": length,
                "batch_size": batch_size,
                "runtime": runtime,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _release_cuda()
        rows.append(row)

    exact_candidates = [
        row
        for row in rows[1:]
        if row.get("status") == "measured"
        and _as_float(row.get("max_abs_error", math.inf)) <= MAX_ABSOLUTE_ERROR
        and _as_float(row.get("prediction_agreement", 0.0)) == 1.0
    ]
    if exact_candidates:
        best_optimized = min(exact_candidates, key=lambda row: _as_float(row["latency_ms"]))
    else:
        best_optimized = eager_row
    optimized = _selected_row(
        best_optimized,
        runtime="optimized_fp32",
        eager=eager_row,
        fallback=not exact_candidates,
    )
    fastest_source = min(
        (eager_row, optimized), key=lambda row: float(cast("float", row["latency_ms"]))
    )
    fastest = _selected_row(
        fastest_source,
        runtime="fastest_exact_fp32",
        eager=eager_row,
        fallback=False,
    )
    rows.extend((optimized, fastest))
    return rows, architecture


def build_comparison_model(
    model_name: ModelName, length: int, batch_size: int
) -> tuple[nn.Module, dict[str, object]]:
    config = _base_config(length, batch_size)
    if model_name in {"efp16", "pa2wp"}:
        requested_dim = 32 if model_name == "efp16" else 64
        internal = "EFP16" if model_name == "efp16" else "PA2WP"
        requested_config = _base_config(length, batch_size, model_dim=requested_dim)
        model = build_efficient_headroom_classifier(
            internal, requested_config, 5, objective="classification"
        )
        parameters = count_parameters(model)
        return model, {
            "display_name": DISPLAY_NAMES[model_name],
            "family": model_name,
            "implementation": "repository-native PAC exact-FP32 inference",
            "internal_spec": internal,
            "requested_model_dim": requested_dim,
            "effective_model_dim": int(model.model_dim),
            "requested_modes": 16,
            "effective_modes": int(model.modes),
            "trainable_parameters": parameters,
            "target_parameters": 5989 if model_name == "pa2wp" else parameters,
            "relative_parameter_difference": (
                abs(parameters - 5989) / 5989 if model_name == "pa2wp" else 0.0
            ),
            "capacity_contract": (
                "final optimized paper model; intentionally not capacity matched"
                if model_name == "pa2wp"
                else "reference capacity"
            ),
            "state_dict_sha256": _state_dict_digest(model),
        }

    target_model = build_efficient_headroom_classifier(
        "EFP16", config, 5, objective="classification"
    )
    target_parameters = count_parameters(target_model)
    del target_model
    trial = SELECTED_TRIALS[model_name]
    matched = match_ucr_baseline(
        model_name,
        config,
        5,
        target_parameters=target_parameters,
        validation_trial=trial,
        tolerance=0.1,
    )
    metadata: dict[str, object]
    if model_name == "minirocket":
        metadata = {
            "family": "minirocket",
            "validation_trial": trial,
            "architecture_label": "deterministic_minirocket_ppv_4_dilations",
            "implementation": "repository-native deterministic MiniRocket-style PPV transform",
        }
    else:
        metadata = confirmatory_implementation_metadata(
            cast("ConfirmatoryFamily", model_name), trial
        )
    return matched.model, {
        "display_name": DISPLAY_NAMES[model_name],
        **metadata,
        "selected_validation_trial": trial,
        "matched_width": matched.width,
        "trainable_parameters": matched.parameters,
        "target_parameters": matched.target_parameters,
        "relative_parameter_difference": matched.relative_error,
        "capacity_contract": "accuracy-selected architecture; width-only EFP16 capacity match",
        "state_dict_sha256": _state_dict_digest(matched.model),
    }


def _measure_runtime(  # noqa: PLR0915 - measurement lifecycle is intentionally linear
    model_name: ModelName,
    length: int,
    batch_size: int,
    runtime: RuntimeName,
    *,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    reference: Tensor | None,
    config: BenchmarkConfig,
    device: str,
) -> tuple[dict[str, object], Tensor]:
    model, _ = build_comparison_model(model_name, length, batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    inputs = cpu_inputs.to(device=device, dtype=torch.float32)

    compile_seconds = 0.0
    if runtime != "eager_fp32":
        start = perf_counter()
        model = _prepare_static_graph_inputs(model_name, model, length=length)
        if runtime == "manual_graph_fp32":
            model = BorrowedEagerCudaGraphInference(model)
        elif runtime == "compiled_graph_fp32":
            model = BorrowedInputCudaGraphInference(
                model,
                compile_mode="max-autotune-no-cudagraphs",
                copy_output=False,
            )
        elif runtime == "specialized_fp32" and model_name == "efp16":
            model = prepare_efp16_ceiling_inference(
                cast("HeadroomPACClassifier", model),
                sequence_length=length,
                batch_size=batch_size,
                copy_output=False,
            )
        elif runtime == "specialized_fp32" and model_name == "pa2wp":
            model = prepare_pa2wp_persistent_core_inference(
                model,
                sequence_length=length,
                batch_size=batch_size,
            )
        else:
            message = f"unsupported runtime {runtime} for {model_name}"
            raise ValueError(message)
        with torch.inference_mode():
            model(inputs)
        torch.cuda.synchronize()
        compile_seconds = perf_counter() - start

    with torch.inference_mode():
        output = model(inputs)
        for _ in range(config.warmups):
            output = model(inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    event_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(config.groups):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_wall = perf_counter()
        start_event.record()
        with torch.inference_mode():
            for _ in range(config.iterations_per_group):
                output = model(inputs)
        end_event.record()
        end_event.synchronize()
        wall_samples.append(
            (perf_counter() - start_wall) * 1000.0 / config.iterations_per_group
        )
        event_samples.append(
            start_event.elapsed_time(end_event) / config.iterations_per_group
        )

    output_cpu = output.detach().float().cpu()
    if reference is None:
        max_abs_error = 0.0
        max_rel_error = 0.0
        prediction_agreement = 1.0
    else:
        absolute = (output_cpu - reference).abs()
        relative = absolute / reference.abs().clamp_min(1.0e-6)
        max_abs_error = float(absolute.max().item())
        max_rel_error = float(relative.max().item())
        prediction_agreement = float(
            (output_cpu.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean().item()
        )
    latency = statistics.median(event_samples)
    quartiles = statistics.quantiles(event_samples, n=4, method="inclusive")
    row: dict[str, object] = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "length": length,
        "batch_size": batch_size,
        "runtime": runtime,
        "status": "measured",
        "latency_ms": latency,
        "latency_iqr_ms": quartiles[2] - quartiles[0],
        "latency_samples_ms": event_samples,
        "wall_latency_ms": statistics.median(wall_samples),
        "wall_latency_samples_ms": wall_samples,
        "compile_seconds": compile_seconds,
        "examples_per_second": batch_size * 1000.0 / latency,
        "tokens_per_second": batch_size * length * 1000.0 / latency,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "prediction_agreement": prediction_agreement,
    }
    del model, inputs
    _release_cuda()
    return row, output_cpu


def _prepare_static_graph_inputs(
    model_name: ModelName, model: nn.Module, *, length: int
) -> nn.Module:
    if model_name == "minirocket":
        if not isinstance(model, ExternalMiniRocketClassifier):
            message = "MiniRocket graph preparation received the wrong model type"
            raise TypeError(message)
        return StaticMiniRocketInference(model)
    if model_name == "transformer":
        device = next(model.parameters()).device
        return StaticTransformerInference(model, length=length, device=device)
    return model


def _selected_row(
    source: dict[str, object],
    *,
    runtime: RuntimeName,
    eager: dict[str, object],
    fallback: bool,
) -> dict[str, object]:
    row = copy.deepcopy(source)
    selected_backend = str(source.get("selected_backend", source["runtime"]))
    row["runtime"] = runtime
    row["selected_backend"] = selected_backend
    row["selection_fallback_to_eager"] = fallback
    row["speedup_vs_eager"] = float(cast("float", eager["latency_ms"])) / float(
        cast("float", source["latency_ms"])
    )
    return row


def _base_config(length: int, batch_size: int, *, model_dim: int = 32) -> PACExperimentConfig:
    return PACExperimentConfig(
        sample_count=64,
        validation_count=16,
        test_count=16,
        sequence_length=length,
        raw_input_dim=1,
        output_dim=5,
        model_dim=model_dim,
        modes=16,
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


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _validate_benchmark_arguments(
    models: tuple[ModelName, ...],
    lengths: tuple[int, ...],
    batches: tuple[int, ...],
    config: BenchmarkConfig,
    device: str,
) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        message = "the model speed comparison requires CUDA"
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
    if min(config.warmups, config.groups, config.iterations_per_group) < 1:
        message = "warmups, groups, and iterations must be positive"
        raise ValueError(message)


def _validate_selection_provenance(models: tuple[ModelName, ...]) -> None:
    public_models = set(models) - {"efp16", "pa2wp", "minirocket"}
    if public_models:
        payload = json.loads(BASELINE_SELECTION.read_text(encoding="utf-8"))
        selected = cast("dict[str, dict[str, object]]", payload.get("selected_trials", {}))
        for model_name in public_models:
            actual = _as_int(selected.get(model_name, {}).get("trial", -1))
            expected = SELECTED_TRIALS[model_name]
            if actual != expected:
                message = (
                    f"selected trial drift for {model_name}: expected {expected}, got {actual}"
                )
                raise ValueError(message)
    if "minirocket" in models:
        payload = json.loads(MINIROCKET_SELECTION.read_text(encoding="utf-8"))
        minirocket = cast("dict[str, object]", payload.get("minirocket", {}))
        selected_trial = _as_int(minirocket.get("trial", -1))
        if selected_trial != SELECTED_TRIALS["minirocket"]:
            message = f"MiniRocket selected trial drift: got {selected_trial}"
            raise ValueError(message)


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


def _parse_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(",") if item)


def _parse_models(raw: str) -> tuple[ModelName, ...]:
    return cast("tuple[ModelName, ...]", tuple(item for item in raw.split(",") if item))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare exact-FP32 model inference speed")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--models", default=",".join(MODELS))
    benchmark_parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    benchmark_parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    benchmark_parser.add_argument("--warmups", type=int, default=20)
    benchmark_parser.add_argument("--groups", type=int, default=9)
    benchmark_parser.add_argument("--iterations-per-group", type=int, default=100)
    benchmark_parser.add_argument("--seed", type=int, default=7)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    subset_parser = subparsers.add_parser("subset")
    subset_parser.add_argument("--input", type=Path, required=True)
    subset_parser.add_argument("--models", required=True)
    subset_parser.add_argument("--output", type=Path, required=True)
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
                seed=args.seed,
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
    if args.command == "subset":
        payload = cast("dict[str, object]", json.loads(args.input.read_text(encoding="utf-8")))
        _write_json(args.output, subset_payload(payload, _parse_models(args.models)))
        return
    payload = cast("dict[str, object]", json.loads(args.input.read_text(encoding="utf-8")))
    evaluation = evaluate_result(payload)
    _write_json(args.output, evaluation)
    if evaluation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
