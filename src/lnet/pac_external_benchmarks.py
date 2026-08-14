# ruff: noqa: EM101, EM102, TRY003
from __future__ import annotations

import gc
import inspect
import math
import warnings
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, Protocol, assert_never, cast, runtime_checkable

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet import Alphabet
from .models import TransformerSequenceBaseline
from .pac_device import resolve_device
from .pac_external_reference_baselines import (
    ExternalCNN1DClassifier,
    ExternalInceptionTimeClassifier,
    ExternalMiniRocketClassifier,
    ExternalS4Classifier,
    ExternalS4DClassifier,
)
from .pac_external_tasks import (
    ExternalDatasetError,
    ExternalDatasetName,
    ExternalObjective,
    ExternalTask,
    ExternalTemporalMetadata,
    load_external_task,
    synthetic_external_task,
)
from .pac_headroom_efficient_models import (
    DUAL_PHASE_WP_PAC_MODEL,
    FINAL_PAC_MODEL,
    LEARNED_PAIR_WP_PAC_MODEL,
    OVERLAPPING_ANTIALIASED_PAC_MODEL,
    PHASE_AUGMENTED_ENSEMBLE_WP_PAC_MODEL,
    PHASE_AUGMENTED_WP_PAC_MODEL,
    PHASE_COMPLETE_WP_PAC_MODEL,
    SPARSE_MULTISCALE_FBFB_PAC_MODEL,
    SPARSE_MULTISCALE_FF_PAC_MODEL,
    SPARSE_MULTISCALE_PAC_MODEL,
    UNDECIMATED_MODAL_DYADIC_PAC_MODEL,
    build_efficient_headroom_classifier,
)
from .pac_metrics import count_parameters
from .pac_overnight_io import prepare_overnight_dirs, write_csv_rows
from .pac_stiefel_variants import DEFAULT_PAC_MODEL
from .pac_tight_frame_models import build_tight_frame_classifier
from .pac_types import PACDevice, PACExperimentConfig
from .pac_unified_models import build_unified_pac_classifier

if TYPE_CHECKING:
    from pathlib import Path

    from .tapped_prl_followup_schema import JsonRow

ExternalModelFamily = Literal[
    "pac",
    "tcn",
    "cnn1d",
    "transformer",
    "mamba",
    "gru",
    "lstm",
    "s4",
    "s4d",
    "minirocket",
    "inception_time",
]


class _ExternalExactSplitRuntime(Protocol):
    def step(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor: ...

    def close(self) -> None: ...

    def activate(self) -> None: ...


@runtime_checkable
class _ExternalExactSplitProvider(Protocol):
    def prepare_external_exact_split_runtime(
        self,
        optimizer: torch.optim.AdamW,
        inputs: Tensor,
        targets: Tensor,
        *,
        objective: ExternalObjective,
        grad_clip_norm: float,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
        metadata_prevalidated: bool = False,
    ) -> _ExternalExactSplitRuntime: ...


def _external_exact_split_provider(model: nn.Module) -> _ExternalExactSplitProvider | None:
    """Return the provider only when the model explicitly opts into the fast path."""
    enabled = getattr(
        model,
        "use_external_exact_split_training",
        getattr(model, "use_efp16_exact_split_training", False),
    )
    if not enabled:
        return None
    if not isinstance(model, _ExternalExactSplitProvider):
        return None
    return cast("_ExternalExactSplitProvider", model)


CANONICAL_MODEL: Final = "pac_stiefel_depth2_norm_autocorr"
FINAL_ALPHABET_MODEL: Final = "alphabet_radial_log_r_affine"
DEFAULT_DATASETS: Final[tuple[ExternalDatasetName, ...]] = (
    "ptb-xl",
    "mit-bih",
    "cwru",
    "speech-commands",
    "pathfinder",
    "ettm1",
    "ettm2",
    "electricity",
    "weather",
    "lra-listops",
    "lra-text",
    "lra-retrieval",
    "lra-image",
    "sequential-mnist",
    "permuted-mnist",
    "sequential-cifar",
    "audioset-balanced",
)
DEFAULT_MODELS: Final[tuple[ExternalModelFamily, ...]] = (
    "pac",
    "tcn",
    "cnn1d",
    "transformer",
    "mamba",
    "gru",
    "lstm",
    "s4",
    "s4d",
    "minirocket",
    "inception_time",
)


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkConfig:
    data_root: Path
    output_root: Path
    datasets: tuple[ExternalDatasetName, ...] = DEFAULT_DATASETS
    models: tuple[ExternalModelFamily, ...] = DEFAULT_MODELS
    model_dim: int = 64
    modes: int = 16
    max_baseline_width: int = 256
    parameter_match_tolerance: float = 0.05
    mitbih_beat_length: int = 256
    cwru_window_length: int = 2048
    forecast_context_length: int = 96
    prediction_length: int = 96
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    patience: int = 8
    seeds: tuple[int, ...] = (7, 11, 19)
    device: PACDevice = "auto"
    latency_warmup: int = 5
    latency_iterations: int = 20
    smoke: bool = False
    pac_model: str = DEFAULT_PAC_MODEL
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True, slots=True)
class ModelMatch:
    family: ExternalModelFamily
    width: int | None
    parameters: int | None
    relative_error: float | None
    status: Literal["matched", "unavailable"]
    reason: str = ""


def run_external_benchmarks(config: ExternalBenchmarkConfig) -> Path:
    _validate_config(config)
    prepare_overnight_dirs(config.output_root)
    device = resolve_device(config.device)
    rows: list[JsonRow] = []
    matches: list[JsonRow] = []
    tasks = _load_tasks(config)
    for dataset_name, task, error in tasks:
        if task is None:
            rows.append(_unavailable_dataset_row(dataset_name, error))
            continue
        pac = _build_model("pac", config.model_dim, task, config)
        target_parameters = count_parameters(pac)
        model_matches = tuple(
            match_external_parameter_budget(
                family,
                target_parameters,
                task,
                config,
            )
            for family in config.models
        )
        matches.extend(_match_row(task, target_parameters, match) for match in model_matches)
        for match in model_matches:
            if match.status == "unavailable" or match.width is None:
                rows.append(_unavailable_model_row(task, target_parameters, match))
                continue
            rows.extend(
                _run_one(task, match, target_parameters, seed, config, device)
                for seed in config.seeds
            )
    write_csv_rows(config.output_root / "results" / "external_parameter_matches.csv", matches)
    write_csv_rows(config.output_root / "results" / "external_comparisons.csv", rows)
    _write_report(config, device, rows, matches)
    return config.output_root


def match_external_parameter_budget(  # noqa: C901 - bounded monotone width search
    family: ExternalModelFamily,
    target_parameters: int,
    task: ExternalTask,
    config: ExternalBenchmarkConfig,
) -> ModelMatch:
    if family == "pac":
        parameters = count_parameters(_build_model("pac", config.model_dim, task, config))
        return ModelMatch(
            family,
            config.model_dim,
            parameters,
            _relative_error(parameters, target_parameters),
            "matched",
        )
    candidates: list[ModelMatch] = []

    def evaluate(width: int) -> ModelMatch:
        try:
            model = _build_model(family, width, task, config)
        except (ImportError, ModuleNotFoundError) as error:
            return ModelMatch(
                family,
                None,
                None,
                None,
                "unavailable",
                f"{type(error).__name__}: {error}",
            )
        parameters = count_parameters(model)
        return ModelMatch(
            family,
            width,
            parameters,
            _relative_error(parameters, target_parameters),
            "matched",
        )

    first = evaluate(1)
    if first.status == "unavailable":
        return first
    candidates.append(first)
    if first.parameters is not None and first.parameters < target_parameters:
        lower = 1
        upper = min(2, config.max_baseline_width)
        while upper > lower:
            candidate = evaluate(upper)
            if candidate.status == "unavailable":
                return candidate
            candidates.append(candidate)
            if candidate.parameters is not None and candidate.parameters >= target_parameters:
                while lower + 1 < upper:
                    middle = (lower + upper) // 2
                    middle_candidate = evaluate(middle)
                    if middle_candidate.status == "unavailable":
                        return middle_candidate
                    candidates.append(middle_candidate)
                    if (
                        middle_candidate.parameters is not None
                        and middle_candidate.parameters < target_parameters
                    ):
                        lower = middle
                    else:
                        upper = middle
                break
            if upper == config.max_baseline_width:
                break
            lower = upper
            upper = min(2 * upper, config.max_baseline_width)
    best = min(candidates, key=_match_key)
    if not config.smoke and (
        best.relative_error is None or best.relative_error > config.parameter_match_tolerance
    ):
        return ModelMatch(
            family,
            best.width,
            best.parameters,
            best.relative_error,
            "unavailable",
            (
                f"closest parameter error {best.relative_error:.6f} exceeds "
                f"tolerance {config.parameter_match_tolerance:.6f}"
            ),
        )
    return best


def external_metric_bundle(
    logits: Tensor, targets: Tensor, objective: ExternalObjective
) -> dict[str, float]:
    if objective == "multiclass":
        predictions = logits.argmax(dim=-1)
        accuracy = float((predictions == targets).to(torch.float32).mean().item())
        multiclass_f1: list[float] = []
        recall_values: list[float] = []
        for class_index in range(logits.shape[-1]):
            predicted = predictions == class_index
            actual = targets == class_index
            true_positive = float((predicted & actual).sum().item())
            precision = true_positive / max(float(predicted.sum().item()), 1.0)
            recall = true_positive / max(float(actual.sum().item()), 1.0)
            multiclass_f1.append(
                0.0
                if precision + recall == 0.0
                else 2.0 * precision * recall / (precision + recall)
            )
            recall_values.append(recall)
        return {
            "accuracy": accuracy,
            "macro_f1": sum(multiclass_f1) / len(multiclass_f1),
            "balanced_accuracy": sum(recall_values) / len(recall_values),
        }
    if objective == "multilabel":
        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= 0.5
        f1_values: list[float] = []
        auc_values: list[float] = []
        ap_values: list[float] = []
        for class_index in range(logits.shape[-1]):
            actual = targets[:, class_index] > 0.5
            predicted = predictions[:, class_index]
            true_positive = float((actual & predicted).sum().item())
            precision = true_positive / max(float(predicted.sum().item()), 1.0)
            recall = true_positive / max(float(actual.sum().item()), 1.0)
            f1_values.append(
                0.0
                if precision + recall == 0.0
                else 2.0 * precision * recall / (precision + recall)
            )
            auc = _binary_auroc(probabilities[:, class_index], actual)
            average_precision = _binary_average_precision(probabilities[:, class_index], actual)
            if not math.isnan(auc):
                auc_values.append(auc)
            if not math.isnan(average_precision):
                ap_values.append(average_precision)
        return {
            "macro_f1": sum(f1_values) / len(f1_values),
            "macro_auroc": _mean_or_nan(auc_values),
            "macro_auprc": _mean_or_nan(ap_values),
        }
    if objective == "forecasting":
        predictions = logits.reshape_as(targets)
        return {
            "mse": float(functional.mse_loss(predictions, targets).item()),
            "mae": float(functional.l1_loss(predictions, targets).item()),
        }
    assert_never(objective)


def _load_tasks(
    config: ExternalBenchmarkConfig,
) -> tuple[tuple[str, ExternalTask | None, str], ...]:
    if config.smoke:
        return (
            ("synthetic-multiclass", synthetic_external_task("multiclass"), ""),
            ("synthetic-multilabel", synthetic_external_task("multilabel"), ""),
            ("synthetic-forecasting", synthetic_external_task("forecasting"), ""),
        )
    loaded: list[tuple[str, ExternalTask | None, str]] = []
    for name in config.datasets:
        try:
            task = load_external_task(
                name,
                config.data_root,
                mitbih_beat_length=config.mitbih_beat_length,
                cwru_window_length=config.cwru_window_length,
                forecast_context_length=config.forecast_context_length,
                prediction_length=config.prediction_length,
            )
        except (ExternalDatasetError, FileNotFoundError, ModuleNotFoundError) as error:
            loaded.append((name, None, f"{type(error).__name__}: {error}"))
        else:
            loaded.append((name, task, ""))
    return tuple(loaded)


def _run_one(
    task: ExternalTask,
    match: ModelMatch,
    target_parameters: int,
    seed: int,
    config: ExternalBenchmarkConfig,
    device: str,
) -> JsonRow:
    common = _common_row(task, target_parameters, match)
    common["seed"] = seed
    try:
        _seed_everything(seed, device)
        if match.width is None:
            raise RuntimeError("matched model has no width")  # noqa: TRY301
        model = _build_model(match.family, match.width, task, config).to(device=device)
        started_at = perf_counter()
        best_epoch, validation_loss = _train_model(model, task, config, device, seed)
        train_seconds = perf_counter() - started_at
        logits, targets = _predict(
            model,
            task.test_inputs,
            task.test_targets,
            config.batch_size,
            device,
            metadata=task.test_metadata,
        )
        metrics = external_metric_bundle(logits, targets, task.objective)
        test_loss = float(_loss(logits, targets, task.objective).item())
        latency_ms, peak_memory_mb = _measure_latency(
            model,
            task.test_inputs,
            config,
            device,
            metadata=task.test_metadata,
        )
        return {  # noqa: TRY300
            **common,
            "status": "done",
            "best_epoch": best_epoch,
            "validation_loss": validation_loss,
            "test_loss": test_loss,
            "train_seconds": train_seconds,
            "latency_ms": latency_ms,
            "peak_memory_mb": peak_memory_mb,
            **metrics,
            "error": "",
        }
    except (AssertionError, RuntimeError, ValueError, torch.OutOfMemoryError) as error:
        return {
            **common,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        _release_device(device)


def _train_model(  # noqa: C901, PLR0912, PLR0915
    model: nn.Module,
    task: ExternalTask,
    config: ExternalBenchmarkConfig,
    device: str,
    seed: int,
    *,
    stage_training_data: bool | None = False,
) -> tuple[int, float]:
    if config.gradient_accumulation_steps < 1:
        message = "gradient_accumulation_steps must be positive"
        raise ValueError(message)
    exact_split_provider = _external_exact_split_provider(model)
    if config.gradient_accumulation_steps != 1:
        exact_split_provider = None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device == "cuda",
        capturable=device == "cuda" and exact_split_provider is not None,
    )
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, Tensor] | None = None
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    (
        train_inputs,
        train_targets,
        train_metadata,
        validation_inputs,
        validation_targets,
        validation_metadata,
        gpu_resident_data,
    ) = _stage_training_splits(task, device, force=stage_training_data)
    model.__dict__["external_gpu_resident_training_data"] = gpu_resident_data
    exact_split_runtime: _ExternalExactSplitRuntime | None = None
    captured_batch_size: int | None = None
    exact_split_full_steps = 0
    exact_split_fallback_steps = 0
    try:
        for epoch in range(config.epochs):
            model.train()
            order = torch.randperm(task.train_inputs.shape[0], generator=generator)
            if gpu_resident_data:
                order = order.to(device=device)
            microbatches = order.split(config.batch_size)
            accumulation = config.gradient_accumulation_steps
            for group_start in range(0, len(microbatches), accumulation):
                group = microbatches[group_start : group_start + accumulation]
                group_size = sum(int(indices.shape[0]) for indices in group)
                eager_batches = 0
                for indices in group:
                    inputs = train_inputs[indices]
                    targets = train_targets[indices]
                    metadata = train_metadata.index_select(indices)
                    if not gpu_resident_data:
                        inputs = inputs.to(device=device)
                        targets = targets.to(device=device)
                        metadata = metadata.to(device=device)
                    use_exact_split = exact_split_provider is not None and (
                        captured_batch_size is None or inputs.shape[0] == captured_batch_size
                    )
                    if use_exact_split and exact_split_runtime is None:
                        active_provider = cast("_ExternalExactSplitProvider", exact_split_provider)
                        try:
                            if metadata.is_empty:
                                exact_split_runtime = (
                                    active_provider.prepare_external_exact_split_runtime(
                                        optimizer,
                                        inputs,
                                        targets,
                                        objective=task.objective,
                                        grad_clip_norm=config.grad_clip_norm,
                                    )
                                )
                            else:
                                exact_split_runtime = (
                                    active_provider.prepare_external_exact_split_runtime(
                                        optimizer,
                                        inputs,
                                        targets,
                                        objective=task.objective,
                                        grad_clip_norm=config.grad_clip_norm,
                                        time_delta=metadata.time_delta,
                                        observation_mask=metadata.observation_mask,
                                        valid_mask=metadata.valid_mask,
                                        metadata_prevalidated=True,
                                    )
                                )
                        except Exception as error:  # graph failures vary by CUDA/runtime version
                            model.__dict__["external_exact_split_capture_error"] = (
                                f"{type(error).__name__}: {error}"
                            )
                            if getattr(model, "require_external_exact_split_training", False):
                                message = (
                                    "required external exact-split runtime failed to initialize"
                                )
                                raise RuntimeError(message) from error
                            exact_split_provider = None
                            use_exact_split = False
                        else:
                            captured_batch_size = int(inputs.shape[0])
                            model.__dict__["external_exact_split_runtime_kind"] = getattr(
                                exact_split_runtime,
                                "training_backend",
                                "external_exact_split",
                            )
                    if use_exact_split and exact_split_runtime is not None:
                        if metadata.is_empty:
                            exact_split_runtime.step(inputs, targets)
                        else:
                            exact_split_runtime.step(
                                inputs,
                                targets,
                                **metadata.model_kwargs(),
                            )
                        exact_split_full_steps += 1
                        continue
                    if exact_split_provider is not None:
                        exact_split_fallback_steps += 1
                    if exact_split_runtime is not None:
                        exact_split_runtime.close()
                    if eager_batches == 0:
                        optimizer.zero_grad(
                            set_to_none=exact_split_runtime is None
                        )
                    logits = _forward_with_metadata(model, inputs, metadata)
                    loss = _loss(logits, targets, task.objective)
                    (loss * (inputs.shape[0] / max(group_size, 1))).backward()
                    eager_batches += 1
                if eager_batches:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip_norm
                    )
                    optimizer.step()
                    _post_optimizer_step(model)
            if exact_split_runtime is not None:
                exact_split_runtime.close()
            validation_logits, validation_targets = _predict(
                model,
                validation_inputs,
                validation_targets,
                config.batch_size,
                device,
                metadata=validation_metadata,
            )
            validation_loss = float(
                _loss(validation_logits, validation_targets, task.objective).item()
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break
    finally:
        if exact_split_runtime is not None:
            exact_split_runtime.close()
        model.__dict__["external_exact_split_full_steps"] = exact_split_full_steps
        model.__dict__["external_exact_split_fallback_steps"] = exact_split_fallback_steps
        model.__dict__["external_exact_split_capture_succeeded"] = exact_split_runtime is not None
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    _finalize_constraints(model)
    return best_epoch, best_validation


def _stage_training_splits(
    task: ExternalTask,
    device: str,
    *,
    force: bool | None = None,
) -> tuple[
    Tensor,
    Tensor,
    ExternalTemporalMetadata,
    Tensor,
    Tensor,
    ExternalTemporalMetadata,
    bool,
]:
    tensors = (
        task.train_inputs,
        task.train_targets,
        task.validation_inputs,
        task.validation_targets,
    )
    if not device.startswith("cuda") or force is False:
        return (
            task.train_inputs,
            task.train_targets,
            task.train_metadata,
            task.validation_inputs,
            task.validation_targets,
            task.validation_metadata,
            False,
        )
    metadata_tensors = (
        *task.train_metadata.model_kwargs().values(),
        *task.validation_metadata.model_kwargs().values(),
    )
    required_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in (*tensors, *metadata_tensors)
    )
    free_bytes, _ = torch.cuda.mem_get_info()
    # Leave ample room for the model, optimizer, activations, and validation logits.
    if force is None and required_bytes > 0.45 * free_bytes:
        return (
            task.train_inputs,
            task.train_targets,
            task.train_metadata,
            task.validation_inputs,
            task.validation_targets,
            task.validation_metadata,
            False,
        )
    try:
        train_inputs, train_targets, validation_inputs, validation_targets = (
            tensor.to(device=device) for tensor in tensors
        )
        train_metadata = task.train_metadata.to(device=device)
        validation_metadata = task.validation_metadata.to(device=device)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if force is True:
            raise
        return (
            task.train_inputs,
            task.train_targets,
            task.train_metadata,
            task.validation_inputs,
            task.validation_targets,
            task.validation_metadata,
            False,
        )
    return (
        train_inputs,
        train_targets,
        train_metadata,
        validation_inputs,
        validation_targets,
        validation_metadata,
        True,
    )


def _predict(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
    device: str,
    *,
    metadata: ExternalTemporalMetadata | None = None,
) -> tuple[Tensor, Tensor]:
    active_metadata = metadata or ExternalTemporalMetadata()
    was_training = model.training
    model.eval()
    outputs: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            batch = inputs[start:stop].to(device=device)
            batch_metadata = active_metadata.batch_slice(start, stop).to(device=device)
            outputs.append(_forward_with_metadata(model, batch, batch_metadata).detach().cpu())
    model.train(was_training)
    return torch.cat(outputs), targets.detach().cpu()


def _loss(logits: Tensor, targets: Tensor, objective: ExternalObjective) -> Tensor:
    if objective == "multiclass":
        return functional.cross_entropy(logits, targets)
    if objective == "multilabel":
        return functional.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype))
    if objective == "forecasting":
        return functional.mse_loss(logits.reshape_as(targets), targets)
    assert_never(objective)


def _forward_with_metadata(
    model: nn.Module,
    inputs: Tensor,
    metadata: ExternalTemporalMetadata,
) -> Tensor:
    if metadata.is_empty:
        return model(inputs)
    return model(inputs, **metadata.model_kwargs())


def _build_model(
    family: ExternalModelFamily,
    width: int,
    task: ExternalTask,
    benchmark: ExternalBenchmarkConfig,
) -> nn.Module:
    packed_temporal_baseline = (
        task.input_encoding == "continuous" and task.has_temporal_metadata and family != "pac"
    )
    if task.input_encoding == "continuous":
        core_input_dim = 2 * task.input_dim + 2 if packed_temporal_baseline else task.input_dim
    else:
        core_input_dim = width
    core_output_dim = width if task.input_encoding == "token_pair" else task.output_dim
    experiment = PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        task.test_inputs.shape[0],
        task.sequence_length,
        raw_input_dim=core_input_dim,
        output_dim=core_output_dim,
        model_dim=benchmark.model_dim if family == "pac" else width,
        modes=benchmark.modes if family == "pac" else max(1, width // 4),
        epochs=benchmark.epochs,
        batch_size=benchmark.batch_size,
        learning_rate=benchmark.learning_rate,
        weight_decay=benchmark.weight_decay,
        grad_clip_norm=benchmark.grad_clip_norm,
        device=benchmark.device,
    )
    core = _build_continuous_model(
        family,
        width,
        core_input_dim,
        core_output_dim,
        experiment,
        benchmark.pac_model,
        coordinate_shape=_coordinate_shape_for_task(task),
        objective="regression" if task.objective == "forecasting" else "classification",
    )
    if task.input_encoding == "continuous" and not task.has_temporal_metadata:
        return core
    if task.input_encoding == "continuous" and family == "pac":
        if not (
            getattr(core, "supports_observation_mask", False)
            and getattr(core, "supports_time_delta", False)
        ):
            raise RuntimeError(
                "PAC candidate does not expose the required native temporal-metadata contract"
            )
        return _NativeTemporalMetadataAdapter(core)
    if task.input_encoding == "continuous":
        return _PackedTemporalMetadataAdapter(core, task.input_dim)
    if task.vocab_size is None:
        raise RuntimeError("token task is missing vocab_size")
    if task.input_encoding == "tokens":
        return _TokenEmbeddingClassifier(task.vocab_size, width, core)
    if task.input_encoding == "token_pair":
        return _TokenPairClassifier(task.vocab_size, width, core, task.output_dim)
    assert_never(task.input_encoding)


def _build_continuous_model(  # noqa: C901, PLR0911, PLR0912 - explicit family registry
    family: ExternalModelFamily,
    width: int,
    input_dim: int,
    output_dim: int,
    experiment: PACExperimentConfig,
    pac_model: str,
    *,
    coordinate_shape: tuple[int, int] | None = None,
    objective: Literal["classification", "regression"] = "classification",
) -> nn.Module:
    match family:
        case "pac":
            if pac_model == FINAL_ALPHABET_MODEL:
                return Alphabet(
                    experiment,
                    output_dim,
                    objective=objective,
                )
            if pac_model == FINAL_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "WP",
                    experiment,
                    output_dim,
                    objective=objective,
                )
            if pac_model == OVERLAPPING_ANTIALIASED_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "OA",
                    experiment,
                    output_dim,
                    objective=objective,
                )
            if pac_model == PHASE_COMPLETE_WP_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "PCWP",
                    experiment,
                    output_dim,
                    objective=objective,
                )
            if pac_model == DUAL_PHASE_WP_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "DPWP",
                    experiment,
                    output_dim,
                    objective=objective,
                )
            if pac_model == PHASE_AUGMENTED_WP_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "PAWP",
                    experiment,
                    output_dim,
                    objective=objective,
                )
            if pac_model == PHASE_AUGMENTED_ENSEMBLE_WP_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "PA2WP",
                    experiment,
                    output_dim,
                    objective=objective,
                )
            if pac_model == LEARNED_PAIR_WP_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "LPWP",
                    experiment,
                    output_dim,
                    objective=objective,
                )
            if pac_model == SPARSE_MULTISCALE_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "SMR",
                    experiment,
                    output_dim,
                    coordinate_shape=coordinate_shape,
                    objective=objective,
                )
            if pac_model == SPARSE_MULTISCALE_FF_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "SMRFF",
                    experiment,
                    output_dim,
                    coordinate_shape=coordinate_shape,
                    objective=objective,
                )
            if pac_model == SPARSE_MULTISCALE_FBFB_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "SMRFBFB",
                    experiment,
                    output_dim,
                    coordinate_shape=coordinate_shape,
                    objective=objective,
                )
            if pac_model == UNDECIMATED_MODAL_DYADIC_PAC_MODEL:
                return build_efficient_headroom_classifier(
                    "UMD",
                    experiment,
                    output_dim,
                    coordinate_shape=coordinate_shape,
                    objective=objective,
                )
            model = build_unified_pac_classifier(
                pac_model,
                experiment,
                output_dim,
                coordinate_shape=coordinate_shape,
                objective=objective,
            )
            if model is None:
                model = build_tight_frame_classifier(pac_model, experiment, output_dim)
            if model is None:
                raise RuntimeError(f"PAC model is unavailable: {pac_model}")
            return model
        case "tcn":
            return _TCNClassifier(input_dim, width, output_dim)
        case "cnn1d":
            return ExternalCNN1DClassifier(input_dim, width, output_dim)
        case "transformer":
            heads = _attention_heads(width)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                encoder = TransformerSequenceBaseline(
                    raw_input_dim=input_dim,
                    model_dim=width,
                    output_dim=width,
                    attention_heads=heads,
                )
            return _MeanHead(encoder, width, output_dim)
        case "mamba":
            return _MambaClassifier(input_dim, width, output_dim)
        case "gru":
            return _RecurrentClassifier("gru", input_dim, width, output_dim)
        case "lstm":
            return _RecurrentClassifier("lstm", input_dim, width, output_dim)
        case "s4":
            return ExternalS4Classifier(input_dim, width, output_dim)
        case "s4d":
            return ExternalS4DClassifier(input_dim, width, output_dim)
        case "minirocket":
            return ExternalMiniRocketClassifier(input_dim, width, output_dim)
        case "inception_time":
            return ExternalInceptionTimeClassifier(input_dim, width, output_dim)
        case unreachable:
            assert_never(unreachable)


class _NativeTemporalMetadataAdapter(nn.Module):
    """Preserve the raw signal path while forwarding physical-time metadata."""

    supports_observation_mask: Final[bool] = True
    supports_time_delta: Final[bool] = True

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core
        self.use_external_exact_split_training = bool(
            getattr(core, "use_external_exact_split_training", False)
        )
        self.require_external_exact_split_training = bool(
            getattr(core, "require_external_exact_split_training", False)
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        return self.core(
            inputs,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )

    def prepare_external_exact_split_runtime(
        self,
        optimizer: torch.optim.AdamW,
        inputs: Tensor,
        targets: Tensor,
        *,
        objective: ExternalObjective,
        grad_clip_norm: float,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
        metadata_prevalidated: bool = False,
    ) -> _ExternalExactSplitRuntime:
        callback = getattr(self.core, "prepare_external_exact_split_runtime", None)
        if not callable(callback):
            raise RuntimeError(  # noqa: TRY004
                "native temporal PAC has no external exact-split runtime"
            )
        return cast(
            "_ExternalExactSplitRuntime",
            callback(
                optimizer,
                inputs,
                targets,
                objective=objective,
                grad_clip_norm=grad_clip_norm,
                time_delta=time_delta,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
                metadata_prevalidated=metadata_prevalidated,
            ),
        )

    def post_optimizer_step(self) -> None:
        _call_constraint_callback(self.core, "post_optimizer_step")

    def finalize_constraints(self) -> None:
        _call_constraint_callback(self.core, "finalize_constraints")


class _PackedTemporalMetadataAdapter(nn.Module):
    """Give discrete baselines the same values, masks, intervals, and validity bits."""

    supports_observation_mask: Final[bool] = True
    supports_time_delta: Final[bool] = True

    def __init__(self, core: nn.Module, raw_input_dim: int) -> None:
        super().__init__()
        self.core = core
        self.raw_input_dim = raw_input_dim
        forward_parameters = inspect.signature(core.forward).parameters.values()
        self.core_accepts_valid_mask = any(
            parameter.name == "valid_mask"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in forward_parameters
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        batch, steps, channels = inputs.shape
        if channels != self.raw_input_dim:
            raise ValueError(f"expected {self.raw_input_dim} raw channels, got {channels}")
        valid = (
            torch.ones((batch, steps, 1), device=inputs.device, dtype=inputs.dtype)
            if valid_mask is None
            else valid_mask.to(device=inputs.device, dtype=inputs.dtype)
        )
        if valid.ndim == 2:
            valid = valid.unsqueeze(-1)
        observed = (
            torch.ones_like(inputs)
            if observation_mask is None
            else observation_mask.to(device=inputs.device, dtype=inputs.dtype)
        )
        if observed.ndim == 2:
            observed = observed.unsqueeze(-1)
        if observed.shape[-1] == 1:
            observed = observed.expand(-1, -1, channels)
        observed = observed * valid
        delta = (
            torch.ones((batch, steps, 1), device=inputs.device, dtype=inputs.dtype)
            if time_delta is None
            else time_delta.to(device=inputs.device, dtype=inputs.dtype)
        )
        if delta.ndim == 2:
            delta = delta.unsqueeze(-1)
        packed = torch.cat(
            (
                inputs * observed,
                observed,
                delta * valid,
                valid,
            ),
            dim=-1,
        )
        if valid_mask is None:
            return self.core(packed)
        valid_bool = valid.squeeze(-1).to(dtype=torch.bool)
        if bool((valid_bool.sum(dim=1) == 0).any()):
            raise ValueError("valid_mask must keep at least one timestep per sample")
        if bool((valid_bool[:, 1:] & ~valid_bool[:, :-1]).any()):
            raise ValueError("valid_mask must be a right-padded prefix mask")
        if (
            getattr(self.core, "requires_exact_length_groups", False)
            or not self.core_accepts_valid_mask
        ):
            return self._forward_length_groups(packed, valid_bool)
        return self.core(packed, valid_mask=valid_bool)

    def _forward_length_groups(self, packed: Tensor, valid_mask: Tensor) -> Tensor:
        lengths = valid_mask.sum(dim=1)
        grouped_outputs: list[Tensor] = []
        grouped_indices: list[Tensor] = []
        for length in torch.unique(lengths, sorted=True):
            indices = torch.nonzero(lengths == length, as_tuple=False).squeeze(-1)
            steps = int(length.item())
            grouped_outputs.append(
                self.core(packed.index_select(0, indices)[:, :steps])
            )
            grouped_indices.append(indices)
        indices = torch.cat(grouped_indices)
        outputs = torch.cat(grouped_outputs)
        return outputs.index_select(0, torch.argsort(indices))

    def post_optimizer_step(self) -> None:
        _call_constraint_callback(self.core, "post_optimizer_step")

    def finalize_constraints(self) -> None:
        _call_constraint_callback(self.core, "finalize_constraints")


class _TokenEmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size: int, width: int, core: nn.Module) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.core = core

    def forward(self, inputs: Tensor) -> Tensor:
        token_ids = inputs.squeeze(-1).to(torch.long)
        features = _token_features(self.embedding, token_ids)
        if getattr(self.core, "supports_observation_mask", False):
            mask = token_ids.ne(0)
            return self.core(features, observation_mask=mask, valid_mask=mask)
        return self.core(features)

    def post_optimizer_step(self) -> None:
        _call_constraint_callback(self.core, "post_optimizer_step")

    def finalize_constraints(self) -> None:
        _call_constraint_callback(self.core, "finalize_constraints")


class _TokenPairClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        width: int,
        core: nn.Module,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.core = core
        self.head = nn.Linear(4 * width, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        token_ids = inputs.to(torch.long)
        left_ids = token_ids[..., 0]
        right_ids = token_ids[..., 1]
        left_features = _token_features(self.embedding, left_ids)
        right_features = _token_features(self.embedding, right_ids)
        if getattr(self.core, "supports_observation_mask", False):
            left_mask = left_ids.ne(0)
            right_mask = right_ids.ne(0)
            left = self.core(left_features, observation_mask=left_mask, valid_mask=left_mask)
            right = self.core(right_features, observation_mask=right_mask, valid_mask=right_mask)
        else:
            left = self.core(left_features)
            right = self.core(right_features)
        interaction = torch.cat((left, right, (left - right).abs(), left * right), dim=-1)
        return self.head(interaction)

    def post_optimizer_step(self) -> None:
        _call_constraint_callback(self.core, "post_optimizer_step")

    def finalize_constraints(self) -> None:
        _call_constraint_callback(self.core, "finalize_constraints")


def _token_features(embedding: nn.Embedding, token_ids: Tensor) -> Tensor:
    features = embedding(token_ids)
    positions = _sinusoidal_positions(
        token_ids.shape[1],
        features.shape[-1],
        device=features.device,
        dtype=features.dtype,
    )
    return (features + positions) * token_ids.ne(0).unsqueeze(-1)


def _coordinate_shape_for_task(task: ExternalTask) -> tuple[int, int] | None:
    shapes = {
        "pathfinder": (32, 32),
        "lra-image": (32, 32),
        "sequential-mnist": (28, 28),
        "sequential-cifar": (32, 32),
    }
    shape = shapes.get(task.name)
    packed_rows = task.name == "sequential-cifar" and shape is not None
    if shape is not None and shape[0] * shape[1] != task.sequence_length and not packed_rows:
        raise RuntimeError(f"coordinate shape does not match {task.name} sequence length")
    return shape


def _sinusoidal_positions(
    length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(width, 1))
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype).unsqueeze(0)


class _MeanHead(nn.Module):
    def __init__(self, encoder: nn.Module, width: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(width, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.encoder(inputs).mean(dim=1))


class _TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, width, 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(width, width, 5, dilation=2**level, groups=1) for level in range(2)
        )
        self.head = nn.Linear(width, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.input_projection(inputs.transpose(1, 2))
        for level, convolution in enumerate(self.layers):
            padding = (2**level) * 4
            update = functional.gelu(convolution(functional.pad(features, (padding, 0))))
            features = features + update
        return self.head(features.mean(dim=-1))


class _RecurrentClassifier(nn.Module):
    def __init__(
        self,
        kind: Literal["gru", "lstm"],
        input_dim: int,
        width: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        recurrent_type = nn.GRU if kind == "gru" else nn.LSTM
        self.recurrent = recurrent_type(input_dim, width, batch_first=True)
        self.head = nn.Linear(width, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        features, _ = self.recurrent(inputs)
        return self.head(features.mean(dim=1))


class _MambaClassifier(nn.Module):
    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        mamba_ssm = import_module("mamba_ssm")
        self.input_projection = nn.Linear(input_dim, width)
        self.mamba = mamba_ssm.Mamba(d_model=width, d_state=16, d_conv=4, expand=2)
        self.head = nn.Linear(width, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.mamba(self.input_projection(inputs)).mean(dim=1))


def _measure_latency(
    model: nn.Module,
    inputs: Tensor,
    config: ExternalBenchmarkConfig,
    device: str,
    *,
    metadata: ExternalTemporalMetadata | None = None,
) -> tuple[float, float]:
    sample_count = min(config.batch_size, inputs.shape[0])
    batch = inputs[:sample_count].to(device=device)
    batch_metadata = (
        (metadata or ExternalTemporalMetadata()).batch_slice(0, sample_count).to(device=device)
    )
    model.eval()
    with torch.no_grad():
        for _ in range(config.latency_warmup):
            _forward_with_metadata(model, batch, batch_metadata)
        _sync(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(config.latency_iterations):
                _forward_with_metadata(model, batch, batch_metadata)
            end.record()
            torch.cuda.synchronize()
            return (
                float(start.elapsed_time(end) / config.latency_iterations),
                float(torch.cuda.max_memory_allocated() / 1_000_000),
            )
        started = perf_counter()
        for _ in range(config.latency_iterations):
            _forward_with_metadata(model, batch, batch_metadata)
        elapsed = perf_counter() - started
        return elapsed * 1_000.0 / config.latency_iterations, 0.0


def _binary_auroc(scores: Tensor, targets: Tensor) -> float:
    positives = int(targets.sum().item())
    negatives = targets.numel() - positives
    if positives == 0 or negatives == 0:
        return math.nan
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order].to(torch.float64)
    true_positive = torch.cat((torch.zeros(1), sorted_targets.cumsum(0))) / positives
    false_positive = (
        torch.cat((torch.zeros(1), (~targets[order]).to(torch.float64).cumsum(0))) / negatives
    )
    return float(torch.trapezoid(true_positive, false_positive).item())


def _binary_average_precision(scores: Tensor, targets: Tensor) -> float:
    positives = int(targets.sum().item())
    if positives == 0:
        return math.nan
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order].to(torch.float64)
    true_positive = sorted_targets.cumsum(0)
    precision = true_positive / torch.arange(1, targets.numel() + 1, dtype=torch.float64)
    return float((precision * sorted_targets).sum().item() / positives)


def _common_row(
    task: ExternalTask,
    target_parameters: int,
    match: ModelMatch,
) -> JsonRow:
    return {
        "dataset": task.name,
        "objective": task.objective,
        "model": match.family,
        "matched_width": match.width,
        "params_trainable": match.parameters,
        "target_params": target_parameters,
        "relative_param_error": match.relative_error,
        "train_samples": task.train_inputs.shape[0],
        "validation_samples": task.validation_inputs.shape[0],
        "test_samples": task.test_inputs.shape[0],
        "sequence_length": task.sequence_length,
        "input_dim": task.input_dim,
        "input_encoding": task.input_encoding,
        "vocab_size": task.vocab_size,
        "output_dim": task.output_dim,
    }


def _match_row(
    task: ExternalTask,
    target_parameters: int,
    match: ModelMatch,
) -> JsonRow:
    return {
        **_common_row(task, target_parameters, match),
        "status": match.status,
        "reason": match.reason,
    }


def _unavailable_dataset_row(name: str, error: str) -> JsonRow:
    return {
        "dataset": name,
        "status": "unavailable",
        "error": error,
    }


def _unavailable_model_row(
    task: ExternalTask,
    target_parameters: int,
    match: ModelMatch,
) -> JsonRow:
    return {
        **_common_row(task, target_parameters, match),
        "status": "unavailable",
        "error": match.reason,
    }


def _write_report(
    config: ExternalBenchmarkConfig,
    device: str,
    rows: list[JsonRow],
    matches: list[JsonRow],
) -> None:
    done = [row for row in rows if row.get("status") == "done"]
    unavailable = [row for row in rows if row.get("status") == "unavailable"]
    failed = [row for row in rows if row.get("status") == "failed"]
    lines = [
        "# PAC External Task Comparisons",
        "",
        f"- PAC model: `{config.pac_model}`",
        f"- selected width: `D={config.model_dim}, M={config.modes}`",
        f"- device: `{device}`",
        f"- seeds: `{', '.join(map(str, config.seeds))}`",
        f"- completed runs: `{len(done)}`",
        f"- unavailable rows: `{len(unavailable)}`",
        f"- failed rows: `{len(failed)}`",
        "",
        "## Mean Results",
        "",
        "| Dataset | Objective | Model | Params | Primary | Secondary | Latency ms | Runs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted({(str(row["dataset"]), str(row["model"])) for row in done}):
        group = [row for row in done if (str(row["dataset"]), str(row["model"])) == key]
        objective = str(group[0]["objective"])
        primary_name, secondary_name = _metric_names(objective, key[0])
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                key[0],
                objective,
                key[1],
                group[0].get("params_trainable", ""),
                _mean_display(group, primary_name),
                _mean_display(group, secondary_name),
                _mean_display(group, "latency_ms"),
                len(group),
            )
        )
    lines.extend(
        (
            "",
            "## Protocol Contract",
            "",
            "- PAC uses one frozen selected architecture; no structural ablation is run here.",
            "- Baseline width minimizes total trainable-parameter error against PAC per task.",
            "- All families share task tensors, splits, optimizer budget, early stopping, and seeds.",  # noqa: E501
            "- Discrete tasks share one learned token embedding and sinusoidal position adapter; retrieval uses one shared dual encoder per family.",  # noqa: E501
            "- Checkpoints are selected by validation loss; test data is never used for selection.",
            "- PTB-XL uses patient-respecting folds 1-8/9/10 and five diagnostic superclasses.",
            "- CWRU requires an explicit group-disjoint manifest; random window splitting is rejected.",  # noqa: E501
            "- Forecasting normalization is fitted on the training interval only.",
            "- Missing optional datasets or model packages remain explicit unavailable rows.",
            "",
            "## Unavailable Or Failed",
            "",
        )
    )
    lines.extend(
        f"- `{row.get('dataset')}` / `{row.get('model', 'dataset')}`: {row.get('error', '')}"
        for row in (*unavailable, *failed)
    )
    lines.extend(("", f"Parameter match rows: `{len(matches)}`", ""))
    report = config.output_root / "reports" / "external_comparisons.md"
    report.write_text("\n".join(lines), encoding="utf-8")


def _metric_names(objective: str, dataset: str = "") -> tuple[str, str]:
    if objective == "multiclass":
        return "accuracy", "macro_f1"
    if objective == "multilabel":
        if dataset == "audioset-balanced":
            return "macro_auprc", "macro_auroc"
        return "macro_auroc", "macro_auprc"
    return "mse", "mae"


def _mean_display(rows: list[JsonRow], key: str) -> str:
    values = [_required_float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return "" if not values else f"{sum(values) / len(values):.6f}"


def _attention_heads(width: int) -> int:
    for heads in (4, 2):
        if width % heads == 0:
            return heads
    return 1


def _relative_error(value: int, target: int) -> float:
    return abs(value - target) / max(target, 1)


def _match_key(match: ModelMatch) -> tuple[float, float]:
    return (
        math.inf if match.relative_error is None else match.relative_error,
        math.inf if match.parameters is None else match.parameters,
    )


def _mean_or_nan(values: list[float]) -> float:
    return math.nan if not values else sum(values) / len(values)


def _required_float(value: object) -> float:
    if isinstance(value, (float, int, str)):
        return float(value)
    message = f"expected numeric CSV value, got {type(value).__name__}"
    raise TypeError(message)


def _post_optimizer_step(model: nn.Module) -> None:
    _call_constraint_callback(model, "post_optimizer_step")


def _finalize_constraints(model: nn.Module) -> None:
    _call_constraint_callback(model, "finalize_constraints")


def _call_constraint_callback(model: nn.Module, name: str) -> None:
    callback = getattr(model, name, None)
    if callable(callback):
        callback()


def _seed_everything(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _release_device(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def _validate_config(config: ExternalBenchmarkConfig) -> None:
    if config.model_dim < 4 or config.modes < 1:
        raise ValueError("model_dim must be at least 4 and modes must be positive")
    if config.epochs < 1 or config.batch_size < 1 or config.patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.latency_warmup < 0 or config.latency_iterations < 1:
        raise ValueError("latency warmup must be nonnegative and iterations positive")
    if config.max_baseline_width < 1:
        raise ValueError("max_baseline_width must be positive")
    if not 0.0 <= config.parameter_match_tolerance < 1.0:
        raise ValueError("parameter_match_tolerance must be in [0, 1)")
    task_lengths = (
        config.mitbih_beat_length,
        config.cwru_window_length,
        config.forecast_context_length,
        config.prediction_length,
    )
    if any(length < 1 for length in task_lengths):
        raise ValueError("all task-specific sequence lengths must be positive")
