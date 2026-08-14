# pyright: reportPrivateUsage=false
"""Raw, same-stack EFP16 training-stage benchmark for the final RTX 4090 audit.

The three reported stages are intentionally different execution contracts:

* ``eager`` is the actual campaign FP32 model, default AdamW, and direct
  autograd path.
* ``previous_best`` replays the frozen absolute-ceiling dispatch with
  ``candidate=True, post_ceiling=False``.
* ``final_best`` adds the post-ceiling shape dispatch with
  ``candidate=True, post_ceiling=True``.

Only the optimized *full train step* is captured.  Standalone forward and
forward+backward measurements use the stage-configured model directly through
autograd, because replaying an inference graph would not measure a training
phase.  Model construction, autotuning, extension compilation, and graph
capture are setup costs and are excluded from every latency sample.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_campaign_utils import canonical_json_sha256
from .pac_efp16_exact_split_training import EFP16ExactSplitTraining
from .pac_metrics import count_parameters
from .pac_training_absolute_benchmark import (
    _EFP_STEM_STRATEGY_DISPATCH,
    _IDENTITY_ELISION_CELLS,
    _MODE_STATIC_POLE_CELLS,
    _RECURRENCE_MODE_DISPATCH,
    _block_modes,
    _build_absolute_context,
    _split_backward,
    _split_backward_modes,
)
from .pac_training_cuda_graph_benchmark import CURRENT_BACKENDS
from .pac_training_exact_split_benchmark import DEFAULT_CONFIG as CAMPAIGN_CONFIG
from .pac_training_speed_comparison import (
    EAGER_BACKEND,
    _configure_backend,
    _post_optimizer_step,
    build_training_model,
)
from .pac_training_ultimate_benchmark import _TrainingContext

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence

    from .pac_training_ultimate_benchmark import _FullGraphTrainingContext

Stage = Literal["eager", "previous_best", "final_best"]
Phase = Literal["forward", "forward_backward", "full"]
Cell = tuple[Literal["efp16"], int, int]

SCHEMA: Final = "pac_efp16_final_training_raw.v1"
MEMORY_SCHEMA: Final = "pac_efp16_final_training_memory.v1"
STAGES: Final[tuple[Stage, ...]] = ("eager", "previous_best", "final_best")
PHASES: Final[tuple[Phase, ...]] = ("forward", "forward_backward", "full")
LENGTHS: Final = (128, 512, 2048)
BATCHES: Final = (1, 64)
MINIMUM_PARITY_STEPS: Final = 75
EXPECTED_PARAMETERS: Final = 5_989


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    warmups: int = 5
    groups: int = 7
    iterations_per_group: int = 10
    parity_steps: int = MINIMUM_PARITY_STEPS
    seed: int = 7
    gpu_clock_ramp_cycles: int = 2_000_000_000
    gpu_clock_precondition_cycles: int = 20_000_000


DEFAULT_CONFIG: Final = BenchmarkConfig()


class _PhaseRuntime(Protocol):
    model: nn.Module
    backend: str
    setup_seconds: float

    def step(self) -> Tensor: ...

    def reset_seed(self, seed: int) -> None: ...

    def prepare_direct_model_readout(self) -> None: ...


class _DirectAutogradRuntime:
    """A direct training-mode phase; no CUDA Graph replay is hidden here."""

    def __init__(
        self,
        model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        phase: Phase,
        stage: Stage,
        cell: Cell,
        *,
        setup_seconds: float,
    ) -> None:
        self.model = model.train()
        self.inputs = inputs
        self.labels = labels
        self.phase = phase
        self.stage: Stage = stage
        self.cell = cell
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CAMPAIGN_CONFIG.learning_rate,
            weight_decay=CAMPAIGN_CONFIG.weight_decay,
        )
        self.backend = (
            "campaign_direct_autograd_default_adamw"
            if stage == "eager"
            else f"{stage}_configured_direct_autograd"
        )
        self.setup_seconds = setup_seconds

    def step(self) -> Tensor:
        with _stage_environment(self.stage, self.cell):
            if self.phase != "forward":
                self.optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(self.model(self.inputs), self.labels)
            if self.phase == "forward":
                return loss
            loss.backward()
            if self.phase == "full":
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), CAMPAIGN_CONFIG.grad_clip_norm
                )
                self.optimizer.step()
                _post_optimizer_step(self.model)
            return loss

    def reset_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_direct_model_readout(self) -> None:
        return


class _CapturedFullStepRuntime:
    """Adapter around the existing frozen/final absolute training contexts."""

    def __init__(
        self,
        context: _TrainingContext | _FullGraphTrainingContext,
        stage: Stage,
        cell: Cell,
        *,
        setup_seconds: float,
    ) -> None:
        self.context = context
        self.model = context.model
        self.stage: Stage = stage
        self.cell = cell
        self.backend = (
            "absolute_context_candidate_post_ceiling_false"
            if stage == "previous_best"
            else "absolute_context_candidate_post_ceiling_true"
        )
        self.setup_seconds = setup_seconds

    def step(self) -> Tensor:
        # The graph/exact-split runtime was captured under this same dispatch.
        with _stage_environment(self.stage, self.cell):
            return self.context.step()

    def reset_seed(self, seed: int) -> None:
        self.context.reset_seed(seed)

    def prepare_direct_model_readout(self) -> None:
        if isinstance(self.context, _TrainingContext) and isinstance(
            self.context.runtime, EFP16ExactSplitTraining
        ):
            self.context.runtime.close()


def benchmark(
    frozen_baseline: dict[str, object],
    *,
    baseline_path: Path | None = None,
    lengths: tuple[int, ...] = LENGTHS,
    batches: tuple[int, ...] = BATCHES,
    config: BenchmarkConfig = DEFAULT_CONFIG,
    isolated_memory: bool = True,
    memory_dir: Path | None = None,
) -> dict[str, object]:
    """Measure all requested EFP16 stages and shapes on one CUDA software stack."""
    _validate_runtime(lengths, batches, config)
    if isolated_memory and baseline_path is None:
        message = "baseline_path is required for isolated memory subprocesses"
        raise ValueError(message)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen_index = _index_frozen_baseline(frozen_baseline, lengths=lengths, batches=batches)
    _precondition_gpu_clock(config.gpu_clock_ramp_cycles)

    rows: list[dict[str, object]] = []
    architectures: dict[str, object] | None = None
    for length in lengths:
        for batch_size in batches:
            cell: Cell = ("efp16", length, batch_size)
            cell_rows, architecture = _benchmark_cell(
                cell,
                frozen_index[cell],
                config=config,
                baseline_path=baseline_path,
                isolated_memory=isolated_memory,
                memory_dir=memory_dir,
            )
            if architectures is None:
                architectures = architecture
            elif _architecture_identity(architectures) != _architecture_identity(architecture):
                message = "EFP16 architecture changed across benchmark shapes"
                raise RuntimeError(message)
            rows.extend(cell_rows)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "schema": SCHEMA,
        "environment": {
            "host": platform.node(),
            "device": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allow_tf32": False,
        },
        "protocol": _protocol(isolated_memory=isolated_memory),
        "config": asdict(config),
        "stages": list(STAGES),
        "lengths": list(lengths),
        "batches": list(batches),
        "architecture": architectures,
        "baseline_schema": frozen_baseline.get("schema"),
        "baseline_sha256": _json_digest(frozen_baseline),
        "rows": rows,
        "summary": summarize(rows),
    }


def _benchmark_cell(
    cell: Cell,
    frozen_row: dict[str, object],
    *,
    config: BenchmarkConfig,
    baseline_path: Path | None,
    isolated_memory: bool,
    memory_dir: Path | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    state_dict, cpu_inputs, cpu_labels, architecture = _make_cell_inputs(cell, config.seed)
    provenance = {
        "initial_state_sha256": _tensor_mapping_digest(state_dict),
        "input_sha256": _tensor_digest(cpu_inputs),
        "label_sha256": _tensor_digest(cpu_labels),
    }
    accuracy_by_stage: dict[Stage, dict[str, object]] = {
        "eager": _zero_accuracy(config.parity_steps)
    }
    for stage in cast("tuple[Stage, ...]", ("previous_best", "final_best")):
        accuracy_by_stage[stage] = _measure_accuracy(
            stage,
            cell,
            frozen_row,
            state_dict,
            cpu_inputs,
            cpu_labels,
            config=config,
        )
        _release_cuda()

    rows: list[dict[str, object]] = []
    for stage in STAGES:
        measurements: dict[Phase, dict[str, object]] = {}
        setup_seconds: dict[Phase, float] = {}
        phase_backends: dict[Phase, str] = {}
        for phase in PHASES:
            runtime = _build_runtime(
                stage,
                phase,
                cell,
                frozen_row,
                state_dict,
                cpu_inputs,
                cpu_labels,
            )
            measurements[phase] = _measure_phase(runtime, config=config)
            setup_seconds[phase] = runtime.setup_seconds
            phase_backends[phase] = runtime.backend
            del runtime
            _release_cuda()
        memory = _measure_memory(
            stage,
            cell,
            frozen_row,
            state_dict,
            cpu_inputs,
            cpu_labels,
            config=config,
            baseline_path=baseline_path,
            isolated=isolated_memory,
            memory_dir=memory_dir,
        )
        rows.append(
            _assemble_row(
                stage,
                cell,
                frozen_row,
                architecture,
                measurements,
                setup_seconds,
                phase_backends,
                accuracy_by_stage[stage],
                provenance,
                memory,
            )
        )
    return rows, architecture


def _make_cell_inputs(
    cell: Cell, seed: int
) -> tuple[dict[str, Tensor], Tensor, Tensor, dict[str, object]]:
    _, length, batch_size = cell
    torch.manual_seed(seed)
    model, architecture = build_training_model("efp16", length, batch_size)
    parameters = count_parameters(model)
    if parameters != EXPECTED_PARAMETERS:
        message = f"unexpected EFP16 parameter count: {parameters}"
        raise RuntimeError(message)
    state_dict = copy.deepcopy(model.state_dict())
    generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * length + 9176 * batch_size)
    cpu_inputs = torch.randn(batch_size, length, 1, generator=generator, dtype=torch.float32)
    cpu_labels = torch.randint(0, 5, (batch_size,), generator=generator, dtype=torch.long)
    del model
    return state_dict, cpu_inputs, cpu_labels, architecture


def _build_runtime(
    stage: Stage,
    phase: Phase,
    cell: Cell,
    frozen_row: dict[str, object],
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
) -> _PhaseRuntime:
    started = perf_counter()
    if stage != "eager" and phase == "full":
        context = _build_absolute_context(
            cell,
            frozen_row,
            copy.deepcopy(state_dict),
            cpu_inputs,
            cpu_labels,
            candidate=True,
            post_ceiling=stage == "final_best",
        )
        return _CapturedFullStepRuntime(
            context,
            stage,
            cell,
            setup_seconds=perf_counter() - started,
        )

    _, length, batch_size = cell
    model, _ = build_training_model("efp16", length, batch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device="cuda", dtype=torch.float32).train()
    if stage == "eager":
        _configure_backend(model, EAGER_BACKEND)
    else:
        _configure_optimized_model(model, stage, cell)
    inputs = cpu_inputs.to(device="cuda", dtype=torch.float32)
    labels = cpu_labels.to(device="cuda", dtype=torch.long)
    return _DirectAutogradRuntime(
        model,
        inputs,
        labels,
        phase,
        stage,
        cell,
        setup_seconds=perf_counter() - started,
    )


def _configure_optimized_model(model: nn.Module, stage: Stage, cell: Cell) -> None:
    backend = CURRENT_BACKENDS[cell]
    _configure_backend(model, backend)
    blocks = (
        getattr(model, "forward_block", None),
        getattr(model, "backward_block", None),
        *getattr(model, "extra_blocks", []),
    )
    for block_index, block in enumerate(blocks):
        if block is None:
            continue
        block.canonical_identity_elision = cell in _IDENTITY_ELISION_CELLS
        block.mode_static_pole_training = cell in _MODE_STATIC_POLE_CELLS
        block.packed_recurrence_moments_training = None
        block.two_pass_reverse_recurrence_moments_training = None
        block.__dict__["fused_recurrence_moments_backward_training"] = (
            stage == "final_best" and block_index == 1 if cell == ("efp16", 2048, 64) else True
        )
    if hasattr(model, "efp16_stem_parameter_gradient_strategy"):
        model.__dict__["efp16_stem_parameter_gradient_strategy"] = _EFP_STEM_STRATEGY_DISPATCH.get(
            cell, "auto"
        )
    if hasattr(model, "use_fused_rmsnorm_mean_training"):
        model.__dict__["use_fused_rmsnorm_mean_training"] = False
    if hasattr(model, "use_fused_rmsnorm_mean_backward_training"):
        model.__dict__["use_fused_rmsnorm_mean_backward_training"] = False
    if stage not in {"previous_best", "final_best"}:
        message = f"unsupported optimized stage: {stage}"
        raise ValueError(message)


@contextmanager
def _stage_environment(stage: Stage, cell: Cell) -> Generator[None]:
    if stage == "eager":
        yield
        return
    block_modes = _RECURRENCE_MODE_DISPATCH[cell]
    split_modes = _split_backward_modes(cell) if stage == "final_best" else None
    with _block_modes(block_modes), _split_backward(split_modes):
        yield


def _measure_phase(runtime: _PhaseRuntime, *, config: BenchmarkConfig) -> dict[str, object]:
    loss = torch.zeros((), device="cuda")
    for warmup in range(config.warmups):
        runtime.reset_seed(config.seed + 50_000 + warmup)
        loss = runtime.step()
    torch.cuda.synchronize()
    wall_samples: list[float] = []
    event_samples: list[float] = []
    for group in range(config.groups):
        runtime.reset_seed(config.seed + 60_000 + group * config.iterations_per_group)
        _precondition_gpu_clock(config.gpu_clock_precondition_cycles)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = perf_counter()
        start_event.record()
        for _ in range(config.iterations_per_group):
            loss = runtime.step()
        end_event.record()
        end_event.synchronize()
        wall_samples.append((perf_counter() - started) * 1000.0 / config.iterations_per_group)
        event_samples.append(start_event.elapsed_time(end_event) / config.iterations_per_group)
    return _sample_summary(wall_samples, event_samples, loss)


def _sample_summary(
    wall_samples: Sequence[float], event_samples: Sequence[float], loss: Tensor
) -> dict[str, object]:
    wall_quartiles = statistics.quantiles(wall_samples, n=4, method="inclusive")
    event_quartiles = statistics.quantiles(event_samples, n=4, method="inclusive")
    return {
        "wall_ms": statistics.median(wall_samples),
        "wall_iqr_ms": wall_quartiles[2] - wall_quartiles[0],
        "wall_samples_ms": list(wall_samples),
        "cuda_event_ms": statistics.median(event_samples),
        "cuda_event_iqr_ms": event_quartiles[2] - event_quartiles[0],
        "cuda_event_samples_ms": list(event_samples),
        "last_loss": float(loss.detach().item()),
    }


def _measure_accuracy(
    stage: Stage,
    cell: Cell,
    frozen_row: dict[str, object],
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    *,
    config: BenchmarkConfig,
) -> dict[str, object]:
    reference = _build_runtime(
        "eager", "full", cell, frozen_row, state_dict, cpu_inputs, cpu_labels
    )
    candidate = _build_runtime(stage, "full", cell, frozen_row, state_dict, cpu_inputs, cpu_labels)
    reference.reset_seed(config.seed + 100_000)
    reference_losses = [reference.step().detach().clone() for _ in range(config.parity_steps)]
    torch.cuda.synchronize()
    candidate.reset_seed(config.seed + 100_000)
    candidate_losses = [candidate.step().detach().clone() for _ in range(config.parity_steps)]
    torch.cuda.synchronize()
    reference.prepare_direct_model_readout()
    candidate.prepare_direct_model_readout()
    result = compare_training_states(
        reference.model,
        candidate.model,
        reference_losses,
        candidate_losses,
        cpu_inputs.to(device="cuda", dtype=torch.float32),
    )
    result["parity_steps"] = config.parity_steps
    result["reference_stage"] = "eager"
    result["threshold_enforced_by_harness"] = False
    del reference, candidate, reference_losses, candidate_losses
    return result


@torch.no_grad()
def compare_training_states(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    reference_losses: Sequence[Tensor],
    candidate_losses: Sequence[Tensor],
    inputs: Tensor,
) -> dict[str, object]:
    """Compare final training state and predictions without applying a pass threshold."""
    if len(reference_losses) != len(candidate_losses) or not reference_losses:
        message = "accuracy comparison requires equal non-empty loss trajectories"
        raise ValueError(message)
    reference_gradients = {
        name: parameter.grad
        for name, parameter in reference_model.named_parameters()
        if parameter.grad is not None
    }
    candidate_gradients = {
        name: parameter.grad
        for name, parameter in candidate_model.named_parameters()
        if parameter.grad is not None
    }
    reference_parameters = dict(reference_model.named_parameters())
    candidate_parameters = dict(candidate_model.named_parameters())
    gradient_abs, gradient_rel = _mapping_errors(reference_gradients, candidate_gradients)
    parameter_abs, parameter_rel = _mapping_errors(reference_parameters, candidate_parameters)
    reference_model.eval()
    candidate_model.eval()
    reference_logits = reference_model(inputs)
    candidate_logits = candidate_model(inputs)
    predictions = reference_logits.argmax(dim=-1)
    candidate_predictions = candidate_logits.argmax(dim=-1)
    return {
        "loss_trajectory_max_abs_error": float(
            (torch.stack(list(candidate_losses)) - torch.stack(list(reference_losses)))
            .abs()
            .max()
            .item()
        ),
        "final_gradient_key_agreement": set(reference_gradients) == set(candidate_gradients),
        "final_gradient_max_abs_error": gradient_abs,
        "final_gradient_max_rel_error": gradient_rel,
        "final_parameter_key_agreement": set(reference_parameters) == set(candidate_parameters),
        "final_parameter_max_abs_error": parameter_abs,
        "final_parameter_max_rel_error": parameter_rel,
        "parameter_update_max_abs_error": parameter_abs,
        "prediction_agreement": float((predictions == candidate_predictions).float().mean().item()),
        "final_logit_max_abs_error": float(
            (reference_logits - candidate_logits).abs().max().item()
        ),
    }


def _mapping_errors(
    reference: Mapping[str, Tensor], candidate: Mapping[str, Tensor]
) -> tuple[float, float]:
    if set(reference) != set(candidate):
        return math.inf, math.inf
    maximum_abs = 0.0
    maximum_rel = 0.0
    for name, reference_value in reference.items():
        difference = (candidate[name] - reference_value).abs()
        relative = difference / reference_value.abs().clamp_min(1.0e-6)
        maximum_abs = max(maximum_abs, float(difference.max().item()))
        maximum_rel = max(maximum_rel, float(relative.max().item()))
    return maximum_abs, maximum_rel


def _zero_accuracy(parity_steps: int) -> dict[str, object]:
    return {
        "parity_steps": parity_steps,
        "reference_stage": "eager",
        "loss_trajectory_max_abs_error": 0.0,
        "final_gradient_key_agreement": True,
        "final_gradient_max_abs_error": 0.0,
        "final_gradient_max_rel_error": 0.0,
        "final_parameter_key_agreement": True,
        "final_parameter_max_abs_error": 0.0,
        "final_parameter_max_rel_error": 0.0,
        "parameter_update_max_abs_error": 0.0,
        "prediction_agreement": 1.0,
        "final_logit_max_abs_error": 0.0,
        "threshold_enforced_by_harness": False,
    }


def _measure_memory(
    stage: Stage,
    cell: Cell,
    frozen_row: dict[str, object],
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    *,
    config: BenchmarkConfig,
    baseline_path: Path | None,
    isolated: bool,
    memory_dir: Path | None,
) -> dict[str, object]:
    if not isolated:
        runtime = _build_runtime(
            stage, "full", cell, frozen_row, state_dict, cpu_inputs, cpu_labels
        )
        result = _measure_runtime_peak_memory(runtime)
        del runtime
        _release_cuda()
        return {
            "peak_memory_mb": result,
            "memory_isolated_process": False,
            "memory_artifact": None,
        }
    if baseline_path is None:
        message = "baseline_path is required for isolated memory"
        raise ValueError(message)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    target_dir = memory_dir
    if target_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="pac-efp16-final-training-memory-")
        target_dir = Path(temporary.name)
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / f"efp16-N{cell[1]}-B{cell[2]}-{stage}.json"
    command = [
        sys.executable,
        "-m",
        "lnet.pac_efp16_final_training_benchmark",
        "memory",
        "--baseline",
        str(baseline_path),
        "--output",
        str(output),
        "--stage",
        stage,
        "--length",
        str(cell[1]),
        "--batch-size",
        str(cell[2]),
        "--seed",
        str(config.seed),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        if temporary is not None:
            temporary.cleanup()
        message = f"isolated memory subprocess failed: {detail}"
        raise RuntimeError(message)
    payload = cast("dict[str, object]", json.loads(output.read_text()))
    result = {
        "peak_memory_mb": payload["peak_memory_mb"],
        "memory_isolated_process": True,
        "memory_artifact": str(output) if temporary is None else None,
    }
    if temporary is not None:
        temporary.cleanup()
    return result


def measure_peak_memory(
    frozen_baseline: dict[str, object],
    stage: Stage,
    length: int,
    batch_size: int,
    *,
    seed: int = DEFAULT_CONFIG.seed,
) -> dict[str, object]:
    """Measure one full-step runtime in a fresh process invoked by ``memory``."""
    _validate_runtime((length,), (batch_size,), BenchmarkConfig(seed=seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen_index = _index_frozen_baseline(frozen_baseline, lengths=(length,), batches=(batch_size,))
    cell: Cell = ("efp16", length, batch_size)
    state_dict, cpu_inputs, cpu_labels, _ = _make_cell_inputs(cell, seed)
    runtime = _build_runtime(
        stage,
        "full",
        cell,
        frozen_index[cell],
        state_dict,
        cpu_inputs,
        cpu_labels,
    )
    peak = _measure_runtime_peak_memory(runtime)
    return {
        "schema": MEMORY_SCHEMA,
        "stage": stage,
        "length": length,
        "batch_size": batch_size,
        "peak_memory_mb": peak,
        "isolated_process": True,
        "pid": os.getpid(),
        "environment": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allow_tf32": False,
        },
    }


def _measure_runtime_peak_memory(runtime: _PhaseRuntime) -> float:
    runtime.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    runtime.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def _assemble_row(
    stage: Stage,
    cell: Cell,
    frozen_row: dict[str, object],
    architecture: dict[str, object],
    measurements: dict[Phase, dict[str, object]],
    setup_seconds: dict[Phase, float],
    phase_backends: dict[Phase, str],
    accuracy: dict[str, object],
    provenance: dict[str, str],
    memory: dict[str, object],
) -> dict[str, object]:
    full_wall = _as_float(measurements["full"]["wall_ms"])
    row: dict[str, object] = {
        "model": "efp16",
        "display_name": "EFP16",
        "stage": stage,
        "length": cell[1],
        "batch_size": cell[2],
        "status": "measured",
        "parameters": _as_int(architecture["trainable_parameters"]),
        "phase_backends": phase_backends,
        "setup_seconds_by_phase": setup_seconds,
        "setup_seconds": sum(setup_seconds.values()),
        "compile_and_capture_cost_included": False,
        "normalized": False,
        "sequences_per_second": cell[2] * 1000.0 / full_wall,
        "tokens_per_second": cell[1] * cell[2] * 1000.0 / full_wall,
        "accuracy": accuracy,
        "provenance": provenance,
        "frozen_row": {
            "selected_runtime": frozen_row.get("selected_runtime"),
            "selected_wall_ms_historical_only": frozen_row.get("selected_wall_ms"),
            "graph_matrix_exp_compute_dtype": frozen_row.get("graph_matrix_exp_compute_dtype"),
        },
        **memory,
    }
    for phase in PHASES:
        prefix = "full_train_step" if phase == "full" else phase
        for key, value in measurements[phase].items():
            if key != "last_loss":
                row[f"{prefix}_{key}"] = value
    row["last_timed_loss"] = measurements["full"]["last_loss"]
    return row


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for stage in STAGES:
        stage_rows = [row for row in rows if row.get("stage") == stage]
        if not stage_rows:
            continue
        eager_index = {
            (_as_int(row["length"]), _as_int(row["batch_size"])): row
            for row in rows
            if row.get("stage") == "eager"
        }
        speedups = [
            _as_float(
                eager_index[(_as_int(row["length"]), _as_int(row["batch_size"]))][
                    "full_train_step_wall_ms"
                ]
            )
            / _as_float(row["full_train_step_wall_ms"])
            for row in stage_rows
        ]
        summary[stage] = {
            "shape_count": len(stage_rows),
            "geometric_mean_full_train_step_wall_ms": _geometric_mean(
                [_as_float(row["full_train_step_wall_ms"]) for row in stage_rows]
            ),
            "geometric_mean_speedup_vs_eager": _geometric_mean(speedups),
            "maximum_peak_memory_mb": max(_as_float(row["peak_memory_mb"]) for row in stage_rows),
        }
    return summary


def _protocol(*, isolated_memory: bool) -> dict[str, object]:
    return {
        "hardware": "NVIDIA GeForce RTX 4090 (enforced by benchmark)",
        "dtype": "float32",
        "tf32": False,
        "autocast": False,
        "compile_and_capture_cost_included": False,
        "normalized_latency": False,
        "timing": "raw CUDA event and synchronized wall samples; median of per-group means",
        "forward": "training-mode direct configured model forward plus cross-entropy",
        "forward_backward": (
            "direct configured model zero_grad + forward + cross-entropy + backward; "
            "no inference graph and no CUDA Graph replay"
        ),
        "full_train_step": (
            "zero_grad + forward + cross-entropy + backward + clip_grad_norm_ + AdamW + "
            "post_optimizer_step"
        ),
        "eager": "actual campaign recurrence auto path and default AdamW",
        "previous_best": (
            "standalone phases use configured direct autograd; full step reuses "
            "_build_absolute_context(candidate=True, post_ceiling=False)"
        ),
        "final_best": (
            "standalone phases use configured direct autograd; full step reuses "
            "_build_absolute_context(candidate=True, post_ceiling=True)"
        ),
        "accuracy": (
            "at least 75 consecutive updates from identical initial state/input/labels; "
            "record loss trajectory, final gradients, final parameters, logits, and predictions"
        ),
        "accuracy_threshold_enforced_by_harness": False,
        "peak_memory": (
            "fresh subprocess, one warm full step then one measured full step"
            if isolated_memory
            else "fresh in-process runtime after cache release"
        ),
    }


def _index_frozen_baseline(
    payload: dict[str, object], *, lengths: Iterable[int], batches: Iterable[int]
) -> dict[Cell, dict[str, object]]:
    rows = cast("list[dict[str, object]]", payload.get("rows", []))
    indexed: dict[Cell, dict[str, object]] = {}
    for row in rows:
        if row.get("model") != "efp16":
            continue
        cell: Cell = ("efp16", _as_int(row.get("length")), _as_int(row.get("batch_size")))
        if cell in indexed:
            message = f"duplicate frozen EFP16 row: {cell}"
            raise ValueError(message)
        indexed[cell] = row
    required: set[Cell] = {("efp16", length, batch) for length in lengths for batch in batches}
    missing = sorted(required - set(indexed))
    if missing:
        message = f"frozen baseline is missing EFP16 cells: {missing}"
        raise ValueError(message)
    for cell in required:
        if not indexed[cell].get("graph_matrix_exp_compute_dtype"):
            message = f"frozen baseline lacks graph compute dtype: {cell}"
            raise ValueError(message)
    return {cell: indexed[cell] for cell in required}


def _validate_runtime(
    lengths: tuple[int, ...], batches: tuple[int, ...], config: BenchmarkConfig
) -> None:
    if not torch.cuda.is_available():
        message = "EFP16 final training benchmark requires CUDA"
        raise RuntimeError(message)
    device_name = torch.cuda.get_device_name()
    if "4090" not in device_name:
        message = f"EFP16 final training benchmark requires RTX 4090, found {device_name}"
        raise RuntimeError(message)
    if not lengths or not batches or set(lengths) - set(LENGTHS) or set(batches) - set(BATCHES):
        message = "lengths and batches must be non-empty subsets of the six canonical shapes"
        raise ValueError(message)
    if min(config.warmups, config.iterations_per_group) < 1 or config.groups < 3:
        message = "warmups/iterations must be positive and groups must be at least three"
        raise ValueError(message)
    if config.parity_steps < MINIMUM_PARITY_STEPS:
        message = f"parity_steps must be at least {MINIMUM_PARITY_STEPS}"
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


def _tensor_digest(tensor: Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_mapping_digest(values: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(_tensor_digest(value).encode())
    return digest.hexdigest()


def _json_digest(payload: dict[str, object]) -> str:
    return canonical_json_sha256(payload)


def _architecture_identity(architecture: dict[str, object]) -> str:
    stable = {key: value for key, value in architecture.items() if key != "state_dict_sha256"}
    return json.dumps(stable, sort_keys=True)


def _geometric_mean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _as_int(value: object) -> int:
    try:
        return int(cast("int | str", value))
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float:
    try:
        return float(cast("float | int | str", value))
    except (TypeError, ValueError):
        return 0.0


def _parse_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _config_from_args(arguments: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        warmups=arguments.warmups,
        groups=arguments.groups,
        iterations_per_group=arguments.iterations_per_group,
        parity_steps=arguments.parity_steps,
        seed=arguments.seed,
        gpu_clock_ramp_cycles=arguments.gpu_clock_ramp_cycles,
        gpu_clock_precondition_cycles=arguments.gpu_clock_precondition_cycles,
    )


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmups", type=int, default=DEFAULT_CONFIG.warmups)
    parser.add_argument("--groups", type=int, default=DEFAULT_CONFIG.groups)
    parser.add_argument(
        "--iterations-per-group", type=int, default=DEFAULT_CONFIG.iterations_per_group
    )
    parser.add_argument("--parity-steps", type=int, default=DEFAULT_CONFIG.parity_steps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
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
    benchmark_parser.add_argument("--baseline", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    benchmark_parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    benchmark_parser.add_argument("--memory-dir", type=Path)
    benchmark_parser.add_argument("--no-isolated-memory", action="store_true")
    _add_config_arguments(benchmark_parser)

    memory_parser = subparsers.add_parser("memory")
    memory_parser.add_argument("--baseline", type=Path, required=True)
    memory_parser.add_argument("--output", type=Path, required=True)
    memory_parser.add_argument("--stage", choices=STAGES, required=True)
    memory_parser.add_argument("--length", type=int, required=True)
    memory_parser.add_argument("--batch-size", type=int, required=True)
    memory_parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    baseline_path = cast("Path", arguments.baseline)
    frozen_baseline = cast("dict[str, object]", json.loads(baseline_path.read_text()))
    if arguments.command == "benchmark":
        result = benchmark(
            frozen_baseline,
            baseline_path=baseline_path,
            lengths=_parse_tuple(arguments.lengths),
            batches=_parse_tuple(arguments.batches),
            config=_config_from_args(arguments),
            isolated_memory=not arguments.no_isolated_memory,
            memory_dir=arguments.memory_dir,
        )
    else:
        result = measure_peak_memory(
            frozen_baseline,
            cast("Stage", arguments.stage),
            arguments.length,
            arguments.batch_size,
            seed=arguments.seed,
        )
    output = cast("Path", arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
