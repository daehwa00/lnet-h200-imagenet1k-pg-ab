# pyright: reportPrivateUsage=false
"""Strict actual-eager accuracy screen for EFP16 B64 training runtimes.

This screen exists because the historical optimized training campaigns used
an exact-split/graph reference, while the final paper comparison requires the
actual campaign eager FP32 path as its reference.  The distinction is
material: an optimization can agree with the former and still drift from the
latter over 75 updates.

Every candidate starts from the same state, input, and labels.  Accuracy is
reported before selection; selection requires loss trajectory, final gradient,
and final parameter errors <=2e-5 plus prediction agreement ==1.0.  Candidate
timings are paired with a fresh eager runtime using alternating group order.
Capture, compilation, and setup are excluded from raw wall/event samples.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_efp16_exact_split_training import (
    EFP16ExactSplitTraining,
    prepare_efp16_exact_split_training,
)
from .pac_efp16_final_training_benchmark import (
    LENGTHS,
    Cell,
    Stage,
    _configure_optimized_model,
    _index_frozen_baseline,
    _make_cell_inputs,
    _precondition_gpu_clock,
    _release_cuda,
    _sample_summary,
    _stage_environment,
    _tensor_digest,
    _tensor_mapping_digest,
    compare_training_states,
)
from .pac_efp16_final_training_benchmark import (
    _build_runtime as _build_final_runtime,
)
from .pac_efp16_training_cuda_graph import (
    EFP16TrainingCudaGraph,
    make_capturable_adamw,
    prepare_efp16_training_cuda_graph,
)
from .pac_training_absolute_benchmark import _EFP_STEM_STRATEGY_DISPATCH
from .pac_training_exact_split_benchmark import DEFAULT_CONFIG as CAMPAIGN_CONFIG
from .pac_training_speed_comparison import (
    EAGER_BACKEND,
    _configure_backend,
    _post_optimizer_step,
    build_training_model,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

CandidateName = Literal[
    "eager_reference",
    "direct_fused_stem",
    "direct_fused_moments",
    "direct_fused_stem_moments",
    "direct_tuned_autograd",
    "generic_cuda_graph_conservative",
    "exact_split_conservative",
    "exact_split_fused_recurrence",
    "exact_split_identity_static",
    "exact_split_specialized_vjp",
    "exact_split_shape_tuned",
    "previous_best",
    "final_best",
]
CandidateKind = Literal["eager", "direct", "generic_graph", "exact_split", "absolute"]

SCHEMA: Final = "pac_efp16_strict_training_screen.v1"
MAXIMUM_ERROR: Final = 2.0e-5
B64_BATCH: Final = 64


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    warmups: int = 5
    groups: int = 7
    iterations_per_group: int = 10
    parity_steps: int = 75
    seed: int = 7
    graph_warmups: int = 3
    gpu_clock_ramp_cycles: int = 2_000_000_000
    gpu_clock_precondition_cycles: int = 20_000_000


DEFAULT_CONFIG: Final = ScreenConfig()


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: CandidateName
    kind: CandidateKind
    description: str
    risky_features: tuple[str, ...] = ()
    direct_fused_stem: bool = False
    direct_fused_moments: bool = False
    direct_tuned: bool = False
    fused_recurrence: bool = False
    identity_static: bool = False
    specialized_vjp: bool = False
    shape_tuned: bool = False
    absolute_stage: Stage | None = None


CANDIDATES: Final[tuple[CandidateSpec, ...]] = (
    CandidateSpec(
        "eager_reference",
        "eager",
        "actual campaign eager FP32 direct autograd and default AdamW",
    ),
    CandidateSpec(
        "direct_fused_stem",
        "direct",
        "actual eager path with only the custom-autograd fused EFP16 stem enabled",
        ("fused_stem",),
        direct_fused_stem=True,
    ),
    CandidateSpec(
        "direct_fused_moments",
        "direct",
        "actual eager path with only fused online-moments backward enabled",
        ("fused_online_moments_backward",),
        direct_fused_moments=True,
    ),
    CandidateSpec(
        "direct_fused_stem_moments",
        "direct",
        "direct autograd with fused stem and fused online-moments backward",
        ("fused_stem", "fused_online_moments_backward"),
        direct_fused_stem=True,
        direct_fused_moments=True,
    ),
    CandidateSpec(
        "direct_tuned_autograd",
        "direct",
        "post-ceiling configured model executed directly without graph/exact-split machinery",
        (
            "fused_stem",
            "fused_online_moments_backward",
            "fused_recurrence_moments_backward",
            "identity_elision",
            "mode_static_pole",
            "shape_dispatch",
        ),
        direct_tuned=True,
    ),
    CandidateSpec(
        "generic_cuda_graph_conservative",
        "generic_graph",
        "generic full-step CUDA Graph with eager recurrence and fused recurrence disabled",
        ("capture_safe_matrix_exp", "capturable_fused_adamw", "fused_stem"),
    ),
    CandidateSpec(
        "exact_split_conservative",
        "exact_split",
        "native matrix-exp exact split; identity/static/fused recurrence/specialized VJP off",
        ("exact_split", "fused_stem", "fused_online_moments_backward"),
    ),
    CandidateSpec(
        "exact_split_fused_recurrence",
        "exact_split",
        "conservative exact split plus fused recurrence+moments backward",
        (
            "exact_split",
            "fused_stem",
            "fused_online_moments_backward",
            "fused_recurrence_moments_backward",
        ),
        fused_recurrence=True,
    ),
    CandidateSpec(
        "exact_split_identity_static",
        "exact_split",
        "fused-recurrence exact split plus identity elision and mode-static pole",
        (
            "exact_split",
            "fused_stem",
            "fused_recurrence_moments_backward",
            "identity_elision",
            "mode_static_pole",
        ),
        fused_recurrence=True,
        identity_static=True,
    ),
    CandidateSpec(
        "exact_split_specialized_vjp",
        "exact_split",
        "identity/static exact split plus specialized native matrix-exp VJP",
        (
            "exact_split",
            "fused_stem",
            "fused_recurrence_moments_backward",
            "identity_elision",
            "mode_static_pole",
            "specialized_matrix_exp_vjp",
        ),
        fused_recurrence=True,
        identity_static=True,
        specialized_vjp=True,
    ),
    CandidateSpec(
        "exact_split_shape_tuned",
        "exact_split",
        "specialized exact split plus absolute-style model flags, stem, and block-mode dispatch",
        (
            "exact_split",
            "fused_stem",
            "fused_recurrence_moments_backward",
            "identity_elision",
            "mode_static_pole",
            "specialized_matrix_exp_vjp",
            "shape_dispatch",
        ),
        fused_recurrence=True,
        identity_static=True,
        specialized_vjp=True,
        shape_tuned=True,
    ),
    CandidateSpec(
        "previous_best",
        "absolute",
        "frozen absolute context with candidate=True and post_ceiling=False",
        ("historical_previous_bundle",),
        absolute_stage="previous_best",
    ),
    CandidateSpec(
        "final_best",
        "absolute",
        "post-ceiling absolute context with candidate=True and post_ceiling=True",
        ("historical_final_bundle",),
        absolute_stage="final_best",
    ),
)
ALL_CANDIDATE_NAMES: Final[tuple[CandidateName, ...]] = tuple(spec.name for spec in CANDIDATES)


class _Runtime(Protocol):
    model: nn.Module
    backend: str
    setup_seconds: float

    def step(self) -> Tensor: ...

    def reset_seed(self, seed: int) -> None: ...

    def prepare_direct_model_readout(self) -> None: ...


class _DirectRuntime:
    def __init__(
        self,
        model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        spec: CandidateSpec,
        cell: Cell,
        *,
        setup_seconds: float,
    ) -> None:
        self.model = model.train()
        self.inputs = inputs
        self.labels = labels
        self.spec = spec
        self.cell = cell
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CAMPAIGN_CONFIG.learning_rate,
            weight_decay=CAMPAIGN_CONFIG.weight_decay,
        )
        self.backend = f"{spec.name}_direct_autograd_default_adamw"
        self.setup_seconds = setup_seconds

    def step(self) -> Tensor:
        environment_stage: Stage | None = "final_best" if self.spec.direct_tuned else None
        manager = (
            _stage_environment(environment_stage, self.cell)
            if environment_stage is not None
            else _null_environment()
        )
        with manager:
            self.optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(self.model(self.inputs), self.labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), CAMPAIGN_CONFIG.grad_clip_norm)
            self.optimizer.step()
            _post_optimizer_step(self.model)
            return loss

    def reset_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_direct_model_readout(self) -> None:
        return


class _GenericGraphRuntime:
    def __init__(
        self,
        model: nn.Module,
        runtime: EFP16TrainingCudaGraph,
        inputs: Tensor,
        labels: Tensor,
        *,
        setup_seconds: float,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.inputs = inputs
        self.labels = labels
        self.backend = "generic_cuda_graph_conservative"
        self.setup_seconds = setup_seconds

    def step(self) -> Tensor:
        return self.runtime.step(self.inputs, self.labels)

    def reset_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_direct_model_readout(self) -> None:
        return


class _ExactSplitRuntime:
    def __init__(
        self,
        model: nn.Module,
        runtime: EFP16ExactSplitTraining,
        inputs: Tensor,
        labels: Tensor,
        spec: CandidateSpec,
        cell: Cell,
        *,
        setup_seconds: float,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.inputs = inputs
        self.labels = labels
        self.spec = spec
        self.cell = cell
        self.backend = spec.name
        self.setup_seconds = setup_seconds

    def step(self) -> Tensor:
        manager = (
            _stage_environment("previous_best", self.cell)
            if self.spec.shape_tuned
            else _null_environment()
        )
        with manager:
            return self.runtime.step(self.inputs, self.labels)

    def reset_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_direct_model_readout(self) -> None:
        self.runtime.close()


class _NullEnvironment:
    def __enter__(self) -> None:
        return

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        return


def _null_environment() -> _NullEnvironment:
    return _NullEnvironment()


def benchmark(
    frozen_baseline: dict[str, object],
    *,
    lengths: tuple[int, ...] = LENGTHS,
    candidates: tuple[CandidateName, ...] = ALL_CANDIDATE_NAMES,
    config: ScreenConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Screen strict eager parity and raw paired latency for EFP16 B64."""
    _validate_runtime(lengths, candidates, config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen_index = _index_frozen_baseline(frozen_baseline, lengths=lengths, batches=(B64_BATCH,))
    requested = {spec.name: spec for spec in CANDIDATES if spec.name in candidates}
    rows: list[dict[str, object]] = []
    for length in lengths:
        cell: Cell = ("efp16", length, B64_BATCH)
        rows.extend(
            _benchmark_cell(
                cell,
                frozen_index[cell],
                tuple(requested[name] for name in candidates),
                config=config,
            )
        )
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
        "protocol": _protocol(),
        "config": asdict(config),
        "lengths": list(lengths),
        "batch_size": B64_BATCH,
        "candidates": [_candidate_payload(requested[name]) for name in candidates],
        "rows": rows,
        "summary": summarize(rows, lengths=lengths),
    }


def _benchmark_cell(
    cell: Cell,
    frozen_row: dict[str, object],
    candidates: tuple[CandidateSpec, ...],
    *,
    config: ScreenConfig,
) -> list[dict[str, object]]:
    state_dict, cpu_inputs, cpu_labels, _ = _make_cell_inputs(cell, config.seed)
    provenance = {
        "initial_state_sha256": _tensor_mapping_digest(state_dict),
        "input_sha256": _tensor_digest(cpu_inputs),
        "label_sha256": _tensor_digest(cpu_labels),
    }
    _precondition_gpu_clock(config.gpu_clock_ramp_cycles)
    reference = _build_candidate(
        _candidate_spec("eager_reference"),
        cell,
        frozen_row,
        state_dict,
        cpu_inputs,
        cpu_labels,
        config=config,
    )
    reference.reset_seed(config.seed + 100_000)
    reference_losses = [reference.step().detach().clone() for _ in range(config.parity_steps)]
    torch.cuda.synchronize()
    reference.prepare_direct_model_readout()

    rows: list[dict[str, object]] = []
    for spec in candidates:
        if spec.name == "eager_reference":
            timing_runtime = _build_candidate(
                spec,
                cell,
                frozen_row,
                state_dict,
                cpu_inputs,
                cpu_labels,
                config=config,
            )
            timing = _measure_single(timing_runtime, config=config)
            rows.append(
                _row(
                    spec,
                    cell,
                    _zero_accuracy(config.parity_steps),
                    timing,
                    timing,
                    timing_runtime.setup_seconds,
                    provenance,
                )
            )
            del timing_runtime
            _release_cuda()
            continue
        try:
            candidate = _build_candidate(
                spec,
                cell,
                frozen_row,
                state_dict,
                cpu_inputs,
                cpu_labels,
                config=config,
            )
            candidate.reset_seed(config.seed + 100_000)
            candidate_losses = [
                candidate.step().detach().clone() for _ in range(config.parity_steps)
            ]
            torch.cuda.synchronize()
            candidate.prepare_direct_model_readout()
            accuracy = compare_training_states(
                reference.model,
                candidate.model,
                reference_losses,
                candidate_losses,
                cpu_inputs.to(device="cuda", dtype=torch.float32),
            )
            accuracy.update(
                {
                    "parity_steps": config.parity_steps,
                    "reference_stage": "actual_campaign_eager",
                    "strict_pass": _accuracy_passes(accuracy),
                }
            )
            del candidate, candidate_losses
            _release_cuda()

            paired_eager = _build_candidate(
                _candidate_spec("eager_reference"),
                cell,
                frozen_row,
                state_dict,
                cpu_inputs,
                cpu_labels,
                config=config,
            )
            paired_candidate = _build_candidate(
                spec,
                cell,
                frozen_row,
                state_dict,
                cpu_inputs,
                cpu_labels,
                config=config,
            )
            paired = _measure_paired(paired_eager, paired_candidate, config=config)
            rows.append(
                _row(
                    spec,
                    cell,
                    accuracy,
                    paired["candidate"],
                    paired["eager"],
                    paired_candidate.setup_seconds,
                    provenance,
                )
            )
            del paired_eager, paired_candidate
        except Exception as error:  # noqa: BLE001 - preserve unsupported screen evidence
            rows.append(
                {
                    "model": "efp16",
                    "length": cell[1],
                    "batch_size": cell[2],
                    "candidate": spec.name,
                    "status": "unsupported",
                    "strict_pass": False,
                    "provenance": provenance,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        _release_cuda()
    del reference, reference_losses
    _release_cuda()
    return rows


def _build_candidate(
    spec: CandidateSpec,
    cell: Cell,
    frozen_row: dict[str, object],
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    *,
    config: ScreenConfig,
) -> _Runtime:
    if spec.kind == "eager":
        return _build_final_runtime(
            "eager", "full", cell, frozen_row, state_dict, cpu_inputs, cpu_labels
        )
    if spec.kind == "absolute":
        if spec.absolute_stage is None:
            message = f"absolute candidate lacks a stage: {spec.name}"
            raise ValueError(message)
        return _build_final_runtime(
            spec.absolute_stage,
            "full",
            cell,
            frozen_row,
            state_dict,
            cpu_inputs,
            cpu_labels,
        )
    if spec.kind == "direct":
        return _build_direct_candidate(spec, cell, state_dict, cpu_inputs, cpu_labels)
    if spec.kind == "generic_graph":
        return _build_generic_graph_candidate(
            spec, cell, state_dict, cpu_inputs, cpu_labels, config=config
        )
    return _build_exact_split_candidate(spec, cell, state_dict, cpu_inputs, cpu_labels)


def _build_direct_candidate(
    spec: CandidateSpec,
    cell: Cell,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
) -> _DirectRuntime:
    started = perf_counter()
    model = _make_model(cell, state_dict)
    if spec.direct_tuned:
        _configure_optimized_model(model, "final_best", cell)
    else:
        _configure_backend(model, EAGER_BACKEND)
        if hasattr(model, "use_fused_efp16_stem_training"):
            model.__dict__["use_fused_efp16_stem_training"] = spec.direct_fused_stem
        for block in _model_blocks(model):
            block.__dict__["fused_moments_backward_training"] = spec.direct_fused_moments
    inputs = cpu_inputs.to(device="cuda", dtype=torch.float32)
    labels = cpu_labels.to(device="cuda", dtype=torch.long)
    return _DirectRuntime(
        model,
        inputs,
        labels,
        spec,
        cell,
        setup_seconds=perf_counter() - started,
    )


def _build_generic_graph_candidate(
    _spec: CandidateSpec,
    cell: Cell,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
    *,
    config: ScreenConfig,
) -> _GenericGraphRuntime:
    started = perf_counter()
    model = _make_model(cell, state_dict)
    _configure_backend(model, EAGER_BACKEND)
    inputs = cpu_inputs.to(device="cuda", dtype=torch.float32)
    labels = cpu_labels.to(device="cuda", dtype=torch.long)
    optimizer = make_capturable_adamw(
        model,
        learning_rate=CAMPAIGN_CONFIG.learning_rate,
        weight_decay=CAMPAIGN_CONFIG.weight_decay,
    )
    runtime = prepare_efp16_training_cuda_graph(
        model,
        optimizer,
        inputs,
        labels,
        grad_clip_norm=CAMPAIGN_CONFIG.grad_clip_norm,
        warmup_steps=config.graph_warmups,
        copy_inputs=True,
        copy_loss=False,
        prepare_model=True,
        fused_recurrence_moments_backward_training=False,
        fused_optimizer_tail=False,
    )
    return _GenericGraphRuntime(
        model,
        runtime,
        inputs,
        labels,
        setup_seconds=perf_counter() - started,
    )


def _build_exact_split_candidate(
    spec: CandidateSpec,
    cell: Cell,
    state_dict: dict[str, Tensor],
    cpu_inputs: Tensor,
    cpu_labels: Tensor,
) -> _ExactSplitRuntime:
    started = perf_counter()
    model = _make_model(cell, state_dict)
    _configure_backend(model, EAGER_BACKEND)
    for block in _model_blocks(model):
        block.__dict__["canonical_identity_elision"] = spec.identity_static
        block.__dict__["mode_static_pole_training"] = spec.identity_static
        block.__dict__["packed_recurrence_moments_training"] = (
            None if spec.identity_static else False
        )
        block.__dict__["two_pass_reverse_recurrence_moments_training"] = (
            None if spec.identity_static else False
        )
    if spec.shape_tuned:
        _configure_optimized_model(model, "previous_best", cell)
        if hasattr(model, "efp16_stem_parameter_gradient_strategy"):
            model.__dict__["efp16_stem_parameter_gradient_strategy"] = (
                _EFP_STEM_STRATEGY_DISPATCH.get(cell, "auto")
            )
    inputs = cpu_inputs.to(device="cuda", dtype=torch.float32)
    labels = cpu_labels.to(device="cuda", dtype=torch.long)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CAMPAIGN_CONFIG.learning_rate,
        weight_decay=CAMPAIGN_CONFIG.weight_decay,
    )
    manager = _stage_environment("previous_best", cell) if spec.shape_tuned else _null_environment()
    with manager:
        runtime = prepare_efp16_exact_split_training(
            model,
            optimizer,
            inputs,
            labels,
            grad_clip_norm=CAMPAIGN_CONFIG.grad_clip_norm,
            warmup_steps=1,
            recurrence_backend="auto",
            fused_recurrence_moments_backward_training=spec.fused_recurrence,
            capture_post_optimizer_step=False,
            specialized_matrix_exp_vjp=spec.specialized_vjp,
            matrix_exp_dispatch="host",
        )
    return _ExactSplitRuntime(
        model,
        runtime,
        inputs,
        labels,
        spec,
        cell,
        setup_seconds=perf_counter() - started,
    )


def _make_model(cell: Cell, state_dict: dict[str, Tensor]) -> nn.Module:
    model, _ = build_training_model("efp16", cell[1], cell[2])
    model.load_state_dict(state_dict, strict=True)
    return model.to(device="cuda", dtype=torch.float32).train()


def _model_blocks(model: nn.Module) -> tuple[nn.Module, ...]:
    blocks = (
        getattr(model, "forward_block", None),
        getattr(model, "backward_block", None),
        *getattr(model, "extra_blocks", []),
    )
    return tuple(block for block in blocks if isinstance(block, nn.Module))


def _measure_single(runtime: _Runtime, *, config: ScreenConfig) -> dict[str, object]:
    loss = torch.zeros((), device="cuda")
    for warmup in range(config.warmups):
        runtime.reset_seed(config.seed + 200_000 + warmup)
        loss = runtime.step()
    torch.cuda.synchronize()
    wall_samples: list[float] = []
    event_samples: list[float] = []
    for group in range(config.groups):
        runtime.reset_seed(config.seed + 210_000 + group * config.iterations_per_group)
        _precondition_gpu_clock(config.gpu_clock_precondition_cycles)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = perf_counter()
        start.record()
        for _ in range(config.iterations_per_group):
            loss = runtime.step()
        end.record()
        end.synchronize()
        wall_samples.append((perf_counter() - started) * 1000.0 / config.iterations_per_group)
        event_samples.append(start.elapsed_time(end) / config.iterations_per_group)
    return _sample_summary(wall_samples, event_samples, loss)


def _measure_paired(
    eager: _Runtime, candidate: _Runtime, *, config: ScreenConfig
) -> dict[str, dict[str, object]]:
    runtimes = {"eager": eager, "candidate": candidate}
    last_loss = {name: torch.zeros((), device="cuda") for name in runtimes}
    for warmup in range(config.warmups):
        order = tuple(runtimes) if warmup % 2 == 0 else tuple(reversed(runtimes))
        for name in order:
            runtimes[name].reset_seed(config.seed + 300_000 + warmup)
            last_loss[name] = runtimes[name].step()
    torch.cuda.synchronize()
    wall: dict[str, list[float]] = {name: [] for name in runtimes}
    event: dict[str, list[float]] = {name: [] for name in runtimes}
    for group in range(config.groups):
        order = tuple(runtimes) if group % 2 == 0 else tuple(reversed(runtimes))
        for name in order:
            runtime = runtimes[name]
            runtime.reset_seed(config.seed + 310_000 + group * config.iterations_per_group)
            _precondition_gpu_clock(config.gpu_clock_precondition_cycles)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            started = perf_counter()
            start.record()
            for _ in range(config.iterations_per_group):
                last_loss[name] = runtime.step()
            end.record()
            end.synchronize()
            wall[name].append((perf_counter() - started) * 1000.0 / config.iterations_per_group)
            event[name].append(start.elapsed_time(end) / config.iterations_per_group)
    return {name: _sample_summary(wall[name], event[name], last_loss[name]) for name in runtimes}


def _accuracy_passes(accuracy: dict[str, object]) -> bool:
    return (
        accuracy.get("final_gradient_key_agreement") is True
        and accuracy.get("final_parameter_key_agreement") is True
        and all(
            _as_float(accuracy.get(metric), math.inf) <= MAXIMUM_ERROR
            for metric in (
                "loss_trajectory_max_abs_error",
                "final_gradient_max_abs_error",
                "final_parameter_max_abs_error",
            )
        )
        and _as_float(accuracy.get("prediction_agreement"), 0.0) == 1.0
    )


def _zero_accuracy(parity_steps: int) -> dict[str, object]:
    return {
        "parity_steps": parity_steps,
        "reference_stage": "actual_campaign_eager",
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
        "strict_pass": True,
    }


def _row(
    spec: CandidateSpec,
    cell: Cell,
    accuracy: dict[str, object],
    candidate_timing: dict[str, object],
    eager_timing: dict[str, object],
    setup_seconds: float,
    provenance: dict[str, str],
) -> dict[str, object]:
    candidate_wall = _as_float(candidate_timing["wall_ms"])
    eager_wall = _as_float(eager_timing["wall_ms"])
    strict_pass = bool(accuracy.get("strict_pass"))
    return {
        "model": "efp16",
        "length": cell[1],
        "batch_size": cell[2],
        "candidate": spec.name,
        "candidate_kind": spec.kind,
        "status": "measured" if strict_pass else "rejected",
        "strict_pass": strict_pass,
        "accuracy": accuracy,
        "provenance": provenance,
        "candidate_timing": candidate_timing,
        "paired_eager_timing": eager_timing,
        "candidate_wall_ms": candidate_wall,
        "candidate_cuda_event_ms": candidate_timing["cuda_event_ms"],
        "paired_eager_wall_ms": eager_wall,
        "speedup_vs_paired_eager": eager_wall / candidate_wall,
        "sequences_per_second": cell[2] * 1000.0 / candidate_wall,
        "tokens_per_second": cell[1] * cell[2] * 1000.0 / candidate_wall,
        "setup_seconds": setup_seconds,
        "compile_and_capture_cost_included": False,
        "normalized": False,
    }


def summarize(rows: Sequence[dict[str, object]], *, lengths: Sequence[int]) -> dict[str, object]:
    result: dict[str, object] = {}
    for length in lengths:
        measured = [
            row
            for row in rows
            if _as_int(row.get("length")) == length
            and row.get("status") in {"measured", "rejected"}
        ]
        passing = [row for row in measured if row.get("strict_pass") is True]
        selected = min(
            passing,
            key=lambda row: _as_float(row.get("candidate_wall_ms"), math.inf),
            default=None,
        )
        result[f"N{length}/B{B64_BATCH}"] = {
            "screened_count": len(measured),
            "strict_pass_count": len(passing),
            "selected_candidate": selected.get("candidate") if selected is not None else None,
            "selected_wall_ms": (
                selected.get("candidate_wall_ms") if selected is not None else None
            ),
            "selected_speedup_vs_paired_eager": (
                selected.get("speedup_vs_paired_eager") if selected is not None else None
            ),
        }
    return result


def _candidate_payload(spec: CandidateSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "kind": spec.kind,
        "description": spec.description,
        "risky_features": list(spec.risky_features),
    }


def _candidate_spec(name: CandidateName) -> CandidateSpec:
    return next(spec for spec in CANDIDATES if spec.name == name)


def _protocol() -> dict[str, object]:
    return {
        "scope": "EFP16 B64 strict actual-eager training accuracy and latency screen",
        "dtype": "float32",
        "tf32": False,
        "autocast": False,
        "normalized_latency": False,
        "compile_and_capture_cost_included": False,
        "reference": "actual campaign eager recurrence-auto direct autograd and default AdamW",
        "accuracy": (
            "75 consecutive updates from identical state/input/labels; raw loss trajectory, "
            "final gradient, final parameter/update, logit, and prediction comparisons"
        ),
        "strict_selection": (
            "loss trajectory, final gradient, and final parameter max abs error <=2e-5 and "
            "prediction agreement ==1.0"
        ),
        "timing": (
            "fresh candidate/eager contexts, alternating paired group order, synchronized raw "
            "wall and CUDA-event samples; setup/capture excluded"
        ),
        "diagnostic_order": (
            "direct fused components -> generic graph -> exact split with successive features "
            "enabled -> historical previous/final bundles"
        ),
    }


def _validate_runtime(
    lengths: tuple[int, ...], candidates: tuple[CandidateName, ...], config: ScreenConfig
) -> None:
    if not torch.cuda.is_available():
        message = "strict EFP16 training screen requires CUDA"
        raise RuntimeError(message)
    device_name = torch.cuda.get_device_name()
    if "4090" not in device_name:
        message = f"strict EFP16 training screen requires RTX 4090, found {device_name}"
        raise RuntimeError(message)
    if not lengths or set(lengths) - set(LENGTHS):
        message = "lengths must be a non-empty subset of 128,512,2048"
        raise ValueError(message)
    known = {spec.name for spec in CANDIDATES}
    unknown = set(candidates) - known
    if not candidates or unknown:
        message = f"unknown or empty candidate set: {sorted(unknown) if candidates else []}"
        raise ValueError(message)
    if len(set(candidates)) != len(candidates):
        message = "candidate names must be unique"
        raise ValueError(message)
    if config.parity_steps < 75:
        message = "strict screen requires at least 75 parity steps"
        raise ValueError(message)
    if min(config.warmups, config.iterations_per_group, config.graph_warmups) < 1:
        message = "warmups and iterations must be positive"
        raise ValueError(message)
    if config.groups < 3:
        message = "strict screen requires at least three timing groups"
        raise ValueError(message)


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


def _parse_lengths(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _parse_candidates(raw: str) -> tuple[CandidateName, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    known = {spec.name for spec in CANDIDATES}
    unknown = sorted(set(values) - known)
    if unknown:
        message = f"unknown candidates: {unknown}"
        raise ValueError(message)
    return cast("tuple[CandidateName, ...]", values)


def _config_from_args(arguments: argparse.Namespace) -> ScreenConfig:
    return ScreenConfig(
        warmups=arguments.warmups,
        groups=arguments.groups,
        iterations_per_group=arguments.iterations_per_group,
        parity_steps=arguments.parity_steps,
        seed=arguments.seed,
        graph_warmups=arguments.graph_warmups,
        gpu_clock_ramp_cycles=arguments.gpu_clock_ramp_cycles,
        gpu_clock_precondition_cycles=arguments.gpu_clock_precondition_cycles,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    parser.add_argument("--candidates", default=",".join(spec.name for spec in CANDIDATES))
    parser.add_argument("--warmups", type=int, default=DEFAULT_CONFIG.warmups)
    parser.add_argument("--groups", type=int, default=DEFAULT_CONFIG.groups)
    parser.add_argument(
        "--iterations-per-group", type=int, default=DEFAULT_CONFIG.iterations_per_group
    )
    parser.add_argument("--parity-steps", type=int, default=DEFAULT_CONFIG.parity_steps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--graph-warmups", type=int, default=DEFAULT_CONFIG.graph_warmups)
    parser.add_argument(
        "--gpu-clock-ramp-cycles", type=int, default=DEFAULT_CONFIG.gpu_clock_ramp_cycles
    )
    parser.add_argument(
        "--gpu-clock-precondition-cycles",
        type=int,
        default=DEFAULT_CONFIG.gpu_clock_precondition_cycles,
    )
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    frozen_baseline = cast(
        "dict[str, object]", json.loads(cast("Path", arguments.baseline).read_text())
    )
    result = benchmark(
        frozen_baseline,
        lengths=_parse_lengths(arguments.lengths),
        candidates=_parse_candidates(arguments.candidates),
        config=_config_from_args(arguments),
    )
    output = cast("Path", arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    gc.collect()


if __name__ == "__main__":
    main()
