"""Exact-matrix-exp split CUDA Graph training runtime for PA2WP.

The full-step PA2WP graph replaces ``torch.matrix_exp`` with a fixed Taylor
map because PyTorch's native CUDA matrix exponential is not capture safe.  A
few high-batch shapes are sensitive to the resulting (small) trajectory
change.  This runtime keeps the native matrix exponential and its native VJP
outside capture, while capturing the much larger PA2WP forward/backward body
and the optimizer tail separately.

One GPU Bernoulli is still drawn per call, before either deterministic phase
body is replayed.  Thus the stochastic training policy and the native
orthogonal parametrization are unchanged; only launch scheduling differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_native_matrix_exp_vjp import (
    MatrixExpDispatch,
    NativeMatrixExpReplay,
    make_native_matrix_exp_replay,
    matrix_exp_one_norm,
    matrix_exp_vjp_branch,
    matrix_exp_vjp_one_norm,
)
from .pac_pa2wp_phase_schedule import _CapturedScalarPhaseSchedule

PhaseName = Literal["original", "shifted"]


class _PA2WPPhaseModel(Protocol):
    def _phase_logits(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> Tensor: ...

    def post_optimizer_step(self) -> None: ...


@dataclass(frozen=True)
class PA2WPExactSplitTrainingStep:
    """Graph-owned loss and the stochastic phase selected for one step."""

    loss: Tensor
    phase: PhaseName


class _CachedNativeOrthogonal(nn.Module):
    """Parametrization proxy whose cache is differentiated by the body graph."""

    base: Tensor
    cached_weight: Tensor

    def __init__(self, shape: torch.Size, base: Tensor) -> None:
        super().__init__()
        self.shape = torch.Size(shape)
        self.register_buffer("base", base)
        self.base = base
        cache = base.new_empty(self.shape).requires_grad_()
        self.register_buffer("cached_weight", cache, persistent=False)
        self.cached_weight = cache

    def native_skew(self, coordinates: Tensor) -> tuple[Tensor, bool, int]:
        """Build torch's skew generator and return its output-shape metadata."""
        rows, columns = coordinates.size(-2), coordinates.size(-1)
        transposed = rows < columns
        working = coordinates.mT if transposed else coordinates
        rows, columns = working.size(-2), working.size(-1)
        lower = working.tril()
        if rows != columns:
            padding = lower.new_zeros(rows, rows - columns).expand(*lower.shape[:-2], -1, -1)
            lower = torch.cat((lower, padding), dim=-1)
        skew = lower - lower.mH
        return skew, transposed, columns

    def finish_native_weight(
        self,
        orthogonal: Tensor,
        *,
        transposed: bool,
        columns: int,
    ) -> Tensor:
        """Apply the truncation and dynamic-trivialization base."""
        if orthogonal.shape[-1] != columns:
            orthogonal = orthogonal[..., :columns]
        orthogonal = self.base @ orthogonal
        return orthogonal.mT if transposed else orthogonal

    def native_weight(self, coordinates: Tensor) -> Tensor:
        """Evaluate the same operation order as torch's matrix-exp map."""
        skew, transposed, columns = self.native_skew(coordinates)
        return self.finish_native_weight(
            torch.matrix_exp(skew),
            transposed=transposed,
            columns=columns,
        )

    def forward(self, _coordinates: Tensor) -> Tensor:
        return self.cached_weight


@dataclass(frozen=True)
class _OrthogonalSplit:
    coordinates: nn.Parameter
    proxy: _CachedNativeOrthogonal


@dataclass(frozen=True)
class _NativeFrameEvaluation:
    split: _OrthogonalSplit
    skew: Tensor
    transposed: bool
    columns: int


@dataclass(frozen=True)
class _MutableStateSnapshot:
    parameters: tuple[tuple[Tensor, Tensor], ...]
    optimizer_tensors: tuple[tuple[Tensor, Tensor], ...]

    @classmethod
    def capture(
        cls, parameters: tuple[nn.Parameter, ...], optimizer: torch.optim.AdamW
    ) -> _MutableStateSnapshot:
        return cls(
            tuple((parameter, parameter.detach().clone()) for parameter in parameters),
            tuple(
                (value, value.detach().clone())
                for state in optimizer.state.values()
                for value in state.values()
                if isinstance(value, Tensor)
            ),
        )

    @torch.no_grad()
    def restore(
        self,
        parameters: tuple[nn.Parameter, ...],
        splits: tuple[_OrthogonalSplit, ...],
    ) -> None:
        for destination, value in self.parameters:
            destination.copy_(value)
        for destination, value in self.optimizer_tensors:
            destination.copy_(value)
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            else:
                parameter.grad.zero_()
        for split in splits:
            gradient = split.proxy.cached_weight.grad
            if gradient is None:
                split.proxy.cached_weight.grad = torch.zeros_like(split.proxy.cached_weight)
            else:
                gradient.zero_()


class PA2WPExactSplitTraining:
    """Train PA2WP with native matrix-exp semantics and split CUDA Graphs.

    The loss tensor is borrowed from the selected phase graph and is
    overwritten by the next replay of that same phase.  Clone it when it must
    outlive another step.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        batch_size: int,
        sequence_length: int,
        learning_rate: float = 3.0e-3,
        weight_decay: float = 1.0e-4,
        grad_clip_norm: float = 1.0,
        warmup_steps_per_phase: int = 3,
        parallel_native_frames: bool = False,
        phase_schedule_capacity: int | None = None,
        fused_recurrence_moments_backward_training: bool = False,
        specialized_matrix_exp_vjp: bool = False,
        matrix_exp_dispatch: MatrixExpDispatch = "host",
    ) -> None:
        parameters, device = _validate_preparation(
            model,
            batch_size=batch_size,
            sequence_length=sequence_length,
            grad_clip_norm=grad_clip_norm,
            warmup_steps_per_phase=warmup_steps_per_phase,
        )
        self.model = model
        self._phase_model = cast("_PA2WPPhaseModel", cast("object", model))
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        self.grad_clip_norm = float(grad_clip_norm)
        self.parallel_native_frames = parallel_native_frames
        self.specialized_matrix_exp_vjp, self.matrix_exp_dispatch = (
            _validate_matrix_exp_options(
                specialized=specialized_matrix_exp_vjp,
                parallel=parallel_native_frames,
                dispatch=matrix_exp_dispatch,
            )
        )
        self.last_phase: PhaseName | None = None
        self.model.train()
        for block_name in ("forward_block", "backward_block"):
            block = getattr(model, block_name, None)
            if block is not None:
                block.__dict__["fused_recurrence_moments_backward_training"] = (
                    fused_recurrence_moments_backward_training
                )
        self._phase_schedule = (
            _CapturedScalarPhaseSchedule(device, phase_schedule_capacity)
            if phase_schedule_capacity is not None
            else None
        )

        self._splits = _install_native_orthogonal_caches_(model)
        if not self._splits:
            message = "PA2WP exact split requires matrix-exp orthogonal parametrizations"
            raise ValueError(message)
        self._matrix_exp_streams = tuple(torch.cuda.Stream(device=device) for _ in self._splits)
        self.replaced_orthogonal_weights = tuple(name for name, _ in _named_split_proxies(model))
        self._captured_matrix_exp_vjps = self._prepare_matrix_exp_replays()

        self._original_inputs = torch.zeros(
            batch_size, sequence_length, 1, device=device, dtype=torch.float32
        )
        self._shifted_inputs = torch.zeros(
            batch_size, sequence_length - 1, 1, device=device, dtype=torch.float32
        )
        self._labels = torch.zeros(batch_size, device=device, dtype=torch.long)
        self._parameters = parameters
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=True,
            capturable=True,
        )

        self._initialize_optimizer_state()
        pristine = _MutableStateSnapshot.capture(parameters, self.optimizer)
        self._warm_up(warmup_steps_per_phase)
        pristine.restore(parameters, self._splits)
        self._refresh_native_weights()
        torch.cuda.synchronize(device)

        self._original_graph, self._original_loss = self._capture_body(self._original_inputs)
        pristine.restore(parameters, self._splits)
        self._refresh_native_weights()
        torch.cuda.synchronize(device)
        self._shifted_graph, self._shifted_loss = self._capture_body(self._shifted_inputs)
        pristine.restore(parameters, self._splits)
        torch.cuda.synchronize(device)
        self._optimizer_graph = self._capture_optimizer_tail()
        pristine.restore(parameters, self._splits)
        self._refresh_native_weights()
        torch.cuda.synchronize(device)

    def __call__(self, inputs: Tensor, labels: Tensor) -> PA2WPExactSplitTrainingStep:
        self._validate_batch(inputs, labels)
        use_shifted = (
            bool(torch.rand((), device=self.device) < 0.5)
            if self._phase_schedule is None
            else self._phase_schedule.next_shifted()
        )
        self._labels.copy_(labels)
        evaluations = self._refresh_native_weights()
        if use_shifted:
            self._shifted_inputs.copy_(inputs[:, 1:])
            self._shifted_graph.replay()
            loss = self._shifted_loss
            phase: PhaseName = "shifted"
        else:
            self._original_inputs.copy_(inputs)
            self._original_graph.replay()
            loss = self._original_loss
            phase = "original"
        self._propagate_native_matrix_exp_gradients(evaluations)
        self._optimizer_graph.replay()
        self.last_phase = phase
        return PA2WPExactSplitTrainingStep(loss, phase)

    def reset_phase_schedule(self) -> None:
        """Invalidate phase prefetch after the caller resets CUDA RNG state."""
        if self._phase_schedule is not None:
            self._phase_schedule.reset()

    @torch.no_grad()
    def _refresh_native_weights(self) -> tuple[_NativeFrameEvaluation, ...]:
        if self.specialized_matrix_exp_vjp:
            specialized_evaluations = tuple(
                _native_frame_evaluation_metadata(split) for split in self._splits
            )
            if self.matrix_exp_dispatch == "cuda_switch":
                for evaluation, runtime in zip(
                    specialized_evaluations,
                    self._captured_matrix_exp_vjps,
                    strict=True,
                ):
                    orthogonal = runtime.replay_forward(evaluation.skew)
                    exact_weight = evaluation.split.proxy.finish_native_weight(
                        orthogonal,
                        transposed=evaluation.transposed,
                        columns=evaluation.columns,
                    )
                    evaluation.split.proxy.cached_weight.copy_(exact_weight)
                return specialized_evaluations
            norms = torch.stack(
                tuple(
                    matrix_exp_one_norm(evaluation.skew) for evaluation in specialized_evaluations
                )
            )
            host_norms = tuple(float(value) for value in norms.cpu().tolist())
            for evaluation, runtime, norm in zip(
                specialized_evaluations,
                self._captured_matrix_exp_vjps,
                host_norms,
                strict=True,
            ):
                branch = matrix_exp_vjp_branch(norm)
                orthogonal = (
                    torch.matrix_exp(evaluation.skew)
                    if branch is None
                    else runtime.replay_forward(evaluation.skew, branch)
                )
                exact_weight = evaluation.split.proxy.finish_native_weight(
                    orthogonal,
                    transposed=evaluation.transposed,
                    columns=evaluation.columns,
                )
                evaluation.split.proxy.cached_weight.copy_(exact_weight)
            return specialized_evaluations
        evaluations: list[_NativeFrameEvaluation] = []
        if not self.parallel_native_frames:
            return tuple(_evaluate_native_frame_(split) for split in self._splits)
        current_stream = torch.cuda.current_stream(self.device)
        for split, stream in zip(self._splits, self._matrix_exp_streams, strict=True):
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                evaluations.append(_evaluate_native_frame_(split))
        for stream in self._matrix_exp_streams:
            current_stream.wait_stream(stream)
        return tuple(evaluations)

    def _propagate_native_matrix_exp_gradients(
        self, evaluations: tuple[_NativeFrameEvaluation, ...]
    ) -> None:
        if self.specialized_matrix_exp_vjp:
            orthogonal_gradients = tuple(
                _native_frame_orthogonal_gradient(evaluation) for evaluation in evaluations
            )
            if self.matrix_exp_dispatch == "cuda_switch":
                with torch.no_grad():
                    for evaluation, runtime, gradient in zip(
                        evaluations,
                        self._captured_matrix_exp_vjps,
                        orthogonal_gradients,
                        strict=True,
                    ):
                        _write_native_frame_gradient_(
                            evaluation,
                            runtime.replay(evaluation.skew, gradient),
                        )
                return
            norms = torch.stack(
                tuple(
                    matrix_exp_vjp_one_norm(evaluation.skew, gradient)
                    for evaluation, gradient in zip(evaluations, orthogonal_gradients, strict=True)
                )
            )
            host_norms = tuple(float(value) for value in norms.cpu().tolist())
            with torch.no_grad():
                for evaluation, runtime, gradient, norm in zip(
                    evaluations,
                    self._captured_matrix_exp_vjps,
                    orthogonal_gradients,
                    host_norms,
                    strict=True,
                ):
                    branch = matrix_exp_vjp_branch(norm)
                    skew_gradient = (
                        torch.ops.aten.matrix_exp_backward(evaluation.skew, gradient)
                        if branch is None
                        else runtime.replay(evaluation.skew, gradient, branch)
                    )
                    _write_native_frame_gradient_(evaluation, skew_gradient)
            return
        if not self.parallel_native_frames:
            with torch.no_grad():
                for evaluation in evaluations:
                    _propagate_native_frame_gradient_(evaluation)
            return
        current_stream = torch.cuda.current_stream(self.device)
        with torch.no_grad():
            for evaluation, stream in zip(evaluations, self._matrix_exp_streams, strict=True):
                stream.wait_stream(current_stream)
                with torch.cuda.stream(stream):
                    _propagate_native_frame_gradient_(evaluation)
            for stream in self._matrix_exp_streams:
                current_stream.wait_stream(stream)

    def _prepare_matrix_exp_replays(self) -> tuple[NativeMatrixExpReplay, ...]:
        if not self.specialized_matrix_exp_vjp:
            return ()
        sizes = tuple(max(split.coordinates.shape[-2:]) for split in self._splits)
        if self.matrix_exp_dispatch == "cuda_switch":
            if len(set(sizes)) != 1:
                message = "shared CUDA SWITCH matrix-exp requires equal frame sizes"
                raise ValueError(message)
            shared = make_native_matrix_exp_replay(
                sizes[0],
                self.device,
                dispatch="cuda_switch",
            )
            return tuple(shared for _ in sizes)
        return tuple(
            make_native_matrix_exp_replay(size, self.device, dispatch="host") for size in sizes
        )

    def _initialize_optimizer_state(self) -> None:
        parameter_values = tuple(parameter.detach().clone() for parameter in self._parameters)
        for parameter in self._parameters:
            parameter.grad = torch.zeros_like(parameter)
        for split in self._splits:
            split.proxy.cached_weight.grad = torch.zeros_like(split.proxy.cached_weight)
        self.optimizer.step()
        with torch.no_grad():
            for parameter, value in zip(self._parameters, parameter_values, strict=True):
                parameter.copy_(value)
            for state in self.optimizer.state.values():
                for value in state.values():
                    if isinstance(value, Tensor):
                        value.zero_()
            self._zero_body_gradients()

    def _warm_up(self, steps_per_phase: int) -> None:
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(steps_per_phase):
                self._uncaptured_full_step(self._original_inputs)
                self._uncaptured_full_step(self._shifted_inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        stream.synchronize()

    def _uncaptured_full_step(self, static_inputs: Tensor) -> Tensor:
        evaluations = self._refresh_native_weights()
        loss = self._uncaptured_body(static_inputs)
        self._propagate_native_matrix_exp_gradients(evaluations)
        self._uncaptured_optimizer_tail()
        return loss

    def _capture_body(self, static_inputs: Tensor) -> tuple[torch.cuda.CUDAGraph, Tensor]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = self._uncaptured_body(static_inputs)
        return graph, loss

    def _capture_optimizer_tail(self) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._uncaptured_optimizer_tail()
        return graph

    def _uncaptured_body(self, static_inputs: Tensor) -> Tensor:
        self._zero_body_gradients()
        logits = self._phase_model._phase_logits(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            static_inputs,
            time_delta=None,
            observation_mask=None,
            valid_mask=None,
        )
        loss = functional.cross_entropy(logits, self._labels)
        loss.backward()
        return loss

    def _zero_body_gradients(self) -> None:
        for parameter in self._parameters:
            gradient = parameter.grad
            if gradient is None:
                message = "PA2WP exact split requires persistent parameter gradients"
                raise RuntimeError(message)
            gradient.zero_()
        for split in self._splits:
            gradient = split.proxy.cached_weight.grad
            if gradient is None:
                message = "PA2WP exact split requires persistent cache gradients"
                raise RuntimeError(message)
            gradient.zero_()

    def _uncaptured_optimizer_tail(self) -> None:
        torch.nn.utils.clip_grad_norm_(
            self._parameters,
            self.grad_clip_norm,
            foreach=True,
        )
        self.optimizer.step()
        self._phase_model.post_optimizer_step()

    def _validate_batch(self, inputs: Tensor, labels: Tensor) -> None:
        expected_inputs = (self.batch_size, self.sequence_length, 1)
        if tuple(inputs.shape) != expected_inputs:
            message = f"inputs must have static shape {expected_inputs}, got {tuple(inputs.shape)}"
            raise ValueError(message)
        if tuple(labels.shape) != (self.batch_size,):
            message = (
                f"labels must have static shape {(self.batch_size,)}, got {tuple(labels.shape)}"
            )
            raise ValueError(message)
        if inputs.device != self.device or labels.device != self.device:
            message = f"inputs and labels must be on captured device {self.device}"
            raise ValueError(message)
        if inputs.dtype != torch.float32:
            message = "PA2WP exact split inputs must be FP32"
            raise ValueError(message)
        if labels.dtype != torch.long:
            message = "PA2WP exact split labels must be torch.long"
            raise ValueError(message)


def prepare_pa2wp_exact_split_training(
    model: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-4,
    grad_clip_norm: float = 1.0,
    warmup_steps_per_phase: int = 3,
    parallel_native_frames: bool = False,
    phase_schedule_capacity: int | None = None,
    fused_recurrence_moments_backward_training: bool = False,
    specialized_matrix_exp_vjp: bool = False,
    matrix_exp_dispatch: MatrixExpDispatch = "host",
) -> PA2WPExactSplitTraining:
    """Prepare exact-native-matrix-exp split-graph PA2WP training."""
    return PA2WPExactSplitTraining(
        model,
        batch_size=batch_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        warmup_steps_per_phase=warmup_steps_per_phase,
        parallel_native_frames=parallel_native_frames,
        phase_schedule_capacity=phase_schedule_capacity,
        fused_recurrence_moments_backward_training=(fused_recurrence_moments_backward_training),
        specialized_matrix_exp_vjp=specialized_matrix_exp_vjp,
        matrix_exp_dispatch=matrix_exp_dispatch,
    )


def _validate_matrix_exp_options(
    *,
    specialized: bool,
    parallel: bool,
    dispatch: MatrixExpDispatch,
) -> tuple[bool, MatrixExpDispatch]:
    if specialized and parallel:
        message = "specialized matrix-exp VJP does not support parallel frame streams"
        raise ValueError(message)
    if dispatch != "host" and not specialized:
        message = "matrix-exp dispatch requires specialized_matrix_exp_vjp=True"
        raise ValueError(message)
    return specialized, dispatch


@torch.no_grad()
def _evaluate_native_frame_(split: _OrthogonalSplit) -> _NativeFrameEvaluation:
    evaluation = _native_frame_evaluation_metadata(split)
    # Keep one native call per frame. PyTorch's batched matrix-exp can select a
    # different adaptive Taylor branch than two independent reference calls.
    orthogonal = torch.matrix_exp(evaluation.skew)
    exact_weight = split.proxy.finish_native_weight(
        orthogonal,
        transposed=evaluation.transposed,
        columns=evaluation.columns,
    )
    split.proxy.cached_weight.copy_(exact_weight)
    return evaluation


def _native_frame_evaluation_metadata(split: _OrthogonalSplit) -> _NativeFrameEvaluation:
    skew, transposed, columns = split.proxy.native_skew(split.coordinates)
    return _NativeFrameEvaluation(split, skew, transposed, columns)


@torch.no_grad()
def _propagate_native_frame_gradient_(evaluation: _NativeFrameEvaluation) -> None:
    orthogonal_gradient = _native_frame_orthogonal_gradient(evaluation)
    skew_gradient = torch.ops.aten.matrix_exp_backward(
        evaluation.skew,
        orthogonal_gradient,
    )
    _write_native_frame_gradient_(evaluation, skew_gradient)


def _native_frame_orthogonal_gradient(evaluation: _NativeFrameEvaluation) -> Tensor:
    split = evaluation.split
    cache_gradient = split.proxy.cached_weight.grad
    if cache_gradient is None:
        message = "captured PA2WP body did not produce a frame gradient"
        raise RuntimeError(message)
    active_gradient = cache_gradient.mT if evaluation.transposed else cache_gradient
    orthogonal_gradient = split.proxy.base.mT @ active_gradient
    matrix_size = evaluation.skew.shape[-1]
    if evaluation.columns != matrix_size:
        padding = orthogonal_gradient.new_zeros(
            matrix_size,
            matrix_size - evaluation.columns,
        ).expand(*orthogonal_gradient.shape[:-2], -1, -1)
        orthogonal_gradient = torch.cat((orthogonal_gradient, padding), dim=-1)
    return orthogonal_gradient


def _write_native_frame_gradient_(
    evaluation: _NativeFrameEvaluation, skew_gradient: Tensor
) -> None:
    split = evaluation.split
    working_gradient = (skew_gradient - skew_gradient.mT).tril()
    working_gradient = working_gradient[..., : evaluation.columns]
    coordinate_gradient = working_gradient.mT if evaluation.transposed else working_gradient
    destination = split.coordinates.grad
    if destination is None:
        split.coordinates.grad = coordinate_gradient
    else:
        destination.copy_(coordinate_gradient)


def _install_native_orthogonal_caches_(
    model: nn.Module,
) -> tuple[_OrthogonalSplit, ...]:
    splits: list[_OrthogonalSplit] = []
    for module in tuple(model.modules()):
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        for name in tuple(parametrizations.keys()):
            parametrization_list = parametrizations[name]
            coordinates = parametrization_list.original
            for index, candidate in enumerate(tuple(parametrization_list)):
                orthogonal_map = getattr(candidate, "orthogonal_map", None)
                if getattr(orthogonal_map, "name", None) != "matrix_exp":
                    continue
                base = candidate._buffers.get("base")  # noqa: SLF001
                shape = getattr(candidate, "shape", None)
                if not isinstance(base, Tensor) or shape is None:
                    message = (
                        "matrix-exp orthogonal parametrization must use dynamic trivialization"
                    )
                    raise ValueError(message)
                proxy = _CachedNativeOrthogonal(torch.Size(shape), base)
                proxy.train(candidate.training)
                parametrization_list[index] = proxy
                splits.append(_OrthogonalSplit(coordinates, proxy))
    return tuple(splits)


def _named_split_proxies(model: nn.Module) -> tuple[tuple[str, _CachedNativeOrthogonal], ...]:
    return tuple(
        (name.removesuffix(".parametrizations.weight.0") + ".weight", module)
        for name, module in model.named_modules()
        if isinstance(module, _CachedNativeOrthogonal)
    )


def _validate_preparation(  # noqa: C901
    model: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
    grad_clip_norm: float,
    warmup_steps_per_phase: int,
) -> tuple[tuple[nn.Parameter, ...], torch.device]:
    if not torch.cuda.is_available():
        message = "PA2WP exact split training requires CUDA"
        raise RuntimeError(message)
    if batch_size < 1:
        message = "batch_size must be positive"
        raise ValueError(message)
    if sequence_length < 2:
        message = "PA2WP exact split requires sequence_length >= 2"
        raise ValueError(message)
    if grad_clip_norm <= 0.0:
        message = "grad_clip_norm must be positive"
        raise ValueError(message)
    if warmup_steps_per_phase < 1:
        message = "warmup_steps_per_phase must be positive"
        raise ValueError(message)
    if not callable(getattr(model, "_phase_logits", None)):
        message = "model must provide PA2WP _phase_logits"
        raise TypeError(message)
    parameters = tuple(model.parameters())
    if not parameters:
        message = "PA2WP training model has no parameters"
        raise ValueError(message)
    device = parameters[0].device
    if device.type != "cuda" or any(parameter.device != device for parameter in parameters):
        message = "PA2WP exact split requires one CUDA device for all parameters"
        raise ValueError(message)
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        message = "PA2WP exact split supports FP32 parameters only"
        raise ValueError(message)
    if any(
        buffer.is_floating_point() and buffer.dtype != torch.float32 for buffer in model.buffers()
    ):
        message = "PA2WP exact split supports FP32 buffers only"
        raise ValueError(message)
    return parameters, device


__all__ = [
    "PA2WPExactSplitTraining",
    "PA2WPExactSplitTrainingStep",
    "prepare_pa2wp_exact_split_training",
]
