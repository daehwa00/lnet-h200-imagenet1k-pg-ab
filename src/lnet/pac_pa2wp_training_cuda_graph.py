"""Full-step CUDA Graph training runtime for stochastic single-phase PA2WP.

The campaign trains PA2WP on exactly one of the two adjacent-pair origins per
step.  A graph cannot contain the Python branch used by the regular model, so
this runtime captures one deterministic graph per phase and keeps the Bernoulli
draw outside both graphs.  Both graphs operate on the same model parameters,
gradient buffers, and capturable AdamW state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_capture_safe_orthogonal import prepare_capture_safe_orthogonal_
from .pac_pa2wp_phase_schedule import _CapturedScalarPhaseSchedule

if TYPE_CHECKING:
    from .pac_cuda_fused_optimizer import FusedClipAdamW

PhaseName = Literal["original", "shifted"]


class _PA2WPPhaseModel(Protocol):
    def prepare_fused_pa2wp_stem_training_(
        self, *, include_large_workloads: bool = False
    ) -> object: ...

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
class PA2WPTrainingGraphStep:
    """Borrowed loss tensor and the stochastic phase selected for one replay."""

    loss: Tensor
    phase: PhaseName


@dataclass(frozen=True)
class _MutableStateSnapshot:
    parameters: tuple[tuple[Tensor, Tensor], ...]
    optimizer_tensors: tuple[tuple[Tensor, Tensor], ...]

    @classmethod
    def capture(
        cls, parameters: tuple[nn.Parameter, ...], optimizer: torch.optim.AdamW
    ) -> _MutableStateSnapshot:
        parameter_values = tuple(
            (parameter, parameter.detach().clone()) for parameter in parameters
        )
        optimizer_values = [
            (value, value.detach().clone())
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, Tensor)
        ]
        return cls(parameter_values, tuple(optimizer_values))

    @torch.no_grad()
    def restore(self, parameters: tuple[nn.Parameter, ...]) -> None:
        for destination, value in self.parameters:
            destination.copy_(value)
        for destination, value in self.optimizer_tensors:
            destination.copy_(value)
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            else:
                parameter.grad.zero_()


class PA2WPTrainingCudaGraph:
    """Capture and replay exact-policy PA2WP FP32 campaign training steps.

    The returned loss is graph-owned and is overwritten the next time its phase
    graph is replayed.  Clone it before another step if it must be retained.
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
        taylor_degree: int = 12,
        scaling_steps: int = 3,
        matrix_exp_compute_dtype: torch.dtype | None = None,
        phase_schedule_capacity: int | None = 64,
        large_fused_stem_training: bool = False,
        fused_recurrence_moments_backward_training: bool = False,
        fused_optimizer_tail: bool = False,
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
        self._phase_model.prepare_fused_pa2wp_stem_training_(
            include_large_workloads=large_fused_stem_training
        )
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        self.grad_clip_norm = float(grad_clip_norm)
        self.last_phase: PhaseName | None = None
        self.replaced_orthogonal_weights = prepare_capture_safe_orthogonal_(
            model,
            taylor_degree=taylor_degree,
            scaling_steps=scaling_steps,
            compute_dtype=matrix_exp_compute_dtype,
        )
        if fused_recurrence_moments_backward_training:
            _enable_fused_recurrence_moments_backward(model)
        self.model.train()

        # The two phases deliberately have different allocations and lengths.
        # Labels and optimizer state are shared by both captures.
        self._original_inputs = torch.zeros(
            batch_size, sequence_length, 1, device=device, dtype=torch.float32
        )
        self._shifted_inputs = torch.zeros(
            batch_size, sequence_length - 1, 1, device=device, dtype=torch.float32
        )
        self._labels = torch.zeros(batch_size, device=device, dtype=torch.long)
        self._parameters = parameters
        self.optimizer = torch.optim.AdamW(
            self._parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=True,
            capturable=True,
        )

        self._initialize_optimizer_state()
        self._fused_optimizer_tail: FusedClipAdamW | None = None
        if fused_optimizer_tail:
            from .pac_cuda_fused_optimizer import FusedClipAdamW  # noqa: PLC0415

            self._fused_optimizer_tail = FusedClipAdamW.from_adamw(
                self.optimizer,
                max_norm=self.grad_clip_norm,
            )
        pristine = _MutableStateSnapshot.capture(self._parameters, self.optimizer)
        self._warm_up(warmup_steps_per_phase)
        pristine.restore(self._parameters)
        torch.cuda.synchronize(device)

        self._original_graph, self._original_loss = self._capture_phase(self._original_inputs)
        pristine.restore(self._parameters)
        torch.cuda.synchronize(device)
        self._shifted_graph, self._shifted_loss = self._capture_phase(self._shifted_inputs)
        pristine.restore(self._parameters)
        torch.cuda.synchronize(device)
        self._phase_schedule = (
            _CapturedScalarPhaseSchedule(device, phase_schedule_capacity)
            if phase_schedule_capacity is not None
            else None
        )

    def __call__(self, inputs: Tensor, labels: Tensor) -> PA2WPTrainingGraphStep:
        """Copy a caller batch, select one GPU Bernoulli, and replay its graph."""
        self._validate_batch(inputs, labels)
        use_shifted = (
            bool(torch.rand((), device=self.device) < 0.5)
            if self._phase_schedule is None
            else self._phase_schedule.next_shifted()
        )
        self._labels.copy_(labels)
        if use_shifted:
            self._shifted_inputs.copy_(inputs[:, 1:])
            self._shifted_graph.replay()
            self.last_phase = "shifted"
            return PA2WPTrainingGraphStep(self._shifted_loss, "shifted")
        self._original_inputs.copy_(inputs)
        self._original_graph.replay()
        self.last_phase = "original"
        return PA2WPTrainingGraphStep(self._original_loss, "original")

    def reset_phase_schedule(self) -> None:
        """Invalidate phase prefetch after the caller resets CUDA RNG state."""
        if self._phase_schedule is not None:
            self._phase_schedule.reset()

    def _validate_batch(self, inputs: Tensor, labels: Tensor) -> None:
        expected_inputs = (self.batch_size, self.sequence_length, 1)
        if tuple(inputs.shape) != expected_inputs:
            message = f"inputs must have static shape {expected_inputs}, got {tuple(inputs.shape)}"
            raise ValueError(message)
        if tuple(labels.shape) != (self.batch_size,):
            expected_labels = (self.batch_size,)
            message = f"labels must have static shape {expected_labels}, got {tuple(labels.shape)}"
            raise ValueError(message)
        if inputs.device != self.device or labels.device != self.device:
            message = f"inputs and labels must be on captured device {self.device}"
            raise ValueError(message)
        if inputs.dtype != torch.float32:
            message = "PA2WP training graph inputs must be FP32"
            raise ValueError(message)
        if labels.dtype != torch.long:
            message = "PA2WP training graph labels must be torch.long"
            raise ValueError(message)

    def _initialize_optimizer_state(self) -> None:
        """Allocate fused AdamW state and persistent gradients without changing state."""
        parameter_values = tuple(parameter.detach().clone() for parameter in self._parameters)
        for parameter in self._parameters:
            parameter.grad = torch.zeros_like(parameter)
        self.optimizer.step()
        with torch.no_grad():
            for parameter, value in zip(self._parameters, parameter_values, strict=True):
                parameter.copy_(value)
            for state in self.optimizer.state.values():
                for value in state.values():
                    if isinstance(value, Tensor):
                        value.zero_()
            for parameter in self._parameters:
                gradient = parameter.grad
                if gradient is None:
                    message = "optimizer initialization lost a persistent gradient buffer"
                    raise RuntimeError(message)
                gradient.zero_()

    def _warm_up(self, steps_per_phase: int) -> None:
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(steps_per_phase):
                self._uncaptured_step(self._original_inputs)
                self._uncaptured_step(self._shifted_inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        stream.synchronize()

    def _capture_phase(self, static_inputs: Tensor) -> tuple[torch.cuda.CUDAGraph, Tensor]:
        graph = torch.cuda.CUDAGraph()
        # Each graph owns a separate private pool: arbitrary stochastic phase
        # order does not satisfy CUDA Graph's ordering rule for shared pools.
        with torch.cuda.graph(graph):
            loss = self._uncaptured_step(static_inputs)
        return graph, loss

    def _uncaptured_step(self, static_inputs: Tensor) -> Tensor:
        self.optimizer.zero_grad(set_to_none=False)
        logits = self._phase_model._phase_logits(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            static_inputs,
            time_delta=None,
            observation_mask=None,
            valid_mask=None,
        )
        loss = functional.cross_entropy(logits, self._labels)
        loss.backward()
        if self._fused_optimizer_tail is None:
            torch.nn.utils.clip_grad_norm_(
                self._parameters,
                self.grad_clip_norm,
                foreach=True,
            )
            self.optimizer.step()
        else:
            self._fused_optimizer_tail.step()
        self._phase_model.post_optimizer_step()
        return loss


def prepare_pa2wp_training_cuda_graph(
    model: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-4,
    grad_clip_norm: float = 1.0,
    warmup_steps_per_phase: int = 3,
    taylor_degree: int = 12,
    scaling_steps: int = 3,
    matrix_exp_compute_dtype: torch.dtype | None = None,
    phase_schedule_capacity: int | None = 64,
    large_fused_stem_training: bool = False,
    fused_recurrence_moments_backward_training: bool = False,
    fused_optimizer_tail: bool = False,
) -> PA2WPTrainingCudaGraph:
    """Prepare the stochastic dual-graph PA2WP full-training runtime."""
    return PA2WPTrainingCudaGraph(
        model,
        batch_size=batch_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        warmup_steps_per_phase=warmup_steps_per_phase,
        taylor_degree=taylor_degree,
        scaling_steps=scaling_steps,
        matrix_exp_compute_dtype=matrix_exp_compute_dtype,
        phase_schedule_capacity=phase_schedule_capacity,
        large_fused_stem_training=large_fused_stem_training,
        fused_recurrence_moments_backward_training=(fused_recurrence_moments_backward_training),
        fused_optimizer_tail=fused_optimizer_tail,
    )


def _enable_fused_recurrence_moments_backward(model: nn.Module) -> None:
    blocks = [getattr(model, "forward_block", None), getattr(model, "backward_block", None)]
    blocks.extend(getattr(model, "extra_blocks", []))
    for block in blocks:
        if block is not None:
            block.__dict__["fused_recurrence_moments_backward_training"] = True


def _validate_preparation(  # noqa: C901
    model: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
    grad_clip_norm: float,
    warmup_steps_per_phase: int,
) -> tuple[tuple[nn.Parameter, ...], torch.device]:
    if not torch.cuda.is_available():
        message = "PA2WP training CUDA Graphs require CUDA"
        raise RuntimeError(message)
    if batch_size < 1:
        message = "batch_size must be positive"
        raise ValueError(message)
    if sequence_length < 2:
        message = "dual-phase PA2WP training requires sequence_length >= 2"
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
        message = "PA2WP training graph requires one CUDA device for all parameters"
        raise ValueError(message)
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        message = "PA2WP training graph supports exact FP32 parameters only"
        raise ValueError(message)
    if any(
        buffer.is_floating_point() and buffer.dtype != torch.float32 for buffer in model.buffers()
    ):
        message = "PA2WP training graph supports exact FP32 buffers only"
        raise ValueError(message)
    return parameters, device
