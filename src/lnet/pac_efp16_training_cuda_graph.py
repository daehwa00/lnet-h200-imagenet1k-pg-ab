from __future__ import annotations

import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_capture_safe_orthogonal import prepare_capture_safe_orthogonal_

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch.optim import AdamW

    from .pac_cuda_fused_optimizer import FusedClipAdamW


class EFP16TrainingCudaGraph:
    """Replay one exact-FP32 EFP16 parameter-update step with a CUDA Graph.

    The runtime owns static input and label buffers by default.  :meth:`step`
    copies caller tensors into those buffers before replay, so the measured API
    includes input-copy overhead and callers may use a different allocation on
    every step.  Set ``copy_inputs=False`` only when the caller owns stable CUDA
    allocations and wants the borrowed-buffer contract.

    ``model`` and ``optimizer`` are not copied.  The graph reads and updates the
    exact parameter, gradient, and AdamW state tensors owned by the caller.
    Consequently they must not be mutated concurrently with :meth:`step`.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: AdamW,
        example_inputs: Tensor,
        example_labels: Tensor,
        *,
        grad_clip_norm: float = 1.0,
        warmup_steps: int = 3,
        copy_inputs: bool = True,
        copy_loss: bool = False,
        prepare_model: bool = True,
        taylor_degree: int = 12,
        scaling_steps: int = 3,
        matrix_exp_compute_dtype: torch.dtype | None = None,
        fused_recurrence_moments_backward_training: bool = False,
        fused_optimizer_tail: bool = False,
    ) -> None:
        _validate_options(grad_clip_norm, warmup_steps)
        _disable_tf32()
        self.model = model
        self.optimizer = optimizer
        self.grad_clip_norm = float(grad_clip_norm)
        self.copy_inputs = copy_inputs
        self.copy_loss = copy_loss
        self._validate_model_optimizer(example_inputs, example_labels)
        names_before = tuple(name for name, _ in model.named_parameters())
        identities_before = tuple(id(parameter) for parameter in model.parameters())
        self.capture_safe_frame_paths = (
            prepare_capture_safe_orthogonal_(
                model,
                taylor_degree=taylor_degree,
                scaling_steps=scaling_steps,
                compute_dtype=matrix_exp_compute_dtype,
            )
            if prepare_model
            else ()
        )
        names_after = tuple(name for name, _ in model.named_parameters())
        identities_after = tuple(id(parameter) for parameter in model.parameters())
        if names_after != names_before or identities_after != identities_before:
            message = "capture-safe orthogonal preparation changed parameter names or identity"
            raise RuntimeError(message)
        if hasattr(model, "use_fused_efp16_stem_training"):
            model.__dict__["use_fused_efp16_stem_training"] = True
        if fused_recurrence_moments_backward_training:
            _enable_fused_recurrence_moments_backward(model)

        self._input_shape = tuple(example_inputs.shape)
        self._label_shape = tuple(example_labels.shape)
        self._device = example_inputs.device
        self._input_dtype = example_inputs.dtype
        self._label_dtype = example_labels.dtype
        self.static_inputs = example_inputs.clone() if copy_inputs else example_inputs
        self.static_labels = example_labels.clone() if copy_inputs else example_labels
        self._borrowed_input_ptr = example_inputs.data_ptr()
        self._borrowed_label_ptr = example_labels.data_ptr()
        self.graph = torch.cuda.CUDAGraph()
        self.loss: Tensor | None = None

        self._materialize_optimizer_state()
        self._fused_optimizer_tail: FusedClipAdamW | None = None
        if fused_optimizer_tail:
            from .pac_cuda_fused_optimizer import FusedClipAdamW  # noqa: PLC0415

            self._fused_optimizer_tail = FusedClipAdamW.from_adamw(
                optimizer,
                max_norm=self.grad_clip_norm,
            )
        self._warmup_and_capture(warmup_steps)

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor:
        """Copy one batch, replay the update, and return its cross-entropy loss."""
        self._validate_step_tensors(inputs, labels)
        if self.copy_inputs:
            self.static_inputs.copy_(inputs, non_blocking=True)
            self.static_labels.copy_(labels, non_blocking=True)
        elif (
            inputs.data_ptr() != self._borrowed_input_ptr
            or labels.data_ptr() != self._borrowed_label_ptr
        ):
            message = "borrowed CUDA Graph training requires the captured input allocations"
            raise ValueError(message)
        _require_tf32_disabled()
        self.graph.replay()
        if self.loss is None:
            message = "EFP16 training CUDA Graph has no captured loss buffer"
            raise RuntimeError(message)
        result = self.loss.detach()
        return result.clone() if self.copy_loss else result

    def _validate_model_optimizer(self, inputs: Tensor, labels: Tensor) -> None:
        self._validate_example_tensors(inputs, labels)
        self._validate_parameter_optimizer(inputs)

    def _validate_example_tensors(self, inputs: Tensor, labels: Tensor) -> None:
        if not inputs.is_cuda or not labels.is_cuda:
            message = "EFP16 training CUDA Graph requires CUDA inputs and labels"
            raise ValueError(message)
        if inputs.device != labels.device:
            message = "EFP16 training inputs and labels must share one CUDA device"
            raise ValueError(message)
        if inputs.dtype != torch.float32:
            message = "EFP16 training CUDA Graph supports exact FP32 inputs only"
            raise ValueError(message)
        if labels.dtype != torch.int64:
            message = "cross-entropy labels must use torch.int64"
            raise ValueError(message)
        if inputs.ndim != 3 or inputs.shape[0] < 1 or inputs.shape[1] < 2:
            message = "EFP16 inputs must have shape [batch>=1,time>=2,channels]"
            raise ValueError(message)
        if inputs.shape[-1] != 1:
            message = "the optimized EFP16 training graph requires scalar raw inputs"
            raise ValueError(message)
        if labels.shape != (inputs.shape[0],):
            message = "EFP16 labels must have shape [batch]"
            raise ValueError(message)
        if not self.model.training:
            message = "EFP16 training CUDA Graph requires model.train()"
            raise ValueError(message)

    def _validate_parameter_optimizer(self, inputs: Tensor) -> None:
        parameters = tuple(self.model.parameters())
        if not parameters:
            message = "EFP16 training CUDA Graph requires trainable parameters"
            raise ValueError(message)
        for parameter in parameters:
            if parameter.device != inputs.device or parameter.dtype != torch.float32:
                message = "all EFP16 parameters must be FP32 tensors on the input CUDA device"
                raise ValueError(message)
        optimizer_parameters = tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in cast("list[Tensor]", group["params"])
        )
        if len(optimizer_parameters) != len(parameters) or any(
            optimizer_parameter is not parameter
            for optimizer_parameter, parameter in zip(optimizer_parameters, parameters, strict=True)
        ):
            message = "AdamW must own the model parameters exactly once and in model order"
            raise ValueError(message)
        for group in self.optimizer.param_groups:
            if group.get("fused") is not True or group.get("capturable") is not True:
                message = "EFP16 graph training requires fused, capturable AdamW"
                raise ValueError(message)
            if group.get("differentiable") is True:
                message = "differentiable AdamW is not supported by CUDA Graph training"
                raise ValueError(message)

    def _validate_step_tensors(self, inputs: Tensor, labels: Tensor) -> None:
        if tuple(inputs.shape) != self._input_shape or tuple(labels.shape) != self._label_shape:
            message = "CUDA Graph training requires the captured input and label shapes"
            raise ValueError(message)
        if inputs.device != self._device or labels.device != self._device:
            message = "CUDA Graph training requires tensors on the captured CUDA device"
            raise ValueError(message)
        if inputs.dtype != self._input_dtype or labels.dtype != self._label_dtype:
            message = "CUDA Graph training requires the captured input and label dtypes"
            raise ValueError(message)

    @torch.no_grad()
    def _materialize_optimizer_state(self) -> None:
        parameters = tuple(self.model.parameters())
        parameter_snapshot = tuple(parameter.detach().clone() for parameter in parameters)
        existing_state = {
            parameter: {
                key: value.detach().clone() if isinstance(value, Tensor) else value
                for key, value in self.optimizer.state.get(parameter, {}).items()
            }
            for parameter in parameters
        }
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            else:
                gradient = parameter.grad
                gradient.zero_()
        self.optimizer.step()
        for parameter, value in zip(parameters, parameter_snapshot, strict=True):
            parameter.copy_(value)
            state = self.optimizer.state[parameter]
            saved = existing_state[parameter]
            for key, state_value in state.items():
                if not isinstance(state_value, Tensor):
                    message = "capturable AdamW state must contain tensors only"
                    raise TypeError(message)
                previous = saved.get(key)
                if isinstance(previous, Tensor):
                    state_value.copy_(previous)
                elif previous is None:
                    state_value.zero_()
                else:
                    message = "capturable AdamW state changed representation during setup"
                    raise TypeError(message)
            gradient = parameter.grad
            if gradient is None:
                message = "EFP16 gradient buffer disappeared during AdamW setup"
                raise RuntimeError(message)
            gradient.zero_()

    def _warmup_and_capture(self, warmup_steps: int) -> None:
        parameters = tuple(self.model.parameters())
        parameter_snapshot = tuple(parameter.detach().clone() for parameter in parameters)
        optimizer_snapshot = _clone_optimizer_tensor_state(self.optimizer, parameters)
        capture_stream = torch.cuda.Stream(device=self._device)
        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(capture_stream):
            for _ in range(warmup_steps):
                self._training_step()
        torch.cuda.current_stream(self._device).wait_stream(capture_stream)
        torch.cuda.synchronize(self._device)
        _restore_training_state(
            parameters,
            parameter_snapshot,
            self.optimizer,
            optimizer_snapshot,
        )

        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        with (
            torch.cuda.stream(capture_stream),
            _capture_safe_edge_projector(self.model),
            torch.cuda.graph(self.graph, stream=capture_stream),
        ):
            self.loss = self._training_step()
        torch.cuda.current_stream(self._device).wait_stream(capture_stream)
        torch.cuda.synchronize(self._device)
        _restore_training_state(
            parameters,
            parameter_snapshot,
            self.optimizer,
            optimizer_snapshot,
        )

    def _training_step(self) -> Tensor:
        self.optimizer.zero_grad(set_to_none=False)
        loss = functional.cross_entropy(self.model(self.static_inputs), self.static_labels)
        loss.backward()
        if self._fused_optimizer_tail is None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm,
                foreach=True,
            )
            self.optimizer.step()
        else:
            self._fused_optimizer_tail.step()
        post_step = getattr(self.model, "post_optimizer_step", None)
        if callable(post_step):
            post_step()
        return loss


def prepare_efp16_training_cuda_graph(
    model: nn.Module,
    optimizer: AdamW,
    example_inputs: Tensor,
    example_labels: Tensor,
    *,
    grad_clip_norm: float = 1.0,
    warmup_steps: int = 3,
    copy_inputs: bool = True,
    copy_loss: bool = False,
    prepare_model: bool = True,
    taylor_degree: int = 12,
    scaling_steps: int = 3,
    matrix_exp_compute_dtype: torch.dtype | None = None,
    fused_recurrence_moments_backward_training: bool = False,
    fused_optimizer_tail: bool = False,
) -> EFP16TrainingCudaGraph:
    """Prepare an EFP16 model and its existing AdamW state for static-shape replay."""
    return EFP16TrainingCudaGraph(
        model,
        optimizer,
        example_inputs,
        example_labels,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        copy_inputs=copy_inputs,
        copy_loss=copy_loss,
        prepare_model=prepare_model,
        taylor_degree=taylor_degree,
        scaling_steps=scaling_steps,
        matrix_exp_compute_dtype=matrix_exp_compute_dtype,
        fused_recurrence_moments_backward_training=(fused_recurrence_moments_backward_training),
        fused_optimizer_tail=fused_optimizer_tail,
    )


def _enable_fused_recurrence_moments_backward(model: nn.Module) -> None:
    blocks = [getattr(model, "forward_block", None), getattr(model, "backward_block", None)]
    blocks.extend(getattr(model, "extra_blocks", []))
    for block in blocks:
        if block is not None:
            block.__dict__["fused_recurrence_moments_backward_training"] = True


def make_capturable_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Construct the fused AdamW variant required by graph replay."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
        capturable=True,
    )


def _validate_options(grad_clip_norm: float, warmup_steps: int) -> None:
    if not torch.isfinite(torch.tensor(grad_clip_norm)) or grad_clip_norm <= 0.0:
        message = "grad_clip_norm must be finite and positive"
        raise ValueError(message)
    if warmup_steps < 1:
        message = "CUDA Graph training requires at least one warmup step"
        raise ValueError(message)


def _disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _require_tf32_disabled() -> None:
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        message = "EFP16 exact-FP32 graph replay requires TF32 to remain disabled"
        raise RuntimeError(message)


def _clone_optimizer_tensor_state(
    optimizer: AdamW,
    parameters: tuple[nn.Parameter, ...],
) -> dict[nn.Parameter, dict[str, Tensor]]:
    snapshot: dict[nn.Parameter, dict[str, Tensor]] = {}
    for parameter in parameters:
        state: dict[str, Tensor] = {}
        for key, value in optimizer.state[parameter].items():
            if not isinstance(value, Tensor):
                message = "capturable AdamW state must contain tensors only"
                raise TypeError(message)
            state[key] = value.detach().clone()
        snapshot[parameter] = state
    return snapshot


@torch.no_grad()
def _restore_training_state(
    parameters: tuple[nn.Parameter, ...],
    parameter_snapshot: tuple[Tensor, ...],
    optimizer: AdamW,
    optimizer_snapshot: dict[nn.Parameter, dict[str, Tensor]],
) -> None:
    for parameter, value in zip(parameters, parameter_snapshot, strict=True):
        parameter.copy_(value)
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        else:
            parameter.grad.zero_()
        for key, saved in optimizer_snapshot[parameter].items():
            current = optimizer.state[parameter].get(key)
            if not isinstance(current, Tensor):
                message = "capturable AdamW state tensor disappeared during capture"
                raise TypeError(message)
            current.copy_(saved)


@contextmanager
def _capture_safe_edge_projector(model: nn.Module) -> Generator[None]:
    stem = getattr(model, "stem", None)
    projection = getattr(stem, "projection", None)
    weight = getattr(projection, "weight", None)
    original = getattr(stem, "project_weight_", None)
    if not isinstance(weight, Tensor) or weight.shape[1] != 2 or not callable(original):
        yield
        return

    @torch.no_grad()
    def project_weight_(self: nn.Module) -> None:
        active_projection = getattr(self, "projection", None)
        active_weight = getattr(active_projection, "weight", None)
        if not isinstance(active_weight, Tensor):
            message = "EFP16 edge projection weight disappeared during graph capture"
            raise TypeError(message)
        first = active_weight[:, 0]
        first = first / torch.linalg.vector_norm(first).clamp_min(1.0e-12)
        second = active_weight[:, 1]
        second = second - first * torch.dot(first, second)
        second = second / torch.linalg.vector_norm(second).clamp_min(1.0e-12)
        active_weight.copy_(torch.stack((first, second), dim=1))

    instance_dict = cast("dict[str, object]", stem.__dict__)
    prior_override = instance_dict.get("project_weight_")
    instance_dict["project_weight_"] = types.MethodType(project_weight_, stem)
    try:
        yield
    finally:
        if prior_override is None:
            instance_dict.pop("project_weight_", None)
        else:
            instance_dict["project_weight_"] = prior_override


__all__ = [
    "EFP16TrainingCudaGraph",
    "make_capturable_adamw",
    "prepare_efp16_training_cuda_graph",
]
