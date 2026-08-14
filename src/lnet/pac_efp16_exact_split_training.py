from __future__ import annotations

import copy
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch._inductor.cudagraph_trees import reset_cudagraph_trees
from torch.nn import functional

from .pac_native_matrix_exp_vjp import (
    MatrixExpDispatch,
    NativeMatrixExpReplay,
    make_native_matrix_exp_replay,
    matrix_exp_one_norm,
    matrix_exp_vjp_branch,
    matrix_exp_vjp_one_norm,
    reset_cuda_graphs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim import AdamW


ExactSplitCompileMode = Literal["default", "max-autotune-no-cudagraphs"]
ExactSplitLossKind = Literal["cross_entropy", "binary_cross_entropy", "mse"]


_active_tf32_guards = 0
_tf32_restore_state: tuple[bool, bool] | None = None


@dataclass(frozen=True, slots=True)
class _ParameterState:
    parameter: nn.Parameter
    value: Tensor
    gradient: Tensor | None
    gradient_reference: Tensor | None


@dataclass(frozen=True, slots=True)
class _BufferState:
    module: nn.Module
    name: str
    reference: Tensor | None
    value: Tensor | None


@dataclass(frozen=True, slots=True)
class _CallerState:
    parameters: tuple[_ParameterState, ...]
    buffers: tuple[_BufferState, ...]
    optimizer: dict[str, object]


@dataclass(frozen=True, slots=True)
class _AttributeState:
    owner: object
    name: str
    present: bool
    value: object


class _OrthogonalMap(Protocol):
    base: Tensor
    orthogonal_map: object


class _WeightParametrizations(Protocol):
    original: nn.Parameter

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _OrthogonalMap: ...


class _Parametrizations(Protocol):
    weight: _WeightParametrizations


class _ParametrizedFrame(Protocol):
    parametrizations: _Parametrizations
    weight: Tensor


class _ModelCrossEntropy(nn.Module):
    """Keep logits and classification loss inside one Dynamo graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        return functional.cross_entropy(self.model(inputs), labels)


class _ModelWeightedCrossEntropy(nn.Module):
    """Keep a fixed class-weighted objective inside the captured graph."""

    def __init__(self, model: nn.Module, weight: Tensor) -> None:
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        return functional.cross_entropy(self.model(inputs), labels, weight=self.weight)


class _ModelObjectiveLoss(nn.Module):
    """Keep an external-task objective inside the captured model graph."""

    def __init__(self, model: nn.Module, loss_kind: ExactSplitLossKind) -> None:
        super().__init__()
        self.model = model
        self.loss_kind = loss_kind

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        logits = self.model(inputs)
        if self.loss_kind == "cross_entropy":
            return functional.cross_entropy(logits, targets)
        if self.loss_kind == "binary_cross_entropy":
            return functional.binary_cross_entropy_with_logits(logits, targets)
        if self.loss_kind == "mse":
            return functional.mse_loss(logits.reshape_as(targets), targets)
        message = f"unsupported exact-split loss: {self.loss_kind}"
        raise AssertionError(message)


class _ModelMetadataObjectiveLoss(nn.Module):
    """Evaluate one fixed metadata signature without changing the regular graph."""

    def __init__(
        self,
        model: nn.Module,
        loss_kind: ExactSplitLossKind,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
        cross_entropy_weight: Tensor | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_kind = loss_kind
        self.time_delta = time_delta
        self.observation_mask = observation_mask
        self.valid_mask = valid_mask
        self.cross_entropy_weight = cross_entropy_weight

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        logits = self.model(
            inputs,
            time_delta=self.time_delta,
            observation_mask=self.observation_mask,
            valid_mask=self.valid_mask,
        )
        if self.loss_kind == "cross_entropy":
            return functional.cross_entropy(
                logits,
                targets,
                weight=self.cross_entropy_weight,
            )
        if self.loss_kind == "binary_cross_entropy":
            return functional.binary_cross_entropy_with_logits(logits, targets)
        if self.loss_kind == "mse":
            return functional.mse_loss(logits.reshape_as(targets), targets)
        message = f"unsupported exact-split loss: {self.loss_kind}"
        raise AssertionError(message)


def _loss_module(
    model: nn.Module,
    loss_kind: ExactSplitLossKind,
    time_delta: Tensor | None,
    observation_mask: Tensor | None,
    valid_mask: Tensor | None,
    cross_entropy_weight: Tensor | None,
) -> nn.Module:
    if time_delta is None and observation_mask is None and valid_mask is None:
        if loss_kind == "cross_entropy":
            return (
                _ModelCrossEntropy(model)
                if cross_entropy_weight is None
                else _ModelWeightedCrossEntropy(model, cross_entropy_weight)
            )
        return _ModelObjectiveLoss(model, loss_kind)
    return _ModelMetadataObjectiveLoss(
        model,
        loss_kind,
        time_delta,
        observation_mask,
        valid_mask,
        cross_entropy_weight,
    )


class _ExactSplitBlock(Protocol):
    frame: _ParametrizedFrame
    __dict__: dict[str, object]

    def set_intervention_frame(self, value: Tensor | None) -> None: ...

    def intervention_frame(self) -> Tensor | None: ...


class EFP16ExactSplitTraining:
    """Train EFP16 with native ``matrix_exp`` around two CUDA Graphs.

    PyTorch's native orthogonal parametrization is not CUDA-Graph capture safe.
    Replacing it with an approximation makes the full step capturable, but can
    drift from the campaign's FP32 training trajectory on high-batch cells.
    This runtime keeps the native parametrization and its native backward
    outside capture.  A first graph computes the rest of forward/backward into
    differentiable frame overrides; native autograd then transfers those frame
    gradients to the original Stiefel parameters. A capturable optimizer gets
    a second graph for clipping and AdamW; otherwise the caller's exact eager
    optimizer implementation runs after the body replay.

    The model and optimizer remain caller-owned.  Input copies are included in
    :meth:`step`, and the original parameter names and identities are retained.

    Construction is an exclusive CUDA-Graph lifecycle boundary: it resets
    Inductor's process-local CUDA Graph tree so previously compiled callables
    may recapture on their next invocation.  Keep this runtime dedicated to a
    training interval and call :meth:`close` before resuming eager execution.
    """

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        model: nn.Module,
        optimizer: AdamW,
        example_inputs: Tensor,
        example_labels: Tensor,
        *,
        example_time_delta: Tensor | None = None,
        example_observation_mask: Tensor | None = None,
        example_valid_mask: Tensor | None = None,
        cross_entropy_weight: Tensor | None = None,
        validate_metadata_values: bool = True,
        grad_clip_norm: float = 1.0,
        warmup_steps: int = 2,
        copy_loss: bool = False,
        recurrence_backend: str = "auto",
        parallel_native_frames: bool = False,
        parallel_specialized_host_frames: bool = False,
        parallel_cuda_switch_frames: bool = False,
        parallel_cuda_switch_lane_dag: bool = False,
        fused_c2_stem_training: bool = False,
        fused_moments_backward_training: bool = True,
        fused_recurrence_moments_backward_training: bool = False,
        capture_post_optimizer_step: bool = False,
        specialized_matrix_exp_vjp: bool = False,
        matrix_exp_dispatch: MatrixExpDispatch = "host",
        matrix_exp_forward_tf32: bool = False,
        direct_skew_matrix_exp_vjp: bool = False,
        allow_multichannel_inputs: bool = False,
        compile_model_body: bool = False,
        compile_training_loss: bool = False,
        training_compile_mode: ExactSplitCompileMode = "default",
        loss_kind: ExactSplitLossKind = "cross_entropy",
    ) -> None:
        _validate_options(grad_clip_norm, warmup_steps, training_compile_mode)
        if loss_kind not in {"cross_entropy", "binary_cross_entropy", "mse"}:
            message = f"unsupported exact-split loss kind: {loss_kind}"
            raise ValueError(message)
        self.model = model
        self.optimizer = optimizer
        self.grad_clip_norm = float(grad_clip_norm)
        self.copy_loss = copy_loss
        self.parallel_native_frames = parallel_native_frames
        self.parallel_specialized_host_frames = parallel_specialized_host_frames
        self.parallel_cuda_switch_frames = parallel_cuda_switch_frames
        self.parallel_cuda_switch_lane_dag = parallel_cuda_switch_lane_dag
        self.fused_c2_stem_training = fused_c2_stem_training
        if specialized_matrix_exp_vjp and parallel_native_frames:
            message = "specialized matrix-exp VJP does not support parallel frame streams"
            raise ValueError(message)
        if parallel_cuda_switch_frames and parallel_native_frames:
            message = "native-frame and CUDA SWITCH parallelism are mutually exclusive"
            raise ValueError(message)
        if parallel_specialized_host_frames and (
            not specialized_matrix_exp_vjp
            or matrix_exp_dispatch != "host"
            or parallel_native_frames
            or parallel_cuda_switch_frames
        ):
            message = (
                "parallel specialized host frames require host-dispatched specialized "
                "matrix-exp VJPs without another frame parallelism mode"
            )
            raise ValueError(message)
        if parallel_cuda_switch_frames and (
            not specialized_matrix_exp_vjp or matrix_exp_dispatch != "cuda_switch"
        ):
            message = (
                "parallel CUDA SWITCH frames require specialized_matrix_exp_vjp=True "
                "and matrix_exp_dispatch='cuda_switch'"
            )
            raise ValueError(message)
        if parallel_cuda_switch_lane_dag and not parallel_cuda_switch_frames:
            message = "lane-local CUDA SWITCH dependencies require parallel frame runtimes"
            raise ValueError(message)
        self.specialized_matrix_exp_vjp = specialized_matrix_exp_vjp
        if matrix_exp_dispatch != "host" and not specialized_matrix_exp_vjp:
            message = "matrix-exp dispatch requires specialized_matrix_exp_vjp=True"
            raise ValueError(message)
        if matrix_exp_forward_tf32 and not specialized_matrix_exp_vjp:
            message = "forward-only TF32 requires specialized_matrix_exp_vjp=True"
            raise ValueError(message)
        if direct_skew_matrix_exp_vjp and (
            not specialized_matrix_exp_vjp
            or matrix_exp_dispatch != "cuda_switch"
            or not parallel_cuda_switch_frames
        ):
            message = (
                "direct skew matrix-exp VJP requires CUDA SWITCH dispatch and "
                "distinct parallel frame runtimes"
            )
            raise ValueError(message)
        self.matrix_exp_dispatch = matrix_exp_dispatch
        self.matrix_exp_forward_tf32 = matrix_exp_forward_tf32
        self.direct_skew_matrix_exp_vjp = direct_skew_matrix_exp_vjp
        self.allow_multichannel_inputs = allow_multichannel_inputs
        self.recurrence_backend = (
            "triton_fused_opaque"
            if (compile_model_body or compile_training_loss) and recurrence_backend == "auto"
            else recurrence_backend
        )
        self.compile_training_loss = compile_training_loss
        self.training_compile_mode = training_compile_mode
        self.loss_kind = loss_kind
        self.validate_metadata_values = validate_metadata_values
        self.fused_moments_backward_training = fused_moments_backward_training
        self.fused_recurrence_moments_backward_training = fused_recurrence_moments_backward_training
        self._validate_inputs(example_inputs, example_labels)
        _validate_metadata_tensors(
            example_inputs,
            example_time_delta,
            example_observation_mask,
            example_valid_mask,
        )
        _validate_cross_entropy_weight(
            example_inputs,
            cross_entropy_weight,
            loss_kind=loss_kind,
        )
        self._validate_model_optimizer(example_inputs)
        self._has_request_metadata = any(
            value is not None
            for value in (
                example_time_delta,
                example_observation_mask,
                example_valid_mask,
            )
        )
        self.blocks = _native_matrix_exp_blocks(model)
        caller_state = _snapshot_caller_state(model, optimizer)
        attribute_state = _snapshot_runtime_attributes(model, self.blocks)
        self._attribute_state = attribute_state
        # ``torch.compile(mode="reduce-overhead")`` keeps an Inductor CUDA
        # Graph tree whose RNG offset can be an inference tensor.  A later
        # manual graph capture then fails before entering capture because that
        # offset would be updated outside inference mode.  Exact-split owns a
        # separate set of manual graphs, so release only Inductor's graph tree
        # before constructing them; compiled callables may recapture lazily.
        reset_cudagraph_trees()
        _acquire_tf32_guard()
        self._tf32_finalizer = weakref.finalize(self, _release_tf32_guard)
        self._active = False
        self._destroyed = False

        try:
            self._device = example_inputs.device
            self._input_shape = tuple(example_inputs.shape)
            self._label_shape = tuple(example_labels.shape)
            self.static_inputs = example_inputs.clone()
            self.static_labels = example_labels.clone()
            self.static_time_delta = (
                None if example_time_delta is None else example_time_delta.clone()
            )
            self.static_observation_mask = (
                None
                if example_observation_mask is None
                else example_observation_mask.clone()
            )
            self.static_valid_mask = (
                None if example_valid_mask is None else example_valid_mask.clone()
            )
            self.static_cross_entropy_weight = (
                None if cross_entropy_weight is None else cross_entropy_weight.clone()
            )
            self.forward_backward_graph = torch.cuda.CUDAGraph(keep_graph=True)
            self.optimizer_graph: torch.cuda.CUDAGraph | None = None
            self.post_optimizer_graph: torch.cuda.CUDAGraph | None = None
            self.capture_optimizer_tail = all(
                group.get("capturable") is True for group in optimizer.param_groups
            )
            self.capture_post_optimizer_step = capture_post_optimizer_step
            self._post_step_in_optimizer_graph = (
                capture_post_optimizer_step and self.capture_optimizer_tail
            )
            self.loss: Tensor | None = None

            self.frame_streams = tuple(torch.cuda.Stream(device=self._device) for _ in self.blocks)
            _apply_runtime_attributes(
                model,
                self.blocks,
                recurrence_backend=self.recurrence_backend,
                fused_moments_backward_training=fused_moments_backward_training,
                fused_recurrence_moments_backward_training=(
                    fused_recurrence_moments_backward_training
                ),
                fused_c2_stem_training=fused_c2_stem_training,
                disable_metadata_validation=self._has_request_metadata,
            )

            self._captured_matrix_exp_vjps = self._prepare_matrix_exp_replays()
            frame_values, _ = self._native_frame_evaluations()
            self.frame_overrides = tuple(
                value.detach().clone().requires_grad_(True)  # noqa: FBT003
                for value in frame_values
            )
            for block, override in zip(self.blocks, self.frame_overrides, strict=True):
                block.set_intervention_frame(override)
            # The manual exact-split graph remains the sole CUDA Graph owner.
            # Compiling through cross-entropy lets Inductor fuse the logits/loss
            # boundary while still emitting ordinary kernels for our capture.
            # When both switches are enabled this wider graph subsumes the old
            # model-only graph instead of compiling the model twice.
            loss_module = _loss_module(
                model,
                loss_kind,
                self.static_time_delta,
                self.static_observation_mask,
                self.static_valid_mask,
                self.static_cross_entropy_weight,
            )
            if compile_training_loss:
                self._training_model = model
                self._training_loss = cast(
                    "Callable[[Tensor, Tensor], Tensor]",
                    _compile_no_cudagraphs(
                        loss_module,
                        mode=training_compile_mode,
                    ),
                )
            else:
                self._training_model = (
                    cast(
                        "nn.Module",
                        _compile_no_cudagraphs(model, mode=training_compile_mode),
                    )
                    if compile_model_body
                    else model
                )
                self._training_loss = cast(
                    "Callable[[Tensor, Tensor], Tensor]",
                    _loss_module(
                        self._training_model,
                        loss_kind,
                        self.static_time_delta,
                        self.static_observation_mask,
                        self.static_valid_mask,
                        self.static_cross_entropy_weight,
                    ),
                )
            self._tf32_finalizer.detach()
            self._tf32_finalizer = _make_runtime_finalizer(
                self,
                self.blocks,
                self.frame_overrides,
                self._attribute_state,
            )
            self._materialize_optimizer_state()
            self._warmup_and_capture(warmup_steps)
            self._active = True
        except BaseException:
            try:
                # Detach overrides installed by this construction attempt
                # before restoring the caller's original buffer references.
                _clear_intervention_frames(model)
                _restore_caller_state(caller_state, optimizer)
                _restore_runtime_attributes(attribute_state)
            finally:
                self._tf32_finalizer()
            raise

    def step(
        self,
        inputs: Tensor,
        labels: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Run one exact-native-matrix-exp FP32 parameter update."""
        self._validate_step_tensors(
            inputs,
            labels,
            time_delta,
            observation_mask,
            valid_mask,
        )
        self.activate()
        _require_tf32_disabled()
        self.static_inputs.copy_(inputs, non_blocking=True)
        self.static_labels.copy_(labels, non_blocking=True)
        if self._has_request_metadata:
            _copy_optional_static_tensor(self.static_time_delta, time_delta)
            _copy_optional_static_tensor(self.static_observation_mask, observation_mask)
            _copy_optional_static_tensor(self.static_valid_mask, valid_mask)

        native_frames, skew_matrices = self._native_frame_evaluations()
        with torch.no_grad():
            for override, frame in zip(self.frame_overrides, native_frames, strict=True):
                override.copy_(frame)
        self.forward_backward_graph.replay()
        self._transfer_frame_gradients(skew_matrices)
        if self.optimizer_graph is None:
            self._optimizer_body()
        else:
            self.optimizer_graph.replay()
        if self.post_optimizer_graph is not None:
            self.post_optimizer_graph.replay()
        elif not self._post_step_in_optimizer_graph:
            self._post_optimizer_step()
        if self.loss is None:
            message = "EFP16 exact split runtime has no captured loss buffer"
            raise RuntimeError(message)
        result = self.loss.detach()
        return result.clone() if self.copy_loss else result

    def close(self) -> None:
        """Detach overrides while retaining graphs for later reactivation."""
        try:
            _clear_owned_intervention_frames(self.blocks, self.frame_overrides)
        finally:
            self._active = False
            self._tf32_finalizer()

    def destroy(self) -> None:
        """Permanently release graph executables, pools, and borrowed buffers."""
        if self._destroyed:
            return
        torch.cuda.synchronize(self._device)
        self.close()
        destroyed_replays: set[int] = set()
        for replay in self._captured_matrix_exp_vjps:
            if id(replay) in destroyed_replays:
                continue
            destroyed_replays.add(id(replay))
            replay.destroy()
        self._captured_matrix_exp_vjps = ()
        reset_cuda_graphs(
            (
                self.forward_backward_graph,
                self.optimizer_graph,
                self.post_optimizer_graph,
            )
        )
        self.optimizer_graph = None
        self.post_optimizer_graph = None
        self.frame_streams = ()
        self.frame_overrides = ()
        self.loss = None
        self.__dict__.pop("static_inputs", None)
        self.__dict__.pop("static_labels", None)
        self.__dict__.pop("static_time_delta", None)
        self.__dict__.pop("static_observation_mask", None)
        self.__dict__.pop("static_valid_mask", None)
        self.__dict__.pop("static_cross_entropy_weight", None)
        self.__dict__.pop("_training_loss", None)
        self.__dict__.pop("_training_model", None)
        self._destroyed = True
        torch.cuda.synchronize(self._device)

    def activate(self) -> None:
        """Restore captured frame overrides after an eager/evaluation interval."""
        if self._destroyed:
            message = "a destroyed exact-split runtime cannot be reactivated"
            raise RuntimeError(message)
        if getattr(self, "_active", False):
            return
        _acquire_tf32_guard()
        self._tf32_finalizer = _make_runtime_finalizer(
            self,
            self.blocks,
            self.frame_overrides,
            self._attribute_state,
        )
        try:
            _apply_runtime_attributes(
                self.model,
                self.blocks,
                recurrence_backend=self.recurrence_backend,
                fused_moments_backward_training=self.fused_moments_backward_training,
                fused_recurrence_moments_backward_training=(
                    self.fused_recurrence_moments_backward_training
                ),
                fused_c2_stem_training=self.fused_c2_stem_training,
                disable_metadata_validation=self._has_request_metadata,
            )
            for block, override in zip(self.blocks, self.frame_overrides, strict=True):
                block.set_intervention_frame(override)
        except BaseException:
            try:
                _clear_intervention_frames(self.model)
            finally:
                self._tf32_finalizer()
            raise
        self._active = True

    @torch.no_grad()
    def _native_frame_evaluations(self) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        if self.specialized_matrix_exp_vjp:
            specialized_skews = tuple(_native_frame_skew(block) for block in self.blocks)
            if self.matrix_exp_dispatch == "cuda_switch":
                switch_frames = tuple(
                    (block.frame.parametrizations.weight[0].base @ runtime.replay_forward(skew))[
                        ..., : block.frame.parametrizations.weight.original.shape[-1]
                    ]
                    for block, runtime, skew in zip(
                        self.blocks,
                        self._captured_matrix_exp_vjps,
                        specialized_skews,
                        strict=True,
                    )
                )
                return switch_frames, specialized_skews
            norms = torch.stack(tuple(matrix_exp_one_norm(skew) for skew in specialized_skews))
            host_norms = tuple(float(value) for value in norms.cpu().tolist())
            if self.parallel_specialized_host_frames:
                return (
                    self._parallel_specialized_frame_evaluations(
                        specialized_skews,
                        host_norms,
                    ),
                    specialized_skews,
                )
            specialized_frames: list[Tensor] = []
            for block, runtime, skew, norm in zip(
                self.blocks,
                self._captured_matrix_exp_vjps,
                specialized_skews,
                host_norms,
                strict=True,
            ):
                branch = matrix_exp_vjp_branch(norm)
                orthogonal = (
                    torch.matrix_exp(skew)
                    if branch is None
                    else runtime.replay_forward(skew, branch)
                )
                frame = block.frame.parametrizations.weight[0].base @ orthogonal
                specialized_frames.append(
                    frame[..., : block.frame.parametrizations.weight.original.shape[-1]]
                )
            return tuple(specialized_frames), specialized_skews
        if self.parallel_native_frames:
            current_stream = torch.cuda.current_stream(self._device)
            evaluations: list[tuple[Tensor, Tensor]] = []
            for block, stream in zip(self.blocks, self.frame_streams, strict=True):
                stream.wait_stream(current_stream)
                with torch.cuda.stream(stream):
                    evaluations.append(_evaluate_native_frame(block))
            for stream in self.frame_streams:
                current_stream.wait_stream(stream)
            return (
                tuple(frame for frame, _ in evaluations),
                tuple(skew for _, skew in evaluations),
            )
        frames: list[Tensor] = []
        skew_matrices: list[Tensor] = []
        for block in self.blocks:
            frame, skew = _evaluate_native_frame(block)
            frames.append(frame)
            skew_matrices.append(skew)
        return tuple(frames), tuple(skew_matrices)

    @torch.no_grad()
    def _transfer_frame_gradients(self, skew_matrices: tuple[Tensor, ...]) -> None:
        if self.specialized_matrix_exp_vjp:
            exponential_gradients = tuple(
                _frame_exponential_gradient(block, override)
                for block, override in zip(self.blocks, self.frame_overrides, strict=True)
            )
            if self.matrix_exp_dispatch == "cuda_switch":
                for block, runtime, skew, gradient in zip(
                    self.blocks,
                    self._captured_matrix_exp_vjps,
                    skew_matrices,
                    exponential_gradients,
                    strict=True,
                ):
                    _write_frame_coordinate_gradient(
                        block,
                        runtime.replay(skew, gradient),
                    )
                return
            norms = torch.stack(
                tuple(
                    matrix_exp_vjp_one_norm(skew, gradient)
                    for skew, gradient in zip(skew_matrices, exponential_gradients, strict=True)
                )
            )
            host_norms = tuple(float(value) for value in norms.cpu().tolist())
            if self.parallel_specialized_host_frames:
                self._parallel_specialized_frame_gradient_transfer(
                    skew_matrices,
                    exponential_gradients,
                    host_norms,
                )
                return
            for block, runtime, skew, gradient, norm in zip(
                self.blocks,
                self._captured_matrix_exp_vjps,
                skew_matrices,
                exponential_gradients,
                host_norms,
                strict=True,
            ):
                branch = matrix_exp_vjp_branch(norm)
                skew_gradient = (
                    torch.ops.aten.matrix_exp_backward(skew, gradient)
                    if branch is None
                    else runtime.replay(skew, gradient, branch)
                )
                _write_frame_coordinate_gradient(block, skew_gradient)
            return
        if self.parallel_native_frames:
            current_stream = torch.cuda.current_stream(self._device)
            for block, override, skew, stream in zip(
                self.blocks,
                self.frame_overrides,
                skew_matrices,
                self.frame_streams,
                strict=True,
            ):
                stream.wait_stream(current_stream)
                with torch.cuda.stream(stream):
                    _transfer_native_frame_gradient(block, override, skew)
            for stream in self.frame_streams:
                current_stream.wait_stream(stream)
            return
        for block, override, skew in zip(
            self.blocks, self.frame_overrides, skew_matrices, strict=True
        ):
            _transfer_native_frame_gradient(block, override, skew)

    def _parallel_specialized_frame_evaluations(
        self,
        skew_matrices: tuple[Tensor, ...],
        host_norms: tuple[float, ...],
    ) -> tuple[Tensor, ...]:
        current_stream = torch.cuda.current_stream(self._device)
        frames: list[Tensor] = []
        for block, runtime, skew, norm, stream in zip(
            self.blocks,
            self._captured_matrix_exp_vjps,
            skew_matrices,
            host_norms,
            self.frame_streams,
            strict=True,
        ):
            branch = matrix_exp_vjp_branch(norm)
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                orthogonal = (
                    torch.matrix_exp(skew)
                    if branch is None
                    else runtime.replay_forward(skew, branch)
                )
                frame = block.frame.parametrizations.weight[0].base @ orthogonal
                frames.append(
                    frame[
                        ...,
                        : block.frame.parametrizations.weight.original.shape[-1],
                    ]
                )
        for stream in self.frame_streams:
            current_stream.wait_stream(stream)
        return tuple(frames)

    def _parallel_specialized_frame_gradient_transfer(
        self,
        skew_matrices: tuple[Tensor, ...],
        exponential_gradients: tuple[Tensor, ...],
        host_norms: tuple[float, ...],
    ) -> None:
        current_stream = torch.cuda.current_stream(self._device)
        for block, runtime, skew, gradient, norm, stream in zip(
            self.blocks,
            self._captured_matrix_exp_vjps,
            skew_matrices,
            exponential_gradients,
            host_norms,
            self.frame_streams,
            strict=True,
        ):
            branch = matrix_exp_vjp_branch(norm)
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                skew_gradient = (
                    torch.ops.aten.matrix_exp_backward(skew, gradient)
                    if branch is None
                    else runtime.replay(skew, gradient, branch)
                )
                _write_frame_coordinate_gradient(block, skew_gradient)
        for stream in self.frame_streams:
            current_stream.wait_stream(stream)

    def _prepare_matrix_exp_replays(self) -> tuple[NativeMatrixExpReplay, ...]:
        if not self.specialized_matrix_exp_vjp:
            return ()
        sizes = tuple(
            max(block.frame.parametrizations.weight.original.shape[-2:]) for block in self.blocks
        )
        if self.matrix_exp_dispatch == "cuda_switch":
            if len(set(sizes)) != 1:
                message = "shared CUDA SWITCH matrix-exp requires equal frame sizes"
                raise ValueError(message)
            if self.parallel_cuda_switch_frames:
                return tuple(
                    make_native_matrix_exp_replay(
                        size,
                        self._device,
                        dispatch="cuda_switch",
                        forward_tf32=self.matrix_exp_forward_tf32,
                        direct_skew_vjp=self.direct_skew_matrix_exp_vjp,
                    )
                    for size in sizes
                )
            shared = make_native_matrix_exp_replay(
                sizes[0],
                self._device,
                dispatch="cuda_switch",
                forward_tf32=self.matrix_exp_forward_tf32,
                direct_skew_vjp=self.direct_skew_matrix_exp_vjp,
            )
            return tuple(shared for _ in sizes)
        return tuple(
            make_native_matrix_exp_replay(
                size,
                self._device,
                dispatch="host",
                forward_tf32=self.matrix_exp_forward_tf32,
                direct_skew_vjp=self.direct_skew_matrix_exp_vjp,
            )
            for size in sizes
        )

    def _validate_inputs(self, inputs: Tensor, labels: Tensor) -> None:
        if not inputs.is_cuda or not labels.is_cuda:
            message = "EFP16 exact split training requires CUDA inputs and labels"
            raise ValueError(message)
        if inputs.device != labels.device:
            message = "EFP16 inputs and labels must share one CUDA device"
            raise ValueError(message)
        if inputs.dtype != torch.float32:
            message = "EFP16 exact split training requires FP32 inputs"
            raise ValueError(message)
        if inputs.ndim != 3 or inputs.shape[0] < 1 or inputs.shape[1] < 2:
            message = "EFP16 inputs must have shape [batch>=1,time>=2,channels]"
            raise ValueError(message)
        if inputs.shape[-1] != 1 and not self.allow_multichannel_inputs:
            message = "EFP16 requires scalar inputs unless multichannel inputs are enabled"
            raise ValueError(message)
        if self.loss_kind == "cross_entropy":
            if labels.dtype != torch.int64 or labels.shape != (inputs.shape[0],):
                message = "cross-entropy exact split requires one int64 label per batch item"
                raise ValueError(message)
        elif labels.dtype != torch.float32 or labels.shape[0] != inputs.shape[0]:
            message = "BCE/MSE exact split requires FP32 targets with the same batch size"
            raise ValueError(message)
        if not self.model.training:
            message = "EFP16 exact split training requires model.train()"
            raise ValueError(message)

    def _validate_model_optimizer(self, inputs: Tensor) -> None:
        parameters = tuple(self.model.parameters())
        if not parameters:
            message = "EFP16 exact split training requires trainable parameters"
            raise ValueError(message)
        if any(
            parameter.device != inputs.device or parameter.dtype != torch.float32
            for parameter in parameters
        ):
            message = "all EFP16 parameters must be FP32 tensors on the input device"
            raise ValueError(message)
        optimizer_parameters = tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in cast("list[Tensor]", group["params"])
        )
        if len(optimizer_parameters) != len(parameters) or any(
            actual is not expected
            for actual, expected in zip(optimizer_parameters, parameters, strict=True)
        ):
            message = "AdamW must own the model parameters exactly once and in model order"
            raise ValueError(message)
        for group in self.optimizer.param_groups:
            if group.get("differentiable") is True:
                message = "exact split training does not support differentiable AdamW"
                raise ValueError(message)

    def _validate_step_tensors(
        self,
        inputs: Tensor,
        labels: Tensor,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> None:
        if tuple(inputs.shape) != self._input_shape or tuple(labels.shape) != self._label_shape:
            message = "exact split training requires the captured input and label shapes"
            raise ValueError(message)
        if inputs.device != self._device or labels.device != self._device:
            message = "exact split training requires tensors on the captured CUDA device"
            raise ValueError(message)
        if inputs.dtype != torch.float32 or labels.dtype != self.static_labels.dtype:
            message = "exact split training requires the captured input and target dtypes"
            raise ValueError(message)
        if not self._has_request_metadata:
            if any(
                value is not None
                for value in (time_delta, observation_mask, valid_mask)
            ):
                message = "exact split training requires the captured metadata presence"
                raise ValueError(message)
            return
        _validate_metadata_tensors(
            inputs,
            time_delta,
            observation_mask,
            valid_mask,
            validate_values=self.validate_metadata_values,
        )
        for name, static, current in (
            ("time_delta", self.static_time_delta, time_delta),
            ("observation_mask", self.static_observation_mask, observation_mask),
            ("valid_mask", self.static_valid_mask, valid_mask),
        ):
            if (static is None) != (current is None):
                message = f"exact split training requires the captured {name} presence"
                raise ValueError(message)
            if static is not None and current is not None and (
                static.shape != current.shape or static.dtype != current.dtype
            ):
                message = f"exact split training requires the captured {name} shape and dtype"
                raise ValueError(message)

    @torch.no_grad()
    def _materialize_optimizer_state(self) -> None:
        parameters = tuple(self.model.parameters())
        parameter_snapshot = tuple(parameter.detach().clone() for parameter in parameters)
        existing_state = _clone_optimizer_state(self.optimizer, parameters, missing_ok=True)
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        self.optimizer.step()
        for parameter, value in zip(parameters, parameter_snapshot, strict=True):
            parameter.copy_(value)
            saved = existing_state.get(parameter, {})
            for key, current in self.optimizer.state[parameter].items():
                if not isinstance(current, Tensor):
                    message = "capturable AdamW state must contain tensors only"
                    raise TypeError(message)
                previous = saved.get(key)
                current.zero_() if previous is None else current.copy_(previous)
            _require_gradient(parameter, "model parameter").zero_()
        for override in self.frame_overrides:
            override.grad = torch.zeros_like(override)

    def _warmup_and_capture(self, warmup_steps: int) -> None:
        parameters = tuple(self.model.parameters())
        parameter_snapshot = tuple(parameter.detach().clone() for parameter in parameters)
        optimizer_snapshot = _clone_optimizer_state(self.optimizer, parameters)

        for _ in range(warmup_steps):
            self._uncaptured_split_step()
        torch.cuda.synchronize(self._device)
        _restore_state(parameters, parameter_snapshot, self.optimizer, optimizer_snapshot)

        native_frames, skew_matrices = self._native_frame_evaluations()
        with torch.no_grad():
            for override, frame in zip(self.frame_overrides, native_frames, strict=True):
                override.copy_(frame)

        capture_stream = torch.cuda.Stream(device=self._device)
        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(capture_stream):
            # Prime the exact allocation pattern that capture will reuse.
            self._forward_backward_body()
        torch.cuda.current_stream(self._device).wait_stream(capture_stream)
        torch.cuda.synchronize(self._device)
        _restore_state(parameters, parameter_snapshot, self.optimizer, optimizer_snapshot)

        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        with (
            torch.cuda.stream(capture_stream),
            torch.cuda.graph(self.forward_backward_graph, stream=capture_stream),
        ):
            self.loss = self._forward_backward_body()
        torch.cuda.current_stream(self._device).wait_stream(capture_stream)
        torch.cuda.synchronize(self._device)

        if self.capture_optimizer_tail:
            self._transfer_frame_gradients(skew_matrices)
            self.optimizer_graph = torch.cuda.CUDAGraph(keep_graph=True)
            capture_stream.wait_stream(torch.cuda.current_stream(self._device))
            with (
                torch.cuda.stream(capture_stream),
                torch.cuda.graph(self.optimizer_graph, stream=capture_stream),
            ):
                self._optimizer_graph_body()
            torch.cuda.current_stream(self._device).wait_stream(capture_stream)
            torch.cuda.synchronize(self._device)
        if self.capture_post_optimizer_step and not self.capture_optimizer_tail:
            self.post_optimizer_graph = torch.cuda.CUDAGraph(keep_graph=True)
            capture_stream.wait_stream(torch.cuda.current_stream(self._device))
            with torch.no_grad(), torch.cuda.stream(capture_stream):
                self._post_optimizer_step()
            torch.cuda.current_stream(self._device).wait_stream(capture_stream)
            torch.cuda.synchronize(self._device)
            capture_stream.wait_stream(torch.cuda.current_stream(self._device))
            with (
                torch.no_grad(),
                torch.cuda.stream(capture_stream),
                torch.cuda.graph(self.post_optimizer_graph, stream=capture_stream),
            ):
                self._post_optimizer_step()
            torch.cuda.current_stream(self._device).wait_stream(capture_stream)
            torch.cuda.synchronize(self._device)
        _restore_state(parameters, parameter_snapshot, self.optimizer, optimizer_snapshot)

    def _uncaptured_split_step(self) -> Tensor:
        native_frames, skew_matrices = self._native_frame_evaluations()
        with torch.no_grad():
            for override, frame in zip(self.frame_overrides, native_frames, strict=True):
                override.copy_(frame)
        loss = self._forward_backward_body()
        self._transfer_frame_gradients(skew_matrices)
        self._optimizer_body()
        self._post_optimizer_step()
        return loss

    def _forward_backward_body(self) -> Tensor:
        self.optimizer.zero_grad(set_to_none=False)
        for override in self.frame_overrides:
            _require_gradient(override, "frame override").zero_()
        loss = self._training_loss(self.static_inputs, self.static_labels)
        loss.backward()
        return loss

    def _optimizer_body(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm, foreach=True)
        self.optimizer.step()

    def _optimizer_graph_body(self) -> None:
        self._optimizer_body()
        if self._post_step_in_optimizer_graph:
            self._post_optimizer_step()

    def _post_optimizer_step(self) -> None:
        post_step = getattr(self.model, "post_optimizer_step", None)
        if callable(post_step):
            post_step()


def prepare_efp16_exact_split_training(
    model: nn.Module,
    optimizer: AdamW,
    example_inputs: Tensor,
    example_labels: Tensor,
    *,
    example_time_delta: Tensor | None = None,
    example_observation_mask: Tensor | None = None,
    example_valid_mask: Tensor | None = None,
    cross_entropy_weight: Tensor | None = None,
    validate_metadata_values: bool = True,
    grad_clip_norm: float = 1.0,
    warmup_steps: int = 2,
    copy_loss: bool = False,
    recurrence_backend: str = "auto",
    parallel_native_frames: bool = False,
    parallel_specialized_host_frames: bool = False,
    parallel_cuda_switch_frames: bool = False,
    parallel_cuda_switch_lane_dag: bool = False,
    fused_c2_stem_training: bool = False,
    fused_moments_backward_training: bool = True,
    fused_recurrence_moments_backward_training: bool = False,
    capture_post_optimizer_step: bool = False,
    specialized_matrix_exp_vjp: bool = False,
    matrix_exp_dispatch: MatrixExpDispatch = "host",
    matrix_exp_forward_tf32: bool = False,
    direct_skew_matrix_exp_vjp: bool = False,
    allow_multichannel_inputs: bool = False,
    compile_model_body: bool = False,
    compile_training_loss: bool = False,
    training_compile_mode: ExactSplitCompileMode = "default",
    loss_kind: ExactSplitLossKind = "cross_entropy",
) -> EFP16ExactSplitTraining:
    """Build an exclusive exact-native-matrix-exp split-graph runtime.

    Setup resets Inductor's process-local CUDA Graph tree. Previously compiled
    callables remain usable but may recapture on their next invocation.
    """
    return EFP16ExactSplitTraining(
        model,
        optimizer,
        example_inputs,
        example_labels,
        example_time_delta=example_time_delta,
        example_observation_mask=example_observation_mask,
        example_valid_mask=example_valid_mask,
        cross_entropy_weight=cross_entropy_weight,
        validate_metadata_values=validate_metadata_values,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        copy_loss=copy_loss,
        recurrence_backend=recurrence_backend,
        parallel_native_frames=parallel_native_frames,
        parallel_specialized_host_frames=parallel_specialized_host_frames,
        parallel_cuda_switch_frames=parallel_cuda_switch_frames,
        parallel_cuda_switch_lane_dag=parallel_cuda_switch_lane_dag,
        fused_c2_stem_training=fused_c2_stem_training,
        fused_moments_backward_training=fused_moments_backward_training,
        fused_recurrence_moments_backward_training=(fused_recurrence_moments_backward_training),
        capture_post_optimizer_step=capture_post_optimizer_step,
        specialized_matrix_exp_vjp=specialized_matrix_exp_vjp,
        matrix_exp_dispatch=matrix_exp_dispatch,
        matrix_exp_forward_tf32=matrix_exp_forward_tf32,
        direct_skew_matrix_exp_vjp=direct_skew_matrix_exp_vjp,
        allow_multichannel_inputs=allow_multichannel_inputs,
        compile_model_body=compile_model_body,
        compile_training_loss=compile_training_loss,
        training_compile_mode=training_compile_mode,
        loss_kind=loss_kind,
    )


def make_exact_split_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Return the fused capturable optimizer used by the split runtime."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
        capturable=True,
    )


def _native_matrix_exp_blocks(model: nn.Module) -> tuple[_ExactSplitBlock, ...]:
    blocks = tuple(
        block
        for block in (
            getattr(model, "forward_block", None),
            getattr(model, "backward_block", None),
        )
        if isinstance(block, nn.Module)
    )
    if len(blocks) != 2:
        message = "EFP16 exact split training requires forward and backward PAC blocks"
        raise ValueError(message)
    for block in blocks:
        frame = getattr(block, "frame", None)
        parametrizations = getattr(frame, "parametrizations", None)
        weight = getattr(parametrizations, "weight", None)
        mapping = weight[0] if weight is not None and len(weight) == 1 else None
        if getattr(getattr(mapping, "orthogonal_map", None), "name", None) != "matrix_exp":
            message = "exact split training requires native matrix_exp parametrizations"
            raise ValueError(message)
        if not callable(getattr(block, "set_intervention_frame", None)):
            message = "PAC block does not expose a frame override"
            raise TypeError(message)
    return cast("tuple[_ExactSplitBlock, ...]", blocks)


def _evaluate_native_frame(block: _ExactSplitBlock) -> tuple[Tensor, Tensor]:
    base = block.frame.parametrizations.weight[0].base
    skew = _native_frame_skew(block)
    frame = base @ torch.matrix_exp(skew)
    return frame[..., : block.frame.parametrizations.weight.original.shape[-1]], skew


def _native_frame_skew(block: _ExactSplitBlock) -> Tensor:
    original = block.frame.parametrizations.weight.original
    lower = original.tril()
    rows, columns = lower.shape[-2:]
    if rows != columns:
        # Match torch.nn.utils.parametrizations._Orthogonal.forward: a tall
        # semi-orthogonal coordinate tensor is zero-padded to a square matrix
        # before the skew-symmetric matrix exponential is evaluated.
        lower = functional.pad(lower, (0, rows - columns))
    return lower - lower.mT


def _transfer_native_frame_gradient(
    block: _ExactSplitBlock, override: Tensor, skew: Tensor
) -> None:
    exponential_gradient = _frame_exponential_gradient(block, override)
    skew_gradient = torch.ops.aten.matrix_exp_backward(skew, exponential_gradient)
    _write_frame_coordinate_gradient(block, skew_gradient)


def _frame_exponential_gradient(block: _ExactSplitBlock, override: Tensor) -> Tensor:
    base = block.frame.parametrizations.weight[0].base
    frame_gradient = _require_gradient(override, "frame override")
    exponential_gradient = base.mT @ frame_gradient
    rows, columns = exponential_gradient.shape[-2:]
    if rows != columns:
        # The forward matrix exponential is square and only its first
        # ``columns`` columns are exposed by the semi-orthogonal map.
        exponential_gradient = functional.pad(exponential_gradient, (0, rows - columns))
    return exponential_gradient


def _write_frame_coordinate_gradient(block: _ExactSplitBlock, skew_gradient: Tensor) -> None:
    original_gradient = torch.tril(skew_gradient - skew_gradient.mT)
    parameter = block.frame.parametrizations.weight.original
    original_gradient = original_gradient[..., : parameter.shape[-1]]
    _require_gradient(parameter, "frame parameter").copy_(original_gradient)


def _compile_no_cudagraphs(
    target: nn.Module,
    *,
    mode: ExactSplitCompileMode,
) -> object:
    if mode == "default":
        return torch.compile(
            target,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
    return torch.compile(
        target,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )


def _validate_options(
    grad_clip_norm: float,
    warmup_steps: int,
    training_compile_mode: ExactSplitCompileMode,
) -> None:
    if not torch.isfinite(torch.tensor(grad_clip_norm)) or grad_clip_norm <= 0.0:
        message = "grad_clip_norm must be finite and positive"
        raise ValueError(message)
    if warmup_steps < 1:
        message = "exact split training requires at least one warmup step"
        raise ValueError(message)
    if training_compile_mode not in {"default", "max-autotune-no-cudagraphs"}:
        message = "unsupported exact-split training compile mode"
        raise ValueError(message)


def _validate_metadata_tensors(
    inputs: Tensor,
    time_delta: Tensor | None,
    observation_mask: Tensor | None,
    valid_mask: Tensor | None,
    *,
    validate_values: bool = True,
) -> None:
    for name, value in (
        ("time_delta", time_delta),
        ("observation_mask", observation_mask),
        ("valid_mask", valid_mask),
    ):
        if value is not None and value.device != inputs.device:
            message = f"exact split {name} must use the input CUDA device"
            raise ValueError(message)
    _validate_time_delta(inputs, time_delta, validate_values=validate_values)
    _validate_observation_mask(
        inputs,
        observation_mask,
        validate_values=validate_values,
    )
    _validate_event_mask(
        inputs,
        valid_mask,
        name="valid_mask",
        validate_values=validate_values,
    )


def _validate_cross_entropy_weight(
    inputs: Tensor,
    weight: Tensor | None,
    *,
    loss_kind: ExactSplitLossKind,
) -> None:
    if weight is None:
        return
    if loss_kind != "cross_entropy":
        message = "class weights are supported only for cross-entropy exact split"
        raise ValueError(message)
    if weight.device != inputs.device:
        message = "exact split class weights must use the input CUDA device"
        raise ValueError(message)
    if weight.ndim != 1 or weight.numel() < 1 or weight.dtype != inputs.dtype:
        message = "exact split class weights must be a nonempty FP32 vector"
        raise ValueError(message)
    if not torch.isfinite(weight).all() or bool((weight < 0).any()) or not bool(weight.sum() > 0):
        message = "exact split class weights must be finite, non-negative, and nonzero"
        raise ValueError(message)


def _validate_time_delta(
    inputs: Tensor,
    time_delta: Tensor | None,
    *,
    validate_values: bool,
) -> None:
    if time_delta is None:
        return
    if time_delta.shape not in (inputs.shape[:2], (*inputs.shape[:2], 1)):
        message = "time_delta must have shape [B,N] or [B,N,1]"
        raise ValueError(message)
    if validate_values and (
        not torch.isfinite(time_delta).all() or bool((time_delta < 0).any())
    ):
        message = "time_delta must contain finite non-negative values"
        raise ValueError(message)


def _validate_observation_mask(
    inputs: Tensor,
    observation_mask: Tensor | None,
    *,
    validate_values: bool,
) -> None:
    if observation_mask is None:
        return
    allowed_shapes = (
        inputs.shape[:2],
        (*inputs.shape[:2], 1),
        inputs.shape,
    )
    if observation_mask.shape not in allowed_shapes:
        message = (
            "observation_mask must have shape [B,N], [B,N,1], or "
            "[B,N,C] matching the raw input"
        )
        raise ValueError(message)
    if validate_values and (
        not torch.isfinite(observation_mask).all()
        or bool(((observation_mask < 0) | (observation_mask > 1)).any())
    ):
        message = "observation_mask must contain finite values in [0,1]"
        raise ValueError(message)


def _validate_event_mask(
    inputs: Tensor,
    mask: Tensor | None,
    *,
    name: str,
    validate_values: bool,
) -> None:
    if mask is None:
        return
    if mask.shape not in (inputs.shape[:2], (*inputs.shape[:2], 1)):
        message = f"{name} must have shape [B,N] or [B,N,1]"
        raise ValueError(message)
    if validate_values and (
        not torch.isfinite(mask).all() or bool(((mask < 0) | (mask > 1)).any())
    ):
        message = f"{name} must contain finite values in [0,1]"
        raise ValueError(message)


def _copy_optional_static_tensor(destination: Tensor | None, source: Tensor | None) -> None:
    if destination is not None and source is not None:
        destination.copy_(source, non_blocking=True)


def _snapshot_caller_state(model: nn.Module, optimizer: AdamW) -> _CallerState:
    parameters = tuple(
        _ParameterState(
            parameter=parameter,
            value=parameter.detach().clone(),
            gradient=(None if parameter.grad is None else parameter.grad.detach().clone()),
            gradient_reference=parameter.grad,
        )
        for parameter in model.parameters()
    )
    buffers = tuple(
        _BufferState(
            module=module,
            name=name,
            reference=buffer,
            value=None if buffer is None else buffer.detach().clone(),
        )
        for module in model.modules()
        for name, buffer in module._buffers.items()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    )
    return _CallerState(
        parameters=parameters,
        buffers=buffers,
        optimizer=copy.deepcopy(optimizer.state_dict()),
    )


@torch.no_grad()
def _restore_caller_state(state: _CallerState, optimizer: AdamW) -> None:
    for saved in state.parameters:
        saved.parameter.copy_(saved.value)
        if saved.gradient is None:
            saved.parameter.grad = None
        elif saved.gradient_reference is None:
            saved.parameter.grad = saved.gradient.clone()
        else:
            saved.gradient_reference.copy_(saved.gradient)
            saved.parameter.grad = saved.gradient_reference
    for saved in state.buffers:
        if saved.reference is not None and saved.value is not None:
            saved.reference.copy_(saved.value)
        setattr(saved.module, saved.name, saved.reference)
    optimizer.load_state_dict(state.optimizer)


def _snapshot_runtime_attributes(
    model: nn.Module,
    blocks: tuple[_ExactSplitBlock, ...],
) -> tuple[_AttributeState, ...]:
    targets: list[tuple[object, str]] = [
        (block, name)
        for block in blocks
        for name in (
            "recurrence_backend",
            "fused_moments_backward_training",
            "fused_recurrence_moments_backward_training",
        )
    ]
    targets.extend(
        (
            (model, "use_fused_efp16_stem_training"),
            (model, "use_fused_efp16_c2_stem_training"),
            (model, "validate_metadata"),
        )
    )
    return tuple(
        _AttributeState(
            owner=owner,
            name=name,
            present=name in owner.__dict__,
            value=owner.__dict__.get(name),
        )
        for owner, name in targets
    )


def _apply_runtime_attributes(
    model: nn.Module,
    blocks: tuple[_ExactSplitBlock, ...],
    *,
    recurrence_backend: str,
    fused_moments_backward_training: bool,
    fused_recurrence_moments_backward_training: bool,
    fused_c2_stem_training: bool,
    disable_metadata_validation: bool,
) -> None:
    for block in blocks:
        block.__dict__["recurrence_backend"] = recurrence_backend
        block.__dict__["fused_moments_backward_training"] = fused_moments_backward_training
        block.__dict__["fused_recurrence_moments_backward_training"] = (
            fused_recurrence_moments_backward_training
        )
    if hasattr(model, "use_fused_efp16_stem_training"):
        model.__dict__["use_fused_efp16_stem_training"] = True
    if hasattr(model, "use_fused_efp16_c2_stem_training"):
        model.__dict__["use_fused_efp16_c2_stem_training"] = fused_c2_stem_training
    if disable_metadata_validation and hasattr(model, "validate_metadata"):
        model.__dict__["validate_metadata"] = False


def _restore_runtime_attributes(state: tuple[_AttributeState, ...]) -> None:
    for saved in state:
        if saved.present:
            saved.owner.__dict__[saved.name] = saved.value
        else:
            saved.owner.__dict__.pop(saved.name, None)


def _clear_intervention_frames(model: nn.Module) -> None:
    seen: set[int] = set()
    for name in ("forward_block", "backward_block"):
        block = getattr(model, name, None)
        if block is None or id(block) in seen:
            continue
        seen.add(id(block))
        setter = getattr(block, "set_intervention_frame", None)
        if callable(setter):
            setter(None)


def _clear_owned_intervention_frames(
    blocks: tuple[_ExactSplitBlock, ...],
    overrides: tuple[Tensor, ...],
) -> None:
    for block, override in zip(blocks, overrides, strict=True):
        if block.intervention_frame() is override:
            block.set_intervention_frame(None)


def _finalize_runtime(
    block_references: tuple[weakref.ReferenceType[_ExactSplitBlock], ...],
    overrides: tuple[Tensor, ...],
    attribute_state: tuple[_AttributeState, ...],
) -> None:
    try:
        try:
            live_pairs = tuple(
                (block, override)
                for block_reference, override in zip(
                    block_references,
                    overrides,
                    strict=True,
                )
                if (block := block_reference()) is not None
            )
            if live_pairs:
                blocks, live_overrides = zip(*live_pairs, strict=True)
                _clear_owned_intervention_frames(tuple(blocks), tuple(live_overrides))
        finally:
            _restore_runtime_attributes(attribute_state)
    finally:
        _release_tf32_guard()


def _make_runtime_finalizer(
    runtime: EFP16ExactSplitTraining,
    blocks: tuple[_ExactSplitBlock, ...],
    overrides: tuple[Tensor, ...],
    attribute_state: tuple[_AttributeState, ...],
) -> weakref.finalize[..., EFP16ExactSplitTraining]:
    block_references = tuple(weakref.ref(block) for block in blocks)
    return weakref.finalize(
        runtime,
        _finalize_runtime,
        block_references,
        overrides,
        attribute_state,
    )


def _disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _acquire_tf32_guard() -> None:
    global _active_tf32_guards, _tf32_restore_state  # noqa: PLW0603
    if _active_tf32_guards == 0:
        _tf32_restore_state = (
            cast("bool", torch.backends.cuda.matmul.allow_tf32),
            torch.backends.cudnn.allow_tf32,
        )
    _active_tf32_guards += 1
    _disable_tf32()


def _release_tf32_guard() -> None:
    global _active_tf32_guards, _tf32_restore_state  # noqa: PLW0603
    if _active_tf32_guards == 0:
        return
    _active_tf32_guards -= 1
    if _active_tf32_guards != 0:
        return
    restore = _tf32_restore_state
    _tf32_restore_state = None
    if restore is None:
        return
    torch.backends.cuda.matmul.allow_tf32 = restore[0]
    torch.backends.cudnn.allow_tf32 = restore[1]


def _require_tf32_disabled() -> None:
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        message = "EFP16 exact FP32 split replay requires TF32 to remain disabled"
        raise RuntimeError(message)


def _require_gradient(tensor: Tensor, label: str) -> Tensor:
    gradient = tensor.grad
    if gradient is None:
        message = f"{label} gradient buffer disappeared"
        raise RuntimeError(message)
    return gradient


def _clone_optimizer_state(
    optimizer: AdamW,
    parameters: tuple[nn.Parameter, ...],
    *,
    missing_ok: bool = False,
) -> dict[nn.Parameter, dict[str, Tensor]]:
    snapshot: dict[nn.Parameter, dict[str, Tensor]] = {}
    for parameter in parameters:
        if parameter not in optimizer.state and missing_ok:
            continue
        values: dict[str, Tensor] = {}
        for key, value in optimizer.state[parameter].items():
            if not isinstance(value, Tensor):
                message = "capturable AdamW state must contain tensors only"
                raise TypeError(message)
            values[key] = value.detach().clone()
        snapshot[parameter] = values
    return snapshot


@torch.no_grad()
def _restore_state(
    parameters: tuple[nn.Parameter, ...],
    parameter_snapshot: tuple[Tensor, ...],
    optimizer: AdamW,
    optimizer_snapshot: dict[nn.Parameter, dict[str, Tensor]],
) -> None:
    for parameter, value in zip(parameters, parameter_snapshot, strict=True):
        parameter.copy_(value)
        _require_gradient(parameter, "model parameter").zero_()
        for key, saved in optimizer_snapshot[parameter].items():
            current = optimizer.state[parameter].get(key)
            if not isinstance(current, Tensor):
                message = "capturable AdamW state tensor disappeared"
                raise TypeError(message)
            current.copy_(saved)


__all__ = [
    "EFP16ExactSplitTraining",
    "ExactSplitCompileMode",
    "make_exact_split_adamw",
    "prepare_efp16_exact_split_training",
]
