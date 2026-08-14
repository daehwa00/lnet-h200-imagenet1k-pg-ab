from __future__ import annotations

from contextlib import contextmanager
from itertools import islice
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_efp16_exact_split_training import prepare_efp16_exact_split_training
from .pac_native_matrix_exp_vjp import cuda_switch_matrix_exp_capability
from .pac_types import (
    PACClassificationMetrics,
    PACClassificationTask,
    PACExperimentConfig,
    PACRegressionTask,
    PACTrainOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@runtime_checkable
class _PostOptimizerStep(Protocol):
    def post_optimizer_step(self) -> None: ...


@runtime_checkable
class _FinalizeConstraints(Protocol):
    def finalize_constraints(self) -> None: ...


class _ExactSplitRuntime(Protocol):
    """Lifecycle shared by the generic and model-specialized CUDA runtimes."""

    def step(self, inputs: Tensor, labels: Tensor) -> Tensor: ...

    def close(self) -> None: ...

    def activate(self) -> None: ...


@runtime_checkable
class _ClassifierExactSplitProvider(Protocol):
    """Optional model-owned factory for a specialized fixed-shape runtime."""

    def prepare_classifier_exact_split_runtime(
        self,
        optimizer: torch.optim.AdamW,
        inputs: Tensor,
        labels: Tensor,
        *,
        grad_clip_norm: float,
    ) -> _ExactSplitRuntime: ...


_LAZY_COMPILE_LOCK = Lock()


def train_regression_model(
    model: nn.Module,
    task: PACRegressionTask,
    config: PACExperimentConfig,
    device: str,
    seed: int,
) -> PACTrainOutcome:
    model.to(device=device)
    runtime_model = _training_runtime_model(model, config)
    compiled_signatures: set[tuple[tuple[int, ...], torch.dtype, torch.device]] = set()
    optimizer = _optimizer(model, config, device)
    train_inputs = task.train_inputs.to(device=device)
    train_targets = task.train_targets.to(device=device)
    batch_generator = _batch_generator(train_inputs, seed)
    started_at = perf_counter()
    last_grad_norm = torch.zeros((), device=device)
    for _ in range(config.epochs):
        for batch_inputs, batch_targets in _regression_batches(
            train_inputs,
            train_targets,
            config.batch_size,
            generator=batch_generator,
        ):
            with _lazy_compile_guard(
                runtime_model,
                model,
                batch_inputs,
                compiled_signatures,
                dynamic=config.compile_mode == "dynamic-no-cudagraph",
            ):
                optimizer.zero_grad(set_to_none=True)
                with _autocast(config, batch_inputs):
                    loss = functional.mse_loss(runtime_model(batch_inputs), batch_targets)
                loss.backward()
                last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip_norm
                )
                optimizer.step()
                _run_post_optimizer_step(model)
    _finalize_constraints(model)
    return PACTrainOutcome(
        train_loss=evaluate_regression_loss(model, train_inputs, train_targets),
        validation_loss=evaluate_regression_loss(
            model,
            task.validation_inputs.to(device=device),
            task.validation_targets.to(device=device),
        ),
        test_loss=evaluate_regression_loss(
            model, task.test_inputs.to(device=device), task.test_targets.to(device=device)
        ),
        grad_norm=float(last_grad_norm.item()),
        elapsed_time=perf_counter() - started_at,
    )


def train_classifier(  # noqa: C901, PLR0912, PLR0915
    model: nn.Module,
    task: PACClassificationTask,
    config: PACExperimentConfig,
    device: str,
    seed: int,
    *,
    # Test metrics are opt-in so validation/selection callers cannot touch the
    # official TEST split merely by omitting a keyword.
    evaluate_test: bool = False,
    restore_best_validation: bool = False,
) -> PACTrainOutcome:
    model.to(device=device)
    runtime_model = _training_runtime_model(model, config)
    compiled_signatures: set[tuple[tuple[int, ...], torch.dtype, torch.device]] = set()
    optimizer = _optimizer(model, config, device)
    train_inputs = task.train_inputs.to(device=device)
    train_labels = task.train_labels.to(device=device)
    batch_generator = _batch_generator(train_inputs, seed)
    started_at = perf_counter()
    last_grad_norm = torch.zeros((), device=device)
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, Tensor] | None = None
    validation_inputs = task.validation_inputs.to(device=device)
    validation_labels = task.validation_labels.to(device=device)
    if restore_best_validation and validation_inputs.shape[0] == 0:
        message = "validation checkpoint restoration requires a non-empty validation fold"
        raise ValueError(message)
    exact_split_runtime: _ExactSplitRuntime | None = None
    if config.gradient_accumulation_steps < 1:
        message = "gradient_accumulation_steps must be positive"
        raise ValueError(message)
    exact_split_unavailable = config.gradient_accumulation_steps != 1
    exact_split_full_steps = 0
    exact_split_fallback_steps = 0
    for epoch in range(config.epochs):
        batches = iter(
            _classification_batches(
                train_inputs,
                train_labels,
                config.batch_size,
                generator=batch_generator,
            )
        )
        while batch_group := tuple(
            islice(batches, config.gradient_accumulation_steps)
        ):
            batch_inputs, batch_labels = batch_group[0]
            use_exact_split = (
                len(batch_group) == 1
                and
                not exact_split_unavailable
                and getattr(model, "use_efp16_exact_split_training", False)
                and batch_inputs.shape[0] == config.batch_size
                and batch_inputs.is_cuda
                and config.precision == "fp32"
            )
            if use_exact_split and exact_split_runtime is None:
                exact_split_runtime = _prepare_classifier_exact_split(
                    model, optimizer, batch_inputs, batch_labels, config
                )
                exact_split_unavailable = exact_split_runtime is None
                use_exact_split = not exact_split_unavailable
            if use_exact_split and exact_split_runtime is not None:
                loss = exact_split_runtime.step(batch_inputs, batch_labels)
                exact_split_full_steps += 1
                continue
            if exact_split_runtime is not None:
                exact_split_runtime.close()
            optimizer.zero_grad(set_to_none=True)
            group_size = sum(int(labels.shape[0]) for _, labels in batch_group)
            for batch_inputs, batch_labels in batch_group:
                if getattr(model, "use_efp16_exact_split_training", False):
                    exact_split_fallback_steps += 1
                with _lazy_compile_guard(
                    runtime_model,
                    model,
                    batch_inputs,
                    compiled_signatures,
                    dynamic=config.compile_mode == "dynamic-no-cudagraph",
                ):
                    with _autocast(config, batch_inputs):
                        loss = functional.cross_entropy(
                            runtime_model(batch_inputs),
                            batch_labels,
                            reduction="sum",
                        ) / max(group_size, 1)
                    loss.backward()
            last_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm
            )
            optimizer.step()
            _run_post_optimizer_step(model)
        if restore_best_validation:
            if exact_split_runtime is not None:
                exact_split_runtime.close()
            current_validation_loss = evaluate_classification_loss(
                model,
                validation_inputs,
                validation_labels,
                batch_size=config.batch_size,
            )
            if current_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
            if exact_split_runtime is not None:
                exact_split_runtime.activate()
    if exact_split_runtime is not None:
        exact_split_runtime.close()
    model.__dict__["efp16_exact_split_full_steps"] = exact_split_full_steps
    model.__dict__["efp16_exact_split_fallback_steps"] = exact_split_fallback_steps
    model.__dict__["efp16_exact_split_capture_succeeded"] = exact_split_runtime is not None
    if restore_best_validation:
        if best_state is None:
            message = "validation checkpoint selection did not produce a checkpoint"
            raise RuntimeError(message)
        model.load_state_dict(best_state)
    _finalize_constraints(model)
    return PACTrainOutcome(
        train_loss=evaluate_classification_loss(
            model, train_inputs, train_labels, batch_size=config.batch_size
        ),
        validation_loss=(
            evaluate_classification_loss(
                model,
                validation_inputs,
                validation_labels,
                batch_size=config.batch_size,
            )
            if validation_inputs.shape[0]
            else float("nan")
        ),
        test_loss=(
            evaluate_classification_loss(
                model,
                task.test_inputs.to(device=device),
                task.test_labels.to(device=device),
                batch_size=config.batch_size,
            )
            if evaluate_test
            else float("nan")
        ),
        grad_norm=float(last_grad_norm.item()),
        elapsed_time=perf_counter() - started_at,
        best_epoch=best_epoch,
    )


def evaluate_regression_loss(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = float(functional.mse_loss(model(inputs), targets).item())
    model.train(was_training)
    return loss


def evaluate_classification_loss(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    *,
    batch_size: int | None = None,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        chunk_size = inputs.shape[0] if batch_size is None else batch_size
        total_loss = torch.zeros((), device=inputs.device, dtype=torch.float64)
        for batch_inputs, batch_labels in zip(
            inputs.split(chunk_size), labels.split(chunk_size), strict=True
        ):
            total_loss += functional.cross_entropy(
                model(batch_inputs), batch_labels, reduction="sum"
            ).to(torch.float64)
        loss = float((total_loss / max(labels.numel(), 1)).item())
    model.train(was_training)
    return loss


def classification_metrics(model: nn.Module, inputs: Tensor, labels: Tensor) -> tuple[float, float]:
    metrics = classification_metric_bundle(model, inputs, labels)
    return metrics.accuracy, metrics.macro_f1


def classification_metric_bundle(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    *,
    batch_size: int | None = None,
) -> PACClassificationMetrics:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        chunk_size = inputs.shape[0] if batch_size is None else batch_size
        predictions = torch.cat(
            [
                torch.argmax(model(batch_inputs), dim=-1).detach().cpu()
                for batch_inputs in inputs.split(chunk_size)
            ]
        )
    model.train(was_training)
    labels_cpu = labels.detach().cpu()
    accuracy = float((predictions == labels_cpu).to(torch.float32).mean().item())
    class_scores: list[float] = []
    recalls: list[float] = []
    supports: list[int] = []
    for class_index in torch.unique(labels_cpu).tolist():
        predicted = predictions == int(class_index)
        actual = labels_cpu == int(class_index)
        tp = (predicted & actual).sum().item()
        precision = tp / max(predicted.sum().item(), 1)
        support = int(actual.sum().item())
        recall = tp / max(support, 1)
        class_scores.append(
            0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        )
        recalls.append(recall)
        supports.append(support)
    total = max(sum(supports), 1)
    return PACClassificationMetrics(
        accuracy=accuracy,
        macro_f1=float(sum(class_scores) / max(len(class_scores), 1)),
        weighted_f1=float(
            sum(score * support for score, support in zip(class_scores, supports, strict=True))
            / total
        ),
        balanced_accuracy=float(sum(recalls) / max(len(recalls), 1)),
    )


def _regression_batches(
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> Generator[tuple[Tensor, Tensor]]:
    # Preserve the device RNG/permutation contract, then move the permutation
    # out of the manual CUDA graph allocator's address space.  A replay may
    # overwrite live eager allocations, so keeping future indices on CUDA is
    # unsafe even when the data batches themselves are streamed lazily.
    order = torch.randperm(inputs.shape[0], device=inputs.device, generator=generator).cpu()
    for index in order.split(batch_size):
        # Materialize one indexed batch at a time.  Keeping the entire epoch's
        # advanced-indexing outputs alive can let a manually captured CUDA
        # graph reuse their allocator addresses and corrupt later batches.
        active_index = index.to(device=inputs.device)
        yield inputs[active_index], targets[active_index]


def _classification_batches(
    inputs: Tensor,
    labels: Tensor,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> Generator[tuple[Tensor, Tensor]]:
    order = torch.randperm(inputs.shape[0], device=inputs.device, generator=generator)
    for index in order.split(batch_size):
        yield inputs[index], labels[index]


def _batch_generator(inputs: Tensor, seed: int) -> torch.Generator:
    return torch.Generator(device=inputs.device).manual_seed(seed)


def _run_post_optimizer_step(model: nn.Module) -> None:
    if isinstance(model, _PostOptimizerStep):
        model.post_optimizer_step()


def _finalize_constraints(model: nn.Module) -> None:
    if isinstance(model, _FinalizeConstraints):
        model.finalize_constraints()


def _training_runtime_model(model: nn.Module, config: PACExperimentConfig) -> nn.Module:
    if config.compile_mode == "dynamic-no-cudagraph":
        return cast(
            "nn.Module",
            torch.compile(
                model,
                fullgraph=True,
                dynamic=True,
                options={"triton.cudagraphs": False},
            ),
        )
    match config.compile_mode:
        case "none":
            return model
        case "default":
            mode = None
        case "reduce-overhead" | "max-autotune":
            mode = config.compile_mode
        case "max-autotune-no-cudagraphs":
            mode = config.compile_mode
    return cast(
        "nn.Module",
        torch.compile(
            model,
            fullgraph=True,
            dynamic=False,
            mode=mode,
        ),
    )


def _optimizer(model: nn.Module, config: PACExperimentConfig, device: str) -> torch.optim.AdamW:
    if getattr(model, "use_efp16_exact_split_training", False) and device.startswith("cuda"):
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
            capturable=True,
        )
    match config.optimizer_mode:
        case "default":
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        case "foreach":
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                foreach=True,
            )
        case "fused":
            if not device.startswith("cuda"):
                message = "fused AdamW requires a CUDA device"
                raise ValueError(message)
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                fused=True,
            )


def _prepare_classifier_exact_split(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    inputs: Tensor,
    labels: Tensor,
    config: PACExperimentConfig,
) -> _ExactSplitRuntime | None:
    """Capture before updates, returning ``None`` with the eager model restored."""
    try:
        if isinstance(model, _ClassifierExactSplitProvider):
            runtime = model.prepare_classifier_exact_split_runtime(
                optimizer,
                inputs,
                labels,
                grad_clip_norm=config.grad_clip_norm,
            )
            model.__dict__["efp16_exact_split_runtime_kind"] = getattr(
                runtime,
                "training_backend",
                "model_specialized_exact_split",
            )
        else:
            switch_available = cuda_switch_matrix_exp_capability()[0]
            runtime = prepare_efp16_exact_split_training(
                model,
                optimizer,
                inputs,
                labels,
                grad_clip_norm=config.grad_clip_norm,
                warmup_steps=1,
                recurrence_backend="auto",
                fused_recurrence_moments_backward_training=True,
                capture_post_optimizer_step=bool(
                    getattr(model, "efp16_exact_split_capture_post_optimizer_step", True)
                ),
                specialized_matrix_exp_vjp=True,
                matrix_exp_dispatch="cuda_switch" if switch_available else "host",
                allow_multichannel_inputs=bool(
                    getattr(model, "efp16_exact_split_allow_multichannel_inputs", False)
                ),
                compile_model_body=bool(
                    getattr(model, "efp16_exact_split_compile_model_body", False)
                ),
            )
            model.__dict__["efp16_exact_split_runtime_kind"] = "generic_exact_split"
    except Exception as error:  # noqa: BLE001 -- graph/driver failures have no stable exception family
        # The factory removes any installed frame interventions. Disable the
        # opt-in as well so later batches cannot retry after updates begin.
        model.__dict__["use_efp16_exact_split_training"] = False
        model.__dict__["efp16_exact_split_capture_error"] = (
            f"{type(error).__name__}: {error}"
        )
        return None
    return runtime


def _autocast(config: PACExperimentConfig, inputs: Tensor) -> torch.autocast:
    return torch.autocast(
        device_type=inputs.device.type,
        dtype=torch.bfloat16,
        enabled=config.precision == "bf16",
    )


@contextmanager
def _lazy_compile_guard(
    runtime_model: nn.Module,
    eager_model: nn.Module,
    inputs: Tensor,
    compiled_signatures: set[tuple[tuple[int, ...], torch.dtype, torch.device]],
    *,
    dynamic: bool,
) -> Generator[None]:
    if runtime_model is eager_model:
        yield
        return
    shape = (-1,) if dynamic else tuple(inputs.shape)
    signature = (shape, inputs.dtype, inputs.device)
    if signature in compiled_signatures:
        yield
        return
    # Inductor compilation is process-global. CUDA Graph modes are restricted to a
    # single queue worker; graph-free modes can execute concurrently after compilation.
    with _LAZY_COMPILE_LOCK:
        yield
    compiled_signatures.add(signature)
