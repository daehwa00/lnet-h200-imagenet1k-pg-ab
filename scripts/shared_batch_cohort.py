"""Lockstep multi-model epochs driven by one ImageNet batch stream."""

from __future__ import annotations

# ruff: noqa: B010, EM101, SLF001, TRY003
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from torch.utils.data import DataLoader


@dataclass(slots=True)
class CohortMember:
    variant: str
    model: nn.Module
    runtime: nn.Module
    optimizer: torch.optim.Optimizer


def _add(current: Tensor | None, value: Tensor) -> Tensor:
    value = value.detach()
    if current is None:
        return value.clone()
    current.add_(value)
    return current


def train_epoch(
    members: list[CohortMember],
    loader: DataLoader[Any],
    *,
    device: torch.device,
    mixup_generator: np.random.Generator,
    permutation_generator: torch.Generator,
    mixup_alpha: float,
    precision: str,
    channels_last: bool,
    objective: Callable[
        [nn.Module, Tensor | tuple[Tensor, ...], Tensor, Tensor, float],
        tuple[Tensor, Tensor, dict[str, Tensor]],
    ],
    after_model_batch: Callable[
        [nn.Module, Tensor | tuple[Tensor, ...], Tensor, Tensor, float],
        None,
    ],
    after_cohort_batch: Callable[[int], None] | None = None,
) -> tuple[dict[str, dict[str, float]], int, float]:
    """Update every model from the same augmented GPU tensor for each batch."""
    if not members:
        raise ValueError("shared-batch cohort cannot be empty")
    for member in members:
        member.model.train()
        member.runtime.train()

    loss_sums: dict[str, Tensor | None] = dict.fromkeys(
        (member.variant for member in members),
        None,
    )
    correct_sums: dict[str, Tensor | None] = dict.fromkeys(
        (member.variant for member in members),
        None,
    )
    diagnostics: dict[str, dict[str, Tensor | None]] = {member.variant: {} for member in members}
    sample_count = 0
    host_input_wait_seconds = 0.0
    batches = iter(harness._device_batches(loader, device, channels_last=channels_last))
    for batch_index in range(len(loader)):
        waiting_started = perf_counter()
        inputs, targets = next(batches)
        host_input_wait_seconds += perf_counter() - waiting_started
        permutation = torch.randperm(
            targets.numel(),
            device=device,
            generator=permutation_generator,
        )
        permuted_targets = targets[permutation]
        mixing = float(mixup_generator.beta(mixup_alpha, mixup_alpha))
        mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
        for member in members:
            member.optimizer.zero_grad(set_to_none=True)
            harness._begin_cudagraph_step(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bfloat16",
            ):
                output = member.runtime(mixed_inputs)
                logits, loss, active_diagnostics = objective(
                    member.model,
                    output,
                    targets,
                    permuted_targets,
                    mixing,
                )
            loss.backward()
            after_model_batch(
                member.model,
                output,
                targets,
                permuted_targets,
                mixing,
            )
            torch.nn.utils.clip_grad_norm_(member.model.parameters(), 1.0)
            member.optimizer.step()
            loss_sums[member.variant] = _add(
                loss_sums[member.variant],
                loss.detach() * targets.numel(),
            )
            correct_sums[member.variant] = _add(
                correct_sums[member.variant],
                logits.detach().argmax(dim=-1).eq(targets).sum(),
            )
            active = diagnostics[member.variant]
            for name, value in active_diagnostics.items():
                active[name] = _add(active.get(name), value.detach() * targets.numel())
            del active_diagnostics, logits, loss, output
        sample_count += targets.numel()
        if after_cohort_batch is not None:
            after_cohort_batch(batch_index + 1)
        del mixed_inputs, permutation, permuted_targets

    metrics: dict[str, dict[str, float]] = {}
    for member in members:
        loss_sum = loss_sums[member.variant]
        correct_sum = correct_sums[member.variant]
        if loss_sum is None or correct_sum is None or sample_count < 1:
            raise RuntimeError("shared-batch cohort produced no metrics")
        member_metrics = {
            "loss": float(loss_sum.double()) / sample_count,
            "mixed_accuracy": int(correct_sum) / sample_count,
        }
        latest_diagnostics = {
            name: float(value.double()) / sample_count
            for name, value in diagnostics[member.variant].items()
            if value is not None
        }
        setattr(member.model, "_latest_training_diagnostics", latest_diagnostics)
        metrics[member.variant] = member_metrics
    return metrics, len(loader), host_input_wait_seconds


def evaluate(
    members: list[CohortMember],
    loader: DataLoader[Any],
    *,
    device: torch.device,
    precision: str,
    channels_last: bool,
    finalize: Callable[[nn.Module, list[list[Tensor]], Tensor, torch.device], dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Evaluate all models from one validation loader traversal."""
    outputs: dict[str, list[list[Tensor]]] = {
        member.variant: [[], [], [], [], []] for member in members
    }
    labels: list[Tensor] = []
    for member in members:
        member.model.eval()
        member.runtime.eval()
    with torch.inference_mode():
        for inputs, targets in harness._device_batches(
            loader,
            device,
            channels_last=channels_last,
        ):
            labels.append(targets.cpu())
            for member in members:
                harness._begin_cudagraph_step(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bfloat16",
                ):
                    output = member.runtime(inputs)
                if not isinstance(output, tuple) or len(output) != 5:
                    raise RuntimeError("shared-batch evaluation requires five branch outputs")
                for index, value in enumerate(output):
                    outputs[member.variant][index].append(value.detach().float().cpu())
                del output
    target = torch.cat(labels)
    return {
        member.variant: finalize(member.model, outputs[member.variant], target, device)
        for member in members
    }


__all__ = ["CohortMember", "evaluate", "train_epoch"]
