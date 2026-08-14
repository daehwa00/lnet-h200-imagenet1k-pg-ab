from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .pac_head_factorial_features import block_context
from .pac_head_factorial_model import PACHeadFactorialClassifier
from .pac_model import PACHybridPRLBlock

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .pac_head_factorial_features import BlockContext
    from .tapped_prl_followup_schema import JsonRow


def real_modal_rows(
    model: nn.Module, inputs: Tensor, labels: Tensor, dataset: str, seed: int, device: str
) -> tuple[tuple[JsonRow, ...], tuple[JsonRow, ...]]:
    if not isinstance(model, PACHeadFactorialClassifier):
        return (), ()
    contexts = _classifier_contexts(model, inputs.to(device=device))
    if not contexts:
        return (), ()
    context = contexts[-1]
    energy = (
        (context.states_real.square() + context.states_imag.square()).mean(dim=1).detach().cpu()
    )
    damping = context.damping.mean(dim=1).detach().cpu()
    labels_cpu = labels.detach().cpu()
    stats = tuple(_class_mode_rows(dataset, seed, labels_cpu, energy, damping))
    attribution = _attribution_rows(model, dataset, seed, context.states_real.shape[-1])
    return stats, attribution


def _classifier_contexts(
    model: PACHeadFactorialClassifier, inputs: Tensor
) -> tuple[BlockContext, ...]:
    features = inputs
    contexts = []
    for module in model.forward_stack.blocks:
        match module:
            case PACHybridPRLBlock() as block:
                context = block_context(block, features)
                contexts.append(context)
                features = context.output
            case _:
                continue
    return tuple(contexts)


def _class_mode_rows(
    dataset: str, seed: int, labels: Tensor, energy: Tensor, damping: Tensor
) -> Iterable[JsonRow]:
    for class_index in torch.unique(labels, sorted=True).tolist():
        mask = labels == int(class_index)
        for mode in range(energy.shape[-1]):
            yield {
                "dataset_or_task": dataset,
                "seed": seed,
                "class_index": int(class_index),
                "mode": mode,
                "mean_modal_energy": float(energy[mask, mode].mean().item()),
                "mean_effective_damping": float(damping[mask, mode].mean().item()),
            }


def _attribution_rows(
    model: PACHeadFactorialClassifier, dataset: str, seed: int, modes: int
) -> tuple[JsonRow, ...]:
    first = model.classifier[1]
    if not isinstance(first, nn.Linear):
        return ()
    weights = first.weight.detach().cpu()[:, : modes * modes]
    scores = torch.linalg.vector_norm(weights, dim=0)
    return tuple(
        {
            "dataset_or_task": dataset,
            "seed": seed,
            "feature_index": index,
            "hermitian_weight_norm": float(scores[index].item()),
        }
        for index in torch.argsort(scores, descending=True)[: min(10, scores.numel())].tolist()
    )
