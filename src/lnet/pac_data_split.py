from __future__ import annotations

import torch
from torch import Tensor


def stratified_partition_indices(
    labels: Tensor,
    validation_ratio: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    if not 0.0 < validation_ratio < 1.0:
        message = "validation_ratio must be strictly between zero and one"
        raise ValueError(message)
    labels_cpu = labels.detach().cpu()
    classes = torch.unique(labels_cpu, sorted=True)
    generator = torch.Generator().manual_seed(seed)
    validation: list[int] = []
    remaining: list[int] = []
    for class_value in classes.tolist():
        indices = torch.nonzero(labels_cpu == int(class_value), as_tuple=False).flatten()
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        # A singleton cannot appear in both folds. Keep it in optimization so
        # the model retains the complete official-TRAIN label space, and stratify
        # only classes for which a leakage-free validation example exists.
        if shuffled.numel() < 2:
            continue
        validation.append(int(shuffled[0]))
        remaining.extend(int(index) for index in shuffled[2:].tolist())
    maximum_validation = labels_cpu.numel() - classes.numel()
    if maximum_validation < 1 or not validation:
        message = "official TRAIN has no class that can support a stratified validation fold"
        raise ValueError(message)
    target = min(
        max(round(labels_cpu.numel() * validation_ratio), len(validation)),
        maximum_validation,
    )
    if len(validation) < target:
        order = torch.randperm(len(remaining), generator=generator).tolist()
        selected_positions = {int(index) for index in order[: target - len(validation)]}
        validation.extend(
            index for position, index in enumerate(remaining) if position in selected_positions
        )
    validation_indices = torch.tensor(sorted(validation), dtype=torch.long)
    mask = torch.ones(labels_cpu.numel(), dtype=torch.bool)
    mask[validation_indices] = False
    train_indices = torch.nonzero(mask, as_tuple=False).flatten()
    return train_indices, validation_indices
