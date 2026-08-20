"""Theory-aligned prototype scoring for A2D modal descriptors."""

from __future__ import annotations

import torch
from torch import Tensor


def prototype_logits(
    features: Tensor,
    prototypes: Tensor,
    *,
    diagonal: Tensor | None = None,
    low_rank: Tensor | None = None,
    logit_scale: Tensor | float = 1.0,
) -> Tensor:
    """Return affine nearest-prototype logits under ``diag(d) + U U^T``."""
    if features.ndim != 2 or prototypes.ndim != 2:
        message = "prototype logits require two matrices"
        raise ValueError(message)
    if features.shape[1] != prototypes.shape[1]:
        message = "feature and prototype dimensions differ"
        raise ValueError(message)
    if diagonal is None:
        diagonal = torch.ones(
            features.shape[1],
            device=features.device,
            dtype=features.dtype,
        )
    if diagonal.shape != (features.shape[1],) or bool((diagonal <= 0).any()):
        message = "prototype diagonal metric must be positive and coordinate-wise"
        raise ValueError(message)
    weighted_prototypes = prototypes * diagonal
    logits = 2.0 * features @ weighted_prototypes.transpose(0, 1) - (
        prototypes * weighted_prototypes
    ).sum(dim=1)
    if low_rank is not None:
        if low_rank.ndim != 2 or low_rank.shape[0] != features.shape[1]:
            message = "prototype low-rank factor has an incompatible shape"
            raise ValueError(message)
        projected_features = features @ low_rank
        projected_prototypes = prototypes @ low_rank
        logits = logits + (
            2.0 * projected_features @ projected_prototypes.transpose(0, 1)
            - projected_prototypes.square().sum(dim=1)
        )
    return logits * torch.as_tensor(
        logit_scale,
        device=features.device,
        dtype=features.dtype,
    )


__all__ = ["prototype_logits"]
