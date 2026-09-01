"""Axis-structured probes for recurrent LM state tensors.

The probes deliberately preserve the two declared state axes.  They answer
whether prediction needs a full map over ``temporal x content`` coordinates or
whether one axis can first be collapsed with a shared linear functional.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

ProbeMode = Literal["temporal", "content", "full"]


class TensorAxisProbe(nn.Module):
    """Map ``[sample, temporal, content]`` state tensors to hidden targets."""

    def __init__(
        self,
        temporal_width: int,
        content_width: int,
        output_width: int,
        *,
        mode: ProbeMode,
        probe_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.temporal_width = int(temporal_width)
        self.content_width = int(content_width)
        self.output_width = int(output_width)
        self.mode = mode
        if mode == "temporal":
            self.axis_weight = nn.Parameter(torch.ones(temporal_width) / temporal_width)
            feature_width = content_width
        elif mode == "content":
            self.axis_weight = nn.Parameter(torch.ones(content_width) / content_width)
            feature_width = temporal_width
        elif mode == "full":
            self.register_parameter("axis_weight", None)
            feature_width = temporal_width * content_width
        else:
            raise ValueError(f"unknown tensor probe mode: {mode}")
        active_rank = int(probe_rank) if probe_rank is not None else feature_width
        self.projection = (
            nn.Linear(feature_width, active_rank, bias=False)
            if active_rank < feature_width
            else nn.Identity()
        )
        projected_width = (
            active_rank if isinstance(self.projection, nn.Linear) else feature_width
        )
        self.output = nn.Linear(projected_width, output_width)

    def forward(self, states: Tensor) -> Tensor:
        expected = (self.temporal_width, self.content_width)
        if states.ndim != 3 or states.shape[1:] != expected:
            message = (
                f"tensor probe expected N,{expected[0]},{expected[1]}; received "
                f"{tuple(states.shape)}"
            )
            raise ValueError(message)
        if self.mode == "temporal":
            features = torch.einsum("ntc,t->nc", states, self.axis_weight)
        elif self.mode == "content":
            features = torch.einsum("ntc,c->nt", states, self.axis_weight)
        else:
            features = states.flatten(1)
        return self.output(self.projection(features))

    @torch.no_grad()
    def weight_tensor(self) -> Tensor:
        """Return the effective map as ``[temporal, content, output]``."""

        output_weight = self.output.weight.t()
        if isinstance(self.projection, nn.Linear):
            output_weight = self.projection.weight.t() @ output_weight
        if self.mode == "temporal":
            return self.axis_weight[:, None, None] * output_weight[None, :, :]
        if self.mode == "content":
            return output_weight[:, None, :] * self.axis_weight[None, :, None]
        return output_weight.reshape(
            self.temporal_width,
            self.content_width,
            self.output_width,
        )


@torch.no_grad()
def axis_spectrum_metrics(weight: Tensor, *, axis: int) -> dict[str, float]:
    """Measure separability of one axis from the remaining weight tensor.

    The selected axis is unfolded against all other dimensions.  Stable rank
    and leading energy fractions are exact; only the reported spectrum is
    truncated when the selected axis is wider than 64.
    """

    if weight.ndim != 3 or axis not in {0, 1}:
        raise ValueError("predictive weight must be T,C,H and axis must be 0 or 1")
    matrix = weight.movedim(axis, 0).flatten(1).float()
    total_energy = matrix.square().sum().clamp_min(1.0e-30)
    axis_width = matrix.shape[0]
    if axis_width <= 64:
        singular = torch.linalg.svdvals(matrix)
    else:
        rank = min(64, matrix.shape[0], matrix.shape[1])
        _u, singular, _v = torch.svd_lowrank(matrix, q=rank, niter=4)
        singular = singular.sort(descending=True).values
    energy = singular.square()
    top_energy = energy[0].clamp_min(1.0e-30)
    metrics = {
        "axis_width": float(axis_width),
        "stable_rank": float((total_energy / top_energy).item()),
        "top1_energy": float((energy[:1].sum() / total_energy).item()),
    }
    for rank in (2, 4, 8, 16, 32, 64):
        if rank <= singular.numel():
            metrics[f"top{rank}_energy"] = float(
                (energy[:rank].sum() / total_energy).item()
            )
    metrics["measured_spectrum_energy"] = float(
        (energy.sum() / total_energy).clamp_max(1.0).item()
    )
    return metrics


@torch.no_grad()
def normalized_states(states: Tensor, epsilon: float = 1.0e-6) -> tuple[Tensor, float]:
    """Apply one scalar RMS normalization without changing tensor geometry."""

    scale = states.float().square().mean().sqrt().clamp_min(epsilon)
    return states.float() / scale, float(scale.item())


__all__ = [
    "ProbeMode",
    "TensorAxisProbe",
    "axis_spectrum_metrics",
    "normalized_states",
]
