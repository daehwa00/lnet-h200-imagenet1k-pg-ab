"""Prototype-aware classifiers for frozen and end-to-end A2D Q descriptors.

The descriptor is assumed to contain radial-log modal energies.  Physical
prototypes are therefore estimated in energy space and mapped back with
``log1p``.  Every prototype score remains affine in Q for fixed prototypes
and a fixed positive-semidefinite metric.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .a2d_spectral_prototype import prototype_logits
from .complex_scan import ModalFusionHead
from .image_layers import LowRankQuadraticModalHead

if TYPE_CHECKING:
    from collections.abc import Sequence


def expected_calibration_error(
    logits: Tensor,
    labels: Tensor,
    *,
    bins: int = 15,
) -> float:
    """Return equal-width expected calibration error."""
    if bins <= 0:
        raise ValueError("ECE requires at least one bin")
    probabilities = logits.float().softmax(dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(labels.to(prediction.device)).float()
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    error = torch.zeros((), device=logits.device)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        active = (confidence > lower) & (confidence <= upper)
        if bool(active.any()):
            error = error + active.float().mean() * (
                correct[active].mean() - confidence[active].mean()
            ).abs()
    return float(error)


def energy_kmeans_prototypes(
    standardized_features: Tensor,
    raw_log_features: Tensor,
    labels: Tensor,
    *,
    components: int,
    iterations: int = 12,
) -> tuple[Tensor, Tensor]:
    """Fit class-wise K-means, then estimate centroids in energy space."""
    if components <= 0 or iterations <= 0:
        raise ValueError("prototype components and iterations must be positive")
    if standardized_features.shape != raw_log_features.shape:
        raise ValueError("standardized and raw prototype features must align")
    classes = torch.unique(labels.to(torch.long), sorted=True)
    all_prototypes: list[Tensor] = []
    all_classes: list[Tensor] = []
    for class_value in classes:
        active_mask = labels == class_value
        active = standardized_features[active_mask]
        active_raw = raw_log_features[active_mask]
        if active.shape[0] < components:
            raise ValueError("every class needs at least K examples")
        # Deterministic farthest-point initialization avoids seed-dependent
        # empty clusters while preserving the requested split-seed protocol.
        first = (active - active.mean(dim=0, keepdim=True)).square().sum(dim=1).argmax()
        center_indices = [int(first)]
        minimum_distance = (active - active[first]).square().sum(dim=1)
        for _ in range(1, components):
            next_index = int(minimum_distance.argmax())
            center_indices.append(next_index)
            next_distance = (active - active[next_index]).square().sum(dim=1)
            minimum_distance = torch.minimum(minimum_distance, next_distance)
        centers = active[torch.tensor(center_indices, device=active.device)]
        assignment = torch.zeros(active.shape[0], device=active.device, dtype=torch.long)
        for _ in range(iterations):
            assignment = torch.cdist(active, centers).argmin(dim=1)
            counts = torch.bincount(assignment, minlength=components)
            if bool((counts == 0).any()):
                # Re-seed empty components with points farthest from their
                # currently assigned center.
                assigned_distance = (active - centers[assignment]).square().sum(dim=1)
                for empty in torch.nonzero(counts == 0, as_tuple=False).flatten():
                    replacement = assigned_distance.argmax()
                    assignment[replacement] = empty
                    assigned_distance[replacement] = -1.0
            centers = torch.stack(
                [active[assignment == index].mean(dim=0) for index in range(components)]
            )
        raw_energy = torch.expm1(active_raw.float()).clamp_min_(0.0)
        for component in range(components):
            all_prototypes.append(
                torch.log1p(raw_energy[assignment == component].mean(dim=0))
            )
            all_classes.append(class_value)
    return torch.stack(all_prototypes), torch.stack(all_classes)


def grouped_logsumexp_logits(
    component_logits: Tensor,
    *,
    classes: int,
    components: int,
    temperature: Tensor | float = 1.0,
) -> Tensor:
    """Reduce ``[B, classes * components]`` mixture scores by log-mean-exp."""
    shaped = component_logits.reshape(component_logits.shape[0], classes, components)
    active_temperature = torch.as_tensor(
        temperature,
        device=component_logits.device,
        dtype=component_logits.dtype,
    ).clamp_min(1.0e-3)
    return active_temperature * (
        torch.logsumexp(shaped / active_temperature, dim=-1) - math.log(components)
    )


class PrototypeMetricHead(nn.Module):
    """Fixed prototypes with a learned shared ``D + U U^T`` metric."""

    def __init__(
        self,
        prototypes: Tensor,
        *,
        classes: int,
        components: int = 1,
        initial_diagonal: Tensor | None = None,
        rank: int = 0,
        learn_temperature: bool = False,
    ) -> None:
        super().__init__()
        if prototypes.ndim != 2 or prototypes.shape[0] != classes * components:
            raise ValueError("prototype tensor does not match C x K")
        self.classes = classes
        self.components = components
        self.input_dim = prototypes.shape[1]
        self.register_buffer("prototypes", prototypes.detach().clone())
        if initial_diagonal is None:
            initial_diagonal = torch.ones(self.input_dim, device=prototypes.device)
        self.log_diagonal = nn.Parameter(initial_diagonal.float().log())
        if rank > 0:
            self.low_rank = nn.Parameter(torch.empty(self.input_dim, rank))
            nn.init.normal_(self.low_rank, std=1.0e-3)
        else:
            self.register_parameter("low_rank", None)
        self.logit_scale = nn.Parameter(torch.zeros(()))
        if learn_temperature and components > 1:
            self.log_temperature = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_temperature", None)

    def diagonal(self) -> Tensor:
        centered = self.log_diagonal - torch.logsumexp(self.log_diagonal, dim=0)
        return centered.exp() * self.input_dim

    def component_logits(self, features: Tensor) -> Tensor:
        return prototype_logits(
            features,
            self.prototypes,
            diagonal=self.diagonal(),
            low_rank=self.low_rank,
            logit_scale=self.logit_scale.clamp(-4.0, 4.0).exp(),
        )

    def forward(self, features: Tensor) -> Tensor:
        logits = self.component_logits(features)
        if self.components == 1:
            return logits
        temperature: Tensor | float = 1.0
        if self.log_temperature is not None:
            temperature = self.log_temperature.clamp(-2.0, 2.0).exp()
        return grouped_logsumexp_logits(
            logits,
            classes=self.classes,
            components=self.components,
            temperature=temperature,
        )


class StagewisePrototypeMetricHead(nn.Module):
    """Sum independently metricized prototype logits for each A2D stage."""

    def __init__(
        self,
        prototypes: Tensor,
        *,
        classes: int,
        components: int,
        stage_dims: Sequence[int],
        rank: int,
    ) -> None:
        super().__init__()
        if sum(stage_dims) != prototypes.shape[1]:
            raise ValueError("stage dimensions do not span the descriptor")
        self.classes = classes
        self.components = components
        self.stage_dims = tuple(int(dimension) for dimension in stage_dims)
        self.stage_heads = nn.ModuleList()
        offset = 0
        for dimension in self.stage_dims:
            self.stage_heads.append(
                PrototypeMetricHead(
                    prototypes[:, offset : offset + dimension],
                    classes=classes,
                    components=components,
                    rank=rank,
                    learn_temperature=False,
                )
            )
            offset += dimension
        self.stage_log_scales = nn.Parameter(torch.zeros(len(self.stage_dims)))
        self.log_temperature = nn.Parameter(torch.zeros(())) if components > 1 else None

    def stage_component_logits(self, features: Tensor) -> tuple[Tensor, ...]:
        pieces = features.split(self.stage_dims, dim=-1)
        scales = self.stage_log_scales.clamp(-3.0, 3.0).exp()
        return tuple(
            scales[index] * head.component_logits(piece)
            for index, (head, piece) in enumerate(zip(self.stage_heads, pieces, strict=True))
        )

    def forward(self, features: Tensor) -> Tensor:
        component_logits = torch.stack(self.stage_component_logits(features), dim=0).sum(dim=0)
        if self.components == 1:
            return component_logits
        temperature = self.log_temperature.clamp(-2.0, 2.0).exp()
        return grouped_logsumexp_logits(
            component_logits,
            classes=self.classes,
            components=self.components,
            temperature=temperature,
        )


class PrototypeResidualHead(nn.Module):
    """Keep prototype logits primary and add one zero-safe residual branch."""

    def __init__(
        self,
        prototype: nn.Module,
        residual: nn.Module,
        *,
        beta_initial: float,
    ) -> None:
        super().__init__()
        self.prototype = prototype
        self.residual = residual
        self.beta = nn.Parameter(torch.tensor(float(beta_initial)))

    def forward(self, features: Tensor) -> Tensor:
        return self.prototype(features) + self.beta * self.residual(features)


class EMAPrototypeHead(nn.Module):
    """End-to-end physical prototypes plus a bounded learnable correction."""

    def __init__(
        self,
        input_dim: int,
        classes: int,
        *,
        components: int,
        rank: int,
        stage_dims: Sequence[int] | None = None,
        ema_momentum: float = 0.05,
        delta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 < ema_momentum <= 1.0:
            raise ValueError("EMA momentum must lie in (0, 1]")
        self.input_dim = input_dim
        self.classes = classes
        self.components = components
        self.rank = rank
        self.stage_dims = tuple(stage_dims or ())
        if self.stage_dims and sum(self.stage_dims) != input_dim:
            raise ValueError("stage dimensions do not span the descriptor")
        self.ema_momentum = float(ema_momentum)
        self.delta_scale = float(delta_scale)
        self.register_buffer("ema_energy", torch.zeros(classes, components, input_dim))
        self.register_buffer("ema_initialized", torch.zeros(classes, components, dtype=torch.bool))
        self.register_buffer("update_count", torch.zeros(classes, components))
        self.delta = nn.Parameter(torch.empty(classes, components, input_dim))
        nn.init.normal_(self.delta, std=1.0e-3)
        metric_count = len(self.stage_dims) if self.stage_dims else 1
        metric_dims = self.stage_dims if self.stage_dims else (input_dim,)
        self.log_diagonals = nn.ParameterList(
            [nn.Parameter(torch.zeros(dimension)) for dimension in metric_dims]
        )
        self.low_ranks = nn.ParameterList()
        for dimension in metric_dims:
            factor = nn.Parameter(torch.empty(dimension, rank))
            nn.init.normal_(factor, std=1.0e-3)
            self.low_ranks.append(factor)
        self.stage_log_scales = nn.Parameter(torch.zeros(metric_count))
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.log_temperature = nn.Parameter(torch.zeros(())) if components > 1 else None

    def physical_prototypes(self) -> Tensor:
        physical = torch.log1p(self.ema_energy)
        return physical + self.delta_scale * self.delta

    @staticmethod
    def _diagonal(log_diagonal: Tensor) -> Tensor:
        centered = log_diagonal - torch.logsumexp(log_diagonal, dim=0)
        return centered.exp() * log_diagonal.numel()

    def _component_logits(self, features: Tensor, prototypes: Tensor) -> Tensor:
        if not self.stage_dims:
            logits = prototype_logits(
                features,
                prototypes.reshape(self.classes * self.components, self.input_dim),
                diagonal=self._diagonal(self.log_diagonals[0]),
                low_rank=self.low_ranks[0],
            )
            return self.logit_scale.clamp(-4.0, 4.0).exp() * logits
        feature_stages = features.split(self.stage_dims, dim=-1)
        prototype_stages = prototypes.split(self.stage_dims, dim=-1)
        scales = self.stage_log_scales.clamp(-3.0, 3.0).exp()
        terms = []
        for index, (feature_stage, prototype_stage) in enumerate(
            zip(feature_stages, prototype_stages, strict=True)
        ):
            terms.append(
                scales[index]
                * prototype_logits(
                    feature_stage,
                    prototype_stage.reshape(
                        self.classes * self.components,
                        self.stage_dims[index],
                    ),
                    diagonal=self._diagonal(self.log_diagonals[index]),
                    low_rank=self.low_ranks[index],
                )
            )
        return self.logit_scale.clamp(-4.0, 4.0).exp() * torch.stack(terms).sum(dim=0)

    def stage_component_logits(self, features: Tensor) -> tuple[Tensor, ...]:
        """Expose exact per-stage component scores for attribution diagnostics."""
        prototypes = self.physical_prototypes()
        scale = self.logit_scale.clamp(-4.0, 4.0).exp()
        if not self.stage_dims:
            return (self._component_logits(features, prototypes),)
        feature_stages = features.split(self.stage_dims, dim=-1)
        prototype_stages = prototypes.split(self.stage_dims, dim=-1)
        stage_scales = self.stage_log_scales.clamp(-3.0, 3.0).exp()
        output = []
        for index, (feature_stage, prototype_stage) in enumerate(
            zip(feature_stages, prototype_stages, strict=True)
        ):
            output.append(
                scale
                * stage_scales[index]
                * prototype_logits(
                    feature_stage,
                    prototype_stage.reshape(
                        self.classes * self.components,
                        self.stage_dims[index],
                    ),
                    diagonal=self._diagonal(self.log_diagonals[index]),
                    low_rank=self.low_ranks[index],
                )
            )
        return tuple(output)

    def affine_component_parameters(self) -> tuple[Tensor, Tensor]:
        """Return exact per-component affine weight and bias under the PSD metric."""
        prototypes = self.physical_prototypes()
        weights = []
        biases = []
        scale = self.logit_scale.clamp(-4.0, 4.0).exp()
        if not self.stage_dims:
            flat = prototypes.reshape(self.classes * self.components, self.input_dim)
            diagonal = self._diagonal(self.log_diagonals[0])
            factor = self.low_ranks[0]
            metric_prototype = flat * diagonal + (flat @ factor) @ factor.T
            return scale * 2.0 * metric_prototype, scale * -(
                flat * metric_prototype
            ).sum(dim=-1)
        prototype_stages = prototypes.split(self.stage_dims, dim=-1)
        stage_scales = self.stage_log_scales.clamp(-3.0, 3.0).exp()
        for index, prototype_stage in enumerate(prototype_stages):
            flat = prototype_stage.reshape(
                self.classes * self.components,
                self.stage_dims[index],
            )
            diagonal = self._diagonal(self.log_diagonals[index])
            factor = self.low_ranks[index]
            metric_prototype = flat * diagonal + (flat @ factor) @ factor.T
            active_scale = scale * stage_scales[index]
            weights.append(active_scale * 2.0 * metric_prototype)
            biases.append(active_scale * -(flat * metric_prototype).sum(dim=-1))
        return torch.cat(weights, dim=-1), torch.stack(biases, dim=0).sum(dim=0)

    def component_logits(self, features: Tensor) -> Tensor:
        return self._component_logits(features, self.physical_prototypes())

    def component_distances(self, features: Tensor) -> Tensor:
        """Return exact squared ``D + U U^T`` distances to every component."""
        prototypes = self.physical_prototypes()
        if not self.stage_dims:
            difference = features[:, None, None, :] - prototypes[None, :, :, :]
            diagonal = self._diagonal(self.log_diagonals[0])
            distance = (difference.square() * diagonal).sum(dim=-1)
            projected = difference @ self.low_ranks[0]
            return distance + projected.square().sum(dim=-1)
        feature_stages = features.split(self.stage_dims, dim=-1)
        prototype_stages = prototypes.split(self.stage_dims, dim=-1)
        scales = self.stage_log_scales.clamp(-3.0, 3.0).exp()
        terms = []
        for index, (feature_stage, prototype_stage) in enumerate(
            zip(feature_stages, prototype_stages, strict=True)
        ):
            difference = feature_stage[:, None, None, :] - prototype_stage[None, :, :, :]
            diagonal = self._diagonal(self.log_diagonals[index])
            distance = (difference.square() * diagonal).sum(dim=-1)
            projected = difference @ self.low_ranks[index]
            terms.append(scales[index] * (distance + projected.square().sum(dim=-1)))
        return torch.stack(terms, dim=0).sum(dim=0)

    def forward(self, features: Tensor) -> Tensor:
        logits = self.component_logits(features)
        if self.components == 1:
            return logits
        temperature = self.log_temperature.clamp(-2.0, 2.0).exp()
        return grouped_logsumexp_logits(
            logits,
            classes=self.classes,
            components=self.components,
            temperature=temperature,
        )

    def compactness(self, features: Tensor, labels: Tensor) -> Tensor:
        prototypes = self.physical_prototypes()[labels]
        if not self.stage_dims:
            difference = features[:, None, :] - prototypes
            diagonal = self._diagonal(self.log_diagonals[0])
            active = (difference.square() * diagonal).sum(dim=-1)
            active = active + (difference @ self.low_ranks[0]).square().sum(dim=-1)
        else:
            feature_stages = features.split(self.stage_dims, dim=-1)
            prototype_stages = prototypes.split(self.stage_dims, dim=-1)
            scales = self.stage_log_scales.clamp(-3.0, 3.0).exp()
            terms = []
            for index, (feature_stage, prototype_stage) in enumerate(
                zip(feature_stages, prototype_stages, strict=True)
            ):
                difference = feature_stage[:, None, :] - prototype_stage
                diagonal = self._diagonal(self.log_diagonals[index])
                distance = (difference.square() * diagonal).sum(dim=-1)
                distance = distance + (
                    difference @ self.low_ranks[index]
                ).square().sum(dim=-1)
                terms.append(scales[index] * distance)
            active = torch.stack(terms, dim=0).sum(dim=0)
        # Smooth minimum keeps all components trainable while approximating the
        # requested nearest physical spectral centroid objective.
        return -torch.logsumexp(-active, dim=-1).mean() / self.input_dim

    @torch.no_grad()
    def update_ema(
        self,
        features: Tensor,
        labels: Tensor,
        *,
        weights: Tensor | None = None,
    ) -> None:
        """Update class/component energy centroids after backward completes."""
        features = features.detach().float()
        labels = labels.detach().to(torch.long)
        if weights is None:
            weights = torch.ones(features.shape[0], device=features.device)
        else:
            weights = weights.detach().float()
        prototypes = self.physical_prototypes().detach()
        target_prototypes = prototypes[labels]
        # Assignment uses an inexpensive shared Euclidean diagnostic.  During
        # cold start, round-robin assignment prevents dead mixture components.
        distances = (features[:, None, :] - target_prototypes).square().mean(dim=-1)
        assignment = distances.argmin(dim=-1)
        cold = ~self.ema_initialized[labels].all(dim=-1)
        if bool(cold.any()):
            assignment[cold] = torch.arange(features.shape[0], device=features.device)[cold].remainder(
                self.components
            )
        group = labels * self.components + assignment
        group_count = self.classes * self.components
        raw_energy = torch.expm1(features.clamp(max=20.0)).clamp_min_(0.0)
        weighted_energy = raw_energy * weights[:, None]
        sums = torch.zeros(group_count, self.input_dim, device=features.device)
        mass = torch.zeros(group_count, device=features.device)
        sums.index_add_(0, group, weighted_energy)
        mass.index_add_(0, group, weights)
        active_indices = torch.nonzero(mass > 0.0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return
        means = sums[active_indices] / mass[active_indices, None]
        flat_energy = self.ema_energy.view(group_count, self.input_dim)
        flat_initialized = self.ema_initialized.view(group_count)
        old = flat_energy[active_indices]
        initialized = flat_initialized[active_indices]
        updated = torch.where(
            initialized[:, None],
            old.lerp(means, self.ema_momentum),
            means,
        )
        flat_energy.index_copy_(0, active_indices, updated)
        flat_initialized[active_indices] = True
        self.update_count.view(-1).index_add_(0, active_indices, mass[active_indices])

    def delta_penalty(self) -> Tensor:
        return self.delta.square().mean()

    def component_usage(self) -> Tensor:
        mass = self.update_count.sum(dim=0)
        return mass / mass.sum().clamp_min(1.0)


class A2DPrototypeClassifier(nn.Module):
    """Compose prototype, Fusion, and LRQ branches for E0--E6."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        prototype: EMAPrototypeHead | None,
        use_fusion: bool,
        use_lrq: bool,
        fusion_width: int = 384,
        lrq_rank: int = 64,
        prototype_main: bool = True,
        beta_fusion: float = 0.1,
        beta_lrq: float = 0.1,
        prototype_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        self.prototype = prototype
        self.fusion = ModalFusionHead(input_dim, fusion_width, output_dim) if use_fusion else None
        self.lrq = LowRankQuadraticModalHead(input_dim, output_dim, lrq_rank) if use_lrq else None
        self.prototype_main = prototype_main
        self.prototype_auxiliary = prototype_auxiliary
        self.beta_fusion = nn.Parameter(torch.tensor(float(beta_fusion))) if use_fusion else None
        self.beta_lrq = nn.Parameter(torch.tensor(float(beta_lrq))) if use_lrq else None

    def branch_logits(self, descriptor: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        reference: Tensor | None = None
        prototype_logits_active = self.prototype(descriptor) if self.prototype is not None else None
        if prototype_logits_active is not None:
            reference = prototype_logits_active
        fusion_logits = self.fusion(descriptor) if self.fusion is not None else None
        if reference is None and fusion_logits is not None:
            reference = fusion_logits
        lrq_logits = self.lrq(descriptor) if self.lrq is not None else None
        if reference is None and lrq_logits is not None:
            reference = lrq_logits
        if reference is None:
            raise RuntimeError("A2D classifier has no active branch")
        zero = torch.zeros_like(reference)
        prototype_term = prototype_logits_active if prototype_logits_active is not None else zero
        fusion_term = fusion_logits if fusion_logits is not None else zero
        lrq_term = lrq_logits if lrq_logits is not None else zero
        if self.prototype_main:
            joint = prototype_term
            if self.beta_fusion is not None:
                joint = joint + self.beta_fusion * fusion_term
            if self.beta_lrq is not None:
                joint = joint + self.beta_lrq * lrq_term
        else:
            joint = fusion_term
            if self.beta_lrq is not None:
                joint = joint + self.beta_lrq * lrq_term
        return joint, prototype_term, fusion_term, lrq_term

    def forward(self, descriptor: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        joint, prototype, fusion, lrq = self.branch_logits(descriptor)
        if self.prototype is not None:
            return joint, prototype, fusion, lrq, descriptor
        return joint


__all__ = [
    "A2DPrototypeClassifier",
    "EMAPrototypeHead",
    "PrototypeMetricHead",
    "PrototypeResidualHead",
    "StagewisePrototypeMetricHead",
    "energy_kmeans_prototypes",
    "expected_calibration_error",
    "grouped_logsumexp_logits",
]
