"""Head-design variants for frozen ALPHABET-2D modal descriptors.

The module deliberately keeps input normalization inside each head.  This lets
the frozen-Q campaign distinguish dataset-affine preprocessing (which preserves
an affine decision surface in the original descriptor) from sample-dependent
normalization such as LayerNorm and RMSNorm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

INPUT_DIM = 576
STAGE_DIM = 192


@dataclass(frozen=True, slots=True)
class HeadDesignSpec:
    """A single frozen-Q head experiment."""

    name: str
    family: Literal[
        "linear",
        "fusion",
        "stage_embedding",
        "stage_logits",
        "stage_residual",
        "stage_residual_only",
        "stage_residual_independent",
        "fusion_residual_depth",
        "grouped_stage_fusion",
        "grouped_direction_fusion",
    ]
    normalizer: str = "batch_fixed"
    widths: tuple[int, ...] = ()
    activation: str = "gelu"
    hidden_norm: str = "rms_after"
    dropout: float = 0.0


SCREEN_SPECS: tuple[HeadDesignSpec, ...] = (
    # Dataset/sample normalization ladder.
    HeadDesignSpec("N0-Raw", "linear", normalizer="identity"),
    HeadDesignSpec("N1-BNFixed", "linear", normalizer="batch_fixed"),
    HeadDesignSpec("N2-BNAffine", "linear", normalizer="batch_affine"),
    HeadDesignSpec("N3-ZScore", "linear", normalizer="fixed_zscore"),
    HeadDesignSpec("N4-RMSScale", "linear", normalizer="fixed_rms"),
    HeadDesignSpec("N5-StageScale", "linear", normalizer="stage_scalar"),
    HeadDesignSpec("N6-Whiten", "linear", normalizer="fixed_whiten"),
    HeadDesignSpec("N7-LayerNorm", "linear", normalizer="layer"),
    HeadDesignSpec("N8-RMSNorm", "linear", normalizer="rms"),
    # Hidden-normalization placement.
    HeadDesignSpec("HN0-NoNorm", "fusion", widths=(256,), hidden_norm="none"),
    HeadDesignSpec("HN1-RMSAfter", "fusion", widths=(256,), hidden_norm="rms_after"),
    HeadDesignSpec("HN2-RMSBefore", "fusion", widths=(256,), hidden_norm="rms_before"),
    HeadDesignSpec("HN3-LNAfter", "fusion", widths=(256,), hidden_norm="layer_after"),
    HeadDesignSpec("HN4-BNAfter", "fusion", widths=(256,), hidden_norm="batch_after"),
    # Activation family at approximately matched width/parameter count.
    HeadDesignSpec("A0-Identity", "fusion", widths=(256,), activation="identity"),
    HeadDesignSpec("A2-SiLU", "fusion", widths=(256,), activation="silu"),
    HeadDesignSpec("A3-ReLU", "fusion", widths=(256,), activation="relu"),
    HeadDesignSpec("A4-ReLU2", "fusion", widths=(256,), activation="relu2"),
    HeadDesignSpec("A5-SwiGLU", "fusion", widths=(136,), activation="swiglu"),
    # Width/depth ladder.
    HeadDesignSpec("W128", "fusion", widths=(128,)),
    HeadDesignSpec("W384", "fusion", widths=(384,)),
    HeadDesignSpec("W576", "fusion", widths=(576,)),
    HeadDesignSpec("W768", "fusion", widths=(768,)),
    HeadDesignSpec("D256x2", "fusion", widths=(256, 256)),
    HeadDesignSpec("D384x2", "fusion", widths=(384, 384)),
    HeadDesignSpec("D512-256", "fusion", widths=(512, 256)),
    # Dropout is isolated on the current 256-wide head.
    HeadDesignSpec("DO005", "fusion", widths=(256,), dropout=0.05),
    HeadDesignSpec("DO010", "fusion", widths=(256,), dropout=0.10),
    HeadDesignSpec("DO020", "fusion", widths=(256,), dropout=0.20),
    # Stage-aware alternatives.
    HeadDesignSpec("S0-StageEmbed", "stage_embedding", widths=(128,)),
    HeadDesignSpec("S1-StageLogits", "stage_logits"),
    HeadDesignSpec("S2-StageResidual", "stage_residual", widths=(64,)),
    HeadDesignSpec("SR-DO020", "stage_residual", widths=(64,), dropout=0.20),
    # Focused follow-up: efficient stage-residual scaling and matched controls.
    HeadDesignSpec("SR16", "stage_residual", widths=(16,)),
    HeadDesignSpec("SR32", "stage_residual", widths=(32,)),
    HeadDesignSpec("SR96", "stage_residual", widths=(96,)),
    HeadDesignSpec("SR128", "stage_residual", widths=(128,)),
    HeadDesignSpec("SR192", "stage_residual", widths=(192,)),
    HeadDesignSpec("Dense168", "fusion", widths=(168,)),
    HeadDesignSpec("SR-Only64", "stage_residual_only", widths=(64,)),
    HeadDesignSpec("SR-Independent64", "stage_residual_independent", widths=(64,)),
    HeadDesignSpec("ResidualDepth384", "fusion_residual_depth", widths=(384,)),
    HeadDesignSpec("StageBlock768", "grouped_stage_fusion", widths=(256,)),
    HeadDesignSpec("DirectionBlock768", "grouped_direction_fusion", widths=(192,)),
    HeadDesignSpec("SR64-FixedZ", "stage_residual", normalizer="fixed_zscore", widths=(64,)),
    HeadDesignSpec("SR64-BNHidden", "stage_residual", widths=(64,), hidden_norm="batch_after"),
)


def _activation(name: str) -> nn.Module:
    if name == "identity":
        return nn.Identity()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "relu2":
        return SquaredReLU()
    message = f"unsupported activation: {name}"
    raise ValueError(message)


class SquaredReLU(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.relu(inputs).square()


class FixedAffineTransform(nn.Module):
    """A fixed dataset-level affine transform, optionally with a dense matrix."""

    def __init__(self, offset: Tensor, scale_or_matrix: Tensor) -> None:
        super().__init__()
        self.register_buffer("offset", offset.detach().clone())
        self.register_buffer("scale_or_matrix", scale_or_matrix.detach().clone())

    def forward(self, inputs: Tensor) -> Tensor:
        centered = inputs - self.offset
        if self.scale_or_matrix.ndim == 1:
            return centered / self.scale_or_matrix
        return centered @ self.scale_or_matrix


class StageScalarTransform(nn.Module):
    def __init__(self, means: Tensor, scales: Tensor) -> None:
        super().__init__()
        self.register_buffer("means", means.detach().clone())
        self.register_buffer("scales", scales.detach().clone())

    def forward(self, inputs: Tensor) -> Tensor:
        shaped = inputs.reshape(inputs.shape[0], 3, STAGE_DIM)
        return ((shaped - self.means[None, :, None]) / self.scales[None, :, None]).flatten(1)


class SwiGLULayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.projection(inputs).chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * value


class FusionHead(nn.Module):
    """Configurable descriptor MLP with observable hidden representation."""

    def __init__(
        self,
        transform: nn.Module,
        classes: int,
        widths: tuple[int, ...],
        *,
        activation: str,
        hidden_norm: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.blocks = nn.ModuleList()
        previous = INPUT_DIM
        for width in widths:
            if activation == "swiglu":
                projection: nn.Module = SwiGLULayer(previous, width)
                activation_module: nn.Module = nn.Identity()
            else:
                projection = nn.Linear(previous, width)
                activation_module = _activation(activation)
            before = nn.RMSNorm(previous) if hidden_norm == "rms_before" else nn.Identity()
            if hidden_norm == "rms_after":
                after: nn.Module = nn.RMSNorm(width)
            elif hidden_norm == "layer_after":
                after = nn.LayerNorm(width)
            elif hidden_norm == "batch_after":
                after = nn.BatchNorm1d(width)
            else:
                after = nn.Identity()
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "before": before,
                        "projection": projection,
                        "activation": activation_module,
                        "after": after,
                        "dropout": nn.Dropout(dropout),
                    }
                )
            )
            previous = width
        self.classifier = nn.Linear(previous, classes)

    def features(self, inputs: Tensor) -> Tensor:
        hidden = self.transform(inputs)
        for block in self.blocks:
            hidden = block["before"](hidden)
            hidden = block["projection"](hidden)
            hidden = block["activation"](hidden)
            hidden = block["after"](hidden)
            hidden = block["dropout"](hidden)
        return hidden

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


class LinearHead(nn.Module):
    def __init__(self, transform: nn.Module, classes: int) -> None:
        super().__init__()
        self.transform = transform
        self.classifier = nn.Linear(INPUT_DIM, classes)

    def features(self, inputs: Tensor) -> Tensor:
        return self.transform(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


class StageEmbeddingHead(nn.Module):
    def __init__(self, transform: nn.Module, classes: int, width: int) -> None:
        super().__init__()
        self.transform = transform
        self.embeddings = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(STAGE_DIM, width), nn.GELU(), nn.RMSNorm(width))
                for _ in range(3)
            ]
        )
        self.classifier = nn.Linear(3 * width, classes)

    def features(self, inputs: Tensor) -> Tensor:
        stages = self.transform(inputs).split(STAGE_DIM, dim=-1)
        return torch.cat(
            [module(stage) for module, stage in zip(self.embeddings, stages, strict=True)],
            dim=-1,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


class StageLogitHead(nn.Module):
    """Exactly affine, but exposes the three stage contributions."""

    def __init__(self, transform: nn.Module, classes: int) -> None:
        super().__init__()
        self.transform = transform
        self.weights = nn.ModuleList([nn.Linear(STAGE_DIM, classes, bias=False) for _ in range(3)])
        self.bias = nn.Parameter(torch.zeros(classes))

    def stage_logits(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stages = self.transform(inputs).split(STAGE_DIM, dim=-1)
        return tuple(  # type: ignore[return-value]
            module(stage) for module, stage in zip(self.weights, stages, strict=True)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return sum(self.stage_logits(inputs), start=self.bias)


def _residual_norm(*, hidden_norm: str, width: int) -> nn.Module:
    if hidden_norm == "rms_after":
        return nn.RMSNorm(width)
    if hidden_norm == "batch_after":
        return nn.BatchNorm1d(width)
    if hidden_norm == "layer_after":
        return nn.LayerNorm(width)
    if hidden_norm == "none":
        return nn.Identity()
    message = f"unsupported residual hidden norm: {hidden_norm}"
    raise ValueError(message)


class StageResidualHead(nn.Module):
    """Affine main decision plus a small cross-stage nonlinear residual."""

    def __init__(
        self,
        transform: nn.Module,
        classes: int,
        width: int,
        dropout: float = 0.0,
        hidden_norm: str = "rms_after",
    ) -> None:
        super().__init__()
        self.transform = transform
        self.affine = nn.Linear(INPUT_DIM, classes)
        self.stage_embeddings = nn.ModuleList(
            [nn.Sequential(nn.Linear(STAGE_DIM, width), nn.GELU()) for _ in range(3)]
        )
        self.residual = nn.Sequential(
            _residual_norm(hidden_norm=hidden_norm, width=3 * width),
            nn.Dropout(dropout),
            nn.Linear(3 * width, classes),
        )
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.transform(inputs)
        stages = normalized.split(STAGE_DIM, dim=-1)
        hidden = torch.cat(
            [module(stage) for module, stage in zip(self.stage_embeddings, stages, strict=True)],
            dim=-1,
        )
        return self.affine(normalized) + self.beta * self.residual(hidden)


class StageResidualOnlyHead(nn.Module):
    """Stage-aware nonlinear branch without the affine main path."""

    def __init__(self, transform: nn.Module, classes: int, width: int) -> None:
        super().__init__()
        self.transform = transform
        self.stage_embeddings = nn.ModuleList(
            [nn.Sequential(nn.Linear(STAGE_DIM, width), nn.GELU()) for _ in range(3)]
        )
        self.classifier = nn.Sequential(nn.RMSNorm(3 * width), nn.Linear(3 * width, classes))

    def forward(self, inputs: Tensor) -> Tensor:
        stages = self.transform(inputs).split(STAGE_DIM, dim=-1)
        hidden = torch.cat(
            [module(stage) for module, stage in zip(self.stage_embeddings, stages, strict=True)],
            dim=-1,
        )
        return self.classifier(hidden)


class IndependentStageResidualHead(nn.Module):
    """Affine main path plus independent nonlinear correction from each stage."""

    def __init__(
        self,
        transform: nn.Module,
        classes: int,
        width: int,
        *,
        input_dim: int = INPUT_DIM,
        stage_dim: int = STAGE_DIM,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or stage_dim <= 0 or input_dim % stage_dim:
            message = "independent stage residual requires equal non-empty stage blocks"
            raise ValueError(message)
        self.transform = transform
        self.input_dim = input_dim
        self.stage_dim = stage_dim
        self.stage_count = input_dim // stage_dim
        self.affine = nn.Linear(input_dim, classes)
        self.stage_residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(stage_dim, width),
                    nn.GELU(),
                    nn.RMSNorm(width),
                    nn.Linear(width, classes, bias=False),
                )
                for _ in range(self.stage_count)
            ]
        )
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.transform(inputs)
        stages = normalized.split(self.stage_dim, dim=-1)
        affine_logits = self.affine(normalized)
        correction = sum(
            (module(stage) for module, stage in zip(self.stage_residuals, stages, strict=True)),
            start=torch.zeros_like(affine_logits),
        )
        return affine_logits + self.beta * correction


class ResidualDepthFusionHead(nn.Module):
    """One wide fusion followed by a zero-initialized residual MLP block."""

    def __init__(self, transform: nn.Module, classes: int, width: int) -> None:
        super().__init__()
        self.transform = transform
        self.input_projection = nn.Sequential(
            nn.Linear(INPUT_DIM, width), nn.GELU(), nn.RMSNorm(width)
        )
        self.residual = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.eta = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Linear(width, classes)

    def features(self, inputs: Tensor) -> Tensor:
        hidden = self.input_projection(self.transform(inputs))
        return hidden + self.eta * self.residual(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


class GroupedFusionHead(nn.Module):
    """Fusion with stage- or direction-restricted input mixing."""

    def __init__(self, transform: nn.Module, classes: int, width: int, grouping: str) -> None:
        super().__init__()
        self.transform = transform
        self.grouping = grouping
        if grouping == "stage":
            count, input_width = 3, STAGE_DIM
        elif grouping == "direction":
            count, input_width = 4, 3 * 48
        else:
            message = f"unsupported descriptor grouping: {grouping}"
            raise ValueError(message)
        self.projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_width, width), nn.GELU()) for _ in range(count)]
        )
        total_width = count * width
        self.hidden_norm = nn.RMSNorm(total_width)
        self.classifier = nn.Linear(total_width, classes)

    def _groups(self, inputs: Tensor) -> tuple[Tensor, ...]:
        if self.grouping == "stage":
            return inputs.split(STAGE_DIM, dim=-1)
        shaped = inputs.reshape(inputs.shape[0], 3, 4, 48)
        return tuple(shaped[:, :, direction, :].flatten(1) for direction in range(4))

    def features(self, inputs: Tensor) -> Tensor:
        groups = self._groups(self.transform(inputs))
        hidden = torch.cat(
            [module(group) for module, group in zip(self.projections, groups, strict=True)],
            dim=-1,
        )
        return self.hidden_norm(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


@dataclass(frozen=True, slots=True)
class DescriptorStatistics:
    mean: Tensor
    std: Tensor
    rms: Tensor
    stage_means: Tensor
    stage_scales: Tensor
    whitening: Tensor


@torch.no_grad()
def descriptor_statistics(features: Tensor, *, shrinkage: float = 0.05) -> DescriptorStatistics:
    """Compute train-only transforms used by the normalization ladder."""

    values = features.float()
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=True).clamp_min(1.0e-5)
    rms = values.square().mean(dim=0).sqrt().clamp_min(1.0e-5)
    stages = values.reshape(values.shape[0], 3, STAGE_DIM)
    stage_means = stages.mean(dim=(0, 2))
    stage_scales = stages.var(dim=(0, 2), unbiased=True).sqrt().clamp_min(1.0e-5)
    centered = values - mean
    covariance = centered.T @ centered / max(1, values.shape[0] - 1)
    average_variance = covariance.diagonal().mean()
    covariance = (1.0 - shrinkage) * covariance
    covariance.diagonal().add_(shrinkage * average_variance)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    whitening = (eigenvectors * eigenvalues.clamp_min(1.0e-6).rsqrt()) @ eigenvectors.T
    return DescriptorStatistics(mean, std, rms, stage_means, stage_scales, whitening)


def build_transform(  # noqa: PLR0911
    name: str, statistics: DescriptorStatistics
) -> nn.Module:
    if name == "identity":
        return nn.Identity()
    if name == "batch_fixed":
        return nn.BatchNorm1d(INPUT_DIM, affine=False)
    if name == "batch_affine":
        return nn.BatchNorm1d(INPUT_DIM, affine=True)
    if name == "fixed_zscore":
        return FixedAffineTransform(statistics.mean, statistics.std)
    if name == "fixed_rms":
        return FixedAffineTransform(torch.zeros_like(statistics.mean), statistics.rms)
    if name == "stage_scalar":
        return StageScalarTransform(statistics.stage_means, statistics.stage_scales)
    if name == "fixed_whiten":
        return FixedAffineTransform(statistics.mean, statistics.whitening)
    if name == "layer":
        return nn.LayerNorm(INPUT_DIM)
    if name == "rms":
        return nn.RMSNorm(INPUT_DIM)
    message = f"unsupported normalizer: {name}"
    raise ValueError(message)


def build_head(  # noqa: C901, PLR0911
    spec: HeadDesignSpec,
    statistics: DescriptorStatistics,
    *,
    classes: int,
) -> nn.Module:
    transform = build_transform(spec.normalizer, statistics)
    if spec.family == "linear":
        return LinearHead(transform, classes)
    if spec.family == "fusion":
        return FusionHead(
            transform,
            classes,
            spec.widths,
            activation=spec.activation,
            hidden_norm=spec.hidden_norm,
            dropout=spec.dropout,
        )
    if spec.family == "stage_embedding":
        return StageEmbeddingHead(transform, classes, spec.widths[0])
    if spec.family == "stage_logits":
        return StageLogitHead(transform, classes)
    if spec.family == "stage_residual":
        return StageResidualHead(
            transform,
            classes,
            spec.widths[0],
            spec.dropout,
            spec.hidden_norm,
        )
    if spec.family == "stage_residual_only":
        return StageResidualOnlyHead(transform, classes, spec.widths[0])
    if spec.family == "stage_residual_independent":
        return IndependentStageResidualHead(transform, classes, spec.widths[0])
    if spec.family == "fusion_residual_depth":
        return ResidualDepthFusionHead(transform, classes, spec.widths[0])
    if spec.family == "grouped_stage_fusion":
        return GroupedFusionHead(transform, classes, spec.widths[0], "stage")
    if spec.family == "grouped_direction_fusion":
        return GroupedFusionHead(transform, classes, spec.widths[0], "direction")
    message = f"unsupported head family: {spec.family}"
    raise ValueError(message)
