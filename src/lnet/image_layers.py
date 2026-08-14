"""Shared image stems, normalization, and descriptor heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """Apply channel-wise LayerNorm to an NCHW feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.norm(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class CifarConvStem(nn.Module):
    """Two overlapping convolutions with an explicit dataset-scale stride."""

    def __init__(
        self,
        output_width: int = 64,
        strides: tuple[int, int] = (1, 1),
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        first = nn.Conv2d(3, 32, 3, stride=strides[0], padding=1, bias=False)
        second = nn.Conv2d(
            32,
            output_width,
            3,
            stride=strides[1],
            padding=1,
            bias=False,
        )
        if bias:
            # Preserve the bias-free initialization while making the offsets trainable.
            first.bias = nn.Parameter(torch.zeros(32))
            second.bias = nn.Parameter(torch.zeros(output_width))
        self.layers = nn.Sequential(
            first,
            LayerNorm2d(32),
            nn.GELU(),
            second,
            LayerNorm2d(output_width),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs).permute(0, 2, 3, 1)


class LowRankQuadraticModalHead(nn.Module):
    """Structured modal classifier with linear and rank-wise attribution."""

    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or rank <= 0:
            message = "modal head dimensions and rank must be positive"
            raise ValueError(message)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.standardizer = nn.BatchNorm1d(input_dim, affine=False)
        self.linear = nn.Linear(input_dim, output_dim)
        self.projection = nn.Linear(input_dim, rank, bias=False)
        self.quadratic = nn.Linear(rank, output_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        nn.init.normal_(self.quadratic.weight, std=1.0e-3)

    def standardized(self, descriptor: Tensor) -> Tensor:
        return self.standardizer(descriptor)

    def forward(self, descriptor: Tensor) -> Tensor:
        standardized = self.standardized(descriptor)
        interaction = self.projection(standardized).square()
        return self.linear(standardized) + self.quadratic(interaction)

    def decompose(self, descriptor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return bias, coordinate-linear, and latent-quadratic logit terms."""
        standardized = self.standardized(descriptor)
        linear = standardized[:, None, :] * self.linear.weight[None, :, :]
        latent = self.projection(standardized).square()
        quadratic = latent[:, None, :] * self.quadratic.weight[None, :, :]
        bias = self.linear.bias
        if bias is None:
            message = "the modal linear term requires a bias"
            raise RuntimeError(message)
        return bias, linear, quadratic


class StandardizedAffineModalHead(nn.Module):
    """Affine modal classifier with parameter-free standardization."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.standardizer = nn.BatchNorm1d(input_dim, affine=False)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, descriptor: Tensor) -> Tensor:
        return self.linear(self.standardizer(descriptor))


__all__ = [
    "CifarConvStem",
    "LayerNorm2d",
    "LowRankQuadraticModalHead",
    "StandardizedAffineModalHead",
]
