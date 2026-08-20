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


class ModeScaledTwoConvStem(nn.Module):
    """Apply 3-to-H-to-2K Conv-LN-GELU blocks and return NHWC features."""

    def __init__(
        self,
        modes: int,
        strides: tuple[int, int] = (2, 2),
        *,
        input_channels: int = 3,
        hidden_width: int = 32,
    ) -> None:
        super().__init__()
        if modes <= 0 or hidden_width <= 0 or input_channels <= 0:
            message = "mode-scaled stem dimensions must be positive"
            raise ValueError(message)
        if len(strides) != 2 or any(stride <= 0 for stride in strides):
            message = "mode-scaled stem requires two positive strides"
            raise ValueError(message)
        self.hidden_width = hidden_width
        self.output_width = 2 * modes
        first = nn.Conv2d(
            input_channels,
            hidden_width,
            3,
            stride=strides[0],
            padding=1,
            bias=True,
        )
        second = nn.Conv2d(
            hidden_width,
            self.output_width,
            3,
            stride=strides[1],
            padding=1,
            bias=True,
        )
        if first.bias is None or second.bias is None:
            raise RuntimeError("mode-scaled stem requires affine convolution biases")
        nn.init.zeros_(first.bias)
        nn.init.zeros_(second.bias)
        self.layers = nn.Sequential(
            first,
            LayerNorm2d(hidden_width),
            nn.GELU(),
            second,
            LayerNorm2d(self.output_width),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        first, first_norm, first_activation, second, second_norm, second_activation = self.layers
        hidden = first_activation(first_norm(first(inputs)))
        features = second(hidden)
        if not isinstance(second_norm, LayerNorm2d):
            message = "mode-scaled stem requires a terminal LayerNorm2d"
            raise TypeError(message)
        features = second_norm.norm(features.permute(0, 2, 3, 1))
        return second_activation(features)


class ResidualPreComplexMixer(nn.Module):
    """Turn the established square two-linear path into a residual mixer."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        if not isinstance(source, nn.Sequential) or len(source) != 4:
            message = "residual pre-complex mixer requires the established two-linear path"
            raise TypeError(message)
        input_projection, activation, output_projection, tail = tuple(source.children())
        if not isinstance(input_projection, nn.Linear) or not isinstance(
            output_projection,
            nn.Linear,
        ):
            message = "residual pre-complex mixer requires two linear projections"
            raise TypeError(message)
        width = input_projection.in_features
        if (
            not isinstance(activation, nn.GELU)
            or not isinstance(tail, nn.Identity)
            or input_projection.out_features != width
            or output_projection.in_features != width
            or output_projection.out_features != width
        ):
            message = "residual pre-complex mixer requires equal input and output widths"
            raise TypeError(message)
        self.width = width
        self.input_projection = input_projection
        self.activation = activation
        self.output_projection = output_projection

    def forward(self, inputs: Tensor) -> Tensor:
        update = self.activation(self.input_projection(inputs))
        flat_inputs = inputs.reshape(-1, self.width)
        flat_update = update.reshape(-1, self.width)
        mixed = torch.addmm(
            flat_inputs,
            flat_update,
            self.output_projection.weight.T,
        )
        if self.output_projection.bias is not None:
            mixed = mixed + self.output_projection.bias
        return mixed.reshape_as(inputs)


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
    "ModeScaledTwoConvStem",
    "ResidualPreComplexMixer",
    "StandardizedAffineModalHead",
]
