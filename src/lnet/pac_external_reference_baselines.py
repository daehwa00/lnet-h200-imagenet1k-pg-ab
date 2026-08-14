from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_confirmatory_baselines import InceptionTimeClassifier, S4DClassifier


class ExternalCNN1DClassifier(nn.Module):
    """Depth-4 noncausal CNN1D adapter for multichannel external tasks."""

    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = input_dim
        for index in range(4):
            output_channels = width if index == 0 else 2 * width
            layers.extend(
                (
                    nn.Conv1d(channels, output_channels, 3, padding=1),
                    nn.BatchNorm1d(output_channels),
                    nn.GELU(),
                )
            )
            channels = output_channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(channels, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.encoder(inputs.transpose(1, 2)).mean(dim=-1)
        return self.head(features)


class ExternalS4DClassifier(S4DClassifier):
    """Reference-faithful S4D-Lin adapter for the external task protocol."""

    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__(
            width,
            output_dim,
            depth=1,
            state_size=8,
            input_dim=input_dim,
        )


class ExternalInceptionTimeClassifier(InceptionTimeClassifier):
    """Canonical three-module InceptionTime block with a task-specific head."""

    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__(
            width,
            output_dim,
            block_count=2,
            kernel_scale=1,
            input_dim=input_dim,
        )


class _S4DPLRKernel(nn.Module):
    """Stable diagonal-plus-low-rank S4 kernel with bilinear discretization."""

    def __init__(self, channels: int, state_size: int = 8) -> None:
        super().__init__()
        if state_size < 2 or state_size % 2:
            message = "S4 state_size must be an even integer >= 2"
            raise ValueError(message)
        modes = state_size // 2
        self.log_dt = nn.Parameter(torch.empty(channels).uniform_(math.log(1e-3), math.log(1e-1)))
        self.log_decay = nn.Parameter(torch.full((channels, modes), math.log(0.5)))
        frequencies = torch.pi * torch.arange(modes, dtype=torch.float32)
        self.frequency = nn.Parameter(frequencies.repeat(channels, 1))
        scale = modes**-0.5
        self.low_rank = nn.Parameter(scale * torch.randn(channels, modes, 2))
        self.input_vector = nn.Parameter(scale * torch.randn(channels, modes, 2))
        self.readout = nn.Parameter(scale * torch.randn(channels, modes, 2))

    def forward(self, length: int) -> Tensor:
        if length < 1:
            message = "S4 kernel length must be positive"
            raise ValueError(message)
        dtype = self.log_dt.dtype
        device = self.log_dt.device
        poles = torch.complex(-torch.exp(self.log_decay), self.frequency)
        low_rank = torch.view_as_complex(self.low_rank.contiguous())
        input_vector = torch.view_as_complex(self.input_vector.contiguous())
        readout = torch.view_as_complex(self.readout.contiguous())
        state = poles.shape[-1]
        diagonal = torch.diag_embed(poles)
        correction = low_rank.unsqueeze(-1) * low_rank.conj().unsqueeze(-2)
        continuous = diagonal - correction
        step = torch.exp(self.log_dt).view(-1, 1, 1)
        identity = torch.eye(state, device=device, dtype=continuous.dtype).unsqueeze(0)
        lhs = identity - 0.5 * step * continuous
        transition = torch.linalg.solve(lhs, identity + 0.5 * step * continuous)
        drive = torch.linalg.solve(lhs, step.squeeze(-1) * input_vector)

        eigenvalues, eigenvectors = torch.linalg.eig(transition)
        modal_drive = torch.linalg.solve(eigenvectors, drive.unsqueeze(-1)).squeeze(-1)
        modal_readout = torch.einsum("hn,hnk->hk", readout, eigenvectors)
        coefficients = modal_readout * modal_drive
        time = torch.arange(length, device=device, dtype=dtype)
        powers = torch.exp(torch.log(eigenvalues).unsqueeze(-1) * time)
        return 2.0 * torch.sum(coefficients.unsqueeze(-1) * powers, dim=1).real


class _S4Layer(nn.Module):
    def __init__(self, width: int, state_size: int = 8) -> None:
        super().__init__()
        self.kernel = _S4DPLRKernel(width, state_size)
        self.skip = nn.Parameter(torch.ones(width))
        self.output_projection = nn.Linear(width, 2 * width)

    def forward(self, inputs: Tensor) -> Tensor:
        original_dtype = inputs.dtype
        values = inputs.transpose(1, 2).to(dtype=torch.float32)
        length = values.shape[-1]
        kernel = self.kernel(length).to(device=values.device, dtype=values.dtype)
        fft_size = 2 * length
        convolved = torch.fft.irfft(
            torch.fft.rfft(values, n=fft_size)
            * torch.fft.rfft(kernel, n=fft_size).unsqueeze(0),
            n=fft_size,
        )[..., :length]
        convolved = convolved + self.skip.unsqueeze(-1) * values
        mixed = self.output_projection(functional.gelu(convolved).transpose(1, 2))
        return functional.glu(mixed, dim=-1).to(dtype=original_dtype)


class ExternalS4Classifier(nn.Module):
    """Repository-native S4-style DPLR baseline for common external tasks."""

    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, width)
        self.norm = nn.LayerNorm(width)
        self.s4 = _S4Layer(width)
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.input_projection(inputs)
        features = features + self.s4(self.norm(features))
        return self.head(self.final_norm(features).mean(dim=1))


class ExternalMiniRocketClassifier(nn.Module):
    """Deterministic MiniRocket-style PPV transform with a learned linear head."""

    _KERNEL_SIZE = 9

    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        if width < 1:
            message = "MiniRocket feature width must be positive"
            raise ValueError(message)
        kernels = self._make_kernels(input_dim, width)
        self.register_buffer("kernels", kernels, persistent=True)
        self.register_buffer("bias", torch.linspace(-1.0, 1.0, width), persistent=True)
        dilations = torch.tensor(tuple(2 ** (index % 4) for index in range(width)))
        self.register_buffer("dilations", dilations, persistent=True)
        self.head = nn.Linear(width, output_dim)

    @classmethod
    def _make_kernels(cls, input_dim: int, width: int) -> Tensor:
        combinations = tuple(itertools.combinations(range(cls._KERNEL_SIZE), 3))
        kernels = torch.empty(width, input_dim, cls._KERNEL_SIZE)
        for feature in range(width):
            pattern = torch.full((cls._KERNEL_SIZE,), -1.0)
            pattern[list(combinations[feature % len(combinations)])] = 2.0
            channel_sign = torch.where(
                (torch.arange(input_dim) + feature) % 2 == 0,
                1.0,
                -1.0,
            )
            kernels[feature] = channel_sign.unsqueeze(-1) * pattern / math.sqrt(input_dim * 18.0)
        return kernels

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs.transpose(1, 2)
        kernels = self.get_buffer("kernels")
        bias = self.get_buffer("bias")
        dilations = self.get_buffer("dilations")
        features = torch.empty(
            inputs.shape[0],
            kernels.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for dilation in (1, 2, 4, 8):
            indices = torch.nonzero(dilations == dilation, as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            padding = dilation * (self._KERNEL_SIZE - 1) // 2
            response = functional.conv1d(
                values,
                kernels[indices].to(dtype=values.dtype),
                dilation=dilation,
                padding=padding,
            )
            threshold = bias[indices].to(dtype=response.dtype).view(1, -1, 1)
            features[:, indices] = (response > threshold).to(response.dtype).mean(dim=-1)
        return self.head(features)
