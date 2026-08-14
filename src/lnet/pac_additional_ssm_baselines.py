"""Compact reference-faithful S5, LRU, and DSS sequence classifiers.

The implementations intentionally share only the FFT convolution utility.  Their
state parameterizations, initializations, and channel contracts remain distinct,
so the Q1 comparison does not collapse three related papers into one generic SSM.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import Tensor, nn

_DT_MIN: Final = 1.0e-3
_DT_MAX: Final = 1.0e-1


def masked_temporal_mean(values: Tensor, valid_mask: Tensor | None) -> Tensor:
    if valid_mask is None:
        return values.mean(dim=1)
    weights = valid_mask.to(device=values.device, dtype=values.dtype)
    if weights.ndim == 2:
        weights = weights.unsqueeze(-1)
    if weights.shape != (*values.shape[:2], 1):
        message = "valid_mask must align with [batch, time]"
        raise ValueError(message)
    count = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / count


def _causal_mimo_convolution(inputs: Tensor, kernel: Tensor) -> Tensor:
    """Apply a real causal kernel [out, in, time] to [batch, time, in]."""
    length = inputs.shape[1]
    fft_size = 2 * length
    signal = torch.fft.rfft(inputs.transpose(1, 2).float(), n=fft_size)
    spectrum = torch.fft.rfft(kernel.float(), n=fft_size)
    output = torch.fft.irfft(
        torch.einsum("bit,oit->bot", signal, spectrum),
        n=fft_size,
    )[..., :length]
    return output.transpose(1, 2).to(dtype=inputs.dtype)


def _causal_depthwise_convolution(inputs: Tensor, kernel: Tensor) -> Tensor:
    length = inputs.shape[1]
    fft_size = 2 * length
    signal = torch.fft.rfft(inputs.transpose(1, 2).float(), n=fft_size)
    spectrum = torch.fft.rfft(kernel.float(), n=fft_size)
    output = torch.fft.irfft(signal * spectrum.unsqueeze(0), n=fft_size)[..., :length]
    return output.transpose(1, 2).to(dtype=inputs.dtype)


def _complex_parameter(*shape: int, scale: float) -> nn.Parameter:
    return nn.Parameter(scale * torch.randn(*shape, 2))


def _as_complex(parameter: Tensor) -> Tensor:
    return torch.view_as_complex(parameter.contiguous())


class _DSSLayer(nn.Module):
    """DSS-exp layer using the paper's length-normalized diagonal exponentials."""

    def __init__(self, width: int, state_size: int) -> None:
        super().__init__()
        modes = max(2, state_size // 2)
        self.log_decay = nn.Parameter(torch.full((width, modes), math.log(0.5)))
        frequencies = math.pi * torch.arange(modes, dtype=torch.float32)
        self.frequency = nn.Parameter(frequencies.repeat(width, 1))
        self.readout = _complex_parameter(width, modes, scale=modes**-0.5)
        self.skip = nn.Parameter(torch.ones(width))
        self.output_projection = nn.Linear(width, 2 * width)

    def _kernel(self, length: int) -> Tensor:
        poles = torch.complex(-torch.exp(self.log_decay), self.frequency)
        time = torch.arange(length, device=poles.device, dtype=poles.real.dtype)
        logits = poles.unsqueeze(-1) * time
        logits = logits - logits.real.amax(dim=-1, keepdim=True)
        exponentials = torch.exp(logits)
        denominator = exponentials.sum(dim=-1, keepdim=True)
        denominator = torch.where(
            denominator.abs() < 1.0e-6,
            torch.ones_like(denominator),
            denominator,
        )
        normalized = exponentials / denominator
        return (
            2.0
            * torch.sum(
                _as_complex(self.readout).unsqueeze(-1) * normalized,
                dim=1,
            ).real
        )

    def forward(self, inputs: Tensor) -> Tensor:
        convolved = _causal_depthwise_convolution(inputs, self._kernel(inputs.shape[1]))
        values = convolved + self.skip * inputs
        return torch.nn.functional.glu(self.output_projection(torch.nn.functional.gelu(values)))


class _LRULayer(nn.Module):
    """Complex diagonal LRU with stable annulus initialization and gamma normalization."""

    def __init__(self, width: int, state_size: int) -> None:
        super().__init__()
        modes = max(2, state_size)
        radius = torch.empty(modes).uniform_(0.90, 0.999)
        self.log_nu = nn.Parameter(torch.log(-torch.log(radius)))
        self.log_theta = nn.Parameter(torch.log(torch.empty(modes).uniform_(1.0e-3, 2.0 * math.pi)))
        self.input_matrix = _complex_parameter(modes, width, scale=width**-0.5)
        self.output_matrix = _complex_parameter(width, modes, scale=modes**-0.5)
        self.skip = nn.Parameter(torch.ones(width))
        self.output_projection = nn.Linear(width, 2 * width)

    def _kernel(self, length: int) -> Tensor:
        radius = torch.exp(-torch.exp(self.log_nu))
        poles = torch.polar(radius, torch.exp(self.log_theta))
        gamma = torch.sqrt((1.0 - radius.square()).clamp_min(1.0e-6))
        time = torch.arange(length, device=poles.device, dtype=poles.real.dtype)
        powers = poles.unsqueeze(-1) ** time
        input_matrix = gamma[:, None] * _as_complex(self.input_matrix)
        output_matrix = _as_complex(self.output_matrix)
        return 2.0 * torch.einsum("om,mt,mi->oit", output_matrix, powers, input_matrix).real

    def forward(self, inputs: Tensor) -> Tensor:
        convolved = _causal_mimo_convolution(inputs, self._kernel(inputs.shape[1]))
        values = convolved + self.skip * inputs
        return torch.nn.functional.glu(self.output_projection(torch.nn.functional.gelu(values)))


class _S5Layer(nn.Module):
    """Single MIMO diagonal SSM with ZOH discretization and S5-style shared state."""

    def __init__(self, width: int, state_size: int) -> None:
        super().__init__()
        modes = max(2, state_size // 2)
        self.log_dt = nn.Parameter(
            torch.empty(modes).uniform_(math.log(_DT_MIN), math.log(_DT_MAX))
        )
        self.log_decay = nn.Parameter(torch.log(torch.arange(1, modes + 1, dtype=torch.float32)))
        # The increasing imaginary spectrum is the diagonal HiPPO/S4D-Lin initialization
        # used by S5 before task-specific training.
        self.frequency = nn.Parameter(math.pi * torch.arange(modes, dtype=torch.float32))
        self.input_matrix = _complex_parameter(modes, width, scale=width**-0.5)
        self.output_matrix = _complex_parameter(width, modes, scale=modes**-0.5)
        self.skip = nn.Parameter(torch.ones(width))
        self.output_projection = nn.Linear(width, 2 * width)

    def _kernel(self, length: int) -> Tensor:
        poles = torch.complex(-torch.exp(self.log_decay), self.frequency)
        step = torch.exp(self.log_dt)
        discrete = step * poles
        transition = torch.exp(discrete)
        zoh = torch.expm1(discrete) / poles
        time = torch.arange(length, device=poles.device, dtype=poles.real.dtype)
        powers = transition.unsqueeze(-1) ** time
        input_matrix = zoh[:, None] * _as_complex(self.input_matrix)
        output_matrix = _as_complex(self.output_matrix)
        return 2.0 * torch.einsum("om,mt,mi->oit", output_matrix, powers, input_matrix).real

    def forward(self, inputs: Tensor) -> Tensor:
        convolved = _causal_mimo_convolution(inputs, self._kernel(inputs.shape[1]))
        values = convolved + self.skip * inputs
        return torch.nn.functional.glu(self.output_projection(torch.nn.functional.gelu(values)))


class _ResidualSSMBlock(nn.Module):
    def __init__(self, layer: nn.Module, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layer = layer

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.layer(self.norm(inputs))


class DiagonalSSMClassifier(nn.Module):
    """Shared sequence-classification shell for the three locked SSM families."""

    def __init__(
        self,
        family: str,
        width: int,
        output_dim: int,
        *,
        depth: int,
        state_size: int,
        input_dim: int = 1,
    ) -> None:
        super().__init__()
        layers = {"dss": _DSSLayer, "lru": _LRULayer, "s5": _S5Layer}
        if family not in layers:
            message = f"unsupported diagonal SSM family: {family}"
            raise ValueError(message)
        layer_type = layers[family]
        self.input_projection = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            _ResidualSSMBlock(layer_type(width, state_size), width) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, output_dim)
        self.family = family
        self.depth = depth
        self.state_size = state_size

    def forward(
        self,
        inputs: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.input_projection(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.classifier(masked_temporal_mean(self.final_norm(hidden), valid_mask))
