"""Small heads for an explicit Fourier-morphology side branch."""

from __future__ import annotations

import math

import numpy as np
from torch import Tensor, nn

from lnet.astronomy.fourier_shape import FourierShape, is_reliable_fourier_shape


def harmonic_feature_vector(
    shape: FourierShape | None,
    *,
    phase_coupling: bool,
) -> np.ndarray:
    """Encode window-corrected harmonic amplitudes and optional phase coupling."""
    reliable = is_reliable_fourier_shape(shape)
    if shape is None:
        amplitude = np.zeros(4, dtype=np.float64)
        quality = np.zeros(3, dtype=np.float64)
        phase = np.zeros(4, dtype=np.float64)
    else:
        amplitude = np.asarray(
            [
                math.log(shape.period_days),
                math.log(max(shape.fundamental_amplitude, 1.0e-6)),
                float(np.clip(math.log(shape.r21), -8.0, 8.0)),
                float(np.clip(math.log(shape.r31), -8.0, 8.0)),
            ],
            dtype=np.float64,
        )
        quality = np.asarray(
            [
                shape.explained_variance,
                math.log1p(shape.observation_count),
                float(reliable),
            ],
            dtype=np.float64,
        )
        phase = np.asarray(
            [
                math.cos(shape.phi21),
                math.sin(shape.phi21),
                math.cos(shape.phi31),
                math.sin(shape.phi31),
            ],
            dtype=np.float64,
        )
        if not reliable:
            phase *= 0.0
    return np.concatenate((amplitude, quality, phase if phase_coupling else []))


class HarmonicFeatureHead(nn.Module):
    """Low-capacity classifier over explicit harmonic morphology coordinates."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 2 or hidden_dim < 1:
            message = "feature, hidden, and output dimensions must be positive"
            raise ValueError(message)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def fuse_logits(
    alphabet_logits: Tensor,
    harmonic_logits: Tensor,
    beta: float,
) -> Tensor:
    """Combine frozen ALPHABET and harmonic logits with a scalar validation weight."""
    if alphabet_logits.shape != harmonic_logits.shape:
        message = "alphabet and harmonic logits must have identical shapes"
        raise ValueError(message)
    if beta < 0.0:
        message = "fusion beta must be nonnegative"
        raise ValueError(message)
    return alphabet_logits + beta * harmonic_logits
