"""Global finite-grid gain normalization for complex product-pole states.

The normalization in this module is deterministic: it derives a finite-grid
operator gain from pole variance tables and never introduces learned scale or
batch statistics.  In contrast to pointwise variance normalization, a single
gain is shared across the spatial grid for each direction and mode.
"""

# The validation messages are part of this small public numerical API; keeping
# them adjacent to their shape checks is clearer than introducing exception
# subclasses solely to satisfy TRY003/EM101.
# ruff: noqa: EM101, TRY003

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

DEFAULT_GAIN_EPSILON: Final = 1.0e-8


def factorized_product_mean_variance(variance_x: Tensor, variance_y: Tensor) -> Tensor:
    """Return finite-grid mean gains for the four separable product directions.

    Args:
        variance_x: Axis gain table shaped ``[..., 2, W, P]``.  Direction zero
            is the positive scan and direction one is the negative scan.
        variance_y: Axis gain table shaped ``[..., 2, H, P]`` with the same
            direction convention.

    Returns:
        A tensor shaped ``[..., 4, P]`` in the product-path order
        ``(+x,+y), (-x,+y), (+x,-y), (-x,-y)``.

    The separable product variance is ``V_q(x, y) = V_x(x) V_y(y)``.  Its
    spatial mean therefore factorizes exactly, avoiding an ``H x W`` tensor.
    """
    if variance_x.ndim < 3 or variance_y.ndim < 3:
        raise ValueError("axis variance tables must have shape [..., 2, length, modes]")
    if variance_x.shape[-3] != 2 or variance_y.shape[-3] != 2:
        raise ValueError("axis variance tables must contain positive and negative directions")
    if variance_x.shape[-1] != variance_y.shape[-1]:
        raise ValueError("x and y variance tables must have the same mode count")
    if variance_x.shape[:-3] != variance_y.shape[:-3]:
        raise ValueError("x and y variance tables must have matching leading dimensions")

    mean_x = variance_x.float().mean(dim=-2)
    mean_y = variance_y.float().mean(dim=-2)
    return torch.stack(
        (
            mean_x[..., 0, :] * mean_y[..., 0, :],
            mean_x[..., 1, :] * mean_y[..., 0, :],
            mean_x[..., 0, :] * mean_y[..., 1, :],
            mean_x[..., 1, :] * mean_y[..., 1, :],
        ),
        dim=-2,
    )


class GlobalFiniteGridGainNorm(nn.Module):
    """Normalize complex states with one finite-grid gain per direction/mode.

    ``mean_variance`` contains analytic operator gains, not empirical batch
    variance.  By default it is detached so the pole response remains the only
    gradient path while the forward scale follows the current learned poles.

    The default spatial layout is ``[B, H, W, ..., P]``.  Alternate layouts can
    specify their spatial dimensions at construction time.
    """

    def __init__(
        self,
        *,
        epsilon: float = DEFAULT_GAIN_EPSILON,
        spatial_dims: tuple[int, ...] = (1, 2),
        detach_gain: bool = True,
        log1p_energy: bool = True,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not spatial_dims:
            raise ValueError("at least one spatial dimension is required")
        if len(set(spatial_dims)) != len(spatial_dims):
            raise ValueError("spatial dimensions must be unique")
        self.epsilon = float(epsilon)
        self.spatial_dims = spatial_dims
        self.detach_gain = detach_gain
        self.log1p_energy = log1p_energy

    def _canonical_spatial_dims(self, ndim: int) -> tuple[int, ...]:
        dims = tuple(dim if dim >= 0 else ndim + dim for dim in self.spatial_dims)
        if any(dim < 0 or dim >= ndim for dim in dims):
            raise ValueError("a spatial dimension is outside the state rank")
        return tuple(sorted(dims))

    def _checked_inputs(
        self,
        real: Tensor,
        imag: Tensor,
        mean_variance: Tensor,
    ) -> tuple[tuple[int, ...], Tensor]:
        if real.shape != imag.shape:
            raise ValueError("real and imaginary states must have matching shapes")
        spatial_dims = self._canonical_spatial_dims(real.ndim)
        non_spatial_rank = real.ndim - len(spatial_dims)
        if mean_variance.ndim > non_spatial_rank:
            raise ValueError("mean variance has too many non-spatial dimensions")

        padded_gain_shape = (1,) * (non_spatial_rank - mean_variance.ndim) + tuple(
            mean_variance.shape
        )
        state_non_spatial_shape = tuple(
            size for dim, size in enumerate(real.shape) if dim not in spatial_dims
        )
        for gain_size, state_size in zip(padded_gain_shape, state_non_spatial_shape, strict=True):
            if gain_size not in {1, state_size}:
                raise ValueError("mean variance is not broadcastable over the state")

        broadcast_shape: list[int] = []
        gain_index = 0
        for dim in range(real.ndim):
            if dim in spatial_dims:
                broadcast_shape.append(1)
            else:
                broadcast_shape.append(padded_gain_shape[gain_index])
                gain_index += 1
        gain = mean_variance.float()
        if self.detach_gain:
            gain = gain.detach()
        return spatial_dims, gain.reshape(broadcast_shape)

    def normalize(
        self,
        real: Tensor,
        imag: Tensor,
        mean_variance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply one inverse-RMS gain to every spatial position in each mode."""
        _, gain = self._checked_inputs(real, imag, mean_variance)
        inverse_gain = torch.rsqrt(gain.clamp_min(self.epsilon))
        return (
            real * inverse_gain.to(dtype=real.dtype),
            imag * inverse_gain.to(dtype=imag.dtype),
        )

    def energy(
        self,
        real: Tensor,
        imag: Tensor,
        mean_variance: Tensor,
        *,
        log1p: bool | None = None,
    ) -> Tensor:
        """Return ``mean(|Z|^2) / mean(V)``, optionally radial-log compressed."""
        spatial_dims, broadcast_gain = self._checked_inputs(real, imag, mean_variance)
        energy = real.float().square().add(imag.float().square()).mean(dim=spatial_dims)

        # Remove the singleton spatial axes inserted by `_checked_inputs` while
        # retaining batch, direction, and mode axes for ordinary broadcasting.
        gain = broadcast_gain
        for dim in reversed(spatial_dims):
            gain = gain.squeeze(dim)
        normalized_energy = energy / gain.clamp_min(self.epsilon)
        use_log1p = self.log1p_energy if log1p is None else log1p
        return torch.log1p(normalized_energy) if use_log1p else normalized_energy

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        mean_variance: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return normalized real/imaginary states and their global energy Q."""
        normalized_real, normalized_imag = self.normalize(real, imag, mean_variance)
        q = self.energy(real, imag, mean_variance)
        return normalized_real, normalized_imag, q


__all__ = [
    "DEFAULT_GAIN_EPSILON",
    "GlobalFiniteGridGainNorm",
    "factorized_product_mean_variance",
]
