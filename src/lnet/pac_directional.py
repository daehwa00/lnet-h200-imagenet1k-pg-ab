"""Direction-aware tensor selection shared by two-dimensional scans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def direction_aligned_endpoints(
    real: Tensor,
    imag: Tensor,
    *,
    direction_x: int,
    direction_y: int,
    stride: int = 2,
) -> tuple[Tensor, Tensor]:
    """Select exact block endpoints from a directional full-resolution state."""
    if direction_x not in {-1, 1} or direction_y not in {-1, 1}:
        message = "directions must be -1 or 1"
        raise ValueError(message)
    if stride < 1 or real.shape[1] % stride or real.shape[2] % stride:
        message = "spatial dimensions must be divisible by the positive stride"
        raise ValueError(message)
    start_x = stride - 1 if direction_x == 1 else 0
    start_y = stride - 1 if direction_y == 1 else 0
    return (
        real[:, start_y::stride, start_x::stride],
        imag[:, start_y::stride, start_x::stride],
    )


def direction_aligned_cells(
    real: Tensor,
    imag: Tensor,
    *,
    direction_x: int,
    direction_y: int,
) -> tuple[Tensor, Tensor]:
    """Gather every 2x2 state in endpoint-relative local order.

    The local axis is ``(q00, q10, q01, q11)``: ``q11`` is the legacy
    direction-aligned endpoint, ``q10`` is its x predecessor, and ``q01`` is
    its y predecessor.  Consequently local index three exactly recovers
    :func:`direction_aligned_endpoints` for every D4 direction.
    """
    if direction_x not in {-1, 1} or direction_y not in {-1, 1}:
        message = "directions must be -1 or 1"
        raise ValueError(message)
    if real.shape != imag.shape or real.ndim != 4:
        message = "direction-aligned cells require matching NHWM tensors"
        raise ValueError(message)
    if real.shape[1] % 2 or real.shape[2] % 2:
        message = "direction-aligned cells require even spatial dimensions"
        raise ValueError(message)

    batch, height, width, modes = real.shape
    physical_real = real.reshape(batch, height // 2, 2, width // 2, 2, modes)
    physical_imag = imag.reshape(batch, height // 2, 2, width // 2, 2, modes)
    predecessor_x, endpoint_x = ((0, 1) if direction_x == 1 else (1, 0))
    predecessor_y, endpoint_y = ((0, 1) if direction_y == 1 else (1, 0))

    def gather(values: Tensor) -> Tensor:
        return torch.stack(
            (
                values[:, :, predecessor_y, :, predecessor_x],
                values[:, :, endpoint_y, :, predecessor_x],
                values[:, :, predecessor_y, :, endpoint_x],
                values[:, :, endpoint_y, :, endpoint_x],
            ),
            dim=-2,
        )

    return gather(physical_real), gather(physical_imag)


__all__ = ["direction_aligned_cells", "direction_aligned_endpoints"]
