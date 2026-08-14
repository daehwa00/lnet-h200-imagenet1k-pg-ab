"""Small hardware-derived contracts shared by Phase-Gated Triton kernels."""

from __future__ import annotations

import torch
from torch import Tensor


def single_warp_dot_tiles_supported(
    widths: tuple[int, ...],
    *,
    warp_size: int,
) -> bool:
    """Return whether every complete dot axis fits one device warp-width tile."""
    supported = warp_size > 0 and len(widths) > 0
    for width in widths:
        supported = supported and 0 < width <= warp_size
    return supported


def device_supports_single_warp_dot_tiles(reference: Tensor, *widths: int) -> bool:
    """Query the current CUDA device instead of assuming an NVIDIA warp width."""
    if not reference.is_cuda:
        return False
    properties = torch.cuda.get_device_properties(reference.device)
    return single_warp_dot_tiles_supported(tuple(widths), warp_size=properties.warp_size)


def diagnostic_sample_rows(reference: Tensor) -> int:
    """Use one square warp tile for bounded CUDA diagnostics, or all CPU rows."""
    rows = reference.numel() // reference.shape[-1]
    if not reference.is_cuda:
        return rows
    warp_size = torch.cuda.get_device_properties(reference.device).warp_size
    return min(rows, warp_size * warp_size)


__all__ = [
    "device_supports_single_warp_dot_tiles",
    "diagnostic_sample_rows",
    "single_warp_dot_tiles_supported",
]
