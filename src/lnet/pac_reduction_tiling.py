"""Hardware-derived row grouping for parameter-gradient reductions."""

from __future__ import annotations

import torch
from torch import Tensor


def workload_parameter_reduction_rows(
    rows: int,
    *,
    multiprocessors: int,
    warp_size: int,
) -> int:
    """Group rows against the device's native half-warp reduction parallelism."""
    if min(rows, multiprocessors, warp_size) <= 0:
        message = "reduction workload and hardware properties must be positive"
        raise ValueError(message)
    target_partials = multiprocessors * max(1, warp_size // 2)
    rows_per_partial = (rows + target_partials - 1) // target_partials
    return 1 << max(0, (rows_per_partial - 1).bit_length())


def device_parameter_reduction_rows(
    reference: Tensor,
    rows: int,
) -> int:
    """Derive grouping directly from the CUDA device executing the reduction."""
    if not reference.is_cuda:
        message = "device reduction grouping requires a CUDA tensor"
        raise ValueError(message)
    properties = torch.cuda.get_device_properties(reference.device)
    return workload_parameter_reduction_rows(
        rows,
        multiprocessors=properties.multi_processor_count,
        warp_size=properties.warp_size,
    )


__all__ = [
    "device_parameter_reduction_rows",
    "workload_parameter_reduction_rows",
]
