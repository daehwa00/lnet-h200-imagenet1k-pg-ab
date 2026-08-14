"""Amortized CUDA scalar phase draws shared by PA2WP training runtimes."""

from __future__ import annotations

import torch


class _CapturedScalarPhaseSchedule:
    """Amortize host synchronization while preserving scalar CUDA RNG draws.

    One graph replay performs the same ordered scalar ``torch.rand`` calls as
    the campaign loop and copies their values to pinned host memory with one
    synchronization. Consuming the schedule assumes the PA2WP phase draw is
    the only use of the default CUDA generator between refills.
    """

    def __init__(self, device: torch.device, capacity: int) -> None:
        if capacity < 2:
            message = "phase schedule capacity must be at least two"
            raise ValueError(message)
        self.capacity = capacity
        self._device = device
        self._device_values = torch.empty(capacity, device=device, dtype=torch.float32)
        self._host_values = torch.empty(capacity, device="cpu", dtype=torch.float32).pin_memory()
        generator_state = torch.cuda.get_rng_state(device)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            for index in range(capacity):
                self._device_values[index].copy_(torch.rand((), device=device))
        torch.cuda.set_rng_state(generator_state, device)
        self._index = capacity

    def reset(self) -> None:
        """Discard prefetched values after an external CUDA RNG reset."""
        self._index = self.capacity

    def next_shifted(self) -> bool:
        """Return the next scalar Bernoulli decision in CUDA RNG order."""
        if self._index == self.capacity:
            self._graph.replay()
            self._host_values.copy_(self._device_values, non_blocking=True)
            torch.cuda.current_stream(self._device).synchronize()
            self._index = 0
        value = float(self._host_values[self._index].item())
        self._index += 1
        return value < 0.5


__all__ = ["_CapturedScalarPhaseSchedule"]
