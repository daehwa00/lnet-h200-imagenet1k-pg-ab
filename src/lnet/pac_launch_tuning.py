"""Measure launch candidates in a real step and persist exact-scope winners.

Only kernels exercised by the supplied step are tuned. Compiled callers must
rebuild the graph for each geometry because Inductor's cache key does not
include project launch overrides.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch.profiler import ProfilerActivity, profile

from .pac_kernel_launch_config import (
    LaunchGeometry,
    LaunchScope,
    capture_launch_scopes,
    override_launch_geometry,
    registered_candidates,
    registered_names,
    resolve,
    store_geometry,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Kernel name in the registry -> the CUDA kernel symbol it launches.
_KERNEL_SYMBOLS: dict[str, str] = {
    "bidirectional_product_scan_forward": "_bidirectional_forward_kernel",
    "bidirectional_product_scan_backward": "_bidirectional_backward_kernel",
    "product_scan_coarse4_forward": "_product_scan_coarse4_associative_forward_kernel",
    "product_scan_descriptor4_forward": "_product_scan_coarse4_associative_forward_kernel",
    "product_scan_coarse4_backward": "_product_scan_coarse4_associative_backward_kernel",
    "product_scan_descriptor4_backward": "_product_scan_coarse4_associative_backward_kernel",
    "product_scan_coarse4_static_variance": "_static_separable_variance_kernel",
    "product_scan_coarse4_global_gain": "_static_global_inverse_gain_kernel",
    "product_scan_coarse4_descriptor_finalize": "_finalize_descriptor_kernel",
    "d4_grouped_path_collapse_forward": "_d4_grouped_path_collapse_forward_kernel",
    "d4_grouped_path_collapse_backward": "_d4_grouped_path_collapse_backward_kernel",
    "packed_complex_rmsnorm_forward": "_packed_complex_rmsnorm_forward_kernel",
    "packed_complex_rmsnorm_backward": "_packed_complex_rmsnorm_backward_kernel",
    "packed_complex_rmsnorm_backward_reduce": "_packed_complex_rmsnorm_backward_reduce_kernel",
    "phase_gate_forward": "_phase_gate_forward_kernel",
    "phase_gate_output_linear_backward": "_phase_gate_output_linear_backward_kernel",
    "phase_gate_output_linear_fused_forward": "_fused_forward_kernel",
    "phase_gate_output_linear_fused_backward": "_fused_backward_kernel",
    "phase_gate_output_residual_fused_forward": "_phase_gate_output_residual_forward_kernel",
    "phase_gate_output_residual_fused_backward": "_phase_gate_output_residual_backward_kernel",
    "phase_gate_output_residual_fused_backward_reduce": (
        "_phase_gate_output_residual_backward_reduce_kernel"
    ),
    "phase_gated_cffn_fused_forward": "_phase_gated_cffn_forward_kernel",
    "phase_gated_cffn_fused_backward": "_phase_gated_cffn_backward_kernel",
    "rmsnorm_input_linear_fused_forward": "_fused_forward_kernel",
    "rmsnorm_input_linear_fused_backward": "_fused_backward_kernel",
    "phase_gate_backward_reduce": "_phase_gate_backward_reduce_kernel",
}


class TuningResult(NamedTuple):
    """One kernel's tuning outcome."""

    name: str
    baseline: LaunchGeometry
    winner: LaunchGeometry
    baseline_ms: float
    winner_ms: float
    scopes: tuple[LaunchScope, ...] = ()

    @property
    def speedup(self) -> float:
        return self.baseline_ms / self.winner_ms if self.winner_ms > 0.0 else 1.0


def kernel_time_ms(step: Callable[[], None], symbol: str, *, iterations: int = 3) -> float:
    """Total CUDA time of one kernel symbol across a step, in milliseconds."""
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as session:
        for _ in range(iterations):
            step()
        torch.cuda.synchronize()
    total = 0.0
    for event in session.key_averages():
        if event.key == symbol and event.device_time_total > 0:
            total += event.device_time_total / iterations / 1000.0
    return total


def _fastest_geometry(
    name: str,
    baseline: LaunchGeometry,
    timed: Callable[[LaunchGeometry], float],
) -> tuple[LaunchGeometry, float, float] | None:
    baseline_ms = timed(baseline)
    if baseline_ms <= 0.0:
        return None
    best_geometry, best_ms = baseline, baseline_ms
    for candidate in registered_candidates(name):
        if candidate == baseline:
            continue
        elapsed = timed(candidate)
        if 0.0 < elapsed < best_ms:
            best_geometry, best_ms = candidate, elapsed
    return best_geometry, baseline_ms, best_ms


def tune_kernel(  # noqa: C901
    name: str,
    step: Callable[[], None],
    *,
    iterations: int = 3,
    store: bool = True,
    on_geometry_change: Callable[[], None] | None = None,
) -> list[TuningResult]:
    """Tune every exact scope exercised by one kernel in a real step.

    Each candidate override applies to one scope only. Other shapes keep their
    own resolved geometry, so a 14x14 winner cannot be copied into 56x56 merely
    because both launches share a CUDA symbol.
    """
    symbol = _KERNEL_SYMBOLS.get(name)
    if symbol is None:
        return []

    if on_geometry_change is not None:
        on_geometry_change()
    with capture_launch_scopes(name) as observed_scopes:
        discovery_ms = kernel_time_ms(step, symbol, iterations=1)
    if discovery_ms <= 0.0:
        return []
    scopes = tuple(sorted(observed_scopes, key=lambda scope: scope.key))

    if scopes:
        results: list[TuningResult] = []
        for scope in scopes:
            baseline = resolve(name, scope=scope)

            def timed_scope(
                geometry: LaunchGeometry,
                active_scope: LaunchScope = scope,
            ) -> float:
                with override_launch_geometry(name, active_scope, geometry):
                    if on_geometry_change is not None:
                        on_geometry_change()
                    return kernel_time_ms(step, symbol, iterations=iterations)

            fastest = _fastest_geometry(name, baseline, timed_scope)
            if fastest is None:
                continue
            winner, baseline_ms, winner_ms = fastest
            if store:
                store_geometry(name, winner, scope=scope)
            results.append(
                TuningResult(
                    name,
                    baseline,
                    winner,
                    baseline_ms,
                    winner_ms,
                    (scope,),
                )
            )
        return results

    def timed_unscoped(geometry: LaunchGeometry) -> float:
        prefix = f"LNET_LAUNCH_{name.upper()}"
        previous = {key: os.environ.get(key) for key in (f"{prefix}_WARPS", f"{prefix}_STAGES")}
        previous.update(
            {
                f"{prefix}_{block.upper()}": os.environ.get(f"{prefix}_{block.upper()}")
                for block in geometry.blocks
            }
        )
        os.environ[f"{prefix}_WARPS"] = str(geometry.num_warps)
        os.environ[f"{prefix}_STAGES"] = str(geometry.num_stages)
        for block, value in geometry.blocks.items():
            os.environ[f"{prefix}_{block.upper()}"] = str(value)
        try:
            # A compiled step bakes the geometry into its graph, so the caller
            # must tear that graph down before the next candidate is timed.
            if on_geometry_change is not None:
                on_geometry_change()
            return kernel_time_ms(step, symbol, iterations=iterations)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    baseline = resolve(name)
    fastest = _fastest_geometry(name, baseline, timed_unscoped)
    if fastest is None:
        return []
    winner, baseline_ms, winner_ms = fastest
    if store:
        # Preserve the unscoped API for external/custom kernels that have not
        # adopted ``make_launch_scope`` yet.
        store_geometry(name, winner)
    return [TuningResult(name, baseline, winner, baseline_ms, winner_ms)]


def tune_all(
    step: Callable[[], None],
    *,
    names: Sequence[str] | None = None,
    iterations: int = 3,
    store: bool = True,
    on_geometry_change: Callable[[], None] | None = None,
) -> list[TuningResult]:
    """Tune every registered kernel the supplied step actually exercises.

    ``on_geometry_change`` runs before each candidate is timed.  A compiled step
    needs it to drop the previous graph; an eager step can leave it unset.
    """
    if not torch.cuda.is_available():
        message = "launch-geometry tuning requires CUDA"
        raise RuntimeError(message)
    selected = tuple(names) if names is not None else registered_names()
    results: list[TuningResult] = []
    for name in selected:
        outcomes = tune_kernel(
            name,
            step,
            iterations=iterations,
            store=store,
            on_geometry_change=on_geometry_change,
        )
        results.extend(outcomes)
    return results
