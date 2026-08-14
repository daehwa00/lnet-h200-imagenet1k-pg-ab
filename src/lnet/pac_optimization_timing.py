from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_metrics import count_parameters

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig, PACModelName
    from .tapped_prl_followup_schema import JsonRow

BenchmarkVariant = Literal[
    "reference_naive",
    "optimized",
    "complex_loop",
    "real2d_loop",
    "compiled_real2d",
    "triton_fused",
    "triton_scan",
    "real2d_e2e",
    "triton_scan_blocks",
    "triton_modal_fused",
    "triton_modal_reduce",
    "triton_modal_reduce_recompute",
    "pac_lite_fast",
    "fixed_real2d_fast",
    "fused_pole_gamma",
    "pac_lite_prl_fused",
    "pac_lite_block_fused",
    "auto",
]


def speed_row(
    model: nn.Module,
    config: PACExperimentConfig,
    device: str,
    model_name: PACModelName | str,
    variant: BenchmarkVariant | str,
    length: int,
    warmup_iters: int,
    timed_iters: int,
) -> JsonRow:
    inputs = torch.randn(config.batch_size, length, config.raw_input_dim, device=device)
    targets = torch.randn(config.batch_size, length, config.output_dim, device=device)
    forward_ms, forward_tps = _forward_timing(model, inputs, device, warmup_iters, timed_iters)
    train_ms, train_tps = _train_timing(model, inputs, targets, device, warmup_iters, timed_iters)
    return {
        "model": model_name,
        "variant": variant,
        "N": length,
        "batch_size": config.batch_size,
        "params_trainable": count_parameters(model),
        "forward_ms": forward_ms,
        "forward_tokens_per_sec": forward_tps,
        "train_step_ms": train_ms,
        "train_tokens_per_sec": train_tps,
        "implementation_type": variant,
    }


def _forward_timing(
    model: nn.Module,
    inputs: Tensor,
    device: str,
    warmup_iters: int,
    timed_iters: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            model(inputs)
        _sync(device)
        start = perf_counter()
        for _ in range(timed_iters):
            model(inputs)
        _sync(device)
    elapsed = (perf_counter() - start) / timed_iters
    return elapsed * 1000.0, _tokens_per_second(inputs, elapsed)


def _train_timing(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    device: str,
    warmup_iters: int,
    timed_iters: int,
) -> tuple[float, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    for _ in range(warmup_iters):
        _train_step(model, inputs, targets, optimizer)
    _sync(device)
    start = perf_counter()
    for _ in range(timed_iters):
        _train_step(model, inputs, targets, optimizer)
    _sync(device)
    elapsed = (perf_counter() - start) / timed_iters
    return elapsed * 1000.0, _tokens_per_second(inputs, elapsed)


def _train_step(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()


def _tokens_per_second(inputs: Tensor, elapsed: float) -> float:
    return float(inputs.shape[0] * inputs.shape[1] / max(elapsed, 1.0e-12))


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
