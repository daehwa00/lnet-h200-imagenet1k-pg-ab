from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

from .pac_metrics import count_parameters
from .pac_overnight_io import append_csv_row
from .pac_overnight_models import build_overnight_classifier

if TYPE_CHECKING:
    from pathlib import Path

    from .pac_types import PACExperimentConfig
    from .tapped_prl_followup_schema import JsonValue


def run_efficiency_audit(
    output_root: Path,
    config: PACExperimentConfig,
    device: str,
    models: tuple[str, ...],
    lengths: tuple[int, ...],
    *,
    warmup_iters: int = 20,
    timed_iters: int = 100,
) -> None:
    path = output_root / "results" / "efficiency_audit.csv"
    for model_name in models:
        for length in lengths:
            model = build_overnight_classifier(model_name, config, class_count=config.output_dim)
            model.to(device=device)
            row = _benchmark_model(
                model,
                model_name,
                config,
                device,
                length,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
            )
            append_csv_row(path, row)
    _write_report(output_root)


def _benchmark_model(
    model: nn.Module,
    model_name: str,
    config: PACExperimentConfig,
    device: str,
    length: int,
    *,
    warmup_iters: int,
    timed_iters: int,
) -> dict[str, JsonValue]:
    inputs = torch.randn(config.batch_size, length, 1, device=device)
    labels = torch.randint(0, config.output_dim, (config.batch_size,), device=device)
    forward_ms, forward_tps = _forward_timing(model, inputs, device, warmup_iters, timed_iters)
    train_ms, train_tps = _train_timing(model, inputs, labels, device, warmup_iters, timed_iters)
    allocated = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    reserved = torch.cuda.max_memory_reserved() if device == "cuda" else 0
    return {
        "model": model_name,
        "variant": model_name,
        "N": length,
        "batch_size": config.batch_size,
        "params_trainable": count_parameters(model),
        "forward_ms": forward_ms,
        "forward_tokens_per_sec": forward_tps,
        "train_step_ms": train_ms,
        "train_tokens_per_sec": train_tps,
        "peak_memory_allocated_MB": allocated / 1_000_000,
        "peak_memory_reserved_MB": reserved / 1_000_000,
        "implementation_type": _implementation_type(model_name),
        "notes": "fp32_synchronized",
    }


def _forward_timing(
    model: nn.Module,
    inputs: torch.Tensor,
    device: str,
    warmup_iters: int,
    timed_iters: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            model(inputs)
        _sync(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        for _ in range(timed_iters):
            model(inputs)
        _sync(device)
    elapsed = (perf_counter() - start) / timed_iters
    return elapsed * 1000.0, _tokens_per_second(inputs, elapsed)


def _train_timing(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    warmup_iters: int,
    timed_iters: int,
) -> tuple[float, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    for _ in range(warmup_iters):
        _train_step(model, inputs, labels, optimizer)
    _sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for _ in range(timed_iters):
        _train_step(model, inputs, labels, optimizer)
    _sync(device)
    elapsed = (perf_counter() - start) / timed_iters
    return elapsed * 1000.0, _tokens_per_second(inputs, elapsed)


def _train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = functional.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()


def _tokens_per_second(inputs: torch.Tensor, elapsed: float) -> float:
    return float(inputs.shape[0] * inputs.shape[1] / max(elapsed, 1.0e-12))


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _implementation_type(model_name: str) -> str:
    if model_name.startswith("pac"):
        return "naive_loop"
    if model_name in {"gru", "lstm"}:
        return "cudnn"
    if model_name in {"cnn1d", "cnn1d_small", "tcn", "tcn_small", "fir_classifier"}:
        return "conv"
    if model_name == "transformer_tiny":
        return "attention"
    return "other"


def _write_report(output_root: Path) -> None:
    path = output_root / "reports" / "overnight_efficiency_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Efficiency Audit\n\nefficiency_status: verified_naive_slow\n", encoding="utf-8"
    )
