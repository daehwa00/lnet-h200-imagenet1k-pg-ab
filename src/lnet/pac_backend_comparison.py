from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, assert_never

import torch
from torch import Tensor, nn

from .pac_model import PACHybridPRLBlock
from .pac_optimization_timing import BenchmarkVariant, speed_row
from .pac_tasks import make_pac_synthetic_tasks
from .pac_training import train_regression_model

if TYPE_CHECKING:
    from .pac_hybrid_backend import HybridBackend
    from .pac_recurrence import RecurrenceBackend
    from .pac_types import PACExperimentConfig, PACModelName
    from .tapped_prl_followup_schema import JsonRow

type BackendRows = dict[str, list[JsonRow]]
ModelPair = tuple[str, nn.Module]


def backend_comparison_rows(config: PACExperimentConfig, device: str) -> BackendRows:
    rows = {
        "backend_correctness.csv": _correctness_rows(config, device),
        "backend_gradient_equivalence.csv": _gradient_rows(config, device),
        "backend_speed_cuda.csv": _speed_rows(config, device),
        "backend_predictive_equivalence.csv": _predictive_rows(config, device),
    }
    advanced = {f"advanced_{name}": values for name, values in rows.items()}
    modal_reduce = {
        f"modal_reduce_{name.removeprefix('backend_')}": values for name, values in rows.items()
    }
    return rows | advanced | modal_reduce


def _correctness_rows(config: PACExperimentConfig, device: str) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for model_name in _pac_models():
        torch.manual_seed(3101)
        reference = _build_model(model_name, config, "complex_loop").to(device=device)
        inputs = torch.randn(
            4,
            min(config.sequence_length, 32),
            config.raw_input_dim,
            device=device,
        )
        with torch.no_grad():
            reference_output = reference(inputs)
        for backend, model in _matching_models(model_name, config, reference, device):
            with torch.no_grad():
                output = model(inputs)
            diff = (output - reference_output).abs()
            rows.append(
                {
                    "model": model_name,
                    "backend": backend,
                    "max_abs_diff_vs_complex_loop": float(diff.max().item()),
                    "mean_abs_diff_vs_complex_loop": float(diff.mean().item()),
                    "allclose_2e_4": torch.allclose(
                        output, reference_output, atol=2.0e-4, rtol=2.0e-4
                    ),
                }
            )
    return rows


def _gradient_rows(config: PACExperimentConfig, device: str) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for model_name in _pac_models():
        torch.manual_seed(3201)
        reference = _build_model(model_name, config, "complex_loop").to(device=device)
        inputs = torch.randn(
            2,
            min(config.sequence_length, 16),
            config.raw_input_dim,
            device=device,
        )
        reference_grads = _gradient_signature(reference, inputs)
        for backend, model in _matching_models(model_name, config, reference, device):
            grads = _gradient_signature(model, inputs)
            max_abs, max_rel = _gradient_error(reference_grads, grads)
            rows.append(
                {
                    "model": model_name,
                    "backend": backend,
                    "max_abs_grad_diff": max_abs,
                    "max_rel_grad_diff": max_rel,
                    "within_5e_3": max_rel <= 5.0e-3 or max_abs <= 5.0e-5,
                }
            )
    return rows


def _speed_rows(config: PACExperimentConfig, device: str) -> list[JsonRow]:
    if device != "cuda":
        return []
    lengths = (128, 512) if _is_smoke(config) else (128, 512, 2048, 4096)
    timed_iters = 1 if _is_smoke(config) else 3
    rows: list[JsonRow] = []
    for model_name in _pac_models():
        for length in lengths:
            baseline = None
            for backend in _speed_backends(device, config):
                model = _build_model(model_name, config, backend).to(device=device)
                row = speed_row(model, config, device, model_name, backend, length, 1, timed_iters)
                if backend == "complex_loop":
                    baseline = _float_row_value(row, "train_tokens_per_sec")
                elif baseline is not None:
                    row["train_speedup_vs_complex_loop"] = _float_row_value(
                        row, "train_tokens_per_sec"
                    ) / max(baseline, 1.0e-12)
                rows.append(row)
    return rows


def _predictive_rows(config: PACExperimentConfig, device: str) -> list[JsonRow]:
    task_config = replace(config, sample_count=32, validation_count=16, test_count=16, epochs=1)
    rows: list[JsonRow] = []
    tasks = [
        task for task in make_pac_synthetic_tasks(task_config, 7) if task.label == "modal_teacher"
    ]
    for task in tasks:
        for model_name in _pac_models():
            for backend in _predictive_backends(device):
                torch.manual_seed(3301)
                model = _build_model(model_name, task_config, backend)
                outcome = train_regression_model(model, task, task_config, device, 7)
                rows.append(
                    {
                        "task": task.label,
                        "model": model_name,
                        "backend": backend,
                        "validation_loss": outcome.validation_loss,
                        "test_loss": outcome.test_loss,
                        "elapsed_time": outcome.elapsed_time,
                    }
                )
    return rows


def _matching_models(
    model_name: PACModelName,
    config: PACExperimentConfig,
    reference: nn.Module,
    device: str,
) -> list[ModelPair]:
    models: list[ModelPair] = []
    state = reference.state_dict()
    for backend in _backends(device):
        model = _build_model(model_name, config, backend).to(device=device)
        model.load_state_dict(state)
        models.append((backend, model))
    return models


def _build_model(
    model_name: PACModelName,
    config: PACExperimentConfig,
    backend: BenchmarkVariant,
) -> PACHybridPRLBlock:
    use_mlp = model_name == "pac_full"
    hybrid_backend = _hybrid_backend(backend)
    return PACHybridPRLBlock(
        raw_input_dim=config.raw_input_dim,
        model_dim=config.model_dim,
        output_dim=config.output_dim,
        modes=config.modes,
        tap_kernel_size=config.tap_kernel_size,
        fir_kernel_size=config.fir_kernel_size,
        use_mlp_branch=use_mlp,
        active_branches=("prl", "fir", "mlp") if use_mlp else ("prl", "fir"),
        recurrence_backend=_recurrence_backend(backend),
        hybrid_backend=hybrid_backend,
    )


def _gradient_signature(model: nn.Module, inputs: Tensor) -> dict[str, Tensor]:
    model.zero_grad(set_to_none=True)
    copied = inputs.detach().clone().requires_grad_()
    model(copied).square().mean().backward()
    grads: dict[str, Tensor] = {}
    if copied.grad is not None:
        grads["inputs"] = copied.grad.detach().cpu()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grads[name] = parameter.grad.detach().cpu()
    return grads


def _gradient_error(
    reference: dict[str, Tensor], candidate: dict[str, Tensor]
) -> tuple[float, float]:
    max_abs = 0.0
    max_rel = 0.0
    for name, ref_grad in reference.items():
        cand_grad = candidate[name]
        diff = (cand_grad - ref_grad).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        scale = ref_grad.abs().clamp_min(1.0e-6)
        max_rel = max(max_rel, float((diff / scale).max().item()))
    return max_abs, max_rel


def _backends(device: str) -> tuple[BenchmarkVariant, ...]:
    if device == "cuda":
        return (
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
        )
    return (
        "complex_loop",
        "real2d_loop",
        "compiled_real2d",
        "triton_scan",
        "real2d_e2e",
        "fixed_real2d_fast",
        "fused_pole_gamma",
        "pac_lite_prl_fused",
        "pac_lite_block_fused",
        "auto",
    )


def _speed_backends(device: str, config: PACExperimentConfig) -> tuple[BenchmarkVariant, ...]:
    if device != "cuda":
        return _backends(device)
    if _is_smoke(config):
        return (
            "complex_loop",
            "real2d_loop",
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
        )
    return _backends(device)


def _predictive_backends(device: str) -> tuple[BenchmarkVariant, ...]:
    if device == "cuda":
        return (
            "complex_loop",
            "triton_fused",
            "real2d_e2e",
            "triton_modal_fused",
            "triton_modal_reduce",
            "triton_modal_reduce_recompute",
            "fixed_real2d_fast",
            "fused_pole_gamma",
            "pac_lite_prl_fused",
            "pac_lite_block_fused",
            "auto",
        )
    return "complex_loop", "real2d_loop", "fixed_real2d_fast", "fused_pole_gamma", "auto"


def _is_smoke(config: PACExperimentConfig) -> bool:
    return config.sample_count <= 32


def _pac_models() -> tuple[PACModelName, ...]:
    return "pac_lite", "pac_full"


def _hybrid_backend(backend: BenchmarkVariant) -> HybridBackend:
    match backend:
        case "pac_lite_prl_fused" | "pac_lite_block_fused":
            return backend
        case (
            "reference_naive"
            | "optimized"
            | "complex_loop"
            | "real2d_loop"
            | "compiled_real2d"
            | "triton_fused"
            | "triton_scan"
            | "real2d_e2e"
            | "triton_scan_blocks"
            | "triton_modal_fused"
            | "triton_modal_reduce"
            | "triton_modal_reduce_recompute"
            | "pac_lite_fast"
            | "fixed_real2d_fast"
            | "fused_pole_gamma"
            | "auto"
        ):
            return "generic"
        case unreachable:
            assert_never(unreachable)


def _recurrence_backend(backend: BenchmarkVariant) -> RecurrenceBackend:
    match backend:
        case "reference_naive" | "optimized" | "pac_lite_prl_fused" | "pac_lite_block_fused":
            return "auto"
        case (
            "complex_loop"
            | "real2d_loop"
            | "compiled_real2d"
            | "triton_fused"
            | "triton_scan"
            | "real2d_e2e"
            | "triton_scan_blocks"
            | "triton_modal_fused"
            | "triton_modal_reduce"
            | "triton_modal_reduce_recompute"
            | "pac_lite_fast"
            | "fixed_real2d_fast"
            | "fused_pole_gamma"
            | "auto"
        ):
            return backend
        case unreachable:
            assert_never(unreachable)


def _float_row_value(row: JsonRow, key: str) -> float:
    value = row[key]
    if isinstance(value, int | float | str):
        return float(value)
    message = f"{key} must be numeric"
    raise TypeError(message)
