from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, assert_never

import torch
from torch import Tensor

from .hybrid_experiment_types import resolve_device
from .pac_backend_comparison import backend_comparison_rows
from .pac_model import PACHybridPRLBlock
from .pac_optimization_timing import speed_row
from .pac_overnight_io import write_csv_rows
from .pac_paper_queue_models import build_paper_regressor
from .pac_tasks import make_pac_synthetic_tasks
from .pac_training import train_regression_model
from .pac_types import PACDevice, PACExperimentConfig, PACModelName

if TYPE_CHECKING:
    from .pac_hybrid_backend import HybridBackend
    from .pac_recurrence import RecurrenceBackend
    from .tapped_prl_followup_schema import JsonRow

OptimizationMode = Literal["smoke", "full"]
BenchmarkVariant = Literal[
    "reference_naive",
    "optimized",
    "complex_loop",
    "real2d_loop",
    "compiled_real2d",
    "triton_fused",
    "triton_scan",
    "pac_lite_prl_fused",
    "pac_lite_block_fused",
    "auto",
]
DEFAULT_OUTPUT_DIR: Final = Path(".omx/results/pac-hybrid-prl/optimization-20260706")
PAC_MODELS: Final[tuple[Literal["pac_lite", "pac_full"], ...]] = ("pac_lite", "pac_full")
SPEED_BASELINES: Final = ("gru", "lstm", "tcn", "cnn1d", "transformer_tiny")
PREDICTIVE_TASKS: Final = frozenset(
    {"modal_teacher", "delayed_oscillatory", "active_damping_teacher", "random_fir_teacher"}
)
BACKEND_BY_VARIANT: Final[dict[BenchmarkVariant, RecurrenceBackend]] = {
    "reference_naive": "auto",
    "optimized": "auto",
    "complex_loop": "complex_loop",
    "real2d_loop": "real2d_loop",
    "compiled_real2d": "compiled_real2d",
    "triton_fused": "triton_fused",
    "triton_scan": "triton_scan",
    "pac_lite_prl_fused": "auto",
    "pac_lite_block_fused": "auto",
    "auto": "auto",
}


class ReferencePACHybridPRLBlock(PACHybridPRLBlock):
    def _prl_output(self, projected: Tensor) -> Tensor:
        if self.prl_branch is None:
            message = "PRL branch is not initialized"
            raise RuntimeError(message)
        return self.prl_branch.forward_reference(projected)


def run_optimization(
    mode: OptimizationMode,
    device: PACDevice,
    output_dir: Path,
    *,
    compare_backends: bool = False,
) -> Path:
    resolved = resolve_device(device)
    config = _config(mode, device, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    correctness = _correctness_rows(config, resolved)
    speed = _speed_rows(config, resolved, mode)
    predictive = _predictive_rows(config, resolved, mode)
    write_csv_rows(output_dir / "correctness_equivalence.csv", correctness)
    write_csv_rows(output_dir / "speed_benchmark.csv", speed)
    write_csv_rows(output_dir / "predictive_equivalence.csv", predictive)
    backend_rows = backend_comparison_rows(config, resolved) if compare_backends else {}
    for name, rows in backend_rows.items():
        write_csv_rows(output_dir / name, rows)
    _write_report(output_dir, correctness, speed, predictive, backend_rows)
    return output_dir


def _config(mode: OptimizationMode, device: PACDevice, output_dir: Path) -> PACExperimentConfig:
    match mode:
        case "smoke":
            return PACExperimentConfig(
                32,
                16,
                16,
                24,
                model_dim=4,
                modes=2,
                tap_kernel_size=5,
                fir_kernel_size=3,
                epochs=3,
                batch_size=16,
                seeds=(7,),
                device=device,
                output_dir=output_dir,
            )
        case "full":
            return PACExperimentConfig(96, 24, 24, 40, epochs=30, seeds=(7, 11, 19))
        case unreachable:
            assert_never(unreachable)


def _correctness_rows(config: PACExperimentConfig, device: str) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for model_name in PAC_MODELS:
        torch.manual_seed(1201)
        reference = build_pac_model(model_name, config, "reference_naive").to(device=device)
        optimized = build_pac_model(model_name, config, "optimized").to(device=device)
        optimized.load_state_dict(reference.state_dict())
        inputs = torch.randn(
            4,
            min(config.sequence_length, 32),
            config.raw_input_dim,
            device=device,
        )
        with torch.no_grad():
            reference_outputs = reference(inputs)
            optimized_outputs = optimized(inputs)
        diff = (optimized_outputs - reference_outputs).abs()
        rows.append(
            {
                "model": model_name,
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
                "allclose_1e_4": torch.allclose(
                    optimized_outputs, reference_outputs, atol=1.0e-4, rtol=1.0e-4
                ),
            }
        )
    return rows


def _speed_rows(config: PACExperimentConfig, device: str, mode: OptimizationMode) -> list[JsonRow]:
    lengths = (32, 64) if mode == "smoke" else (128, 512, 2048, 4096)
    timed_iters = 3 if mode == "smoke" else 20
    warmup_iters = 1 if mode == "smoke" else 5
    rows: list[JsonRow] = []
    for model_name in PAC_MODELS:
        for length in lengths:
            baseline_speed = None
            for variant in (
                "reference_naive",
                "optimized",
                "pac_lite_prl_fused",
                "pac_lite_block_fused",
            ):
                model = build_pac_model(model_name, config, variant).to(device=device)
                row = speed_row(
                    model,
                    config,
                    device,
                    model_name,
                    variant,
                    length,
                    warmup_iters,
                    timed_iters,
                )
                if variant == "reference_naive":
                    baseline_speed = _float_row_value(row, "train_tokens_per_sec")
                elif baseline_speed is not None:
                    row["train_speedup_vs_reference"] = _float_row_value(
                        row, "train_tokens_per_sec"
                    ) / max(baseline_speed, 1.0e-12)
                rows.append(row)
    for baseline_name in SPEED_BASELINES:
        for length in lengths:
            model = build_paper_regressor(baseline_name, config).to(device=device)
            rows.append(
                speed_row(
                    model,
                    config,
                    device,
                    baseline_name,
                    "baseline",
                    length,
                    warmup_iters,
                    timed_iters,
                )
            )
    return rows


def _predictive_rows(
    config: PACExperimentConfig, device: str, mode: OptimizationMode
) -> list[JsonRow]:
    task_config = replace(config, sample_count=32, validation_count=16, test_count=16, epochs=3)
    if mode == "full":
        task_config = replace(
            config,
            sample_count=96,
            validation_count=24,
            test_count=24,
            epochs=10,
        )
    rows: list[JsonRow] = []
    for task in make_pac_synthetic_tasks(task_config, 7):
        if task.label not in PREDICTIVE_TASKS:
            continue
        for model_name in PAC_MODELS:
            for variant in (
                "reference_naive",
                "optimized",
                "pac_lite_prl_fused",
                "pac_lite_block_fused",
            ):
                torch.manual_seed(1301)
                model = build_pac_model(model_name, task_config, variant)
                outcome = train_regression_model(model, task, task_config, device, 7)
                rows.append(
                    {
                        "task": task.label,
                        "model": model_name,
                        "variant": variant,
                        "validation_loss": outcome.validation_loss,
                        "test_loss": outcome.test_loss,
                        "elapsed_time": outcome.elapsed_time,
                    }
                )
    return rows


def build_pac_model(
    model_name: PACModelName,
    config: PACExperimentConfig,
    variant: BenchmarkVariant,
) -> PACHybridPRLBlock:
    block_type = ReferencePACHybridPRLBlock if variant == "reference_naive" else PACHybridPRLBlock
    use_mlp = model_name == "pac_full"
    recurrence_backend = _variant_backend(variant)
    return block_type(
        raw_input_dim=config.raw_input_dim,
        model_dim=config.model_dim,
        output_dim=config.output_dim,
        modes=config.modes,
        tap_kernel_size=config.tap_kernel_size,
        fir_kernel_size=config.fir_kernel_size,
        use_mlp_branch=use_mlp,
        active_branches=("prl", "fir", "mlp") if use_mlp else ("prl", "fir"),
        recurrence_backend=recurrence_backend,
        hybrid_backend=_variant_hybrid_backend(variant),
    )


def _write_report(
    output_dir: Path,
    correctness: list[JsonRow],
    speed: list[JsonRow],
    predictive: list[JsonRow],
    backend_rows: dict[str, list[JsonRow]],
) -> None:
    max_diff = max(_float_row_value(row, "max_abs_diff") for row in correctness)
    speedups = [
        _float_row_value(row, "train_speedup_vs_reference")
        for row in speed
        if "train_speedup_vs_reference" in row
    ]
    min_speedup = min(speedups) if speedups else 0.0
    report = (
        "# PAC-Hybrid PRL Optimization Report\n\n"
        f"- correctness_max_abs_diff: {max_diff:.6g}\n"
        f"- min_train_speedup_vs_reference: {min_speedup:.3f}x\n"
        f"- gru_comparison_status: {_gru_comparison_status(speed)}\n"
        f"- predictive_status: {_predictive_status(predictive)}\n"
        f"- rows: correctness={len(correctness)}, "
        f"speed={len(speed)}, predictive={len(predictive)}\n"
    )
    if backend_rows:
        report += "\n## Backend Comparison\n\n"
        for name, rows in backend_rows.items():
            report += f"- {name}: rows={len(rows)}\n"
    output_dir.joinpath("optimization_report.md").write_text(report, encoding="utf-8")


def _predictive_status(rows: list[JsonRow]) -> str:
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["model"]))
        by_key.setdefault(key, {})[str(row["variant"])] = _float_row_value(row, "test_loss")
    regressions = [
        losses["optimized"] / max(losses["reference_naive"], 1.0e-12)
        for losses in by_key.values()
        if "optimized" in losses and "reference_naive" in losses
    ]
    if not regressions:
        return "missing"
    return "verified" if max(regressions) <= 1.1 else f"regressed_max_ratio={max(regressions):.3f}"


def _gru_comparison_status(rows: list[JsonRow]) -> str:
    pac: dict[int, float] = {}
    gru: dict[int, float] = {}
    for row in rows:
        length = int(_float_row_value(row, "N"))
        speed = _float_row_value(row, "train_tokens_per_sec")
        if row["model"] == "pac_lite" and row["variant"] == "pac_lite_block_fused":
            pac[length] = speed
        elif row["model"] == "gru":
            gru[length] = speed
    shared = sorted(set(pac) & set(gru))
    if not shared:
        return "missing"
    ratios = [pac[length] / max(gru[length], 1.0e-12) for length in shared]
    if min(ratios) >= 1.05:
        return f"beats_gru_min_ratio={min(ratios):.3f}"
    return f"below_gru_min_ratio={min(ratios):.3f}"


def _float_row_value(row: JsonRow, key: str) -> float:
    value = row[key]
    if isinstance(value, int | float | str):
        return float(value)
    message = f"{key} must be numeric"
    raise TypeError(message)


def _variant_backend(variant: BenchmarkVariant) -> RecurrenceBackend:
    return BACKEND_BY_VARIANT[variant]


def _variant_hybrid_backend(variant: BenchmarkVariant) -> HybridBackend:
    match variant:
        case "pac_lite_prl_fused" | "pac_lite_block_fused":
            return variant
        case (
            "reference_naive"
            | "optimized"
            | "complex_loop"
            | "real2d_loop"
            | "compiled_real2d"
            | "triton_fused"
            | "triton_scan"
            | "auto"
        ):
            return "generic"
        case unreachable:
            assert_never(unreachable)
