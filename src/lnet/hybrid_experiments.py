from __future__ import annotations

from lnet.benchmarks import BenchmarkSection, BenchmarkSummary
from lnet.hybrid_experiment_tables import (
    branch_ablation_rows,
    delay_rows,
    delayed_modal_rows,
    full_hybrid_by_task,
    gate_rows,
    parameter_rows,
)
from lnet.hybrid_experiment_types import (
    HybridExperimentConfig,
    make_task,
    resolve_device,
    training_config,
)

__all__ = ["HybridExperimentConfig", "run_hybrid_experiment_suite"]


def run_hybrid_experiment_suite(
    config: HybridExperimentConfig | None = None,
) -> BenchmarkSummary:
    active_config = config or HybridExperimentConfig()
    device = resolve_device(active_config.device)
    training = training_config(active_config, device)
    tasks = tuple(make_task(kind, active_config) for kind in active_config.task_kinds)
    trained_models, ablation_rows = branch_ablation_rows(tasks, active_config, training)
    full_models = full_hybrid_by_task(trained_models, tasks, active_config, training)
    sections = [
        BenchmarkSection(
            title="Hybrid Experiment Device",
            headers=("Metric", "Value"),
            rows=(("Device", device), ("Epochs", str(active_config.epochs))),
        ),
        BenchmarkSection(
            title="Hybrid Branch Ablation",
            headers=("Teacher", "Active Branches", "Params", "Validation Loss", "Pole Proxy"),
            rows=ablation_rows,
        ),
    ]
    if active_config.run_parameter_match:
        sections.append(
            BenchmarkSection(
                title="Parameter-Matched Baselines",
                headers=("Teacher", "Model", "Params", "Validation Loss"),
                rows=parameter_rows(tasks, active_config, training, full_models),
            ),
        )
    if active_config.run_gate_diagnostics:
        sections.append(
            BenchmarkSection(
                title="Hybrid Gate Diagnostics",
                headers=(
                    "Teacher",
                    "alpha_PRL",
                    "alpha_FIR",
                    "alpha_MLP",
                    "Entropy",
                    "PRL Norm",
                    "FIR Norm",
                    "MLP Norm",
                ),
                rows=gate_rows(tasks, full_models),
            ),
        )
    if active_config.run_delay_sweep:
        sections.append(
            BenchmarkSection(
                title="Strict Delay Family",
                headers=(
                    "Delay",
                    "FIR Kernel",
                    "Params",
                    "Validation Loss",
                    "Pole Proxy",
                    "Metadata Status",
                ),
                rows=delay_rows(active_config, training),
            ),
        )
        sections.append(
            BenchmarkSection(
                title="Delayed Modal Teacher",
                headers=(
                    "Teacher",
                    "True Delay",
                    "Tap Peak",
                    "Tap Mass Near Delay",
                    "Tap Peak Error",
                    "Mean Pole Error",
                    "Max Pole Error",
                    "Validation Loss",
                ),
                rows=delayed_modal_rows(active_config, training),
            ),
        )
    return BenchmarkSummary(sections=tuple(sections))
