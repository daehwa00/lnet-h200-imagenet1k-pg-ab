from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from pathlib import Path
from statistics import mean
from time import time

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lnet.advanced_experiments import (
    make_delay_teacher_task,
    make_fir_baseline,
    make_fir_teacher_task,
    make_fixed_prl,
    make_gated_prl,
    make_gru_baseline,
    make_hybrid_modal_prl,
    make_switching_teacher_task,
    make_transformer_baseline,
    train_regression_model,
)
from lnet.experiment import (
    SyntheticTaskConfig,
    TrainingConfig,
    make_synthetic_task,
    train_linear_recurrent_baseline,
    train_mlp_baseline,
    train_synthetic_model,
)
from lnet.hybrid import HybridModalPRLSequenceClassifier
from lnet.models import (
    FIRSequenceClassifier,
    GatedPRLSequenceClassifier,
    GRUSequenceClassifier,
    LinearRecurrentClassifier,
    MLPSequenceClassifier,
    PRLSequenceClassifier,
    TransformerSequenceClassifier,
)


@dataclass(frozen=True, slots=True)
class BenchmarkSection:
    title: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    sections: tuple[BenchmarkSection, ...]


@dataclass(frozen=True, slots=True)
class ClassificationConfig:
    train_examples: int = 4000
    validation_examples: int = 1000
    batch_size: int = 128
    epochs: int = 2
    learning_rate: float = 3.0e-3
    device: str = "cpu"
    seed: int = 7
    patch_size: int = 4
    classifier_model_dim: int = 32
    classifier_modes: int = 8
    classifier_state_dim: int = 16
    classifier_kernel_size: int = 5
    classifier_attention_heads: int = 4


@dataclass(frozen=True, slots=True)
class BenchmarkRunConfig:
    seeds: tuple[int, ...] = (7, 11, 19, 23, 31)
    mode_values: tuple[int, ...] = (1, 2, 3, 4, 6)
    model_dim_values: tuple[int, ...] = (2, 4, 6, 8, 12)
    noise_values: tuple[float, ...] = (0.0, 0.01, 0.05)
    regression_epochs: int = 180
    mismatch_model_dim: int = 6
    mismatch_modes: int = 3
    classification: ClassificationConfig = ClassificationConfig()


def render_markdown_report(summary: BenchmarkSummary) -> str:
    sections: list[str] = []
    for section in summary.sections:
        header = "| " + " | ".join(section.headers) + " |"
        divider = "| " + " | ".join("---" for _ in section.headers) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in section.rows]
        sections.append("\n".join((f"## {section.title}", header, divider, *rows)))
    return "\n\n".join(sections)


def _format_float(value: float) -> str:
    if isnan(value):
        return "n/a"
    return f"{value:.6f}"


def _mnist_training_dataset() -> datasets.MNIST:
    root = str(Path(".cache") / "mnist")
    try:
        return datasets.MNIST(
            root=root,
            train=True,
            download=False,
            transform=transforms.ToTensor(),
        )
    except RuntimeError:
        return datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )


def _mnist_patch_loader(
    *,
    use_training_split: bool,
    config: ClassificationConfig,
) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = _mnist_training_dataset()
    if use_training_split:
        subset = Subset(dataset, range(config.train_examples))
        return DataLoader(subset, batch_size=config.batch_size, shuffle=True)
    start = config.train_examples
    stop = config.train_examples + config.validation_examples
    subset = Subset(dataset, range(start, stop))
    return DataLoader(subset, batch_size=config.batch_size, shuffle=False)


def _mnist_patches(images: Tensor, patch_size: int) -> Tensor:
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_count = (images.shape[-2] // patch_size) * (images.shape[-1] // patch_size)
    return patches.reshape(images.shape[0], patch_count, patch_size * patch_size)


def _fit_classifier(model: nn.Module, config: ClassificationConfig) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_loader = _mnist_patch_loader(use_training_split=True, config=config)
    validation_loader = _mnist_patch_loader(use_training_split=False, config=config)
    model.to(device=config.device)
    torch.manual_seed(config.seed)
    for _ in range(config.epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(_mnist_patches(images, config.patch_size).to(device=config.device))
            loss = functional.cross_entropy(logits, labels.to(device=config.device))
            loss.backward()
            optimizer.step()
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in validation_loader:
            logits = model(_mnist_patches(images, config.patch_size).to(device=config.device))
            predictions = torch.argmax(logits, dim=1).cpu()
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / total


def _seed_rows(config: BenchmarkRunConfig) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for seed in config.seeds:
        task = make_synthetic_task(SyntheticTaskConfig(seed=seed))
        result = train_synthetic_model(
            task,
            TrainingConfig(seed=seed, epochs=config.regression_epochs),
        )
        rows.append(
            (
                str(seed),
                _format_float(result.summary.final_loss),
                _format_float(result.summary.validation_loss),
                _format_float(result.summary.pole_mae),
            )
        )
    return tuple(rows)


def _mode_rows(config: BenchmarkRunConfig) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    teacher_task = make_synthetic_task(SyntheticTaskConfig(modes=3, model_dim=6, seed=7))
    for modes in config.mode_values:
        result = train_synthetic_model(
            teacher_task,
            TrainingConfig(seed=7, epochs=config.regression_epochs, student_modes=modes),
        )
        rows.append(
            (
                str(modes),
                _format_float(result.summary.final_loss),
                _format_float(result.summary.validation_loss),
                _format_float(result.summary.pole_mae),
            )
        )
    return tuple(rows)


def _model_dim_rows(config: BenchmarkRunConfig) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    teacher_task = make_synthetic_task(SyntheticTaskConfig(modes=3, model_dim=6, seed=7))
    for model_dim in config.model_dim_values:
        result = train_synthetic_model(
            teacher_task,
            TrainingConfig(seed=7, epochs=config.regression_epochs, student_model_dim=model_dim),
        )
        rows.append(
            (
                str(model_dim),
                _format_float(result.summary.final_loss),
                _format_float(result.summary.validation_loss),
                _format_float(result.summary.pole_mae),
            )
        )
    return tuple(rows)


def _baseline_rows(config: BenchmarkRunConfig) -> tuple[tuple[str, ...], ...]:
    task = make_synthetic_task(SyntheticTaskConfig(seed=7))
    rows = []
    training_config = TrainingConfig(seed=7, epochs=config.regression_epochs)
    baseline_results = (
        ("Projected PRL", train_synthetic_model(task, training_config).summary),
        (
            "Hybrid Modal PRL",
            train_regression_model(
                make_hybrid_modal_prl(
                    raw_input_dim=task.train_inputs.shape[-1],
                    model_dim=6,
                    output_dim=task.train_targets.shape[-1],
                    modes=3,
                    fir_kernel_size=9,
                ),
                task,
                training_config,
            ),
        ),
        ("Per-step MLP", train_mlp_baseline(task, training_config).summary),
        (
            "Linear recurrent",
            train_linear_recurrent_baseline(task, training_config).summary,
        ),
        (
            "FIR baseline",
            train_regression_model(
                make_fir_baseline(
                    raw_input_dim=task.train_inputs.shape[-1],
                    model_dim=6,
                    output_dim=task.train_targets.shape[-1],
                    kernel_size=9,
                ),
                task,
                training_config,
            ),
        ),
        (
            "GRU baseline",
            train_regression_model(
                make_gru_baseline(
                    raw_input_dim=task.train_inputs.shape[-1],
                    model_dim=6,
                    output_dim=task.train_targets.shape[-1],
                ),
                task,
                training_config,
            ),
        ),
        (
            "Transformer baseline",
            train_regression_model(
                make_transformer_baseline(
                    raw_input_dim=task.train_inputs.shape[-1],
                    model_dim=8,
                    output_dim=task.train_targets.shape[-1],
                    attention_heads=2,
                ),
                task,
                training_config,
            ),
        ),
    )
    for name, summary in baseline_results:
        rows.append(
            (
                name,
                _format_float(summary.final_loss),
                _format_float(summary.validation_loss),
                _format_float(summary.pole_mae),
            )
        )
    return tuple(rows)


def _stress_rows(config: BenchmarkRunConfig) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for noise_scale in config.noise_values:
        task = make_synthetic_task(SyntheticTaskConfig(seed=7, noise_scale=noise_scale))
        result = train_synthetic_model(
            task,
            TrainingConfig(seed=7, epochs=config.regression_epochs),
        )
        rows.append(
            (
                f"noise={noise_scale:.2f}",
                _format_float(result.summary.validation_loss),
                _format_float(result.summary.pole_mae),
            )
        )
    mismatch_task = make_synthetic_task(SyntheticTaskConfig(seed=7, modes=5, model_dim=6))
    result = train_synthetic_model(
        mismatch_task,
        TrainingConfig(seed=7, epochs=config.regression_epochs, student_modes=3),
    )
    rows.append(
        (
            "teacher_modes=5 -> student_modes=3",
            _format_float(result.summary.validation_loss),
            _format_float(result.summary.pole_mae),
        )
    )
    return tuple(rows)


def _teacher_mismatch_rows(config: BenchmarkRunConfig) -> tuple[tuple[str, ...], ...]:
    tasks = (
        make_fir_teacher_task(
            sample_count=128,
            validation_count=32,
            sequence_length=32,
            raw_input_dim=2,
            output_dim=2,
            seed=7,
        ),
        make_switching_teacher_task(
            sample_count=128,
            validation_count=32,
            sequence_length=32,
            raw_input_dim=2,
            output_dim=2,
            seed=11,
        ),
    )
    rows: list[tuple[str, ...]] = []
    for task in tasks:
        model_builders = (
            (
                "Projected PRL",
                make_fixed_prl(
                    raw_input_dim=2,
                    model_dim=config.mismatch_model_dim,
                    output_dim=2,
                    modes=config.mismatch_modes,
                ),
            ),
            (
                "Gated PRL",
                make_gated_prl(
                    raw_input_dim=2,
                    model_dim=config.mismatch_model_dim,
                    output_dim=2,
                    modes=config.mismatch_modes,
                    gate_variant="input_output",
                ),
            ),
            (
                "Hybrid Modal PRL",
                make_hybrid_modal_prl(
                    raw_input_dim=2,
                    model_dim=config.mismatch_model_dim,
                    output_dim=2,
                    modes=config.mismatch_modes,
                    fir_kernel_size=9,
                ),
            ),
            (
                "FIR baseline",
                make_fir_baseline(
                    raw_input_dim=2,
                    model_dim=6,
                    output_dim=2,
                    kernel_size=9,
                ),
            ),
            ("GRU baseline", make_gru_baseline(raw_input_dim=2, model_dim=6, output_dim=2)),
            (
                "Transformer baseline",
                make_transformer_baseline(
                    raw_input_dim=2,
                    model_dim=8,
                    output_dim=2,
                    attention_heads=2,
                ),
            ),
        )
        for name, model in model_builders:
            outcome = train_regression_model(
                model,
                task,
                TrainingConfig(seed=7, epochs=config.regression_epochs),
            )
            rows.append((task.teacher_label, name, _format_float(outcome.validation_loss)))
    return tuple(rows)


def _delay_gated_rows(config: BenchmarkRunConfig) -> tuple[tuple[str, ...], ...]:
    task = make_delay_teacher_task(
        sample_count=128,
        validation_count=32,
        sequence_length=40,
        raw_input_dim=2,
        output_dim=2,
        seed=13,
        delay_steps=6,
    )
    rows: list[tuple[str, ...]] = []
    models = (
        ("Fixed PRL", make_fixed_prl(raw_input_dim=2, model_dim=6, output_dim=2, modes=3)),
        (
            "Input-gated PRL",
            make_gated_prl(
                raw_input_dim=2,
                model_dim=6,
                output_dim=2,
                modes=3,
                gate_variant="input",
            ),
        ),
        (
            "Output-gated PRL",
            make_gated_prl(
                raw_input_dim=2,
                model_dim=6,
                output_dim=2,
                modes=3,
                gate_variant="output",
            ),
        ),
        (
            "Input+Output-gated PRL",
            make_gated_prl(
                raw_input_dim=2,
                model_dim=6,
                output_dim=2,
                modes=3,
                gate_variant="input_output",
            ),
        ),
        (
            "Hybrid Modal PRL",
            make_hybrid_modal_prl(
                raw_input_dim=2,
                model_dim=6,
                output_dim=2,
                modes=3,
                fir_kernel_size=13,
            ),
        ),
        (
            "FIR baseline",
            make_fir_baseline(
                raw_input_dim=2,
                model_dim=6,
                output_dim=2,
                kernel_size=13,
            ),
        ),
    )
    for name, model in models:
        outcome = train_regression_model(
            model,
            task,
            TrainingConfig(seed=7, epochs=config.regression_epochs),
        )
        rows.append((name, _format_float(outcome.validation_loss), _format_float(outcome.pole_mae)))
    return tuple(rows)


def _mnist_patch_rows(config: ClassificationConfig) -> tuple[tuple[str, ...], ...]:
    patch_dim = config.patch_size * config.patch_size
    models = (
        (
            "Projected PRL",
            PRLSequenceClassifier(
                raw_input_dim=patch_dim,
                model_dim=config.classifier_model_dim,
                modes=config.classifier_modes,
                class_count=10,
            ),
        ),
        (
            "Gated PRL",
            GatedPRLSequenceClassifier(
                raw_input_dim=patch_dim,
                model_dim=config.classifier_model_dim,
                modes=config.classifier_modes,
                class_count=10,
                gate_variant="input_output",
            ),
        ),
        (
            "Hybrid Modal PRL",
            HybridModalPRLSequenceClassifier(
                raw_input_dim=patch_dim,
                model_dim=config.classifier_model_dim,
                modes=config.classifier_modes,
                class_count=10,
                fir_kernel_size=config.classifier_kernel_size,
            ),
        ),
        (
            "Per-step MLP",
            MLPSequenceClassifier(
                raw_input_dim=patch_dim,
                model_dim=config.classifier_model_dim,
                class_count=10,
            ),
        ),
        (
            "FIR baseline",
            FIRSequenceClassifier(
                raw_input_dim=patch_dim,
                model_dim=config.classifier_model_dim,
                class_count=10,
                kernel_size=config.classifier_kernel_size,
            ),
        ),
        (
            "GRU baseline",
            GRUSequenceClassifier(
                raw_input_dim=patch_dim,
                model_dim=config.classifier_model_dim,
                class_count=10,
            ),
        ),
        (
            "Linear recurrent",
            LinearRecurrentClassifier(
                raw_input_dim=patch_dim,
                model_dim=config.classifier_model_dim,
                state_dim=config.classifier_state_dim,
                class_count=10,
            ),
        ),
        (
            "Transformer baseline",
            TransformerSequenceClassifier(
                raw_input_dim=patch_dim,
                model_dim=config.classifier_model_dim,
                class_count=10,
                attention_heads=config.classifier_attention_heads,
            ),
        ),
    )
    rows: list[tuple[str, ...]] = []
    for name, model in models:
        started_at = time()
        accuracy = _fit_classifier(model, config)
        rows.append((name, f"{accuracy:.4f}", f"{time() - started_at:.1f}s"))
    return tuple(rows)


def run_full_benchmark_suite(config: BenchmarkRunConfig | None = None) -> BenchmarkSummary:
    active_config = config or BenchmarkRunConfig()
    seed_rows = _seed_rows(active_config)
    mode_rows = _mode_rows(active_config)
    model_rows = _model_dim_rows(active_config)
    baseline_rows = _baseline_rows(active_config)
    stress_rows = _stress_rows(active_config)
    mismatch_rows = _teacher_mismatch_rows(active_config)
    delay_rows = _delay_gated_rows(active_config)
    patch_rows = _mnist_patch_rows(active_config.classification)
    aggregate_seed_loss = mean(float(row[2]) for row in seed_rows)
    return BenchmarkSummary(
        sections=(
            BenchmarkSection(
                title="Seed Sweep Aggregate",
                headers=("Metric", "Value"),
                rows=(("Mean validation loss", _format_float(aggregate_seed_loss)),),
            ),
            BenchmarkSection(
                title="Seed Sweep",
                headers=("Seed", "Final Train Loss", "Validation Loss", "Pole MAE"),
                rows=seed_rows,
            ),
            BenchmarkSection(
                title="Mode Sweep",
                headers=("Student Modes", "Final Train Loss", "Validation Loss", "Pole MAE"),
                rows=mode_rows,
            ),
            BenchmarkSection(
                title="Model Dimension Sweep",
                headers=("Student Model Dim", "Final Train Loss", "Validation Loss", "Pole MAE"),
                rows=model_rows,
            ),
            BenchmarkSection(
                title="Synthetic Baseline Comparison",
                headers=("Model", "Final Train Loss", "Validation Loss", "Pole MAE"),
                rows=baseline_rows,
            ),
            BenchmarkSection(
                title="Noise And Mismatch Stress Tests",
                headers=("Condition", "Validation Loss", "Pole MAE"),
                rows=stress_rows,
            ),
            BenchmarkSection(
                title="Teacher Mismatch Comparison",
                headers=("Teacher", "Model", "Validation Loss"),
                rows=mismatch_rows,
            ),
            BenchmarkSection(
                title="Delay Stress And Gated Ablation",
                headers=("Model", "Validation Loss", "Pole Proxy"),
                rows=delay_rows,
            ),
            BenchmarkSection(
                title="MNIST Patch-Sequence Classification",
                headers=("Model", "Validation Accuracy", "Elapsed"),
                rows=patch_rows,
            ),
        ),
    )
