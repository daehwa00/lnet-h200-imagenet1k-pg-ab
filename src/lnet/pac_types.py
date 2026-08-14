from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Tensor

PACMode = Literal["smoke", "synthetic", "ablation", "ood", "efficiency", "real", "full"]
PACDevice = str
PACCompileMode = Literal[
    "none",
    "default",
    "dynamic-no-cudagraph",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
PACPrecision = Literal["fp32", "bf16"]
PACOptimizerMode = Literal["default", "foreach", "fused"]
PACBranchName = Literal["prl", "fir", "mlp"]
PACFusion = Literal[
    "sum",
    "learned_scalar_sum",
    "softmax",
    "temperature_softmax",
    "sigmoid_gates",
]
PACModelName = Literal[
    "pac_full",
    "pac_lite",
    "controlled_tapped_prl_only",
    "tapped_prl_fixed",
    "fixed_prl",
    "fir_only",
    "mlp_only",
    "linear_recurrent",
    "gru",
    "lstm",
    "transformer",
    "selective_diagonal_ssm",
]


@dataclass(frozen=True, slots=True)
class PACExperimentConfig:
    sample_count: int
    validation_count: int
    test_count: int
    sequence_length: int
    raw_input_dim: int = 2
    output_dim: int = 2
    model_dim: int = 16
    modes: int = 8
    tap_kernel_size: int = 16
    fir_kernel_size: int = 9
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    seeds: tuple[int, ...] = (7, 11, 19, 23, 31)
    device: PACDevice = "auto"
    output_dir: Path = Path(".omx/results/pac-hybrid-prl/full")
    compile_mode: PACCompileMode = "none"
    precision: PACPrecision = "fp32"
    optimizer_mode: PACOptimizerMode = "default"
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True, slots=True)
class PACRegressionTask:
    label: str
    train_inputs: Tensor
    train_targets: Tensor
    validation_inputs: Tensor
    validation_targets: Tensor
    test_inputs: Tensor
    test_targets: Tensor
    true_delay: int | None = None
    true_frequency: float | None = None
    train_teacher_damping: Tensor | None = None
    validation_teacher_damping: Tensor | None = None
    test_teacher_damping: Tensor | None = None
    validation_regime: Tensor | None = None
    true_frequencies: tuple[float, ...] = ()
    true_dampings: tuple[float, ...] = ()
    diagnostic_inputs: Tensor | None = None
    diagnostic_targets: Tensor | None = None
    mechanism_expectation: Literal["positive", "negative", "neutral"] = "neutral"


@dataclass(frozen=True, slots=True)
class PACClassificationTask:
    label: str
    train_inputs: Tensor
    train_labels: Tensor
    validation_inputs: Tensor
    validation_labels: Tensor
    test_inputs: Tensor
    test_labels: Tensor
    class_count: int


@dataclass(frozen=True, slots=True)
class PACTrainOutcome:
    train_loss: float
    validation_loss: float
    test_loss: float
    grad_norm: float
    elapsed_time: float
    best_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class PACClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float


@dataclass(frozen=True, slots=True)
class UCRDataset:
    name: str
    train_inputs: Tensor
    train_labels: Tensor
    test_inputs: Tensor
    test_labels: Tensor
    class_count: int
