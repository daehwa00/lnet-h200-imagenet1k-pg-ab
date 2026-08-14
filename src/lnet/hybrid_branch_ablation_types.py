from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .hybrid import BranchName

BranchAblationMode = Literal["smoke", "full"]
BranchTaskName = Literal[
    "modal_teacher",
    "random_fir_teacher",
    "strict_delay_6",
    "delayed_exponential_4",
    "delayed_oscillatory_4",
    "switching_teacher",
]


@dataclass(frozen=True, slots=True)
class BranchSpec:
    name: str
    branches: tuple[BranchName, ...]


BRANCH_SPECS: tuple[BranchSpec, ...] = (
    BranchSpec(name="prl_only", branches=("prl",)),
    BranchSpec(name="fir_only", branches=("fir",)),
    BranchSpec(name="mlp_only", branches=("mlp",)),
    BranchSpec(name="prl_fir", branches=("prl", "fir")),
    BranchSpec(name="prl_mlp", branches=("prl", "mlp")),
    BranchSpec(name="fir_mlp", branches=("fir", "mlp")),
    BranchSpec(name="prl_fir_mlp", branches=("prl", "fir", "mlp")),
)


@dataclass(frozen=True, slots=True)
class HybridBranchAblationConfig:
    sample_count: int = 96
    validation_count: int = 24
    sequence_length: int = 40
    raw_input_dim: int = 2
    output_dim: int = 2
    model_dim: int = 8
    modes: int = 4
    fir_kernel_size: int = 13
    prl_tap_kernel_size: int = 8
    epochs: int = 30
    learning_rate: float = 4.0e-2
    weight_decay: float = 1.0e-4
    seeds: tuple[int, ...] = (7, 11, 19)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    task_names: tuple[BranchTaskName, ...] = (
        "modal_teacher",
        "random_fir_teacher",
        "strict_delay_6",
        "delayed_exponential_4",
        "delayed_oscillatory_4",
        "switching_teacher",
    )
    branch_specs: tuple[BranchSpec, ...] = BRANCH_SPECS
