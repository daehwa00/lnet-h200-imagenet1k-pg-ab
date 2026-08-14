from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .pac_confirmatory_baselines import ConfirmatoryFamily
    from .pac_types import PACCompileMode, PACDevice, PACOptimizerMode, PACPrecision

LowDataStage = Literal[
    "sanity",
    "enqueue",
    "workers",
    "report",
    "enqueue-selected-test",
    "enqueue-unseen-final",
    "enqueue-unseen-validation",
]
LowDataEvaluationSplit = Literal["test", "validation"]
LowDataProtocol = Literal["historical_ordered", "clean_stratified"]
LowDataPreset = Literal[
    "smoke",
    "full",
    "matched10k",
    "matched10k_real_dynamical",
    "matched10k_low_data",
    "local_stem_overnight",
    "convstem2_only",
    "pac_design_stack",
    "qprl_depth2",
    "implicit_complex_depth2",
    "tight_frame_depth2",
    "stiefel_depth2",
    "stiefel_factorial",
    "stiefel_optimized_ablation",
    "stiefel_capacity_scaling",
    "stiefel_large_capacity_scaling",
    "stiefel_validation_capacity_selection",
    "stiefel_core_component_ablation",
    "matched_pac_d64_m16_test",
]
LowDataStatus = Literal["running", "done", "failed"]


@dataclass(frozen=True, slots=True)
class LowDataQueueConfig:
    output_root: Path = Path(".omx/results/pac-hybrid-prl/recommended-low-data-20260708")
    preset: LowDataPreset = "full"
    seeds: tuple[int, ...] = (7, 11, 19, 23, 31)
    device: PACDevice = "auto"
    workers: int = 4
    total_slots: int = 8
    compile_mode: PACCompileMode = "none"
    precision: PACPrecision = "fp32"
    optimizer_mode: PACOptimizerMode = "default"

    def __post_init__(self) -> None:
        graph_modes = {"reduce-overhead", "max-autotune"}
        if self.compile_mode in graph_modes:
            message = f"{self.compile_mode} uses CUDA Graphs and is inference-only in this queue"
            raise ValueError(message)
        if self.compile_mode != "none" and self.workers > 1:
            message = "compiled queue execution requires workers=1 to avoid Dynamo/FX thread races"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class LowDataJob:
    key: str
    seed: int
    model: str
    dataset: str
    ratio: float
    slots: int = 1
    evaluation_split: LowDataEvaluationSplit = "test"
    refit_full_train: bool = False
    data_protocol: LowDataProtocol = "historical_ordered"
    restore_best_validation: bool = False
    evaluation_collection: str = "development_13_ucr"
    baseline_family: ConfirmatoryFamily | None = None
    reference_model: str | None = None
    validation_trial: int | None = None
    architecture_metadata_json: str = ""
    refit_epochs: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    parameter_match_tolerance: float | None = None
    official_test_contract_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LowDataEvent:
    key: str
    status: LowDataStatus
    notes: str = ""
