from __future__ import annotations

from typing import Final, assert_never

from .pac_design_stack_models import PAC_DESIGN_MODELS
from .pac_implicit_complex_models import IMPLICIT_COMPLEX_MODELS
from .pac_local_stem_models import CONVSTEM2_MODELS, LOCAL_STEM_MODELS
from .pac_qprl_models import QPRL_MODELS
from .pac_recommended_low_data_models import (
    MATCHED_PAC_D64_M16_MODELS,
    RECOMMENDED_MODEL,
)
from .pac_recommended_low_data_types import (
    LowDataEvaluationSplit,
    LowDataJob,
    LowDataPreset,
    LowDataQueueConfig,
)
from .pac_stiefel_variants import (
    STIEFEL_CAPACITY_MODELS,
    STIEFEL_CONFIRMATORY_CAPACITY_MODELS,
    STIEFEL_CORE_COMPONENT_ABLATION_MODELS,
    STIEFEL_FACTORIAL_MODELS,
    STIEFEL_LARGE_CAPACITY_MODELS,
    STIEFEL_MODELS,
    STIEFEL_OPTIMIZED_ABLATION_MODELS,
)
from .pac_tight_frame_models import TIGHT_FRAME_MODELS

DATASETS: Final[tuple[str, ...]] = ("ECG5000", "FordA", "FordB", "Wafer")
REAL_DYNAMICAL_DATASETS: Final[tuple[str, ...]] = (
    "ECG5000",
    "FordA",
    "FordB",
    "Wafer",
    "TwoLeadECG",
    "ECG200",
    "GunPoint",
    "ItalyPowerDemand",
    "ArrowHead",
    "Plane",
    "ECGFiveDays",
    "Trace",
    "MoteStrain",
)
DATA_RATIOS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
MODELS: Final[tuple[str, ...]] = (
    RECOMMENDED_MODEL,
    "tcn",
    "cnn1d",
    "gru",
    "lstm",
    "mamba_ssm",
    "transformer_tiny",
    "pac_lite_stack2_pyramid",
)
MATCHED_10K_MODELS: Final[tuple[str, ...]] = (
    RECOMMENDED_MODEL,
    "tcn_matched10k",
    "cnn1d_matched10k",
    "gru_matched10k",
    "lstm_matched10k",
    "mamba_ssm_matched10k",
    "transformer_matched10k",
)
LOCAL_STEM_OVERNIGHT_MODELS: Final[tuple[str, ...]] = LOCAL_STEM_MODELS + MATCHED_10K_MODELS


def build_low_data_jobs(config: LowDataQueueConfig) -> tuple[LowDataJob, ...]:
    match config.preset:
        case "smoke":
            seeds = (config.seeds[0],)
            datasets = ("ECG5000",)
            ratios = (0.1, 1.0)
            models = (RECOMMENDED_MODEL, "tcn", "gru")
        case "full":
            seeds = config.seeds
            datasets = DATASETS
            ratios = DATA_RATIOS
            models = MODELS
        case "matched10k":
            seeds = config.seeds
            datasets = DATASETS
            ratios = (1.0,)
            models = MATCHED_10K_MODELS
        case "matched10k_real_dynamical":
            seeds = config.seeds
            datasets = REAL_DYNAMICAL_DATASETS
            ratios = (1.0,)
            models = MATCHED_10K_MODELS
        case "matched10k_low_data":
            seeds = config.seeds
            datasets = DATASETS
            ratios = DATA_RATIOS
            models = MATCHED_10K_MODELS
        case "local_stem_overnight":
            seeds = config.seeds
            datasets = REAL_DYNAMICAL_DATASETS
            ratios = (1.0,)
            models = LOCAL_STEM_OVERNIGHT_MODELS
        case "convstem2_only":
            seeds = config.seeds
            datasets = REAL_DYNAMICAL_DATASETS
            ratios = (1.0,)
            models = CONVSTEM2_MODELS
        case "pac_design_stack":
            seeds = config.seeds
            datasets = REAL_DYNAMICAL_DATASETS
            ratios = (1.0,)
            models = PAC_DESIGN_MODELS
        case (
            "qprl_depth2"
            | "implicit_complex_depth2"
            | "tight_frame_depth2"
            | "stiefel_depth2"
            | "stiefel_factorial"
            | "stiefel_optimized_ablation"
            | "stiefel_capacity_scaling"
            | "stiefel_large_capacity_scaling"
            | "stiefel_validation_capacity_selection"
            | "stiefel_core_component_ablation"
            | "matched_pac_d64_m16_test"
        ):
            seeds = config.seeds
            datasets = REAL_DYNAMICAL_DATASETS
            ratios = (1.0,)
            models = {
                "qprl_depth2": QPRL_MODELS,
                "implicit_complex_depth2": IMPLICIT_COMPLEX_MODELS,
                "tight_frame_depth2": TIGHT_FRAME_MODELS,
                "stiefel_depth2": STIEFEL_MODELS,
                "stiefel_factorial": STIEFEL_FACTORIAL_MODELS,
                "stiefel_optimized_ablation": STIEFEL_OPTIMIZED_ABLATION_MODELS,
                "stiefel_capacity_scaling": STIEFEL_CAPACITY_MODELS,
                "stiefel_large_capacity_scaling": STIEFEL_LARGE_CAPACITY_MODELS,
                "stiefel_validation_capacity_selection": STIEFEL_CONFIRMATORY_CAPACITY_MODELS,
                "stiefel_core_component_ablation": STIEFEL_CORE_COMPONENT_ABLATION_MODELS,
                "matched_pac_d64_m16_test": MATCHED_PAC_D64_M16_MODELS,
            }[config.preset]
        case unreachable:
            assert_never(unreachable)
    evaluation_split, key_namespace, refit_full_train = _job_protocol(config.preset)
    clean_selection = config.preset == "stiefel_validation_capacity_selection"
    return tuple(
        LowDataJob(
            key=f"{key_namespace}:{seed}:{model}:{dataset}:{ratio}",
            seed=seed,
            model=model,
            dataset=dataset,
            ratio=ratio,
            slots=_slots(model),
            evaluation_split=evaluation_split,
            refit_full_train=refit_full_train,
            data_protocol="clean_stratified" if clean_selection else "historical_ordered",
            restore_best_validation=clean_selection,
        )
        for seed in seeds
        for dataset in datasets
        for model in models
        for ratio in ratios
    )


def _slots(model: str) -> int:
    return 2 if model in {"tcn", "mamba_ssm", "tcn_matched10k", "mamba_ssm_matched10k"} else 1


def _job_protocol(preset: LowDataPreset) -> tuple[LowDataEvaluationSplit, str, bool]:
    if preset == "stiefel_validation_capacity_selection":
        return "validation", "low_data:validation_capacity_selection", False
    if preset == "matched_pac_d64_m16_test":
        return "test", "low_data:matched_pac_d64_m16_test", True
    return "test", "low_data", False
