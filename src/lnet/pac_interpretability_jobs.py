from __future__ import annotations

from typing import Final

from .pac_interpretability_types import InterpretabilityJob, InterpretabilityQueueConfig

SYNTHETIC_TASKS: Final[tuple[str, ...]] = (
    "modal_teacher",
    "delayed_exponential",
    "delayed_oscillatory",
    "multi_mode_delayed_resonance",
    "active_damping_teacher",
    "context_damped_exponential",
    "delayed_context_damped_exponential",
    "random_fir_teacher",
)
SYNTHETIC_MODELS: Final[tuple[str, ...]] = (
    "pac_full_depth2_all_learned_mix_hermitian_realmean_max_seqreg",
    "pac_lite",
    "controlled_tapped_prl_only",
    "fixed_prl",
    "tcn_seqreg_matched",
    "gru_seqreg_matched",
)
REAL_DATASETS: Final[tuple[str, ...]] = (
    "ECG5000",
    "ECGFiveDays",
    "Trace",
    "Wafer",
    "ArrowHead",
    "Plane",
    "ItalyPowerDemand",
    "FordA",
    "FordB",
)
REAL_MODEL: Final[str] = "pac_full_depth2_causal_all_learned_mix_hermitian_realmean_max"


def build_interpretability_jobs(
    config: InterpretabilityQueueConfig,
) -> tuple[InterpretabilityJob, ...]:
    seeds = config.seeds if config.preset == "full" else (config.seeds[0],)
    tasks = SYNTHETIC_TASKS if config.preset == "full" else ("modal_teacher",)
    datasets = REAL_DATASETS if config.preset == "full" else ("Tiny",)
    models = (
        SYNTHETIC_MODELS if config.preset == "full" else (SYNTHETIC_MODELS[0], "gru_seqreg_matched")
    )
    return (
        *(
            InterpretabilityJob(
                f"{config.preset}:synthetic:{seed}:{model}:{task}",
                "synthetic_mechanism",
                seed,
                model,
                task,
                slots=_slots(model),
            )
            for seed in seeds
            for task in tasks
            for model in models
        ),
        *(
            InterpretabilityJob(
                f"{config.preset}:real:{seed}:{REAL_MODEL}:{dataset}",
                "real_modal",
                seed,
                REAL_MODEL,
                dataset,
                slots=2,
            )
            for seed in seeds
            for dataset in datasets
        ),
    )


def _slots(model: str) -> int:
    return 2 if model.startswith(("tcn", "gru")) else 1
