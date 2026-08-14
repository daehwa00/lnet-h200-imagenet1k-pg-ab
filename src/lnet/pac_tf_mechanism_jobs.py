from __future__ import annotations

from .pac_tf_mechanism_models import MECHANISM_MODELS
from .pac_tf_mechanism_types import MechanismJob, MechanismQueueConfig

MECHANISM_TASKS: tuple[str, ...] = (
    "damped_oscillator",
    "known_damping_regime",
    "delayed_oscillation",
    "multi_mode_delayed_resonance",
    "pure_fir_negative_control",
    "context_dependent_damping",
    "random_local_pattern",
)


def build_mechanism_jobs(config: MechanismQueueConfig) -> tuple[MechanismJob, ...]:
    return tuple(
        MechanismJob(
            key=f"mechanism__{task}__{model}__seed{seed}",
            seed=seed,
            model=model,
            task=task,
            slots=2,
        )
        for seed in config.seeds
        for task in MECHANISM_TASKS
        for model in MECHANISM_MODELS
    )
