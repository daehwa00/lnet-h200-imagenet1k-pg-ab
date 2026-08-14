from __future__ import annotations

from typing import Final

from .pac_paper_queue_types import PaperJob, PaperQueueConfig

REAL_DATASETS: Final[tuple[str, ...]] = ("ECG5000", "FordA", "FordB", "Wafer")


def build_jobs(config: PaperQueueConfig) -> tuple[PaperJob, ...]:
    seeds = config.seeds if config.preset == "full" else (config.seeds[0],)
    models = _models(config.preset)
    jobs: list[PaperJob] = [PaperJob("param_audit", "param_audit", seeds[0], "audit", "audit")]
    for seed in seeds:
        jobs.extend(_sampling_jobs(seed, models))
        jobs.append(
            PaperJob(
                f"damping_counterfactual:{seed}",
                "damping_counterfactual",
                seed,
                "pac_full",
                "active_damping_teacher",
            )
        )
        jobs.extend(_synthetic_jobs(seed, models))
        jobs.extend(_expanded_jobs(seed, models))
        jobs.extend(_role_jobs(seed))
        jobs.extend(_low_data_jobs(seed, models, config.preset))
        if config.preset == "full":
            jobs.extend(_real_jobs(seed, models))
    jobs.append(
        PaperJob(
            "speed_correctness:exclusive", "speed_correctness", seeds[0], "speed", "speed", slots=4
        )
    )
    return tuple(jobs)


def _sampling_jobs(seed: int, models: tuple[str, ...]) -> tuple[PaperJob, ...]:
    jobs = [
        PaperJob(
            f"sampling:{seed}:{model}:{delta}",
            "sampling_rate_ood",
            seed,
            model,
            "modal_teacher",
            value=delta,
        )
        for model in models
        for delta in (0.5, 0.75, 1.5, 2.0)
    ]
    jobs.extend(
        PaperJob(
            f"irregular:{seed}:{model}",
            "irregular_time_ood",
            seed,
            model,
            "modal_teacher",
            value=1.0,
        )
        for model in models
    )
    return tuple(jobs)


def _synthetic_jobs(seed: int, models: tuple[str, ...]) -> tuple[PaperJob, ...]:
    tasks = ("modal_teacher", "delayed_oscillatory", "active_damping_teacher", "random_fir_teacher")
    return tuple(
        PaperJob(
            f"synthetic:{seed}:{model}:{task}", "strong_baselines_synthetic", seed, model, task
        )
        for model in models
        for task in tasks
    )


def _expanded_jobs(seed: int, models: tuple[str, ...]) -> tuple[PaperJob, ...]:
    tasks = ("ood_length_128", "ood_noise_0.1")
    return tuple(
        PaperJob(f"ood:{seed}:{model}:{task}", "expanded_ood", seed, model, task)
        for model in models
        for task in tasks
    )


def _role_jobs(seed: int) -> tuple[PaperJob, ...]:
    knockouts = ("prl_off", "fir_off", "mlp_off", "direct_term_off", "fir_pointwise_off")
    return tuple(
        PaperJob(
            f"role:{seed}:{knockout}", "role_ablation", seed, knockout, "active_damping_teacher"
        )
        for knockout in knockouts
    )


def _low_data_jobs(seed: int, models: tuple[str, ...], preset: str) -> tuple[PaperJob, ...]:
    ratios = (0.1,) if preset == "smoke" else (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    return tuple(
        PaperJob(
            f"low:{seed}:{model}:{ratio}",
            "low_data_scaling",
            seed,
            model,
            "active_damping_teacher",
            ratio=ratio,
        )
        for model in models
        for ratio in ratios
    )


def _real_jobs(seed: int, models: tuple[str, ...]) -> tuple[PaperJob, ...]:
    baseline = tuple(
        PaperJob(
            f"real:{seed}:{model}:{dataset}",
            "strong_baselines_real",
            seed,
            model,
            dataset,
            dataset=dataset,
        )
        for model in models
        for dataset in REAL_DATASETS
    )
    low = tuple(
        PaperJob(
            f"real_low:{seed}:{model}:{dataset}:{ratio}",
            "low_data_scaling",
            seed,
            model,
            dataset,
            ratio=ratio,
            dataset=dataset,
        )
        for model in ("pac_lite", "pac_full", "gru", "tcn")
        for dataset in REAL_DATASETS
        for ratio in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    )
    return baseline + low


def _models(preset: str) -> tuple[str, ...]:
    if preset == "smoke":
        return ("pac_lite", "gru")
    return (
        "pac_lite",
        "pac_full",
        "gru",
        "lstm",
        "cnn1d",
        "tcn",
        "transformer_tiny",
        "selective_diagonal_ssm",
    )
