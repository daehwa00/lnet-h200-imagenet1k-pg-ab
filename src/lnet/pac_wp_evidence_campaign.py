# ruff: noqa: TC003
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

from .pac_recommended_low_data_types import LowDataJob
from .pac_tf_evidence_queue import EvidenceJob, EvidenceKind
from .pac_tf_p1p2_types import P1P2Job

WP_MODEL: Final = "WP"
WP_REFERENCE_MODEL: Final = "pac_headroom_wp_d64_m16"
SEEDS: Final = (7, 11, 19, 23, 31)
UCR_DATASETS: Final = (
    "ArrowHead",
    "CinCECGTorso",
    "CricketX",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "Earthquakes",
    "FordA",
    "FordB",
    "GunPoint",
    "ItalyPowerDemand",
    "MoteStrain",
    "Phoneme",
    "Plane",
    "StarLightCurves",
    "Trace",
    "TwoLeadECG",
    "Wafer",
)
LOW_DATASETS: Final = (
    "CinCECGTorso",
    "CricketX",
    "Earthquakes",
    "Phoneme",
    "StarLightCurves",
)
LOW_RATIOS: Final = (0.01, 0.05, 0.10, 0.25, 0.50)
INTERPRETABILITY_DATASETS: Final = (
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
MECHANISM_TASKS: Final = (
    "damped_oscillator",
    "delayed_oscillation",
    "multi_mode_delayed_resonance",
    "pure_fir_negative_control",
    "random_local_pattern",
)


def official_test_jobs() -> tuple[LowDataJob, ...]:
    return tuple(
        LowDataJob(
            key=f"wp_official_test:{dataset}:seed{seed}",
            seed=seed,
            model=WP_MODEL,
            dataset=dataset,
            ratio=1.0,
            evaluation_split="test",
            refit_full_train=True,
            data_protocol="clean_stratified",
            restore_best_validation=False,
            evaluation_collection="wp_official_ucr_test",
            reference_model=WP_REFERENCE_MODEL,
            refit_epochs=75,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
        )
        for dataset in UCR_DATASETS
        for seed in SEEDS
    )


def low_data_jobs() -> tuple[LowDataJob, ...]:
    return tuple(
        LowDataJob(
            key=f"wp_low_data:{dataset}:ratio{ratio:g}:seed{seed}",
            seed=seed,
            model=WP_MODEL,
            dataset=dataset,
            ratio=ratio,
            evaluation_split="test",
            refit_full_train=False,
            data_protocol="clean_stratified",
            restore_best_validation=True,
            evaluation_collection="wp_low_data_boundary",
            reference_model=WP_REFERENCE_MODEL,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
        )
        for dataset in LOW_DATASETS
        for ratio in LOW_RATIOS
        for seed in SEEDS
    )


def enqueue_low_data_shards(root: Path, *, shard_count: int = 24) -> dict[str, object]:
    if shard_count < 1:
        message = "shard_count must be positive"
        raise ValueError(message)
    jobs = [*official_test_jobs(), *low_data_jobs()]
    weights = [_job_weight(job) for job in jobs]
    shards: list[list[LowDataJob]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    ordered = sorted(zip(jobs, weights, strict=True), key=lambda row: row[1], reverse=True)
    for job, weight in ordered:
        index = min(range(shard_count), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += weight
    shard_root = root / "low-data-shards"
    for index, shard in enumerate(shards):
        active = shard_root / f"shard-{index:02d}"
        active.mkdir(parents=True, exist_ok=True)
        (active / "queue_manifest.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard),
            encoding="utf-8",
        )
    contract: dict[str, object] = {
        "schema": "pac_wp_evidence_campaign.v3.stable_autocorr_d64_m16",
        "paper_model": "PAC",
        "internal_model": WP_MODEL,
        "reference_model": WP_REFERENCE_MODEL,
        "selected_model_dim": 64,
        "selected_modes": 16,
        "official_test_jobs": len(official_test_jobs()),
        "low_data_jobs": len(low_data_jobs()),
        "shards": shard_count,
        "estimated_loads": loads,
        "official_test_accessed_at_enqueue": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "low_data_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def p1p2_jobs() -> tuple[P1P2Job, ...]:
    jobs = [
        P1P2Job(
            key=f"wp_p1p2:low_data:{dataset}:ratio{ratio:g}:seed{seed}",
            package="low_data",
            seed=seed,
            model="wp_pac",
            reference_model=WP_REFERENCE_MODEL,
            dataset=dataset,
            ratio=ratio,
            slots=2,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
        )
        for dataset in LOW_DATASETS
        for ratio in LOW_RATIOS
        for seed in SEEDS
    ]
    jobs.extend(
        P1P2Job(
            key=f"wp_p1p2:real_diagnostics:{dataset}:seed{seed}",
            package="real_diagnostics",
            seed=seed,
            model="wp_pac",
            reference_model=WP_REFERENCE_MODEL,
            dataset=dataset,
            ratio=1.0,
            slots=2,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
        )
        for dataset in LOW_DATASETS
        for seed in SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"wp_p1p2:synthetic_ood:seed{seed}",
            package="synthetic_ood",
            seed=seed,
            model="wp_pac",
            reference_model=WP_REFERENCE_MODEL,
            slots=2,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
        )
        for seed in SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"wp_p1p2:real_domain_ood:mit_bih:seed{seed}",
            package="real_domain_ood",
            seed=seed,
            model="wp_pac",
            reference_model=WP_REFERENCE_MODEL,
            dataset="mit-bih-ds1-ds2",
            slots=2,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
        )
        for seed in SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"wp_p1p2:efficiency:{runtime}:n{length}:b{batch_size}",
            package="efficiency",
            seed=SEEDS[0],
            model="wp_pac",
            reference_model=WP_REFERENCE_MODEL,
            length=length,
            batch_size=batch_size,
            runtime=runtime,
            slots=16,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
        )
        for length in (128, 500, 2_000, 8_000, 32_000)
        for batch_size in (1, 8, 32)
        for runtime in ("train", "eager", "compiled")
    )
    return tuple(jobs)


def enqueue_p1p2_shards(
    root: Path,
    *,
    training_shards: int = 16,
    efficiency_shards: int = 5,
) -> dict[str, object]:
    jobs = p1p2_jobs()
    training = [job for job in jobs if job.package != "efficiency"]
    efficiency = [job for job in jobs if job.package == "efficiency"]
    _write_round_robin_shards(root / "p1p2-training-shards", training, training_shards)
    _write_round_robin_shards(root / "p1p2-efficiency-shards", efficiency, efficiency_shards)
    contract: dict[str, object] = {
        "schema": "pac_wp_p1p2.v2.stable_autocorr",
        "model": "wp_pac",
        "reference_model": WP_REFERENCE_MODEL,
        "jobs": len(jobs),
        "low_data": sum(job.package == "low_data" for job in jobs),
        "real_diagnostics": sum(job.package == "real_diagnostics" for job in jobs),
        "synthetic_ood": sum(job.package == "synthetic_ood" for job in jobs),
        "real_domain_ood": sum(job.package == "real_domain_ood" for job in jobs),
        "efficiency": len(efficiency),
        "training_shards": training_shards,
        "efficiency_shards": efficiency_shards,
    }
    (root / "p1p2_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def _write_round_robin_shards(root: Path, jobs: list[P1P2Job], count: int) -> None:
    if count < 1:
        message = "shard count must be positive"
        raise ValueError(message)
    shards: list[list[P1P2Job]] = [[] for _ in range(count)]
    for index, job in enumerate(jobs):
        shards[index % count].append(job)
    for index, shard in enumerate(shards):
        active = root / f"shard-{index:02d}"
        active.mkdir(parents=True, exist_ok=True)
        (active / "p1p2_manifest.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard),
            encoding="utf-8",
        )


def enqueue_evidence_shards(  # noqa: C901
    root: Path, *, shard_count: int = 12
) -> dict[str, object]:
    shards: list[dict[str, list[EvidenceJob]]] = [
        {"training": [], "interpretability": [], "sensitivity": []}
        for _ in range(shard_count)
    ]
    classifier_interventions = (
        "forward_direction_removal",
        "backward_direction_removal",
        "moment_head_intervention",
        "lag1_intervention",
        "lag4_intervention",
        "low_band_removal",
        "detail_band_removal",
        "uniform_band_fusion",
    )
    classifier_index = 0
    for dataset in INTERPRETABILITY_DATASETS:
        for seed in SEEDS:
            checkpoint_key = f"wp_classifier_checkpoint:{dataset}:seed{seed}"
            shard = shards[classifier_index % shard_count]
            classifier_index += 1
            shard["training"].append(
                _evidence_job(
                    checkpoint_key,
                    "core_ablation",
                    seed,
                    dataset,
                    "reference",
                    architecture_surface="wp_classifier",
                )
            )
            shard["interpretability"].extend(
                _evidence_job(
                    f"wp_classifier_intervention:{intervention}:{dataset}:seed{seed}",
                    "interpretability",
                    seed,
                    dataset,
                    intervention,
                    architecture_surface="wp_classifier",
                    checkpoint_key=checkpoint_key,
                )
                for intervention in classifier_interventions
            )
    mechanism_index = 0
    frequency_tasks = {
        "damped_oscillator": 1,
        "delayed_oscillation": 1,
        "multi_mode_delayed_resonance": 3,
    }
    for task in MECHANISM_TASKS:
        for seed in SEEDS:
            checkpoint_key = f"wp_mechanism_checkpoint:{task}:seed{seed}"
            shard = shards[mechanism_index % shard_count]
            mechanism_index += 1
            shard["training"].append(
                _evidence_job(
                    checkpoint_key,
                    "mechanism_checkpoint",
                    seed,
                    task,
                    "checkpoint_training",
                    architecture_surface="wp_endpoint_regressor",
                )
            )
            shard["interpretability"].append(
                _evidence_job(
                    f"wp_mechanism_intervention:frame:{task}:seed{seed}",
                    "interpretability",
                    seed,
                    task,
                    "frame_subspace_perturbation",
                    architecture_surface="wp_endpoint_regressor",
                    checkpoint_key=checkpoint_key,
                )
            )
            if task in frequency_tasks:
                shard["interpretability"].append(
                    _evidence_job(
                        f"wp_mechanism_intervention:frequency:{task}:seed{seed}",
                        "interpretability",
                        seed,
                        task,
                        "teacher_frequency_recovery",
                        architecture_surface="wp_endpoint_regressor",
                        checkpoint_key=checkpoint_key,
                    )
                )
                for teacher in range(frequency_tasks[task]):
                    for intervention in ("matched_mode_knockout", "random_mode_knockout"):
                        shard["interpretability"].append(
                            _evidence_job(
                                f"wp_mechanism_intervention:{intervention}:teacher{teacher}:{task}:seed{seed}",
                                "interpretability",
                                seed,
                                task,
                                intervention,
                                architecture_surface="wp_endpoint_regressor",
                                checkpoint_key=checkpoint_key,
                                teacher_mode_index=teacher,
                            )
                        )
    settings = (
        ("depth", "2"),
        ("stem_kernel", "5"),
        ("stem_kernel", "13"),
        ("local_kernel", "3"),
        ("local_kernel", "9"),
        ("stride", "1"),
        ("stride", "4"),
        ("depth", "1"),
        ("depth", "3"),
        ("model_dim", "32"),
        ("model_dim", "128"),
        ("modes", "8"),
        ("modes", "32"),
    )
    sensitivity_index = 0
    for dataset in LOW_DATASETS:
        for seed in SEEDS[:3]:
            for factor, level in settings:
                shards[sensitivity_index % shard_count]["sensitivity"].append(
                    _evidence_job(
                        f"wp_sensitivity:{factor}:{level}:{dataset}:seed{seed}",
                        "sensitivity",
                        seed,
                        dataset,
                        factor,
                        level=level,
                        reference_level=factor == "depth" and level == "2",
                        architecture_surface="wp_classifier",
                    )
                )
                sensitivity_index += 1
    root_shards = root / "evidence-shards"
    for index, phases in enumerate(shards):
        active = root_shards / f"shard-{index:02d}"
        active.mkdir(parents=True, exist_ok=True)
        for phase, jobs in phases.items():
            (active / f"{phase}_manifest.jsonl").write_text(
                "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
                encoding="utf-8",
            )
    counts = {
        phase: sum(len(shard[phase]) for shard in shards)
        for phase in ("training", "interpretability", "sensitivity")
    }
    contract: dict[str, object] = {
        "schema": "pac_wp_interpretability_sensitivity.v2.stable_autocorr",
        "model": WP_MODEL,
        "shards": shard_count,
        **counts,
        "damping_regime_recovery": False,
        "retrained_core_ablation": False,
    }
    (root / "evidence_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def _evidence_job(
    key: str,
    kind: EvidenceKind,
    seed: int,
    scope: str,
    intervention: str,
    *,
    level: str = "reference",
    architecture_surface: str,
    checkpoint_key: str = "",
    teacher_mode_index: int | None = None,
    reference_level: bool = False,
) -> EvidenceJob:
    return EvidenceJob(
        key=key,
        protocol_id="pac-wp-evidence-20260712-v2-stable-autocorr",
        kind=kind,
        seed=seed,
        scope=scope,
        intervention=intervention,
        model=WP_MODEL,
        level=level,
        architecture_surface=architecture_surface,
        checkpoint_key=checkpoint_key,
        slots=2,
        protocol_sha256="wp-stable-autocorr-20260712",
        capacity_artifact_sha256="wp-d64-m16-stable-autocorr-20260712",
        selected_model=WP_REFERENCE_MODEL,
        selected_model_dim=64,
        selected_modes=16,
        teacher_mode_index=teacher_mode_index,
        reference_level=reference_level,
    )


def _job_weight(job: LowDataJob) -> float:
    dataset_weight = {
        "FordA": 10.0,
        "FordB": 10.0,
        "ECG5000": 5.0,
        "StarLightCurves": 3.0,
        "CricketX": 2.0,
    }.get(job.dataset, 1.0)
    ratio_weight = max(0.25, job.ratio)
    return dataset_weight * ratio_weight * (1.5 if job.refit_full_train else 1.0)
