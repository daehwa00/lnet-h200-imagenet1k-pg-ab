"""Prospectively selected UCR writer/reader capacity ablation campaign."""

# pyright: reportPrivateUsage=false
# ruff: noqa: EM101, EM102, PERF401, SLF001, TRY003

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Final, cast

import torch
from torch import nn

from . import pac_baseline_fairness_maximal as runner
from .pac_baseline_fairness_maximal import FairnessJob, ResourceLane
from .pac_campaign_utils import canonical_json_sha256, file_sha256, write_once
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_efp16_final_campaign import UCR_DATASETS
from .pac_final_validation import UCR_SECONDS
from .pac_types import PACExperimentConfig
from .pac_writer_reader_capacity_ablation import (
    VARIANTS,
    WriterReaderCapacityAblationPAC,
    WriterReaderVariant,
)

if TYPE_CHECKING:
    from .pac_types import PACDevice


DEFAULT_ROOT: Final = Path(
    ".omx/results/pac-writer-reader-capacity-ablation-ucr18-20260724"
)
CAPACITY_PAIRS: Final = (
    (64, 8),
    (64, 4),
    (64, 16),
    (64, 32),
    (128, 16),
    (128, 32),
)
TRIALS: Final = (2, 4, 6)
SELECTION_SEEDS: Final = (7, 11, 19)
FINAL_SEEDS: Final = (61, 67, 71, 73, 79, 83, 89, 97, 101, 103)
MODEL_PREFIX: Final = "wr_capacity_"


def _model_name(variant: WriterReaderVariant) -> str:
    return f"{MODEL_PREFIX}{variant}"


def _variant(model_name: str) -> WriterReaderVariant:
    value = model_name.removeprefix(MODEL_PREFIX)
    if model_name == f"{MODEL_PREFIX}selection_full":
        return "full"
    if value not in VARIANTS:
        raise ValueError(f"unknown writer/reader model name: {model_name}")
    return cast("WriterReaderVariant", value)


def default_lanes(count: int = 4) -> tuple[ResourceLane, ...]:
    return tuple(
        ResourceLane(
            f"local_gpu-gpu{index // 2}-lane{index % 2:02d}",
            "local_gpu",
            index // 2,
            index % 2,
            1.0,
        )
        for index in range(count)
    )


def selection_jobs() -> list[FairnessJob]:
    jobs: list[FairnessJob] = []
    for dataset in UCR_DATASETS:
        for capacity_index, (width, modes) in enumerate(CAPACITY_PAIRS, start=1):
            for trial in TRIALS:
                recipe = confirmatory_trial_spec("pac_tf", trial)
                for seed in SELECTION_SEEDS:
                    jobs.append(
                        FairnessJob(
                            stage="stage1",
                            suite="ucr",
                            dataset=dataset,
                            model=f"{MODEL_PREFIX}selection_full",
                            width_tier=capacity_index,
                            width=width,
                            modes=modes,
                            trial=trial,
                            split_seed=seed,
                            train_seed=seed,
                            epochs=100,
                            batch_size=recipe.batch_size,
                            learning_rate=recipe.learning_rate,
                            weight_decay=recipe.weight_decay,
                            grad_clip_norm=recipe.grad_clip_norm,
                            evaluation_split="validation",
                            estimated_seconds=UCR_SECONDS[dataset]
                            * max(0.75, width / 32.0),
                        )
                    )
    return jobs


def _source_manifest() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    names = (
        "src/lnet/alphabet_backbone.py",
        "src/lnet/pac_writer_reader_capacity_ablation.py",
        "src/lnet/pac_writer_reader_capacity_campaign.py",
        "src/lnet/pac_writer_reader_capacity_cli.py",
        "src/lnet/pac_baseline_fairness_maximal.py",
        "src/lnet/pac_training.py",
    )
    hashes = {
        name: file_sha256(project / name)
        for name in names
    }
    body: dict[str, object] = {
        "schema": "pac_writer_reader_capacity_source_manifest.v1",
        "source_sha256": hashes,
        "variants": list(VARIANTS),
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def enqueue_selection(
    root: Path = DEFAULT_ROOT,
    *,
    lane_count: int = 4,
) -> dict[str, object]:
    jobs = selection_jobs()
    if len(jobs) != len(UCR_DATASETS) * len(CAPACITY_PAIRS) * len(TRIALS) * len(
        SELECTION_SEEDS
    ):
        raise RuntimeError("selection grid is incomplete")
    loads = runner._write_manifests(root, "stage1", jobs, default_lanes(lane_count))
    source = _source_manifest()
    write_once(
        root / "reports/source_manifest.json",
        json.dumps(source, indent=2, sort_keys=True) + "\n",
    )
    contract: dict[str, object] = {
        "schema": "pac_writer_reader_capacity_contract.v1",
        "purpose": "capacity-matched writer-reader ablation on UCR18",
        "selection": {
            "split": "TRAIN-derived validation",
            "capacities": list(CAPACITY_PAIRS),
            "trials": list(TRIALS),
            "seeds": list(SELECTION_SEEDS),
            "jobs": len(jobs),
            "official_test_accessed": False,
        },
        "final": {
            "variants": list(VARIANTS),
            "seeds": list(FINAL_SEEDS),
            "jobs": len(UCR_DATASETS) * len(VARIANTS) * len(FINAL_SEEDS),
            "selection_frozen_before_test": True,
        },
        "primary_contrast": ["full", "one_scan_writer"],
        "parameter_tolerance": 0.03,
        "source_manifest_sha256": source["sha256"],
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def freeze_selection(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    status = cast("dict[str, object]", runner.campaign_status(root)["stage1"])
    if not status["done"]:
        raise RuntimeError(
            "selection is incomplete: "
            f"{status['completed']}/{status['expected']} completed"
        )
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "stage1" / "completed").glob("*.json"))
    ]
    grouped: dict[tuple[str, int, int, int], list[float]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            int(row["width"]),
            int(row["modes"]),
            int(row["trial"]),
        )
        grouped.setdefault(key, []).append(float(row["selection_score"]))

    selected: dict[str, dict[str, object]] = {}
    for dataset in UCR_DATASETS:
        candidates = [
            (
                -mean(scores),
                width,
                modes,
                trial,
                scores,
            )
            for (active_dataset, width, modes, trial), scores in grouped.items()
            if active_dataset == dataset and len(scores) == len(SELECTION_SEEDS)
        ]
        if len(candidates) != len(CAPACITY_PAIRS) * len(TRIALS):
            raise RuntimeError(f"incomplete selection candidates for {dataset}")
        _, width, modes, trial, scores = min(candidates)
        selected[dataset] = {
            "width": width,
            "modes": modes,
            "trial": trial,
            "mean_validation_balanced_accuracy": mean(scores),
            "validation_scores": scores,
        }
    payload: dict[str, object] = {
        "schema": "pac_writer_reader_capacity_selection.v1",
        "selection_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "tie_break": "higher mean score, then smaller D, smaller M, smaller trial id",
        "selected": selected,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_once(root / "stage1" / "selection.json", encoded)
    # The shared final runner records this conventional path in each final row.
    write_once(root / "stage2" / "selection.json", encoded)
    return payload


def final_jobs(root: Path = DEFAULT_ROOT) -> list[FairnessJob]:
    selection = json.loads(
        (root / "stage2" / "selection.json").read_text(encoding="utf-8")
    )
    selected = cast("dict[str, dict[str, object]]", selection["selected"])
    jobs: list[FairnessJob] = []
    for dataset in UCR_DATASETS:
        choice = selected[dataset]
        width = int(choice["width"])
        modes = int(choice["modes"])
        trial = int(choice["trial"])
        recipe = confirmatory_trial_spec("pac_tf", trial)
        for variant in VARIANTS:
            for seed in FINAL_SEEDS:
                jobs.append(
                    FairnessJob(
                        stage="final",
                        suite="ucr",
                        dataset=dataset,
                        model=_model_name(variant),
                        width_tier=1,
                        width=width,
                        modes=modes,
                        trial=trial,
                        split_seed=seed,
                        train_seed=seed,
                        epochs=100,
                        batch_size=recipe.batch_size,
                        learning_rate=recipe.learning_rate,
                        weight_decay=recipe.weight_decay,
                        grad_clip_norm=recipe.grad_clip_norm,
                        evaluation_split="test",
                        estimated_seconds=UCR_SECONDS[dataset]
                        * max(0.75, width / 32.0),
                    )
                )
    return jobs


def enqueue_final(
    root: Path = DEFAULT_ROOT,
    *,
    lane_count: int = 4,
) -> dict[str, object]:
    jobs = final_jobs(root)
    loads = runner._write_manifests(root, "final", jobs, default_lanes(lane_count))
    return {
        "schema": "pac_writer_reader_capacity_final_enqueue.v1",
        "jobs": len(jobs),
        "seeds": list(FINAL_SEEDS),
        "variants": list(VARIANTS),
        "estimated_normalized_lane_seconds": loads,
    }


_hooks_installed = False


def install_runner_hooks() -> None:
    global _hooks_installed  # noqa: PLW0603
    if _hooks_installed:
        return
    original_ucr_builder = runner._build_ucr_model
    original_experiment_config = runner._experiment_config
    original_enable = runner._enable_pac_optimized_training

    def build_ucr(job: FairnessJob, config: PACExperimentConfig, output_dim: int):  # noqa: ANN202
        if job.model.startswith(MODEL_PREFIX):
            return WriterReaderCapacityAblationPAC(
                config,
                output_dim,
                variant=_variant(job.model),
                objective="classification",
            )
        return original_ucr_builder(job, config, output_dim)

    def experiment_config(
        *,
        train_count: int,
        validation_count: int,
        test_count: int,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        job: FairnessJob,
        device: PACDevice,
    ) -> PACExperimentConfig:
        active = (
            replace(job, model="h_compact_lag124_tied")
            if job.model.startswith(MODEL_PREFIX)
            else job
        )
        config = original_experiment_config(
            train_count=train_count,
            validation_count=validation_count,
            test_count=test_count,
            sequence_length=sequence_length,
            input_dim=input_dim,
            output_dim=output_dim,
            job=active,
            device=device,
        )
        return replace(config, optimizer_mode="default")

    def enable_optimized(model: nn.Module, model_name: str) -> None:
        if model_name.startswith(MODEL_PREFIX):
            return
        original_enable(model, model_name)

    runner._build_ucr_model = build_ucr
    runner._experiment_config = experiment_config
    runner._enable_pac_optimized_training = enable_optimized
    _hooks_installed = True


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: str = "cuda",
    ucr_data_root: Path = Path(".omx/data/ucr"),
) -> None:
    install_runner_hooks()
    runner.run_manifest(
        root,
        manifest,
        device=cast("PACDevice", device),
        ucr_data_root=ucr_data_root,
    )


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    shared = runner.campaign_status(root)
    return {
        "schema": "pac_writer_reader_capacity_status.v1",
        "stage1": shared["stage1"],
        "final": shared["final"],
    }


def synthetic_preflight(device: str = "cpu") -> dict[str, object]:
    active_device = torch.device(device)
    checks: list[dict[str, object]] = []
    for variant in VARIANTS:
        config = PACExperimentConfig(
            8,
            2,
            0,
            33,
            raw_input_dim=2,
            output_dim=5,
            model_dim=64,
            modes=16,
            epochs=1,
            batch_size=4,
        )
        model = WriterReaderCapacityAblationPAC(
            config,
            5,
            variant=variant,
        ).to(active_device)
        inputs = torch.randn(4, 33, 2, device=active_device, requires_grad=True)
        outputs = model(inputs)
        outputs.square().mean().backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            raise RuntimeError(f"{variant} has disconnected parameters: {missing}")
        checks.append(
            {
                "variant": variant,
                **asdict(model.capacity_match),
                "output_shape": list(outputs.shape),
            }
        )
    return {
        "schema": "pac_writer_reader_capacity_preflight.v1",
        "device": device,
        "checks": checks,
    }


__all__ = [
    "DEFAULT_ROOT",
    "FINAL_SEEDS",
    "SELECTION_SEEDS",
    "enqueue_final",
    "enqueue_selection",
    "final_jobs",
    "freeze_selection",
    "run_manifest",
    "selection_jobs",
    "status",
    "synthetic_preflight",
]
