"""Five-seed, 30-task validation screen for controlled H-compact variants."""

# pyright: reportPrivateUsage=false
# ruff: noqa: EM102, SLF001, TRY003

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Final, Literal, cast

import torch
from torch import nn

from . import pac_baseline_fairness_maximal as runner
from .pac_baseline_fairness_maximal import FairnessJob, ResourceLane
from .pac_campaign_utils import canonical_json_sha256, file_sha256, write_once
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_efp16_final_campaign import EXTERNAL_DATASETS, UCR_DATASETS
from .pac_final_validation import EXTERNAL_SECONDS, UCR_SECONDS
from .pac_h_compact_overnight_variants import (
    ALL_VARIANT_SPECS,
    CONTROL_VARIANT,
    IDENTITY_CAPACITY_VARIANT_SPECS,
    VARIANT_SPECS,
    VARIANTS,
    build_overnight_variant,
)
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from .pac_external_tasks import ExternalSelectionTask, ExternalTask
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACDevice

DEFAULT_ROOT: Final = Path(".omx/results/pac-h-compact-overnight-30task-20260722")
SEEDS: Final = (23, 31, 43, 47, 59)
TRIAL: Final = 4
STAGE: Final = "stage1"
LANE_COUNT: Final = 14


def default_lanes() -> tuple[ResourceLane, ...]:
    return tuple(
        ResourceLane(f"worker-{index:02d}", "pro6000", 0, index, 1.0) for index in range(LANE_COUNT)
    )


def _source_manifest() -> dict[str, object]:
    root = Path(__file__).resolve().parent
    names = (
        "pac_h_compact_overnight_campaign.py",
        "pac_h_compact_overnight_variants.py",
        "pac_baseline_fairness_maximal.py",
        "pac_efp_writer_reader.py",
        "pac_h_compact_lag124.py",
        "pac_h_compact_lag124_tied.py",
        "pac_headroom_efficient_models.py",
        "pac_laplace_native_input.py",
        "pac_tight_frame_models.py",
        "pac_training.py",
    )
    hashes = {name: file_sha256(root / name) for name in names}
    body: dict[str, object] = {
        "schema": "pac_h_compact_overnight_source_manifest.v1",
        "source_sha256": hashes,
        "variants": list(VARIANTS),
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def _estimated_seconds(
    suite: Literal["ucr", "external"],
    dataset: str,
    variant: str,
) -> float:
    spec = VARIANT_SPECS[variant]
    base = UCR_SECONDS[dataset] if suite == "ucr" else EXTERNAL_SECONDS[dataset]
    width_factor = max(0.75, spec.model_dim / 32.0)
    mode_factor = max(0.75, math.sqrt(spec.modes / 16.0))
    head_factor = 1.1 if spec.residual_head_width is not None else 1.0
    return base * width_factor * mode_factor * head_factor


def jobs() -> list[FairnessJob]:
    recipe = confirmatory_trial_spec("pac_tf", TRIAL)
    return [
        FairnessJob(
            stage="stage1",
            suite=cast("Literal['ucr', 'external']", suite),
            dataset=dataset,
            model=variant,
            width_tier=1,
            width=VARIANT_SPECS[variant].model_dim,
            trial=TRIAL,
            split_seed=seed,
            train_seed=seed,
            epochs=100 if suite == "ucr" else 60,
            batch_size=recipe.batch_size,
            learning_rate=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
            grad_clip_norm=recipe.grad_clip_norm,
            evaluation_split="validation",
            estimated_seconds=_estimated_seconds(
                cast("Literal['ucr', 'external']", suite), dataset, variant
            ),
            modes=VARIANT_SPECS[variant].modes,
        )
        for suite, datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS))
        for dataset in datasets
        for variant in VARIANTS
        for seed in SEEDS
    ]


def enqueue(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    active_jobs = jobs()
    loads = runner._write_manifests(root, STAGE, active_jobs, default_lanes())
    manifest = _source_manifest()
    write_once(
        root / "reports/source_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    contract: dict[str, object] = {
        "schema": "pac_h_compact_overnight_30task_contract.v1",
        "purpose": "validation-only controlled architecture screen",
        "control": CONTROL_VARIANT,
        "variants": {name: asdict(spec) for name, spec in VARIANT_SPECS.items()},
        "tasks": 30,
        "ucr_tasks": len(UCR_DATASETS),
        "external_tasks": len(EXTERNAL_DATASETS),
        "seeds": list(SEEDS),
        "optimizer_trial": TRIAL,
        "jobs": len(active_jobs),
        "selection_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "source_manifest_sha256": manifest["sha256"],
        "estimated_normalized_lane_seconds": loads,
        "failure_policy": (
            "each manifest continues after a failed job; supervisor restarts pending manifests "
            "and preserves per-attempt failure records"
        ),
    }
    write_once(
        root / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


_hooks_installed = False


def install_runner_hooks() -> None:
    """Install an isolated builder adapter without changing the shared runner source."""
    global _hooks_installed  # noqa: PLW0603
    if _hooks_installed:
        return
    original_ucr_builder = runner._build_ucr_model
    original_external_builder = runner._build_external_model
    original_experiment_config = runner._experiment_config
    original_enable = runner._enable_pac_optimized_training

    def build_ucr(job: FairnessJob, config: PACExperimentConfig, output_dim: int):  # noqa: ANN202
        if job.model in IDENTITY_CAPACITY_VARIANT_SPECS:
            from .alphabet_backbone import AlphabetBackbone  # noqa: PLC0415

            return AlphabetBackbone(config, output_dim, objective="classification")
        if job.model in ALL_VARIANT_SPECS:
            return build_overnight_variant(
                config,
                output_dim,
                job.model,
                objective="classification",
            )
        return original_ucr_builder(job, config, output_dim)

    def build_external(
        job: FairnessJob,
        config: PACExperimentConfig,
        task: ExternalTask | ExternalSelectionTask,
    ) -> nn.Module:
        if job.model in IDENTITY_CAPACITY_VARIANT_SPECS:
            from .alphabet_backbone import AlphabetBackbone  # noqa: PLC0415

            objective = "regression" if task.objective == "forecasting" else "classification"
            return AlphabetBackbone(config, task.output_dim, objective=objective)
        if job.model in ALL_VARIANT_SPECS:
            objective = "regression" if task.objective == "forecasting" else "classification"
            return build_overnight_variant(
                config,
                task.output_dim,
                job.model,
                objective=objective,
            )
        return original_external_builder(job, config, task)

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
        active_job = (
            replace(job, model="h_compact_lag124_tied")
            if job.model in ALL_VARIANT_SPECS
            else job
        )
        return original_experiment_config(
            train_count=train_count,
            validation_count=validation_count,
            test_count=test_count,
            sequence_length=sequence_length,
            input_dim=input_dim,
            output_dim=output_dim,
            job=active_job,
            device=device,
        )

    def enable_optimized(model: nn.Module, model_name: str) -> None:
        original_enable(
            model,
            "h_compact_lag124_tied" if model_name in ALL_VARIANT_SPECS else model_name,
        )

    runner._build_ucr_model = build_ucr
    runner._build_external_model = build_external
    runner._experiment_config = experiment_config
    runner._enable_pac_optimized_training = enable_optimized
    _hooks_installed = True


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: str,
    ucr_data_root: Path,
    external_data_root: Path,
) -> None:
    install_runner_hooks()
    runner.run_manifest(
        root,
        manifest,
        device=cast("PACDevice", device),
        ucr_data_root=ucr_data_root,
        external_data_root=external_data_root,
    )


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    return {
        "schema": "pac_h_compact_overnight_status.v1",
        STAGE: runner.campaign_status(root)[STAGE],
    }


def report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    snapshot = status(root)[STAGE]
    if not cast("dict[str, object]", snapshot).get("done"):
        raise RuntimeError(f"refusing to report incomplete campaign: {snapshot}")
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / STAGE / "completed").glob("*.json"))
    ]
    expected = len(jobs())
    if len(rows) != expected:
        raise RuntimeError(f"completed row mismatch: {len(rows)}/{expected}")
    by_cell: dict[tuple[str, str, str], list[float]] = {}
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("job_key", ""))
        variant = str(row.get("model", ""))
        score = row.get("selection_score")
        if (
            not key
            or key in seen
            or variant not in VARIANT_SPECS
            or row.get("evaluation_split") != "validation"
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise RuntimeError(f"invalid validation row: {key}")
        seen.add(key)
        cell = (str(row["suite"]), str(row["dataset"]), variant)
        by_cell.setdefault(cell, []).append(float(score))
    task_means: dict[str, dict[str, float]] = {variant: {} for variant in VARIANTS}
    task_sds: dict[str, dict[str, float]] = {variant: {} for variant in VARIANTS}
    for (suite, dataset, variant), scores in by_cell.items():
        if len(scores) != len(SEEDS):
            raise RuntimeError(f"{suite}/{dataset}/{variant} has {len(scores)} seeds")
        task_key = f"{suite}:{dataset}"
        task_means[variant][task_key] = mean(scores)
        task_sds[variant][task_key] = stdev(scores)
    control = task_means[CONTROL_VARIANT]
    aggregate: dict[str, object] = {}
    for variant in VARIANTS:
        means = task_means[variant]
        deltas = [means[key] - control[key] for key in sorted(control)]
        aggregate[variant] = {
            "mean_selection_score": mean(means.values()),
            "mean_delta_vs_control": mean(deltas),
            "wins_ties_losses_vs_control": {
                "wins": sum(delta > 1.0e-8 for delta in deltas),
                "ties": sum(abs(delta) <= 1.0e-8 for delta in deltas),
                "losses": sum(delta < -1.0e-8 for delta in deltas),
            },
        }
    payload: dict[str, object] = {
        "schema": "pac_h_compact_overnight_30task_report.v1",
        "control": CONTROL_VARIANT,
        "jobs": expected,
        "official_test_accessed": False,
        "aggregate": aggregate,
        "task_means": task_means,
        "task_sample_sds": task_sds,
    }
    path = root / "reports/summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def synthetic_preflight(device: str = "cpu") -> dict[str, object]:
    """Exercise classification and regression forward/backward for every variant."""
    checks: list[dict[str, object]] = []
    active_device = torch.device(device)
    for variant, spec in VARIANT_SPECS.items():
        for objective, output_dim in (("classification", 5), ("regression", 3)):
            config = PACExperimentConfig(
                8,
                2,
                0,
                33,
                raw_input_dim=2,
                output_dim=output_dim,
                model_dim=spec.model_dim,
                modes=spec.modes,
                epochs=1,
                batch_size=4,
                learning_rate=1.0e-3,
                weight_decay=1.0e-4,
                grad_clip_norm=1.0,
                seeds=(7,),
                device=cast("PACDevice", device),
            )
            model = build_overnight_variant(
                config,
                output_dim,
                variant,
                objective=cast("HeadroomObjective", objective),
            ).to(active_device)
            inputs = torch.randn(4, 33, 2, device=active_device, requires_grad=True)
            outputs = model(inputs)
            if outputs.shape != (4, output_dim) or not torch.isfinite(outputs).all():
                raise RuntimeError(f"invalid {variant}/{objective} output")
            outputs.float().square().mean().backward()
            if inputs.grad is None or not torch.isfinite(inputs.grad).all():
                raise RuntimeError(f"invalid {variant}/{objective} input gradient")
            parameter_gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not parameter_gradients or not all(
                torch.isfinite(gradient).all() for gradient in parameter_gradients
            ):
                raise RuntimeError(f"invalid {variant}/{objective} parameter gradients")
            checks.append(
                {
                    "variant": variant,
                    "objective": objective,
                    "output_shape": list(outputs.shape),
                    "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                }
            )
    return {
        "schema": "pac_h_compact_overnight_preflight.v1",
        "device": device,
        "checks": checks,
        "passed": len(checks),
    }


__all__ = [
    "DEFAULT_ROOT",
    "SEEDS",
    "default_lanes",
    "enqueue",
    "install_runner_hooks",
    "jobs",
    "report",
    "run_manifest",
    "status",
    "synthetic_preflight",
]
