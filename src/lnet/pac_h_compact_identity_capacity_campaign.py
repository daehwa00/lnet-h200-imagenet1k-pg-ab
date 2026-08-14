"""Validation-only five-seed sweep for six identity H-compact capacities."""

# pyright: reportPrivateUsage=false
# ruff: noqa: EM102, SLF001, TRY003

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Final, Literal, cast

import torch

from . import pac_baseline_fairness_maximal as runner
from .pac_baseline_fairness_maximal import FairnessJob
from .pac_campaign_utils import canonical_json_sha256, file_sha256, write_once
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_efp16_final_campaign import EXTERNAL_DATASETS, UCR_DATASETS
from .pac_final_validation import EXTERNAL_SECONDS, UCR_SECONDS
from .pac_h_compact_overnight_campaign import SEEDS, default_lanes, install_runner_hooks
from .pac_h_compact_overnight_variants import (
    IDENTITY_CAPACITY_VARIANT_SPECS,
    IDENTITY_CAPACITY_VARIANTS,
    build_overnight_variant,
)
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACDevice

DEFAULT_ROOT: Final = Path(
    ".omx/results/pac-h-compact-identity-capacity-validation-20260801"
)
DEFAULT_TRIALS: Final = (4,)
STAGE: Final = "final"
REGRESSION_DATASETS: Final = frozenset({"electricity", "ettm1", "ettm2", "weather"})


def _estimated_seconds(suite: Literal["ucr", "external"], dataset: str, variant: str) -> float:
    spec = IDENTITY_CAPACITY_VARIANT_SPECS[variant]
    base = UCR_SECONDS[dataset] if suite == "ucr" else EXTERNAL_SECONDS[dataset]
    return base * max(0.75, spec.model_dim / 32.0) * max(
        0.75, math.sqrt(spec.modes / 16.0)
    )


def jobs(trials: tuple[int, ...] = DEFAULT_TRIALS) -> list[FairnessJob]:
    return [
        FairnessJob(
            stage=STAGE,
            suite=cast("Literal['ucr', 'external']", suite),
            dataset=dataset,
            model=variant,
            width_tier=1,
            width=IDENTITY_CAPACITY_VARIANT_SPECS[variant].model_dim,
            trial=trial,
            split_seed=seed,
            train_seed=seed,
            epochs=100 if suite == "ucr" else 60,
            batch_size=confirmatory_trial_spec("pac_tf", trial).batch_size,
            learning_rate=confirmatory_trial_spec("pac_tf", trial).learning_rate,
            weight_decay=confirmatory_trial_spec("pac_tf", trial).weight_decay,
            grad_clip_norm=confirmatory_trial_spec("pac_tf", trial).grad_clip_norm,
            evaluation_split="validation",
            estimated_seconds=_estimated_seconds(
                cast("Literal['ucr', 'external']", suite), dataset, variant
            ),
            modes=IDENTITY_CAPACITY_VARIANT_SPECS[variant].modes,
        )
        for suite, datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS))
        for dataset in datasets
        for variant in IDENTITY_CAPACITY_VARIANTS
        for trial in trials
        for seed in SEEDS
    ]


def _source_manifest() -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    names = (
        "src/lnet/alphabet_backbone.py",
        "src/lnet/pac_h_compact_identity_capacity_campaign.py",
        "src/lnet/pac_h_compact_overnight_campaign.py",
        "src/lnet/pac_h_compact_overnight_variants.py",
        "src/lnet/pac_baseline_fairness_maximal.py",
        "src/lnet/pac_h_compact_lag124_tied.py",
        "src/lnet/pac_training.py",
        "src/lnet/pac_efp16_exact_split_training.py",
        "src/lnet/pac_triton_recurrence_op.py",
        "src/lnet/pac_recurrence.py",
        "src/lnet/pac_native_matrix_exp_vjp.py",
        "src/lnet/pac_cuda_conditional_matrix_exp.py",
        "src/lnet/pac_cuda_outer_graph.py",
    )
    hashes = {
        name: file_sha256(project_root / name) for name in names
    }
    body: dict[str, object] = {
        "schema": "pac_h_compact_identity_capacity_source_manifest.v1",
        "source_sha256": hashes,
        "variants": list(IDENTITY_CAPACITY_VARIANTS),
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def enqueue(
    root: Path = DEFAULT_ROOT,
    *,
    trials: tuple[int, ...] = DEFAULT_TRIALS,
) -> dict[str, object]:
    active_jobs = jobs(trials)
    loads = runner._write_manifests(root, STAGE, active_jobs, default_lanes())
    source = _source_manifest()
    write_once(
        root / "reports/source_manifest.json",
        json.dumps(source, indent=2, sort_keys=True) + "\n",
    )
    contract: dict[str, object] = {
        "schema": "pac_h_compact_identity_capacity_q1_final_contract.v1",
        "purpose": "validation-only sweep over six identity H-compact capacities",
        "variants": {
            name: asdict(spec) for name, spec in IDENTITY_CAPACITY_VARIANT_SPECS.items()
        },
        "tasks": len(UCR_DATASETS) + len(EXTERNAL_DATASETS),
        "ucr_tasks": len(UCR_DATASETS),
        "external_tasks": len(EXTERNAL_DATASETS),
        "seeds": list(SEEDS),
        "optimizer_trials": list(trials),
        "jobs": len(active_jobs),
        "selection_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "claim_status": "validation diagnostic; not confirmatory Q1 evidence",
        "source_manifest_sha256": source["sha256"],
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(root / "contract.json", json.dumps(contract, indent=2) + "\n")
    return contract


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
        "schema": "pac_h_compact_identity_capacity_status.v1",
        STAGE: runner.campaign_status(root)[STAGE],
    }


def _score(row: dict[str, object]) -> float:
    if row["suite"] == "ucr":
        value = row.get("balanced_accuracy")
    elif row["dataset"] == "audioset-balanced":
        value = row.get("macro_auprc")
    elif row["dataset"] == "ptb-xl":
        value = row.get("macro_auroc")
    elif row["dataset"] in REGRESSION_DATASETS:
        mse = row.get("mse")
        value = -float(mse) if isinstance(mse, (int, float)) else None
    else:
        value = row.get("accuracy")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TypeError(f"missing finite primary validation metric for {row.get('job_key')}")
    return float(value)


def report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    snapshot = cast("dict[str, object]", status(root)[STAGE])
    if not snapshot.get("done"):
        raise RuntimeError(f"refusing to report incomplete campaign: {snapshot}")
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    trials = tuple(int(value) for value in contract["optimizer_trials"])
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for path in (root / STAGE / "completed").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        score = _score(row)
        task = f"{row['suite']}:{row['dataset']}"
        grouped.setdefault((task, str(row["model"]), int(row["trial"])), []).append(float(score))
    task_means = {
        f"{variant}:t{trial}": {}
        for variant in IDENTITY_CAPACITY_VARIANTS
        for trial in trials
    }
    task_sds = {
        f"{variant}:t{trial}": {}
        for variant in IDENTITY_CAPACITY_VARIANTS
        for trial in trials
    }
    for (task, variant, trial), scores in grouped.items():
        if len(scores) != len(SEEDS):
            raise RuntimeError(f"{task}/{variant}/t{trial} has {len(scores)} seeds")
        key = f"{variant}:t{trial}"
        task_means[key][task] = mean(scores)
        task_sds[key][task] = stdev(scores)
    payload: dict[str, object] = {
        "schema": "pac_h_compact_identity_capacity_validation_report.v2",
        "jobs": len(jobs(trials)),
        "official_test_accessed": False,
        "claim_status": "validation diagnostic; taskwise winner is selected on validation",
        "mean_selection_score": {
            variant: mean(scores.values()) for variant, scores in task_means.items()
        },
        "task_means": task_means,
        "task_sample_sds": task_sds,
    }
    path = root / "reports/summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def synthetic_preflight(device: str = "cpu") -> dict[str, object]:
    checks: list[dict[str, object]] = []
    active_device = torch.device(device)
    for variant, spec in IDENTITY_CAPACITY_VARIANT_SPECS.items():
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
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError(f"invalid {variant}/{objective} parameter gradients")
            checks.append({"variant": variant, "objective": objective})
    return {
        "schema": "pac_h_compact_identity_capacity_preflight.v1",
        "device": device,
        "passed": len(checks),
        "checks": checks,
    }
