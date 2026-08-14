"""Validation-only 30-task sweep for pointwise Identity ALPHABET capacities."""

# pyright: reportPrivateUsage=false
# ruff: noqa: EM101, EM102, PLC0415, SLF001, TRY003

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
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
from .pac_h_compact_overnight_campaign import SEEDS
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from .alphabet_backbone import AlphabetBackbone
    from .pac_external_tasks import ExternalSelectionTask, ExternalTask
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACDevice


@dataclass(frozen=True, slots=True)
class CapacitySpec:
    model_dim: int
    modes: int


DEFAULT_ROOT: Final = Path(
    ".omx/results/pac-pointwise-identity-capacity-validation-20260801"
)
CAPACITY_PAIRS: Final = (
    (64, 8),
    (64, 4),
    (64, 16),
    (64, 32),
    (128, 16),
    (128, 32),
)
CAPACITY_SPECS: Final = {
    f"pointwise_identity_d{model_dim}m{modes}": CapacitySpec(model_dim, modes)
    for model_dim, modes in CAPACITY_PAIRS
}
VARIANTS: Final = tuple(CAPACITY_SPECS)
TRIALS: Final = (2, 4, 6)
STAGE: Final = "final"
REGRESSION_DATASETS: Final = frozenset({"electricity", "ettm1", "ettm2", "weather"})


def default_lanes() -> tuple[ResourceLane, ...]:
    """Logical lanes; fixed-shape redistribution assigns physical 8/4/2 workers."""
    return tuple(
        ResourceLane(f"worker-{index:02d}", "pro6000", 0, index, 1.0)
        for index in range(14)
    )


def _estimated_seconds(
    suite: Literal["ucr", "external"], dataset: str, variant: str
) -> float:
    spec = CAPACITY_SPECS[variant]
    base = UCR_SECONDS[dataset] if suite == "ucr" else EXTERNAL_SECONDS[dataset]
    return base * max(0.75, spec.model_dim / 32.0) * max(
        0.75, math.sqrt(spec.modes / 16.0)
    )


def jobs() -> list[FairnessJob]:
    return [
        FairnessJob(
            stage=STAGE,
            suite=cast("Literal['ucr', 'external']", suite),
            dataset=dataset,
            model=variant,
            width_tier=capacity_index,
            width=CAPACITY_SPECS[variant].model_dim,
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
            modes=CAPACITY_SPECS[variant].modes,
        )
        for suite, datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS))
        for dataset in datasets
        for capacity_index, variant in enumerate(VARIANTS, start=1)
        for trial in TRIALS
        for seed in SEEDS
    ]


def _source_manifest() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    names = (
        "src/lnet/alphabet_backbone.py",
        "src/lnet/pac_pointwise_identity_capacity_campaign.py",
        "src/lnet/pac_pointwise_identity_capacity_cli.py",
        "src/lnet/pac_baseline_fairness_maximal.py",
        "src/lnet/pac_external_benchmarks.py",
        "src/lnet/pac_external_tasks.py",
        "src/lnet/pac_training.py",
        "src/lnet/pac_h_compact_lag124_tied.py",
        "src/lnet/pac_tight_frame_models.py",
        "src/lnet/pac_efp16_exact_split_training.py",
        "src/lnet/pac_recurrence.py",
        "src/lnet/pac_triton_recurrence_op.py",
        "src/lnet/pac_triton_parallel_static_recurrence.py",
        "src/lnet/pac_triton_parallel_static_recurrence_lag124_training.py",
        "src/lnet/pac_triton_direct_stem_training.py",
        "src/lnet/pac_triton_terminal_reader_local_training.py",
        "src/lnet/pac_triton_terminal_reader_scan_training.py",
        "src/lnet/pac_triton_writer_reader_local_training.py",
        "src/lnet/pac_native_matrix_exp_vjp.py",
        "src/lnet/pac_cuda_conditional_matrix_exp.py",
        "src/lnet/pac_cuda_fused_optimizer.py",
        "src/lnet/pac_cuda_fused_optimizer_runtime.py",
        "src/lnet/pac_cuda_outer_graph.py",
    )
    hashes = {
        name: file_sha256(project / name)
        for name in names
    }
    body: dict[str, object] = {
        "schema": "pac_pointwise_identity_capacity_source_manifest.v1",
        "source_sha256": hashes,
        "variants": list(VARIANTS),
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def enqueue(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    active_jobs = jobs()
    if len(active_jobs) != 2_700 or len({job.key for job in active_jobs}) != 2_700:
        raise RuntimeError("sealed pointwise Q1-final grid must contain 2,700 unique jobs")
    loads = runner._write_manifests(root, STAGE, active_jobs, default_lanes())
    source = _source_manifest()
    write_once(
        root / "reports/source_manifest.json",
        json.dumps(source, indent=2, sort_keys=True) + "\n",
    )
    contract: dict[str, object] = {
        "schema": "pac_pointwise_identity_capacity_q1_final_contract.v1",
        "purpose": "validation-only sweep over pointwise Identity capacities",
        "variants": {name: asdict(spec) for name, spec in CAPACITY_SPECS.items()},
        "tasks": len(UCR_DATASETS) + len(EXTERNAL_DATASETS),
        "ucr_tasks": len(UCR_DATASETS),
        "external_tasks": len(EXTERNAL_DATASETS),
        "seeds": list(SEEDS),
        "optimizer_trials": list(TRIALS),
        "jobs": len(active_jobs),
        "selection_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "claim_status": "validation diagnostic; not confirmatory Q1 evidence",
        "source_manifest_sha256": source["sha256"],
        "physical_worker_policy": {"pro6000": 8, "local_gpu": 4, "secondary_gpu": 2},
        "estimated_normalized_lane_seconds": loads,
        "failure_policy": (
            "fixed-shape groups are restart-safe; completed rows are immutable; "
            "lane supervisors retry infrastructure/job failures without discarding progress"
        ),
    }
    write_once(root / "contract.json", json.dumps(contract, indent=2) + "\n")
    return contract


_hooks_installed = False


def _build_candidate(
    config: PACExperimentConfig,
    output_dim: int,
    *,
    objective: HeadroomObjective,
) -> AlphabetBackbone:
    """Build the canonical optimized direct-stem Identity ALPHABET.

    The historical ``PointwiseAlphabetBackbone`` subclass duplicated the
    same high-level stem but bypassed the canonical class' fused-stem type
    dispatch.  Keeping one production class prevents the Q1 campaign from
    silently falling back to the generic PyTorch stem.
    """
    from .alphabet_backbone import AlphabetBackbone

    return AlphabetBackbone(config, output_dim, objective=objective)


def install_runner_hooks() -> None:
    """Install the pointwise builder while retaining the verified Identity runtime."""
    global _hooks_installed  # noqa: PLW0603
    if _hooks_installed:
        return
    original_ucr_builder = runner._build_ucr_model
    original_external_builder = runner._build_external_model
    original_experiment_config = runner._experiment_config
    original_enable = runner._enable_pac_optimized_training

    def build_ucr(job: FairnessJob, config: PACExperimentConfig, output_dim: int):  # noqa: ANN202
        if job.model in CAPACITY_SPECS:
            return _build_candidate(config, output_dim, objective="classification")
        return original_ucr_builder(job, config, output_dim)

    def build_external(
        job: FairnessJob,
        config: PACExperimentConfig,
        task: ExternalTask | ExternalSelectionTask,
    ) -> nn.Module:
        if job.model in CAPACITY_SPECS:
            objective = "regression" if task.objective == "forecasting" else "classification"
            return _build_candidate(config, task.output_dim, objective=objective)
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
        active = replace(job, model="h_compact_lag124_tied") if job.model in CAPACITY_SPECS else job
        return original_experiment_config(
            train_count=train_count,
            validation_count=validation_count,
            test_count=test_count,
            sequence_length=sequence_length,
            input_dim=input_dim,
            output_dim=output_dim,
            job=active,
            device=device,
        )

    def enable_optimized(model: nn.Module, model_name: str) -> None:
        original_enable(
            model,
            "h_compact_lag124_tied" if model_name in CAPACITY_SPECS else model_name,
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
        "schema": "pac_pointwise_identity_capacity_status.v1",
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
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for path in (root / STAGE / "completed").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        task = f"{row['suite']}:{row['dataset']}"
        grouped.setdefault((task, str(row["model"]), int(row["trial"])), []).append(
            _score(row)
        )
    task_means = {f"{variant}:t{trial}": {} for variant in VARIANTS for trial in TRIALS}
    task_sds = {f"{variant}:t{trial}": {} for variant in VARIANTS for trial in TRIALS}
    for (task, variant, trial), scores in grouped.items():
        if len(scores) != len(SEEDS):
            raise RuntimeError(f"{task}/{variant}/t{trial} has {len(scores)} seeds")
        key = f"{variant}:t{trial}"
        task_means[key][task] = mean(scores)
        task_sds[key][task] = stdev(scores)
    payload: dict[str, object] = {
        "schema": "pac_pointwise_identity_capacity_validation_report.v2",
        "jobs": len(jobs()),
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
    active_device = torch.device(device)
    checks: list[dict[str, object]] = []
    for variant, spec in CAPACITY_SPECS.items():
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
            model = _build_candidate(
                config,
                output_dim,
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
            model.post_optimizer_step()
            checks.append({"variant": variant, "objective": objective})
    return {
        "schema": "pac_pointwise_identity_capacity_preflight.v1",
        "device": device,
        "passed": len(checks),
        "jobs": len(jobs()),
        "checks": checks,
    }


__all__ = [
    "CAPACITY_PAIRS",
    "CAPACITY_SPECS",
    "DEFAULT_ROOT",
    "TRIALS",
    "VARIANTS",
    "enqueue",
    "jobs",
    "report",
    "run_manifest",
    "status",
    "synthetic_preflight",
]
