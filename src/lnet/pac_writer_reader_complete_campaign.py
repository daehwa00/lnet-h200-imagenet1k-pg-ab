"""Confirmatory ten-task campaign for complete writer/reader controls.

This campaign has no architecture-selection stage.  The task set, D/M
capacity, optimizer recipe, and seeds are fixed prospectively.  Shared runner
hooks are isolated behind a unique model prefix and preserve the runner's raw
manifests, attempt events, completed rows, failures, and restart semantics.
"""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallInDefaultInitializer=false
# pyright: reportImplicitStringConcatenation=false
# pyright: reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# pyright: reportUnnecessaryCast=false
# ruff: noqa: EM101, EM102, SLF001, TRY003

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Final, Literal, cast

import torch
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp
from torch import nn

from . import pac_baseline_fairness_maximal as runner
from .pac_baseline_fairness_maximal import FairnessJob, ResourceLane
from .pac_campaign_utils import canonical_json_sha256, file_sha256, write_once
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_external_tasks import ExternalTask, load_external_task
from .pac_real_data import ensure_ucr_dataset
from .pac_types import PACExperimentConfig
from .pac_writer_reader_complete_ablation import (
    CAPACITY_MATCHED_VARIANTS,
    VARIANTS,
    CompleteWriterReaderAblationPAC,
    CompleteWriterReaderVariant,
)

if TYPE_CHECKING:
    from .pac_external_tasks import ExternalDatasetName, ExternalSelectionTask
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACDevice


DEFAULT_ROOT: Final = Path(
    ".omx/results/pac-writer-reader-complete-10task-20260724"
)
DEFAULT_REUSE_SOURCE: Final = Path(
    ".omx/results/pac-h-compact-identity-capacity-q1-final-20260722/final/completed"
)
MODEL_PREFIX: Final = "wr_complete_"
MODEL_DIM: Final = 64
MODES: Final = 16
TRIAL: Final = 4
STANDARD_SEEDS: Final = (23, 31, 43, 47, 59)
PRIMARY_EXTRA_SEEDS: Final = (61, 67, 71, 73, 79)
PRIMARY_SEEDS: Final = (*STANDARD_SEEDS, *PRIMARY_EXTRA_SEEDS)
PRIMARY_VARIANTS: Final[tuple[CompleteWriterReaderVariant, ...]] = (
    "full",
    "one_scan_writer",
)
EXACT_ALIASES: Final[dict[CompleteWriterReaderVariant, CompleteWriterReaderVariant]] = {
    "learned_poles": "full",
}
EXECUTED_VARIANTS: Final[tuple[CompleteWriterReaderVariant, ...]] = tuple(
    variant for variant in VARIANTS if variant not in EXACT_ALIASES
)


@dataclass(frozen=True, slots=True)
class CampaignTask:
    suite: Literal["ucr", "external"]
    dataset: str
    role: str


TASKS: Final[tuple[CampaignTask, ...]] = (
    CampaignTask("ucr", "GunPoint", "short, few-class time series"),
    CampaignTask("ucr", "FordA", "long univariate time series"),
    CampaignTask("ucr", "Phoneme", "many-class speech-like time series"),
    CampaignTask("ucr", "ECG200", "small ECG classification"),
    CampaignTask("external", "ettm1", "multivariate forecasting"),
    CampaignTask("external", "weather", "multivariate forecasting"),
    CampaignTask("external", "sequential-mnist", "sequential image classification"),
    CampaignTask("external", "sequential-cifar", "hard sequential image classification"),
    CampaignTask("external", "speech-commands", "speech classification"),
    CampaignTask("external", "ptb-xl", "multilabel ECG classification"),
)


CONTROL_DEFINITIONS: Final[dict[str, str]] = {
    "full": "canonical learned-pole writer and learned-pole reader",
    "learned_poles": "exact result alias of full; no duplicate training",
    "one_scan_writer": "writer only; no reader local lift or reader pole bank",
    "reader_statistics_removed": (
        "writer moments plus pooled reader-local trajectory; no reader pole statistics"
    ),
    "energy_only": (
        "shared pooled-real D plus both banks' log-energy coordinates (D+2M)"
    ),
    "log_energy_only": (
        "strict two-bank log-energy coordinates and one linear head (2M); "
        "descriptive, intentionally not capacity matched"
    ),
    "lag_only": "shared pooled-real D plus both banks' lag coordinates (D+12M)",
    "pooled_real_only": "reader-local pooled real trajectory only (D)",
    "fixed_random_poles": (
        "decay and frequency (pole locations) fixed random; frames remain learned"
    ),
    "no_semi_orthogonality": (
        "stem and modal frames are unconstrained and receive no retraction"
    ),
    "local_convolution_only": "pole-free local temporal convolution stack",
    "wider_one_scan": "writer-only graph matched by genuinely wider D and M",
    "trajectory_mlp_reader": "parameter-matched pointwise trajectory MLP reader",
    "trajectory_convolution_reader": (
        "parameter-matched trajectory convolution reader"
    ),
}


def _model_name(variant: CompleteWriterReaderVariant) -> str:
    return f"{MODEL_PREFIX}{variant}"


def _variant(model_name: str) -> CompleteWriterReaderVariant:
    value = model_name.removeprefix(MODEL_PREFIX)
    if value not in VARIANTS:
        message = f"unknown complete writer/reader model: {model_name}"
        raise ValueError(message)
    return cast("CompleteWriterReaderVariant", value)


def default_lanes(count: int = 1) -> tuple[ResourceLane, ...]:
    if count != 1:
        message = (
            "the current local_gpu inventory exposes one shared GPU; "
            "lane_count must be 1 to avoid oversubscription"
        )
        raise ValueError(message)
    return (
        ResourceLane(
            name="local_gpu-gpu0-lane00",
            host="local_gpu",
            gpu=0,
            lane=0,
            relative_speed=1.0,
        ),
    )


def _estimated_seconds(task: CampaignTask) -> float:
    table = runner.UCR_SECONDS if task.suite == "ucr" else runner.EXTERNAL_SECONDS
    return float(table.get(task.dataset, 120.0)) * 2.0


def campaign_jobs() -> list[FairnessJob]:
    """Return the immutable 750-run grid.

    ``full`` and ``one_scan_writer`` receive ten seeds.  Every other executed
    condition receives the same five standard seeds.  ``learned_poles`` is an
    exact alias of ``full`` and therefore schedules no duplicate runs.
    """

    recipe = confirmatory_trial_spec("pac_tf", TRIAL)
    jobs: list[FairnessJob] = []
    for task in TASKS:
        epochs = 100 if task.suite == "ucr" else 60
        for variant in EXECUTED_VARIANTS:
            seeds = PRIMARY_SEEDS if variant in PRIMARY_VARIANTS else STANDARD_SEEDS
            jobs.extend(
                [
                    FairnessJob(
                        stage="final",
                        suite=task.suite,
                        dataset=task.dataset,
                        model=_model_name(variant),
                        width_tier=1,
                        width=MODEL_DIM,
                        modes=MODES,
                        trial=TRIAL,
                        split_seed=seed,
                        train_seed=seed,
                        epochs=epochs,
                        batch_size=recipe.batch_size,
                        learning_rate=recipe.learning_rate,
                        weight_decay=recipe.weight_decay,
                        grad_clip_norm=recipe.grad_clip_norm,
                        evaluation_split="test",
                        estimated_seconds=_estimated_seconds(task),
                    )
                    for seed in seeds
                ]
            )
    expected = len(TASKS) * (
        len(PRIMARY_VARIANTS) * len(PRIMARY_SEEDS)
        + (len(EXECUTED_VARIANTS) - len(PRIMARY_VARIANTS))
        * len(STANDARD_SEEDS)
    )
    if len(jobs) != expected or len({job.key for job in jobs}) != expected:
        raise RuntimeError("complete writer/reader job grid is not unique and complete")
    return jobs


def _source_manifest() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    names = (
        "src/lnet/alphabet_backbone.py",
        "src/lnet/pac_writer_reader_complete_ablation.py",
        "src/lnet/pac_writer_reader_complete_campaign.py",
        "src/lnet/pac_baseline_fairness_maximal.py",
        "src/lnet/pac_training.py",
    )
    hashes = {
        name: file_sha256(project / name)
        for name in names
        if (project / name).exists()
    }
    body: dict[str, object] = {
        "schema": "pac_writer_reader_complete_source_manifest.v1",
        "source_sha256": hashes,
        "variants": list(VARIANTS),
        "executed_variants": list(EXECUTED_VARIANTS),
        "exact_aliases": EXACT_ALIASES,
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def verify_prepared_tasks(
    *,
    ucr_data_root: Path = Path(".omx/data/ucr"),
    external_data_root: Path = Path("data/external"),
) -> dict[str, object]:
    """Load every fixed task without downloads and record shape/file evidence."""

    checks: list[dict[str, object]] = []
    for task_spec in TASKS:
        if task_spec.suite == "ucr":
            task = ensure_ucr_dataset(
                task_spec.dataset,
                ucr_data_root,
                allow_download=False,
                require_train_label_space=True,
            )
            dataset_root = ucr_data_root / task_spec.dataset
            files = sorted(dataset_root.glob(f"{task_spec.dataset}_*"))
            checks.append(
                {
                    "suite": "ucr",
                    "dataset": task_spec.dataset,
                    "role": task_spec.role,
                    "train_shape": list(task.train_inputs.shape),
                    "test_shape": list(task.test_inputs.shape),
                    "output_dim": task.class_count,
                    "files": {
                        str(path): file_sha256(path)
                        for path in files
                        if path.is_file()
                    },
                    "loader_verified": True,
                }
            )
            continue
        external = load_external_task(
            cast("ExternalDatasetName", task_spec.dataset),
            external_data_root,
        )
        prepared = external_data_root / f"{task_spec.dataset}.pt"
        checks.append(
            {
                "suite": "external",
                "dataset": task_spec.dataset,
                "role": task_spec.role,
                "objective": external.objective,
                "train_shape": list(external.train_inputs.shape),
                "validation_shape": list(external.validation_inputs.shape),
                "test_shape": list(external.test_inputs.shape),
                "output_dim": external.output_dim,
                "input_encoding": external.input_encoding,
                "prepared_path": str(prepared),
                "prepared_sha256": (
                    file_sha256(prepared) if prepared.exists() else None
                ),
                "loader_verified": True,
            }
        )
        del external
    return {
        "schema": "pac_writer_reader_complete_dataset_audit.v1",
        "task_count": len(checks),
        "all_local_loaders_verified": len(checks) == len(TASKS),
        "tasks": checks,
    }


def enqueue_campaign(
    root: Path = DEFAULT_ROOT,
    *,
    lane_count: int = 1,
    verify_data: bool = True,
    ucr_data_root: Path = Path(".omx/data/ucr"),
    external_data_root: Path = Path("data/external"),
) -> dict[str, object]:
    """Write immutable manifests and contract; never starts a worker."""

    jobs = campaign_jobs()
    loads = runner._write_manifests(
        root,
        "final",
        jobs,
        default_lanes(lane_count),
    )
    source = _source_manifest()
    write_once(
        root / "reports" / "source_manifest.json",
        json.dumps(source, indent=2, sort_keys=True) + "\n",
    )
    dataset_audit = (
        verify_prepared_tasks(
            ucr_data_root=ucr_data_root,
            external_data_root=external_data_root,
        )
        if verify_data
        else {
            "schema": "pac_writer_reader_complete_dataset_audit.v1",
            "all_local_loaders_verified": False,
            "reason": "enqueue called with verify_data=False",
        }
    )
    write_once(
        root / "reports" / "dataset_audit.json",
        json.dumps(dataset_audit, indent=2, sort_keys=True) + "\n",
    )
    recipe = confirmatory_trial_spec("pac_tf", TRIAL)
    contract: dict[str, object] = {
        "schema": "pac_writer_reader_complete_contract.v1",
        "purpose": "complete capacity-matched writer/reader ablation",
        "tasks": [asdict(task) for task in TASKS],
        "task_count": len(TASKS),
        "model_dim": MODEL_DIM,
        "modes": MODES,
        "trial": TRIAL,
        "variants": list(VARIANTS),
        "executed_variants": list(EXECUTED_VARIANTS),
        "exact_aliases": EXACT_ALIASES,
        "control_definitions": CONTROL_DEFINITIONS,
        "capacity_matched_variants": list(CAPACITY_MATCHED_VARIANTS),
        "parameter_tolerance": 0.03,
        "flop_tolerance": 0.10,
        "flop_policy": (
            "major-operation estimate; structural exceptions retained explicitly"
        ),
        "standard_seeds": list(STANDARD_SEEDS),
        "primary_contrast": list(PRIMARY_VARIANTS),
        "primary_seeds": list(PRIMARY_SEEDS),
        "jobs": len(jobs),
        "optimizer_recipe": {
            "selection_budget_per_variant": 0,
            "recipe_is_fixed_before_test": True,
            "batch_size": recipe.batch_size,
            "learning_rate": recipe.learning_rate,
            "weight_decay": recipe.weight_decay,
            "grad_clip_norm": recipe.grad_clip_norm,
            "ucr_epochs": 100,
            "external_epochs": 60,
            "equal_within_every_task_across_variants": True,
        },
        "official_test_policy": (
            "fixed architecture/recipe/task registry; official TEST used only "
            "for final reporting"
        ),
        "source_manifest_sha256": source["sha256"],
        "all_local_loaders_verified": dataset_audit.get(
            "all_local_loaders_verified",
            False,
        ),
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def _capacity_metadata(
    row: dict[str, object],
    job: FairnessJob,
) -> dict[str, object]:
    objective: HeadroomObjective = (
        "regression"
        if row.get("objective") == "forecasting"
        else "classification"
    )
    config = PACExperimentConfig(
        sample_count=int(row["train_count"]),
        validation_count=int(row["validation_count"]),
        test_count=int(row["test_count"]),
        sequence_length=int(row["sequence_length"]),
        raw_input_dim=int(row["input_dim"]),
        output_dim=int(row["output_dim"]),
        model_dim=job.width,
        modes=job.modes,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=job.learning_rate,
        weight_decay=job.weight_decay,
        grad_clip_norm=job.grad_clip_norm,
        seeds=(job.train_seed,),
        device="cpu",
    )
    model = CompleteWriterReaderAblationPAC(
        config,
        int(row["output_dim"]),
        variant=_variant(job.model),
        objective=objective,
        random_seed=job.train_seed,
    )
    return asdict(model.capacity_match)


_hooks_installed = False


def install_runner_hooks() -> None:  # noqa: C901
    """Install unique-prefix hooks for both UCR and external runner paths."""

    global _hooks_installed  # noqa: PLW0603
    if _hooks_installed:
        return
    original_ucr_builder = runner._build_ucr_model
    original_external_builder = runner._build_external_model
    original_experiment_config = runner._experiment_config
    original_enable = runner._enable_pac_optimized_training
    original_run_job = runner.run_job

    def build_ucr(
        job: FairnessJob,
        config: PACExperimentConfig,
        output_dim: int,
    ) -> nn.Module:
        if job.model.startswith(MODEL_PREFIX):
            return CompleteWriterReaderAblationPAC(
                config,
                output_dim,
                variant=_variant(job.model),
                objective="classification",
                random_seed=job.train_seed,
            )
        return original_ucr_builder(job, config, output_dim)

    def build_external(
        job: FairnessJob,
        config: PACExperimentConfig,
        task: ExternalTask | ExternalSelectionTask,
    ) -> nn.Module:
        if job.model.startswith(MODEL_PREFIX):
            objective: HeadroomObjective = (
                "regression"
                if task.objective == "forecasting"
                else "classification"
            )
            return CompleteWriterReaderAblationPAC(
                config,
                task.output_dim,
                variant=_variant(job.model),
                objective=objective,
                random_seed=job.train_seed,
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

    def run_job(
        job: FairnessJob,
        *,
        device: PACDevice,
        ucr_data_root: Path = runner.UCR_DATA_ROOT,
        external_data_root: Path = runner.EXTERNAL_DATA_ROOT,
    ) -> dict[str, object]:
        row = original_run_job(
            job,
            device=device,
            ucr_data_root=ucr_data_root,
            external_data_root=external_data_root,
        )
        if job.model.startswith(MODEL_PREFIX):
            row["writer_reader_variant"] = _variant(job.model)
            row["control_definition"] = CONTROL_DEFINITIONS[
                _variant(job.model)
            ]
            row["capacity_match"] = _capacity_metadata(row, job)
        return row

    runner._build_ucr_model = build_ucr
    runner._build_external_model = build_external
    runner._experiment_config = experiment_config
    runner._enable_pac_optimized_training = enable_optimized
    runner.run_job = run_job
    _hooks_installed = True


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: str = "cuda",
    ucr_data_root: Path = Path(".omx/data/ucr"),
    external_data_root: Path = Path("data/external"),
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
    shared = cast("dict[str, object]", runner.campaign_status(root)["final"])
    return {
        "schema": "pac_writer_reader_complete_status.v1",
        "final": shared,
        "expected_jobs": len(campaign_jobs()),
        "task_count": len(TASKS),
        "executed_variant_count": len(EXECUTED_VARIANTS),
        "exact_aliases": EXACT_ALIASES,
    }


def synthetic_preflight(device: str = "cpu") -> dict[str, object]:
    active_device = torch.device(device)
    checks: list[dict[str, object]] = []
    config = PACExperimentConfig(
        8,
        2,
        0,
        33,
        raw_input_dim=2,
        output_dim=5,
        model_dim=MODEL_DIM,
        modes=MODES,
        epochs=1,
        batch_size=4,
    )
    for variant in VARIANTS:
        model = CompleteWriterReaderAblationPAC(
            config,
            5,
            variant=variant,
            random_seed=23,
        ).to(active_device)
        inputs = torch.randn(4, 33, 2, device=active_device)
        outputs = model(inputs)
        outputs.square().mean().backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            message = f"{variant} has disconnected parameters: {missing}"
            raise RuntimeError(message)
        match = model.capacity_match
        if match.parameter_match_required and abs(match.relative_error) > 0.03:
            raise RuntimeError(f"{variant} failed parameter preflight")
        checks.append(
            {
                "variant": variant,
                "control_definition": CONTROL_DEFINITIONS[variant],
                **asdict(match),
                "output_shape": list(outputs.shape),
                "all_trainable_parameters_active": True,
            }
        )
        del model, inputs, outputs
    return {
        "schema": "pac_writer_reader_complete_preflight.v1",
        "device": device,
        "checks": checks,
    }


def _paired_summary(values: list[float]) -> dict[str, object]:
    if not values:
        return {
            "pairs": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "t_statistic": None,
            "one_sided_p_value": None,
            "alternative": "full_greater_than_ablation",
        }
    average = mean(values)
    spread = stdev(values) if len(values) > 1 else 0.0
    critical = (
        float(student_t.ppf(0.975, len(values) - 1))
        if len(values) > 1
        else 0.0
    )
    half_width = critical * spread / math.sqrt(len(values))
    if spread == 0.0:
        statistic = math.inf if average > 0.0 else (
            -math.inf if average < 0.0 else 0.0
        )
        p_value = 0.0 if average > 0.0 else (1.0 if average < 0.0 else 0.5)
    else:
        test = ttest_1samp(values, popmean=0.0, alternative="greater")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    return {
        "pairs": len(values),
        "mean": average,
        "std": spread,
        "ci95_low": average - half_width,
        "ci95_high": average + half_width,
        "wins": sum(value > 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
        "losses": sum(value < 0.0 for value in values),
        "t_statistic": statistic,
        "one_sided_p_value": p_value,
        "alternative": "full_greater_than_ablation",
    }


def _metric(row: dict[str, object]) -> tuple[str, float]:
    objective = str(row["objective"])
    if objective == "forecasting":
        return "mse", float(row["mse"])
    if objective == "multilabel":
        return "macro_auprc", float(row["macro_auprc"])
    return "balanced_accuracy", float(row["balanced_accuracy"])


def _completed_rows(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "final" / "completed").glob("*.json"))
        if json.loads(path.read_text(encoding="utf-8")).get(
            "model",
            "",
        ).startswith(MODEL_PREFIX)
    ]


def build_report(
    root: Path = DEFAULT_ROOT,
    *,
    allow_partial: bool = False,
) -> dict[str, object]:
    active_status = cast("dict[str, object]", status(root)["final"])
    if not allow_partial and not active_status["done"]:
        message = (
            "campaign is incomplete: "
            f"{active_status['completed_expected']}/"
            f"{active_status['expected']} expected rows"
        )
        raise RuntimeError(message)
    rows = _completed_rows(root)
    by_cell_seed: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["suite"]),
            str(row["dataset"]),
            str(row["writer_reader_variant"]),
            int(row["train_seed"]),
        )
        by_cell_seed[key] = row

    taskwise: list[dict[str, object]] = []
    for task in TASKS:
        full_rows = {
            seed: row
            for (suite, dataset, variant, seed), row in by_cell_seed.items()
            if suite == task.suite
            and dataset == task.dataset
            and variant == "full"
        }
        for variant in VARIANTS:
            source_variant = EXACT_ALIASES.get(variant, variant)
            variant_rows = {
                seed: row
                for (suite, dataset, active_variant, seed), row in by_cell_seed.items()
                if suite == task.suite
                and dataset == task.dataset
                and active_variant == source_variant
            }
            common = sorted(set(full_rows) & set(variant_rows))
            deltas: list[float] = []
            relative_mse_reductions: list[float] = []
            metric_name: str | None = None
            for seed in common:
                full_metric_name, full_value = _metric(full_rows[seed])
                active_metric_name, active_value = _metric(variant_rows[seed])
                if full_metric_name != active_metric_name:
                    raise RuntimeError("paired rows disagree on metric")
                metric_name = full_metric_name
                if metric_name == "mse":
                    relative = (active_value - full_value) / max(
                        abs(active_value),
                        1.0e-12,
                    )
                    deltas.append(relative)
                    relative_mse_reductions.append(relative)
                else:
                    deltas.append(full_value - active_value)
            capacity_rows = [
                row.get("capacity_match")
                for row in variant_rows.values()
                if isinstance(row.get("capacity_match"), dict)
            ]
            taskwise.append(
                {
                    "suite": task.suite,
                    "dataset": task.dataset,
                    "variant": variant,
                    "source_variant": source_variant,
                    "exact_alias": variant in EXACT_ALIASES,
                    "metric": metric_name,
                    "paired_full_minus_ablation": _paired_summary(deltas),
                    "forecasting_full_relative_mse_reduction": (
                        _paired_summary(relative_mse_reductions)
                        if relative_mse_reductions
                        else None
                    ),
                    "capacity_match": (
                        capacity_rows[0] if capacity_rows else None
                    ),
                }
            )

    variant_summary: list[dict[str, object]] = []
    for variant in VARIANTS:
        cells = [
            row
            for row in taskwise
            if row["variant"] == variant
            and cast("dict[str, object]", row["paired_full_minus_ablation"])[
                "mean"
            ]
            is not None
        ]
        task_means = [
            float(
                cast("dict[str, object]", row["paired_full_minus_ablation"])[
                    "mean"
                ]
            )
            for row in cells
        ]
        variant_summary.append(
            {
                "variant": variant,
                "tasks_with_pairs": len(cells),
                "task_wins": sum(value > 0.0 for value in task_means),
                "task_ties": sum(value == 0.0 for value in task_means),
                "task_losses": sum(value < 0.0 for value in task_means),
                "mean_taskwise_full_advantage": (
                    mean(task_means) if task_means else None
                ),
            }
        )

    primary = [
        row
        for row in taskwise
        if row["variant"] == "one_scan_writer"
    ]
    primary_effects = [
        float(cast("dict[str, object]", row["paired_full_minus_ablation"])["mean"])
        for row in primary
        if cast("dict[str, object]", row["paired_full_minus_ablation"])["mean"]
        is not None
    ]
    primary_task_wins = sum(value > 0.0 for value in primary_effects)
    forecasting_primary = [
        float(
            cast(
                "dict[str, object]",
                row["forecasting_full_relative_mse_reduction"],
            )["mean"]
        )
        for row in primary
        if isinstance(row["forecasting_full_relative_mse_reduction"], dict)
        and cast(
            "dict[str, object]",
            row["forecasting_full_relative_mse_reduction"],
        )["mean"]
        is not None
    ]
    primary_verdict = {
        "mean_taskwise_full_advantage": (
            mean(primary_effects) if primary_effects else None
        ),
        "task_wins": primary_task_wins,
        "task_count": len(primary_effects),
        "criterion_mean_at_least_1pp": (
            bool(primary_effects) and mean(primary_effects) >= 0.01
        ),
        "criterion_at_least_7_of_10_task_wins": primary_task_wins >= 7,
        "primary_success": (
            bool(primary_effects)
            and (
                mean(primary_effects) >= 0.01
                or primary_task_wins >= 7
            )
        ),
        "mean_forecasting_full_relative_mse_reduction": (
            mean(forecasting_primary) if forecasting_primary else None
        ),
        "forecasting_criterion_at_least_3pct": (
            bool(forecasting_primary)
            and mean(forecasting_primary) >= 0.03
        ),
        "paired_test_direction": "one-sided: full > one_scan_writer",
    }
    parameter_checks = [
        {
            "suite": row["suite"],
            "dataset": row["dataset"],
            "variant": row["variant"],
            "relative_parameter_error": cast(
                "dict[str, object]",
                row["capacity_match"],
            )["relative_error"],
            "relative_flop_error": cast(
                "dict[str, object]",
                row["capacity_match"],
            )["relative_flop_error"],
            "flop_match_status": cast(
                "dict[str, object]",
                row["capacity_match"],
            )["flop_match_status"],
            "structural_exception": cast(
                "dict[str, object]",
                row["capacity_match"],
            )["structural_exception"],
        }
        for row in taskwise
        if isinstance(row["capacity_match"], dict)
    ]
    payload: dict[str, object] = {
        "schema": "pac_writer_reader_complete_report.v1",
        "status": active_status,
        "raw_rows": len(rows),
        "task_count": len(TASKS),
        "variants": list(VARIANTS),
        "exact_aliases": EXACT_ALIASES,
        "primary_full_vs_one_scan": primary,
        "primary_verdict": primary_verdict,
        "taskwise": taskwise,
        "variant_summary": variant_summary,
        "parameter_and_flop_checks": parameter_checks,
        "forecasting_sign_convention": (
            "positive means full has lower MSE than the ablation"
        ),
        "classification_sign_convention": (
            "positive means full has higher score than the ablation"
        ),
    }
    runner._write_json(root / "reports" / "complete_report.json", payload)
    lines = [
        "# Complete writer/reader ablation",
        "",
        f"- Raw completed rows: {len(rows)} / {len(campaign_jobs())}",
        f"- Tasks: {len(TASKS)}",
        "- Learned-poles is an exact alias of full; it was not retrained.",
        (
            "- `energy_only`: pooled real D + writer/reader log-energy "
            "(D+2M)."
        ),
        (
            "- `log_energy_only`: strict 2M linear diagnostic; deliberately "
            "not capacity matched."
        ),
        "",
        "## Taskwise win counts versus full",
        "",
        "| Variant | Wins | Ties | Losses | Tasks |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {summary['variant']} | {summary['task_wins']} | "
        f"{summary['task_ties']} | {summary['task_losses']} | "
        f"{summary['tasks_with_pairs']} |"
        for summary in variant_summary
    )
    runner._write_json(
        root / "reports" / "complete_report_manifest.json",
        {
            "schema": "pac_writer_reader_complete_report_manifest.v1",
            "json": "complete_report.json",
            "markdown": "complete_report.md",
        },
    )
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "complete_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return payload


def audit_and_import_reusable_full_rows(
    root: Path = DEFAULT_ROOT,
    *,
    source_root: Path = DEFAULT_REUSE_SOURCE,
) -> dict[str, object]:
    """Import only byte-auditable, graph-identical full rows.

    The known historical D64/M16 rows differ in canonical trainable parameter
    count, so the expected outcome for that source is an explicit rejection,
    not silent reuse.
    """

    target_jobs = {
        (job.suite, job.dataset, job.train_seed): job
        for job in campaign_jobs()
        if _variant(job.model) == "full"
        and job.train_seed in STANDARD_SEEDS
    }
    current_code_sha256 = runner._code_sha256()
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for source_path in sorted(source_root.glob("*.json")):
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes)
        key = (
            source.get("suite"),
            source.get("dataset"),
            source.get("train_seed"),
        )
        job = target_jobs.get(cast("tuple[str, str, int]", key))
        if job is None or source.get("model") != "hco_identity_d64m16":
            continue
        reasons: list[str] = []
        for field, expected in (
            ("width", MODEL_DIM),
            ("modes", MODES),
            ("trial", TRIAL),
            ("train_seed", job.train_seed),
            ("split_seed", job.split_seed),
            ("epochs", job.epochs),
            ("batch_size", job.batch_size),
            ("learning_rate", job.learning_rate),
            ("weight_decay", job.weight_decay),
            ("grad_clip_norm", job.grad_clip_norm),
            ("evaluation_split", "test"),
            ("status", "done"),
        ):
            if source.get(field) != expected:
                reasons.append(f"{field}_mismatch")
        if source.get("code_sha256") != current_code_sha256:
            reasons.append("code_identity_not_equal")
        metadata = _capacity_metadata(source, job)
        if int(source.get("params_trainable", -1)) != int(
            metadata["actual_parameters"]
        ):
            reasons.append("parameter_count_mismatch")
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if reasons:
            rejected.append(
                {
                    "source": str(source_path),
                    "source_sha256": source_sha256,
                    "target_job_key": job.key,
                    "reasons": sorted(set(reasons)),
                    "source_params": source.get("params_trainable"),
                    "expected_params": metadata["actual_parameters"],
                }
            )
            continue
        transformed = {
            **source,
            "job_key": job.key,
            "cell_key": job.cell_key,
            "config_key": job.config_key,
            **asdict(job),
            "model": job.model,
            "writer_reader_variant": "full",
            "control_definition": CONTROL_DEFINITIONS["full"],
            "capacity_match": metadata,
            "reuse_provenance": {
                "schema": "pac_writer_reader_complete_reuse.v1",
                "source_path": str(source_path),
                "source_sha256": source_sha256,
                "source_job_key": source["job_key"],
                "source_model": source["model"],
                "exact_config_fields_verified": True,
                "code_sha256_equal": True,
                "parameter_count_equal": True,
            },
        }
        destination = runner._result_path(root, job, failed=False)
        if not destination.exists():
            runner._write_json(destination, transformed)
        accepted.append(
            {
                "source": str(source_path),
                "source_sha256": source_sha256,
                "destination": str(destination),
                "target_job_key": job.key,
            }
        )
    payload: dict[str, object] = {
        "schema": "pac_writer_reader_complete_reuse_audit.v1",
        "source_root": str(source_root),
        "eligible_target_rows": len(target_jobs),
        "accepted": accepted,
        "accepted_count": len(accepted),
        "rejected": rejected,
        "rejected_count": len(rejected),
        "policy": (
            "reuse requires exact job recipe, current code identity, and exact "
            "canonical trainable parameter count"
        ),
    }
    runner._write_json(root / "reports" / "reuse_audit.json", payload)
    return payload


__all__ = [
    "CONTROL_DEFINITIONS",
    "DEFAULT_REUSE_SOURCE",
    "DEFAULT_ROOT",
    "EXACT_ALIASES",
    "EXECUTED_VARIANTS",
    "MODEL_DIM",
    "MODES",
    "PRIMARY_SEEDS",
    "STANDARD_SEEDS",
    "TASKS",
    "audit_and_import_reusable_full_rows",
    "build_report",
    "campaign_jobs",
    "enqueue_campaign",
    "install_runner_hooks",
    "run_manifest",
    "status",
    "synthetic_preflight",
    "verify_prepared_tasks",
]
