from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

import numpy as np
from scipy.stats import wilcoxon

from .pac_stiefel_variants import CORRECTED_VARIANT, REVISED_UNTIED_VARIANT, StiefelVariant
from .pac_tight_frame_models import TightFrameClassifier
from .pac_types import PACDevice, PACExperimentConfig

PROTOCOL_PATH = Path(".omx/protocols/pac_tf_confirmatory_20260711.json")
EXPLORATORY_ROOT = Path(".omx/results/pac-tf-confirmatory-evidence-20260711")
DEFAULT_ROOT = Path(".omx/results/pac-tf-confirmatory-selected-evidence-20260711")
CAPACITY_SELECTION_PATH = (
    Path(".omx/results/pac-tf-confirmatory-clean-selection-20260711")
    / "reports"
    / "stiefel_validation_capacity_selection.json"
)
SELECTION_SCHEMA = "pac_validation_capacity_selection.v1"
CONTRACT_SCHEMA = "pac_tf_selected_evidence_contract.v2"
_SELECTED_MODEL_PATTERN = re.compile(
    r"^pac_stiefel_depth2_norm_autocorr_d(?P<model_dim>[0-9]+)_m(?P<modes>[0-9]+)$"
)
CANONICAL_MECHANISM_TASKS = (
    "damped_oscillator",
    "known_damping_regime",
    "delayed_oscillation",
    "multi_mode_delayed_resonance",
    "pure_fir_negative_control",
    "context_dependent_damping",
    "random_local_pattern",
)
TEACHER_MODE_COUNTS = {
    "damped_oscillator": 1,
    "known_damping_regime": 1,
    "delayed_oscillation": 1,
    "multi_mode_delayed_resonance": 3,
    "pure_fir_negative_control": 0,
    "context_dependent_damping": 1,
    "random_local_pattern": 0,
}

EvidenceKind = Literal[
    "core_ablation",
    "mechanism_checkpoint",
    "interpretability",
    "sensitivity",
]


class ProtocolStatistics(TypedDict):
    confidence_interval: str
    paired_test: str
    multiple_comparison: str


class ConfirmatoryProtocol(TypedDict):
    protocol_id: str
    locked_before_final_evaluation: bool
    seeds: list[int]
    development_datasets: list[str]
    untouched_final_datasets: list[str]
    core_ablations: list[str]
    interpretability_interventions: list[str]
    architecture_sensitivity_factors: list[str]
    statistics: ProtocolStatistics


class JobProvenance(TypedDict):
    protocol_sha256: str
    capacity_artifact_sha256: str
    selected_model: str
    selected_model_dim: int
    selected_modes: int


class ComparisonRow(TypedDict):
    comparison: str
    baseline: str
    paired_macro_nrmse_improvement: float
    ci95_low: float
    ci95_high: float
    wilcoxon_p: float
    paired_runs: int
    inferential_units: int
    paired_observations: int
    tasks: int
    fdr_q: float
    fdr_reject_0_05: bool


class StatisticsReport(TypedDict):
    protocol_id: str
    evidence_design: Literal[
        "selected_capacity_confirmatory", "exploratory_historical"
    ]
    exploratory: bool
    source_artifact: str
    source_rows: int
    estimand: str
    confidence_interval: str
    paired_test: str
    multiple_comparison: str
    comparisons: list[ComparisonRow]


class ValidationComparison(TypedDict):
    comparison: str
    paired_balanced_accuracy_delta: float
    ci95_low: float
    ci95_high: float
    wilcoxon_p: float
    paired_runs: int
    inferential_units: int
    paired_observations: int
    datasets: int
    fdr_q: float
    fdr_reject_0_05: bool


class ValidationReport(TypedDict):
    protocol_id: str
    evaluation_split: str
    official_test_read: bool
    confidence_interval: str
    paired_test: str
    multiple_comparison: str
    core_ablation: list[ValidationComparison]
    sensitivity: list[ValidationComparison]


class _WilcoxonResult(Protocol):
    pvalue: float

_CORE_MODEL_STEMS: dict[str, str] = {
    "reference": "pac_stiefel_depth2_norm_autocorr",
    "unconstrained_projection": "pac_stiefel_ablate_unconstrained_frame",
    "forward_only": "pac_stiefel_ablate_forward_only",
    "no_local_convolution": "pac_stiefel_ablate_no_local_conv",
    "no_modal_gate": "pac_stiefel_ablate_no_mode_gate",
    "no_modal_moments": "pac_stiefel_ablate_no_modal_moments",
    "unnormalized_correlation": "pac_stiefel_ablate_unnormalized_correlation",
    "global_mean_pooling": "pac_stiefel_ablate_mean_pool",
    "untied_synthesis": "pac_stiefel_ablate_untied_synthesis_learned",
}

_STATIC_SENSITIVITY_LEVELS: dict[str, tuple[str, str, str]] = {
    "stem_kernel": ("5", "9", "13"),
    "local_kernel": ("3", "5", "9"),
    "stride": ("1", "2", "4"),
    "depth": ("1", "2", "3"),
    "moment_lags": ("1", "1,4", "1,4,8"),
    "pooling_scales": ("1", "1,2,4", "1,2,4,8"),
    "alpha_min": ("0.0001", "0.001", "0.01"),
    "alpha_max": ("1.0", "2.0", "4.0"),
    "gate_range": ("0,1", "0,2", "0,4"),
}


@dataclass(frozen=True, slots=True)
class SelectionBinding:
    protocol_sha256: str
    capacity_artifact: str
    capacity_artifact_sha256: str
    selected_model: str
    model_dim: int
    modes: int


def core_models(model_dim: int, modes: int) -> dict[str, str]:
    suffix = f"_d{model_dim}_m{modes}"
    return {name: stem + suffix for name, stem in _CORE_MODEL_STEMS.items()}


# Explicitly retained only for the already-running exploratory D64/M16 queue.
CORE_MODELS: dict[str, str] = core_models(64, 16)


def core_variant(intervention: str) -> StiefelVariant:
    changes: dict[str, dict[str, object]] = {
        "reference": {},
        "unconstrained_projection": {"frame_parameterization": "unconstrained"},
        "forward_only": {"use_backward_block": False},
        "no_local_convolution": {"use_local_convolution": False},
        "no_modal_gate": {"use_mode_gate": False},
        "no_modal_moments": {"use_modal_moments": False},
        "unnormalized_correlation": {"normalize_autocorrelation": False},
        "global_mean_pooling": {"use_ordered_pool": False},
        "untied_synthesis": {"tie_analysis_synthesis": False},
        "revised_fixed_mean_nogate_untied": {
            "use_mode_gate": False,
            "use_ordered_pool": False,
            "tie_analysis_synthesis": False,
        },
        "revised_add_mode_gate": {
            "use_ordered_pool": False,
            "tie_analysis_synthesis": False,
        },
        "revised_add_ordered_pool": {
            "use_mode_gate": False,
            "tie_analysis_synthesis": False,
        },
        "revised_tied_synthesis": {
            "use_mode_gate": False,
            "use_ordered_pool": False,
        },
    }
    try:
        return replace(CORRECTED_VARIANT, **changes[intervention])
    except KeyError as error:
        message = f"unsupported core intervention: {intervention}"
        raise KeyError(message) from error


def sensitivity_levels(model_dim: int, modes: int) -> dict[str, tuple[str, str, str]]:
    if model_dim < 4 or modes < 1 or 2 * modes > model_dim:
        message = f"invalid selected PAC-TF capacity D={model_dim}, M={modes}"
        raise ValueError(message)
    lower_dim = max(4, model_dim // 2, 2 * modes)
    upper_dim = max(model_dim + 1, model_dim * 2)
    # At the full-frame boundary D=2M there is no smaller valid D with M held
    # fixed.  Use a disclosed one-sided neighbour while keeping the selected
    # reference as the middle manifest level.
    if lower_dim >= model_dim:
        lower_dim = model_dim + max(4, model_dim // 2)
        upper_dim = model_dim * 2
    lower_modes = max(1, modes // 2)
    upper_modes = min(model_dim // 2, max(modes + 1, modes * 2))
    if upper_modes <= modes:
        lower_modes = max(1, modes // 4)
        upper_modes = max(1, modes // 2)
    return {
        **_STATIC_SENSITIVITY_LEVELS,
        "model_dim": (str(lower_dim), str(model_dim), str(upper_dim)),
        "modes": (str(lower_modes), str(modes), str(upper_modes)),
    }


# Compatibility surface for report code/tests; selected manifests use the
# per-selection mapping returned by ``sensitivity_levels``.
SENSITIVITY_LEVELS = sensitivity_levels(64, 16)


@dataclass(frozen=True, slots=True)
class EvidenceJob:
    key: str
    protocol_id: str
    kind: EvidenceKind
    seed: int
    scope: str
    intervention: str
    model: str = "pac_stiefel_depth2_norm_autocorr_d64_m16"
    level: str = "reference"
    evaluation_split: str = "validation"
    source_artifact: str = ""
    architecture_surface: str = "tight_frame_classifier"
    applicable: bool = True
    checkpoint_key: str = ""
    slots: int = 2
    protocol_sha256: str = ""
    capacity_artifact_sha256: str = ""
    selected_model: str = ""
    selected_model_dim: int = 0
    selected_modes: int = 0
    teacher_mode_index: int | None = None
    reference_level: bool = False


def full_evidence_config(
    root: Path,
    *,
    model_dim: int,
    modes: int,
    device: PACDevice = "auto",
) -> PACExperimentConfig:
    return PACExperimentConfig(
        1,
        1,
        0,
        1,
        model_dim=model_dim,
        modes=modes,
        epochs=100,
        batch_size=64,
        device=device,
        output_dir=root,
    )


def mechanism_checkpoint_config(
    root: Path,
    *,
    model_dim: int,
    modes: int,
    device: PACDevice = "auto",
) -> PACExperimentConfig:
    """Train fresh selected-capacity checkpoints for mechanism evidence."""
    return PACExperimentConfig(
        sample_count=1024,
        validation_count=256,
        test_count=256,
        sequence_length=64,
        model_dim=model_dim,
        modes=modes,
        epochs=60,
        batch_size=64,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        device=device,
        output_dir=root,
    )


def sensitivity_configuration(  # noqa: C901, PLR0912 - one branch per locked factor
    factor: str,
    level: str,
    config: PACExperimentConfig,
    *,
    base_variant: StiefelVariant = CORRECTED_VARIANT,
) -> tuple[PACExperimentConfig, StiefelVariant]:
    variant = base_variant
    active_config = config
    match factor:
        case "stem_kernel":
            variant = replace(variant, stem_kernel=int(level))
        case "local_kernel":
            variant = replace(variant, local_kernel=int(level))
        case "stride":
            variant = replace(variant, stem_stride=int(level))
        case "depth":
            variant = replace(variant, depth=int(level))
        case "model_dim":
            active_config = replace(config, model_dim=int(level))
        case "modes":
            active_config = replace(config, modes=int(level))
        case "moment_lags":
            variant = replace(
                variant,
                moment_lags=tuple(int(value) for value in level.split(",")),
            )
        case "pooling_scales":
            variant = replace(
                variant,
                pooling_scales=tuple(int(value) for value in level.split(",")),
            )
        case "alpha_min":
            variant = replace(variant, damping_min=float(level))
        case "alpha_max":
            variant = replace(variant, damping_max=float(level))
        case "gate_range":
            lower, upper = (float(value) for value in level.split(","))
            if lower != 0.0:
                message = "PAC-TF positive sigmoid gate ranges must start at zero"
                raise ValueError(message)
            variant = replace(variant, gate_max=upper)
        case _:
            message = f"unsupported sensitivity factor: {factor}"
            raise KeyError(message)
    return active_config, variant


def build_sensitivity_classifier(
    factor: str,
    level: str,
    config: PACExperimentConfig,
    class_count: int,
    *,
    revised: bool = False,
) -> TightFrameClassifier:
    active_config, variant = sensitivity_configuration(
        factor,
        level,
        config,
        base_variant=REVISED_UNTIED_VARIANT if revised else CORRECTED_VARIANT,
    )
    return TightFrameClassifier(
        active_config,
        class_count,
        variant,
        full_modal_frame=active_config.modes > active_config.model_dim // 4,
    )


def load_protocol(path: Path = PROTOCOL_PATH) -> ConfirmatoryProtocol:
    payload = cast("ConfirmatoryProtocol", json.loads(path.read_text(encoding="utf-8")))
    if not payload.get("locked_before_final_evaluation"):
        message = "confirmatory protocol must be locked before manifest generation"
        raise ValueError(message)
    return payload


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selection_binding(
    capacity_selection: Path = CAPACITY_SELECTION_PATH,
    protocol_path: Path = PROTOCOL_PATH,
) -> SelectionBinding:
    if not capacity_selection.is_file():
        message = f"complete P0 capacity artifact is missing: {capacity_selection}"
        raise FileNotFoundError(message)
    payload = json.loads(capacity_selection.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SELECTION_SCHEMA:
        message = "unexpected P0 capacity-selection schema"
        raise ValueError(message)
    if payload.get("status") != "complete":
        message = "P0 capacity selection is not complete"
        raise ValueError(message)
    if payload.get("official_test_observed") is not False:
        message = "P0 capacity selection must be TRAIN-derived and TEST-blind"
        raise ValueError(message)
    expected = payload.get("expected_jobs")
    completed = payload.get("completed_jobs")
    if not isinstance(expected, int) or expected <= 0 or completed != expected:
        message = "P0 capacity artifact does not account for every locked job"
        raise ValueError(message)
    selected_model = payload.get("selected_model")
    if not isinstance(selected_model, str):
        message = "P0 capacity artifact has no selected model"
        raise TypeError(message)
    match = _SELECTED_MODEL_PATTERN.fullmatch(selected_model)
    if match is None:
        message = f"cannot parse selected PAC-TF capacity: {selected_model}"
        raise ValueError(message)
    protocol = load_protocol(protocol_path)
    if not protocol.get("protocol_id"):
        message = "locked protocol has no protocol id"
        raise ValueError(message)
    return SelectionBinding(
        protocol_sha256=file_sha256(protocol_path),
        capacity_artifact=str(capacity_selection),
        capacity_artifact_sha256=file_sha256(capacity_selection),
        selected_model=selected_model,
        model_dim=int(match.group("model_dim")),
        modes=int(match.group("modes")),
    )


def build_evidence_jobs(
    protocol_path: Path = PROTOCOL_PATH,
    capacity_selection: Path = CAPACITY_SELECTION_PATH,
    mechanism_artifact: Path | None = None,
) -> tuple[EvidenceJob, ...]:
    protocol = load_protocol(protocol_path)
    binding = load_selection_binding(capacity_selection, protocol_path)
    protocol_id = str(protocol["protocol_id"])
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    datasets = tuple(str(name) for name in protocol["development_datasets"])
    provenance = _job_provenance(binding)
    jobs = list(_ablation_jobs(protocol_id, seeds, datasets, binding, provenance))
    tasks = (
        CANONICAL_MECHANISM_TASKS
        if mechanism_artifact is None
        else _artifact_tasks(mechanism_artifact)
    )
    jobs.extend(_mechanism_checkpoint_jobs(protocol_id, seeds, tasks, provenance))
    jobs.extend(
        _interpretability_jobs(
            protocol_id,
            seeds,
            tasks,
            datasets,
            provenance,
        )
    )
    jobs.extend(
        _sensitivity_jobs(
            protocol_id,
            seeds,
            datasets,
            binding,
            provenance,
        )
    )
    return tuple(jobs)


def _ablation_jobs(
    protocol_id: str,
    seeds: tuple[int, ...],
    datasets: tuple[str, ...],
    binding: SelectionBinding,
    provenance: JobProvenance,
) -> tuple[EvidenceJob, ...]:
    models = core_models(binding.model_dim, binding.modes)
    return tuple(
        EvidenceJob(
            f"ablation__{intervention}__{dataset}__seed{seed}",
            protocol_id,
            "core_ablation",
            seed,
            dataset,
            intervention,
            model=model,
            **provenance,
        )
        for seed in seeds
        for dataset in datasets
        for intervention, model in models.items()
    )


def _interpretability_jobs(
    protocol_id: str,
    seeds: tuple[int, ...],
    tasks: tuple[str, ...],
    datasets: tuple[str, ...],
    provenance: JobProvenance,
) -> tuple[EvidenceJob, ...]:
    recovery_interventions = (
        "teacher_frequency_damping_recovery",
        "untrained_initialization_recovery",
        "random_grid_recovery",
        "frame_subspace_perturbation",
    )
    classifier_interventions = (
        "moment_head_intervention",
        "forward_direction_removal",
        "backward_direction_removal",
        "lag1_intervention",
        "lag4_intervention",
    )
    recovery = tuple(
        EvidenceJob(
            f"interpretability__mechanism__{intervention}__{model}__{task}__seed{seed}",
            protocol_id,
            "interpretability",
            seed,
            task,
            intervention,
            model=model,
            evaluation_split="postfit_test",
            architecture_surface="causal_tight_frame_sequence_regressor",
            checkpoint_key=f"mechanism_checkpoint__{model}__{task}__seed{seed}",
            **provenance,
        )
        for seed in seeds
        for task in tasks
        for model in ("pac_tf", "pac_tf_fixed_damping")
        for intervention in recovery_interventions
    )
    knockout = tuple(
        EvidenceJob(
            (
                "interpretability__mechanism__"
                f"{intervention}__teacher{teacher_mode_index}__{model}__{task}__seed{seed}"
            ),
            protocol_id,
            "interpretability",
            seed,
            task,
            intervention,
            model=model,
            evaluation_split="postfit_test",
            architecture_surface="causal_tight_frame_sequence_regressor",
            checkpoint_key=f"mechanism_checkpoint__{model}__{task}__seed{seed}",
            teacher_mode_index=teacher_mode_index,
            **provenance,
        )
        for seed in seeds
        for task in tasks
        for model in ("pac_tf", "pac_tf_fixed_damping")
        for teacher_mode_index in range(TEACHER_MODE_COUNTS.get(task, 0))
        for intervention in ("matched_mode_knockout", "random_mode_knockout")
    )
    classifier = tuple(
        EvidenceJob(
            f"interpretability__classifier__{intervention}__{dataset}__seed{seed}",
            protocol_id,
            "interpretability",
            seed,
            dataset,
            intervention,
            model=provenance["selected_model"],
            evaluation_split="postfit_validation",
            architecture_surface="tight_frame_classifier",
            checkpoint_key=f"ablation__reference__{dataset}__seed{seed}",
            **provenance,
        )
        for seed in seeds
        for dataset in datasets
        for intervention in classifier_interventions
    )
    return recovery + knockout + classifier


def _mechanism_checkpoint_jobs(
    protocol_id: str,
    seeds: tuple[int, ...],
    tasks: tuple[str, ...],
    provenance: JobProvenance,
) -> tuple[EvidenceJob, ...]:
    return tuple(
        EvidenceJob(
            f"mechanism_checkpoint__{model}__{task}__seed{seed}",
            protocol_id,
            "mechanism_checkpoint",
            seed,
            task,
            "checkpoint_training",
            model=model,
            evaluation_split="validation_and_test",
            architecture_surface="causal_tight_frame_sequence_regressor",
            **provenance,
        )
        for seed in seeds
        for task in tasks
        for model in ("pac_tf", "pac_tf_fixed_damping")
    )


def _sensitivity_jobs(
    protocol_id: str,
    seeds: tuple[int, ...],
    datasets: tuple[str, ...],
    binding: SelectionBinding,
    provenance: JobProvenance,
) -> tuple[EvidenceJob, ...]:
    levels_by_factor = sensitivity_levels(binding.model_dim, binding.modes)
    return tuple(
        EvidenceJob(
            f"sensitivity__{factor}__{level}__{dataset}__seed{seed}",
            protocol_id,
            "sensitivity",
            seed,
            dataset,
            factor,
            model=provenance["selected_model"],
            level=level,
            reference_level=index == 1,
            **provenance,
        )
        for seed in seeds
        for dataset in datasets
        for factor, levels in levels_by_factor.items()
        for index, level in enumerate(levels)
    )


def _job_provenance(binding: SelectionBinding) -> JobProvenance:
    return {
        "protocol_sha256": binding.protocol_sha256,
        "capacity_artifact_sha256": binding.capacity_artifact_sha256,
        "selected_model": binding.selected_model,
        "selected_model_dim": binding.model_dim,
        "selected_modes": binding.modes,
    }


def write_manifests(
    root: Path = DEFAULT_ROOT,
    protocol_path: Path = PROTOCOL_PATH,
    capacity_selection: Path = CAPACITY_SELECTION_PATH,
    mechanism_artifact: Path | None = None,
) -> dict[str, int]:
    if root.resolve() == EXPLORATORY_ROOT.resolve():
        message = (
            "the D64/M16 exploratory evidence root is immutable; enqueue selected "
            f"evidence under {DEFAULT_ROOT} or another fresh root"
        )
        raise ValueError(message)
    protocol = load_protocol(protocol_path)
    binding = load_selection_binding(capacity_selection, protocol_path)
    jobs = build_evidence_jobs(protocol_path, capacity_selection, mechanism_artifact)
    root.mkdir(parents=True, exist_ok=True)
    counts = {kind: sum(job.kind == kind for job in jobs) for kind in _kinds()}
    for kind in _kinds():
        selected = (job for job in jobs if job.kind == kind)
        (root / f"{kind}_manifest.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in selected),
            encoding="utf-8",
        )
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": binding.protocol_sha256,
        "capacity_artifact": binding.capacity_artifact,
        "capacity_artifact_sha256": binding.capacity_artifact_sha256,
        "selected_model": binding.selected_model,
        "selected_model_dim": binding.model_dim,
        "selected_modes": binding.modes,
        "evidence_design": "selected_capacity_confirmatory",
        "exploratory": False,
        "mechanism_checkpoint_training": "fresh_selected_capacity_rerun",
        "mechanism_artifact_reused_without_training": False,
        "mechanism_task_inventory": list(CANONICAL_MECHANISM_TASKS),
        "sensitivity_factor_refinement": {
            "capacity": ["model_dim", "modes"],
            "damping_range": ["alpha_min", "alpha_max"],
            "reason": "one-factor selected-capacity audit; locked semantics refined, not retuned",
        },
        "sensitivity_reference_levels": {
            factor: levels[1]
            for factor, levels in sensitivity_levels(binding.model_dim, binding.modes).items()
        },
        "counts": counts,
        "statistics": protocol["statistics"],
        "execution_status": "resumable_workers_ready_no_gpu_launch",
    }
    (root / "evidence_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return counts


def validate_selected_evidence_root(  # noqa: C901, PLR0912 - fail-closed provenance audit
    root: Path = DEFAULT_ROOT,
    *,
    kind: EvidenceKind | None = None,
) -> SelectionBinding:
    contract_path = root / "evidence_contract.json"
    if not contract_path.is_file():
        message = f"selected evidence contract is missing: {contract_path}"
        raise FileNotFoundError(message)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        message = "worker refused non-selected or stale evidence contract"
        raise ValueError(message)
    if contract.get("evidence_design") != "selected_capacity_confirmatory":
        message = "worker refused exploratory evidence root"
        raise ValueError(message)
    protocol_path = Path(str(contract.get("protocol_path", "")))
    capacity_path = Path(str(contract.get("capacity_artifact", "")))
    binding = load_selection_binding(capacity_path, protocol_path)
    expected_contract = {
        "protocol_sha256": binding.protocol_sha256,
        "capacity_artifact_sha256": binding.capacity_artifact_sha256,
        "selected_model": binding.selected_model,
        "selected_model_dim": binding.model_dim,
        "selected_modes": binding.modes,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            message = f"selected evidence contract mismatch: {field}"
            raise ValueError(message)
    kinds = (kind,) if kind is not None else _kinds()
    counts = contract.get("counts")
    if not isinstance(counts, dict):
        message = "selected evidence contract has no manifest counts"
        raise TypeError(message)
    for active_kind in kinds:
        manifest = root / f"{active_kind}_manifest.jsonl"
        rows = _read_jsonl(manifest)
        if not rows or counts.get(active_kind) != len(rows):
            message = f"selected evidence manifest count mismatch: {active_kind}"
            raise ValueError(message)
        keys = [str(row.get("key", "")) for row in rows]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            message = f"selected evidence manifest keys are invalid: {active_kind}"
            raise ValueError(message)
        for row in rows:
            if row.get("kind") != active_kind:
                message = f"selected evidence manifest kind mismatch: {active_kind}"
                raise ValueError(message)
            if row.get("protocol_id") != contract.get("protocol_id"):
                message = f"selected evidence protocol id mismatch: {active_kind}"
                raise ValueError(message)
            for field, expected in expected_contract.items():
                if row.get(field) != expected:
                    message = f"selected evidence job mismatch: {active_kind}.{field}"
                    raise ValueError(message)
    return binding


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        return ()
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def paired_hierarchical_bootstrap(
    pairs: dict[str, list[float]], *, samples: int = 10_000, seed: int = 20260711
) -> tuple[float, float, float]:
    """Bootstrap a macro task effect, resampling tasks then paired seeds within task."""
    if not pairs or any(not values for values in pairs.values()):
        message = "paired hierarchical bootstrap requires non-empty task/seed effects"
        raise ValueError(message)
    task_names = tuple(sorted(pairs))
    observed = float(np.mean([np.mean(pairs[task]) for task in task_names]))
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        sampled_tasks = generator.choice(task_names, size=len(task_names), replace=True)
        task_effects = []
        for task in sampled_tasks:
            values = np.asarray(pairs[str(task)], dtype=np.float64)
            sampled = generator.choice(values, size=len(values), replace=True)
            task_effects.append(float(sampled.mean()))
        draws[draw] = float(np.mean(task_effects))
    low, high = np.quantile(draws, (0.025, 0.975))
    return observed, float(low), float(high)


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 1.0
    count = len(ordered)
    for reverse_index, (name, value) in enumerate(reversed(ordered), start=1):
        rank = count - reverse_index + 1
        running = min(running, value * count / rank)
        adjusted[name] = min(running, 1.0)
    return adjusted


def write_mechanism_statistics(
    root: Path = DEFAULT_ROOT,
    protocol_path: Path = PROTOCOL_PATH,
) -> StatisticsReport:
    """Report fresh selected-capacity mechanism evidence only."""
    _validate_selected_report_root(root, protocol_path)
    source = root / "results" / "mechanism_checkpoint.csv"
    return _write_mechanism_statistics(
        root,
        protocol_path,
        source,
        evidence_design="selected_capacity_confirmatory",
        exploratory=False,
        task_field="dataset_or_task",
        baselines=("pac_tf_fixed_damping",),
    )


def write_exploratory_mechanism_statistics(
    root: Path,
    protocol_path: Path,
    mechanism_artifact: Path,
) -> StatisticsReport:
    """Explicitly report a labelled historical exploratory artifact."""
    if not mechanism_artifact.is_file():
        message = f"exploratory mechanism artifact is missing: {mechanism_artifact}"
        raise FileNotFoundError(message)
    return _write_mechanism_statistics(
        root,
        protocol_path,
        mechanism_artifact,
        evidence_design="exploratory_historical",
        exploratory=True,
        task_field="task",
        baselines=None,
    )


def _write_mechanism_statistics(
    root: Path,
    protocol_path: Path,
    source: Path,
    *,
    evidence_design: Literal[
        "selected_capacity_confirmatory", "exploratory_historical"
    ],
    exploratory: bool,
    task_field: Literal["dataset_or_task", "task"],
    baselines: tuple[str, ...] | None,
) -> StatisticsReport:
    protocol = load_protocol(protocol_path)
    rows = _read_csv_if_present(source)
    indexed = {
        (row.get(task_field, ""), row["model"], int(row["seed"])): float(
            row["test_nrmse"]
        )
        for row in rows
        if row.get("status") == "done"
        and row.get(task_field)
        and row.get("test_nrmse")
    }
    active_baselines = (
        tuple(sorted({row["model"] for row in rows} - {"pac_tf"}))
        if baselines is None
        else baselines
    )
    comparisons: list[ComparisonRow] = []
    raw_p: dict[str, float] = {}
    for baseline in active_baselines:
        task_pairs: dict[str, list[float]] = {}
        flat: list[float] = []
        for task in sorted({row.get(task_field, "") for row in rows} - {""}):
            effects = []
            for seed in protocol["seeds"]:
                pac_key = (task, "pac_tf", int(seed))
                baseline_key = (task, baseline, int(seed))
                if pac_key in indexed and baseline_key in indexed:
                    # Positive means PAC-TF has lower NRMSE.
                    effects.append(indexed[baseline_key] - indexed[pac_key])
            if effects:
                task_pairs[task] = effects
                flat.extend(effects)
        if not task_pairs:
            continue
        effect, low, high = paired_hierarchical_bootstrap(task_pairs)
        task_means = _group_means(task_pairs)
        p_value = _wilcoxon_p(task_means)
        raw_p[baseline] = p_value
        comparisons.append(
            {
                "comparison": f"pac_tf_vs_{baseline}",
                "baseline": baseline,
                "paired_macro_nrmse_improvement": effect,
                "ci95_low": low,
                "ci95_high": high,
                "wilcoxon_p": p_value,
                "paired_runs": len(flat),
                "inferential_units": len(task_means),
                "paired_observations": len(flat),
                "tasks": len(task_pairs),
                "fdr_q": 1.0,
                "fdr_reject_0_05": False,
            }
        )
    adjusted = benjamini_hochberg(raw_p)
    for row in comparisons:
        row["fdr_q"] = adjusted[str(row["baseline"])]
        row["fdr_reject_0_05"] = bool(row["fdr_q"] <= 0.05)
    report: StatisticsReport = {
        "protocol_id": protocol["protocol_id"],
        "evidence_design": evidence_design,
        "exploratory": exploratory,
        "source_artifact": "<local-path>",
        "source_rows": len(rows),
        "estimand": "macro task mean of paired baseline NRMSE minus PAC-TF NRMSE",
        "confidence_interval": protocol["statistics"]["confidence_interval"],
        "paired_test": protocol["statistics"]["paired_test"],
        "multiple_comparison": protocol["statistics"]["multiple_comparison"],
        "comparisons": comparisons,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "mechanism_statistics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "mechanism_statistics.md").write_text(
        _statistics_markdown(report), encoding="utf-8"
    )
    return report


def write_validation_statistics(
    root: Path = DEFAULT_ROOT,
    protocol_path: Path = PROTOCOL_PATH,
) -> ValidationReport:
    _validate_selected_report_root(root, protocol_path)
    protocol = load_protocol(protocol_path)
    core = _read_csv_if_present(root / "results" / "core_ablation.csv")
    sensitivity = _read_csv_if_present(root / "results" / "sensitivity.csv")
    payload: ValidationReport = {
        "protocol_id": protocol["protocol_id"],
        "evaluation_split": "clean_stratified_validation",
        "official_test_read": False,
        "confidence_interval": protocol["statistics"]["confidence_interval"],
        "paired_test": protocol["statistics"]["paired_test"],
        "multiple_comparison": protocol["statistics"]["multiple_comparison"],
        "core_ablation": _validation_comparisons(core, family="core_ablation"),
        "sensitivity": _validation_comparisons(sensitivity, family="sensitivity"),
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "validation_statistics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "validation_statistics.md").write_text(
        _validation_markdown(payload), encoding="utf-8"
    )
    return payload


def _validation_comparisons(
    rows: list[dict[str, str]], *, family: str
) -> list[ValidationComparison]:
    complete = [
        row
        for row in rows
        if row.get("status") == "done"
        and row.get("evaluation_split") == "validation"
        and row.get("validation_balanced_accuracy")
    ]
    if not complete:
        return []
    values = {
        (_comparison_group(row, family), row["dataset_or_task"], int(row["seed"])): float(
            row["validation_balanced_accuracy"]
        )
        for row in complete
    }
    references = {
        (_reference_group(row, family), row["dataset_or_task"], int(row["seed"])): float(
            row["validation_balanced_accuracy"]
        )
        for row in complete
        if _is_reference(row, family)
    }
    groups = sorted(
        {_comparison_group(row, family) for row in complete if not _is_reference(row, family)}
    )
    output: list[ValidationComparison] = []
    raw_p: dict[str, float] = {}
    for group in groups:
        by_dataset: dict[str, list[float]] = {}
        flat: list[float] = []
        for _, dataset, seed in sorted(key for key in values if key[0] == group):
            reference_group = "reference" if family == "core_ablation" else group.split("=", 1)[0]
            if family != "core_ablation":
                reference_group = f"{reference_group}={SENSITIVITY_LEVELS[reference_group][1]}"
            reference = references.get((reference_group, dataset, seed))
            if reference is not None:
                effect = values[(group, dataset, seed)] - reference
                by_dataset.setdefault(dataset, []).append(effect)
                flat.append(effect)
        if not flat:
            continue
        effect, low, high = paired_hierarchical_bootstrap(by_dataset)
        dataset_means = _group_means(by_dataset)
        p_value = _wilcoxon_p(dataset_means)
        raw_p[group] = p_value
        output.append(
            {
                "comparison": group,
                "paired_balanced_accuracy_delta": effect,
                "ci95_low": low,
                "ci95_high": high,
                "wilcoxon_p": p_value,
                "paired_runs": len(flat),
                "inferential_units": len(dataset_means),
                "paired_observations": len(flat),
                "datasets": len(by_dataset),
                "fdr_q": 1.0,
                "fdr_reject_0_05": False,
            }
        )
    adjusted = benjamini_hochberg(raw_p)
    for row in output:
        row["fdr_q"] = adjusted[str(row["comparison"])]
        row["fdr_reject_0_05"] = bool(float(row["fdr_q"]) <= 0.05)
    return output


def _group_means(grouped_values: dict[str, list[float]]) -> list[float]:
    """Return one predeclared Wilcoxon unit per task or dataset."""
    return [float(np.mean(grouped_values[name])) for name in sorted(grouped_values)]


def _comparison_group(row: dict[str, str], family: str) -> str:
    intervention = row.get("intervention", "")
    return (
        intervention
        if family == "core_ablation"
        else f"{intervention}={row.get('level', '')}"
    )


def _reference_group(row: dict[str, str], family: str) -> str:
    if family == "core_ablation":
        return "reference"
    intervention = row["intervention"]
    return f"{intervention}={_selected_sensitivity_levels(row)[intervention][1]}"


def _is_reference(row: dict[str, str], family: str) -> bool:
    if family == "core_ablation":
        return row.get("intervention") == "reference"
    if row.get("reference_level"):
        return row.get("reference_level", "").lower() == "true"
    intervention = row.get("intervention", "")
    return row.get("level") == _selected_sensitivity_levels(row).get(
        intervention, ("", "", "")
    )[1]


def _selected_sensitivity_levels(row: dict[str, str]) -> dict[str, tuple[str, str, str]]:
    try:
        return sensitivity_levels(
            int(row.get("selected_model_dim", "64")),
            int(row.get("selected_modes", "16")),
        )
    except ValueError:
        return SENSITIVITY_LEVELS


def _read_csv_if_present(path: Path) -> list[dict[str, str]]:
    return _read_csv(path) if path.exists() else []


def _validation_markdown(payload: ValidationReport) -> str:
    lines = [
        "# PAC-TF validation-only evidence statistics",
        "",
        f"- Protocol: `{payload['protocol_id']}`",
        "- Official TEST read: no",
        "- Metric: validation balanced accuracy",
        "- Runs are paired seed runs; units are dataset means used by Wilcoxon.",
    ]
    for family in ("core_ablation", "sensitivity"):
        lines.extend(
            (
                "",
                f"## {family.replace('_', ' ').title()}",
                "",
                "| Comparison | Delta | 95% CI | Runs | Units | Wilcoxon p | BH-FDR q |",
                "|---|---:|---:|---:|---:|---:|---:|",
            )
        )
        lines.extend(
            _validation_row_markdown(row)
            for row in payload[family]
        )
    return "\n".join(lines) + "\n"


def _validation_row_markdown(row: ValidationComparison) -> str:
    return "| {} | {:.6f} | [{:.6f}, {:.6f}] | {} | {} | {:.6g} | {:.6g} |".format(
        row["comparison"],
        row["paired_balanced_accuracy_delta"],
        row["ci95_low"],
        row["ci95_high"],
        row["paired_runs"],
        row["inferential_units"],
        row["wilcoxon_p"],
        row["fdr_q"],
    )


def _validate_selected_report_root(root: Path, protocol_path: Path) -> SelectionBinding:
    binding = validate_selected_evidence_root(root)
    if file_sha256(protocol_path) != binding.protocol_sha256:
        message = "selected evidence report protocol does not match its locked contract"
        raise ValueError(message)
    return binding


def _wilcoxon_p(values: list[float]) -> float:
    if not values or all(abs(value) <= 1.0e-15 for value in values):
        return 1.0
    result = cast(
        "_WilcoxonResult",
        cast("object", wilcoxon(values, alternative="two-sided", zero_method="wilcox")),
    )
    return float(result.pvalue)


def _artifact_tasks(path: Path) -> tuple[str, ...]:
    rows = _read_csv(path)
    tasks = sorted({row["task"] for row in rows if row.get("status") == "done"})
    if not tasks:
        message = "canonical mechanism artifact contains no completed tasks"
        raise ValueError(message)
    return tuple(tasks)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _statistics_markdown(report: StatisticsReport) -> str:
    lines = [
        "# PAC-TF paired mechanism statistics",
        "",
        f"- Protocol: `{report['protocol_id']}`",
        f"- Evidence design: `{report['evidence_design']}`",
        f"- Exploratory: {'yes' if report['exploratory'] else 'no'}",
        f"- Source artifact: `{report['source_artifact']}`",
        f"- Source rows: {report['source_rows']}",
        "- Positive effect means lower NRMSE for PAC-TF.",
        "- Wilcoxon unit: one task mean across paired seed runs.",
        "",
        "| Comparison | Effect | 95% CI | Runs | Units | Wilcoxon p | BH-FDR q | Reject |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    lines.extend(
        (
            f"| PAC-TF vs {row['baseline']} | {row['paired_macro_nrmse_improvement']:.6f} "
            f"| [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] "
            f"| {row['paired_runs']} | {row['inferential_units']} "
            f"| {row['wilcoxon_p']:.6g} | {row['fdr_q']:.6g} "
            f"| {'yes' if row['fdr_reject_0_05'] else 'no'} |"
        )
        for row in report["comparisons"]
    )
    return "\n".join(lines) + "\n"


def _kinds() -> tuple[EvidenceKind, ...]:
    return (
        "core_ablation",
        "mechanism_checkpoint",
        "interpretability",
        "sensitivity",
    )
