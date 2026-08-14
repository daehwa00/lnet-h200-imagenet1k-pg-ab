"""Pre-selection queue contract for the 27-task balanced ALPHABET campaign.

This module is intentionally queue-only.  It freezes logical job identities and
the task exclusion decision without starting training or importing official
test tensors.  Execution remains blocked until the ALPHABET source snapshot and
the cross-host software environment are frozen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

from .pac_campaign_utils import write_once
from .pac_campaign_utils import canonical_json_sha256
from .pac_efp16_final_campaign import EXTERNAL_DATASETS as ORIGINAL_EXTERNAL_DATASETS
from .pac_efp16_final_campaign import UCR_DATASETS
from .pac_final_validation import EXTERNAL_SECONDS, UCR_SECONDS

DEFAULT_ROOT: Final = Path(".omx/results/alphabet-balanced-hpo-27task-20260725")

EXCLUDED_EXTERNAL_DATASETS: Final = (
    "speech-commands",
    "sequential-mnist",
    "permuted-mnist",
)
EXTERNAL_DATASETS: Final = tuple(
    dataset for dataset in ORIGINAL_EXTERNAL_DATASETS if dataset not in EXCLUDED_EXTERNAL_DATASETS
)

MODELS: Final = (
    "alphabet",
    "cnn1d",
    "tcn",
    "transformer",
    "mamba",
    "s4d",
    "s5",
    "lru",
    "gru",
    "lstm",
)
BASELINES: Final = MODELS[1:]
ALPHABET_CAPACITIES: Final = (
    (32, 8),
    (32, 16),
    (64, 16),
    (64, 32),
    (128, 16),
    (128, 32),
)
BASELINE_WIDTHS: Final = (32, 64, 128)
SEARCH_SEED: Final = 7
CONFIRMATION_SEEDS: Final = (11, 19)
FINAL_SEEDS: Final = (23, 31, 43, 47, 59)
TOP_K: Final = 6

Suite = Literal["ucr", "external"]
JobClass = Literal["short", "medium", "long"]


@dataclass(frozen=True, slots=True)
class OptimizerRecipe:
    name: Literal["A", "B", "C"]
    learning_rate: float
    weight_decay: float
    batch_size: int
    grad_clip_norm: float


OPTIMIZER_RECIPES: Final = (
    OptimizerRecipe("A", 1.0e-3, 1.0e-4, 64, 0.5),
    OptimizerRecipe("B", 3.0e-3, 1.0e-4, 64, 1.0),
    OptimizerRecipe("C", 1.0e-2, 1.0e-4, 64, 2.0),
)


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    label: str
    settings: tuple[tuple[str, int], ...]


BASELINE_ARCHITECTURES: Final[dict[str, tuple[ArchitectureSpec, ArchitectureSpec]]] = {
    "cnn1d": (
        ArchitectureSpec("depth2-kernel3", (("depth", 2), ("kernel_size", 3))),
        ArchitectureSpec("depth4-kernel5", (("depth", 4), ("kernel_size", 5))),
    ),
    "tcn": (
        ArchitectureSpec("depth3-kernel3", (("depth", 3), ("kernel_size", 3))),
        ArchitectureSpec("depth5-kernel5", (("depth", 5), ("kernel_size", 5))),
    ),
    "transformer": (
        ArchitectureSpec("depth1-heads2", (("depth", 1), ("attention_heads", 2))),
        ArchitectureSpec("depth2-heads4", (("depth", 2), ("attention_heads", 4))),
    ),
    "mamba": (
        ArchitectureSpec("state16-conv3", (("state_size", 16), ("conv_size", 3))),
        ArchitectureSpec("state32-conv4", (("state_size", 32), ("conv_size", 4))),
    ),
    "s4d": (
        ArchitectureSpec("depth1-state16", (("depth", 1), ("state_size", 16))),
        ArchitectureSpec("depth3-state16", (("depth", 3), ("state_size", 16))),
    ),
    "s5": (
        ArchitectureSpec("depth1-state16", (("depth", 1), ("state_size", 16))),
        ArchitectureSpec("depth2-state32", (("depth", 2), ("state_size", 32))),
    ),
    "lru": (
        ArchitectureSpec("depth1-state16", (("depth", 1), ("state_size", 16))),
        ArchitectureSpec("depth2-state32", (("depth", 2), ("state_size", 32))),
    ),
    "gru": (
        ArchitectureSpec("depth1-state16", (("depth", 1), ("state_size", 16))),
        ArchitectureSpec("depth2-state32", (("depth", 2), ("state_size", 32))),
    ),
    "lstm": (
        ArchitectureSpec("depth1-state16", (("depth", 1), ("state_size", 16))),
        ArchitectureSpec("depth2-state32", (("depth", 2), ("state_size", 32))),
    ),
}

MODEL_RUNTIME_FACTORS: Final = {
    "alphabet": 1.0,
    "cnn1d": 1.0,
    "tcn": 1.0,
    "transformer": 1.5,
    "mamba": 1.3,
    "s4d": 1.4,
    "s5": 1.5,
    "lru": 1.5,
    "gru": 1.3,
    "lstm": 1.5,
}


@dataclass(frozen=True, slots=True)
class Stage1Job:
    key: str
    stage: Literal["stage1"]
    suite: Suite
    dataset: str
    model: str
    candidate_id: str
    recipe: OptimizerRecipe
    width: int
    modes: int | None
    architecture: str
    architecture_settings: tuple[tuple[str, int], ...]
    split_seed: int
    train_seed: int
    epochs: int
    evaluation_split: Literal["validation"]
    official_test_accessed: Literal[False]
    job_class: JobClass
    estimated_seconds: float

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["architecture_settings"] = dict(self.architecture_settings)
        return payload


def _dataset_seconds(suite: Suite, dataset: str) -> float:
    return float(UCR_SECONDS[dataset] if suite == "ucr" else EXTERNAL_SECONDS[dataset])


def _job_class(suite: Suite, dataset: str) -> JobClass:
    seconds = _dataset_seconds(suite, dataset)
    if seconds < 100.0:
        return "short"
    if seconds <= 600.0:
        return "medium"
    return "long"


def _alphabet_jobs(suite: Suite, dataset: str) -> list[Stage1Job]:
    jobs: list[Stage1Job] = []
    base_seconds = _dataset_seconds(suite, dataset)
    for width, modes in ALPHABET_CAPACITIES:
        capacity_factor = (width / 64.0) * (0.75 + 0.25 * modes / 16.0)
        for recipe in OPTIMIZER_RECIPES:
            candidate_id = f"d{width}-m{modes}-recipe{recipe.name.lower()}"
            jobs.append(
                Stage1Job(
                    key=(
                        f"balanced-hpo:stage1:{suite}:{dataset}:alphabet:{candidate_id}:"
                        f"split{SEARCH_SEED}:seed{SEARCH_SEED}"
                    ),
                    stage="stage1",
                    suite=suite,
                    dataset=dataset,
                    model="alphabet",
                    candidate_id=candidate_id,
                    recipe=recipe,
                    width=width,
                    modes=modes,
                    architecture="radial-log-r-affine",
                    architecture_settings=(),
                    split_seed=SEARCH_SEED,
                    train_seed=SEARCH_SEED,
                    epochs=100 if suite == "ucr" else 60,
                    evaluation_split="validation",
                    official_test_accessed=False,
                    job_class=_job_class(suite, dataset),
                    estimated_seconds=base_seconds * capacity_factor,
                )
            )
    return jobs


def _baseline_jobs(suite: Suite, dataset: str, model: str) -> list[Stage1Job]:
    jobs: list[Stage1Job] = []
    base_seconds = _dataset_seconds(suite, dataset)
    for width in BASELINE_WIDTHS:
        width_factor = max(0.35, min(width / 64.0, 4.0))
        for architecture_index, architecture in enumerate(
            BASELINE_ARCHITECTURES[model],
            start=1,
        ):
            architecture_factor = 1.0 if architecture_index == 1 else 1.35
            for recipe in OPTIMIZER_RECIPES:
                candidate_id = f"w{width}-arch{architecture_index}-recipe{recipe.name.lower()}"
                jobs.append(
                    Stage1Job(
                        key=(
                            f"balanced-hpo:stage1:{suite}:{dataset}:{model}:{candidate_id}:"
                            f"split{SEARCH_SEED}:seed{SEARCH_SEED}"
                        ),
                        stage="stage1",
                        suite=suite,
                        dataset=dataset,
                        model=model,
                        candidate_id=candidate_id,
                        recipe=recipe,
                        width=width,
                        modes=None,
                        architecture=architecture.label,
                        architecture_settings=architecture.settings,
                        split_seed=SEARCH_SEED,
                        train_seed=SEARCH_SEED,
                        epochs=100 if suite == "ucr" else 60,
                        evaluation_split="validation",
                        official_test_accessed=False,
                        job_class=_job_class(suite, dataset),
                        estimated_seconds=(
                            base_seconds
                            * MODEL_RUNTIME_FACTORS[model]
                            * width_factor
                            * architecture_factor
                        ),
                    )
                )
    return jobs


def stage1_jobs() -> list[Stage1Job]:
    jobs: list[Stage1Job] = []
    registries: tuple[tuple[Suite, tuple[str, ...]], ...] = (
        ("ucr", UCR_DATASETS),
        ("external", EXTERNAL_DATASETS),
    )
    for suite, datasets in registries:
        for dataset in datasets:
            jobs.extend(_alphabet_jobs(suite, dataset))
            for model in BASELINES:
                jobs.extend(_baseline_jobs(suite, dataset, model))
    return jobs


def expected_counts() -> dict[str, int]:
    tasks = len(UCR_DATASETS) + len(EXTERNAL_DATASETS)
    return {
        "tasks": tasks,
        "models": len(MODELS),
        "stage1": tasks * len(MODELS) * 18,
        "stage2": tasks * len(MODELS) * TOP_K * len(CONFIRMATION_SEEDS),
        "final": tasks * len(MODELS) * len(FINAL_SEEDS),
    }


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def enqueue_stage1(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    jobs = stage1_jobs()
    counts = expected_counts()
    if len(jobs) != counts["stage1"]:
        message = f"stage1 queue has {len(jobs)} jobs; expected {counts['stage1']}"
        raise RuntimeError(message)
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        message = "stage1 queue contains duplicate logical keys"
        raise RuntimeError(message)

    queue_counts = {
        job_class: sum(job.job_class == job_class for job in jobs)
        for job_class in ("short", "medium", "long")
    }
    estimated_hours = {
        job_class: sum(job.estimated_seconds for job in jobs if job.job_class == job_class) / 3600.0
        for job_class in ("short", "medium", "long")
    }
    contract: dict[str, object] = {
        "schema": "pac.balanced_hpo_27task_queue.v1",
        "state": "prepared_not_released",
        "scope_decision": {
            "timing": "pre-Stage-1 and independent of campaign scores",
            "original_task_count": 30,
            "included_task_count": counts["tasks"],
            "excluded_external_datasets": list(EXCLUDED_EXTERNAL_DATASETS),
            "reason": (
                "predeclared computational-scope reduction; no result from the "
                "balanced HPO campaign was observed"
            ),
        },
        "datasets": {
            "ucr": list(UCR_DATASETS),
            "external": list(EXTERNAL_DATASETS),
        },
        "models": list(MODELS),
        "architecture_status": {
            "state": "frozen",
            "model": "ALPHABET",
            "implementation": "Alphabet",
            "allowed_post_freeze_changes": (
                "numerically exact execution/runtime optimizations only"
            ),
        },
        "alphabet": {
            "representation": "radial_log_autocorrelation",
            "capacities": [
                {"width": width, "modes": modes} for width, modes in ALPHABET_CAPACITIES
            ],
        },
        "baselines": {
            "widths": list(BASELINE_WIDTHS),
            "architectures": {
                model: [
                    {
                        "label": architecture.label,
                        "settings": dict(architecture.settings),
                    }
                    for architecture in architectures
                ]
                for model, architectures in BASELINE_ARCHITECTURES.items()
            },
        },
        "optimizer_recipes": [asdict(recipe) for recipe in OPTIMIZER_RECIPES],
        "seeds": {
            "stage1": [SEARCH_SEED],
            "stage2": list(CONFIRMATION_SEEDS),
            "final": list(FINAL_SEEDS),
        },
        "expected_logical_jobs": counts,
        "stage1_queue": {
            "jobs_by_class": queue_counts,
            "estimated_prior_worker_hours_by_class": estimated_hours,
            "estimate_policy": "queue-weight prior only; never a selection signal",
            "ordering": "longest estimated job first, stable logical key tie-break",
        },
        "execution_policy": {
            "rtx4090_processes_per_gpu": {"short": 3, "medium": 2, "long": 1},
            "rtx3080ti_processes_per_gpu": {
                "short": 1,
                "medium": 1,
                "long": 1,
                "status": "conservative until host crossover is completed",
            },
            "cpu_math_threads_per_worker": 1,
            "independent_fit_processes": True,
            "ddp": False,
            "optimized_training": {
                "alphabet": "model-provided exact-split runtime with eager fallback",
                "validated_baseline_cuda_graph": {
                    "families": ["s4d", "s5", "lru"],
                    "scope": "UCR validation fits satisfying the production shape gate",
                    "failure_policy": (
                        "restore untouched initial state and retry with fused eager training"
                    ),
                },
                "other_baselines": "fused eager training",
            },
        },
        "selection_policy": {
            "stage1": "retain top six per task-model cell using validation only",
            "stage2": ("select by mean validation score over seeds 7, 11, and 19"),
            "final": "five frozen seeds; official TEST available only here",
        },
        "test_access_policy": {
            "stage1": "forbidden",
            "stage2": "forbidden",
            "final": "allowed only after configuration freeze",
        },
        "release_blockers": [
            (
                "capture and distribute one immutable hash-addressed snapshot "
                "of the already-frozen ALPHABET implementation"
            ),
            "freeze and preflight each declared runtime profile on every assigned host",
            "complete the RTX 3080 Ti host crossover before increasing its concurrency",
        ],
        "source_anchors": {
            "balanced_hpo_plan": {
                "path": ".omx/plans/final-alphabet-balanced-hpo.md",
                "sha256": _sha256(Path(".omx/plans/final-alphabet-balanced-hpo.md")),
            },
            "queue_module": {
                "path": "src/lnet/pac_balanced_hpo_queue.py",
                "sha256": _sha256(Path(__file__)),
            },
        },
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)

    ordered = sorted(jobs, key=lambda job: (-job.estimated_seconds, job.key))
    write_once(
        root / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    write_once(
        root / "stage1" / "master.jsonl",
        "".join(json.dumps(job.payload(), sort_keys=True) + "\n" for job in ordered),
    )
    for job_class in ("short", "medium", "long"):
        class_jobs = [job for job in ordered if job.job_class == job_class]
        write_once(
            root / "stage1" / "queues" / f"{job_class}.jsonl",
            "".join(json.dumps(job.payload(), sort_keys=True) + "\n" for job in class_jobs),
        )
    write_once(
        root / "PREPARED_NOT_RELEASED.json",
        json.dumps(
            {
                "contract_sha256": contract["contract_sha256"],
                "jobs": len(jobs),
                "release_blockers": contract["release_blockers"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return contract
