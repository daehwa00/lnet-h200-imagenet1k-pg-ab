# ruff: noqa: SLF001
"""Corrected-path ALPHABET-only HPO for the post-selection UCR-19 extension."""

# pyright: reportPrivateLocalImportUsage=false

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final, cast

from . import pac_balanced_hpo_alphabet_27task_recovery as recovery
from .pac_balanced_hpo_campaign import BalancedHPOJob
from .pac_balanced_hpo_queue import (
    SEARCH_SEED,
    UCR_SECONDS,
    JobClass,
    _alphabet_jobs,  # pyright: ignore[reportPrivateUsage]
)
from .pac_balanced_hpo_ucr19_baselines import UCR19_DATASETS, UCR19_SECONDS

DEFAULT_ROOT: Final = Path(".omx/results/alphabet-ucr19-corrected-path-20260726")
REFERENCE_DATASET: Final = "ArrowHead"


def _job_class(seconds: float) -> str:
    if seconds < 100.0:
        return "short"
    if seconds <= 600.0:
        return "medium"
    return "long"


def _dataset_jobs(_suite: str, dataset: str) -> list[BalancedHPOJob]:
    base_seconds = UCR19_SECONDS[dataset]
    reference_seconds = float(UCR_SECONDS[REFERENCE_DATASET])
    jobs = []
    for template in _alphabet_jobs("ucr", REFERENCE_DATASET):
        candidate = replace(
            template,
            key=(
                f"balanced-hpo:stage1:ucr:{dataset}:alphabet:{template.candidate_id}:"
                f"split{SEARCH_SEED}:seed{SEARCH_SEED}"
            ),
            dataset=dataset,
            job_class=cast("JobClass", _job_class(base_seconds)),
            estimated_seconds=template.estimated_seconds * base_seconds / reference_seconds,
        )
        jobs.append(BalancedHPOJob.from_payload(candidate.payload()))
    return jobs


def _configure() -> None:
    recovery.UCR_DATASETS = UCR19_DATASETS
    recovery.EXTERNAL_DATASETS = ()
    recovery.SCHEMA_PREFIX = "pac.balanced_hpo_alphabet_ucr19"
    recovery.RELATIONSHIP_TO_PRIMARY = {
        "primary_registry_unchanged": True,
        "baseline_results_reused": True,
        "reporting": "same-table post-selection UCR-19 extension",
        "reason": "fill the reserved ALPHABET cells using the corrected frozen path",
    }
    recovery._alphabet_jobs = _dataset_jobs  # pyright: ignore[reportPrivateUsage]


def expected_counts() -> dict[str, int]:
    _configure()
    return recovery.expected_counts()


def enqueue_stage1(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    _configure()
    return recovery.enqueue_stage1(root)


def select_stage1(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    _configure()
    return recovery.select_stage1(root)


def select_stage2(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    _configure()
    return recovery.select_stage2(root)


def audit_extension(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    _configure()
    return recovery.audit_extension(root)


__all__ = [
    "DEFAULT_ROOT",
    "audit_extension",
    "enqueue_stage1",
    "expected_counts",
    "select_stage1",
    "select_stage2",
]
