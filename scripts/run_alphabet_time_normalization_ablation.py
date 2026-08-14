# ruff: noqa: EM101, EM102, T201, TRY003
"""Paired validation ablation for TRAIN-fitted irregular-time normalization."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import cast

import torch

from lnet.pac_balanced_hpo_campaign import BalancedHPOJob, run_job
from lnet.pac_balanced_hpo_queue import OptimizerRecipe
from lnet.pac_external_tasks import ExternalSelectionTask, load_external_selection_task
from lnet.pac_time_normalization import fit_characteristic_time_scale

DATASETS = ("human-activity", "ushcn-daily")
SEEDS = (7, 11, 19)


def _job(dataset: str, seed: int, *, final: bool = False) -> BalancedHPOJob:
    if dataset == "human-activity":
        candidate_id = "d32-m8-recipeb"
        recipe = OptimizerRecipe("B", 3.0e-3, 1.0e-4, 64, 1.0)
    elif dataset == "ushcn-daily":
        candidate_id = "d32-m8-recipec"
        recipe = OptimizerRecipe("C", 1.0e-2, 1.0e-4, 64, 2.0)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return BalancedHPOJob(
        key=(
            f"time-normalization:{'final' if final else 'validation'}:"
            f"{dataset}:{candidate_id}:seed{seed}"
        ),
        stage="final" if final else "stage2",
        suite="external",
        dataset=dataset,
        model="alphabet",
        candidate_id=candidate_id,
        recipe=recipe,
        width=32,
        modes=8,
        architecture="radial-log-r-affine",
        architecture_settings=(),
        split_seed=7,
        train_seed=seed,
        epochs=30,
        evaluation_split="test" if final else "validation",
        official_test_accessed=final,
        job_class="medium",
        estimated_seconds=300.0,
        microbatch_size=64,
        gradient_accumulation_steps=1,
    )


def _fit_coordinate_scale(task: ExternalSelectionTask) -> float:
    metadata = task.train_metadata
    if metadata.time_delta is None:
        raise ValueError(f"{task.name} has no time_delta")
    fit_valid = metadata.valid_mask
    if task.name == "ushcn-daily":
        if fit_valid is None:
            raise ValueError("USHCN requires a valid mask")
        fit_valid = fit_valid.clone().bool()
        for row in range(fit_valid.shape[0]):
            last = int(fit_valid[row].sum().item()) - 1
            fit_valid[row, last] = False
    return fit_characteristic_time_scale(metadata.time_delta, fit_valid)


def _relative_time_support(task: ExternalSelectionTask) -> dict[str, object]:
    metadata = task.train_metadata
    if metadata.time_delta is None or metadata.valid_mask is None:
        raise ValueError(f"{task.name} requires time_delta and valid_mask")
    delta = metadata.time_delta.squeeze(-1)
    valid = metadata.valid_mask.bool()
    if valid.ndim == 3:
        valid = valid.squeeze(-1)
    time = delta.cumsum(dim=1)
    first = torch.where(valid, time, torch.inf).amin(dim=1, keepdim=True)
    last = torch.where(valid, time, -torch.inf).amax(dim=1, keepdim=True)
    denominator = int(valid.sum())

    def fraction(active: torch.Tensor) -> float:
        return float((active & valid).sum()) / max(denominator, 1)

    return {
        "physical_moment_causal_support": {
            str(lag): fraction(time - lag >= first) for lag in (1, 2, 4)
        },
        "dilated_dwconv_offset_support": {
            str(offset): fraction(
                (time + offset >= first) & (time + offset <= last)
            )
            for offset in (-8, -4, 4, 8)
        },
    }


def _assert_paired_data(
    original: ExternalSelectionTask,
    normalized: ExternalSelectionTask,
) -> None:
    for name in (
        "train_inputs",
        "train_targets",
        "validation_inputs",
        "validation_targets",
    ):
        if not torch.equal(getattr(original, name), getattr(normalized, name)):
            raise RuntimeError(f"paired ablation changed {name}")
    for name in ("train_groups", "validation_groups"):
        if getattr(original, name) != getattr(normalized, name):
            raise RuntimeError(f"paired ablation changed {name}")
    for split in ("train", "validation"):
        left = getattr(original, f"{split}_metadata")
        right = getattr(normalized, f"{split}_metadata")
        for name in ("observation_mask", "valid_mask"):
            if not torch.equal(getattr(left, name), getattr(right, name)):
                raise RuntimeError(f"paired ablation changed {split} {name}")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "completed").glob("*.json")):
        row = cast(
            "dict[str, object]",
            json.loads(path.read_text(encoding="utf-8")),
        )
        if row.get("status") == "done":
            rows.append(row)
    return rows


def _score(
    row: dict[str, object],
    dataset: str,
    *,
    final: bool,
) -> float:
    if not final:
        return float(cast("str | int | float", row["selection_score"]))
    if dataset == "human-activity":
        return float(cast("str | int | float", row["balanced_accuracy"]))
    return -float(cast("str | int | float", row["mse"]))


def _summary(
    rows: list[dict[str, object]],
    *,
    final: bool,
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for dataset in DATASETS:
        by_condition = {
            condition: {
                int(cast("str | int", row["train_seed"])): _score(
                    row,
                    dataset,
                    final=final,
                )
                for row in rows
                if row["dataset"] == dataset and row["condition"] == condition
            }
            for condition in ("original", "train-fitted")
        }
        seeds = sorted(set(by_condition["original"]) & set(by_condition["train-fitted"]))
        differences = [
            by_condition["train-fitted"][seed] - by_condition["original"][seed]
            for seed in seeds
        ]
        comparisons[dataset] = {
            "metric": (
                "balanced_accuracy"
                if dataset == "human-activity"
                else ("negative_test_mse" if final else "negative_validation_mse")
            ),
            "higher_is_better": True,
            "paired_seeds": seeds,
            "original_scores": [by_condition["original"][seed] for seed in seeds],
            "train_fitted_scores": [
                by_condition["train-fitted"][seed] for seed in seeds
            ],
            "mean_original": mean(by_condition["original"].values()),
            "mean_train_fitted": mean(by_condition["train-fitted"].values()),
            "paired_differences": differences,
            "mean_paired_difference": mean(differences),
            "sample_sd_paired_difference": (
                stdev(differences) if len(differences) > 1 else 0.0
            ),
        }
    return {
        "schema": "alphabet.time_normalization_ablation.summary.v1",
        "selection_only": not final,
        "official_test_accessed": final,
        "datasets": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-data-root", type=Path, required=True)
    parser.add_argument("--normalized-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--final", action="store_true")
    arguments = parser.parse_args()
    run_root = (
        arguments.output_root / "final"
        if arguments.final
        else arguments.output_root
    )

    data_roots = {
        "original": arguments.original_data_root,
        "train-fitted": arguments.normalized_data_root,
    }
    contract_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        original = load_external_selection_task(dataset, data_roots["original"])
        normalized = load_external_selection_task(dataset, data_roots["train-fitted"])
        _assert_paired_data(original, normalized)
        original_scale = _fit_coordinate_scale(original)
        normalized_scale = _fit_coordinate_scale(normalized)
        if not math.isclose(normalized_scale, 1.0, rel_tol=0.0, abs_tol=2.0e-5):
            raise RuntimeError(
                f"{dataset} normalized TRAIN median is {normalized_scale}, expected 1"
            )
        contract_rows.append(
            {
                "dataset": dataset,
                "original_coordinate_train_median": original_scale,
                "normalized_coordinate_train_median": normalized_scale,
                "physical_characteristic_time_scale": (
                    normalized.characteristic_time_scale
                ),
                "original_relative_time_support": _relative_time_support(original),
                "normalized_relative_time_support": _relative_time_support(normalized),
                "train_count": int(original.train_inputs.shape[0]),
                "validation_count": int(original.validation_inputs.shape[0]),
            }
        )
    _atomic_json(
        run_root / "contract.json",
        {
            "schema": "alphabet.time_normalization_ablation.contract.v1",
            "datasets": contract_rows,
            "seeds": arguments.seeds,
            "configuration": {
                dataset: asdict(
                    _job(dataset, arguments.seeds[0], final=arguments.final)
                )
                for dataset in DATASETS
            },
            "selection_only": not arguments.final,
            "official_test_accessed": arguments.final,
        },
    )

    for dataset in DATASETS:
        for seed in arguments.seeds:
            for condition, data_root in data_roots.items():
                output = (
                    run_root
                    / "completed"
                    / f"{dataset}__{condition}__seed{seed}.json"
                )
                if output.is_file():
                    continue
                job = _job(dataset, seed, final=arguments.final)
                started = perf_counter()
                row = run_job(
                    job,
                    device=arguments.device,
                    external_data_root=data_root,
                )
                row.update(
                    {
                        "condition": condition,
                        "elapsed_seconds": perf_counter() - started,
                        "data_root": str(data_root),
                    }
                )
                _atomic_json(output, row)
                print(
                    dataset,
                    condition,
                    seed,
                    _score(row, dataset, final=arguments.final),
                    row["elapsed_seconds"],
                    flush=True,
                )
    rows = _completed_rows(run_root)
    summary = _summary(rows, final=arguments.final)
    _atomic_json(run_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
