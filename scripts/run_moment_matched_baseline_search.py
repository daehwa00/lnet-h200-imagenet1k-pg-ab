"""Exact 18-candidate selection grid for the moment-matched synthetic task."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import cast

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_moment_matched_spectral_experiment import (  # noqa: E402
    EPSILONS,
    EPOCHS,
    MODEL_DIM,
    MODES,
    SEEDS,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    make_task,
)
from lnet.pac_balanced_hpo_campaign import (  # noqa: E402
    BalancedHPOJob,
    build_balanced_sequence_model,
)
from lnet.pac_balanced_hpo_queue import (  # noqa: E402
    BASELINE_ARCHITECTURES,
    BASELINE_WIDTHS,
    OPTIMIZER_RECIPES,
)
from lnet.pac_metrics import count_parameters  # noqa: E402
from lnet.pac_training import classification_metric_bundle, train_classifier  # noqa: E402
from lnet.pac_types import PACExperimentConfig  # noqa: E402

FAMILIES = ("s4d", "gru", "transformer")
SEARCH_SEED = 7
CONFIRMATION_SEEDS = (11, 19)
DEFAULT_ROOT = Path(".omx/results/pac-moment-matched-baseline-search-20260727")


@dataclass(frozen=True, slots=True)
class Candidate:
    family: str
    width: int
    architecture_index: int
    recipe_index: int

    @property
    def key(self) -> str:
        return f"w{self.width}-arch{self.architecture_index}-recipe{self.recipe_index}"


def candidates(family: str) -> list[Candidate]:
    return [
        Candidate(family, width, architecture_index, recipe_index)
        for width in BASELINE_WIDTHS
        for architecture_index in (1, 2)
        for recipe_index in (1, 2, 3)
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{random.randrange(1 << 30)}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _job(candidate: Candidate, epsilon: float, seed: int) -> BalancedHPOJob:
    architecture = BASELINE_ARCHITECTURES[candidate.family][candidate.architecture_index - 1]
    recipe = OPTIMIZER_RECIPES[candidate.recipe_index - 1]
    return BalancedHPOJob(
        key=f"moment:{epsilon}:{seed}:{candidate.family}:{candidate.key}",
        stage="stage1",
        suite="external",
        dataset=f"MomentMatched-eps{epsilon}",
        model=candidate.family,
        candidate_id=candidate.key,
        recipe=recipe,
        width=candidate.width,
        modes=None,
        architecture=architecture.label,
        architecture_settings=architecture.settings,
        split_seed=seed,
        train_seed=seed,
        epochs=EPOCHS,
        evaluation_split="validation",
        official_test_accessed=False,
        job_class="short",
        estimated_seconds=1.0,
    )


def _config(job: BalancedHPOJob, epsilon: float, seed: int) -> PACExperimentConfig:
    del epsilon
    return PACExperimentConfig(
        TRAIN_COUNT,
        VALIDATION_COUNT,
        0,
        128,
        raw_input_dim=1,
        output_dim=2,
        model_dim=job.width,
        modes=MODES,
        epochs=EPOCHS,
        batch_size=job.recipe.batch_size,
        learning_rate=job.recipe.learning_rate,
        weight_decay=job.recipe.weight_decay,
        grad_clip_norm=job.recipe.grad_clip_norm,
        seeds=(seed,),
        device="cuda",
        optimizer_mode="fused",
    )


def result_path(root: Path, stage: str, epsilon: float, seed: int, candidate: Candidate) -> Path:
    eps = f"{epsilon:.3f}".replace(".", "p")
    return root / stage / f"eps{eps}__seed{seed}__{candidate.family}__{candidate.key}.json"


def run_one(root: Path, stage: str, epsilon: float, seed: int, candidate: Candidate) -> None:
    output = result_path(root, stage, epsilon, seed, candidate)
    if output.exists() and output.stat().st_size:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    task = make_task(seed, epsilon)
    contract = _job(candidate, epsilon, seed)
    config = _config(contract, epsilon, seed)
    model = build_balanced_sequence_model(contract, config, 2).cuda()
    outcome = train_classifier(
        model,
        task,
        config,
        "cuda",
        seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    train = classification_metric_bundle(
        model,
        task.train_inputs.cuda(),
        task.train_labels.cuda(),
        batch_size=config.batch_size,
    )
    valid = classification_metric_bundle(
        model,
        task.validation_inputs.cuda(),
        task.validation_labels.cuda(),
        batch_size=config.batch_size,
    )
    _write_json(
        output,
        {
            "stage": stage,
            "epsilon": epsilon,
            "seed": seed,
            "candidate": asdict(candidate),
            "candidate_key": candidate.key,
            "architecture": contract.architecture,
            "recipe": asdict(contract.recipe),
            "parameters": count_parameters(model),
            "best_epoch": outcome.best_epoch,
            "train_balanced_accuracy": train.balanced_accuracy,
            "validation_balanced_accuracy": valid.balanced_accuracy,
            "official_test_accessed": False,
        },
    )


def run_stage1(root: Path, shard_index: int, shard_count: int) -> None:
    work = [
        (epsilon, family, candidate)
        for epsilon in EPSILONS
        for family in FAMILIES
        for candidate in candidates(family)
    ]
    for index, (epsilon, _family, candidate) in enumerate(work):
        if index % shard_count == shard_index:
            run_one(root, "stage1", epsilon, SEARCH_SEED, candidate)


def stage1_selection(root: Path) -> dict[str, object]:
    selected: dict[str, list[str]] = {}
    for epsilon in EPSILONS:
        for family in FAMILIES:
            rows = [
                json.loads(result_path(root, "stage1", epsilon, SEARCH_SEED, candidate).read_text())
                for candidate in candidates(family)
            ]
            ranked = sorted(
                rows,
                key=lambda row: (
                    -row["validation_balanced_accuracy"],
                    row["parameters"],
                    row["candidate_key"],
                ),
            )
            selected[f"{epsilon:.3f}:{family}"] = [
                row["candidate_key"] for row in ranked[:6]
            ]
    payload = {"top_k": 6, "selected": selected, "official_test_accessed": False}
    _write_json(root / "stage1_selection.json", payload)
    return payload


def _candidate_by_key(family: str, key: str) -> Candidate:
    return next(candidate for candidate in candidates(family) if candidate.key == key)


def run_stage2(root: Path, shard_index: int, shard_count: int) -> None:
    selection = json.loads((root / "stage1_selection.json").read_text())["selected"]
    work = []
    for epsilon in EPSILONS:
        for family in FAMILIES:
            for key in selection[f"{epsilon:.3f}:{family}"]:
                for seed in CONFIRMATION_SEEDS:
                    work.append((epsilon, seed, _candidate_by_key(family, key)))
    for index, (epsilon, seed, candidate) in enumerate(work):
        if index % shard_count == shard_index:
            run_one(root, "stage2", epsilon, seed, candidate)


def stage2_selection(root: Path) -> dict[str, object]:
    stage1 = json.loads((root / "stage1_selection.json").read_text())["selected"]
    selected = {}
    for epsilon in EPSILONS:
        for family in FAMILIES:
            ranked = []
            for key in stage1[f"{epsilon:.3f}:{family}"]:
                candidate = _candidate_by_key(family, key)
                rows = [
                    json.loads(result_path(root, "stage1", epsilon, SEARCH_SEED, candidate).read_text()),
                    *[
                        json.loads(result_path(root, "stage2", epsilon, seed, candidate).read_text())
                        for seed in CONFIRMATION_SEEDS
                    ],
                ]
                ranked.append(
                    (
                        mean(row["validation_balanced_accuracy"] for row in rows),
                        key,
                        round(median(row["best_epoch"] for row in rows)),
                    )
                )
            score, key, best_epoch = min(ranked, key=lambda item: (-item[0], item[1]))
            selected[f"{epsilon:.3f}:{family}"] = {
                "candidate_key": key,
                "selection_mean": score,
                "median_best_epoch": best_epoch,
            }
    payload = {"selected": selected, "official_test_accessed": False}
    _write_json(root / "stage2_selection.json", payload)
    return payload


def run_final(root: Path, shard_index: int, shard_count: int) -> None:
    selection = json.loads((root / "stage2_selection.json").read_text())["selected"]
    work = []
    for epsilon in EPSILONS:
        for family in FAMILIES:
            candidate = _candidate_by_key(
                family, selection[f"{epsilon:.3f}:{family}"]["candidate_key"]
            )
            for seed in SEEDS:
                work.append((epsilon, seed, candidate))
    for index, (epsilon, seed, candidate) in enumerate(work):
        if index % shard_count == shard_index:
            run_one(root, "final", epsilon, seed, candidate)


def report(root: Path) -> dict[str, object]:
    selection = json.loads((root / "stage2_selection.json").read_text())["selected"]
    summary = {}
    for epsilon in EPSILONS:
        summary[f"{epsilon:.3f}"] = {}
        for family in FAMILIES:
            candidate = _candidate_by_key(
                family, selection[f"{epsilon:.3f}:{family}"]["candidate_key"]
            )
            rows = [
                json.loads(result_path(root, "final", epsilon, seed, candidate).read_text())
                for seed in SEEDS
            ]
            train = [row["train_balanced_accuracy"] for row in rows]
            valid = [row["validation_balanced_accuracy"] for row in rows]
            summary[f"{epsilon:.3f}"][family] = {
                "candidate_key": candidate.key,
                "parameters": rows[0]["parameters"],
                "train_mean": mean(train),
                "train_sample_sd": stdev(train),
                "validation_mean": mean(valid),
                "validation_sample_sd": stdev(valid),
            }
    payload = {
        "schema": "alphabet.moment_matched_baseline_search.summary.v1",
        "search": "18 candidates; top 6 confirmed on two additional seeds",
        "summary": summary,
        "official_test_accessed": False,
    }
    _write_json(root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("stage1", "select1", "stage2", "select2", "final", "report"),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.command == "stage1":
        result = run_stage1(args.root, args.shard_index, args.shard_count)
    elif args.command == "select1":
        result = stage1_selection(args.root)
    elif args.command == "stage2":
        result = run_stage2(args.root, args.shard_index, args.shard_count)
    elif args.command == "select2":
        result = stage2_selection(args.root)
    elif args.command == "final":
        result = run_final(args.root, args.shard_index, args.shard_count)
    else:
        result = report(args.root)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
