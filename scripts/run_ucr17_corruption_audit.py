"""Run and summarize the score-blind complete UCR-17 corruption audit."""

# ruff: noqa: EM101, EM102, PERF401, T201, TRY003

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import cast

from scipy.stats import t as student_t

from lnet import pac_direct_stem_corruption as corruption
from lnet.pac_confirmatory_baselines import (
    ConfirmatoryFamily,
    confirmatory_trial_spec,
)
from lnet.pac_direct_stem_corruption import DirectStemCorruptionJob, run_job

DATASETS = (
    "ArrowHead",
    "CinCECGTorso",
    "CricketX",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "Earthquakes",
    "FordA",
    "FordB",
    "GunPoint",
    "ItalyPowerDemand",
    "MoteStrain",
    "Plane",
    "StarLightCurves",
    "Trace",
    "TwoLeadECG",
    "Wafer",
)
CORE_MODELS = ("cnn1d", "tcn", "mamba", "gru", "lstm", "transformer")
SSM_MODELS = ("s4d", "s5", "lru")
MODELS = ("alphabet", *CORE_MODELS, *SSM_MODELS)
SEEDS = (23, 31, 43, 47, 59)
CONDITIONS = (
    "id",
    "noise_std_0.1",
    "noise_std_0.2",
    "missing_rate_0.1",
    "missing_rate_0.3",
    "amplitude_0.5",
    "amplitude_1.5",
    "resample_half_restore",
)
DEFAULT_ROOT = Path("results/corruption-audit")
DEFAULT_ALPHABET_SELECTION = Path("selection/alphabet.json")
DEFAULT_CORE_SELECTION = Path("selection/core.json")
DEFAULT_SSM_SELECTION = Path("selection/ssm.json")


def _load_selection(path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid selection record: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"selection record is not an object: {path}")
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise TypeError(f"{path} does not contain a selected mapping")
    test_flags = (
        "official_test_accessed_during_selection",
        "test_evidence_used_for_architecture_choice",
    )
    present_test_flags = [name for name in test_flags if name in payload]
    if not present_test_flags or any(payload[name] is not False for name in present_test_flags):
        raise RuntimeError(f"{path} is not marked as TEST-free selection")
    if (
        "configuration_frozen_before_test" in payload
        and payload.get("configuration_frozen_before_test") is not True
    ):
        raise RuntimeError(f"{path} is not frozen before TEST")
    if payload.get("schema") not in {
        "pac.balanced_hpo_alphabet_27task_recovery_stage2_selection.v1",
        "pac.balanced_hpo_stage2_selection.v1",
        "pac_alphabet_q1_final_freeze.v1",
    }:
        raise RuntimeError(f"{path} has an unsupported selection schema")
    for name, row in selected.items():
        if not isinstance(name, str) or not isinstance(row, dict):
            raise RuntimeError(f"{path} contains an invalid selected cell")
        parts = name.split(":")
        if len(parts) != 3 or not all(parts):
            raise RuntimeError(f"{path} contains a malformed cell key: {name!r}")
        if not isinstance(row.get("config_key"), str) or not row["config_key"]:
            raise RuntimeError(f"{path} contains a cell without config_key: {name}")
        width = row.get("width")
        if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 4096:
            raise RuntimeError(f"{path} contains an invalid width: {name}")
        row_seeds = row.get("selection_seeds")
        if row_seeds is not None and row_seeds != [7, 11, 19]:
            raise RuntimeError(f"{path} contains non-canonical selection seeds: {name}")
        score = row.get("mean_validation_score", row.get("mean_selection_score"))
        if score is not None and (
            isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score))
        ):
            raise RuntimeError(f"{path} contains an invalid validation score: {name}")
        if parts[2] == "alphabet":
            modes = row.get("modes")
            recipe = row.get("recipe")
            if isinstance(modes, bool) or not isinstance(modes, int) or not 1 <= modes <= width:
                raise RuntimeError(f"{path} contains an invalid ALPHABET mode count: {name}")
            if not isinstance(recipe, dict) or recipe.get("name") not in {"A", "B", "C"}:
                raise RuntimeError(f"{path} contains an invalid ALPHABET recipe: {name}")
        elif parts[2] in CORE_MODELS:
            trial = row.get("trial")
            if trial is not None and (
                isinstance(trial, bool) or not isinstance(trial, int) or not 1 <= trial <= 6
            ):
                raise RuntimeError(f"{path} contains an invalid baseline trial: {name}")
            if trial is None and (
                not isinstance(row.get("recipe"), dict)
                or row["recipe"].get("name") not in {"A", "B", "C"}
            ):
                raise RuntimeError(f"{path} contains an invalid baseline recipe: {name}")
        elif parts[2] in SSM_MODELS:
            recipe = row.get("recipe")
            architecture = row.get("architecture_settings")
            if not isinstance(recipe, dict) or recipe.get("name") not in {"A", "B", "C"}:
                raise RuntimeError(f"{path} contains an invalid SSM recipe: {name}")
            if not isinstance(architecture, dict):
                raise RuntimeError(f"{path} contains invalid SSM architecture settings: {name}")
    return cast("dict[str, dict[str, object]]", selected)


def _ssm_trial(model: str, settings: dict[str, object]) -> int:
    depth = int(cast("int", settings["depth"]))
    state = int(cast("int", settings["state_size"]))
    architectures = {
        "s4d": ((1, 8), (2, 8), (3, 8), (1, 16), (2, 16), (3, 16)),
        "s5": ((1, 8), (2, 8), (1, 16), (2, 16), (1, 32), (2, 32)),
        "lru": ((1, 8), (2, 8), (1, 16), (2, 16), (1, 32), (2, 32)),
    }
    return architectures[model].index((depth, state)) + 1


def jobs(
    alphabet_selection: Path,
    core_selection: Path,
    ssm_selection: Path,
) -> list[DirectStemCorruptionJob]:
    alphabet_rows = _load_selection(alphabet_selection)
    core_rows = _load_selection(core_selection)
    ssm_rows = _load_selection(ssm_selection)
    result: list[DirectStemCorruptionJob] = []
    for dataset in DATASETS:
        alphabet = alphabet_rows[f"ucr:{dataset}:alphabet"]
        alphabet_recipe = cast("dict[str, object]", alphabet["recipe"])
        for seed in SEEDS:
            result.append(
                DirectStemCorruptionJob(
                    key=f"alphabet_radial_log_corruption:alphabet:{dataset}:seed{seed}",
                    dataset=dataset,
                    model="alphabet",
                    seed=seed,
                    model_dim=int(cast("int", alphabet["width"])),
                    modes=int(cast("int", alphabet["modes"])),
                    trial=None,
                    config_key=str(alphabet["config_key"]),
                    recipe_name=str(alphabet_recipe["name"]),
                    epochs=100,
                    batch_size=int(cast("int", alphabet_recipe["batch_size"])),
                    learning_rate=float(cast("float", alphabet_recipe["learning_rate"])),
                    weight_decay=float(cast("float", alphabet_recipe["weight_decay"])),
                    grad_clip_norm=float(cast("float", alphabet_recipe["grad_clip_norm"])),
                    selection_source=(
                        f"TEST-free validation-frozen ALPHABET selection:ucr:{dataset}:alphabet"
                    ),
                )
            )
        for model in CORE_MODELS:
            cell = core_rows[f"ucr:{dataset}:{model}"]
            trial = int(cast("int", cell["trial"]))
            spec = confirmatory_trial_spec(cast("ConfirmatoryFamily", model), trial)
            for seed in SEEDS:
                result.append(
                    DirectStemCorruptionJob(
                        key=f"alphabet_radial_log_corruption:{model}:{dataset}:seed{seed}",
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        model_dim=int(cast("int", cell["width"])),
                        modes=16,
                        trial=trial,
                        config_key=str(cell["config_key"]),
                        recipe_name=f"trial-{trial}",
                        epochs=100,
                        batch_size=spec.batch_size,
                        learning_rate=spec.learning_rate,
                        weight_decay=spec.weight_decay,
                        grad_clip_norm=spec.grad_clip_norm,
                        selection_source=(
                            f"TEST-free validation-frozen baseline selection:ucr:{dataset}:{model}"
                        ),
                    )
                )
        for model in SSM_MODELS:
            cell = ssm_rows[f"ucr:{dataset}:{model}"]
            recipe = cast("dict[str, object]", cell["recipe"])
            architecture = cast("dict[str, object]", cell["architecture_settings"])
            for seed in SEEDS:
                result.append(
                    DirectStemCorruptionJob(
                        key=f"alphabet_radial_log_corruption:{model}:{dataset}:seed{seed}",
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        model_dim=int(cast("int", cell["width"])),
                        modes=16,
                        trial=_ssm_trial(model, architecture),
                        config_key=str(cell["config_key"]),
                        recipe_name=str(recipe["name"]),
                        epochs=100,
                        batch_size=int(cast("int", recipe["batch_size"])),
                        learning_rate=float(cast("float", recipe["learning_rate"])),
                        weight_decay=float(cast("float", recipe["weight_decay"])),
                        grad_clip_norm=float(cast("float", recipe["grad_clip_norm"])),
                        selection_source=(
                            f"TEST-free validation-frozen SSM selection:ucr:{dataset}:{model}"
                        ),
                    )
                )
    expected = len(DATASETS) * len(MODELS) * len(SEEDS)
    if len(result) != expected or len({job.key for job in result}) != expected:
        raise RuntimeError(f"expected {expected} unique jobs, built {len(result)}")
    return result


def _result_path(root: Path, job_key: str, bucket: str = "completed") -> Path:
    return root / bucket / f"{job_key.replace(':', '__')}.json"


def _completed_keys(root: Path, expected: set[str] | None = None) -> set[str]:
    keys: set[str] = set()
    for path in (root / "completed").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or row.get("status") != "done":
            continue
        job_key = row.get("job_key")
        if isinstance(job_key, str) and (expected is None or job_key in expected):
            keys.add(job_key)
    return keys


def run(args: argparse.Namespace) -> None:
    corruption.BASELINE_MODELS = (*CORE_MODELS, *SSM_MODELS)
    corruption.MODELS = MODELS
    all_jobs = jobs(args.alphabet_selection, args.core_selection, args.ssm_selection)
    selected = [
        job for index, job in enumerate(all_jobs) if index % args.shard_count == args.shard_index
    ]
    completed = _completed_keys(args.output_root)
    (args.output_root / "completed").mkdir(parents=True, exist_ok=True)
    (args.output_root / "failed").mkdir(parents=True, exist_ok=True)
    for job in selected:
        if job.key in completed:
            continue
        try:
            row = run_job(
                job,
                data_root=args.data_root,
                output_root=args.output_root,
                device=args.device,
            )
            _result_path(args.output_root, job.key).write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n"
            )
        except Exception as error:  # noqa: BLE001
            failure = {
                "schema": "alphabet_corruption_ucr17_failure.v1",
                **asdict(job),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            _result_path(args.output_root, job.key, "failed").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n"
            )
    print(json.dumps(status(args), sort_keys=True))


def status(args: argparse.Namespace) -> dict[str, object]:
    expected = {
        job.key for job in jobs(args.alphabet_selection, args.core_selection, args.ssm_selection)
    }
    completed = _completed_keys(args.output_root, expected)
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "remaining": len(expected - completed),
        "done": expected <= completed,
    }


def _ci95(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    center = mean(values)
    radius = float(
        student_t.ppf(0.975, df=len(values) - 1) * stdev(values) / math.sqrt(len(values))
    )
    return [center - radius, center + radius]


def _validated_completed_rows(
    root: Path, expected: dict[str, DirectStemCorruptionJob]
) -> dict[str, dict[str, object]]:
    """Load only rows at their job-derived paths and validate their identity."""
    completed_dir = root / "completed"
    rows: dict[str, dict[str, object]] = {}
    for key, job in expected.items():
        path = _result_path(root, key)
        if not path.is_file():
            raise RuntimeError(f"missing completed result for {key}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid completed result for {key}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"completed result for {key} is not an object")
        if value.get("status") != "done" or value.get("job_key") != key:
            raise RuntimeError(f"completed result for {key} has invalid status or job key")
        for field in (
            "dataset",
            "model",
            "seed",
            "model_dim",
            "modes",
            "trial",
            "config_key",
            "recipe_name",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "grad_clip_norm",
        ):
            if value.get(field) != getattr(job, field):
                raise RuntimeError(f"completed result for {key} has wrong {field}")
        if value.get("selection_test_evidence_used") is not False:
            raise RuntimeError(f"completed result for {key} used TEST selection evidence")
        try:
            items = json.loads(str(value["corruption_balanced_accuracy_json"]))
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"completed result for {key} has invalid corruption metrics") from error
        if not isinstance(items, list) or len(items) != len(CONDITIONS):
            raise RuntimeError(f"completed result for {key} has incomplete corruption metrics")
        shifts: set[str] = set()
        for item in items:
            if not isinstance(item, dict) or item.get("shift") not in CONDITIONS:
                raise RuntimeError(f"completed result for {key} has an unknown corruption condition")
            shift = str(item["shift"])
            if shift in shifts:
                raise RuntimeError(f"completed result for {key} repeats corruption condition {shift}")
            shifts.add(shift)
            for metric in ("balanced_accuracy", "absolute_balanced_accuracy_drop"):
                metric_value = item.get(metric)
                if isinstance(metric_value, bool) or not isinstance(metric_value, (float, int)):
                    raise RuntimeError(f"completed result for {key} has invalid {metric}")
                numeric = float(metric_value)
                if not math.isfinite(numeric) or (metric == "balanced_accuracy" and not 0.0 <= numeric <= 1.0):
                    raise RuntimeError(f"completed result for {key} has non-finite {metric}")
        if shifts != set(CONDITIONS):
            raise RuntimeError(f"completed result for {key} has incomplete corruption conditions")
        rows[key] = cast("dict[str, object]", value)

    extras = []
    for path in completed_dir.glob("*.json"):
        if path not in {_result_path(root, key) for key in expected}:
            extras.append(path.name)
    if extras:
        raise RuntimeError(f"completed directory contains unexpected result files: {sorted(extras)}")
    return rows


def report(args: argparse.Namespace) -> dict[str, object]:
    expected_jobs = {
        job.key: job for job in jobs(args.alphabet_selection, args.core_selection, args.ssm_selection)
    }
    campaign_status = status(args)
    if not campaign_status["done"]:
        raise RuntimeError(f"campaign is incomplete: {campaign_status}")
    rows_by_key = _validated_completed_rows(args.output_root, expected_jobs)
    indexed = {
        (str(row["dataset"]), str(row["model"]), int(row["seed"])): row
        for row in rows_by_key.values()
    }
    expected_cells = len(DATASETS) * len(MODELS) * len(SEEDS)
    if len(indexed) != expected_cells:
        raise RuntimeError(
            f"expected {expected_cells} unique completed cells, found {len(indexed)}"
        )

    def score(dataset: str, model: str, seed: int, condition: str) -> float:
        items = json.loads(
            str(indexed[(dataset, model, seed)]["corruption_balanced_accuracy_json"])
        )
        return next(
            float(item["balanced_accuracy"]) for item in items if item["shift"] == condition
        )

    conditions: dict[str, object] = {}
    for condition in CONDITIONS:
        dataset_rows: dict[str, object] = {}
        task_deltas: list[float] = []
        task_ranks: list[float] = []
        wins = ties = losses = 0
        for dataset in DATASETS:
            model_means = {
                model: mean(score(dataset, model, seed, condition) for seed in SEEDS)
                for model in MODELS
            }
            best_control = max(CORE_MODELS + SSM_MODELS, key=model_means.__getitem__)
            delta = model_means["alphabet"] - model_means[best_control]
            tolerance = 1.0e-12
            wins += int(delta > tolerance)
            ties += int(abs(delta) <= tolerance)
            losses += int(delta < -tolerance)
            rank = 1 + sum(
                value > model_means["alphabet"] + tolerance
                for model, value in model_means.items()
                if model != "alphabet"
            )
            task_deltas.append(delta)
            task_ranks.append(float(rank))
            dataset_rows[dataset] = {
                "alphabet_balanced_accuracy": model_means["alphabet"],
                "best_control": best_control,
                "best_control_balanced_accuracy": model_means[best_control],
                "alphabet_minus_best_control": delta,
                "alphabet_rank": rank,
            }
        model_task_means = {
            model: mean(
                mean(score(dataset, model, seed, condition) for seed in SEEDS)
                for dataset in DATASETS
            )
            for model in MODELS
        }
        conditions[condition] = {
            "dataset_unit": "mean over five seeds within each dataset",
            "datasets": dataset_rows,
            "alphabet_minus_best_control": {
                "mean": mean(task_deltas),
                "ci95": _ci95(task_deltas),
                "wins_ties_losses": [wins, ties, losses],
                "tasks": len(task_deltas),
            },
            "alphabet_mean_rank": mean(task_ranks),
            "model_mean_balanced_accuracy": model_task_means,
        }
    payload = {
        "schema": "alphabet_corruption_ucr17_report.v1",
        "status": campaign_status,
        "completed_rows_validated": True,
        "paper_comparable": False,
        "scope": "deterministic synthetic corruption on UCR classification; not domain OOD",
        "dataset_selection": (
            "score-blind completeness: every UCR task with TEST-free frozen selections "
            "for ALPHABET and all nine controls"
        ),
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "uncertainty_unit": "dataset mean after averaging five seeds",
        "conditions": conditions,
    }
    reports = args.output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "CORRUPTION_UCR17.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("run", "status", "report"))
    result.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--data-root", type=Path, default=Path(".omx/data/ucr"))
    result.add_argument("--alphabet-selection", type=Path, default=DEFAULT_ALPHABET_SELECTION)
    result.add_argument("--core-selection", type=Path, default=DEFAULT_CORE_SELECTION)
    result.add_argument("--ssm-selection", type=Path, default=DEFAULT_SSM_SELECTION)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--shard-count", type=int, default=1)
    result.add_argument("--device", default="auto")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must satisfy 0 <= index < count")
    if args.stage == "run":
        run(args)
    elif args.stage == "report":
        print(json.dumps(report(args), indent=2, sort_keys=True))
    else:
        print(json.dumps(status(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
