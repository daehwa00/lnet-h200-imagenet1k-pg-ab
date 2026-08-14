# ruff: noqa: E501, EM101, EM102, T201, TRY003
"""Same-recipe, parameter-matched synthetic OOD comparison for ALPHABET.

The completed ALPHABET rows are reused verbatim.  Six active controls are
trained with the identical fixed recipe and nearest real width, then evaluated
on the same matched-ZOH conditions and metadata interventions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn

from .pac_alphabet_synthetic_ood import (
    SEEDS,
    VARIANTS,
    _permute_delta,  # pyright: ignore[reportPrivateUsage]
)
from .pac_external_benchmarks import (
    _build_continuous_model,  # pyright: ignore[reportPrivateUsage]
)
from .pac_matched_zoh_ood import matched_zoh_conditions, matched_zoh_training_task
from .pac_metrics import count_parameters, nrmse
from .pac_training import train_regression_model
from .pac_types import PACDevice, PACExperimentConfig, PACRegressionTask

if TYPE_CHECKING:
    from .pac_external_benchmarks import ExternalModelFamily

ModelName = Literal["alphabet", "cnn1d", "tcn", "mamba", "gru", "lstm", "transformer"]
BaselineName = Literal["cnn1d", "tcn", "mamba", "gru", "lstm", "transformer"]

MODELS: tuple[ModelName, ...] = (
    "alphabet",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
BASELINES: tuple[BaselineName, ...] = (
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
WIDTHS: dict[BaselineName, int] = {
    "cnn1d": 10,
    "tcn": 17,
    "mamba": 15,
    "gru": 29,
    "lstm": 25,
    "transformer": 18,
}
EXPECTED_PARAMS: dict[ModelName, int] = {
    "alphabet": 3_138,
    "cnn1d": 3_372,
    "tcn": 3_045,
    "mamba": 3_167,
    "gru": 3_105,
    "lstm": 3_152,
    "transformer": 3_260,
}
TARGET_PARAMS = EXPECTED_PARAMS["alphabet"]
PARAMETER_TOLERANCE = 0.08
DEFAULT_ROOT = Path(
    ".omx/results/alphabet-final-radial-log-synthetic-ood-baselines-20260726"
)
ALPHABET_ROOT = Path(".omx/results/alphabet-final-radial-log-synthetic-ood-20260726")


@dataclass(frozen=True, slots=True)
class BaselineJob:
    model: BaselineName
    seed: int

    @property
    def key(self) -> str:
        return f"alphabet_synthetic_ood_baseline__{self.model}__seed{self.seed}"


def jobs() -> tuple[BaselineJob, ...]:
    return tuple(BaselineJob(model, seed) for model in BASELINES for seed in SEEDS)


def _config(root: Path, seed: int, *, smoke: bool) -> PACExperimentConfig:
    return PACExperimentConfig(
        sample_count=64 if smoke else 2048,
        validation_count=32 if smoke else 512,
        test_count=32 if smoke else 512,
        sequence_length=60,
        raw_input_dim=4,
        output_dim=2,
        model_dim=32,
        modes=16,
        epochs=1 if smoke else 100,
        batch_size=16 if smoke else 64,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        seeds=(seed,),
        device=cast("PACDevice", "cuda"),
        output_dir=root,
        compile_mode="none",
        precision="fp32",
    )


def _endpoint_task(config: PACExperimentConfig, seed: int) -> PACRegressionTask:
    task = matched_zoh_training_task(config, seed)
    return PACRegressionTask(
        label="matched_zoh_regular_endpoint",
        train_inputs=task.train_inputs,
        train_targets=task.train_targets[:, -1],
        validation_inputs=task.validation_inputs,
        validation_targets=task.validation_targets[:, -1],
        test_inputs=task.test_inputs,
        test_targets=task.test_targets[:, -1],
        true_delay=task.true_delay,
        true_frequency=task.true_frequency,
        true_frequencies=task.true_frequencies,
        true_dampings=task.true_dampings,
        mechanism_expectation=task.mechanism_expectation,
    )


def _build_baseline(model: BaselineName, config: PACExperimentConfig, seed: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        built = _build_continuous_model(
            cast("ExternalModelFamily", model),
            WIDTHS[model],
            4,
            2,
            config,
            "",
            objective="regression",
        )
    actual = count_parameters(built)
    if actual != EXPECTED_PARAMS[model]:
        raise RuntimeError(
            f"parameter lock changed for {model}: {actual} != {EXPECTED_PARAMS[model]}"
        )
    if abs(actual - TARGET_PARAMS) / TARGET_PARAMS > PARAMETER_TOLERANCE:
        raise RuntimeError(f"parameter match exceeds tolerance for {model}")
    return built


def transform_metadata(inputs: Tensor, variant: str, *, seed: int) -> Tensor:
    if variant not in VARIANTS:
        raise ValueError(f"unknown metadata variant: {variant}")
    values = inputs[..., :2]
    delta = inputs[..., 2:3]
    mask = inputs[..., 3:4]
    if variant in {"unit_dt_mask", "unit_dt_no_mask"}:
        delta = torch.ones_like(delta)
    elif variant == "shuffled_dt_mask":
        delta = _permute_delta(delta, seed)
    if variant in {"correct_dt_no_mask", "unit_dt_no_mask"}:
        mask = torch.ones_like(mask)
    return torch.cat((values, delta, mask), dim=-1)


@torch.no_grad()
def _metrics(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    variant: str,
    seed: int,
    batch_size: int,
) -> tuple[float, float]:
    transformed = transform_metadata(inputs, variant, seed=seed)
    predictions = torch.cat(
        [model(batch).detach().cpu() for batch in transformed.split(batch_size)], dim=0
    )
    target = targets.detach().cpu()
    mse = float(torch.nn.functional.mse_loss(predictions, target).item())
    return mse, nrmse(mse, target)


def run_job(root: Path, job: BaselineJob, *, device: str, smoke: bool) -> dict[str, object]:
    config = _config(root, job.seed, smoke=smoke)
    model = _build_baseline(job.model, config, job.seed)
    task = _endpoint_task(config, job.seed)
    outcome = train_regression_model(model, task, config, device, job.seed)
    model.eval()
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(matched_zoh_conditions(config, job.seed)):
        inputs = condition.ood_inputs.to(device)
        targets = condition.ood_targets[:, -1]
        for variant in VARIANTS:
            mse, active_nrmse = _metrics(
                model,
                inputs,
                targets,
                variant=variant,
                seed=job.seed + 10_000 + condition_index,
                batch_size=config.batch_size,
            )
            rows.append(
                {
                    "family": condition.family,
                    "level": condition.level,
                    "variant": variant,
                    "mse": mse,
                    "nrmse": active_nrmse,
                }
            )
    return {
        "schema": "alphabet.synthetic_ood_baseline_result.v1",
        "job_key": job.key,
        "model": job.model,
        "seed": job.seed,
        "status": "done",
        "smoke": smoke,
        "width": WIDTHS[job.model],
        "params_trainable": count_parameters(model),
        "target_params": TARGET_PARAMS,
        "relative_parameter_error": abs(count_parameters(model) - TARGET_PARAMS) / TARGET_PARAMS,
        "recipe": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "grad_clip_norm": config.grad_clip_norm,
        },
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "id_test_loss": outcome.test_loss,
        "id_test_nrmse": nrmse(outcome.test_loss, task.test_targets),
        "elapsed_seconds": outcome.elapsed_time,
        "conditions": rows,
    }


def enqueue(root: Path, *, shards: int = 8) -> dict[str, object]:
    if shards < 1:
        raise ValueError("shards must be positive")
    if not (ALPHABET_ROOT / "COMPLETE").is_file():
        raise RuntimeError("completed ALPHABET synthetic OOD artifact is required")
    root.mkdir(parents=True, exist_ok=True)
    (root / "completed").mkdir(exist_ok=True)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    alphabet_contract = json.loads((ALPHABET_ROOT / "contract.json").read_text(encoding="utf-8"))
    contract = {
        "schema": "alphabet.synthetic_ood_baseline_contract.v1",
        "public_models": list(MODELS),
        "baseline_jobs": len(jobs()),
        "reused_alphabet_jobs": len(SEEDS),
        "seeds": list(SEEDS),
        "shards": shards,
        "variants": list(VARIANTS),
        "conditions_per_job": 19,
        "target_params": TARGET_PARAMS,
        "parameter_tolerance": PARAMETER_TOLERANCE,
        "widths": WIDTHS,
        "expected_params": EXPECTED_PARAMS,
        "capacity_policy": "nearest real width; no dummy, adapter, or inert parameters",
        "comparison_estimand": "same-recipe parameter-matched synthetic matched-ZOH diagnostic",
        "recipe": {
            "epochs": 100,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "grad_clip_norm": 1.0,
            "precision": "fp32",
        },
        "metadata_parity": "all models receive value2, elapsed-time, and observation-mask information; baselines concatenate channels while ALPHABET uses native semantic arguments",
        "family_optimized": False,
        "test_or_ood_used_for_selection": False,
        "alphabet_source_root": str(ALPHABET_ROOT),
        "alphabet_contract_sha256": hashlib.sha256(
            (ALPHABET_ROOT / "contract.json").read_bytes()
        ).hexdigest(),
        "alphabet_source_sha256": alphabet_contract["source_sha256"],
        "source_sha256": source_hash,
        "restart_safe": True,
    }
    _atomic_json(root / "contract.json", contract)
    all_jobs = jobs()
    for shard in range(shards):
        (root / f"manifest-shard{shard}.jsonl").write_text(
            "".join(
                json.dumps(asdict(job), sort_keys=True) + "\n"
                for index, job in enumerate(all_jobs)
                if index % shards == shard
            ),
            encoding="utf-8",
        )
    return {"jobs": len(all_jobs), "shards": shards}


def worker(root: Path, shard: int, *, device: str, smoke: bool, max_jobs: int | None) -> int:
    manifest = root / f"manifest-shard{shard}.jsonl"
    completed_now = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        job = BaselineJob(cast("BaselineName", payload["model"]), int(payload["seed"]))
        destination = root / "completed" / f"{job.key}.json"
        if destination.is_file():
            continue
        _atomic_json(destination, run_job(root, job, device=device, smoke=smoke))
        completed_now += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if max_jobs is not None and completed_now >= max_jobs:
            break
    return completed_now


def status(root: Path) -> dict[str, object]:
    completed = 0
    failed = 0
    for path in (root / "completed").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        completed += row.get("status") == "done" and not row.get("smoke")
        failed += row.get("status") != "done"
    expected = len(jobs())
    return {
        "completed": completed,
        "expected": expected,
        "remaining": expected - completed,
        "failed": failed,
        "done": completed == expected and failed == 0,
    }


def _average_ranks(scores: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores, key=scores.__getitem__)
    ranks: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and math.isclose(
            scores[ordered[cursor]], scores[ordered[end]], rel_tol=1e-5, abs_tol=1e-8
        ):
            end += 1
        rank = ((cursor + 1) + end) / 2
        for model in ordered[cursor:end]:
            ranks[model] = rank
        cursor = end
    return ranks


def _load_final_payloads(root: Path) -> list[dict[str, object]]:
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]
    for path in sorted((ALPHABET_ROOT / "completed").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "done" and not row.get("smoke"):
            row["model"] = "alphabet"
            row["linked_from"] = str(path)
            payloads.append(row)
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    actual = {
        (str(row["model"]), int(row["seed"]))
        for row in payloads
        if row.get("status") == "done" and not row.get("smoke")
    }
    if actual != expected:
        raise RuntimeError(
            f"incomplete final comparison: missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    return [row for row in payloads if row.get("status") == "done" and not row.get("smoke")]


def report(root: Path) -> dict[str, object]:
    payloads = _load_final_payloads(root)
    long_rows = [
        {"model": payload["model"], "seed": payload["seed"], **condition}
        for payload in payloads
        for condition in cast("list[dict[str, object]]", payload["conditions"])
    ]
    profile_means: dict[tuple[str, str, str, str], list[float]] = {}
    for row in long_rows:
        key = (str(row["family"]), str(row["level"]), str(row["variant"]), str(row["model"]))
        profile_means.setdefault(key, []).append(float(cast("float | int | str", row["nrmse"])))
    variants: dict[str, object] = {}
    for variant in VARIANTS:
        rank_values = {model: [] for model in MODELS}
        top = dict.fromkeys(MODELS, 0)
        profiles: list[dict[str, object]] = []
        for family, level in sorted({(str(row["family"]), str(row["level"])) for row in long_rows}):
            scores = {
                model: mean(profile_means[(family, level, variant, model)]) for model in MODELS
            }
            ranks = _average_ranks(scores)
            best = min(scores.values())
            winners = [
                model
                for model in MODELS
                if math.isclose(scores[model], best, rel_tol=1e-5, abs_tol=1e-8)
            ]
            for model in MODELS:
                rank_values[model].append(ranks[model])
                top[model] += model in winners
            profiles.append(
                {
                    "family": family,
                    "level": level,
                    "mean_nrmse": scores,
                    "average_tie_rank": ranks,
                    "joint_top1": winners,
                }
            )
        # Equal-weight the seven shift families instead of overweighting sampling rate.
        macro: dict[str, float] = {}
        for model in MODELS:
            family_means = []
            for family in sorted({str(row["family"]) for row in long_rows}):
                values = [
                    float(cast("float | int | str", row["nrmse"]))
                    for row in long_rows
                    if row["model"] == model
                    and row["variant"] == variant
                    and row["family"] == family
                ]
                family_means.append(mean(values))
            macro[model] = mean(family_means)
        variants[variant] = {
            "models": {
                model: {
                    "equal_family_mean_nrmse": macro[model],
                    "mean_rank_19": mean(rank_values[model]),
                    "top1_19": top[model],
                }
                for model in MODELS
            },
            "profiles": profiles,
        }
    parameters = {
        str(row["model"]): int(cast("float | int | str", row["params_trainable"]))
        for row in payloads
    }
    id_nrmse = {
        model: mean(
            float(cast("float | int | str", row["id_test_nrmse"]))
            for row in payloads
            if row["model"] == model
        )
        for model in MODELS
    }
    summary = {
        "schema": "alphabet.synthetic_ood_baseline_summary.v1",
        **status(root),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "condition_rows": len(long_rows),
        "parameters": parameters,
        "id_mean_nrmse": id_nrmse,
        "variants": variants,
        "claim_boundary": "same-recipe parameter-matched diagnostic under one synthetic exact-ZOH teacher; not universal OOD superiority evidence",
    }
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    if long_rows:
        with (reports / "synthetic_ood_baseline_long.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
            writer.writeheader()
            writer.writerows(long_rows)
    _atomic_json(reports / "summary.json", summary)
    if summary["done"]:
        (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return summary


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("enqueue", "worker", "status", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    if args.stage == "enqueue":
        payload = enqueue(args.root, shards=args.shards)
    elif args.stage == "worker":
        payload = {
            "completed_now": worker(
                args.root, args.shard, device=args.device, smoke=args.smoke, max_jobs=args.max_jobs
            )
        }
    elif args.stage == "status":
        payload = status(args.root)
    else:
        payload = report(args.root)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
