"""Run the L=4 moment-matched spectral separation experiment.

Class zero is white Gaussian noise. Class one is the stationary MA(5)

    X_t = a Z_t + b Z_{t-5},

with ``a^2+b^2=1`` and ``2ab=epsilon``. Consequently, the two classes have
identical population autocovariances at lags 0,...,4 and first differ at lag 5.
Only TRAIN-derived validation data are used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Literal, cast

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lnet.pac_confirmatory_baselines import confirmatory_trial_spec  # noqa: E402
from lnet.pac_final_two_scan_ablation import FinalTwoScanAblation  # noqa: E402
from lnet.pac_metrics import count_parameters  # noqa: E402
from lnet.pac_training import classification_metric_bundle, train_classifier  # noqa: E402
from lnet.pac_types import PACClassificationTask, PACDevice, PACExperimentConfig  # noqa: E402

Variant = Literal["energy_only", "full"]

DEFAULT_ROOT = Path("results/spectral-control")
EPSILONS = (0.1, 0.2, 0.4, 0.8)
SEEDS = (23, 31, 43, 47, 59)
VARIANTS: tuple[Variant, ...] = ("energy_only", "full")
MATCHED_LAG = 4
FIRST_DIFFERING_LAG = MATCHED_LAG + 1
SEQUENCE_LENGTH = 128
TRAIN_COUNT = 512
VALIDATION_COUNT = 256
MODEL_DIM = 64
MODES = 16
EPOCHS = 60
TRIAL = 4


@dataclass(frozen=True, slots=True)
class Job:
    epsilon: float
    seed: int
    variant: Variant

    @property
    def key(self) -> str:
        epsilon_label = f"{self.epsilon:.3f}".replace(".", "p")
        return f"eps{epsilon_label}__seed{self.seed}__{self.variant}"


def ma_coefficients(epsilon: float) -> tuple[float, float]:
    """Return positive MA coefficients with unit variance and 2ab=epsilon."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    discriminant = math.sqrt(1.0 - epsilon * epsilon)
    a = math.sqrt((1.0 + discriminant) / 2.0)
    b = math.sqrt((1.0 - discriminant) / 2.0)
    return a, b


def population_autocovariances(epsilon: float, max_lag: int) -> Tensor:
    """Population autocovariances of the MA(5) class."""
    result = torch.zeros(max_lag + 1, dtype=torch.float64)
    result[0] = 1.0
    if max_lag >= FIRST_DIFFERING_LAG:
        result[FIRST_DIFFERING_LAG] = epsilon / 2.0
    return result


def moment_matched_samples(
    count: int,
    length: int,
    epsilon: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample balanced stationary white-noise and MA(5) sequences."""
    if count % 2:
        raise ValueError("count must be even")
    labels = torch.arange(count, dtype=torch.long) % 2
    labels = labels[torch.randperm(count, generator=generator)]
    innovations = torch.randn(
        count,
        length + FIRST_DIFFERING_LAG,
        generator=generator,
    )
    values = innovations[:, FIRST_DIFFERING_LAG:].clone()
    a, b = ma_coefficients(epsilon)
    selected = labels == 1
    values[selected] = (
        a * innovations[selected, FIRST_DIFFERING_LAG:]
        + b * innovations[selected, :length]
    )
    return values.unsqueeze(-1), labels


def make_task(
    seed: int,
    epsilon: float,
    *,
    train_count: int = TRAIN_COUNT,
    validation_count: int = VALIDATION_COUNT,
) -> PACClassificationTask:
    generator = torch.Generator().manual_seed(seed)
    train_inputs, train_labels = moment_matched_samples(
        train_count,
        SEQUENCE_LENGTH,
        epsilon,
        generator,
    )
    validation_inputs, validation_labels = moment_matched_samples(
        validation_count,
        SEQUENCE_LENGTH,
        epsilon,
        generator,
    )
    return PACClassificationTask(
        f"MomentMatchedL4-eps{epsilon:.3f}",
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        train_inputs[:0],
        train_labels[:0],
        2,
    )


def empirical_autocovariances(inputs: Tensor, max_lag: int) -> Tensor:
    values = inputs.squeeze(-1)
    return torch.stack(
        [
            (values[:, lag:] * values[:, : values.shape[1] - lag]).mean(dim=1)
            if lag
            else values.square().mean(dim=1)
            for lag in range(max_lag + 1)
        ],
        dim=1,
    )


def balanced_accuracy(predictions: Tensor, labels: Tensor) -> float:
    recalls = [
        float((predictions[labels == label] == label).float().mean())
        for label in (0, 1)
    ]
    return mean(recalls)


def raw_moment_nearest_prototype_bacc(task: PACClassificationTask) -> float:
    train = empirical_autocovariances(task.train_inputs, MATCHED_LAG)
    validation = empirical_autocovariances(task.validation_inputs, MATCHED_LAG)
    prototypes = torch.stack(
        [train[task.train_labels == label].mean(dim=0) for label in (0, 1)]
    )
    scale = train.std(dim=0, unbiased=True).clamp_min(1.0e-6)
    distances = ((validation[:, None, :] - prototypes[None, :, :]) / scale).square().sum(-1)
    return balanced_accuracy(distances.argmin(dim=1), task.validation_labels)


def _class_covariance(length: int, epsilon: float, label: int) -> Tensor:
    covariance = torch.eye(length, dtype=torch.float64)
    if label == 1:
        delta = epsilon / 2.0
        index = torch.arange(length - FIRST_DIFFERING_LAG)
        covariance[index, index + FIRST_DIFFERING_LAG] = delta
        covariance[index + FIRST_DIFFERING_LAG, index] = delta
    return covariance


def bayes_balanced_accuracy(task: PACClassificationTask, epsilon: float) -> float:
    values = task.validation_inputs.squeeze(-1).to(torch.float64)
    scores = []
    for label in (0, 1):
        covariance = _class_covariance(values.shape[1], epsilon, label)
        cholesky = torch.linalg.cholesky(covariance)
        solved = torch.cholesky_solve(values.T, cholesky).T
        quadratic = (values * solved).sum(dim=1)
        log_determinant = 2.0 * torch.log(torch.diagonal(cholesky)).sum()
        scores.append(-0.5 * (quadratic + log_determinant))
    predictions = torch.stack(scores, dim=1).argmax(dim=1)
    return balanced_accuracy(predictions, task.validation_labels)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jobs() -> list[Job]:
    return jobs_for(False)


def jobs_for(quick: bool) -> list[Job]:
    epsilons = (EPSILONS[0],) if quick else EPSILONS
    seeds = SEEDS[:2] if quick else SEEDS
    return [
        Job(epsilon, seed, variant)
        for epsilon in epsilons
        for seed in seeds
        for variant in VARIANTS
    ]


def prepare(root: Path, *, quick: bool = False) -> dict[str, object]:
    active = jobs_for(quick)
    root.mkdir(parents=True, exist_ok=True)
    (root / "queue.jsonl").write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in active),
        encoding="utf-8",
    )
    contract = {
        "schema": "alphabet.moment_matched_l4.contract.v1",
        "status": "frozen",
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "construction": {
            "class_0": "white Gaussian noise",
            "class_1": "unit-variance stationary MA(5)",
            "matched_population_autocovariance_lags": list(range(MATCHED_LAG + 1)),
            "first_differing_lag": FIRST_DIFFERING_LAG,
            "gamma_5": "epsilon/2",
        },
        "epsilons": list(EPSILONS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "sequence_length": SEQUENCE_LENGTH,
        "train_count": TRAIN_COUNT,
        "validation_count": VALIDATION_COUNT,
        "model_dim": MODEL_DIM,
        "modes": MODES,
        "epochs": EPOCHS,
        "optimizer_trial": TRIAL,
        "jobs": len(active),
        "quick": quick,
    }
    _write_json(root / "contract.json", contract)
    return status(root)


def result_path(root: Path, job: Job) -> Path:
    return root / "completed" / f"{job.key}.json"


def run_job(
    job: Job,
    root: Path,
    device: PACDevice,
    *,
    quick: bool = False,
) -> dict[str, object]:
    output = result_path(root, job)
    if output.exists():
        return cast("dict[str, object]", json.loads(output.read_text(encoding="utf-8")))
    runtime_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    _seed_everything(job.seed)
    train_count = 64 if quick else TRAIN_COUNT
    validation_count = 64 if quick else VALIDATION_COUNT
    task = make_task(
        job.seed,
        job.epsilon,
        train_count=train_count,
        validation_count=validation_count,
    )
    spec = confirmatory_trial_spec("pac_tf", TRIAL)
    config = PACExperimentConfig(
        train_count,
        validation_count,
        0,
        SEQUENCE_LENGTH,
        raw_input_dim=1,
        output_dim=2,
        model_dim=MODEL_DIM,
        modes=MODES,
        epochs=3 if quick else EPOCHS,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        seeds=(job.seed,),
        device=cast("PACDevice", runtime_device),
        optimizer_mode="fused" if runtime_device == "cuda" else "default",
    )
    model = FinalTwoScanAblation(
        config,
        2,
        variant=job.variant,
        objective="classification",
    ).to(runtime_device)
    started = perf_counter()
    outcome = train_classifier(
        model,
        task,
        config,
        runtime_device,
        job.seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    metrics = classification_metric_bundle(
        model,
        task.validation_inputs.to(runtime_device),
        task.validation_labels.to(runtime_device),
        batch_size=spec.batch_size,
    )
    row = {
        "schema": "alphabet.moment_matched_l4.result.v1",
        "status": "done",
        "job_key": job.key,
        "epsilon": job.epsilon,
        "seed": job.seed,
        "variant": job.variant,
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "matched_lags": list(range(MATCHED_LAG + 1)),
        "first_differing_lag": FIRST_DIFFERING_LAG,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        "validation_accuracy": metrics.accuracy,
        "bayes_balanced_accuracy": bayes_balanced_accuracy(task, job.epsilon),
        "raw_moment_0_4_nearest_prototype_balanced_accuracy": (
            raw_moment_nearest_prototype_bacc(task)
        ),
        "best_epoch": outcome.best_epoch,
        "elapsed_seconds": perf_counter() - started,
        "parameters": count_parameters(model),
        "parameter_relative_error": model.capacity_audit.relative_error,
    }
    _write_json(output, row)
    return row


def _queued_jobs(root: Path) -> list[Job]:
    return [
        Job(**json.loads(line))
        for line in (root / "queue.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def run(root: Path, device: PACDevice, *, quick: bool = False) -> dict[str, object]:
    queued = _queued_jobs(root)
    for job in queued:
        run_job(job, root, device, quick=quick)
    return report(root)


def status(root: Path) -> dict[str, object]:
    active = _queued_jobs(root) if (root / "queue.jsonl").is_file() else jobs()
    expected = len(active)
    completed = sum(1 for job in active if result_path(root, job).exists())
    return {
        "expected": expected,
        "completed": completed,
        "remaining": expected - completed,
        "done": completed == expected,
    }


def report(root: Path) -> dict[str, object]:
    current = status(root)
    if not current["done"]:
        return current
    active = _queued_jobs(root)
    rows = [json.loads(result_path(root, job).read_text(encoding="utf-8")) for job in active]
    summary: dict[str, object] = {}
    epsilons = sorted({job.epsilon for job in active})
    variants = tuple(dict.fromkeys(job.variant for job in active))
    seeds = sorted({job.seed for job in active})
    reference_variant = "full" if "full" in variants else variants[-1]
    for epsilon in epsilons:
        selected = [row for row in rows if row["epsilon"] == epsilon]
        by_variant = {}
        for variant in variants:
            values = [
                float(row["validation_balanced_accuracy"])
                for row in selected
                if row["variant"] == variant
            ]
            by_variant[variant] = {
                "mean": mean(values),
                "sample_sd": stdev(values),
            }
        unique_seeds = [
            next(
                row for row in selected
                if row["seed"] == seed and row["variant"] == reference_variant
            )
            for seed in seeds
        ]
        summary[f"{epsilon:.3f}"] = {
            "bayes": {
                "mean": mean(float(row["bayes_balanced_accuracy"]) for row in unique_seeds),
                "sample_sd": stdev(
                    float(row["bayes_balanced_accuracy"]) for row in unique_seeds
                ),
            },
            "raw_moment_0_4": {
                "mean": mean(
                    float(row["raw_moment_0_4_nearest_prototype_balanced_accuracy"])
                    for row in unique_seeds
                ),
                "sample_sd": stdev(
                    float(row["raw_moment_0_4_nearest_prototype_balanced_accuracy"])
                    for row in unique_seeds
                ),
            },
            **by_variant,
        }
    payload = {
        "schema": "alphabet.moment_matched_l4.summary.v1",
        "status": current,
        "estimand": (
            "balanced accuracy versus spectral separation epsilon when raw population "
            "autocovariances at lags 0,...,4 are identical"
        ),
        "summary": summary,
        "rows": len(rows),
        "official_test_accessed": False,
    }
    _write_json(root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "status", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare(args.root, quick=args.quick)
    elif args.command == "run":
        payload = run(args.root, cast("PACDevice", args.device), quick=args.quick)
    elif args.command == "report":
        payload = report(args.root)
    else:
        payload = status(args.root)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
