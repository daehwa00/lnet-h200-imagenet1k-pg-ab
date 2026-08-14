"""Epsilon-zero pole-drift control and epsilon=.4 long-T sweep."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_moment_matched_extended_diagnostics import (  # noqa: E402
    phase_extremum_alignment,
    pole_phases,
)
from scripts.run_moment_matched_spectral_experiment import (  # noqa: E402
    EPOCHS,
    MODEL_DIM,
    MODES,
    SEEDS,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    bayes_balanced_accuracy,
    moment_matched_samples,
)
from lnet.pac_confirmatory_baselines import confirmatory_trial_spec  # noqa: E402
from lnet.pac_final_two_scan_ablation import FinalTwoScanAblation  # noqa: E402
from lnet.pac_training import classification_metric_bundle, train_classifier  # noqa: E402
from lnet.pac_types import PACClassificationTask, PACExperimentConfig  # noqa: E402

DEFAULT_ROOT = Path(".omx/results/pac-moment-matched-null-long-t-20260727")
LENGTHS = (256, 512, 1024, 2048)


@dataclass(frozen=True, slots=True)
class Job:
    kind: str
    seed: int
    length: int

    @property
    def key(self) -> str:
        return f"{self.kind}__T{self.length}__seed{self.seed}"


def jobs() -> list[Job]:
    return [Job("null", seed, 128) for seed in SEEDS] + [
        Job("long_t", seed, length) for length in LENGTHS for seed in SEEDS
    ]


def task_for(job: Job) -> PACClassificationTask:
    generator = torch.Generator().manual_seed(job.seed)
    epsilon = 0.4 if job.kind == "long_t" else 0.0
    if epsilon:
        train_x, train_y = moment_matched_samples(TRAIN_COUNT, job.length, epsilon, generator)
        valid_x, valid_y = moment_matched_samples(
            VALIDATION_COUNT, job.length, epsilon, generator
        )
    else:
        train_y = torch.arange(TRAIN_COUNT) % 2
        train_y = train_y[torch.randperm(TRAIN_COUNT, generator=generator)]
        valid_y = torch.arange(VALIDATION_COUNT) % 2
        valid_y = valid_y[torch.randperm(VALIDATION_COUNT, generator=generator)]
        train_x = torch.randn(TRAIN_COUNT, job.length, 1, generator=generator)
        valid_x = torch.randn(VALIDATION_COUNT, job.length, 1, generator=generator)
    return PACClassificationTask(
        job.key,
        train_x,
        train_y,
        valid_x,
        valid_y,
        train_x[:0],
        train_y[:0],
        2,
    )


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.tmp-{random.randrange(1 << 30)}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_job(job: Job, root: Path) -> None:
    output = root / "completed" / f"{job.key}.json"
    if output.exists() and output.stat().st_size:
        return
    random.seed(job.seed)
    torch.manual_seed(job.seed)
    torch.cuda.manual_seed_all(job.seed)
    task = task_for(job)
    spec = confirmatory_trial_spec("pac_tf", 4)
    batch_size = 16 if job.length >= 1024 else 32
    config = PACExperimentConfig(
        TRAIN_COUNT,
        VALIDATION_COUNT,
        0,
        job.length,
        raw_input_dim=1,
        output_dim=2,
        model_dim=MODEL_DIM,
        modes=MODES,
        epochs=EPOCHS,
        batch_size=batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        seeds=(job.seed,),
        device="cuda",
        optimizer_mode="fused",
    )
    model = FinalTwoScanAblation(
        config, 2, variant="full", objective="classification"
    ).cuda()
    initial = pole_phases(model)
    outcome = train_classifier(
        model,
        task,
        config,
        "cuda",
        job.seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    final = pole_phases(model)
    train = classification_metric_bundle(
        model, task.train_inputs.cuda(), task.train_labels.cuda(), batch_size=batch_size
    )
    valid = classification_metric_bundle(
        model,
        task.validation_inputs.cuda(),
        task.validation_labels.cuda(),
        batch_size=batch_size,
    )
    _write(
        output,
        {
            "kind": job.kind,
            "seed": job.seed,
            "length": job.length,
            "epsilon": 0.0 if job.kind == "null" else 0.4,
            "train_balanced_accuracy": train.balanced_accuracy,
            "validation_balanced_accuracy": valid.balanced_accuracy,
            "bayes_balanced_accuracy": (
                0.5 if job.kind == "null" else bayes_balanced_accuracy(task, 0.4)
            ),
            "initial_phase_alignment": phase_extremum_alignment(initial),
            "final_phase_alignment": phase_extremum_alignment(final),
            "best_epoch": outcome.best_epoch,
            "official_test_accessed": False,
        },
    )


def run(root: Path, shard_index: int, shard_count: int) -> None:
    for index, job in enumerate(jobs()):
        if index % shard_count == shard_index:
            run_job(job, root)


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "sample_sd": stdev(values)}


def report(root: Path) -> dict[str, object]:
    rows = [
        json.loads((root / "completed" / f"{job.key}.json").read_text()) for job in jobs()
    ]
    null = [row for row in rows if row["kind"] == "null"]
    length = {}
    for active_length in LENGTHS:
        chosen = [
            row for row in rows if row["kind"] == "long_t" and row["length"] == active_length
        ]
        length[str(active_length)] = {
            "train": _stats([row["train_balanced_accuracy"] for row in chosen]),
            "validation": _stats([row["validation_balanced_accuracy"] for row in chosen]),
            "bayes": _stats([row["bayes_balanced_accuracy"] for row in chosen]),
        }
    payload = {
        "null": {
            "train": _stats([row["train_balanced_accuracy"] for row in null]),
            "validation": _stats([row["validation_balanced_accuracy"] for row in null]),
            "initial_phase_alignment": _stats(
                [row["initial_phase_alignment"] for row in null]
            ),
            "final_phase_alignment": _stats([row["final_phase_alignment"] for row in null]),
        },
        "epsilon_0_4_length": length,
        "official_test_accessed": False,
    }
    _write(root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.root, args.shard_index, args.shard_count)
    else:
        result = report(args.root)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
