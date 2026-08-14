"""Run the preregistered radial-log lag battery on TRAIN-derived validation folds."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lnet.alphabet import Alphabet  # noqa: E402
from lnet.pac_confirmatory_baselines import confirmatory_trial_spec  # noqa: E402
from lnet.pac_eval_sections import clean_validation_classification_task  # noqa: E402
from lnet.pac_metrics import count_parameters  # noqa: E402
from lnet.pac_real_data import ensure_ucr_train_only  # noqa: E402
from lnet.pac_training import classification_metric_bundle, train_classifier  # noqa: E402
from lnet.pac_types import PACDevice, PACExperimentConfig  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

Variant = Literal["full", "energy_mlp"]
DATASETS = (
    "CricketX",
    "CricketY",
    "CricketZ",
    "ECGFiveDays",
    "TwoLeadECG",
    "FordA",
    "FordB",
    "Phoneme",
    "ShapeletSim",
    "SyntheticControl",
)
SEEDS = (23, 31, 43, 47, 59)
MODEL_DIM = 64
MODES = 16
EPOCHS = 100
TRIAL = 4


@dataclass(frozen=True, slots=True)
class Job:
    dataset: str
    seed: int
    variant: Variant

    @property
    def key(self) -> str:
        return f"{self.dataset}__seed{self.seed}__{self.variant}"


class EnergyMLPClassifier(nn.Module):
    """MLP exposing affine-like weight and bias attributes for ALPHABET dispatch."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.first = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.second = nn.Linear(hidden_dim, output_dim)

    @property
    def weight(self) -> Tensor:
        return self.second.weight

    @property
    def bias(self) -> Tensor | None:
        return self.second.bias

    def forward(self, inputs: Tensor) -> Tensor:
        return self.second(self.activation(self.first(inputs)))


class EnergyOnlyMLPHead(nn.Module):
    """Capacity-matched nonlinear control over writer/reader log energy only."""

    def __init__(self, modes: int, output_dim: int) -> None:
        super().__init__()
        input_dim = 2 * modes
        target = output_dim * (14 * modes + 1)
        hidden = max(1, round((target - output_dim) / (input_dim + output_dim + 1)))
        self.modes = modes
        self.hidden_dim = hidden
        self.mode_map = None
        self.classifier = EnergyMLPClassifier(input_dim, hidden, output_dim)

    def feature_group_slices(self) -> dict[str, tuple[slice, ...]]:
        return {"raw_modal": (slice(0, 2 * self.modes),), "mode_branch": ()}

    def forward(self, writer_moments: Tensor, reader_moments: Tensor) -> Tensor:
        energy = torch.cat(
            (writer_moments[:, : self.modes], reader_moments[:, : self.modes]),
            dim=-1,
        )
        return self.classifier(energy)


class EnergyOnlyAlphabet(Alphabet):
    """Canonical backbone with the capacity-matched energy-only control head."""

    def __init__(self, config: PACExperimentConfig, output_dim: int) -> None:
        super().__init__(config, output_dim, objective="classification")
        self.head = EnergyOnlyMLPHead(self.modes, output_dim)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _head_parameters(model: Alphabet) -> int:
    return sum(
        parameter.numel()
        for parameter in model.head.parameters()
        if parameter.requires_grad
    )


def run_job(
    job: Job,
    *,
    output_root: Path,
    data_root: Path,
    device: PACDevice,
) -> dict[str, object]:
    output = output_root / "completed" / f"{job.key}.json"
    if output.exists():
        return cast("dict[str, object]", json.loads(output.read_text(encoding="utf-8")))
    runtime_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    _seed_everything(job.seed)
    task = clean_validation_classification_task(
        ensure_ucr_train_only(job.dataset, data_root, allow_download=False),
        job.seed,
    )
    spec = confirmatory_trial_spec("pac_tf", TRIAL)
    config = PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        0,
        task.train_inputs.shape[1],
        raw_input_dim=task.train_inputs.shape[-1],
        output_dim=task.class_count,
        model_dim=MODEL_DIM,
        modes=MODES,
        epochs=EPOCHS,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        seeds=(job.seed,),
        device=cast("PACDevice", runtime_device),
        optimizer_mode="fused" if runtime_device == "cuda" else "default",
    )
    model: Alphabet
    if job.variant == "full":
        model = Alphabet(config, task.class_count, objective="classification")
    else:
        model = EnergyOnlyAlphabet(config, task.class_count)
    model.use_efp16_exact_split_training = False
    model.require_external_exact_split_training = False
    full_reference = Alphabet(config, task.class_count, objective="classification")
    target_total = count_parameters(full_reference)
    target_head = _head_parameters(full_reference)
    del full_reference
    actual_total = count_parameters(model)
    actual_head = _head_parameters(model)
    model.to(runtime_device)
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
    row: dict[str, object] = {
        "schema": "finite_bank_lag_battery_result.v1",
        "status": "done",
        "dataset": job.dataset,
        "seed": job.seed,
        "variant": job.variant,
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "best_epoch": outcome.best_epoch,
        "elapsed_seconds": perf_counter() - started,
        "balanced_accuracy": metrics.balanced_accuracy,
        "accuracy": metrics.accuracy,
        "train_count": int(task.train_inputs.shape[0]),
        "validation_count": int(task.validation_inputs.shape[0]),
        "class_count": task.class_count,
        "sequence_length": int(task.train_inputs.shape[1]),
        "parameters": {
            "actual_total": actual_total,
            "target_total": target_total,
            "total_relative_error": (actual_total - target_total) / target_total,
            "actual_head": actual_head,
            "target_head": target_head,
            "head_relative_error": (actual_head - target_head) / target_head,
        },
    }
    _write_json(output, row)
    return row


def _paired_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    indexed = {
        (str(row["dataset"]), int(row["seed"]), str(row["variant"])): row
        for row in rows
    }
    datasets: dict[str, object] = {}
    all_deltas: list[float] = []
    for dataset in DATASETS:
        deltas = []
        pairs = []
        for seed in SEEDS:
            full = float(indexed[(dataset, seed, "full")]["balanced_accuracy"])
            energy = float(indexed[(dataset, seed, "energy_mlp")]["balanced_accuracy"])
            delta = 100 * (full - energy)
            deltas.append(delta)
            all_deltas.append(delta)
            pairs.append({"seed": seed, "energy_mlp": energy, "full": full, "delta_pp": delta})
        mean_delta = sum(deltas) / len(deltas)
        sample_sd = math.sqrt(
            sum((value - mean_delta) ** 2 for value in deltas) / (len(deltas) - 1)
        )
        datasets[dataset] = {
            "pairs": pairs,
            "mean_delta_pp": mean_delta,
            "sample_sd_delta_pp": sample_sd,
            "wins": sum(value > 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
        }
    return {
        "schema": "finite_bank_lag_battery_summary.v1",
        "protocol": (
            "fixed 10-dataset battery; five independent seeds; paired full "
            "radial-log affine versus capacity-matched energy-only MLP"
        ),
        "datasets": datasets,
        "dataset_count": len(DATASETS),
        "seed_count": len(SEEDS),
        "official_test_accessed": False,
        "mean_delta_pp_over_all_pairs": sum(all_deltas) / len(all_deltas),
        "wins": sum(value > 0 for value in all_deltas),
        "ties": sum(value == 0 for value in all_deltas),
        "losses": sum(value < 0 for value in all_deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / ".omx/data/ucr")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=("full", "energy_mlp"),
        default=["full", "energy_mlp"],
    )
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    selected_datasets = tuple(args.datasets)
    selected_seeds = tuple(args.seeds)
    selected_variants = tuple(args.variants)
    jobs = [
        Job(dataset, seed, variant)
        for dataset in selected_datasets
        for seed in selected_seeds
        for variant in selected_variants
    ]
    contract = {
        "schema": "finite_bank_lag_battery_contract.v1",
        "datasets": list(selected_datasets),
        "seeds": list(selected_seeds),
        "variants": list(selected_variants),
        "model_dim": MODEL_DIM,
        "modes": MODES,
        "epochs": EPOCHS,
        "optimizer_trial": TRIAL,
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "jobs": [asdict(job) for job in jobs],
    }
    _write_json(args.output_root / "contract.json", contract)
    if not 0 <= args.worker_index < args.worker_count:
        message = "worker-index must be in [0, worker-count)"
        raise ValueError(message)
    active_jobs = jobs[args.worker_index :: args.worker_count]
    if not args.report_only:
        for job in active_jobs:
            run_job(
                job,
                output_root=args.output_root,
                data_root=args.data_root,
                device=cast("PACDevice", args.device),
            )
    completed_paths = sorted((args.output_root / "completed").glob("*.json"))
    rows = [
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        for path in completed_paths
    ]
    complete = len(rows) == len(jobs)
    if complete and (
        selected_datasets == DATASETS
        and selected_seeds == SEEDS
        and set(selected_variants) == {"full", "energy_mlp"}
    ):
        summary = _paired_summary(rows)
        _write_json(args.output_root / "summary.json", summary)
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        status = {
            "expected": len(jobs),
            "completed": len(rows),
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
        }
        sys.stdout.write(json.dumps(status, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
