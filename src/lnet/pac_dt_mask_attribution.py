# ruff: noqa: E501, T201
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn

from .pac_headroom_efficient_models import build_efficient_headroom_classifier
from .pac_matched_zoh_ood import _build_case, _forcing_parameters
from .pac_metrics import count_parameters, nrmse
from .pac_training import evaluate_regression_loss, train_regression_model
from .pac_types import PACDevice, PACExperimentConfig, PACRegressionTask

SEEDS = (7, 11, 19, 23, 31)
VARIANTS = (
    "correct_dt_mask",
    "unit_dt_no_mask",
    "shuffled_dt_mask",
    "correct_dt_no_mask",
    "constant_dt_mask",
)
Variant = Literal[
    "correct_dt_mask",
    "unit_dt_no_mask",
    "shuffled_dt_mask",
    "correct_dt_no_mask",
    "constant_dt_mask",
]


@dataclass(frozen=True, slots=True)
class AttributionJob:
    seed: int
    variant: Variant

    @property
    def key(self) -> str:
        return f"dt_mask__{self.variant}__seed{self.seed}"


class _AttributedAlphabet(nn.Module):
    def __init__(self, config: PACExperimentConfig, variant: Variant) -> None:
        super().__init__()
        self.variant = variant
        self.model = build_efficient_headroom_classifier(
            "PA2WP", replace(config, raw_input_dim=2), config.output_dim, objective="regression"
        )

    def forward(self, inputs: Tensor) -> Tensor:
        time_delta = inputs[..., 2:3]
        mask: Tensor | None = inputs[..., 3:4]
        if self.variant in {"unit_dt_no_mask", "constant_dt_mask"}:
            time_delta = torch.ones_like(time_delta)
        if self.variant in {"unit_dt_no_mask", "correct_dt_no_mask"}:
            mask = None
        return self.model(inputs[..., :2], time_delta=time_delta, observation_mask=mask)

    def post_optimizer_step(self) -> None:
        self.model.post_optimizer_step()

    def finalize_constraints(self) -> None:
        self.model.finalize_constraints()


def attribution_jobs() -> list[AttributionJob]:
    return [AttributionJob(seed, cast("Variant", variant)) for seed in SEEDS for variant in VARIANTS]


def _case(
    count: int,
    seed: int,
    *,
    delta: float = 1.0,
    irregularity: float = 0.45,
    missing_rate: float = 0.15,
) -> tuple[Tensor, Tensor]:
    return _build_case(
        _forcing_parameters(count, seed),
        delta=delta,
        irregularity=irregularity,
        missing_rate=missing_rate,
        perturbation_seed=seed + 10_000,
    )


def _permute_delta(inputs: Tensor, seed: int) -> Tensor:
    result = inputs.clone()
    generator = torch.Generator().manual_seed(seed)
    for row in range(result.shape[0]):
        order = torch.randperm(result.shape[1], generator=generator)
        result[row, :, 2] = result[row, order, 2]
    return result


def _variant_inputs(inputs: Tensor, variant: Variant, seed: int) -> Tensor:
    if variant == "shuffled_dt_mask":
        return _permute_delta(inputs, seed)
    return inputs


def attribution_task(config: PACExperimentConfig, job: AttributionJob) -> PACRegressionTask:
    train_x, train_y = _case(config.sample_count, job.seed + 101)
    val_x, val_y = _case(config.validation_count, job.seed + 211)
    test_x, test_y = _case(config.test_count, job.seed + 307)
    return PACRegressionTask(
        label=f"variable_dt_mask_attribution_{job.variant}",
        train_inputs=_variant_inputs(train_x, job.variant, job.seed + 1001),
        train_targets=train_y[:, -1],
        validation_inputs=_variant_inputs(val_x, job.variant, job.seed + 2001),
        validation_targets=val_y[:, -1],
        test_inputs=_variant_inputs(test_x, job.variant, job.seed + 3001),
        test_targets=test_y[:, -1],
        true_delay=0,
        true_frequency=math.pi / 4,
        true_frequencies=(math.pi / 4,),
        true_dampings=(0.8,),
        mechanism_expectation="positive",
    )


def evaluation_conditions(config: PACExperimentConfig, job: AttributionJob) -> list[tuple[str, Tensor, Tensor]]:
    specifications = (
        ("id_variable_dt", 1.0, 0.45, 0.15),
        ("unseen_dt_mild", 1.0, 0.70, 0.15),
        ("unseen_dt_hard", 1.0, 0.90, 0.30),
        ("unseen_base_dt_0.5", 0.5, 0.45, 0.15),
        ("unseen_base_dt_2.0", 2.0, 0.45, 0.15),
    )
    rows: list[tuple[str, Tensor, Tensor]] = []
    for index, (name, delta, irregularity, missing_rate) in enumerate(specifications):
        inputs, targets = _case(
            config.test_count,
            job.seed + 401,
            delta=delta,
            irregularity=irregularity,
            missing_rate=missing_rate,
        )
        rows.append(
            (
                name,
                _variant_inputs(inputs, job.variant, job.seed + 4001 + index),
                targets[:, -1],
            )
        )
    return rows


def run_job(root: Path, job: AttributionJob, *, device: str = "cuda", smoke: bool = False) -> dict[str, object]:
    config = PACExperimentConfig(
        sample_count=128 if smoke else 2048,
        validation_count=64 if smoke else 512,
        test_count=64 if smoke else 512,
        sequence_length=60,
        raw_input_dim=4,
        output_dim=2,
        model_dim=16 if smoke else 64,
        modes=4 if smoke else 16,
        epochs=1 if smoke else 100,
        batch_size=32 if smoke else 64,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        seeds=(job.seed,),
        device=cast("PACDevice", device),
        output_dir=root,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.seed)
        model = _AttributedAlphabet(config, job.variant)
    task = attribution_task(config, job)
    outcome = train_regression_model(model, task, config, device, job.seed)
    condition_rows: list[dict[str, object]] = []
    for name, inputs, targets in evaluation_conditions(config, job):
        loss = evaluate_regression_loss(model, inputs.to(device), targets.to(device))
        condition_rows.append({"condition": name, "mse": loss, "nrmse": nrmse(loss, targets)})
    return {
        "schema_version": "pac_dt_mask_attribution.v1",
        "job_key": job.key,
        "seed": job.seed,
        "variant": job.variant,
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "id_test_loss": outcome.test_loss,
        "id_test_nrmse": nrmse(outcome.test_loss, task.test_targets),
        "elapsed_seconds": outcome.elapsed_time,
        "params_trainable": count_parameters(model),
        "conditions": condition_rows,
        "training_distribution": {"base_dt": 1.0, "irregularity": 0.45, "missing_rate": 0.15},
        "status": "done",
    }


def enqueue(root: Path, *, shards: int = 4) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(exist_ok=True)
    jobs = attribution_jobs()
    _atomic_json(root / "contract.json", {
        "schema_version": "pac_dt_mask_attribution_contract.v1",
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "jobs": len(jobs),
        "shards": shards,
        "locked_before_execution": True,
        "model": "ALPHABET/PA2WP",
    })
    for shard in range(shards):
        path = root / f"manifest-shard{shard}.jsonl"
        path.write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for index, job in enumerate(jobs) if index % shards == shard),
            encoding="utf-8",
        )
    return {"jobs": len(jobs), "shards": shards}


def worker(root: Path, shard: int, *, device: str = "cuda", smoke: bool = False) -> int:
    manifest = root / f"manifest-shard{shard}.jsonl"
    completed = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        job = AttributionJob(int(payload["seed"]), cast("Variant", payload["variant"]))
        destination = root / "jobs" / f"{job.key}.json"
        if destination.is_file():
            continue
        _atomic_json(destination, run_job(root, job, device=device, smoke=smoke))
        completed += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return completed


def report(root: Path) -> dict[str, object]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "jobs").glob("*.json"))]
    long_rows = [
        {"job_key": row["job_key"], "seed": row["seed"], "variant": row["variant"], **condition}
        for row in payloads
        for condition in row["conditions"]
    ]
    report_root = root / "reports"
    report_root.mkdir(exist_ok=True)
    if long_rows:
        with (report_root / "dt_mask_attribution.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
            writer.writeheader()
            writer.writerows(long_rows)
    summary_rows = []
    for variant in VARIANTS:
        for condition in ("id_variable_dt", "unseen_dt_mild", "unseen_dt_hard", "unseen_base_dt_0.5", "unseen_base_dt_2.0"):
            values = [float(row["nrmse"]) for row in long_rows if row["variant"] == variant and row["condition"] == condition]
            if values:
                summary_rows.append({"variant": variant, "condition": condition, "mean_nrmse": sum(values) / len(values), "seeds": len(values)})
    summary = {"schema_version": "pac_dt_mask_attribution_report.v1", "completed_jobs": len(payloads), "expected_jobs": len(attribution_jobs()), "complete": len(payloads) == len(attribution_jobs()), "summary": summary_rows}
    _atomic_json(report_root / "summary.json", summary)
    if summary["complete"]:
        (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return summary


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("enqueue", "worker", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "enqueue":
        print(json.dumps(enqueue(args.root, shards=args.shards), sort_keys=True))
    elif args.stage == "worker":
        print(json.dumps({"completed": worker(args.root, args.shard, device=args.device, smoke=args.smoke)}))
    else:
        print(json.dumps(report(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
