# ruff: noqa: E501, EM101, T201, TRY003
"""Restart-safe synthetic OOD attribution for the final ALPHABET architecture.

The model is trained once per seed on regular, fully observed physical time.
Every OOD condition is then evaluated through metadata-only views of the same
weights, which isolates whether correct elapsed-time and mask information is
actually used rather than confounding the comparison with model capacity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn

from .alphabet import Alphabet
from .pac_matched_zoh_ood import matched_zoh_conditions, matched_zoh_training_task
from .pac_metrics import count_parameters, nrmse
from .pac_training import train_regression_model
from .pac_types import PACDevice, PACExperimentConfig, PACRegressionTask

SEEDS = (23, 31, 43, 47, 59)
VARIANTS = (
    "correct_dt_mask",
    "unit_dt_mask",
    "shuffled_dt_mask",
    "correct_dt_no_mask",
    "unit_dt_no_mask",
)
Variant = Literal[
    "correct_dt_mask",
    "unit_dt_mask",
    "shuffled_dt_mask",
    "correct_dt_no_mask",
    "unit_dt_no_mask",
]
EXPECTED_ALPHABET_PARAMS = 3_138
DEFAULT_ROOT = Path(".omx/results/alphabet-final-radial-log-synthetic-ood-20260726")
SOURCE_FILES = (
    "src/lnet/alphabet.py",
    "src/lnet/alphabet_backbone.py",
    "src/lnet/pac_tight_frame_models.py",
    "src/lnet/pac_alphabet_synthetic_ood.py",
    "src/lnet/pac_matched_zoh_ood.py",
)


@dataclass(frozen=True, slots=True)
class SyntheticOODJob:
    seed: int

    @property
    def key(self) -> str:
        return f"alphabet_synthetic_ood__seed{self.seed}"


class _AlphabetMetadataAttribution(nn.Module):
    def __init__(self, config: PACExperimentConfig) -> None:
        super().__init__()
        active = replace(config, raw_input_dim=2, model_dim=32, modes=16)
        self.model = Alphabet(active, active.output_dim, objective="regression")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_variant(inputs, "correct_dt_mask", seed=0)

    def forward_variant(self, inputs: Tensor, variant: Variant, *, seed: int) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != 4:
            raise ValueError("synthetic ALPHABET inputs must have [value2, dt, mask]")
        delta = inputs[..., 2:3]
        mask: Tensor | None = inputs[..., 3:4]
        if variant in {"unit_dt_mask", "unit_dt_no_mask"}:
            delta = torch.ones_like(delta)
        elif variant == "shuffled_dt_mask":
            delta = _permute_delta(delta, seed)
        if variant in {"correct_dt_no_mask", "unit_dt_no_mask"}:
            mask = None
        return self.model(inputs[..., :2], time_delta=delta, observation_mask=mask)

    def post_optimizer_step(self) -> None:
        self.model.post_optimizer_step()

    def finalize_constraints(self) -> None:
        self.model.finalize_constraints()


def _permute_delta(delta: Tensor, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    order = torch.rand(
        delta.shape[0], delta.shape[1], generator=generator, device="cpu"
    ).argsort(dim=1).to(delta.device)
    return delta.gather(1, order.unsqueeze(-1))


def _endpoint_task(config: PACExperimentConfig, seed: int) -> PACRegressionTask:
    task = matched_zoh_training_task(config, seed)
    return PACRegressionTask(
        label="alphabet_regular_dt_exact_zoh_endpoint",
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
        mechanism_expectation="positive",
    )


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


@torch.no_grad()
def _condition_metrics(
    model: _AlphabetMetadataAttribution,
    inputs: Tensor,
    targets: Tensor,
    *,
    variant: Variant,
    seed: int,
    batch_size: int,
) -> tuple[float, float]:
    predictions = [
        model.forward_variant(batch, variant, seed=seed).detach().cpu()
        for batch in inputs.split(batch_size)
    ]
    predicted = torch.cat(predictions)
    active_targets = targets.detach().cpu()
    mse = float(torch.nn.functional.mse_loss(predicted, active_targets).item())
    return mse, nrmse(mse, active_targets)


def run_job(root: Path, job: SyntheticOODJob, *, device: str, smoke: bool) -> dict[str, object]:
    config = _config(root, job.seed, smoke=smoke)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.seed)
        model = _AlphabetMetadataAttribution(config)
    actual_parameters = count_parameters(model)
    if actual_parameters != EXPECTED_ALPHABET_PARAMS:
        message = (
            "final ALPHABET parameter lock changed: "
            f"{actual_parameters} != {EXPECTED_ALPHABET_PARAMS}"
        )
        raise RuntimeError(
            message
        )
    task = _endpoint_task(config, job.seed)
    outcome = train_regression_model(model, task, config, device, job.seed)
    model.eval()
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(matched_zoh_conditions(config, job.seed)):
        inputs = condition.ood_inputs.to(device)
        targets = condition.ood_targets[:, -1]
        for variant in VARIANTS:
            mse, active_nrmse = _condition_metrics(
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
        "schema": "alphabet.synthetic_ood_result.v1",
        "job_key": job.key,
        "seed": job.seed,
        "status": "done",
        "smoke": smoke,
        "model": "ALPHABET",
        "internal_spec": "radial_log_r_affine",
        "model_dim": 32,
        "modes": 16,
        "params_trainable": actual_parameters,
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "id_test_loss": outcome.test_loss,
        "id_test_nrmse": nrmse(outcome.test_loss, task.test_targets),
        "elapsed_seconds": outcome.elapsed_time,
        "training_distribution": {
            "physical_horizon": 60.0,
            "base_dt": 1.0,
            "irregularity": 0.0,
            "missing_rate": 0.0,
        },
        "conditions": rows,
    }


def enqueue(root: Path, *, shards: int = 5) -> dict[str, object]:
    if shards < 1:
        raise ValueError("shards must be positive")
    root.mkdir(parents=True, exist_ok=True)
    (root / "completed").mkdir(exist_ok=True)
    jobs = [SyntheticOODJob(seed) for seed in SEEDS]
    repository = Path(__file__).resolve().parents[2]
    contract = {
        "schema": "alphabet.synthetic_ood_contract.v1",
        "public_model": "ALPHABET",
        "internal_spec": "radial_log_r_affine",
        "model_class": "lnet.alphabet.Alphabet",
        "model_dim": 32,
        "modes": 16,
        "expected_params": EXPECTED_ALPHABET_PARAMS,
        "seeds": list(SEEDS),
        "jobs": len(jobs),
        "shards": shards,
        "variants": list(VARIANTS),
        "training_distribution": "regular dt=1, fully observed, physical horizon 60",
        "evaluation_source": "pac_matched_zoh_ood.matched_zoh_conditions",
        "same_weight_metadata_attribution": True,
        "test_evidence_used_for_training_or_selection": False,
        "source_sha256": {
            relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
            for relative in SOURCE_FILES
        },
        "locked_before_execution": True,
    }
    _atomic_json(root / "contract.json", contract)
    for shard in range(shards):
        manifest = root / f"manifest-shard{shard}.jsonl"
        manifest.write_text(
            "".join(
                json.dumps(asdict(job), sort_keys=True) + "\n"
                for index, job in enumerate(jobs)
                if index % shards == shard
            ),
            encoding="utf-8",
        )
    return {"jobs": len(jobs), "shards": shards}


def worker(root: Path, shard: int, *, device: str, smoke: bool) -> int:
    manifest = root / f"manifest-shard{shard}.jsonl"
    completed = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        job = SyntheticOODJob(seed=int(payload["seed"]))
        destination = root / "completed" / f"{job.key}.json"
        if destination.is_file():
            continue
        result = run_job(root, job, device=device, smoke=smoke)
        _atomic_json(destination, result)
        completed += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return completed


def status(root: Path) -> dict[str, object]:
    rows = []
    for path in sorted((root / "completed").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "done" and not row.get("smoke"):
            rows.append(row)
    expected = len(SEEDS)
    return {
        "completed": len(rows),
        "expected": expected,
        "remaining": expected - len(rows),
        "done": len(rows) == expected,
    }


def report(root: Path) -> dict[str, object]:
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
        if not json.loads(path.read_text(encoding="utf-8")).get("smoke")
    ]
    rows = [
        {"seed": payload["seed"], **condition}
        for payload in payloads
        for condition in payload["conditions"]
    ]
    report_root = root / "reports"
    report_root.mkdir(exist_ok=True)
    if rows:
        with (report_root / "synthetic_ood_long.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["level"]), str(row["variant"]))
        grouped.setdefault(key, []).append(float(row["nrmse"]))
    summary_rows = [
        {
            "family": key[0],
            "level": key[1],
            "variant": key[2],
            "mean_nrmse": sum(values) / len(values),
            "seeds": len(values),
        }
        for key, values in sorted(grouped.items())
    ]
    summary = {
        "schema": "alphabet.synthetic_ood_summary.v1",
        **status(root),
        "summary": summary_rows,
    }
    _atomic_json(report_root / "summary.json", summary)
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
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "enqueue":
        payload = enqueue(args.root, shards=args.shards)
    elif args.stage == "worker":
        payload = {"completed_now": worker(args.root, args.shard, device=args.device, smoke=args.smoke)}
    elif args.stage == "status":
        payload = status(args.root)
    else:
        payload = report(args.root)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
