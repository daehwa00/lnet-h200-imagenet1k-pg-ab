"""Restart-safe UCR-16 validation screen for lag versus pole-attention readers."""

# ruff: noqa: BLE001, EM101, EM102, T201, TRY003

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, cast

import torch

from lnet.pac_device import resolve_device
from lnet.pac_eval_sections import clean_validation_classification_task
from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_metrics import count_parameters
from lnet.pac_real_data import ensure_ucr_train_only
from lnet.pac_training import classification_metric_bundle, train_classifier
from lnet.pac_types import PACDevice, PACExperimentConfig
from optimization.learned_two_tap_pole_attention import (
    LagReaderALPHABET,
    PoleAttentionALPHABET,
)

if TYPE_CHECKING:
    from torch import nn

ROOT: Final = Path(".omx/results/pac-pole-attention-ucr16-fast-20260720")
DATA_ROOT: Final = Path(".omx/data/ucr")
DATASETS: Final = (
    "ArrowHead",
    "CinCECGTorso",
    "CricketX",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "Earthquakes",
    "GunPoint",
    "ItalyPowerDemand",
    "MoteStrain",
    "Phoneme",
    "Plane",
    "StarLightCurves",
    "Trace",
    "TwoLeadECG",
    "Wafer",
)
SEEDS: Final = (7, 11, 19, 23, 31)
ReaderVariant = Literal["lag_reader", "pole_attention"]
VARIANTS: Final[tuple[ReaderVariant, ...]] = ("lag_reader", "pole_attention")
MODEL_DIM: Final = 32
MODES: Final = 16
HEADS: Final = 2
EPOCHS: Final = 100
BATCH_SIZE: Final = 64
LEARNING_RATE: Final = 3.0e-3
WEIGHT_DECAY: Final = 1.0e-4
GRAD_CLIP_NORM: Final = 1.0


@dataclass(frozen=True, slots=True)
class Job:
    key: str
    dataset: str
    variant: ReaderVariant
    split_seed: int
    train_seed: int
    model_dim: int
    modes: int
    heads: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    evaluation_split: Literal["validation"]
    estimated_seconds: float
    design_sha256: str


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("optimization/learned_two_tap_alphabet.py"),
        Path("optimization/learned_two_tap_pole_attention.py"),
        Path("optimization/masked_modal_moments.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
    )
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _design() -> dict[str, object]:
    return {
        "schema": "pac_pole_attention_ucr16_fast_contract.v1",
        "purpose": "validation-only replacement screen for the fixed lag-1/lag-4 reader descriptor",
        "official_test_accessed": False,
        "datasets": list(DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "model_dim": MODEL_DIM,
        "modes": MODES,
        "attention_heads": HEADS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "hyperparameter_tuning": False,
        "controlled_difference": (
            "the optimized learned-two-tap writer, reader local lift, exact-pole scan, "
            "writer moments, pooled real stream, D, M, and classifier input width are fixed; "
            "reader lag-1/lag-4 coherences are replaced by two-head per-mode normalized "
            "Hermitian last-query pooling with RMS-normalized real values"
        ),
        "attention_contract": {
            "query": "last valid reader complex state",
            "similarity": (
                "FP32 per-mode normalized Hermitian products; all real and imaginary "
                "relative-phase coordinates feed every bounded head projection"
            ),
            "logits": (
                "bounded learned real/imaginary mixture, positive temperature, "
                "bounded relative-time bias"
            ),
            "values": "RMS-normalized reader real features",
            "all_padding": "zero descriptor",
            "complexity": "linear in sequence length",
            "reader_descriptor_width": "5M for both variants",
        },
        "source_sha256": _source_hashes(),
    }


def design_sha256() -> str:
    encoded = json.dumps(_design(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jobs() -> list[Job]:
    digest = design_sha256()
    return [
        Job(
            key=f"pole_attention_ucr16_fast:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=variant,
            split_seed=seed,
            train_seed=seed,
            model_dim=MODEL_DIM,
            modes=MODES,
            heads=HEADS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            grad_clip_norm=GRAD_CLIP_NORM,
            evaluation_split="validation",
            estimated_seconds=UCR_SECONDS[dataset],
            design_sha256=digest,
        )
        for dataset in DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    ]


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _result_path(root: Path, job: Job, *, failed: bool) -> Path:
    bucket = "failed" if failed else "completed"
    return root / bucket / f"{job.key.replace(':', '__')}.json"


def _result_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(item.read_text(encoding="utf-8"))["job_key"])
        for item in path.glob("*.json")
    }


def enqueue(root: Path, workers: int) -> dict[str, object]:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    expected = jobs()
    completed = _result_keys(root / "completed")
    pending = [job for job in expected if job.key not in completed]
    shards: list[list[Job]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(pending, key=lambda item: (-item.estimated_seconds, item.key)):
        index = min(range(workers), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for stale in manifests.glob("worker-*.jsonl"):
        stale.unlink()
    for index, shard in enumerate(shards):
        (manifests / f"worker-{index:02d}.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard),
            encoding="utf-8",
        )
    contract = {
        **_design(),
        "design_sha256": design_sha256(),
        "jobs": len(expected),
        "workers": workers,
        "estimated_worker_seconds": loads,
        "restart_safe": True,
    }
    _atomic_json(root / "contract.json", contract)
    return {"jobs": len(expected), "pending": len(pending), "workers": workers, "loads": loads}


def _build_model(job: Job, input_dim: int, output_dim: int) -> nn.Module:
    if job.variant == "lag_reader":
        return LagReaderALPHABET(input_dim, job.model_dim, job.modes, output_dim)
    return PoleAttentionALPHABET(
        input_dim,
        job.model_dim,
        job.modes,
        output_dim,
        heads=job.heads,
    )


def run_job(job: Job, *, device: PACDevice, data_root: Path) -> dict[str, object]:
    if job.evaluation_split != "validation" or job.split_seed != job.train_seed:
        raise ValueError("job violates the sealed validation-only split contract")
    if job.design_sha256 != design_sha256():
        raise RuntimeError("job design hash does not match the current candidate sources")
    runtime_device = resolve_device(device)
    dataset = ensure_ucr_train_only(job.dataset, data_root, allow_download=True)
    task = clean_validation_classification_task(dataset, job.split_seed)
    if task.test_inputs.shape[0] or task.test_labels.shape[0]:
        raise RuntimeError("validation worker unexpectedly received UCR TEST examples")
    config = PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        0,
        task.train_inputs.shape[1],
        raw_input_dim=task.train_inputs.shape[-1],
        output_dim=task.class_count,
        model_dim=job.model_dim,
        modes=job.modes,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=job.learning_rate,
        weight_decay=job.weight_decay,
        grad_clip_norm=job.grad_clip_norm,
        seeds=(job.train_seed,),
        device=device,
        optimizer_mode="fused" if runtime_device == "cuda" else "default",
    )
    torch.manual_seed(job.train_seed)
    if runtime_device == "cuda":
        torch.cuda.manual_seed_all(job.train_seed)
        torch.cuda.reset_peak_memory_stats()
    model = _build_model(job, task.train_inputs.shape[-1], task.class_count).to(
        device=runtime_device
    )
    started = perf_counter()
    outcome = train_classifier(
        model,
        task,
        config,
        runtime_device,
        job.train_seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    elapsed = perf_counter() - started
    metrics = classification_metric_bundle(
        model,
        task.validation_inputs.to(device=runtime_device),
        task.validation_labels.to(device=runtime_device),
        batch_size=job.batch_size,
    )
    return {
        "schema": "pac_pole_attention_ucr16_fast_result.v1",
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "official_test_accessed": False,
        "test_evaluated": False,
        "test_count": 0,
        "checkpoint_policy": "minimum TRAIN-derived validation loss",
        "train_count": int(task.train_inputs.shape[0]),
        "validation_count": int(task.validation_inputs.shape[0]),
        "best_epoch": outcome.best_epoch,
        "validation_loss": outcome.validation_loss,
        "validation_accuracy": metrics.accuracy,
        "validation_macro_f1": metrics.macro_f1,
        "validation_weighted_f1": metrics.weighted_f1,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        "params_trainable": count_parameters(model),
        "train_seconds": elapsed,
        "peak_memory_mb": (
            float(torch.cuda.max_memory_allocated() / 1_000_000)
            if runtime_device == "cuda"
            else 0.0
        ),
    }


def run_manifest(root: Path, manifest: Path, *, device: PACDevice, data_root: Path) -> None:
    scheduled = [
        Job(**json.loads(line))
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for job in scheduled:
        completed = _result_path(root, job, failed=False)
        if completed.exists():
            continue
        try:
            row = run_job(job, device=device, data_root=data_root)
        except Exception as error:
            _atomic_json(
                _result_path(root, job, failed=True),
                {
                    "schema": "pac_pole_attention_ucr16_fast_failure.v1",
                    "job_key": job.key,
                    **asdict(job),
                    "status": "failed",
                    "official_test_accessed": False,
                    "test_evaluated": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
        else:
            _atomic_json(completed, row)
            _result_path(root, job, failed=True).unlink(missing_ok=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def status(root: Path) -> dict[str, object]:
    expected = {job.key for job in jobs()}
    completed = _result_keys(root / "completed")
    failed = _result_keys(root / "failed") - completed
    unexpected = (completed | failed) - expected
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed - failed),
        "unexpected": len(unexpected),
        "done": expected == completed and not failed and not unexpected,
    }


def _ranks(scores: dict[str, float]) -> dict[str, float]:
    first, second = VARIANTS
    if math.isclose(scores[first], scores[second], rel_tol=1.0e-5, abs_tol=1.0e-8):
        return {first: 1.5, second: 1.5}
    winner = max(VARIANTS, key=scores.__getitem__)
    return {variant: 1.0 if variant == winner else 2.0 for variant in VARIANTS}


def report(root: Path) -> dict[str, object]:
    # Workers only train models; keep SciPy as a report-time dependency so
    # minimal CUDA environments can execute manifests.
    from scipy.stats import t as student_t  # noqa: PLC0415
    from scipy.stats import ttest_1samp  # noqa: PLC0415

    campaign_status = status(root)
    if campaign_status["done"] is not True:
        raise RuntimeError(f"refusing to report incomplete results: {campaign_status}")
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]
    expected = {job.key: job for job in jobs()}
    if len(rows) != len(expected):
        raise RuntimeError("completed row count disagrees with the sealed design")
    for row in rows:
        job = expected.get(str(row.get("job_key")))
        if (
            job is None
            or row.get("schema") != "pac_pole_attention_ucr16_fast_result.v1"
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or row.get("test_count") != 0
            or row.get("design_sha256") != design_sha256()
        ):
            raise RuntimeError(f"invalid or contaminated row: {row.get('job_key')}")
    rank_sums = dict.fromkeys(VARIANTS, 0.0)
    top_counts = dict.fromkeys(VARIANTS, 0)
    deltas: list[float] = []
    datasets: list[dict[str, object]] = []
    for dataset in DATASETS:
        scores: dict[str, float] = {}
        sample_sds: dict[str, float] = {}
        for variant in VARIANTS:
            values = [
                float(row["validation_balanced_accuracy"])
                for row in rows
                if row["dataset"] == dataset and row["variant"] == variant
            ]
            if len(values) != len(SEEDS):
                raise RuntimeError(f"incomplete cell: {dataset}/{variant}")
            scores[variant] = mean(values)
            sample_sds[variant] = stdev(values)
        ranks = _ranks(scores)
        for variant in VARIANTS:
            rank_sums[variant] += ranks[variant]
            top_counts[variant] += int(ranks[variant] <= 1.5)
        delta = scores["pole_attention"] - scores["lag_reader"]
        deltas.append(delta)
        datasets.append(
            {
                "dataset": dataset,
                "means": scores,
                "sample_sds": sample_sds,
                "ranks": ranks,
                "attention_minus_lag": delta,
            }
        )
    aggregate: dict[str, object] = {}
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            "row_mean_balanced_accuracy": mean(
                float(row["validation_balanced_accuracy"]) for row in variant_rows
            ),
            "mean_rank": rank_sums[variant] / len(DATASETS),
            "joint_top1": top_counts[variant],
            "params_trainable": sorted({int(row["params_trainable"]) for row in variant_rows}),
            "mean_train_seconds": mean(float(row["train_seconds"]) for row in variant_rows),
        }
    paired = ttest_1samp(deltas, popmean=0.0)
    sem = stdev(deltas) / math.sqrt(len(deltas))
    critical = float(student_t.ppf(0.975, len(deltas) - 1))
    delta_mean = mean(deltas)
    payload: dict[str, object] = {
        "schema": "pac_pole_attention_ucr16_fast_report.v1",
        "status": campaign_status,
        "official_test_accessed": False,
        "rows": len(rows),
        "aggregate": aggregate,
        "paired_inference": {
            "unit": "dataset-level five-seed mean",
            "attention_minus_lag_mean": delta_mean,
            "ci95": [delta_mean - critical * sem, delta_mean + critical * sem],
            "two_sided_t_pvalue": float(cast("float", paired[1])),
            "wins_ties_losses": {
                "wins": sum(delta > 1.0e-8 for delta in deltas),
                "ties": sum(abs(delta) <= 1.0e-8 for delta in deltas),
                "losses": sum(delta < -1.0e-8 for delta in deltas),
            },
        },
        "datasets": datasets,
    }
    _atomic_json(root / "reports/summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("enqueue", "worker", "status", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    if args.command == "enqueue":
        payload = enqueue(args.root, args.workers)
    elif args.command == "worker":
        if args.manifest is None:
            parser.error("worker requires --manifest")
        run_manifest(
            args.root,
            args.manifest,
            device=cast("PACDevice", args.device),
            data_root=args.data_root,
        )
        payload = status(args.root)
    elif args.command == "report":
        payload = report(args.root)
    else:
        payload = status(args.root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
