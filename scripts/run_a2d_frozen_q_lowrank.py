#!/usr/bin/env python3
"""Measure the effective rank of the frozen A2D affine Q classifier."""

# ruff: noqa: ANN401, C901, EM101, EM102, I001, PLC0415, PLR0912, PLR0915, SLF001, T201, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import run_a2d_frozen_q_suite as frozen


DEFAULT_RANKS = (4, 8, 16, 32, 48, 64, 80, 96, 100)


class FactorizedAffineHead(nn.Module):
    """Affine classifier constrained to a rank-r weight matrix."""

    def __init__(self, input_dim: int, classes: int, rank: int) -> None:
        super().__init__()
        maximum_rank = min(input_dim, classes)
        if not 0 < rank <= maximum_rank:
            raise ValueError(f"rank must lie in [1, {maximum_rank}]")
        self.input_projection = nn.Linear(input_dim, rank, bias=False)
        self.class_projection = nn.Linear(rank, classes, bias=True)

    def forward(self, features: Tensor) -> Tensor:
        return self.class_projection(self.input_projection(features))

    def equivalent_parameters(self) -> tuple[Tensor, Tensor]:
        weight = self.class_projection.weight @ self.input_projection.weight
        return weight, self.class_projection.bias


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(frozen.SEEDS))
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "alphabet2d-imagenet100"),
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", "daehwa"),
    )
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _wandb_run(
    args: argparse.Namespace,
    *,
    name: str,
    seed: int,
    parameters: int,
) -> Any:
    if not args.wandb_project or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        return None
    key = f"{args.output_root.resolve()}::{name}::{seed}"
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group="A2D-QRank",
        job_type="frozen-q-lowrank",
        name=f"{name}-s{seed}",
        id=hashlib.sha256(key.encode()).hexdigest()[:16],
        resume="allow",
        dir=str(args.output_root / "wandb"),
        config={
            "head": name,
            "seed": seed,
            "parameters": parameters,
            "backbone": "A2D-D4-PathMix",
            "descriptor_dim": 576,
        },
    )


def _fit(
    model: nn.Module,
    data: frozen.SeedData,
    args: argparse.Namespace,
    *,
    seed: int,
    run: Any,
) -> tuple[nn.Module, list[dict[str, float]]]:
    def callback(row: dict[str, float]) -> None:
        if run is not None:
            run.log(
                {
                    "epoch": row["epoch"],
                    "calibration/nll": row["calibration_nll"],
                    "calibration/accuracy": row["calibration_accuracy"],
                },
                step=int(row["epoch"]),
            )

    return frozen._fit_model(
        model,
        data,
        epochs=args.head_epochs,
        batch_size=args.head_batch_size,
        seed=seed,
        epoch_callback=callback,
    )


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    return frozen._metrics(logits, labels)


def _save_artifact(
    path: Path,
    *,
    model: nn.Module,
    data: frozen.SeedData,
    logits: Tensor,
    metadata: dict[str, Any],
) -> None:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    _atomic_torch_save(
        path,
        {
            "schema": "lnet.a2d.frozen_q_lowrank_artifact.v1",
            "state_dict": state,
            "normalization_mean": data.mean.detach().cpu(),
            "normalization_std": data.std.detach().cpu(),
            "validation_logits": logits.detach().cpu().to(torch.float16),
            "validation_labels": data.validation_labels.detach().cpu(),
            "cache": metadata,
        },
    )


def _svd_diagnostics(
    model: nn.Linear,
    data: frozen.SeedData,
    ranks: list[int],
) -> dict[str, Any]:
    with torch.inference_mode():
        weight = model.weight.float()
        bias = model.bias.float()
        left, singular_values, right = torch.linalg.svd(weight, full_matrices=False)
        energy = singular_values.square()
        cumulative = energy.cumsum(0) / energy.sum().clamp_min(1.0e-12)
        truncations: dict[str, Any] = {}
        for rank in ranks:
            active_rank = min(rank, singular_values.numel())
            approximation = (
                left[:, :active_rank] * singular_values[:active_rank]
            ) @ right[:active_rank]
            logits = data.validation @ approximation.T + bias
            truncations[str(rank)] = {
                "rank": active_rank,
                "parameters": 576 * active_rank + active_rank * 100 + 100,
                "retained_weight_energy": float(cumulative[active_rank - 1]),
                "validation": _metrics(logits, data.validation_labels),
            }
    return {
        "singular_values": singular_values.detach().cpu().tolist(),
        "cumulative_weight_energy": cumulative.detach().cpu().tolist(),
        "truncated_svd": truncations,
    }


def _run_full_linear(
    data: frozen.SeedData,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    *,
    seed: int,
    ranks: list[int],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = nn.Linear(576, int(data.classes.numel())).to(data.fit.device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    run = _wandb_run(args, name="QFull", seed=seed, parameters=parameters)
    started = time.perf_counter()
    model, history = _fit(model, data, args, seed=seed, run=run)
    with torch.inference_mode():
        logits = model(data.validation)
    validation = _metrics(logits, data.validation_labels)
    result = {
        "schema": "lnet.a2d.frozen_q_lowrank.v1",
        "name": "QFull",
        "seed": seed,
        "rank": min(576, int(data.classes.numel())),
        "parameters": parameters,
        "cache": metadata,
        "validation": validation,
        "history": history,
        "svd": _svd_diagnostics(model, data, ranks),
        "seconds": time.perf_counter() - started,
    }
    _save_artifact(
        args.output_root / "artifacts" / f"QFull__s{seed}.pt",
        model=model,
        data=data,
        logits=logits,
        metadata=metadata,
    )
    if run is not None:
        run.log({f"validation/{key}": value for key, value in validation.items()})
        for rank, row in result["svd"]["truncated_svd"].items():
            run.log(
                {
                    "svd/rank": int(rank),
                    "svd/accuracy": row["validation"]["accuracy"],
                    "svd/retained_weight_energy": row["retained_weight_energy"],
                }
            )
        run.finish()
    return result


def _run_factorized(
    data: frozen.SeedData,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    *,
    seed: int,
    rank: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    name = f"QRank{rank}"
    model = FactorizedAffineHead(576, int(data.classes.numel()), rank).to(data.fit.device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    run = _wandb_run(args, name=name, seed=seed, parameters=parameters)
    started = time.perf_counter()
    model, history = _fit(model, data, args, seed=seed, run=run)
    with torch.inference_mode():
        logits = model(data.validation)
        weight, _ = model.equivalent_parameters()
        effective_rank = int(torch.linalg.matrix_rank(weight.float()).item())
    validation = _metrics(logits, data.validation_labels)
    result = {
        "schema": "lnet.a2d.frozen_q_lowrank.v1",
        "name": name,
        "seed": seed,
        "rank": rank,
        "effective_weight_rank": effective_rank,
        "parameters": parameters,
        "cache": metadata,
        "validation": validation,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    _save_artifact(
        args.output_root / "artifacts" / f"{name}__s{seed}.pt",
        model=model,
        data=data,
        logits=logits,
        metadata=metadata,
    )
    if run is not None:
        run.log({f"validation/{key}": value for key, value in validation.items()})
        run.summary.update(validation)
        run.finish()
    return result


def _write_summary(root: Path, expected: int) -> None:
    rows = []
    for path in sorted((root / "results").glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("schema") == "lnet.a2d.frozen_q_lowrank.v1":
            rows.append(row)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["name"], []).append(row["validation"]["accuracy"])
    _atomic_json(
        root / "summary.json",
        {
            "schema": "lnet.a2d.frozen_q_lowrank_summary.v1",
            "completed_jobs": len(rows),
            "expected_jobs": expected,
            "accuracy": {
                name: {
                    "runs": len(values),
                    "mean": sum(values) / len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for name, values in sorted(grouped.items())
            },
        },
    )


def main() -> None:
    args = _parser().parse_args()
    if args.head_epochs <= 0 or args.head_batch_size <= 0:
        raise ValueError("head epochs and batch size must be positive")
    if args.retry_count < 0:
        raise ValueError("retry count must be non-negative")
    if args.smoke_test:
        cache = frozen._synthetic_cache()
        args.ranks = [rank for rank in args.ranks if rank <= 4] or [1, 2, 4]
        args.seeds = [args.seeds[0]]
        args.head_epochs = 1
        args.head_batch_size = min(args.head_batch_size, 32)
        args.wandb_project = ""
    else:
        cache = frozen._load_cache(args.cache_root)
    ranks = sorted(set(args.ranks))
    train_features, train_labels, validation_features, validation_labels, metadata = cache
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("frozen-Q low-rank worker cannot see a CUDA GPU")
    classes = int(train_labels.unique().numel())
    if any(rank <= 0 or rank > min(576, classes) for rank in ranks):
        raise ValueError(f"ranks must lie in [1, {min(576, classes)}]")
    args.output_root.mkdir(parents=True, exist_ok=True)
    expected = len(args.seeds) * (1 + len(ranks))
    _atomic_json(
        args.output_root / "contract.json",
        {
            "schema": "lnet.a2d.frozen_q_lowrank_contract.v1",
            "ranks": ranks,
            "seeds": args.seeds,
            "head_epochs": args.head_epochs,
            "head_batch_size": args.head_batch_size,
            "expected_jobs": expected,
            "cache": metadata,
        },
    )
    random.seed(0)
    torch.set_float32_matmul_precision("high")
    seed_cache: dict[int, frozen.SeedData] = {}
    failures = 0
    for seed in args.seeds:
        if seed not in seed_cache:
            seed_cache[seed] = frozen._seed_data(
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                seed=seed,
                device=device,
            )
        data = seed_cache[seed]
        jobs = [("QFull", None), *[(f"QRank{rank}", rank) for rank in ranks]]
        for name, rank in jobs:
            result_path = args.output_root / "results" / f"{name}__s{seed}.json"
            if result_path.exists():
                continue
            for attempt in range(args.retry_count + 1):
                try:
                    result = (
                        _run_full_linear(data, metadata, args, seed=seed, ranks=ranks)
                        if rank is None
                        else _run_factorized(data, metadata, args, seed=seed, rank=rank)
                    )
                    _atomic_json(result_path, result)
                    print(
                        json.dumps(
                            {
                                "event": "job_complete",
                                "name": name,
                                "seed": seed,
                                "accuracy": result["validation"]["accuracy"],
                                "seconds": result["seconds"],
                            }
                        ),
                        flush=True,
                    )
                    break
                except Exception as error:  # noqa: BLE001
                    failure = {
                        "event": "job_failure",
                        "name": name,
                        "seed": seed,
                        "attempt": attempt,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                    _atomic_json(
                        args.output_root
                        / "failures"
                        / f"{name}__s{seed}__a{attempt}.json",
                        failure,
                    )
                    print(json.dumps(failure), flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if attempt == args.retry_count:
                        failures += 1
            _write_summary(args.output_root, expected)
    _write_summary(args.output_root, expected)
    if failures:
        raise RuntimeError(f"{failures} low-rank jobs exhausted all retries")


if __name__ == "__main__":
    main()
