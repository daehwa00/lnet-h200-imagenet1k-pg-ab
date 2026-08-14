#!/usr/bin/env python3
"""Sequential, memory-guarded frozen-Q head-design campaign on one GPU."""

# Experimental runner intentionally owns error messages, lazy optional W&B import,
# and JSON-shaped dynamic payloads in one self-contained process.
# ruff: noqa: ANN401, EM101, PLC0415, PLR0915, T201, TRY003, TRY300

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.a2d_head_design import (
    SCREEN_SPECS,
    DescriptorStatistics,
    HeadDesignSpec,
    build_head,
    descriptor_statistics,
)
from lnet.a2d_q_heads import expected_calibration_error
from lnet.a2d_spectral_prototype import classification_metrics, stratified_fit_calibration_split

DEFAULT_SEEDS = (501, 509, 521)


@dataclass(frozen=True, slots=True)
class PreparedSeed:
    fit_indices: Tensor
    calibration_indices: Tensor
    statistics: DescriptorStatistics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--screen-seed", type=int, default=501)
    parser.add_argument("--screen-epochs", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument(
        "--promote-all",
        action="store_true",
        help="run the final protocol for every screened head",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--minimum-batch-size", type=int, default=256)
    parser.add_argument("--min-free-gib", type=float, default=3.0)
    parser.add_argument("--memory-poll-seconds", type=float, default=15.0)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "alphabet2d-imagenet100")
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "daehwa"))
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_cache(root: Path) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    train = torch.load(root / "train-center-crop.pt", map_location="cpu", weights_only=True)
    validation = torch.load(root / "val-center-crop.pt", map_location="cpu", weights_only=True)
    required = {"features", "labels", "checkpoint_sha256"}
    if not required <= train.keys() or not required <= validation.keys():
        raise RuntimeError("descriptor cache is incomplete")
    if train["checkpoint_sha256"] != validation["checkpoint_sha256"]:
        raise RuntimeError("train and validation descriptors use different checkpoints")
    train_features = cast("Tensor", train["features"]).float().contiguous()
    validation_features = cast("Tensor", validation["features"]).float().contiguous()
    train_labels = cast("Tensor", train["labels"]).long().contiguous()
    validation_labels = cast("Tensor", validation["labels"]).long().contiguous()
    if train_features.shape[1:] != (576,) or validation_features.shape[1:] != (576,):
        raise RuntimeError("head-design suite requires 576-dimensional A2D descriptors")
    if not torch.isfinite(train_features).all() or not torch.isfinite(validation_features).all():
        raise RuntimeError("descriptor cache contains non-finite values")
    return (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        {
            "checkpoint_sha256": train["checkpoint_sha256"],
            "train_samples": train_features.shape[0],
            "validation_samples": validation_features.shape[0],
        },
    )


def _synthetic_cache() -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    generator = torch.Generator().manual_seed(17)
    classes = 4
    labels = torch.arange(classes).repeat_interleave(24)
    validation_labels = torch.arange(classes).repeat_interleave(8)
    centers = torch.randn(classes, 576, generator=generator) * 0.2
    train = centers[labels] + torch.randn(labels.shape[0], 576, generator=generator) * 0.05
    validation = (
        centers[validation_labels]
        + torch.randn(validation_labels.shape[0], 576, generator=generator) * 0.05
    )
    return (
        train,
        labels,
        validation,
        validation_labels,
        {
            "checkpoint_sha256": "synthetic",
            "train_samples": train.shape[0],
            "validation_samples": validation.shape[0],
        },
    )


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    output = classification_metrics(logits.float(), labels)
    output["nll"] = output.pop("cross_entropy")
    output["ece"] = expected_calibration_error(logits.float(), labels)
    return output


def _wait_for_memory(device: torch.device, minimum_gib: float, poll_seconds: float) -> None:
    if device.type != "cuda":
        return
    minimum = int(minimum_gib * 1024**3)
    while True:
        free, total = torch.cuda.mem_get_info(device)
        if free >= minimum:
            print(
                json.dumps(
                    {
                        "event": "memory_ready",
                        "free_gib": free / 1024**3,
                        "total_gib": total / 1024**3,
                    }
                ),
                flush=True,
            )
            return
        print(
            json.dumps(
                {
                    "event": "memory_wait",
                    "free_gib": free / 1024**3,
                    "required_gib": minimum_gib,
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def _to_device(statistics: DescriptorStatistics, device: torch.device) -> DescriptorStatistics:
    return DescriptorStatistics(
        statistics.mean.to(device),
        statistics.std.to(device),
        statistics.rms.to(device),
        statistics.stage_means.to(device),
        statistics.stage_scales.to(device),
        statistics.whitening.to(device),
    )


def _prepare_seed(train_features: Tensor, train_labels: Tensor, seed: int) -> PreparedSeed:
    fit_indices, calibration_indices = stratified_fit_calibration_split(
        train_labels, calibration_fraction=0.1, seed=seed
    )
    return PreparedSeed(
        fit_indices,
        calibration_indices,
        descriptor_statistics(train_features[fit_indices]),
    )


def _wandb_run(
    args: argparse.Namespace,
    spec: HeadDesignSpec,
    *,
    seed: int,
    phase: str,
    parameters: int,
) -> Any:
    if not args.wandb_project or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        return None
    identity = f"{args.output_root.resolve()}::{phase}::{spec.name}::{seed}"
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group="A2D-HeadDesign",
        job_type=phase,
        name=f"{spec.name}-{phase}-s{seed}",
        id=hashlib.sha256(identity.encode()).hexdigest()[:16],
        resume="allow",
        dir=str(args.output_root / "wandb"),
        config={
            **asdict(spec),
            "seed": seed,
            "phase": phase,
            "parameters": parameters,
            "backbone": "A2D-D4-PathMix",
        },
    )


def _hidden_diagnostics(model: nn.Module, features: Tensor) -> dict[str, float]:
    if not hasattr(model, "features"):
        return {}
    model.eval()
    with torch.inference_mode():
        hidden = model.features(features[: min(4096, features.shape[0])])
    return {
        "hidden_rms": float(hidden.square().mean().sqrt()),
        "hidden_mean": float(hidden.mean()),
        "hidden_std": float(hidden.std(unbiased=False)),
        "hidden_zero_fraction": float(hidden.eq(0).float().mean()),
        "hidden_abs_gt_3_fraction": float(hidden.abs().gt(3).float().mean()),
    }


def _evaluate_batched(
    model: nn.Module,
    features: Tensor,
    labels: Tensor,
    *,
    device: torch.device,
    batch_size: int,
    keep_logits: bool = False,
) -> tuple[dict[str, float], Tensor | None]:
    model.eval()
    outputs: list[Tensor] = []
    targets: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size].to(device, non_blocking=True)
            outputs.append(model(batch).float().cpu())
            targets.append(labels[start : start + batch_size])
    logits = torch.cat(outputs)
    target = torch.cat(targets)
    return _metrics(logits, target), logits if keep_logits else None


def _fit_one(
    spec: HeadDesignSpec,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    device: torch.device,
    args: argparse.Namespace,
    phase: str,
    metadata: dict[str, Any],
    prepared: PreparedSeed,
) -> dict[str, Any]:
    started = time.perf_counter()
    fit_features = train_features[prepared.fit_indices]
    fit_labels = train_labels[prepared.fit_indices]
    calibration_features = train_features[prepared.calibration_indices]
    calibration_labels = train_labels[prepared.calibration_indices]
    statistics = _to_device(prepared.statistics, device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_head(spec, statistics, classes=int(train_labels.max()) + 1).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    run = _wandb_run(args, spec, seed=seed, phase=phase, parameters=parameters)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(fit_features.shape[0], generator=generator)
        correct = 0
        seen = 0
        train_loss = 0.0
        for start in range(0, permutation.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            inputs = fit_features[indices].to(device, non_blocking=True)
            targets = fit_labels[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            count = targets.shape[0]
            seen += count
            train_loss += float(loss.detach()) * count
            correct += int(logits.detach().argmax(dim=-1).eq(targets).sum())
        calibration, _ = _evaluate_batched(
            model,
            calibration_features,
            calibration_labels,
            device=device,
            batch_size=batch_size,
        )
        row = {
            "epoch": epoch + 1,
            "train_nll": train_loss / seen,
            "train_accuracy": correct / seen,
            "calibration_nll": calibration["nll"],
            "calibration_accuracy": calibration["accuracy"],
        }
        history.append(row)
        if run is not None:
            run.log(row, step=epoch + 1)
        if calibration["nll"] < best_loss:
            best_loss = calibration["nll"]
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    validation, validation_logits = _evaluate_batched(
        model,
        validation_features,
        validation_labels,
        device=device,
        batch_size=batch_size,
        keep_logits=True,
    )
    train_metrics, _ = _evaluate_batched(
        model,
        fit_features,
        fit_labels,
        device=device,
        batch_size=batch_size,
    )
    diagnostics = _hidden_diagnostics(
        model, validation_features[: min(4096, validation_features.shape[0])].to(device)
    )
    result = {
        "schema": "lnet.a2d.frozen_q_head_design.v1",
        "spec": asdict(spec),
        "phase": phase,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "parameters": parameters,
        "cache": metadata,
        "train": train_metrics,
        "validation": validation,
        "generalization_gap": train_metrics["accuracy"] - validation["accuracy"],
        "hidden": diagnostics,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    artifact = args.output_root / "predictions" / f"{phase}__{spec.name}__s{seed}.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logits": validation_logits,
            "labels": validation_labels,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "classifier_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
                if "classifier" in key or "weights" in key or key == "bias"
            },
        },
        artifact,
    )
    if run is not None:
        run.log({f"validation/{key}": value for key, value in validation.items()})
        run.summary.update(validation)
        run.finish()
    del model, optimizer, statistics, validation_logits
    return result


def _result_path(root: Path, phase: str, spec: HeadDesignSpec, seed: int) -> Path:
    return root / "results" / f"{phase}__{spec.name}__s{seed}.json"


def _run_guarded(
    spec: HeadDesignSpec,
    *,
    phase: str,
    seed: int,
    epochs: int,
    args: argparse.Namespace,
    cache: tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]],
    device: torch.device,
    prepared_cache: dict[int, PreparedSeed],
) -> dict[str, Any]:
    output = _result_path(args.output_root, phase, spec, seed)
    if output.exists():
        return json.loads(output.read_text())
    train_features, train_labels, validation_features, validation_labels, metadata = cache
    if seed not in prepared_cache:
        prepared_cache[seed] = _prepare_seed(train_features, train_labels, seed)
    prepared = prepared_cache[seed]
    batch_size = args.batch_size
    for attempt in range(args.retry_count + 1):
        _wait_for_memory(device, args.min_free_gib, args.memory_poll_seconds)
        try:
            result = _fit_one(
                spec,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                validation_labels=validation_labels,
                device=device,
                args=args,
                phase=phase,
                metadata=metadata,
                prepared=prepared,
            )
            _atomic_json(output, result)
            print(
                json.dumps(
                    {
                        "event": "job_complete",
                        "phase": phase,
                        "name": spec.name,
                        "seed": seed,
                        "accuracy": result["validation"]["accuracy"],
                        "nll": result["validation"]["nll"],
                        "batch_size": batch_size,
                    }
                ),
                flush=True,
            )
            return result
        except torch.OutOfMemoryError:
            failure = traceback.format_exc()
            batch_size //= 2
            if batch_size < args.minimum_batch_size:
                raise RuntimeError("head batch size fell below the configured minimum") from None
            print(
                json.dumps(
                    {
                        "event": "oom_retry",
                        "name": spec.name,
                        "seed": seed,
                        "next_batch_size": batch_size,
                    }
                ),
                flush=True,
            )
            (args.output_root / "failures").mkdir(parents=True, exist_ok=True)
            (args.output_root / "failures" / f"{phase}__{spec.name}__s{seed}__oom.txt").write_text(
                failure
            )
        except Exception:
            failure = traceback.format_exc()
            _atomic_json(
                args.output_root / "failures" / f"{phase}__{spec.name}__s{seed}__a{attempt}.json",
                {"traceback": failure, "attempt": attempt},
            )
            if attempt == args.retry_count:
                raise
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
    raise RuntimeError("head run exhausted retries")


def _write_summary(root: Path) -> None:
    rows = []
    for path in (root / "results").glob("*.json"):
        try:
            rows.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["spec"]["name"], []).append(row)
    summary = {
        "schema": "lnet.a2d.frozen_q_head_design_summary.v1",
        "completed_jobs": len(rows),
        "heads": {
            name: {
                "runs": len(group),
                "mean_accuracy": sum(item["validation"]["accuracy"] for item in group) / len(group),
                "mean_nll": sum(item["validation"]["nll"] for item in group) / len(group),
                "mean_ece": sum(item["validation"]["ece"] for item in group) / len(group),
                "mean_gap": sum(item["generalization_gap"] for item in group) / len(group),
                "parameters": group[0]["parameters"],
            }
            for name, group in sorted(grouped.items())
        },
    }
    _atomic_json(root / "summary.json", summary)


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size < args.minimum_batch_size:
        raise ValueError("initial batch size must not be below minimum batch size")
    if args.finalists <= 0 or args.screen_epochs <= 0 or args.final_epochs <= 0:
        raise ValueError("epochs and finalist count must be positive")
    random.seed(0)
    torch.set_float32_matmul_precision("high")
    cache = _synthetic_cache() if args.smoke_test else _load_cache(args.cache_root)
    if args.smoke_test:
        args.screen_epochs = 1
        args.final_epochs = 1
        args.finalists = min(2, args.finalists)
        args.batch_size = min(32, args.batch_size)
        args.minimum_batch_size = min(16, args.minimum_batch_size)
        args.min_free_gib = 0.0
        args.wandb_project = ""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    selected = [spec for spec in SCREEN_SPECS if not args.only or spec.name in set(args.only)]
    if not selected:
        raise ValueError("no head-design specs selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.output_root / "contract.json",
        {
            "schema": "lnet.a2d.frozen_q_head_design_contract.v1",
            "backbone": "A2D-D4-PathMix",
            "execution": "strictly-sequential",
            "screen_specs": [asdict(spec) for spec in selected],
            "screen_seed": args.screen_seed,
            "screen_epochs": args.screen_epochs,
            "final_epochs": args.final_epochs,
            "final_seeds": args.seeds,
            "finalists": args.finalists,
            "promote_all": args.promote_all,
            "memory_guard_gib": args.min_free_gib,
            "cache": cache[-1],
        },
    )
    screen_rows = []
    prepared_cache: dict[int, PreparedSeed] = {}
    for spec in selected:
        screen_rows.append(
            _run_guarded(
                spec,
                phase="screen",
                seed=args.screen_seed,
                epochs=args.screen_epochs,
                args=args,
                cache=cache,
                device=device,
                prepared_cache=prepared_cache,
            )
        )
        _write_summary(args.output_root)
    ranked = sorted(
        screen_rows,
        key=lambda row: (row["validation"]["nll"], -row["validation"]["accuracy"]),
    )
    finalists = (
        {row["spec"]["name"] for row in ranked}
        if args.promote_all
        else {row["spec"]["name"] for row in ranked[: min(args.finalists, len(ranked))]}
    )
    _atomic_json(
        args.output_root / "promotion.json",
        {
            "criterion": (
                "all screened heads"
                if args.promote_all
                else "lowest validation NLL, accuracy tie-break"
            ),
            "finalists": [
                row["spec"]["name"] for row in ranked if row["spec"]["name"] in finalists
            ],
            "ranking": [
                {
                    "name": row["spec"]["name"],
                    "accuracy": row["validation"]["accuracy"],
                    "nll": row["validation"]["nll"],
                }
                for row in ranked
            ],
        },
    )
    for spec in selected:
        if spec.name not in finalists:
            continue
        for seed in args.seeds:
            _run_guarded(
                spec,
                phase="final",
                seed=seed,
                epochs=args.final_epochs,
                args=args,
                cache=cache,
                device=device,
                prepared_cache=prepared_cache,
            )
            _write_summary(args.output_root)


if __name__ == "__main__":
    main()
