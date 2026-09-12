#!/usr/bin/env python3
"""Bounded, non-production readiness proof for dense-transfer training.

This command never invokes the production ``train`` mode.  Its disposable
model receives at most two optimizer updates; readiness is granted only after
finite train/eval, checkpoint restore, and (when enabled) continuation parity.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import numpy as np

import run_dense_transfer as training
from dense_transfer.engine import (
    TrainProgress, atomic_json, atomic_torch, checkpoint_payload, contract_hash,
    load_checkpoint, require_idle_cuda, rng_state, set_seed, train_batches, validate_device,
)


def parser() -> argparse.ArgumentParser:
    command = training.parser()
    command.description = __doc__
    command.add_argument("--max-probe-updates", type=int, default=2)
    command.add_argument("--parity-rtol", type=float, default=1e-5)
    command.add_argument("--parity-atol", type=float, default=1e-6)
    return command


class _LimitedLoader:
    """Preserve ``dataset`` while exposing only an evaluation label subset."""

    def __init__(self, loader: Any, batches: int):
        self.batches = list(itertools.islice(iter(loader), batches))
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.batches)


def _same(value: object, other: object) -> bool:
    if isinstance(value, torch.Tensor) and isinstance(other, torch.Tensor):
        return torch.equal(value.cpu(), other.cpu())
    if isinstance(value, np.ndarray) and isinstance(other, np.ndarray):
        return np.array_equal(value, other)
    if isinstance(value, float) and isinstance(other, float):
        return value == other or (math.isnan(value) and math.isnan(other))
    if isinstance(value, dict) and isinstance(other, dict):
        return value.keys() == other.keys() and all(_same(value[key], other[key]) for key in value)
    if isinstance(value, (list, tuple)) and isinstance(other, (list, tuple)):
        return len(value) == len(other) and all(_same(left, right) for left, right in zip(value, other, strict=True))
    return value == other


def _weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _assert_close(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor], rtol: float, atol: float) -> None:
    if expected.keys() != actual.keys():
        raise AssertionError("model state keys changed during restore proof")
    for name in expected:
        try:
            torch.testing.assert_close(actual[name], expected[name], rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise AssertionError(f"continuation parity failed for {name}: {exc}") from exc


def readiness_payload(contract: dict[str, object], evidence: dict[str, object], passed: bool) -> dict[str, object]:
    required = ("fullinput_train_finite", "eval_subset_finite", "restore_model_exact", "restore_optimizer_exact", "restore_scheduler_exact", "restore_rng_exact", "restore_eval_exact")
    passed = passed and all(evidence.get(key) is True for key in required) and isinstance(evidence.get("inspect_contract_hash"), str)
    return {
        "status": "ready" if passed else "needs_attention",
        "contract_hash": contract_hash(contract),
        "inspect_contract_hash": evidence.get("inspect_contract_hash"),
        "contract": contract,
        "evidence": evidence,
        "autostart": False,
    }


def _finite(value: object) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return False


def _finite_aggregate_metrics(metrics: dict[str, object]) -> bool:
    """Only scalar aggregates gate readiness; absent-class detail may be NaN."""
    scalars = [value for value in metrics.values() if isinstance(value, (int, float))]
    return bool(scalars) and all(math.isfinite(float(value)) for value in scalars)


def write_readiness(output_root: Path, contract: dict[str, object], evidence: dict[str, object], passed: bool) -> Path:
    path = output_root / "queue" / "readiness.json"
    atomic_json(path, readiness_payload(contract, evidence, passed))
    return path


def _take(iterator: Iterator[Any], count: int) -> list[Any]:
    batches = list(itertools.islice(iterator, count))
    if len(batches) != count:
        raise RuntimeError(f"training loader supplied {len(batches)} batches; need {count} for accumulation proof")
    return batches


def run_validation(args: argparse.Namespace) -> dict[str, object]:
    if args.max_probe_updates <= 0 or args.max_probe_updates > 2:
        raise ValueError("--max-probe-updates must be 1 or 2")
    settings, plan = training.settings_from_args(args), training.plan_from_args(args)
    try:
        inspect_contract_hash = contract_hash(training.runtime_contract(settings, plan))
        # Guard before build_runtime/model allocation, regardless of base --mode.
        device = validate_device(settings.device)
        require_idle_cuda(device, False)
        set_seed(settings.seed)
        model, train_loader, val_loader, optimizer, scheduler, contract, runtime_device = training.build_runtime(settings, plan)
    except Exception as exc:
        fallback = {"schema": "dense-transfer.readiness-preflight.v1", "task": settings.task, "model_key": settings.model_key}
        early = {"production_train_started": False, "inspect_contract_hash": contract_hash(fallback), "error_type": type(exc).__name__, "error": str(exc), "cuda_oom": isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()}
        path = write_readiness(settings.output_root, fallback, early, False)
        return {"status": "needs_attention", "readiness_path": str(path), "evidence": early}
    probe_root = settings.output_root / "readiness_probe"
    probe_checkpoint = probe_root / "resume-proof.pt"
    evidence: dict[str, object] = {
        "probe_mode": args.mode, "production_train_started": False, "max_probe_updates": args.max_probe_updates,
        "runtime_contract_hash": contract_hash(contract),
        "inspect_contract_hash": inspect_contract_hash,
        "source_dataset_pretrain_hashes": {key: contract.get(key) for key in ("checkpoint_sha256", "checkpoint_prevalidation", "dataset", "backbone", "runtime_source_sha256")},
        "parity_batch_source": "fixed_in_memory_batches_no_dataloader_replay",
        "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
    }
    initial = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict()), rng_state())
    try:
        iterator = iter(train_loader)
        first = _take(iterator, settings.accumulation_steps)
        start = time.perf_counter()
        if runtime_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime_device)
            torch.cuda.synchronize(runtime_device)
        progress = TrainProgress()
        train_result = train_batches(model, first, optimizer, scheduler, runtime_device, settings.accumulation_steps, progress=progress, max_updates=1, bf16=settings.bf16, channels_last=settings.channels_last, physical_batch_size=settings.physical_batch_size)
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)
        evidence["fullinput_train_finite"] = math.isfinite(float(train_result["loss"]))
        evidence["train_seconds"] = time.perf_counter() - start
        if runtime_device.type == "cuda":
            evidence["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(runtime_device))
            evidence["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(runtime_device))

        generator = getattr(train_loader, "dense_transfer_generator", None)
        atomic_torch(probe_checkpoint, checkpoint_payload(model, optimizer, scheduler, progress, contract, generator))
        saved_rng = rng_state()
        data_module = importlib.import_module("dense_transfer.data")
        eval_subset = _LimitedLoader(val_loader, 1)
        eval_before = training.evaluate(model, eval_subset, runtime_device, data_module, settings)
        evidence["eval_subset_batches"] = 1
        evidence["eval_subset_finite"] = _finite_aggregate_metrics(eval_before)

        # Verify exact model/optimizer/scheduler/RNG restoration before optional
        # continuation parity.  The probe's fixed in-memory batches eliminate
        # dataloader-worker RNG ambiguity from this proof.
        mutated = _weights(model)
        load_checkpoint(probe_checkpoint, model, optimizer, scheduler, contract, generator)
        evidence["restore_model_exact"] = _same(mutated, _weights(model))
        evidence["restore_optimizer_exact"] = _same(torch.load(probe_checkpoint, map_location="cpu", weights_only=False)["optimizer"], optimizer.state_dict())
        evidence["restore_scheduler_exact"] = _same(torch.load(probe_checkpoint, map_location="cpu", weights_only=False)["scheduler"], scheduler.state_dict())
        evidence["restore_rng_exact"] = _same(saved_rng, rng_state())
        eval_after = training.evaluate(model, eval_subset, runtime_device, data_module, settings)
        evidence["restore_eval_exact"] = _same(eval_before, eval_after)

        evidence["continuation_parity_verified"] = False
        if args.max_probe_updates == 2:
            # Materialize fixed CPU batches first.  Creating/evaluating loader
            # iterators may consume main Torch RNG, so both branches must load
            # the exact saved probe state *after* that materialization.
            second = _take(iterator, settings.accumulation_steps)
            baseline_progress = load_checkpoint(probe_checkpoint, model, optimizer, scheduler, contract, generator)
            train_batches(model, second, optimizer, scheduler, runtime_device, settings.accumulation_steps, progress=baseline_progress, max_updates=2, bf16=settings.bf16, channels_last=settings.channels_last, physical_batch_size=settings.physical_batch_size)
            expected = _weights(model)
            replay_progress = load_checkpoint(probe_checkpoint, model, optimizer, scheduler, contract, generator)
            train_batches(model, second, optimizer, scheduler, runtime_device, settings.accumulation_steps, progress=replay_progress, max_updates=2, bf16=settings.bf16, channels_last=settings.channels_last, physical_batch_size=settings.physical_batch_size)
            _assert_close(expected, _weights(model), args.parity_rtol, args.parity_atol)
            evidence["continuation_parity_verified"] = True
        passed = all(bool(evidence[key]) for key in ("fullinput_train_finite", "eval_subset_finite", "restore_model_exact", "restore_optimizer_exact", "restore_scheduler_exact", "restore_rng_exact", "restore_eval_exact"))
        path = write_readiness(settings.output_root, contract, evidence, passed)
        return {"status": "ready" if passed else "needs_attention", "readiness_path": str(path), "evidence": evidence}
    except Exception as exc:
        message = str(exc)
        evidence.update({"error_type": type(exc).__name__, "error": message, "cuda_oom": isinstance(exc, torch.OutOfMemoryError) or "out of memory" in message.lower()})
        path = write_readiness(settings.output_root, contract, evidence, False)
        return {"status": "needs_attention", "readiness_path": str(path), "evidence": evidence}
    finally:
        model.load_state_dict(initial[0])
        optimizer.load_state_dict(initial[1])
        scheduler.load_state_dict(initial[2])
        from dense_transfer.engine import restore_rng_state
        restore_rng_state(initial[3])


def main() -> None:
    result = run_validation(parser().parse_args())
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
