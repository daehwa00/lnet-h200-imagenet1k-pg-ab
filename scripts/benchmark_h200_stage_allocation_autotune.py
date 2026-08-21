#!/usr/bin/env python3
"""Measure H200 batch size and single-process CUDA-stream concurrency."""

from __future__ import annotations

# ruff: noqa: EM101, EM102, SLF001, T201, TRY003
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import numpy as np
import run_a2d_affine_qhead_imagenet100 as heads
import run_a2d_qhead_e2e_imagenet100 as structured
import run_a2d_r2k3_stage_allocation_cohort_imagenet100 as cohort
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as stage
import run_a2d_resaux1_imagenet100 as prepare
import run_alphabet2d_imagenet100_nano as harness
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.utils.data import DataLoader


SCHEMA = "lnet.h200.stage_allocation.autotune.v1"
SEED = 501
WARMUP_STEPS = 5
MEASURED_STEPS = 20
TOTAL_MODELS = len(stage.VARIANTS)
MAXIMUM_RESERVED_RATIO = 0.85
MINIMUM_FREE_BYTES = 24 * 2**30
MAXIMUM_DRIFT_FLOOR_BYTES = 512 * 2**20
MAXIMUM_DRIFT_RATIO = 0.005
MINIMUM_GAIN = 0.10
CANDIDATE_MATRIX = (
    (128, 1),
    (128, 2),
    (128, 4),
    (128, 6),
    (128, 7),
    (128, 8),
    (256, 1),
    (256, 2),
    (256, 4),
    (512, 1),
    (512, 2),
)


class OwnerStopRequestedError(RuntimeError):
    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__(str(record["reason"]))
        self.record = record


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measured-steps", type=int, default=MEASURED_STEPS)
    parser.add_argument("--worker-batch-size", type=int, choices=(128, 256, 512))
    return parser.parse_args()


def _partitions(total: int, lanes: int) -> list[int]:
    if total < 1 or lanes < 1:
        raise ValueError("total and lanes must be positive")
    full, remainder = divmod(total, lanes)
    return [lanes] * full + ([remainder] if remainder else [])


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "scripts/benchmark_h200_stage_allocation_autotune.py",
        "scripts/run_a2d_r2k3_stage_allocation_screen_imagenet100.py",
        "scripts/run_a2d_r2k3_stage_allocation_cohort_imagenet100.py",
        "scripts/run_a2d_affine_qhead_imagenet100.py",
        "scripts/run_alphabet2d_imagenet100_nano.py",
        "scripts/a2d_r2k3_runtime.py",
    ):
        path = Path(relative)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _owner_stop_record() -> dict[str, Any] | None:
    configured = os.environ.get("H200_CONTROL_FAST_STOP_MARKER")
    if not configured:
        return None
    path = Path(configured)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError("H200 owner-stop marker is unreadable") from error
    if (
        payload.get("schema") != "lnet.h200.owner_stop.v1"
        or payload.get("target_commit") != os.environ.get("H200_EXPECTED_COMMIT")
        or not isinstance(payload.get("generation"), int)
        or not isinstance(payload.get("reason"), str)
    ):
        raise RuntimeError("H200 owner-stop marker identity changed")
    return cast("dict[str, Any]", payload)


def _raise_if_owner_stopped() -> None:
    record = _owner_stop_record()
    if record is not None:
        raise OwnerStopRequestedError(record)


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    active_workers = harness._active_loader_workers(args.workers)
    data_stat = args.data_root.stat()
    filesystem = os.statvfs(args.data_root)
    packages = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "torch", "torchvision", "triton")
    }
    return {
        "schema": SCHEMA,
        "target_commit": os.environ["H200_EXPECTED_COMMIT"],
        "source_sha256": _source_sha256(),
        "dataset_identity_sha256": os.environ["LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256"],
        "gpu": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "driver_version": subprocess.run(
            ["/usr/bin/nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "packages": packages,
        "precision": args.precision,
        "compile_mode": os.environ.get("LNET_COMPILE_MODE", "default"),
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "input_pipeline": {
            "workers_requested": args.workers,
            "workers_effective": active_workers,
            "prefetch_factor": harness._active_loader_prefetch_factor(),
            "persistent_workers": harness._persistent_loader_workers(active_workers),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_affinity_evidence": os.environ.get("LNET_CPU_AFFINITY_ACTIVE"),
        },
        "data_path_context": {
            "resolved_root": str(args.data_root.resolve()),
            "device": data_stat.st_dev,
            "filesystem_id": getattr(filesystem, "f_fsid", None),
            "block_size": filesystem.f_bsize,
        },
        "candidate_matrix": [
            {"batch_size": batch_size, "lanes": lanes} for batch_size, lanes in CANDIDATE_MATRIX
        ],
        "safety": {
            "maximum_reserved_ratio": MAXIMUM_RESERVED_RATIO,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "maximum_drift_floor_bytes": MAXIMUM_DRIFT_FLOOR_BYTES,
            "maximum_drift_ratio": MAXIMUM_DRIFT_RATIO,
            "minimum_gain": MINIMUM_GAIN,
        },
    }


def _cache_hit(path: Path, identity_sha256: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity_sha256") != identity_sha256
        or not isinstance(payload.get("candidates"), list)
        or not isinstance(payload.get("recommendation"), dict)
    ):
        return None
    return cast("dict[str, Any]", payload)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _maximum_drift(total_memory: int) -> int:
    return max(MAXIMUM_DRIFT_FLOOR_BYTES, int(MAXIMUM_DRIFT_RATIO * total_memory))


def _safety_reasons(candidate: dict[str, Any], total_memory: int) -> list[str]:
    if candidate.get("status") != "complete":
        return [str(candidate.get("status", "incomplete"))]
    reasons: list[str] = []
    if not candidate.get("finite", False):
        reasons.append("nonfinite")
    if int(candidate["peak_reserved_bytes"]) / total_memory >= MAXIMUM_RESERVED_RATIO:
        reasons.append("reserved_headroom")
    if int(candidate["free_memory_bytes"]) < MINIMUM_FREE_BYTES:
        reasons.append("free_memory")
    if int(candidate["allocated_drift_bytes"]) > _maximum_drift(total_memory):
        reasons.append("allocated_drift")
    return reasons


def _annotate_candidate(candidate: dict[str, Any], total_memory: int) -> dict[str, Any]:
    result = dict(candidate)
    reasons = _safety_reasons(result, total_memory)
    result["safety_reasons"] = reasons
    result["safe"] = not reasons
    if result.get("status") == "complete":
        lanes = int(result["lanes"])
        seconds_per_batch = float(result["elapsed_seconds"]) / int(result["measured_steps"])
        result["partition"] = _partitions(TOTAL_MODELS, lanes)
        result["estimated_full_cohort_seconds_per_epoch"] = (
            int(result["loader_batches"]) * seconds_per_batch
        )
    return result


def _select(candidates: list[dict[str, Any]], total_memory: int) -> dict[str, Any]:
    annotated = [_annotate_candidate(candidate, total_memory) for candidate in candidates]
    safe = [candidate for candidate in annotated if candidate["safe"]]
    if not safe:
        return {
            "status": "no_safe_candidate",
            "selected": None,
            "minimum_gain": MINIMUM_GAIN,
        }
    best = min(
        safe,
        key=lambda candidate: (
            candidate["estimated_full_cohort_seconds_per_epoch"],
            candidate["batch_size"],
            candidate["lanes"],
        ),
    )
    baseline = next(
        (
            candidate
            for candidate in safe
            if candidate["batch_size"] == 128 and candidate["lanes"] == 1
        ),
        None,
    )
    gain = None
    selected = best
    gated = False
    if baseline is not None:
        baseline_seconds = float(baseline["estimated_full_cohort_seconds_per_epoch"])
        gain = (baseline_seconds - float(best["estimated_full_cohort_seconds_per_epoch"])) / (
            baseline_seconds
        )
        if gain < MINIMUM_GAIN:
            selected = baseline
            gated = best is not baseline
    return {
        "status": "selected",
        "selected": {
            "batch_size": selected["batch_size"],
            "lanes": selected["lanes"],
            "partition": selected["partition"],
            "model_images_per_second": selected["model_images_per_second"],
            "estimated_full_cohort_seconds_per_epoch": selected[
                "estimated_full_cohort_seconds_per_epoch"
            ],
        },
        "best_observed": {"batch_size": best["batch_size"], "lanes": best["lanes"]},
        "gain_over_baseline": gain,
        "minimum_gain": MINIMUM_GAIN,
        "minimum_gain_gate_applied": gated,
        "recipe_note": (
            "batch size changes optimizer-step count; this is a hardware recommendation "
            "and must be frozen into the next scientific contract"
        ),
    }


def _attempt(
    batch_size: int,
    lanes: int,
    runner: Callable[[], dict[str, Any]],
    recover: Callable[[], None],
) -> dict[str, Any]:
    try:
        return runner()
    except torch.cuda.OutOfMemoryError as error:
        recover()
        return {
            "batch_size": batch_size,
            "lanes": lanes,
            "status": "oom",
            "error": type(error).__name__,
        }


def _build_member(
    variant: str,
    recipe: dict[str, Any],
    device: torch.device,
) -> cohort.MemberState:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = stage._build(variant, runtime.model_config()).to(device)
    model = prepare._prepare_model(model, recipe)
    optimizer = runtime.build_optimizer(model, recipe)
    return cohort.MemberState(
        variant=variant,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0),
        runtime=harness._build_runtime(model, recipe),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
    )


def _assert_disjoint_parameters(members: list[cohort.MemberState]) -> None:
    owners: dict[int, str] = {}
    for member in members:
        for parameter in member.model.parameters():
            previous = owners.setdefault(id(parameter), member.variant)
            if previous != member.variant:
                raise RuntimeError(
                    f"autotune models share a parameter: {previous} and {member.variant}"
                )


def _launch_shared_batch(
    members: list[cohort.MemberState],
    streams: list[torch.cuda.Stream],
    inputs: Tensor,
    targets: Tensor,
    *,
    permutation_generator: torch.Generator,
    mixup_generator: np.random.Generator,
    mixup_alpha: float,
    precision: str,
    device: torch.device,
) -> list[Tensor]:
    default_stream = torch.cuda.current_stream(device)
    permutation = torch.randperm(
        targets.numel(),
        device=device,
        generator=permutation_generator,
    )
    permuted_targets = targets[permutation]
    mixing = float(mixup_generator.beta(mixup_alpha, mixup_alpha))
    mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
    losses: list[Tensor] = []
    for wave_start in range(0, len(members), len(streams)):
        wave = members[wave_start : wave_start + len(streams)]
        for member, stream in zip(wave, streams, strict=True):
            if member.runtime is None:
                raise RuntimeError(f"missing runtime for {member.variant}")
            stream.wait_stream(default_stream)
            mixed_inputs.record_stream(stream)
            targets.record_stream(stream)
            permuted_targets.record_stream(stream)
            with torch.cuda.stream(stream):
                member.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bfloat16",
                ):
                    output = member.runtime(mixed_inputs)
                    _logits, loss, _diagnostics = heads._training_objective(
                        member.model,
                        output,
                        targets,
                        permuted_targets,
                        mixing,
                    )
                loss.backward()
                heads._after_training_batch(
                    member.model,
                    output,
                    targets,
                    permuted_targets,
                    mixing,
                )
                torch.nn.utils.clip_grad_norm_(member.model.parameters(), 1.0)
                member.optimizer.step()
                losses.append(loss.detach())
        for stream in streams[: len(wave)]:
            default_stream.wait_stream(stream)
    return losses


def _finite(members: list[cohort.MemberState], losses: list[Tensor]) -> bool:
    if not all(bool(torch.isfinite(loss).all()) for loss in losses):
        return False
    return all(
        bool(torch.isfinite(parameter).all())
        and (parameter.grad is None or bool(torch.isfinite(parameter.grad).all()))
        for member in members
        for parameter in member.model.parameters()
    )


def _benchmark_candidate(
    members: list[cohort.MemberState],
    loader: DataLoader[Any],
    *,
    batch_size: int,
    lanes: int,
    warmup_steps: int,
    measured_steps: int,
    recipe: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    active = members
    streams = (
        [torch.cuda.current_stream(device)]
        if lanes == 1
        else [torch.cuda.Stream(device=device) for _ in range(lanes)]
    )
    batches = iter(
        harness._device_batches(
            loader,
            device,
            channels_last=bool(recipe.get("channels_last", False)),
        )
    )
    permutation_generator = torch.Generator(device=device).manual_seed(
        SEED + batch_size * 100 + lanes
    )
    mixup_generator = np.random.default_rng(SEED + batch_size * 100 + lanes)
    host_wait = 0.0
    last_losses: list[Tensor] = []

    def step() -> None:
        nonlocal host_wait, last_losses
        _raise_if_owner_stopped()
        waiting_started = time.perf_counter()
        inputs, targets = next(batches)
        host_wait += time.perf_counter() - waiting_started
        last_losses = _launch_shared_batch(
            active,
            streams,
            inputs,
            targets,
            permutation_generator=permutation_generator,
            mixup_generator=mixup_generator,
            mixup_alpha=float(recipe["mixup_alpha"]),
            precision=str(recipe["precision"]),
            device=device,
        )

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize(device)
    start_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    host_wait = 0.0
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record(torch.cuda.current_stream(device))
    for _ in range(measured_steps):
        step()
    finished.record(torch.cuda.current_stream(device))
    finished.synchronize()
    elapsed_seconds = started.elapsed_time(finished) / 1000.0
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    end_allocated = torch.cuda.memory_allocated(device)
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    finite = _finite(active, last_losses)
    result = {
        "batch_size": batch_size,
        "lanes": lanes,
        "status": "complete",
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "loader_batches": len(loader),
        "elapsed_seconds": elapsed_seconds,
        "host_input_wait_seconds": host_wait,
        "model_images_per_second": batch_size * len(active) * measured_steps / elapsed_seconds,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "free_memory_bytes": free_memory,
        "total_memory_bytes": total_memory,
        "allocated_drift_bytes": max(0, end_allocated - start_allocated),
        "finite": finite,
        "partition": _partitions(len(active), lanes),
        "variants": [member.variant for member in active],
    }
    print("H200_AUTOTUNE_CANDIDATE=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def _recover_oom(device: torch.device, members: list[cohort.MemberState]) -> None:
    for member in members:
        member.optimizer.zero_grad(set_to_none=True)
    with contextlib.suppress(RuntimeError):
        torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


def _base_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        root=args.root,
        data_root=args.data_root,
        variants=list(stage.VARIANTS),
        run_seeds=[SEED],
        epochs=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        workers=args.workers,
        precision=args.precision,
        minimum_cohort_model_images_per_second=800.0,
        initialize_only=False,
    )


def _configure_runtime() -> None:
    runtime.configure(stage.VARIANTS, (SEED,))
    heads.VARIANTS = stage.VARIANTS
    heads.SEEDS = (SEED,)
    structured._training_objective = heads._training_objective
    structured._after_training_batch = heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    dynamo_config = cast("Any", torch)._dynamo.config
    dynamo_config.recompile_limit = 64
    dynamo_config.cache_size_limit = 64


def _batch_result_path(root: Path, identity_sha256: str, batch_size: int) -> Path:
    return root / identity_sha256 / f"batch-{batch_size}.json"


def _batch_cache_hit(
    path: Path,
    *,
    identity_sha256: str,
    batch_size: int,
) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    candidates = payload.get("candidates")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "batch_complete"
        or payload.get("identity_sha256") != identity_sha256
        or payload.get("batch_size") != batch_size
        or not isinstance(candidates, list)
        or len(candidates)
        != sum(candidate_batch == batch_size for candidate_batch, _ in CANDIDATE_MATRIX)
    ):
        return None
    return cast("list[dict[str, Any]]", candidates)


def _run_batch(
    args: argparse.Namespace,
    *,
    batch_size: int,
    identity_sha256: str,
) -> Path:
    result_path = _batch_result_path(args.root, identity_sha256, batch_size)
    cached = _batch_cache_hit(
        result_path,
        identity_sha256=identity_sha256,
        batch_size=batch_size,
    )
    if cached is not None:
        print(f"H200_AUTOTUNE_BATCH_CACHE_HIT={result_path}", flush=True)
        return result_path

    _configure_runtime()
    base_args = _base_arguments(args)
    contract = cohort._contract(base_args)
    recipe = cast("dict[str, Any]", contract["recipe"])
    evidence_root = args.root / identity_sha256
    harness._configure_compile_runtime(evidence_root, recipe)
    parameter_counts = cast("dict[str, int]", contract["parameter_counts"])
    representatives = sorted(
        stage.VARIANTS,
        key=lambda variant: (-parameter_counts[variant], variant),
    )
    device = torch.device("cuda")
    lane_candidates = [lanes for batch, lanes in CANDIDATE_MATRIX if batch == batch_size]
    batch_recipe = dict(recipe)
    batch_recipe["batch_size"] = batch_size
    batch_recipe["effective_batch_size"] = batch_size
    training_generator = torch.Generator().manual_seed(SEED + batch_size)
    loader, validation_loader = harness._loaders(
        args.data_root,
        batch_size=batch_size,
        workers=args.workers,
        training_generator=training_generator,
    )
    del validation_loader
    members: list[cohort.MemberState] = []
    candidates: list[dict[str, Any]] = []
    try:
        members.extend(_build_member(variant, batch_recipe, device) for variant in representatives)
        _assert_disjoint_parameters(members)
    except torch.cuda.OutOfMemoryError as error:
        _recover_oom(device, members)
        candidates = [
            {
                "batch_size": batch_size,
                "lanes": lanes,
                "status": "oom",
                "error": type(error).__name__,
                "phase": "model_build",
            }
            for lanes in lane_candidates
        ]
    else:
        batch_oom = False
        for lanes in lane_candidates:
            if batch_oom:
                candidates.append(
                    {"batch_size": batch_size, "lanes": lanes, "status": "pruned_after_oom"}
                )
                continue
            result = _attempt(
                batch_size,
                lanes,
                lambda lanes=lanes: _benchmark_candidate(
                    members,
                    loader,
                    batch_size=batch_size,
                    lanes=lanes,
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                    recipe=batch_recipe,
                    device=device,
                ),
                lambda: _recover_oom(device, members),
            )
            candidates.append(result)
            if result["status"] == "oom":
                batch_oom = True
    payload = {
        "schema": SCHEMA,
        "status": "batch_complete",
        "identity_sha256": identity_sha256,
        "batch_size": batch_size,
        "candidates": candidates,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(result_path, payload)
    print(f"H200_AUTOTUNE_BATCH_COMPLETE={result_path}", flush=True)
    return result_path


def _worker_command(args: argparse.Namespace, batch_size: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(args.root),
        "--data-root",
        str(args.data_root),
        "--workers",
        str(args.workers),
        "--precision",
        str(args.precision),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measured-steps",
        str(args.measured_steps),
        "--worker-batch-size",
        str(batch_size),
    ]


def _run_worker_process(args: argparse.Namespace, batch_size: int) -> None:
    process = subprocess.Popen(_worker_command(args, batch_size))  # noqa: S603
    while process.poll() is None:
        record = _owner_stop_record()
        if record is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise OwnerStopRequestedError(record)
        time.sleep(0.5)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    _raise_if_owner_stopped()


def _run(args: argparse.Namespace) -> Path:
    _raise_if_owner_stopped()
    if args.warmup_steps < 1 or args.measured_steps < 1:
        raise ValueError("autotune timing counts must be positive")
    if os.environ.get("LNET_COMPILE_MODE", "default") != "default":
        raise ValueError("autotune requires the production default compile mode")
    identity = _identity(args)
    identity_sha256 = _canonical_sha256(identity)
    evidence_root = args.root / identity_sha256
    result_path = evidence_root / "result.json"
    cached = _cache_hit(result_path, identity_sha256)
    if cached is not None:
        print(f"H200_AUTOTUNE_CACHE_HIT={result_path}", flush=True)
        return result_path
    if args.worker_batch_size is not None:
        return _run_batch(
            args,
            batch_size=args.worker_batch_size,
            identity_sha256=identity_sha256,
        )

    candidates: list[dict[str, Any]] = []
    for batch_size in (128, 256, 512):
        batch_path = _batch_result_path(args.root, identity_sha256, batch_size)
        batch_candidates = _batch_cache_hit(
            batch_path,
            identity_sha256=identity_sha256,
            batch_size=batch_size,
        )
        if batch_candidates is None:
            _run_worker_process(args, batch_size)
            batch_candidates = _batch_cache_hit(
                batch_path,
                identity_sha256=identity_sha256,
                batch_size=batch_size,
            )
        if batch_candidates is None:
            raise RuntimeError(
                f"autotune batch worker did not publish valid evidence: {batch_size}"
            )
        candidates.extend(batch_candidates)

    total_memory = int(identity["gpu_total_memory_bytes"])

    annotated = [_annotate_candidate(candidate, total_memory) for candidate in candidates]
    recommendation = _select(candidates, total_memory)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "identity_sha256": identity_sha256,
        "identity": identity,
        "candidates": annotated,
        "recommendation": recommendation,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(result_path, payload)
    print("H200_AUTOTUNE_RECOMMENDATION=" + json.dumps(recommendation, sort_keys=True), flush=True)
    return result_path


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("autotune requires exactly one visible CUDA GPU")
    harness._configure_cpu_affinity()
    args = _arguments()
    try:
        result_path = _run(args)
    except OwnerStopRequestedError as error:
        print(
            "H200_AUTOTUNE_STOPPED="
            + json.dumps(
                {
                    "generation": error.record["generation"],
                    "reason": error.record["reason"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if args.worker_batch_size is None:
        print(f"H200_AUTOTUNE_COMPLETE={result_path}", flush=True)


if __name__ == "__main__":
    main()
