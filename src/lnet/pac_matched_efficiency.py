# ruff: noqa: BLE001, E501, EM101, FBT001, FBT003, PLC0415, T201, TRY003
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_confirmatory_baselines import ConfirmatoryFamily, build_confirmatory_family
from .pac_headroom_efficient_models import build_efficient_headroom_classifier
from .pac_metrics import count_parameters
from .pac_types import PACExperimentConfig

MODELS = (
    "alphabet",
    "fixed_origin",
    "cnn1d",
    "s4d",
    "mamba",
    "tcn",
    "gru",
    "lstm",
    "transformer",
    "inception_time",
)
LENGTHS = (128, 512, 2048)
BATCHES = (1, 64)
SELECTION_PATH = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
_MATCHED_WIDTHS: dict[tuple[str, int, int], tuple[int, int, float]] = {}


@dataclass(frozen=True, slots=True)
class EfficiencyJob:
    model: str
    length: int
    batch_size: int

    @property
    def key(self) -> str:
        return f"efficiency__{self.model}__n{self.length}__b{self.batch_size}"


def efficiency_jobs() -> list[EfficiencyJob]:
    return [EfficiencyJob(model, length, batch) for model in MODELS for length in LENGTHS for batch in BATCHES]


def _selection(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("baseline selection is not complete")
    return payload


def build_model(job: EfficiencyJob, selection_path: Path = SELECTION_PATH) -> tuple[nn.Module, dict[str, object]]:
    config = PACExperimentConfig(
        sample_count=64,
        validation_count=16,
        test_count=16,
        sequence_length=job.length,
        raw_input_dim=1,
        output_dim=5,
        model_dim=64,
        modes=16,
    )
    if job.model in {"alphabet", "fixed_origin"}:
        spec = "PA2WP" if job.model == "alphabet" else "WP"
        model = build_efficient_headroom_classifier(spec, config, 5, objective="classification")
        return model, {
            "family": job.model,
            "internal_spec": spec,
            "model_dim": 64,
            "modes": 16,
            "parameter_matching": "current ALPHABET capacity",
        }
    selected = _selection(selection_path)
    trials = cast("dict[str, dict[str, object]]", selected["selected_trials"])
    trial = int(trials[job.model]["trial"])
    reference_model = build_efficient_headroom_classifier(
        "PA2WP", config, 5, objective="classification"
    )
    target_params = count_parameters(reference_model)
    cache_key = (job.model, trial, target_params)
    if cache_key not in _MATCHED_WIDTHS:
        candidates: list[tuple[float, int, int]] = []
        limit = 2048 if job.model == "inception_time" else 256
        for width in range(1, limit + 1):
            candidate = build_confirmatory_family(
                cast("ConfirmatoryFamily", job.model),
                width,
                config,
                5,
                validation_trial=trial,
            )
            parameters = count_parameters(candidate)
            error = abs(parameters - target_params) / target_params
            candidates.append((error, width, parameters))
            if parameters >= target_params:
                break
        error, width, parameters = min(candidates)
        if error > 0.055:
            message = f"{job.model} parameter error {error:.4f} exceeds 0.055"
            raise ValueError(message)
        _MATCHED_WIDTHS[cache_key] = width, parameters, error
    width, parameters, error = _MATCHED_WIDTHS[cache_key]
    model = build_confirmatory_family(
        cast("ConfirmatoryFamily", job.model),
        width,
        config,
        5,
        validation_trial=trial,
    )
    return model, {
        "family": job.model,
        "validation_trial": trial,
        "reference_model": "ALPHABET/PA2WP D64 M16",
        "matched_width": width,
        "target_params": target_params,
        "matched_params": parameters,
        "relative_param_error": error,
        "parameter_matching": "current ALPHABET trainable-parameter target",
    }


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _run_forward(model: nn.Module, inputs: Tensor) -> None:
    with torch.inference_mode():
        model(inputs)


def _median_iqr(samples: list[float]) -> tuple[float, float]:
    values = torch.tensor(samples, dtype=torch.float64)
    median = float(torch.quantile(values, 0.5).item())
    iqr = float((torch.quantile(values, 0.75) - torch.quantile(values, 0.25)).item())
    return median, iqr


def _repetitions(job: EfficiencyJob, training: bool) -> tuple[int, int]:
    tokens = job.length * job.batch_size
    if training:
        return (2, 7) if tokens <= 32_768 else (1, 3)
    return (8, 30) if tokens <= 32_768 else (3, 10)


def measure_inference(model: nn.Module, inputs: Tensor, job: EfficiencyJob, device: str) -> dict[str, object]:
    model.eval()
    warmups, repeats = _repetitions(job, False)
    for _ in range(warmups):
        _run_forward(model, inputs)
    _sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeats):
        _sync(device)
        start = perf_counter()
        _run_forward(model, inputs)
        _sync(device)
        samples.append((perf_counter() - start) * 1000.0)
    latency, iqr = _median_iqr(samples)
    examples_per_second = job.batch_size * 1000.0 / latency
    return {
        "latency_ms": latency,
        "latency_iqr_ms": iqr,
        "latency_samples_ms": samples,
        "examples_per_second": examples_per_second,
        "tokens_per_second": examples_per_second * job.length,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
    }


def _post_step(model: nn.Module) -> None:
    callback = getattr(model, "post_optimizer_step", None)
    if callable(callback):
        callback()


def measure_training(model: nn.Module, inputs: Tensor, labels: Tensor, job: EfficiencyJob, device: str) -> dict[str, object]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    warmups, repeats = _repetitions(job, True)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        functional.cross_entropy(model(inputs), labels).backward()
        optimizer.step()
        _post_step(model)

    for _ in range(warmups):
        step()
    _sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeats):
        _sync(device)
        start = perf_counter()
        step()
        _sync(device)
        samples.append((perf_counter() - start) * 1000.0)
    latency, iqr = _median_iqr(samples)
    examples_per_second = job.batch_size * 1000.0 / latency
    return {
        "step_latency_ms": latency,
        "step_latency_iqr_ms": iqr,
        "step_latency_samples_ms": samples,
        "examples_per_second": examples_per_second,
        "tokens_per_second": examples_per_second * job.length,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
    }


def profile_forward_flops(
    model: nn.Module, inputs: Tensor, *, batch_multiplier: int = 1
) -> dict[str, object]:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            model(inputs)
        measured_flops = int(counter.get_total_flops())
    except Exception as error:  # profiler support must not invalidate runtime measurements
        primary_error = f"{type(error).__name__}: {error}"
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if inputs.is_cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            model.eval()
            with torch.inference_mode(), torch.profiler.profile(
                activities=activities, with_flops=True
            ) as profiler:
                model(inputs)
            measured_flops = int(sum(event.flops for event in profiler.key_averages()))
            method = "torch.profiler(with_flops=True) fallback"
        except Exception as fallback_error:
            return {
                "forward_flops": None,
                "forward_mac_equivalent": None,
                "method": "unavailable",
                "error": (
                    f"flop_counter={primary_error}; torch.profiler="
                    f"{type(fallback_error).__name__}: {fallback_error}"
                ),
            }
    else:
        method = "torch.utils.flop_counter"
    flops = measured_flops * batch_multiplier
    return {
        "forward_flops": flops,
        "forward_mac_equivalent": flops / 2.0,
        "profiled_batch_size": 1,
        "batch_multiplier": batch_multiplier,
        "method": f"{method}; batch-linear scaling; MAC-equivalent=FLOP/2",
        "coverage": "operator-dispatch count; unsupported custom operators may be omitted",
    }


def run_job(job: EfficiencyJob, *, device: str = "cuda", selection_path: Path = SELECTION_PATH, smoke: bool = False) -> dict[str, object]:
    torch.manual_seed(7)
    model, architecture = build_model(job, selection_path)
    model = model.to(device)
    active_job = EfficiencyJob(job.model, min(job.length, 32), min(job.batch_size, 2)) if smoke else job
    inputs = torch.randn(active_job.batch_size, active_job.length, 1, device=device)
    labels = torch.randint(0, 5, (active_job.batch_size,), device=device)
    try:
        flops = profile_forward_flops(
            model, inputs[:1], batch_multiplier=active_job.batch_size
        )
        inference = measure_inference(model, inputs, active_job, device)
        training = measure_training(model, inputs, labels, active_job, device)
        outcome = "measured"
        reason = ""
    except (torch.cuda.OutOfMemoryError, MemoryError) as error:
        flops = {}
        inference = {}
        training = {}
        outcome = "resource_limit"
        reason = f"{type(error).__name__}: {error}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device()) if device == "cuda" else None
    return {
        "schema_version": "pac_matched_efficiency.v1",
        "job_key": job.key,
        "model": job.model,
        "sequence_length": job.length,
        "batch_size": job.batch_size,
        "runtime": "eager_fp32",
        "outcome_status": outcome,
        "resource_limit_reason": reason,
        "params_trainable": count_parameters(model),
        "architecture": architecture,
        "forward_profile": flops,
        "inference": inference,
        "training": training,
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": properties.name if properties is not None else "cpu",
            "device_total_memory_bytes": properties.total_memory if properties is not None else None,
            "cuda": torch.version.cuda,
        },
        "status": "done",
    }


def enqueue(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(exist_ok=True)
    jobs = efficiency_jobs()
    _atomic_json(root / "contract.json", {
        "schema_version": "pac_matched_efficiency_contract.v1",
        "models": list(MODELS),
        "lengths": list(LENGTHS),
        "batch_sizes": list(BATCHES),
        "runtime": "eager_fp32",
        "jobs": len(jobs),
        "exclusive_gpu_required": True,
        "locked_before_execution": True,
    })
    (root / "manifest.jsonl").write_text("".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs), encoding="utf-8")
    return {"jobs": len(jobs), "workers": 1}


def worker(root: Path, *, device: str = "cuda", selection_path: Path = SELECTION_PATH, smoke: bool = False) -> int:
    completed = 0
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        job = EfficiencyJob(str(payload["model"]), int(payload["length"]), int(payload["batch_size"]))
        destination = root / "jobs" / f"{job.key}.json"
        if destination.is_file():
            continue
        _atomic_json(destination, run_job(job, device=device, selection_path=selection_path, smoke=smoke))
        completed += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return completed


def report(root: Path) -> dict[str, object]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "jobs").glob("*.json"))]
    rows: list[dict[str, object]] = []
    for item in payloads:
        profile = item.get("forward_profile", {})
        inference = item.get("inference", {})
        training = item.get("training", {})
        rows.append({
            "model": item["model"], "sequence_length": item["sequence_length"], "batch_size": item["batch_size"],
            "params_trainable": item["params_trainable"], "outcome_status": item["outcome_status"],
            "forward_flops": profile.get("forward_flops"), "forward_mac_equivalent": profile.get("forward_mac_equivalent"),
            "inference_latency_ms": inference.get("latency_ms"), "inference_examples_per_second": inference.get("examples_per_second"),
            "inference_peak_memory_mb": inference.get("peak_memory_mb"), "training_step_latency_ms": training.get("step_latency_ms"),
            "training_examples_per_second": training.get("examples_per_second"), "training_peak_memory_mb": training.get("peak_memory_mb"),
        })
    report_root = root / "reports"
    report_root.mkdir(exist_ok=True)
    if rows:
        with (report_root / "matched_efficiency.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"schema_version": "pac_matched_efficiency_report.v1", "completed_jobs": len(payloads), "expected_jobs": len(efficiency_jobs()), "complete": len(payloads) == len(efficiency_jobs()), "resource_limited": sum(row["outcome_status"] != "measured" for row in rows)}
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-path", type=Path, default=SELECTION_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "enqueue":
        print(json.dumps(enqueue(args.root), sort_keys=True))
    elif args.stage == "worker":
        print(json.dumps({"completed": worker(args.root, device=args.device, selection_path=args.selection_path, smoke=args.smoke)}))
    else:
        print(json.dumps(report(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
