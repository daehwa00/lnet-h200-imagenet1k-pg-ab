# ruff: noqa: BLE001, E501, EM101, EM102, TRY003
from __future__ import annotations

import csv
import json
import platform
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, cast

import torch
from torch import nn

from .pac_confirmatory_baselines import ConfirmatoryFamily, build_confirmatory_family
from .pac_headroom_efficient_models import (
    EfficientHeadroomSpec,
    build_efficient_headroom_classifier,
)
from .pac_matched_efficiency import (
    EfficiencyJob as LegacyEfficiencyJob,
)
from .pac_matched_efficiency import (
    measure_inference,
    measure_training,
    profile_forward_flops,
)
from .pac_metrics import count_parameters
from .pac_types import PACExperimentConfig

DEFAULT_ROOT: Final = Path(".omx/results/pac-efp16-matched-efficiency-20260713")
DEFAULT_SELECTION_PATH: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
PUBLIC_BASELINES: Final = (
    "tcn",
    "cnn1d",
    "gru",
    "lstm",
    "transformer",
    "mamba",
    "s4d",
    "inception_time",
)
MODELS: Final = ("efp16", "pa2wp", "fixed_origin", *PUBLIC_BASELINES)
LENGTHS: Final = (128, 512, 2048)
BATCHES: Final = (1, 64)
PARAMETER_TOLERANCE: Final = 0.05
PUBLIC_VALIDATION_TRIALS: Final = tuple(range(1, 7))
_PUBLIC_MATCH_CACHE: dict[tuple[str, int], tuple[float, int, int, int]] = {}


@dataclass(frozen=True, slots=True)
class EFP16EfficiencyJob:
    model: str
    length: int
    batch_size: int

    @property
    def key(self) -> str:
        return f"efp16_efficiency__{self.model}__n{self.length}__b{self.batch_size}"


def efficiency_jobs() -> list[EFP16EfficiencyJob]:
    return [
        EFP16EfficiencyJob(model, length, batch_size)
        for model in MODELS
        for length in LENGTHS
        for batch_size in BATCHES
    ]


def _base_config(length: int) -> PACExperimentConfig:
    return PACExperimentConfig(
        64,
        16,
        16,
        length,
        raw_input_dim=1,
        output_dim=5,
        model_dim=64,
        modes=16,
        epochs=1,
        batch_size=1,
        device="cpu",
    )


def _selection(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("baseline selection is not complete")
    return cast("dict[str, object]", payload)


def _match_predecessor(
    spec: str,
    config: PACExperimentConfig,
    target_params: int,
) -> tuple[nn.Module, dict[str, object]]:
    candidates: list[tuple[float, int, int, nn.Module]] = []
    for model_dim in range(16, 129):
        candidate = build_efficient_headroom_classifier(
            cast("EfficientHeadroomSpec", spec),
            replace(config, model_dim=model_dim, modes=16),
            5,
            objective="classification",
        )
        parameters = count_parameters(candidate)
        error = abs(parameters - target_params) / target_params
        candidates.append((error, model_dim, parameters, candidate))
        if parameters >= target_params:
            break
    error, model_dim, parameters, model = min(candidates, key=lambda item: (item[0], item[1]))
    if error > PARAMETER_TOLERANCE:
        raise ValueError(f"{spec} parameter error {error:.4f} exceeds 0.05")
    return model, {
        "family": "pa2wp" if spec == "PA2WP" else "fixed_origin",
        "internal_spec": spec,
        "matched_model_dim": model_dim,
        "matched_modes": 16,
        "target_params": target_params,
        "matched_params": parameters,
        "relative_param_error": error,
        "parameter_target": "EFP16 D32 M16 trainable parameters",
    }


def _match_public_baseline(
    family: str,
    config: PACExperimentConfig,
    target_params: int,
    selection_path: Path,
) -> tuple[nn.Module, dict[str, object]]:
    selected = cast(
        "dict[str, dict[str, object]]", _selection(selection_path)["selected_trials"]
    )
    selected_trial = int(cast("int", selected[family]["trial"]))
    cache_key = (family, target_params)
    matched = _PUBLIC_MATCH_CACHE.get(cache_key)
    if matched is None:
        candidates: list[tuple[float, int, int, int]] = []
        limit = 2048 if family == "inception_time" else 256
        for trial in PUBLIC_VALIDATION_TRIALS:
            for width in range(1, limit + 1):
                candidate = build_confirmatory_family(
                    cast("ConfirmatoryFamily", family),
                    width,
                    config,
                    5,
                    validation_trial=trial,
                )
                parameters = count_parameters(candidate)
                error = abs(parameters - target_params) / target_params
                candidates.append((error, trial, width, parameters))
                if parameters >= target_params:
                    break
        matched = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        _PUBLIC_MATCH_CACHE[cache_key] = matched
    error, trial, width, parameters = matched
    if error > PARAMETER_TOLERANCE:
        raise ValueError(f"{family} parameter error {error:.4f} exceeds 0.05")
    model = build_confirmatory_family(
        cast("ConfirmatoryFamily", family),
        width,
        config,
        5,
        validation_trial=trial,
    )
    return model, {
        "family": family,
        "selected_validation_trial": selected_trial,
        "capacity_match_trial": trial,
        "selection_trial_changed": trial != selected_trial,
        "matched_width": width,
        "target_params": target_params,
        "matched_params": parameters,
        "relative_param_error": error,
        "parameter_target": "EFP16 D32 M16 trainable parameters",
        "evidence_scope": (
            "capacity-only efficiency proxy; the capacity-match trial is never used "
            "as predictive-accuracy evidence"
        ),
    }


def build_model(
    job: EFP16EfficiencyJob,
    selection_path: Path = DEFAULT_SELECTION_PATH,
) -> tuple[nn.Module, dict[str, object]]:
    config = _base_config(job.length)
    reference = build_efficient_headroom_classifier(
        "EFP16", config, 5, objective="classification"
    )
    target_params = count_parameters(reference)
    if job.model == "efp16":
        return reference, {
            "family": "efp16",
            "internal_spec": "EFP16",
            "model_dim": 32,
            "modes": 16,
            "target_params": target_params,
            "matched_params": target_params,
            "relative_param_error": 0.0,
            "parameter_target": "self",
        }
    del reference
    if job.model == "pa2wp":
        return _match_predecessor("PA2WP", config, target_params)
    if job.model == "fixed_origin":
        return _match_predecessor("WP", config, target_params)
    if job.model not in PUBLIC_BASELINES:
        raise ValueError(f"unknown EFP16 efficiency model: {job.model}")
    return _match_public_baseline(job.model, config, target_params, selection_path)


def run_job(
    job: EFP16EfficiencyJob,
    *,
    device: str,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    smoke: bool = False,
) -> dict[str, object]:
    torch.manual_seed(7)
    model, architecture = build_model(job, selection_path)
    model = model.to(device)
    active = (
        EFP16EfficiencyJob(job.model, min(job.length, 32), min(job.batch_size, 2))
        if smoke
        else job
    )
    inputs = torch.randn(active.batch_size, active.length, 1, device=device)
    labels = torch.randint(0, 5, (active.batch_size,), device=device)
    try:
        flops = profile_forward_flops(model, inputs[:1], batch_multiplier=active.batch_size)
        measurement_job = LegacyEfficiencyJob(
            active.model, active.length, active.batch_size
        )
        inference = measure_inference(model, inputs, measurement_job, device)
        training = measure_training(model, inputs, labels, measurement_job, device)
        outcome, reason = "measured", ""
    except (torch.cuda.OutOfMemoryError, MemoryError) as error:
        flops, inference, training = {}, {}, {}
        outcome, reason = "resource_limit", f"{type(error).__name__}: {error}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    properties = (
        torch.cuda.get_device_properties(torch.cuda.current_device())
        if device == "cuda"
        else None
    )
    return {
        "schema": "pac_efp16_matched_efficiency_result.v1",
        "job_key": job.key,
        **asdict(job),
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


def enqueue(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    jobs = efficiency_jobs()
    _atomic_json(
        root / "contract.json",
        {
            "schema": "pac_efp16_matched_efficiency_contract.v1",
            "reference_model": "EFP16 D32 M16",
            "models": list(MODELS),
            "predecessor_controls": ["pa2wp", "fixed_origin"],
            "public_baselines": list(PUBLIC_BASELINES),
            "parameter_target": "per-shape EFP16 trainable parameter count",
            "parameter_tolerance": PARAMETER_TOLERANCE,
            "public_baseline_capacity_policy": (
                "choose width and one of the six predeclared TRAIN-derived trial "
                "architectures solely by absolute parameter-count error; record any "
                "difference from the accuracy-selected trial; never interpret this "
                "choice as accuracy selection"
            ),
            "lengths": list(LENGTHS),
            "batch_sizes": list(BATCHES),
            "runtime": "eager_fp32",
            "exclusive_gpu_required": True,
            "workers": 1,
            "jobs": len(jobs),
            "restart_safe": True,
        },
    )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    return {"jobs": len(jobs), "workers": 1}


def worker(
    root: Path = DEFAULT_ROOT,
    *,
    device: str,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    smoke: bool = False,
) -> None:
    jobs = [
        EFP16EfficiencyJob(**json.loads(line))
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    for job in jobs:
        completed = _result_path(root, job, failed=False)
        if completed.exists():
            continue
        try:
            row = run_job(job, device=device, selection_path=selection_path, smoke=smoke)
        except Exception as error:  # durable scientific queue; preserve the failing cell
            _atomic_json(
                _result_path(root, job, failed=True),
                {
                    "schema": "pac_efp16_matched_efficiency_failure.v1",
                    "job_key": job.key,
                    **asdict(job),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
        else:
            _atomic_json(completed, row)
            _result_path(root, job, failed=True).unlink(missing_ok=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    expected = {job.key for job in efficiency_jobs()}
    completed = _result_keys(root / "completed")
    failed = _result_keys(root / "failed") - completed
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed - failed),
        "done": expected <= completed and not (expected & failed),
    }


def report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "completed").glob("*.json"))]
    report_root = root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    table = [
        {
            "model": row["model"],
            "sequence_length": row["length"],
            "batch_size": row["batch_size"],
            "params_trainable": row["params_trainable"],
            "relative_param_error": row["architecture"]["relative_param_error"],
            "outcome_status": row["outcome_status"],
            "forward_flops": row["forward_profile"].get("forward_flops"),
            "inference_latency_ms": row["inference"].get("latency_ms"),
            "training_step_latency_ms": row["training"].get("step_latency_ms"),
        }
        for row in rows
    ]
    if table:
        with (report_root / "efp16_matched_efficiency.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    payload: dict[str, object] = {
        "schema": "pac_efp16_matched_efficiency_report.v1",
        "status": status(root),
        "rows": len(rows),
        "resource_limited": sum(row["outcome_status"] != "measured" for row in rows),
    }
    _atomic_json(report_root / "summary.json", payload)
    return payload


def _result_path(root: Path, job: EFP16EfficiencyJob, *, failed: bool) -> Path:
    return root / ("failed" if failed else "completed") / f"{job.key}.json"


def _result_keys(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        str(json.loads(path.read_text(encoding="utf-8"))["job_key"])
        for path in directory.glob("*.json")
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
