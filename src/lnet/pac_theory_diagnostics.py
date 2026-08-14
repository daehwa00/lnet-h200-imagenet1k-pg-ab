from __future__ import annotations

import gc
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Final, cast

import torch
from torch import Tensor, nn

from .pac_eval_sections import clean_validation_classification_task
from .pac_headroom_efficient_models import (
    PhaseAugmentedWaveletPacketPAC,
    _weighted_haar,  # pyright: ignore[reportPrivateUsage]
    build_efficient_headroom_classifier,
)
from .pac_real_data import ensure_ucr_train_only
from .pac_retrained_ablation_campaign import DATASETS, SEEDS
from .pac_types import PACDevice, PACExperimentConfig

DEFAULT_ROOT: Final = Path(".omx/results/pac-alphabet-theory-diagnostics-20260713")
DEFAULT_CHECKPOINT_ROOT: Final = Path(
    ".omx/results/pac-retrained-core-ablation-pro6000-20260713/checkpoints"
)
DELTA_GRID: Final = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True, slots=True)
class TheoryDiagnosticJob:
    key: str
    dataset: str
    seed: int
    checkpoint: str


def theory_diagnostic_jobs(
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> list[TheoryDiagnosticJob]:
    return [
        TheoryDiagnosticJob(
            key=f"alphabet_theory:{dataset}:seed{seed}",
            dataset=dataset,
            seed=seed,
            checkpoint=str((checkpoint_root / dataset / f"alphabet_dual_seed{seed}.pt").resolve()),
        )
        for dataset in DATASETS
        for seed in SEEDS
    ]


def prepare_theory_diagnostics(
    root: Path = DEFAULT_ROOT,
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, object]:
    jobs = theory_diagnostic_jobs(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "queue_manifest.jsonl").write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    available = sum(Path(job.checkpoint).exists() for job in jobs)
    contract = {
        "schema": "pac_alphabet_theory_diagnostics.v1",
        "purpose": "connect ALPHABET's mathematical constraints to trained behavior",
        "training_performed": False,
        "evaluation_split": "TRAIN-derived validation inputs only",
        "official_test_accessed": False,
        "checkpoint_policy": "best TRAIN-derived validation checkpoint",
        "checkpoint_root": str(checkpoint_root.resolve()),
        "jobs": len(jobs),
        "checkpoints_available_at_prepare": available,
        "diagnostics": [
            "Stiefel Gram residual and singular values",
            "learned modal damping and contraction factors",
            "Haar pair-energy preservation residual",
            "task-level low/detail fusion weights",
        ],
        "delta_grid": list(DELTA_GRID),
    }
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return theory_diagnostic_status(root, checkpoint_root=checkpoint_root)


def run_theory_diagnostics(
    root: Path = DEFAULT_ROOT,
    *,
    device: str = "cuda",
) -> None:
    jobs = [
        TheoryDiagnosticJob(**json.loads(line))
        for line in (root / "queue_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    for job in jobs:
        path = _result_path(root, job, failed=False)
        if path.exists():
            continue
        try:
            row = diagnose_checkpoint(job, device=device)
        except Exception as error:  # noqa: BLE001 - restart-safe durable diagnostics
            row = {
                "job_key": job.key,
                **asdict(job),
                "status": "failed",
                "training_performed": False,
                "official_test_accessed": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _write_result(root, job, row, failed=True)
        else:
            _write_result(root, job, row, failed=False)
            failed = _result_path(root, job, failed=True)
            if failed.exists():
                failed.unlink()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_theory_diagnostic_report(root)


def diagnose_checkpoint(job: TheoryDiagnosticJob, *, device: str) -> dict[str, object]:
    checkpoint = torch.load(job.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "pac_alphabet_validation_checkpoint.v1":
        message = f"unsupported checkpoint schema for {job.key}"
        raise ValueError(message)
    config = _config_from_checkpoint(checkpoint, device)
    class_count = int(checkpoint["config"]["class_count"])
    model = build_efficient_headroom_classifier(
        "PA2WP", config, class_count, objective="classification"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device).eval()
    if not isinstance(model, PhaseAugmentedWaveletPacketPAC):
        message = "checkpoint did not reconstruct PA2WP"
        raise TypeError(message)

    dataset = ensure_ucr_train_only(job.dataset, Path(".omx/data/ucr"), allow_download=True)
    task = clean_validation_classification_task(dataset, job.seed)
    inputs = task.validation_inputs[:256].to(device=device)
    diagnostics = diagnose_alphabet_model(model, inputs)
    return {
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "training_performed": False,
        "evaluation_split": "TRAIN-derived validation inputs only",
        "official_test_accessed": False,
        "validation_examples": int(inputs.shape[0]),
        "best_epoch": checkpoint.get("best_epoch"),
        **diagnostics,
    }


@torch.no_grad()
def diagnose_alphabet_model(
    model: PhaseAugmentedWaveletPacketPAC, inputs: Tensor
) -> dict[str, object]:
    low, detail, _ = _weighted_haar(inputs, None)
    return {
        "stiefel": {
            "forward": _frame_diagnostics(model.forward_block.frame_matrix()),
            "backward": _frame_diagnostics(model.backward_block.frame_matrix()),
        },
        "modal": {
            "forward": _modal_diagnostics(model.forward_block),
            "backward": _modal_diagnostics(model.backward_block),
        },
        "pair_energy": _pair_energy_diagnostics(inputs, low, detail),
        "fusion_weights": [
            float(value)
            for value in torch.softmax(model.band_logits.detach(), dim=0).cpu().tolist()
        ],
    }


def theory_diagnostic_status(
    root: Path = DEFAULT_ROOT,
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, object]:
    jobs = theory_diagnostic_jobs(checkpoint_root)
    completed = _result_keys(root / "completed")
    failed = _result_keys(root / "failed") - completed
    return {
        "expected": len(jobs),
        "checkpoints_available": sum(Path(job.checkpoint).exists() for job in jobs),
        "completed": len(completed),
        "failed": len(failed),
        "remaining": len(jobs) - len(completed) - len(failed),
        "done": len(completed) == len(jobs) and not failed,
    }


def write_theory_diagnostic_report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    rows = _completed_rows(root / "completed")
    tasks: dict[str, object] = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        low = [float(row["fusion_weights"][0]) for row in selected]
        detail = [float(row["fusion_weights"][1]) for row in selected]
        tasks[dataset] = {
            "seeds": len(selected),
            "fusion_low_mean": mean(low) if low else None,
            "fusion_low_std": pstdev(low) if len(low) > 1 else 0.0 if low else None,
            "fusion_detail_mean": mean(detail) if detail else None,
            "fusion_detail_std": pstdev(detail) if len(detail) > 1 else 0.0 if detail else None,
            "stiefel_gram_frobenius_mean": _nested_mean(
                selected, "stiefel", "forward", "gram_frobenius"
            ),
            "pair_energy_relative_max": _nested_max(selected, "pair_energy", "relative_max"),
            "forward_contraction_dt1_mean": _nested_mean(
                selected, "modal", "forward", "contraction_mean_dt_1"
            ),
        }
    payload = {
        "schema": "pac_alphabet_theory_diagnostic_report.v1",
        "status": theory_diagnostic_status(root),
        "rows": len(rows),
        "tasks": tasks,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "THEORY_DIAGNOSTICS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _frame_diagnostics(frame: Tensor) -> dict[str, object]:
    active = frame.detach().float()
    identity = torch.eye(active.shape[1], device=active.device, dtype=active.dtype)
    residual = active.transpose(0, 1) @ active - identity
    singular = torch.linalg.svdvals(active)
    return {
        "gram_frobenius": float(torch.linalg.matrix_norm(residual).item()),
        "gram_max_abs": float(residual.abs().max().item()),
        "singular_values": [float(value) for value in singular.cpu().tolist()],
        "singular_min": float(singular.min().item()),
        "singular_max": float(singular.max().item()),
        "singular_max_abs_deviation_from_one": float((singular - 1.0).abs().max().item()),
    }


def _modal_diagnostics(block: nn.Module) -> dict[str, object]:
    damping = block.damping_values().detach().float()  # type: ignore[attr-defined]
    per_mode = damping
    payload: dict[str, object] = {
        "damping_per_mode": [float(value) for value in per_mode.cpu().tolist()],
        "damping_min": float(damping.min().item()),
        "damping_max": float(damping.max().item()),
        "frequency_per_mode": [
            float(value)
            for value in block.frequency_values().detach().float().cpu().tolist()  # type: ignore[attr-defined]
        ],
    }
    for delta in DELTA_GRID:
        contraction = torch.exp(-per_mode * delta)
        label = _delta_label(delta)
        payload[f"contraction_per_mode_dt_{label}"] = [
            float(value) for value in contraction.cpu().tolist()
        ]
        payload[f"contraction_mean_dt_{label}"] = float(contraction.mean().item())
        payload[f"contraction_min_dt_{label}"] = float(contraction.min().item())
        payload[f"contraction_max_dt_{label}"] = float(contraction.max().item())
    return payload


def _pair_energy_diagnostics(inputs: Tensor, low: Tensor, detail: Tensor) -> dict[str, float]:
    padded = inputs if inputs.shape[1] % 2 == 0 else torch.nn.functional.pad(inputs, (0, 0, 0, 1))
    original = padded[:, 0::2].square() + padded[:, 1::2].square()
    transformed = low.square() + detail.square()
    residual = (transformed - original).abs()
    scale = original.abs().clamp_min(torch.finfo(original.dtype).eps)
    relative = residual / scale
    return {
        "absolute_mean": float(residual.mean().item()),
        "absolute_max": float(residual.max().item()),
        "relative_mean": float(relative.mean().item()),
        "relative_max": float(relative.max().item()),
    }


def _config_from_checkpoint(checkpoint: dict[str, object], device: str) -> PACExperimentConfig:
    values = cast("dict[str, object]", checkpoint["config"])
    return PACExperimentConfig(
        int(values["sample_count"]),
        int(values["validation_count"]),
        int(values["test_count"]),
        int(values["sequence_length"]),
        raw_input_dim=int(values["raw_input_dim"]),
        output_dim=int(values["output_dim"]),
        model_dim=int(values["model_dim"]),
        modes=int(values["modes"]),
        epochs=int(values["epochs"]),
        batch_size=int(values["batch_size"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        grad_clip_norm=float(values["grad_clip_norm"]),
        seeds=(int(values["seed"]),),
        device=cast("PACDevice", device),
    )


def _delta_label(delta: float) -> str:
    return str(int(delta)) if delta.is_integer() else str(delta).replace(".", "p")


def _safe(key: str) -> str:
    return key.replace(":", "_").replace("/", "_")


def _result_path(root: Path, job: TheoryDiagnosticJob, *, failed: bool) -> Path:
    return root / ("failed" if failed else "completed") / f"{_safe(job.key)}.json"


def _write_result(
    root: Path, job: TheoryDiagnosticJob, row: dict[str, object], *, failed: bool
) -> None:
    path = _result_path(root, job, failed=failed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _result_keys(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        str(json.loads(path.read_text(encoding="utf-8"))["job_key"])
        for path in directory.glob("*.json")
    }


def _completed_rows(directory: Path) -> list[dict[str, object]]:
    if not directory.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")]


def _nested_mean(rows: list[dict[str, object]], *keys: str) -> float | None:
    values = [_nested_float(row, keys) for row in rows]
    return mean(values) if values else None


def _nested_max(rows: list[dict[str, object]], *keys: str) -> float | None:
    values = [_nested_float(row, keys) for row in rows]
    return max(values) if values else None


def _nested_float(row: dict[str, object], keys: tuple[str, ...]) -> float:
    value: object = row
    for key in keys:
        value = cast("dict[str, object]", value)[key]
    return float(cast("float", value))
