# ruff: noqa: BLE001, E501, EM101, EM102, TRY003
from __future__ import annotations

import gc
import hashlib
import json
import math
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Final, Literal, cast

import torch
from torch import Tensor, nn

from .pac_eval_sections import clean_validation_classification_task
from .pac_final_validation import UCR_SECONDS
from .pac_headroom_efficient_models import (
    EDGE_FRAME_VARIANT,
    EdgeFramePAC,
    _apply_raw_mask,  # pyright: ignore[reportPrivateUsage]
    _combined_edge_mask,  # pyright: ignore[reportPrivateUsage]
    _degree_normalized_edge_analysis,  # pyright: ignore[reportPrivateUsage]
    _edge_or_singleton_delta,  # pyright: ignore[reportPrivateUsage]
    _edge_or_singleton_mask,  # pyright: ignore[reportPrivateUsage]
    build_efficient_headroom_classifier,
)
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_train_only
from .pac_retrained_ablation_campaign import DATASETS, SEEDS
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACDevice, PACExperimentConfig

DEFAULT_ROOT: Final = Path(".omx/results/pac-efp16-retrained-ablation-20260713")
EFP16AblationVariant = Literal[
    "efp16_full",
    "unnormalized_edges",
    "unconstrained_projection",
    "pointwise_local",
    "efp8_modal",
    "tied_synthesis",
]
ABLATION_VARIANTS: Final[tuple[EFP16AblationVariant, ...]] = (
    "efp16_full",
    "unnormalized_edges",
    "unconstrained_projection",
    "pointwise_local",
    "efp8_modal",
    "tied_synthesis",
)


@dataclass(frozen=True, slots=True)
class EFP16AblationMetadata:
    variant: EFP16AblationVariant
    edge_analysis: str
    projection_constraint: str
    local_map: str
    synthesis_map: str
    modes: int
    params_trainable: int
    target_params: int
    relative_param_error: float
    capacity_interpretation: str


@dataclass(frozen=True, slots=True)
class EFP16AblationJob:
    key: str
    dataset: str
    variant: EFP16AblationVariant
    seed: int
    epochs: int = 100
    batch_size: int = 64
    estimated_seconds: float = 60.0


class _EFP16ControlPAC(EdgeFramePAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        normalized_edges: bool,
        semi_orthogonal: bool,
        modes: int,
        pointwise_local: bool,
        tied_synthesis: bool,
    ) -> None:
        super().__init__(
            config,
            output_dim,
            modes=modes,
            semi_orthogonal=semi_orthogonal,
            objective="classification",
            pac_variant=replace(
                EDGE_FRAME_VARIANT,
                tie_analysis_synthesis=tied_synthesis,
            ),
        )
        self.normalized_edges = normalized_edges
        if pointwise_local:
            self.stem.local = nn.Conv1d(32, 32, kernel_size=1, groups=32)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        if self.normalized_edges:
            low, detail, active_delta = _degree_normalized_edge_analysis(
                stem_inputs, time_delta
            )
        else:
            low, detail, active_delta = _unnormalized_edge_analysis(
                stem_inputs, time_delta
            )
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        edge_mask = _combined_edge_mask(active_observation, active_valid)
        edge_features = torch.cat((low, detail), dim=-1)
        if edge_mask is not None:
            edge_features = edge_features * edge_mask.to(
                device=edge_features.device, dtype=edge_features.dtype
            )
        encoded = self.stem(edge_features)
        if active_valid is not None:
            encoded = encoded * active_valid.to(
                device=encoded.device, dtype=encoded.dtype
            )
        encoded, forward_moments = self.forward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        encoded, backward_moments = self.backward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        return self._readout(
            encoded, forward_moments, backward_moments, active_valid
        )


def _unnormalized_edge_analysis(
    inputs: Tensor, time_delta: Tensor | None
) -> tuple[Tensor, Tensor, Tensor | None]:
    if inputs.ndim != 3 or inputs.shape[1] < 1:
        raise ValueError("edge analysis requires inputs with shape [B,N>=1,D]")
    if inputs.shape[1] == 1:
        return inputs, torch.zeros_like(inputs), _edge_or_singleton_delta(time_delta)
    first, second = inputs[:, :-1], inputs[:, 1:]
    if time_delta is None:
        scale = inputs.new_tensor(1.0 / math.sqrt(2.0))
        return scale * (first + second), scale * (first - second), None
    delta = time_delta if time_delta.ndim == 3 else time_delta.unsqueeze(-1)
    delta = delta.clamp_min(0.0)
    first_delta, second_delta = delta[:, :-1], delta[:, 1:]
    total = first_delta + second_delta
    safe_total = total.clamp_min(torch.finfo(inputs.dtype).eps)
    first_weight = torch.sqrt(first_delta / safe_total).to(dtype=inputs.dtype)
    second_weight = torch.sqrt(second_delta / safe_total).to(dtype=inputs.dtype)
    empty = total <= 0.0
    equal = inputs.new_tensor(1.0 / math.sqrt(2.0))
    first_weight = torch.where(empty, equal, first_weight)
    second_weight = torch.where(empty, equal, second_weight)
    return (
        first_weight * first + second_weight * second,
        second_weight * first - first_weight * second,
        second_delta,
    )


def build_ablation_model(
    variant: EFP16AblationVariant,
    config: PACExperimentConfig,
    output_dim: int,
) -> tuple[nn.Module, EFP16AblationMetadata]:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"unknown EFP16 ablation variant: {variant}")
    target = build_efficient_headroom_classifier(
        "EFP16", config, output_dim, objective="classification"
    )
    target_params = count_parameters(target)
    del target
    normalized = variant != "unnormalized_edges"
    semi_orthogonal = variant != "unconstrained_projection"
    pointwise = variant == "pointwise_local"
    modes = 8 if variant == "efp8_modal" else 16
    tied_synthesis = variant == "tied_synthesis"
    model = _EFP16ControlPAC(
        config,
        output_dim,
        normalized_edges=normalized,
        semi_orthogonal=semi_orthogonal,
        modes=modes,
        pointwise_local=pointwise,
        tied_synthesis=tied_synthesis,
    )
    parameters = count_parameters(model)
    error = abs(parameters - target_params) / target_params
    return model, EFP16AblationMetadata(
        variant=variant,
        edge_analysis=(
            "degree-normalized Parseval adjacent-edge analysis"
            if normalized
            else "adjacent-edge analysis without node-degree normalization"
        ),
        projection_constraint=(
            "semi-orthogonal QR retraction" if semi_orthogonal else "unconstrained"
        ),
        local_map=(
            "depthwise pointwise control"
            if pointwise
            else "depthwise kernel-5 dilation-4 context"
        ),
        synthesis_map=(
            "analysis transpose shared for synthesis"
            if tied_synthesis
            else "independently learned synthesis"
        ),
        modes=modes,
        params_trainable=parameters,
        target_params=target_params,
        relative_param_error=error,
        capacity_interpretation=(
            "exact parameter-count control"
            if error == 0.0
            else "component-removal control; parameter difference is reported, not hidden"
        ),
    )


def ablation_jobs() -> list[EFP16AblationJob]:
    runtime = {
        "efp16_full": 1.0,
        "unnormalized_edges": 1.0,
        "unconstrained_projection": 1.0,
        "pointwise_local": 0.9,
        "efp8_modal": 0.8,
        "tied_synthesis": 1.0,
    }
    return [
        EFP16AblationJob(
            key=f"efp16_ucr_validation:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=variant,
            seed=seed,
            estimated_seconds=UCR_SECONDS[dataset] * runtime[variant],
        )
        for dataset in DATASETS
        for variant in ABLATION_VARIANTS
        for seed in SEEDS
    ]


def enqueue(root: Path = DEFAULT_ROOT, *, workers: int = 6) -> dict[str, object]:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    jobs = ablation_jobs()
    completed = _result_keys(root / "completed")
    pending = [job for job in jobs if job.key not in completed]
    shards: list[list[EFP16AblationJob]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(pending, key=lambda item: item.estimated_seconds, reverse=True):
        worker_index = min(range(workers), key=loads.__getitem__)
        shards[worker_index].append(job)
        loads[worker_index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for stale in manifests.glob("worker-*.jsonl"):
        stale.unlink()
    for worker_index, shard in enumerate(shards):
        (manifests / f"worker-{worker_index:02d}.jsonl").write_text(
            "".join(
                json.dumps(asdict(job), sort_keys=True) + "\n"
                for job in sorted(shard, key=lambda item: (item.estimated_seconds, item.key))
            ),
            encoding="utf-8",
        )
    contract = {
        "schema": "pac_efp16_retrained_ablation_contract.v1",
        "purpose": "retrained component controls for the EFP16 Parseval edge-frame architecture",
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "variants": list(ABLATION_VARIANTS),
        "jobs": len(jobs),
        "pending": len(pending),
        "workers": workers,
        "estimated_worker_seconds": loads,
        "restart_safe": True,
        "shared_recipe": {
            "epochs": 100,
            "batch_size": 64,
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "checkpoint": "minimum TRAIN-derived validation loss",
        },
        "factorization": {
            "efp16_full": "all EFP16 components",
            "unnormalized_edges": "remove node-degree normalization only",
            "unconstrained_projection": "remove semi-orthogonal QR retraction only",
            "pointwise_local": "replace dilated local context with a pointwise depthwise map",
            "efp8_modal": "halve the modal frame from 16 to 8 modes",
            "tied_synthesis": "reuse each analysis frame transpose for synthesis (S=R^T)",
        },
        "capacity_policy": (
            "full, unnormalized, and unconstrained variants are exact-count controls; "
            "pointwise, eight-mode, and tied-synthesis controls report their lower counts explicitly"
        ),
    }
    _atomic_json(root / "contract.json", contract)
    return {
        "jobs": len(jobs),
        "pending": len(pending),
        "workers": workers,
        "estimated_worker_seconds": loads,
    }


def run_job(
    job: EFP16AblationJob,
    *,
    device: str,
    checkpoint_root: Path | None = None,
) -> dict[str, object]:
    dataset = ensure_ucr_train_only(job.dataset, Path(".omx/data/ucr"), allow_download=True)
    task = clean_validation_classification_task(dataset, job.seed)
    config = PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        0,
        task.train_inputs.shape[1],
        raw_input_dim=task.train_inputs.shape[-1],
        output_dim=task.class_count,
        model_dim=32,
        modes=16,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        seeds=(job.seed,),
        device=cast("PACDevice", device),
    )
    torch.manual_seed(job.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(job.seed)
        torch.cuda.reset_peak_memory_stats()
    model, metadata = build_ablation_model(job.variant, config, task.class_count)
    started = perf_counter()
    outcome = train_classifier(
        model,
        task,
        config,
        device,
        job.seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    train_seconds = perf_counter() - started
    metrics = classification_metric_bundle(
        model,
        task.validation_inputs.to(device=device),
        task.validation_labels.to(device=device),
        batch_size=job.batch_size,
    )
    checkpoint: dict[str, object] = {}
    if job.variant == "efp16_full" and checkpoint_root is not None:
        checkpoint = save_checkpoint(
            checkpoint_root,
            job,
            model,
            config,
            class_count=task.class_count,
            best_epoch=outcome.best_epoch,
        )
    return {
        "schema": "pac_efp16_retrained_ablation_result.v1",
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "evaluation_split": "validation",
        "official_test_accessed": False,
        "normalization_fit": "optimization fold only",
        "checkpoint_policy": "minimum validation loss",
        "best_epoch": outcome.best_epoch,
        "validation_loss": outcome.validation_loss,
        "validation_accuracy": metrics.accuracy,
        "validation_macro_f1": metrics.macro_f1,
        "validation_weighted_f1": metrics.weighted_f1,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        "train_seconds": train_seconds,
        "peak_memory_mb": (
            float(torch.cuda.max_memory_allocated() / 1.0e6) if device == "cuda" else 0.0
        ),
        "params_trainable": count_parameters(model),
        "model_metadata": asdict(metadata),
        **checkpoint,
    }


def run_manifest(root: Path, manifest: Path, *, device: str) -> None:
    jobs = [
        EFP16AblationJob(**json.loads(line))
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for job in jobs:
        completed = _result_path(root, job, failed=False)
        if completed.exists():
            continue
        try:
            row = run_job(job, device=device, checkpoint_root=root / "checkpoints")
        except Exception as error:  # durable restart-safe queue cell
            _atomic_json(
                _result_path(root, job, failed=True),
                {
                    "schema": "pac_efp16_retrained_ablation_failure.v1",
                    "job_key": job.key,
                    **asdict(job),
                    "status": "failed",
                    "official_test_accessed": False,
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


def save_checkpoint(
    checkpoint_root: Path,
    job: EFP16AblationJob,
    model: nn.Module,
    config: PACExperimentConfig,
    *,
    class_count: int,
    best_epoch: int | None,
) -> dict[str, object]:
    path = checkpoint_root / job.dataset / f"efp16_full_seed{job.seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    payload: dict[str, object] = {
        "schema": "pac_efp16_validation_checkpoint.v1",
        "job": asdict(job),
        "architecture": "EFP16",
        "objective": "classification",
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "checkpoint_policy": "minimum TRAIN-derived validation loss",
        "best_epoch": best_epoch,
        "config": {
            "sample_count": config.sample_count,
            "validation_count": config.validation_count,
            "test_count": config.test_count,
            "sequence_length": config.sequence_length,
            "raw_input_dim": config.raw_input_dim,
            "output_dim": config.output_dim,
            "model_dim": 32,
            "modes": 16,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "grad_clip_norm": config.grad_clip_norm,
            "seed": job.seed,
            "class_count": class_count,
        },
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    expected = {job.key for job in ablation_jobs()}
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
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in (root / "completed").glob("*.json")]
    variants: dict[str, object] = {}
    for variant in ABLATION_VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        variants[variant] = {
            "mean_balanced_accuracy": (
                mean(float(row["validation_balanced_accuracy"]) for row in selected)
                if selected
                else None
            ),
            "rows": len(selected),
            "mean_params_trainable": (
                mean(int(row["params_trainable"]) for row in selected) if selected else None
            ),
        }
    payload: dict[str, object] = {
        "schema": "pac_efp16_retrained_ablation_report.v1",
        "status": status(root),
        "variants": variants,
    }
    _atomic_json(root / "reports" / "summary.json", payload)
    return payload


def _result_path(root: Path, job: EFP16AblationJob, *, failed: bool) -> Path:
    safe = job.key.replace(":", "_").replace("/", "_")
    return root / ("failed" if failed else "completed") / f"{safe}.json"


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
