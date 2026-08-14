"""Restart-safe component controls for the frozen Compact H-only ALPHABET.

The campaign is deliberately independent of the historical EFP16 and PA2WP
ablation artifacts.  It freezes each UCR task's final architecture-selection
cell, keeps the CompactEFPHOnlyTerminalPAC writer/reader contract unless that
component is the named intervention, and evaluates TRAIN-derived validation
splits only.  Removed parameters are reported rather than replaced by inert
capacity.
"""

# ruff: noqa: BLE001, EM101, EM102, TRY003
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Final, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_device import resolve_device
from .pac_efp16_final_campaign import UCR_DATASETS
from .pac_efp_writer_reader import CompactEFPHOnlyTerminalPAC
from .pac_eval_sections import clean_validation_classification_task
from .pac_final_validation import UCR_SECONDS
from .pac_headroom_efficient_models import (
    StrideOneConvStem,
    _apply_raw_mask,  # pyright: ignore[reportPrivateUsage]
    _combined_edge_mask,  # pyright: ignore[reportPrivateUsage]
    _degree_normalized_edge_analysis,  # pyright: ignore[reportPrivateUsage]
    _edge_or_singleton_delta,  # pyright: ignore[reportPrivateUsage]
    _edge_or_singleton_mask,  # pyright: ignore[reportPrivateUsage]
)
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_train_only
from .pac_tight_frame_models import (
    _InvariantMomentHead,  # pyright: ignore[reportPrivateUsage]
)
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACClassificationTask, PACDevice, PACExperimentConfig

DEFAULT_ROOT: Final = Path(".omx/results/pac-compact-h-only-ablation-20260719")
DEFAULT_SELECTION: Final = Path(
    ".omx/results/pac-alphabet-q1q2-final-20260719/stage2/selection.json"
)
UCR_DATA_ROOT: Final = Path(".omx/data/ucr")
FINAL_SEEDS: Final = (23, 31, 43, 47, 59)
SELECTION_SEEDS: Final = (7, 11, 19)

CompactAblationVariant = Literal[
    "full",
    "no_degree_normalization",
    "level_only",
    "detail_only",
    "unconstrained_two_tap",
    "no_second_projection",
    "no_terminal_reader",
]
VARIANTS: Final[tuple[CompactAblationVariant, ...]] = (
    "full",
    "no_degree_normalization",
    "level_only",
    "detail_only",
    "unconstrained_two_tap",
    "no_second_projection",
    "no_terminal_reader",
)


class IncompleteCampaignError(RuntimeError):
    """Raised instead of publishing incomplete or contaminated evidence."""


@dataclass(frozen=True, slots=True)
class SelectedCompactConfig:
    dataset: str
    model_dim: int
    modes: int
    trial: int
    epochs: int
    width_tier: int
    config_key: str
    selection_sha256: str


@dataclass(frozen=True, slots=True)
class CompactAblationJob:
    key: str
    dataset: str
    variant: CompactAblationVariant
    split_seed: int
    train_seed: int
    model_dim: int
    modes: int
    trial: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    width_tier: int
    selected_config_key: str
    selection_sha256: str
    evaluation_split: Literal["validation"]
    estimated_seconds: float


@dataclass(frozen=True, slots=True)
class CompactAblationMetadata:
    variant: CompactAblationVariant
    edge_analysis: str
    stem: str
    writer: str
    terminal_reader: str
    second_projection: str
    model_dim: int
    modes: int
    params_trainable: int
    full_params_trainable: int
    parameter_delta_vs_full: int
    inert_or_padding_parameters_added: bool = False


class _OneCoordinateStem(nn.Module):
    """Semi-orthogonal one-coordinate stem with the canonical local map."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, model_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.local = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=5,
            dilation=4,
            padding=8,
            groups=model_dim,
        )
        self.project_weight_()

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.projection(inputs)
        local = self.local(projected.transpose(1, 2)).transpose(1, 2)
        return functional.silu(local)

    @torch.no_grad()
    def project_weight_(self) -> None:
        weight = self.projection.weight
        active = weight.float() if weight.shape[0] >= weight.shape[1] else weight.float().T
        frame, upper = torch.linalg.qr(active, mode="reduced")
        signs = torch.where(
            torch.diagonal(upper) >= 0.0,
            torch.ones((), device=weight.device),
            -torch.ones((), device=weight.device),
        )
        projected = frame * signs.unsqueeze(0)
        if weight.shape[0] < weight.shape[1]:
            projected = projected.T
        weight.copy_(projected.to(dtype=weight.dtype))


def _unnormalized_edge_analysis(
    inputs: Tensor,
    time_delta: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Canonical adjacent coordinates without node-degree normalization."""
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


class _CompactControlPAC(CompactEFPHOnlyTerminalPAC):
    """One active intervention around the frozen compact writer/reader."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        variant: CompactAblationVariant,
    ) -> None:
        if variant == "full":
            raise ValueError("the full variant must use the canonical class directly")
        super().__init__(config, output_dim, objective="classification")
        self.ablation_variant = variant
        if variant in {"level_only", "detail_only"}:
            self.stem = _OneCoordinateStem(config.raw_input_dim, self.model_dim)
        elif variant == "unconstrained_two_tap":
            self.stem = StrideOneConvStem(config.raw_input_dim, self.model_dim)
        elif variant == "no_second_projection":
            self.second_projection = nn.Identity()
        elif variant == "no_terminal_reader":
            full_head = self.head
            self.second_projection = nn.Identity()
            self.second_local = nn.Identity()
            self.backward_block = nn.Identity()
            reduced_head = _InvariantMomentHead(
                self.model_dim,
                self.modes,
                output_dim,
                use_modal_moments=True,
                use_backward_moments=False,
                lags=self.forward_block.moment_lags,
            )
            with torch.no_grad():
                retained = reduced_head.classifier.in_features
                reduced_head.classifier.weight.copy_(full_head.classifier.weight[:, :retained])
                reduced_head.classifier.bias.copy_(full_head.classifier.bias)
            self.head = reduced_head

    def _edge_coordinates(
        self,
        inputs: Tensor,
        time_delta: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        variant = self.ablation_variant
        if variant == "unconstrained_two_tap":
            return inputs, _edge_or_singleton_delta(time_delta)
        if variant == "no_degree_normalization":
            level, detail, active_delta = _unnormalized_edge_analysis(inputs, time_delta)
        else:
            level, detail, active_delta = _degree_normalized_edge_analysis(
                inputs,
                time_delta,
            )
        if variant == "level_only":
            return level, active_delta
        if variant == "detail_only":
            return detail, active_delta
        return torch.cat((level, detail), dim=-1), active_delta

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        edge_features, active_delta = self._edge_coordinates(stem_inputs, time_delta)
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        edge_mask = _combined_edge_mask(active_observation, active_valid)
        if edge_mask is not None:
            edge_features = edge_features * edge_mask.to(
                device=edge_features.device,
                dtype=edge_features.dtype,
            )

        first_local = self._mask_features(self.stem(edge_features), active_valid)
        first_stream, first_moments = self.forward_block(
            first_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        if self.ablation_variant == "no_terminal_reader":
            # The replacement head ignores its third argument and contains no
            # inactive terminal-reader columns.
            return self._readout(
                first_stream,
                first_moments,
                first_moments,
                active_valid,
            )

        second_projected = self.second_projection(first_stream)
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        encoded = self._mask_features(second_local, active_valid)
        second_moments = self.backward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_moments_only=True,
        )
        return self._readout(encoded, first_moments, second_moments, active_valid)

    def post_optimizer_step(self) -> None:
        self.forward_block.retract_frame()
        if self.ablation_variant != "no_terminal_reader":
            self.backward_block.retract_frame()
        projector = getattr(self.stem, "project_weight_", None)
        if callable(projector):
            projector()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        if self.ablation_variant != "no_terminal_reader":
            self.backward_block.finalize_frame()
        projector = getattr(self.stem, "project_weight_", None)
        if callable(projector):
            projector()


def build_ablation_model(
    variant: CompactAblationVariant,
    config: PACExperimentConfig,
    output_dim: int,
) -> tuple[nn.Module, CompactAblationMetadata]:
    """Build one control without capacity padding or width rematching."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown compact H-only ablation variant: {variant}")
    random_state = torch.random.get_rng_state()
    canonical = CompactEFPHOnlyTerminalPAC(
        config,
        output_dim,
        objective="classification",
    )
    full_params = count_parameters(canonical)
    if variant == "full":
        model: nn.Module = canonical
    else:
        del canonical
        # Keep every unaffected tensor on the canonical same-seed
        # initialization.  The temporary full model above is used only for the
        # active parameter-count reference and must not advance the control's
        # initialization stream.
        torch.random.set_rng_state(random_state)
        model = _CompactControlPAC(config, output_dim, variant=variant)
    params = count_parameters(model)
    metadata = CompactAblationMetadata(
        variant=variant,
        edge_analysis={
            "full": "degree-normalized complementary level/detail coordinates",
            "no_degree_normalization": "level/detail coordinates without node-degree normalization",
            "level_only": "degree-normalized level coordinate only",
            "detail_only": "degree-normalized detail coordinate only",
            "unconstrained_two_tap": "overlapping learned unconstrained two-tap map",
            "no_second_projection": "degree-normalized complementary level/detail coordinates",
            "no_terminal_reader": "degree-normalized complementary level/detail coordinates",
        }[variant],
        stem=(
            "unconstrained learned two-tap projection plus canonical local map"
            if variant == "unconstrained_two_tap"
            else (
                "single-coordinate semi-orthogonal projection plus canonical local map"
                if variant in {"level_only", "detail_only"}
                else "canonical joint semi-orthogonal projection plus local map"
            )
        ),
        writer="frozen forward exact-pole writer with synthesis and residual update",
        terminal_reader=(
            "removed; first writer stream and moments feed a reduced active head"
            if variant == "no_terminal_reader"
            else "H-only local lift and read-only forward exact-pole scan"
        ),
        second_projection=(
            "identity with no trainable parameters"
            if variant in {"no_second_projection", "no_terminal_reader"}
            else "learned identity-initialized dense projection"
        ),
        model_dim=config.model_dim,
        modes=config.modes,
        params_trainable=params,
        full_params_trainable=full_params,
        parameter_delta_vs_full=params - full_params,
    )
    return model, metadata


def _sealed_selection_payload(selection_path: Path) -> tuple[dict[str, object], str]:
    if not selection_path.exists():
        raise FileNotFoundError(f"final selection artifact is missing: {selection_path}")
    raw = selection_path.read_bytes()
    selection_sha256 = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("final selection must be a JSON object")
    if payload.get("schema") != "pac_alphabet_q1_final_freeze.v1":
        raise ValueError("final selection has an incompatible schema")
    if payload.get("chosen_internal_model") != "compact_h_only":
        raise ValueError("final selection does not freeze compact_h_only")
    if payload.get("test_evidence_used_for_architecture_choice") is not False:
        raise ValueError("architecture selection is not sealed from TEST evidence")
    if tuple(payload.get("selection_seeds", ())) != SELECTION_SEEDS:
        raise ValueError("final selection does not use the sealed selection seeds")
    if tuple(payload.get("final_seeds", ())) != FINAL_SEEDS:
        raise ValueError("final selection does not freeze the final five seeds")
    return payload, selection_sha256


def _selected_config(
    dataset: str,
    row: object,
    selection_sha256: str,
) -> SelectedCompactConfig:
    if not isinstance(row, dict):
        raise TypeError(f"final selection is missing compact_h_only UCR cell: {dataset}")
    model_dim = int(row["model_dim"])
    modes = int(row["modes"])
    trial = int(row["trial"])
    epochs = int(row["ucr_refit_epochs"])
    valid = (
        model_dim >= 1
        and modes >= 1
        and 2 * modes <= model_dim
        and trial in range(1, 7)
        and epochs >= 1
        and tuple(row.get("selection_seeds", ())) == SELECTION_SEEDS
    )
    if not valid:
        raise ValueError(f"invalid frozen compact configuration: {dataset}")
    return SelectedCompactConfig(
        dataset=dataset,
        model_dim=model_dim,
        modes=modes,
        trial=trial,
        epochs=epochs,
        width_tier=int(row["width_tier"]),
        config_key=str(row["config_key"]),
        selection_sha256=selection_sha256,
    )


def load_selected_configs(
    selection_path: Path = DEFAULT_SELECTION,
) -> dict[str, SelectedCompactConfig]:
    payload, selection_sha256 = _sealed_selection_payload(selection_path)
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise TypeError("final selection lacks selected cells")
    return {
        dataset: _selected_config(
            dataset,
            selected.get(f"ucr:{dataset}:compact_h_only"),
            selection_sha256,
        )
        for dataset in UCR_DATASETS
    }


def ablation_jobs(
    selection_path: Path = DEFAULT_SELECTION,
) -> list[CompactAblationJob]:
    selected = load_selected_configs(selection_path)
    runtime_factor: dict[CompactAblationVariant, float] = {
        "full": 1.0,
        "no_degree_normalization": 1.0,
        "level_only": 1.0,
        "detail_only": 1.0,
        "unconstrained_two_tap": 1.0,
        "no_second_projection": 1.0,
        "no_terminal_reader": 0.6,
    }
    jobs: list[CompactAblationJob] = []
    for dataset in UCR_DATASETS:
        frozen = selected[dataset]
        recipe = confirmatory_trial_spec("pac_tf", frozen.trial)
        capacity_factor = max(
            0.25,
            (frozen.model_dim / 32.0) * (frozen.modes / 16.0),
        )
        for variant in VARIANTS:
            jobs.extend(
                CompactAblationJob(
                    key=f"compact_h_only_ablation:{dataset}:{variant}:seed{seed}",
                    dataset=dataset,
                    variant=variant,
                    split_seed=seed,
                    train_seed=seed,
                    model_dim=frozen.model_dim,
                    modes=frozen.modes,
                    trial=frozen.trial,
                    epochs=frozen.epochs,
                    batch_size=recipe.batch_size,
                    learning_rate=recipe.learning_rate,
                    weight_decay=recipe.weight_decay,
                    grad_clip_norm=recipe.grad_clip_norm,
                    width_tier=frozen.width_tier,
                    selected_config_key=frozen.config_key,
                    selection_sha256=frozen.selection_sha256,
                    evaluation_split="validation",
                    estimated_seconds=(
                        UCR_SECONDS[dataset]
                        * capacity_factor
                        * (frozen.epochs / 100.0)
                        * runtime_factor[variant]
                    ),
                )
                for seed in FINAL_SEEDS
            )
    return jobs


def enqueue(
    root: Path = DEFAULT_ROOT,
    *,
    selection_path: Path = DEFAULT_SELECTION,
    workers: int = 8,
) -> dict[str, object]:
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    jobs = ablation_jobs(selection_path)
    completed = _result_keys(root / "completed")
    pending = [job for job in jobs if job.key not in completed]
    shards: list[list[CompactAblationJob]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(pending, key=lambda item: item.estimated_seconds, reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for stale in manifests.glob("worker-*.jsonl"):
        stale.unlink()
    for index, shard in enumerate(shards):
        _atomic_text(
            manifests / f"worker-{index:02d}.jsonl",
            "".join(
                json.dumps(asdict(job), sort_keys=True) + "\n"
                for job in sorted(shard, key=lambda item: item.key)
            ),
        )
    selection_sha256 = next(iter(jobs)).selection_sha256
    contract = {
        "schema": "pac_compact_h_only_ablation_contract.v1",
        "purpose": "component controls for the frozen Compact H-only ALPHABET",
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": selection_sha256,
        "selection_schema": "pac_alphabet_q1_final_freeze.v1",
        "chosen_internal_model": "compact_h_only",
        "evaluation_split": "official UCR TRAIN-derived validation only",
        "official_test_accessed": False,
        "test_evidence_used_for_selection": False,
        "datasets": list(UCR_DATASETS),
        "variants": list(VARIANTS),
        "seeds": list(FINAL_SEEDS),
        "jobs": len(jobs),
        "pending": len(pending),
        "workers": workers,
        "estimated_total_gpu_seconds": sum(job.estimated_seconds for job in jobs),
        "estimated_worker_seconds": loads,
        "restart_safe": True,
        "report_policy": "fail closed until every expected job succeeds",
        "architecture_policy": (
            "full is the canonical CompactEFPHOnlyTerminalPAC; each control changes "
            "one named component while retaining the selected D/M and writer-reader "
            "contract except for the explicit no_terminal_reader intervention"
        ),
        "parameter_policy": (
            "no width rematching, dummy, padding, frozen, or otherwise inert trainable "
            "parameters; every active count and signed delta from full are reported"
        ),
        "training_policy": (
            "task-specific final trial and refit epochs; split_seed=train_seed in "
            "{23,31,43,47,59}; validation checkpoint; evaluate_test=False"
        ),
    }
    _atomic_json(root / "contract.json", contract)
    return {
        "jobs": len(jobs),
        "pending": len(pending),
        "workers": workers,
        "estimated_total_gpu_seconds": contract["estimated_total_gpu_seconds"],
        "estimated_worker_seconds": loads,
    }


def _experiment_config(
    job: CompactAblationJob,
    task: PACClassificationTask,
    device: PACDevice,
) -> PACExperimentConfig:
    return PACExperimentConfig(
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
        optimizer_mode="fused" if device == "cuda" else "default",
    )


def _configure_training_backends(model: nn.Module, dataset: str) -> None:
    for name in ("forward_block", "backward_block"):
        block = getattr(model, name, None)
        if block is None or not hasattr(block, "recurrence_backend"):
            continue
        block.recurrence_backend = "auto"
        block.fused_moments_backward_training = dataset != "CinCECGTorso"


def run_job(
    job: CompactAblationJob,
    *,
    device: PACDevice,
    data_root: Path = UCR_DATA_ROOT,
) -> dict[str, object]:
    if job.evaluation_split != "validation" or job.split_seed != job.train_seed:
        raise ValueError("ablation job violates the sealed validation split contract")
    runtime_device = resolve_device(device)
    dataset = ensure_ucr_train_only(job.dataset, data_root, allow_download=True)
    task = clean_validation_classification_task(dataset, job.split_seed)
    if task.test_inputs.shape[0] != 0 or task.test_labels.shape[0] != 0:
        raise RuntimeError("validation-only task unexpectedly contains TEST examples")
    config = _experiment_config(job, task, device)
    torch.manual_seed(job.train_seed)
    if runtime_device == "cuda":
        torch.cuda.manual_seed_all(job.train_seed)
        torch.cuda.reset_peak_memory_stats()
    model, metadata = build_ablation_model(job.variant, config, task.class_count)
    model = model.to(device=runtime_device)
    _configure_training_backends(model, job.dataset)
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
    actual_params = count_parameters(model)
    if actual_params != metadata.params_trainable:
        raise RuntimeError("trainable parameter count changed after model construction")
    return {
        "schema": "pac_compact_h_only_ablation_result.v1",
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "official_test_accessed": False,
        "test_evaluated": False,
        "normalization_fit": "optimization fold only",
        "checkpoint_policy": "minimum TRAIN-derived validation loss",
        "train_count": int(task.train_inputs.shape[0]),
        "validation_count": int(task.validation_inputs.shape[0]),
        "test_count": 0,
        "best_epoch": outcome.best_epoch,
        "validation_loss": outcome.validation_loss,
        "validation_accuracy": metrics.accuracy,
        "validation_macro_f1": metrics.macro_f1,
        "validation_weighted_f1": metrics.weighted_f1,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        "train_seconds": elapsed,
        "peak_memory_mb": (
            float(torch.cuda.max_memory_allocated() / 1_000_000)
            if runtime_device == "cuda"
            else 0.0
        ),
        "params_trainable": actual_params,
        "model_metadata": asdict(metadata),
    }


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: PACDevice,
    data_root: Path = UCR_DATA_ROOT,
) -> None:
    jobs = [
        CompactAblationJob(**json.loads(line))
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for job in jobs:
        completed = _result_path(root, job, failed=False)
        if completed.exists():
            continue
        try:
            row = run_job(job, device=device, data_root=data_root)
        except Exception as error:
            _atomic_json(
                _result_path(root, job, failed=True),
                {
                    "schema": "pac_compact_h_only_ablation_failure.v1",
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


def status(
    root: Path = DEFAULT_ROOT,
    *,
    selection_path: Path = DEFAULT_SELECTION,
) -> dict[str, object]:
    expected = {job.key for job in ablation_jobs(selection_path)}
    completed_list = _result_key_list(root / "completed")
    failed_list = _result_key_list(root / "failed")
    completed = set(completed_list)
    failed = set(failed_list) - completed
    unexpected = (completed | failed) - expected
    duplicate_files = (len(completed_list) - len(completed)) + (
        len(failed_list) - len(set(failed_list))
    )
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed - failed),
        "unexpected": len(unexpected),
        "duplicate_files": duplicate_files,
        "done": (expected == completed and not failed and not unexpected and duplicate_files == 0),
    }


def report(
    root: Path = DEFAULT_ROOT,
    *,
    selection_path: Path = DEFAULT_SELECTION,
) -> dict[str, object]:
    campaign_status = status(root, selection_path=selection_path)
    if not campaign_status["done"]:
        raise IncompleteCampaignError(
            f"refusing to publish partial compact H-only ablation: {campaign_status}"
        )
    expected_jobs = {job.key: job for job in ablation_jobs(selection_path)}
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]
    for row in rows:
        key = str(row.get("job_key", ""))
        job = expected_jobs.get(key)
        metadata = row.get("model_metadata")
        if (
            job is None
            or any(row.get(field) != value for field, value in asdict(job).items())
            or row.get("schema") != "pac_compact_h_only_ablation_result.v1"
            or row.get("status") != "done"
            or row.get("evaluation_split") != "validation"
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or row.get("test_count") != 0
            or int(row.get("split_seed", -1)) != job.split_seed
            or int(row.get("train_seed", -1)) != job.train_seed
            or row.get("selection_sha256") != job.selection_sha256
            or not isinstance(metadata, dict)
        ):
            raise ValueError(f"invalid or contaminated completed result: {key}")
        if (
            metadata.get("inert_or_padding_parameters_added") is not False
            or int(metadata["params_trainable"]) != int(row["params_trainable"])
            or int(metadata["parameter_delta_vs_full"])
            != int(metadata["params_trainable"]) - int(metadata["full_params_trainable"])
        ):
            raise ValueError(f"invalid parameter audit: {key}")

    full = {
        (str(row["dataset"]), int(row["train_seed"])): float(row["validation_balanced_accuracy"])
        for row in rows
        if row["variant"] == "full"
    }
    variants: dict[str, object] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        variants[variant] = {
            "rows": len(selected),
            "mean_balanced_accuracy": mean(
                float(row["validation_balanced_accuracy"]) for row in selected
            ),
            "mean_params_trainable": mean(int(row["params_trainable"]) for row in selected),
            "mean_parameter_delta_vs_full": mean(
                int(row["model_metadata"]["parameter_delta_vs_full"]) for row in selected
            ),
            "mean_paired_delta_vs_full": mean(
                float(row["validation_balanced_accuracy"])
                - full[(str(row["dataset"]), int(row["train_seed"]))]
                for row in selected
            ),
        }
    payload: dict[str, object] = {
        "schema": "pac_compact_h_only_ablation_report.v1",
        "status": campaign_status,
        "rows": len(rows),
        "variants": variants,
        "selection_sha256": next(iter(expected_jobs.values())).selection_sha256,
        "official_test_accessed": False,
        "test_evaluated": False,
    }
    _atomic_json(root / "reports" / "summary.json", payload)
    return payload


def smoke() -> dict[str, object]:
    config = PACExperimentConfig(
        8,
        4,
        0,
        17,
        raw_input_dim=1,
        output_dim=3,
        model_dim=16,
        modes=8,
        epochs=1,
        batch_size=4,
        seeds=(23,),
        device="cpu",
    )
    inputs = torch.randn(4, 17, 1)
    targets = torch.tensor([0, 1, 2, 1])
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        torch.manual_seed(23)
        model, metadata = build_ablation_model(variant, config, 3)
        logits = model(inputs)
        functional.cross_entropy(logits, targets).backward()
        if logits.shape != (4, 3) or not torch.isfinite(logits).all():
            raise RuntimeError(f"smoke failed for {variant}")
        rows.append(
            {
                "variant": variant,
                "logits_shape": list(logits.shape),
                "params_trainable": metadata.params_trainable,
                "parameter_delta_vs_full": metadata.parameter_delta_vs_full,
            }
        )
    return {"schema": "pac_compact_h_only_ablation_smoke.v1", "rows": rows}


def _result_path(root: Path, job: CompactAblationJob, *, failed: bool) -> Path:
    safe = job.key.replace(":", "_").replace("/", "_")
    return root / ("failed" if failed else "completed") / f"{safe}.json"


def _result_keys(directory: Path) -> set[str]:
    return set(_result_key_list(directory))


def _result_key_list(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return [
        str(json.loads(path.read_text(encoding="utf-8"))["job_key"])
        for path in directory.glob("*.json")
    ]


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("enqueue", "worker", "status", "report", "smoke"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=UCR_DATA_ROOT)
    arguments = parser.parse_args()
    if arguments.command == "enqueue":
        payload = enqueue(
            arguments.root,
            selection_path=arguments.selection,
            workers=arguments.workers,
        )
    elif arguments.command == "worker":
        if arguments.manifest is None:
            parser.error("worker requires --manifest")
        run_manifest(
            arguments.root,
            arguments.manifest,
            device=cast("PACDevice", arguments.device),
            data_root=arguments.data_root,
        )
        payload = status(arguments.root, selection_path=arguments.selection)
    elif arguments.command == "status":
        payload = status(arguments.root, selection_path=arguments.selection)
    elif arguments.command == "report":
        payload = report(arguments.root, selection_path=arguments.selection)
    else:
        payload = smoke()
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
