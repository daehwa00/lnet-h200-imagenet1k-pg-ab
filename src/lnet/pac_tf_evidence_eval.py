from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .hybrid_experiment_types import resolve_device
from .pac_eval_sections import clean_validation_classification_task
from .pac_headroom_efficient_models import WaveletPacketPAC
from .pac_headroom_models import HeadroomPACClassifier
from .pac_metrics import count_parameters, nrmse
from .pac_real_data import load_ucr_train_only
from .pac_stiefel_variants import REVISED_UNTIED_MODEL
from .pac_tasks import make_pac_tf_mechanism_tasks
from .pac_tf_evidence_queue import (
    EvidenceJob,
    build_sensitivity_classifier,
    core_variant,
    sensitivity_configuration,
)
from .pac_tf_mechanism_eval import pac_tf_recovery
from .pac_tight_frame_models import TightFrameClassifier, TightFrameSequenceRegressor
from .pac_training import (
    classification_metric_bundle,
    evaluate_regression_loss,
    train_classifier,
    train_regression_model,
)
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from .pac_types import PACRegressionTask
    from .tapped_prl_followup_schema import JsonRow

_MODEL_BUILD_LOCK = Lock()
_WP_MODEL = "WP"


def checkpoint_path(root: Path, key: str) -> Path:
    return root / "checkpoints" / f"{key}.pt"


def run_evidence_job(
    root: Path, config: PACExperimentConfig, job: EvidenceJob
) -> JsonRow:
    match job.kind:
        case "core_ablation" | "sensitivity":
            return _run_classification_training(root, config, job)
        case "mechanism_checkpoint":
            return _run_mechanism_checkpoint(root, config, job)
        case "interpretability":
            return _run_intervention(root, config, job)


def _run_classification_training(
    root: Path, config: PACExperimentConfig, job: EvidenceJob
) -> JsonRow:
    device = resolve_device(config.device)
    dataset = load_ucr_train_only(job.scope, Path(".omx/data/ucr"))
    task = clean_validation_classification_task(dataset, job.seed)
    run_config = replace(
        config,
        sequence_length=task.train_inputs.shape[1],
        raw_input_dim=1,
        output_dim=task.class_count,
        seeds=(job.seed,),
    )
    model = _seeded_classifier(job, run_config, task.class_count)
    started = perf_counter()
    outcome = train_classifier(
        model,
        task,
        run_config,
        device,
        job.seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    metrics = classification_metric_bundle(
        model,
        task.validation_inputs.to(device=device),
        task.validation_labels.to(device=device),
    )
    if job.kind == "core_ablation" and job.intervention == "reference":
        _save_checkpoint(root, job, model, run_config, task.class_count)
    return {
        "queue_key": job.key,
        "protocol_id": job.protocol_id,
        "protocol_sha256": job.protocol_sha256,
        "capacity_artifact_sha256": job.capacity_artifact_sha256,
        "selected_model": job.selected_model,
        "selected_model_dim": job.selected_model_dim,
        "selected_modes": job.selected_modes,
        "experiment_group": job.kind,
        "dataset_or_task": job.scope,
        "seed": job.seed,
        "model": job.model,
        "intervention": job.intervention,
        "level": job.level,
        "reference_level": job.reference_level,
        "evaluation_split": "validation",
        "official_test_read": False,
        "data_protocol": "raw_train_clean_stratified",
        "checkpoint_policy": "best_validation_loss",
        "validation_loss": outcome.validation_loss,
        "validation_accuracy": metrics.accuracy,
        "validation_macro_f1": metrics.macro_f1,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        "best_epoch": outcome.best_epoch,
        "params_trainable": count_parameters(model),
        "elapsed_total_time": perf_counter() - started,
        "status": "done",
    }


def _seeded_classifier(
    job: EvidenceJob, config: PACExperimentConfig, class_count: int
) -> nn.Module:
    with _MODEL_BUILD_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.seed)
        if job.model == _WP_MODEL:
            active_config = config
            variant = core_variant("revised_fixed_mean_nogate_untied")
            if job.kind == "sensitivity":
                active_config, variant = sensitivity_configuration(
                    job.intervention,
                    job.level,
                    config,
                    base_variant=variant,
                )
            return WaveletPacketPAC(
                active_config,
                class_count,
                objective="classification",
                pac_variant=variant,
            )
        if job.kind == "sensitivity":
            return build_sensitivity_classifier(
                job.intervention,
                job.level,
                config,
                class_count,
                revised=job.model == REVISED_UNTIED_MODEL,
            )
        if job.model == REVISED_UNTIED_MODEL:
            intervention = (
                "revised_fixed_mean_nogate_untied"
                if job.intervention == "reference"
                else job.intervention
            )
            return TightFrameClassifier(
                config,
                class_count,
                core_variant(intervention),
                full_modal_frame=config.modes > config.model_dim // 4,
            )
        return TightFrameClassifier(
            config,
            class_count,
            core_variant(job.intervention),
            full_modal_frame=config.modes > config.model_dim // 4,
        )


def _run_mechanism_checkpoint(
    root: Path, config: PACExperimentConfig, job: EvidenceJob
) -> JsonRow:
    device = resolve_device(config.device)
    task = next(
        task
        for task in make_pac_tf_mechanism_tasks(config, job.seed)
        if task.label == job.scope
    )
    if job.model == _WP_MODEL:
        task = _endpoint_mechanism_task(task)
    with _MODEL_BUILD_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.seed)
        model = _build_selected_mechanism_model(job.model, config)
    started = perf_counter()
    outcome = train_regression_model(model, task, config, device, job.seed)
    recovery = (
        _wp_frequency_recovery(model, task)
        if job.model == _WP_MODEL
        else pac_tf_recovery(model, task, device)
    )
    _save_checkpoint(root, job, model, config, config.output_dim)
    return {
        "queue_key": job.key,
        "protocol_id": job.protocol_id,
        "protocol_sha256": job.protocol_sha256,
        "capacity_artifact_sha256": job.capacity_artifact_sha256,
        "selected_model": job.selected_model,
        "selected_model_dim": job.selected_model_dim,
        "selected_modes": job.selected_modes,
        "experiment_group": job.kind,
        "dataset_or_task": job.scope,
        "seed": job.seed,
        "model": job.model,
        "validation_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "test_nrmse": nrmse(outcome.test_loss, task.test_targets),
        "params_trainable": count_parameters(model),
        "checkpoint": str(checkpoint_path(root, job.key)),
        "elapsed_total_time": perf_counter() - started,
        "status": "done",
        **recovery,
    }


def _save_checkpoint(
    root: Path,
    job: EvidenceJob,
    model: nn.Module,
    config: PACExperimentConfig,
    output_count: int,
) -> None:
    path = checkpoint_path(root, job.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "pac_tf_evidence_checkpoint.v2",
            "protocol_id": job.protocol_id,
            "protocol_sha256": job.protocol_sha256,
            "capacity_artifact_sha256": job.capacity_artifact_sha256,
            "selected_model": job.selected_model,
            "selected_model_dim": job.selected_model_dim,
            "selected_modes": job.selected_modes,
            "queue_key": job.key,
            "model": job.model,
            "scope": job.scope,
            "seed": job.seed,
            "output_count": output_count,
            "config": asdict(config),
            "state_dict": model.state_dict(),
        },
        path,
    )


def _build_selected_mechanism_model(
    name: str, config: PACExperimentConfig
) -> nn.Module:
    if name == _WP_MODEL:
        return WaveletPacketPAC(
            config,
            config.output_dim,
            objective="regression",
            pac_variant=core_variant("revised_fixed_mean_nogate_untied"),
        )
    intervention = {
        "pac_tf": "reference",
        "pac_tf_fixed_damping": "reference",
        "pac_tf_revised": "revised_fixed_mean_nogate_untied",
    }.get(name)
    if intervention is None:
        message = f"unsupported selected-capacity mechanism model: {name}"
        raise KeyError(message)
    model = TightFrameSequenceRegressor(config, core_variant(intervention))
    if name == "pac_tf_fixed_damping":
        model.block1.raw_decay.requires_grad_(requires_grad=False)
        model.block2.raw_decay.requires_grad_(requires_grad=False)
    return model


def _validate_checkpoint_provenance(source: Path, job: EvidenceJob) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    expected = {
        "schema_version": "pac_tf_evidence_checkpoint.v2",
        "protocol_id": job.protocol_id,
        "protocol_sha256": job.protocol_sha256,
        "capacity_artifact_sha256": job.capacity_artifact_sha256,
        "selected_model": job.selected_model,
        "selected_model_dim": job.selected_model_dim,
        "selected_modes": job.selected_modes,
        "queue_key": job.checkpoint_key,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            message = f"checkpoint provenance mismatch: {source}.{field}"
            raise ValueError(message)


def _recovery_control_metrics(
    trained_model: TightFrameSequenceRegressor,
    task: PACRegressionTask,
    config: PACExperimentConfig,
    job: EvidenceJob,
    device: str,
) -> JsonRow:
    trained = pac_tf_recovery(trained_model, task, device)
    with _MODEL_BUILD_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.seed)
        control = _build_selected_mechanism_model(job.model, config)
        if job.intervention == "random_grid_recovery":
            generator = torch.Generator(device="cpu").manual_seed(job.seed + 190_811)
            with torch.no_grad():
                for block in (control.block1, control.block2):
                    frequency = 0.95 * torch.rand(
                        block.raw_frequency.shape,
                        generator=generator,
                        dtype=block.raw_frequency.dtype,
                    )
                    block.raw_frequency.copy_(torch.atanh(frequency.clamp(max=0.999)))
                    block.raw_decay.copy_(
                        -4.0
                        + 6.0
                        * torch.rand(
                            block.raw_decay.shape,
                            generator=generator,
                            dtype=block.raw_decay.dtype,
                        )
                    )
    control.to(device=device).eval()
    control_metrics = pac_tf_recovery(control, task, device)
    control_loss = evaluate_regression_loss(
        control,
        task.test_inputs.to(device=device),
        task.test_targets.to(device=device),
    )
    return {
        "recovery_control": job.intervention,
        "control_test_loss": control_loss,
        "control_frequency_recovery_mae": control_metrics.get("frequency_recovery_mae"),
        "control_damping_recovery_mae": control_metrics.get("damping_recovery_mae"),
        "frequency_recovery_improvement_from_control": _mae_improvement(
            control_metrics.get("frequency_recovery_mae"),
            trained.get("frequency_recovery_mae"),
        ),
        "damping_recovery_improvement_from_control": _mae_improvement(
            control_metrics.get("damping_recovery_mae"),
            trained.get("damping_recovery_mae"),
        ),
        **trained,
    }


def _mae_improvement(control: object, trained: object) -> float | None:
    if not isinstance(control, int | float) or not isinstance(trained, int | float):
        return None
    return float(control) - float(trained)


def _run_intervention(
    root: Path, config: PACExperimentConfig, job: EvidenceJob
) -> JsonRow:
    source = checkpoint_path(root, job.checkpoint_key)
    if not source.exists():
        message = f"required checkpoint is missing: {source}"
        raise FileNotFoundError(message)
    _validate_checkpoint_provenance(source, job)
    if job.architecture_surface in {
        "causal_tight_frame_sequence_regressor",
        "wp_endpoint_regressor",
    }:
        return _mechanism_intervention(source, config, job)
    return _classifier_intervention(source, config, job)


def _mechanism_intervention(
    source: Path, config: PACExperimentConfig, job: EvidenceJob
) -> JsonRow:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    run_config = replace(config, **_config_values(payload["config"]))
    task = next(
        task
        for task in make_pac_tf_mechanism_tasks(run_config, job.seed)
        if task.label == job.scope
    )
    if job.model == _WP_MODEL:
        task = _endpoint_mechanism_task(task)
    model = _build_selected_mechanism_model(job.model, run_config)
    model.load_state_dict(payload["state_dict"])
    device = resolve_device(run_config.device)
    model.to(device=device).eval()
    baseline = evaluate_regression_loss(
        model,
        task.test_inputs.to(device=device),
        task.test_targets.to(device=device),
    )
    if job.intervention in {"teacher_frequency_damping_recovery", "teacher_frequency_recovery"}:
        metrics = (
            _wp_frequency_recovery(model, task)
            if job.model == _WP_MODEL
            else pac_tf_recovery(model, task, device)
        )
        intervention_loss = baseline
    elif job.intervention in {
        "untrained_initialization_recovery",
        "random_grid_recovery",
    }:
        metrics = _recovery_control_metrics(model, task, run_config, job, device)
        intervention_loss = baseline
    elif job.intervention == "frame_subspace_perturbation":
        with _mechanism_patch(model, job.intervention, 0, job.seed):
            intervention_loss = evaluate_regression_loss(
                model,
                task.test_inputs.to(device=device),
                task.test_targets.to(device=device),
            )
        metrics = {}
    else:
        mode = _intervention_mode(model, task, job)
        with _mechanism_patch(model, job.intervention, mode, job.seed):
            intervention_loss = evaluate_regression_loss(
                model,
                task.test_inputs.to(device=device),
                task.test_targets.to(device=device),
            )
        metrics = {
            "intervened_mode": mode,
            "teacher_mode_index": job.teacher_mode_index,
            "teacher_frequency": (
                task.true_frequencies[job.teacher_mode_index]
                if job.teacher_mode_index is not None
                else None
            ),
        }
    return {
        "queue_key": job.key,
        "protocol_id": job.protocol_id,
        "protocol_sha256": job.protocol_sha256,
        "capacity_artifact_sha256": job.capacity_artifact_sha256,
        "selected_model": job.selected_model,
        "selected_model_dim": job.selected_model_dim,
        "selected_modes": job.selected_modes,
        "experiment_group": "interpretability",
        "architecture_surface": job.architecture_surface,
        "dataset_or_task": job.scope,
        "seed": job.seed,
        "model": job.model,
        "intervention": job.intervention,
        "baseline_metric": baseline,
        "intervention_metric": intervention_loss,
        "metric_delta": intervention_loss - baseline,
        "checkpoint": str(source),
        "applicable": True,
        "status": "done",
        **metrics,
    }


def _intervention_mode(
    model: nn.Module,
    task: PACRegressionTask,
    job: EvidenceJob,
) -> int:
    frequencies = _frequency_values(model).detach().cpu().abs()
    truths = task.true_frequencies
    teacher_index = job.teacher_mode_index
    if teacher_index is None or teacher_index < 0 or teacher_index >= len(truths):
        message = f"invalid teacher mode index for {job.scope}: {teacher_index}"
        raise ValueError(message)
    target = float(truths[teacher_index])
    matched = int(torch.argmin(torch.abs(frequencies - abs(target))).item())
    if job.intervention != "random_mode_knockout" or frequencies.numel() == 1:
        return matched
    generator = torch.Generator().manual_seed(job.seed + 88_171)
    candidates = torch.tensor(
        [index for index in range(frequencies.numel()) if index != matched]
    )
    return int(candidates[torch.randint(candidates.numel(), (), generator=generator)].item())


@contextmanager
def _mechanism_patch(
    model: nn.Module,
    intervention: str,
    mode: int,
    seed: int,
) -> Generator[None]:
    originals: list[Tensor | None] = []
    blocks = _mechanism_blocks(model)
    for block in blocks:
        originals.append(block.intervention_frame())
        frame = block.frame_matrix().detach()
        if intervention in {"matched_mode_knockout", "random_mode_knockout"}:
            override = frame.clone()
            override[:, mode] = 0
            override[:, mode + block.raw_frequency.numel()] = 0
        elif intervention == "frame_subspace_perturbation":
            override = _perturbed_frame(frame, seed)
        else:
            override = frame
        block.set_intervention_frame(override)
    try:
        yield
    finally:
        for block, original in zip(blocks, originals, strict=True):
            block.set_intervention_frame(original)


def _perturbed_frame(frame: Tensor, seed: int, angle_degrees: float = 30.0) -> Tensor:
    generator = torch.Generator(device=frame.device).manual_seed(seed + 71_911)
    random = torch.randn(frame.shape, generator=generator, device=frame.device, dtype=frame.dtype)
    residual = random - frame @ (frame.transpose(0, 1) @ random)
    complement, _ = torch.linalg.qr(residual, mode="reduced")
    angle = torch.tensor(angle_degrees * torch.pi / 180.0, device=frame.device)
    return torch.cos(angle) * frame + torch.sin(angle) * complement


def _classifier_intervention(
    source: Path, config: PACExperimentConfig, job: EvidenceJob
) -> JsonRow:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    run_config = replace(config, **_config_values(payload["config"]))
    dataset = load_ucr_train_only(job.scope, Path(".omx/data/ucr"))
    task = clean_validation_classification_task(dataset, job.seed)
    if job.model == _WP_MODEL:
        model: nn.Module = WaveletPacketPAC(
            run_config,
            task.class_count,
            objective="classification",
            pac_variant=core_variant("revised_fixed_mean_nogate_untied"),
        )
    else:
        model = TightFrameClassifier(
            run_config,
            task.class_count,
            core_variant(
                "revised_fixed_mean_nogate_untied"
                if job.model == REVISED_UNTIED_MODEL
                else "reference"
            ),
            full_modal_frame=run_config.modes > run_config.model_dim // 4,
        )
    model.load_state_dict(payload["state_dict"])
    device = resolve_device(run_config.device)
    model.to(device=device).eval()
    inputs = task.validation_inputs.to(device=device)
    labels = task.validation_labels.to(device=device)
    baseline = classification_metric_bundle(model, inputs, labels).balanced_accuracy
    with _classifier_patch(model, job.intervention):
        intervened = classification_metric_bundle(model, inputs, labels).balanced_accuracy
    return {
        "queue_key": job.key,
        "protocol_id": job.protocol_id,
        "protocol_sha256": job.protocol_sha256,
        "capacity_artifact_sha256": job.capacity_artifact_sha256,
        "selected_model": job.selected_model,
        "selected_model_dim": job.selected_model_dim,
        "selected_modes": job.selected_modes,
        "experiment_group": "interpretability",
        "architecture_surface": job.architecture_surface,
        "dataset_or_task": job.scope,
        "seed": job.seed,
        "model": job.model,
        "intervention": job.intervention,
        "baseline_metric": baseline,
        "intervention_metric": intervened,
        "metric_delta": intervened - baseline,
        "evaluation_split": "validation",
        "official_test_read": False,
        "checkpoint": str(source),
        "applicable": True,
        "status": "done",
    }


@contextmanager
def _classifier_patch(
    model: nn.Module, intervention: str
) -> Generator[None]:
    if isinstance(model, WaveletPacketPAC) and intervention in {
        "low_band_removal",
        "detail_band_removal",
        "uniform_band_fusion",
    }:
        original = model.band_logits.detach().clone()
        with torch.no_grad():
            if intervention == "low_band_removal":
                model.band_logits.copy_(model.band_logits.new_tensor((-20.0, 0.0)))
            elif intervention == "detail_band_removal":
                model.band_logits.copy_(model.band_logits.new_tensor((0.0, -20.0)))
            else:
                model.band_logits.zero_()
        try:
            yield
        finally:
            with torch.no_grad():
                model.band_logits.copy_(original)
        return
    if not isinstance(model, HeadroomPACClassifier | TightFrameClassifier):
        message = "classifier intervention requires a PAC classifier"
        raise TypeError(message)
    direction_block = {
        "forward_direction_removal": model.forward_block,
        "backward_direction_removal": model.backward_block,
    }.get(intervention)
    if direction_block is not None:
        handle = direction_block.register_forward_hook(
            lambda _module, args, output: (args[0], torch.zeros_like(output[1]))
        )
        try:
            yield
        finally:
            handle.remove()
        return
    weight = model.head.classifier.weight
    original = weight.detach().clone()
    columns = _moment_columns(model, intervention)
    try:
        with torch.no_grad():
            weight[:, columns] = 0
        yield
    finally:
        with torch.no_grad():
            weight.copy_(original)


def _moment_columns(model: nn.Module, intervention: str) -> Tensor:
    if isinstance(model, HeadroomPACClassifier):
        pooled = model.model_dim
    elif isinstance(model, TightFrameClassifier):
        pooled = (
            sum(model.pooling_scales) * model.model_dim
            if model.use_ordered_pool
            else model.model_dim
        )
    else:
        message = "moment intervention requires a PAC classifier"
        raise TypeError(message)
    modes = model.modes
    per_copy = modes * 5
    offsets = (pooled, pooled + per_copy)
    if intervention == "moment_head_intervention":
        ranges = [(offset, offset + per_copy) for offset in offsets]
    elif intervention == "lag1_intervention":
        ranges = [(offset + modes, offset + 3 * modes) for offset in offsets]
    elif intervention == "lag4_intervention":
        ranges = [(offset + 3 * modes, offset + 5 * modes) for offset in offsets]
    else:
        message = f"unsupported classifier intervention: {intervention}"
        raise KeyError(message)
    return torch.cat([torch.arange(start, stop) for start, stop in ranges])


def _endpoint_mechanism_task(task: PACRegressionTask) -> PACRegressionTask:
    return type(task)(
        task.label + "_endpoint",
        task.train_inputs,
        task.train_targets[:, -1],
        task.validation_inputs,
        task.validation_targets[:, -1],
        task.test_inputs,
        task.test_targets[:, -1],
        true_delay=task.true_delay,
        true_frequency=task.true_frequency,
        true_frequencies=task.true_frequencies,
        true_dampings=task.true_dampings,
        mechanism_expectation=task.mechanism_expectation,
    )


def _frequency_values(model: nn.Module) -> Tensor:
    if isinstance(model, WaveletPacketPAC):
        return model.forward_block.frequency_values()
    if isinstance(model, TightFrameSequenceRegressor):
        return model.frequency_values()
    message = "frequency recovery requires a PAC mechanism model"
    raise TypeError(message)


def _wp_frequency_recovery(model: nn.Module, task: PACRegressionTask) -> JsonRow:
    from .pac_tf_mechanism_eval import frequency_recovery  # noqa: PLC0415

    frequencies = _frequency_values(model).detach().cpu()
    error, _ = frequency_recovery(frequencies, task.true_frequencies)
    scale = (
        sum(abs(value) for value in task.true_frequencies) / len(task.true_frequencies)
        if task.true_frequencies
        else None
    )
    return {
        "frequency_recovery_mae": error,
        "frequency_recovery_relative_mae": (
            error / max(scale, 1.0e-12) if error is not None and scale is not None else None
        ),
        "learned_frequency_count": int(frequencies.numel()),
    }


def _mechanism_blocks(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    if isinstance(model, WaveletPacketPAC):
        return model.forward_block, model.backward_block
    if isinstance(model, TightFrameSequenceRegressor):
        return model.block1, model.block2
    message = "mechanism intervention requires a PAC mechanism model"
    raise TypeError(message)


def _config_values(payload: dict[str, object]) -> dict[str, object]:
    allowed = {field.name for field in PACExperimentConfig.__dataclass_fields__.values()}
    runtime_owned = {"device", "output_dir"}
    return {key: value for key, value in payload.items() if key in allowed - runtime_owned}
