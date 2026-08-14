"""Preregistered PLAsTiCC three-class Phase-0 training harness."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader

from lnet.alphabet import Alphabet
from lnet.astronomy.damped_spectrum import DampedSpectrumClassifier
from lnet.astronomy.plasticc import (
    LightCurveBatch,
    PlasticcDataset,
    collate_light_curves,
)
from lnet.astronomy.poles import (
    configure_astronomy_impulse_poles,
    configure_astronomy_poles,
)
from lnet.pac_irregular_models import GRUDClassifier
from lnet.pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

ModelName = Literal["alphabet", "gru", "grud", "dls"]
LagMode = Literal["physical", "token", "energy"]
InjectionMode = Literal["zoh", "impulse"]


@dataclass(frozen=True, slots=True)
class Phase0RunConfig:
    model: ModelName
    seed: int
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 8
    model_dim: int = 64
    modes: int = 16
    classes: int = 3
    lag_mode: LagMode = "physical"
    injection_mode: InjectionMode = "zoh"
    near_undamped_modes: int = 0
    near_undamped_alpha_per_day: float = 1.0e-6
    point_sample_local_convolution: bool = False
    freeze_spectrum_frequencies: bool = False
    class_weights: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    loss: float
    weighted_log_loss: float
    balanced_accuracy: float
    macro_f1: float
    expected_calibration_error: float


class DeltaTimeGRU(nn.Module):
    """GRU control receiving raw flux, masks, and log time intervals."""

    def __init__(self, model_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(13, model_dim, batch_first=True)
        self.classifier = nn.Linear(model_dim, output_dim)

    def forward(
        self,
        flux: Tensor,
        *,
        time_delta: Tensor,
        observation_mask: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        inputs = torch.cat((flux, observation_mask, torch.log1p(time_delta)), dim=-1)
        states, _ = self.gru(inputs)
        lengths = valid_mask.squeeze(-1).sum(dim=1).long().clamp_min(1)
        final = states[torch.arange(states.shape[0], device=states.device), lengths - 1]
        return self.classifier(final)


class DeltaTimeGRUD(nn.Module):
    """GRU-D-like control using one inter-epoch decay interval for all bands.

    This is not a faithful feature-delta GRU-D: missing-band intervals do not
    accumulate independently. The legacy ``grud`` CLI name is retained for
    frozen-result compatibility.
    """

    def __init__(self, model_dim: int, output_dim: int) -> None:
        super().__init__()
        self.model = GRUDClassifier(6, model_dim, output_dim)

    def set_feature_mean(self, mean: Tensor) -> None:
        self.model.set_feature_mean(mean)

    def forward(
        self,
        flux: Tensor,
        *,
        time_delta: Tensor,
        observation_mask: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        return self.model(
            flux,
            observation_mask,
            time_delta.expand_as(flux),
            valid_mask,
        )


class EnergyOnlyHead(nn.Module):
    """Affine control restricted to the R0 coordinate of both pole banks."""

    def __init__(self, modes: int, output_dim: int) -> None:
        super().__init__()
        self.modes = modes
        self.mode_map = None
        self.classifier = nn.Linear(2 * modes, output_dim)

    def forward(self, writer_moments: Tensor, reader_moments: Tensor) -> Tensor:
        energy = torch.cat(
            (writer_moments[:, : self.modes], reader_moments[:, : self.modes]),
            dim=-1,
        )
        return self.classifier(energy)


def build_model(config: Phase0RunConfig, sequence_length: int) -> nn.Module:
    if config.model == "gru":
        return DeltaTimeGRU(config.model_dim, config.classes)
    if config.model == "grud":
        return DeltaTimeGRUD(config.model_dim, config.classes)
    if config.model == "dls":
        return DampedSpectrumClassifier(
            6,
            config.modes,
            config.classes,
            near_undamped_modes=config.near_undamped_modes,
            near_undamped_alpha_per_day=config.near_undamped_alpha_per_day,
            freeze_frequencies=config.freeze_spectrum_frequencies,
        )
    experiment = PACExperimentConfig(
        sample_count=1,
        validation_count=1,
        test_count=1,
        sequence_length=sequence_length,
        raw_input_dim=6,
        output_dim=config.classes,
        model_dim=config.model_dim,
        modes=config.modes,
    )
    model = Alphabet(experiment, config.classes)
    if config.injection_mode == "impulse":
        model = configure_astronomy_impulse_poles(
            model,
            near_undamped_modes=config.near_undamped_modes,
            near_undamped_alpha_per_day=config.near_undamped_alpha_per_day,
            point_sample_local_convolution=config.point_sample_local_convolution,
        )
    else:
        model = configure_astronomy_poles(model)
    if config.lag_mode == "token":
        for block in (model.forward_block, model.backward_block):
            block.physical_time_lag_moments = False
    elif config.lag_mode == "energy":
        model.head = EnergyOnlyHead(  # pyright: ignore[reportAttributeAccessIssue]
            config.modes,
            config.classes,
        )
    return model


def _move(batch: LightCurveBatch, device: torch.device) -> LightCurveBatch:
    return LightCurveBatch(
        flux=batch.flux.to(device, non_blocking=True),
        time_delta=batch.time_delta.to(device, non_blocking=True),
        observation_mask=batch.observation_mask.to(device, non_blocking=True),
        valid_mask=batch.valid_mask.to(device, non_blocking=True),
        target=batch.target.to(device, non_blocking=True),
        object_id=batch.object_id.to(device, non_blocking=True),
    )


def _forward(model: nn.Module, batch: LightCurveBatch) -> Tensor:
    return model(
        batch.flux,
        time_delta=batch.time_delta,
        observation_mask=batch.observation_mask,
        valid_mask=batch.valid_mask,
    )


def _set_grud_training_mean(
    model: DeltaTimeGRUD,
    train_dataset: PlasticcDataset,
    device: torch.device,
) -> None:
    observed_sum = torch.zeros(6)
    observed_count = torch.zeros(6)
    for object_id in train_dataset.object_ids:
        curve = train_dataset.curves[object_id]
        observed = torch.from_numpy(curve.observation_mask)
        observed_sum += (torch.from_numpy(curve.flux) * observed).sum(dim=0)
        observed_count += observed.sum(dim=0)
    model.set_feature_mean((observed_sum / observed_count.clamp_min(1.0)).to(device))


def evaluate(
    model: nn.Module,
    loader: Iterable[LightCurveBatch],
    device: torch.device,
    classes: int,
    class_weights: Tensor | None = None,
) -> ClassificationMetrics:
    model.eval()
    loss_sum = 0.0
    count = 0
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    class_loss_sum = torch.zeros(classes, dtype=torch.float64)
    class_count = torch.zeros(classes, dtype=torch.long)
    confidence_chunks: list[Tensor] = []
    correctness_chunks: list[Tensor] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move(cpu_batch, device)
            logits = _forward(model, batch)
            batch_loss = functional.cross_entropy(logits, batch.target, reduction="sum")
            loss_sum += float(batch_loss.item())
            count += batch.target.numel()
            prediction = logits.argmax(dim=-1).cpu()
            probability = torch.softmax(logits, dim=-1)
            confidence, _ = probability.max(dim=-1)
            confidence_chunks.append(confidence.cpu())
            correctness_chunks.append(prediction.eq(batch.target.cpu()))
            per_example_loss = functional.cross_entropy(
                logits,
                batch.target,
                reduction="none",
            ).cpu()
            for truth, predicted, example_loss in zip(
                batch.target.cpu(),
                prediction,
                per_example_loss,
                strict=True,
            ):
                confusion[int(truth), int(predicted)] += 1
                class_loss_sum[int(truth)] += float(example_loss)
                class_count[int(truth)] += 1
    recall = confusion.diag() / confusion.sum(dim=1).clamp_min(1)
    precision = confusion.diag() / confusion.sum(dim=0).clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float32).eps)
    weights = (
        torch.ones(classes, dtype=torch.float64)
        if class_weights is None
        else class_weights.detach().cpu().to(torch.float64)
    )
    present = class_count > 0
    weighted_log_loss = (
        weights[present] * class_loss_sum[present] / class_count[present]
    ).sum() / weights[present].sum()
    confidence = torch.cat(confidence_chunks)
    correctness = torch.cat(correctness_chunks)
    calibration_error = 0.0
    for bin_index in range(15):
        lower = bin_index / 15.0
        upper = (bin_index + 1) / 15.0
        in_bin = (confidence > lower) & (
            confidence <= upper if bin_index < 14 else confidence <= 1.0
        )
        if in_bin.any():
            calibration_error += float(in_bin.float().mean()) * abs(
                float(correctness[in_bin].float().mean())
                - float(confidence[in_bin].mean())
            )
    return ClassificationMetrics(
        loss_sum / count,
        float(weighted_log_loss),
        float(recall.mean()),
        float(f1.mean()),
        calibration_error,
    )


def train_one_seed(
    config: Phase0RunConfig,
    train_dataset: PlasticcDataset,
    validation_dataset: PlasticcDataset,
    test_dataset: PlasticcDataset,
    output_dir: Path,
    *,
    device: torch.device,
    evaluation_split: str = "test",
) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_light_curves,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_light_curves,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_light_curves,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    maximum_length = max(curve.flux.shape[0] for curve in train_dataset.curves.values())
    model = build_model(config, maximum_length).to(device)
    if isinstance(model, DeltaTimeGRUD):
        _set_grud_training_mean(model, train_dataset, device)
    class_weights = (
        torch.tensor(config.class_weights, dtype=torch.float32, device=device)
        if config.class_weights
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{config.model}-seed{config.seed}.pt"
    for epoch in range(1, config.epochs + 1):
        model.train()
        training_loss = 0.0
        training_count = 0
        for cpu_batch in train_loader:
            batch = _move(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch)
            loss = functional.cross_entropy(logits, batch.target, weight=class_weights)
            if not torch.isfinite(loss):
                message = f"non-finite training loss at epoch {epoch}"
                raise FloatingPointError(message)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss += float(loss.item()) * batch.target.numel()
            training_count += batch.target.numel()
        validation = evaluate(
            model,
            validation_loader,
            device,
            config.classes,
            class_weights,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss / training_count,
                "validation_loss": validation.loss,
                "validation_weighted_log_loss": validation.weighted_log_loss,
                "validation_balanced_accuracy": validation.balanced_accuracy,
                "validation_macro_f1": validation.macro_f1,
            }
        )
        if (
            selection_loss := (
                validation.weighted_log_loss
                if class_weights is not None
                else validation.loss
            )
        ) < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, device, config.classes, class_weights)
    result: dict[str, object] = {
        "config": asdict(config),
        "best_epoch": best_epoch,
        "history": history,
        "test": asdict(test),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "evaluation_split": evaluation_split,
        "checkpoint_selection_metric": (
            "weighted_log_loss" if class_weights is not None else "mean_cross_entropy"
        ),
    }
    result_path = output_dir / f"{config.model}-seed{config.seed}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
