"""Training and evaluation backend for the fixed Raindrop P19/PAM splits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet import Alphabet
from .pac_irregular_models import (
    GRUDClassifier,
    LatentODEClassifier,
    MTANClassifier,
    NeuralCDEClassifier,
    ODERNNClassifier,
    RaindropClassifier,
)
from .pac_physionet2012 import binary_metrics

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .pac_irregular_data import IrregularSplit, IrregularTask
    from .pac_types import PACExperimentConfig

IrregularModelName = Literal[
    "alphabet",
    "gru-d",
    "ode-rnn",
    "latent-ode",
    "mtan",
    "neural-cde",
    "raindrop",
]


@dataclass(frozen=True, slots=True)
class IrregularMetrics:
    accuracy: float
    macro_f1: float
    auroc: float | None = None
    auprc: float | None = None


@dataclass(frozen=True, slots=True)
class FitResult:
    best_epoch: int
    train_loss: float
    validation: IrregularMetrics | None


def _macro_f1(predictions: Tensor, labels: Tensor, classes: int) -> float:
    scores: list[float] = []
    for label in range(classes):
        predicted = predictions == label
        actual = labels == label
        true_positive = int((predicted & actual).sum())
        false_positive = int((predicted & ~actual).sum())
        false_negative = int((~predicted & actual).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return sum(scores) / classes


def classification_metrics(probabilities: Tensor, labels: Tensor) -> IrregularMetrics:
    probabilities = probabilities.detach().cpu()
    labels = labels.detach().cpu()
    predictions = probabilities.argmax(dim=-1)
    classes = probabilities.shape[-1]
    accuracy = float((predictions == labels).to(torch.float32).mean())
    macro_f1 = _macro_f1(predictions, labels, classes)
    if classes != 2:
        return IrregularMetrics(accuracy=accuracy, macro_f1=macro_f1)
    binary = binary_metrics(probabilities[:, 1], labels)
    return IrregularMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        auroc=binary.auroc,
        auprc=binary.auprc,
    )


def selection_scores(metrics: IrregularMetrics, output_dim: int) -> tuple[float, float]:
    if output_dim == 2:
        return cast("float", metrics.auprc), cast("float", metrics.auroc)
    return metrics.macro_f1, metrics.accuracy


def _moments(
    splits: tuple[IrregularSplit, ...],
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    feature_sum = torch.zeros(splits[0].values.shape[-1], dtype=torch.float64)
    feature_square = torch.zeros_like(feature_sum)
    feature_count = torch.zeros_like(feature_sum)
    static_sum: Tensor | None = None
    static_square: Tensor | None = None
    static_count = 0
    for split in splits:
        mask = split.observed.to(torch.float64)
        values = split.values.to(torch.float64)
        feature_sum += (values * mask).sum(dim=(0, 1))
        feature_square += (values.square() * mask).sum(dim=(0, 1))
        feature_count += mask.sum(dim=(0, 1))
        if split.static is not None:
            static = split.static.to(torch.float64)
            if static_sum is None:
                static_sum = torch.zeros(static.shape[-1], dtype=torch.float64)
                static_square = torch.zeros_like(static_sum)
            static_square = cast("Tensor", static_square)
            static_sum += static.sum(dim=0)
            static_square += static.square().sum(dim=0)
            static_count += static.shape[0]
    mean = feature_sum / feature_count.clamp_min(1)
    variance = feature_square / feature_count.clamp_min(1) - mean.square()
    scale = variance.clamp_min(1.0e-12).sqrt().clamp_min(1.0e-6)
    if static_sum is None or static_square is None:
        return mean.float(), scale.float(), None, None
    static_mean = static_sum / max(static_count, 1)
    static_variance = static_square / max(static_count, 1) - static_mean.square()
    static_scale = static_variance.clamp_min(1.0e-12).sqrt().clamp_min(1.0e-6)
    return mean.float(), scale.float(), static_mean.float(), static_scale.float()


def normalization_moments(
    splits: tuple[IrregularSplit, ...],
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Return train-only value and static normalization statistics."""
    return _moments(splits)


class IrregularAlphabet(nn.Module):
    """Final Alphabet with an additive affine path for dataset static covariates."""

    static_head: nn.Linear | None

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        feature_mean: Tensor,
        feature_scale: Tensor,
        static_mean: Tensor | None,
        static_scale: Tensor | None,
    ) -> None:
        super().__init__()
        self.core = Alphabet(
            replace(config, raw_input_dim=feature_mean.numel()),
            output_dim,
            objective="classification",
        )
        self.register_buffer("_feature_mean", feature_mean)
        self.register_buffer("_feature_scale", feature_scale)
        self.register_buffer(
            "_static_mean",
            torch.empty(0) if static_mean is None else static_mean,
        )
        self.register_buffer(
            "_static_scale",
            torch.empty(0) if static_scale is None else static_scale,
        )
        self.static_head = (
            None if static_mean is None else nn.Linear(static_mean.numel(), output_dim, bias=False)
        )

    @property
    def feature_mean(self) -> Tensor:
        return self.get_buffer("_feature_mean")

    @property
    def feature_scale(self) -> Tensor:
        return self.get_buffer("_feature_scale")

    @property
    def static_mean(self) -> Tensor:
        return self.get_buffer("_static_mean")

    @property
    def static_scale(self) -> Tensor:
        return self.get_buffer("_static_scale")

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        interval_delta: Tensor,
        _feature_delta: Tensor,
        valid: Tensor,
        static: Tensor | None,
    ) -> Tensor:
        standardized = (values - self.feature_mean) / self.feature_scale
        standardized = standardized * observed.to(standardized.dtype)
        logits = self.core(
            standardized,
            time_delta=interval_delta,
            observation_mask=observed,
            valid_mask=valid,
        )
        if self.static_head is not None:
            if static is None:
                message = "static covariates are required by this Alphabet instance"
                raise ValueError(message)
            normalized_static = (static - self.static_mean) / self.static_scale
            logits = logits + self.static_head(normalized_static)
        return logits

    def post_optimizer_step(self) -> None:
        self.core.post_optimizer_step()

    def finalize_constraints(self) -> None:
        self.core.finalize_constraints()


class IrregularGRUD(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        output_dim: int,
        *,
        depth: int,
        feature_mean: Tensor,
        feature_scale: Tensor,
        static_mean: Tensor | None,
        static_scale: Tensor | None,
    ) -> None:
        super().__init__()
        self.register_buffer("_feature_mean", feature_mean)
        self.register_buffer("_feature_scale", feature_scale)
        self.register_buffer(
            "_static_mean",
            torch.empty(0) if static_mean is None else static_mean,
        )
        self.register_buffer(
            "_static_scale",
            torch.empty(0) if static_scale is None else static_scale,
        )
        self.core = GRUDClassifier(
            input_dim,
            width,
            output_dim,
            static_dim=0 if static_mean is None else static_mean.numel(),
            depth=depth,
        )

    @property
    def feature_mean(self) -> Tensor:
        return self.get_buffer("_feature_mean")

    @property
    def feature_scale(self) -> Tensor:
        return self.get_buffer("_feature_scale")

    @property
    def static_mean(self) -> Tensor:
        return self.get_buffer("_static_mean")

    @property
    def static_scale(self) -> Tensor:
        return self.get_buffer("_static_scale")

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        _interval_delta: Tensor,
        feature_delta: Tensor,
        valid: Tensor,
        static: Tensor | None,
    ) -> Tensor:
        standardized = (values - self.feature_mean) / self.feature_scale
        standardized = standardized * observed.to(standardized.dtype)
        normalized_static = (
            None
            if static is None
            else (static - self.static_mean) / self.static_scale
        )
        return self.core(
            standardized,
            observed,
            feature_delta,
            valid,
            normalized_static,
        )


class IrregularNormalizedBaseline(nn.Module):
    """Apply train-only normalization before a common irregular model core."""

    def __init__(
        self,
        core: nn.Module,
        feature_mean: Tensor,
        feature_scale: Tensor,
        static_mean: Tensor | None,
        static_scale: Tensor | None,
    ) -> None:
        super().__init__()
        self.core = core
        self.register_buffer("_feature_mean", feature_mean)
        self.register_buffer("_feature_scale", feature_scale)
        self.register_buffer(
            "_static_mean",
            torch.empty(0) if static_mean is None else static_mean,
        )
        self.register_buffer(
            "_static_scale",
            torch.empty(0) if static_scale is None else static_scale,
        )

    @property
    def feature_mean(self) -> Tensor:
        return self.get_buffer("_feature_mean")

    @property
    def feature_scale(self) -> Tensor:
        return self.get_buffer("_feature_scale")

    @property
    def static_mean(self) -> Tensor:
        return self.get_buffer("_static_mean")

    @property
    def static_scale(self) -> Tensor:
        return self.get_buffer("_static_scale")

    @property
    def auxiliary_loss(self) -> object:
        return getattr(self.core, "auxiliary_loss", None)

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        interval_delta: Tensor,
        feature_delta: Tensor,
        valid: Tensor,
        static: Tensor | None,
    ) -> Tensor:
        standardized = (values - self.feature_mean) / self.feature_scale
        standardized = standardized * observed.to(standardized.dtype)
        normalized_static = (
            None
            if static is None
            else (static - self.static_mean) / self.static_scale
        )
        return self.core(
            standardized,
            observed,
            interval_delta,
            feature_delta,
            valid,
            normalized_static,
        )


class IrregularPackedSequenceBaseline(nn.Module):
    """Expose equal irregular metadata to a standard sequence-model baseline."""

    def __init__(
        self,
        core: nn.Module,
        output_dim: int,
        feature_mean: Tensor,
        feature_scale: Tensor,
        static_mean: Tensor | None,
        static_scale: Tensor | None,
    ) -> None:
        super().__init__()
        self.core = core
        self.register_buffer("_feature_mean", feature_mean)
        self.register_buffer("_feature_scale", feature_scale)
        self.register_buffer(
            "_static_mean",
            torch.empty(0) if static_mean is None else static_mean,
        )
        self.register_buffer(
            "_static_scale",
            torch.empty(0) if static_scale is None else static_scale,
        )
        self.static_head = (
            None
            if static_mean is None
            else nn.Linear(static_mean.numel(), output_dim, bias=False)
        )

    @property
    def feature_mean(self) -> Tensor:
        return self.get_buffer("_feature_mean")

    @property
    def feature_scale(self) -> Tensor:
        return self.get_buffer("_feature_scale")

    @property
    def static_mean(self) -> Tensor:
        return self.get_buffer("_static_mean")

    @property
    def static_scale(self) -> Tensor:
        return self.get_buffer("_static_scale")

    @property
    def auxiliary_loss(self) -> object:
        return getattr(self.core, "auxiliary_loss", None)

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        interval_delta: Tensor,
        _feature_delta: Tensor,
        valid: Tensor,
        static: Tensor | None,
    ) -> Tensor:
        valid_values = valid.to(values.dtype)
        observed_values = observed.to(values.dtype) * valid_values
        standardized = (values - self.feature_mean) / self.feature_scale
        packed = torch.cat(
            (
                standardized * observed_values,
                observed_values,
                interval_delta.to(values.dtype) * valid_values,
                valid_values,
            ),
            dim=-1,
        )
        logits = self.core(packed)
        if self.static_head is not None:
            if static is None:
                message = "static covariates are required by this baseline"
                raise ValueError(message)
            normalized_static = (static - self.static_mean) / self.static_scale
            logits = logits + self.static_head(normalized_static)
        return logits


def build_model(
    name: IrregularModelName,
    task: IrregularTask,
    config: PACExperimentConfig,
    *,
    width: int,
    depth: int,
    architecture: str | None = None,
    fit_splits: tuple[IrregularSplit, ...],
    seed: int,
) -> nn.Module:
    feature_mean, feature_scale, static_mean, static_scale = _moments(fit_splits)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name == "alphabet":
            return IrregularAlphabet(
                config,
                task.output_dim,
                feature_mean,
                feature_scale,
                static_mean,
                static_scale,
            )
        input_dim = task.train.values.shape[-1]
        static_dim = 0 if static_mean is None else static_mean.numel()
        if name == "gru-d":
            return IrregularGRUD(
                input_dim,
                width,
                task.output_dim,
                depth=depth,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                static_mean=static_mean,
                static_scale=static_scale,
            )
        if architecture is None:
            message = f"{name} requires an architecture variant"
            raise ValueError(message)
        if name == "ode-rnn":
            core: nn.Module = ODERNNClassifier(
                input_dim,
                width,
                task.output_dim,
                solver=architecture,
                static_dim=static_dim,
            )
        elif name == "latent-ode":
            core = LatentODEClassifier(
                input_dim,
                width,
                task.output_dim,
                encoder=architecture,
                static_dim=static_dim,
            )
        elif name == "mtan":
            core = MTANClassifier(
                input_dim,
                width,
                task.output_dim,
                heads={"h2": 2, "h4": 4}[architecture],
                static_dim=static_dim,
            )
        elif name == "neural-cde":
            core = NeuralCDEClassifier(
                input_dim,
                width,
                task.output_dim,
                interpolation=architecture,
                static_dim=static_dim,
            )
        elif name == "raindrop":
            core = RaindropClassifier(
                input_dim,
                width,
                task.output_dim,
                propagation_layers={"layer1": 1, "layer2": 2}[architecture],
                static_dim=static_dim,
            )
        else:
            raise AssertionError(name)
        return IrregularNormalizedBaseline(
            core,
            feature_mean,
            feature_scale,
            static_mean,
            static_scale,
        )


def _batch(
    split: IrregularSplit,
    indices: Tensor,
    device: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]:
    return (
        split.values[indices].to(device),
        split.observed[indices].to(device),
        split.interval_delta[indices].to(device),
        split.time_delta[indices].to(device),
        split.valid[indices].to(device),
        None if split.static is None else split.static[indices].to(device),
        split.labels[indices].to(device),
    )


def _indices(
    splits: tuple[IrregularSplit, ...],
    generator: torch.Generator,
) -> Tensor:
    rows = torch.cat(
        [
            torch.stack(
                (
                    torch.full((split.labels.numel(),), index, dtype=torch.long),
                    torch.arange(split.labels.numel()),
                ),
                dim=-1,
            )
            for index, split in enumerate(splits)
        ]
    )
    return rows[torch.randperm(rows.shape[0], generator=generator)]


def _iter_batches(
    splits: tuple[IrregularSplit, ...],
    order: Tensor,
    batch_size: int,
    device: str,
) -> Iterable[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]]:
    for rows in order.split(batch_size):
        parts = [
            _batch(
                splits[int(index)],
                rows[rows[:, 0] == index, 1],
                device,
            )
            for index in rows[:, 0].unique()
        ]
        columns: list[Tensor | None] = []
        for position in range(7):
            values = [part[position] for part in parts]
            if values[0] is None:
                columns.append(None)
            else:
                columns.append(torch.cat(cast("list[Tensor]", values)))
        yield cast(
            "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]",
            tuple(columns),
        )


def predict(
    model: nn.Module,
    split: IrregularSplit,
    *,
    batch_size: int,
    device: str,
) -> Tensor:
    was_training = model.training
    model.eval()
    rows: list[Tensor] = []
    with torch.no_grad():
        for indices in torch.arange(split.labels.numel()).split(batch_size):
            batch = _batch(split, indices, device)
            rows.append(torch.softmax(model(*batch[:-1]), dim=-1).cpu())
    model.train(was_training)
    return torch.cat(rows)


def _loss(
    model: nn.Module,
    splits: tuple[IrregularSplit, ...],
    *,
    batch_size: int,
    device: str,
) -> float:
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for split in splits:
            for indices in torch.arange(split.labels.numel()).split(batch_size):
                batch = _batch(split, indices, device)
                total += float(
                    functional.cross_entropy(
                        model(*batch[:-1]),
                        batch[-1],
                        reduction="sum",
                    )
                )
                count += indices.numel()
    return total / count


def fit(
    model: nn.Module,
    train_splits: tuple[IrregularSplit, ...],
    validation: IrregularSplit | None,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    seed: int,
    device: str,
    patience: int = 8,
) -> FitResult:
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    labels = torch.cat(tuple(split.labels for split in train_splits))
    counts = torch.bincount(labels, minlength=int(labels.max()) + 1).float()
    weights = (counts.sum() / (len(counts) * counts.clamp_min(1))).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_score = -math.inf
    best_epoch = epochs
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in _iter_batches(
            train_splits,
            _indices(train_splits, generator),
            batch_size,
            device,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:-1])
            loss = functional.cross_entropy(logits, batch[-1], weight=weights)
            auxiliary = getattr(model, "auxiliary_loss", None)
            if isinstance(auxiliary, Tensor):
                loss = loss + auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            callback = getattr(model, "post_optimizer_step", None)
            if callable(callback):
                callback()
        if validation is None:
            continue
        metrics = classification_metrics(
            predict(model, validation, batch_size=batch_size, device=device),
            validation.labels,
        )
        score, _ = selection_scores(metrics, counts.numel())
        if score > best_score + 1.0e-12:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if patience and stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    callback = getattr(model, "finalize_constraints", None)
    if callable(callback):
        callback()
    train_loss = _loss(model, train_splits, batch_size=batch_size, device=device)
    validation_metrics = (
        None
        if validation is None
        else classification_metrics(
            predict(model, validation, batch_size=batch_size, device=device),
            validation.labels,
        )
    )
    return FitResult(best_epoch, train_loss, validation_metrics)


def result_payload(result: FitResult) -> dict[str, object]:
    return {
        "best_epoch": result.best_epoch,
        "train_loss": result.train_loss,
        "validation": None if result.validation is None else asdict(result.validation),
    }


__all__ = [
    "FitResult",
    "IrregularAlphabet",
    "IrregularGRUD",
    "IrregularMetrics",
    "IrregularPackedSequenceBaseline",
    "build_model",
    "classification_metrics",
    "fit",
    "normalization_moments",
    "predict",
    "result_payload",
    "selection_scores",
]
