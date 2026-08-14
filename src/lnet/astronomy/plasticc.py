"""PLAsTiCC light-curve representation and object-level splits."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

PASSBAND_COUNT = 6
PHASE0_TARGETS = (16, 88, 92)
PHASE0_CLASS_NAMES = ("Eclipsing Binary", "AGN/QSO", "RR Lyrae")
PLASTICC_KNOWN_TARGETS = (6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95)
PLASTICC_KNOWN_CLASS_WEIGHTS = tuple(
    2.0 if target in (15, 64) else 1.0 for target in PLASTICC_KNOWN_TARGETS
)


@dataclass(frozen=True, slots=True)
class LightCurveRow:
    object_id: int
    mjd: float
    passband: int
    flux: float
    flux_error: float


@dataclass(frozen=True, slots=True)
class LightCurve:
    object_id: int
    time_delta: NDArray[np.float32]
    flux: NDArray[np.float32]
    flux_error: NDArray[np.float32]
    observation_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ObjectSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LightCurveBatch:
    flux: Tensor
    time_delta: Tensor
    observation_mask: Tensor
    valid_mask: Tensor
    target: Tensor
    object_id: Tensor


class PlasticcDataset(Dataset[tuple[LightCurve, int]]):
    def __init__(
        self,
        curves: Mapping[int, LightCurve],
        labels: Mapping[int, int],
        object_ids: Sequence[int],
    ) -> None:
        self.curves = curves
        self.labels = labels
        self.object_ids = tuple(object_ids)

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, index: int) -> tuple[LightCurve, int]:
        object_id = self.object_ids[index]
        return self.curves[object_id], self.labels[object_id]


def build_light_curve(rows: Sequence[LightCurveRow]) -> LightCurve:
    """Group simultaneous observations into sparse six-band epoch tokens."""
    if not rows:
        message = "a light curve requires at least one observation"
        raise ValueError(message)
    object_ids = {row.object_id for row in rows}
    if len(object_ids) != 1:
        message = "all observations must belong to one object"
        raise ValueError(message)
    epochs: dict[float, list[LightCurveRow]] = {}
    for row in rows:
        if row.passband not in range(PASSBAND_COUNT):
            message = f"passband must be in [0, 5], got {row.passband}"
            raise ValueError(message)
        epochs.setdefault(row.mjd, []).append(row)
    mjd = np.asarray(sorted(epochs), dtype=np.float64)
    time_delta = np.diff(mjd, prepend=mjd[0]).astype(np.float32)
    flux = np.zeros((len(mjd), PASSBAND_COUNT), dtype=np.float32)
    flux_error = np.zeros_like(flux)
    observation_mask = np.zeros_like(flux, dtype=np.bool_)
    for epoch_index, epoch in enumerate(mjd):
        for row in epochs[float(epoch)]:
            if observation_mask[epoch_index, row.passband]:
                message = "duplicate passband observation at one object epoch"
                raise ValueError(message)
            flux[epoch_index, row.passband] = row.flux
            flux_error[epoch_index, row.passband] = row.flux_error
            observation_mask[epoch_index, row.passband] = True
    return LightCurve(
        object_id=rows[0].object_id,
        time_delta=time_delta,
        flux=flux,
        flux_error=flux_error,
        observation_mask=observation_mask,
    )


def stratified_object_split(
    labels: Mapping[int, int],
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> ObjectSplit:
    """Create a deterministic class-stratified object-disjoint split."""
    if not labels:
        message = "labels must not be empty"
        raise ValueError(message)
    if not (0.0 < train_fraction < 1.0):
        message = "train_fraction must lie in (0, 1)"
        raise ValueError(message)
    if not (0.0 < validation_fraction < 1.0 - train_fraction):
        message = "validation_fraction must leave a positive test fraction"
        raise ValueError(message)
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for object_id, class_id in labels.items():
        by_class.setdefault(class_id, []).append(object_id)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for class_id in sorted(by_class):
        object_ids = np.asarray(sorted(by_class[class_id]), dtype=np.int64)
        rng.shuffle(object_ids)
        train_end = round(len(object_ids) * train_fraction)
        validation_end = train_end + round(len(object_ids) * validation_fraction)
        train.extend(object_ids[:train_end].tolist())
        validation.extend(object_ids[train_end:validation_end].tolist())
        test.extend(object_ids[validation_end:].tolist())
    return ObjectSplit(tuple(sorted(train)), tuple(sorted(validation)), tuple(sorted(test)))


def stratified_train_validation_split(
    labels: Mapping[int, int],
    *,
    seed: int,
    validation_fraction: float = 0.1,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Create a deterministic class-stratified train/validation object split."""
    if not labels:
        message = "labels must not be empty"
        raise ValueError(message)
    if not 0.0 < validation_fraction < 1.0:
        message = "validation_fraction must lie in (0, 1)"
        raise ValueError(message)
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for object_id, class_id in labels.items():
        by_class.setdefault(class_id, []).append(object_id)
    train: list[int] = []
    validation: list[int] = []
    for class_id in sorted(by_class):
        object_ids = np.asarray(sorted(by_class[class_id]), dtype=np.int64)
        rng.shuffle(object_ids)
        validation_count = max(1, round(validation_fraction * len(object_ids)))
        validation.extend(object_ids[:validation_count].tolist())
        train.extend(object_ids[validation_count:].tolist())
    return tuple(sorted(train)), tuple(sorted(validation))


def read_phase0_labels(
    metadata_path: Path,
    *,
    targets: tuple[int, ...] = PHASE0_TARGETS,
    max_objects_per_class: int = 5000,
    seed: int = 20260729,
    target_column: str = "target",
) -> dict[int, int]:
    """Read and deterministically cap Phase-0 objects from PLAsTiCC metadata."""
    if max_objects_per_class < 1:
        message = "max_objects_per_class must be positive"
        raise ValueError(message)
    target_to_index = {target: index for index, target in enumerate(targets)}
    by_target: dict[int, list[int]] = {target: [] for target in targets}
    with gzip.open(metadata_path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            target = int(row[target_column])
            if target in target_to_index:
                by_target[target].append(int(row["object_id"]))
    rng = np.random.default_rng(seed)
    labels: dict[int, int] = {}
    for target in targets:
        object_ids = np.asarray(sorted(by_target[target]), dtype=np.int64)
        rng.shuffle(object_ids)
        for object_id in object_ids[:max_objects_per_class]:
            labels[int(object_id)] = target_to_index[target]
    return labels


def read_light_curves(
    light_curve_path: Path,
    object_ids: Iterable[int],
) -> dict[int, LightCurve]:
    """Read only requested objects from the row-oriented PLAsTiCC archive."""
    requested = set(object_ids)
    grouped: dict[int, list[LightCurveRow]] = {object_id: [] for object_id in requested}
    with gzip.open(light_curve_path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            object_id = int(row["object_id"])
            if object_id not in requested:
                continue
            grouped[object_id].append(
                LightCurveRow(
                    object_id=object_id,
                    mjd=float(row["mjd"]),
                    passband=int(row["passband"]),
                    flux=float(row["flux"]),
                    flux_error=float(row["flux_err"]),
                )
            )
    missing = sorted(object_id for object_id, rows in grouped.items() if not rows)
    if missing:
        message = f"missing light curves for {len(missing)} requested objects"
        raise ValueError(message)
    return {object_id: build_light_curve(rows) for object_id, rows in grouped.items()}


def iter_light_curves(
    light_curve_paths: Sequence[Path],
    object_ids: set[int] | None = None,
) -> Iterable[LightCurve]:
    """Stream object-grouped curves across ordered CSV shard boundaries."""
    current_object_id: int | None = None
    current_rows: list[LightCurveRow] = []
    for light_curve_path in light_curve_paths:
        with gzip.open(light_curve_path, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                object_id = int(row["object_id"])
                if current_object_id is not None and object_id != current_object_id:
                    if object_ids is None or current_object_id in object_ids:
                        yield build_light_curve(current_rows)
                    current_rows = []
                current_object_id = object_id
                if object_ids is None or object_id in object_ids:
                    current_rows.append(
                        LightCurveRow(
                            object_id=object_id,
                            mjd=float(row["mjd"]),
                            passband=int(row["passband"]),
                            flux=float(row["flux"]),
                            flux_error=float(row["flux_err"]),
                        )
                    )
    if current_object_id is not None and current_rows:
        yield build_light_curve(current_rows)


def collate_light_curves(
    examples: Sequence[tuple[LightCurve, int]],
) -> LightCurveBatch:
    """Right-pad variable-length curves while keeping raw signed flux unchanged."""
    if not examples:
        message = "cannot collate an empty batch"
        raise ValueError(message)
    maximum_length = max(curve.flux.shape[0] for curve, _ in examples)
    batch_size = len(examples)
    flux = torch.zeros(batch_size, maximum_length, PASSBAND_COUNT)
    time_delta = torch.zeros(batch_size, maximum_length, 1)
    observation_mask = torch.zeros(batch_size, maximum_length, PASSBAND_COUNT)
    valid_mask = torch.zeros(batch_size, maximum_length, 1)
    target = torch.empty(batch_size, dtype=torch.long)
    object_id = torch.empty(batch_size, dtype=torch.long)
    for index, (curve, class_index) in enumerate(examples):
        length = curve.flux.shape[0]
        flux[index, :length] = torch.from_numpy(curve.flux)
        time_delta[index, :length, 0] = torch.from_numpy(curve.time_delta)
        observation_mask[index, :length] = torch.from_numpy(curve.observation_mask)
        valid_mask[index, :length] = 1.0
        target[index] = class_index
        object_id[index] = curve.object_id
    return LightCurveBatch(
        flux=flux,
        time_delta=time_delta,
        observation_mask=observation_mask,
        valid_mask=valid_mask,
        target=target,
        object_id=object_id,
    )
