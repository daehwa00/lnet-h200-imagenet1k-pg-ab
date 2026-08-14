# ruff: noqa: EM101, EM102, TC003, TRY003
"""Wave-1 public irregular-task adapters.

The adapters in this module intentionally remain separate from the historical
``pac_external_tasks`` registry.  They produce the same ``ExternalTask`` contract
while preserving irregular intervals, channel missingness, and right-padding.

Human Activity uses the event representation from the Latent ODE reference code:
four 3-D tags, 10 ms timestamp bins, 50-event windows, stride 25, and the same
seven merged activity classes.  Since ``ExternalTask`` is sequence-level, the
window target is the modal per-event activity. This is an explicit window-level
variant, and complete recording IDs are partitioned before windows are assigned
so overlapping windows never cross train, validation, or TEST boundaries.

USHCN follows the GRU-ODE-Bayes climate protocol: observations through time 150
form the context and the first three later event rows form the forecast target.
The reference metric masks unobserved target channels.  To retain that metric
with the dense ``ExternalTask`` forecasting loss, each observed future scalar is
unfolded into one query-conditioned sample.  Averaging MSE over these samples is
equivalent to averaging over the reference target mask.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import numpy as np
import torch
from torch import Tensor

from .pac_external_tasks import ExternalTask, ExternalTemporalMetadata
from .pac_time_normalization import (
    fit_characteristic_time_scale,
    normalize_time_delta,
)

HUMAN_ACTIVITY_NAME: Final = "human-activity"
USHCN_DAILY_NAME: Final = "ushcn-daily"

HUMAN_ACTIVITY_UPSTREAM_COMMIT: Final = "c0682d4f52b806fb88d965755892eadd9783f936"
USHCN_UPSTREAM_COMMIT: Final = "ddd0b34e884dbee1c09b6a3927d1e9ab10443af8"


@dataclass(frozen=True, slots=True)
class PublicSource:
    dataset: str
    filename: str
    url: str
    sha256: str
    license_name: str
    upstream_commit: str | None = None


HUMAN_ACTIVITY_SOURCE: Final = PublicSource(
    dataset=HUMAN_ACTIVITY_NAME,
    filename="localization-data-for-person-activity.zip",
    url=(
        "https://archive.ics.uci.edu/static/public/196/"
        "localization+data+for+person+activity.zip"
    ),
    sha256="6660124b4d1fd98963c56a6804bbb0ca0c5ee12edc9b772c8a867148af20d0da",
    license_name="CC BY 4.0",
    upstream_commit=HUMAN_ACTIVITY_UPSTREAM_COMMIT,
)
USHCN_DAILY_SOURCE: Final = PublicSource(
    dataset=USHCN_DAILY_NAME,
    filename="small_chunked_sporadic.csv",
    url=(
        "https://raw.githubusercontent.com/edebrouwer/gru_ode_bayes/"
        f"{USHCN_UPSTREAM_COMMIT}/gru_ode_bayes/datasets/Climate/"
        "small_chunked_sporadic.csv"
    ),
    sha256="671eb8d121522e98891c84197742a6c9e9bb5015e42b328a93ebdf2cfd393ecf",
    license_name="MIT (upstream processed artifact); NOAA source data are public",
    upstream_commit=USHCN_UPSTREAM_COMMIT,
)
PUBLIC_SOURCES: Final = (HUMAN_ACTIVITY_SOURCE, USHCN_DAILY_SOURCE)

_ACTIVITY_MEMBER: Final = "ConfLongDemo_JSI.txt"
_ACTIVITY_TAGS: Final = (
    "010-000-024-033",  # left ankle
    "010-000-030-096",  # right ankle
    "020-000-033-111",  # chest
    "020-000-032-221",  # belt
)
_ACTIVITY_TAG_INDEX: Final = {tag: index for index, tag in enumerate(_ACTIVITY_TAGS)}
ACTIVITY_CLASS_NAMES: Final = (
    "walking",
    "falling",
    "lying",
    "sitting",
    "standing-up",
    "on-all-fours",
    "sitting-on-ground",
)
_ACTIVITY_LABELS: Final = {
    "walking": 0,
    "falling": 1,
    "lying": 2,
    "lying down": 2,
    "sitting": 3,
    "sitting down": 3,
    "standing up from lying": 4,
    "standing up from sitting": 4,
    "standing up from sit on grnd": 4,
    "standing up from sitting on the ground": 4,
    "on all fours": 5,
    "sitting on the ground": 6,
}
_ACTIVITY_TEST_SEED: Final = 42
_ACTIVITY_VALIDATION_SEED: Final = 43
_USHCN_SPLIT_SEED: Final = 432
_USHCN_VARIABLES: Final = ("Value_0", "Value_1", "Value_2", "Value_3", "Value_4")
_USHCN_MASKS: Final = ("Mask_0", "Mask_1", "Mask_2", "Mask_3", "Mask_4")
_USHCN_CONTEXT_END: Final = 150.0


@dataclass(frozen=True, slots=True)
class _ActivityWindow:
    values: Tensor
    observed: Tensor
    time_delta: Tensor
    target: int
    group: str


@dataclass(frozen=True, slots=True)
class _ClimateStation:
    station_id: str
    time: Tensor
    values: Tensor
    observed: Tensor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_public_source(source: PublicSource, destination: Path) -> dict[str, object]:
    """Download one immutable source and fail closed on a digest mismatch."""
    parsed = urlparse(source.url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("public benchmark sources must use an absolute HTTPS URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_sha256(destination) == source.sha256:
        return _source_audit(source, destination, downloaded=False)

    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with (
            urllib.request.urlopen(source.url, timeout=120) as response,  # noqa: S310
            temporary.open("wb") as output,
        ):
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
        actual = file_sha256(temporary)
        if actual != source.sha256:
            msg = f"checksum mismatch for {source.dataset}: {actual} != {source.sha256}"
            raise RuntimeError(msg)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _source_audit(source, destination, downloaded=True)


def _source_audit(
    source: PublicSource,
    destination: Path,
    *,
    downloaded: bool,
) -> dict[str, object]:
    return {
        "dataset": source.dataset,
        "url": source.url,
        "path": str(destination),
        "sha256": file_sha256(destination),
        "expected_sha256": source.sha256,
        "bytes": destination.stat().st_size,
        "license": source.license_name,
        "upstream_commit": source.upstream_commit,
        "downloaded": downloaded,
    }


def _locked_partition(
    count: int,
    *,
    test_fraction: float,
    validation_fraction_of_development: float,
    test_seed: int,
    validation_seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 3:
        raise ValueError("at least three samples are required for train/validation/test")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")
    if not 0.0 < validation_fraction_of_development < 1.0:
        raise ValueError("validation fraction must lie in (0, 1)")

    test_generator = np.random.RandomState(test_seed)
    permutation = test_generator.permutation(count)
    test_count = math.ceil(count * test_fraction)
    test = permutation[:test_count]
    development = permutation[test_count:]

    validation_generator = (
        test_generator
        if validation_seed is None
        else np.random.RandomState(validation_seed)
    )
    validation_permutation = validation_generator.permutation(len(development))
    validation_count = math.ceil(len(development) * validation_fraction_of_development)
    validation = development[validation_permutation[:validation_count]]
    train = development[validation_permutation[validation_count:]]
    if min(len(train), len(validation), len(test)) < 1:
        raise ValueError("partition produced an empty split")
    return train, validation, test


def _masked_channel_statistics(values: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
    if values.ndim != 3 or observed.shape != values.shape:
        raise ValueError("statistics inputs must be aligned [S,N,C] tensors")
    weights = observed.to(dtype=torch.float64)
    values64 = values.to(dtype=torch.float64)
    count = weights.sum(dim=(0, 1))
    if bool((count == 0).any()):
        raise ValueError("every channel must be observed in the training split")
    mean = (values64 * weights).sum(dim=(0, 1)) / count
    variance = ((values64 - mean).square() * weights).sum(dim=(0, 1)) / count
    return mean.to(torch.float32), variance.sqrt().clamp_min(1.0e-6).to(torch.float32)


def _normalize_masked(values: Tensor, observed: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return ((values - mean) / std) * observed


def _activity_record_windows(
    record_id: str,
    rows: list[tuple[str, int, tuple[float, float, float], str]],
    *,
    segment_length: int,
    stride: int,
) -> list[_ActivityWindow]:
    if not rows:
        return []
    rows.sort(key=lambda row: row[1])
    first_timestamp = rows[0][1]
    event_bins: list[int] = []
    event_values: list[Tensor] = []
    event_counts: list[Tensor] = []
    event_observed: list[Tensor] = []
    event_labels: list[Tensor] = []

    for tag, timestamp, coordinates, label in rows:
        bin_index = round((timestamp - first_timestamp) / 100_000)
        if event_bins and bin_index < event_bins[-1]:
            raise ValueError(f"{record_id} activity timestamps are not monotonic")
        if not event_bins or bin_index != event_bins[-1]:
            event_bins.append(bin_index)
            event_values.append(torch.zeros((4, 3), dtype=torch.float32))
            event_counts.append(torch.zeros(4, dtype=torch.float32))
            event_observed.append(torch.zeros((4, 3), dtype=torch.float32))
            event_labels.append(torch.zeros(len(ACTIVITY_CLASS_NAMES), dtype=torch.float32))
        tag_index = _ACTIVITY_TAG_INDEX[tag]
        count = event_counts[-1][tag_index]
        coordinate_tensor = torch.tensor(coordinates, dtype=torch.float32)
        event_values[-1][tag_index] = (
            event_values[-1][tag_index] * count + coordinate_tensor
        ) / (count + 1.0)
        event_counts[-1][tag_index] = count + 1.0
        event_observed[-1][tag_index] = 1.0
        event_labels[-1][_ACTIVITY_LABELS[label]] += 1.0

    values = torch.stack(event_values).reshape(-1, 12)
    observed = torch.stack(event_observed).reshape(-1, 12)
    event_time = torch.tensor(event_bins, dtype=torch.float64) * 0.01
    windows: list[_ActivityWindow] = []
    offset = 0
    while offset + segment_length < len(event_bins):
        stop = offset + segment_length
        active_time = event_time[offset:stop]
        delta = torch.empty(segment_length, dtype=torch.float32)
        delta[0] = 0.0
        delta[1:] = (active_time[1:] - active_time[:-1]).to(torch.float32)
        votes = torch.stack(event_labels[offset:stop]).sum(dim=0)
        windows.append(
            _ActivityWindow(
                values=values[offset:stop],
                observed=observed[offset:stop],
                time_delta=delta,
                target=int(votes.argmax().item()),
                group=f"{record_id}:{offset}",
            )
        )
        offset += stride
    return windows


def _read_activity_windows(  # noqa: C901
    archive: Path,
    *,
    segment_length: int,
    stride: int,
) -> list[_ActivityWindow]:
    if segment_length < 2 or stride < 1:
        raise ValueError("activity segment_length must be >=2 and stride must be positive")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    windows: list[_ActivityWindow] = []
    with zipfile.ZipFile(archive) as bundle:
        member = bundle.getinfo(_ACTIVITY_MEMBER)
        if member.file_size > 64 * 1024 * 1024:
            raise RuntimeError("Human Activity member exceeds the audited size bound")
        current_id: str | None = None
        current_rows: list[tuple[str, int, tuple[float, float, float], str]] = []
        with bundle.open(member) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                columns = raw_line.decode("utf-8").strip().split(",")
                if len(columns) != 8:
                    raise ValueError(f"invalid Human Activity row {line_number}")
                record_id, tag, timestamp_text = columns[:3]
                label = columns[7].strip()
                if tag not in _ACTIVITY_TAG_INDEX:
                    raise ValueError(f"unknown Human Activity tag at row {line_number}: {tag}")
                if label not in _ACTIVITY_LABELS:
                    raise ValueError(
                        f"unknown Human Activity label at row {line_number}: {label}"
                    )
                if current_id is not None and record_id != current_id:
                    windows.extend(
                        _activity_record_windows(
                            current_id,
                            current_rows,
                            segment_length=segment_length,
                            stride=stride,
                        )
                    )
                    current_rows = []
                current_id = record_id
                current_rows.append(
                    (
                        tag,
                        int(timestamp_text),
                        (float(columns[4]), float(columns[5]), float(columns[6])),
                        label,
                    )
                )
        if current_id is not None:
            windows.extend(
                _activity_record_windows(
                    current_id,
                    current_rows,
                    segment_length=segment_length,
                    stride=stride,
                )
            )
    if not windows:
        raise RuntimeError("Human Activity preprocessing produced no windows")
    return windows


def _activity_split(
    windows: list[_ActivityWindow],
    indices: np.ndarray,
    mean: Tensor,
    std: Tensor,
    characteristic_time_scale: float,
) -> tuple[Tensor, Tensor, tuple[str, ...], ExternalTemporalMetadata]:
    selected = [windows[int(index)] for index in indices]
    values = torch.stack([window.values for window in selected])
    observed = torch.stack([window.observed for window in selected])
    valid = torch.ones(values.shape[:2], dtype=torch.float32)
    metadata = ExternalTemporalMetadata(
        time_delta=normalize_time_delta(
            torch.stack([window.time_delta for window in selected]),
            characteristic_time_scale,
            valid,
        ),
        observation_mask=observed,
        valid_mask=valid,
    )
    return (
        _normalize_masked(values, observed, mean, std),
        torch.tensor([window.target for window in selected], dtype=torch.long),
        tuple(window.group for window in selected),
        metadata,
    )


def _activity_record_partition(
    windows: list[_ActivityWindow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = sorted({window.group.partition(":")[0] for window in windows})
    train_records, validation_records, test_records = _locked_partition(
        len(records),
        test_fraction=0.20,
        validation_fraction_of_development=0.20,
        test_seed=_ACTIVITY_TEST_SEED,
        validation_seed=_ACTIVITY_VALIDATION_SEED,
    )
    record_sets = tuple(
        {records[int(index)] for index in indices}
        for indices in (train_records, validation_records, test_records)
    )

    def indices(selected_records: set[str]) -> np.ndarray:
        return np.asarray(
            [
                index
                for index, window in enumerate(windows)
                if window.group.partition(":")[0] in selected_records
            ],
            dtype=np.int64,
        )

    return indices(record_sets[0]), indices(record_sets[1]), indices(record_sets[2])


def load_human_activity(
    archive: Path,
    *,
    segment_length: int = 50,
    stride: int = 25,
) -> ExternalTask:
    """Build the seven-class sequence-level Human Activity task."""
    windows = _read_activity_windows(
        archive,
        segment_length=segment_length,
        stride=stride,
    )
    train_indices, validation_indices, test_indices = _activity_record_partition(windows)
    train_values = torch.stack([windows[int(index)].values for index in train_indices])
    train_observed = torch.stack(
        [windows[int(index)].observed for index in train_indices]
    )
    mean, std = _masked_channel_statistics(train_values, train_observed)
    train_delta = torch.stack(
        [windows[int(index)].time_delta for index in train_indices]
    )
    characteristic_time_scale = fit_characteristic_time_scale(train_delta)
    train = _activity_split(
        windows,
        train_indices,
        mean,
        std,
        characteristic_time_scale,
    )
    validation = _activity_split(
        windows,
        validation_indices,
        mean,
        std,
        characteristic_time_scale,
    )
    test = _activity_split(
        windows,
        test_indices,
        mean,
        std,
        characteristic_time_scale,
    )
    return ExternalTask(
        name=HUMAN_ACTIVITY_NAME,
        objective="multiclass",
        train_inputs=train[0],
        train_targets=train[1],
        validation_inputs=validation[0],
        validation_targets=validation[1],
        test_inputs=test[0],
        test_targets=test[1],
        output_dim=len(ACTIVITY_CLASS_NAMES),
        class_names=ACTIVITY_CLASS_NAMES,
        characteristic_time_scale=characteristic_time_scale,
        train_groups=train[2],
        validation_groups=validation[2],
        test_groups=test[2],
        train_metadata=train[3],
        validation_metadata=validation[3],
        test_metadata=test[3],
    )


def _station_id(value: str) -> str:
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"USHCN station ID must be a finite integer: {value}")
    return str(int(numeric))


def _read_ushcn_stations(path: Path) -> dict[str, _ClimateStation]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, list[tuple[float, tuple[float, ...], tuple[float, ...]]]] = (
        defaultdict(list)
    )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ID", "Time", *_USHCN_VARIABLES, *_USHCN_MASKS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("USHCN CSV does not contain the audited columns")
        for line_number, row in enumerate(reader, start=2):
            station_id = _station_id(row["ID"])
            time = float(row["Time"])
            values = tuple(float(row[name]) for name in _USHCN_VARIABLES)
            observed = tuple(float(row[name]) for name in _USHCN_MASKS)
            if not math.isfinite(time) or not all(map(math.isfinite, values)):
                raise ValueError(f"non-finite USHCN value at row {line_number}")
            if any(mask not in (0.0, 1.0) for mask in observed):
                raise ValueError(f"non-binary USHCN mask at row {line_number}")
            rows[station_id].append((time, values, observed))

    stations: dict[str, _ClimateStation] = {}
    for station_id, station_rows in rows.items():
        station_rows.sort(key=lambda item: item[0])
        stations[station_id] = _ClimateStation(
            station_id=station_id,
            time=torch.tensor([item[0] for item in station_rows], dtype=torch.float32),
            values=torch.tensor([item[1] for item in station_rows], dtype=torch.float32),
            observed=torch.tensor([item[2] for item in station_rows], dtype=torch.float32),
        )
    if len(stations) < 3:
        raise RuntimeError("USHCN preprocessing requires at least three stations")
    return stations


def _ushcn_training_statistics(
    stations: dict[str, _ClimateStation],
    train_ids: list[str],
) -> tuple[Tensor, Tensor]:
    values: list[Tensor] = []
    masks: list[Tensor] = []
    for station_id in train_ids:
        station = stations[station_id]
        context = station.time <= _USHCN_CONTEXT_END
        values.append(station.values[context])
        masks.append(station.observed[context])
    joined_values = torch.cat(values).unsqueeze(0)
    joined_masks = torch.cat(masks).unsqueeze(0)
    return _masked_channel_statistics(joined_values, joined_masks)


def _ushcn_training_time_scale(
    stations: dict[str, _ClimateStation],
    train_ids: list[str],
) -> float:
    interval_rows: list[Tensor] = []
    for station_id in train_ids:
        context_time = stations[station_id].time
        context_time = context_time[context_time <= _USHCN_CONTEXT_END]
        if context_time.numel() > 1:
            interval_rows.append(context_time.diff())
    if not interval_rows:
        raise ValueError("USHCN TRAIN context contains no time transition")
    intervals = torch.cat(interval_rows)
    return fit_characteristic_time_scale(
        intervals.unsqueeze(0),
        exclude_first=False,
    )


def _ushcn_split(
    stations: dict[str, _ClimateStation],
    station_ids: list[str],
    *,
    sequence_length: int,
    mean: Tensor,
    std: Tensor,
    characteristic_time_scale: float,
) -> tuple[Tensor, Tensor, tuple[str, ...], ExternalTemporalMetadata]:
    inputs: list[Tensor] = []
    targets: list[Tensor] = []
    deltas: list[Tensor] = []
    observations: list[Tensor] = []
    valid_masks: list[Tensor] = []
    groups: list[str] = []
    for station_id in station_ids:
        station = stations[station_id]
        context_selector = station.time <= _USHCN_CONTEXT_END
        context_time = station.time[context_selector]
        context_values = station.values[context_selector]
        context_observed = station.observed[context_selector]
        future_indices = torch.nonzero(
            station.time > _USHCN_CONTEXT_END,
            as_tuple=False,
        ).flatten()[:3]
        if context_time.numel() == 0 or future_indices.numel() == 0:
            continue
        normalized_context = _normalize_masked(
            context_values,
            context_observed,
            mean,
            std,
        )
        context_count = int(context_time.numel())
        for future_index in future_indices.tolist():
            target_observed = station.observed[future_index]
            for channel in torch.nonzero(target_observed, as_tuple=False).flatten().tolist():
                model_input = torch.zeros((sequence_length, 10), dtype=torch.float32)
                observed = torch.zeros_like(model_input)
                valid = torch.zeros(sequence_length, dtype=torch.float32)
                time_delta = torch.zeros(sequence_length, dtype=torch.float32)

                model_input[:context_count, :5] = normalized_context
                model_input[:context_count, 5:] = context_observed
                observed[:context_count, :5] = context_observed
                observed[:context_count, 5:] = 1.0
                valid[: context_count + 1] = 1.0

                query_index = context_count
                model_input[query_index, 5 + channel] = 1.0
                observed[query_index, 5:] = 1.0

                time_delta[0] = context_time[0]
                if context_count > 1:
                    time_delta[1:context_count] = (
                        context_time[1:] - context_time[:-1]
                    )
                time_delta[query_index] = (
                    station.time[future_index] - context_time[-1]
                )

                normalized_target = (
                    station.values[future_index, channel] - mean[channel]
                ) / std[channel]
                inputs.append(model_input)
                targets.append(normalized_target.reshape(1, 1))
                deltas.append(
                    normalize_time_delta(
                        time_delta.unsqueeze(0),
                        characteristic_time_scale,
                        valid.unsqueeze(0),
                    ).squeeze(0)
                )
                observations.append(observed)
                valid_masks.append(valid)
                groups.append(station_id)
    if not inputs:
        raise RuntimeError("USHCN split contains no forecast queries")
    return (
        torch.stack(inputs),
        torch.stack(targets),
        tuple(groups),
        ExternalTemporalMetadata(
            time_delta=torch.stack(deltas),
            observation_mask=torch.stack(observations),
            valid_mask=torch.stack(valid_masks),
        ),
    )


def load_ushcn_daily(path: Path) -> ExternalTask:
    """Build the query-unfolded GRU-ODE-Bayes USHCN forecasting task."""
    stations = _read_ushcn_stations(path)
    station_ids = sorted(stations, key=int)
    train_indices, validation_indices, test_indices = _locked_partition(
        len(station_ids),
        test_fraction=0.10,
        validation_fraction_of_development=0.20,
        test_seed=_USHCN_SPLIT_SEED,
        validation_seed=None,
    )
    train_ids = [station_ids[int(index)] for index in train_indices]
    validation_ids = [station_ids[int(index)] for index in validation_indices]
    test_ids = [station_ids[int(index)] for index in test_indices]
    mean, std = _ushcn_training_statistics(stations, train_ids)
    characteristic_time_scale = _ushcn_training_time_scale(stations, train_ids)
    max_context = max(
        int((station.time <= _USHCN_CONTEXT_END).sum().item())
        for station in stations.values()
    )
    sequence_length = max_context + 1
    train = _ushcn_split(
        stations,
        train_ids,
        sequence_length=sequence_length,
        mean=mean,
        std=std,
        characteristic_time_scale=characteristic_time_scale,
    )
    validation = _ushcn_split(
        stations,
        validation_ids,
        sequence_length=sequence_length,
        mean=mean,
        std=std,
        characteristic_time_scale=characteristic_time_scale,
    )
    test = _ushcn_split(
        stations,
        test_ids,
        sequence_length=sequence_length,
        mean=mean,
        std=std,
        characteristic_time_scale=characteristic_time_scale,
    )
    return ExternalTask(
        name=USHCN_DAILY_NAME,
        objective="forecasting",
        train_inputs=train[0],
        train_targets=train[1],
        validation_inputs=validation[0],
        validation_targets=validation[1],
        test_inputs=test[0],
        test_targets=test[1],
        output_dim=1,
        characteristic_time_scale=characteristic_time_scale,
        train_groups=train[2],
        validation_groups=validation[2],
        test_groups=test[2],
        train_metadata=train[3],
        validation_metadata=validation[3],
        test_metadata=test[3],
    )
