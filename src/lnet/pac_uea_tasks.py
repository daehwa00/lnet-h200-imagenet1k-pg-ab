# ruff: noqa: EM101, EM102, TRY003
from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch import Tensor

from lnet.pac_external_tasks import (
    ExternalDatasetError,
    ExternalSelectionTask,
    ExternalTask,
    ExternalTemporalMetadata,
    load_external_selection_task,
    load_prepared_task,
    save_prepared_task,
    write_external_selection_task,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import BinaryIO

UEA_30_DATASETS: Final = (
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
)

# Fixed records in the official TSML Zenodo community. These are the same records used
# by aeon, the reference Python toolkit maintained by the archive authors.
UEA_ZENODO_RECORDS: Final = {
    "ArticularyWordRecognition": 11204924,
    "AtrialFibrillation": 11206175,
    "BasicMotions": 11206179,
    "CharacterTrajectories": 18723007,
    "Cricket": 11206185,
    "DuckDuckGeese": 11206189,
    "EigenWorms": 11206196,
    "Epilepsy": 11206204,
    "ERing": 11206210,
    "EthanolConcentration": 11206212,
    "FaceDetection": 11206216,
    "FingerMovements": 11206220,
    "HandMovementDirection": 11206224,
    "Handwriting": 11206227,
    "Heartbeat": 11206229,
    "InsectWingbeat": 11206234,
    "JapaneseVowels": 18735628,
    "Libras": 11206239,
    "LSST": 11206243,
    "MotorImagery": 11206246,
    "NATOPS": 11206248,
    "PEMS-SF": 11206252,
    "PenDigits": 11206259,
    "PhonemeSpectra": 11206261,
    "RacketSports": 11206263,
    "SelfRegulationSCP1": 11206265,
    "SelfRegulationSCP2": 11206269,
    "SpokenArabicDigits": 18734026,
    "StandWalkJump": 11206278,
    "UWaveGestureLibrary": 11206282,
}

_ZENODO_API: Final = "https://zenodo.org/api/records"
_UEA_ARCHIVE_RECORD_ID: Final = 11206331
_UEA_ARCHIVE_DOI: Final = "10.5281/zenodo.11206331"
_USER_AGENT: Final = "lnet-uea-preparer/1.0 (research dataset downloader)"
_CHUNK_BYTES: Final = 1024 * 1024
_MISSING_VALUES: Final = {"?", "nan", "NaN", "NAN"}


@dataclass(frozen=True, slots=True)
class ParsedTsCase:
    values: Tensor
    observed: Tensor
    time_delta: Tensor | None
    label: str


@dataclass(frozen=True, slots=True)
class ParsedTsFile:
    problem_name: str
    class_names: tuple[str, ...]
    cases: tuple[ParsedTsCase, ...]
    timestamps: bool

    @property
    def dimensions(self) -> int:
        return int(self.cases[0].values.shape[1])


@dataclass(frozen=True, slots=True)
class _PaddedSplit:
    inputs: Tensor
    targets: Tensor
    observation_mask: Tensor
    valid_mask: Tensor
    time_delta: Tensor | None

    def index_select(self, indices: Tensor) -> _PaddedSplit:
        return _PaddedSplit(
            inputs=self.inputs[indices],
            targets=self.targets[indices],
            observation_mask=self.observation_mask[indices],
            valid_mask=self.valid_mask[indices],
            time_delta=None if self.time_delta is None else self.time_delta[indices],
        )


def load_uea_task(
    dataset: str,
    source_root: Path,
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 20260727,
) -> ExternalTask:
    """Load one original UEA TRAIN/TEST pair without touching held-out statistics."""
    _validate_dataset_name(dataset)
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie strictly between 0 and 0.5")
    dataset_root = source_root / dataset
    train_file = parse_ts_file(dataset_root / f"{dataset}_TRAIN.ts")
    test_file = parse_ts_file(dataset_root / f"{dataset}_TEST.ts")
    _validate_pair(dataset, train_file, test_file)

    label_to_index = {label: index for index, label in enumerate(train_file.class_names)}
    official_train = _pad_cases(train_file.cases, label_to_index)
    official_test = _pad_cases(test_file.cases, label_to_index)
    train_indices, validation_indices = stratified_train_validation_indices(
        official_train.targets,
        validation_fraction=validation_fraction,
        seed=split_seed,
        class_names=train_file.class_names,
    )
    train = official_train.index_select(train_indices)
    validation = official_train.index_select(validation_indices)
    mean, scale = _training_channel_statistics(train.inputs, train.observation_mask)
    train = _normalize_split(train, mean, scale)
    validation = _normalize_split(validation, mean, scale)
    official_test = _normalize_split(official_test, mean, scale)

    train_groups = tuple(f"official-train:{int(index)}" for index in train_indices)
    validation_groups = tuple(
        f"official-train:{int(index)}" for index in validation_indices
    )
    test_groups = tuple(f"official-test:{index}" for index in range(len(test_file.cases)))
    return ExternalTask(
        name=dataset,
        objective="multiclass",
        train_inputs=train.inputs,
        train_targets=train.targets,
        validation_inputs=validation.inputs,
        validation_targets=validation.targets,
        test_inputs=official_test.inputs,
        test_targets=official_test.targets,
        output_dim=len(train_file.class_names),
        class_names=train_file.class_names,
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
        train_metadata=_metadata(train),
        validation_metadata=_metadata(validation),
        test_metadata=_metadata(official_test),
    )


def prepare_uea_task(
    dataset: str,
    source_root: Path,
    output_root: Path,
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 20260727,
) -> dict[str, object]:
    """Write full and physically sealed selection artifacts plus an audit record."""
    task = load_uea_task(
        dataset,
        source_root,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    full_path = output_root / f"{dataset}.pt"
    selection_path = output_root / "selection-only" / f"{dataset}.pt"
    _atomic_save_prepared_task(task, full_path)
    write_external_selection_task(task, selection_path)
    loaded_full = load_prepared_task(full_path)
    loaded_selection = load_external_selection_task(dataset, output_root)
    _assert_task_identity(task, loaded_full)
    _assert_selection_identity(loaded_full, loaded_selection)
    source_paths = (
        source_root / dataset / f"{dataset}_TRAIN.ts",
        source_root / dataset / f"{dataset}_TEST.ts",
    )
    download_manifest_path = source_root / dataset / "download-manifest.json"
    source_revision = _source_revision(download_manifest_path, dataset)
    record: dict[str, object] = {
        "schema": "lnet.uea-preparation.v1",
        "dataset": dataset,
        "official_source": f"https://zenodo.org/records/{UEA_ZENODO_RECORDS[dataset]}",
        "zenodo_record_id": UEA_ZENODO_RECORDS[dataset],
        "source_revision": source_revision,
        "reference_archive": {
            "record_id": _UEA_ARCHIVE_RECORD_ID,
            "doi": _UEA_ARCHIVE_DOI,
            "usage": "reference only; exact per-dataset original split files were used",
        },
        "source_sha256": {path.name: _hash_path(path, "sha256") for path in source_paths},
        "source_bytes": {path.name: path.stat().st_size for path in source_paths},
        "full_artifact": str(full_path),
        "full_artifact_sha256": _hash_path(full_path, "sha256"),
        "full_artifact_bytes": full_path.stat().st_size,
        "selection_artifact": str(selection_path),
        "selection_artifact_sha256": _hash_path(selection_path, "sha256"),
        "selection_artifact_bytes": selection_path.stat().st_size,
        "selection_full_identity": {
            "verified": True,
            "selection_split_sha256": loaded_selection.selection_split_sha256,
            "scope": "TRAIN/validation tensors, labels, groups, and temporal metadata",
        },
        "split": {
            "source": "official TRAIN only",
            "algorithm": "per-class stable SHA-256 ordering",
            "validation_fraction": validation_fraction,
            "seed": split_seed,
        },
        "normalization": "per-channel mean/std fitted on derived training subset only",
        "counts": {
            "train": int(task.train_inputs.shape[0]),
            "validation": int(task.validation_inputs.shape[0]),
            "test": int(task.test_inputs.shape[0]),
        },
        "shape": {
            "train_steps": int(task.train_inputs.shape[1]),
            "test_steps": int(task.test_inputs.shape[1]),
            "channels": task.input_dim,
            "classes": task.output_dim,
        },
    }
    audit_path = output_root / "provenance" / f"{dataset}.json"
    _atomic_json(audit_path, record)
    return record


def download_uea_dataset(dataset: str, source_root: Path) -> dict[str, object]:
    """Download only the two original split files from a fixed official record."""
    _validate_dataset_name(dataset)
    record_id = UEA_ZENODO_RECORDS[dataset]
    api_url = f"{_ZENODO_API}/{record_id}"
    record = _read_json_url(api_url)
    returned_record_id = record.get("id")
    if not isinstance(returned_record_id, int) or returned_record_id != record_id:
        raise ExternalDatasetError(f"Zenodo returned the wrong record for {dataset}")
    files = record.get("files")
    if not isinstance(files, list):
        raise ExternalDatasetError(f"Zenodo record {record_id} has no file listing")
    by_name = {
        str(item.get("key")): item
        for item in files
        if isinstance(item, dict) and item.get("key") is not None
    }
    expected_names = (f"{dataset}_TRAIN.ts", f"{dataset}_TEST.ts")
    destination = source_root / dataset
    destination.mkdir(parents=True, exist_ok=True)
    file_records: list[dict[str, object]] = []
    for filename in expected_names:
        item = by_name.get(filename)
        if item is None:
            raise ExternalDatasetError(
                f"official record {record_id} is missing original split {filename}"
            )
        file_records.append(_download_zenodo_file(item, destination / filename))
    manifest: dict[str, object] = {
        "schema": "lnet.uea-download.v1",
        "dataset": dataset,
        "record_id": record_id,
        "record_url": f"https://zenodo.org/records/{record_id}",
        "doi": record.get("doi"),
        "files": file_records,
        "extraction": "none; exact split files downloaded individually",
    }
    _atomic_json(destination / "download-manifest.json", manifest)
    return manifest


def parse_ts_file(path: Path) -> ParsedTsFile:  # noqa: C901
    if not path.is_file():
        raise ExternalDatasetError(f"UEA split does not exist: {path}")
    metadata: dict[str, list[str]] = {}
    cases: list[ParsedTsCase] = []
    data_started = False
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not data_started:
                if not line.startswith("@"):
                    raise ExternalDatasetError(
                        f"{path}:{line_number}: metadata line must start with @"
                    )
                key, *values = shlex.split(line[1:], comments=False, posix=True)
                lowered = key.lower()
                if lowered == "data":
                    data_started = True
                else:
                    metadata[lowered] = values
                continue
            try:
                cases.append(_parse_case(line, metadata))
            except (TypeError, ValueError) as error:
                raise ExternalDatasetError(
                    f"{path}:{line_number}: invalid .ts data row: {error}"
                ) from error
    if not data_started or not cases:
        raise ExternalDatasetError(f"{path} has no non-empty @data section")
    timestamps = _metadata_bool(metadata, "timestamps")
    class_values = metadata.get("classlabel")
    if not class_values or len(class_values) < 2 or class_values[0].lower() != "true":
        raise ExternalDatasetError(f"{path} must declare @classLabel true with classes")
    class_names = tuple(class_values[1:])
    unknown = sorted({case.label for case in cases} - set(class_names))
    if unknown:
        raise ExternalDatasetError(f"{path} contains labels absent from its header: {unknown}")
    dimensions = {int(case.values.shape[1]) for case in cases}
    if len(dimensions) != 1:
        raise ExternalDatasetError(f"{path} changes channel count between cases")
    return ParsedTsFile(
        problem_name=" ".join(metadata.get("problemname", (path.stem,))),
        class_names=class_names,
        cases=tuple(cases),
        timestamps=timestamps,
    )


def stratified_train_validation_indices(
    targets: Tensor,
    *,
    validation_fraction: float,
    seed: int,
    class_names: Sequence[str],
) -> tuple[Tensor, Tensor]:
    if targets.ndim != 1 or targets.dtype != torch.long:
        raise ValueError("targets must be rank-1 torch.long")
    train: list[int] = []
    validation: list[int] = []
    for class_index, class_name in enumerate(class_names):
        members = [
            int(value)
            for value in (targets == class_index).nonzero(as_tuple=False).flatten().tolist()
        ]
        if not members:
            raise ExternalDatasetError(f"official TRAIN has no case for class {class_name!r}")
        members.sort(key=lambda index: _stable_key(f"{seed}:{class_name}:{index}"))
        count = (
            0
            if len(members) == 1
            else min(
                len(members) - 1,
                max(1, math.floor(len(members) * validation_fraction + 0.5)),
            )
        )
        validation.extend(members[:count])
        train.extend(members[count:])
    if not validation:
        raise ExternalDatasetError("TRAIN is too small to derive a validation split")
    train.sort()
    validation.sort()
    return torch.tensor(train, dtype=torch.long), torch.tensor(validation, dtype=torch.long)


def _parse_case(line: str, metadata: dict[str, list[str]]) -> ParsedTsCase:
    parts = _split_top_level(line, ":")
    class_values = metadata.get("classlabel", ())
    if not class_values or class_values[0].lower() != "true" or len(parts) < 3:
        raise ValueError("classification row must contain dimensions and a class label")
    label = parts[-1].strip()
    channels = parts[:-1]
    timestamps = _metadata_bool(metadata, "timestamps")
    if timestamps:
        parsed_channels = [_parse_timestamp_channel(channel) for channel in channels]
        reference_times = parsed_channels[0][0]
        if any(times != reference_times for times, _ in parsed_channels[1:]):
            raise ValueError("timestamps must align across channels")
        channel_values = [values for _, values in parsed_channels]
        time_delta = _timestamp_deltas(reference_times)
    else:
        channel_values = [_parse_plain_channel(channel) for channel in channels]
        time_delta = None
    lengths = {len(values) for values in channel_values}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("all channels in a case must have the same positive length")
    array = np.stack(channel_values, axis=1).astype(np.float32, copy=False)
    values = torch.from_numpy(array)
    observed = torch.isfinite(values)
    values = torch.where(observed, values, torch.zeros_like(values))
    return ParsedTsCase(values=values, observed=observed, time_delta=time_delta, label=label)


def _parse_plain_channel(channel: str) -> np.ndarray:
    stripped = channel.strip()
    if not stripped:
        return np.empty(0, dtype=np.float32)
    normalized = stripped
    for marker in _MISSING_VALUES:
        normalized = normalized.replace(marker, "nan")
    values = np.fromstring(normalized, dtype=np.float64, sep=",")
    expected = stripped.count(",") + 1
    if values.size != expected:
        raise ValueError(f"could not parse all {expected} values in channel")
    return values


def _parse_timestamp_channel(channel: str) -> tuple[tuple[float, ...], np.ndarray]:
    pairs = _parenthesized_items(channel)
    timestamps: list[float] = []
    values: list[float] = []
    for item in pairs:
        timestamp, separator, value = item.partition(",")
        if not separator:
            raise ValueError("timestamp tuple must contain time,value")
        timestamps.append(_parse_timestamp(timestamp.strip()))
        value = value.strip()
        values.append(math.nan if value in _MISSING_VALUES else float(value))
    return tuple(timestamps), np.asarray(values, dtype=np.float64)


def _parse_timestamp(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"unsupported timestamp {value!r}") from error
        return parsed.timestamp()


def _timestamp_deltas(timestamps: Sequence[float]) -> Tensor:
    values = torch.tensor(timestamps, dtype=torch.float64)
    deltas = torch.zeros_like(values)
    if values.numel() > 1:
        deltas[1:] = values[1:] - values[:-1]
    if not torch.isfinite(deltas).all() or bool((deltas < 0).any()):
        raise ValueError("timestamps must be finite and non-decreasing")
    return deltas.to(dtype=torch.float32)


def _parenthesized_items(value: str) -> list[str]:
    items: list[str] = []
    index = 0
    while index < len(value):
        while index < len(value) and (value[index].isspace() or value[index] == ","):
            index += 1
        if index == len(value):
            break
        if value[index] != "(":
            raise ValueError("timestamp channel must contain parenthesized tuples")
        end = value.find(")", index + 1)
        if end < 0:
            raise ValueError("unterminated timestamp tuple")
        items.append(value[index + 1 : end])
        index = end + 1
    if not items:
        raise ValueError("timestamp channel is empty")
    return items


def _split_top_level(value: str, separator: str) -> list[str]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced parentheses")
        elif character == separator and depth == 0:
            result.append(value[start:index])
            start = index + 1
    if depth:
        raise ValueError("unbalanced parentheses")
    result.append(value[start:])
    return result


def _pad_cases(
    cases: Sequence[ParsedTsCase],
    label_to_index: dict[str, int],
) -> _PaddedSplit:
    maximum = max(int(case.values.shape[0]) for case in cases)
    channels = int(cases[0].values.shape[1])
    inputs = torch.zeros((len(cases), maximum, channels), dtype=torch.float32)
    observed = torch.zeros_like(inputs, dtype=torch.bool)
    valid = torch.zeros((len(cases), maximum), dtype=torch.bool)
    has_timestamps = cases[0].time_delta is not None
    if any((case.time_delta is not None) != has_timestamps for case in cases):
        raise ExternalDatasetError("a split mixes timestamped and index-sampled cases")
    time_delta = torch.zeros((len(cases), maximum), dtype=torch.float32) if has_timestamps else None
    targets = torch.empty(len(cases), dtype=torch.long)
    for index, case in enumerate(cases):
        length = int(case.values.shape[0])
        inputs[index, :length] = case.values
        observed[index, :length] = case.observed
        valid[index, :length] = True
        if time_delta is not None and case.time_delta is not None:
            time_delta[index, :length] = case.time_delta
        try:
            targets[index] = label_to_index[case.label]
        except KeyError as error:
            raise ExternalDatasetError(f"unknown class label {case.label!r}") from error
    return _PaddedSplit(inputs, targets, observed, valid, time_delta)


def _training_channel_statistics(inputs: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
    weights = observed.to(dtype=inputs.dtype)
    counts = weights.sum(dim=(0, 1))
    if bool((counts == 0).any()):
        raise ExternalDatasetError("derived training split has an entirely missing channel")
    mean = (inputs * weights).sum(dim=(0, 1)) / counts
    variance = ((inputs - mean).square() * weights).sum(dim=(0, 1)) / counts
    scale = variance.sqrt()
    scale = torch.where(scale > 1.0e-6, scale, torch.ones_like(scale))
    return mean, scale


def _normalize_split(split: _PaddedSplit, mean: Tensor, scale: Tensor) -> _PaddedSplit:
    normalized = torch.where(
        split.observation_mask,
        (split.inputs - mean) / scale,
        torch.zeros_like(split.inputs),
    )
    return _PaddedSplit(
        normalized,
        split.targets,
        split.observation_mask,
        split.valid_mask,
        split.time_delta,
    )


def _metadata(split: _PaddedSplit) -> ExternalTemporalMetadata:
    valid_mask = split.valid_mask if not bool(split.valid_mask.all()) else None
    active = split.valid_mask.unsqueeze(-1).expand_as(split.observation_mask)
    observation_mask = (
        split.observation_mask
        if bool((split.observation_mask != active).any())
        else None
    )
    return ExternalTemporalMetadata(
        time_delta=split.time_delta,
        observation_mask=observation_mask,
        valid_mask=valid_mask,
    )


def _assert_task_identity(expected: ExternalTask, actual: ExternalTask) -> None:
    scalar_fields = (
        "name",
        "objective",
        "output_dim",
        "class_names",
        "train_groups",
        "validation_groups",
        "test_groups",
    )
    if any(getattr(expected, field) != getattr(actual, field) for field in scalar_fields):
        raise ExternalDatasetError("full UEA artifact changed scalar task metadata")
    for split in ("train", "validation", "test"):
        if not torch.equal(
            getattr(expected, f"{split}_inputs"),
            getattr(actual, f"{split}_inputs"),
        ) or not torch.equal(
            getattr(expected, f"{split}_targets"),
            getattr(actual, f"{split}_targets"),
        ):
            raise ExternalDatasetError(f"full UEA artifact changed {split} tensors")
        _assert_metadata_identity(
            getattr(expected, f"{split}_metadata"),
            getattr(actual, f"{split}_metadata"),
            split,
        )


def _assert_selection_identity(
    full: ExternalTask,
    selection: ExternalSelectionTask,
) -> None:
    scalar_fields = (
        "name",
        "objective",
        "output_dim",
        "class_names",
        "train_groups",
        "validation_groups",
    )
    if any(getattr(full, field) != getattr(selection, field) for field in scalar_fields):
        raise ExternalDatasetError("selection UEA artifact changed scalar task metadata")
    if selection.test_count != full.test_inputs.shape[0]:
        raise ExternalDatasetError("selection UEA artifact changed held-out count")
    for split in ("train", "validation"):
        if not torch.equal(
            getattr(full, f"{split}_inputs"),
            getattr(selection, f"{split}_inputs"),
        ) or not torch.equal(
            getattr(full, f"{split}_targets"),
            getattr(selection, f"{split}_targets"),
        ):
            raise ExternalDatasetError(f"selection/full UEA {split} tensors differ")
        _assert_metadata_identity(
            getattr(full, f"{split}_metadata"),
            getattr(selection, f"{split}_metadata"),
            split,
        )


def _assert_metadata_identity(
    expected: ExternalTemporalMetadata,
    actual: ExternalTemporalMetadata,
    split: str,
) -> None:
    for name in ("time_delta", "observation_mask", "valid_mask"):
        left = getattr(expected, name)
        right = getattr(actual, name)
        if (left is None) != (right is None) or (
            left is not None and right is not None and not torch.equal(left, right)
        ):
            raise ExternalDatasetError(f"selection/full UEA {split} {name} differs")


def _validate_pair(dataset: str, train: ParsedTsFile, test: ParsedTsFile) -> None:
    if train.class_names != test.class_names:
        raise ExternalDatasetError(f"{dataset} TRAIN/TEST class declarations differ")
    if train.dimensions != test.dimensions or train.dimensions < 2:
        raise ExternalDatasetError(f"{dataset} is not a consistent multivariate task")
    if train.timestamps != test.timestamps:
        raise ExternalDatasetError(f"{dataset} TRAIN/TEST timestamp declarations differ")


def _metadata_bool(metadata: dict[str, list[str]], name: str) -> bool:
    values = metadata.get(name)
    if values is None or len(values) != 1 or values[0].lower() not in {"true", "false"}:
        raise ValueError(f"metadata must declare @{name} true or false")
    return values[0].lower() == "true"


def _stable_key(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _validate_dataset_name(dataset: str) -> None:
    if dataset not in UEA_ZENODO_RECORDS or dataset not in UEA_30_DATASETS:
        raise ValueError(f"unsupported UEA-30 dataset: {dataset!r}")


def _read_json_url(url: str) -> dict[str, object]:
    request = Request(  # noqa: S310 - caller supplies a fixed Zenodo HTTPS URL
        url,
        headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS host
            payload = json.load(response)
    except (OSError, TimeoutError, json.JSONDecodeError) as error:
        raise ExternalDatasetError(f"failed to retrieve official metadata from {url}") from error
    if not isinstance(payload, dict):
        raise ExternalDatasetError(f"official metadata at {url} is not an object")
    return payload


def _download_zenodo_file(item: dict[str, object], destination: Path) -> dict[str, object]:
    key = item.get("key")
    checksum = item.get("checksum")
    size = item.get("size")
    links = item.get("links")
    if (
        not isinstance(key, str)
        or key != destination.name
        or not isinstance(checksum, str)
        or not isinstance(size, int)
        or size < 1
        or not isinstance(links, dict)
    ):
        raise ExternalDatasetError(f"invalid Zenodo file metadata for {destination.name}")
    url = links.get("self") or links.get("content")
    if not isinstance(url, str) or urlparse(url).scheme != "https":
        raise ExternalDatasetError(f"invalid Zenodo download URL for {destination.name}")
    if urlparse(url).hostname not in {"zenodo.org", "www.zenodo.org"}:
        raise ExternalDatasetError(f"unexpected download host for {destination.name}")
    algorithm, separator, expected = checksum.partition(":")
    if not separator or algorithm not in {"md5", "sha256"}:
        raise ExternalDatasetError(f"unsupported checksum for {destination.name}: {checksum}")
    if (
        destination.is_file()
        and destination.stat().st_size == size
        and _hash_path(destination, algorithm) == expected.lower()
    ):
        return _file_provenance(destination, url, size, checksum)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    request = Request(  # noqa: S310 - URL scheme and host are validated above
        url,
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:  # noqa: S310
            _copy_exact(response, handle, expected_size=size)
        if _hash_path(temporary, algorithm) != expected.lower():
            raise ExternalDatasetError(f"checksum mismatch for {destination.name}")
        if _looks_like_html(temporary):
            raise ExternalDatasetError(f"downloaded HTML instead of {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_provenance(destination, url, size, checksum)


def _copy_exact(source: BinaryIO, destination: BinaryIO, *, expected_size: int) -> None:
    written = 0
    while True:
        chunk = source.read(_CHUNK_BYTES)
        if not chunk:
            break
        written += len(chunk)
        if written > expected_size:
            raise ExternalDatasetError("download exceeded declared Zenodo size")
        destination.write(chunk)
    if written != expected_size:
        raise ExternalDatasetError(
            f"download size mismatch: expected {expected_size} bytes, received {written}"
        )


def _file_provenance(
    path: Path,
    url: str,
    size: int,
    official_checksum: str,
) -> dict[str, object]:
    return {
        "name": path.name,
        "bytes": size,
        "url": url,
        "official_checksum": official_checksum,
        "sha256": _hash_path(path, "sha256"),
    }


def _hash_path(path: Path, algorithm: str) -> str:
    if algorithm == "md5":
        digest = hashlib.md5(usedforsecurity=False)
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_html(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(256).lstrip().lower()
    return prefix.startswith((b"<!doctype html", b"<html"))


def _source_revision(path: Path, dataset: str) -> dict[str, object]:
    if not path.is_file():
        return {
            "verified_download_manifest": False,
            "reason": "source files were supplied locally without downloader provenance",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalDatasetError(f"invalid UEA download manifest: {path}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "lnet.uea-download.v1"
        or payload.get("dataset") != dataset
        or payload.get("record_id") != UEA_ZENODO_RECORDS[dataset]
    ):
        raise ExternalDatasetError(f"UEA download manifest does not match {dataset}")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ExternalDatasetError(f"UEA download manifest has no files: {path}")
    by_name = {
        item.get("name"): item
        for item in files
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    for suffix in ("TRAIN", "TEST"):
        filename = f"{dataset}_{suffix}.ts"
        item = by_name.get(filename)
        source = path.parent / filename
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("bytes"), int)
            or not isinstance(item.get("sha256"), str)
            or source.stat().st_size != item["bytes"]
            or _hash_path(source, "sha256") != item["sha256"]
        ):
            raise ExternalDatasetError(
                f"UEA source no longer matches verified download manifest: {filename}"
            )
    return {
        "verified_download_manifest": True,
        "manifest_sha256": _hash_path(path, "sha256"),
        "record_id": payload["record_id"],
        "doi": payload.get("doi"),
        "files": files,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_prepared_task(task: ExternalTask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        save_prepared_task(task, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "UEA_30_DATASETS",
    "UEA_ZENODO_RECORDS",
    "ParsedTsCase",
    "ParsedTsFile",
    "download_uea_dataset",
    "load_uea_task",
    "parse_ts_file",
    "prepare_uea_task",
    "stratified_train_validation_indices",
]
