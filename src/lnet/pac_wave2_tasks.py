# ruff: noqa: C901, EM101, EM102, PLR0912, TRY003
"""Leakage-safe common task builder for Wave-2 sensor datasets.

Dataset-specific extractors write a compact manifest plus NumPy arrays.  This
module performs the shared group split, TRAIN-only normalization, windowing,
and serialization consumed unchanged by all ten benchmark models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly

from .pac_external_tasks import (
    ExternalTask,
    save_prepared_task,
    write_external_selection_task,
)

if TYPE_CHECKING:
    from pathlib import Path

Split = Literal["train", "validation", "test"]
SPLITS: Final = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class Wave2Policy:
    name: str
    target_rate_hz: int
    window_seconds: float
    stride_seconds: float
    channels: tuple[str, ...] = ()


POLICIES: Final = {
    "sleepedf-78": Wave2Policy("sleepedf-78", 100, 30.0, 30.0, ("EEG Fpz-Cz",)),
    "isruc-sleep": Wave2Policy("isruc-sleep", 100, 30.0, 30.0, ("F3-A2",)),
    "chb-mit": Wave2Policy("chb-mit", 128, 4.0, 2.0),
    "bci-iv-2a": Wave2Policy("bci-iv-2a", 250, 4.0, 4.0),
    "mfpt-bearing": Wave2Policy("mfpt-bearing", 12_000, 2048 / 12_000, 2048 / 12_000),
    "paderborn-kat": Wave2Policy("paderborn-kat", 12_800, 2048 / 12_800, 2048 / 12_800),
    "xjtu-sy": Wave2Policy("xjtu-sy", 12_800, 2048 / 12_800, 2048 / 12_800),
    "ims-bearing": Wave2Policy("ims-bearing", 10_000, 2048 / 10_000, 2048 / 10_000),
    "chapman-shaoxing": Wave2Policy("chapman-shaoxing", 250, 10.0, 10.0),
    "cpsc-2018": Wave2Policy("cpsc-2018", 250, 10.0, 10.0),
}


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    path: Path
    label: str
    group: str
    split: Split | None
    sample_rate_hz: float
    signal_key: str | None
    channels: tuple[int, ...] | None
    start_sample: int | None
    stop_sample: int | None


def _stable_fraction(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assigned_split(group: str, seed: int = 20260727) -> Split:
    value = _stable_fraction(group, seed)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def read_manifest(path: Path) -> tuple[ManifestRecord, ...]:
    root = path.parent
    records: list[ManifestRecord] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_split = row.get("split", "").strip()
            if raw_split and raw_split not in SPLITS:
                raise ValueError(f"invalid split {raw_split!r} in {path}")
            raw_channels = row.get("channels", "").strip()
            records.append(
                ManifestRecord(
                    path=root / row["path"],
                    label=row["label"],
                    group=row["group"],
                    split=raw_split or None,  # type: ignore[arg-type]
                    sample_rate_hz=float(row["sample_rate_hz"]),
                    signal_key=row.get("signal_key") or None,
                    channels=(
                        tuple(int(value) for value in raw_channels.split(";"))
                        if raw_channels
                        else None
                    ),
                    start_sample=(
                        int(row["start_sample"])
                        if row.get("start_sample", "").strip()
                        else None
                    ),
                    stop_sample=(
                        int(row["stop_sample"])
                        if row.get("stop_sample", "").strip()
                        else None
                    ),
                )
            )
    if not records:
        raise ValueError(f"empty Wave-2 manifest: {path}")
    return tuple(records)


def _load_signal(record: ManifestRecord) -> np.ndarray:
    suffix = record.path.suffix.lower()
    if suffix == ".npy":
        signal = np.load(record.path, mmap_mode="r")
    elif suffix == ".npz":
        payload = np.load(record.path)
        key = record.signal_key or next(iter(payload.files))
        signal = payload[key]
    elif suffix == ".mat":
        payload = loadmat(record.path)
        key = record.signal_key
        if key is None:
            candidates = [
                name
                for name, value in payload.items()
                if not name.startswith("__") and isinstance(value, np.ndarray)
            ]
            if len(candidates) != 1:
                raise ValueError(f"{record.path} requires an explicit signal_key")
            key = candidates[0]
        signal = payload[key]
    elif suffix in {".csv", ".txt", ""} or record.signal_key == "text":
        first_line = record.path.open(encoding="utf-8", errors="ignore").readline()
        has_header = any(character.isalpha() for character in first_line)
        signal = np.genfromtxt(
            record.path,
            delimiter="," if suffix == ".csv" else None,
            usecols=record.channels,
            invalid_raise=False,
            skip_header=int(has_header),
        )
    else:
        raise ValueError(f"unsupported Wave-2 signal file: {record.path}")
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2:
        raise ValueError(f"{record.path} must contain a rank-2 signal")
    if signal.shape[0] < signal.shape[1] and signal.shape[0] <= 64:
        signal = signal.T
    if (
        record.channels is not None
        and suffix not in {".csv", ".txt", ""}
        and record.signal_key != "text"
    ):
        signal = signal[:, record.channels]
    if record.start_sample is not None or record.stop_sample is not None:
        signal = signal[record.start_sample : record.stop_sample]
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal)
    return signal


def _resample(signal: np.ndarray, source: float, target: int) -> np.ndarray:
    if math.isclose(source, target):
        return signal
    source_i = round(source)
    divisor = math.gcd(source_i, target)
    return resample_poly(signal, target // divisor, source_i // divisor, axis=0).astype(
        np.float32
    )


def _windows(signal: np.ndarray, length: int, stride: int) -> list[np.ndarray]:
    if signal.shape[0] < length:
        padded = np.zeros((length, signal.shape[1]), dtype=np.float32)
        padded[: signal.shape[0]] = signal
        return [padded]
    return [
        np.array(signal[start : start + length], dtype=np.float32, order="C", copy=True)
        for start in range(0, signal.shape[0] - length + 1, stride)
    ]


def prepare_manifest_task(
    manifest: Path,
    output_root: Path,
    *,
    policy: Wave2Policy,
    split_seed: int = 20260727,
) -> ExternalTask:
    records = read_manifest(manifest)
    class_names = tuple(sorted({record.label for record in records}))
    label_index = {label: index for index, label in enumerate(class_names)}
    length = max(1, round(policy.window_seconds * policy.target_rate_hz))
    stride = max(1, round(policy.stride_seconds * policy.target_rate_hz))
    values: dict[Split, list[np.ndarray]] = {split: [] for split in SPLITS}
    labels: dict[Split, list[int]] = {split: [] for split in SPLITS}
    groups: dict[Split, list[str]] = {split: [] for split in SPLITS}
    group_splits: dict[str, Split] = {}
    for record in records:
        split = record.split or assigned_split(record.group, split_seed)
        previous = group_splits.setdefault(record.group, split)
        if previous != split:
            raise ValueError(f"group {record.group!r} crosses Wave-2 splits")
        signal = _resample(
            _load_signal(record), record.sample_rate_hz, policy.target_rate_hz
        )
        for window in _windows(signal, length, stride):
            values[split].append(window)
            labels[split].append(label_index[record.label])
            groups[split].append(record.group)
    if any(not values[split] for split in SPLITS):
        raise ValueError("Wave-2 group split produced an empty partition")
    tensors = {
        split: torch.from_numpy(np.stack(values[split])).to(torch.float32)
        for split in SPLITS
    }
    train = tensors["train"]
    mean = train.mean(dim=(0, 1), keepdim=True)
    scale = train.std(dim=(0, 1), keepdim=True).clamp_min(1.0e-6)
    tensors = {split: (tensor - mean) / scale for split, tensor in tensors.items()}
    task = ExternalTask(
        name=policy.name,
        objective="multiclass",
        train_inputs=tensors["train"],
        train_targets=torch.tensor(labels["train"], dtype=torch.long),
        validation_inputs=tensors["validation"],
        validation_targets=torch.tensor(labels["validation"], dtype=torch.long),
        test_inputs=tensors["test"],
        test_targets=torch.tensor(labels["test"], dtype=torch.long),
        output_dim=len(class_names),
        class_names=class_names,
        train_groups=tuple(groups["train"]),
        validation_groups=tuple(groups["validation"]),
        test_groups=tuple(groups["test"]),
        sample_rate_hz=float(policy.target_rate_hz),
    )
    save_prepared_task(task, output_root / f"{policy.name}.pt")
    write_external_selection_task(
        task, output_root / "selection-only" / f"{policy.name}.pt"
    )
    return task


def task_summary(task: ExternalTask) -> dict[str, object]:
    return {
        "dataset": task.name,
        "input_dim": task.input_dim,
        "sequence_length": task.sequence_length,
        "output_dim": task.output_dim,
        "split_counts": {
            "train": int(task.train_inputs.shape[0]),
            "validation": int(task.validation_inputs.shape[0]),
            "test": int(task.test_inputs.shape[0]),
        },
        "groups": {
            "train": len(set(task.train_groups)),
            "validation": len(set(task.validation_groups)),
            "test": len(set(task.test_groups)),
        },
    }


def write_summary(task: ExternalTask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task_summary(task), indent=2, sort_keys=True) + "\n")


__all__ = [
    "POLICIES",
    "ManifestRecord",
    "Wave2Policy",
    "assigned_split",
    "prepare_manifest_task",
    "read_manifest",
    "task_summary",
    "write_summary",
]
