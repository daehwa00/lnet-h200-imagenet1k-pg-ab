# ruff: noqa: EM102, TRY003
"""Minimal EDF/EDF+ reader used by the public Wave-2 EEG adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def _number(raw: bytes, default: float = 0.0) -> float:
    text = raw.decode("ascii", errors="ignore").strip()
    return float(text) if text else default


@dataclass(frozen=True, slots=True)
class EdfHeader:
    header_bytes: int
    records: int
    record_seconds: float
    labels: tuple[str, ...]
    physical_min: tuple[float, ...]
    physical_max: tuple[float, ...]
    digital_min: tuple[float, ...]
    digital_max: tuple[float, ...]
    samples_per_record: tuple[int, ...]


def read_header(path: Path) -> EdfHeader:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"truncated EDF header: {path}")
        header_bytes = int(_number(fixed[184:192]))
        records = int(_number(fixed[236:244], -1))
        record_seconds = _number(fixed[244:252], 1.0)
        signals = int(_number(fixed[252:256]))
        variable = handle.read(header_bytes - 256)
    cursor = 0

    def fields(width: int) -> tuple[bytes, ...]:
        nonlocal cursor
        result = tuple(
            variable[cursor + width * index : cursor + width * (index + 1)]
            for index in range(signals)
        )
        cursor += width * signals
        return result

    labels = tuple(value.decode("ascii", errors="ignore").strip() for value in fields(16))
    fields(80)
    fields(8)
    physical_min = tuple(_number(value) for value in fields(8))
    physical_max = tuple(_number(value) for value in fields(8))
    digital_min = tuple(_number(value) for value in fields(8))
    digital_max = tuple(_number(value) for value in fields(8))
    fields(80)
    samples = tuple(int(_number(value)) for value in fields(8))
    return EdfHeader(
        header_bytes,
        records,
        record_seconds,
        labels,
        physical_min,
        physical_max,
        digital_min,
        digital_max,
        samples,
    )


def read_signals(path: Path, labels: tuple[str, ...]) -> tuple[np.ndarray, float]:
    header = read_header(path)
    indices = []
    for label in labels:
        matches = [
            index
            for index, observed in enumerate(header.labels)
            if observed.casefold() == label.casefold()
        ]
        if not matches:
            raise ValueError(f"{path} does not contain EDF signal {label!r}")
        indices.append(matches[0])
    sample_counts = {header.samples_per_record[index] for index in indices}
    if len(sample_counts) != 1:
        raise ValueError(f"{path} selected EDF signals have different sample rates")
    samples = sample_counts.pop()
    record_bytes = 2 * sum(header.samples_per_record)
    records = header.records
    available_records = (path.stat().st_size - header.header_bytes) // record_bytes
    records = available_records if records < 0 else min(records, available_records)
    output = np.empty((records * samples, len(indices)), dtype=np.float32)
    offsets = np.cumsum((0, *header.samples_per_record))
    with path.open("rb") as handle:
        handle.seek(header.header_bytes)
        for record in range(records):
            raw = np.frombuffer(handle.read(record_bytes), dtype="<i2")
            for column, index in enumerate(indices):
                values = raw[offsets[index] : offsets[index + 1]].astype(np.float32)
                digital_range = header.digital_max[index] - header.digital_min[index]
                physical_range = header.physical_max[index] - header.physical_min[index]
                output[record * samples : (record + 1) * samples, column] = (
                    (values - header.digital_min[index]) * physical_range / digital_range
                    + header.physical_min[index]
                )
    return output, samples / header.record_seconds


def read_annotations(path: Path) -> tuple[tuple[float, float, str], ...]:
    header = read_header(path)
    annotation = next(
        (
            index
            for index, label in enumerate(header.labels)
            if "annotation" in label.casefold()
        ),
        None,
    )
    if annotation is None:
        raise ValueError(f"{path} has no EDF+ annotation signal")
    record_bytes = 2 * sum(header.samples_per_record)
    offsets = np.cumsum((0, *header.samples_per_record))
    records = header.records
    available_records = (path.stat().st_size - header.header_bytes) // record_bytes
    records = available_records if records < 0 else min(records, available_records)
    chunks = []
    with path.open("rb") as handle:
        handle.seek(header.header_bytes)
        for _ in range(records):
            raw = handle.read(record_bytes)
            start = 2 * offsets[annotation]
            stop = 2 * offsets[annotation + 1]
            chunks.append(raw[start:stop])
    text = b"".join(chunks).replace(b"\x00", b"\n").decode("latin1", errors="ignore")
    events = []
    for block in text.splitlines():
        parts = block.split("\x14")
        timing = parts[0].split("\x15")
        match = re.match(r"([+-]?\d+(?:\.\d+)?)", timing[0])
        if match is None:
            continue
        onset = float(match.group(1))
        duration = float(timing[1]) if len(timing) > 1 and timing[1] else 0.0
        for annotation_label in parts[1:]:
            cleaned = annotation_label.strip()
            if cleaned:
                events.append((onset, duration, cleaned))
    return tuple(events)


__all__ = ["EdfHeader", "read_annotations", "read_header", "read_signals"]
