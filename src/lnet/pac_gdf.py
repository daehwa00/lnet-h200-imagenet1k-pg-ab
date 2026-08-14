# ruff: noqa: EM102, TC003, TRY003
"""Minimal GDF 1.x reader for the public BCI Competition IV-2a files."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class GDFRecording:
    signals: np.ndarray
    sample_rate_hz: float
    event_positions: np.ndarray
    event_types: np.ndarray


def read_gdf(path: Path) -> GDFRecording:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if not fixed.startswith(b"GDF 1."):
            raise ValueError(f"unsupported GDF version in {path}")
        header_bytes = struct.unpack_from("<H", fixed, 184)[0] * 256
        records = struct.unpack_from("<q", fixed, 236)[0]
        duration_numerator = struct.unpack_from("<I", fixed, 244)[0]
        duration_denominator = struct.unpack_from("<I", fixed, 248)[0]
        signal_count = struct.unpack_from("<H", fixed, 252)[0]
        handle.seek(256 + signal_count * (16 + 80 + 8 + 8 * 4 + 80))
        samples_per_record = np.frombuffer(
            handle.read(4 * signal_count), dtype="<i4"
        ).copy()
        data_types = np.frombuffer(handle.read(4 * signal_count), dtype="<u4").copy()
    if not np.all(samples_per_record == 1) or not np.all(data_types == 3):
        raise ValueError(f"{path} does not use the expected interleaved int16 layout")
    data_values = records * signal_count
    raw = np.memmap(
        path,
        mode="r",
        dtype="<i2",
        offset=header_bytes,
        shape=(records, signal_count),
    )
    event_offset = header_bytes + data_values * 2
    with path.open("rb") as handle:
        handle.seek(event_offset)
        mode_raw = handle.read(1)
        if not mode_raw:
            raise ValueError(f"{path} has no GDF event table")
        mode = mode_raw[0]
        event_count = int.from_bytes(handle.read(3), "little")
        event_rate = struct.unpack("<f", handle.read(4))[0]
        positions = np.frombuffer(
            handle.read(4 * event_count), dtype="<u4"
        ).astype(np.int64)
        event_types = np.frombuffer(
            handle.read(2 * event_count), dtype="<u2"
        ).astype(np.int64)
    if mode not in {1, 3}:
        raise ValueError(f"unsupported GDF event mode {mode} in {path}")
    sample_rate = duration_denominator / duration_numerator
    if not np.isclose(event_rate, sample_rate):
        raise ValueError(f"event and signal rates disagree in {path}")
    return GDFRecording(
        signals=raw,
        sample_rate_hz=float(sample_rate),
        event_positions=positions - 1,
        event_types=event_types,
    )


__all__ = ["GDFRecording", "read_gdf"]
