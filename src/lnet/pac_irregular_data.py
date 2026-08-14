"""Adapters for the fixed Raindrop irregular-series benchmark splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import torch
from torch import Tensor

from .pac_time_normalization import (
    fit_characteristic_time_scale,
    normalize_time_delta,
)

if TYPE_CHECKING:
    from pathlib import Path

IrregularDatasetName = Literal["physionet-2019", "pam"]

RAINDROP_FIGSHARE: Final = {
    "physionet-2019": {
        "article": "19514338",
        "file": "34683070",
        "directory": "P19data",
    },
    "pam": {
        "article": "19514347",
        "file": "34683103",
        "directory": "PAMAP2data",
    },
}


@dataclass(frozen=True, slots=True)
class IrregularSplit:
    values: Tensor
    observed: Tensor
    interval_delta: Tensor
    time_delta: Tensor
    valid: Tensor
    labels: Tensor
    static: Tensor | None = None

    def index_select(self, indices: Tensor) -> IrregularSplit:
        return IrregularSplit(
            values=self.values[indices],
            observed=self.observed[indices],
            interval_delta=self.interval_delta[indices],
            time_delta=self.time_delta[indices],
            valid=self.valid[indices],
            labels=self.labels[indices],
            static=None if self.static is None else self.static[indices],
        )


@dataclass(frozen=True, slots=True)
class IrregularTask:
    name: IrregularDatasetName
    train: IrregularSplit
    validation: IrregularSplit
    test: IrregularSplit
    output_dim: int
    source_article: str
    characteristic_time_scale: float = 1.0


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load split indices without executing NumPy pickle payloads.

    New datasets store variable-length splits in a numeric ``.npz`` with the
    keys ``train``, ``validation`` and ``test``.  A legacy numeric ``.npy``
    with shape ``(3, n)`` remains accepted for fixtures with equal-sized
    partitions.  Object arrays are deliberately rejected: NumPy implements
    those arrays with pickle deserialization, which is code execution for an
    untrusted data directory.
    """
    source = path if path.is_file() else path.with_suffix(".npz")
    if not source.is_file():
        raise FileNotFoundError(path)
    if source.suffix == ".npz":
        try:
            with np.load(source, allow_pickle=False) as archive:
                required = ("train", "validation", "test")
                if set(archive.files) != set(required):
                    raise ValueError(
                        f"safe split archive {source} must contain exactly {required}"
                    )
                raw_parts = tuple(np.asarray(archive[name]) for name in required)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid safe split archive: {source}") from error
    else:
        try:
            payload = np.load(source, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"unsafe or invalid split file {source}; convert object arrays to .npz"
            ) from error
        if not isinstance(payload, np.ndarray) or payload.ndim != 2 or payload.shape[0] != 3:
            raise ValueError(
                f"legacy numeric split file {source} must have shape (3, n); "
                "variable-length splits must use .npz"
            )
        raw_parts = tuple(np.asarray(payload[index]) for index in range(3))
    if any(
        part.ndim != 1 or not np.issubdtype(part.dtype, np.integer)
        for part in raw_parts
    ):
        raise ValueError(f"split indices in {source} must be one-dimensional integers")
    parts = tuple(np.asarray(part, dtype=np.int64) for part in raw_parts)
    if len({int(index) for part in parts for index in part}) != sum(map(len, parts)):
        message = f"split indices overlap in {source}"
        raise ValueError(message)
    if any(int(part.min(initial=0)) < 0 for part in parts):
        raise ValueError(f"split indices in {source} must be non-negative")
    return parts


def _load_numeric_array(path: Path) -> np.ndarray:
    """Load an ndarray while rejecting object/pickle-backed files."""
    try:
        payload = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"unsafe or invalid NumPy input {path}; object arrays are not supported"
        ) from error
    if not isinstance(payload, np.ndarray) or payload.dtype.kind == "O":
        raise ValueError(f"NumPy input {path} must be a numeric ndarray")
    return payload


def _load_p19_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the converted P19 numeric bundle, never a pickled record list."""
    npz_path = root / "processed_data" / "PT_dict_list_6.npz"
    legacy_path = root / "processed_data" / "PT_dict_list_6.npy"
    source = npz_path if npz_path.is_file() else legacy_path
    if not source.is_file():
        raise FileNotFoundError(npz_path)
    if source.suffix != ".npz":
        # Loading with allow_pickle=False is intentional: it gives a clear
        # conversion error for the historical object-array distribution.
        _load_numeric_array(source)
        raise ValueError(
            f"P19 record bundle {source} is not a safe numeric archive; "
            "convert it to PT_dict_list_6.npz with arr, length, time, "
            "and extended_static arrays"
        )
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = ("arr", "length", "time", "extended_static")
            if set(archive.files) != set(required):
                raise ValueError(
                    f"safe P19 archive {source} must contain exactly {required}"
                )
            arrays = tuple(np.asarray(archive[name]) for name in required)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid safe P19 archive: {source}") from error
    if any(array.dtype.kind == "O" for array in arrays):
        raise ValueError(f"safe P19 archive {source} contains an object array")
    values, lengths, timestamps, static = arrays
    if values.ndim != 3 or lengths.ndim != 1 or timestamps.ndim != 2 or static.ndim != 2:
        raise ValueError("P19 safe arrays have incompatible ranks")
    if not (
        values.shape[0] == lengths.shape[0] == timestamps.shape[0] == static.shape[0]
        and values.shape[1] == timestamps.shape[1]
    ):
        raise ValueError("P19 safe arrays have incompatible sample counts or lengths")
    if not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("P19 lengths must be integers")
    lengths = np.asarray(lengths, dtype=np.int64)
    if np.any(lengths < 1) or np.any(lengths > values.shape[1]):
        raise ValueError("P19 sequence lengths are outside the values array")
    for name, array in (("values", values), ("timestamps", timestamps), ("static", static)):
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"P19 {name} must contain finite numeric values")
    return (
        np.asarray(values, dtype=np.float32),
        lengths,
        np.asarray(timestamps, dtype=np.float32),
        np.asarray(static, dtype=np.float32),
    )


def _feature_delta(observed: Tensor, interval_delta: Tensor, valid: Tensor) -> Tensor:
    """Return time since each feature's previous observation."""
    batch, steps, features = observed.shape
    delta = torch.zeros((batch, steps, features), dtype=torch.float32)
    for step in range(1, steps):
        delta[:, step] = torch.where(
            observed[:, step - 1],
            interval_delta[:, step],
            delta[:, step - 1] + interval_delta[:, step],
        )
    return delta * valid.to(torch.float32)


def _partition(
    values: Tensor,
    observed: Tensor,
    interval_delta: Tensor,
    valid: Tensor,
    labels: Tensor,
    indices: np.ndarray,
    static: Tensor | None,
) -> IrregularSplit:
    selected = torch.from_numpy(indices)
    return IrregularSplit(
        values=values[selected],
        observed=observed[selected],
        interval_delta=interval_delta[selected],
        time_delta=_feature_delta(
            observed[selected],
            interval_delta[selected],
            valid[selected],
        ),
        valid=valid[selected],
        labels=labels[selected],
        static=None if static is None else static[selected],
    )


def load_raindrop_task(
    name: IrregularDatasetName,
    root: Path,
    *,
    split: int = 1,
) -> IrregularTask:
    if split not in range(1, 6):
        message = "Raindrop split must be between 1 and 5"
        raise ValueError(message)
    if name == "physionet-2019":
        return _load_p19(root / "P19data", split)
    return _load_pam(root / "PAMAP2data", split)


def _load_p19(root: Path, split: int) -> IrregularTask:
    values_array, lengths_array, timestamps_array, static_array = _load_p19_arrays(root)
    outcomes = _load_numeric_array(root / "processed_data" / "arr_outcomes_6.npy")
    indices = _load_split(root / "splits" / f"phy19_split{split}_new.npy")
    values = torch.from_numpy(values_array)
    lengths = torch.from_numpy(lengths_array)
    valid = torch.arange(values.shape[1])[None, :, None] < lengths[:, None, None]
    observed = values.ne(0) & valid
    timestamps = torch.from_numpy(timestamps_array)
    interval_delta = torch.zeros_like(timestamps)
    interval_delta[:, 0] = timestamps[:, 0]
    interval_delta[:, 1:] = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(0)
    interval_delta *= valid.to(interval_delta.dtype)
    static = torch.from_numpy(static_array)
    if outcomes.ndim < 2 or outcomes.shape[0] != values.shape[0]:
        raise ValueError("P19 outcomes do not match the safe record bundle")
    labels = torch.from_numpy(outcomes[:, 0].astype(np.int64, copy=False))
    parts = tuple(np.asarray(value, dtype=np.int64) for value in indices)
    characteristic_time_scale = fit_characteristic_time_scale(
        interval_delta[torch.from_numpy(parts[0])],
        valid[torch.from_numpy(parts[0])],
    )
    interval_delta = normalize_time_delta(
        interval_delta,
        characteristic_time_scale,
        valid,
    )
    return IrregularTask(
        name="physionet-2019",
        train=_partition(values, observed, interval_delta, valid, labels, parts[0], static),
        validation=_partition(values, observed, interval_delta, valid, labels, parts[1], static),
        test=_partition(values, observed, interval_delta, valid, labels, parts[2], static),
        output_dim=2,
        source_article=RAINDROP_FIGSHARE["physionet-2019"]["article"],
        characteristic_time_scale=characteristic_time_scale,
    )


def _load_pam(root: Path, split: int) -> IrregularTask:
    values_array = _load_numeric_array(root / "processed_data" / "PTdict_list.npy")
    outcomes = _load_numeric_array(root / "processed_data" / "arr_outcomes.npy")
    if values_array.ndim != 3 or outcomes.ndim < 2 or outcomes.shape[0] != values_array.shape[0]:
        raise ValueError("PAM arrays have incompatible shapes")
    indices = _load_split(root / "splits" / f"PAMAP2_split_{split}.npy")
    values = torch.from_numpy(values_array.astype(np.float32, copy=False))
    observed = values.ne(0)
    valid = torch.ones((*values.shape[:2], 1), dtype=torch.bool)
    interval_delta = torch.ones((*values.shape[:2], 1), dtype=torch.float32)
    interval_delta[:, 0] = 0
    labels = torch.from_numpy(outcomes[:, 0].astype(np.int64, copy=False))
    parts = tuple(np.asarray(value, dtype=np.int64) for value in indices)
    characteristic_time_scale = fit_characteristic_time_scale(
        interval_delta[torch.from_numpy(parts[0])],
        valid[torch.from_numpy(parts[0])],
    )
    interval_delta = normalize_time_delta(
        interval_delta,
        characteristic_time_scale,
        valid,
    )
    return IrregularTask(
        name="pam",
        train=_partition(values, observed, interval_delta, valid, labels, parts[0], None),
        validation=_partition(values, observed, interval_delta, valid, labels, parts[1], None),
        test=_partition(values, observed, interval_delta, valid, labels, parts[2], None),
        output_dim=8,
        source_article=RAINDROP_FIGSHARE["pam"]["article"],
        characteristic_time_scale=characteristic_time_scale,
    )


__all__ = [
    "RAINDROP_FIGSHARE",
    "IrregularDatasetName",
    "IrregularSplit",
    "IrregularTask",
    "load_raindrop_task",
]
