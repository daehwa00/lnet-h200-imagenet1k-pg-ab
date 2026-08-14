from __future__ import annotations

import math
import os
import re
import stat
from typing import TYPE_CHECKING, Final
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import torch
from torch import Tensor

from .pac_types import UCRDataset

if TYPE_CHECKING:
    from pathlib import Path

RAW_SPLIT_URLS: Final[dict[str, dict[str, str]]] = {
    "ECG5000": {
        "TRAIN": "https://raw.githubusercontent.com/tejaslodaya/timeseries-clustering-vae/master/data/ECG5000/ECG5000_TRAIN",
        "TEST": "https://raw.githubusercontent.com/tejaslodaya/timeseries-clustering-vae/master/data/ECG5000/ECG5000_TEST",
    },
    "FordA": {
        "TRAIN": "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv",
        "TEST": "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv",
    },
}
UCR_ARCHIVE_URL: Final[str] = (
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
)
UCR_ARCHIVE_NAME: Final[str] = "UCRArchive_2018.zip"
UCR_ARCHIVE_PASSWORD: Final[bytes] = b"someone"
MAX_DOWNLOAD_BYTES: Final[int] = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES: Final[int] = 256 * 1024 * 1024
DATASET_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class UCRDatasetUnavailableError(RuntimeError):
    def __init__(self, dataset: str, root: Path) -> None:
        super().__init__(f"UCR dataset {dataset!r} is unavailable under {root}")


def ensure_ucr_dataset(
    dataset: str,
    root: Path,
    *,
    allow_download: bool,
    require_train_label_space: bool = True,
) -> UCRDataset:
    _validate_dataset_name(dataset)
    try:
        return load_ucr_dataset(
            dataset,
            root,
            require_train_label_space=require_train_label_space,
        )
    except FileNotFoundError as missing:
        if not allow_download:
            raise UCRDatasetUnavailableError(dataset, root) from missing
    try:
        _download_ucr_zip(dataset, root)
    except UCRDatasetUnavailableError:
        try:
            download_ucr_archive_files(dataset, root)
        except UCRDatasetUnavailableError:
            _download_raw_split_files(dataset, root)
    return load_ucr_dataset(
        dataset,
        root,
        require_train_label_space=require_train_label_space,
    )


def ensure_ucr_train_only(dataset: str, root: Path, *, allow_download: bool) -> UCRDataset:
    """Ensure and load official TRAIN without opening the official TEST payload."""
    _validate_dataset_name(dataset)
    try:
        return load_ucr_train_only(dataset, root)
    except FileNotFoundError as missing:
        if not allow_download:
            raise UCRDatasetUnavailableError(dataset, root) from missing
    try:
        _download_ucr_train_zip(dataset, root)
    except UCRDatasetUnavailableError:
        try:
            download_ucr_archive_train_file(dataset, root)
        except UCRDatasetUnavailableError:
            _download_raw_train_file(dataset, root)
    return load_ucr_train_only(dataset, root)


def load_ucr_dataset(
    dataset: str,
    root: Path,
    *,
    require_train_label_space: bool = True,
) -> UCRDataset:
    _validate_dataset_name(dataset)
    dataset_dir = root / dataset
    train_path = _find_split(dataset_dir, dataset, "TRAIN")
    test_path = _find_split(dataset_dir, dataset, "TEST")
    train_values, train_labels, labels = _parse_split(train_path, ())
    test_values, test_labels, observed_labels = _parse_split(test_path, labels)
    if require_train_label_space and observed_labels != labels:
        unseen = observed_labels[len(labels) :]
        message = (
            f"UCR TEST for {dataset!r} contains labels absent from official TRAIN: "
            f"{list(unseen)}"
        )
        raise ValueError(message)
    return UCRDataset(
        name=dataset,
        train_inputs=train_values.unsqueeze(-1),
        train_labels=train_labels,
        test_inputs=test_values.unsqueeze(-1),
        test_labels=test_labels,
        class_count=len(labels if require_train_label_space else observed_labels),
    )


def load_ucr_train_only(dataset: str, root: Path) -> UCRDataset:
    """Load official TRAIN without opening or parsing the official TEST file."""
    _validate_dataset_name(dataset)
    train_path = _find_split(root / dataset, dataset, "TRAIN")
    train_values, train_labels, labels = _parse_split(train_path, ())
    empty_inputs = train_values[:0].unsqueeze(-1)
    return UCRDataset(
        name=dataset,
        train_inputs=train_values.unsqueeze(-1),
        train_labels=train_labels,
        test_inputs=empty_inputs,
        test_labels=train_labels[:0],
        class_count=len(labels),
    )


def write_tiny_ucr_fixture(root: Path) -> None:
    dataset_dir = root / "Tiny"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "Tiny_TRAIN.txt").write_text("0 0.0 1.0 0.0\n1 1.0 0.0 1.0\n", encoding="utf-8")
    (dataset_dir / "Tiny_TEST.txt").write_text("0 0.1 1.1 0.1\n1 1.1 0.1 1.1\n", encoding="utf-8")


def _find_split(dataset_dir: Path, dataset: str, split: str) -> Path:
    candidates = (
        dataset_dir / f"{dataset}_{split}.txt",
        dataset_dir / f"{dataset}_{split}.tsv",
        dataset_dir / f"{dataset}_{split}.ts",
        dataset_dir / f"{dataset}_{split}",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


def _parse_split(
    path: Path, known_labels: tuple[str, ...]
) -> tuple[Tensor, Tensor, tuple[str, ...]]:
    values: list[list[float]] = []
    labels = list(known_labels)
    encoded: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(("@", "#")):
            continue
        label, series = _parse_line(line)
        if not label or not series or not all(math.isfinite(value) for value in series):
            raise ValueError(f"invalid UCR row at {path}:{line_number}")
        if values and len(series) != len(values[0]):
            raise ValueError(f"UCR rows have inconsistent lengths at {path}:{line_number}")
        if label not in labels:
            labels.append(label)
        encoded.append(labels.index(label))
        values.append(series)
    if not values:
        raise ValueError(f"UCR split is empty: {path}")
    return (
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(encoded, dtype=torch.long),
        tuple(labels),
    )


def _parse_line(line: str) -> tuple[str, list[float]]:
    if ":" in line:
        parts = line.split(":")
        return parts[-1].strip(), [float(value) for value in parts[0].split(",") if value]
    parts = line.replace(",", " ").split()
    return parts[0], [float(value) for value in parts[1:]]


def _validate_dataset_name(dataset: str) -> None:
    if not isinstance(dataset, str) or DATASET_NAME_PATTERN.fullmatch(dataset) is None:
        raise ValueError(f"invalid UCR dataset name: {dataset!r}")


def _download_url(url: str, target: Path, *, limit: int) -> None:
    """Download a fixed, HTTPS URL with bounded, atomic writes."""
    if not url.startswith("https://"):
        raise ValueError("UCR downloads require HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as handle:  # noqa: S310
            advertised = response.headers.get("Content-Length")
            if advertised is not None and int(advertised) > limit:
                raise ValueError(f"download exceeds {limit} bytes: {url}")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise ValueError(f"download exceeds {limit} bytes: {url}")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _archive_member_bytes(zip_file: ZipFile, member: str) -> bytes:
    """Read one allowlisted regular member without extracting arbitrary paths."""
    info = zip_file.getinfo(member)
    mode = (info.external_attr >> 16) & 0xFFFF
    if info.is_dir() or stat.S_ISLNK(mode):
        raise ValueError(f"archive member is not a regular file: {member}")
    if info.file_size < 0 or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
        raise ValueError(f"archive member is too large: {member}")
    return zip_file.read(info, pwd=UCR_ARCHIVE_PASSWORD)


def _write_archive_split(zip_file: ZipFile, member: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_archive_member_bytes(zip_file, member))
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _find_split_member(zip_file: ZipFile, dataset: str, split: str) -> str:
    prefix = f"{dataset}_{split}"
    candidates = [
        info.filename
        for info in zip_file.infolist()
        if info.filename.rsplit("/", 1)[-1].startswith(prefix)
        and _split_suffix(info.filename) in {".txt", ".tsv", ".ts", ""}
    ]
    if len(candidates) != 1:
        raise KeyError(f"archive does not contain exactly one {dataset}_{split} member")
    return candidates[0]


def _download_ucr_zip(dataset: str, root: Path) -> None:
    _validate_dataset_name(dataset)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{dataset}.zip"
    try:
        if not archive.exists():
            _download_url(
                f"https://www.timeseriesclassification.com/aeon-toolkit/{dataset}.zip",
                archive,
                limit=MAX_DOWNLOAD_BYTES,
            )
        with ZipFile(archive) as zip_file:
            for split in ("TRAIN", "TEST"):
                member = _find_split_member(zip_file, dataset, split)
                suffix = _split_suffix(member)
                _write_archive_split(
                    zip_file,
                    member,
                    root / dataset / f"{dataset}_{split}{suffix}",
                )
    except (KeyError, ValueError, URLError, BadZipFile, OSError) as error:
        raise UCRDatasetUnavailableError(dataset, root) from error


def _download_ucr_train_zip(dataset: str, root: Path) -> None:
    """Extract only the TRAIN member from the per-dataset archive."""
    _validate_dataset_name(dataset)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{dataset}.zip"
    try:
        if not archive.exists():
            _download_url(
                f"https://www.timeseriesclassification.com/aeon-toolkit/{dataset}.zip",
                archive,
                limit=MAX_DOWNLOAD_BYTES,
            )
        with ZipFile(archive) as zip_file:
            member = _find_train_member(zip_file, dataset)
            suffix = _split_suffix(member)
            _write_archive_split(zip_file, member, root / dataset / f"{dataset}_TRAIN{suffix}")
    except (KeyError, ValueError, URLError, BadZipFile, OSError) as error:
        raise UCRDatasetUnavailableError(dataset, root) from error


def _download_raw_split_files(dataset: str, root: Path) -> None:
    _validate_dataset_name(dataset)
    split_urls = RAW_SPLIT_URLS.get(dataset)
    if split_urls is None:
        raise UCRDatasetUnavailableError(dataset, root)
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split, url in split_urls.items():
        suffix = ".tsv" if url.endswith(".tsv") else ""
        _download_url(url, dataset_dir / f"{dataset}_{split}{suffix}", limit=MAX_DOWNLOAD_BYTES)


def _download_raw_train_file(dataset: str, root: Path) -> None:
    _validate_dataset_name(dataset)
    split_urls = RAW_SPLIT_URLS.get(dataset)
    if split_urls is None:
        raise UCRDatasetUnavailableError(dataset, root)
    url = split_urls["TRAIN"]
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".tsv" if url.endswith(".tsv") else ""
    _download_url(url, dataset_dir / f"{dataset}_TRAIN{suffix}", limit=MAX_DOWNLOAD_BYTES)


def download_ucr_archive_files(dataset: str, root: Path) -> None:
    _validate_dataset_name(dataset)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / UCR_ARCHIVE_NAME
    if not archive.exists():
        _download_url(UCR_ARCHIVE_URL, archive, limit=MAX_DOWNLOAD_BYTES)
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(archive) as zip_file:
            for split in ("TRAIN", "TEST"):
                member = f"UCRArchive_2018/{dataset}/{dataset}_{split}.tsv"
                _write_archive_split(zip_file, member, dataset_dir / f"{dataset}_{split}.tsv")
    except (KeyError, RuntimeError, ValueError, BadZipFile, OSError) as error:
        raise UCRDatasetUnavailableError(dataset, root) from error


def download_ucr_archive_train_file(dataset: str, root: Path) -> None:
    """Extract official TRAIN only; the TEST archive member is never read."""
    _validate_dataset_name(dataset)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / UCR_ARCHIVE_NAME
    if not archive.exists():
        _download_url(UCR_ARCHIVE_URL, archive, limit=MAX_DOWNLOAD_BYTES)
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(archive) as zip_file:
            member = f"UCRArchive_2018/{dataset}/{dataset}_TRAIN.tsv"
            _write_archive_split(zip_file, member, dataset_dir / f"{dataset}_TRAIN.tsv")
    except (KeyError, RuntimeError, ValueError, BadZipFile, OSError) as error:
        raise UCRDatasetUnavailableError(dataset, root) from error


def _find_train_member(zip_file: ZipFile, dataset: str) -> str:
    return _find_split_member(zip_file, dataset, "TRAIN")


def _split_suffix(member: str) -> str:
    name = member.rsplit("/", 1)[-1]
    for suffix in (".txt", ".tsv", ".ts"):
        if name.endswith(suffix):
            return suffix
    return ""
