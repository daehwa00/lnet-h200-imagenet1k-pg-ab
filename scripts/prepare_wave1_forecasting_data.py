# ruff: noqa: EM102, T201, TRY003
"""Download, seal, and audit the five Wave-1 forecasting datasets.

The source revisions are immutable.  Selection artifacts are written before
full artifacts so Stage 1/2 workers never need a container that holds TEST.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import torch

from lnet.pac_external_tasks import (
    ExternalDatasetName,
    ExternalTask,
    _selection_split_sha256,  # pyright: ignore[reportPrivateUsage]
    load_external_selection_task,
    load_forecasting_csv,
    load_prepared_task,
    save_prepared_task,
    write_external_selection_task,
)

SCHEMA = "alphabet.wave1_forecasting_data.v1"
ETT_REVISION = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
THUML_REVISION = "2b66e59ee19dac8f6f19fb5d4997f289fdfea357"
ETT_RAW_BASE = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{ETT_REVISION}"
THUML_DATA_BASE = (
    f"https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/{THUML_REVISION}"
)


@dataclass(frozen=True, slots=True)
class ForecastSource:
    dataset: ExternalDatasetName
    filename: str
    url: str
    sha256: str
    protocol: str


SOURCES = (
    ForecastSource(
        "etth1",
        "etth1.csv",
        f"{ETT_RAW_BASE}/ETT-small/ETTh1.csv",
        "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
        "M, context=96, horizon=96, fixed 12/4/4-month hourly split",
    ),
    ForecastSource(
        "etth2",
        "etth2.csv",
        f"{ETT_RAW_BASE}/ETT-small/ETTh2.csv",
        "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
        "M, context=96, horizon=96, fixed 12/4/4-month hourly split",
    ),
    ForecastSource(
        "traffic",
        "traffic.csv",
        f"{THUML_DATA_BASE}/traffic/traffic.csv",
        "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16",
        "M, context=96, horizon=96, chronological 70/10/20 split",
    ),
    ForecastSource(
        "ili",
        "ili.csv",
        f"{THUML_DATA_BASE}/illness/national_illness.csv",
        "93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a",
        "M, context=36, horizon=24, chronological 70/10/20 split",
    ),
    ForecastSource(
        "exchange-rate",
        "exchange-rate.csv",
        f"{THUML_DATA_BASE}/exchange_rate/exchange_rate.csv",
        "48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842",
        "M, context=96, horizon=96, chronological 70/10/20 split",
    ),
)
SOURCE_BY_DATASET = {source.dataset: source for source in SOURCES}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_sources_verified(rows: list[dict[str, object]]) -> bool:
    return (
        len(rows) == len(SOURCES)
        and {
            str(row["dataset"])
            for row in rows
            if (
                str(row["dataset"]) in SOURCE_BY_DATASET
                and row.get("raw_sha256")
                == SOURCE_BY_DATASET[str(row["dataset"])].sha256
            )
        }
        == {source.dataset for source in SOURCES}
    )


def download_source(source: ForecastSource, raw_root: Path) -> Path:
    destination = raw_root / source.filename
    if destination.is_file():
        observed = file_sha256(destination)
        if observed != source.sha256:
            message = (
                f"existing source checksum mismatch for {source.dataset}: "
                f"{observed} != {source.sha256}"
            )
            raise RuntimeError(message)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    request = Request(  # noqa: S310 - manifest permits immutable HTTPS URLs only
        source.url,
        headers={"User-Agent": "alphabet-wave1-preparer/1"},
    )
    try:
        with (
            urlopen(  # noqa: S310 - manifest permits immutable HTTPS URLs only
                request,
                timeout=120,
            ) as response,
            temporary.open("wb") as output,
        ):
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                output.write(chunk)
        observed = file_sha256(temporary)
        if observed != source.sha256:
            message = (
                f"download checksum mismatch for {source.dataset}: "
                f"{observed} != {source.sha256}"
            )
            raise RuntimeError(message)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _load_fresh_task(source: ForecastSource, data_root: Path) -> ExternalTask:
    split_kind = "etth" if source.dataset in {"etth1", "etth2"} else "ratio"
    return load_forecasting_csv(
        data_root / "forecasting" / source.filename,
        name=source.dataset,
        context_length=36 if source.dataset == "ili" else 96,
        prediction_length=24 if source.dataset == "ili" else 96,
        split_kind=split_kind,
    )


def _assert_full_identity(expected: ExternalTask, actual: ExternalTask) -> None:
    scalar_fields = (
        "name",
        "objective",
        "output_dim",
        "class_names",
        "train_groups",
        "validation_groups",
        "test_groups",
        "sample_rate_hz",
        "input_encoding",
        "vocab_size",
    )
    if any(getattr(expected, field) != getattr(actual, field) for field in scalar_fields):
        raise RuntimeError(f"prepared full task metadata is stale for {expected.name}")
    if _selection_split_sha256(expected) != _selection_split_sha256(actual):
        raise RuntimeError(f"prepared TRAIN/validation task is stale for {expected.name}")
    if not torch.equal(expected.test_inputs, actual.test_inputs) or not torch.equal(
        expected.test_targets,
        actual.test_targets,
    ):
        raise RuntimeError(f"prepared TEST task is stale for {expected.name}")
    for name in ("time_delta", "observation_mask", "valid_mask"):
        left = getattr(expected.test_metadata, name)
        right = getattr(actual.test_metadata, name)
        if (left is None) != (right is None) or (
            left is not None and right is not None and not torch.equal(left, right)
        ):
            raise RuntimeError(f"prepared TEST metadata is stale for {expected.name}")


def prepare_dataset(source: ForecastSource, data_root: Path) -> dict[str, object]:
    raw = download_source(source, data_root / "forecasting")
    task = _load_fresh_task(source, data_root)
    selection_digest = _selection_split_sha256(task)
    selection_path = data_root / "selection-only" / f"{source.dataset}.pt"
    if not selection_path.is_file():
        write_external_selection_task(task, selection_path)
    sealed = load_external_selection_task(source.dataset, data_root)
    if sealed.selection_split_sha256 != selection_digest:
        message = f"selection artifact differs from raw task: {source.dataset}"
        raise RuntimeError(message)

    full_path = data_root / f"{source.dataset}.pt"
    if full_path.is_file():
        _assert_full_identity(task, load_prepared_task(full_path))
    else:
        temporary = full_path.with_suffix(full_path.suffix + f".tmp-{os.getpid()}")
        try:
            save_prepared_task(task, temporary)
            temporary.replace(full_path)
        finally:
            temporary.unlink(missing_ok=True)

    row: dict[str, object] = {
        "dataset": source.dataset,
        "source_url": source.url,
        "raw_path": str(raw),
        "raw_sha256": file_sha256(raw),
        "raw_bytes": raw.stat().st_size,
        "protocol": source.protocol,
        "train_shape": list(task.train_inputs.shape),
        "validation_shape": list(task.validation_inputs.shape),
        "test_shape": list(task.test_inputs.shape),
        "selection_path": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "selection_split_sha256": selection_digest,
        "selection_full_identity": {
            "scope": "TRAIN/validation tensors, labels, groups, and temporal metadata",
            "selection_split_sha256": selection_digest,
            "verified": True,
        },
        "selection_contains_test_tensors": False,
        "raw_full_identity_verified": True,
        "full_path": str(full_path),
        "full_sha256": file_sha256(full_path),
        "full_bytes": full_path.stat().st_size,
    }
    del sealed, task
    gc.collect()
    return row


def prepare(
    data_root: Path,
    datasets: tuple[str, ...] = (),
    *,
    download_only: bool = False,
) -> dict[str, object]:
    selected = tuple(
        source for source in SOURCES if not datasets or source.dataset in datasets
    )
    if len(selected) != len(datasets) and datasets:
        known = {source.dataset for source in SOURCES}
        unknown = sorted(set(datasets) - known)
        message = f"unknown Wave-1 forecasting datasets: {unknown}"
        raise ValueError(message)

    rows: list[dict[str, object]] = []
    for source in selected:
        raw = download_source(source, data_root / "forecasting")
        if download_only:
            rows.append(
                {
                    "dataset": source.dataset,
                    "source_url": source.url,
                    "raw_path": str(raw),
                    "raw_sha256": file_sha256(raw),
                    "raw_bytes": raw.stat().st_size,
                    "protocol": source.protocol,
                }
            )
        else:
            rows.append(prepare_dataset(source, data_root))

    report = data_root / (
        "wave1-forecasting-download-audit.json"
        if download_only
        else "wave1-forecasting-audit.json"
    )
    if datasets and report.is_file():
        previous = json.loads(report.read_text(encoding="utf-8"))
        previous_rows = {
            str(row["dataset"]): row for row in previous.get("datasets", ())
        }
        previous_rows.update({str(row["dataset"]): row for row in rows})
        rows = [
            previous_rows[source.dataset]
            for source in SOURCES
            if source.dataset in previous_rows
        ]

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "source_revisions": {
            "zhouhaoyi/ETDataset": ETT_REVISION,
            "thuml/Time-Series-Library": THUML_REVISION,
        },
        "download_only": download_only,
        "datasets": rows,
        "all_source_checksums_verified": all_sources_verified(rows),
        "scientific_contract": (
            "multivariate-to-multivariate forecasting; normalization is fitted "
            "on TRAIN only; validation and TEST targets are chronologically "
            "disjoint; selection artifacts physically omit TEST tensors"
        ),
    }
    temporary = report.with_suffix(report.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=tuple(source.dataset for source in SOURCES),
    )
    parser.add_argument("--download-only", action="store_true")
    arguments = parser.parse_args()
    payload = prepare(
        arguments.data_root,
        tuple(arguments.datasets or ()),
        download_only=arguments.download_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
