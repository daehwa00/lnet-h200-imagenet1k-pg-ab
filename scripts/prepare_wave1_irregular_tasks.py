# ruff: noqa: EM102, T201, TRY003
"""Download, materialize, and audit the public Wave-1 irregular tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lnet.pac_external_tasks import (
    _selection_split_sha256,  # pyright: ignore[reportPrivateUsage]
    load_external_selection_task,
    save_prepared_task,
    write_external_selection_task,
)
from lnet.pac_wave1_irregular_tasks import (
    HUMAN_ACTIVITY_NAME,
    HUMAN_ACTIVITY_SOURCE,
    USHCN_DAILY_NAME,
    USHCN_DAILY_SOURCE,
    PublicSource,
    download_public_source,
    file_sha256,
    load_human_activity,
    load_ushcn_daily,
)

DATASETS = (HUMAN_ACTIVITY_NAME, USHCN_DAILY_NAME)
SCHEMA = "pac_wave1_irregular_preparation.v2"


def _atomic_save_task(task_name: str, data_root: Path, source_path: Path) -> dict[str, object]:
    if task_name == HUMAN_ACTIVITY_NAME:
        task = load_human_activity(source_path)
    elif task_name == USHCN_DAILY_NAME:
        task = load_ushcn_daily(source_path)
    else:
        raise ValueError(f"unknown Wave-1 irregular task: {task_name}")

    output = data_root / f"{task_name}.pt"
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        save_prepared_task(task, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    selection = data_root / "selection-only" / f"{task_name}.pt"
    write_external_selection_task(task, selection)
    expected_selection_identity = _selection_split_sha256(task)
    sealed = load_external_selection_task(task_name, data_root)
    if sealed.selection_split_sha256 != expected_selection_identity:
        raise RuntimeError(f"selection/full identity mismatch for {task_name}")
    return {
        "dataset": task_name,
        "prepared_path": str(output),
        "prepared_sha256": file_sha256(output),
        "prepared_bytes": output.stat().st_size,
        "selection_path": str(selection),
        "selection_sha256": file_sha256(selection),
        "selection_contains_test_tensors": False,
        "selection_full_identity": {
            "scope": "TRAIN/validation tensors, labels, groups, and temporal metadata",
            "selection_split_sha256": expected_selection_identity,
            "verified": True,
        },
        "train_count": int(task.train_inputs.shape[0]),
        "validation_count": int(task.validation_inputs.shape[0]),
        "test_count": int(task.test_inputs.shape[0]),
        "sequence_length": task.sequence_length,
        "input_dim": task.input_dim,
        "output_dim": task.output_dim,
        "has_temporal_metadata": task.has_temporal_metadata,
        "characteristic_time_scale": task.characteristic_time_scale,
    }


def prepare(
    data_root: Path,
    datasets: tuple[str, ...] = DATASETS,
    *,
    download: bool = True,
) -> dict[str, object]:
    sources: dict[str, PublicSource] = {
        HUMAN_ACTIVITY_NAME: HUMAN_ACTIVITY_SOURCE,
        USHCN_DAILY_NAME: USHCN_DAILY_SOURCE,
    }
    raw_root = data_root / "wave1-irregular" / "raw"
    source_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    for dataset in datasets:
        source = sources[dataset]
        source_path = raw_root / source.filename
        if download:
            source_rows.append(download_public_source(source, source_path))
        else:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            actual = file_sha256(source_path)
            if actual != source.sha256:
                raise RuntimeError(
                    f"checksum mismatch for {dataset}: {actual} != {source.sha256}"
                )
            source_rows.append(
                {
                    "dataset": dataset,
                    "url": source.url,
                    "path": str(source_path),
                    "sha256": actual,
                    "expected_sha256": source.sha256,
                    "bytes": source_path.stat().st_size,
                    "license": source.license_name,
                    "upstream_commit": source.upstream_commit,
                    "downloaded": False,
                }
            )
        task_rows.append(_atomic_save_task(dataset, data_root, source_path))

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "sources": source_rows,
        "tasks": task_rows,
        "scientific_contract": {
            "human_activity": (
                "Latent ODE event representation with 10 ms bins, 50-event windows, "
                "stride 25, and seven merged classes; this custom sequence-level "
                "variant uses the modal event class"
            ),
            "ushcn_daily": (
                "GRU-ODE-Bayes context Time<=150 and first three later rows; sparse "
                "future masks are represented by query-unfolded observed scalars"
            ),
            "normalization": (
                "channel statistics are fitted only from observed training-context "
                "values; one median positive observation-to-observation interval is "
                "fitted only from TRAIN, then all elapsed times are divided by that "
                "frozen scalar for validation, test, and forecast queries"
            ),
            "splits": (
                "Human Activity partitions complete recording IDs before assigning "
                "overlapping windows, using locked seeds 42/43; USHCN uses locked 10% "
                "test and 20%-of-development validation station splits with seed 432"
            ),
        },
    }
    audit = data_root / "wave1-irregular-preparation-audit.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit.with_suffix(audit.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(audit)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/external"))
    parser.add_argument("--datasets", nargs="*", choices=DATASETS)
    parser.add_argument("--skip-download", action="store_true")
    arguments = parser.parse_args()
    datasets = tuple(arguments.datasets) if arguments.datasets else DATASETS
    payload = prepare(
        arguments.data_root,
        datasets,
        download=not arguments.skip_download,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
