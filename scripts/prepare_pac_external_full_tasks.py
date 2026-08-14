# ruff: noqa: T201
"""Materialize and audit full external-task tensors for final evaluation.

Calibration workers consume sealed TRAIN/validation-only artifacts under
``data/external/selection-only``.  Final workers require the corresponding
full TRAIN/validation/TEST task.  This script serializes the full task once and
fails closed unless its TRAIN/validation tensors have exactly the same content
fingerprint as the sealed selection artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from lnet.pac_external_tasks import (
    ExternalDatasetName,
    ExternalTask,
    _selection_split_sha256,  # pyright: ignore[reportPrivateUsage]
    load_external_selection_task,
    load_external_task,
    load_prepared_task,
    save_prepared_task,
)

SCHEMA = "pac_external_full_preparation.v1"
DEFAULT_DATASETS: tuple[ExternalDatasetName, ...] = (
    "ptb-xl",
    "mit-bih",
    "cwru",
    "ettm1",
    "ettm2",
    "electricity",
    "weather",
    "etth1",
    "etth2",
    "traffic",
    "ili",
    "exchange-rate",
    "sequential-mnist",
    "permuted-mnist",
    "sequential-cifar",
    "audioset-balanced",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_selection_identity(task: ExternalTask, expected_sha256: str) -> str:
    """Fail unless full-task TRAIN/validation tensors equal calibration data."""
    actual = _selection_split_sha256(task)
    if actual != expected_sha256:
        message = (
            "full-task TRAIN/validation fingerprint changed: "
            f"{actual} != {expected_sha256}"
        )
        raise RuntimeError(message)
    return actual


def prepare_dataset(dataset: ExternalDatasetName, data_root: Path) -> dict[str, object]:
    output = data_root / f"{dataset}.pt"
    selection = load_external_selection_task(dataset, data_root)
    if output.is_file():
        task = load_prepared_task(output)
        source = "existing prepared tensor"
    else:
        task = load_external_task(dataset, data_root)
        verify_selection_identity(task, selection.selection_split_sha256)
        temporary = output.with_suffix(output.suffix + ".tmp")
        save_prepared_task(task, temporary)
        temporary.replace(output)
        source = "raw loader, then one-time serialization"

    selection_sha256 = verify_selection_identity(
        task, selection.selection_split_sha256
    )
    payload: dict[str, object] = {
        "dataset": dataset,
        "prepared_path": str(output),
        "prepared_sha256": _sha256(output),
        "prepared_bytes": output.stat().st_size,
        "selection_split_sha256": selection_sha256,
        "selection_identity_verified": True,
        "source": source,
        "train_count": int(task.train_inputs.shape[0]),
        "validation_count": int(task.validation_inputs.shape[0]),
        "test_count": int(task.test_inputs.shape[0]),
        "sequence_length": task.sequence_length,
        "input_dim": task.input_dim,
        "output_dim": task.output_dim,
    }
    del task, selection
    gc.collect()
    return payload


def prepare(
    data_root: Path,
    datasets: tuple[ExternalDatasetName, ...] = DEFAULT_DATASETS,
) -> dict[str, object]:
    rows = [prepare_dataset(dataset, data_root) for dataset in datasets]
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "datasets": rows,
        "selection_identity_verified": all(
            row["selection_identity_verified"] is True for row in rows
        ),
        "scientific_contract": (
            "serialization-only optimization; every full task's TRAIN/validation "
            "tensors equal its sealed calibration artifact by full-content SHA-256"
        ),
    }
    report = data_root / "full-prepared-audit.json"
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(report)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/external"))
    parser.add_argument("--datasets", nargs="*", choices=DEFAULT_DATASETS)
    args = parser.parse_args()
    datasets = tuple(args.datasets) if args.datasets else DEFAULT_DATASETS
    payload = prepare(args.data_root, datasets)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
