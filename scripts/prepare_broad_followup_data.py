# ruff: noqa: T201
"""Prepare and audit every local dataset used by the broad follow-up queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from lnet.pac_broad_followup_queue import EXTERNAL_TASKS, NEW_UCR_83
from lnet.pac_external_tasks import (
    load_external_selection_task,
    load_prepared_task,
    write_external_selection_task,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_ucr_remainder(archive: Path, root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with ZipFile(archive) as source:
        names = set(source.namelist())
        for dataset in NEW_UCR_83:
            destination = root / dataset
            files: dict[str, object] = {}
            for split in ("TRAIN", "TEST"):
                member = f"UCRArchive_2018/{dataset}/{dataset}_{split}.tsv"
                pure = PurePosixPath(member)
                if pure.is_absolute() or ".." in pure.parts or member not in names:
                    message = f"UCR archive is missing a safe member: {member}"
                    raise RuntimeError(message)
                path = destination / f"{dataset}_{split}.tsv"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
                    # The official UCR-2018 archive uses its published password
                    # ``someone``; the extracted scientific files are public.
                    with (
                        source.open(member, pwd=b"someone") as input_handle,
                        temporary.open("wb") as output_handle,
                    ):
                        for chunk in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                            output_handle.write(chunk)
                    temporary.replace(path)
                if path.stat().st_size < 1:
                    message = f"empty UCR split: {path}"
                    raise RuntimeError(message)
                files[split.lower()] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            rows.append({"dataset": dataset, "files": files})
    return rows


def _prepare_external_selection(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in EXTERNAL_TASKS:
        full_path = root / spec.source_artifact
        if not full_path.is_file():
            message = f"missing prepared external task: {full_path}"
            raise FileNotFoundError(message)
        selection_path = root / "selection-only" / f"{spec.key}.pt"
        if not selection_path.exists():
            task = load_prepared_task(full_path)
            write_external_selection_task(task, selection_path)
            del task
        full = load_prepared_task(full_path)
        selection = load_external_selection_task(spec.key, root)
        if (
            full.name != selection.name
            or full.objective != selection.objective
            or full.input_encoding != selection.input_encoding
            or full.output_dim != selection.output_dim
            or full.train_inputs.shape != selection.train_inputs.shape
            or full.validation_inputs.shape != selection.validation_inputs.shape
            or not full.train_inputs.equal(selection.train_inputs)
            or not full.train_targets.equal(selection.train_targets)
            or not full.validation_inputs.equal(selection.validation_inputs)
            or not full.validation_targets.equal(selection.validation_targets)
        ):
            message = f"selection-only artifact differs from full task: {spec.key}"
            raise RuntimeError(message)
        rows.append(
            {
                "dataset": spec.key,
                "objective": full.objective,
                "input_encoding": full.input_encoding,
                "train_shape": list(full.train_inputs.shape),
                "validation_shape": list(full.validation_inputs.shape),
                "test_count_sealed": selection.test_count,
                "full_path": str(full_path),
                "full_sha256": _sha256(full_path),
                "selection_path": str(selection_path),
                "selection_sha256": _sha256(selection_path),
                "selection_split_sha256": selection.selection_split_sha256,
            }
        )
        del full, selection
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ucr-archive",
        type=Path,
        default=Path(".omx/data/ucr/UCRArchive_2018.zip"),
    )
    parser.add_argument("--ucr-root", type=Path, default=Path(".omx/data/ucr"))
    parser.add_argument("--external-root", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            ".omx/results/alphabet-broad-new-datasets-3gpu-20260727/data-audit.json"
        ),
    )
    arguments = parser.parse_args()
    ucr_rows = _extract_ucr_remainder(arguments.ucr_archive, arguments.ucr_root)
    external_rows = _prepare_external_selection(arguments.external_root)
    counts = {
        "ucr": len(ucr_rows),
        "external": len(external_rows),
    }
    payload: dict[str, object] = {
        "schema": "alphabet.broad_new_datasets.data_audit.v1",
        "ucr_archive": {
            "path": str(arguments.ucr_archive),
            "sha256": _sha256(arguments.ucr_archive),
        },
        "ucr": ucr_rows,
        "external": external_rows,
        "counts": counts,
    }
    arguments.audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.audit.with_suffix(arguments.audit.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.audit)
    print(json.dumps({"audit": str(arguments.audit), **counts}, indent=2))


if __name__ == "__main__":
    main()
