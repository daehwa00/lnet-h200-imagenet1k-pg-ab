from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import csv
import gzip
import json
from pathlib import Path

import h5py
import numpy as np

from lnet.astronomy.plasticc import (
    PLASTICC_KNOWN_TARGETS,
    read_phase0_labels,
    stratified_train_validation_split,
)

FILTER_NAMES = ("u", "g", "r", "i", "z", "Y")


def _convert(data_dir: Path, output_dir: Path, *, split_seed: int) -> None:
    metadata_path = data_dir / "plasticc_train_metadata.csv.gz"
    light_curve_path = data_dir / "plasticc_train_lightcurves.csv.gz"
    labels = read_phase0_labels(
        metadata_path,
        targets=PLASTICC_KNOWN_TARGETS,
        max_objects_per_class=10_000_000,
        seed=split_seed,
    )
    train_ids, validation_ids = stratified_train_validation_split(
        labels,
        seed=split_seed,
    )
    requested = set(labels)
    peak: dict[int, tuple[float, float]] = {}
    with gzip.open(light_curve_path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            object_id = int(row["object_id"])
            if object_id not in requested:
                continue
            flux_error = float(row["flux_err"])
            score = float(
                abs(float(row["flux"])) / max(flux_error, np.finfo(np.float32).tiny)
            )
            previous = peak.get(object_id)
            if previous is None or score > previous[0]:
                peak[object_id] = (score, float(row["mjd"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / "PLASTICC_HEAD.csv.gz"
    with (
        gzip.open(metadata_path, "rt", newline="") as source,
        gzip.open(head_path, "wt", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=("SNID", "SNTYPE", "PEAKMJD"))
        writer.writeheader()
        for row in reader:
            object_id = int(row["object_id"])
            if object_id not in requested:
                continue
            writer.writerow(
                {
                    "SNID": object_id,
                    "SNTYPE": int(row["target"]),
                    "PEAKMJD": peak[object_id][1],
                }
            )

    phot_path = output_dir / "PLASTICC_PHOT.csv.gz"
    with (
        gzip.open(light_curve_path, "rt", newline="") as source,
        gzip.open(phot_path, "wt", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(
            destination,
            fieldnames=("SNID", "MJD", "FLUXCAL", "FLUXCALERR", "FLT"),
        )
        writer.writeheader()
        for row in reader:
            object_id = int(row["object_id"])
            if object_id not in requested:
                continue
            writer.writerow(
                {
                    "SNID": object_id,
                    "MJD": row["mjd"],
                    "FLUXCAL": row["flux"],
                    "FLUXCALERR": row["flux_err"],
                    "FLT": FILTER_NAMES[int(row["passband"])],
                }
            )

    validation_path = output_dir / "validation_ids.csv"
    with validation_path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("SNID",))
        writer.writerows((object_id,) for object_id in validation_ids)
    split = {
        "split_seed": split_seed,
        "known_targets": PLASTICC_KNOWN_TARGETS,
        "train_object_ids": train_ids,
        "validation_object_ids": validation_ids,
        "peak_estimator": "MJD of maximum absolute flux/flux_err",
        "supernnova_filter_order": FILTER_NAMES,
    }
    (output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")


def _patch_hdf5(
    hdf5_path: Path,
    split_path: Path,
    *,
    validation_role: str,
) -> None:
    split = json.loads(split_path.read_text())
    train_ids = {str(value) for value in split["train_object_ids"]}
    validation_ids = {str(value) for value in split["validation_object_ids"]}
    with h5py.File(hdf5_path, "r+") as dataset:
        object_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in dataset["SNID"][:]
        ]
        assignment = np.full(len(object_ids), -1, dtype=np.int8)
        for index, object_id in enumerate(object_ids):
            if object_id in train_ids:
                assignment[index] = 0
            elif object_id in validation_ids:
                assignment[index] = 1 if validation_role == "validation" else 2
        missing = len(train_ids | validation_ids) - int((assignment >= 0).sum())
        if missing:
            message = f"{missing} split objects are absent from the SuperNNova dataset"
            raise ValueError(message)
        dataset["dataset_photometry_14classes"][:] = assignment
        dataset.attrs["plasticc_split_seed"] = int(split["split_seed"])
        dataset.attrs["plasticc_split_contract"] = (
            f"shared 90/10 object split; validation role={validation_role}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert")
    convert.add_argument("--data-dir", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument("--split-seed", type=int, default=20260729)
    patch = subparsers.add_parser("patch-hdf5")
    patch.add_argument("--hdf5", type=Path, required=True)
    patch.add_argument("--split", type=Path, required=True)
    patch.add_argument(
        "--validation-role",
        choices=("validation", "test"),
        default="validation",
    )
    args = parser.parse_args()
    if args.command == "convert":
        _convert(args.data_dir, args.output_dir, split_seed=args.split_seed)
    else:
        _patch_hdf5(
            args.hdf5,
            args.split,
            validation_role=args.validation_role,
        )


if __name__ == "__main__":
    main()
