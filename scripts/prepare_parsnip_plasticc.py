from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import json
from pathlib import Path

import lcdata
import numpy as np
from lcdata.scripts.lcdata_download_plasticc import (
    read_file,
    update_bands,
    update_classes,
    update_object_id,
)


def _format_object_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"expected integer object id, got {value!r}"
        raise TypeError(message)
    return f"PLAsTiCC {value:09d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = read_file(args.data_dir, "plasticc_train_metadata.csv.gz")
    observations = read_file(args.data_dir, "plasticc_train_lightcurves.csv.gz")
    update_classes(metadata)
    update_object_id(metadata)
    update_object_id(observations)
    update_bands(observations)
    dataset = lcdata.from_observations(metadata, observations)

    split = json.loads(args.split.read_text())
    train_ids = {_format_object_id(value) for value in split["train_object_ids"]}
    validation_ids = {
        _format_object_id(value) for value in split["validation_object_ids"]
    }
    object_ids = np.asarray(dataset.meta["object_id"])
    train_mask = np.asarray([value in train_ids for value in object_ids])
    validation_mask = np.asarray([value in validation_ids for value in object_ids])
    if int(train_mask.sum()) != len(train_ids):
        message = "ParSNIP train objects do not match the shared split"
        raise ValueError(message)
    if int(validation_mask.sum()) != len(validation_ids):
        message = "ParSNIP validation objects do not match the shared split"
        raise ValueError(message)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset[train_mask].write_hdf5(
        args.output_dir / "plasticc_train.h5",
        overwrite=True,
    )
    dataset[validation_mask].write_hdf5(
        args.output_dir / "plasticc_validation.h5",
        overwrite=True,
    )
    manifest = {
        "train_objects": int(train_mask.sum()),
        "validation_objects": int(validation_mask.sum()),
        "split_seed": split["split_seed"],
        "converter": "lcdata 1.1.2 official PLAsTiCC mapping",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
