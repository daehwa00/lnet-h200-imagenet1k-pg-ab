from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import torch

from lnet.astronomy.phase0 import LagMode, ModelName, Phase0RunConfig, train_one_seed
from lnet.astronomy.plasticc import (
    PLASTICC_KNOWN_CLASS_WEIGHTS,
    PLASTICC_KNOWN_TARGETS,
    PlasticcDataset,
    read_light_curves,
    read_phase0_labels,
    stratified_train_validation_split,
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("alphabet", "gru", "grud"), required=True)
    parser.add_argument(
        "--lag-mode",
        choices=("physical", "token", "energy"),
        default="physical",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 31])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=20260729)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        message = "Phase-1 training requires a CUDA host"
        raise RuntimeError(message)
    metadata = args.data_dir / "plasticc_train_metadata.csv.gz"
    light_curves = args.data_dir / "plasticc_train_lightcurves.csv.gz"
    labels = read_phase0_labels(
        metadata,
        targets=PLASTICC_KNOWN_TARGETS,
        max_objects_per_class=10_000_000,
        seed=args.split_seed,
    )
    curves = read_light_curves(light_curves, labels)
    train_ids, validation_ids = stratified_train_validation_split(
        labels,
        seed=args.split_seed,
    )
    train = PlasticcDataset(curves, labels, train_ids)
    validation = PlasticcDataset(curves, labels, validation_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "official_test_accessed": False,
        "metadata_sha256": _digest(metadata),
        "light_curves_sha256": _digest(light_curves),
        "known_targets": PLASTICC_KNOWN_TARGETS,
        "class_weights": PLASTICC_KNOWN_CLASS_WEIGHTS,
        "train_count": len(train),
        "validation_count": len(validation),
        "train_object_ids": train_ids,
        "validation_object_ids": validation_ids,
        "split_seed": args.split_seed,
        "lag_mode": args.lag_mode,
    }
    (args.output_dir / "train-validation-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    device = torch.device("cuda")
    for seed in args.seeds:
        train_one_seed(
            Phase0RunConfig(
                model=cast("ModelName", args.model),
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                classes=len(PLASTICC_KNOWN_TARGETS),
                lag_mode=cast("LagMode", args.lag_mode),
                class_weights=PLASTICC_KNOWN_CLASS_WEIGHTS,
            ),
            train,
            validation,
            validation,
            args.output_dir,
            device=device,
            evaluation_split="validation",
        )


if __name__ == "__main__":
    main()
