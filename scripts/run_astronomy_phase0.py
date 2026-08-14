from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import torch

from lnet.astronomy.phase0 import LagMode, Phase0RunConfig, train_one_seed
from lnet.astronomy.plasticc import (
    PHASE0_TARGETS,
    PlasticcDataset,
    read_light_curves,
    read_phase0_labels,
    stratified_object_split,
)
from lnet.astronomy.robustness import (
    interpolate_uniform_grid,
    replace_with_unit_intervals,
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("alphabet", "gru", "grud", "dls"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 31])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=20260729)
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=list(PHASE0_TARGETS),
        help="Ordered PLAsTiCC target ids; their positions define classifier labels.",
    )
    parser.add_argument(
        "--time-mode",
        choices=("actual", "unit", "uniform-grid"),
        default="actual",
    )
    parser.add_argument("--grid-step-days", type=float, default=1.0)
    parser.add_argument(
        "--lag-mode",
        choices=("physical", "token", "energy"),
        default="physical",
    )
    parser.add_argument(
        "--injection-mode",
        choices=("zoh", "impulse"),
        default="zoh",
    )
    parser.add_argument("--near-undamped-modes", type=int, default=0)
    parser.add_argument("--near-undamped-alpha-per-day", type=float, default=1.0e-6)
    parser.add_argument("--point-sample-local-convolution", action="store_true")
    parser.add_argument("--class-weights", type=float, nargs="*", default=[])
    parser.add_argument("--freeze-spectrum-frequencies", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata = args.data_dir / "plasticc_train_metadata.csv.gz"
    light_curves = args.data_dir / "plasticc_train_lightcurves.csv.gz"
    targets = tuple(args.targets)
    if len(targets) < 2 or len(set(targets)) != len(targets):
        message = "--targets requires at least two distinct class ids"
        raise ValueError(message)
    if args.class_weights and len(args.class_weights) != len(targets):
        message = "--class-weights must provide one value per target"
        raise ValueError(message)
    labels = read_phase0_labels(metadata, targets=targets, seed=args.split_seed)
    curves = read_light_curves(light_curves, labels)
    if args.time_mode == "unit":
        curves = {
            object_id: replace_with_unit_intervals(curve)
            for object_id, curve in curves.items()
        }
    elif args.time_mode == "uniform-grid":
        curves = {
            object_id: interpolate_uniform_grid(
                curve,
                step_days=args.grid_step_days,
            )
            for object_id, curve in curves.items()
        }
    split = stratified_object_split(labels, seed=args.split_seed)
    datasets = (
        PlasticcDataset(curves, labels, split.train),
        PlasticcDataset(curves, labels, split.validation),
        PlasticcDataset(curves, labels, split.test),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "metadata_sha256": _digest(metadata),
        "light_curves_sha256": _digest(light_curves),
        "split_seed": args.split_seed,
        "targets": list(targets),
        "time_mode": args.time_mode,
        "grid_step_days": (
            args.grid_step_days if args.time_mode == "uniform-grid" else None
        ),
        "lag_mode": args.lag_mode,
        "injection_mode": args.injection_mode,
        "near_undamped_modes": args.near_undamped_modes,
        "near_undamped_alpha_per_day": args.near_undamped_alpha_per_day,
        "point_sample_local_convolution": args.point_sample_local_convolution,
        "class_weights": args.class_weights,
        "freeze_spectrum_frequencies": args.freeze_spectrum_frequencies,
        "counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "object_ids": {
            "train": list(split.train),
            "validation": list(split.validation),
            "test": list(split.test),
        },
    }
    manifest_path = args.output_dir / "split-manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest != manifest:
            message = f"output directory contains a conflicting manifest: {manifest_path}"
            raise FileExistsError(message)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not torch.cuda.is_available():
        message = "Phase 0 is GPU-only; CUDA is unavailable"
        raise RuntimeError(message)
    device = torch.device("cuda")
    for seed in args.seeds:
        train_one_seed(
            Phase0RunConfig(
                model=args.model,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                modes=args.modes,
                lag_mode=cast("LagMode", args.lag_mode),
                injection_mode=args.injection_mode,
                near_undamped_modes=args.near_undamped_modes,
                near_undamped_alpha_per_day=args.near_undamped_alpha_per_day,
                point_sample_local_convolution=args.point_sample_local_convolution,
                classes=len(targets),
                class_weights=tuple(args.class_weights),
                freeze_spectrum_frequencies=args.freeze_spectrum_frequencies,
            ),
            *datasets,
            args.output_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
