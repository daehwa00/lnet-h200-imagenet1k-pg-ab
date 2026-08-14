# pyright: reportExplicitAny=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from lnet.astronomy.damped_spectrum import DampedSpectrumClassifier
from lnet.astronomy.phase0 import Phase0RunConfig, build_model
from lnet.astronomy.plasticc import (
    LightCurveBatch,
    PlasticcDataset,
    collate_light_curves,
    read_light_curves,
    read_phase0_labels,
    stratified_object_split,
)
from lnet.astronomy.pole_audit import (
    finite_period_spearman,
    lomb_scargle_period_days,
)
from lnet.astronomy.robustness import (
    interpolate_uniform_grid,
    replace_with_unit_intervals,
)


def _move(batch: LightCurveBatch) -> LightCurveBatch:
    return LightCurveBatch(
        flux=batch.flux.cuda(non_blocking=True),
        time_delta=batch.time_delta.cuda(non_blocking=True),
        observation_mask=batch.observation_mask.cuda(non_blocking=True),
        valid_mask=batch.valid_mask.cuda(non_blocking=True),
        target=batch.target.cuda(non_blocking=True),
        object_id=batch.object_id.cuda(non_blocking=True),
    )


def _correlations(
    rows: list[dict[str, float | int]],
    targets: tuple[int, ...],
) -> dict[str, dict[str, float | int]]:
    correlations: dict[str, dict[str, float | int]] = {}
    for class_index, target_id in enumerate(targets):
        selected = [
            row
            for row in rows
            if row["target"] == class_index
            and np.isfinite(float(row["spectrum_period_days"]))
            and np.isfinite(float(row["lomb_scargle_period_days"]))
        ]
        if not selected:
            continue
        count, statistic, pvalue = finite_period_spearman(
            [float(row["spectrum_period_days"]) for row in selected],
            [float(row["lomb_scargle_period_days"]) for row in selected],
        )
        correlations[str(target_id)] = {
            "count": count,
            "statistic": statistic,
            "pvalue": pvalue,
        }
    return correlations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[16, 92])
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--near-undamped-modes", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initialized-control", action="store_true")
    parser.add_argument("--evaluation-split", choices=("test", "all"), default="test")
    return parser.parse_args()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_data_files(data_dir: Path, manifest: dict[str, Any]) -> None:
    paths = {
        "metadata_sha256": data_dir / "plasticc_train_metadata.csv.gz",
        "light_curves_sha256": data_dir / "plasticc_train_lightcurves.csv.gz",
    }
    for key, path in paths.items():
        expected = manifest.get(key)
        if expected is not None and _digest(path) != expected:
            message = f"data file does not match training manifest: {path}"
            raise ValueError(message)


def _resolve_training_contract(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, tuple[int, ...], int, dict[str, Any] | None]:
    if args.initialized_control:
        return None, tuple(args.targets), 20260729, None
    result_payload = cast(
        "dict[str, Any]",
        json.loads((args.results_dir / f"dls-seed{args.seed}.json").read_text()),
    )
    stored_config = cast("dict[str, Any]", result_payload["config"])
    stored_variant = (
        str(stored_config["model"]),
        int(stored_config["seed"]),
        int(stored_config["modes"]),
        int(stored_config.get("near_undamped_modes", 0)),
    )
    requested_variant = (
        "dls",
        args.seed,
        args.modes,
        args.near_undamped_modes,
    )
    if requested_variant != stored_variant:
        message = f"audit variant {requested_variant} does not match checkpoint {stored_variant}"
        raise ValueError(message)
    manifest = cast(
        "dict[str, Any]",
        json.loads((args.results_dir / "split-manifest.json").read_text()),
    )
    targets = tuple(int(value) for value in manifest.get("targets", args.targets))
    if targets != tuple(args.targets):
        message = "audit targets do not match the training manifest"
        raise ValueError(message)
    return stored_config, targets, int(manifest["split_seed"]), manifest


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        message = "damped-spectrum audit requires a CUDA host"
        raise RuntimeError(message)
    stored_config, targets, split_seed, manifest = _resolve_training_contract(args)
    if manifest is not None:
        _validate_data_files(args.data_dir, manifest)
    labels = read_phase0_labels(
        args.data_dir / "plasticc_train_metadata.csv.gz",
        targets=targets,
    )
    curves = read_light_curves(
        args.data_dir / "plasticc_train_lightcurves.csv.gz",
        labels,
    )
    if manifest is not None and manifest.get("time_mode") == "unit":
        curves = {
            object_id: replace_with_unit_intervals(curve)
            for object_id, curve in curves.items()
        }
    elif manifest is not None and manifest.get("time_mode") == "uniform-grid":
        curves = {
            object_id: interpolate_uniform_grid(
                curve,
                step_days=float(manifest["grid_step_days"]),
            )
            for object_id, curve in curves.items()
        }
    split = stratified_object_split(labels, seed=split_seed)
    object_ids = split.test if args.evaluation_split == "test" else tuple(sorted(labels))
    dataset = PlasticcDataset(curves, labels, object_ids)
    near_undamped_alpha = (
        float(stored_config.get("near_undamped_alpha_per_day", 1.0e-6))
        if stored_config is not None
        else 1.0e-6
    )
    freeze_frequencies = (
        bool(stored_config.get("freeze_spectrum_frequencies", False))
        if stored_config is not None
        else False
    )
    model = build_model(
        Phase0RunConfig(
            model="dls",
            seed=args.seed,
            classes=len(targets),
            modes=args.modes,
            near_undamped_modes=args.near_undamped_modes,
            near_undamped_alpha_per_day=near_undamped_alpha,
            freeze_spectrum_frequencies=freeze_frequencies,
        ),
        max(curve.flux.shape[0] for curve in curves.values()),
    )
    if not isinstance(model, DampedSpectrumClassifier):
        message = "damped-spectrum builder returned an incompatible model"
        raise TypeError(message)
    model = model.cuda()
    checkpoint = args.results_dir / f"dls-seed{args.seed}.pt"
    if not args.initialized_control:
        model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    model.eval()
    frequencies = model.frequency_values().detach().cpu()
    rows: list[dict[str, float | int]] = []
    subbank_rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            examples = [dataset[index] for index in range(start, min(start + 64, len(dataset)))]
            cpu_batch = collate_light_curves(examples)
            batch = _move(cpu_batch)
            *_, power = model.modal_coordinates(
                batch.flux,
                time_delta=batch.time_delta,
                observation_mask=batch.observation_mask,
                valid_mask=batch.valid_mask,
            )
            modes = power.argmax(dim=-1).cpu()
            subbank_modes = (
                power[:, : args.near_undamped_modes].argmax(dim=-1).cpu()
                if args.near_undamped_modes
                else None
            )
            for index, object_id in enumerate(cpu_batch.object_id.tolist()):
                target_index = int(cpu_batch.target[index])
                target_id = targets[target_index]
                minimum_period = 0.2 if target_id == 92 else 0.05
                maximum_period = 1.0 if target_id == 92 else 10.0
                rows.append(
                    {
                        "object_id": object_id,
                        "target": target_index,
                        "mode": int(modes[index]),
                        "spectrum_period_days": float(
                            2.0 * math.pi / frequencies[int(modes[index])]
                        ),
                        "lomb_scargle_period_days": lomb_scargle_period_days(
                            curves[object_id],
                            minimum_period_days=minimum_period,
                            maximum_period_days=maximum_period,
                        ),
                    }
                )
                if subbank_modes is not None:
                    subbank_rows.append(
                        {
                            "object_id": object_id,
                            "target": target_index,
                            "mode": int(subbank_modes[index]),
                            "spectrum_period_days": float(
                                2.0 * math.pi / frequencies[int(subbank_modes[index])]
                            ),
                            "lomb_scargle_period_days": rows[-1][
                                "lomb_scargle_period_days"
                            ],
                        }
                    )
    correlations = _correlations(rows, targets)
    subbank_correlations = _correlations(subbank_rows, targets)
    payload = {
        "schema": "lnet.astronomy.damped_spectrum_audit.v1",
        "checkpoint": None if args.initialized_control else str(checkpoint),
        "seed": args.seed,
        "targets": targets,
        "modes": args.modes,
        "near_undamped_modes": args.near_undamped_modes,
        "evaluation_split": args.evaluation_split,
        "class_spearman": correlations,
        "near_undamped_subbank_spearman": subbank_correlations,
        "objects": rows,
        "near_undamped_subbank_objects": subbank_rows,
    }
    output = args.results_dir / (
        (
            f"spectrum-audit-initialized-m{args.modes}-"
            f"u{args.near_undamped_modes}-{args.evaluation_split}-seed{args.seed}.json"
        )
        if args.initialized_control
        else f"spectrum-audit-seed{args.seed}.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n")
    sys.stdout.write(json.dumps(correlations, indent=2) + "\n")


if __name__ == "__main__":
    main()
