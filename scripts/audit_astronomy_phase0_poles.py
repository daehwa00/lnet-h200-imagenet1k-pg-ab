# pyright: reportExplicitAny=false
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from lnet.astronomy.phase0 import (
    InjectionMode,
    LagMode,
    Phase0RunConfig,
    build_model,
)
from lnet.astronomy.plasticc import (
    PHASE0_TARGETS,
    LightCurveBatch,
    PlasticcDataset,
    collate_light_curves,
    read_light_curves,
    read_phase0_labels,
    stratified_object_split,
)
from lnet.astronomy.pole_audit import (
    ObjectPoleAudit,
    bank_mode_attribution,
    finite_period_spearman,
    lomb_scargle_period_days,
    modal_representations,
    mode_attribution,
    pole_period_days,
)
from lnet.astronomy.robustness import (
    interpolate_uniform_grid,
    replace_with_unit_intervals,
)

if TYPE_CHECKING:
    from lnet.alphabet import Alphabet


def _move(batch: LightCurveBatch) -> LightCurveBatch:
    return LightCurveBatch(
        flux=batch.flux.cuda(non_blocking=True),
        time_delta=batch.time_delta.cuda(non_blocking=True),
        observation_mask=batch.observation_mask.cuda(non_blocking=True),
        valid_mask=batch.valid_mask.cuda(non_blocking=True),
        target=batch.target.cuda(non_blocking=True),
        object_id=batch.object_id.cuda(non_blocking=True),
    )


def _class_correlations(
    rows: list[ObjectPoleAudit],
    targets: tuple[int, ...],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for class_index, target_id in enumerate(targets):
        class_rows = [
            row
            for row in rows
            if row.target == class_index
            and np.isfinite(row.attributed_period_days)
            and np.isfinite(row.lomb_scargle_period_days)
        ]
        count, statistic, pvalue = finite_period_spearman(
            [row.attributed_period_days for row in class_rows],
            [row.lomb_scargle_period_days for row in class_rows],
        )
        result[str(target_id)] = {
            "count": count,
            "statistic": statistic,
            "pvalue": pvalue,
        }
    return result


def _load_run_config(results_dir: Path, seed: int) -> Phase0RunConfig:
    payload = cast(
        "dict[str, Any]",
        json.loads((results_dir / f"alphabet-seed{seed}.json").read_text()),
    )
    stored = cast("dict[str, Any]", payload["config"])
    if stored.get("model") != "alphabet":
        message = "pole attribution requires an ALPHABET checkpoint"
        raise ValueError(message)
    class_weights = cast("list[float]", stored.get("class_weights", []))
    return Phase0RunConfig(
        model="alphabet",
        seed=int(stored["seed"]),
        epochs=int(stored.get("epochs", 50)),
        batch_size=int(stored.get("batch_size", 64)),
        learning_rate=float(stored.get("learning_rate", 3.0e-3)),
        weight_decay=float(stored.get("weight_decay", 1.0e-4)),
        patience=int(stored.get("patience", 8)),
        model_dim=int(stored.get("model_dim", 64)),
        modes=int(stored.get("modes", 16)),
        classes=int(stored.get("classes", 3)),
        lag_mode=cast("LagMode", stored.get("lag_mode", "physical")),
        injection_mode=cast(
            "InjectionMode",
            stored.get("injection_mode", "zoh"),
        ),
        near_undamped_modes=int(stored.get("near_undamped_modes", 0)),
        near_undamped_alpha_per_day=float(
            stored.get("near_undamped_alpha_per_day", 1.0e-6)
        ),
        point_sample_local_convolution=bool(
            stored.get("point_sample_local_convolution", False)
        ),
        class_weights=tuple(class_weights),
    )


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=list(PHASE0_TARGETS),
        help="Ordered target ids used to train the checkpoint.",
    )
    parser.add_argument(
        "--injection-mode",
        choices=("zoh", "impulse"),
        default="zoh",
    )
    parser.add_argument("--near-undamped-modes", type=int, default=0)
    parser.add_argument("--near-undamped-alpha-per-day", type=float, default=1.0e-6)
    parser.add_argument("--point-sample-local-convolution", action="store_true")
    parser.add_argument(
        "--lag-mode",
        choices=("physical", "token", "energy"),
        default="physical",
    )
    return parser.parse_args()


def _validate_audit_contract(
    args: argparse.Namespace,
    run_config: Phase0RunConfig,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    if run_config.seed != args.seed:
        message = "result JSON seed does not match the requested checkpoint"
        raise ValueError(message)
    requested_variant = (
        args.injection_mode,
        args.near_undamped_modes,
        args.near_undamped_alpha_per_day,
        args.point_sample_local_convolution,
        args.lag_mode,
    )
    stored_variant = (
        run_config.injection_mode,
        run_config.near_undamped_modes,
        run_config.near_undamped_alpha_per_day,
        run_config.point_sample_local_convolution,
        run_config.lag_mode,
    )
    if requested_variant != stored_variant:
        message = (
            f"audit variant {requested_variant} does not match checkpoint {stored_variant}"
        )
        raise ValueError(message)
    manifest_path = args.results_dir / "split-manifest.json"
    manifest = cast("dict[str, Any]", json.loads(manifest_path.read_text()))
    targets = tuple(int(value) for value in manifest.get("targets", args.targets))
    if targets != tuple(args.targets):
        message = "audit targets do not match the training manifest"
        raise ValueError(message)
    if len(targets) < 2 or len(set(targets)) != len(targets):
        message = "--targets requires at least two distinct class ids"
        raise ValueError(message)
    return targets, manifest


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        message = "pole attribution requires a CUDA host"
        raise RuntimeError(message)
    run_config = _load_run_config(args.results_dir, args.seed)
    targets, manifest = _validate_audit_contract(args, run_config)
    _validate_data_files(args.data_dir, manifest)
    labels = read_phase0_labels(
        args.data_dir / "plasticc_train_metadata.csv.gz",
        targets=targets,
    )
    curves = read_light_curves(args.data_dir / "plasticc_train_lightcurves.csv.gz", labels)
    time_mode = str(manifest.get("time_mode", "actual"))
    if time_mode == "unit":
        curves = {
            object_id: replace_with_unit_intervals(curve)
            for object_id, curve in curves.items()
        }
    elif time_mode == "uniform-grid":
        curves = {
            object_id: interpolate_uniform_grid(
                curve,
                step_days=float(manifest["grid_step_days"]),
            )
            for object_id, curve in curves.items()
        }
    split = stratified_object_split(labels, seed=int(manifest["split_seed"]))
    dataset = PlasticcDataset(curves, labels, split.test)
    model = cast(
        "Alphabet",
        build_model(
            run_config,
            max(curve.flux.shape[0] for curve in curves.values()),
        ),
    ).cuda()
    checkpoint = args.results_dir / f"alphabet-seed{args.seed}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    model.eval()
    rows: list[ObjectPoleAudit] = []
    writer_rows: list[ObjectPoleAudit] = []
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            examples = [dataset[index] for index in range(start, min(start + 64, len(dataset)))]
            cpu_batch = collate_light_curves(examples)
            batch = _move(cpu_batch)
            writer, reader = modal_representations(model, batch)
            banks, modes = mode_attribution(model, writer, reader, batch.target)
            periods = pole_period_days(model, banks, modes).cpu()
            writer_modes = bank_mode_attribution(
                model,
                writer,
                batch.target,
                bank=0,
            )
            writer_periods = pole_period_days(
                model,
                torch.zeros_like(writer_modes),
                writer_modes,
            ).cpu()
            for index, object_id in enumerate(cpu_batch.object_id.tolist()):
                target_index = int(cpu_batch.target[index])
                minimum_period = 0.2 if targets[target_index] == 92 else 0.05
                maximum_period = 1.0 if targets[target_index] == 92 else 10.0
                lomb_period = lomb_scargle_period_days(
                    curves[object_id],
                    minimum_period_days=minimum_period,
                    maximum_period_days=maximum_period,
                )
                rows.append(
                    ObjectPoleAudit(
                        object_id=object_id,
                        target=target_index,
                        attributed_bank=("writer" if int(banks[index]) == 0 else "reader"),
                        attributed_mode=int(modes[index]),
                        attributed_period_days=float(periods[index]),
                        lomb_scargle_period_days=lomb_period,
                    )
                )
                writer_rows.append(
                    ObjectPoleAudit(
                        object_id=object_id,
                        target=target_index,
                        attributed_bank="writer",
                        attributed_mode=int(writer_modes[index]),
                        attributed_period_days=float(writer_periods[index]),
                        lomb_scargle_period_days=lomb_period,
                    )
                )
    class_spearman = _class_correlations(rows, targets)
    writer_class_spearman = _class_correlations(writer_rows, targets)
    rr_spearman = class_spearman.get(
        "92",
        {"count": 0, "statistic": float("nan"), "pvalue": float("nan")},
    )
    payload = {
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "targets": targets,
        "rr_lyrae_count": rr_spearman["count"],
        "rr_spearman": rr_spearman,
        "class_spearman": class_spearman,
        "writer_class_spearman": writer_class_spearman,
        "objects": [asdict(row) for row in rows],
        "writer_objects": [asdict(row) for row in writer_rows],
    }
    output = args.results_dir / f"pole-audit-seed{args.seed}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    sys.stdout.write(json.dumps(payload["class_spearman"], indent=2) + "\n")


if __name__ == "__main__":
    main()
