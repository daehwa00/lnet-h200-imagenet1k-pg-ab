from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from lnet.astronomy.phase0 import ModelName, Phase0RunConfig, build_model, evaluate
from lnet.astronomy.plasticc import (
    LightCurve,
    PlasticcDataset,
    collate_light_curves,
    read_light_curves,
    read_phase0_labels,
    stratified_object_split,
)
from lnet.astronomy.robustness import insert_seasonal_gap, truncate_after_days


def _evaluate_curves(
    *,
    model_name: str,
    seed: int,
    curves: dict[int, LightCurve],
    labels: dict[int, int],
    object_ids: tuple[int, ...],
    results_dir: Path,
) -> dict[str, float]:
    config = Phase0RunConfig(model=cast("ModelName", model_name), seed=seed)
    model = build_model(config, max(curve.flux.shape[0] for curve in curves.values())).cuda()
    checkpoint = results_dir / f"{model_name}-seed{seed}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    dataset = PlasticcDataset(curves, labels, object_ids)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_light_curves)
    return asdict(evaluate(model, loader, torch.device("cuda"), 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        message = "Phase-2 curve evaluation requires a CUDA host"
        raise RuntimeError(message)
    labels = read_phase0_labels(args.data_dir / "plasticc_train_metadata.csv.gz")
    curves = read_light_curves(args.data_dir / "plasticc_train_lightcurves.csv.gz", labels)
    split = stratified_object_split(labels, seed=20260729)
    payload: dict[str, object] = {"early_classification": [], "seasonal_gap": []}
    for model_name in ("alphabet", "gru"):
        for seed in (7, 11, 19, 23, 31):
            for days in (3.0, 7.0, 14.0, 30.0, 60.0):
                truncated = {
                    object_id: truncate_after_days(curves[object_id], days)
                    for object_id in split.test
                }
                metrics = _evaluate_curves(
                    model_name=model_name,
                    seed=seed,
                    curves=truncated,
                    labels=labels,
                    object_ids=split.test,
                    results_dir=args.results_dir,
                )
                cast("list[object]", payload["early_classification"]).append(
                    {"model": model_name, "seed": seed, "days": days, **metrics}
                )
            for gap_days in (60.0, 120.0, 180.0):
                shifted = {
                    object_id: insert_seasonal_gap(curves[object_id], gap_days)
                    for object_id in split.test
                }
                metrics = _evaluate_curves(
                    model_name=model_name,
                    seed=seed,
                    curves=shifted,
                    labels=labels,
                    object_ids=split.test,
                    results_dir=args.results_dir,
                )
                cast("list[object]", payload["seasonal_gap"]).append(
                    {"model": model_name, "seed": seed, "gap_days": gap_days, **metrics}
                )
    output = args.results_dir / "phase2-curves.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
