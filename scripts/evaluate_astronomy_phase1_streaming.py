from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import torch

from lnet.astronomy.metrics import MetricAccumulator
from lnet.astronomy.phase0 import LagMode, ModelName, Phase0RunConfig, build_model
from lnet.astronomy.plasticc import (
    PLASTICC_KNOWN_CLASS_WEIGHTS,
    PLASTICC_KNOWN_TARGETS,
    LightCurve,
    collate_light_curves,
    iter_light_curves,
    read_phase0_labels,
)


def _load_models(
    results_dir: Path,
    model_name: ModelName,
    lag_mode: LagMode,
) -> list[torch.nn.Module]:
    models: list[torch.nn.Module] = []
    for seed in (7, 11, 19, 23, 31):
        model = build_model(
            Phase0RunConfig(
                model=model_name,
                seed=seed,
                classes=len(PLASTICC_KNOWN_TARGETS),
                lag_mode=lag_mode,
                class_weights=PLASTICC_KNOWN_CLASS_WEIGHTS,
            ),
            sequence_length=400,
        ).cuda()
        checkpoint = results_dir / f"{model_name}-seed{seed}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
        model.eval()
        models.append(model)
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("alphabet", "gru", "grud"), required=True)
    parser.add_argument(
        "--lag-mode",
        choices=("physical", "token", "energy"),
        default="physical",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--expected-shards", type=int, default=11)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        message = "official streaming evaluation requires a CUDA host"
        raise RuntimeError(message)
    labels = read_phase0_labels(
        args.data_dir / "plasticc_test_metadata.csv.gz",
        targets=PLASTICC_KNOWN_TARGETS,
        max_objects_per_class=10_000_000,
        seed=20260729,
        target_column="true_target",
    )
    paths = tuple(sorted(args.data_dir.glob("plasticc_test_lightcurves_*.csv.gz")))
    if len(paths) != args.expected_shards:
        message = f"expected {args.expected_shards} official test shards, found {len(paths)}"
        raise ValueError(message)
    models = _load_models(
        args.results_dir,
        cast("ModelName", args.model),
        cast("LagMode", args.lag_mode),
    )
    accumulators = [MetricAccumulator.create(len(PLASTICC_KNOWN_TARGETS)) for _ in models]
    ensemble = MetricAccumulator.create(len(PLASTICC_KNOWN_TARGETS))
    started = time.perf_counter()
    examples: list[tuple[LightCurve, int]] = []
    with torch.no_grad():
        for curve in iter_light_curves(paths, set(labels)):
            examples.append((curve, labels[curve.object_id]))
            if len(examples) < args.batch_size:
                continue
            batch = collate_light_curves(examples)
            flux = batch.flux.cuda(non_blocking=True)
            delta = batch.time_delta.cuda(non_blocking=True)
            observation = batch.observation_mask.cuda(non_blocking=True)
            valid = batch.valid_mask.cuda(non_blocking=True)
            probabilities = [
                model(
                    flux,
                    time_delta=delta,
                    observation_mask=observation,
                    valid_mask=valid,
                ).softmax(dim=-1)
                for model in models
            ]
            for accumulator, probability in zip(accumulators, probabilities, strict=True):
                accumulator.update(probability, batch.target)
            ensemble.update(torch.stack(probabilities).mean(dim=0), batch.target)
            examples.clear()
        if examples:
            batch = collate_light_curves(examples)
            probabilities = [
                model(
                    batch.flux.cuda(),
                    time_delta=batch.time_delta.cuda(),
                    observation_mask=batch.observation_mask.cuda(),
                    valid_mask=batch.valid_mask.cuda(),
                ).softmax(dim=-1)
                for model in models
            ]
            for accumulator, probability in zip(accumulators, probabilities, strict=True):
                accumulator.update(probability, batch.target)
            ensemble.update(torch.stack(probabilities).mean(dim=0), batch.target)
    elapsed = time.perf_counter() - started
    evaluated_objects = int(ensemble.class_count.sum())
    if evaluated_objects != len(labels):
        message = (
            f"official test object mismatch: evaluated {evaluated_objects}, "
            f"expected {len(labels)}"
        )
        raise RuntimeError(message)
    payload = {
        "model": args.model,
        "lag_mode": args.lag_mode,
        "official_unknown_classes_excluded": [991, 992, 993, 994],
        "metric_scope": "known 14 classes; not the 15-class class-99 competition metric",
        "per_seed": [accumulator.finalize() for accumulator in accumulators],
        "ensemble": ensemble.finalize(),
        "elapsed_seconds": elapsed,
        "expected_known14_objects": len(labels),
        "objects_per_second": evaluated_objects / elapsed,
    }
    output = args.results_dir / "official-test-known14.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
