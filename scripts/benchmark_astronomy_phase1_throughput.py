from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import torch

from lnet.astronomy.phase0 import LagMode, ModelName, Phase0RunConfig, build_model
from lnet.astronomy.plasticc import (
    PLASTICC_KNOWN_CLASS_WEIGHTS,
    PLASTICC_KNOWN_TARGETS,
    LightCurveBatch,
    collate_light_curves,
    read_light_curves,
    read_phase0_labels,
)


def _move(batch: LightCurveBatch, device: torch.device) -> LightCurveBatch:
    return LightCurveBatch(
        flux=batch.flux.to(device),
        time_delta=batch.time_delta.to(device),
        observation_mask=batch.observation_mask.to(device),
        valid_mask=batch.valid_mask.to(device),
        target=batch.target.to(device),
        object_id=batch.object_id.to(device),
    )


def _best_seed(results_dir: Path, model_name: str) -> int:
    candidates: list[tuple[float, int]] = []
    for seed in (7, 11, 19, 23, 31):
        result = json.loads((results_dir / f"{model_name}-seed{seed}.json").read_text())
        candidates.append((float(result["test"]["weighted_log_loss"]), seed))
    return min(candidates)[1]


def _benchmark(
    *,
    model_name: ModelName,
    lag_mode: LagMode,
    results_dir: Path,
    cpu_batches: list[LightCurveBatch],
    device: torch.device,
    repetitions: int,
) -> dict[str, object]:
    seed = _best_seed(results_dir, model_name)
    model = build_model(
        Phase0RunConfig(
            model=model_name,
            seed=seed,
            classes=len(PLASTICC_KNOWN_TARGETS),
            lag_mode=lag_mode,
            class_weights=PLASTICC_KNOWN_CLASS_WEIGHTS,
        ),
        max(batch.flux.shape[1] for batch in cpu_batches),
    ).to(device)
    model.load_state_dict(
        torch.load(
            results_dir / f"{model_name}-seed{seed}.pt",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    batches = [_move(batch, device) for batch in cpu_batches]
    with torch.inference_mode():
        for batch in batches:
            model(
                batch.flux,
                time_delta=batch.time_delta,
                observation_mask=batch.observation_mask,
                valid_mask=batch.valid_mask,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repetitions):
            for batch in batches:
                model(
                    batch.flux,
                    time_delta=batch.time_delta,
                    observation_mask=batch.observation_mask,
                    valid_mask=batch.valid_mask,
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    objects = repetitions * sum(batch.flux.shape[0] for batch in batches)
    return {
        "model": model_name,
        "lag_mode": lag_mode,
        "seed": seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "objects": objects,
        "elapsed_seconds": elapsed,
        "objects_per_second": objects / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--gru-dir", type=Path, required=True)
    parser.add_argument("--grud-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = read_phase0_labels(
        args.data_dir / "plasticc_train_metadata.csv.gz",
        targets=PLASTICC_KNOWN_TARGETS,
        max_objects_per_class=10_000_000,
        seed=20260729,
    )
    curves = read_light_curves(args.data_dir / "plasticc_train_lightcurves.csv.gz", labels)
    object_ids = sorted(labels)[:1024]
    examples = [(curves[object_id], labels[object_id]) for object_id in object_ids]
    batches = [
        collate_light_curves(examples[start : start + 64])
        for start in range(0, len(examples), 64)
    ]
    jobs = (
        ("alphabet", "physical", args.physical_dir),
        ("alphabet", "token", args.token_dir),
        ("gru", "physical", args.gru_dir),
        ("grud", "physical", args.grud_dir),
    )
    results: list[dict[str, object]] = []
    for model_name, lag_mode, results_dir in jobs:
        results.append(
            _benchmark(
                model_name=cast("ModelName", model_name),
                lag_mode=cast("LagMode", lag_mode),
                results_dir=results_dir,
                cpu_batches=batches,
                device=torch.device("cpu"),
                repetitions=1,
            )
        )
        if torch.cuda.is_available():
            results.append(
                _benchmark(
                    model_name=cast("ModelName", model_name),
                    lag_mode=cast("LagMode", lag_mode),
                    results_dir=results_dir,
                    cpu_batches=batches,
                    device=torch.device("cuda"),
                    repetitions=10,
                )
            )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
