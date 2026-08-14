"""Run the three missing SSM controls for the frozen corruption audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from lnet import pac_direct_stem_corruption as corruption
from lnet.pac_direct_stem_corruption import DirectStemCorruptionJob, run_job


DATASETS = ("CinCECGTorso", "CricketX", "Earthquakes", "StarLightCurves")
MODELS = ("s4d", "s5", "lru")
SEEDS = (23, 31, 43, 47, 59)
SELECTION = Path(
    ".omx/results/alphabet-balanced-hpo-27task-20260725/stage2/selection.json"
)
ROOT = Path(".omx/results/alphabet-corruption-missing-ssm-20260729")


def _trial(model: str, settings: dict[str, int]) -> int:
    depth, state = settings["depth"], settings["state_size"]
    architectures = {
        "s4d": ((1, 8), (2, 8), (3, 8), (1, 16), (2, 16), (3, 16)),
        "s5": ((1, 8), (2, 8), (1, 16), (2, 16), (1, 32), (2, 32)),
        "lru": ((1, 8), (2, 8), (1, 16), (2, 16), (1, 32), (2, 32)),
    }
    return architectures[model].index((depth, state)) + 1


def jobs() -> list[DirectStemCorruptionJob]:
    raw = SELECTION.read_bytes()
    selected = json.loads(raw)["selected"]
    digest = hashlib.sha256(raw).hexdigest()
    result: list[DirectStemCorruptionJob] = []
    for dataset in DATASETS:
        for model in MODELS:
            cell = selected[f"ucr:{dataset}:{model}"]
            recipe = cell["recipe"]
            for seed in SEEDS:
                result.append(
                    DirectStemCorruptionJob(
                        key=f"alphabet_radial_log_corruption:{model}:{dataset}:seed{seed}",
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        model_dim=int(cell["width"]),
                        modes=16,
                        trial=_trial(model, cell["architecture_settings"]),
                        config_key=str(cell["config_key"]),
                        recipe_name=str(recipe["name"]),
                        epochs=100,
                        batch_size=int(recipe["batch_size"]),
                        learning_rate=float(recipe["learning_rate"]),
                        weight_decay=float(recipe["weight_decay"]),
                        grad_clip_norm=float(recipe["grad_clip_norm"]),
                        selection_artifact_sha256=digest,
                        selection_source=(
                            "validation-frozen balanced-HPO selection:"
                            f"ucr:{dataset}:{model}"
                        ),
                    )
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    corruption.BASELINE_MODELS = (*corruption.BASELINE_MODELS, *MODELS)
    selected_jobs = [
        job for index, job in enumerate(jobs())
        if index % args.shard_count == args.shard_index
    ]
    completed = ROOT / "completed"
    failed = ROOT / "failed"
    completed.mkdir(parents=True, exist_ok=True)
    failed.mkdir(parents=True, exist_ok=True)
    for job in selected_jobs:
        target = completed / f"{hashlib.sha256(job.key.encode()).hexdigest()}.json"
        if target.exists():
            continue
        try:
            row = run_job(
                job,
                data_root=args.data_root,
                output_root=ROOT,
                device="cuda",
            )
            target.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        except Exception as error:  # noqa: BLE001
            payload = {**asdict(job), "error": f"{type(error).__name__}: {error}"}
            (failed / target.name).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
    print(json.dumps({"scheduled": len(selected_jobs), "completed": len(list(completed.glob("*.json")))}))


if __name__ == "__main__":
    main()
