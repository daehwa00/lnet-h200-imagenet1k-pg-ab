#!/usr/bin/env python3
"""Summarize validated three-seed H200 baseline results."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# ruff: noqa: T201
import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
    }


def summarize(campaign_path: Path, root: Path) -> dict[str, Any]:
    campaign_bytes = campaign_path.read_bytes()
    campaign = json.loads(campaign_bytes)
    seeds = [int(seed) for seed in campaign["seeds"]]
    models: dict[str, Any] = {}
    complete_models = 0
    for model in campaign["models"]:
        model_key = str(model["key"])
        rows: list[dict[str, Any]] = []
        missing: list[int] = []
        for seed in seeds:
            path = root / "full" / model_key / f"seed_{seed}" / "result.json"
            if not path.is_file():
                missing.append(seed)
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if (
                result.get("status") != "completed"
                or result.get("model_key") != model_key
                or result.get("seed") != seed
                or result.get("completed_epochs") != campaign["full_training"]["epochs"]
            ):
                missing.append(seed)
                continue
            metrics = result["metrics"]
            values = [
                float(metrics["validation_top1"]),
                float(metrics["validation_top5"]),
                float(metrics["images_per_second"]),
                float(result["training_seconds"]),
            ]
            if not all(math.isfinite(value) for value in values):
                missing.append(seed)
                continue
            rows.append(
                {
                    "seed": seed,
                    "learning_rate": result["learning_rate"],
                    "parameters": result["parameters"],
                    "validation_top1": values[0],
                    "validation_top5": values[1],
                    "images_per_second": values[2],
                    "training_seconds": values[3],
                    "contract_sha256": result["contract_sha256"],
                }
            )
        payload: dict[str, Any] = {
            "display_name": model["display_name"],
            "status": "complete" if len(rows) == len(seeds) else "incomplete",
            "missing_seeds": missing,
            "runs": rows,
        }
        if len(rows) == len(seeds):
            payload.update(
                {
                    "parameters": rows[0]["parameters"],
                    "validation_top1": _mean_std(
                        [float(row["validation_top1"]) for row in rows]
                    ),
                    "validation_top5": _mean_std(
                        [float(row["validation_top5"]) for row in rows]
                    ),
                    "images_per_second": _mean_std(
                        [float(row["images_per_second"]) for row in rows]
                    ),
                    "training_seconds": _mean_std(
                        [float(row["training_seconds"]) for row in rows]
                    ),
                }
            )
            complete_models += 1
        models[model_key] = payload
    return {
        "schema": "lnet.h200.imagenet1k.matched_baselines.summary.v1",
        "campaign_id": campaign["campaign_id"],
        "campaign_manifest_sha256": hashlib.sha256(campaign_bytes).hexdigest(),
        "expected_models": len(campaign["models"]),
        "complete_models": complete_models,
        "complete": complete_models == len(campaign["models"]),
        "seeds": seeds,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.campaign, args.root)
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
