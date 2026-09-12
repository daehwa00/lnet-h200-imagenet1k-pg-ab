#!/usr/bin/env python3
"""Generate a disarmed, offline-only dense-transfer queue.

This command creates manifests; it never launches a process, chooses CUDA
indices, or marks a task complete.  A separate operator must satisfy the
recorded prerequisites and explicitly arm a launcher outside this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dense_transfer.engine import (
    VALID_MODELS, atomic_json, canonical_json, default_optimizer_plan,
    dataset_identity, prevalidate_backbone_checkpoint, sha256_file,
)


SCHEMA = "dense-transfer.queue.v1"
P0: tuple[tuple[str, str, int], ...] = (
    ("ade20k", "va_k128", 501), ("ade20k", "convnextv2_atto", 501), ("ade20k", "tinyvim_s", 501),
    ("coco", "va_k128", 501), ("coco", "convnextv2_atto", 501), ("coco", "tinyvim_s", 501),
)
P1: tuple[tuple[str, str, int], ...] = (("ade20k", "va_k128", 509), ("ade20k", "tinyvim_s", 509))


def digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def checkpoint_for(root: Path, model_key: str) -> Path:
    # A is the only checkpoint permitted for the VA comparison and is explicitly
    # named so a queue cannot silently substitute an unvalidated epoch/seed.
    name = f"{model_key}_seed501_ep100.pt"
    return root / name


def source_contract(task: str, model_key: str, downstream_seed: int, checkpoint: Path, data_root: Path, source_root: Path | None) -> dict[str, object]:
    plan = default_optimizer_plan(task)
    checkpoint_exists = checkpoint.is_file()
    data_exists = data_root.is_dir()
    source_exists = model_key != "tinyvim_s" or (source_root is not None and source_root.is_dir())
    validation: dict[str, object] | None = None
    if checkpoint_exists:
        try:
            validation = prevalidate_backbone_checkpoint(checkpoint, model_key)
        except Exception:  # corrupt/untrusted offline artifacts remain unsatisfied, never abort preparation
            validation = None
    prerequisites = [
        {"name": "checkpoint", "path": str(checkpoint.resolve()), "satisfied": checkpoint_exists},
        {"name": "checkpoint_prevalidated_completed_epochs_100", "path": str(checkpoint.resolve()), "satisfied": validation is not None},
        {"name": "dataset", "path": str(data_root.resolve()), "satisfied": data_exists},
        {"name": "tinyvim_source", "required": model_key == "tinyvim_s", "path": None if source_root is None else str(source_root.resolve()), "satisfied": source_exists},
    ]
    if model_key == "va_k128":
        prerequisites.append({
            "name": "checkpoint_A_prevalidated_seed501_ep100", "path": str(checkpoint.resolve()),
            "satisfied": validation is not None,
            "how": "checkpoint was loaded and verified as completed_epochs=100; its filename binds checkpoint A to seed=501.",
        })
    settings = {
        "task": task, "model_key": model_key, "downstream_seed": downstream_seed, "pretrain_seed": 501, "checkpoint": str(checkpoint.resolve()),
        "source_root": None if source_root is None else str(source_root.resolve()), "data_root": str(data_root.resolve()),
        "physical_batch_size": 2, "effective_batch_size": 16, "accumulation_steps": 8,
        "workers": 2, "bf16": True, "compiled_backbone": False, "channels_last": False,
    }
    return {
        "schema": SCHEMA, "checkpoint_sha256": sha256_file(checkpoint) if checkpoint_exists else None,
        "dataset_identity": dataset_identity(data_root) if data_exists else None,
        "checkpoint_prevalidation": validation,
        "settings": settings, "optimizer": {"epochs": plan.epochs, "learning_rate": plan.learning_rate,
        "weight_decay": plan.weight_decay, "warmup_updates": plan.warmup_updates, "drop_epochs": list(plan.drop_epochs)},
        "prerequisites": prerequisites,
    }


def manifest(priority: str, ordinal: int, task: str, model_key: str, downstream_seed: int, checkpoint_root: Path, data_root: Path, source_root: Path | None) -> dict[str, object]:
    contract = source_contract(task, model_key, downstream_seed, checkpoint_for(checkpoint_root, model_key), data_root, source_root)
    return {
        "schema": SCHEMA, "id": f"{priority.lower()}-{ordinal:02d}-{task}-{model_key}-seed{downstream_seed}", "priority": priority,
        "downstream_seed": downstream_seed, "pretrain_seed": 501,
        "optional": priority == "P1",
        "state": "DISARMED", "autostart": False,
        "device_profile": {
            "device": "cuda", "allowed_device_forms": ["cuda", "cuda:0"],
            "exclusive_required": True, "allow_busy": False,
            "note": "No CUDA index is assigned by this queue; the operator supplies an idle device.",
        },
        "source_contract": contract, "contract_hash": digest(contract),
        "readiness": "incomplete", "result": None,
    }


def _legacy_roots(data_root: Path) -> tuple[Path, Path]:
    coco_candidates = (data_root / "datasets" / "coco", data_root / "coco", data_root)
    ade_candidates = (data_root / "ADEChallengeData2016", data_root / "datasets" / "ADEChallengeData2016", data_root)
    return next(path for path in coco_candidates if path.exists()), next(path for path in ade_candidates if path.exists())


def prepare(output_root: Path, checkpoint_root: Path, coco_root: Path, ade_root: Path, source_root: Path | None) -> list[Path]:
    queue_dir = output_root / "queue"
    paths: list[Path] = []
    all_manifests: list[dict[str, object]] = []
    for priority, tasks in (("P0", P0), ("P1", P1)):
        for ordinal, (task, model_key, downstream_seed) in enumerate(tasks, start=1):
            data_root = ade_root if task == "ade20k" else coco_root
            payload = manifest(priority, ordinal, task, model_key, downstream_seed, checkpoint_root, data_root, source_root)
            payload["output_directory"] = str((output_root / "runs" / payload["id"]).resolve())
            path = queue_dir / f"{payload['id']}.json"
            atomic_json(path, payload)
            paths.append(path)
            all_manifests.append(payload)
    # This is deliberately not a readiness grant.  The training CLI also checks
    # an exact runtime contract, so an offline manifest cannot start a long run.
    atomic_json(queue_dir / "readiness.json", {
        "schema": SCHEMA, "status": "incomplete", "autostart": False,
        "reason": "offline queue only; validate dataset/checkpoints and create a runtime-specific readiness record",
        "manifests": [{"id": item["id"], "contract_hash": item["contract_hash"]} for item in all_manifests],
    })
    return paths


def main() -> None:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--checkpoint-root", type=Path, required=True)
    command.add_argument("--data-root", type=Path, help="legacy common root; resolves datasets/coco and ADEChallengeData2016")
    command.add_argument("--coco-root", type=Path)
    command.add_argument("--ade-root", type=Path)
    command.add_argument("--source-root", type=Path)
    args = command.parse_args()
    if args.coco_root is None or args.ade_root is None:
        if args.data_root is None:
            command.error("provide both --coco-root and --ade-root, or legacy --data-root")
        legacy_coco, legacy_ade = _legacy_roots(args.data_root)
        coco_root = args.coco_root or legacy_coco
        ade_root = args.ade_root or legacy_ade
    else:
        coco_root, ade_root = args.coco_root, args.ade_root
    paths = prepare(args.output_root, args.checkpoint_root, coco_root, ade_root, args.source_root)
    print(json.dumps({"status": "DISARMED", "manifests": [str(path) for path in paths]}, sort_keys=True))


if __name__ == "__main__":
    main()
