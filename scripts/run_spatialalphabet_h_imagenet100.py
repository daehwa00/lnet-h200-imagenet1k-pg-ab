# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the frozen SPATIALPHABET-H ImageNet-100 protocol."""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_imagenet100_nano as harness

from lnet.spatialalphabet_h import (
    SpatialAlphabetHConfig,
    build_spatialalphabet_h,
    descriptor_coordinates,
)

if TYPE_CHECKING:
    from argparse import Namespace

VARIANT = "spatialalphabet_h"
SEEDS = (501, 509, 521)


def _contract(args: Namespace) -> dict[str, Any]:
    config = SpatialAlphabetHConfig()
    learning_rate = float(os.environ.get("SPATIALH_PEAK_LR", "0.0003"))
    data_digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    descriptor_dim = len(descriptor_coordinates(config.modes))
    model = build_spatialalphabet_h(config)
    formal = args.epochs >= 100
    payload = {
        "schema": (
            "lnet.spatialalphabet_h.imagenet100.identity_descriptor.formal.v2"
            if formal
            else "lnet.spatialalphabet_h.imagenet100.stability.v3"
        ),
        "evidence_status": (
            "100-epoch three-seed confirmatory architecture evaluation"
            if formal
            else "10-epoch restored-standardizer fixed-maximum-LR gate"
        ),
        "variants": [VARIANT],
        "seeds": list(SEEDS),
        "model": asdict(config),
        "architecture": {
            "descriptor_dim": descriptor_dim,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "fixed_learned_poles": "checkerboard 50:50 per stage",
            "direction_mixer": "input-independent complex L2",
            "readout": ("stage1-2 global Q/R; stage3-4 global+2x2 Q/R/C; single affine head"),
            "cross_mode_graph": (
                "same-scale orientation cycle plus adjacent-scale same-orientation edges"
            ),
        },
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "modal_learning_rate_multiplier": 1.0 / 3.0,
            "pole_geometry_learning_rate_multiplier": 0.1,
            "weight_decay": 0.05,
            "warmup_epochs": 0,
            "schedule": "cosine decay from the initial maximum LR",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "precision": args.precision,
            "augmentation": ("RandomResizedCrop(224,bicubic)+HFlip+RandAugment(2,9)+RandomErasing"),
            "selection": "fixed final epoch; validation is not used for selection",
            "resume": ("epoch-boundary exact RNG restore; non-persistent augmentation workers"),
            "regularization_note": (
                "the conditioned Q/R/C descriptor has no running standardizer; "
                "analysis/synthesis and direction mixing use one-third LR, pole "
                "geometry uses one-tenth LR, and modal/norm/bias parameters have "
                "no weight decay"
            ),
        },
        "data": {
            "manifest_sha256": data_digest,
            "train_images": train_count,
            "validation_images": validation_count,
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "model": harness._digest(Path("src/lnet/spatialalphabet_h.py")),
            "scan": harness._digest(Path("src/lnet/alphabet2d.py")),
            "recurrence": harness._digest(Path("src/lnet/pac_recurrence.py")),
            "harness": harness._digest(Path("scripts/run_alphabet2d_imagenet100_nano.py")),
        },
    }
    # Match the persisted JSON representation exactly so tuple-valued config
    # fields do not invalidate an otherwise identical immutable contract.
    return json.loads(json.dumps(payload))


def _build(_variant: str, config: SpatialAlphabetHConfig) -> torch.nn.Module:
    if _variant != VARIANT:
        message = f"unknown SPATIALPHABET-H variant: {_variant}"
        raise ValueError(message)
    return build_spatialalphabet_h(config)


def _build_optimizer(
    model: torch.nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    modal = []
    pole_geometry = []
    geometry_tokens = ("log_damping_offset", "frequency_offset")
    modal_tokens = (".field.analysis.", "direction_mix", "pole_scale")
    no_decay_tokens = ("bias", "norm")
    for name, parameter in model.named_parameters():
        if any(token in name for token in geometry_tokens):
            pole_geometry.append(parameter)
        elif any(token in name for token in modal_tokens):
            modal.append(parameter)
        elif parameter.ndim < 2 or any(token in name for token in no_decay_tokens):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    learning_rate = recipe["learning_rate"]
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "lr": learning_rate,
                "weight_decay": recipe["weight_decay"],
            },
            {
                "params": no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": modal,
                "lr": learning_rate * recipe["modal_learning_rate_multiplier"],
                "weight_decay": 0.0,
            },
            {
                "params": pole_geometry,
                "lr": learning_rate * recipe["pole_geometry_learning_rate_multiplier"],
                "weight_decay": 0.0,
            },
        ]
    )


def _learning_rate_factor(epoch: int, epochs: int) -> float:
    progress = epoch / max(1, epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [root / "results" / f"{VARIANT}__seed{seed}.json" for seed in SEEDS]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    accuracies = [row["final_validation"]["accuracy"] for row in rows]
    mean = sum(accuracies) / len(accuracies)
    variance = sum((value - mean) ** 2 for value in accuracies) / (len(accuracies) - 1)
    payload = {
        "schema": contract["schema"],
        "final_validation_accuracy": {
            "per_seed": dict(zip((str(seed) for seed in SEEDS), accuracies, strict=True)),
            "mean": mean,
            "sample_standard_deviation": variance**0.5,
        },
        "parameter_count": sorted({row["parameters"] for row in rows}),
        "mean_training_examples_per_second": sum(
            row["complete_training_examples_per_second"] for row in rows
        )
        / len(rows),
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = (VARIANT,)
    harness.SEEDS = SEEDS
    harness.ImageNetNanoConfig = SpatialAlphabetHConfig
    harness.build_imagenet_nano = _build
    harness._build_optimizer = _build_optimizer
    harness._learning_rate_factor = _learning_rate_factor
    harness._contract = _contract
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
