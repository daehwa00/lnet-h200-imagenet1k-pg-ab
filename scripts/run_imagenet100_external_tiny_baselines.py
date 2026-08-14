# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Train external tiny-image baselines under the H-ALPHABET protocol."""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import timm
import torch
from torch import nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_imagenet100_nano as harness

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = (
    "mobilenet_v3_small",
    "shufflenet_v2_x1_0",
    "convnext_atto",
    "efficientformerv2_s0",
)
SEEDS = (501, 509, 521)


@dataclass(frozen=True, slots=True)
class ExternalBaselineConfig:
    output_dim: int = 100


def _build(variant: str, config: ExternalBaselineConfig) -> nn.Module:
    if variant == "mobilenet_v3_small":
        return models.mobilenet_v3_small(
            weights=None,
            num_classes=config.output_dim,
        )
    if variant == "shufflenet_v2_x1_0":
        return models.shufflenet_v2_x1_0(
            weights=None,
            num_classes=config.output_dim,
        )
    if variant in {"convnext_atto", "efficientformerv2_s0"}:
        return timm.create_model(
            variant,
            pretrained=False,
            num_classes=config.output_dim,
        )
    message = f"unknown external tiny baseline: {variant}"
    raise ValueError(message)


def _contract(args: Namespace) -> dict[str, Any]:
    config = ExternalBaselineConfig()
    data_digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload = {
        "schema": "lnet.imagenet100.external_tiny_baselines.shared_recipe.v1",
        "evidence_status": "external 100-epoch three-seed comparison",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": parameter_counts,
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": 3.0e-4,
            "weight_decay": 0.05,
            "warmup_epochs": 0,
            "schedule": "cosine decay from the initial maximum LR",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "precision": args.precision,
            "augmentation": ("RandomResizedCrop(224,bicubic)+HFlip+RandAugment(2,9)+RandomErasing"),
            "selection": "fixed final epoch; validation is not used for selection",
            "resume": ("epoch-boundary exact RNG restore; non-persistent augmentation workers"),
            "fairness": (
                "same data, budget, augmentation, optimizer family, and schedule "
                "as H-ALPHABET; architecture-specific official-recipe confirmation "
                "is a separate follow-up"
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
            "torchvision": getattr(models, "__version__", "module"),
            "timm": timm.__version__,
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_imagenet100_nano.py")),
        },
    }
    return json.loads(json.dumps(payload))


def _build_optimizer(
    model: nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer_options: dict[str, Any] = {}
    if bool(recipe.get("fused_optimizer", False)):
        optimizer_options["fused"] = True
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "lr": recipe["learning_rate"],
                "weight_decay": recipe["weight_decay"],
            },
            {
                "params": no_decay,
                "lr": recipe["learning_rate"],
                "weight_decay": 0.0,
            },
        ],
        **optimizer_options,
    )


def _learning_rate_factor(epoch: int, epochs: int) -> float:
    progress = epoch / max(1, epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [
        root / "results" / f"{variant}__seed{seed}.json" for variant in VARIANTS for seed in SEEDS
    ]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    variants = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        accuracies = [row["final_validation"]["accuracy"] for row in selected]
        mean = sum(accuracies) / len(accuracies)
        variance = sum((value - mean) ** 2 for value in accuracies) / (len(accuracies) - 1)
        variants[variant] = {
            "parameters": sorted({row["parameters"] for row in selected}),
            "accuracy_per_seed": dict(
                zip(
                    (str(seed) for seed in SEEDS),
                    accuracies,
                    strict=True,
                )
            ),
            "mean_final_accuracy": mean,
            "sample_standard_deviation": variance**0.5,
            "mean_training_examples_per_second": sum(
                row["complete_training_examples_per_second"] for row in selected
            )
            / len(selected),
        }
    payload = {
        "schema": contract["schema"],
        "variants": variants,
        "accuracy_ranking": sorted(
            VARIANTS,
            key=lambda variant: float(variants[variant]["mean_final_accuracy"]),
            reverse=True,
        ),
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.ImageNetNanoConfig = ExternalBaselineConfig
    harness.build_imagenet_nano = _build
    harness._contract = _contract
    harness._build_optimizer = _build_optimizer
    harness._learning_rate_factor = _learning_rate_factor
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
