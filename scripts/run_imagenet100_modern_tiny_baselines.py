#!/usr/bin/env python3
"""Train modern tiny classifiers under the matched D4 ImageNet-100 recipe."""

# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_alphabet2d_imagenet100_nano as harness
import run_imagenet100_external_tiny_baselines as external
import timm
import torch
from torch import nn

import lnet.modern_tiny_models as modern_models
from lnet.modern_tiny_models import RepViTM09, TinyNeXtT

if TYPE_CHECKING:
    from argparse import Namespace


TINY_NEXT_T = "tinynext_t"
FAST_VIT_T8 = "fastvit_t8"
REP_VIT_M09 = "repvit_m0_9"
VARIANTS = (TINY_NEXT_T, FAST_VIT_T8, REP_VIT_M09)
SEEDS = (501,)

OFFICIAL_SOURCES = {
    TINY_NEXT_T: {
        "repository": "https://github.com/yuffeenn/TinyNeXt",
        "commit": "3eb30a847f8e5916b975f139d101a0da1f0d7e67",
        "implementation": "vendored official TinyNeXt-T topology",
    },
    FAST_VIT_T8: {
        "repository": "https://github.com/apple/ml-fastvit",
        "commit": "8af5928238cab99c45f64fc3e4e7b1516b8224ba",
        "implementation": "timm FastViT-T8 topology verified against the official builder",
    },
    REP_VIT_M09: {
        "repository": "https://github.com/THU-MIG/RepViT",
        "commit": "298f42075eda5d2e6102559fad260c970769d34e",
        "implementation": "vendored official RepViT-M0.9 training topology",
    },
}


@dataclass(frozen=True, slots=True)
class ModernTinyConfig:
    output_dim: int = 100


def _build(variant: str, config: ModernTinyConfig) -> nn.Module:
    if variant == TINY_NEXT_T:
        model = TinyNeXtT(num_classes=config.output_dim)
    elif variant == FAST_VIT_T8:
        model = timm.create_model(
            FAST_VIT_T8,
            pretrained=False,
            num_classes=config.output_dim,
        )
    elif variant == REP_VIT_M09:
        model = RepViTM09(num_classes=config.output_dim)
    else:
        message = f"unknown modern tiny baseline: {variant}"
        raise ValueError(message)
    # These backbones replay the same compiled block topology many times in a
    # single forward. Classic CUDA Graph replay preserves saved activations;
    # CUDA Graph Trees may recycle a stage output before autograd consumes it.
    model._lnet_requires_classic_cudagraph = True  # type: ignore[attr-defined]
    return model


def _variant_config(variant: str, parameters: int) -> dict[str, Any]:
    return {
        "name": variant,
        "num_classes": 100,
        "parameters": parameters,
        "pretrained": False,
        "distillation": False,
        "source": OFFICIAL_SOURCES[variant],
    }


def _contract(args: Namespace) -> dict[str, Any]:
    payload = harness._contract(args)
    config = ModernTinyConfig()
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload.update(
        {
            "schema": "lnet.imagenet100.modern_tiny_baselines.d4_matched_bf16.v1",
            "evidence_status": "one-seed modern tiny comparison under the D4 recipe",
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "model": asdict(config),
            "parameter_counts": parameter_counts,
            "variant_configs": {
                variant: _variant_config(variant, parameter_counts[variant]) for variant in VARIANTS
            },
        }
    )
    payload["recipe"].update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "optimizer": "AdamW (fused, norm/bias excluded from weight decay)",
            "fused_optimizer": True,
            "learning_rate": 3.0e-3,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "five-epoch warmup plus cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "precision": args.precision,
            "compile_mode": os.environ.get("LNET_COMPILE_MODE", "default"),
            "matmul_precision": "high (TF32 enabled)",
            "channels_last": True,
            "augmentation": "matched D4 ImageNet-100 public recipe",
            "selection": "fixed epoch 100; no within-run validation selection",
            "resume": "epoch-boundary exact RNG restore",
        }
    )
    payload["runtime"].update(
        {
            "hostname": platform.node(),
            "timm": timm.__version__,
        }
    )
    payload["source_sha256"] = {
        "runner": harness._digest(Path(__file__)),
        "harness": harness._digest(Path(harness.__file__)),
        "modern_tiny_models": harness._digest(Path(modern_models.__file__)),
    }
    return json.loads(json.dumps(payload))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [root / "results" / f"{variant}__seed501.json" for variant in VARIANTS]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    variants = {
        row["variant"]: {
            "parameters": row["parameters"],
            "final_validation_accuracy": row["final_validation"]["accuracy"],
            "training_examples_per_second": row["complete_training_examples_per_second"],
        }
        for row in rows
    }
    payload = {
        "schema": contract["schema"],
        "seed": SEEDS[0],
        "variants": variants,
        "accuracy_ranking": sorted(
            VARIANTS,
            key=lambda variant: float(variants[variant]["final_validation_accuracy"]),
            reverse=True,
        ),
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.WANDB_MODEL_ALIASES.update(
        {
            TINY_NEXT_T: "TinyNeXt-T",
            FAST_VIT_T8: "FastViT-T8",
            REP_VIT_M09: "RepViT-M0.9",
        }
    )
    harness.main(
        harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ModernTinyConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=external._build_optimizer,
            prepare_model=harness._prepare_model,
            train_epoch=harness._train_epoch,
            evaluate=harness._evaluate,
            wandb_model_metrics=harness._wandb_model_metrics,
            summarize=_summarize,
        )
    )


if __name__ == "__main__":
    main()
