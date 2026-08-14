#!/usr/bin/env python3
"""End-to-end confirmation of the frozen-Q A2D head finalists."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import complex_scan_imagenet100_runtime as runtime
import run_a2d_d4_pathmix_imagenet100 as backbone
import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import nn

from lnet.a2d_head_design import FusionHead, StageResidualHead
from lnet.complex_scan import ComplexScanConfig, ModalFusionHead

if TYPE_CHECKING:
    from argparse import Namespace


VARIANTS = ("A2D-W768", "A2D-StageResidual", "A2D-Drop020")
SEEDS = (501,)


def _variant_config(variant: str, config: ComplexScanConfig) -> ComplexScanConfig:
    if variant not in VARIANTS:
        message = f"unknown A2D head-design finalist: {variant}"
        raise ValueError(message)
    return backbone._variant_config(backbone.VARIANT, config)


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    model = backbone._build(backbone.VARIANT, config)
    if variant == "A2D-W768":
        model.classifier = ModalFusionHead(model.descriptor_dim, 768, config.output_dim)
    elif variant == "A2D-StageResidual":
        model.classifier = StageResidualHead(
            nn.BatchNorm1d(model.descriptor_dim, affine=False),
            config.output_dim,
            64,
        )
    elif variant == "A2D-Drop020":
        model.classifier = FusionHead(
            nn.BatchNorm1d(model.descriptor_dim, affine=False),
            config.output_dim,
            (256,),
            activation="gelu",
            hidden_norm="rms_after",
            dropout=0.2,
        )
    else:
        message = f"unsupported finalist: {variant}"
        raise ValueError(message)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    active = backbone._variant_config(backbone.VARIANT, config)
    models = {variant: _build(variant, config) for variant in VARIANTS}
    digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    payload = {
        "schema": "lnet.a2d.head_design_e2e.imagenet100.v1",
        "evidence_status": "one-seed 100-epoch end-to-end head confirmation",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "backbone": {
            "name": "A2D-D4-PathMix",
            "config": asdict(active),
            "descriptor": "3 stages x 4 directions x 48 Q energies = 576",
        },
        "heads": {
            "A2D-W768": "BN(affine=False)-Linear768-GELU-RMSNorm-Linear100",
            "A2D-StageResidual": (
                "BN-affine main plus three shared-width stage embeddings and "
                "a beta-scaled cross-stage residual"
            ),
            "A2D-Drop020": "BN-Linear256-GELU-RMSNorm-Dropout0.2-Linear100",
        },
        "parameter_counts": {
            variant: sum(parameter.numel() for parameter in model.parameters())
            for variant, model in models.items()
        },
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "precision": args.precision,
            "optimizer": "matched fused AdamW with pole-aware groups",
            "fused_optimizer": True,
            "learning_rate": 3.0e-3,
            "modal_learning_rate_multiplier": 1.0 / 3.0,
            "pole_geometry_learning_rate_multiplier": 0.1,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "warmup plus cosine",
            "mixup_alpha": 0.8,
            "channels_last": True,
            "compile_mode": "default",
            "augmentation": "matched A2D ImageNet-100 recipe",
            "selection": "fixed final epoch; no validation selection",
        },
        "data": {
            "manifest_sha256": digest,
            "train_images": train_count,
            "validation_images": validation_count,
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_imagenet100_nano.py")),
            "backbone": harness._digest(Path("scripts/run_a2d_d4_pathmix_imagenet100.py")),
            "head": harness._digest(Path("src/lnet/a2d_head_design.py")),
        },
    }
    return json.loads(json.dumps(payload))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [root / "results" / f"{variant}__seed501.json" for variant in VARIANTS]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    payload = {
        "schema": contract["schema"],
        "variants": {
            row["variant"]: {
                "parameters": row["parameters"],
                "final_validation": row["final_validation"],
                "best_validation_accuracy_diagnostic": row["best_validation_accuracy_diagnostic"],
                "training_examples_per_second": row["complete_training_examples_per_second"],
            }
            for row in rows
        },
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=runtime.optimizer_source._build_optimizer,
            prepare_model=runtime.base._prepare_model,
            summarize=_summarize,
        )
    )


if __name__ == "__main__":
    main()
