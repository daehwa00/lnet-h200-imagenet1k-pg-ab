# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
"""Run the prioritized complex scan ImageNet-100 follow-ups."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_alphabet2d_imagenet100_nano as harness
import run_complex_scan_augmented_cifar100 as optimizer_source
import run_complex_scan_zero_init_imagenet100 as base
import torch

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = (
    "dual_fusion256_lrq64",
    "coarse_heavy_fusion256",
    "capacity_fusion384",
    "capacity_dual_fusion384_lrq64",
)
SEEDS = (501, 509, 521)
MANIFEST_ENV = "LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256"


def _expected_imagenet100_manifest() -> str:
    value = os.environ.get(MANIFEST_ENV, base.IMAGENET100_MANIFEST_SHA256)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        message = f"{MANIFEST_ENV} must be a lowercase SHA-256 digest"
        raise ValueError(message)
    return value


def _variant_config(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanConfig:
    common = base._variant_config(config)
    if variant == "dual_fusion256_lrq64":
        return replace(common, fusion_width=256, dual_fusion_lrq_head=True)
    if variant == "coarse_heavy_fusion256":
        return replace(
            common,
            modes=(24, 32, 48),
            augmented_widths=(64, 80),
            fusion_width=256,
        )
    if variant == "capacity_fusion384":
        return replace(
            common,
            stem_width=96,
            modes=(48, 48, 48),
            augmented_widths=(96, 96),
            fusion_width=384,
        )
    if variant == "capacity_dual_fusion384_lrq64":
        return replace(
            common,
            stem_width=96,
            modes=(48, 48, 48),
            augmented_widths=(96, 96),
            fusion_width=384,
            dual_fusion_lrq_head=True,
        )
    message = f"unknown complex scan follow-up variant: {variant}"
    raise ValueError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return ComplexScanBackbone(_variant_config(variant, config))


def _contract(args: Namespace) -> dict[str, Any]:
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    variants = {variant: _variant_config(variant, config) for variant in VARIANTS}
    models = {variant: ComplexScanBackbone(active) for variant, active in variants.items()}
    data_digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    expected_manifest = _expected_imagenet100_manifest()
    if data_digest != expected_manifest:
        message = "ImageNet-100 data manifest does not match the existing baselines"
        raise RuntimeError(message)
    return json.loads(
        json.dumps(
            {
                "schema": "lnet.complex_scan.followups.imagenet100.v1",
                "evidence_status": "queued 100-epoch three-seed prioritized follow-ups",
                "variants": list(VARIANTS),
                "seeds": list(SEEDS),
                "priority": list(VARIANTS),
                "model": asdict(config),
                "variant_configs": {name: asdict(active) for name, active in variants.items()},
                "parameter_counts": {
                    name: sum(parameter.numel() for parameter in model.parameters())
                    for name, model in models.items()
                },
                "recipe": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": 1,
                    "effective_batch_size": args.batch_size,
                    "optimizer": "AdamW",
                    "fused_optimizer": True,
                    "learning_rate": 3.0e-3,
                    "modal_learning_rate_multiplier": 1.0 / 3.0,
                    "pole_geometry_learning_rate_multiplier": 0.1,
                    "weight_decay": 0.05,
                    "warmup_epochs": 5,
                    "schedule": "warmup plus cosine",
                    "label_smoothing": 0.1,
                    "mixup_alpha": 0.8,
                    "precision": args.precision,
                    "loader_prefetch_factor": harness.PREFETCH_FACTOR,
                    "device_prefetch_stream": True,
                    "device_prefetch_scope": "copy_only",
                    "fused_h2d_channels_last": False,
                    "compile_mode": "default",
                    "channels_last": True,
                    "emulate_precision_casts": True,
                    "augmentation": (
                        "RandomResizedCrop(224,bicubic)+HFlip+RandAugment(2,9)+RandomErasing"
                    ),
                    "selection": "fixed final epoch; validation is not used for selection",
                    "resume": "epoch-boundary exact RNG restore",
                    "kernel": "fused state-plus-stop-gradient-variance Triton recurrence",
                    "runtime_bundle": (
                        "conjugate-pole reuse + split vertical materialization + capture-safe "
                        "orthogonal + channels-last + default torch.compile + fused AdamW"
                    ),
                },
                "architecture": {
                    "backbone": "S2D pole-main plus Augmented Complex FFN",
                    "stem": "two 3x3 stride-2 convolutions",
                    "scan_boundary": "zero",
                    "descriptors": {
                        "dual_fusion256_lrq64": "global 384; Fusion256 + beta*LRQ64",
                        "coarse_heavy_fusion256": "global 416; modes 24/32/48",
                        "capacity_fusion384": "global 576; modes 48/48/48",
                        "capacity_dual_fusion384_lrq64": (
                            "global 576; modes 48/48/48; Fusion384 + beta*LRQ64"
                        ),
                    },
                },
                "data": {
                    "manifest_sha256": data_digest,
                    "train_images": train_count,
                    "validation_images": validation_count,
                },
                "comparison": {
                    "existing_baseline_schema": (
                        "lnet.imagenet100.external_tiny_baselines.shared_recipe.v1"
                    ),
                    "existing_baseline_manifest_sha256": expected_manifest,
                    "existing_final_validation_means": base.EXISTING_BASELINES,
                },
                "source_sha256": {
                    "runner": harness._digest(Path(__file__)),
                    "harness": harness._digest(Path("scripts/run_alphabet2d_imagenet100_nano.py")),
                    "base_runner": harness._digest(
                        Path("scripts/run_complex_scan_zero_init_imagenet100.py")
                    ),
                    "model": harness._digest(Path("src/lnet/complex_scan.py")),
                },
            }
        )
    )


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [
        root / "results" / f"{variant}__seed{seed}.json" for variant in VARIANTS for seed in SEEDS
    ]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        accuracies = [float(row["final_validation"]["accuracy"]) for row in selected]
        mean = sum(accuracies) / len(accuracies)
        variance = sum((value - mean) ** 2 for value in accuracies) / (len(accuracies) - 1)
        summaries[variant] = {
            "parameters": selected[0]["parameters"],
            "accuracy_per_seed": dict(zip(map(str, SEEDS), accuracies, strict=True)),
            "mean_final_accuracy": mean,
            "sample_standard_deviation": variance**0.5,
            "mean_training_examples_per_second": sum(
                float(row["complete_training_examples_per_second"]) for row in selected
            )
            / len(selected),
        }
    payload = {"schema": contract["schema"], "variants": summaries}
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.ImageNetNanoConfig = ComplexScanConfig
    harness.build_imagenet_nano = _build
    harness._contract = _contract
    harness._build_optimizer = optimizer_source._build_optimizer
    harness._prepare_model = base._prepare_model
    harness._summarize = _summarize
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main()


if __name__ == "__main__":
    main()
