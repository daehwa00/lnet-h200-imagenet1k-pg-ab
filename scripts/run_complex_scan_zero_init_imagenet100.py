# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the CIFAR-selected complex zero-init complex scan backbone on ImageNet-100."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_imagenet100_nano as harness
import run_complex_scan_augmented_cifar100 as optimizer_source
import run_complex_scan_stage_carry_cifar100 as architecture_source

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
from lnet.pac_capture_safe_orthogonal import prepare_capture_safe_orthogonal_

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("s2d_pole_main_zero_init",)
SEEDS = (501, 509, 521)
MODES = (32, 32, 32)
AUGMENTED_WIDTHS = (64, 64)
QUADRATIC_RANK = 64
IMAGENET100_MANIFEST_SHA256 = "4772b7e5311ad9b6912c1e78f2923cd1a9a74b154d7cec9644c4338155842db9"
EXISTING_BASELINES = {
    "mobilenet_v3_small": {"accuracy": 0.6678, "parameters": 1_620_356},
    "shufflenet_v2_x1_0": {"accuracy": 0.6927333333333333, "parameters": 1_356_104},
    "convnext_atto": {"accuracy": 0.7219333333333333, "parameters": 3_406_620},
    "efficientformerv2_s0": {"accuracy": 0.7984, "parameters": 3_281_656},
    "previous_full_s2d_complex_scan": {"accuracy": 0.6452, "parameters": 322_072},
}
_BASE_PREPARE_MODEL = harness._prepare_model


def _variant_config(config: ComplexScanConfig) -> ComplexScanConfig:
    selected = architecture_source._variant_config("s2d_pole_main", config)
    return replace(
        selected,
        modes=MODES,
        augmented_widths=AUGMENTED_WIDTHS,
        quadratic_rank=QUADRATIC_RANK,
    )


def _build(_variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return ComplexScanBackbone(_variant_config(config))


def _prepare_model(model: torch.nn.Module, recipe: dict[str, Any]) -> torch.nn.Module:
    prepared = _BASE_PREPARE_MODEL(model, recipe)
    replaced_paths = prepare_capture_safe_orthogonal_(prepared)
    if not replaced_paths:
        message = "optimized ImageNet-100 runtime found no matrix-exp parametrizations"
        raise RuntimeError(message)
    return prepared


def _contract(args: Namespace) -> dict[str, Any]:
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
        modes=MODES,
        quadratic_rank=QUADRATIC_RANK,
    )
    variant_config = _variant_config(config)
    model = ComplexScanBackbone(variant_config)
    data_digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    payload = {
        "schema": "lnet.complex_scan.zero_init.imagenet100.optimized.v3",
        "evidence_status": "100-epoch three-seed external-baseline comparison",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "variant_configs": {VARIANTS[0]: asdict(variant_config)},
        "parameter_counts": {
            VARIANTS[0]: sum(parameter.numel() for parameter in model.parameters())
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
            "compile_mode": "default",
            "channels_last": True,
            "emulate_precision_casts": True,
            "augmentation": ("RandomResizedCrop(224,bicubic)+HFlip+RandAugment(2,9)+RandomErasing"),
            "selection": "fixed final epoch; validation is not used for selection",
            "resume": "epoch-boundary exact RNG restore",
            "kernel": "fused state-plus-stop-gradient-variance Triton recurrence",
            "runtime_bundle": (
                "conjugate-pole reuse + split vertical materialization + capture-safe "
                "orthogonal + channels-last + default torch.compile + fused AdamW"
            ),
        },
        "architecture": {
            "backbone": "S2D pole-main plus width-64 Augmented Complex FFN",
            "stem": "3x3 stride-2 convolution twice: 224 -> 112 -> 56",
            "scan_boundary": "zero for every axis, direction, stage, and mode",
            "stages": "static 32->32->32 complex modes plus terminal pole bank",
            "augmented_hidden_widths": list(AUGMENTED_WIDTHS),
            "descriptor": "three-stage raw directional energy, 384 coordinates",
            "head": "LRQ64",
            "runtime_semantics": "parameter-identical BF16 training path",
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
            "existing_baseline_manifest_sha256": (IMAGENET100_MANIFEST_SHA256),
            "shared_controls": [
                "dataset manifest",
                "224px transforms",
                "100 epochs",
                "seeds 501/509/521",
                "AdamW family and cosine schedule",
                "fixed-final-epoch validation",
            ],
            "recipe_difference": (
                "batch size 256 matches the archived external baselines; the model retains "
                "its architecture-specific learning-rate and parameter-group recipe"
            ),
            "existing_final_validation_means": EXISTING_BASELINES,
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_imagenet100_nano.py")),
            "optimizer": harness._digest(Path("scripts/run_complex_scan_augmented_cifar100.py")),
            "architecture": harness._digest(
                Path("scripts/run_complex_scan_stage_carry_cifar100.py")
            ),
            "model": harness._digest(Path("src/lnet/complex_scan.py")),
            "recurrence": harness._digest(Path("src/lnet/pac_triton_recurrence_op.py")),
            "capture_safe_orthogonal": harness._digest(
                Path("src/lnet/pac_capture_safe_orthogonal.py")
            ),
        },
    }
    if data_digest != IMAGENET100_MANIFEST_SHA256:
        message = "ImageNet-100 data manifest does not match the existing baselines"
        raise RuntimeError(message)
    return json.loads(json.dumps(payload))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [root / "results" / f"{VARIANTS[0]}__seed{seed}.json" for seed in SEEDS]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    accuracies = [float(row["final_validation"]["accuracy"]) for row in rows]
    mean = sum(accuracies) / len(accuracies)
    variance = sum((value - mean) ** 2 for value in accuracies) / (len(accuracies) - 1)
    baseline_deltas = {
        name: 100.0 * (mean - float(reference["accuracy"]))
        for name, reference in EXISTING_BASELINES.items()
    }
    ranking = sorted(
        {VARIANTS[0]: mean}
        | {name: float(reference["accuracy"]) for name, reference in EXISTING_BASELINES.items()},
        key=lambda name: (
            mean if name == VARIANTS[0] else float(EXISTING_BASELINES[name]["accuracy"])
        ),
        reverse=True,
    )
    payload = {
        "schema": contract["schema"],
        "variant": VARIANTS[0],
        "parameters": rows[0]["parameters"],
        "accuracy_per_seed": dict(zip((str(seed) for seed in SEEDS), accuracies, strict=True)),
        "mean_final_accuracy": mean,
        "sample_standard_deviation": variance**0.5,
        "mean_training_examples_per_second": sum(
            float(row["complete_training_examples_per_second"]) for row in rows
        )
        / len(rows),
        "delta_to_existing_baselines_pp": baseline_deltas,
        "accuracy_ranking": ranking,
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.ImageNetNanoConfig = ComplexScanConfig
    harness.build_imagenet_nano = _build
    harness._contract = _contract
    harness._build_optimizer = optimizer_source._build_optimizer
    harness._prepare_model = _prepare_model
    harness._summarize = _summarize
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main()


if __name__ == "__main__":
    main()
