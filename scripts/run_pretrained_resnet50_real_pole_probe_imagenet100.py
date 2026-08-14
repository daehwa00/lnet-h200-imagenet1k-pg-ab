#!/usr/bin/env python3
"""Probe a frozen pretrained real CNN feature map with one terminal D4 pole scan."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# ruff: noqa: SLF001
import json
import math
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, override

import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import Tensor, nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import InterpolationMode

from lnet.complex_scan_stage import ComplexScanStage
from lnet.image_layers import StandardizedAffineModalHead

if TYPE_CHECKING:
    from argparse import Namespace


GAP_VARIANT = "RN50L3-GAP"
ENERGY_VARIANT = "RN50L3-Energy96"
REAL_POLE_VARIANT = "RN50L3-RealPole96"
LAYER2_ENERGY_VARIANT = "RN50L2-Energy96"
LAYER2_REAL_POLE_VARIANT = "RN50L2-RealPole96"
VARIANTS = (
    GAP_VARIANT,
    ENERGY_VARIANT,
    REAL_POLE_VARIANT,
    LAYER2_ENERGY_VARIANT,
    LAYER2_REAL_POLE_VARIANT,
)
SEEDS = (501,)

POLE_MODES = 96
POLE_DESCRIPTOR_DIM = 4 * POLE_MODES
MAXIMUM_PHASE = math.pi * 0.65
PRETRAINED_WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


@dataclass(frozen=True, slots=True)
class ProbeVariantSpec:
    """Frozen feature boundary and descriptor used by one controlled probe."""

    feature_node: Literal["layer2", "layer3"]
    feature_channels: int
    feature_size: int
    descriptor: Literal["gap", "energy", "real_pole"]


VARIANT_SPECS = {
    GAP_VARIANT: ProbeVariantSpec("layer3", 1024, 14, "gap"),
    ENERGY_VARIANT: ProbeVariantSpec("layer3", 1024, 14, "energy"),
    REAL_POLE_VARIANT: ProbeVariantSpec("layer3", 1024, 14, "real_pole"),
    LAYER2_ENERGY_VARIANT: ProbeVariantSpec("layer2", 512, 28, "energy"),
    LAYER2_REAL_POLE_VARIANT: ProbeVariantSpec("layer2", 512, 28, "real_pole"),
}


@dataclass(frozen=True, slots=True)
class FrozenRealFeatureProbeConfig:
    """Only the task output width is configurable in this controlled probe."""

    output_dim: int = 100


def _initialize_projection_(projection: nn.Conv2d) -> None:
    with torch.no_grad():
        nn.init.orthogonal_(projection.weight.flatten(1))


class FrozenResNet50FeatureProbe(nn.Module):
    """Keep one external real feature map fixed while training a small probe."""

    def __init__(
        self,
        variant: str,
        output_dim: int,
        *,
        weights: ResNet50_Weights | None = PRETRAINED_WEIGHTS,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_SPECS:
            message = f"unsupported frozen ResNet-50 probe: {variant}"
            raise ValueError(message)
        if output_dim <= 0:
            message = "frozen real-feature probe output width must be positive"
            raise ValueError(message)

        self.spec = VARIANT_SPECS[variant]
        backbone = resnet50(weights=weights)
        self.backbone = create_feature_extractor(
            backbone,
            return_nodes={self.spec.feature_node: "feature"},
        )
        self.backbone.requires_grad_(requires_grad=False)
        self.backbone.eval()
        self.variant = variant

        self.projection: nn.Conv2d | None = None
        self.scan: ComplexScanStage | None = None
        if self.spec.descriptor != "gap":
            self.projection = nn.Conv2d(
                self.spec.feature_channels,
                POLE_MODES,
                kernel_size=1,
                bias=False,
            )
            _initialize_projection_(self.projection)
        if self.spec.descriptor == "real_pole":
            self.scan = ComplexScanStage(
                POLE_MODES,
                maximum_phase=MAXIMUM_PHASE,
                output_modes=None,
                scan_memory_policy="recompute",
            )

        descriptor_dim = {
            "gap": self.spec.feature_channels,
            "energy": POLE_MODES,
            "real_pole": POLE_DESCRIPTOR_DIM,
        }[self.spec.descriptor]
        self.classifier = StandardizedAffineModalHead(descriptor_dim, output_dim)

    @override
    def train(self, mode: bool = True) -> FrozenResNet50FeatureProbe:
        """Train the probe while permanently retaining pretrained BN statistics."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _feature_map(self, inputs: Tensor) -> Tensor:
        # no_grad, rather than inference_mode, leaves an ordinary tensor that a
        # trainable projection may safely save for its weight gradient.
        with torch.no_grad():
            feature = self.backbone(inputs)["feature"]
        if feature.ndim != 4 or feature.shape[1] != self.spec.feature_channels:
            message = (
                f"pretrained ResNet-50 {self.spec.feature_node} returned an incompatible "
                "feature map"
            )
            raise RuntimeError(message)
        return cast("Tensor", feature)

    def forward(self, inputs: Tensor) -> Tensor:
        feature = self._feature_map(inputs)
        if self.spec.descriptor == "gap":
            descriptor = feature.float().mean(dim=(2, 3))
        else:
            if self.projection is None:
                message = "projected frozen-CNN probe is missing its real projection"
                raise RuntimeError(message)
            drive_nchw = self.projection(feature)
            if self.spec.descriptor == "energy":
                descriptor = torch.log1p(drive_nchw.float().square().mean(dim=(2, 3)))
            else:
                if self.scan is None:
                    message = "real-pole probe is missing its terminal D4 scan"
                    raise RuntimeError(message)
                drive = drive_nchw.movedim(1, -1).contiguous()
                _, descriptor = self.scan(drive, torch.zeros_like(drive))
        return self.classifier(descriptor)


def _build_with_weights(
    variant: str,
    config: FrozenRealFeatureProbeConfig,
    *,
    weights: ResNet50_Weights | None,
) -> FrozenResNet50FeatureProbe:
    return FrozenResNet50FeatureProbe(
        variant,
        config.output_dim,
        weights=weights,
    )


def _build(
    variant: str,
    config: FrozenRealFeatureProbeConfig,
) -> FrozenResNet50FeatureProbe:
    return _build_with_weights(variant, config, weights=PRETRAINED_WEIGHTS)


def _assert_model(model: FrozenResNet50FeatureProbe, variant: str) -> None:
    if model.variant != variant:
        message = "frozen-CNN probe builder returned the wrong variant"
        raise RuntimeError(message)
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        message = "pretrained ResNet-50 feature extractor was not frozen"
        raise RuntimeError(message)
    if model.backbone.training:
        message = "pretrained ResNet-50 feature extractor must remain in eval mode"
        raise RuntimeError(message)
    spec = VARIANT_SPECS[variant]
    expected_dim = {
        "gap": spec.feature_channels,
        "energy": POLE_MODES,
        "real_pole": POLE_DESCRIPTOR_DIM,
    }[spec.descriptor]
    if (
        model.classifier.input_dim != expected_dim
        or model.classifier.standardizer.affine
        or model.classifier.linear.in_features != expected_dim
    ):
        message = f"{variant} changed the standardized affine-head contract"
        raise RuntimeError(message)
    if spec.descriptor == "gap" and (model.projection is not None or model.scan is not None):
        message = "GAP control unexpectedly contains a projection or pole scan"
        raise RuntimeError(message)
    if spec.descriptor == "energy" and (model.projection is None or model.scan is not None):
        message = "energy control changed its projection-only contract"
        raise RuntimeError(message)
    if spec.descriptor == "real_pole":
        if model.projection is None or model.scan is None:
            message = "real-pole probe is missing its projection or D4 scan"
            raise RuntimeError(message)
        if (
            model.scan.modes != POLE_MODES
            or model.scan.output_modes is not None
            or model.scan.scan_memory_policy != "recompute"
        ):
            message = "real-pole probe changed the terminal D4 descriptor contract"
            raise RuntimeError(message)


def _transforms() -> tuple[transforms.Compose, nn.Module]:
    mean = (0.485, 0.456, 0.406)
    standard_deviation = (0.229, 0.224, 0.225)
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
            transforms.RandomErasing(p=0.25),
        ]
    )
    return train, PRETRAINED_WEIGHTS.transforms()


def _trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _parameter_counts() -> tuple[dict[str, int], dict[str, int]]:
    total: dict[str, int] = {}
    trainable: dict[str, int] = {}
    config = FrozenRealFeatureProbeConfig()
    for variant in VARIANTS:
        model = _build_with_weights(variant, config, weights=None)
        _assert_model(model, variant)
        total[variant] = sum(parameter.numel() for parameter in model.parameters())
        trainable[variant] = _trainable_parameter_count(model)
    return total, trainable


def _variant_contract(variant: str) -> dict[str, str]:
    spec = VARIANT_SPECS[variant]
    feature = (
        f"frozen torchvision ResNet50 IMAGENET1K_V2 {spec.feature_node} "
        f"[{spec.feature_channels},{spec.feature_size},{spec.feature_size}]"
    )
    if spec.descriptor == "gap":
        return {
            "feature": feature,
            "descriptor": f"GAP over real {spec.feature_channels}-channel feature",
            "head": (
                f"BatchNorm{spec.feature_channels}-affine-false then "
                f"Linear{spec.feature_channels}-to-100"
            ),
        }
    reader = (
        "orthogonal-initialized bias-free real "
        f"Linear{spec.feature_channels}-to-{POLE_MODES}"
    )
    if spec.descriptor == "energy":
        return {
            "feature": feature,
            "reader": reader,
            "descriptor": "log1p spatial mean square; no scan",
            "head": f"BatchNorm{POLE_MODES}-affine-false then Linear{POLE_MODES}-to-100",
        }
    return {
        "feature": feature,
        "reader": f"{reader}; U is injected as U+i0",
        "scan": (
            "one optimized terminal associative D4 product scan at "
            f"{spec.feature_size}x{spec.feature_size}; {POLE_MODES} poles; "
            f"full-grid raw directional log-energy Q{POLE_DESCRIPTOR_DIM}"
        ),
        "head": (
            f"BatchNorm{POLE_DESCRIPTOR_DIM}-affine-false then "
            f"Linear{POLE_DESCRIPTOR_DIM}-to-100"
        ),
    }


def _contract(args: Namespace) -> dict[str, Any]:
    if args.gradient_accumulation_steps < 1:
        message = "gradient_accumulation_steps must be positive"
        raise ValueError(message)
    loader_workers = harness._active_loader_workers(args.workers)
    persistent_workers = harness._persistent_loader_workers(loader_workers)
    data_digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    selected_variants = tuple(args.variants)
    if not selected_variants or not set(selected_variants) <= set(VARIANTS):
        message = "frozen real-feature probe selected an unsupported variant set"
        raise ValueError(message)
    total_parameters, trainable_parameters = _parameter_counts()
    recipe = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "optimizer": "AdamW",
        "learning_rate": 3.0e-3,
        "pole_geometry_learning_rate_multiplier": 1.0,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "schedule": "cosine",
        "label_smoothing": 0.1,
        "mixup_alpha": 0.8,
        "precision": args.precision,
        "loader_workers": loader_workers,
        "loader_persistent_workers": persistent_workers,
        "validation_loader_persistent_workers": False,
        "loader_prefetch_factor": harness.PREFETCH_FACTOR,
        "cpu_affinity": None,
        "device_prefetch_stream": True,
        "channels_last": True,
        "compile_mode": "reduce-overhead",
        "fused_optimizer": True,
        "augmentation": (
            "RandomResizedCrop224-bilinear+HFlip+RandAugment(2,9)+RandomErasing; ImageNet mean/std"
        ),
        "evaluation_transform": "ResNet50_Weights.IMAGENET1K_V2.transforms",
        "selection": "fixed final epoch; validation is not used for selection",
        "resume": "epoch-boundary RNG restore; frozen pretrained weights are checkpointed",
    }
    continuation_source = os.environ.get("PROBE_CONTINUATION_SOURCE_ROOT")
    if continuation_source:
        recipe["continuation"] = {
            "source_root": continuation_source,
            "source_epoch": int(os.environ["PROBE_CONTINUATION_SOURCE_EPOCH"]),
            "target_epoch": args.epochs,
            "state": "model, optimizer moments, RNG, global step, and history preserved",
            "scheduler": (
                "resume at the source run's last nonzero training LR, then follow a "
                "rescaled target-epoch cosine tail without an LR discontinuity"
            ),
        }
    return {
        "schema": "lnet.pretrained_resnet50.feature_level_real_pole_probe.imagenet100.v2",
        "evidence_status": "frozen external-feature diagnostic",
        "variants": list(selected_variants),
        "seeds": list(SEEDS),
        "model": asdict(FrozenRealFeatureProbeConfig()),
        "parameter_counts": {variant: total_parameters[variant] for variant in selected_variants},
        "trainable_parameter_counts": {
            variant: trainable_parameters[variant] for variant in selected_variants
        },
        "variant_configs": {
            variant: _variant_contract(variant) for variant in selected_variants
        },
        "controlled_comparison": {
            "layer3_Energy96_minus_GAP": "value of a learned 96-D local-energy readout",
            "layer3_RealPole96_minus_Energy96": (
                "incremental value of spatial pole recurrence at 14x14"
            ),
            "layer2_RealPole96_minus_Energy96": (
                "incremental value of spatial pole recurrence at 28x28"
            ),
            "backbone": "identical frozen external pretrained CNN in every variant",
        },
        "recipe": recipe,
        "data": {
            "manifest_sha256": data_digest,
            "train_images": train_count,
            "validation_images": validation_count,
            "pretraining_overlap_warning": (
                "ImageNet-1K supervision includes the ImageNet-100 classes; absolute accuracy "
                "is not comparable to a from-scratch native-backbone result"
            ),
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision_weights": "ResNet50_Weights.IMAGENET1K_V2",
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path(harness.__file__)),
            "scan_stage": harness._digest(Path("src/lnet/complex_scan_stage.py")),
            "scan_pipeline": harness._digest(Path("src/lnet/pac_product_scan_pipeline.py")),
        },
    }


_POLE_PARAMETER_NAMES = frozenset(
    {
        "scan.damping_logits_x",
        "scan.damping_logits_y",
        "scan.phase_x",
        "scan.phase_y",
    }
)


def _build_optimizer(
    model: nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    pole_geometry: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in _POLE_PARAMETER_NAMES:
            pole_geometry.append(parameter)
        elif parameter.ndim < 2 or name.endswith(".bias") or "standardizer" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": recipe["learning_rate"],
                "weight_decay": recipe["weight_decay"],
                "group_name": "probe_decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": recipe["learning_rate"],
                "weight_decay": 0.0,
                "group_name": "probe_no_decay",
            }
        )
    if pole_geometry:
        groups.append(
            {
                "params": pole_geometry,
                "lr": (recipe["learning_rate"] * recipe["pole_geometry_learning_rate_multiplier"]),
                "weight_decay": 0.0,
                "group_name": "pole_geometry",
            }
        )
    return torch.optim.AdamW(groups, fused=bool(recipe.get("fused_optimizer", False)))


def _summarize(root: Path, contract_payload: dict[str, Any]) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for variant in contract_payload["variants"]:
        result = root / "results" / f"{variant}__seed{SEEDS[0]}.json"
        if not result.exists():
            return None
        rows.append(json.loads(result.read_text()))
    accuracy = {row["variant"]: row["final_validation"]["accuracy"] for row in rows}
    payload: dict[str, Any] = {
        "schema": "lnet.pretrained_resnet50.feature_level_real_pole_probe.summary.v2",
        "final_validation_accuracy": accuracy,
        "interpretation": (
            "Within each frozen feature level, RealPole96-Energy96 isolates the incremental "
            "value of D4 spatial recurrence; absolute accuracy remains "
            "ImageNet-1K-pretraining-contaminated."
        ),
    }
    if GAP_VARIANT in accuracy and ENERGY_VARIANT in accuracy:
        payload["energy_minus_gap_pp"] = 100.0 * (accuracy[ENERGY_VARIANT] - accuracy[GAP_VARIANT])
    if ENERGY_VARIANT in accuracy and REAL_POLE_VARIANT in accuracy:
        payload["real_pole_minus_energy_pp"] = 100.0 * (
            accuracy[REAL_POLE_VARIANT] - accuracy[ENERGY_VARIANT]
        )
        payload["layer3_real_pole_minus_energy_pp"] = payload[
            "real_pole_minus_energy_pp"
        ]
    if LAYER2_ENERGY_VARIANT in accuracy and LAYER2_REAL_POLE_VARIANT in accuracy:
        payload["layer2_real_pole_minus_energy_pp"] = 100.0 * (
            accuracy[LAYER2_REAL_POLE_VARIANT] - accuracy[LAYER2_ENERGY_VARIANT]
        )
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    # The shared harness owns loading, resume, logging, and fixed-budget training.
    # This process-local transform replacement pins the external weight's expected
    # evaluation preprocessing without changing other experiment runners.
    harness._transforms = _transforms
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=FrozenRealFeatureProbeConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=_build_optimizer,
            prepare_model=harness._prepare_model,
            train_epoch=harness._train_epoch,
            evaluate=harness._evaluate,
            wandb_model_metrics=harness._wandb_model_metrics,
            summarize=_summarize,
        )
    )


if __name__ == "__main__":
    main()
