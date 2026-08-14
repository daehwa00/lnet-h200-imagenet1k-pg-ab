# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Train parameter-matched ultra-tiny CNN baselines on CIFAR-100."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn
from torchvision.models import mobilenet_v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alphabet2d_cifar100_nano as harness
import cifar100_packed_data as packed_data

if TYPE_CHECKING:
    from argparse import Namespace

VARIANTS = ("resnet20", "resnet32", "mobilenetv2_015x")
SEEDS = (401,)


class _CifarResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, input_width: int, output_width: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_width, output_width, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_width, output_width, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_width)
        if stride != 1 or input_width != output_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_width, output_width, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_width),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + residual)


class CifarResNet(nn.Module):
    def __init__(self, blocks_per_stage: int, output_dim: int = 100) -> None:
        super().__init__()
        self.width = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._stage(16, blocks_per_stage, stride=1)
        self.stage2 = self._stage(32, blocks_per_stage, stride=2)
        self.stage3 = self._stage(64, blocks_per_stage, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, output_dim)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def _stage(self, width: int, blocks: int, *, stride: int) -> nn.Sequential:
        layers = [_CifarResidualBlock(self.width, width, stride)]
        self.width = width
        layers.extend(_CifarResidualBlock(width, width) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.stage3(self.stage2(self.stage1(self.stem(inputs))))
        return self.classifier(self.pool(features).flatten(1))


@dataclass(frozen=True, slots=True)
class UltraTinyBaselineConfig:
    output_dim: int = 100
    mobilenet_width_mult: float = 0.15
    mobilenet_first_stride: int = 1


def _build(variant: str, config: UltraTinyBaselineConfig) -> nn.Module:
    if variant == "resnet20":
        return CifarResNet(3, config.output_dim)
    if variant == "resnet32":
        return CifarResNet(5, config.output_dim)
    if variant == "mobilenetv2_015x":
        model = mobilenet_v2(
            width_mult=config.mobilenet_width_mult,
            num_classes=config.output_dim,
        )
        first_block = cast("nn.Sequential", model.features[0])
        first_conv = cast("nn.Conv2d", first_block[0])
        first_conv.stride = (
            config.mobilenet_first_stride,
            config.mobilenet_first_stride,
        )
        return model
    message = f"unknown ultra-tiny baseline: {variant}"
    raise ValueError(message)


def _contract(args: Namespace) -> dict[str, Any]:
    config = UltraTinyBaselineConfig()
    payload = {
        "schema": "lnet.ultratiny_baselines.cifar100.matched.v1",
        "evidence_status": "single-seed matched ultra-tiny external baselines",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "parameter_counts": {
            variant: sum(p.numel() for p in _build(variant, config).parameters())
            for variant in VARIANTS
        },
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW with no decay on norm and bias",
            "learning_rate": 3.0e-3,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "warmup plus cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "augmentation": "RandomCrop(32,pad4)+HFlip+RandAugment(2,9)+RandomErasing",
            "validation": "fixed stratified 5k from CIFAR-100 train",
            "test_selection": False,
        },
        "data_sha256": {
            "cifar100_packed.pt": harness._digest(args.data_root / "cifar100_packed.pt")
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_cifar100_nano.py")),
            "packed_data": harness._digest(Path("scripts/run_polepyramid_a_tiny_cifar100.py")),
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
        if parameter.ndim < 2 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "lr": recipe["learning_rate"],
                "weight_decay": recipe["weight_decay"],
            },
            {"params": no_decay, "lr": recipe["learning_rate"], "weight_decay": 0.0},
        ]
    )


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    result_paths = [root / "results" / f"{variant}__seed{SEEDS[0]}.json" for variant in VARIANTS]
    available = [path for path in result_paths if path.exists()]
    if not available:
        return None
    rows = [json.loads(path.read_text()) for path in available]
    payload = {
        "schema": contract["schema"],
        "results": {
            row["variant"]: {
                "parameters": row["parameters"],
                "best_validation_accuracy": row["best_validation_accuracy"],
                "test": row["test"],
                "training_seconds": row["training_seconds"],
            }
            for row in rows
        },
    }
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.CifarNanoConfig = UltraTinyBaselineConfig
    harness.build_cifar_nano = _build
    harness._contract = _contract
    harness._build_optimizer = _build_optimizer
    harness._loaders = packed_data.build_loaders
    harness._summarize = _summarize
    harness.main()


if __name__ == "__main__":
    main()
