#!/usr/bin/env python3
"""Canonical model registry for the H200 ImageNet-1K baseline campaign."""

from __future__ import annotations

# pyright: reportAny=false, reportMissingImports=false
# ruff: noqa: TC003
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from torch import Tensor, nn

TIMM_VERSION = "1.0.26"

PUBLIC_MODEL_KEYS = (
    "parc_net_xs",
    "parc_net_s",
    "mobilevitv2_050",
    "mobilevitv2_075",
    "mobilevitv2_100",
    "sret_tiny",
    "moganet_xt",
    "uniconvnet_a",
    "convnextv2_atto",
    "efficientmod_xxs",
    "emov2_1m",
    "emov2_2m",
    "mobileone_s0",
    "mobileone_s1",
    "efficientformerv2_s0",
    "swiftformer_xs",
    "fastvit_t8",
    "tinynext_t",
    "tinynext_s",
    "tinynext_m",
    "tinyvim_s",
    "efficientvim_m1",
    "mambaout_femto",
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A stable public key and the implementation used by the campaign."""

    key: str
    display_name: str
    backend: Literal["timm", "external"]
    implementation_key: str
    create_kwargs: dict[str, Any] = field(default_factory=dict)
    single_head: bool = False
    precision: Literal["bfloat16", "float32"] = "bfloat16"


_TIMM_SPECS = {
    "mobilevitv2_050": ModelSpec("mobilevitv2_050", "MobileViTv2-0.50", "timm", "mobilevitv2_050"),
    "mobilevitv2_075": ModelSpec("mobilevitv2_075", "MobileViTv2-0.75", "timm", "mobilevitv2_075"),
    "mobilevitv2_100": ModelSpec("mobilevitv2_100", "MobileViTv2-1.00", "timm", "mobilevitv2_100"),
    "convnextv2_atto": ModelSpec("convnextv2_atto", "ConvNeXt V2 Atto", "timm", "convnextv2_atto"),
    # timm's MobileOne builders retain all training branches unless the model
    # is explicitly reparameterized after construction. Do not call that path.
    "mobileone_s0": ModelSpec("mobileone_s0", "MobileOne-S0", "timm", "mobileone_s0"),
    "mobileone_s1": ModelSpec("mobileone_s1", "MobileOne-S1", "timm", "mobileone_s1"),
    # timm 1.0.26 leaves head_dist=None when distillation=False but its stock
    # forward_head still calls it. _SingleHeadTimmClassifier intentionally
    # invokes only the primary classifier.
    "efficientformerv2_s0": ModelSpec(
        "efficientformerv2_s0",
        "EfficientFormerV2-S0",
        "timm",
        "efficientformerv2_s0",
        {"distillation": False},
        single_head=True,
    ),
    # SwiftFormer always constructs two classifier heads. The matched campaign
    # disables distillation and trains the primary head only.
    "swiftformer_xs": ModelSpec(
        "swiftformer_xs",
        "SwiftFormer-XS",
        "timm",
        "swiftformer_xs",
        {"distillation": False},
        single_head=True,
    ),
    # inference_mode=False is the unfused training topology.
    "fastvit_t8": ModelSpec(
        "fastvit_t8",
        "FastViT-T8",
        "timm",
        "fastvit_t8",
        {"inference_mode": False},
    ),
}

_DISPLAY_NAMES = {
    "parc_net_xs": "ParC-Net XS",
    "parc_net_s": "ParC-Net S",
    "sret_tiny": "SReT-Tiny",
    "moganet_xt": "MogaNet-XT",
    "uniconvnet_a": "UniConvNet-A",
    "efficientmod_xxs": "EfficientMod-XXS",
    "emov2_1m": "EMOv2-1M",
    "emov2_2m": "EMOv2-2M",
    "tinynext_t": "TinyNeXt-T",
    "tinynext_s": "TinyNeXt-S",
    "tinynext_m": "TinyNeXt-M",
    "tinyvim_s": "TinyViM-S",
    "efficientvim_m1": "EfficientViM-M1",
    "mambaout_femto": "MambaOut-Femto",
}

MODEL_SPECS = {
    key: (
        _TIMM_SPECS[key]
        if key in _TIMM_SPECS
        else ModelSpec(
            key,
            _DISPLAY_NAMES[key],
            "external",
            key,
            precision="float32" if key == "uniconvnet_a" else "bfloat16",
        )
    )
    for key in PUBLIC_MODEL_KEYS
}


class _SingleHeadTimmClassifier(nn.Module):
    """Use the primary timm classifier without a distillation branch."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.model.forward_features(inputs)  # type: ignore[attr-defined]
        global_pool = getattr(self.model, "global_pool", "avg")
        if global_pool == "avg":
            features = features.mean(dim=(2, 3))
        features = self.model.head_drop(features)  # type: ignore[attr-defined]
        return self.model.head(features)  # type: ignore[attr-defined, no-any-return]


def model_spec(key: str) -> ModelSpec:
    """Return a declared model spec, failing closed for misspelled tasks."""
    try:
        return MODEL_SPECS[key]
    except KeyError as error:
        supported = ", ".join(PUBLIC_MODEL_KEYS)
        message = f"unknown H200 baseline model {key!r}; supported: {supported}"
        raise ValueError(message) from error


def _build_timm_model(spec: ModelSpec, num_classes: int) -> nn.Module:
    try:
        import timm  # noqa: PLC0415
    except ModuleNotFoundError as error:
        message = f"timm=={TIMM_VERSION} is required for {spec.key}"
        raise RuntimeError(message) from error
    if timm.__version__ != TIMM_VERSION:
        message = f"{spec.key} requires timm=={TIMM_VERSION}, found {timm.__version__}"
        raise RuntimeError(message)
    model = timm.create_model(
        spec.implementation_key,
        pretrained=False,
        num_classes=num_classes,
        **spec.create_kwargs,
    )
    if spec.single_head:
        if hasattr(model, "head_dist"):
            model.head_dist = None
        model = _SingleHeadTimmClassifier(model)
    return model


def build_model(
    key: str,
    source_root: str | Path | None = None,
    num_classes: int = 1000,
) -> nn.Module:
    """Build one public baseline using timm or a pinned external source tree."""
    spec = model_spec(key)
    if num_classes <= 0:
        message = f"num_classes must be positive, got {num_classes}"
        raise ValueError(message)
    if spec.backend == "timm":
        return _build_timm_model(spec, num_classes)

    try:
        from h200_external_models import build_external_model  # noqa: PLC0415
    except ModuleNotFoundError as error:
        message = f"{key} requires scripts/h200_external_models.py and its pinned source checkout"
        raise RuntimeError(message) from error
    return build_external_model(key, source_root, num_classes)


__all__ = [
    "MODEL_SPECS",
    "PUBLIC_MODEL_KEYS",
    "TIMM_VERSION",
    "ModelSpec",
    "build_model",
    "model_spec",
]
