# Copyright (c) 2026

"""Task heads shared by the dense-transfer training and evaluation code.

The segmentation decoder follows the structure of OpenMMLab's UPerHead:
Pyramid Pooling on the deepest feature, lateral/top-down FPN fusion, and a
final concatenation bottleneck.  This is a clean-room PyTorch implementation
so that the experiment does not depend on mmcv/mmseg.  The reference source is
the Apache-2.0 licensed mmsegmentation implementation:

https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/uper_head.py
https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/psp_head.py
https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE

The COCO detector is assembled from torchvision's public MaskRCNN,
FeaturePyramidNetwork, and detection utility modules.  No pretrained weights
are requested by this module.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

__all__ = [
    "ADE20KUPerNet",
    "COCOMaskRCNN",
    "FPNBackbone",
    "UPerNet",
    "build_task_model",
]


def _group_count(channels: int, maximum: int = 32) -> int:
    """Choose a GroupNorm32-compatible divisor for ``channels``.

    Standard decoder widths (256/512) use exactly 32 groups.  Small toy
    configurations used by CPU tests may have widths that are not divisible by
    32, so the largest valid divisor is selected instead.
    """

    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1  # pragma: no cover - every positive integer is divisible by one


class _ConvGNReLU(nn.Sequential):
    """A decoder convolution with the shared GroupNorm32 convention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


def _interpolate_like(source: Tensor, target: Tensor) -> Tensor:
    """Resize a feature to another feature's spatial size."""

    return functional.interpolate(
        source,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _as_feature_list(features: Mapping[str, Tensor] | Sequence[Tensor]) -> list[Tensor]:
    """Normalize a backbone output to the four ordered feature levels."""

    if isinstance(features, Mapping):
        # Backbones in this project use string keys ``0`` ... ``3``.  Sorting
        # numerically also handles a regular dict assembled by a toy backbone.
        try:
            values = [features[str(index)] for index in range(4)]
        except KeyError as exc:
            raise ValueError("backbone must provide feature keys '0', '1', '2', '3'") from exc
    else:
        values = list(features)
        if len(values) != 4:
            raise ValueError(f"backbone must provide four feature levels, got {len(values)}")
    if not all(isinstance(value, Tensor) and value.ndim == 4 for value in values):
        raise TypeError("backbone feature levels must be four-dimensional tensors")
    return values


class _PyramidPooling(nn.Module):
    """PSP module used by UPerNet on the deepest backbone feature."""

    def __init__(self, in_channels: int, channels: int, pool_scales: Sequence[int]) -> None:
        super().__init__()
        if not pool_scales:
            raise ValueError("pool_scales must contain at least one bin")
        if any(scale <= 0 for scale in pool_scales):
            raise ValueError(f"pool_scales must be positive, got {tuple(pool_scales)}")
        self.pool_scales = tuple(int(scale) for scale in pool_scales)
        self.stages = nn.ModuleList(
            [
                _ConvGNReLU(
                    in_channels,
                    channels,
                    kernel_size=1,
                    padding=0,
                )
                for _ in self.pool_scales
            ],
        )
        self.bottleneck = _ConvGNReLU(
            in_channels + len(self.pool_scales) * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, feature: Tensor) -> Tensor:
        outputs = [feature]
        target_size = feature.shape[-2:]
        for scale, stage in zip(self.pool_scales, self.stages, strict=True):
            pooled = functional.adaptive_avg_pool2d(feature, output_size=(scale, scale))
            pooled = stage(pooled)
            pooled = functional.interpolate(
                pooled,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            outputs.append(pooled)
        return self.bottleneck(torch.cat(outputs, dim=1))


class UPerNet(nn.Module):
    """UPerNet decoder for ADE20K-style semantic segmentation.

    ``backbone`` must return an ordered mapping with four levels at strides
    4/8/16/32 and expose ``feature_channels``.  Logits are deliberately kept
    at the finest backbone resolution (stride 4); the training engine owns the
    resize-to-label operation.  The auxiliary FCN reads the stride-16 level,
    matching the standard UPerNet auxiliary head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int = 150,
        decoder_channels: int = 512,
        aux_channels: int = 256,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        aux_loss_weight: float = 0.4,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if decoder_channels <= 0 or aux_channels <= 0:
            raise ValueError("decoder_channels and aux_channels must be positive")
        feature_channels = _feature_channels(backbone)
        self.backbone = backbone
        self.feature_channels = feature_channels
        self.num_classes = int(num_classes)
        self.decoder_channels = int(decoder_channels)
        self.aux_channels = int(aux_channels)
        self.aux_loss_weight = float(aux_loss_weight)

        self.psp = _PyramidPooling(
            feature_channels[-1],
            self.decoder_channels,
            pool_scales,
        )
        self.lateral_convs = nn.ModuleList(
            [
                _ConvGNReLU(
                    channels,
                    self.decoder_channels,
                    kernel_size=1,
                    padding=0,
                )
                for channels in feature_channels[:-1]
            ],
        )
        self.fpn_convs = nn.ModuleList(
            [
                _ConvGNReLU(
                    self.decoder_channels,
                    self.decoder_channels,
                    kernel_size=3,
                    padding=1,
                )
                for _ in feature_channels[:-1]
            ],
        )
        self.fpn_bottleneck = _ConvGNReLU(
            len(feature_channels) * self.decoder_channels,
            self.decoder_channels,
            kernel_size=3,
            padding=1,
        )
        # Classifier projections intentionally have no normalization: they are
        # the final linear maps from decoder features to class logits.
        self.classifier = nn.Conv2d(self.decoder_channels, self.num_classes, kernel_size=1)
        self.auxiliary = nn.Sequential(
            _ConvGNReLU(
                feature_channels[2],
                self.aux_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(self.aux_channels, self.num_classes, kernel_size=1),
        )

    def _decode(self, features: list[Tensor]) -> Tensor:
        laterals = [
            lateral(feature)
            for lateral, feature in zip(self.lateral_convs, features[:-1], strict=True)
        ]
        laterals.append(self.psp(features[-1]))

        # UPerHead's top-down path: every level receives the resized next
        # coarser level before its FPN convolution.
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + _interpolate_like(
                laterals[index],
                laterals[index - 1],
            )

        fpn_outputs = [
            fpn_conv(laterals[index])
            for index, fpn_conv in enumerate(self.fpn_convs)
        ]
        finest = fpn_outputs[0]
        fpn_outputs.append(laterals[-1])
        fpn_outputs = [
            output if output.shape[-2:] == finest.shape[-2:] else _interpolate_like(output, finest)
            for output in fpn_outputs
        ]
        return self.fpn_bottleneck(torch.cat(fpn_outputs, dim=1))

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        features = _as_feature_list(self.backbone(inputs))
        decoded = self._decode(features)
        return {
            "logits": self.classifier(decoded),
            "aux_logits": self.auxiliary(features[2]),
        }


# A descriptive alias is useful to callers that name task classes by dataset.
ADE20KUPerNet = UPerNet


class FPNBackbone(nn.Module):
    """Adapt a four-level dense-transfer backbone to torchvision detection."""

    def __init__(self, body: nn.Module, *, out_channels: int = 256) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        self.body = body
        self.feature_channels = _feature_channels(body)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(self.feature_channels),
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        # MaskRCNN uses this public attribute when constructing its RPN and
        # ROI heads.  The extra max-pool block adds the fifth ``pool`` level.
        self.out_channels = int(out_channels)

    def forward(self, inputs: Tensor) -> OrderedDict[str, Tensor]:
        features = self.body(inputs)
        if not isinstance(features, Mapping):
            raise TypeError("detection backbone must return a mapping of feature tensors")
        return self.fpn(features)


class COCOMaskRCNN(MaskRCNN):
    """Torchvision Mask R-CNN with a dense-transfer FPN backbone.

    The defaults match the standard COCO setup: 81 classes including
    background, five FPN anchor levels with 32..512 pixel anchors, and the
    torchvision ImageNet RGB normalization.  ``min_size``/``max_size`` remain
    configurable so CPU shape tests can use tiny images without changing the
    production defaults.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int = 81,
        fpn_channels: int = 256,
        min_size: int | Sequence[int] = 800,
        max_size: int = 1333,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        **kwargs: Any,
    ) -> None:
        fpn_backbone = FPNBackbone(backbone, out_channels=fpn_channels)
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2,
        )
        super().__init__(
            fpn_backbone,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
            image_mean=list(image_mean),
            image_std=list(image_std),
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=box_roi_pool,
            mask_roi_pool=mask_roi_pool,
            **kwargs,
        )


def _feature_channels(backbone: nn.Module) -> tuple[int, int, int, int]:
    """Read and validate the channel contract exposed by a dense backbone."""

    try:
        channels = tuple(int(value) for value in backbone.feature_channels)  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("backbone must expose feature_channels") from exc
    if len(channels) != 4 or any(value <= 0 for value in channels):
        raise ValueError(f"backbone.feature_channels must contain four positives, got {channels}")
    return channels  # type: ignore[return-value]


def build_task_model(task: str, backbone: nn.Module, **kwargs: Any) -> nn.Module:
    """Build the requested downstream task model around ``backbone``.

    Args:
        task: ``"ade20k"`` for UPerNet semantic segmentation or ``"coco"``
            for torchvision Mask R-CNN instance segmentation.
        backbone: Four-level dense-transfer backbone.
        **kwargs: Task-specific model options documented by the corresponding
            class.  No pretrained weights are loaded implicitly.
    """

    normalized_task = task.lower()
    if normalized_task == "ade20k":
        return UPerNet(backbone, **kwargs)
    if normalized_task == "coco":
        return COCOMaskRCNN(backbone, **kwargs)
    raise ValueError(f"unsupported task {task!r}; expected 'ade20k' or 'coco'")
