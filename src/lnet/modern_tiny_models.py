"""Official-topology TinyNeXt-T and RepViT-M0.9 classifiers.

TinyNeXt is adapted from yuffeenn/TinyNeXt commit
3eb30a847f8e5916b975f139d101a0da1f0d7e67 (MIT). RepViT is adapted from
THU-MIG/RepViT commit 298f42075eda5d2e6102559fad260c970769d34e (Apache-2.0).
The corresponding notices are recorded in ``THIRD_PARTY_NOTICES.md``.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import torch
from timm.layers import SqueezeExcite, trunc_normal_
from torch import Tensor, nn

TINY_NEXT_T_CONFIGURATION = (
    ("mv2", 32, 3, 2.0),
    ("mv2", 64, 3, 2.0),
    ("former", 96, 6, 2.0),
    ("se", 192, 2, 2.0),
)

REP_VIT_M09_CONFIGURATION = (
    (3, 2.0, 48, True, False, 1),
    (3, 2.0, 48, False, False, 1),
    (3, 2.0, 48, False, False, 1),
    (3, 2.0, 96, False, False, 2),
    (3, 2.0, 96, True, False, 1),
    (3, 2.0, 96, False, False, 1),
    (3, 2.0, 96, False, False, 1),
    (3, 2.0, 192, False, True, 2),
    (3, 2.0, 192, True, True, 1),
    (3, 2.0, 192, False, True, 1),
    (3, 2.0, 192, True, True, 1),
    (3, 2.0, 192, False, True, 1),
    (3, 2.0, 192, True, True, 1),
    (3, 2.0, 192, False, True, 1),
    (3, 2.0, 192, True, True, 1),
    (3, 2.0, 192, False, True, 1),
    (3, 2.0, 192, True, True, 1),
    (3, 2.0, 192, False, True, 1),
    (3, 2.0, 192, True, True, 1),
    (3, 2.0, 192, False, True, 1),
    (3, 2.0, 192, True, True, 1),
    (3, 2.0, 192, False, True, 1),
    (3, 2.0, 192, False, True, 1),
    (3, 2.0, 384, False, True, 2),
    (3, 2.0, 384, True, True, 1),
    (3, 2.0, 384, False, True, 1),
)


class _Add(nn.Module):
    def forward(self, identity: Tensor, update: Tensor) -> Tensor:
        return identity + update


class _Multiply(nn.Module):
    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        return first * second


class _MatrixMultiply(nn.Module):
    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        return torch.matmul(first, second)


class _ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class _TinyNeXtStem(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            _ConvBNReLU(input_channels, output_channels // 2, 3, 2),
            _ConvBNReLU(output_channels // 2, output_channels, 3, 2),
        )


class _MV2Block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        expanded_channels: int,
    ) -> None:
        super().__init__()
        self.shortcut = stride == 1 and input_channels == output_channels
        self.layers = nn.Sequential(
            _ConvBNReLU(input_channels, expanded_channels, kernel_size=1),
            _ConvBNReLU(
                expanded_channels,
                expanded_channels,
                stride=stride,
                groups=expanded_channels,
            ),
            nn.Conv2d(expanded_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.add = _Add() if self.shortcut else None

    def forward(self, inputs: Tensor) -> Tensor:
        update = self.layers(inputs)
        return self.add(inputs, update) if self.add is not None else update


class _TinyNeXtEmbed(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(_MV2Block(input_channels, output_channels, 2, output_channels))


class _LeanSingleHeadAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = channels**-0.5
        self.linear1 = nn.Linear(channels, channels, bias=False)
        self.linear2 = nn.Linear(channels, channels, bias=False)
        self.matmul1 = _MatrixMultiply()
        self.matmul2 = _MatrixMultiply()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        tokens = inputs.view(batch, channels, -1).transpose(-1, -2).contiguous()
        attention = self.matmul1(self.linear1(tokens), tokens.transpose(-2, -1))
        attention = self.softmax(attention * self.scale)
        output = self.matmul2(attention, self.linear2(tokens))
        return output.transpose(-1, -2).view(batch, channels, height, width).contiguous()


class _TinyNeXtMLP(nn.Sequential):
    def __init__(self, channels: int, ratio: float) -> None:
        hidden_channels = int(ratio * channels)
        super().__init__(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )


class _FormerBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.BatchNorm2d(channels),
            _LeanSingleHeadAttention(channels),
        )
        self.local = nn.Conv2d(
            channels,
            channels,
            3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(channels),
            _TinyNeXtMLP(channels, mlp_ratio),
        )
        self.add1 = _Add()
        self.add2 = _Add()
        self.add3 = _Add()

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.add1(inputs, self.attention(inputs))
        output = self.add2(output, self.local(output))
        return self.add3(output, self.mlp(output))


class _TinyNeXtSEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.Hardsigmoid(),
        )
        self.mul = _Multiply()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.mul(inputs, self.se(inputs))


class _TinyNeXtSEBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float, reduction: int = 4) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.BatchNorm2d(channels),
            _TinyNeXtSEModule(channels, reduction),
        )
        self.local = nn.Conv2d(
            channels,
            channels,
            3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(channels),
            _TinyNeXtMLP(channels, mlp_ratio),
        )
        self.add1 = _Add()
        self.add2 = _Add()
        self.add3 = _Add()

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.add1(inputs, self.se(inputs))
        output = self.add2(output, self.local(output))
        return self.add3(output, self.mlp(output))


def _tiny_next_block(name: str, channels: int, ratio: float) -> nn.Module:
    expanded_channels = int(ratio * channels)
    if name == "mv2":
        return _MV2Block(channels, channels, 1, expanded_channels)
    if name == "former":
        return _FormerBlock(channels, ratio)
    if name == "se":
        return _TinyNeXtSEBlock(channels, ratio)
    message = f"unsupported TinyNeXt block: {name}"
    raise ValueError(message)


class TinyNeXtT(nn.Module):
    """Smallest official TinyNeXt classification variant."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        input_channels = TINY_NEXT_T_CONFIGURATION[0][1]
        self.embeds = nn.ModuleList([_TinyNeXtStem(3, input_channels)])
        self.stages = nn.ModuleList()
        for index, (name, width, depth, ratio) in enumerate(TINY_NEXT_T_CONFIGURATION):
            if index > 0:
                self.embeds.append(_TinyNeXtEmbed(input_channels, width))
            self.stages.append(
                nn.Sequential(*(_tiny_next_block(name, width, ratio) for _ in range(depth)))
            )
            input_channels = width
        self.norm = nn.BatchNorm2d(input_channels)
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.class_head = nn.Linear(input_channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(
                module,
                (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d),
            ):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for embed, stage in zip(self.embeds, self.stages, strict=True):
            output = stage(embed(output))
        output = self.global_pool(self.norm(output))
        return self.class_head(output)


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class _Conv2dBN(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        batch_norm_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=False,
            ),
        )
        batch_norm = nn.BatchNorm2d(output_channels)
        self.add_module("bn", batch_norm)
        nn.init.constant_(batch_norm.weight, batch_norm_weight)
        nn.init.constant_(batch_norm.bias, 0.0)


class _Residual(nn.Module):
    def __init__(self, module: nn.Module, drop: float = 0.0) -> None:
        super().__init__()
        self.m = module
        self.drop = drop

    def forward(self, inputs: Tensor) -> Tensor:
        update = self.m(inputs)
        if self.training and self.drop > 0.0:
            keep = (
                torch.rand(inputs.shape[0], 1, 1, 1, device=inputs.device)
                .ge_(self.drop)
                .div_(1.0 - self.drop)
                .detach()
            )
            update = update * keep
        return inputs + update


class _RepVGGDepthwise(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = _Conv2dBN(channels, channels, 3, 1, 1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, groups=channels)
        self.dim = channels
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.bn(self.conv(inputs) + self.conv1(inputs) + inputs)


class _RepViTBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        *,
        use_se: bool,
    ) -> None:
        super().__init__()
        if stride not in {1, 2} or hidden_channels != 2 * input_channels:
            message = "invalid official RepViT block dimensions"
            raise ValueError(message)
        identity = stride == 1 and input_channels == output_channels
        if stride == 2:
            self.token_mixer = nn.Sequential(
                _Conv2dBN(
                    input_channels,
                    input_channels,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=input_channels,
                ),
                SqueezeExcite(input_channels, 0.25) if use_se else nn.Identity(),
                _Conv2dBN(input_channels, output_channels),
            )
            mixer_input = output_channels
        else:
            if not identity:
                message = "stride-one RepViT blocks require an identity path"
                raise ValueError(message)
            self.token_mixer = nn.Sequential(
                _RepVGGDepthwise(input_channels),
                SqueezeExcite(input_channels, 0.25) if use_se else nn.Identity(),
            )
            mixer_input = input_channels
        self.channel_mixer = _Residual(
            nn.Sequential(
                _Conv2dBN(mixer_input, 2 * mixer_input),
                nn.GELU(),
                _Conv2dBN(
                    2 * mixer_input,
                    output_channels,
                    batch_norm_weight=0.0,
                ),
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.channel_mixer(self.token_mixer(inputs))


class _BNLinear(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.add_module("bn", nn.BatchNorm1d(input_dim))
        linear = nn.Linear(input_dim, output_dim)
        self.add_module("l", linear)
        trunc_normal_(linear.weight, std=0.02)
        nn.init.constant_(linear.bias, 0.0)


class _RepViTClassifier(nn.Module):
    def __init__(self, dimension: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = _BNLinear(dimension, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(inputs)


class RepViTM09(nn.Module):
    """Official RepViT-M0.9 training-time topology without distillation."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        input_channels = REP_VIT_M09_CONFIGURATION[0][2]
        patch_embed = nn.Sequential(
            _Conv2dBN(3, input_channels // 2, 3, 2, 1),
            nn.GELU(),
            _Conv2dBN(input_channels // 2, input_channels, 3, 2, 1),
        )
        layers: list[nn.Module] = [patch_embed]
        output_channels = input_channels
        for kernel, ratio, channels, use_se, _use_hs, stride in REP_VIT_M09_CONFIGURATION:
            output_channels = _make_divisible(float(channels))
            hidden_channels = _make_divisible(input_channels * ratio)
            layers.append(
                _RepViTBlock(
                    input_channels,
                    hidden_channels,
                    output_channels,
                    kernel,
                    stride,
                    use_se=use_se,
                )
            )
            input_channels = output_channels
        self.features = nn.ModuleList(layers)
        self.classifier = _RepViTClassifier(output_channels, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for feature in self.features:
            output = feature(output)
        output = torch.nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
        return self.classifier(output)


__all__ = [
    "REP_VIT_M09_CONFIGURATION",
    "TINY_NEXT_T_CONFIGURATION",
    "RepViTM09",
    "TinyNeXtT",
]
