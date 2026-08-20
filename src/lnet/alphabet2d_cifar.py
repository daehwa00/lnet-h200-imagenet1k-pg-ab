"""CIFAR-100 Nano controls for separating pole mixing from the shared backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet2d import (
    Alphabet2D,
    Alphabet2DConfig,
    ProductPoleField2D,
    spatial_modal_moments,
)


@dataclass(frozen=True, slots=True)
class CifarNanoConfig:
    """Frozen sub-1M ALPHABET-2D Nano contract."""

    model_dim: int = 128
    depth: int = 6
    modes: int = 12
    patch_size: int = 4
    mlp_ratio: float = 2.0
    classes: int = 100

    def alphabet_config(self) -> Alphabet2DConfig:
        return Alphabet2DConfig(
            input_channels=3,
            output_dim=self.classes,
            image_size=32,
            patch_size=self.patch_size,
            model_dim=self.model_dim,
            modes=self.modes,
            depth=self.depth,
            mlp_ratio=self.mlp_ratio,
            windows="global_2x2",
            recurrence_backend="auto",
        )


@dataclass(frozen=True, slots=True)
class ImageNetNanoConfig:
    """Frozen sub-1M ALPHABET-2D Nano contract for 224px ImageNet-100."""

    model_dim: int = 128
    depth: int = 6
    modes: int = 12
    patch_size: int = 16
    image_size: int = 224
    mlp_ratio: float = 2.0
    classes: int = 100

    def alphabet_config(self) -> Alphabet2DConfig:
        return Alphabet2DConfig(
            input_channels=3,
            output_dim=self.classes,
            image_size=self.image_size,
            patch_size=self.patch_size,
            model_dim=self.model_dim,
            modes=self.modes,
            depth=self.depth,
            mlp_ratio=self.mlp_ratio,
            windows="global_2x2",
            recurrence_backend="auto",
        )


class _PoleFreeField(nn.Module):
    """Pointwise complex frame with the same modal width but no spatial poles."""

    def __init__(self, model_dim: int, modes: int) -> None:
        super().__init__()
        self.analysis = nn.Linear(model_dim, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        real, imag = self.analysis(inputs).chunk(2, dim=-1)
        return (
            real[:, None].expand(-1, 4, -1, -1, -1),
            imag[:, None].expand(-1, 4, -1, -1, -1),
        )

    def synthesize(self, real: Tensor, imag: Tensor) -> Tensor:
        mean_real = real.mean(dim=1)
        mean_imag = imag.mean(dim=1)
        frame_real, frame_imag = self.analysis.weight.chunk(2, dim=0)
        return torch.matmul(mean_real, frame_real) + torch.matmul(
            mean_imag,
            frame_imag,
        )


class _PoleFreeBlock(nn.Module):
    def __init__(self, config: CifarNanoConfig | ImageNetNanoConfig) -> None:
        super().__init__()
        hidden = round(config.model_dim * config.mlp_ratio)
        self.local = nn.Conv2d(
            config.model_dim,
            config.model_dim,
            3,
            padding=1,
            groups=config.model_dim,
        )
        self.norm = nn.RMSNorm(config.model_dim)
        self.field = _PoleFreeField(config.model_dim, config.modes)
        self.mixer_scale = nn.Parameter(torch.full((config.model_dim,), 1.0e-2))
        self.direct_scale = nn.Parameter(torch.zeros(config.model_dim))
        self.mlp_norm = nn.RMSNorm(config.model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.model_dim),
        )
        self.mlp_scale = nn.Parameter(torch.full((config.model_dim,), 1.0e-2))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        local = self.local(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        normalized = self.norm(functional.silu(local))
        real, imag = self.field(normalized)
        update = self.field.synthesize(real, imag)
        output = inputs + self.mixer_scale * (update + self.direct_scale * normalized)
        output = output + self.mlp_scale * self.mlp(self.mlp_norm(output))
        return output, real, imag


class PoleFreeNano(nn.Module):
    """Parameter-matched pointwise-modal control for four-scan Nano."""

    def __init__(self, config: CifarNanoConfig | ImageNetNanoConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            3,
            config.model_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.blocks = nn.ModuleList([_PoleFreeBlock(config) for _ in range(config.depth)])
        self.reader_local = nn.Conv2d(
            config.model_dim,
            config.model_dim,
            3,
            padding=1,
            groups=config.model_dim,
        )
        self.reader_norm = nn.RMSNorm(config.model_dim)
        self.reader = _PoleFreeField(config.model_dim, config.modes)
        coordinates = 1 + 2 * 4
        self.descriptor_dim = 2 * 5 * 4 * config.modes * coordinates
        self.classifier = nn.Linear(self.descriptor_dim, config.classes)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.patch_embed(inputs).permute(0, 2, 3, 1)
        direct_real: Tensor | None = None
        direct_imag: Tensor | None = None
        for block in self.blocks:
            features, direct_real, direct_imag = block(features)
        if direct_real is None or direct_imag is None:
            message = "pole-free Nano requires at least one block"
            raise RuntimeError(message)
        reader_input = self.reader_local(features.permute(0, 3, 1, 2)).permute(
            0,
            2,
            3,
            1,
        )
        reader_input = self.reader_norm(functional.silu(reader_input))
        reader_real, reader_imag = self.reader(reader_input)
        direct = spatial_modal_moments(
            direct_real,
            direct_imag,
            windows="global_2x2",
        )
        reader = spatial_modal_moments(
            reader_real,
            reader_imag,
            windows="global_2x2",
        )
        return self.classifier(torch.cat((direct, reader), dim=-1))


class HybridReaderNano(nn.Module):
    """Pole-free feature backbone with one terminal product-pole measurement."""

    def __init__(self, config: CifarNanoConfig | ImageNetNanoConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            3,
            config.model_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.blocks = nn.ModuleList([_PoleFreeBlock(config) for _ in range(config.depth)])
        self.reader_local = nn.Conv2d(
            config.model_dim,
            config.model_dim,
            3,
            padding=1,
            groups=config.model_dim,
        )
        self.reader_norm = nn.RMSNorm(config.model_dim)
        self.reader = ProductPoleField2D(
            config.model_dim,
            config.modes,
            recurrence_backend="auto",
        )
        coordinates = 1 + 2 * 4
        self.descriptor_dim = 2 * 5 * 4 * config.modes * coordinates
        self.classifier = nn.Linear(self.descriptor_dim, config.classes)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.patch_embed(inputs).permute(0, 2, 3, 1)
        direct_real: Tensor | None = None
        direct_imag: Tensor | None = None
        for block in self.blocks:
            features, direct_real, direct_imag = block(features)
        if direct_real is None or direct_imag is None:
            message = "hybrid reader Nano requires at least one block"
            raise RuntimeError(message)
        reader_input = self.reader_local(features.permute(0, 3, 1, 2)).permute(
            0,
            2,
            3,
            1,
        )
        reader_input = self.reader_norm(functional.silu(reader_input))
        reader_real, reader_imag = self.reader(reader_input)
        direct = spatial_modal_moments(
            direct_real,
            direct_imag,
            windows="global_2x2",
        )
        reader = spatial_modal_moments(
            reader_real,
            reader_imag,
            windows="global_2x2",
        )
        return self.classifier(torch.cat((direct, reader), dim=-1))


class SecondOrderNano(nn.Module):
    """One-bank second-order spatial spectrum model with an affine head."""

    def __init__(
        self,
        config: CifarNanoConfig | ImageNetNanoConfig,
        *,
        product_poles: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            3,
            config.model_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.stem_local = nn.Conv2d(
            config.model_dim,
            config.model_dim,
            3,
            padding=1,
            groups=config.model_dim,
        )
        self.stem_norm = nn.RMSNorm(config.model_dim)
        self.field: ProductPoleField2D | _PoleFreeField
        if product_poles:
            self.field = ProductPoleField2D(
                config.model_dim,
                config.modes,
                recurrence_backend="auto",
            )
        else:
            self.field = _PoleFreeField(config.model_dim, config.modes)
        coordinates = 1 + 2 * 4
        self.descriptor_dim = 4 * config.modes * coordinates
        self.classifier = nn.Linear(self.descriptor_dim, config.classes)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.patch_embed(inputs)
        features = self.stem_local(features).permute(0, 2, 3, 1)
        features = self.stem_norm(functional.silu(features))
        states_real, states_imag = self.field(features)
        descriptor = spatial_modal_moments(
            states_real,
            states_imag,
            windows="global",
        )
        return self.classifier(descriptor)


def build_cifar_nano(
    variant: str,
    config: CifarNanoConfig | None = None,
) -> nn.Module:
    """Build the headline four-scan Nano or its pole-free matched control."""
    active = config or CifarNanoConfig()
    if variant == "product_four":
        return Alphabet2D(active.alphabet_config())
    if variant == "pole_free":
        return PoleFreeNano(active)
    if variant == "hybrid_reader":
        return HybridReaderNano(active)
    if variant == "second_order_product":
        return SecondOrderNano(active, product_poles=True)
    if variant == "second_order_pointwise":
        return SecondOrderNano(active, product_poles=False)
    message = f"unknown CIFAR Nano variant: {variant}"
    raise ValueError(message)


def build_imagenet_nano(
    variant: str,
    config: ImageNetNanoConfig | None = None,
) -> nn.Module:
    """Build the 224px four-scan Nano or its matched pole-free control."""
    active = config or ImageNetNanoConfig()
    if variant == "product_four":
        return Alphabet2D(active.alphabet_config())
    if variant == "pole_free":
        return PoleFreeNano(active)
    if variant == "hybrid_reader":
        return HybridReaderNano(active)
    if variant == "second_order_product":
        return SecondOrderNano(active, product_poles=True)
    if variant == "second_order_pointwise":
        return SecondOrderNano(active, product_poles=False)
    message = f"unknown ImageNet Nano variant: {variant}"
    raise ValueError(message)


__all__ = [
    "CifarNanoConfig",
    "HybridReaderNano",
    "ImageNetNanoConfig",
    "PoleFreeNano",
    "SecondOrderNano",
    "build_cifar_nano",
    "build_imagenet_nano",
]
