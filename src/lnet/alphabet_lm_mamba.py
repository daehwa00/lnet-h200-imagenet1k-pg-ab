"""Pinned Mamba-1 baseline adapter for ALPHABET-LM."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace

from torch import Tensor, nn
from torch.nn import functional

MAMBA_GIT_COMMIT = "10b5d6358f27966f6a40e4bf0baa17a460688128"


@dataclass(frozen=True, slots=True)
class MambaLMConfig:
    vocab_size: int = 32_768
    model_width: int = 512
    layers: int = 11
    state_size: int = 16
    conv_width: int = 4
    expand: int = 2
    rms_epsilon: float = 1.0e-6


def _mamba_class() -> type[nn.Module]:
    try:
        return importlib.import_module("mamba_ssm").Mamba
    except (AttributeError, ModuleNotFoundError) as error:
        raise RuntimeError("pinned mamba-ssm is unavailable") from error


class MambaResidualBlock(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(config.model_width, eps=config.rms_epsilon)
        self.mixer = _mamba_class()(
            d_model=config.model_width,
            d_state=config.state_size,
            d_conv=config.conv_width,
            expand=config.expand,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.mixer(self.norm(hidden))


class MambaLM(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.model_width)
        self.blocks = nn.ModuleList(MambaResidualBlock(config) for _ in range(config.layers))
        self.final_norm = nn.RMSNorm(config.model_width, eps=config.rms_epsilon)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return functional.linear(self.final_norm(hidden), self.embedding.weight)


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_parameter_matched_mamba(
    target_parameters: int,
    base: MambaLMConfig,
    *,
    tolerance: float = 0.03,
) -> tuple[MambaLM, int, float]:
    best: tuple[MambaLM, int, float] | None = None
    for layers in range(8, 17):
        model = MambaLM(replace(base, layers=layers))
        parameters = trainable_parameters(model)
        error = abs(parameters - target_parameters) / target_parameters
        if best is None or error < best[2]:
            best = model, parameters, error
        else:
            del model
    if best is None or best[2] > tolerance:
        raise RuntimeError(f"no Mamba model satisfies ±{tolerance:.1%}")
    return best


__all__ = [
    "MAMBA_GIT_COMMIT", "MambaLM", "MambaLMConfig",
    "build_parameter_matched_mamba", "trainable_parameters",
]
