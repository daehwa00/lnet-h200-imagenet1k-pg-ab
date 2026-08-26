"""Pinned Mamba-1 baseline adapter for ALPHABET-LM."""

# pyright: reportExplicitAny=false
from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import Any, cast

from torch import Tensor, nn

MAMBA_GIT_COMMIT = "10b5d6358f27966f6a40e4bf0baa17a460688128"


@dataclass(frozen=True, slots=True)
class MambaLMConfig:
    vocab_size: int = 32_768
    model_width: int = 512
    layers: int = 11
    state_size: int = 16
    conv_width: int = 4
    expand: int = 2


def _mamba_lm_components() -> tuple[type[Any], type[nn.Module]]:
    try:
        config_type = importlib.import_module("mamba_ssm.models.config_mamba").MambaConfig
        model_type = importlib.import_module(
            "mamba_ssm.models.mixer_seq_simple"
        ).MambaLMHeadModel
    except (AttributeError, ModuleNotFoundError) as error:
        raise RuntimeError("pinned mamba-ssm is unavailable") from error
    return cast("type[Any]", config_type), cast("type[nn.Module]", model_type)


class MambaLM(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        config_type, model_type = _mamba_lm_components()
        official_config = config_type(
            d_model=config.model_width,
            d_intermediate=0,
            n_layer=config.layers,
            vocab_size=config.vocab_size,
            ssm_cfg={
                "layer": "Mamba1",
                "d_state": config.state_size,
                "d_conv": config.conv_width,
                "expand": config.expand,
            },
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            pad_vocab_size_multiple=8,
            tie_embeddings=True,
        )
        self.model = model_type(
            official_config,
            initializer_cfg={
                "initializer_range": 0.02,
                "rescale_prenorm_residual": True,
            },
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        output = self.model(input_ids)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise TypeError("official Mamba LM returned no logits")
        return logits


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
