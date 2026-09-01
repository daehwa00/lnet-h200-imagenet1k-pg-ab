"""Pinned Mamba-1 baseline adapter for ALPHABET-LM."""

# pyright: reportExplicitAny=false
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .alphabet_lm import FixedComplexPoleMemory1D

MAMBA_GIT_COMMIT = "10b5d6358f27966f6a40e4bf0baa17a460688128"


@dataclass(frozen=True, slots=True)
class MambaLMConfig:
    vocab_size: int = 32_768
    model_width: int = 512
    layers: int = 11
    state_size: int = 16
    conv_width: int = 4
    expand: int = 2
    architecture: Literal["Mamba1", "Mamba2"] = "Mamba1"
    head_dim: int = 64
    groups: int = 1
    mamba2_use_mem_eff_path: bool = False


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
        ssm_cfg: dict[str, object] = {
            "layer": config.architecture,
            "d_state": config.state_size,
            "d_conv": config.conv_width,
            "expand": config.expand,
        }
        if config.architecture == "Mamba2":
            ssm_cfg.update(
                {
                    "headdim": config.head_dim,
                    "ngroups": config.groups,
                    "use_mem_eff_path": config.mamba2_use_mem_eff_path,
                }
            )
        official_config = config_type(
            d_model=config.model_width,
            d_intermediate=0,
            n_layer=config.layers,
            vocab_size=config.vocab_size,
            ssm_cfg=ssm_cfg,
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


@dataclass(frozen=True, slots=True)
class LaplaceSSDMamba2Config:
    vocab_size: int = 32_768
    model_width: int = 512
    layers: int = 18
    pole_modes: int = 8
    conv_width: int = 4
    expand: int = 2
    head_dim: int = 64
    groups: int = 1
    context_length: int = 2_048
    minimum_half_life: float = 16.0
    maximum_half_life: float = 4_096.0
    scan_fp32: bool = True
    parallel_static_scan: bool = True
    activation_checkpoint: bool = True

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.model_width,
            self.layers,
            self.pole_modes,
            self.conv_width,
            self.expand,
            self.head_dim,
            self.groups,
            self.context_length,
        ) <= 0:
            raise ValueError("invalid Laplace-SSD Mamba-2 configuration")
        inner = self.model_width * self.expand
        if inner % self.head_dim or (inner // self.head_dim) % self.groups:
            raise ValueError("Laplace-SSD heads and groups do not divide the inner width")


class LaplaceSSDMamba2Mixer(nn.Module):
    """Official Mamba-2 scaffold with fixed complex Laplace SSD transport."""

    def __init__(self, config: LaplaceSSDMamba2Config, *, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.d_model = config.model_width
        self.d_inner = config.model_width * config.expand
        self.headdim = config.head_dim
        self.nheads = self.d_inner // self.headdim
        self.ngroups = config.groups
        self.poles = config.pole_modes
        projection_width = 2 * self.d_inner + 3 * self.ngroups * self.poles
        self.in_proj = nn.Linear(self.d_model, projection_width, bias=False)
        conv_width = self.d_inner + 3 * self.ngroups * self.poles
        self.conv1d = nn.Conv1d(
            conv_width,
            conv_width,
            kernel_size=config.conv_width,
            groups=conv_width,
            padding=config.conv_width - 1,
            bias=True,
        )
        self.memory = FixedComplexPoleMemory1D(
            self.nheads * self.poles,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
            banks=self.nheads,
            parallel_static_scan=config.parallel_static_scan,
        )
        self.D = nn.Parameter(torch.ones(self.nheads))
        try:
            norm_type = importlib.import_module(
                "mamba_ssm.ops.triton.layernorm_gated"
            ).RMSNorm
        except (AttributeError, ModuleNotFoundError) as error:
            raise RuntimeError("official Mamba-2 gated RMSNorm is unavailable") from error
        self.norm = norm_type(
            self.d_inner,
            eps=1.0e-5,
            norm_before_gate=False,
            group_size=self.d_inner // self.ngroups,
        )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5.0))
        with torch.no_grad():
            self.out_proj.weight.div_(math.sqrt(config.layers))

    def _group_routes(self, route: Tensor) -> Tensor:
        batch, steps, _groups, _poles = route.shape
        heads_per_group = self.nheads // self.ngroups
        return (
            route.unsqueeze(3)
            .expand(batch, steps, self.ngroups, heads_per_group, self.poles)
            .reshape(batch, steps, self.nheads, self.poles)
        )

    def _forward_impl(
        self,
        hidden: Tensor,
    ) -> Tensor:
        batch, steps, _width = hidden.shape
        projected = self.in_proj(hidden)
        z, active = projected.split((self.d_inner, projected.shape[-1] - self.d_inner), dim=-1)
        active = functional.silu(
            self.conv1d(active.transpose(1, 2)).transpose(1, 2)[:, :steps]
        )
        x, write, read_real, read_imag = active.split(
            (
                self.d_inner,
                self.ngroups * self.poles,
                self.ngroups * self.poles,
                self.ngroups * self.poles,
            ),
            dim=-1,
        )
        x = x.reshape(batch, steps, self.nheads, self.headdim)
        route_shape = (batch, steps, self.ngroups, self.poles)
        write = self._group_routes(write.reshape(route_shape))
        read_real = self._group_routes(read_real.reshape(route_shape))
        read_imag = self._group_routes(read_imag.reshape(route_shape))
        drive_real = (write.unsqueeze(-1) * x.unsqueeze(-2)).reshape(
            batch, steps, self.nheads * self.poles, self.headdim
        )
        drive_imag = torch.zeros_like(drive_real)
        state_real, state_imag = self.memory(drive_real, drive_imag)
        state_shape = (batch, steps, self.nheads, self.poles, self.headdim)
        state_real = state_real.reshape(state_shape)
        state_imag = state_imag.reshape(state_shape)
        y = (
            read_real.unsqueeze(-1) * state_real
            + read_imag.unsqueeze(-1) * state_imag
        ).sum(dim=-2)
        y = y + self.D.to(y.dtype).view(1, 1, self.nheads, 1) * x
        y = self.norm(y.flatten(-2), z)
        return self.out_proj(y)

    def forward(
        self,
        hidden: Tensor,
        inference_params: object | None = None,
        **_kwargs: object,
    ) -> Tensor:
        if inference_params is not None:
            raise NotImplementedError("Laplace-SSD inference cache is not implemented")
        if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
            return cast(
                "Tensor",
                activation_checkpoint(self._forward_impl, hidden, use_reentrant=False),
            )
        return self._forward_impl(hidden)


class LaplaceSSDMamba2LM(nn.Module):
    """Official Mamba-2 LM backbone with Laplace-SSD mixers."""

    def __init__(self, config: LaplaceSSDMamba2Config) -> None:
        super().__init__()
        self.config = config
        official = MambaLM(
            MambaLMConfig(
                vocab_size=config.vocab_size,
                model_width=config.model_width,
                layers=config.layers,
                state_size=128,
                conv_width=config.conv_width,
                expand=config.expand,
                architecture="Mamba2",
                head_dim=config.head_dim,
                groups=config.groups,
                mamba2_use_mem_eff_path=False,
            )
        )
        backbone = cast("Any", official.model).backbone
        for index, block in enumerate(backbone.layers):
            block.mixer = LaplaceSSDMamba2Mixer(config, layer_idx=index)
        self.model = official.model

    def forward(self, input_ids: Tensor) -> Tensor:
        output = self.model(input_ids)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise TypeError("Laplace-SSD Mamba-2 returned no logits")
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
    for layers in range(8, 25):
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
    "MAMBA_GIT_COMMIT",
    "LaplaceSSDMamba2Config",
    "LaplaceSSDMamba2LM",
    "LaplaceSSDMamba2Mixer",
    "MambaLM",
    "MambaLMConfig",
    "build_parameter_matched_mamba",
    "trainable_parameters",
]
