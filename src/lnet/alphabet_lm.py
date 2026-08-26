"""Causal fixed-complex-pole language model built from the PAC recurrence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_layers import PackedComplexLinear
from .pac_gated_post_fusion import GatedComplexPostFusion
from .pac_real2d_math import discrete_pole_real2d
from .pac_triton_recurrence_op import pac_triton_recurrence_opaque_op

ComplexField = tuple[Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class AlphabetLMConfig:
    vocab_size: int = 32_768
    modes: int = 256
    pole_modes: int = 320
    layers: int = 12
    reader_rank: int = 2
    reader_kernel: int = 3
    post_hidden: int = 384
    context_length: int = 2_048
    scan_fp32: bool = True
    rms_epsilon: float = 1.0e-6
    pole_initialization: Literal["legacy", "lifetime_palette"] = "legacy"
    minimum_half_life: float = 2.0
    maximum_half_life: float = 8_192.0
    decay_dominant_fraction: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.vocab_size, self.modes, self.pole_modes, self.layers, self.reader_rank,
            self.reader_kernel, self.post_hidden, self.context_length,
        )
        if any(value <= 0 for value in values) or self.reader_kernel % 2 == 0:
            raise ValueError("invalid ALPHABET-LM configuration")
        if (
            self.pole_initialization not in {"legacy", "lifetime_palette"}
            or self.minimum_half_life >= self.maximum_half_life
            or not 0.0 <= self.decay_dominant_fraction <= 1.0
        ):
            raise ValueError("invalid ALPHABET-LM pole initialization")

    @property
    def model_width(self) -> int:
        return 2 * self.modes


class CausalFactorizedComplexConv1dReader(nn.Module):
    def __init__(
        self,
        input_modes: int,
        output_modes: int,
        *,
        rank: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_modes = int(input_modes)
        self.output_modes = int(output_modes)
        self.rank = int(rank)
        self.kernel_size = int(kernel_size)
        if min(input_modes, output_modes, rank, kernel_size) <= 0 or kernel_size % 2 == 0:
            raise ValueError("invalid causal reader configuration")
        self.input_norm = ComplexRMSNorm(input_modes)
        self.point_weight_real = nn.Parameter(torch.empty(output_modes, rank, input_modes))
        self.point_weight_imag = nn.Parameter(torch.empty(output_modes, rank, input_modes))
        self.temporal_weight_real = nn.Parameter(torch.empty(output_modes, rank, kernel_size))
        self.temporal_weight_imag = nn.Parameter(torch.empty(output_modes, rank, kernel_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.point_weight_real)
        nn.init.xavier_uniform_(self.point_weight_imag)
        with torch.no_grad():
            self.point_weight_real.mul_(math.sqrt(0.5))
            self.point_weight_imag.mul_(math.sqrt(0.5))
            self.temporal_weight_real.zero_()
            self.temporal_weight_imag.zero_()
            self.temporal_weight_real[:, :, -1].fill_(1.0 / math.sqrt(self.rank))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.input_modes:
            raise ValueError("causal reader expects matching B,T,K coordinates")
        real, imag = self.input_norm(real, imag)
        point_real = torch.einsum("btk,prk->btpr", real, self.point_weight_real) - torch.einsum(
            "btk,prk->btpr", imag, self.point_weight_imag
        )
        point_imag = torch.einsum("btk,prk->btpr", real, self.point_weight_imag) + torch.einsum(
            "btk,prk->btpr", imag, self.point_weight_real
        )
        batch, steps, _, _ = point_real.shape
        channels = self.output_modes * self.rank

        def causal(source: Tensor, weight: Tensor) -> Tensor:
            packed = source.permute(0, 2, 3, 1).reshape(batch, channels, steps)
            padded = functional.pad(packed, (self.kernel_size - 1, 0))
            return functional.conv1d(
                padded, weight, groups=self.output_modes
            ).transpose(1, 2)

        return (
            causal(point_real, self.temporal_weight_real)
            - causal(point_imag, self.temporal_weight_imag),
            causal(point_real, self.temporal_weight_imag)
            + causal(point_imag, self.temporal_weight_real),
        )


class FixedComplexPoleMemory1D(nn.Module):
    minimum_damping = 1.0e-5

    def __init__(
        self,
        modes: int,
        *,
        context_length: int,
        scan_fp32: bool,
        initialization: Literal["legacy", "lifetime_palette"] = "legacy",
        minimum_half_life: float = 2.0,
        maximum_half_life: float = 8_192.0,
        decay_dominant_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        self.modes = int(modes)
        self.context_length = int(context_length)
        self.scan_fp32 = bool(scan_fp32)
        if initialization == "legacy":
            damping = torch.logspace(
                math.log10(1.0 / context_length), math.log10(0.5), modes
            )
            periods = torch.logspace(
                math.log10(4.0), math.log10(float(context_length)), modes
            ).flip(0)
            frequency = (2.0 * math.pi / periods).clamp_max(0.95 * math.pi)
        elif initialization == "lifetime_palette":
            octaves = round(math.log2(maximum_half_life / minimum_half_life))
            if not math.isclose(minimum_half_life * 2**octaves, maximum_half_life):
                raise ValueError("lifetime palette bounds must span complete octaves")
            anchors = minimum_half_life * 2.0 ** torch.arange(
                octaves + 1, dtype=torch.float32
            )
            anchor_indices = torch.arange(modes) * anchors.numel() // modes
            half_lives = anchors[anchor_indices]
            damping = math.log(2.0) / half_lives
            frequency = torch.zeros(modes)
            decay_modes = round(modes * decay_dominant_fraction)
            decay_indices = torch.linspace(0, modes - 1, decay_modes).round().long()
            oscillatory = torch.ones(modes, dtype=torch.bool)
            oscillatory[decay_indices] = False
            periods = torch.logspace(
                math.log10(4.0),
                math.log10(maximum_half_life),
                int(oscillatory.sum()),
            )
            frequency[oscillatory] = 2.0 * math.pi / periods
        else:
            raise ValueError("unknown fixed-pole initialization")
        self.raw_damping = nn.Parameter(torch.log(torch.expm1(damping - self.minimum_damping)))
        self.raw_frequency = nn.Parameter(torch.atanh((frequency / math.pi).clamp(-0.999, 0.999)))

    def damping(self) -> Tensor:
        return self.minimum_damping + functional.softplus(self.raw_damping)

    def frequency(self) -> Tensor:
        return math.pi * torch.tanh(self.raw_frequency)

    def coefficients(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return discrete_pole_real2d(self.damping(), self.frequency(), 1.0)

    def forward(self, drive_real: Tensor, drive_imag: Tensor) -> ComplexField:
        decay_real, decay_imag, gamma_real, gamma_imag = self.coefficients()
        scan_dtype = torch.float32 if self.scan_fp32 else drive_real.dtype
        active_real, active_imag = drive_real.to(scan_dtype), drive_imag.to(scan_dtype)
        values = tuple(
            value.to(device=drive_real.device, dtype=scan_dtype)
            for value in (decay_real, decay_imag, gamma_real, gamma_imag)
        )
        dr, di, gr, gi = values
        input_real = gr * active_real - gi * active_imag
        input_imag = gi * active_real + gr * active_imag
        shape = (1, 1, self.modes)
        state_real, state_imag = pac_triton_recurrence_opaque_op(
            dr.view(shape).expand_as(input_real),
            di.view(shape).expand_as(input_imag),
            input_real,
            input_imag,
        )
        return state_real.to(drive_real.dtype), state_imag.to(drive_imag.dtype)


class AlphabetLMBlock(nn.Module):
    def __init__(self, config: AlphabetLMConfig) -> None:
        super().__init__()
        self.reader = CausalFactorizedComplexConv1dReader(
            config.modes, config.pole_modes, rank=config.reader_rank,
            kernel_size=config.reader_kernel,
        )
        self.memory = FixedComplexPoleMemory1D(
            config.pole_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization=config.pole_initialization,
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
            decay_dominant_fraction=config.decay_dominant_fraction,
        )
        self.writer = PackedComplexLinear(config.pole_modes, config.modes)
        self.post_fusion = GatedComplexPostFusion(config.modes, config.post_hidden)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        drive = self.reader(real, imag)
        state = self.memory(*drive)
        memory = self.writer(*state)
        return self.post_fusion(real + memory[0], imag + memory[1])


class AlphabetLM(nn.Module):
    def __init__(self, config: AlphabetLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.model_width)
        self.analysis = nn.Linear(config.model_width, config.model_width, bias=False)
        self.blocks = nn.ModuleList(AlphabetLMBlock(config) for _ in range(config.layers))
        self.final_norm = nn.RMSNorm(config.model_width, eps=config.rms_epsilon)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.orthogonal_(self.analysis.weight)

    def hidden(self, input_ids: Tensor) -> Tensor:
        packed = self.analysis(self.embedding(input_ids))
        real, imag = packed.split(self.config.modes, dim=-1)
        for block in self.blocks:
            real, imag = block(real, imag)
        return self.final_norm(torch.cat((real, imag), dim=-1))

    def forward(self, input_ids: Tensor) -> Tensor:
        return functional.linear(self.hidden(input_ids), self.embedding.weight)


__all__ = [
    "AlphabetLM", "AlphabetLMBlock", "AlphabetLMConfig",
    "CausalFactorizedComplexConv1dReader", "FixedComplexPoleMemory1D",
]
