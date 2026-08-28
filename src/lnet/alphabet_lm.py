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
from .pac_real2d_math import discrete_pole_real2d, pole_gamma_from_control_real2d
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
    reader_type: Literal["r2k3", "dense_k3"] = "r2k3"
    pole_routing: Literal["static", "dynamic_write", "dynamic_write_read"] = "static"
    pole_dynamics: Literal["fixed", "delta_select"] = "fixed"
    write_map: Literal["static", "dynamic_low_rank"] = "static"
    memory_layout: Literal["flat", "tensor_product", "local_only", "local_sidecar"] = "flat"
    memory_readout: Literal["fixed", "query_low_rank"] = "fixed"
    router_hidden: int = 32
    delta_select_rank: int = 16
    delta_select_initial_scale: float = 0.1
    delta_select_control_bound: float = 1.0
    dynamic_write_rank: int = 4
    dynamic_write_initial_scale: float = 0.06
    query_read_rank: int = 32
    query_read_initial_scale: float = 0.05
    minimum_half_life: float = 2.0
    maximum_half_life: float = 8_192.0
    decay_dominant_fraction: float = 0.5
    memory_banks: int = 1
    bank_pole_modes: int = 128
    tensor_temporal_modes: int = 8
    tensor_initial_read_gain: float = 0.6
    sidecar_initial_scale: float = 0.01
    sidecar_normalize_memory: bool = False
    sidecar_channelwise_scale: bool = True
    sidecar_use_recurrence: bool = True
    chunk_memory: bool = False
    chunk_size: int = 32
    chunk_summary_width: int = 128
    chunk_pole_modes: int = 128
    chunk_upper_blocks: int = 4
    chunk_beta_initial: float = 0.01
    chunk_minimum_half_life: float = 1.0
    chunk_maximum_half_life: float = 128.0
    semantic_edge_memory: bool = False
    semantic_edge_stride: int = 16
    semantic_edge_pole_modes: int = 128
    semantic_edge_upper_blocks: int = 4
    semantic_edge_beta_initial: float = 0.01
    semantic_edge_use_recurrence: bool = True
    semantic_edge_minimum_half_life: float = 1.0
    semantic_edge_maximum_half_life: float = 256.0
    cnn_pole_memory: bool = False
    cnn_pole_interval: int = 2
    cnn_pole_modes: int = 128
    cnn_pole_evidence_width: int = 512
    cnn_pole_kernel_size: int = 4
    cnn_pole_beta_initial: float = 0.01
    cnn_pole_use_recurrence: bool = True
    cnn_pole_minimum_half_life: float = 8.0
    cnn_pole_maximum_half_life: float = 4_096.0
    slow_cnn_pole_memory: bool = False
    slow_cnn_pole_stride: int = 16
    slow_cnn_pole_modes: int = 128
    slow_cnn_pole_evidence_width: int = 512
    slow_cnn_pole_kernel_size: int = 4
    slow_cnn_pole_upper_blocks: int = 4
    slow_cnn_pole_beta_initial: float = 0.01
    slow_cnn_pole_use_recurrence: bool = True
    slow_cnn_pole_minimum_half_life: float = 1.0
    slow_cnn_pole_maximum_half_life: float = 256.0
    slow_cnn_pole_query: Literal["none", "anchor", "token"] = "none"
    slow_cnn_pole_query_rho: float = 0.5
    slow_cnn_pole_key: bool = False
    slow_cnn_pole_key_rho: float = 0.5
    tensor_half_lives: tuple[float, ...] = (
        4.0,
        16.0,
        64.0,
        256.0,
        1_024.0,
        4_096.0,
        16_384.0,
        65_536.0,
    )

    def __post_init__(self) -> None:
        values = (
            self.vocab_size,
            self.modes,
            self.pole_modes,
            self.layers,
            self.reader_rank,
            self.reader_kernel,
            self.post_hidden,
            self.context_length,
            self.memory_banks,
            self.bank_pole_modes,
            self.router_hidden,
            self.delta_select_rank,
            self.dynamic_write_rank,
            self.query_read_rank,
            self.tensor_temporal_modes,
            self.chunk_size,
            self.chunk_summary_width,
            self.chunk_pole_modes,
            self.chunk_upper_blocks,
            self.semantic_edge_stride,
            self.semantic_edge_pole_modes,
            self.semantic_edge_upper_blocks,
            self.cnn_pole_interval,
            self.cnn_pole_modes,
            self.cnn_pole_evidence_width,
            self.cnn_pole_kernel_size,
            self.slow_cnn_pole_stride,
            self.slow_cnn_pole_modes,
            self.slow_cnn_pole_evidence_width,
            self.slow_cnn_pole_kernel_size,
            self.slow_cnn_pole_upper_blocks,
        )
        if any(value <= 0 for value in values) or self.reader_kernel % 2 == 0:
            raise ValueError("invalid ALPHABET-LM configuration")
        if (
            self.pole_initialization not in {"legacy", "lifetime_palette"}
            or self.minimum_half_life >= self.maximum_half_life
            or not 0.0 <= self.decay_dominant_fraction <= 1.0
        ):
            raise ValueError("invalid ALPHABET-LM pole initialization")
        if self.modes % self.memory_banks:
            raise ValueError("ALPHABET-LM modes must divide evenly across memory banks")
        if self.reader_type == "dense_k3" and self.memory_banks != 1:
            raise ValueError("dense K3 reader requires the single-bank configuration")
        if self.pole_routing != "static" and self.memory_banks != 1:
            raise ValueError("dynamic pole routing currently requires a single memory bank")
        if (
            self.pole_routing not in {"static", "dynamic_write", "dynamic_write_read"}
            or self.pole_dynamics not in {"fixed", "delta_select"}
            or self.memory_layout not in {"flat", "tensor_product", "local_only", "local_sidecar"}
            or self.write_map not in {"static", "dynamic_low_rank"}
            or self.memory_readout not in {"fixed", "query_low_rank"}
        ):
            raise ValueError("invalid ALPHABET-LM dynamic configuration")
        if (
            not 0.0 < self.delta_select_initial_scale < 1.0
            or self.delta_select_control_bound <= 0.0
            or not 0.0 < self.query_read_initial_scale < 1.0
            or not 0.0 < self.dynamic_write_initial_scale < 1.0
            or not 0.0 < self.sidecar_initial_scale < 1.0
        ):
            raise ValueError("invalid ALPHABET-LM dynamic initialization")
        self._validate_memory_layout()
        self._validate_chunk_memory()
        self._validate_semantic_edge_memory()
        self._validate_cnn_pole_memory()
        self._validate_slow_cnn_pole_memory()
        if self.write_map == "dynamic_low_rank" and (
            self.reader_type != "dense_k3"
            or self.memory_layout != "flat"
            or self.memory_banks != 1
            or self.pole_routing != "static"
        ):
            raise ValueError("invalid low-rank dynamic write configuration")

    def _validate_chunk_memory(self) -> None:
        if not self.chunk_memory:
            return
        if (
            self.memory_layout != "local_only"
            or self.reader_type != "dense_k3"
            or self.chunk_summary_width % 2
            or self.chunk_upper_blocks > self.layers
            or not 0.0 < self.chunk_beta_initial < 1.0
            or self.chunk_minimum_half_life >= self.chunk_maximum_half_life
        ):
            raise ValueError("invalid chunk semantic memory configuration")

    def _validate_semantic_edge_memory(self) -> None:
        if not self.semantic_edge_memory:
            return
        if (
            self.chunk_memory
            or self.memory_layout != "local_only"
            or self.reader_type != "dense_k3"
            or self.semantic_edge_upper_blocks > self.layers
            or not 0.0 < self.semantic_edge_beta_initial < 1.0
            or self.semantic_edge_minimum_half_life >= self.semantic_edge_maximum_half_life
        ):
            raise ValueError("invalid semantic edge memory configuration")

    def _validate_cnn_pole_memory(self) -> None:
        if not self.cnn_pole_memory:
            return
        if (
            self.chunk_memory
            or self.semantic_edge_memory
            or self.memory_layout != "local_only"
            or self.reader_type != "dense_k3"
            or self.layers % self.cnn_pole_interval
            or not 0.0 < self.cnn_pole_beta_initial < 1.0
            or self.cnn_pole_minimum_half_life >= self.cnn_pole_maximum_half_life
        ):
            raise ValueError("invalid CNN pole memory configuration")

    def _validate_slow_cnn_pole_memory(self) -> None:
        if not self.slow_cnn_pole_memory:
            if self.slow_cnn_pole_query != "none" or self.slow_cnn_pole_key:
                raise ValueError("slow pole addressing requires a slow memory bank")
            return
        if (
            self.chunk_memory
            or self.semantic_edge_memory
            or self.memory_layout != "local_only"
            or self.reader_type != "dense_k3"
            or self.slow_cnn_pole_upper_blocks > self.layers
            or not 0.0 < self.slow_cnn_pole_beta_initial < 1.0
            or self.slow_cnn_pole_minimum_half_life
            >= self.slow_cnn_pole_maximum_half_life
            or self.slow_cnn_pole_query not in {"none", "anchor", "token"}
            or not 0.0 < self.slow_cnn_pole_query_rho < 1.0
            or not 0.0 < self.slow_cnn_pole_key_rho < 1.0
        ):
            raise ValueError("invalid slow CNN pole memory configuration")

    def _validate_memory_layout(self) -> None:
        if self.memory_layout == "tensor_product" and (
            self.reader_type != "dense_k3"
            or self.memory_banks != 1
            or self.pole_routing != "static"
            or self.pole_dynamics != "fixed"
            or self.memory_readout != "fixed"
            or len(self.tensor_half_lives) != self.tensor_temporal_modes
            or any(value <= 0.0 for value in self.tensor_half_lives)
            or self.tensor_initial_read_gain <= 0.0
            or any(
                left >= right
                for left, right in zip(
                    self.tensor_half_lives,
                    self.tensor_half_lives[1:],
                    strict=False,
                )
            )
        ):
            raise ValueError("invalid tensor-product memory configuration")
        if self.memory_layout in {"local_only", "local_sidecar"} and (
            self.reader_type != "dense_k3"
            or self.memory_banks != 1
            or self.pole_routing != "static"
            or self.pole_dynamics != "fixed"
            or self.write_map != "static"
            or self.memory_readout != "fixed"
        ):
            raise ValueError("invalid local-only memory configuration")

    @property
    def model_width(self) -> int:
        return 2 * self.modes

    @property
    def total_pole_modes(self) -> int:
        if self.memory_banks == 1:
            return self.pole_modes
        return self.memory_banks * self.bank_pole_modes

    @property
    def recurrent_state_modes(self) -> int:
        if self.memory_layout == "tensor_product":
            return self.modes * self.tensor_temporal_modes
        return self.total_pole_modes


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
            return functional.conv1d(padded, weight, groups=self.output_modes).transpose(1, 2)

        return (
            causal(point_real, self.temporal_weight_real)
            - causal(point_imag, self.temporal_weight_imag),
            causal(point_real, self.temporal_weight_imag)
            + causal(point_imag, self.temporal_weight_real),
        )


class GroupedCausalFactorizedComplexConv1dReader(nn.Module):
    def __init__(
        self,
        input_modes: int,
        poles_per_bank: int,
        *,
        banks: int,
        rank: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if input_modes % banks or min(input_modes, poles_per_bank, banks, rank) <= 0:
            raise ValueError("invalid grouped causal reader dimensions")
        self.input_modes = input_modes
        self.banks = banks
        self.input_modes_per_bank = input_modes // banks
        self.poles_per_bank = poles_per_bank
        self.rank = rank
        self.kernel_size = kernel_size
        self.norm_weight = nn.Parameter(torch.ones(banks, self.input_modes_per_bank))
        point_shape = (banks, poles_per_bank, rank, self.input_modes_per_bank)
        temporal_shape = (banks, poles_per_bank, rank, kernel_size)
        self.point_weight_real = nn.Parameter(torch.empty(point_shape))
        self.point_weight_imag = nn.Parameter(torch.empty(point_shape))
        self.temporal_weight_real = nn.Parameter(torch.empty(temporal_shape))
        self.temporal_weight_imag = nn.Parameter(torch.empty(temporal_shape))
        for bank in range(banks):
            nn.init.xavier_uniform_(self.point_weight_real[bank])
            nn.init.xavier_uniform_(self.point_weight_imag[bank])
        with torch.no_grad():
            self.point_weight_real.mul_(math.sqrt(0.5))
            self.point_weight_imag.mul_(math.sqrt(0.5))
            self.temporal_weight_real.zero_()
            self.temporal_weight_imag.zero_()
            self.temporal_weight_real[..., -1].fill_(1.0 / math.sqrt(rank))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.input_modes:
            raise ValueError("grouped reader expects matching B,T,K coordinates")
        batch, steps, _modes = real.shape
        shape = (batch, steps, self.banks, self.input_modes_per_bank)
        grouped_real = real.reshape(shape)
        grouped_imag = imag.reshape(shape)
        energy = (grouped_real.float().square() + grouped_imag.float().square()).mean(
            dim=-1, keepdim=True
        )
        scale = torch.rsqrt(energy + 1.0e-6).to(real.dtype)
        weight = self.norm_weight.to(real.dtype).view(1, 1, self.banks, -1)
        grouped_real = grouped_real * scale * weight
        grouped_imag = grouped_imag * scale * weight
        point_real = torch.einsum(
            "bthk,hprk->bthpr", grouped_real, self.point_weight_real
        ) - torch.einsum("bthk,hprk->bthpr", grouped_imag, self.point_weight_imag)
        point_imag = torch.einsum(
            "bthk,hprk->bthpr", grouped_real, self.point_weight_imag
        ) + torch.einsum("bthk,hprk->bthpr", grouped_imag, self.point_weight_real)
        channels = self.banks * self.poles_per_bank * self.rank

        def causal(source: Tensor, weight_source: Tensor) -> Tensor:
            packed = source.permute(0, 2, 3, 4, 1).reshape(batch, channels, steps)
            padded = functional.pad(packed, (self.kernel_size - 1, 0))
            weights = weight_source.reshape(
                self.banks * self.poles_per_bank, self.rank, self.kernel_size
            )
            output = functional.conv1d(padded, weights, groups=self.banks * self.poles_per_bank)
            return output.transpose(1, 2).reshape(batch, steps, -1)

        return (
            causal(point_real, self.temporal_weight_real)
            - causal(point_imag, self.temporal_weight_imag),
            causal(point_real, self.temporal_weight_imag)
            + causal(point_imag, self.temporal_weight_real),
        )


class DenseComplexConv1dReader(nn.Module):
    def __init__(
        self,
        input_modes: int,
        output_modes: int,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if min(input_modes, output_modes, kernel_size) <= 0 or kernel_size % 2 == 0:
            raise ValueError("invalid dense complex reader configuration")
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.kernel_size = kernel_size
        self.input_norm = ComplexRMSNorm(input_modes)
        shape = (output_modes, input_modes, kernel_size)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))

    @classmethod
    def from_factorized(
        cls, source: CausalFactorizedComplexConv1dReader
    ) -> DenseComplexConv1dReader:
        dense = cls(source.input_modes, source.output_modes, kernel_size=source.kernel_size)
        with torch.no_grad():
            dense.input_norm.weight.copy_(source.input_norm.weight)
            dense.weight_real.copy_(
                torch.einsum("prl,prk->pkl", source.temporal_weight_real, source.point_weight_real)
                - torch.einsum(
                    "prl,prk->pkl", source.temporal_weight_imag, source.point_weight_imag
                )
            )
            dense.weight_imag.copy_(
                torch.einsum("prl,prk->pkl", source.temporal_weight_real, source.point_weight_imag)
                + torch.einsum(
                    "prl,prk->pkl", source.temporal_weight_imag, source.point_weight_real
                )
            )
        return dense

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.input_modes:
            raise ValueError("dense complex reader expects matching B,T,K coordinates")
        real, imag = self.input_norm(real, imag)
        packed_input = torch.cat((real, imag), dim=-1).transpose(1, 2)
        packed_weight = torch.cat(
            (
                torch.cat((self.weight_real, -self.weight_imag), dim=1),
                torch.cat((self.weight_imag, self.weight_real), dim=1),
            ),
            dim=0,
        )
        padded = functional.pad(packed_input, (self.kernel_size - 1, 0))
        packed_output = functional.conv1d(padded, packed_weight).transpose(1, 2)
        output_real, output_imag = packed_output.split(self.output_modes, dim=-1)
        return output_real, output_imag


class GroupedPackedComplexLinear(nn.Module):
    def __init__(self, poles_per_bank: int, output_modes: int, *, banks: int) -> None:
        super().__init__()
        if output_modes % banks:
            raise ValueError("grouped writer output modes must divide across banks")
        self.banks = banks
        self.poles_per_bank = poles_per_bank
        self.output_modes = output_modes
        self.output_modes_per_bank = output_modes // banks
        shape = (banks, self.output_modes_per_bank, poles_per_bank)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        for bank in range(banks):
            nn.init.xavier_uniform_(self.weight_real[bank])
            nn.init.xavier_uniform_(self.weight_imag[bank])
        with torch.no_grad():
            self.weight_real.mul_(math.sqrt(0.5))
            self.weight_imag.mul_(math.sqrt(0.5))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        expected = self.banks * self.poles_per_bank
        if real.shape != imag.shape or real.shape[-1] != expected:
            raise ValueError("grouped writer inputs have incompatible shapes")
        shape = (*real.shape[:-1], self.banks, self.poles_per_bank)
        grouped_real = real.reshape(shape)
        grouped_imag = imag.reshape(shape)
        output_real = torch.einsum(
            "...hp,hkp->...hk", grouped_real, self.weight_real
        ) - torch.einsum("...hp,hkp->...hk", grouped_imag, self.weight_imag)
        output_imag = torch.einsum(
            "...hp,hkp->...hk", grouped_real, self.weight_imag
        ) + torch.einsum("...hp,hkp->...hk", grouped_imag, self.weight_real)
        return output_real.flatten(-2), output_imag.flatten(-2)


class _ComplexIdentity(nn.Module):
    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        return real, imag


class IdentityComplexMemory1D(nn.Module):
    """Preserve the learned local drive while removing recurrent transport."""

    def forward(
        self,
        drive_real: Tensor,
        drive_imag: Tensor,
        *,
        damping_control: Tensor | None = None,
    ) -> ComplexField:
        if drive_real.shape != drive_imag.shape:
            raise ValueError("identity memory expects matching complex drives")
        if damping_control is not None:
            raise ValueError("identity memory does not accept damping control")
        return drive_real, drive_imag


class FixedPoleResidualSidecar(nn.Module):
    """Add fixed-pole memory after an otherwise unchanged local trunk."""

    def __init__(
        self,
        modes: int,
        pole_modes: int,
        *,
        context_length: int,
        scan_fp32: bool,
        initialization: Literal["legacy", "lifetime_palette"],
        minimum_half_life: float,
        maximum_half_life: float,
        decay_dominant_fraction: float,
        initial_scale: float,
        normalize_memory: bool,
        channelwise_scale: bool,
        epsilon: float,
        use_recurrence: bool,
    ) -> None:
        super().__init__()
        self.modes = int(modes)
        self.norm = ComplexRMSNorm(modes)
        self.reader = PackedComplexLinear(modes, pole_modes)
        self.memory = FixedComplexPoleMemory1D(
            pole_modes,
            context_length=context_length,
            scan_fp32=scan_fp32,
            initialization=initialization,
            minimum_half_life=minimum_half_life,
            maximum_half_life=maximum_half_life,
            decay_dominant_fraction=decay_dominant_fraction,
        )
        self.writer = PackedComplexLinear(pole_modes, modes)
        beta_shape = (modes,) if channelwise_scale else ()
        self.beta = nn.Parameter(torch.full(beta_shape, float(initial_scale)))
        self.normalize_memory = bool(normalize_memory)
        self.epsilon = float(epsilon)
        self.use_recurrence = bool(use_recurrence)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            raise ValueError("fixed-pole sidecar expects matching B,T,K coordinates")
        drive = self.reader(*self.norm(real, imag))
        state = self.memory(*drive) if self.use_recurrence else drive
        memory = self.writer(*state)
        if self.normalize_memory:
            trunk_rms = torch.sqrt(
                real.float().square().add(imag.float().square()).mean(dim=-1, keepdim=True)
            )
            memory_rms = torch.sqrt(
                memory[0]
                .float()
                .square()
                .add(memory[1].float().square())
                .mean(dim=-1, keepdim=True)
            )
            scale = (trunk_rms / (memory_rms + self.epsilon)).detach().to(real.dtype)
            memory = memory[0] * scale, memory[1] * scale
        beta = self.beta.to(dtype=real.dtype)
        return real + beta * memory[0], imag + beta * memory[1]


class ChunkedSemanticPoleMemory(nn.Module):
    """Compress completed semantic chunks into a slow fixed-pole history."""

    def __init__(
        self,
        modes: int,
        *,
        chunk_size: int,
        summary_width: int,
        pole_modes: int,
        upper_blocks: int,
        beta_initial: float,
        context_length: int,
        scan_fp32: bool,
        minimum_half_life: float,
        maximum_half_life: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.modes = int(modes)
        self.chunk_size = int(chunk_size)
        self.summary_modes = int(summary_width) // 2
        self.epsilon = float(epsilon)
        self.summary_norm = nn.RMSNorm(2 * modes, eps=epsilon)
        self.summary = nn.Linear(4 * modes, summary_width, bias=False)
        self.reader = PackedComplexLinear(self.summary_modes, pole_modes)
        self.memory = FixedComplexPoleMemory1D(
            pole_modes,
            context_length=context_length,
            scan_fp32=scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=minimum_half_life,
            maximum_half_life=maximum_half_life,
            decay_dominant_fraction=0.5,
        )
        self.writer = PackedComplexLinear(pole_modes, modes)
        self.beta = nn.Parameter(torch.full((upper_blocks,), float(beta_initial)))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.modes:
            raise ValueError("chunk memory expects matching B,T,K complex coordinates")
        batch, steps, _modes = real.shape
        full_chunks = steps // self.chunk_size
        if full_chunks == 0:
            return torch.zeros_like(real), torch.zeros_like(imag)
        packed = self.summary_norm(torch.cat((real, imag), dim=-1))
        chunks = packed[:, : full_chunks * self.chunk_size].reshape(
            batch,
            full_chunks,
            self.chunk_size,
            2 * self.modes,
        )
        evidence = torch.cat((chunks[:, :, -1], chunks.mean(dim=2)), dim=-1)
        summary_real, summary_imag = self.summary(evidence).split(self.summary_modes, dim=-1)
        state = self.memory(*self.reader(summary_real, summary_imag))
        zero = torch.zeros_like(state[0][:, :1]), torch.zeros_like(state[1][:, :1])
        delayed_state = (
            torch.cat((zero[0], state[0]), dim=1),
            torch.cat((zero[1], state[1]), dim=1),
        )
        memory_real, memory_imag = self.writer(*delayed_state)
        return (
            memory_real.repeat_interleave(self.chunk_size, dim=1)[:, :steps],
            memory_imag.repeat_interleave(self.chunk_size, dim=1)[:, :steps],
        )

    def inject(
        self,
        real: Tensor,
        imag: Tensor,
        memory_real: Tensor,
        memory_imag: Tensor,
        upper_index: int,
    ) -> ComplexField:
        trunk_rms = torch.sqrt(
            real.float().square().add(imag.float().square()).mean(dim=-1, keepdim=True)
        )
        memory_rms = torch.sqrt(
            memory_real.float()
            .square()
            .add(memory_imag.float().square())
            .mean(dim=-1, keepdim=True)
        )
        scale = (trunk_rms / (memory_rms + self.epsilon)).detach().to(real.dtype)
        beta = self.beta[upper_index].to(real.dtype)
        return (
            real + beta * memory_real * scale,
            imag + beta * memory_imag * scale,
        )


class SemanticEdgePoleMemory(nn.Module):
    """Persist complementary level/detail evidence on a slower semantic clock."""

    def __init__(
        self,
        modes: int,
        *,
        stride: int,
        pole_modes: int,
        upper_blocks: int,
        beta_initial: float,
        use_recurrence: bool,
        context_length: int,
        scan_fp32: bool,
        minimum_half_life: float,
        maximum_half_life: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.modes = int(modes)
        self.stride = int(stride)
        self.use_recurrence = bool(use_recurrence)
        self.epsilon = float(epsilon)
        self.norm = ComplexRMSNorm(modes)
        self.excitation = PackedComplexLinear(2 * modes, pole_modes)
        with torch.no_grad():
            nn.init.orthogonal_(self.excitation.weight_real)
            self.excitation.weight_imag.zero_()
        self.memory = FixedComplexPoleMemory1D(
            pole_modes,
            context_length=context_length,
            scan_fp32=scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=minimum_half_life,
            maximum_half_life=maximum_half_life,
            decay_dominant_fraction=0.5,
        )
        self.writer = PackedComplexLinear(pole_modes, modes)
        self.beta = nn.Parameter(torch.full((upper_blocks,), float(beta_initial)))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.modes:
            raise ValueError("semantic edge memory expects matching B,T,K coordinates")
        steps = real.shape[1]
        full_anchors = steps // self.stride
        if full_anchors == 0:
            return torch.zeros_like(real), torch.zeros_like(imag)
        unit_real, unit_imag = self.norm(real, imag)
        anchors_real = unit_real[:, self.stride - 1 : full_anchors * self.stride : self.stride]
        anchors_imag = unit_imag[:, self.stride - 1 : full_anchors * self.stride : self.stride]
        previous_real = torch.cat(
            (torch.zeros_like(anchors_real[:, :1]), anchors_real[:, :-1]), dim=1
        )
        previous_imag = torch.cat(
            (torch.zeros_like(anchors_imag[:, :1]), anchors_imag[:, :-1]), dim=1
        )
        inverse_sqrt_two = math.sqrt(0.5)
        level_real = inverse_sqrt_two * (previous_real + anchors_real)
        level_imag = inverse_sqrt_two * (previous_imag + anchors_imag)
        detail_real = inverse_sqrt_two * (anchors_real - previous_real)
        detail_imag = inverse_sqrt_two * (anchors_imag - previous_imag)
        drive = self.excitation(
            torch.cat((level_real, detail_real), dim=-1),
            torch.cat((level_imag, detail_imag), dim=-1),
        )
        state = self.memory(*drive) if self.use_recurrence else drive
        zero = torch.zeros_like(state[0][:, :1]), torch.zeros_like(state[1][:, :1])
        delayed_state = (
            torch.cat((zero[0], state[0]), dim=1),
            torch.cat((zero[1], state[1]), dim=1),
        )
        memory_real, memory_imag = self.writer(*delayed_state)
        return (
            memory_real.repeat_interleave(self.stride, dim=1)[:, :steps],
            memory_imag.repeat_interleave(self.stride, dim=1)[:, :steps],
        )

    def inject(
        self,
        real: Tensor,
        imag: Tensor,
        memory_real: Tensor,
        memory_imag: Tensor,
        upper_index: int,
    ) -> ComplexField:
        trunk_rms = torch.sqrt(
            real.float().square().add(imag.float().square()).mean(dim=-1, keepdim=True)
        )
        memory_rms = torch.sqrt(
            memory_real.float()
            .square()
            .add(memory_imag.float().square())
            .mean(dim=-1, keepdim=True)
        )
        scale = (trunk_rms / (memory_rms + self.epsilon)).detach().to(real.dtype)
        beta = self.beta[upper_index].to(real.dtype)
        return (
            real + beta * memory_real * scale,
            imag + beta * memory_imag * scale,
        )


class CausalCNNPoleMemory(nn.Module):
    """Build local evidence with a causal CNN and persist it in a pole sidecar."""

    def __init__(
        self,
        modes: int,
        *,
        evidence_width: int,
        kernel_size: int,
        pole_modes: int,
        beta_initial: float,
        use_recurrence: bool,
        context_length: int,
        scan_fp32: bool,
        minimum_half_life: float,
        maximum_half_life: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.modes = int(modes)
        self.evidence_width = int(evidence_width)
        self.kernel_size = int(kernel_size)
        self.use_recurrence = bool(use_recurrence)
        self.epsilon = float(epsilon)
        packed_width = 2 * modes
        self.norm = nn.RMSNorm(packed_width, eps=epsilon)
        self.pointwise = nn.Linear(packed_width, evidence_width, bias=False)
        self.temporal = nn.Conv1d(
            evidence_width,
            evidence_width,
            kernel_size,
            groups=evidence_width,
            bias=False,
        )
        self.analysis = nn.Linear(evidence_width, 2 * pole_modes, bias=False)
        self.memory = FixedComplexPoleMemory1D(
            pole_modes,
            context_length=context_length,
            scan_fp32=scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=minimum_half_life,
            maximum_half_life=maximum_half_life,
            decay_dominant_fraction=0.5,
        )
        self.synthesis = nn.Linear(2 * pole_modes, packed_width, bias=False)
        self.beta = nn.Parameter(torch.tensor(float(beta_initial)))
        nn.init.orthogonal_(self.pointwise.weight)
        with torch.no_grad():
            self.temporal.weight.zero_()
            self.temporal.weight[:, 0, -1] = 1.0
            nn.init.orthogonal_(self.analysis.weight)
        # The learned pointwise map can absorb any analysis basis. Keeping this
        # final map fixed preserves the non-expansive row-Stiefel contract
        # without a costly matrix parametrization in every forward pass.
        self.analysis.weight.requires_grad = False
        nn.init.xavier_uniform_(self.synthesis.weight)

    def pole_drive(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.modes:
            raise ValueError("CNN pole memory expects matching B,T,K coordinates")
        packed = self.norm(torch.cat((real, imag), dim=-1))
        evidence = self.pointwise(packed)
        evidence = functional.pad(evidence.transpose(1, 2), (self.kernel_size - 1, 0))
        evidence = functional.silu(self.temporal(evidence).transpose(1, 2))
        return self.analysis(evidence).chunk(2, dim=-1)

    def synthesize(self, state: ComplexField) -> ComplexField:
        return self.synthesis(torch.cat(state, dim=-1)).chunk(2, dim=-1)

    def raw_memory(self, real: Tensor, imag: Tensor) -> ComplexField:
        drive_real, drive_imag = self.pole_drive(real, imag)
        state = (
            self.memory(drive_real, drive_imag)
            if self.use_recurrence
            else (drive_real, drive_imag)
        )
        return self.synthesize(state)

    def inject_memory(
        self,
        real: Tensor,
        imag: Tensor,
        memory_real: Tensor,
        memory_imag: Tensor,
        beta: Tensor,
    ) -> ComplexField:
        trunk_rms = torch.sqrt(
            real.float().square().add(imag.float().square()).mean(dim=-1, keepdim=True)
        )
        memory_rms = torch.sqrt(
            memory_real.float()
            .square()
            .add(memory_imag.float().square())
            .mean(dim=-1, keepdim=True)
        )
        scale = (trunk_rms / (memory_rms + self.epsilon)).detach().to(real.dtype)
        active_beta = beta.to(real.dtype)
        return (
            real + active_beta * memory_real * scale,
            imag + active_beta * memory_imag * scale,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        memory_real, memory_imag = self.raw_memory(real, imag)
        return self.inject_memory(real, imag, memory_real, memory_imag, self.beta)


class SlowCausalCNNPoleMemory(CausalCNNPoleMemory):
    """Persist CNN evidence on a slower clock and inject it into upper blocks."""

    def __init__(
        self,
        modes: int,
        *,
        stride: int,
        upper_blocks: int,
        evidence_width: int,
        kernel_size: int,
        pole_modes: int,
        beta_initial: float,
        use_recurrence: bool,
        context_length: int,
        scan_fp32: bool,
        minimum_half_life: float,
        maximum_half_life: float,
        epsilon: float,
        query_mode: Literal["none", "anchor", "token"] = "none",
        query_rho: float = 0.5,
        key_enabled: bool = False,
        key_rho: float = 0.5,
    ) -> None:
        super().__init__(
            modes,
            evidence_width=evidence_width,
            kernel_size=kernel_size,
            pole_modes=pole_modes,
            beta_initial=beta_initial,
            use_recurrence=use_recurrence,
            context_length=context_length,
            scan_fp32=scan_fp32,
            minimum_half_life=minimum_half_life,
            maximum_half_life=maximum_half_life,
            epsilon=epsilon,
        )
        self.stride = int(stride)
        self.query_mode = query_mode
        self.query_rho = float(query_rho)
        self.key_rho = float(key_rho)
        initial_beta = float(self.beta.detach())
        self.beta = nn.Parameter(torch.full((upper_blocks,), initial_beta))
        if query_mode == "none":
            self.query_norm = None
            self.query = None
        else:
            self.query_norm = nn.RMSNorm(2 * modes, eps=epsilon)
            self.query = nn.Linear(2 * modes, pole_modes, bias=False)
            nn.init.zeros_(self.query.weight)
        if key_enabled:
            self.key_norm = nn.RMSNorm(2 * modes, eps=epsilon)
            self.key = nn.Linear(2 * modes, pole_modes, bias=False)
            nn.init.zeros_(self.key.weight)
        else:
            self.key_norm = None
            self.key = None

    def query_gate(self, packed: Tensor) -> Tensor:
        if self.query_norm is None or self.query is None:
            return torch.ones(
                *packed.shape[:-1],
                self.memory.modes,
                device=packed.device,
                dtype=packed.dtype,
            )
        logits = self.query(self.query_norm(packed))
        gate = 1.0 + self.query_rho * torch.tanh(logits)
        return gate / gate.mean(dim=-1, keepdim=True)

    def key_gate(self, packed: Tensor) -> Tensor:
        if self.key_norm is None or self.key is None:
            return torch.ones(
                *packed.shape[:-1],
                self.memory.modes,
                device=packed.device,
                dtype=packed.dtype,
            )
        logits = self.key(self.key_norm(packed))
        gate = 1.0 + self.key_rho * torch.tanh(logits)
        return gate / gate.mean(dim=-1, keepdim=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        steps = real.shape[1]
        full_anchors = steps // self.stride
        if full_anchors == 0:
            return torch.zeros_like(real), torch.zeros_like(imag)
        drive_real, drive_imag = self.pole_drive(real, imag)
        anchors = (
            drive_real[:, self.stride - 1 : full_anchors * self.stride : self.stride],
            drive_imag[:, self.stride - 1 : full_anchors * self.stride : self.stride],
        )
        packed = torch.cat((real, imag), dim=-1)
        anchor_features = packed[
            :, self.stride - 1 : full_anchors * self.stride : self.stride
        ]
        key_gate = self.key_gate(anchor_features)
        anchors = anchors[0] * key_gate, anchors[1] * key_gate
        state = self.memory(*anchors) if self.use_recurrence else anchors
        zero = torch.zeros_like(state[0][:, :1]), torch.zeros_like(state[1][:, :1])
        delayed_state = (
            torch.cat((zero[0], state[0]), dim=1),
            torch.cat((zero[1], state[1]), dim=1),
        )
        if self.query_mode == "anchor":
            anchor_gate = self.query_gate(anchor_features)
            delayed_gate = torch.cat((torch.ones_like(anchor_gate[:, :1]), anchor_gate), dim=1)
            delayed_state = (
                delayed_state[0] * delayed_gate,
                delayed_state[1] * delayed_gate,
            )
        if self.query_mode == "token":
            repeated_state = (
                delayed_state[0].repeat_interleave(self.stride, dim=1)[:, :steps],
                delayed_state[1].repeat_interleave(self.stride, dim=1)[:, :steps],
            )
            token_gate = self.query_gate(torch.cat((real, imag), dim=-1))
            return self.synthesize(
                (repeated_state[0] * token_gate, repeated_state[1] * token_gate)
            )
        memory_real, memory_imag = self.synthesize(delayed_state)
        return (
            memory_real.repeat_interleave(self.stride, dim=1)[:, :steps],
            memory_imag.repeat_interleave(self.stride, dim=1)[:, :steps],
        )

    def inject(
        self,
        real: Tensor,
        imag: Tensor,
        memory_real: Tensor,
        memory_imag: Tensor,
        upper_index: int,
    ) -> ComplexField:
        return self.inject_memory(
            real,
            imag,
            memory_real,
            memory_imag,
            self.beta[upper_index],
        )


class TensorProductPoleMemory1D(nn.Module):
    """Separate content coordinates from a small shared temporal pole basis."""

    minimum_damping = 1.0e-7

    def __init__(
        self,
        content_modes: int,
        temporal_modes: int,
        *,
        half_lives: tuple[float, ...],
        scan_fp32: bool,
        initial_read_gain: float,
    ) -> None:
        super().__init__()
        if (
            min(content_modes, temporal_modes) <= 0
            or len(half_lives) != temporal_modes
            or any(value <= 0.0 for value in half_lives)
            or initial_read_gain <= 0.0
        ):
            raise ValueError("invalid tensor-product pole memory configuration")
        self.content_modes = int(content_modes)
        self.temporal_modes = int(temporal_modes)
        self.scan_fp32 = bool(scan_fp32)
        self.initial_read_gain = float(initial_read_gain)
        damping = math.log(2.0) / torch.tensor(half_lives, dtype=torch.float32)
        self.raw_damping = nn.Parameter(torch.log(torch.expm1(damping - self.minimum_damping)))
        self.raw_frequency = nn.Parameter(torch.zeros(temporal_modes))
        shape = (content_modes, temporal_modes)
        self.write_real = nn.Parameter(torch.empty(shape))
        self.write_imag = nn.Parameter(torch.empty(shape))
        self.read_real = nn.Parameter(torch.empty(shape))
        self.read_imag = nn.Parameter(torch.empty(shape))
        write_real = torch.randn(shape)
        write_imag = torch.randn(shape)
        inverse_norm = torch.rsqrt(
            write_real.square().add(write_imag.square()).sum(dim=-1, keepdim=True)
        )
        with torch.no_grad():
            self.write_real.copy_(write_real * inverse_norm)
            self.write_imag.copy_(write_imag * inverse_norm)
            self.read_real.copy_(self.initial_read_gain * self.write_real)
            self.read_imag.copy_(-self.initial_read_gain * self.write_imag)

    def damping(self) -> Tensor:
        return self.minimum_damping + functional.softplus(self.raw_damping)

    def frequency(self) -> Tensor:
        return math.pi * torch.tanh(self.raw_frequency)

    def half_lives(self) -> Tensor:
        return math.log(2.0) / self.damping()

    def forward(
        self,
        drive_real: Tensor,
        drive_imag: Tensor,
        damping_control: Tensor | None = None,
    ) -> ComplexField:
        if (
            drive_real.shape != drive_imag.shape
            or drive_real.ndim != 3
            or drive_real.shape[-1] != self.content_modes
        ):
            raise ValueError("tensor-product memory expects matching B,T,K coordinates")
        if damping_control is not None:
            raise ValueError("tensor-product memory uses token-independent poles")
        drive_dtype = drive_real.dtype
        write_real = self.write_real.to(drive_dtype)
        write_imag = self.write_imag.to(drive_dtype)
        expanded_real = (
            drive_real.unsqueeze(-1) * write_real - drive_imag.unsqueeze(-1) * write_imag
        )
        expanded_imag = (
            drive_real.unsqueeze(-1) * write_imag + drive_imag.unsqueeze(-1) * write_real
        )
        scan_dtype = torch.float32 if self.scan_fp32 else drive_dtype
        damping = self.damping().to(device=drive_real.device, dtype=scan_dtype)
        frequency = self.frequency().to(device=drive_real.device, dtype=scan_dtype)
        decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
            damping,
            frequency,
            1.0,
            threshold=1.0e-4,
        )
        active_real = expanded_real.to(scan_dtype)
        active_imag = expanded_imag.to(scan_dtype)
        gamma_shape = (1, 1, 1, self.temporal_modes)
        gamma_real = gamma_real.view(gamma_shape)
        gamma_imag = gamma_imag.view(gamma_shape)
        input_real = gamma_real * active_real - gamma_imag * active_imag
        input_imag = gamma_imag * active_real + gamma_real * active_imag
        flattened_real = input_real.flatten(-2).contiguous()
        flattened_imag = input_imag.flatten(-2).contiguous()
        repeated_decay_real = decay_real.repeat(self.content_modes)
        repeated_decay_imag = decay_imag.repeat(self.content_modes)
        decay_shape = (1, 1, self.content_modes * self.temporal_modes)
        state_real, state_imag = pac_triton_recurrence_opaque_op(
            repeated_decay_real.view(decay_shape).expand_as(flattened_real),
            repeated_decay_imag.view(decay_shape).expand_as(flattened_imag),
            flattened_real,
            flattened_imag,
        )
        state_shape = (*drive_real.shape, self.temporal_modes)
        state_real = state_real.to(drive_dtype).reshape(state_shape)
        state_imag = state_imag.to(drive_dtype).reshape(state_shape)
        read_real = self.read_real.to(drive_dtype)
        read_imag = self.read_imag.to(drive_dtype)
        return (
            (state_real * read_real - state_imag * read_imag).sum(dim=-1),
            (state_real * read_imag + state_imag * read_real).sum(dim=-1),
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
        banks: int = 1,
    ) -> None:
        super().__init__()
        self.modes = int(modes)
        self.context_length = int(context_length)
        self.scan_fp32 = bool(scan_fp32)
        if modes % banks:
            raise ValueError("fixed-pole modes must divide evenly across banks")
        modes_per_bank = modes // banks
        if initialization == "legacy":
            damping = torch.logspace(
                math.log10(1.0 / context_length), math.log10(0.5), modes_per_bank
            )
            periods = torch.logspace(
                math.log10(4.0), math.log10(float(context_length)), modes_per_bank
            ).flip(0)
            frequency = (2.0 * math.pi / periods).clamp_max(0.95 * math.pi)
        elif initialization == "lifetime_palette":
            octaves = round(math.log2(maximum_half_life / minimum_half_life))
            if not math.isclose(minimum_half_life * 2**octaves, maximum_half_life):
                raise ValueError("lifetime palette bounds must span complete octaves")
            anchors = minimum_half_life * 2.0 ** torch.arange(octaves + 1, dtype=torch.float32)
            anchor_indices = torch.arange(modes_per_bank) * anchors.numel() // modes_per_bank
            half_lives = anchors[anchor_indices]
            damping = math.log(2.0) / half_lives
            frequency = torch.zeros(modes_per_bank)
            decay_modes = round(modes_per_bank * decay_dominant_fraction)
            decay_indices = torch.linspace(0, modes_per_bank - 1, decay_modes).round().long()
            oscillatory = torch.ones(modes_per_bank, dtype=torch.bool)
            oscillatory[decay_indices] = False
            periods = torch.logspace(
                math.log10(4.0),
                math.log10(maximum_half_life),
                int(oscillatory.sum()),
            )
            frequency[oscillatory] = 2.0 * math.pi / periods
        else:
            raise ValueError("unknown fixed-pole initialization")
        if banks > 1:
            damping = damping.repeat(banks)
            frequency = frequency.repeat(banks)
        self.raw_damping = nn.Parameter(torch.log(torch.expm1(damping - self.minimum_damping)))
        self.raw_frequency = nn.Parameter(torch.atanh((frequency / math.pi).clamp(-0.999, 0.999)))

    def damping(self) -> Tensor:
        return self.minimum_damping + functional.softplus(self.raw_damping)

    def frequency(self) -> Tensor:
        return math.pi * torch.tanh(self.raw_frequency)

    def coefficients(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return discrete_pole_real2d(self.damping(), self.frequency(), 1.0)

    def forward(
        self,
        drive_real: Tensor,
        drive_imag: Tensor,
        damping_control: Tensor | None = None,
    ) -> ComplexField:
        scan_dtype = torch.float32 if self.scan_fp32 else drive_real.dtype
        active_real, active_imag = drive_real.to(scan_dtype), drive_imag.to(scan_dtype)
        if damping_control is None:
            coefficients = self.coefficients()
        else:
            if damping_control.shape != drive_real.shape:
                raise ValueError("dynamic damping control must match the pole drive")
            frequency = self.frequency().to(device=drive_real.device, dtype=scan_dtype)
            coefficients = pole_gamma_from_control_real2d(
                self.raw_damping.to(device=drive_real.device, dtype=scan_dtype),
                frequency,
                damping_control.to(scan_dtype),
                self.minimum_damping,
                1.0,
            )
        values = tuple(
            value.to(device=drive_real.device, dtype=scan_dtype) for value in coefficients
        )
        dr, di, gr, gi = values
        input_real = gr * active_real - gi * active_imag
        input_imag = gi * active_real + gr * active_imag
        shape = (1, 1, self.modes)
        active_decay_real = dr.view(shape).expand_as(input_real) if dr.ndim == 1 else dr
        active_decay_imag = di.view(shape).expand_as(input_imag) if di.ndim == 1 else di
        state_real, state_imag = pac_triton_recurrence_opaque_op(
            active_decay_real,
            active_decay_imag,
            input_real,
            input_imag,
        )
        return state_real.to(drive_real.dtype), state_imag.to(drive_imag.dtype)


class LowRankPoleRouter(nn.Module):
    def __init__(
        self,
        modes: int,
        pole_modes: int,
        *,
        hidden_modes: int,
        read_gate: bool,
    ) -> None:
        super().__init__()
        self.pole_modes = pole_modes
        self.read_gate = read_gate
        self.norm = ComplexRMSNorm(modes)
        self.input = nn.Linear(2 * modes, hidden_modes, bias=False)
        outputs = pole_modes * (2 if read_gate else 1)
        self.output = nn.Linear(hidden_modes, outputs, bias=False)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor | None]:
        unit_real, unit_imag = self.norm(real, imag)
        hidden = functional.silu(self.input(torch.cat((unit_real, unit_imag), dim=-1)))
        gates = 1.0 + torch.tanh(self.output(hidden))
        if self.read_gate:
            write_gate, read_gate = gates.split(self.pole_modes, dim=-1)
            return write_gate, read_gate
        return gates, None


class LowRankDecaySelector(nn.Module):
    """Produce bounded token-conditioned offsets in raw damping space."""

    def __init__(
        self,
        modes: int,
        pole_modes: int,
        *,
        rank: int,
        initial_scale: float,
        control_bound: float,
    ) -> None:
        super().__init__()
        if (
            min(modes, pole_modes, rank) <= 0
            or not 0.0 < initial_scale < 1.0
            or control_bound <= 0.0
        ):
            raise ValueError("invalid low-rank decay selector configuration")
        self.modes = int(modes)
        self.pole_modes = int(pole_modes)
        self.rank = int(rank)
        self.control_bound = float(control_bound)
        self.norm = ComplexRMSNorm(modes)
        self.input = nn.Linear(2 * modes, rank, bias=False)
        self.output = nn.Linear(rank, pole_modes, bias=False)
        initial_logit = math.log(initial_scale / (1.0 - initial_scale))
        self.raw_scale = nn.Parameter(torch.tensor(initial_logit))
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def scale(self) -> Tensor:
        return torch.sigmoid(self.raw_scale)

    def forward(self, real: Tensor, imag: Tensor) -> Tensor:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            raise ValueError("low-rank decay selector expects matching B,T,K coordinates")
        unit_real, unit_imag = self.norm(real, imag)
        hidden = functional.silu(self.input(torch.cat((unit_real, unit_imag), dim=-1)))
        bounded = torch.tanh(self.output(hidden))
        return self.control_bound * self.scale().to(bounded.dtype) * bounded


class DynamicLowRankWrite(nn.Module):
    """Add a token-conditioned low-rank update to a static complex write map."""

    def __init__(
        self,
        modes: int,
        pole_modes: int,
        *,
        rank: int,
        initial_scale: float,
    ) -> None:
        super().__init__()
        if min(modes, pole_modes, rank) <= 0 or not 0.0 < initial_scale < 1.0:
            raise ValueError("invalid dynamic low-rank write configuration")
        self.modes = int(modes)
        self.pole_modes = int(pole_modes)
        self.rank = int(rank)
        self.norm = ComplexRMSNorm(modes)
        self.content = PackedComplexLinear(modes, rank)
        self.gate = nn.Linear(2 * modes, rank, bias=False)
        self.direction = PackedComplexLinear(rank, pole_modes)
        initial_logit = math.log(initial_scale / (1.0 - initial_scale))
        self.raw_scale = nn.Parameter(torch.tensor(initial_logit))
        nn.init.xavier_uniform_(self.gate.weight)

    def scale(self) -> Tensor:
        return torch.sigmoid(self.raw_scale)

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        base_real: Tensor,
        base_imag: Tensor,
    ) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            raise ValueError("dynamic write expects matching B,T,K coordinates")
        if (
            base_real.shape != base_imag.shape
            or base_real.shape[:-1] != real.shape[:-1]
            or base_real.shape[-1] != self.pole_modes
        ):
            raise ValueError("dynamic write base drive has incompatible shapes")
        unit_real, unit_imag = self.norm(real, imag)
        content_real, content_imag = self.content(unit_real, unit_imag)
        gate = functional.silu(self.gate(torch.cat((unit_real, unit_imag), dim=-1)))
        dynamic_real, dynamic_imag = self.direction(
            content_real * gate,
            content_imag * gate,
        )
        scale = self.scale().to(base_real.dtype)
        return base_real + scale * dynamic_real, base_imag + scale * dynamic_imag


class QueryConditionedLowRankReadout(nn.Module):
    """Add a small content-conditioned bilinear read from pole state.

    The established fixed writer remains the primary readout.  The current
    token queries a low-rank summary of the recurrent pole state, so the
    experiment changes state access without changing pole dynamics or the
    downstream PostFusion decoder.
    """

    def __init__(
        self,
        modes: int,
        pole_modes: int,
        *,
        rank: int,
        initial_scale: float,
    ) -> None:
        super().__init__()
        if min(modes, pole_modes, rank) <= 0 or not 0.0 < initial_scale < 1.0:
            raise ValueError("invalid query-conditioned readout configuration")
        self.modes = int(modes)
        self.pole_modes = int(pole_modes)
        self.rank = int(rank)
        self.query_norm = ComplexRMSNorm(modes)
        self.query = nn.Linear(2 * modes, rank, bias=False)
        self.state_projection = PackedComplexLinear(pole_modes, rank)
        self.output_projection = PackedComplexLinear(rank, modes)
        initial_logit = math.log(initial_scale / (1.0 - initial_scale))
        self.raw_scale = nn.Parameter(torch.tensor(initial_logit))
        nn.init.xavier_uniform_(self.query.weight)

    def scale(self) -> Tensor:
        return torch.sigmoid(self.raw_scale)

    def forward(
        self,
        query_real: Tensor,
        query_imag: Tensor,
        state_real: Tensor,
        state_imag: Tensor,
        base_real: Tensor,
        base_imag: Tensor,
    ) -> ComplexField:
        if query_real.shape != query_imag.shape or query_real.shape[-1] != self.modes:
            raise ValueError("query-conditioned readout query has incompatible shapes")
        if state_real.shape != state_imag.shape or state_real.shape[-1] != self.pole_modes:
            raise ValueError("query-conditioned readout state has incompatible shapes")
        if base_real.shape != base_imag.shape or base_real.shape != query_real.shape:
            raise ValueError("query-conditioned readout base memory has incompatible shapes")
        unit_real, unit_imag = self.query_norm(query_real, query_imag)
        query = functional.silu(self.query(torch.cat((unit_real, unit_imag), dim=-1)))
        summary_real, summary_imag = self.state_projection(state_real, state_imag)
        dynamic_real, dynamic_imag = self.output_projection(
            summary_real * query,
            summary_imag * query,
        )
        scale = self.scale().to(dtype=base_real.dtype)
        return base_real + scale * dynamic_real, base_imag + scale * dynamic_imag


def _make_memory(config: AlphabetLMConfig) -> nn.Module:
    if config.memory_layout == "flat":
        return FixedComplexPoleMemory1D(
            config.total_pole_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization=config.pole_initialization,
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
            decay_dominant_fraction=config.decay_dominant_fraction,
            banks=config.memory_banks,
        )
    if config.memory_layout in {"local_only", "local_sidecar"}:
        return IdentityComplexMemory1D()
    return TensorProductPoleMemory1D(
        config.modes,
        config.tensor_temporal_modes,
        half_lives=config.tensor_half_lives,
        scan_fp32=config.scan_fp32,
        initial_read_gain=config.tensor_initial_read_gain,
    )


def _make_sidecar(config: AlphabetLMConfig) -> FixedPoleResidualSidecar | None:
    if config.memory_layout != "local_sidecar":
        return None
    with torch.random.fork_rng(devices=[]):
        return FixedPoleResidualSidecar(
            config.modes,
            config.pole_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization=config.pole_initialization,
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
            decay_dominant_fraction=config.decay_dominant_fraction,
            initial_scale=config.sidecar_initial_scale,
            normalize_memory=config.sidecar_normalize_memory,
            channelwise_scale=config.sidecar_channelwise_scale,
            epsilon=config.rms_epsilon,
            use_recurrence=config.sidecar_use_recurrence,
        )


class AlphabetLMBlock(nn.Module):
    def __init__(self, config: AlphabetLMConfig) -> None:
        super().__init__()
        if config.memory_layout == "tensor_product":
            factorized_reader = CausalFactorizedComplexConv1dReader(
                config.modes,
                config.modes,
                rank=config.reader_rank,
                kernel_size=config.reader_kernel,
            )
            self.reader = DenseComplexConv1dReader.from_factorized(factorized_reader)
            self.writer = _ComplexIdentity()
        elif config.memory_banks == 1:
            factorized_reader = CausalFactorizedComplexConv1dReader(
                config.modes,
                config.pole_modes,
                rank=config.reader_rank,
                kernel_size=config.reader_kernel,
            )
            self.reader = (
                factorized_reader
                if config.reader_type == "r2k3"
                else DenseComplexConv1dReader.from_factorized(factorized_reader)
            )
            self.writer = PackedComplexLinear(config.pole_modes, config.modes)
        else:
            self.reader = GroupedCausalFactorizedComplexConv1dReader(
                config.modes,
                config.bank_pole_modes,
                banks=config.memory_banks,
                rank=config.reader_rank,
                kernel_size=config.reader_kernel,
            )
            self.writer = GroupedPackedComplexLinear(
                config.bank_pole_modes, config.modes, banks=config.memory_banks
            )
        self.memory = _make_memory(config)
        self.post_fusion = GatedComplexPostFusion(config.modes, config.post_hidden)
        if config.memory_readout == "query_low_rank":
            with torch.random.fork_rng(devices=[]):
                self.query_readout = QueryConditionedLowRankReadout(
                    config.modes,
                    config.total_pole_modes,
                    rank=config.query_read_rank,
                    initial_scale=config.query_read_initial_scale,
                )
        else:
            self.query_readout = None
        if config.pole_dynamics == "delta_select":
            with torch.random.fork_rng(devices=[]):
                self.decay_selector = LowRankDecaySelector(
                    config.modes,
                    config.total_pole_modes,
                    rank=config.delta_select_rank,
                    initial_scale=config.delta_select_initial_scale,
                    control_bound=config.delta_select_control_bound,
                )
        else:
            self.decay_selector = None
        if config.write_map == "dynamic_low_rank":
            with torch.random.fork_rng(devices=[]):
                self.dynamic_write = DynamicLowRankWrite(
                    config.modes,
                    config.total_pole_modes,
                    rank=config.dynamic_write_rank,
                    initial_scale=config.dynamic_write_initial_scale,
                )
        else:
            self.dynamic_write = None
        if config.pole_routing == "static":
            self.router = None
        else:
            with torch.random.fork_rng(devices=[]):
                self.router = LowRankPoleRouter(
                    config.modes,
                    config.total_pole_modes,
                    hidden_modes=config.router_hidden,
                    read_gate=config.pole_routing == "dynamic_write_read",
                )
        self.sidecar = _make_sidecar(config)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        drive = self.reader(real, imag)
        if self.dynamic_write is not None:
            drive = self.dynamic_write(real, imag, *drive)
        read_gate = None
        if self.router is not None:
            write_gate, read_gate = self.router(real, imag)
            drive = drive[0] * write_gate, drive[1] * write_gate
        damping_control = (
            self.decay_selector(real, imag) if self.decay_selector is not None else None
        )
        state = self.memory(*drive, damping_control=damping_control)
        if read_gate is not None:
            state = state[0] * read_gate, state[1] * read_gate
        memory = self.writer(*state)
        if self.query_readout is not None:
            memory = self.query_readout(real, imag, *state, *memory)
        output = self.post_fusion(real + memory[0], imag + memory[1])
        return output if self.sidecar is None else self.sidecar(*output)


class AlphabetLM(nn.Module):
    def __init__(self, config: AlphabetLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.model_width)
        self.analysis = nn.Linear(config.model_width, config.model_width, bias=False)
        self.blocks = nn.ModuleList(AlphabetLMBlock(config) for _ in range(config.layers))
        self.final_norm = nn.RMSNorm(config.model_width, eps=config.rms_epsilon)
        if config.chunk_memory:
            with torch.random.fork_rng(devices=[]):
                self.chunk_memory = ChunkedSemanticPoleMemory(
                    config.modes,
                    chunk_size=config.chunk_size,
                    summary_width=config.chunk_summary_width,
                    pole_modes=config.chunk_pole_modes,
                    upper_blocks=config.chunk_upper_blocks,
                    beta_initial=config.chunk_beta_initial,
                    context_length=config.context_length,
                    scan_fp32=config.scan_fp32,
                    minimum_half_life=config.chunk_minimum_half_life,
                    maximum_half_life=config.chunk_maximum_half_life,
                    epsilon=config.rms_epsilon,
                )
        else:
            self.chunk_memory = None
        if config.semantic_edge_memory:
            with torch.random.fork_rng(devices=[]):
                self.semantic_edge_memory = SemanticEdgePoleMemory(
                    config.modes,
                    stride=config.semantic_edge_stride,
                    pole_modes=config.semantic_edge_pole_modes,
                    upper_blocks=config.semantic_edge_upper_blocks,
                    beta_initial=config.semantic_edge_beta_initial,
                    use_recurrence=config.semantic_edge_use_recurrence,
                    context_length=config.context_length,
                    scan_fp32=config.scan_fp32,
                    minimum_half_life=config.semantic_edge_minimum_half_life,
                    maximum_half_life=config.semantic_edge_maximum_half_life,
                    epsilon=config.rms_epsilon,
                )
        else:
            self.semantic_edge_memory = None
        if config.cnn_pole_memory:
            with torch.random.fork_rng(devices=[]):
                self.cnn_pole_memories = nn.ModuleList(
                    CausalCNNPoleMemory(
                        config.modes,
                        evidence_width=config.cnn_pole_evidence_width,
                        kernel_size=config.cnn_pole_kernel_size,
                        pole_modes=config.cnn_pole_modes,
                        beta_initial=config.cnn_pole_beta_initial,
                        use_recurrence=config.cnn_pole_use_recurrence,
                        context_length=config.context_length,
                        scan_fp32=config.scan_fp32,
                        minimum_half_life=config.cnn_pole_minimum_half_life,
                        maximum_half_life=config.cnn_pole_maximum_half_life,
                        epsilon=config.rms_epsilon,
                    )
                    for _ in range(config.layers // config.cnn_pole_interval)
                )
        else:
            self.cnn_pole_memories = None
        if config.slow_cnn_pole_memory:
            with torch.random.fork_rng(devices=[]):
                self.slow_cnn_pole_memory = SlowCausalCNNPoleMemory(
                    config.modes,
                    stride=config.slow_cnn_pole_stride,
                    upper_blocks=config.slow_cnn_pole_upper_blocks,
                    evidence_width=config.slow_cnn_pole_evidence_width,
                    kernel_size=config.slow_cnn_pole_kernel_size,
                    pole_modes=config.slow_cnn_pole_modes,
                    beta_initial=config.slow_cnn_pole_beta_initial,
                    use_recurrence=config.slow_cnn_pole_use_recurrence,
                    context_length=config.context_length,
                    scan_fp32=config.scan_fp32,
                    minimum_half_life=config.slow_cnn_pole_minimum_half_life,
                    maximum_half_life=config.slow_cnn_pole_maximum_half_life,
                    epsilon=config.rms_epsilon,
                    query_mode=config.slow_cnn_pole_query,
                    query_rho=config.slow_cnn_pole_query_rho,
                    key_enabled=config.slow_cnn_pole_key,
                    key_rho=config.slow_cnn_pole_key_rho,
                )
        else:
            self.slow_cnn_pole_memory = None
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.orthogonal_(self.analysis.weight)

    def hidden(self, input_ids: Tensor) -> Tensor:
        packed = self.analysis(self.embedding(input_ids))
        real, imag = packed.split(self.config.modes, dim=-1)
        chunk_memory: ComplexField | None = None
        active_chunk_memory = self.chunk_memory
        active_edge_memory = self.semantic_edge_memory
        active_slow_memory = self.slow_cnn_pole_memory
        upper_blocks = (
            self.config.chunk_upper_blocks
            if active_chunk_memory is not None
            else self.config.semantic_edge_upper_blocks
        )
        memory_start = self.config.layers - upper_blocks
        slow_memory: ComplexField | None = None
        slow_memory_start = self.config.layers - self.config.slow_cnn_pole_upper_blocks
        for index, block in enumerate(self.blocks):
            if active_chunk_memory is not None and index == memory_start:
                chunk_memory = active_chunk_memory(real, imag)
            if active_chunk_memory is not None and chunk_memory is not None:
                real, imag = active_chunk_memory.inject(
                    real,
                    imag,
                    chunk_memory[0],
                    chunk_memory[1],
                    index - memory_start,
                )
            if active_edge_memory is not None and index == memory_start:
                chunk_memory = active_edge_memory(real, imag)
            if active_edge_memory is not None and chunk_memory is not None:
                real, imag = active_edge_memory.inject(
                    real,
                    imag,
                    chunk_memory[0],
                    chunk_memory[1],
                    index - memory_start,
                )
            if active_slow_memory is not None and index == slow_memory_start:
                slow_memory = active_slow_memory(real, imag)
            if active_slow_memory is not None and slow_memory is not None:
                real, imag = active_slow_memory.inject(
                    real,
                    imag,
                    slow_memory[0],
                    slow_memory[1],
                    index - slow_memory_start,
                )
            real, imag = block(real, imag)
            if (
                self.cnn_pole_memories is not None
                and (index + 1) % self.config.cnn_pole_interval == 0
            ):
                bank_index = (index + 1) // self.config.cnn_pole_interval - 1
                real, imag = self.cnn_pole_memories[bank_index](real, imag)
        return self.final_norm(torch.cat((real, imag), dim=-1))

    def forward(self, input_ids: Tensor) -> Tensor:
        return functional.linear(self.hidden(input_ids), self.embedding.weight)


__all__ = [
    "AlphabetLM",
    "AlphabetLMBlock",
    "AlphabetLMConfig",
    "CausalCNNPoleMemory",
    "CausalFactorizedComplexConv1dReader",
    "ChunkedSemanticPoleMemory",
    "DenseComplexConv1dReader",
    "DynamicLowRankWrite",
    "FixedComplexPoleMemory1D",
    "FixedPoleResidualSidecar",
    "GroupedCausalFactorizedComplexConv1dReader",
    "GroupedPackedComplexLinear",
    "IdentityComplexMemory1D",
    "LowRankDecaySelector",
    "LowRankPoleRouter",
    "QueryConditionedLowRankReadout",
    "SemanticEdgePoleMemory",
    "SlowCausalCNNPoleMemory",
    "TensorProductPoleMemory1D",
]
