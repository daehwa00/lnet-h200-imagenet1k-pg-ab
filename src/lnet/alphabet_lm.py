"""Causal fixed-complex-pole language model built from the PAC recurrence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .complex_scan_transitions import ComplexRMSNorm, complex_rms_unit
from .pac_complex_layers import PackedComplexLinear, WidelyLinear
from .pac_gated_post_fusion import GatedComplexPostFusion
from .pac_real2d_math import discrete_pole_real2d, pole_gamma_from_control_real2d
from .pac_triton_parallel_static_recurrence import (
    chunked_parallel_static_recurrence_packed,
)
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
    slow_cnn_pole_value_width: int = 1
    slow_cnn_pole_matrix_key_width: int = 1
    slow_cnn_pole_independent_matrix_value: bool = False
    slow_cnn_pole_vector_width: int = 1
    slow_cnn_pole_complex_vector_excitation: bool = False
    slow_cnn_pole_complex_vector_query: bool = False
    slow_cnn_pole_coordinate_read: bool = False
    slow_cnn_pole_dynamic_transport: bool = False
    slow_cnn_pole_transport_rank: int = 16
    slow_cnn_pole_transport_scale: float = 0.1
    slow_cnn_pole_transport_bound: float = 1.0
    slow_cnn_pole_specific_reader: bool = False
    slow_cnn_pole_reader_kernel: int = 3
    slow_cnn_pole_write_scheduler: bool = False
    slow_cnn_pole_innovation: bool = False
    slow_cnn_pole_innovation_kernel: int = 3
    slow_cnn_pole_semantic_clock: bool = False
    repeated_vector_pole_memory: bool = False
    repeated_vector_pole_interval: int = 1
    repeated_vector_pole_modes: int = 32
    repeated_vector_pole_width: int = 4
    repeated_vector_pole_reader_kernel: int = 3
    repeated_vector_pole_beta_initial: float = 0.01
    repeated_vector_pole_minimum_half_life: float = 16.0
    repeated_vector_pole_maximum_half_life: float = 4_096.0
    repeated_vector_pole_factorized: bool = False
    repeated_vector_pole_write_rank: int = 4
    repeated_vector_pole_query_rank: int = 4
    repeated_vector_pole_synthesis_rank: int = 16
    repeated_vector_pole_retain_factor_state: bool = False
    repeated_vector_pole_learned_factor_read: bool = False
    repeated_vector_pole_factor_read_rho: float = 0.5
    repeated_vector_pole_factor_write_law: Literal[
        "row_specific", "shared_outer", "pole_outer"
    ] = "row_specific"
    repeated_vector_pole_mamba_outer: bool = False
    repeated_vector_pole_outer_direct: bool = False
    repeated_vector_pole_outer_gate: bool = False
    repeated_vector_pole_outer_kernel: int = 4
    repeated_vector_pole_activation_checkpoint: bool = False
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
            self.slow_cnn_pole_value_width,
            self.slow_cnn_pole_matrix_key_width,
            self.slow_cnn_pole_vector_width,
            self.slow_cnn_pole_transport_rank,
            self.slow_cnn_pole_reader_kernel,
            self.slow_cnn_pole_innovation_kernel,
            self.repeated_vector_pole_interval,
            self.repeated_vector_pole_modes,
            self.repeated_vector_pole_width,
            self.repeated_vector_pole_reader_kernel,
            self.repeated_vector_pole_write_rank,
            self.repeated_vector_pole_query_rank,
            self.repeated_vector_pole_synthesis_rank,
            self.repeated_vector_pole_outer_kernel,
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
        self._validate_repeated_vector_pole_memory()
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
            if (
                self.slow_cnn_pole_query != "none"
                or self.slow_cnn_pole_key
                or self.slow_cnn_pole_value_width != 1
                or self.slow_cnn_pole_matrix_key_width != 1
                or self.slow_cnn_pole_independent_matrix_value
                or self.slow_cnn_pole_vector_width != 1
                or self.slow_cnn_pole_complex_vector_excitation
                or self.slow_cnn_pole_complex_vector_query
                or self.slow_cnn_pole_coordinate_read
                or self.slow_cnn_pole_dynamic_transport
                or self.slow_cnn_pole_specific_reader
                or self.slow_cnn_pole_write_scheduler
                or self.slow_cnn_pole_innovation
                or self.slow_cnn_pole_semantic_clock
            ):
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
        if self.slow_cnn_pole_matrix_key_width > 1 and (
            self.slow_cnn_pole_value_width <= 1 or self.slow_cnn_pole_query != "token"
        ):
            raise ValueError("matrix memory requires token query and vector value state")
        if self.slow_cnn_pole_independent_matrix_value and (
            self.slow_cnn_pole_matrix_key_width <= 1
            or self.slow_cnn_pole_value_width <= 1
        ):
            raise ValueError("independent matrix value requires matrix memory")
        if self.slow_cnn_pole_vector_width > 1 and (
            self.slow_cnn_pole_query != "token"
            or self.slow_cnn_pole_key
            or not self.slow_cnn_pole_use_recurrence
            or self.slow_cnn_pole_value_width != 1
            or self.slow_cnn_pole_matrix_key_width != 1
        ):
            raise ValueError("vector pole memory requires token query without value axes")
        if self.slow_cnn_pole_complex_vector_excitation and (
            self.slow_cnn_pole_vector_width <= 1
        ):
            raise ValueError("complex vector excitation requires vector pole memory")
        if self.slow_cnn_pole_complex_vector_query and (
            self.slow_cnn_pole_vector_width <= 1
            or self.slow_cnn_pole_query != "token"
        ):
            raise ValueError("complex vector query requires token-addressed vector memory")
        if self.slow_cnn_pole_coordinate_read and not (
            self.slow_cnn_pole_complex_vector_query
            and self.slow_cnn_pole_vector_width > 1
        ):
            raise ValueError("coordinate read requires complex vector query")
        self._validate_slow_dynamic_transport()
        self._validate_slow_pole_reader()

    def _validate_slow_pole_reader(self) -> None:
        if self.slow_cnn_pole_write_scheduler and not self.slow_cnn_pole_specific_reader:
            raise ValueError("write scheduler requires a pole-specific reader")
        if self.slow_cnn_pole_innovation and not self.slow_cnn_pole_specific_reader:
            raise ValueError("innovation filter requires a pole-specific reader")
        if self.slow_cnn_pole_innovation and self.slow_cnn_pole_write_scheduler:
            raise ValueError("innovation and write scheduler controls must be isolated")
        if self.slow_cnn_pole_semantic_clock and not self.slow_cnn_pole_specific_reader:
            raise ValueError("semantic clock requires a pole-specific reader")
        if self.slow_cnn_pole_semantic_clock and (
            self.slow_cnn_pole_write_scheduler
            or self.slow_cnn_pole_innovation
            or self.slow_cnn_pole_dynamic_transport
        ):
            raise ValueError("semantic clock must be tested without other transport controls")
        if not self.slow_cnn_pole_specific_reader:
            return
        if (
            self.slow_cnn_pole_stride != 1
            or self.slow_cnn_pole_vector_width <= 1
            or not self.slow_cnn_pole_complex_vector_query
            or not self.slow_cnn_pole_coordinate_read
        ):
            raise ValueError("pole-specific reader requires token-rate vector late fusion")

    def _validate_slow_dynamic_transport(self) -> None:
        if not self.slow_cnn_pole_dynamic_transport:
            return
        if (
            self.slow_cnn_pole_vector_width <= 1
            or not self.slow_cnn_pole_use_recurrence
            or not 0.0 < self.slow_cnn_pole_transport_scale < 1.0
            or self.slow_cnn_pole_transport_bound <= 0.0
        ):
            raise ValueError("dynamic transport requires recurrent vector pole memory")

    def _validate_repeated_vector_pole_memory(self) -> None:
        if not self.repeated_vector_pole_memory:
            if (
                self.repeated_vector_pole_factorized
                or self.repeated_vector_pole_retain_factor_state
                or self.repeated_vector_pole_learned_factor_read
                or self.repeated_vector_pole_mamba_outer
                or self.repeated_vector_pole_outer_direct
                or self.repeated_vector_pole_outer_gate
                or self.repeated_vector_pole_activation_checkpoint
            ):
                raise ValueError("factorized VectorPole requires repeated memory")
            return
        if (
            self.memory_layout != "local_only"
            or self.reader_type != "dense_k3"
            or self.chunk_memory
            or self.semantic_edge_memory
            or self.cnn_pole_memory
            or self.slow_cnn_pole_memory
            or self.layers % self.repeated_vector_pole_interval
            or self.repeated_vector_pole_width <= 1
            or (
                self.repeated_vector_pole_factorized
                and (
                    self.repeated_vector_pole_write_rank < min(
                        4, self.repeated_vector_pole_width
                    )
                    or self.repeated_vector_pole_query_rank < min(
                        4, self.repeated_vector_pole_width
                    )
                    or
                    self.repeated_vector_pole_write_rank
                    > self.repeated_vector_pole_width
                    or self.repeated_vector_pole_query_rank
                    > self.repeated_vector_pole_width
                )
            )
            or (
                self.repeated_vector_pole_retain_factor_state
                and not self.repeated_vector_pole_factorized
            )
            or (
                self.repeated_vector_pole_learned_factor_read
                and not self.repeated_vector_pole_retain_factor_state
            )
            or (
                (
                    self.repeated_vector_pole_outer_direct
                    or self.repeated_vector_pole_outer_gate
                )
                and not self.repeated_vector_pole_mamba_outer
            )
            or (
                self.repeated_vector_pole_mamba_outer
                and not self.repeated_vector_pole_retain_factor_state
            )
            or self.repeated_vector_pole_factor_write_law
            not in {"row_specific", "shared_outer", "pole_outer"}
            or (
                self.repeated_vector_pole_factor_write_law != "row_specific"
                and not self.repeated_vector_pole_retain_factor_state
            )
            or not 0.0 < self.repeated_vector_pole_factor_read_rho <= 1.0
            or not 0.0 < self.repeated_vector_pole_beta_initial < 1.0
            or self.repeated_vector_pole_minimum_half_life
            >= self.repeated_vector_pole_maximum_half_life
        ):
            raise ValueError("invalid repeated token-rate VectorPole configuration")

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


@dataclass(frozen=True, slots=True)
class LaplaceMambaLMConfig:
    vocab_size: int = 32_768
    model_width: int = 512
    layers: int = 19
    pole_modes: int = 32
    state_size: int = 4
    head_width: int = 16
    aligned_content_rank: int = 1
    observer_count: int = 8
    content_preserving_heads: int = 4
    content_preserving_poles_per_head: int = 8
    content_preserving_width_per_head: int = 64
    hybrid_dense_poles: int = 32
    hybrid_dense_width: int = 8
    dynamic_delta_hidden: int = 32
    dynamic_delta_log_bound: float = math.log(2.0)
    conv_width: int = 4
    context_length: int = 2_048
    scan_fp32: bool = True
    rms_epsilon: float = 1.0e-6
    minimum_half_life: float = 16.0
    maximum_half_life: float = 4_096.0
    activation_checkpoint: bool = True
    parallel_static_scan: bool = False

    def __post_init__(self) -> None:
        integers = (
            self.vocab_size,
            self.model_width,
            self.layers,
            self.pole_modes,
            self.state_size,
            self.head_width,
            self.aligned_content_rank,
            self.observer_count,
            self.content_preserving_heads,
            self.content_preserving_poles_per_head,
            self.content_preserving_width_per_head,
            self.hybrid_dense_poles,
            self.hybrid_dense_width,
            self.dynamic_delta_hidden,
            self.conv_width,
            self.context_length,
        )
        if any(value <= 0 for value in integers):
            raise ValueError("invalid Laplace-Mamba configuration")
        if (
            self.model_width % 2
            or self.minimum_half_life >= self.maximum_half_life
            or self.dynamic_delta_log_bound <= 0.0
        ):
            raise ValueError("invalid Laplace-Mamba width or pole lifetime")

    @property
    def inner_complex_width(self) -> int:
        return self.pole_modes * self.head_width

    @property
    def inner_real_width(self) -> int:
        return 2 * self.inner_complex_width

    @property
    def content_preserving_width(self) -> int:
        return self.content_preserving_heads * self.content_preserving_width_per_head

    @property
    def content_preserving_poles(self) -> int:
        return self.content_preserving_heads * self.content_preserving_poles_per_head


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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)
        with torch.no_grad():
            self.weight_real.mul_(math.sqrt(0.5))
            self.weight_imag.mul_(math.sqrt(0.5))

    @classmethod
    def from_factorized(
        cls, source: CausalFactorizedComplexConv1dReader
    ) -> DenseComplexConv1dReader:
        with torch.random.fork_rng(devices=[]):
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


class StrictComplexCausalConv1d(nn.Module):
    """Dense strict-complex causal convolution without an implicit normalization."""

    def __init__(self, input_modes: int, output_modes: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        if min(input_modes, output_modes, kernel_size) <= 0:
            raise ValueError("invalid strict complex causal convolution")
        self.input_modes = int(input_modes)
        self.output_modes = int(output_modes)
        self.kernel_size = int(kernel_size)
        shape = (self.output_modes, self.input_modes, self.kernel_size)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)
        with torch.no_grad():
            self.weight_real.mul_(math.sqrt(0.5))
            self.weight_imag.mul_(math.sqrt(0.5))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.input_modes:
            raise ValueError("strict complex causal convolution inputs are incompatible")
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


class PoleAlignedComplexCausalConv1d(nn.Module):
    """Give every pole a causal filter over the same semantic coordinates."""

    def __init__(self, poles: int, content_modes: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        if min(poles, content_modes, kernel_size) <= 0:
            raise ValueError("invalid pole-aligned causal convolution")
        self.poles = int(poles)
        self.content_modes = int(content_modes)
        self.kernel_size = int(kernel_size)
        shape = (self.poles, self.content_modes, self.kernel_size)
        self.weight_real = nn.Parameter(torch.zeros(shape))
        self.weight_imag = nn.Parameter(torch.zeros(shape))
        with torch.no_grad():
            self.weight_real[..., -1].fill_(1.0)

    def packed_weight(self) -> Tensor:
        real = self.weight_real.permute(1, 0, 2)
        imag = self.weight_imag.permute(1, 0, 2)
        real_rows = torch.stack((real, -imag), dim=2)
        imag_rows = torch.stack((imag, real), dim=2)
        return torch.cat((real_rows, imag_rows), dim=1).reshape(
            2 * self.content_modes * self.poles,
            2,
            self.kernel_size,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.content_modes:
            raise ValueError("pole-aligned causal convolution inputs are incompatible")
        batch, steps, _modes = real.shape
        packed = torch.stack(
            (real.transpose(1, 2), imag.transpose(1, 2)),
            dim=2,
        ).reshape(batch, 2 * self.content_modes, steps)
        padded = functional.pad(packed, (self.kernel_size - 1, 0))
        output = functional.conv1d(
            padded,
            self.packed_weight(),
            groups=self.content_modes,
        ).reshape(batch, self.content_modes, 2 * self.poles, steps)
        output_real, output_imag = output.split(self.poles, dim=2)
        return (
            output_real.permute(0, 3, 2, 1),
            output_imag.permute(0, 3, 2, 1),
        )


class LowRankPoleAlignedComplexCausalConv1d(nn.Module):
    """Mix a small shared semantic family independently at every fixed pole."""

    def __init__(
        self,
        poles: int,
        content_modes: int,
        basis_rank: int,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if min(poles, content_modes, basis_rank, kernel_size) <= 0:
            raise ValueError("invalid low-rank pole-aligned causal convolution")
        self.poles = int(poles)
        self.content_modes = int(content_modes)
        self.basis_rank = int(basis_rank)
        self.kernel_size = int(kernel_size)
        shape = (self.poles, self.content_modes, self.basis_rank, self.kernel_size)
        self.weight_real = nn.Parameter(torch.zeros(shape))
        self.weight_imag = nn.Parameter(torch.zeros(shape))
        with torch.no_grad():
            current_real = torch.randn(self.poles, self.content_modes, self.basis_rank)
            current_imag = torch.randn_like(current_real)
            norm = current_real.square().add(current_imag.square()).sum(-1, keepdim=True).sqrt()
            self.weight_real[..., -1].copy_(current_real / norm)
            self.weight_imag[..., -1].copy_(current_imag / norm)

    def packed_weight(self) -> Tensor:
        real = self.weight_real.permute(1, 0, 2, 3)
        imag = self.weight_imag.permute(1, 0, 2, 3)
        real_rows = torch.cat((real, -imag), dim=2)
        imag_rows = torch.cat((imag, real), dim=2)
        return torch.cat((real_rows, imag_rows), dim=1).reshape(
            2 * self.content_modes * self.poles,
            2 * self.basis_rank,
            self.kernel_size,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        expected = self.basis_rank * self.content_modes
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != expected:
            raise ValueError("low-rank pole-aligned inputs are incompatible")
        batch, steps, _modes = real.shape
        shape = (batch, steps, self.basis_rank, self.content_modes)
        grouped_real = real.reshape(shape).permute(0, 3, 2, 1)
        grouped_imag = imag.reshape(shape).permute(0, 3, 2, 1)
        packed = torch.cat((grouped_real, grouped_imag), dim=2).reshape(
            batch,
            2 * self.basis_rank * self.content_modes,
            steps,
        )
        padded = functional.pad(packed, (self.kernel_size - 1, 0))
        output = functional.conv1d(
            padded,
            self.packed_weight(),
            groups=self.content_modes,
        ).reshape(batch, self.content_modes, 2 * self.poles, steps)
        output_real, output_imag = output.split(self.poles, dim=2)
        return (
            output_real.permute(0, 3, 2, 1),
            output_imag.permute(0, 3, 2, 1),
        )


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


class PoleSpecificCausalVectorReader(nn.Module):
    """Dense complex FIR reader with an independent filter per pole coordinate."""

    def __init__(
        self,
        modes: int,
        pole_modes: int,
        vector_width: int,
        *,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if min(modes, pole_modes, vector_width, kernel_size) <= 0:
            raise ValueError("invalid pole-specific reader dimensions")
        self.modes = modes
        self.pole_modes = pole_modes
        self.vector_width = vector_width
        self.kernel_size = kernel_size
        outputs = pole_modes * vector_width
        self.norm = ComplexRMSNorm(modes)
        self.weight_real = nn.Parameter(torch.zeros(outputs, modes, kernel_size))
        self.weight_imag = nn.Parameter(torch.zeros(outputs, modes, kernel_size))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.weight_real[:, :, -1])
            nn.init.xavier_uniform_(self.weight_imag[:, :, -1])
            self.weight_real.mul_(math.sqrt(0.5))
            self.weight_imag.mul_(math.sqrt(0.5))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.modes:
            raise ValueError("pole-specific reader expects matching B,T,K coordinates")
        real, imag = self.norm(real, imag)
        packed_real = functional.pad(real.transpose(1, 2), (self.kernel_size - 1, 0))
        packed_imag = functional.pad(imag.transpose(1, 2), (self.kernel_size - 1, 0))
        output_real = functional.conv1d(packed_real, self.weight_real) - functional.conv1d(
            packed_imag, self.weight_imag
        )
        output_imag = functional.conv1d(packed_real, self.weight_imag) + functional.conv1d(
            packed_imag, self.weight_real
        )
        shape = (real.shape[0], real.shape[1], self.pole_modes, self.vector_width)
        output_real = output_real.transpose(1, 2)
        output_imag = output_imag.transpose(1, 2)
        return output_real.reshape(shape), output_imag.reshape(shape)


class ComplexDepthwiseCausalPredictor(nn.Module):
    """Predict each pole coordinate from only its own recent complex history."""

    def __init__(self, pole_modes: int, vector_width: int, *, kernel_size: int) -> None:
        super().__init__()
        if min(pole_modes, vector_width, kernel_size) <= 0:
            raise ValueError("invalid complex predictor dimensions")
        self.pole_modes = int(pole_modes)
        self.vector_width = int(vector_width)
        self.kernel_size = int(kernel_size)
        shape = (pole_modes, vector_width, kernel_size)
        self.weight_real = nn.Parameter(torch.zeros(shape))
        self.weight_imag = nn.Parameter(torch.zeros(shape))
        with torch.no_grad():
            self.weight_real[..., 0] = 1.0

    def normalized_weights(self) -> ComplexField:
        norm = (
            self.weight_real.float()
            .square()
            .add(self.weight_imag.float().square())
            .sum(dim=-1, keepdim=True)
            .sqrt()
            .clamp_min(1.0e-12)
        )
        return self.weight_real / norm, self.weight_imag / norm

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        expected = (self.pole_modes, self.vector_width)
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-2:] != expected:
            raise ValueError("complex predictor expects matching B,T,P,R coordinates")
        batch, steps = real.shape[:2]
        channels = self.pole_modes * self.vector_width
        active_real = functional.pad(
            real.flatten(2).transpose(1, 2), (self.kernel_size, 0)
        )
        active_imag = functional.pad(
            imag.flatten(2).transpose(1, 2), (self.kernel_size, 0)
        )
        weight_real, weight_imag = self.normalized_weights()
        weight_real = weight_real.flip(-1).reshape(channels, 1, self.kernel_size)
        weight_imag = weight_imag.flip(-1).reshape(channels, 1, self.kernel_size)
        predicted_real = functional.conv1d(
            active_real, weight_real, groups=channels
        ) - functional.conv1d(active_imag, weight_imag, groups=channels)
        predicted_imag = functional.conv1d(
            active_real, weight_imag, groups=channels
        ) + functional.conv1d(active_imag, weight_real, groups=channels)
        shape = (batch, steps, self.pole_modes, self.vector_width)
        return (
            predicted_real[:, :, :steps].transpose(1, 2).reshape(shape),
            predicted_imag[:, :, :steps].transpose(1, 2).reshape(shape),
        )


class PoleInnovationFilter(nn.Module):
    """Remove a learned fraction of predictable excitation before transport."""

    def __init__(self, pole_modes: int, vector_width: int, *, kernel_size: int) -> None:
        super().__init__()
        self.pole_modes = int(pole_modes)
        self.predictor = ComplexDepthwiseCausalPredictor(
            pole_modes, vector_width, kernel_size=kernel_size
        )
        self.raw_strength = nn.Parameter(torch.zeros(pole_modes))

    def strength(self) -> Tensor:
        return self.raw_strength.clamp(0.0, 1.0)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        predicted_real, predicted_imag = self.predictor(real, imag)
        strength = self.strength().to(real.dtype).view(1, 1, self.pole_modes, 1)
        return real - strength * predicted_real, imag - strength * predicted_imag


class LearnedSemanticClock(nn.Module):
    """Produce one bounded Laplace-time increment shared by every pole."""

    def __init__(self, modes: int, *, epsilon: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(2 * modes, eps=epsilon)
        with torch.random.fork_rng(devices=[]):
            self.hold = nn.Linear(2 * modes, 1, bias=True)
        nn.init.zeros_(self.hold.weight)
        nn.init.zeros_(self.hold.bias)

    def forward(self, packed: Tensor) -> Tensor:
        hold = self.hold(self.norm(packed)).squeeze(-1).clamp(0.0, 1.0)
        return 1.0 - hold


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
        value_width: int = 1,
        matrix_key_width: int = 1,
        independent_matrix_value: bool = False,
        vector_width: int = 1,
        complex_vector_excitation: bool = False,
        complex_vector_query: bool = False,
        coordinate_read: bool = False,
        dynamic_transport: bool = False,
        transport_rank: int = 16,
        transport_scale: float = 0.1,
        transport_bound: float = 1.0,
        pole_specific_reader: bool = False,
        reader_kernel: int = 16,
        write_scheduler: bool = False,
        innovation: bool = False,
        innovation_kernel: int = 3,
        semantic_clock: bool = False,
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
        self.value_width = int(value_width)
        self.matrix_key_width = int(matrix_key_width)
        self.independent_matrix_value = bool(independent_matrix_value)
        self.vector_width = int(vector_width)
        self.complex_vector_excitation = bool(complex_vector_excitation)
        self.complex_vector_query = bool(complex_vector_query)
        self.coordinate_read = bool(coordinate_read)
        self.pole_specific_reader = (
            PoleSpecificCausalVectorReader(
                modes,
                pole_modes,
                vector_width,
                kernel_size=reader_kernel,
            )
            if pole_specific_reader
            else None
        )
        self._configure_forcing_controls(
            modes=modes,
            pole_modes=pole_modes,
            vector_width=vector_width,
            epsilon=epsilon,
            write_scheduler=write_scheduler,
            innovation=innovation,
            innovation_kernel=innovation_kernel,
            semantic_clock=semantic_clock,
        )
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
        if value_width > 1:
            self.value_norm = nn.RMSNorm(2 * modes, eps=epsilon)
            self.value = nn.Linear(2 * modes, value_width - 1, bias=False)
            self.extra_synthesis = nn.Linear(
                2 * pole_modes * (value_width - 1),
                2 * modes,
                bias=False,
            )
            nn.init.xavier_uniform_(self.value.weight)
            nn.init.zeros_(self.extra_synthesis.weight)
        else:
            self.value_norm = None
            self.value = None
            self.extra_synthesis = None
        self._configure_matrix_axes(
            modes=modes,
            pole_modes=pole_modes,
            key_width=matrix_key_width,
            value_width=value_width,
            epsilon=epsilon,
            independent_value=independent_matrix_value,
        )
        self._configure_vector_poles(
            modes=modes,
            pole_modes=pole_modes,
            vector_width=vector_width,
            epsilon=epsilon,
            complex_excitation=complex_vector_excitation,
            complex_query=complex_vector_query,
            coordinate_read=coordinate_read,
        )
        if dynamic_transport:
            self.transport_selector = LowRankDecaySelector(
                modes,
                pole_modes,
                rank=transport_rank,
                initial_scale=transport_scale,
                control_bound=transport_bound,
            )
            nn.init.zeros_(self.transport_selector.output.weight)
        else:
            self.transport_selector = None
        if self.pole_specific_reader is not None:
            for module in (
                self.norm,
                self.pointwise,
                self.temporal,
                self.analysis,
                self.vector_excitation_norm,
                self.vector_excitation,
                self.vector_excitation_imag_norm,
                self.vector_excitation_imag,
            ):
                if module is not None:
                    module.requires_grad_(requires_grad=False)

    def _configure_forcing_controls(
        self,
        *,
        modes: int,
        pole_modes: int,
        vector_width: int,
        epsilon: float,
        write_scheduler: bool,
        innovation: bool,
        innovation_kernel: int,
        semantic_clock: bool,
    ) -> None:
        self.write_scheduler_norm = None
        self.write_scheduler = None
        if write_scheduler:
            self.write_scheduler_norm = nn.RMSNorm(2 * modes, eps=epsilon)
            with torch.random.fork_rng(devices=[]):
                self.write_scheduler = nn.Linear(2 * modes, pole_modes, bias=True)
            nn.init.zeros_(self.write_scheduler.weight)
            nn.init.zeros_(self.write_scheduler.bias)
        self.innovation_filter = (
            PoleInnovationFilter(
                pole_modes,
                vector_width,
                kernel_size=innovation_kernel,
            )
            if innovation
            else None
        )
        self.semantic_clock = (
            LearnedSemanticClock(modes, epsilon=epsilon) if semantic_clock else None
        )

    def _configure_matrix_axes(
        self,
        *,
        modes: int,
        pole_modes: int,
        key_width: int,
        value_width: int,
        epsilon: float,
        independent_value: bool,
    ) -> None:
        self.matrix_key_norm = None
        self.matrix_key = None
        self.matrix_query_norm = None
        self.matrix_query = None
        self.matrix_value_norm = None
        self.matrix_value = None
        if key_width <= 1:
            return
        extra_axes = pole_modes * (key_width - 1)
        self.matrix_key_norm = nn.RMSNorm(2 * modes, eps=epsilon)
        self.matrix_key = nn.Linear(2 * modes, extra_axes, bias=False)
        self.matrix_query_norm = nn.RMSNorm(2 * modes, eps=epsilon)
        self.matrix_query = nn.Linear(2 * modes, extra_axes, bias=False)
        nn.init.xavier_uniform_(self.matrix_key.weight)
        nn.init.zeros_(self.matrix_query.weight)
        if independent_value:
            self.matrix_value_norm = nn.RMSNorm(2 * modes, eps=epsilon)
            self.matrix_value = nn.Linear(2 * modes, key_width * value_width, bias=False)
            nn.init.zeros_(self.matrix_value.weight)

    def _configure_vector_poles(
        self,
        *,
        modes: int,
        pole_modes: int,
        vector_width: int,
        epsilon: float,
        complex_excitation: bool,
        complex_query: bool,
        coordinate_read: bool,
    ) -> None:
        self.vector_excitation_norm = None
        self.vector_excitation = None
        self.vector_query_norm = None
        self.vector_query = None
        self.vector_excitation_imag_norm = None
        self.vector_excitation_imag = None
        self.vector_query_imag_norm = None
        self.vector_query_imag = None
        self.coordinate_synthesis = None
        if vector_width <= 1:
            return
        extra_coordinates = pole_modes * (vector_width - 1)
        self.vector_excitation_norm = nn.RMSNorm(2 * modes, eps=epsilon)
        self.vector_excitation = nn.Linear(2 * modes, extra_coordinates, bias=False)
        self.vector_query_norm = nn.RMSNorm(2 * modes, eps=epsilon)
        self.vector_query = nn.Linear(2 * modes, extra_coordinates, bias=False)
        nn.init.xavier_uniform_(self.vector_excitation.weight)
        nn.init.zeros_(self.vector_query.weight)
        if complex_excitation:
            self.vector_excitation_imag_norm = nn.RMSNorm(2 * modes, eps=epsilon)
            self.vector_excitation_imag = nn.Linear(
                2 * modes, extra_coordinates, bias=False
            )
            nn.init.xavier_uniform_(self.vector_excitation_imag.weight)
        if complex_query:
            self.vector_query_imag_norm = nn.RMSNorm(2 * modes, eps=epsilon)
            self.vector_query_imag = nn.Linear(
                2 * modes, pole_modes * vector_width, bias=False
            )
            nn.init.xavier_uniform_(self.vector_query_imag.weight)
        if coordinate_read:
            self.synthesis.weight.requires_grad = False
            self.coordinate_synthesis = nn.Linear(
                2 * pole_modes * vector_width, 2 * modes, bias=False
            )
            nn.init.xavier_uniform_(self.coordinate_synthesis.weight)

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

    def write_gate(self, packed: Tensor) -> Tensor:
        if self.write_scheduler_norm is None or self.write_scheduler is None:
            return torch.ones(
                *packed.shape[:-1],
                self.memory.modes,
                device=packed.device,
                dtype=packed.dtype,
            )
        logits = self.write_scheduler(self.write_scheduler_norm(packed))
        return 2.0 * torch.sigmoid(logits)

    def anchor_value(self, packed: Tensor) -> Tensor:
        ones = torch.ones(*packed.shape[:-1], 1, device=packed.device, dtype=packed.dtype)
        if self.value_norm is None or self.value is None:
            return ones
        extra = torch.tanh(self.value(self.value_norm(packed)))
        return torch.cat((ones, extra), dim=-1)

    def matrix_key_axes(self, packed: Tensor) -> Tensor:
        shape = (*packed.shape[:-1], self.memory.modes, 1)
        ones = torch.ones(shape, device=packed.device, dtype=packed.dtype)
        if self.matrix_key_norm is None or self.matrix_key is None:
            return ones
        extra = torch.tanh(self.matrix_key(self.matrix_key_norm(packed)))
        extra = extra.reshape(*packed.shape[:-1], self.memory.modes, self.matrix_key_width - 1)
        return torch.cat((ones, extra), dim=-1)

    def matrix_query_axes(self, packed: Tensor, scalar_query: Tensor) -> Tensor:
        base = scalar_query.unsqueeze(-1)
        if self.matrix_query_norm is None or self.matrix_query is None:
            return base
        extra = torch.tanh(self.matrix_query(self.matrix_query_norm(packed)))
        extra = extra.reshape(*packed.shape[:-1], self.memory.modes, self.matrix_key_width - 1)
        return torch.cat((base, extra), dim=-1)

    def matrix_value_axes(self, packed: Tensor, shared_value: Tensor) -> Tensor:
        base = shared_value.unsqueeze(-2).expand(
            *shared_value.shape[:-1], self.matrix_key_width, self.value_width
        )
        if self.matrix_value_norm is None or self.matrix_value is None:
            return base
        delta = torch.tanh(self.matrix_value(self.matrix_value_norm(packed)))
        return base + delta.reshape(
            *packed.shape[:-1], self.matrix_key_width, self.value_width
        )

    def vector_excitation_axes(self, packed: Tensor) -> Tensor:
        shape = (*packed.shape[:-1], self.memory.modes, 1)
        ones = torch.ones(shape, device=packed.device, dtype=packed.dtype)
        if self.vector_excitation_norm is None or self.vector_excitation is None:
            return ones
        extra = torch.tanh(self.vector_excitation(self.vector_excitation_norm(packed)))
        extra = extra.reshape(*packed.shape[:-1], self.memory.modes, self.vector_width - 1)
        return torch.cat((ones, extra), dim=-1)

    def vector_query_axes(self, packed: Tensor, scalar_query: Tensor) -> Tensor:
        base = scalar_query.unsqueeze(-1)
        if self.vector_query_norm is None or self.vector_query is None:
            return base
        extra = torch.tanh(self.vector_query(self.vector_query_norm(packed)))
        extra = extra.reshape(*packed.shape[:-1], self.memory.modes, self.vector_width - 1)
        raw = torch.cat((base, extra), dim=-1)
        norm = raw.float().square().sum(dim=-1, keepdim=True).sqrt().clamp_min(self.epsilon)
        return raw * (base.abs().float() / norm).to(raw.dtype)

    def vector_query_components(
        self, packed: Tensor, scalar_query: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.vector_query_imag_norm is None or self.vector_query_imag is None:
            real = self.vector_query_axes(packed, scalar_query)
            return real, torch.zeros_like(real)
        base = scalar_query.unsqueeze(-1)
        if self.vector_query_norm is None or self.vector_query is None:
            raise RuntimeError("complex vector query is missing its real coordinates")
        extra_real = torch.tanh(self.vector_query(self.vector_query_norm(packed)))
        extra_real = extra_real.reshape(
            *packed.shape[:-1], self.memory.modes, self.vector_width - 1
        )
        raw_real = torch.cat((base, extra_real), dim=-1)
        raw_imag = torch.tanh(
            self.vector_query_imag(self.vector_query_imag_norm(packed))
        ).reshape(*packed.shape[:-1], self.memory.modes, self.vector_width)
        norm = (
            raw_real.float()
            .square()
            .add(raw_imag.float().square())
            .sum(dim=-1, keepdim=True)
            .sqrt()
            .clamp_min(self.epsilon)
        )
        scale = (base.abs().float() / norm).to(raw_real.dtype)
        return raw_real * scale, raw_imag * scale

    def vector_excitation_imag_axes(self, packed: Tensor) -> Tensor:
        shape = (*packed.shape[:-1], self.memory.modes, 1)
        zero = torch.zeros(shape, device=packed.device, dtype=packed.dtype)
        if self.vector_excitation_imag_norm is None or self.vector_excitation_imag is None:
            return zero.expand(*packed.shape[:-1], self.memory.modes, self.vector_width)
        extra = torch.tanh(
            self.vector_excitation_imag(self.vector_excitation_imag_norm(packed))
        )
        extra = extra.reshape(*packed.shape[:-1], self.memory.modes, self.vector_width - 1)
        return torch.cat((zero, extra), dim=-1)

    def synthesize_vector(self, state: ComplexField) -> ComplexField:
        state_real, state_imag = state
        base = self.synthesize((state_real[..., 0], state_imag[..., 0]))
        if self.extra_synthesis is None:
            return base
        extra_real = state_real[..., 1:].flatten(-2)
        extra_imag = state_imag[..., 1:].flatten(-2)
        projected = self.extra_synthesis(torch.cat((extra_real, extra_imag), dim=-1))
        projected_real, projected_imag = projected.chunk(2, dim=-1)
        return base[0] + projected_real, base[1] + projected_imag

    def transport_control(
        self, real: Tensor, imag: Tensor, full_anchors: int
    ) -> Tensor | None:
        if self.transport_selector is None:
            return None
        anchor_real = real[
            :, self.stride - 1 : full_anchors * self.stride : self.stride
        ]
        anchor_imag = imag[
            :, self.stride - 1 : full_anchors * self.stride : self.stride
        ]
        return self.transport_selector(anchor_real, anchor_imag)

    def build_transport_drive(
        self,
        real: Tensor,
        imag: Tensor,
        anchor_features: Tensor,
        full_anchors: int,
    ) -> ComplexField:
        if self.pole_specific_reader is not None:
            return self.pole_specific_reader(real, imag)
        drive_real, drive_imag = self.pole_drive(real, imag)
        anchors = (
            drive_real[:, self.stride - 1 : full_anchors * self.stride : self.stride],
            drive_imag[:, self.stride - 1 : full_anchors * self.stride : self.stride],
        )
        key_gate = self.key_gate(anchor_features)
        anchors = anchors[0] * key_gate, anchors[1] * key_gate
        if self.vector_width > 1:
            excitation_real = self.vector_excitation_axes(anchor_features)
            excitation_imag = self.vector_excitation_imag_axes(anchor_features)
            return (
                anchors[0].unsqueeze(-1) * excitation_real
                - anchors[1].unsqueeze(-1) * excitation_imag,
                anchors[0].unsqueeze(-1) * excitation_imag
                + anchors[1].unsqueeze(-1) * excitation_real,
            )
        value = self.anchor_value(anchor_features)
        if self.matrix_key_width <= 1:
            return (
                anchors[0].unsqueeze(-1) * value.unsqueeze(-2),
                anchors[1].unsqueeze(-1) * value.unsqueeze(-2),
            )
        matrix_key = self.matrix_key_axes(anchor_features)
        matrix_value = self.matrix_value_axes(anchor_features, value)
        return (
            anchors[0].unsqueeze(-1).unsqueeze(-1)
            * matrix_key.unsqueeze(-1)
            * matrix_value.unsqueeze(-3),
            anchors[1].unsqueeze(-1).unsqueeze(-1)
            * matrix_key.unsqueeze(-1)
            * matrix_value.unsqueeze(-3),
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        steps = real.shape[1]
        full_anchors = steps // self.stride
        if full_anchors == 0:
            return torch.zeros_like(real), torch.zeros_like(imag)
        packed = torch.cat((real, imag), dim=-1)
        anchor_features = packed[
            :, self.stride - 1 : full_anchors * self.stride : self.stride
        ]
        transported_drive = self.build_transport_drive(
            real, imag, anchor_features, full_anchors
        )
        if self.innovation_filter is not None:
            transported_drive = self.innovation_filter(*transported_drive)
        if self.write_scheduler is not None:
            write_gate = self.write_gate(anchor_features)
            for _ in range(transported_drive[0].ndim - write_gate.ndim):
                write_gate = write_gate.unsqueeze(-1)
            transported_drive = (
                transported_drive[0] * write_gate,
                transported_drive[1] * write_gate,
            )
        damping_control = self.transport_control(real, imag, full_anchors)
        clock_step = (
            self.semantic_clock(anchor_features)
            if self.semantic_clock is not None
            else None
        )
        state = (
            self.memory(
                *transported_drive,
                damping_control=damping_control,
                clock_step=clock_step,
            )
            if self.use_recurrence
            else transported_drive
        )
        zero = torch.zeros_like(state[0][:, :1]), torch.zeros_like(state[1][:, :1])
        delayed_state = (
            torch.cat((zero[0], state[0]), dim=1),
            torch.cat((zero[1], state[1]), dim=1),
        )
        if self.query_mode == "anchor":
            anchor_gate = self.query_gate(anchor_features)
            delayed_gate = torch.cat((torch.ones_like(anchor_gate[:, :1]), anchor_gate), dim=1)
            delayed_state = (
                delayed_state[0] * delayed_gate.unsqueeze(-1),
                delayed_state[1] * delayed_gate.unsqueeze(-1),
            )
        if self.query_mode == "token":
            repeated_state = (
                delayed_state[0].repeat_interleave(self.stride, dim=1)[:, :steps],
                delayed_state[1].repeat_interleave(self.stride, dim=1)[:, :steps],
            )
            token_gate = self.query_gate(torch.cat((real, imag), dim=-1))
            if self.vector_width > 1:
                query_real, query_imag = self.vector_query_components(
                    torch.cat((real, imag), dim=-1), token_gate
                )
                gated_real = repeated_state[0] * query_real + repeated_state[1] * query_imag
                gated_imag = repeated_state[1] * query_real - repeated_state[0] * query_imag
                if self.coordinate_synthesis is not None:
                    projected = self.coordinate_synthesis(
                        torch.cat(
                            (gated_real.flatten(-2), gated_imag.flatten(-2)), dim=-1
                        )
                    )
                    return projected.chunk(2, dim=-1)
                contracted_state = (
                    gated_real.sum(dim=-1),
                    gated_imag.sum(dim=-1),
                )
                return self.synthesize(contracted_state)
            if self.matrix_key_width > 1:
                matrix_query = self.matrix_query_axes(
                    torch.cat((real, imag), dim=-1), token_gate
                )
                repeated_state = (
                    (repeated_state[0] * matrix_query.unsqueeze(-1)).sum(dim=-2),
                    (repeated_state[1] * matrix_query.unsqueeze(-1)).sum(dim=-2),
                )
                return self.synthesize_vector(repeated_state)
            return self.synthesize_vector(
                (
                    repeated_state[0] * token_gate.unsqueeze(-1),
                    repeated_state[1] * token_gate.unsqueeze(-1),
                )
            )
        memory_real, memory_imag = self.synthesize_vector(delayed_state)
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
        parallel_static_scan: bool = False,
    ) -> None:
        super().__init__()
        self.modes = int(modes)
        self.context_length = int(context_length)
        self.scan_fp32 = bool(scan_fp32)
        self.parallel_static_scan = bool(parallel_static_scan)
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

    def _active_coefficients(
        self,
        drive_real: Tensor,
        *,
        scan_dtype: torch.dtype,
        vector_width: int,
        damping_control: Tensor | None,
        clock_step: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if damping_control is not None and clock_step is not None:
            raise ValueError("dynamic damping and semantic clock are mutually exclusive")
        if clock_step is not None:
            shared_shape = drive_real.shape[:2]
            pole_shape = (*shared_shape, self.modes)
            if clock_step.shape == shared_shape:
                active_clock = clock_step.unsqueeze(-1)
            elif clock_step.shape == pole_shape:
                active_clock = clock_step
            else:
                raise ValueError("semantic clock must match B,T or B,T,P pole axes")
            damping = self.damping().to(device=drive_real.device, dtype=scan_dtype)
            frequency = self.frequency().to(device=drive_real.device, dtype=scan_dtype)
            coefficients = discrete_pole_real2d(
                damping.view(1, 1, -1),
                frequency.view(1, 1, -1),
                active_clock.to(scan_dtype),
            )
        elif damping_control is None:
            coefficients = self.coefficients()
        else:
            if damping_control.shape != drive_real.shape:
                raise ValueError("dynamic damping control must match the pole drive")
            frequency = self.frequency().to(device=drive_real.device, dtype=scan_dtype)
            raw_damping = self.raw_damping.to(device=drive_real.device, dtype=scan_dtype)
            if vector_width > 1:
                frequency = frequency.repeat_interleave(vector_width)
                raw_damping = raw_damping.repeat_interleave(vector_width)
            coefficients = pole_gamma_from_control_real2d(
                raw_damping,
                frequency,
                damping_control.to(scan_dtype),
                self.minimum_damping,
                1.0,
            )
        dr, di, gr, gi = (
            value.to(device=drive_real.device, dtype=scan_dtype)
            for value in coefficients
        )
        if vector_width > 1 and damping_control is None:
            dr = dr.repeat_interleave(vector_width, dim=-1)
            di = di.repeat_interleave(vector_width, dim=-1)
            gr = gr.repeat_interleave(vector_width, dim=-1)
            gi = gi.repeat_interleave(vector_width, dim=-1)
        return dr, di, gr, gi

    def forward(
        self,
        drive_real: Tensor,
        drive_imag: Tensor,
        damping_control: Tensor | None = None,
        clock_step: Tensor | None = None,
    ) -> ComplexField:
        if drive_real.shape != drive_imag.shape:
            raise ValueError("fixed-pole memory expects matching complex coordinates")
        if self.parallel_static_scan and (
            damping_control is not None or clock_step is not None
        ):
            raise ValueError("parallel static scan does not accept dynamic pole controls")
        vector_width = 1
        output_shape = drive_real.shape
        if drive_real.ndim >= 4 and drive_real.shape[2] == self.modes:
            vector_width = math.prod(drive_real.shape[3:])
            if damping_control is not None:
                expected_control_shape = drive_real.shape[:3]
                if damping_control.shape != expected_control_shape:
                    raise ValueError(
                        "vector dynamic damping control must match B,T,P coordinates"
                    )
                damping_control = (
                    damping_control.unsqueeze(-1)
                    .expand(*expected_control_shape, vector_width)
                    .flatten(2)
                )
            drive_real = drive_real.flatten(2)
            drive_imag = drive_imag.flatten(2)
        elif drive_real.ndim != 3 or drive_real.shape[-1] != self.modes:
            raise ValueError("fixed-pole memory expects B,T,P or B,T,P,... coordinates")
        scan_dtype = torch.float32 if self.scan_fp32 else drive_real.dtype
        active_real, active_imag = drive_real.to(scan_dtype), drive_imag.to(scan_dtype)
        dr, di, gr, gi = self._active_coefficients(
            drive_real,
            scan_dtype=scan_dtype,
            vector_width=vector_width,
            damping_control=damping_control,
            clock_step=clock_step,
        )
        input_real = gr * active_real - gi * active_imag
        input_imag = gi * active_real + gr * active_imag
        shape = (1, 1, self.modes * vector_width)
        active_decay_real = dr.view(shape).expand_as(input_real) if dr.ndim == 1 else dr
        active_decay_imag = di.view(shape).expand_as(input_imag) if di.ndim == 1 else di
        if self.parallel_static_scan:
            if dr.ndim != 1 or di.ndim != 1:
                raise RuntimeError("parallel static scan requires mode-static decay")
            packed_states = chunked_parallel_static_recurrence_packed(
                dr,
                di,
                torch.cat((input_real, input_imag), dim=-1),
            )
            state_real, state_imag = packed_states.split(input_real.shape[-1], dim=-1)
        else:
            state_real, state_imag = pac_triton_recurrence_opaque_op(
                active_decay_real,
                active_decay_imag,
                input_real,
                input_imag,
            )
        state_real = state_real.to(drive_real.dtype).reshape(output_shape)
        state_imag = state_imag.to(drive_imag.dtype).reshape(output_shape)
        return state_real, state_imag


class LaplaceMambaBlock(nn.Module):
    """Mamba-shaped block with fixed complex Laplace transport."""

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.RMSNorm(config.model_width, eps=config.rms_epsilon)
        self.gate_width = config.inner_complex_width
        self.value_width = config.inner_real_width
        self.address_width = 2 * config.pole_modes * config.state_size
        projection_width = (
            self.gate_width + self.value_width + 2 * self.address_width
        )
        self.input_projection = nn.Linear(
            config.model_width,
            projection_width,
            bias=False,
        )
        conv_channels = self.value_width + 2 * self.address_width
        self.conv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size=config.conv_width,
            groups=conv_channels,
            bias=True,
        )
        self.memory = FixedComplexPoleMemory1D(
            config.pole_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
        )
        shape = (config.pole_modes, config.head_width)
        self.direct_scale = nn.Parameter(torch.ones(shape))
        self.output_norm_weight = nn.Parameter(torch.ones(shape))
        self.output_projection = nn.Linear(
            config.inner_real_width,
            config.model_width,
            bias=False,
        )
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        with torch.no_grad():
            self.output_projection.weight.div_(math.sqrt(config.layers))

    def _split_projection(
        self,
        projected: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        gate, active = projected.split(
            (self.gate_width, projected.shape[-1] - self.gate_width),
            dim=-1,
        )
        causal = functional.silu(
            self.conv(
                functional.pad(
                    active.transpose(1, 2),
                    (self.config.conv_width - 1, 0),
                )
            ).transpose(1, 2)
        )
        value, write, read = causal.split(
            (self.value_width, self.address_width, self.address_width),
            dim=-1,
        )
        return gate, value, write, read

    def _complex_axes(
        self,
        value: Tensor,
        write: Tensor,
        read: Tensor,
    ) -> tuple[ComplexField, ComplexField, ComplexField]:
        batch, steps, _width = value.shape
        value_shape = (
            batch,
            steps,
            self.config.pole_modes,
            self.config.head_width,
        )
        address_shape = (
            batch,
            steps,
            self.config.pole_modes,
            self.config.state_size,
        )
        value_real, value_imag = value.chunk(2, dim=-1)
        write_real, write_imag = write.chunk(2, dim=-1)
        read_real, read_imag = read.chunk(2, dim=-1)
        return (
            (value_real.reshape(value_shape), value_imag.reshape(value_shape)),
            (write_real.reshape(address_shape), write_imag.reshape(address_shape)),
            (read_real.reshape(address_shape), read_imag.reshape(address_shape)),
        )

    def _transport_and_read(
        self,
        value: ComplexField,
        write: ComplexField,
        read: ComplexField,
    ) -> ComplexField:
        drive_real = (
            write[0].unsqueeze(-1) * value[0].unsqueeze(-2)
            - write[1].unsqueeze(-1) * value[1].unsqueeze(-2)
        )
        drive_imag = (
            write[0].unsqueeze(-1) * value[1].unsqueeze(-2)
            + write[1].unsqueeze(-1) * value[0].unsqueeze(-2)
        )
        state_real, state_imag = self.memory(drive_real, drive_imag)
        output_real = (
            state_real * read[0].unsqueeze(-1)
            + state_imag * read[1].unsqueeze(-1)
        ).sum(dim=-2)
        output_imag = (
            state_imag * read[0].unsqueeze(-1)
            - state_real * read[1].unsqueeze(-1)
        ).sum(dim=-2)
        scale = self.direct_scale.to(output_real.dtype)
        return output_real + scale * value[0], output_imag + scale * value[1]

    def _normalize_and_gate(
        self,
        output: ComplexField,
        gate: Tensor,
    ) -> Tensor:
        energy = output[0].float().square().add(output[1].float().square()).mean(
            dim=(-2, -1),
            keepdim=True,
        )
        scale = torch.rsqrt(energy + self.config.rms_epsilon).to(output[0].dtype)
        weight = self.output_norm_weight.to(output[0].dtype)
        gate_shape = (*gate.shape[:-1], self.config.pole_modes, self.config.head_width)
        active_gate = functional.silu(gate.reshape(gate_shape))
        normalized_real = output[0] * scale * weight * active_gate
        normalized_imag = output[1] * scale * weight * active_gate
        return torch.cat(
            (normalized_real.flatten(-2), normalized_imag.flatten(-2)),
            dim=-1,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        projected = self.input_projection(self.input_norm(hidden))
        gate, value_packed, write_packed, read_packed = self._split_projection(projected)
        value, write, read = self._complex_axes(value_packed, write_packed, read_packed)
        memory = self._transport_and_read(value, write, read)
        update = self.output_projection(self._normalize_and_gate(memory, gate))
        return hidden + update


class ComplexDepthwiseCausalConv1d(nn.Module):
    """Strict-complex depthwise causal convolution evaluated as one grouped conv."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if channels <= 0 or kernel_size <= 0:
            raise ValueError("invalid complex causal convolution dimensions")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        shape = (self.channels, self.kernel_size)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        self.bias_real = nn.Parameter(torch.empty(self.channels))
        self.bias_imag = nn.Parameter(torch.empty(self.channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(2.0 * self.kernel_size)
        nn.init.uniform_(self.weight_real, -bound, bound)
        nn.init.uniform_(self.weight_imag, -bound, bound)
        nn.init.uniform_(self.bias_real, -bound, bound)
        nn.init.uniform_(self.bias_imag, -bound, bound)

    def packed_weight(self) -> Tensor:
        real_row = torch.stack((self.weight_real, -self.weight_imag), dim=1)
        imag_row = torch.stack((self.weight_imag, self.weight_real), dim=1)
        return torch.stack((real_row, imag_row), dim=1).reshape(
            2 * self.channels,
            2,
            self.kernel_size,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 3 or real.shape[-1] != self.channels:
            raise ValueError("complex causal convolution inputs have incompatible shapes")
        batch, steps, _channels = real.shape
        packed = torch.stack((real, imag), dim=-1).permute(0, 2, 3, 1).reshape(
            batch,
            2 * self.channels,
            steps,
        )
        packed = functional.pad(packed, (self.kernel_size - 1, 0))
        bias = torch.stack((self.bias_real, self.bias_imag), dim=-1).reshape(-1)
        output = functional.conv1d(
            packed,
            self.packed_weight(),
            bias,
            groups=self.channels,
        ).reshape(batch, self.channels, 2, steps)
        return output[:, :, 0].transpose(1, 2), output[:, :, 1].transpose(1, 2)


class ComplexHighwayLaplaceMambaBlock(nn.Module):
    """Fixed-Laplace matrix memory embedded in a persistent complex highway."""

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.value_width = config.inner_complex_width
        self.address_width = config.pole_modes * config.state_size
        self.active_width = self.value_width + 2 * self.address_width
        self.input_norm = ComplexRMSNorm(
            self.complex_width,
            epsilon=config.rms_epsilon,
        )
        self.analysis_projection = WidelyLinear(
            self.complex_width,
            self.active_width,
            bias=False,
        )
        self.gate_projection = nn.Linear(
            config.model_width,
            self.value_width,
            bias=False,
        )
        self.conv = ComplexDepthwiseCausalConv1d(
            self.active_width,
            config.conv_width,
        )
        self.memory = FixedComplexPoleMemory1D(
            config.pole_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
        )
        shape = (config.pole_modes, config.head_width)
        self.direct_scale = nn.Parameter(torch.ones(shape))
        self.output_norm_weight = nn.Parameter(torch.ones(shape))
        self.synthesis = WidelyLinear(
            self.value_width,
            self.complex_width,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        projection_width = self.value_width + 2 * self.active_width
        joint_weight = torch.empty(projection_width, self.config.model_width)
        nn.init.xavier_uniform_(joint_weight)
        with torch.no_grad():
            self.gate_projection.weight.copy_(joint_weight[: self.value_width])
        self.analysis_projection.load_real_affine(joint_weight[self.value_width :])

        synthesis_weight = torch.empty(
            self.config.model_width,
            2 * self.value_width,
        )
        nn.init.xavier_uniform_(synthesis_weight)
        synthesis_weight.div_(math.sqrt(self.config.layers))
        self.synthesis.load_real_affine(synthesis_weight)
        with torch.no_grad():
            self.input_norm.weight.fill_(math.sqrt(2.0))

    def _analyze(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, ComplexField, ComplexField, ComplexField]:
        normalized = self.input_norm(real, imag)
        gate = self.gate_projection(torch.cat(normalized, dim=-1))
        active_real, active_imag = self.analysis_projection(*normalized)
        active_real, active_imag = self.conv(active_real, active_imag)
        active_real = functional.silu(active_real)
        active_imag = functional.silu(active_imag)
        value_real, write_real, read_real = active_real.split(
            (self.value_width, self.address_width, self.address_width),
            dim=-1,
        )
        value_imag, write_imag, read_imag = active_imag.split(
            (self.value_width, self.address_width, self.address_width),
            dim=-1,
        )
        batch, steps, _width = value_real.shape
        value_shape = (batch, steps, self.config.pole_modes, self.config.head_width)
        address_shape = (batch, steps, self.config.pole_modes, self.config.state_size)
        return (
            gate,
            (value_real.reshape(value_shape), value_imag.reshape(value_shape)),
            (write_real.reshape(address_shape), write_imag.reshape(address_shape)),
            (read_real.reshape(address_shape), read_imag.reshape(address_shape)),
        )

    def _transport_and_read(
        self,
        value: ComplexField,
        write: ComplexField,
        read: ComplexField,
    ) -> ComplexField:
        drive_real = (
            write[0].unsqueeze(-1) * value[0].unsqueeze(-2)
            - write[1].unsqueeze(-1) * value[1].unsqueeze(-2)
        )
        drive_imag = (
            write[0].unsqueeze(-1) * value[1].unsqueeze(-2)
            + write[1].unsqueeze(-1) * value[0].unsqueeze(-2)
        )
        state_real, state_imag = self.memory(drive_real, drive_imag)
        output_real = (
            state_real * read[0].unsqueeze(-1)
            + state_imag * read[1].unsqueeze(-1)
        ).sum(dim=-2)
        output_imag = (
            state_imag * read[0].unsqueeze(-1)
            - state_real * read[1].unsqueeze(-1)
        ).sum(dim=-2)
        scale = self.direct_scale.to(output_real.dtype)
        return output_real + scale * value[0], output_imag + scale * value[1]

    def _normalize_and_gate(
        self,
        output: ComplexField,
        gate: Tensor,
    ) -> ComplexField:
        energy = output[0].float().square().add(output[1].float().square()).mean(
            dim=(-2, -1),
            keepdim=True,
        )
        scale = torch.rsqrt(energy + self.config.rms_epsilon).to(output[0].dtype)
        weight = self.output_norm_weight.to(output[0].dtype)
        gate_shape = (*gate.shape[:-1], self.config.pole_modes, self.config.head_width)
        active_gate = functional.silu(gate.reshape(gate_shape))
        return (
            output[0] * scale * weight * active_gate,
            output[1] * scale * weight * active_gate,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        gate, value, write, read = self._analyze(real, imag)
        memory = self._transport_and_read(value, write, read)
        normalized = self._normalize_and_gate(memory, gate)
        update_real, update_imag = self.synthesis(
            normalized[0].flatten(-2),
            normalized[1].flatten(-2),
        )
        return real + update_real, imag + update_imag


class ComplexHighwayLaplaceMambaLM(nn.Module):
    """Language model whose embedding, residual stream, and tied head stay complex."""

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.embedding_real = nn.Embedding(config.vocab_size, self.complex_width)
        self.embedding_imag = nn.Embedding(config.vocab_size, self.complex_width)
        self.blocks = nn.ModuleList(
            ComplexHighwayLaplaceMambaBlock(config) for _ in range(config.layers)
        )
        self.final_norm = ComplexRMSNorm(
            self.complex_width,
            epsilon=config.rms_epsilon,
        )
        nn.init.normal_(self.embedding_real.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embedding_imag.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.final_norm.weight.fill_(math.sqrt(2.0))

    def complex_hidden(self, input_ids: Tensor) -> ComplexField:
        real = self.embedding_real(input_ids)
        imag = self.embedding_imag(input_ids)
        for block in self.blocks:
            if (
                self.config.activation_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                result = activation_checkpoint(
                    block,
                    real,
                    imag,
                    use_reentrant=False,
                )
                real, imag = cast("ComplexField", result)
            else:
                real, imag = block(real, imag)
        return self.final_norm(real, imag)

    def hidden(self, input_ids: Tensor) -> Tensor:
        return torch.cat(self.complex_hidden(input_ids), dim=-1)

    def forward(self, input_ids: Tensor) -> Tensor:
        real, imag = self.complex_hidden(input_ids)
        return functional.linear(real, self.embedding_real.weight) + functional.linear(
            imag,
            self.embedding_imag.weight,
        )


@torch.no_grad()
def _scale_image_postfusion_output_(
    post_fusion: GatedComplexPostFusion,
    layers: int,
) -> None:
    output = cast("WidelyLinear", post_fusion.out)
    residual_scale = math.sqrt(layers)
    for parameter in (
        output.weight_real,
        output.weight_imag,
        output.conjugate_real,
        output.conjugate_imag,
    ):
        parameter.div_(residual_scale)


def _rms_matched_complex_residual(
    hidden: ComplexField,
    memory: ComplexField,
    gain: Tensor | float,
    epsilon: float,
) -> ComplexField:
    hidden_energy = hidden[0].float().square().add(hidden[1].float().square()).mean(
        dim=-1,
        keepdim=True,
    )
    memory_energy = memory[0].float().square().add(memory[1].float().square()).mean(
        dim=-1,
        keepdim=True,
    )
    rms_ratio = torch.sqrt(hidden_energy + epsilon) * torch.rsqrt(memory_energy + epsilon)
    active_gain = torch.as_tensor(gain, device=hidden[0].device, dtype=hidden[0].dtype)
    scale = active_gain * rms_ratio.detach().to(dtype=hidden[0].dtype)
    return hidden[0] + scale * memory[0], hidden[1] + scale * memory[1]


class ImagePostFusionAlphabet2Block(nn.Module):
    """Image-ALPHABET block with content-addressable fixed-Laplace memory."""

    memory_scale_initial = 0.01

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.value_width = config.inner_complex_width
        self.address_width = config.pole_modes * config.state_size
        self.active_width = self.value_width + 2 * self.address_width
        self.input_norm = ComplexRMSNorm(
            self.complex_width,
            epsilon=config.rms_epsilon,
        )
        self.analysis = PackedComplexLinear(
            self.complex_width,
            self.active_width,
        )
        self.reader = ComplexDepthwiseCausalConv1d(
            self.active_width,
            config.conv_width,
        )
        self.memory = FixedComplexPoleMemory1D(
            config.pole_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
        )
        self.synthesis = PackedComplexLinear(
            self.value_width,
            self.complex_width,
        )
        self.memory_scale = nn.Parameter(torch.tensor(self.memory_scale_initial))
        self.post_fusion = GatedComplexPostFusion(
            self.complex_width,
            self.complex_width,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.input_norm.weight.fill_(math.sqrt(2.0))
        _scale_image_postfusion_output_(self.post_fusion, self.config.layers)

    def _analyze(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[ComplexField, ComplexField, ComplexField]:
        normalized = self.input_norm(real, imag)
        active_real, active_imag = self.analysis(*normalized)
        active_real, active_imag = self.reader(active_real, active_imag)
        value_real, write_real, read_real = active_real.split(
            (self.value_width, self.address_width, self.address_width),
            dim=-1,
        )
        value_imag, write_imag, read_imag = active_imag.split(
            (self.value_width, self.address_width, self.address_width),
            dim=-1,
        )
        batch, steps, _width = value_real.shape
        value_shape = (batch, steps, self.config.pole_modes, self.config.head_width)
        address_shape = (
            batch,
            steps,
            self.config.pole_modes,
            self.config.state_size,
        )
        value = value_real.reshape(value_shape), value_imag.reshape(value_shape)
        write = complex_rms_unit(
            write_real.reshape(address_shape),
            write_imag.reshape(address_shape),
            self.config.rms_epsilon,
        )
        read = complex_rms_unit(
            read_real.reshape(address_shape),
            read_imag.reshape(address_shape),
            self.config.rms_epsilon,
        )
        return value, write, read

    def _transport_and_read(
        self,
        value: ComplexField,
        write: ComplexField,
        read: ComplexField,
    ) -> ComplexField:
        drive_real = (
            write[0].unsqueeze(-1) * value[0].unsqueeze(-2)
            - write[1].unsqueeze(-1) * value[1].unsqueeze(-2)
        )
        drive_imag = (
            write[0].unsqueeze(-1) * value[1].unsqueeze(-2)
            + write[1].unsqueeze(-1) * value[0].unsqueeze(-2)
        )
        state_real, state_imag = self.memory(drive_real, drive_imag)
        return (
            (
                state_real * read[0].unsqueeze(-1)
                + state_imag * read[1].unsqueeze(-1)
            ).sum(dim=-2),
            (
                state_imag * read[0].unsqueeze(-1)
                - state_real * read[1].unsqueeze(-1)
            ).sum(dim=-2),
        )

    def _merge_memory(
        self,
        hidden: ComplexField,
        memory: ComplexField,
    ) -> ComplexField:
        return _rms_matched_complex_residual(
            hidden,
            memory,
            self.memory_scale,
            self.config.rms_epsilon,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        value, write, read = self._analyze(real, imag)
        retrieved = self._transport_and_read(value, write, read)
        memory = self.synthesis(
            retrieved[0].flatten(-2),
            retrieved[1].flatten(-2),
        )
        merged = self._merge_memory((real, imag), memory)
        return self.post_fusion(*merged)


class ImagePostFusionAlphabet2LM(nn.Module):
    """Complex LM repeating the image-ALPHABET analysis/memory/fusion block."""

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.embedding_real = nn.Embedding(config.vocab_size, self.complex_width)
        self.embedding_imag = nn.Embedding(config.vocab_size, self.complex_width)
        self.blocks = nn.ModuleList(
            ImagePostFusionAlphabet2Block(config) for _ in range(config.layers)
        )
        self.final_norm = ComplexRMSNorm(
            self.complex_width,
            epsilon=config.rms_epsilon,
        )
        nn.init.normal_(self.embedding_real.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embedding_imag.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.final_norm.weight.fill_(math.sqrt(2.0))

    def complex_hidden(self, input_ids: Tensor) -> ComplexField:
        real = self.embedding_real(input_ids)
        imag = self.embedding_imag(input_ids)
        for block in self.blocks:
            if (
                self.config.activation_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                result = activation_checkpoint(
                    block,
                    real,
                    imag,
                    use_reentrant=False,
                )
                real, imag = cast("ComplexField", result)
            else:
                real, imag = block(real, imag)
        return self.final_norm(real, imag)

    def hidden(self, input_ids: Tensor) -> Tensor:
        return torch.cat(self.complex_hidden(input_ids), dim=-1)

    def forward(self, input_ids: Tensor) -> Tensor:
        real, imag = self.complex_hidden(input_ids)
        return functional.linear(real, self.embedding_real.weight) + functional.linear(
            imag,
            self.embedding_imag.weight,
        )


class VectorImagePostFusionAlphabet2Block(nn.Module):
    """Image-ALPHABET block with a selectively read vector Laplace state."""

    memory_scale_initial = 0.01

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.value_width = config.inner_complex_width
        self.active_width = 2 * self.value_width
        self.reader = DenseComplexConv1dReader(
            self.complex_width,
            self.active_width,
            kernel_size=config.conv_width,
        )
        self.memory = FixedComplexPoleMemory1D(
            config.pole_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
        )
        self.synthesis = PackedComplexLinear(
            self.value_width,
            self.complex_width,
        )
        self.memory_scale = nn.Parameter(torch.tensor(self.memory_scale_initial))
        self.post_fusion = GatedComplexPostFusion(
            self.complex_width,
            self.complex_width,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.reader.input_norm.weight.fill_(math.sqrt(2.0))
        _scale_image_postfusion_output_(self.post_fusion, self.config.layers)

    def _analyze(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[ComplexField, ComplexField]:
        active_real, active_imag = self.reader(real, imag)
        excitation_real, query_real = active_real.split(self.value_width, dim=-1)
        excitation_imag, query_imag = active_imag.split(self.value_width, dim=-1)
        batch, steps, _width = excitation_real.shape
        shape = (batch, steps, self.config.pole_modes, self.config.head_width)
        excitation = excitation_real.reshape(shape), excitation_imag.reshape(shape)
        query = query_real.reshape(shape), query_imag.reshape(shape)
        return excitation, query

    def _transport_and_read(
        self,
        excitation: ComplexField,
        query: ComplexField,
    ) -> ComplexField:
        state_real, state_imag = self.memory(*excitation)
        return (
            query[0] * state_real + query[1] * state_imag,
            query[0] * state_imag - query[1] * state_real,
        )

    def _merge_memory(
        self,
        hidden: ComplexField,
        memory: ComplexField,
    ) -> ComplexField:
        return _rms_matched_complex_residual(
            hidden,
            memory,
            self.memory_scale,
            self.config.rms_epsilon,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        excitation, query = self._analyze(real, imag)
        selected = self._transport_and_read(excitation, query)
        memory = self.synthesis(
            selected[0].flatten(-2),
            selected[1].flatten(-2),
        )
        merged = self._merge_memory((real, imag), memory)
        return self.post_fusion(*merged)


class ContentPreservingImagePostFusionAlphabet2Block(nn.Module):
    """Keep content intact while every coordinate owns its write/read/poles."""

    memory_scale_initial = 0.01

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.heads = config.content_preserving_heads
        self.poles_per_head = config.content_preserving_poles_per_head
        self.width_per_head = config.content_preserving_width_per_head
        self.total_poles = self.heads * self.poles_per_head
        self.content_width = self.heads * self.width_per_head
        self.state_modes = self.content_width * self.poles_per_head
        if self.total_poles != config.pole_modes:
            raise ValueError(
                "content-preserving heads x poles must match configured pole modes"
            )
        if self.content_width != self.complex_width:
            raise ValueError(
                "projection-free content width must match the complex highway width"
            )
        self.feature_reader = DenseComplexConv1dReader(
            self.complex_width,
            self.content_width,
            kernel_size=config.conv_width,
        )
        self.write_router = nn.Linear(2 * self.content_width, self.total_poles)
        self.read_router = PackedComplexLinear(self.content_width, self.total_poles)
        self.memory = FixedComplexPoleMemory1D(
            self.state_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
            banks=self.content_width,
            parallel_static_scan=config.parallel_static_scan,
        )
        self.synthesis = PackedComplexLinear(self.content_width, self.complex_width)
        self.memory_scale = nn.Parameter(torch.tensor(self.memory_scale_initial))
        self.post_fusion = GatedComplexPostFusion(
            self.complex_width,
            self.complex_width,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.feature_reader.input_norm.weight.fill_(math.sqrt(2.0))
        _scale_image_postfusion_output_(self.post_fusion, self.config.layers)

    def _analyze(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[ComplexField, Tensor, ComplexField]:
        content_real, content_imag = self.feature_reader(real, imag)
        packed_content = torch.cat((content_real, content_imag), dim=-1)
        write = self.write_router(packed_content)
        read_real, read_imag = self.read_router(content_real, content_imag)
        batch, steps, _width = content_real.shape
        content_shape = (batch, steps, self.heads, self.width_per_head)
        route_shape = (batch, steps, self.heads, self.poles_per_head)
        return (
            (content_real.reshape(content_shape), content_imag.reshape(content_shape)),
            write.reshape(route_shape),
            (read_real.reshape(route_shape), read_imag.reshape(route_shape)),
        )

    def _transport_and_read(
        self,
        content: ComplexField,
        write: Tensor,
        read: ComplexField,
    ) -> ComplexField:
        drive_real = (write.unsqueeze(-2) * content[0].unsqueeze(-1)).flatten(2)
        drive_imag = (write.unsqueeze(-2) * content[1].unsqueeze(-1)).flatten(2)
        state_real, state_imag = self.memory(drive_real, drive_imag)
        state_shape = (
            *state_real.shape[:2],
            self.heads,
            self.width_per_head,
            self.poles_per_head,
        )
        state_real = state_real.reshape(state_shape)
        state_imag = state_imag.reshape(state_shape)
        selected_real = (
            read[0].unsqueeze(-2) * state_real
            + read[1].unsqueeze(-2) * state_imag
        ).sum(dim=-1)
        selected_imag = (
            read[0].unsqueeze(-2) * state_imag
            - read[1].unsqueeze(-2) * state_real
        ).sum(dim=-1)
        return selected_real.flatten(-2), selected_imag.flatten(-2)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        content, write, read = self._analyze(real, imag)
        selected = self._transport_and_read(content, write, read)
        memory = self.synthesis(*selected)
        merged = _rms_matched_complex_residual(
            (real, imag), memory, self.memory_scale, self.config.rms_epsilon
        )
        return self.post_fusion(*merged)


class HybridContentDenseImagePostFusionAlphabet2Block(nn.Module):
    """Fuse a content carrier and compact pole-specific Dense memory once."""

    memory_scale_initial = 0.01

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.heads = config.content_preserving_heads
        self.content_poles = config.content_preserving_poles_per_head
        self.width_per_head = config.content_preserving_width_per_head
        self.content_width = self.heads * self.width_per_head
        self.content_state_modes = self.content_width * self.content_poles
        self.dense_poles = config.hybrid_dense_poles
        self.dense_width = config.hybrid_dense_width
        self.dense_state_modes = self.dense_poles * self.dense_width
        if self.content_width != self.complex_width:
            raise ValueError("hybrid content width must match the complex highway")

        self.content_reader = DenseComplexConv1dReader(
            self.complex_width,
            self.content_width,
            kernel_size=config.conv_width,
        )
        content_routes = self.heads * self.content_poles
        self.content_write = nn.Linear(2 * self.content_width, content_routes)
        self.content_read = PackedComplexLinear(self.content_width, content_routes)
        self.content_memory = FixedComplexPoleMemory1D(
            self.content_state_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
            banks=self.content_width,
        )
        self.content_synthesis = PackedComplexLinear(
            self.content_width, self.complex_width
        )

        self.dense_reader = DenseComplexConv1dReader(
            self.complex_width,
            2 * self.dense_state_modes,
            kernel_size=config.conv_width,
        )
        self.dense_memory = FixedComplexPoleMemory1D(
            self.dense_poles,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
        )
        self.dense_synthesis = PackedComplexLinear(
            self.dense_state_modes, self.complex_width
        )

        self.content_scale = nn.Parameter(torch.tensor(self.memory_scale_initial))
        self.dense_scale = nn.Parameter(torch.tensor(self.memory_scale_initial))
        self.post_fusion = GatedComplexPostFusion(
            self.complex_width, self.complex_width
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.content_reader.input_norm.weight.fill_(math.sqrt(2.0))
            self.dense_reader.input_norm.weight.fill_(math.sqrt(2.0))
        _scale_image_postfusion_output_(self.post_fusion, self.config.layers)

    def _content_branch(self, real: Tensor, imag: Tensor) -> ComplexField:
        content_real, content_imag = self.content_reader(real, imag)
        packed = torch.cat((content_real, content_imag), dim=-1)
        write = self.content_write(packed)
        read_real, read_imag = self.content_read(content_real, content_imag)
        batch, steps, _width = content_real.shape
        content_shape = (batch, steps, self.heads, self.width_per_head)
        route_shape = (batch, steps, self.heads, self.content_poles)
        content_real = content_real.reshape(content_shape)
        content_imag = content_imag.reshape(content_shape)
        write = write.reshape(route_shape)
        read_real = read_real.reshape(route_shape)
        read_imag = read_imag.reshape(route_shape)
        drive_real = (write.unsqueeze(-2) * content_real.unsqueeze(-1)).flatten(2)
        drive_imag = (write.unsqueeze(-2) * content_imag.unsqueeze(-1)).flatten(2)
        state_real, state_imag = self.content_memory(drive_real, drive_imag)
        state_shape = (
            batch,
            steps,
            self.heads,
            self.width_per_head,
            self.content_poles,
        )
        state_real = state_real.reshape(state_shape)
        state_imag = state_imag.reshape(state_shape)
        selected_real = (
            read_real.unsqueeze(-2) * state_real
            + read_imag.unsqueeze(-2) * state_imag
        ).sum(dim=-1)
        selected_imag = (
            read_real.unsqueeze(-2) * state_imag
            - read_imag.unsqueeze(-2) * state_real
        ).sum(dim=-1)
        return self.content_synthesis(
            selected_real.flatten(-2), selected_imag.flatten(-2)
        )

    def _dense_branch(self, real: Tensor, imag: Tensor) -> ComplexField:
        active_real, active_imag = self.dense_reader(real, imag)
        excitation_real, query_real = active_real.split(self.dense_state_modes, dim=-1)
        excitation_imag, query_imag = active_imag.split(self.dense_state_modes, dim=-1)
        batch, steps, _width = excitation_real.shape
        state_shape = (batch, steps, self.dense_poles, self.dense_width)
        state_real, state_imag = self.dense_memory(
            excitation_real.reshape(state_shape), excitation_imag.reshape(state_shape)
        )
        query_real = query_real.reshape(state_shape)
        query_imag = query_imag.reshape(state_shape)
        selected_real = query_real * state_real + query_imag * state_imag
        selected_imag = query_real * state_imag - query_imag * state_real
        return self.dense_synthesis(selected_real.flatten(-2), selected_imag.flatten(-2))

    def _merge(
        self,
        hidden: ComplexField,
        content_memory: ComplexField,
        dense_memory: ComplexField,
    ) -> ComplexField:
        hidden_energy = hidden[0].float().square().add(hidden[1].float().square()).mean(
            dim=-1, keepdim=True
        )

        def matched(memory: ComplexField, gain: Tensor) -> ComplexField:
            energy = memory[0].float().square().add(memory[1].float().square()).mean(
                dim=-1, keepdim=True
            )
            ratio = torch.sqrt(hidden_energy + self.config.rms_epsilon) * torch.rsqrt(
                energy + self.config.rms_epsilon
            )
            scale = gain.to(hidden[0].dtype) * ratio.detach().to(hidden[0].dtype)
            return scale * memory[0], scale * memory[1]

        content_update = matched(content_memory, self.content_scale)
        dense_update = matched(dense_memory, self.dense_scale)
        return (
            hidden[0] + content_update[0] + dense_update[0],
            hidden[1] + content_update[1] + dense_update[1],
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        content_memory = self._content_branch(real, imag)
        dense_memory = self._dense_branch(real, imag)
        return self.post_fusion(*self._merge((real, imag), content_memory, dense_memory))


class ContentAlignedImagePostFusionAlphabet2Block(nn.Module):
    """Transport one shared semantic basis through pole-specific temporal filters."""

    memory_scale_initial = 0.01

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.content_width = config.head_width
        self.content_rank = config.aligned_content_rank
        self.value_width = config.inner_complex_width
        self.input_norm = ComplexRMSNorm(
            self.complex_width,
            epsilon=config.rms_epsilon,
        )
        self.content_analysis = PackedComplexLinear(
            self.complex_width,
            self.content_rank * self.content_width,
        )
        self.excitation_reader = (
            PoleAlignedComplexCausalConv1d(
                config.pole_modes,
                self.content_width,
                kernel_size=config.conv_width,
            )
            if self.content_rank == 1
            else LowRankPoleAlignedComplexCausalConv1d(
                config.pole_modes,
                self.content_width,
                self.content_rank,
                kernel_size=config.conv_width,
            )
        )
        self.query_reader = StrictComplexCausalConv1d(
            self.complex_width,
            self.value_width,
            kernel_size=config.conv_width,
        )
        self.memory = FixedComplexPoleMemory1D(
            config.pole_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
        )
        self.synthesis = PackedComplexLinear(
            self.value_width,
            self.complex_width,
        )
        self.memory_scale = nn.Parameter(torch.tensor(self.memory_scale_initial))
        self.post_fusion = GatedComplexPostFusion(
            self.complex_width,
            self.complex_width,
        )
        with torch.no_grad():
            self.input_norm.weight.fill_(math.sqrt(2.0))
        _scale_image_postfusion_output_(self.post_fusion, config.layers)

    def _analyze(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[ComplexField, ComplexField]:
        normalized = self.input_norm(real, imag)
        content = self.content_analysis(*normalized)
        excitation = self.excitation_reader(*content)
        query_real, query_imag = self.query_reader(*normalized)
        shape = (
            query_real.shape[0],
            query_real.shape[1],
            self.config.pole_modes,
            self.config.head_width,
        )
        query = query_real.reshape(shape), query_imag.reshape(shape)
        return excitation, query

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        excitation, query = self._analyze(real, imag)
        state_real, state_imag = self.memory(*excitation)
        selected = (
            query[0] * state_real + query[1] * state_imag,
            query[0] * state_imag - query[1] * state_real,
        )
        memory = self.synthesis(
            selected[0].flatten(-2),
            selected[1].flatten(-2),
        )
        merged = _rms_matched_complex_residual(
            (real, imag),
            memory,
            self.memory_scale,
            self.config.rms_epsilon,
        )
        return self.post_fusion(*merged)


class ComplexMultiObserverRead(nn.Module):
    """Read each VectorPole through several learned complex bilinear forms."""

    def __init__(self, poles: int, observers: int, vector_width: int) -> None:
        super().__init__()
        if min(poles, observers, vector_width) <= 0:
            raise ValueError("invalid complex multi-observer dimensions")
        self.poles = int(poles)
        self.observers = int(observers)
        self.vector_width = int(vector_width)
        shape = (self.poles, self.observers, self.vector_width, self.vector_width)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        standard_deviation = 1.0 / math.sqrt(2.0 * self.vector_width)
        nn.init.normal_(self.weight_real, std=standard_deviation)
        nn.init.normal_(self.weight_imag, std=standard_deviation)

    def forward(self, query: ComplexField, state: ComplexField) -> ComplexField:
        if (
            query[0].shape != query[1].shape
            or state[0].shape != state[1].shape
            or query[0].shape != state[0].shape
            or query[0].shape[-2:] != (self.poles, self.vector_width)
        ):
            raise ValueError("complex multi-observer inputs are incompatible")
        transformed_real = torch.einsum(
            "pjrs,btps->btpjr",
            self.weight_real,
            state[0],
        ) - torch.einsum("pjrs,btps->btpjr", self.weight_imag, state[1])
        transformed_imag = torch.einsum(
            "pjrs,btps->btpjr",
            self.weight_real,
            state[1],
        ) + torch.einsum("pjrs,btps->btpjr", self.weight_imag, state[0])
        return (
            (
                query[0].unsqueeze(-2) * transformed_real
                + query[1].unsqueeze(-2) * transformed_imag
            ).sum(dim=-1),
            (
                query[0].unsqueeze(-2) * transformed_imag
                - query[1].unsqueeze(-2) * transformed_real
            ).sum(dim=-1),
        )


class PoleWiseComplexReadAdapter(nn.Module):
    """Learn a strict-complex read basis independently at every pole."""

    def __init__(self, poles: int, vector_width: int) -> None:
        super().__init__()
        if min(poles, vector_width) <= 0:
            raise ValueError("invalid pole-wise read adapter dimensions")
        self.poles = int(poles)
        self.vector_width = int(vector_width)
        shape = (self.poles, self.vector_width, self.vector_width)
        self.weight_real = nn.Parameter(torch.zeros(shape))
        self.weight_imag = nn.Parameter(torch.zeros(shape))
        with torch.no_grad():
            identity = torch.eye(self.vector_width).expand(self.poles, -1, -1)
            self.weight_real.copy_(identity)

    def forward(self, state: ComplexField) -> ComplexField:
        if (
            state[0].shape != state[1].shape
            or state[0].shape[-2:] != (self.poles, self.vector_width)
        ):
            raise ValueError("pole-wise read adapter inputs are incompatible")
        return (
            torch.einsum("prs,btps->btpr", self.weight_real, state[0])
            - torch.einsum("prs,btps->btpr", self.weight_imag, state[1]),
            torch.einsum("prs,btps->btpr", self.weight_real, state[1])
            + torch.einsum("prs,btps->btpr", self.weight_imag, state[0]),
        )


class PoleAxisTemporalWhitening(nn.Module):
    """Recondition fixed exponential histories along the temporal pole axis."""

    def __init__(
        self,
        memory: FixedComplexPoleMemory1D,
        *,
        horizon: int = 1_536,
        relative_epsilon: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if horizon <= 0 or relative_epsilon <= 0.0:
            raise ValueError("invalid temporal whitening configuration")
        self.poles = memory.modes
        self.horizon = int(horizon)
        self.relative_epsilon = float(relative_epsilon)
        with torch.no_grad():
            damping = memory.damping().double()
            frequency = memory.frequency().double()
            continuous = torch.complex(-damping, frequency)
            decay = torch.exp(continuous)
            injection = (decay - 1.0) / continuous
            lag = torch.arange(self.horizon, dtype=torch.float64)
            kernels = injection[:, None] * decay[:, None].pow(lag[None, :])
            gram = kernels @ kernels.conj().transpose(0, 1)
            regularization = self.relative_epsilon * gram.diagonal().real.mean()
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            inverse_root = torch.rsqrt(eigenvalues.clamp_min(0.0) + regularization)
            transform = (eigenvectors * inverse_root.unsqueeze(0)) @ eigenvectors.conj().T
            whitened_gram = transform @ gram @ transform.conj().T
            output_scale = torch.sqrt(
                torch.tensor(float(self.poles), dtype=torch.float64)
                / whitened_gram.diagonal().real.sum().clamp_min(1.0e-30)
            )
            transform = transform * output_scale
        self.weight_real = nn.Parameter(transform.real.float())
        self.weight_imag = nn.Parameter(transform.imag.float())
        self.register_buffer("initial_gram_eigenvalues", eigenvalues.real.float())
        self.register_buffer("initial_regularization", regularization.float())

    def forward(self, state: ComplexField) -> ComplexField:
        if state[0].shape != state[1].shape or state[0].shape[-2] != self.poles:
            raise ValueError("pole-axis temporal whitening inputs are incompatible")
        return (
            torch.einsum("qp,btpr->btqr", self.weight_real, state[0])
            - torch.einsum("qp,btpr->btqr", self.weight_imag, state[1]),
            torch.einsum("qp,btpr->btqr", self.weight_real, state[1])
            + torch.einsum("qp,btpr->btqr", self.weight_imag, state[0]),
        )


class PoleWiseDynamicDelta(nn.Module):
    """Advance each fixed Laplace mode on a content-conditioned positive clock."""

    def __init__(
        self,
        modes: int,
        poles: int,
        *,
        hidden: int,
        log_bound: float,
    ) -> None:
        super().__init__()
        if min(modes, poles, hidden) <= 0 or log_bound <= 0.0:
            raise ValueError("invalid pole-wise dynamic delta configuration")
        self.poles = int(poles)
        self.log_bound = float(log_bound)
        self.norm = ComplexRMSNorm(modes)
        self.input = nn.Linear(2 * modes, hidden, bias=False)
        self.output = nn.Linear(hidden, self.poles, bias=False)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, real: Tensor, imag: Tensor) -> Tensor:
        unit_real, unit_imag = self.norm(real, imag)
        hidden = functional.silu(self.input(torch.cat((unit_real, unit_imag), dim=-1)))
        log_delta = self.log_bound * torch.tanh(self.output(hidden))
        return torch.exp(log_delta)


class MultiObserverImagePostFusionAlphabet2Block(nn.Module):
    """Keep dense writes and observe VectorPoles with full bilinear forms."""

    memory_scale_initial = 0.01

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.value_width = config.inner_complex_width
        self.reader = DenseComplexConv1dReader(
            self.complex_width,
            2 * self.value_width,
            kernel_size=config.conv_width,
        )
        self.memory = FixedComplexPoleMemory1D(
            config.pole_modes,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=config.minimum_half_life,
            maximum_half_life=config.maximum_half_life,
        )
        self.observer = ComplexMultiObserverRead(
            config.pole_modes,
            config.observer_count,
            config.head_width,
        )
        self.synthesis = PackedComplexLinear(
            config.pole_modes * config.observer_count,
            self.complex_width,
        )
        self.memory_scale = nn.Parameter(torch.tensor(self.memory_scale_initial))
        self.post_fusion = GatedComplexPostFusion(
            self.complex_width,
            self.complex_width,
        )
        _scale_image_postfusion_output_(self.post_fusion, config.layers)

    def _analyze(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[ComplexField, ComplexField]:
        active_real, active_imag = self.reader(real, imag)
        excitation_real, query_real = active_real.split(self.value_width, dim=-1)
        excitation_imag, query_imag = active_imag.split(self.value_width, dim=-1)
        shape = (
            active_real.shape[0],
            active_real.shape[1],
            self.config.pole_modes,
            self.config.head_width,
        )
        return (
            (excitation_real.reshape(shape), excitation_imag.reshape(shape)),
            (query_real.reshape(shape), query_imag.reshape(shape)),
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        excitation, query = self._analyze(real, imag)
        state = self.memory(*excitation)
        observed = self.observer(query, state)
        memory = self.synthesis(
            observed[0].flatten(-2),
            observed[1].flatten(-2),
        )
        merged = _rms_matched_complex_residual(
            (real, imag),
            memory,
            self.memory_scale,
            self.config.rms_epsilon,
        )
        return self.post_fusion(*merged)


class ReadAdaptedImagePostFusionAlphabet2Block(VectorImagePostFusionAlphabet2Block):
    """Keep Dense VectorPole intact and align only its read coordinate system."""

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__(config)
        self.read_adapter = PoleWiseComplexReadAdapter(
            config.pole_modes,
            config.head_width,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        excitation, query = self._analyze(real, imag)
        state = self.memory(*excitation)
        adapted_state = self.read_adapter(state)
        selected = (
            query[0] * adapted_state[0] + query[1] * adapted_state[1],
            query[0] * adapted_state[1] - query[1] * adapted_state[0],
        )
        memory = self.synthesis(
            selected[0].flatten(-2),
            selected[1].flatten(-2),
        )
        merged = self._merge_memory((real, imag), memory)
        return self.post_fusion(*merged)


class TemporallyWhitenedImagePostFusionAlphabet2Block(
    ContentAlignedImagePostFusionAlphabet2Block
):
    """Keep J2 transport diagonal and expose a conditioned pole observation basis."""

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        if config.aligned_content_rank != 2:
            raise ValueError("temporally whitened ALPHABET requires content rank 2")
        super().__init__(config)
        self.temporal_whitening = PoleAxisTemporalWhitening(self.memory)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        excitation, query = self._analyze(real, imag)
        state = self.memory(*excitation)
        observed_state = self.temporal_whitening(state)
        selected = (
            query[0] * observed_state[0] + query[1] * observed_state[1],
            query[0] * observed_state[1] - query[1] * observed_state[0],
        )
        memory = self.synthesis(
            selected[0].flatten(-2),
            selected[1].flatten(-2),
        )
        merged = _rms_matched_complex_residual(
            (real, imag),
            memory,
            self.memory_scale,
            self.config.rms_epsilon,
        )
        return self.post_fusion(*merged)


class DynamicDeltaImagePostFusionAlphabet2Block(VectorImagePostFusionAlphabet2Block):
    """Use dense writes and reads with exact content-conditioned Laplace time."""

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__(config)
        # The zero-output controller must not perturb the established Dense
        # initialization sequence merely by being constructed.
        with torch.random.fork_rng(devices=[]):
            self.delta = PoleWiseDynamicDelta(
                self.complex_width,
                config.pole_modes,
                hidden=config.dynamic_delta_hidden,
                log_bound=config.dynamic_delta_log_bound,
            )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        excitation, query = self._analyze(real, imag)
        clock_step = self.delta(real, imag)
        state = self.memory(*excitation, clock_step=clock_step)
        selected = (
            query[0] * state[0] + query[1] * state[1],
            query[0] * state[1] - query[1] * state[0],
        )
        memory = self.synthesis(
            selected[0].flatten(-2),
            selected[1].flatten(-2),
        )
        merged = self._merge_memory((real, imag), memory)
        return self.post_fusion(*merged)


class VectorImagePostFusionAlphabet2LM(nn.Module):
    """Complex LM with image-ALPHABET fusion around selective VectorPoles."""

    block_type: type[nn.Module] = VectorImagePostFusionAlphabet2Block

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.complex_width = config.model_width // 2
        self.embedding_real = nn.Embedding(config.vocab_size, self.complex_width)
        self.embedding_imag = nn.Embedding(config.vocab_size, self.complex_width)
        self.blocks = nn.ModuleList(
            self.block_type(config) for _ in range(config.layers)
        )
        self.final_norm = ComplexRMSNorm(
            self.complex_width,
            epsilon=config.rms_epsilon,
        )
        nn.init.normal_(self.embedding_real.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embedding_imag.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.final_norm.weight.fill_(math.sqrt(2.0))

    def complex_hidden(self, input_ids: Tensor) -> ComplexField:
        real = self.embedding_real(input_ids)
        imag = self.embedding_imag(input_ids)
        for block in self.blocks:
            if (
                self.config.activation_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                result = activation_checkpoint(
                    block,
                    real,
                    imag,
                    use_reentrant=False,
                )
                real, imag = cast("ComplexField", result)
            else:
                real, imag = block(real, imag)
        return self.final_norm(real, imag)

    def hidden(self, input_ids: Tensor) -> Tensor:
        return torch.cat(self.complex_hidden(input_ids), dim=-1)

    def forward(self, input_ids: Tensor) -> Tensor:
        real, imag = self.complex_hidden(input_ids)
        return functional.linear(real, self.embedding_real.weight) + functional.linear(
            imag,
            self.embedding_imag.weight,
        )


class ContentAlignedImagePostFusionAlphabet2LM(VectorImagePostFusionAlphabet2LM):
    """Vector ALPHABET with a shared content basis across fixed temporal modes."""

    block_type = ContentAlignedImagePostFusionAlphabet2Block


class ContentPreservingImagePostFusionAlphabet2LM(VectorImagePostFusionAlphabet2LM):
    """ALPHABET with a persistent content axis and grouped fixed-pole workspaces."""

    block_type = ContentPreservingImagePostFusionAlphabet2Block


class HybridContentDenseImagePostFusionAlphabet2LM(VectorImagePostFusionAlphabet2LM):
    """ALPHABET combining persistent content and compact Dense modal evidence."""

    block_type = HybridContentDenseImagePostFusionAlphabet2Block


class MultiObserverImagePostFusionAlphabet2LM(VectorImagePostFusionAlphabet2LM):
    """Dense-write Vector ALPHABET with a direct multi-observer read."""

    block_type = MultiObserverImagePostFusionAlphabet2Block


class ReadAdaptedImagePostFusionAlphabet2LM(VectorImagePostFusionAlphabet2LM):
    """Dense Vector ALPHABET with identity-initialized pole-wise read bases."""

    block_type = ReadAdaptedImagePostFusionAlphabet2Block


class TemporallyWhitenedImagePostFusionAlphabet2LM(VectorImagePostFusionAlphabet2LM):
    """J2 ALPHABET with separate transport and temporal observation bases."""

    block_type = TemporallyWhitenedImagePostFusionAlphabet2Block


class DynamicDeltaImagePostFusionAlphabet2LM(VectorImagePostFusionAlphabet2LM):
    """Dense Vector ALPHABET with mandatory exact pole-wise dynamic time."""

    block_type = DynamicDeltaImagePostFusionAlphabet2Block


class LaplaceMambaLM(nn.Module):
    """Token LM built only from integrated Laplace-Mamba blocks."""

    def __init__(self, config: LaplaceMambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.model_width)
        self.blocks = nn.ModuleList(
            LaplaceMambaBlock(config) for _ in range(config.layers)
        )
        self.final_norm = nn.RMSNorm(config.model_width, eps=config.rms_epsilon)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            if (
                self.config.activation_checkpoint
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = cast(
                    "Tensor",
                    activation_checkpoint(
                        block,
                        hidden,
                        use_reentrant=False,
                    ),
                )
            else:
                hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, input_ids: Tensor) -> Tensor:
        return functional.linear(self.hidden(input_ids), self.embedding.weight)


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


class TokenRateVectorPoleBlock(nn.Module):
    """Detect, transport, and selectively read one depth-local modal state."""

    def __init__(
        self,
        modes: int,
        *,
        pole_modes: int,
        vector_width: int,
        reader_kernel: int,
        beta_initial: float,
        context_length: int,
        scan_fp32: bool,
        minimum_half_life: float,
        maximum_half_life: float,
        epsilon: float,
        query_rho: float = 0.5,
    ) -> None:
        super().__init__()
        self.modes = int(modes)
        self.pole_modes = int(pole_modes)
        self.vector_width = int(vector_width)
        self.query_rho = float(query_rho)
        self.epsilon = float(epsilon)
        self.reader = PoleSpecificCausalVectorReader(
            modes,
            pole_modes,
            vector_width,
            kernel_size=reader_kernel,
        )
        self.pole_memory = FixedComplexPoleMemory1D(
            pole_modes,
            context_length=context_length,
            scan_fp32=scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=minimum_half_life,
            maximum_half_life=maximum_half_life,
        )
        self.query_norm = nn.RMSNorm(2 * modes, eps=epsilon)
        self.query = nn.Linear(2 * modes, pole_modes, bias=False)
        self.vector_query_real = nn.Linear(
            2 * modes, pole_modes * (vector_width - 1), bias=False
        )
        self.vector_query_imag = nn.Linear(
            2 * modes, pole_modes * vector_width, bias=False
        )
        self.synthesis = nn.Linear(
            2 * pole_modes * vector_width, 2 * modes, bias=False
        )
        self.beta = nn.Parameter(torch.tensor(float(beta_initial)))
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.vector_query_real.weight)
        nn.init.xavier_uniform_(self.vector_query_imag.weight)
        nn.init.xavier_uniform_(self.synthesis.weight)

    def query_components(self, packed: Tensor) -> ComplexField:
        normalized = self.query_norm(packed)
        scalar = 1.0 + self.query_rho * torch.tanh(self.query(normalized))
        scalar = scalar / scalar.mean(dim=-1, keepdim=True)
        extra_real = torch.tanh(self.vector_query_real(normalized)).reshape(
            *packed.shape[:-1], self.pole_modes, self.vector_width - 1
        )
        query_real = torch.cat((scalar.unsqueeze(-1), extra_real), dim=-1)
        query_imag = torch.tanh(self.vector_query_imag(normalized)).reshape(
            *packed.shape[:-1], self.pole_modes, self.vector_width
        )
        norm = (
            query_real.float()
            .square()
            .add(query_imag.float().square())
            .sum(dim=-1, keepdim=True)
            .sqrt()
            .clamp_min(self.epsilon)
        )
        scale = (scalar.abs().float().unsqueeze(-1) / norm).to(query_real.dtype)
        return query_real * scale, query_imag * scale

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        excitation = self.reader(real, imag)
        state_real, state_imag = self.pole_memory(*excitation)
        query_real, query_imag = self.query_components(torch.cat((real, imag), dim=-1))
        selected_real = state_real * query_real + state_imag * query_imag
        selected_imag = state_imag * query_real - state_real * query_imag
        projected = self.synthesis(
            torch.cat((selected_real.flatten(-2), selected_imag.flatten(-2)), dim=-1)
        )
        memory_real, memory_imag = projected.chunk(2, dim=-1)
        trunk_rms = real.float().square().add(imag.float().square()).mean(
            dim=-1, keepdim=True
        ).sqrt()
        memory_rms = memory_real.float().square().add(memory_imag.float().square()).mean(
            dim=-1, keepdim=True
        ).sqrt()
        scale = (trunk_rms / (memory_rms + self.epsilon)).detach().to(real.dtype)
        beta = self.beta.to(real.dtype)
        return real + beta * memory_real * scale, imag + beta * memory_imag * scale


class FactorizedTokenRateVectorPoleBlock(nn.Module):
    """Expand modal state width without a dense KxPxR parameterization."""

    def __init__(  # noqa: PLR0915
        self,
        modes: int,
        *,
        pole_modes: int,
        vector_width: int,
        write_rank: int,
        query_rank: int,
        synthesis_rank: int,
        retain_factor_state: bool,
        learned_factor_read: bool,
        factor_read_rho: float,
        factor_write_law: Literal["row_specific", "shared_outer", "pole_outer"],
        mamba_outer: bool,
        outer_direct: bool,
        outer_gate: bool,
        outer_kernel: int,
        reader_kernel: int,
        beta_initial: float,
        context_length: int,
        scan_fp32: bool,
        minimum_half_life: float,
        maximum_half_life: float,
        epsilon: float,
        query_rho: float = 0.5,
    ) -> None:
        super().__init__()
        self.modes = int(modes)
        self.pole_modes = int(pole_modes)
        self.vector_width = int(vector_width)
        self.write_rank = int(write_rank)
        self.query_rank = int(query_rank)
        self.synthesis_rank = int(synthesis_rank)
        self.retain_factor_state = bool(retain_factor_state)
        self.learned_factor_read = bool(learned_factor_read)
        self.factor_read_rho = float(factor_read_rho)
        self.factor_write_law = factor_write_law
        self.mamba_outer = bool(mamba_outer)
        self.outer_direct = bool(outer_direct)
        self.outer_gate = bool(outer_gate)
        self.outer_kernel = int(outer_kernel)
        self.baseline_width = min(4, vector_width)
        self.query_rho = float(query_rho)
        self.epsilon = float(epsilon)
        self.reader = PoleSpecificCausalVectorReader(
            modes,
            pole_modes,
            self.baseline_width,
            kernel_size=reader_kernel,
        )
        self.extra_reader = (
            PoleSpecificCausalVectorReader(
                modes,
                pole_modes,
                write_rank - self.baseline_width,
                kernel_size=reader_kernel,
            )
            if write_rank > self.baseline_width
            else None
        )
        self.pole_memory = FixedComplexPoleMemory1D(
            pole_modes,
            context_length=context_length,
            scan_fp32=scan_fp32,
            initialization="lifetime_palette",
            minimum_half_life=minimum_half_life,
            maximum_half_life=maximum_half_life,
        )
        self.query_norm = nn.RMSNorm(2 * modes, eps=epsilon)
        self.query = nn.Linear(2 * modes, pole_modes, bias=False)
        self.vector_query_real = nn.Linear(
            2 * modes, pole_modes * (self.baseline_width - 1), bias=False
        )
        self.vector_query_imag = nn.Linear(
            2 * modes, pole_modes * self.baseline_width, bias=False
        )
        extra_query_width = query_rank - self.baseline_width
        if extra_query_width > 0:
            self.extra_query_norm = nn.RMSNorm(2 * modes, eps=epsilon)
            self.extra_query_real = nn.Linear(
                2 * modes, pole_modes * extra_query_width, bias=False
            )
            self.extra_query_imag = nn.Linear(
                2 * modes, pole_modes * extra_query_width, bias=False
            )
        else:
            self.extra_query_norm = None
            self.extra_query_real = None
            self.extra_query_imag = None
        self.synthesis = nn.Linear(
            2 * pole_modes * self.baseline_width,
            2 * modes,
            bias=False,
        )
        self.content_basis_real = nn.Parameter(torch.zeros(write_rank, vector_width))
        self.content_basis_imag = nn.Parameter(torch.zeros(write_rank, vector_width))
        self.content_delta_norm = nn.RMSNorm(2 * modes, eps=epsilon)
        self.content_delta = nn.Linear(
            2 * modes, 2 * write_rank * vector_width, bias=False
        )
        self.query_basis_real = nn.Parameter(torch.zeros(query_rank, vector_width))
        self.query_basis_imag = nn.Parameter(torch.zeros(query_rank, vector_width))
        self.query_basis_delta_norm = nn.RMSNorm(2 * modes, eps=epsilon)
        self.query_basis_delta = nn.Linear(
            2 * modes, 2 * query_rank * vector_width, bias=False
        )
        self.extra_projection_basis = nn.Parameter(
            torch.zeros(pole_modes, synthesis_rank, vector_width)
        )
        self.extra_synthesis = nn.Linear(
            2 * pole_modes * synthesis_rank, 2 * modes, bias=False
        )
        if self.learned_factor_read:
            with torch.random.fork_rng(devices=[]):
                self.factor_read_norm = nn.RMSNorm(2 * modes, eps=epsilon)
                self.factor_read_real = nn.Linear(
                    2 * modes, pole_modes * write_rank, bias=False
                )
                self.factor_read_imag = nn.Linear(
                    2 * modes, pole_modes * write_rank, bias=False
                )
        else:
            self.factor_read_norm = None
            self.factor_read_real = None
            self.factor_read_imag = None
        extra_width = vector_width - self.baseline_width
        if self.factor_write_law == "pole_outer" and extra_width > 0:
            with torch.random.fork_rng(devices=[]):
                self.pole_value_norm = nn.RMSNorm(2 * modes, eps=epsilon)
                self.pole_value_real = nn.Linear(
                    2 * modes, pole_modes * extra_width, bias=False
                )
                self.pole_value_imag = nn.Linear(
                    2 * modes, pole_modes * extra_width, bias=False
                )
        else:
            self.pole_value_norm = None
            self.pole_value_real = None
            self.pole_value_imag = None
        self._configure_outer(modes, epsilon)
        self.beta = nn.Parameter(torch.tensor(float(beta_initial)))
        self._initialize_factorized_expansion()

    def _configure_outer(self, modes: int, epsilon: float) -> None:
        if not self.mamba_outer:
            self.outer_input_norm = None
            self.outer_input_projection = None
            self.outer_conv = None
            self.register_parameter("outer_direct_scale", None)
            self.outer_post_norm = None
            self.outer_output = None
            return
        packed_modes = 2 * modes
        with torch.random.fork_rng(devices=[]):
            self.outer_input_norm = nn.RMSNorm(packed_modes, eps=epsilon)
            self.outer_input_projection = nn.Linear(
                packed_modes, 2 * packed_modes, bias=False
            )
            self.outer_conv = nn.Conv1d(
                packed_modes,
                packed_modes,
                kernel_size=self.outer_kernel,
                groups=packed_modes,
                bias=True,
            )
            self.outer_direct_scale = nn.Parameter(torch.ones(packed_modes))
            self.outer_post_norm = nn.RMSNorm(packed_modes, eps=epsilon)
            self.outer_output = nn.Linear(packed_modes, packed_modes, bias=False)

    def _initialize_factorized_expansion(self) -> None:
        baseline_width = self.baseline_width
        with torch.no_grad():
            identity = torch.eye(baseline_width)
            self.content_basis_real[:baseline_width, :baseline_width].copy_(identity)
            self.query_basis_real[:baseline_width, :baseline_width].copy_(identity)
            projection_start = (
                baseline_width if self.vector_width > baseline_width else 0
            )
            projection_width = self.vector_width - projection_start
            self.extra_projection_basis[:, :, projection_start:].normal_(
                std=1.0 / math.sqrt(projection_width)
            )
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.vector_query_real.weight)
        nn.init.xavier_uniform_(self.vector_query_imag.weight)
        if self.extra_query_real is not None and self.extra_query_imag is not None:
            nn.init.xavier_uniform_(self.extra_query_real.weight)
            nn.init.xavier_uniform_(self.extra_query_imag.weight)
        nn.init.xavier_uniform_(self.synthesis.weight)
        nn.init.xavier_uniform_(self.content_delta.weight)
        nn.init.xavier_uniform_(self.query_basis_delta.weight)
        with torch.no_grad():
            content_mask = torch.ones(
                2, self.write_rank, self.vector_width, 1
            )
            content_mask[:, :, :baseline_width] = 0.0
            self.content_delta.weight.mul_(content_mask.flatten(0, 2))
            query_mask = torch.ones(
                2, self.query_rank, self.vector_width, 1
            )
            query_mask[:, :, :baseline_width] = 0.0
            self.query_basis_delta.weight.mul_(query_mask.flatten(0, 2))
        nn.init.zeros_(self.extra_synthesis.weight)
        if self.factor_read_real is not None and self.factor_read_imag is not None:
            nn.init.zeros_(self.factor_read_real.weight)
            nn.init.zeros_(self.factor_read_imag.weight)
        if self.pole_value_real is not None and self.pole_value_imag is not None:
            with torch.random.fork_rng(devices=[]):
                nn.init.xavier_uniform_(self.pole_value_real.weight)
                nn.init.xavier_uniform_(self.pole_value_imag.weight)
        if self.outer_output is not None:
            nn.init.zeros_(self.outer_output.weight)

    @staticmethod
    def complex_factor_product(
        left_real: Tensor,
        left_imag: Tensor,
        right_real: Tensor,
        right_imag: Tensor,
    ) -> ComplexField:
        return (
            torch.einsum("btpj,btjr->btpr", left_real, right_real)
            - torch.einsum("btpj,btjr->btpr", left_imag, right_imag),
            torch.einsum("btpj,btjr->btpr", left_real, right_imag)
            + torch.einsum("btpj,btjr->btpr", left_imag, right_real),
        )

    def content_basis(self, packed: Tensor) -> ComplexField:
        delta_real, delta_imag = self.content_delta(
            self.content_delta_norm(packed)
        ).chunk(2, dim=-1)
        shape = (*packed.shape[:-1], self.write_rank, self.vector_width)
        return (
            self.content_basis_real + delta_real.reshape(shape),
            self.content_basis_imag + delta_imag.reshape(shape),
        )

    def query_factors(self, packed: Tensor) -> ComplexField:
        normalized = self.query_norm(packed)
        scalar = 1.0 + self.query_rho * torch.tanh(self.query(normalized))
        scalar = scalar / scalar.mean(dim=-1, keepdim=True)
        baseline_extra_real = torch.tanh(self.vector_query_real(normalized)).reshape(
            *packed.shape[:-1], self.pole_modes, self.baseline_width - 1
        )
        baseline_real = torch.cat((scalar.unsqueeze(-1), baseline_extra_real), dim=-1)
        baseline_imag = torch.tanh(self.vector_query_imag(normalized)).reshape(
            *packed.shape[:-1], self.pole_modes, self.baseline_width
        )
        baseline_norm = baseline_real.float().square().add(
            baseline_imag.float().square()
        ).sum(dim=-1, keepdim=True).sqrt().clamp_min(self.epsilon)
        baseline_scale = (
            scalar.abs().float().unsqueeze(-1) / baseline_norm
        ).to(baseline_real.dtype)
        baseline_real = baseline_real * baseline_scale
        baseline_imag = baseline_imag * baseline_scale
        if (
            self.extra_query_norm is not None
            and self.extra_query_real is not None
            and self.extra_query_imag is not None
        ):
            extra_normalized = self.extra_query_norm(packed)
            extra_shape = (
                *packed.shape[:-1],
                self.pole_modes,
                self.query_rank - self.baseline_width,
            )
            extra_real = torch.tanh(self.extra_query_real(extra_normalized)).reshape(
                extra_shape
            )
            extra_imag = torch.tanh(self.extra_query_imag(extra_normalized)).reshape(
                extra_shape
            )
            extra_norm = extra_real.float().square().add(extra_imag.float().square()).sum(
                dim=-1, keepdim=True
            ).sqrt().clamp_min(self.epsilon)
            extra_scale = (
                scalar.abs().float().unsqueeze(-1) / extra_norm
            ).to(extra_real.dtype)
            factor_real = torch.cat((baseline_real, extra_real * extra_scale), dim=-1)
            factor_imag = torch.cat((baseline_imag, extra_imag * extra_scale), dim=-1)
        else:
            factor_real, factor_imag = baseline_real, baseline_imag
        return factor_real, factor_imag

    def query_basis(self, packed: Tensor) -> ComplexField:
        delta_real, delta_imag = self.query_basis_delta(
            self.query_basis_delta_norm(packed)
        ).chunk(2, dim=-1)
        shape = (*packed.shape[:-1], self.query_rank, self.vector_width)
        return (
            self.query_basis_real + delta_real.reshape(shape),
            self.query_basis_imag + delta_imag.reshape(shape),
        )

    def factor_state_drive(
        self,
        packed: Tensor,
        coefficient_real: Tensor,
        coefficient_imag: Tensor,
        basis_real: Tensor,
        basis_imag: Tensor,
    ) -> ComplexField:
        row_specific = (
            coefficient_real.unsqueeze(-1) * basis_real.unsqueeze(-3)
            - coefficient_imag.unsqueeze(-1) * basis_imag.unsqueeze(-3),
            coefficient_real.unsqueeze(-1) * basis_imag.unsqueeze(-3)
            + coefficient_imag.unsqueeze(-1) * basis_real.unsqueeze(-3),
        )
        if self.factor_write_law == "row_specific" or self.vector_width == self.baseline_width:
            return row_specific
        baseline = (
            row_specific[0][..., : self.baseline_width],
            row_specific[1][..., : self.baseline_width],
        )
        if self.factor_write_law == "shared_outer":
            value_real = basis_real[..., self.baseline_width :].mean(dim=-2)
            value_imag = basis_imag[..., self.baseline_width :].mean(dim=-2)
            value_real = value_real.unsqueeze(-2).unsqueeze(-3)
            value_imag = value_imag.unsqueeze(-2).unsqueeze(-3)
        else:
            if (
                self.pole_value_norm is None
                or self.pole_value_real is None
                or self.pole_value_imag is None
            ):
                raise RuntimeError("pole-specific outer value projection disappeared")
            active = self.pole_value_norm(packed)
            shape = (
                *packed.shape[:-1],
                self.pole_modes,
                self.vector_width - self.baseline_width,
            )
            value_real = self.pole_value_real(active).reshape(shape).unsqueeze(-2)
            value_imag = self.pole_value_imag(active).reshape(shape).unsqueeze(-2)
        coefficient_real = coefficient_real.unsqueeze(-1)
        coefficient_imag = coefficient_imag.unsqueeze(-1)
        extra = (
            coefficient_real * value_real - coefficient_imag * value_imag,
            coefficient_real * value_imag + coefficient_imag * value_real,
        )
        return (
            torch.cat((baseline[0], extra[0]), dim=-1),
            torch.cat((baseline[1], extra[1]), dim=-1),
        )

    def factor_read(self, packed: Tensor) -> ComplexField | None:
        if self.factor_read_real is None or self.factor_read_imag is None:
            return None
        normalized = self.factor_read_norm
        if normalized is None:
            raise RuntimeError("learned factor read normalization disappeared")
        shape = (*packed.shape[:-1], self.pole_modes, self.write_rank)
        active = normalized(packed)
        return (
            1.0
            + self.factor_read_rho
            * torch.tanh(self.factor_read_real(active)).reshape(shape),
            self.factor_read_rho
            * torch.tanh(self.factor_read_imag(active)).reshape(shape),
        )

    def contract_factor_state(
        self,
        packed: Tensor,
        state_real: Tensor,
        state_imag: Tensor,
    ) -> ComplexField:
        factor_read = self.factor_read(packed)
        if factor_read is None:
            return state_real.sum(dim=-2), state_imag.sum(dim=-2)
        read_real, read_imag = factor_read
        return (
            (state_real * read_real.unsqueeze(-1)).sum(dim=-2)
            + (state_imag * read_imag.unsqueeze(-1)).sum(dim=-2),
            (state_imag * read_real.unsqueeze(-1)).sum(dim=-2)
            - (state_real * read_imag.unsqueeze(-1)).sum(dim=-2),
        )

    def _synthesize(self, selected_real: Tensor, selected_imag: Tensor) -> ComplexField:
        baseline_width = self.synthesis.in_features // (2 * self.pole_modes)
        baseline = self.synthesis(
            torch.cat(
                (
                    selected_real[..., :baseline_width].flatten(-2),
                    selected_imag[..., :baseline_width].flatten(-2),
                ),
                dim=-1,
            )
        )
        compressed_real = torch.einsum(
            "btpr,pjr->btpj", selected_real, self.extra_projection_basis
        )
        compressed_imag = torch.einsum(
            "btpr,pjr->btpj", selected_imag, self.extra_projection_basis
        )
        extra = self.extra_synthesis(
            torch.cat((compressed_real.flatten(-2), compressed_imag.flatten(-2)), dim=-1)
        )
        return tuple(
            baseline_part + extra_part
            for baseline_part, extra_part in zip(
                baseline.chunk(2, dim=-1), extra.chunk(2, dim=-1), strict=True
            )
        )

    def _outer_refine(self, packed: Tensor, memory: ComplexField) -> ComplexField:
        if self.outer_output is None:
            return memory
        if (
            self.outer_input_norm is None
            or self.outer_input_projection is None
            or self.outer_conv is None
            or self.outer_post_norm is None
            or self.outer_direct_scale is None
        ):
            raise RuntimeError("Mamba-shaped outer scaffold disappeared")
        local, gate = self.outer_input_projection(
            self.outer_input_norm(packed)
        ).chunk(2, dim=-1)
        local = functional.silu(
            self.outer_conv(
                functional.pad(
                    local.transpose(1, 2),
                    (self.outer_kernel - 1, 0),
                )
            ).transpose(1, 2)
        )
        base = torch.cat(memory, dim=-1)
        mixed = (
            base + local * self.outer_direct_scale.to(local.dtype)
            if self.outer_direct
            else base
        )
        refined = self.outer_post_norm(mixed)
        if self.outer_gate:
            refined = refined * functional.silu(gate)
        delta = self.outer_output(refined)
        return tuple(
            base_part + delta_part
            for base_part, delta_part in zip(
                base.chunk(2, dim=-1),
                delta.chunk(2, dim=-1),
                strict=True,
            )
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        packed = torch.cat((real, imag), dim=-1)
        coefficient_real, coefficient_imag = self.reader(real, imag)
        if self.extra_reader is not None:
            extra_real, extra_imag = self.extra_reader(real, imag)
            coefficient_real = torch.cat((coefficient_real, extra_real), dim=-1)
            coefficient_imag = torch.cat((coefficient_imag, extra_imag), dim=-1)
        content_basis = self.content_basis(packed)
        if self.retain_factor_state:
            excitation = self.factor_state_drive(
                packed,
                coefficient_real,
                coefficient_imag,
                *content_basis,
            )
            factor_state = self.pole_memory(*excitation)
            state_real, state_imag = self.contract_factor_state(
                packed,
                *factor_state,
            )
        else:
            excitation = self.complex_factor_product(
                coefficient_real,
                coefficient_imag,
                *content_basis,
            )
            state_real, state_imag = self.pole_memory(*excitation)
        query = self.complex_factor_product(
            *self.query_factors(packed),
            *self.query_basis(packed),
        )
        selected_real = state_real * query[0] + state_imag * query[1]
        selected_imag = state_imag * query[0] - state_real * query[1]
        memory_real, memory_imag = self._synthesize(selected_real, selected_imag)
        memory_real, memory_imag = self._outer_refine(
            packed,
            (memory_real, memory_imag),
        )
        trunk_rms = real.float().square().add(imag.float().square()).mean(
            dim=-1, keepdim=True
        ).sqrt()
        memory_rms = memory_real.float().square().add(memory_imag.float().square()).mean(
            dim=-1, keepdim=True
        ).sqrt()
        scale = (trunk_rms / (memory_rms + self.epsilon)).detach().to(real.dtype)
        beta = self.beta.to(real.dtype)
        return real + beta * memory_real * scale, imag + beta * memory_imag * scale


def _make_repeated_vector_pole_block(
    config: AlphabetLMConfig,
) -> TokenRateVectorPoleBlock | FactorizedTokenRateVectorPoleBlock:
    if config.repeated_vector_pole_factorized:
        return FactorizedTokenRateVectorPoleBlock(
            config.modes,
            pole_modes=config.repeated_vector_pole_modes,
            vector_width=config.repeated_vector_pole_width,
            write_rank=config.repeated_vector_pole_write_rank,
            query_rank=config.repeated_vector_pole_query_rank,
            synthesis_rank=config.repeated_vector_pole_synthesis_rank,
            retain_factor_state=config.repeated_vector_pole_retain_factor_state,
            learned_factor_read=config.repeated_vector_pole_learned_factor_read,
            factor_read_rho=config.repeated_vector_pole_factor_read_rho,
            factor_write_law=config.repeated_vector_pole_factor_write_law,
            mamba_outer=config.repeated_vector_pole_mamba_outer,
            outer_direct=config.repeated_vector_pole_outer_direct,
            outer_gate=config.repeated_vector_pole_outer_gate,
            outer_kernel=config.repeated_vector_pole_outer_kernel,
            reader_kernel=config.repeated_vector_pole_reader_kernel,
            beta_initial=config.repeated_vector_pole_beta_initial,
            context_length=config.context_length,
            scan_fp32=config.scan_fp32,
            minimum_half_life=config.repeated_vector_pole_minimum_half_life,
            maximum_half_life=config.repeated_vector_pole_maximum_half_life,
            epsilon=config.rms_epsilon,
        )
    return TokenRateVectorPoleBlock(
        config.modes,
        pole_modes=config.repeated_vector_pole_modes,
        vector_width=config.repeated_vector_pole_width,
        reader_kernel=config.repeated_vector_pole_reader_kernel,
        beta_initial=config.repeated_vector_pole_beta_initial,
        context_length=config.context_length,
        scan_fp32=config.scan_fp32,
        minimum_half_life=config.repeated_vector_pole_minimum_half_life,
        maximum_half_life=config.repeated_vector_pole_maximum_half_life,
        epsilon=config.rms_epsilon,
    )


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
                    value_width=config.slow_cnn_pole_value_width,
                    matrix_key_width=config.slow_cnn_pole_matrix_key_width,
                    independent_matrix_value=(
                        config.slow_cnn_pole_independent_matrix_value
                    ),
                    vector_width=config.slow_cnn_pole_vector_width,
                    complex_vector_excitation=(
                        config.slow_cnn_pole_complex_vector_excitation
                    ),
                    complex_vector_query=config.slow_cnn_pole_complex_vector_query,
                    coordinate_read=config.slow_cnn_pole_coordinate_read,
                    dynamic_transport=config.slow_cnn_pole_dynamic_transport,
                    transport_rank=config.slow_cnn_pole_transport_rank,
                    transport_scale=config.slow_cnn_pole_transport_scale,
                    transport_bound=config.slow_cnn_pole_transport_bound,
                    pole_specific_reader=config.slow_cnn_pole_specific_reader,
                    reader_kernel=config.slow_cnn_pole_reader_kernel,
                    write_scheduler=config.slow_cnn_pole_write_scheduler,
                    innovation=config.slow_cnn_pole_innovation,
                    innovation_kernel=config.slow_cnn_pole_innovation_kernel,
                    semantic_clock=config.slow_cnn_pole_semantic_clock,
                )
        else:
            self.slow_cnn_pole_memory = None
        if config.repeated_vector_pole_memory:
            with torch.random.fork_rng(devices=[]):
                self.repeated_vector_pole_memories = nn.ModuleList(
                    _make_repeated_vector_pole_block(config)
                    for _ in range(
                        config.layers // config.repeated_vector_pole_interval
                    )
                )
        else:
            self.repeated_vector_pole_memories = None
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.orthogonal_(self.analysis.weight)

    def _checkpointed_repeated_hidden(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> ComplexField:
        banks = self.repeated_vector_pole_memories
        if banks is None:
            raise RuntimeError("checkpointed repeated memory disappeared")
        bank_index = 0
        for index, block in enumerate(self.blocks):
            active_bank: nn.Module | None = None
            if (index + 1) % self.config.repeated_vector_pole_interval == 0:
                active_bank = banks[bank_index]
                bank_index += 1

            def stage(
                active_real: Tensor,
                active_imag: Tensor,
                block: nn.Module = block,
                bank: nn.Module | None = active_bank,
            ) -> ComplexField:
                output = cast("ComplexField", block(active_real, active_imag))
                return output if bank is None else cast("ComplexField", bank(*output))

            real, imag = cast(
                "ComplexField",
                activation_checkpoint(
                    stage,
                    real,
                    imag,
                    use_reentrant=False,
                ),
            )
        return real, imag

    def _use_repeated_activation_checkpoint(self) -> bool:
        return (
            self.config.repeated_vector_pole_activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
        )

    def hidden(self, input_ids: Tensor) -> Tensor:  # noqa: C901
        packed = self.analysis(self.embedding(input_ids))
        real, imag = packed.split(self.config.modes, dim=-1)
        if self._use_repeated_activation_checkpoint():
            real, imag = self._checkpointed_repeated_hidden(real, imag)
            return self.final_norm(torch.cat((real, imag), dim=-1))
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
            if (
                self.repeated_vector_pole_memories is not None
                and (index + 1) % self.config.repeated_vector_pole_interval == 0
            ):
                bank_index = (
                    (index + 1) // self.config.repeated_vector_pole_interval - 1
                )
                real, imag = self.repeated_vector_pole_memories[bank_index](real, imag)
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
    "ComplexDepthwiseCausalConv1d",
    "ComplexHighwayLaplaceMambaBlock",
    "ComplexHighwayLaplaceMambaLM",
    "ComplexMultiObserverRead",
    "ContentAlignedImagePostFusionAlphabet2Block",
    "ContentAlignedImagePostFusionAlphabet2LM",
    "ContentPreservingImagePostFusionAlphabet2Block",
    "ContentPreservingImagePostFusionAlphabet2LM",
    "DenseComplexConv1dReader",
    "DynamicDeltaImagePostFusionAlphabet2Block",
    "DynamicDeltaImagePostFusionAlphabet2LM",
    "DynamicLowRankWrite",
    "FactorizedTokenRateVectorPoleBlock",
    "FixedComplexPoleMemory1D",
    "FixedPoleResidualSidecar",
    "GroupedCausalFactorizedComplexConv1dReader",
    "GroupedPackedComplexLinear",
    "IdentityComplexMemory1D",
    "ImagePostFusionAlphabet2Block",
    "ImagePostFusionAlphabet2LM",
    "HybridContentDenseImagePostFusionAlphabet2Block",
    "HybridContentDenseImagePostFusionAlphabet2LM",
    "LaplaceMambaBlock",
    "LaplaceMambaLM",
    "LaplaceMambaLMConfig",
    "LowRankDecaySelector",
    "LowRankPoleAlignedComplexCausalConv1d",
    "LowRankPoleRouter",
    "MultiObserverImagePostFusionAlphabet2Block",
    "MultiObserverImagePostFusionAlphabet2LM",
    "PoleAlignedComplexCausalConv1d",
    "PoleAxisTemporalWhitening",
    "PoleWiseComplexReadAdapter",
    "PoleWiseDynamicDelta",
    "QueryConditionedLowRankReadout",
    "ReadAdaptedImagePostFusionAlphabet2Block",
    "ReadAdaptedImagePostFusionAlphabet2LM",
    "SemanticEdgePoleMemory",
    "SlowCausalCNNPoleMemory",
    "StrictComplexCausalConv1d",
    "TemporallyWhitenedImagePostFusionAlphabet2Block",
    "TemporallyWhitenedImagePostFusionAlphabet2LM",
    "TensorProductPoleMemory1D",
    "TokenRateVectorPoleBlock",
    "VectorImagePostFusionAlphabet2Block",
    "VectorImagePostFusionAlphabet2LM",
]
