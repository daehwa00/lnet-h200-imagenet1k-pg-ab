#!/usr/bin/env python3
"""Measure fixed-pole memory and effective context use from trained checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.alphabet_lm import (
    AlphabetLM,
    AlphabetLMConfig,
    CausalCNNPoleMemory,
    ChunkedSemanticPoleMemory,
    ComplexHighwayLaplaceMambaLM,
    ContentAlignedImagePostFusionAlphabet2LM,
    DynamicLowRankWrite,
    FactorizedTokenRateVectorPoleBlock,
    FixedComplexPoleMemory1D,
    FixedPoleResidualSidecar,
    GroupedPackedComplexLinear,
    ImagePostFusionAlphabet2LM,
    LaplaceMambaLM,
    LaplaceMambaLMConfig,
    LowRankDecaySelector,
    MultiObserverImagePostFusionAlphabet2LM,
    QueryConditionedLowRankReadout,
    ReadAdaptedImagePostFusionAlphabet2LM,
    SemanticEdgePoleMemory,
    SlowCausalCNNPoleMemory,
    TemporallyWhitenedImagePostFusionAlphabet2LM,
    TensorProductPoleMemory1D,
    TokenRateVectorPoleBlock,
    VectorImagePostFusionAlphabet2LM,
)
from lnet.alphabet_lm_data import TokenBlockDataset
from lnet.alphabet_lm_mamba import MambaLM, MambaLMConfig
from lnet.pac_complex_layers import PackedComplexLinear


def _build(kind: str) -> nn.Module:
    if kind in {
        "laplace_mamba",
        "laplace_mamba_complex_highway",
        "alphabet2_image_postfusion",
        "alphabet2_vector_image_postfusion",
        "alphabet2_content_aligned_image_postfusion",
        "alphabet2_content_aligned_j2",
        "alphabet2_content_aligned_j4",
        "alphabet2_multi_observer_image_postfusion",
        "alphabet2_read_adapter_image_postfusion",
        "alphabet2_temporal_whitening_image_postfusion",
    }:
        model_type: type[nn.Module]
        if kind == "laplace_mamba_complex_highway":
            model_type = ComplexHighwayLaplaceMambaLM
        elif kind == "alphabet2_image_postfusion":
            model_type = ImagePostFusionAlphabet2LM
        elif kind == "alphabet2_vector_image_postfusion":
            model_type = VectorImagePostFusionAlphabet2LM
        elif kind in {
            "alphabet2_content_aligned_image_postfusion",
            "alphabet2_content_aligned_j2",
            "alphabet2_content_aligned_j4",
        }:
            model_type = ContentAlignedImagePostFusionAlphabet2LM
        elif kind == "alphabet2_multi_observer_image_postfusion":
            model_type = MultiObserverImagePostFusionAlphabet2LM
        elif kind == "alphabet2_read_adapter_image_postfusion":
            model_type = ReadAdaptedImagePostFusionAlphabet2LM
        elif kind == "alphabet2_temporal_whitening_image_postfusion":
            model_type = TemporallyWhitenedImagePostFusionAlphabet2LM
        else:
            model_type = LaplaceMambaLM
        config = (
            LaplaceMambaLMConfig(conv_width=3)
            if kind
            in {
                "alphabet2_vector_image_postfusion",
                "alphabet2_content_aligned_image_postfusion",
                "alphabet2_content_aligned_j2",
                "alphabet2_content_aligned_j4",
                "alphabet2_multi_observer_image_postfusion",
                "alphabet2_read_adapter_image_postfusion",
                "alphabet2_temporal_whitening_image_postfusion",
            }
            else LaplaceMambaLMConfig()
        )
        if kind == "alphabet2_content_aligned_j2":
            config = LaplaceMambaLMConfig(conv_width=3, aligned_content_rank=2)
        elif kind == "alphabet2_content_aligned_j4":
            config = LaplaceMambaLMConfig(conv_width=3, aligned_content_rank=4)
        elif kind == "alphabet2_temporal_whitening_image_postfusion":
            config = LaplaceMambaLMConfig(conv_width=3, aligned_content_rank=2)
        return model_type(config)
    if kind == "mamba":
        return MambaLM(MambaLMConfig())
    if kind == "mamba2":
        return MambaLM(
            MambaLMConfig(
                layers=18,
                state_size=128,
                architecture="Mamba2",
            )
        )
    if kind == "mamba1_49m":
        return MambaLM(MambaLMConfig(layers=19))
    if kind == "mamba2_48m":
        return MambaLM(
            MambaLMConfig(
                layers=18,
                state_size=128,
                architecture="Mamba2",
            )
        )
    config = AlphabetLMConfig()
    if kind == "grouped":
        config = AlphabetLMConfig(
            pole_initialization="lifetime_palette",
            memory_banks=8,
            bank_pole_modes=128,
        )
    elif kind == "wide":
        config = AlphabetLMConfig(post_hidden=512)
    elif kind == "qread":
        config = AlphabetLMConfig(memory_readout="query_low_rank", query_read_rank=32)
    elif kind == "dense_fixed":
        config = AlphabetLMConfig(reader_type="dense_k3")
    elif kind == "dense_delta":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            pole_dynamics="delta_select",
            delta_select_rank=16,
            delta_select_initial_scale=0.3,
        )
    elif kind == "tensorpole":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="tensor_product",
            tensor_temporal_modes=8,
        )
    elif kind == "dense_dynamic_write_r4":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            write_map="dynamic_low_rank",
            dynamic_write_rank=4,
            dynamic_write_initial_scale=0.06,
        )
    elif kind == "dense_local_only":
        config = AlphabetLMConfig(reader_type="dense_k3", memory_layout="local_only")
    elif kind == "dense_local_sidecar":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_sidecar",
            sidecar_initial_scale=0.01,
        )
    elif kind == "dense_local_sidecar_normalized":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_sidecar",
            sidecar_initial_scale=0.01,
            sidecar_normalize_memory=True,
            sidecar_channelwise_scale=False,
        )
    elif kind == "dense_local_sidecar_normalized_no_recurrence":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_sidecar",
            sidecar_initial_scale=0.01,
            sidecar_normalize_memory=True,
            sidecar_channelwise_scale=False,
            sidecar_use_recurrence=False,
        )
    elif kind == "chunked_semantic_p128":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            chunk_memory=True,
            chunk_size=32,
            chunk_summary_width=128,
            chunk_pole_modes=128,
            chunk_upper_blocks=4,
            chunk_beta_initial=0.01,
            chunk_minimum_half_life=1.0,
            chunk_maximum_half_life=128.0,
        )
    elif kind in {"semantic_edge_p128", "semantic_edge_p128_no_recurrence"}:
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            semantic_edge_memory=True,
            semantic_edge_stride=16,
            semantic_edge_pole_modes=128,
            semantic_edge_upper_blocks=4,
            semantic_edge_beta_initial=0.01,
            semantic_edge_use_recurrence=kind == "semantic_edge_p128",
            semantic_edge_minimum_half_life=1.0,
            semantic_edge_maximum_half_life=256.0,
        )
    elif kind in {"cnn_pole_p128_6bank", "cnn_pole_p128_6bank_no_recurrence"}:
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=kind == "cnn_pole_p128_6bank",
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
        )
    elif kind in {
        "cnn_pole_p128_6bank_slow_p128",
        "cnn_pole_p128_6bank_slow_p128_no_recurrence",
    }:
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=kind == "cnn_pole_p128_6bank_slow_p128",
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
        )
    elif kind in {
        "alphabet2_retained_factor_fixed_p32r32_js4",
        "alphabet2_retained_factor_learned_p32r32_js4",
        "alphabet2_mamba_outer_post_p32j4r32",
        "alphabet2_mamba_outer_direct_p32j4r32",
        "alphabet2_mamba_outer_gate_p32j4r32",
        "alphabet2_mamba_outer_both_p32j4r32",
        "alphabet2_write_row_specific_p32j4r32",
        "alphabet2_write_shared_outer_p32j4r32",
        "alphabet2_write_pole_outer_p32j4r32",
    }:
        outer = kind.startswith("alphabet2_mamba_outer_")
        write_law = (
            "shared_outer"
            if kind.startswith("alphabet2_write_shared_outer")
            else "pole_outer"
            if kind.startswith("alphabet2_write_pole_outer")
            else "row_specific"
        )
        write_law_campaign = kind.startswith("alphabet2_write_")
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            repeated_vector_pole_memory=True,
            repeated_vector_pole_interval=1,
            repeated_vector_pole_modes=32,
            repeated_vector_pole_width=32,
            repeated_vector_pole_reader_kernel=3,
            repeated_vector_pole_beta_initial=0.01,
            repeated_vector_pole_minimum_half_life=16.0,
            repeated_vector_pole_maximum_half_life=4_096.0,
            repeated_vector_pole_factorized=True,
            repeated_vector_pole_write_rank=4,
            repeated_vector_pole_query_rank=4,
            repeated_vector_pole_synthesis_rank=4,
            repeated_vector_pole_retain_factor_state=True,
            repeated_vector_pole_learned_factor_read=(
                kind.endswith("learned_p32r32_js4") or outer or write_law_campaign
            ),
            repeated_vector_pole_factor_read_rho=0.5,
            repeated_vector_pole_factor_write_law=write_law,
            repeated_vector_pole_mamba_outer=outer,
            repeated_vector_pole_outer_direct=(
                kind.endswith(("direct_p32j4r32", "both_p32j4r32"))
            ),
            repeated_vector_pole_outer_gate=(
                kind.endswith(("gate_p32j4r32", "both_p32j4r32"))
            ),
            repeated_vector_pole_outer_kernel=4,
        )
    elif kind == "alphabet2_repeated_vector_pole_p32r4":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            repeated_vector_pole_memory=True,
            repeated_vector_pole_interval=1,
            repeated_vector_pole_modes=32,
            repeated_vector_pole_width=4,
            repeated_vector_pole_reader_kernel=3,
            repeated_vector_pole_beta_initial=0.01,
            repeated_vector_pole_minimum_half_life=16.0,
            repeated_vector_pole_maximum_half_life=4_096.0,
        )
    elif kind in {
        "alphabet2_factorized_vector_pole_p32r4_interface",
        "alphabet2_factorized_vector_pole_p32r32",
        "alphabet2_factorized_vector_pole_p32r32_js4",
        "alphabet2_factorized_vector_pole_p32r32_j8q4",
        "alphabet2_factorized_vector_pole_p32r32_j4q8",
        "alphabet2_factorized_vector_pole_p32r32_j8q8",
    }:
        write_rank = 8 if kind.endswith(("j8q4", "j8q8")) else 4
        query_rank = 8 if kind.endswith(("j4q8", "j8q8")) else 4
        vector_width = 4 if kind.endswith("p32r4_interface") else 32
        synthesis_rank = 4 if kind.endswith("p32r32_js4") else 16
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            repeated_vector_pole_memory=True,
            repeated_vector_pole_interval=1,
            repeated_vector_pole_modes=32,
            repeated_vector_pole_width=vector_width,
            repeated_vector_pole_reader_kernel=3,
            repeated_vector_pole_beta_initial=0.01,
            repeated_vector_pole_minimum_half_life=16.0,
            repeated_vector_pole_maximum_half_life=4_096.0,
            repeated_vector_pole_factorized=True,
            repeated_vector_pole_write_rank=write_rank,
            repeated_vector_pole_query_rank=query_rank,
            repeated_vector_pole_synthesis_rank=synthesis_rank,
        )
    elif kind == "alphabet2_dynamic_transport_r16":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_vector_width=16,
            slow_cnn_pole_complex_vector_excitation=True,
            slow_cnn_pole_complex_vector_query=True,
            slow_cnn_pole_coordinate_read=True,
            slow_cnn_pole_dynamic_transport=True,
            slow_cnn_pole_transport_rank=16,
            slow_cnn_pole_transport_scale=0.1,
            slow_cnn_pole_transport_bound=1.0,
        )
    elif kind in {
        "alphabet2_pole_reader_r16",
        "alphabet2_pole_reader_write_scheduler_r16",
        "alphabet2_pole_reader_innovation_r16",
        "alphabet2_pole_reader_semantic_clock_r16",
    }:
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=1,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=16.0,
            slow_cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_vector_width=16,
            slow_cnn_pole_complex_vector_query=True,
            slow_cnn_pole_coordinate_read=True,
            slow_cnn_pole_specific_reader=True,
            slow_cnn_pole_reader_kernel=3,
            slow_cnn_pole_write_scheduler=(
                kind == "alphabet2_pole_reader_write_scheduler_r16"
            ),
            slow_cnn_pole_innovation=(
                kind == "alphabet2_pole_reader_innovation_r16"
            ),
            slow_cnn_pole_innovation_kernel=3,
            slow_cnn_pole_semantic_clock=(
                kind == "alphabet2_pole_reader_semantic_clock_r16"
            ),
        )
    elif kind in {"alphabet2_anchor_q", "alphabet2_token_q"}:
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            slow_cnn_pole_query="anchor" if kind == "alphabet2_anchor_q" else "token",
            slow_cnn_pole_query_rho=0.5,
        )
    elif kind == "alphabet2_qk":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_key=True,
            slow_cnn_pole_key_rho=0.5,
        )
    elif kind == "alphabet2_vector_d4":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_value_width=4,
        )
    elif kind == "alphabet2_matrix_k4v4":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_value_width=4,
            slow_cnn_pole_matrix_key_width=4,
        )
    elif kind == "alphabet2_nonseparable_k4v4":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_value_width=4,
            slow_cnn_pole_matrix_key_width=4,
            slow_cnn_pole_independent_matrix_value=True,
        )
    elif kind == "alphabet2_vector_pole_r4":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_vector_width=4,
        )
    elif kind == "alphabet2_complex_vector_r4":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_vector_width=4,
            slow_cnn_pole_complex_vector_excitation=True,
        )
    elif kind == "alphabet2_complex_vector_r16":
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_vector_width=16,
            slow_cnn_pole_complex_vector_excitation=True,
        )
    elif kind in {
        "alphabet2_complex_query_r16",
        "alphabet2_token_rate_r16",
        "alphabet2_coordinate_read_r16",
    }:
        token_rate = kind == "alphabet2_token_rate_r16"
        config = AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_only",
            cnn_pole_memory=True,
            cnn_pole_interval=2,
            cnn_pole_modes=128,
            cnn_pole_evidence_width=512,
            cnn_pole_kernel_size=4,
            cnn_pole_beta_initial=0.01,
            cnn_pole_use_recurrence=False,
            cnn_pole_minimum_half_life=8.0,
            cnn_pole_maximum_half_life=4_096.0,
            slow_cnn_pole_memory=True,
            slow_cnn_pole_stride=1 if token_rate else 16,
            slow_cnn_pole_modes=128,
            slow_cnn_pole_evidence_width=512,
            slow_cnn_pole_kernel_size=4,
            slow_cnn_pole_upper_blocks=4,
            slow_cnn_pole_beta_initial=0.01,
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=16.0 if token_rate else 1.0,
            slow_cnn_pole_maximum_half_life=4_096.0 if token_rate else 256.0,
            slow_cnn_pole_query="token",
            slow_cnn_pole_query_rho=0.5,
            slow_cnn_pole_vector_width=16,
            slow_cnn_pole_complex_vector_excitation=True,
            slow_cnn_pole_complex_vector_query=True,
            slow_cnn_pole_coordinate_read=(
                kind == "alphabet2_coordinate_read_r16"
            ),
        )
    return AlphabetLM(config)


def _zero_memory(model: nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []

    def zero_output(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        real, imag = cast("tuple[Tensor, Tensor]", output)
        return torch.zeros_like(real), torch.zeros_like(imag)

    chunk_memories = [
        module for module in model.modules() if isinstance(module, ChunkedSemanticPoleMemory)
    ]
    edge_memories = [
        module for module in model.modules() if isinstance(module, SemanticEdgePoleMemory)
    ]
    sidecars = [
        module for module in model.modules() if isinstance(module, FixedPoleResidualSidecar)
    ]
    cnn_memories = [
        module for module in model.modules() if isinstance(module, CausalCNNPoleMemory)
    ]
    if cnn_memories:
        handles = [
            memory.synthesis.register_forward_hook(
                lambda _module, _inputs, output: torch.zeros_like(cast("Tensor", output))
            )
            for memory in cnn_memories
        ]
        for memory in cnn_memories:
            if isinstance(memory, SlowCausalCNNPoleMemory) and memory.extra_synthesis is not None:
                handles.append(
                    memory.extra_synthesis.register_forward_hook(
                        lambda _module, _inputs, output: torch.zeros_like(
                            cast("Tensor", output)
                        )
                    )
                )
        return handles
    writers = [memory.writer for memory in (*chunk_memories, *edge_memories)]
    if not writers:
        writers = [sidecar.writer for sidecar in sidecars]
    if not writers:
        writers = [
            module
            for module in model.modules()
            if isinstance(
                module,
                (PackedComplexLinear, GroupedPackedComplexLinear, TensorProductPoleMemory1D),
            )
        ]
    return [writer.register_forward_hook(zero_output) for writer in writers]


def _zero_slow_cnn_memory(model: nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM) or not isinstance(
        model.slow_cnn_pole_memory, SlowCausalCNNPoleMemory
    ):
        return []
    handles = [
        model.slow_cnn_pole_memory.synthesis.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(cast("Tensor", output))
        )
    ]
    if model.slow_cnn_pole_memory.extra_synthesis is not None:
        handles.append(
            model.slow_cnn_pole_memory.extra_synthesis.register_forward_hook(
                lambda _module, _inputs, output: torch.zeros_like(cast("Tensor", output))
            )
        )
    return handles


def _bypass_repeated_vector_poles(
    model: nn.Module,
    *,
    bank_index: int | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM) or model.repeated_vector_pole_memories is None:
        return []
    banks = model.repeated_vector_pole_memories
    selected = range(len(banks)) if bank_index is None else (bank_index,)

    def bypass(_module: nn.Module, inputs: tuple[object, ...], _output: object) -> object:
        return cast("tuple[Tensor, Tensor]", inputs)

    return [banks[index].register_forward_hook(bypass) for index in selected]


def _factorized_state_memories(
    model: nn.Module,
) -> list[FixedComplexPoleMemory1D]:
    if not isinstance(model, AlphabetLM) or model.repeated_vector_pole_memories is None:
        return []
    banks = list(model.repeated_vector_pole_memories)
    if not banks or not all(
        isinstance(bank, FactorizedTokenRateVectorPoleBlock) for bank in banks
    ):
        return []
    return [
        cast("FactorizedTokenRateVectorPoleBlock", bank).pole_memory
        for bank in banks
    ]


def _factorized_extra_coordinate_override(
    model: nn.Module,
) -> list[torch.utils.hooks.RemovableHandle]:
    memories = _factorized_state_memories(model)
    if not memories:
        return []
    banks = cast("AlphabetLM", model).repeated_vector_pole_memories
    if banks is None:
        raise RuntimeError("factorized VectorPole banks disappeared")

    def truncate(
        baseline_width: int,
        output: object,
    ) -> tuple[Tensor, Tensor]:
        real, imag = cast("tuple[Tensor, Tensor]", output)
        truncated_real = real.clone()
        truncated_imag = imag.clone()
        truncated_real[..., baseline_width:] = 0
        truncated_imag[..., baseline_width:] = 0
        return truncated_real, truncated_imag

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for bank, memory in zip(banks, memories, strict=True):
        baseline_width = cast(
            "FactorizedTokenRateVectorPoleBlock", bank
        ).baseline_width

        def override(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: object,
            *,
            active_width: int = baseline_width,
        ) -> tuple[Tensor, Tensor]:
            return truncate(active_width, output)

        handles.append(memory.register_forward_hook(override))
    return handles


@torch.no_grad()
def _calibrate_factorized_pca_bases(
    model: nn.Module,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> list[Tensor]:
    memories = _factorized_state_memories(model)
    if not memories:
        return []
    grams: list[Tensor] = []

    def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        real, imag = cast("tuple[Tensor, Tensor]", output)
        rows = torch.complex(real.float(), imag.float()).flatten(0, 2)
        grams.append(rows.mH @ rows / max(1, rows.shape[0]))

    handles = [memory.register_forward_hook(capture) for memory in memories]
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(sample)
    for handle in handles:
        handle.remove()
    if len(grams) != len(memories):
        raise RuntimeError("failed to calibrate every factorized VectorPole bank")
    return [torch.linalg.eigh(gram).eigenvectors for gram in grams]


def _factorized_pca_override(
    model: nn.Module,
    bases: list[Tensor],
    *,
    rank: int,
) -> list[torch.utils.hooks.RemovableHandle]:
    memories = _factorized_state_memories(model)
    if not memories:
        return []
    if len(bases) != len(memories):
        raise ValueError("PCA basis count must match factorized VectorPole banks")

    def project(basis: Tensor, output: object) -> tuple[Tensor, Tensor]:
        real, imag = cast("tuple[Tensor, Tensor]", output)
        active = basis[:, -min(rank, basis.shape[1]) :]
        state = torch.complex(real.float(), imag.float())
        coordinates = torch.matmul(state, active)
        projected = torch.matmul(coordinates, active.mH)
        return projected.real.to(real.dtype), projected.imag.to(imag.dtype)

    return [
        memory.register_forward_hook(
            lambda _module, _inputs, output, basis=basis: project(basis, output)
        )
        for memory, basis in zip(memories, bases, strict=True)
    ]


def _factorized_factor_read_override(
    model: nn.Module,
    *,
    shift: bool,
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM) or model.repeated_vector_pole_memories is None:
        return []
    modules: list[nn.Linear] = []
    for bank in model.repeated_vector_pole_memories:
        if not isinstance(bank, FactorizedTokenRateVectorPoleBlock):
            return []
        if bank.factor_read_real is None or bank.factor_read_imag is None:
            return []
        modules.extend((bank.factor_read_real, bank.factor_read_imag))

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        logits = cast("Tensor", output)
        return torch.roll(logits, shifts=1, dims=1) if shift else torch.zeros_like(logits)

    return [module.register_forward_hook(override) for module in modules]


def _mamba_outer_override(
    model: nn.Module,
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM) or model.repeated_vector_pole_memories is None:
        return []
    outputs: list[nn.Linear] = []
    for bank in model.repeated_vector_pole_memories:
        if not isinstance(bank, FactorizedTokenRateVectorPoleBlock):
            return []
        if bank.outer_output is None:
            return []
        outputs.append(bank.outer_output)

    def zero(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        return torch.zeros_like(cast("Tensor", output))

    return [module.register_forward_hook(zero) for module in outputs]


def _slow_query_override(
    model: nn.Module,
    *,
    shuffle: bool,
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory) or slow.query is None:
        return []

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        logits = cast("Tensor", output)
        return torch.roll(logits, shifts=1, dims=1) if shuffle else torch.zeros_like(logits)

    return [slow.query.register_forward_hook(override)]


def _slow_key_override(
    model: nn.Module,
    *,
    shift: bool,
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory) or slow.key is None:
        return []

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        logits = cast("Tensor", output)
        return torch.roll(logits, shifts=1, dims=1) if shift else torch.zeros_like(logits)

    return [slow.key.register_forward_hook(override)]


def _slow_value_override(
    model: nn.Module,
    *,
    mode: Literal["off", "shift", "time_mean"],
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory) or slow.value is None:
        return []

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        value = cast("Tensor", output)
        if mode == "off":
            return torch.zeros_like(value)
        if mode == "shift":
            return torch.roll(value, shifts=1, dims=1)
        return value.mean(dim=1, keepdim=True).expand_as(value)

    return [slow.value.register_forward_hook(override)]


def _slow_matrix_override(
    model: nn.Module,
    *,
    target: Literal["key", "query"],
    mode: Literal["off", "shift", "time_mean"],
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory):
        return []
    module = slow.matrix_key if target == "key" else slow.matrix_query
    if module is None:
        return []

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        value = cast("Tensor", output)
        if mode == "off":
            return torch.zeros_like(value)
        if mode == "shift":
            return torch.roll(value, shifts=1, dims=1)
        return value.mean(dim=1, keepdim=True).expand_as(value)

    return [module.register_forward_hook(override)]


def _slow_independent_value_override(
    model: nn.Module,
    *,
    mode: Literal["off", "shift", "time_mean"],
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory) or slow.matrix_value is None:
        return []

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        value = cast("Tensor", output)
        if mode == "off":
            return torch.zeros_like(value)
        if mode == "shift":
            return torch.roll(value, shifts=1, dims=1)
        return value.mean(dim=1, keepdim=True).expand_as(value)

    return [slow.matrix_value.register_forward_hook(override)]


def _slow_vector_pole_override(
    model: nn.Module,
    *,
    target: Literal["excitation", "excitation_imag", "query", "query_imag"],
    mode: Literal["off", "shift", "time_mean"],
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory):
        return []
    if target == "excitation":
        module = slow.vector_excitation
    elif target == "excitation_imag":
        module = slow.vector_excitation_imag
    elif target == "query_imag":
        module = slow.vector_query_imag
    else:
        module = slow.vector_query
    if module is None:
        return []

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        value = cast("Tensor", output)
        if mode == "off":
            return torch.zeros_like(value)
        if mode == "shift":
            return torch.roll(value, shifts=1, dims=1)
        return value.mean(dim=1, keepdim=True).expand_as(value)

    return [module.register_forward_hook(override)]


def _slow_transport_override(
    model: nn.Module,
    *,
    mode: Literal["off", "shift", "time_mean"],
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory) or slow.transport_selector is None:
        return []
    module = slow.transport_selector.output

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        value = cast("Tensor", output)
        if mode == "off":
            return torch.zeros_like(value)
        if mode == "shift":
            return torch.roll(value, shifts=1, dims=1)
        return value.mean(dim=1, keepdim=True).expand_as(value)

    return [module.register_forward_hook(override)]


def _slow_pole_reader_override(
    model: nn.Module,
    *,
    mode: Literal["off", "shift", "time_mean"],
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory) or slow.pole_specific_reader is None:
        return []

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        real, imag = cast("tuple[Tensor, Tensor]", output)
        if mode == "off":
            return torch.zeros_like(real), torch.zeros_like(imag)
        if mode == "shift":
            return torch.roll(real, shifts=1, dims=1), torch.roll(imag, shifts=1, dims=1)
        return (
            real.mean(dim=1, keepdim=True).expand_as(real),
            imag.mean(dim=1, keepdim=True).expand_as(imag),
        )

    return [slow.pole_specific_reader.register_forward_hook(override)]


def _slow_write_scheduler_override(
    model: nn.Module,
    *,
    mode: Literal["neutral", "shuffle"],
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory) or slow.write_scheduler is None:
        return []

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        logits = cast("Tensor", output)
        if mode == "neutral":
            return torch.zeros_like(logits)
        return torch.roll(logits, shifts=max(1, logits.shape[1] // 2), dims=1)

    return [slow.write_scheduler.register_forward_hook(override)]


def _slow_innovation_override(
    model: nn.Module,
    *,
    mode: Literal["neutral", "shuffle"],
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory) or slow.innovation_filter is None:
        return []
    if mode == "neutral":
        def restore_drive(
            _module: nn.Module,
            inputs: tuple[object, ...],
            _output: object,
        ) -> object:
            return cast("tuple[Tensor, Tensor]", inputs)

        return [slow.innovation_filter.register_forward_hook(restore_drive)]

    def shuffle_prediction(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> object:
        real, imag = cast("tuple[Tensor, Tensor]", output)
        shift = max(1, real.shape[1] // 2)
        return torch.roll(real, shifts=shift, dims=1), torch.roll(
            imag, shifts=shift, dims=1
        )

    return [slow.innovation_filter.predictor.register_forward_hook(shuffle_prediction)]


def _slow_semantic_clock_override(
    model: nn.Module,
    *,
    shuffle: bool,
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory) or slow.semantic_clock is None:
        return []

    def override(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        step = cast("Tensor", output)
        if not shuffle:
            return torch.ones_like(step)
        return torch.roll(step, shifts=max(1, step.shape[1] // 2), dims=1)

    return [slow.semantic_clock.register_forward_hook(override)]


def _complex_autocorrelation(
    real: Tensor,
    imag: Tensor,
    lag: int,
) -> float:
    left_real = real[:, :-lag].float()
    left_imag = imag[:, :-lag].float()
    right_real = real[:, lag:].float()
    right_imag = imag[:, lag:].float()
    correlation_real = (left_real * right_real + left_imag * right_imag).mean()
    correlation_imag = (left_real * right_imag - left_imag * right_real).mean()
    left_energy = left_real.square().add(left_imag.square()).mean()
    right_energy = right_real.square().add(right_imag.square()).mean()
    return float(
        correlation_real.square().add(correlation_imag.square()).sqrt()
        / (left_energy * right_energy).sqrt().clamp_min(1.0e-12)
    )


@torch.no_grad()
def _sidecar_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, object] | None:
    modules = [module for module in model.modules() if isinstance(module, FixedPoleResidualSidecar)]
    if not modules:
        return None
    branch_ratios: list[float] = []

    def capture(_module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        real, imag = cast("tuple[Tensor, Tensor]", inputs)
        output_real, output_imag = cast("tuple[Tensor, Tensor]", output)
        trunk_energy = real.float().square().add(imag.float().square()).mean()
        branch_energy = (
            (output_real - real).float().square() + (output_imag - imag).float().square()
        ).mean()
        branch_ratios.append(float(torch.sqrt(branch_energy / trunk_energy.clamp_min(1.0e-12))))

    handles = [module.register_forward_hook(capture) for module in modules]
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model.hidden(sample)
    for handle in handles:
        handle.remove()
    beta = torch.stack([module.beta.float() for module in modules])
    return {
        "branch_to_trunk_rms_mean": sum(branch_ratios) / len(branch_ratios),
        "branch_to_trunk_rms_by_layer": branch_ratios,
        "beta_mean": float(beta.mean()),
        "beta_abs_mean": float(beta.abs().mean()),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
        "beta_by_layer": beta.detach().cpu().tolist(),
    }


@torch.no_grad()
def _chunk_memory_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, object] | None:
    chunk_memory = model.chunk_memory
    if not isinstance(chunk_memory, ChunkedSemanticPoleMemory):
        return None
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        packed = model.analysis(model.embedding(sample))
        real, imag = packed.split(model.config.modes, dim=-1)
        memory_start = model.config.layers - model.config.chunk_upper_blocks
        for block in model.blocks[:memory_start]:
            real, imag = block(real, imag)
        memory = chunk_memory(real, imag)
        branch_ratios: list[float] = []
        for upper_index, block in enumerate(model.blocks[memory_start:]):
            injected_real, injected_imag = chunk_memory.inject(
                real,
                imag,
                memory[0],
                memory[1],
                upper_index,
            )
            trunk_energy = real.float().square().add(imag.float().square()).mean()
            branch_energy = (
                (injected_real - real).float().square() + (injected_imag - imag).float().square()
            ).mean()
            branch_ratios.append(float(torch.sqrt(branch_energy / trunk_energy.clamp_min(1.0e-12))))
            real, imag = block(injected_real, injected_imag)
    half_lives = math.log(2.0) / chunk_memory.memory.damping().float()
    return {
        "chunk_size": chunk_memory.chunk_size,
        "summary_modes": chunk_memory.summary_modes,
        "pole_modes": chunk_memory.memory.modes,
        "beta_by_upper_block": chunk_memory.beta.float().detach().cpu().tolist(),
        "beta_mean": float(chunk_memory.beta.float().mean()),
        "branch_to_trunk_rms_by_upper_block": branch_ratios,
        "branch_to_trunk_rms_mean": sum(branch_ratios) / len(branch_ratios),
        "half_life_chunks_min": float(half_lives.min()),
        "half_life_chunks_median": float(half_lives.median()),
        "half_life_chunks_max": float(half_lives.max()),
    }


@torch.no_grad()
def _semantic_edge_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, object] | None:
    edge_memory = model.semantic_edge_memory
    if not isinstance(edge_memory, SemanticEdgePoleMemory):
        return None
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        packed = model.analysis(model.embedding(sample))
        real, imag = packed.split(model.config.modes, dim=-1)
        memory_start = model.config.layers - model.config.semantic_edge_upper_blocks
        for block in model.blocks[:memory_start]:
            real, imag = block(real, imag)
        memory = edge_memory(real, imag)
        branch_ratios: list[float] = []
        for upper_index, block in enumerate(model.blocks[memory_start:]):
            injected_real, injected_imag = edge_memory.inject(
                real,
                imag,
                memory[0],
                memory[1],
                upper_index,
            )
            trunk_energy = real.float().square().add(imag.float().square()).mean()
            branch_energy = (
                (injected_real - real).float().square() + (injected_imag - imag).float().square()
            ).mean()
            branch_ratios.append(float(torch.sqrt(branch_energy / trunk_energy.clamp_min(1.0e-12))))
            real, imag = block(injected_real, injected_imag)
    half_lives = math.log(2.0) / edge_memory.memory.damping().float()
    weight_real = edge_memory.excitation.weight_real.float()
    weight_imag = edge_memory.excitation.weight_imag.float()
    gram = weight_real @ weight_real.T + weight_imag @ weight_imag.T
    identity = torch.eye(gram.shape[0], device=gram.device)
    return {
        "stride": edge_memory.stride,
        "pole_modes": edge_memory.memory.modes,
        "use_recurrence": edge_memory.use_recurrence,
        "beta_by_upper_block": edge_memory.beta.float().detach().cpu().tolist(),
        "beta_mean": float(edge_memory.beta.float().mean()),
        "branch_to_trunk_rms_by_upper_block": branch_ratios,
        "branch_to_trunk_rms_mean": sum(branch_ratios) / len(branch_ratios),
        "half_life_anchors_min": float(half_lives.min()),
        "half_life_anchors_median": float(half_lives.median()),
        "half_life_anchors_max": float(half_lives.max()),
        "excitation_row_gram_max_abs": float((gram - identity).abs().max()),
    }


@torch.no_grad()
def _cnn_pole_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, object] | None:
    memories = model.cnn_pole_memories
    if memories is None:
        return None
    branch_ratios: list[float] = []

    def capture(_module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        real, imag = cast("tuple[Tensor, Tensor]", inputs)
        output_real, output_imag = cast("tuple[Tensor, Tensor]", output)
        trunk_energy = real.float().square().add(imag.float().square()).mean()
        branch_energy = (
            (output_real - real).float().square() + (output_imag - imag).float().square()
        ).mean()
        branch_ratios.append(float(torch.sqrt(branch_energy / trunk_energy.clamp_min(1.0e-12))))

    handles = [memory.register_forward_hook(capture) for memory in memories]
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model.hidden(sample)
    for handle in handles:
        handle.remove()
    typed_memories = [cast("CausalCNNPoleMemory", memory) for memory in memories]
    beta = torch.stack([memory.beta.float() for memory in typed_memories])
    gram_errors = []
    for memory in typed_memories:
        weight = memory.analysis.weight.float()
        identity = torch.eye(weight.shape[0], device=weight.device)
        gram_errors.append(float((weight @ weight.T - identity).abs().max()))
    half_lives = [
        math.log(2.0) / memory.memory.damping().float() for memory in typed_memories
    ]
    return {
        "banks": len(memories),
        "interval": model.config.cnn_pole_interval,
        "kernel_size": model.config.cnn_pole_kernel_size,
        "pole_modes": model.config.cnn_pole_modes,
        "use_recurrence": model.config.cnn_pole_use_recurrence,
        "beta_by_bank": beta.detach().cpu().tolist(),
        "beta_mean": float(beta.mean()),
        "branch_to_trunk_rms_by_bank": branch_ratios,
        "branch_to_trunk_rms_mean": sum(branch_ratios) / len(branch_ratios),
        "analysis_row_gram_max_abs_by_bank": gram_errors,
        "half_life_tokens_min": min(float(value.min()) for value in half_lives),
        "half_life_tokens_median": sum(float(value.median()) for value in half_lives)
        / len(half_lives),
        "half_life_tokens_max": max(float(value.max()) for value in half_lives),
    }


@torch.no_grad()
def _slow_cnn_pole_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, object] | None:
    slow = model.slow_cnn_pole_memory
    if not isinstance(slow, SlowCausalCNNPoleMemory):
        return None
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        packed = model.analysis(model.embedding(sample))
        real, imag = packed.split(model.config.modes, dim=-1)
        memory_start = model.config.layers - model.config.slow_cnn_pole_upper_blocks
        for index, block in enumerate(model.blocks[:memory_start]):
            real, imag = block(real, imag)
            if (
                model.cnn_pole_memories is not None
                and (index + 1) % model.config.cnn_pole_interval == 0
            ):
                bank_index = (index + 1) // model.config.cnn_pole_interval - 1
                real, imag = model.cnn_pole_memories[bank_index](real, imag)
        def gate_metrics(gate: Tensor) -> dict[str, float]:
            active = gate.float()
            probability = active / active.sum(dim=-1, keepdim=True)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
            temporal_delta = (
                (active[:, 1:] - active[:, :-1]).abs().mean()
                if active.shape[1] > 1
                else torch.zeros((), device=active.device)
            )
            return {
                "gate_mean": float(active.mean()),
                "gate_std": float(active.std()),
                "gate_min": float(active.min()),
                "gate_max": float(active.max()),
                "normalized_entropy_mean": float(
                    entropy.mean() / math.log(active.shape[-1])
                ),
                "temporal_delta_mean": float(temporal_delta),
                "pole_usage_std": float(active.mean(dim=(0, 1)).std()),
            }

        packed_source = torch.cat((real, imag), dim=-1)
        full_anchors = packed_source.shape[1] // slow.stride
        anchor_source = packed_source[
            :, slow.stride - 1 : full_anchors * slow.stride : slow.stride
        ]
        write_metrics: dict[str, object] | None = None
        innovation_metrics: dict[str, object] | None = None
        write_gate: Tensor | None = None
        if slow.write_scheduler is not None:
            write_gate = slow.write_gate(anchor_source)
            active_gate = write_gate.float()
            sample_count = active_gate.shape[0] * active_gate.shape[1]
            gate_by_pole = active_gate.flatten(0, 1)
            effective_fraction = gate_by_pole.sum(dim=0).square() / (
                sample_count
                * gate_by_pole.square().sum(dim=0).clamp_min(1.0e-12)
            )
            write_metrics = {
                **gate_metrics(write_gate),
                "effective_write_fraction_mean": float(effective_fraction.mean()),
                "effective_write_fraction_min": float(effective_fraction.min()),
                "effective_write_fraction_max": float(effective_fraction.max()),
                "gate_below_0_5_fraction": float((active_gate < 0.5).float().mean()),
                "gate_above_1_5_fraction": float((active_gate > 1.5).float().mean()),
            }
        query_metrics: dict[str, float | str] | None = None
        if slow.query is not None:
            query_source = anchor_source if slow.query_mode == "anchor" else packed_source
            query_metrics = {
                "mode": slow.query_mode,
                **gate_metrics(slow.query_gate(query_source)),
            }
        key_metrics = gate_metrics(slow.key_gate(anchor_source)) if slow.key is not None else None
        vector_metrics: dict[str, object] | None = None
        transport_metrics: dict[str, float] | None = None
        clock_metrics: dict[str, float] | None = None
        def spectrum(rows: Tensor) -> tuple[Tensor, Tensor]:
            gram = rows.mH @ rows / max(1, rows.shape[0])
            eigenvalues = torch.linalg.eigvalsh(gram).real.clamp_min(0.0)
            effective_rank = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(
                1e-12
            )
            return eigenvalues, effective_rank

        if slow.vector_width > 1:
            if slow.pole_specific_reader is not None:
                transported_real, transported_imag = slow.pole_specific_reader(real, imag)
                excitation_magnitude = transported_real.float().square().add(
                    transported_imag.float().square()
                ).sqrt()
                excitation_extra_rms = float(
                    transported_real[..., 1:]
                    .float()
                    .square()
                    .add(transported_imag[..., 1:].float().square())
                    .mean()
                    .sqrt()
                )
                excitation_imag_rms = float(
                    transported_imag[..., 1:].float().square().mean().sqrt()
                )
                reader_metrics = {
                    "magnitude_mean": float(excitation_magnitude.mean()),
                    "magnitude_std": float(excitation_magnitude.std()),
                    "near_zero_fraction": float(
                        (excitation_magnitude < 0.1 * excitation_magnitude.mean())
                        .float()
                        .mean()
                    ),
                    "temporal_delta_mean": float(
                        (excitation_magnitude[:, 1:] - excitation_magnitude[:, :-1])
                        .abs()
                        .mean()
                    ),
                }
                if write_gate is not None and write_metrics is not None:
                    gated_real = transported_real * write_gate.unsqueeze(-1)
                    gated_imag = transported_imag * write_gate.unsqueeze(-1)
                    excitation_magnitude_by_pole = (
                        transported_real.float()
                        .square()
                        .add(transported_imag.float().square())
                        .mean(dim=-1)
                        .sqrt()
                    )
                    centered_excitation = excitation_magnitude_by_pole - (
                        excitation_magnitude_by_pole.mean(dim=1, keepdim=True)
                    )
                    centered_gate = write_gate.float() - write_gate.float().mean(
                        dim=1, keepdim=True
                    )
                    correlation = (centered_excitation * centered_gate).mean(dim=1) / (
                        centered_excitation.square().mean(dim=1).sqrt()
                        * centered_gate.square().mean(dim=1).sqrt()
                    ).clamp_min(1.0e-12)
                    write_metrics.update(
                        {
                            "excitation_temporal_variance_mean": float(
                                excitation_magnitude_by_pole.var(dim=1).mean()
                            ),
                            "gate_temporal_variance_mean": float(
                                write_gate.float().var(dim=1).mean()
                            ),
                            "excitation_gate_correlation_mean": float(
                                correlation.mean()
                            ),
                            "excitation_gate_correlation_abs_mean": float(
                                correlation.abs().mean()
                            ),
                        }
                    )
                    lags = (1, 2, 4, 8, 16)
                    write_metrics["excitation_autocorrelation"] = {
                        str(lag): _complex_autocorrelation(
                            transported_real, transported_imag, lag
                        )
                        for lag in lags
                    }
                    write_metrics["gated_excitation_autocorrelation"] = {
                        str(lag): _complex_autocorrelation(
                            gated_real, gated_imag, lag
                        )
                        for lag in lags
                    }
                    transported_real, transported_imag = gated_real, gated_imag
                if slow.innovation_filter is not None:
                    innovation = slow.innovation_filter
                    predicted_real, predicted_imag = innovation.predictor(
                        transported_real, transported_imag
                    )
                    innovation_real, innovation_imag = innovation(
                        transported_real, transported_imag
                    )
                    strength = innovation.strength().float()
                    weight_real, weight_imag = innovation.predictor.normalized_weights()
                    lag_magnitude = weight_real.float().square().add(
                        weight_imag.float().square()
                    ).mean(dim=(0, 1)).sqrt()
                    excitation_rms = transported_real.float().square().add(
                        transported_imag.float().square()
                    ).mean().sqrt()
                    prediction_rms = predicted_real.float().square().add(
                        predicted_imag.float().square()
                    ).mean().sqrt()
                    innovation_rms = innovation_real.float().square().add(
                        innovation_imag.float().square()
                    ).mean().sqrt()
                    lags = (1, 2, 4, 8, 16)
                    innovation_metrics = {
                        "strength_mean": float(strength.mean()),
                        "strength_min": float(strength.min()),
                        "strength_max": float(strength.max()),
                        "active_pole_fraction": float((strength > 0.0).float().mean()),
                        "prediction_to_excitation_rms": float(
                            prediction_rms / excitation_rms.clamp_min(1.0e-12)
                        ),
                        "innovation_to_excitation_rms": float(
                            innovation_rms / excitation_rms.clamp_min(1.0e-12)
                        ),
                        "predictor_lag_magnitude": lag_magnitude.detach().cpu().tolist(),
                        "excitation_autocorrelation": {
                            str(lag): _complex_autocorrelation(
                                transported_real, transported_imag, lag
                            )
                            for lag in lags
                        },
                        "innovation_autocorrelation": {
                            str(lag): _complex_autocorrelation(
                                innovation_real, innovation_imag, lag
                            )
                            for lag in lags
                        },
                    }
                    transported_real, transported_imag = (
                        innovation_real,
                        innovation_imag,
                    )
            else:
                reader_metrics = None
                drive_real, drive_imag = slow.pole_drive(real, imag)
                anchors = (
                    drive_real[:, slow.stride - 1 : full_anchors * slow.stride : slow.stride],
                    drive_imag[:, slow.stride - 1 : full_anchors * slow.stride : slow.stride],
                )
                excitation_real = slow.vector_excitation_axes(anchor_source)
                excitation_imag = slow.vector_excitation_imag_axes(anchor_source)
                transported_real = (
                    anchors[0].unsqueeze(-1) * excitation_real
                    - anchors[1].unsqueeze(-1) * excitation_imag
                )
                transported_imag = (
                    anchors[0].unsqueeze(-1) * excitation_imag
                    + anchors[1].unsqueeze(-1) * excitation_real
                )
                excitation_extra_rms = float(
                    excitation_real[..., 1:]
                    .float()
                    .square()
                    .add(excitation_imag[..., 1:].float().square())
                    .mean()
                    .sqrt()
                )
                excitation_imag_rms = float(
                    excitation_imag[..., 1:].float().square().mean().sqrt()
                )
            damping_control = None
            clock_step = None
            if slow.transport_selector is not None:
                anchor_real = real[
                    :, slow.stride - 1 : full_anchors * slow.stride : slow.stride
                ]
                anchor_imag = imag[
                    :, slow.stride - 1 : full_anchors * slow.stride : slow.stride
                ]
                damping_control = slow.transport_selector(anchor_real, anchor_imag)
                active_control = damping_control.float()
                transport_metrics = {
                    "scale": float(slow.transport_selector.scale()),
                    "control_mean": float(active_control.mean()),
                    "control_std": float(active_control.std()),
                    "control_abs_mean": float(active_control.abs().mean()),
                    "temporal_delta_mean": float(
                        (active_control[:, 1:] - active_control[:, :-1]).abs().mean()
                    ),
                }
            if slow.semantic_clock is not None:
                clock_step = slow.semantic_clock(anchor_source)
                active_step = clock_step.float()
                clock_metrics = {
                    "step_mean": float(active_step.mean()),
                    "step_std": float(active_step.std()),
                    "step_min": float(active_step.min()),
                    "step_max": float(active_step.max()),
                    "hold_below_0_1_fraction": float(
                        (active_step < 0.1).float().mean()
                    ),
                    "hold_below_0_5_fraction": float(
                        (active_step < 0.5).float().mean()
                    ),
                    "semantic_time_ratio": float(active_step.sum() / active_step.numel()),
                    "temporal_delta_mean": float(
                        (active_step[:, 1:] - active_step[:, :-1]).abs().mean()
                    ),
                }
            state_real, state_imag = slow.memory(
                transported_real,
                transported_imag,
                damping_control=damping_control,
                clock_step=clock_step,
            )
            rows = torch.complex(state_real.float(), state_imag.float()).flatten(0, 2)
            eigenvalues, effective_rank = spectrum(rows)
            coordinate_energy = state_real.float().square().add(state_imag.float().square()).mean(
                dim=(0, 1, 2)
            )
            scalar_query = slow.query_gate(packed_source)
            query_real, query_imag = slow.vector_query_components(
                packed_source, scalar_query
            )
            vector_metrics = {
                "vector_width": slow.vector_width,
                "excitation_extra_rms": excitation_extra_rms,
                "excitation_imag_rms": excitation_imag_rms,
                "query_extra_rms": float(
                    query_real[..., 1:]
                    .float()
                    .square()
                    .add(query_imag[..., 1:].float().square())
                    .mean()
                    .sqrt()
                ),
                "query_imag_rms": float(
                    query_imag.float().square().mean().sqrt()
                ),
                "query_base_energy_fraction": float(
                    query_real[..., 0]
                    .float()
                    .square()
                    .add(query_imag[..., 0].float().square())
                    .sum()
                    / query_real
                    .float()
                    .square()
                    .add(query_imag.float().square())
                    .sum()
                    .clamp_min(1e-12)
                ),
                "state_effective_rank": float(effective_rank),
                "state_eigenvalues": eigenvalues.detach().cpu().tolist(),
                "state_coordinate_rms": coordinate_energy.sqrt().detach().cpu().tolist(),
                "dominant_energy_fraction": float(
                    coordinate_energy.max() / coordinate_energy.sum().clamp_min(1e-12)
                ),
            }
            if reader_metrics is not None:
                vector_metrics["pole_specific_reader"] = reader_metrics
        elif slow.value_width > 1:
            drive_real, drive_imag = slow.pole_drive(real, imag)
            anchors = (
                drive_real[:, slow.stride - 1 : full_anchors * slow.stride : slow.stride],
                drive_imag[:, slow.stride - 1 : full_anchors * slow.stride : slow.stride],
            )
            value = slow.anchor_value(anchor_source)
            if slow.matrix_key_width > 1:
                matrix_key = slow.matrix_key_axes(anchor_source)
                matrix_value = slow.matrix_value_axes(anchor_source, value)
                state_real, state_imag = slow.memory(
                    anchors[0].unsqueeze(-1).unsqueeze(-1)
                    * matrix_key.unsqueeze(-1)
                    * matrix_value.unsqueeze(-3),
                    anchors[1].unsqueeze(-1).unsqueeze(-1)
                    * matrix_key.unsqueeze(-1)
                    * matrix_value.unsqueeze(-3),
                )
                complex_state = torch.complex(state_real.float(), state_imag.float())
                key_rows = complex_state.permute(0, 1, 2, 4, 3).reshape(
                    -1, slow.matrix_key_width
                )
                value_rows = complex_state.reshape(-1, slow.value_width)
                key_eigenvalues, key_rank = spectrum(key_rows)
                value_eigenvalues, value_rank = spectrum(value_rows)
                scalar_query = slow.query_gate(packed_source)
                matrix_query = slow.matrix_query_axes(packed_source, scalar_query)
                vector_metrics = {
                    "value_width": slow.value_width,
                    "matrix_key_width": slow.matrix_key_width,
                    "value_extra_rms": float(value[..., 1:].float().square().mean().sqrt()),
                    "matrix_key_extra_rms": float(
                        matrix_key[..., 1:].float().square().mean().sqrt()
                    ),
                    "matrix_query_extra_rms": float(
                        matrix_query[..., 1:].float().square().mean().sqrt()
                    ),
                    "independent_value_delta_rms": float(
                        (matrix_value - value.unsqueeze(-2)).float().square().mean().sqrt()
                    ),
                    "state_key_effective_rank": float(key_rank),
                    "state_key_eigenvalues": key_eigenvalues.detach().cpu().tolist(),
                    "state_value_effective_rank": float(value_rank),
                    "state_value_eigenvalues": value_eigenvalues.detach().cpu().tolist(),
                }
            else:
                state_real, state_imag = slow.memory(
                    anchors[0].unsqueeze(-1) * value.unsqueeze(-2),
                    anchors[1].unsqueeze(-1) * value.unsqueeze(-2),
                )
                coordinate_rms = torch.sqrt(
                    state_real.float().square().add(state_imag.float().square()).mean(
                        dim=(0, 1, 2)
                    )
                )
                rows = torch.complex(state_real.float(), state_imag.float()).flatten(0, 2)
                eigenvalues, effective_rank = spectrum(rows)
                vector_metrics = {
                    "value_width": slow.value_width,
                    "value_extra_rms": float(value[..., 1:].float().square().mean().sqrt()),
                    "state_coordinate_rms": coordinate_rms.detach().cpu().tolist(),
                    "state_effective_rank": float(effective_rank),
                    "state_eigenvalues": eigenvalues.detach().cpu().tolist(),
                }
        memory = slow(real, imag)
        branch_ratios: list[float] = []
        for upper_index, block in enumerate(model.blocks[memory_start:]):
            injected_real, injected_imag = slow.inject(
                real,
                imag,
                memory[0],
                memory[1],
                upper_index,
            )
            trunk_energy = real.float().square().add(imag.float().square()).mean()
            branch_energy = (
                (injected_real - real).float().square()
                + (injected_imag - imag).float().square()
            ).mean()
            branch_ratios.append(float(torch.sqrt(branch_energy / trunk_energy.clamp_min(1e-12))))
            real, imag = block(injected_real, injected_imag)
            block_index = memory_start + upper_index
            if (
                model.cnn_pole_memories is not None
                and (block_index + 1) % model.config.cnn_pole_interval == 0
            ):
                bank_index = (block_index + 1) // model.config.cnn_pole_interval - 1
                real, imag = model.cnn_pole_memories[bank_index](real, imag)
    weight = slow.analysis.weight.float()
    identity = torch.eye(weight.shape[0], device=weight.device)
    half_lives = math.log(2.0) / slow.memory.damping().float()
    payload: dict[str, object] = {
        "stride": slow.stride,
        "pole_modes": slow.memory.modes,
        "use_recurrence": slow.use_recurrence,
        "beta_by_upper_block": slow.beta.float().detach().cpu().tolist(),
        "beta_mean": float(slow.beta.float().mean()),
        "branch_to_trunk_rms_by_upper_block": branch_ratios,
        "branch_to_trunk_rms_mean": sum(branch_ratios) / len(branch_ratios),
        "half_life_tokens_min": float(half_lives.min()) * slow.stride,
        "half_life_tokens_median": float(half_lives.median()) * slow.stride,
        "half_life_tokens_max": float(half_lives.max()) * slow.stride,
        "analysis_row_gram_max_abs": float((weight @ weight.T - identity).abs().max()),
    }
    if query_metrics is not None:
        payload["query"] = query_metrics
    if key_metrics is not None:
        payload["key"] = key_metrics
    if vector_metrics is not None:
        payload["vector_state"] = vector_metrics
    if transport_metrics is not None:
        payload["dynamic_transport"] = transport_metrics
    if write_metrics is not None:
        payload["write_scheduler"] = write_metrics
    if innovation_metrics is not None:
        payload["innovation"] = innovation_metrics
    if clock_metrics is not None:
        payload["semantic_clock"] = clock_metrics
    return payload


@torch.no_grad()
def _repeated_vector_pole_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, object] | None:
    banks = model.repeated_vector_pole_memories
    if banks is None:
        return None
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    beta: list[float] = []
    branch_ratios: list[float] = []
    state_ranks: list[float] = []
    factor_state_ranks: list[float] = []
    factor_read_delta_rms: list[float] = []
    reader_temporal_delta: list[float] = []
    query_extra_rms: list[float] = []
    half_life_medians: list[float] = []
    instantaneous_write_ranks: list[float] = []
    new_coordinate_rms: list[float] = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        packed = model.analysis(model.embedding(sample))
        real, imag = packed.split(model.config.modes, dim=-1)
        bank_index = 0
        for index, block in enumerate(model.blocks):
            real, imag = block(real, imag)
            if (index + 1) % model.config.repeated_vector_pole_interval:
                continue
            bank = banks[bank_index]
            typed_bank = cast(
                "TokenRateVectorPoleBlock | FactorizedTokenRateVectorPoleBlock",
                bank,
            )
            packed_source = torch.cat((real, imag), dim=-1)
            if isinstance(bank, FactorizedTokenRateVectorPoleBlock):
                coefficient = bank.reader(real, imag)
                if bank.extra_reader is not None:
                    extra_coefficient = bank.extra_reader(real, imag)
                    coefficient = (
                        torch.cat((coefficient[0], extra_coefficient[0]), dim=-1),
                        torch.cat((coefficient[1], extra_coefficient[1]), dim=-1),
                    )
                content_basis = bank.content_basis(packed_source)
                collapsed_excitation = bank.complex_factor_product(
                    coefficient[0],
                    coefficient[1],
                    content_basis[0],
                    content_basis[1],
                )
                if bank.retain_factor_state:
                    factor_excitation = bank.factor_state_drive(
                        packed_source,
                        coefficient[0],
                        coefficient[1],
                        content_basis[0],
                        content_basis[1],
                    )
                    factor_state = bank.pole_memory(*factor_excitation)
                    factor_rows = torch.complex(
                        factor_state[0].float(), factor_state[1].float()
                    ).permute(0, 1, 2, 4, 3).reshape(-1, bank.write_rank)
                    factor_gram = factor_rows.mH @ factor_rows / max(
                        1, factor_rows.shape[0]
                    )
                    factor_eigenvalues = torch.linalg.eigvalsh(
                        factor_gram
                    ).real.clamp_min(0.0)
                    factor_state_ranks.append(
                        float(
                            factor_eigenvalues.sum().square()
                            / factor_eigenvalues.square().sum().clamp_min(1.0e-12)
                        )
                    )
                    factor_read = bank.factor_read(packed_source)
                    if factor_read is None:
                        factor_read_delta_rms.append(0.0)
                    else:
                        factor_read_delta_rms.append(
                            float(
                                (factor_read[0] - 1.0)
                                .float()
                                .square()
                                .add(factor_read[1].float().square())
                                .mean()
                                .sqrt()
                            )
                        )
                    excitation_real, excitation_imag = collapsed_excitation
                    state_real, state_imag = bank.contract_factor_state(
                        packed_source,
                        *factor_state,
                    )
                else:
                    excitation_real, excitation_imag = collapsed_excitation
                    state_real, state_imag = bank.pole_memory(
                        excitation_real, excitation_imag
                    )
                query_factors = bank.query_factors(packed_source)
                query_basis = bank.query_basis(packed_source)
                query_real, query_imag = bank.complex_factor_product(
                    query_factors[0],
                    query_factors[1],
                    query_basis[0],
                    query_basis[1],
                )
                singular = torch.linalg.svdvals(
                    torch.complex(
                        excitation_real[0, :32].float(),
                        excitation_imag[0, :32].float(),
                    )
                )
                instantaneous_write_ranks.append(
                    float(
                        (
                            singular.sum(dim=-1).square()
                            / singular.square().sum(dim=-1).clamp_min(1.0e-12)
                        ).mean()
                    )
                )
                if bank.vector_width > bank.write_rank:
                    new_coordinate_rms.append(
                        float(
                            excitation_real[..., bank.write_rank :]
                            .float()
                            .square()
                            .add(
                                excitation_imag[..., bank.write_rank :]
                                .float()
                                .square()
                            )
                            .mean()
                            .sqrt()
                        )
                    )
                else:
                    new_coordinate_rms.append(0.0)
            else:
                dense_bank = cast("TokenRateVectorPoleBlock", bank)
                excitation_real, excitation_imag = dense_bank.reader(real, imag)
                query_real, query_imag = dense_bank.query_components(packed_source)
                state_real, state_imag = typed_bank.pole_memory(
                    excitation_real, excitation_imag
                )
            rows = torch.complex(state_real.float(), state_imag.float()).flatten(0, 2)
            gram = rows.mH @ rows / max(1, rows.shape[0])
            eigenvalues = torch.linalg.eigvalsh(gram).real.clamp_min(0.0)
            state_ranks.append(
                float(
                    eigenvalues.sum().square()
                    / eigenvalues.square().sum().clamp_min(1.0e-12)
                )
            )
            magnitude = excitation_real.float().square().add(
                excitation_imag.float().square()
            ).sqrt()
            reader_temporal_delta.append(
                float((magnitude[:, 1:] - magnitude[:, :-1]).abs().mean())
            )
            query_extra_rms.append(
                float(
                    query_real[..., 1:]
                    .float()
                    .square()
                    .add(query_imag[..., 1:].float().square())
                    .mean()
                    .sqrt()
                )
            )
            half_lives = math.log(2.0) / typed_bank.pole_memory.damping().float()
            half_life_medians.append(float(half_lives.median()))
            output_real, output_imag = typed_bank(real, imag)
            trunk_energy = real.float().square().add(imag.float().square()).mean()
            branch_energy = (output_real - real).float().square().add(
                (output_imag - imag).float().square()
            ).mean()
            branch_ratios.append(
                float(torch.sqrt(branch_energy / trunk_energy.clamp_min(1.0e-12)))
            )
            beta.append(float(typed_bank.beta))
            real, imag = output_real, output_imag
            bank_index += 1
    return {
        "banks": len(banks),
        "interval": model.config.repeated_vector_pole_interval,
        "pole_modes": model.config.repeated_vector_pole_modes,
        "vector_width": model.config.repeated_vector_pole_width,
        "beta_by_bank": beta,
        "branch_to_trunk_rms_by_bank": branch_ratios,
        "state_effective_rank_by_bank": state_ranks,
        "factor_state_effective_rank_by_bank": factor_state_ranks,
        "factor_read_delta_rms_by_bank": factor_read_delta_rms,
        "reader_temporal_delta_by_bank": reader_temporal_delta,
        "query_extra_rms_by_bank": query_extra_rms,
        "half_life_median_by_bank": half_life_medians,
        "instantaneous_write_effective_rank_by_bank": instantaneous_write_ranks,
        "new_coordinate_rms_by_bank": new_coordinate_rms,
    }


@torch.no_grad()
def _query_readout_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, float] | None:
    modules = [
        module for module in model.modules() if isinstance(module, QueryConditionedLowRankReadout)
    ]
    if not modules:
        return None
    rows: list[tuple[float, float, float, float]] = []

    def capture(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        readout = cast("QueryConditionedLowRankReadout", module)
        query_real, query_imag, _state_real, _state_imag, base_real, base_imag = cast(
            "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]", inputs
        )
        output_real, output_imag = cast("tuple[Tensor, Tensor]", output)
        unit_real, unit_imag = readout.query_norm(query_real, query_imag)
        query = functional.silu(readout.query(torch.cat((unit_real, unit_imag), dim=-1))).float()
        base_energy = base_real.float().square().add(base_imag.float().square()).mean()
        branch_energy = (
            (output_real - base_real).float().square() + (output_imag - base_imag).float().square()
        ).mean()
        rows.append(
            (
                float(torch.sqrt(branch_energy / base_energy.clamp_min(1.0e-12))),
                float(query.square().mean().sqrt()),
                float(query.std()),
                float(readout.scale()),
            )
        )

    handles = [module.register_forward_hook(capture) for module in modules]
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model.hidden(sample)
    for handle in handles:
        handle.remove()
    labels = ("branch_to_base_rms", "query_rms", "query_std", "scale")
    return {
        f"{label}_mean": sum(row[index] for row in rows) / len(rows)
        for index, label in enumerate(labels)
    }


@torch.no_grad()
def _delta_select_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, float] | None:
    pairs = [
        (block.decay_selector, block.memory)
        for block in model.blocks
        if isinstance(block.decay_selector, LowRankDecaySelector)
    ]
    if not pairs:
        return None
    rows: list[tuple[float, float, float, float, float, float, float, float, float]] = []

    def capture(
        selector: LowRankDecaySelector,
        memory: nn.Module,
        control: Tensor,
    ) -> None:
        fixed_memory = cast("FixedComplexPoleMemory1D", memory)
        active = control.float()
        base = fixed_memory.damping().float().view(1, 1, -1)
        effective = fixed_memory.minimum_damping + functional.softplus(
            fixed_memory.raw_damping.float().view(1, 1, -1) + active
        )
        frequency = fixed_memory.frequency().float().view(1, 1, -1)
        relative = effective / base - 1.0
        rows.append(
            (
                float(active.square().mean().sqrt()),
                float(active.std()),
                float(active.abs().max()),
                float(selector.scale()),
                float(relative.square().mean().sqrt()),
                float(
                    (
                        active.abs()
                        > 0.95 * selector.control_bound * selector.scale().to(active.dtype)
                    )
                    .float()
                    .mean()
                ),
                float(effective.min()),
                float(torch.exp(-effective).max()),
                float(torch.sqrt(effective.square() + frequency.square()).min()),
            )
        )

    handles = [
        selector.register_forward_hook(
            lambda module, _inputs, output, memory=memory: capture(
                cast("LowRankDecaySelector", module),
                memory,
                cast("Tensor", cast("object", output)),
            )
        )
        for selector, memory in pairs
    ]
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model.hidden(sample)
    for handle in handles:
        handle.remove()
    labels = (
        "control_rms",
        "control_std",
        "control_abs_max",
        "scale",
        "relative_damping_change_rms",
        "saturation_fraction",
        "effective_damping_min",
        "discrete_decay_abs_max",
        "continuous_pole_abs_min",
    )
    return {
        f"{label}_mean": sum(row[index] for row in rows) / len(rows)
        for index, label in enumerate(labels)
    }


@torch.no_grad()
def _tensorpole_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, object] | None:
    modules = [
        module for module in model.modules() if isinstance(module, TensorProductPoleMemory1D)
    ]
    if not modules:
        return None
    memory_to_drive: list[float] = []

    def capture(_module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        drive_real, drive_imag, *_rest = cast("tuple[Tensor, Tensor, object]", inputs)
        memory_real, memory_imag = cast("tuple[Tensor, Tensor]", output)
        drive_energy = drive_real.float().square().add(drive_imag.float().square()).mean()
        memory_energy = memory_real.float().square().add(memory_imag.float().square()).mean()
        memory_to_drive.append(float(torch.sqrt(memory_energy / drive_energy.clamp_min(1.0e-12))))

    handles = [module.register_forward_hook(capture) for module in modules]
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model.hidden(sample)
    for handle in handles:
        handle.remove()
    half_lives = torch.stack([module.half_lives().float() for module in modules])
    frequencies = torch.stack([module.frequency().float() for module in modules])
    write_energy = torch.stack(
        [
            module.write_real.float().square().add(module.write_imag.float().square()).sum(dim=-1)
            for module in modules
        ]
    )
    read_energy = torch.stack(
        [
            module.read_real.float().square().add(module.read_imag.float().square()).sum(dim=-1)
            for module in modules
        ]
    )
    write_mode_rms = torch.stack(
        [
            module.write_real.float()
            .square()
            .add(module.write_imag.float().square())
            .mean(dim=0)
            .sqrt()
            for module in modules
        ]
    )
    read_mode_rms = torch.stack(
        [
            module.read_real.float()
            .square()
            .add(module.read_imag.float().square())
            .mean(dim=0)
            .sqrt()
            for module in modules
        ]
    )
    transport_abs = torch.stack(
        [
            (
                (module.read_real * module.write_real - module.read_imag * module.write_imag)
                .float()
                .square()
                + (module.read_real * module.write_imag + module.read_imag * module.write_real)
                .float()
                .square()
            )
            .sqrt()
            .mean(dim=0)
            for module in modules
        ]
    )
    return {
        "memory_to_drive_rms_mean": sum(memory_to_drive) / len(memory_to_drive),
        "memory_to_drive_rms_by_layer": memory_to_drive,
        "half_life_min": float(half_lives.min()),
        "half_life_median": float(half_lives.median()),
        "half_life_max": float(half_lives.max()),
        "frequency_abs_mean": float(frequencies.abs().mean()),
        "write_row_energy_mean": float(write_energy.mean()),
        "read_row_energy_mean": float(read_energy.mean()),
        "half_lives_by_layer": half_lives.detach().cpu().tolist(),
        "frequencies_by_layer": frequencies.detach().cpu().tolist(),
        "write_mode_rms_by_layer": write_mode_rms.detach().cpu().tolist(),
        "read_mode_rms_by_layer": read_mode_rms.detach().cpu().tolist(),
        "transport_abs_mean_by_layer_mode": transport_abs.detach().cpu().tolist(),
    }


@torch.no_grad()
def _dynamic_write_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> dict[str, object] | None:
    modules = [module for module in model.modules() if isinstance(module, DynamicLowRankWrite)]
    if not modules:
        return None
    rows: list[tuple[float, float, float, float, float, float]] = []

    def capture(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        write = cast("DynamicLowRankWrite", module)
        real, imag, base_real, base_imag = cast("tuple[Tensor, Tensor, Tensor, Tensor]", inputs)
        output_real, output_imag = cast("tuple[Tensor, Tensor]", output)
        unit_real, unit_imag = write.norm(real, imag)
        gate = functional.silu(write.gate(torch.cat((unit_real, unit_imag), dim=-1))).float()
        content_real, content_imag = write.content(unit_real, unit_imag)
        content_energy = content_real.float().square().add(content_imag.float().square()).mean()
        base_energy = base_real.float().square().add(base_imag.float().square()).mean()
        branch_energy = (
            (output_real - base_real).float().square() + (output_imag - base_imag).float().square()
        ).mean()
        rows.append(
            (
                float(torch.sqrt(branch_energy / base_energy.clamp_min(1.0e-12))),
                float(gate.square().mean().sqrt()),
                float(gate.std()),
                float(gate.mean()),
                float(content_energy.sqrt()),
                float(write.scale()),
            )
        )

    handles = [module.register_forward_hook(capture) for module in modules]
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model.hidden(sample)
    for handle in handles:
        handle.remove()
    labels = (
        "branch_to_static_rms",
        "gate_rms",
        "gate_std",
        "gate_mean",
        "content_rms",
        "scale",
    )
    payload: dict[str, object] = {
        f"{label}_mean": sum(row[index] for row in rows) / len(rows)
        for index, label in enumerate(labels)
    }
    payload.update(
        {f"{label}_by_layer": [row[index] for row in rows] for index, label in enumerate(labels)}
    )
    return payload


def _loss_sum(model: nn.Module, inputs: Tensor, labels: Tensor, pad_id: int) -> tuple[float, int]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=pad_id, reduction="sum"
        )
    count = int((labels != pad_id).sum())
    return float(loss), count


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataset: TokenBlockDataset,
    *,
    segment: int,
    token_limit: int,
    sequence_limit: int | None,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()
    sequence_index = 0
    token_batch = 16_384
    segment_batch = max(1, min(256, token_batch // segment))
    pending_inputs: list[Tensor] = []
    pending_labels: list[Tensor] = []
    maximum_sequences = min(len(dataset), sequence_limit or len(dataset))
    while sequence_index < maximum_sequences and (
        sequence_limit is not None or total_tokens < token_limit
    ):
        tokens = dataset[sequence_index]
        sequence_index += 1
        inputs = tokens[:-1].reshape(-1, segment)
        labels = tokens[1:].reshape(-1, segment)
        pending_inputs.extend(inputs)
        pending_labels.extend(labels)
        while len(pending_inputs) >= segment_batch or (
            sequence_index == maximum_sequences and pending_inputs
        ):
            take = min(segment_batch, len(pending_inputs))
            active_inputs = torch.stack(pending_inputs[:take]).to(device, non_blocking=True)
            active_labels = torch.stack(pending_labels[:take]).to(device, non_blocking=True)
            del pending_inputs[:take], pending_labels[:take]
            loss, count = _loss_sum(model, active_inputs, active_labels, dataset.manifest.pad_id)
            total_loss += loss
            total_tokens += count
            if sequence_limit is None and total_tokens >= token_limit:
                break
    torch.cuda.synchronize()
    return {
        "loss": total_loss / total_tokens,
        "tokens": total_tokens,
        "segment": segment,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=(
            "legacy",
            "grouped",
            "wide",
            "qread",
            "dense_fixed",
            "dense_delta",
            "tensorpole",
            "dense_dynamic_write_r4",
            "dense_local_only",
            "dense_local_sidecar",
            "dense_local_sidecar_normalized",
            "dense_local_sidecar_normalized_no_recurrence",
            "chunked_semantic_p128",
            "semantic_edge_p128",
            "semantic_edge_p128_no_recurrence",
            "cnn_pole_p128_6bank",
            "cnn_pole_p128_6bank_no_recurrence",
            "cnn_pole_p128_6bank_slow_p128",
            "cnn_pole_p128_6bank_slow_p128_no_recurrence",
            "alphabet2_anchor_q",
            "alphabet2_token_q",
            "alphabet2_qk",
            "alphabet2_vector_d4",
            "alphabet2_matrix_k4v4",
            "alphabet2_nonseparable_k4v4",
            "alphabet2_vector_pole_r4",
            "alphabet2_complex_vector_r4",
            "alphabet2_complex_vector_r16",
            "alphabet2_complex_query_r16",
            "alphabet2_token_rate_r16",
            "alphabet2_coordinate_read_r16",
            "alphabet2_dynamic_transport_r16",
            "alphabet2_pole_reader_r16",
            "alphabet2_pole_reader_write_scheduler_r16",
            "alphabet2_pole_reader_innovation_r16",
            "alphabet2_pole_reader_semantic_clock_r16",
            "alphabet2_repeated_vector_pole_p32r4",
            "alphabet2_factorized_vector_pole_p32r4_interface",
            "alphabet2_factorized_vector_pole_p32r32",
            "alphabet2_factorized_vector_pole_p32r32_js4",
            "alphabet2_factorized_vector_pole_p32r32_j8q4",
            "alphabet2_factorized_vector_pole_p32r32_j4q8",
            "alphabet2_factorized_vector_pole_p32r32_j8q8",
            "alphabet2_retained_factor_fixed_p32r32_js4",
            "alphabet2_retained_factor_learned_p32r32_js4",
            "alphabet2_mamba_outer_post_p32j4r32",
            "alphabet2_mamba_outer_direct_p32j4r32",
            "alphabet2_mamba_outer_gate_p32j4r32",
            "alphabet2_mamba_outer_both_p32j4r32",
            "alphabet2_write_row_specific_p32j4r32",
            "alphabet2_write_shared_outer_p32j4r32",
            "alphabet2_write_pole_outer_p32j4r32",
            "mamba",
            "mamba2",
            "mamba1_49m",
            "mamba2_48m",
            "laplace_mamba",
            "laplace_mamba_complex_highway",
            "alphabet2_image_postfusion",
            "alphabet2_vector_image_postfusion",
            "alphabet2_content_aligned_image_postfusion",
            "alphabet2_content_aligned_j2",
            "alphabet2_content_aligned_j4",
            "alphabet2_multi_observer_image_postfusion",
            "alphabet2_read_adapter_image_postfusion",
            "alphabet2_temporal_whitening_image_postfusion",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--token-limit", type=int, default=1_000_000)
    parser.add_argument("--sequence-limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    model = _build(args.kind)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    dataset = TokenBlockDataset(args.validation_manifest, verify_sha256=True)
    results = {
        "normal": _evaluate(
            model,
            dataset,
            segment=2_048,
            token_limit=args.token_limit,
            sequence_limit=args.sequence_limit,
            device=device,
        )
    }
    factorized_pca_bases = (
        []
        if isinstance(model, AlphabetLM)
        and model.config.repeated_vector_pole_retain_factor_state
        else _calibrate_factorized_pca_bases(model, dataset, device)
    )
    if factorized_pca_bases:
        extra_coordinate_handles = _factorized_extra_coordinate_override(model)
        results["factorized_extra_coordinates_off"] = _evaluate(
            model,
            dataset,
            segment=2_048,
            token_limit=args.token_limit,
            sequence_limit=args.sequence_limit,
            device=device,
        )
        for handle in extra_coordinate_handles:
            handle.remove()
        for rank in (4, 8):
            pca_handles = _factorized_pca_override(
                model,
                factorized_pca_bases,
                rank=rank,
            )
            results[f"factorized_state_pca_rank_{rank}"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in pca_handles:
                handle.remove()
    neutral_factor_handles = _factorized_factor_read_override(model, shift=False)
    if neutral_factor_handles:
        results["factor_read_neutral"] = _evaluate(
            model,
            dataset,
            segment=2_048,
            token_limit=args.token_limit,
            sequence_limit=args.sequence_limit,
            device=device,
        )
        for handle in neutral_factor_handles:
            handle.remove()
        shifted_factor_handles = _factorized_factor_read_override(model, shift=True)
        results["factor_read_shifted"] = _evaluate(
            model,
            dataset,
            segment=2_048,
            token_limit=args.token_limit,
            sequence_limit=args.sequence_limit,
            device=device,
        )
        for handle in shifted_factor_handles:
            handle.remove()
    outer_handles = _mamba_outer_override(model)
    if outer_handles:
        results["mamba_outer_off"] = _evaluate(
            model,
            dataset,
            segment=2_048,
            token_limit=args.token_limit,
            sequence_limit=args.sequence_limit,
            device=device,
        )
        for handle in outer_handles:
            handle.remove()
    if args.kind not in {
        "mamba",
        "mamba2",
        "mamba1_49m",
        "mamba2_48m",
        "laplace_mamba",
        "laplace_mamba_complex_highway",
        "alphabet2_image_postfusion",
        "alphabet2_vector_image_postfusion",
        "alphabet2_content_aligned_image_postfusion",
        "alphabet2_content_aligned_j2",
        "alphabet2_content_aligned_j4",
        "alphabet2_multi_observer_image_postfusion",
        "alphabet2_read_adapter_image_postfusion",
        "alphabet2_temporal_whitening_image_postfusion",
    }:
        handles = _zero_memory(model)
        results["memory_zero"] = _evaluate(
            model,
            dataset,
            segment=2_048,
            token_limit=args.token_limit,
            sequence_limit=args.sequence_limit,
            device=device,
        )
        for handle in handles:
            handle.remove()
        slow_handles = _zero_slow_cnn_memory(model)
        if slow_handles:
            results["slow_memory_zero"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in slow_handles:
                handle.remove()
        repeated_handles = _bypass_repeated_vector_poles(model)
        if repeated_handles:
            results["repeated_memory_zero"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in repeated_handles:
                handle.remove()
            repeated_banks = cast("AlphabetLM", model).repeated_vector_pole_memories
            if repeated_banks is None:
                raise RuntimeError("repeated VectorPole banks disappeared")
            for bank_index in range(len(repeated_banks)):
                bank_handles = _bypass_repeated_vector_poles(
                    model,
                    bank_index=bank_index,
                )
                results[f"repeated_bank_{bank_index + 1}_off"] = _evaluate(
                    model,
                    dataset,
                    segment=2_048,
                    token_limit=args.token_limit,
                    sequence_limit=args.sequence_limit,
                    device=device,
                )
                for handle in bank_handles:
                    handle.remove()
        neutral_query_handles = _slow_query_override(model, shuffle=False)
        if neutral_query_handles:
            results["query_neutral"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in neutral_query_handles:
                handle.remove()
            shuffled_query_handles = _slow_query_override(model, shuffle=True)
            results["query_shuffled"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in shuffled_query_handles:
                handle.remove()
        neutral_key_handles = _slow_key_override(model, shift=False)
        if neutral_key_handles:
            results["key_neutral"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in neutral_key_handles:
                handle.remove()
            shifted_key_handles = _slow_key_override(model, shift=True)
            results["key_shifted"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in shifted_key_handles:
                handle.remove()
            both_neutral_handles = [
                *_slow_query_override(model, shuffle=False),
                *_slow_key_override(model, shift=False),
            ]
            results["query_key_neutral"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in both_neutral_handles:
                handle.remove()
        value_off_handles = _slow_value_override(model, mode="off")
        if value_off_handles:
            results["vector_extra_off"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in value_off_handles:
                handle.remove()
            shifted_value_handles = _slow_value_override(model, mode="shift")
            results["value_shifted"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in shifted_value_handles:
                handle.remove()
            mean_value_handles = _slow_value_override(model, mode="time_mean")
            results["value_time_mean"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in mean_value_handles:
                handle.remove()
        matrix_off_handles = _slow_matrix_override(model, target="query", mode="off")
        if matrix_off_handles:
            results["matrix_extra_off"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in matrix_off_handles:
                handle.remove()
            shifted_matrix_query = _slow_matrix_override(
                model, target="query", mode="shift"
            )
            results["matrix_query_shifted"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in shifted_matrix_query:
                handle.remove()
            shifted_matrix_key = _slow_matrix_override(model, target="key", mode="shift")
            results["matrix_key_shifted"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in shifted_matrix_key:
                handle.remove()
            mean_matrix_key = _slow_matrix_override(model, target="key", mode="time_mean")
            results["matrix_key_time_mean"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in mean_matrix_key:
                handle.remove()
        independent_value_off = _slow_independent_value_override(model, mode="off")
        if independent_value_off:
            results["independent_value_off"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in independent_value_off:
                handle.remove()
            shifted_independent_value = _slow_independent_value_override(
                model, mode="shift"
            )
            results["independent_value_shifted"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in shifted_independent_value:
                handle.remove()
            mean_independent_value = _slow_independent_value_override(
                model, mode="time_mean"
            )
            results["independent_value_time_mean"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in mean_independent_value:
                handle.remove()
        vector_query_off = _slow_vector_pole_override(
            model, target="query", mode="off"
        )
        if vector_query_off:
            results["vector_pole_extra_off"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in vector_query_off:
                handle.remove()
            for target in ("excitation", "query"):
                for mode in ("shift", "time_mean"):
                    handles = _slow_vector_pole_override(
                        model,
                        target=target,
                        mode=mode,
                    )
                    results[f"vector_{target}_{mode}"] = _evaluate(
                        model,
                        dataset,
                        segment=2_048,
                        token_limit=args.token_limit,
                        sequence_limit=args.sequence_limit,
                        device=device,
                    )
                    for handle in handles:
                        handle.remove()
            complex_query_off = _slow_vector_pole_override(
                model, target="query_imag", mode="off"
            )
            if complex_query_off:
                results["complex_query_off"] = _evaluate(
                    model,
                    dataset,
                    segment=2_048,
                    token_limit=args.token_limit,
                    sequence_limit=args.sequence_limit,
                    device=device,
                )
                for handle in complex_query_off:
                    handle.remove()
                for mode in ("shift", "time_mean"):
                    handles = _slow_vector_pole_override(
                        model, target="query_imag", mode=mode
                    )
                    results[f"complex_query_{mode}"] = _evaluate(
                        model,
                        dataset,
                        segment=2_048,
                        token_limit=args.token_limit,
                        sequence_limit=args.sequence_limit,
                        device=device,
                    )
                    for handle in handles:
                        handle.remove()
            complex_excitation_off = _slow_vector_pole_override(
                model, target="excitation_imag", mode="off"
            )
            if complex_excitation_off:
                results["complex_excitation_off"] = _evaluate(
                    model,
                    dataset,
                    segment=2_048,
                    token_limit=args.token_limit,
                    sequence_limit=args.sequence_limit,
                    device=device,
                )
                for handle in complex_excitation_off:
                    handle.remove()
                for mode in ("shift", "time_mean"):
                    handles = _slow_vector_pole_override(
                        model, target="excitation_imag", mode=mode
                    )
                    results[f"complex_excitation_{mode}"] = _evaluate(
                        model,
                        dataset,
                        segment=2_048,
                        token_limit=args.token_limit,
                        sequence_limit=args.sequence_limit,
                        device=device,
                    )
                    for handle in handles:
                        handle.remove()
        transport_off = _slow_transport_override(model, mode="off")
        if transport_off:
            results["dynamic_transport_off"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in transport_off:
                handle.remove()
            for mode in ("shift", "time_mean"):
                handles = _slow_transport_override(model, mode=mode)
                results[f"dynamic_transport_{mode}"] = _evaluate(
                    model,
                    dataset,
                    segment=2_048,
                    token_limit=args.token_limit,
                    sequence_limit=args.sequence_limit,
                    device=device,
                )
                for handle in handles:
                    handle.remove()
        pole_reader_off = _slow_pole_reader_override(model, mode="off")
        if pole_reader_off:
            results["pole_reader_off"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in pole_reader_off:
                handle.remove()
            for mode in ("shift", "time_mean"):
                handles = _slow_pole_reader_override(model, mode=mode)
                results[f"pole_reader_{mode}"] = _evaluate(
                    model,
                    dataset,
                    segment=2_048,
                    token_limit=args.token_limit,
                    sequence_limit=args.sequence_limit,
                    device=device,
                )
                for handle in handles:
                    handle.remove()
        scheduler_neutral = _slow_write_scheduler_override(model, mode="neutral")
        if scheduler_neutral:
            results["write_scheduler_neutral"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in scheduler_neutral:
                handle.remove()
            scheduler_shuffled = _slow_write_scheduler_override(
                model,
                mode="shuffle",
            )
            results["write_scheduler_shuffled"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in scheduler_shuffled:
                handle.remove()
        innovation_neutral = _slow_innovation_override(model, mode="neutral")
        if innovation_neutral:
            results["innovation_neutral"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in innovation_neutral:
                handle.remove()
            innovation_shuffled = _slow_innovation_override(
                model,
                mode="shuffle",
            )
            results["innovation_shuffled"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in innovation_shuffled:
                handle.remove()
        clock_identity = _slow_semantic_clock_override(model, shuffle=False)
        if clock_identity:
            results["semantic_clock_identity"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in clock_identity:
                handle.remove()
            clock_shuffled = _slow_semantic_clock_override(model, shuffle=True)
            results["semantic_clock_shuffled"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in clock_shuffled:
                handle.remove()
    for segment in (512, 128, 32, 1):
        results[f"reset_{segment}"] = _evaluate(
            model,
            dataset,
            segment=segment,
            token_limit=args.token_limit,
            sequence_limit=args.sequence_limit,
            device=device,
        )
    normal_loss = float(results["normal"]["loss"])
    deltas = {
        f"{name}_minus_normal": float(result["loss"]) - normal_loss
        for name, result in results.items()
        if name != "normal"
    }
    payload = {
        "schema": "lnet.kau.lm_context_diagnostic.v1",
        "kind": args.kind,
        **results,
        "deltas": deltas,
    }
    if "query_key_neutral" in results:
        loss_00 = float(results["query_key_neutral"]["loss"])
        loss_10 = float(results["query_neutral"]["loss"])
        loss_01 = float(results["key_neutral"]["loss"])
        loss_11 = normal_loss
        payload["query_key_factorial"] = {
            "L00_query_off_key_off": loss_00,
            "L10_query_off_key_on": loss_10,
            "L01_query_on_key_off": loss_01,
            "L11_query_on_key_on": loss_11,
            "interaction": loss_10 + loss_01 - loss_00 - loss_11,
        }
    if isinstance(model, AlphabetLM):
        query_metrics = _query_readout_metrics(model, dataset, device)
        if query_metrics is not None:
            payload["query_readout"] = query_metrics
        delta_metrics = _delta_select_metrics(model, dataset, device)
        if delta_metrics is not None:
            payload["delta_select"] = delta_metrics
        tensor_metrics = _tensorpole_metrics(model, dataset, device)
        if tensor_metrics is not None:
            payload["tensorpole"] = tensor_metrics
        dynamic_write_metrics = _dynamic_write_metrics(model, dataset, device)
        if dynamic_write_metrics is not None:
            payload["dynamic_write"] = dynamic_write_metrics
        sidecar_metrics = _sidecar_metrics(model, dataset, device)
        if sidecar_metrics is not None:
            payload["sidecar"] = sidecar_metrics
        chunk_memory_metrics = _chunk_memory_metrics(model, dataset, device)
        if chunk_memory_metrics is not None:
            payload["chunk_memory"] = chunk_memory_metrics
        semantic_edge_metrics = _semantic_edge_metrics(model, dataset, device)
        if semantic_edge_metrics is not None:
            payload["semantic_edge"] = semantic_edge_metrics
        cnn_pole_metrics = _cnn_pole_metrics(model, dataset, device)
        if cnn_pole_metrics is not None:
            payload["cnn_pole"] = cnn_pole_metrics
        slow_cnn_pole_metrics = _slow_cnn_pole_metrics(model, dataset, device)
        if slow_cnn_pole_metrics is not None:
            payload["slow_cnn_pole"] = slow_cnn_pole_metrics
        repeated_metrics = _repeated_vector_pole_metrics(model, dataset, device)
        if repeated_metrics is not None:
            payload["repeated_vector_pole"] = repeated_metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("KAU_LM_CONTEXT=" + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
