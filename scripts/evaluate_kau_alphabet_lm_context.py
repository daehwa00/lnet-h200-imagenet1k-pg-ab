#!/usr/bin/env python3
"""Measure fixed-pole memory and effective context use from trained checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.alphabet_lm import (
    AlphabetLM,
    AlphabetLMConfig,
    CausalCNNPoleMemory,
    ChunkedSemanticPoleMemory,
    DynamicLowRankWrite,
    FixedComplexPoleMemory1D,
    FixedPoleResidualSidecar,
    GroupedPackedComplexLinear,
    LowRankDecaySelector,
    QueryConditionedLowRankReadout,
    SemanticEdgePoleMemory,
    SlowCausalCNNPoleMemory,
    TensorProductPoleMemory1D,
)
from lnet.alphabet_lm_data import TokenBlockDataset
from lnet.alphabet_lm_mamba import MambaLM, MambaLMConfig
from lnet.pac_complex_layers import PackedComplexLinear


def _build(kind: str) -> nn.Module:
    if kind == "mamba":
        return MambaLM(MambaLMConfig())
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
        "cnn_pole_p128_6bank_cascade_p128",
        "cnn_pole_p128_6bank_cascade_p128_no_recurrence",
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
            slow_cnn_pole_use_recurrence=True,
            slow_cnn_pole_minimum_half_life=1.0,
            slow_cnn_pole_maximum_half_life=256.0,
            additional_slow_cnn_pole_depths=(4,),
            additional_slow_cnn_pole_beta_initial=0.01,
            additional_slow_cnn_pole_use_recurrence=(
                kind == "cnn_pole_p128_6bank_cascade_p128"
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
        return [
            memory.synthesis.register_forward_hook(
                lambda _module, _inputs, output: torch.zeros_like(cast("Tensor", output))
            )
            for memory in cnn_memories
        ]
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
    return [
        model.slow_cnn_pole_memory.synthesis.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(cast("Tensor", output))
        )
    ]


def _zero_additional_slow_cnn_memories(
    model: nn.Module,
) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM) or model.additional_slow_cnn_pole_memories is None:
        return []
    return [
        cast("SlowCausalCNNPoleMemory", bank).synthesis.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(cast("Tensor", output))
        )
        for bank in model.additional_slow_cnn_pole_memories
    ]


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
    return {
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


@torch.no_grad()
def _additional_slow_cnn_pole_metrics(
    model: AlphabetLM,
    dataset: TokenBlockDataset,
    device: torch.device,
) -> list[dict[str, object]] | None:
    memories = model.additional_slow_cnn_pole_memories
    if memories is None:
        return None
    banks = [cast("SlowCausalCNNPoleMemory", bank) for bank in memories]
    depths = model.config.additional_slow_cnn_pole_depths
    base_depth = model.config.layers - model.config.slow_cnn_pole_upper_blocks
    boundaries = (*depths[1:], base_depth)
    branch_ratios: list[list[float]] = [[] for _ in banks]
    active_memory: list[tuple[Tensor, Tensor] | None] = [None for _ in banks]
    sample = dataset[0][:-1].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        packed = model.analysis(model.embedding(sample))
        real, imag = packed.split(model.config.modes, dim=-1)
        for index, block in enumerate(model.blocks[:base_depth]):
            for bank_index, (bank, depth, boundary) in enumerate(
                zip(banks, depths, boundaries, strict=True)
            ):
                if index == depth:
                    active_memory[bank_index] = bank(real, imag)
                memory = active_memory[bank_index]
                if memory is not None and depth <= index < boundary:
                    injected_real, injected_imag = bank.inject(
                        real,
                        imag,
                        memory[0],
                        memory[1],
                        index - depth,
                    )
                    trunk_energy = real.float().square().add(imag.float().square()).mean()
                    branch_energy = (
                        (injected_real - real).float().square()
                        + (injected_imag - imag).float().square()
                    ).mean()
                    branch_ratios[bank_index].append(
                        float(torch.sqrt(branch_energy / trunk_energy.clamp_min(1e-12)))
                    )
                    real, imag = injected_real, injected_imag
            real, imag = block(real, imag)
            if (
                model.cnn_pole_memories is not None
                and (index + 1) % model.config.cnn_pole_interval == 0
            ):
                fast_index = (index + 1) // model.config.cnn_pole_interval - 1
                real, imag = model.cnn_pole_memories[fast_index](real, imag)
    rows = []
    for bank, depth, boundary, ratios in zip(
        banks,
        depths,
        boundaries,
        branch_ratios,
        strict=True,
    ):
        weight = bank.analysis.weight.float()
        identity = torch.eye(weight.shape[0], device=weight.device)
        half_lives = math.log(2.0) / bank.memory.damping().float()
        rows.append(
            {
                "depth": depth,
                "next_depth": boundary,
                "stride": bank.stride,
                "pole_modes": bank.memory.modes,
                "use_recurrence": bank.use_recurrence,
                "beta_by_block": bank.beta.float().detach().cpu().tolist(),
                "beta_mean": float(bank.beta.float().mean()),
                "branch_to_trunk_rms_by_block": ratios,
                "branch_to_trunk_rms_mean": sum(ratios) / len(ratios),
                "half_life_tokens_min": float(half_lives.min()) * bank.stride,
                "half_life_tokens_median": float(half_lives.median()) * bank.stride,
                "half_life_tokens_max": float(half_lives.max()) * bank.stride,
                "analysis_row_gram_max_abs": float(
                    (weight @ weight.T - identity).abs().max()
                ),
            }
        )
    return rows


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
            "cnn_pole_p128_6bank_cascade_p128",
            "cnn_pole_p128_6bank_cascade_p128_no_recurrence",
            "mamba",
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
    if args.kind != "mamba":
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
        additional_handles = _zero_additional_slow_cnn_memories(model)
        if additional_handles:
            results["additional_slow_memory_zero"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in additional_handles:
                handle.remove()
            all_slow_handles = [
                *_zero_slow_cnn_memory(model),
                *_zero_additional_slow_cnn_memories(model),
            ]
            results["all_slow_memory_zero"] = _evaluate(
                model,
                dataset,
                segment=2_048,
                token_limit=args.token_limit,
                sequence_limit=args.sequence_limit,
                device=device,
            )
            for handle in all_slow_handles:
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
    if "all_slow_memory_zero" in results:
        loss_00 = float(results["all_slow_memory_zero"]["loss"])
        loss_10 = float(results["slow_memory_zero"]["loss"])
        loss_01 = float(results["additional_slow_memory_zero"]["loss"])
        loss_11 = normal_loss
        payload["slow_bank_factorial"] = {
            "L00_both_off": loss_00,
            "L10_additional_only": loss_10,
            "L01_anchor_only": loss_01,
            "L11_both_on": loss_11,
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
        additional_slow_metrics = _additional_slow_cnn_pole_metrics(model, dataset, device)
        if additional_slow_metrics is not None:
            payload["additional_slow_cnn_poles"] = additional_slow_metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("KAU_LM_CONTEXT=" + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
