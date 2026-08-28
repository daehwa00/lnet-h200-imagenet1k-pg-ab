#!/usr/bin/env python3
"""Restart-safe 10M-token H200 trainer for ALPHABET-LM and official Mamba."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportMissingImports=false
import argparse
import hashlib
import json
import math
import os
import random
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.alphabet_lm import (
    AlphabetLM,
    AlphabetLMConfig,
    CausalCNNPoleMemory,
    DynamicLowRankWrite,
    LowRankDecaySelector,
    QueryConditionedLowRankReadout,
    SlowCausalCNNPoleMemory,
)
from lnet.alphabet_lm_data import TokenBlockDataset, sha256_file
from lnet.alphabet_lm_mamba import MambaLMConfig, build_parameter_matched_mamba

RUNTIME_SCHEMA = "lnet.h200.alphabet_lm.viability_10m.runtime.v1"
KAU_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.pole_init_10m.runtime.v1"
KAU_GROUPED_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.grouped_h8p128_10m.runtime.v1"
KAU_DENSE_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.dense_k3_p320_10m.runtime.v1"
KAU_ROUTING_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.dynamic_routing_2m.runtime.v1"
KAU_STEP_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.step_control_2m.runtime.v1"
KAU_DECODER_READOUT_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.decoder_readout_screen_2m.runtime.v1"
KAU_DENSE_DELTA_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.dense_delta_screen_2m.runtime.v1"
KAU_TENSORPOLE_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.tensorpole_m8_2m.runtime.v1"
KAU_DYNAMIC_WRITE_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.dynamic_write_r4_2m.runtime.v1"
KAU_CONTEXT_CONTROL_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.context_controls_2m.runtime.v1"
KAU_LOCAL_SIDECAR_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.local_sidecar_2m.runtime.v1"
KAU_FROZEN_NORMALIZED_SIDECAR_RUNTIME_SCHEMA = (
    "lnet.kau.alphabet_lm.frozen_normalized_sidecar_1m.runtime.v1"
)
KAU_FROZEN_LOCAL_SIDECAR_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.frozen_local_sidecar_1m.runtime.v1"
KAU_LOCAL_ONLY_10M_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.local_only_10m.runtime.v1"
KAU_CHUNKED_SEMANTIC_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.chunked_semantic_1m.runtime.v1"
KAU_SEMANTIC_EDGE_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.semantic_edge_1m.runtime.v1"
KAU_SEMANTIC_EDGE_EXTENSION_RUNTIME_SCHEMA = (
    "lnet.kau.alphabet_lm.semantic_edge_extension.runtime.v1"
)
KAU_CNN_POLE_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.cnn_pole_1m.runtime.v1"
KAU_SLOW_CNN_POLE_RUNTIME_SCHEMA = "lnet.kau.alphabet_lm.slow_cnn_pole_1m.runtime.v1"
KAU_SLOW_CNN_POLE_EXTENSION_RUNTIME_SCHEMA = (
    "lnet.kau.alphabet_lm.slow_cnn_pole_extension.runtime.v1"
)
KAU_CASCADED_SLOW_CNN_POLE_RUNTIME_SCHEMA = (
    "lnet.kau.alphabet_lm.cascaded_slow_cnn_pole.runtime.v1"
)
_STOP_EVENT = threading.Event()


def _signal_stop(_signum: int, _frame: object) -> None:
    _STOP_EVENT.set()


class DeterministicBatcher:
    def __init__(self, dataset: TokenBlockDataset, *, seed: int, batch_size: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.epoch = 0
        self.position = 0
        self._order = self._permutation()

    def _permutation(self) -> Tensor:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=generator)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, payload: dict[str, int]) -> None:
        self.epoch = int(payload["epoch"])
        self.position = int(payload["position"])
        self._order = self._permutation()
        if not 0 <= self.position < len(self.dataset):
            raise RuntimeError("restored LM data cursor is invalid")

    def next(self) -> Tensor:
        indices: list[int] = []
        while len(indices) < self.batch_size:
            remaining = len(self.dataset) - self.position
            take = min(self.batch_size - len(indices), remaining)
            indices.extend(self._order[self.position : self.position + take].tolist())
            self.position += take
            if self.position == len(self.dataset):
                self.epoch += 1
                self.position = 0
                self._order = self._permutation()
        return torch.stack([self.dataset[index] for index in indices])


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def _copy_matching_legacy_initialization(
    model: AlphabetLM,
    *,
    vocab_size: int,
    seed: int,
    reader_type: str = "r2k3",
) -> tuple[int, int]:
    """Make shape-compatible parameters identical to a seeded fixed-pole control."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = AlphabetLM(
            AlphabetLMConfig(vocab_size=vocab_size, reader_type=cast("Any", reader_type))
        )
    source = dict(reference.named_parameters())
    copied_tensors = 0
    copied_parameters = 0
    for name, parameter in model.named_parameters():
        candidate = source.get(name)
        if candidate is None or candidate.shape != parameter.shape:
            continue
        parameter.copy_(candidate)
        copied_tensors += 1
        copied_parameters += parameter.numel()
    return copied_tensors, copied_parameters


def _initialize_sidecar_from_trunk(
    model: nn.Module,
    checkpoint_path: Path | None,
    *,
    freeze_trunk: bool,
) -> dict[str, object]:
    if checkpoint_path is None:
        if freeze_trunk:
            raise RuntimeError("freezing the trunk requires an initialization checkpoint")
        return {"enabled": False, "frozen": False}
    if not isinstance(model, AlphabetLM) or model.config.memory_layout != "local_sidecar":
        raise RuntimeError("trunk initialization requires a LocalSidecar ALPHABET model")
    payload = cast(
        "dict[str, Any]", torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    source = cast("dict[str, Tensor]", payload["model"])
    expected_missing = {name for name in model.state_dict() if ".sidecar." in name}
    incompatible = model.load_state_dict(source, strict=False)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError("LocalOnly checkpoint does not match the LocalSidecar trunk")
    if freeze_trunk:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(".sidecar." in name)
    return {
        "enabled": True,
        "frozen": freeze_trunk,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "missing_sidecar_tensors": len(expected_missing),
    }


def _initialize_chunk_memory_from_trunk(
    model: nn.Module,
    checkpoint_path: Path | None,
    *,
    train_upper_blocks: int,
) -> dict[str, object]:
    if checkpoint_path is None:
        if train_upper_blocks:
            raise RuntimeError("training upper blocks requires a chunk-trunk checkpoint")
        return {"enabled": False, "train_upper_blocks": 0}
    if not isinstance(model, AlphabetLM) or model.chunk_memory is None:
        raise RuntimeError("chunk-trunk initialization requires chunk semantic memory")
    if not 0 < train_upper_blocks <= model.config.chunk_upper_blocks:
        raise RuntimeError("invalid number of trainable upper blocks")
    payload = cast(
        "dict[str, Any]", torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    source = cast("dict[str, Tensor]", payload["model"])
    expected_missing = {name for name in model.state_dict() if name.startswith("chunk_memory.")}
    incompatible = model.load_state_dict(source, strict=False)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError("LocalOnly checkpoint does not match the chunk-memory trunk")
    first_upper = model.config.layers - train_upper_blocks
    for name, parameter in model.named_parameters():
        block_index = None
        if name.startswith("blocks."):
            block_index = int(name.split(".", 2)[1])
        parameter.requires_grad_(
            name.startswith("chunk_memory.")
            or (block_index is not None and block_index >= first_upper)
        )
    return {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "missing_chunk_tensors": len(expected_missing),
        "train_upper_blocks": train_upper_blocks,
    }


def _initialize_semantic_edge_from_trunk(
    model: nn.Module,
    checkpoint_path: Path | None,
) -> dict[str, object]:
    if checkpoint_path is None:
        return {"enabled": False}
    if not isinstance(model, AlphabetLM) or model.semantic_edge_memory is None:
        raise RuntimeError("semantic-edge initialization requires semantic edge memory")
    payload = cast(
        "dict[str, Any]", torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    source = cast("dict[str, Tensor]", payload["model"])
    expected_missing = {
        name for name in model.state_dict() if name.startswith("semantic_edge_memory.")
    }
    incompatible = model.load_state_dict(source, strict=False)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError("LocalOnly checkpoint does not match the semantic-edge trunk")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("semantic_edge_memory."))
    return {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "missing_edge_tensors": len(expected_missing),
        "trunk_frozen": True,
    }


def _initialize_cnn_pole_from_trunk(
    model: nn.Module,
    checkpoint_path: Path | None,
) -> dict[str, object]:
    if checkpoint_path is None:
        return {"enabled": False}
    if not isinstance(model, AlphabetLM) or model.cnn_pole_memories is None:
        raise RuntimeError("CNN-pole initialization requires repeated CNN pole memory")
    payload = cast(
        "dict[str, Any]", torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    source = cast("dict[str, Tensor]", payload["model"])
    expected_missing = {
        name for name in model.state_dict() if name.startswith("cnn_pole_memories.")
    }
    incompatible = model.load_state_dict(source, strict=False)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError("LocalOnly checkpoint does not match the CNN-pole trunk")
    for name, parameter in model.named_parameters():
        trainable = name.startswith("cnn_pole_memories.") and not name.endswith(
            ".analysis.weight"
        )
        parameter.requires_grad_(trainable)
    return {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "missing_cnn_pole_tensors": len(expected_missing),
        "trunk_frozen": True,
    }


def _initialize_slow_cnn_pole_from_trunk(
    model: nn.Module,
    checkpoint_path: Path | None,
) -> dict[str, object]:
    if checkpoint_path is None:
        return {"enabled": False}
    if not isinstance(model, AlphabetLM) or model.slow_cnn_pole_memory is None:
        raise RuntimeError("slow CNN-pole initialization requires a slow memory bank")
    payload = cast(
        "dict[str, Any]", torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    source = cast("dict[str, Tensor]", payload["model"])
    expected_missing = {
        name for name in model.state_dict() if name.startswith("slow_cnn_pole_memory.")
    }
    incompatible = model.load_state_dict(source, strict=False)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError("CNNx6 checkpoint does not match the slow CNN-pole model")
    for name, parameter in model.named_parameters():
        trainable = name.startswith("slow_cnn_pole_memory.") and not name.endswith(
            ".analysis.weight"
        )
        parameter.requires_grad_(trainable)
    return {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "missing_slow_cnn_pole_tensors": len(expected_missing),
        "fast_cnn_and_trunk_frozen": True,
    }


def _initialize_additional_slow_cnn_poles_from_trunk(
    model: nn.Module,
    checkpoint_path: Path | None,
) -> dict[str, object]:
    if checkpoint_path is None:
        return {"enabled": False}
    if not isinstance(model, AlphabetLM) or model.additional_slow_cnn_pole_memories is None:
        raise RuntimeError("additional slow-bank initialization requires additional banks")
    payload = cast(
        "dict[str, Any]", torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    source = cast("dict[str, Tensor]", payload["model"])
    prefix = "additional_slow_cnn_pole_memories."
    expected_missing = {name for name in model.state_dict() if name.startswith(prefix)}
    incompatible = model.load_state_dict(source, strict=False)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError("anchor-bank checkpoint does not match the cascaded model")
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefix) and not name.endswith(".analysis.weight")
        parameter.requires_grad_(trainable)
    return {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "missing_additional_slow_bank_tensors": len(expected_missing),
        "anchor_bank_and_trunk_frozen": True,
    }


def _loss_sum(model: nn.Module, tokens: Tensor, pad_id: int) -> tuple[Tensor, int]:
    labels = tokens[:, 1:]
    logits = model(tokens[:, :-1])
    loss = functional.cross_entropy(
        logits.flatten(0, 1),
        labels.flatten(),
        ignore_index=pad_id,
        reduction="sum",
    )
    return loss, int((labels != pad_id).sum())


def _build(
    model_name: str,
    vocab_size: int,
    *,
    pole_initialization: str = "legacy",
    memory_banks: int = 1,
    bank_pole_modes: int = 128,
    reader_type: str = "r2k3",
    pole_routing: str = "static",
    post_hidden: int = 384,
    memory_readout: str = "fixed",
    query_read_rank: int = 32,
    query_read_initial_scale: float = 0.05,
    pole_dynamics: str = "fixed",
    delta_select_rank: int = 16,
    delta_select_initial_scale: float = 0.1,
    delta_select_control_bound: float = 1.0,
    memory_layout: str = "flat",
    tensor_temporal_modes: int = 8,
    tensor_initial_read_gain: float = 0.6,
    sidecar_initial_scale: float = 0.01,
    sidecar_normalize_memory: bool = False,
    sidecar_channelwise_scale: bool = True,
    sidecar_use_recurrence: bool = True,
    chunk_memory: bool = False,
    chunk_size: int = 32,
    chunk_summary_width: int = 128,
    chunk_pole_modes: int = 128,
    chunk_upper_blocks: int = 4,
    chunk_beta_initial: float = 0.01,
    chunk_minimum_half_life: float = 1.0,
    chunk_maximum_half_life: float = 128.0,
    semantic_edge_memory: bool = False,
    semantic_edge_stride: int = 16,
    semantic_edge_pole_modes: int = 128,
    semantic_edge_upper_blocks: int = 4,
    semantic_edge_beta_initial: float = 0.01,
    semantic_edge_use_recurrence: bool = True,
    semantic_edge_minimum_half_life: float = 1.0,
    semantic_edge_maximum_half_life: float = 256.0,
    cnn_pole_memory: bool = False,
    cnn_pole_interval: int = 2,
    cnn_pole_modes: int = 128,
    cnn_pole_evidence_width: int = 512,
    cnn_pole_kernel_size: int = 4,
    cnn_pole_beta_initial: float = 0.01,
    cnn_pole_use_recurrence: bool = True,
    cnn_pole_minimum_half_life: float = 8.0,
    cnn_pole_maximum_half_life: float = 4_096.0,
    slow_cnn_pole_memory: bool = False,
    slow_cnn_pole_stride: int = 16,
    slow_cnn_pole_modes: int = 128,
    slow_cnn_pole_evidence_width: int = 512,
    slow_cnn_pole_kernel_size: int = 4,
    slow_cnn_pole_upper_blocks: int = 4,
    slow_cnn_pole_beta_initial: float = 0.01,
    slow_cnn_pole_use_recurrence: bool = True,
    slow_cnn_pole_minimum_half_life: float = 1.0,
    slow_cnn_pole_maximum_half_life: float = 256.0,
    additional_slow_cnn_pole_depths: tuple[int, ...] = (),
    additional_slow_cnn_pole_beta_initial: float = 0.01,
    additional_slow_cnn_pole_use_recurrence: bool = True,
    write_map: str = "static",
    dynamic_write_rank: int = 4,
    dynamic_write_initial_scale: float = 0.06,
) -> tuple[nn.Module, dict[str, Any]]:
    alphabet_config = AlphabetLMConfig(
        vocab_size=vocab_size,
        modes=256,
        pole_modes=320,
        layers=12,
        post_hidden=post_hidden,
        context_length=2_048,
        scan_fp32=True,
        pole_initialization=cast("Any", pole_initialization),
        memory_banks=memory_banks,
        bank_pole_modes=bank_pole_modes,
        reader_type=cast("Any", reader_type),
        pole_routing=cast("Any", pole_routing),
        memory_readout=cast("Any", memory_readout),
        query_read_rank=query_read_rank,
        query_read_initial_scale=query_read_initial_scale,
        pole_dynamics=cast("Any", pole_dynamics),
        delta_select_rank=delta_select_rank,
        delta_select_initial_scale=delta_select_initial_scale,
        delta_select_control_bound=delta_select_control_bound,
        memory_layout=cast("Any", memory_layout),
        tensor_temporal_modes=tensor_temporal_modes,
        tensor_initial_read_gain=tensor_initial_read_gain,
        sidecar_initial_scale=sidecar_initial_scale,
        sidecar_normalize_memory=sidecar_normalize_memory,
        sidecar_channelwise_scale=sidecar_channelwise_scale,
        sidecar_use_recurrence=sidecar_use_recurrence,
        chunk_memory=chunk_memory,
        chunk_size=chunk_size,
        chunk_summary_width=chunk_summary_width,
        chunk_pole_modes=chunk_pole_modes,
        chunk_upper_blocks=chunk_upper_blocks,
        chunk_beta_initial=chunk_beta_initial,
        chunk_minimum_half_life=chunk_minimum_half_life,
        chunk_maximum_half_life=chunk_maximum_half_life,
        semantic_edge_memory=semantic_edge_memory,
        semantic_edge_stride=semantic_edge_stride,
        semantic_edge_pole_modes=semantic_edge_pole_modes,
        semantic_edge_upper_blocks=semantic_edge_upper_blocks,
        semantic_edge_beta_initial=semantic_edge_beta_initial,
        semantic_edge_use_recurrence=semantic_edge_use_recurrence,
        semantic_edge_minimum_half_life=semantic_edge_minimum_half_life,
        semantic_edge_maximum_half_life=semantic_edge_maximum_half_life,
        cnn_pole_memory=cnn_pole_memory,
        cnn_pole_interval=cnn_pole_interval,
        cnn_pole_modes=cnn_pole_modes,
        cnn_pole_evidence_width=cnn_pole_evidence_width,
        cnn_pole_kernel_size=cnn_pole_kernel_size,
        cnn_pole_beta_initial=cnn_pole_beta_initial,
        cnn_pole_use_recurrence=cnn_pole_use_recurrence,
        cnn_pole_minimum_half_life=cnn_pole_minimum_half_life,
        cnn_pole_maximum_half_life=cnn_pole_maximum_half_life,
        slow_cnn_pole_memory=slow_cnn_pole_memory,
        slow_cnn_pole_stride=slow_cnn_pole_stride,
        slow_cnn_pole_modes=slow_cnn_pole_modes,
        slow_cnn_pole_evidence_width=slow_cnn_pole_evidence_width,
        slow_cnn_pole_kernel_size=slow_cnn_pole_kernel_size,
        slow_cnn_pole_upper_blocks=slow_cnn_pole_upper_blocks,
        slow_cnn_pole_beta_initial=slow_cnn_pole_beta_initial,
        slow_cnn_pole_use_recurrence=slow_cnn_pole_use_recurrence,
        slow_cnn_pole_minimum_half_life=slow_cnn_pole_minimum_half_life,
        slow_cnn_pole_maximum_half_life=slow_cnn_pole_maximum_half_life,
        additional_slow_cnn_pole_depths=additional_slow_cnn_pole_depths,
        additional_slow_cnn_pole_beta_initial=additional_slow_cnn_pole_beta_initial,
        additional_slow_cnn_pole_use_recurrence=additional_slow_cnn_pole_use_recurrence,
        write_map=cast("Any", write_map),
        dynamic_write_rank=dynamic_write_rank,
        dynamic_write_initial_scale=dynamic_write_initial_scale,
    )
    if model_name == "alphabet":
        return AlphabetLM(alphabet_config), {
            "model": "alphabet",
            "config": asdict(alphabet_config),
        }
    target = _parameter_count(AlphabetLM(alphabet_config))
    model, parameters, relative_error = build_parameter_matched_mamba(
        target,
        MambaLMConfig(vocab_size=vocab_size, model_width=512),
        tolerance=0.03,
    )
    return model, {
        "model": "mamba",
        "config": asdict(model.config),
        "official_mamba_lm": True,
        "alphabet_target_parameters": target,
        "parameters": parameters,
        "relative_parameter_error": relative_error,
    }


def _scheduler(
    optimizer: torch.optim.Optimizer, *, warmup: int, horizon: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(update: int) -> float:
        if update < warmup:
            return (update + 1) / max(1, warmup)
        progress = min(1.0, (update - warmup) / max(1, horizon - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataset: TokenBlockDataset,
    *,
    microbatch: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(dataset), microbatch):
        batch = torch.stack(
            [dataset[index] for index in range(start, min(start + microbatch, len(dataset)))]
        ).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, count = _loss_sum(model, batch, dataset.manifest.pad_id)
        total_loss += float(loss)
        total_tokens += count
    model.train()
    return total_loss / total_tokens


def _wandb_run(
    runtime: dict[str, Any], run_label: str, root: Path, contract: dict[str, Any]
) -> Any:
    import wandb

    record = runtime["runs"][run_label]
    if runtime["schema"] == RUNTIME_SCHEMA:
        expected = {
            "WANDB_API_KEY": "0" * 40,
            "WANDB_APP_URL": runtime["wandb_app_url"],
            "WANDB_BASE_URL": runtime["wandb_base_url"],
            "WANDB_ENTITY": runtime["entity"],
            "WANDB_PROJECT": runtime["project"],
            "WANDB_GROUP": runtime["group"],
            "WANDB_CONSOLE": runtime["console"],
        }
        if any(os.environ.get(name) != value for name, value in expected.items()):
            raise RuntimeError("H200 LM training W&B environment changed")
    tracking = root / "wandb"
    tracking.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=runtime["project"],
        entity=runtime["entity"],
        group=runtime["group"],
        name=record["display_name"],
        id=record["id"],
        tags=record["tags"],
        resume="allow",
        dir=str(tracking),
        mode="online",
        anonymous="never",
        force=True,
        settings=wandb.Settings(
            disable_code=True,
            console="off",
            disable_git=True,
            disable_job_creation=True,
            init_timeout=30,
            save_code=False,
            x_disable_meta=True,
            x_disable_stats=True,
            x_disable_viewer=True,
            x_extra_http_headers={"User-Agent": "Mozilla/5.0 lnet-h200-wandb-client/1"},
            x_save_requirements=False,
        ),
        config=contract,
    )
    if run is None or not run.url:
        raise RuntimeError("required H200 LM training W&B run did not initialize")
    print(f"WANDB_RUN_URL={run.url}", flush=True)
    return run


def _save_checkpoint(
    path: Path,
    *,
    contract_sha: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batcher: DeterministicBatcher,
    update: int,
    tokens_seen: int,
    history: list[dict[str, float | int]],
) -> None:
    _atomic_torch(
        path,
        {
            "contract_sha256": contract_sha,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "batcher": batcher.state_dict(),
            "update": update,
            "tokens_seen": tokens_seen,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("alphabet", "mamba"), required=True)
    parser.add_argument("--run-label")
    parser.add_argument(
        "--pole-initialization",
        choices=("legacy", "lifetime_palette"),
        default="legacy",
    )
    parser.add_argument("--memory-banks", type=int, default=1)
    parser.add_argument("--bank-pole-modes", type=int, default=128)
    parser.add_argument("--reader-type", choices=("r2k3", "dense_k3"), default="r2k3")
    parser.add_argument(
        "--pole-routing",
        choices=("static", "dynamic_write", "dynamic_write_read"),
        default="static",
    )
    parser.add_argument("--post-hidden", type=int, default=384)
    parser.add_argument("--memory-readout", choices=("fixed", "query_low_rank"), default="fixed")
    parser.add_argument("--query-read-rank", type=int, default=32)
    parser.add_argument("--query-read-initial-scale", type=float, default=0.05)
    parser.add_argument("--paired-legacy-initialization", action="store_true")
    parser.add_argument("--paired-dense-initialization", action="store_true")
    parser.add_argument("--pole-dynamics", choices=("fixed", "delta_select"), default="fixed")
    parser.add_argument("--delta-select-rank", type=int, default=16)
    parser.add_argument("--delta-select-initial-scale", type=float, default=0.1)
    parser.add_argument("--delta-select-control-bound", type=float, default=1.0)
    parser.add_argument(
        "--memory-layout",
        choices=("flat", "tensor_product", "local_only", "local_sidecar"),
        default="flat",
    )
    parser.add_argument("--tensor-temporal-modes", type=int, default=8)
    parser.add_argument("--tensor-initial-read-gain", type=float, default=0.6)
    parser.add_argument("--sidecar-initial-scale", type=float, default=0.01)
    parser.add_argument(
        "--sidecar-normalize-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--sidecar-channelwise-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--initialize-trunk-checkpoint", type=Path)
    parser.add_argument("--freeze-trunk", action="store_true")
    parser.add_argument(
        "--sidecar-use-recurrence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--chunk-memory", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-summary-width", type=int, default=128)
    parser.add_argument("--chunk-pole-modes", type=int, default=128)
    parser.add_argument("--chunk-upper-blocks", type=int, default=4)
    parser.add_argument("--chunk-beta-initial", type=float, default=0.01)
    parser.add_argument("--chunk-minimum-half-life", type=float, default=1.0)
    parser.add_argument("--chunk-maximum-half-life", type=float, default=128.0)
    parser.add_argument("--initialize-chunk-trunk-checkpoint", type=Path)
    parser.add_argument("--train-upper-blocks", type=int, default=0)
    parser.add_argument("--upper-block-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--semantic-edge-memory", action="store_true")
    parser.add_argument("--semantic-edge-stride", type=int, default=16)
    parser.add_argument("--semantic-edge-pole-modes", type=int, default=128)
    parser.add_argument("--semantic-edge-upper-blocks", type=int, default=4)
    parser.add_argument("--semantic-edge-beta-initial", type=float, default=0.01)
    parser.add_argument(
        "--semantic-edge-use-recurrence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--semantic-edge-minimum-half-life", type=float, default=1.0)
    parser.add_argument("--semantic-edge-maximum-half-life", type=float, default=256.0)
    parser.add_argument("--initialize-semantic-edge-trunk-checkpoint", type=Path)
    parser.add_argument("--cnn-pole-memory", action="store_true")
    parser.add_argument("--cnn-pole-interval", type=int, default=2)
    parser.add_argument("--cnn-pole-modes", type=int, default=128)
    parser.add_argument("--cnn-pole-evidence-width", type=int, default=512)
    parser.add_argument("--cnn-pole-kernel-size", type=int, default=4)
    parser.add_argument("--cnn-pole-beta-initial", type=float, default=0.01)
    parser.add_argument(
        "--cnn-pole-use-recurrence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cnn-pole-minimum-half-life", type=float, default=8.0)
    parser.add_argument("--cnn-pole-maximum-half-life", type=float, default=4_096.0)
    parser.add_argument("--initialize-cnn-pole-trunk-checkpoint", type=Path)
    parser.add_argument("--slow-cnn-pole-memory", action="store_true")
    parser.add_argument("--slow-cnn-pole-stride", type=int, default=16)
    parser.add_argument("--slow-cnn-pole-modes", type=int, default=128)
    parser.add_argument("--slow-cnn-pole-evidence-width", type=int, default=512)
    parser.add_argument("--slow-cnn-pole-kernel-size", type=int, default=4)
    parser.add_argument("--slow-cnn-pole-upper-blocks", type=int, default=4)
    parser.add_argument("--slow-cnn-pole-beta-initial", type=float, default=0.01)
    parser.add_argument(
        "--slow-cnn-pole-use-recurrence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--slow-cnn-pole-minimum-half-life", type=float, default=1.0)
    parser.add_argument("--slow-cnn-pole-maximum-half-life", type=float, default=256.0)
    parser.add_argument("--initialize-slow-cnn-pole-trunk-checkpoint", type=Path)
    parser.add_argument(
        "--additional-slow-cnn-pole-depths",
        type=int,
        nargs="*",
        default=(),
    )
    parser.add_argument("--additional-slow-cnn-pole-beta-initial", type=float, default=0.01)
    parser.add_argument(
        "--additional-slow-cnn-pole-use-recurrence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--initialize-additional-slow-cnn-pole-trunk-checkpoint", type=Path)
    parser.add_argument("--write-map", choices=("static", "dynamic_low_rank"), default="static")
    parser.add_argument("--dynamic-write-rank", type=int, default=4)
    parser.add_argument("--dynamic-write-initial-scale", type=float, default=0.06)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target-tokens-override", type=int)
    parser.add_argument("--resume-extension-checkpoint", type=Path)
    args = parser.parse_args()
    runtime = cast("dict[str, Any]", json.loads(args.runtime.read_text(encoding="utf-8")))
    if runtime.get("schema") not in {
        RUNTIME_SCHEMA,
        KAU_RUNTIME_SCHEMA,
        KAU_GROUPED_RUNTIME_SCHEMA,
        KAU_DENSE_RUNTIME_SCHEMA,
        KAU_ROUTING_RUNTIME_SCHEMA,
        KAU_STEP_RUNTIME_SCHEMA,
        KAU_DECODER_READOUT_RUNTIME_SCHEMA,
        KAU_DENSE_DELTA_RUNTIME_SCHEMA,
        KAU_TENSORPOLE_RUNTIME_SCHEMA,
        KAU_DYNAMIC_WRITE_RUNTIME_SCHEMA,
        KAU_CONTEXT_CONTROL_RUNTIME_SCHEMA,
        KAU_LOCAL_SIDECAR_RUNTIME_SCHEMA,
        KAU_FROZEN_NORMALIZED_SIDECAR_RUNTIME_SCHEMA,
        KAU_FROZEN_LOCAL_SIDECAR_RUNTIME_SCHEMA,
        KAU_LOCAL_ONLY_10M_RUNTIME_SCHEMA,
        KAU_CHUNKED_SEMANTIC_RUNTIME_SCHEMA,
        KAU_SEMANTIC_EDGE_RUNTIME_SCHEMA,
        KAU_SEMANTIC_EDGE_EXTENSION_RUNTIME_SCHEMA,
        KAU_CNN_POLE_RUNTIME_SCHEMA,
        KAU_SLOW_CNN_POLE_RUNTIME_SCHEMA,
        KAU_SLOW_CNN_POLE_EXTENSION_RUNTIME_SCHEMA,
        KAU_CASCADED_SLOW_CNN_POLE_RUNTIME_SCHEMA,
    }:
        raise RuntimeError("invalid H200/KAU LM training runtime")
    if runtime["training"]["scan_fp32"] is not True:
        raise RuntimeError("invalid or non-FP32 H200 LM training runtime")
    training = runtime["training"]
    run_label = args.run_label or f"{args.model}-10m"
    args.root.mkdir(parents=True, exist_ok=True)
    completed = args.root / "completed.json"
    if completed.is_file():
        print(f"ALPHABET_LM_TRAINING_REUSED={completed}", flush=True)
        return
    train = TokenBlockDataset(args.train_manifest, verify_sha256=True)
    validation = TokenBlockDataset(args.validation_manifest, verify_sha256=True)
    if (
        train.manifest.context_length != 2_048
        or validation.manifest.context_length != 2_048
        or train.manifest.tokenizer_sha256 != validation.manifest.tokenizer_sha256
        or train.manifest.vocab_size != 32_768
    ):
        raise RuntimeError("train/validation token contracts differ")
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model, model_contract = _build(
        args.model,
        train.manifest.vocab_size,
        pole_initialization=args.pole_initialization,
        memory_banks=args.memory_banks,
        bank_pole_modes=args.bank_pole_modes,
        reader_type=args.reader_type,
        pole_routing=args.pole_routing,
        post_hidden=args.post_hidden,
        memory_readout=args.memory_readout,
        query_read_rank=args.query_read_rank,
        query_read_initial_scale=args.query_read_initial_scale,
        pole_dynamics=args.pole_dynamics,
        delta_select_rank=args.delta_select_rank,
        delta_select_initial_scale=args.delta_select_initial_scale,
        delta_select_control_bound=args.delta_select_control_bound,
        memory_layout=args.memory_layout,
        tensor_temporal_modes=args.tensor_temporal_modes,
        tensor_initial_read_gain=args.tensor_initial_read_gain,
        sidecar_initial_scale=args.sidecar_initial_scale,
        sidecar_normalize_memory=args.sidecar_normalize_memory,
        sidecar_channelwise_scale=args.sidecar_channelwise_scale,
        sidecar_use_recurrence=args.sidecar_use_recurrence,
        chunk_memory=args.chunk_memory,
        chunk_size=args.chunk_size,
        chunk_summary_width=args.chunk_summary_width,
        chunk_pole_modes=args.chunk_pole_modes,
        chunk_upper_blocks=args.chunk_upper_blocks,
        chunk_beta_initial=args.chunk_beta_initial,
        chunk_minimum_half_life=args.chunk_minimum_half_life,
        chunk_maximum_half_life=args.chunk_maximum_half_life,
        semantic_edge_memory=args.semantic_edge_memory,
        semantic_edge_stride=args.semantic_edge_stride,
        semantic_edge_pole_modes=args.semantic_edge_pole_modes,
        semantic_edge_upper_blocks=args.semantic_edge_upper_blocks,
        semantic_edge_beta_initial=args.semantic_edge_beta_initial,
        semantic_edge_use_recurrence=args.semantic_edge_use_recurrence,
        semantic_edge_minimum_half_life=args.semantic_edge_minimum_half_life,
        semantic_edge_maximum_half_life=args.semantic_edge_maximum_half_life,
        cnn_pole_memory=args.cnn_pole_memory,
        cnn_pole_interval=args.cnn_pole_interval,
        cnn_pole_modes=args.cnn_pole_modes,
        cnn_pole_evidence_width=args.cnn_pole_evidence_width,
        cnn_pole_kernel_size=args.cnn_pole_kernel_size,
        cnn_pole_beta_initial=args.cnn_pole_beta_initial,
        cnn_pole_use_recurrence=args.cnn_pole_use_recurrence,
        cnn_pole_minimum_half_life=args.cnn_pole_minimum_half_life,
        cnn_pole_maximum_half_life=args.cnn_pole_maximum_half_life,
        slow_cnn_pole_memory=args.slow_cnn_pole_memory,
        slow_cnn_pole_stride=args.slow_cnn_pole_stride,
        slow_cnn_pole_modes=args.slow_cnn_pole_modes,
        slow_cnn_pole_evidence_width=args.slow_cnn_pole_evidence_width,
        slow_cnn_pole_kernel_size=args.slow_cnn_pole_kernel_size,
        slow_cnn_pole_upper_blocks=args.slow_cnn_pole_upper_blocks,
        slow_cnn_pole_beta_initial=args.slow_cnn_pole_beta_initial,
        slow_cnn_pole_use_recurrence=args.slow_cnn_pole_use_recurrence,
        slow_cnn_pole_minimum_half_life=args.slow_cnn_pole_minimum_half_life,
        slow_cnn_pole_maximum_half_life=args.slow_cnn_pole_maximum_half_life,
        additional_slow_cnn_pole_depths=tuple(args.additional_slow_cnn_pole_depths),
        additional_slow_cnn_pole_beta_initial=args.additional_slow_cnn_pole_beta_initial,
        additional_slow_cnn_pole_use_recurrence=(
            args.additional_slow_cnn_pole_use_recurrence
        ),
        write_map=args.write_map,
        dynamic_write_rank=args.dynamic_write_rank,
        dynamic_write_initial_scale=args.dynamic_write_initial_scale,
    )
    paired_initialization: dict[str, int | bool | str] = {"enabled": False}
    if args.paired_legacy_initialization and args.paired_dense_initialization:
        raise RuntimeError("paired initialization reference is ambiguous")
    if args.paired_legacy_initialization or args.paired_dense_initialization:
        if (
            args.initialize_trunk_checkpoint is not None
            or args.initialize_chunk_trunk_checkpoint is not None
            or args.initialize_semantic_edge_trunk_checkpoint is not None
            or args.initialize_cnn_pole_trunk_checkpoint is not None
            or args.initialize_slow_cnn_pole_trunk_checkpoint is not None
            or args.initialize_additional_slow_cnn_pole_trunk_checkpoint is not None
        ):
            raise RuntimeError("paired and checkpoint trunk initialization are mutually exclusive")
        if not isinstance(model, AlphabetLM):
            raise RuntimeError("paired initialization requires ALPHABET-LM")
        reference_reader = "dense_k3" if args.paired_dense_initialization else "r2k3"
        copied_tensors, copied_parameters = _copy_matching_legacy_initialization(
            model,
            vocab_size=train.manifest.vocab_size,
            seed=seed,
            reader_type=reference_reader,
        )
        paired_initialization = {
            "enabled": True,
            "copied_tensors": copied_tensors,
            "copied_parameters": copied_parameters,
            "reference_reader": reference_reader,
        }
    trunk_initialization = _initialize_sidecar_from_trunk(
        model,
        args.initialize_trunk_checkpoint,
        freeze_trunk=args.freeze_trunk,
    )
    chunk_trunk_initialization = _initialize_chunk_memory_from_trunk(
        model,
        args.initialize_chunk_trunk_checkpoint,
        train_upper_blocks=args.train_upper_blocks,
    )
    semantic_edge_initialization = _initialize_semantic_edge_from_trunk(
        model,
        args.initialize_semantic_edge_trunk_checkpoint,
    )
    cnn_pole_initialization = _initialize_cnn_pole_from_trunk(
        model,
        args.initialize_cnn_pole_trunk_checkpoint,
    )
    slow_cnn_pole_initialization = _initialize_slow_cnn_pole_from_trunk(
        model,
        args.initialize_slow_cnn_pole_trunk_checkpoint,
    )
    additional_slow_cnn_pole_initialization = (
        _initialize_additional_slow_cnn_poles_from_trunk(
            model,
            args.initialize_additional_slow_cnn_pole_trunk_checkpoint,
        )
    )
    variant_contract = runtime.get("architecture", {}).get("variants", {}).get(run_label)
    if variant_contract is not None:
        active_arguments = {
            "post_hidden": args.post_hidden,
            "memory_readout": args.memory_readout,
            "query_read_rank": args.query_read_rank,
            "query_read_initial_scale": args.query_read_initial_scale,
            "reader_type": args.reader_type,
            "pole_dynamics": args.pole_dynamics,
            "delta_select_rank": args.delta_select_rank,
            "delta_select_initial_scale": args.delta_select_initial_scale,
            "delta_select_control_bound": args.delta_select_control_bound,
            "memory_layout": args.memory_layout,
            "tensor_temporal_modes": args.tensor_temporal_modes,
            "tensor_initial_read_gain": args.tensor_initial_read_gain,
            "sidecar_initial_scale": args.sidecar_initial_scale,
            "sidecar_normalize_memory": args.sidecar_normalize_memory,
            "sidecar_channelwise_scale": args.sidecar_channelwise_scale,
            "sidecar_use_recurrence": args.sidecar_use_recurrence,
            "chunk_memory": args.chunk_memory,
            "chunk_size": args.chunk_size,
            "chunk_summary_width": args.chunk_summary_width,
            "chunk_pole_modes": args.chunk_pole_modes,
            "chunk_upper_blocks": args.chunk_upper_blocks,
            "chunk_beta_initial": args.chunk_beta_initial,
            "chunk_minimum_half_life": args.chunk_minimum_half_life,
            "chunk_maximum_half_life": args.chunk_maximum_half_life,
            "train_upper_blocks": args.train_upper_blocks,
            "upper_block_lr_multiplier": args.upper_block_lr_multiplier,
            "semantic_edge_memory": args.semantic_edge_memory,
            "semantic_edge_stride": args.semantic_edge_stride,
            "semantic_edge_pole_modes": args.semantic_edge_pole_modes,
            "semantic_edge_upper_blocks": args.semantic_edge_upper_blocks,
            "semantic_edge_beta_initial": args.semantic_edge_beta_initial,
            "semantic_edge_use_recurrence": args.semantic_edge_use_recurrence,
            "semantic_edge_minimum_half_life": args.semantic_edge_minimum_half_life,
            "semantic_edge_maximum_half_life": args.semantic_edge_maximum_half_life,
            "cnn_pole_memory": args.cnn_pole_memory,
            "cnn_pole_interval": args.cnn_pole_interval,
            "cnn_pole_modes": args.cnn_pole_modes,
            "cnn_pole_evidence_width": args.cnn_pole_evidence_width,
            "cnn_pole_kernel_size": args.cnn_pole_kernel_size,
            "cnn_pole_beta_initial": args.cnn_pole_beta_initial,
            "cnn_pole_use_recurrence": args.cnn_pole_use_recurrence,
            "cnn_pole_minimum_half_life": args.cnn_pole_minimum_half_life,
            "cnn_pole_maximum_half_life": args.cnn_pole_maximum_half_life,
            "slow_cnn_pole_memory": args.slow_cnn_pole_memory,
            "slow_cnn_pole_stride": args.slow_cnn_pole_stride,
            "slow_cnn_pole_modes": args.slow_cnn_pole_modes,
            "slow_cnn_pole_evidence_width": args.slow_cnn_pole_evidence_width,
            "slow_cnn_pole_kernel_size": args.slow_cnn_pole_kernel_size,
            "slow_cnn_pole_upper_blocks": args.slow_cnn_pole_upper_blocks,
            "slow_cnn_pole_beta_initial": args.slow_cnn_pole_beta_initial,
            "slow_cnn_pole_use_recurrence": args.slow_cnn_pole_use_recurrence,
            "slow_cnn_pole_minimum_half_life": args.slow_cnn_pole_minimum_half_life,
            "slow_cnn_pole_maximum_half_life": args.slow_cnn_pole_maximum_half_life,
            "additional_slow_cnn_pole_depths": list(
                args.additional_slow_cnn_pole_depths
            ),
            "additional_slow_cnn_pole_beta_initial": (
                args.additional_slow_cnn_pole_beta_initial
            ),
            "additional_slow_cnn_pole_use_recurrence": (
                args.additional_slow_cnn_pole_use_recurrence
            ),
            "freeze_trunk": args.freeze_trunk,
            "write_map": args.write_map,
            "dynamic_write_rank": args.dynamic_write_rank,
            "dynamic_write_initial_scale": args.dynamic_write_initial_scale,
        }
        expected_variant = {
            key: value for key, value in active_arguments.items() if key in variant_contract
        }
        if any(variant_contract.get(key) != value for key, value in expected_variant.items()):
            raise RuntimeError("campaign variant arguments changed")
        expected_copied = next(
            (
                variant_contract[key]
                for key in (
                    "paired_legacy_copied_parameters",
                    "paired_dense_copied_parameters",
                )
                if key in variant_contract
            ),
            None,
        )
        if paired_initialization.get("copied_parameters") != expected_copied:
            raise RuntimeError("paired initialization coverage changed")
    model = model.to(device)
    parameters = _parameter_count(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    default_parameters = 34_794_496 if args.model == "alphabet" else 35_425_280
    expected_parameters = runtime.get("parameter_counts", {}).get(run_label, default_parameters)
    if parameters != expected_parameters:
        raise RuntimeError(f"{args.model} parameter contract changed: {parameters}")
    expected_total = runtime.get("total_parameter_counts", {}).get(run_label, total_parameters)
    if total_parameters != expected_total:
        raise RuntimeError(f"{args.model} total parameter contract changed: {total_parameters}")
    named_trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_parameters = [parameter for _name, parameter in named_trainable]
    optimizer_parameters: object = trainable_parameters
    if args.initialize_chunk_trunk_checkpoint is not None:
        if not 0.0 < args.upper_block_lr_multiplier <= 1.0:
            raise RuntimeError("invalid upper-block learning-rate multiplier")
        memory_parameters = [
            parameter for name, parameter in named_trainable if name.startswith("chunk_memory.")
        ]
        upper_parameters = [
            parameter for name, parameter in named_trainable if name.startswith("blocks.")
        ]
        if (
            not memory_parameters
            or not upper_parameters
            or len(memory_parameters) + len(upper_parameters) != len(named_trainable)
        ):
            raise RuntimeError("chunk-memory optimizer ownership is incomplete")
        optimizer_parameters = [
            {"params": memory_parameters, "lr": float(training["learning_rate"])},
            {
                "params": upper_parameters,
                "lr": float(training["learning_rate"]) * args.upper_block_lr_multiplier,
            },
        ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    tokens_per_update = int(training["global_sequences"]) * train.manifest.context_length
    horizon_updates = math.ceil(int(training["horizon_tokens"]) / tokens_per_update)
    scheduler = _scheduler(
        optimizer,
        warmup=int(training["warmup_updates"]),
        horizon=horizon_updates,
    )
    batcher = DeterministicBatcher(
        train,
        seed=seed,
        batch_size=int(training["global_sequences"]),
    )
    target_tokens = (
        args.target_tokens_override
        if args.target_tokens_override is not None
        else int(training["target_tokens"])
    )
    if target_tokens <= 0:
        raise RuntimeError("target token count must be positive")
    extension_source: dict[str, object] = {"enabled": False}
    if args.resume_extension_checkpoint is not None:
        extension_source = {
            "enabled": True,
            "checkpoint": str(args.resume_extension_checkpoint),
            "checkpoint_sha256": sha256_file(args.resume_extension_checkpoint),
        }
    contract = {
        "schema": "lnet.alphabet_lm.viability_10m.v1",
        "campaign_id": runtime["campaign_id"],
        "campaign_manifest_sha256": runtime["campaign_manifest_sha256"],
        "source_commit": os.environ["H200_EXPECTED_COMMIT"],
        "model": model_contract,
        "pole_initialization": args.pole_initialization,
        "memory_banks": args.memory_banks,
        "bank_pole_modes": args.bank_pole_modes,
        "reader_type": args.reader_type,
        "pole_routing": args.pole_routing,
        "post_hidden": args.post_hidden,
        "memory_readout": args.memory_readout,
        "query_read_rank": args.query_read_rank,
        "query_read_initial_scale": args.query_read_initial_scale,
        "pole_dynamics": args.pole_dynamics,
        "delta_select_rank": args.delta_select_rank,
        "delta_select_initial_scale": args.delta_select_initial_scale,
        "delta_select_control_bound": args.delta_select_control_bound,
        "memory_layout": args.memory_layout,
        "tensor_temporal_modes": args.tensor_temporal_modes,
        "tensor_initial_read_gain": args.tensor_initial_read_gain,
        "sidecar_initial_scale": args.sidecar_initial_scale,
        "sidecar_normalize_memory": args.sidecar_normalize_memory,
        "sidecar_channelwise_scale": args.sidecar_channelwise_scale,
        "sidecar_use_recurrence": args.sidecar_use_recurrence,
        "chunk_memory": args.chunk_memory,
        "chunk_size": args.chunk_size,
        "chunk_summary_width": args.chunk_summary_width,
        "chunk_pole_modes": args.chunk_pole_modes,
        "chunk_upper_blocks": args.chunk_upper_blocks,
        "chunk_beta_initial": args.chunk_beta_initial,
        "chunk_minimum_half_life": args.chunk_minimum_half_life,
        "chunk_maximum_half_life": args.chunk_maximum_half_life,
        "chunk_trunk_initialization": chunk_trunk_initialization,
        "train_upper_blocks": args.train_upper_blocks,
        "upper_block_lr_multiplier": args.upper_block_lr_multiplier,
        "semantic_edge_memory": args.semantic_edge_memory,
        "semantic_edge_stride": args.semantic_edge_stride,
        "semantic_edge_pole_modes": args.semantic_edge_pole_modes,
        "semantic_edge_upper_blocks": args.semantic_edge_upper_blocks,
        "semantic_edge_beta_initial": args.semantic_edge_beta_initial,
        "semantic_edge_use_recurrence": args.semantic_edge_use_recurrence,
        "semantic_edge_minimum_half_life": args.semantic_edge_minimum_half_life,
        "semantic_edge_maximum_half_life": args.semantic_edge_maximum_half_life,
        "semantic_edge_initialization": semantic_edge_initialization,
        "cnn_pole_memory": args.cnn_pole_memory,
        "cnn_pole_interval": args.cnn_pole_interval,
        "cnn_pole_modes": args.cnn_pole_modes,
        "cnn_pole_evidence_width": args.cnn_pole_evidence_width,
        "cnn_pole_kernel_size": args.cnn_pole_kernel_size,
        "cnn_pole_beta_initial": args.cnn_pole_beta_initial,
        "cnn_pole_use_recurrence": args.cnn_pole_use_recurrence,
        "cnn_pole_minimum_half_life": args.cnn_pole_minimum_half_life,
        "cnn_pole_maximum_half_life": args.cnn_pole_maximum_half_life,
        "cnn_pole_initialization": cnn_pole_initialization,
        "slow_cnn_pole_memory": args.slow_cnn_pole_memory,
        "slow_cnn_pole_stride": args.slow_cnn_pole_stride,
        "slow_cnn_pole_modes": args.slow_cnn_pole_modes,
        "slow_cnn_pole_evidence_width": args.slow_cnn_pole_evidence_width,
        "slow_cnn_pole_kernel_size": args.slow_cnn_pole_kernel_size,
        "slow_cnn_pole_upper_blocks": args.slow_cnn_pole_upper_blocks,
        "slow_cnn_pole_beta_initial": args.slow_cnn_pole_beta_initial,
        "slow_cnn_pole_use_recurrence": args.slow_cnn_pole_use_recurrence,
        "slow_cnn_pole_minimum_half_life": args.slow_cnn_pole_minimum_half_life,
        "slow_cnn_pole_maximum_half_life": args.slow_cnn_pole_maximum_half_life,
        "slow_cnn_pole_initialization": slow_cnn_pole_initialization,
        "additional_slow_cnn_pole_depths": list(args.additional_slow_cnn_pole_depths),
        "additional_slow_cnn_pole_beta_initial": (
            args.additional_slow_cnn_pole_beta_initial
        ),
        "additional_slow_cnn_pole_use_recurrence": (
            args.additional_slow_cnn_pole_use_recurrence
        ),
        "additional_slow_cnn_pole_initialization": additional_slow_cnn_pole_initialization,
        "write_map": args.write_map,
        "dynamic_write_rank": args.dynamic_write_rank,
        "dynamic_write_initial_scale": args.dynamic_write_initial_scale,
        "paired_legacy_initialization": paired_initialization,
        "trunk_initialization": trunk_initialization,
        "parameters": parameters,
        "total_parameters": total_parameters,
        "target_tokens": target_tokens,
        "extension_source": extension_source,
        "training": training,
        "paper": runtime.get("paper"),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
    }
    contract_sha = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract["contract_sha256"] = contract_sha
    contract_path = args.root / "contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("LM run contract changed under an existing root")
    _atomic_json(contract_path, contract)
    checkpoint = args.root / "checkpoint.pt"
    update = 0
    tokens_seen = 0
    history: list[dict[str, float | int]] = []
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("contract_sha256") != contract_sha:
            raise RuntimeError("LM checkpoint is not bound to the active contract")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        batcher.load_state_dict(payload["batcher"])
        update = int(payload["update"])
        tokens_seen = int(payload["tokens_seen"])
        history = payload["history"]
        torch.set_rng_state(payload["torch_rng_state"])
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        random.setstate(payload["python_rng_state"])
    elif args.resume_extension_checkpoint is not None:
        payload = torch.load(
            args.resume_extension_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        batcher.load_state_dict(payload["batcher"])
        update = int(payload["update"])
        tokens_seen = int(payload["tokens_seen"])
        history = payload["history"]
        torch.set_rng_state(payload["torch_rng_state"])
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        random.setstate(payload["python_rng_state"])
        if tokens_seen >= target_tokens:
            raise RuntimeError("extension checkpoint already reached the requested target")
    wandb_run = _wandb_run(runtime, run_label, args.root, contract)
    runtime_model = cast(
        "nn.Module", torch.compile(model, mode="default", fullgraph=False, dynamic=False)
    )
    grad_steps = int(training["global_sequences"]) // int(training["microbatch"])
    if grad_steps < 1 or int(training["global_sequences"]) % int(training["microbatch"]):
        raise RuntimeError("invalid H200 LM gradient accumulation contract")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    starting_tokens = tokens_seen
    signal.signal(signal.SIGTERM, _signal_stop)
    while tokens_seen < target_tokens:
        batch = batcher.next()
        labels = batch[:, 1:]
        valid_tokens = int((labels != train.manifest.pad_id).sum())
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for micro_cpu in batch.split(int(training["microbatch"])):
            micro = micro_cpu.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_sum, _count = _loss_sum(runtime_model, micro, train.manifest.pad_id)
                scaled = loss_sum / valid_tokens
            scaled.backward()
            loss_total += float(loss_sum.detach())
        torch.nn.utils.clip_grad_norm_(trainable_parameters, float(training["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        update += 1
        tokens_seen += valid_tokens
        elapsed = time.perf_counter() - started
        row: dict[str, float | int] = {
            "update": update,
            "tokens_seen": tokens_seen,
            "train_loss": loss_total / valid_tokens,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "tokens_per_second": (tokens_seen - starting_tokens) / max(elapsed, 1.0e-9),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "gradient_accumulation_steps": grad_steps,
        }
        if len(optimizer.param_groups) > 1:
            row["upper_block_learning_rate"] = optimizer.param_groups[1]["lr"]
        if isinstance(model, AlphabetLM):
            query_scales = []
            decay_scales = []
            dynamic_write_scales = []
            for block in model.blocks:
                readout = block.query_readout
                if isinstance(readout, QueryConditionedLowRankReadout):
                    query_scales.append(float(readout.scale().detach()))
                selector = block.decay_selector
                if isinstance(selector, LowRankDecaySelector):
                    decay_scales.append(float(selector.scale().detach()))
                dynamic_write = block.dynamic_write
                if isinstance(dynamic_write, DynamicLowRankWrite):
                    dynamic_write_scales.append(float(dynamic_write.scale().detach()))
            if query_scales:
                row["query_read_scale_mean"] = sum(query_scales) / len(query_scales)
            if decay_scales:
                row["delta_select_scale_mean"] = sum(decay_scales) / len(decay_scales)
            if dynamic_write_scales:
                row["dynamic_write_scale_mean"] = sum(dynamic_write_scales) / len(
                    dynamic_write_scales
                )
            if model.cnn_pole_memories is not None:
                beta_values = [
                    float(cast("CausalCNNPoleMemory", bank).beta.detach())
                    for bank in model.cnn_pole_memories
                ]
                row["cnn_pole_beta_mean"] = sum(beta_values) / len(beta_values)
            if isinstance(model.slow_cnn_pole_memory, SlowCausalCNNPoleMemory):
                row["slow_cnn_pole_beta_mean"] = float(
                    model.slow_cnn_pole_memory.beta.detach().mean()
                )
            if model.additional_slow_cnn_pole_memories is not None:
                additional_beta = [
                    cast("SlowCausalCNNPoleMemory", bank).beta.detach().mean()
                    for bank in model.additional_slow_cnn_pole_memories
                ]
                row["additional_slow_cnn_pole_beta_mean"] = float(
                    torch.stack(additional_beta).mean()
                )
        if update == 1:
            uniform_loss = math.log(train.manifest.vocab_size)
            if not 0.5 * uniform_loss <= row["train_loss"] <= 2.0 * uniform_loss:
                raise RuntimeError(
                    f"{args.model} first-update loss is invalid: {row['train_loss']}"
                )
        history.append(row)
        checkpoint_due = (
            update % int(training["checkpoint_updates"]) == 0
            or tokens_seen >= target_tokens
            or _STOP_EVENT.is_set()
        )
        if checkpoint_due:
            _save_checkpoint(
                checkpoint,
                contract_sha=contract_sha,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                batcher=batcher,
                update=update,
                tokens_seen=tokens_seen,
                history=history,
            )
            _atomic_json(args.root / "history.json", history)
        wandb_metrics = {
            "train/loss": row["train_loss"],
            "training/tokens": tokens_seen,
            "training/update": update,
            "training/learning_rate": row["learning_rate"],
            "performance/tokens_per_second": row["tokens_per_second"],
            "performance/peak_memory_bytes": row["peak_memory_bytes"],
        }
        if "query_read_scale_mean" in row:
            wandb_metrics["model/query_read_scale_mean"] = row["query_read_scale_mean"]
        if "delta_select_scale_mean" in row:
            wandb_metrics["model/delta_select_scale_mean"] = row["delta_select_scale_mean"]
        if "dynamic_write_scale_mean" in row:
            wandb_metrics["model/dynamic_write_scale_mean"] = row["dynamic_write_scale_mean"]
        if "cnn_pole_beta_mean" in row:
            wandb_metrics["model/cnn_pole_beta_mean"] = row["cnn_pole_beta_mean"]
        if "slow_cnn_pole_beta_mean" in row:
            wandb_metrics["model/slow_cnn_pole_beta_mean"] = row[
                "slow_cnn_pole_beta_mean"
            ]
        if "additional_slow_cnn_pole_beta_mean" in row:
            wandb_metrics["model/additional_slow_cnn_pole_beta_mean"] = row[
                "additional_slow_cnn_pole_beta_mean"
            ]
        wandb_run.log(wandb_metrics, step=update)
        print("ALPHABET_LM_PROGRESS=" + json.dumps(row, sort_keys=True), flush=True)
        if _STOP_EVENT.is_set():
            wandb_run.summary["status"] = "stopped"
            wandb_run.finish(exit_code=143)
            raise SystemExit(143)
    validation_loss = _evaluate(
        runtime_model,
        validation,
        microbatch=int(training["microbatch"]),
        device=device,
    )
    receipt = {
        "schema": "lnet.h200.alphabet_lm.training_receipt.v1",
        "status": "completed",
        "model": args.model,
        "parameters": parameters,
        "total_parameters": total_parameters,
        "tokens_seen": tokens_seen,
        "updates": update,
        "validation_loss": validation_loss,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
        "source_commit": os.environ["H200_EXPECTED_COMMIT"],
        "contract_sha256": contract_sha,
        "checkpoint": str(checkpoint),
        "wandb_url": wandb_run.url,
    }
    _atomic_json(completed, receipt)
    wandb_run.summary.update(receipt)
    wandb_run.finish()
    print("ALPHABET_LM_TRAINING_COMPLETE=" + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
