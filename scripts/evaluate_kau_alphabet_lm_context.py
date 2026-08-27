#!/usr/bin/env python3
"""Measure fixed-pole memory and effective context use from trained checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.alphabet_lm import (
    AlphabetLM,
    AlphabetLMConfig,
    DynamicLowRankWrite,
    FixedComplexPoleMemory1D,
    GroupedPackedComplexLinear,
    LowRankDecaySelector,
    QueryConditionedLowRankReadout,
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
    return AlphabetLM(config)


def _zero_memory(model: nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []

    def zero_output(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        real, imag = cast("tuple[Tensor, Tensor]", output)
        return torch.zeros_like(real), torch.zeros_like(imag)

    writers = [
        module
        for module in model.modules()
        if isinstance(
            module,
            (PackedComplexLinear, GroupedPackedComplexLinear, TensorProductPoleMemory1D),
        )
    ]
    return [writer.register_forward_hook(zero_output) for writer in writers]


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
    for segment in (128, 32, 1):
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("KAU_LM_CONTEXT=" + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
