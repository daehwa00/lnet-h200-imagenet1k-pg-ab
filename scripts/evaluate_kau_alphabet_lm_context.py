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

from lnet.alphabet_lm import AlphabetLM, AlphabetLMConfig, GroupedPackedComplexLinear
from lnet.alphabet_lm_data import TokenBlockDataset
from lnet.alphabet_lm_mamba import MambaLM, MambaLMConfig
from lnet.pac_complex_layers import PackedComplexLinear


def _build(kind: str) -> nn.Module:
    if kind == "legacy":
        return AlphabetLM(AlphabetLMConfig())
    if kind == "grouped":
        return AlphabetLM(
            AlphabetLMConfig(
                pole_initialization="lifetime_palette",
                memory_banks=8,
                bank_pole_modes=128,
            )
        )
    return MambaLM(MambaLMConfig())


def _zero_memory(model: nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
    if not isinstance(model, AlphabetLM):
        return []

    def zero_output(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        real, imag = cast("tuple[Tensor, Tensor]", output)
        return torch.zeros_like(real), torch.zeros_like(imag)

    writers = [
        module
        for module in model.modules()
        if isinstance(module, (PackedComplexLinear, GroupedPackedComplexLinear))
    ]
    return [writer.register_forward_hook(zero_output) for writer in writers]


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
    while total_tokens < token_limit and sequence_index < len(dataset):
        tokens = dataset[sequence_index]
        sequence_index += 1
        inputs = tokens[:-1].reshape(-1, segment)
        labels = tokens[1:].reshape(-1, segment)
        pending_inputs.extend(inputs)
        pending_labels.extend(labels)
        while len(pending_inputs) >= segment_batch or (
            sequence_index == len(dataset) and pending_inputs
        ):
            take = min(segment_batch, len(pending_inputs))
            active_inputs = torch.stack(pending_inputs[:take]).to(device, non_blocking=True)
            active_labels = torch.stack(pending_labels[:take]).to(device, non_blocking=True)
            del pending_inputs[:take], pending_labels[:take]
            loss, count = _loss_sum(model, active_inputs, active_labels, dataset.manifest.pad_id)
            total_loss += loss
            total_tokens += count
            if total_tokens >= token_limit:
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
    parser.add_argument("--kind", choices=("legacy", "grouped", "mamba"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--token-limit", type=int, default=1_000_000)
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
            model, dataset, segment=2_048, token_limit=args.token_limit, device=device
        )
    }
    if args.kind != "mamba":
        handles = _zero_memory(model)
        results["memory_zero"] = _evaluate(
            model, dataset, segment=2_048, token_limit=args.token_limit, device=device
        )
        for handle in handles:
            handle.remove()
    for segment in (128, 32, 1):
        results[f"reset_{segment}"] = _evaluate(
            model, dataset, segment=segment, token_limit=args.token_limit, device=device
        )
    payload = {"schema": "lnet.kau.lm_context_diagnostic.v1", "kind": args.kind, **results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("KAU_LM_CONTEXT=" + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
