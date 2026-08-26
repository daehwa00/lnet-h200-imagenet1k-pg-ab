#!/usr/bin/env python3
"""Compiled full-context RTX 4090 gate for pole-init variants and Mamba."""

from __future__ import annotations

import argparse
import json
import math
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.alphabet_lm import AlphabetLM, AlphabetLMConfig
from lnet.alphabet_lm_mamba import MambaLMConfig, build_parameter_matched_mamba


def _loss(model: nn.Module, tokens: Tensor) -> Tensor:
    logits = model(tokens[:, :-1])
    return functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())


def _step(model: nn.Module, tokens: Tensor) -> dict[str, float]:
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.1, fused=True)
    compiled = cast("nn.Module", torch.compile(model, fullgraph=False, dynamic=False))
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = _loss(compiled, tokens).float()
    initial_loss = float(loss.detach())
    if not 0.5 * math.log(32_768) <= initial_loss <= 2.0 * math.log(32_768):
        raise RuntimeError(f"invalid RTX 4090 initial loss: {initial_loss}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "initial_loss": initial_loss,
        "peak_memory_bytes": float(torch.cuda.max_memory_allocated()),
        "parameters": float(sum(parameter.numel() for parameter in model.parameters())),
    }
    del compiled, optimizer, model
    torch.compiler.reset()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("all", "palette", "grouped", "dense", "routing"),
        default="all",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name().upper():
        raise RuntimeError("KAU ALPHABET-LM smoke requires the RTX 4090")
    torch.manual_seed(501)
    tokens = torch.randint(32_768, (2, 2_049), device="cuda")
    results = {}
    alphabet_variants = (
        ("alphabet-legacy", "legacy"),
        ("alphabet-palette", "lifetime_palette"),
    )
    if args.only == "palette":
        alphabet_variants = alphabet_variants[1:]
    elif args.only in {"grouped", "dense", "routing"}:
        alphabet_variants = ()
    for label, initialization in alphabet_variants:
        torch.manual_seed(501)
        results[label] = _step(
            AlphabetLM(
                AlphabetLMConfig(
                    pole_initialization=cast(
                        "Literal['legacy', 'lifetime_palette']", initialization
                    )
                )
            ),
            tokens,
        )
    if args.only == "grouped":
        torch.manual_seed(501)
        grouped = AlphabetLM(
            AlphabetLMConfig(
                pole_initialization="lifetime_palette",
                memory_banks=8,
                bank_pole_modes=128,
            )
        )
        if sum(parameter.numel() for parameter in grouped.parameters()) != 31_373_824:
            raise RuntimeError("grouped H8P128 parameter contract changed")
        results["alphabet-grouped-h8p128"] = _step(grouped, tokens)
    if args.only == "dense":
        torch.manual_seed(501)
        dense = AlphabetLM(AlphabetLMConfig(reader_type="dense_k3"))
        if sum(parameter.numel() for parameter in dense.parameters()) != 36_714_496:
            raise RuntimeError("dense K3 P320 parameter contract changed")
        results["alphabet-dense-k3-p320"] = _step(dense, tokens)
    if args.only == "routing":
        torch.manual_seed(501)
        routed = AlphabetLM(AlphabetLMConfig(pole_routing="dynamic_write_read"))
        if sum(parameter.numel() for parameter in routed.parameters()) != 35_239_936:
            raise RuntimeError("dynamic write/read parameter contract changed")
        results["alphabet-dynamic-write-read"] = _step(routed, tokens)
    if args.only == "all":
        torch.manual_seed(501)
        mamba, parameters, relative_error = build_parameter_matched_mamba(
            34_794_496, MambaLMConfig()
        )
        if parameters != 35_425_280 or relative_error >= 0.03:
            raise RuntimeError("RTX 4090 Mamba parameter match changed")
        results["mamba"] = _step(mamba, tokens)
    print("KAU_ALPHABET_LM_SMOKE=" + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
