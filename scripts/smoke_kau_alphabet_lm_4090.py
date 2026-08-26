#!/usr/bin/env python3
"""Compiled full-context RTX 4090 gate for pole-init variants and Mamba."""

from __future__ import annotations

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
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name().upper():
        raise RuntimeError("KAU ALPHABET-LM smoke requires the RTX 4090")
    torch.manual_seed(501)
    tokens = torch.randint(32_768, (2, 2_049), device="cuda")
    results = {}
    for label, initialization in (
        ("alphabet-legacy", "legacy"),
        ("alphabet-palette", "lifetime_palette"),
    ):
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
