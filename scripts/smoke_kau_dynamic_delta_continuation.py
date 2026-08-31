#!/usr/bin/env python3
"""Smoke Dense-checkpoint continuation with frozen poles and zero-init dynamic delta."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
from typing import cast

import torch
from torch import nn
from torch.nn import functional
from train_h200_alphabet_lm_10m import _initialize_laplace_continuation

from lnet.alphabet_lm import (
    DynamicDeltaImagePostFusionAlphabet2Block,
    DynamicDeltaImagePostFusionAlphabet2LM,
    LaplaceMambaLMConfig,
    VectorImagePostFusionAlphabet2LM,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.manual_seed(501)
    config = LaplaceMambaLMConfig(conv_width=3)
    static = VectorImagePostFusionAlphabet2LM(config)
    dynamic = DynamicDeltaImagePostFusionAlphabet2LM(config)
    _initialize_laplace_continuation(
        static,
        args.checkpoint,
        freeze_poles=True,
    )
    metadata, _payload = _initialize_laplace_continuation(
        dynamic,
        args.checkpoint,
        freeze_poles=True,
    )
    static = static.cuda().eval()
    dynamic = dynamic.cuda().eval()
    tokens = torch.randint(0, config.vocab_size, (1, 33), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        static_logits = static(tokens)
        dynamic_logits = dynamic(tokens)
    torch.testing.assert_close(dynamic_logits, static_logits)
    dynamic.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in dynamic.parameters() if parameter.requires_grad],
        lr=3.0e-5,
        fused=True,
    )
    compiled = cast("nn.Module", torch.compile(dynamic, fullgraph=False, dynamic=False))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = compiled(tokens[:, :-1])
        loss = functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    first_block = cast("DynamicDeltaImagePostFusionAlphabet2Block", dynamic.blocks[0])
    controller = first_block.delta.output.weight
    if controller.grad is None or not torch.isfinite(controller.grad).all():
        raise RuntimeError("dynamic continuation controller has no finite gradient")
    optimizer.step()
    if any(
        parameter.requires_grad
        for name, parameter in dynamic.named_parameters()
        if name.endswith(("memory.raw_damping", "memory.raw_frequency"))
    ):
        raise RuntimeError("dynamic continuation did not freeze Laplace poles")
    details = f"loss={float(loss.detach()):.6f},missing={metadata['missing_delta_tensors']}"
    print(f"DYNAMIC_DELTA_CONTINUATION_SMOKE={details}", flush=True)


if __name__ == "__main__":
    main()
