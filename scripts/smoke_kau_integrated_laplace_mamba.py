#!/usr/bin/env python3
"""CUDA train/save/load smoke for the integrated Laplace-Mamba block."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from lnet.alphabet_lm import LaplaceMambaLM, LaplaceMambaLMConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address-norm-bias", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(501)
    torch.cuda.manual_seed_all(501)
    config = LaplaceMambaLMConfig(
        vocab_size=256,
        model_width=64,
        layers=2,
        pole_modes=4,
        state_size=2,
        head_width=8,
        conv_width=4,
        context_length=64,
        minimum_half_life=4.0,
        maximum_half_life=64.0,
        address_norm_bias=args.address_norm_bias,
    )
    model = LaplaceMambaLM(config).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, fused=True)
    compiled = cast("nn.Module", torch.compile(model, fullgraph=False, dynamic=False))
    tokens = torch.randint(0, config.vocab_size, (2, 65), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = compiled(tokens[:, :-1])
        loss = functional.cross_entropy(
            logits.flatten(0, 1),
            tokens[:, 1:].flatten(),
        )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = LaplaceMambaLM(config).cuda().eval()
        restored.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            restored_logits = restored(tokens[:, :-1])
    if not torch.isfinite(restored_logits).all():
        raise RuntimeError("restored Laplace-Mamba output is non-finite")
    details = (
        f"loss={float(loss.detach()):.6f},shape={tuple(logits.shape)},"
        f"address_norm_bias={args.address_norm_bias}"
    )
    print(f"INTEGRATED_LAPLACE_MAMBA_SMOKE={details}", flush=True)


if __name__ == "__main__":
    main()
