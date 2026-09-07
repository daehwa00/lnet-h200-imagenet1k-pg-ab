"""Bounded K64 memory/finite train-eval smoke; never writes training checkpoints."""
from __future__ import annotations

import argparse
import json
import time

import torch
import run_lnet_k64_p80_d2262_imagenet1k as runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memory-gib", type=float, default=16.0)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(501)
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    cap = min(args.memory_gib * 1024**3, total * 0.9)
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = runner._build_model(runner.MODEL_KEY, None, 1000).to(device)
    model = model.to(memory_format=torch.channels_last)
    active = model if args.eager else torch.compile(model, mode="default", dynamic=False)
    optimizer = runner.worker._build_optimizer(model, 0.003, device)
    inputs = torch.randn(args.batch_size, 3, 224, 224, device=device).contiguous(memory_format=torch.channels_last)
    targets = torch.randint(1000, (args.batch_size,), device=device)
    started = time.monotonic()
    losses = []
    for cycle in range(2):
        active.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = active(inputs)
            loss = torch.nn.functional.cross_entropy(logits.float(), targets)
        loss.backward()
        if not bool(torch.isfinite(loss)) or not all(
            bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
        ):
            raise FloatingPointError("nonfinite K64 training loss or gradients")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        active.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in (inputs, inputs[:80]):
                logits = active(batch)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("nonfinite K64 evaluation logits")
        print(f"K64_SMOKE_CYCLE={cycle + 1}", flush=True)
    torch.cuda.synchronize()
    print("K64_MIG1_SMOKE_JSON=" + json.dumps({
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "parameters": runner.EXPECTED_PARAMETERS,
        "compile": not args.eager,
        "memory_cap_gib": cap / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "seconds": time.monotonic() - started,
        "losses": losses,
        "finite": True,
    }), flush=True)


if __name__ == "__main__":
    main()
