#!/usr/bin/env python3
"""Smoke the two canonical K64 P-allocation H200 cells without redefining them."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import argparse
import json
import random
from pathlib import Path
from typing import Any

import a2d_r2k3_runtime as runtime
import imagenet100_checkpoint_runtime as checkpoint_runtime
import numpy as np
import run_a2d_r2k3_k64_p_allocation_d2262_imagenet100 as experiment
import smoke_a2d_r2k3_same_resolution_factorial as shared
import torch

VARIANTS = experiment.H200_VARIANTS
RECIPE = {
    "fused_optimizer": True,
    "learning_rate": 3.0e-3,
    "modal_learning_rate_multiplier": 1.0 / 3.0,
    "pole_geometry_learning_rate_multiplier": 0.1,
    "weight_decay": 0.05,
}


def _logits(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _optimizer_resume_roundtrip(variant: str, root: Path, batch_size: int) -> dict[str, Any]:
    torch.compiler.reset()
    torch.manual_seed(501)
    random.seed(501)
    model = experiment._build(variant, runtime.model_config()).cuda()
    model.prepare_for_compiled_training_()
    optimizer = runtime.build_optimizer(model, RECIPE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    inputs = torch.randn(batch_size, 3, 224, 224, device="cuda")
    compiled = torch.compile(model, mode="default", fullgraph=False, dynamic=False)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = _logits(compiled(inputs)).float().square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if not bool(torch.isfinite(loss)) or not optimizer.state:
        raise RuntimeError(f"{variant} optimizer smoke did not create finite state")

    training_generator = torch.Generator().manual_seed(501)
    mixup_generator = np.random.default_rng(501)
    checkpoint = root / f"{variant}__resume.pt"
    saved_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint_runtime.atomic_torch(
        checkpoint,
        {
            "variant": variant,
            "seed": 501,
            "epoch": 1,
            "global_step": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": [{"epoch": 1.0, "validation_accuracy": 0.0}],
            "training_seconds": 1.0,
            "training_generator_state": training_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
            "mixup_rng_state": mixup_generator.bit_generator.state,
        },
    )

    del compiled, optimizer, model
    torch.cuda.empty_cache()
    torch.compiler.reset()

    restored = experiment._build(variant, runtime.model_config()).cuda()
    restored.prepare_for_compiled_training_()
    restored_optimizer = runtime.build_optimizer(restored, RECIPE)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lambda _epoch: 1.0,
    )
    restored_training_generator = torch.Generator().manual_seed(999)
    restored_mixup_generator = np.random.default_rng(999)
    progress = {"global_step": -1}
    start_epoch, history, training_seconds = checkpoint_runtime.restore_checkpoint(
        checkpoint,
        variant=variant,
        seed=501,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        training_generator=restored_training_generator,
        mixup_generator=restored_mixup_generator,
        progress=progress,
        optimizer_steps_per_epoch=1,
    )
    checkpoint_runtime.restore_optimizer_runtime_options(
        restored_optimizer,
        RECIPE,
    )
    maximum_error = max(
        float((value.detach().cpu() - saved_model[name]).abs().max())
        for name, value in restored.state_dict().items()
    )
    if (
        start_epoch != 1
        or progress["global_step"] != 1
        or len(history) != 1
        or training_seconds != 1.0
        or maximum_error != 0.0
        or not restored_optimizer.state
    ):
        raise RuntimeError(f"{variant} exact optimizer resume contract changed")

    resumed = torch.compile(restored, mode="default", fullgraph=False, dynamic=False)
    restored_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        resumed_loss = _logits(resumed(inputs)).float().square().mean()
    resumed_loss.backward()
    restored_optimizer.step()
    if not bool(torch.isfinite(resumed_loss)):
        raise RuntimeError(f"{variant} resumed optimizer step is non-finite")

    payload = {
        "checkpoint_max_abs": maximum_error,
        "initial_loss": float(loss.detach()),
        "resumed_loss": float(resumed_loss.detach()),
        "start_epoch": start_epoch,
        "global_step": progress["global_step"],
        "optimizer_state_entries": len(restored_optimizer.state),
    }
    checkpoint.unlink(missing_ok=True)
    del inputs, resumed, restored_optimizer, restored_scheduler, restored
    torch.cuda.empty_cache()
    torch.compiler.reset()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeat-steps", type=int, default=3)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    shared.runner = experiment  # type: ignore[reportConstantRedefinition]
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested without an available GPU")

    payload: dict[str, Any] = {"device": str(device), "variants": {}}
    for variant in VARIANTS:
        evidence = shared._run_candidate(
            variant,
            device,
            args.root,
            compile_model=False,
        )
        if device.type == "cuda":
            evidence["full_batch"] = shared._full_batch_cuda_step(
                variant,
                args.batch_size,
                args.repeat_steps,
            )
            evidence["optimizer_resume"] = _optimizer_resume_roundtrip(
                variant,
                args.root,
                args.batch_size,
            )
        payload["variants"][variant] = evidence
        print(json.dumps({"event": "smoke_passed", "variant": variant}), flush=True)
    payload["status"] = "passed"
    (args.root / "smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
