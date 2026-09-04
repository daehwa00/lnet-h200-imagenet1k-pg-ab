from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
from pathlib import Path

import torch

from scripts import run_lnet_k128_h200_imagenet1k_queue as queue
from scripts import run_lnet_k128_p160_160_160_128_d2262_h200_imagenet1k as runner


def test_k128_h200_contract_is_exact() -> None:
    assert runner.VARIANT == "A-K128-P160-160-160-128-D2262"
    assert runner.SEEDS == (509, 521)
    assert runner.EXPECTED_PARAMETERS == 5_083_176
    assert queue.SEEDS == runner.SEEDS
    assert queue.MODEL_KEY == runner.MODEL_KEY


def test_k128_h200_task_uses_one_physical_batch(tmp_path: Path) -> None:
    task = runner._task(
        argparse.Namespace(
            output_root=tmp_path,
            data_root=tmp_path / "data",
            seed=509,
            batch_size=256,
            workers=8,
            wandb_mode="disabled",
        )
    )
    assert task.batch_size == 256
    assert task.gradient_accumulation_steps == 1
    assert task.epochs == 100
    assert task.resume is False


def test_k128_h200_build_and_primary_logits() -> None:
    model = runner._build_model(runner.MODEL_KEY, None, 1000).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_083_176
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 1000)


def test_h200_entrypoint_uses_reduce_overhead() -> None:
    source = (Path(__file__).resolve().parents[1] / "h200/run_baselines.sh").read_text()
    assert "refs/heads/control/imagenet1k-lnet-k128" in source
    assert "H200_LNET_K128_ONLY" in source
    assert "H200_BASELINE_TORCH_COMPILE_MODE=reduce-overhead" in source
    assert "H200_BASELINE_COMPILED_TRAINING_PREPARATION=1" in source
    assert "--batch-size 256" in source
