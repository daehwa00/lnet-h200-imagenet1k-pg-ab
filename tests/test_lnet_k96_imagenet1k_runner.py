from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
from pathlib import Path

import torch

from scripts import run_lnet_k96_imagenet1k_queue as queue
from scripts import run_lnet_k96_p128_d2262_imagenet1k as runner


def test_lnet_k96_imagenet1k_contract_is_frozen() -> None:
    assert runner.VARIANT == "XL-K96-U125"
    assert runner.MODEL_KEY == "lnet_k96_p128x4_d2262_optimized_v2"
    assert runner.SEEDS == (501, 509, 521)
    assert runner.EPOCHS == 100
    assert runner.LEARNING_RATE == 3.0e-3
    assert runner.EXPECTED_PARAMETERS == 3_253_224
    assert runner.WANDB_GROUP == "h200-imagenet1k-moga-emo-100ep-s501-v2"
    assert queue.SEEDS == runner.SEEDS


def test_lnet_k96_imagenet1k_uses_matched_baseline_task(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_root=tmp_path / "output",
        data_root=tmp_path / "imagenet",
        seed=509,
        batch_size=256,
        workers=8,
        wandb_mode="disabled",
    )
    task = runner._task(args)
    assert task.epochs == 100
    assert task.batch_size == 256
    assert task.gradient_accumulation_steps == 1
    assert task.workers == 8
    assert task.learning_rate == 3.0e-3
    assert task.seed == 509
    assert task.resume is False


def test_lnet_k96_imagenet1k_builder_has_exact_parameter_count() -> None:
    model = runner._build_model(runner.MODEL_KEY, None, 1000)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_253_224


def test_lnet_k96_imagenet1k_exposes_only_primary_logits() -> None:
    model = runner._build_model(runner.MODEL_KEY, None, 1000).eval()
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 1000)
    assert bool(torch.isfinite(logits).all())


def test_h200_entrypoint_selects_only_lnet_k96_queue() -> None:
    source = (Path(__file__).resolve().parents[1] / "h200/run_baselines.sh").read_text()
    assert "H200_LNET_K96_ONLY:-0" in source
    assert "scripts/run_lnet_k96_imagenet1k_queue.py" in source
    assert "scripts/run_lnet_k96_p128_d2262_imagenet1k.py" in source
    assert "refs/heads/control/imagenet1k-lnet-k96" in source
    assert "--batch-size 256" in source
    assert "--workers 8" in source
    assert "H200_BASELINE_TORCH_COMPILE_MODE=default" in source
    assert "H200_BASELINE_COMPILED_TRAINING_PREPARATION=1" in source
    assert "H200_LNET_K96_RESUME_ROOT" in source
    assert "H200_ALLOW_PERFORMANCE_ONLY_CHECKPOINT_MIGRATION=1" in source
