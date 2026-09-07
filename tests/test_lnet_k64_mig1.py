from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_lnet_k64_mig1_imagenet1k_queue as queue
from scripts import run_lnet_k64_p80_d2262_imagenet1k as runner

ROOT = Path(__file__).resolve().parents[1]


def test_model_and_task_contract(tmp_path: Path) -> None:
    assert runner.VARIANT == "L-K64-U125"
    assert runner.family.SPECS[runner.VARIANT].excitation_modes == (64,) * 4
    assert runner.family.SPECS[runner.VARIANT].pole_modes == (80,) * 4
    assert runner.family.SPECS[runner.VARIANT].depth == (2, 2, 6, 2)
    assert runner.SEEDS == queue.SEEDS == (501, 509, 521)
    assert runner.MODEL_KEY == queue.MODEL_KEY
    args = argparse.Namespace(output_root=tmp_path, data_root=tmp_path / "data",
        seed=501, batch_size=256, workers=8, wandb_mode="disabled")
    task = runner._task(args)
    assert task.epochs == 100 and task.learning_rate == 0.003
    assert task.batch_size == 256 and task.gradient_accumulation_steps == 1
    assert not task.resume
    task.checkpoint_path.parent.mkdir(parents=True)
    task.checkpoint_path.touch()
    assert runner._task(args).resume


def test_forward_and_source_closure() -> None:
    model = runner._build_model(runner.MODEL_KEY, None, 1000).eval()
    assert sum(p.numel() for p in model.parameters()) == 1_562_760
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 1000) and bool(torch.isfinite(output).all())
    assert len(runner._source_fingerprint()) == 64


def test_wandb_identity_matches_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = json.loads((ROOT / "h200/baselines/wandb.runtime.json").read_text())
    with monkeypatch.context() as isolated:
        isolated.setattr(os, "environ", os.environ.copy())
        for seed in runner.SEEDS:
            runner._configure_wandb("online", seed)
            record = runtime["runs"][runner.MODEL_KEY]["seeds"][str(seed)]
            assert os.environ["H200_BASELINE_RUN_ID"] == record["id"]
            assert os.environ["H200_BASELINE_DISPLAY_NAME"] == record["display_name"]
            assert json.loads(os.environ["H200_BASELINE_TAGS_JSON"]) == record["tags"]
            assert record["id"] == hashlib.sha256(
                f"{runner.WANDB_GROUP}:{runner.MODEL_KEY}:seed{seed}".encode()).hexdigest()[:16]


def result(seed: int, epochs: int = 100) -> dict:
    return dict(model_key=queue.MODEL_KEY, seed=seed, phase="full", status="completed",
        completed_epochs=epochs, requested_epochs=100, stopped_at_max_steps=False)


def test_result_completion_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    assert not queue._complete_result(path, 501)
    path.write_text(json.dumps(result(501, 2)))
    assert not queue._complete_result(path, 501)
    path.write_text(json.dumps(result(501)))
    assert queue._complete_result(path, 501)
    with pytest.raises(RuntimeError, match="identity"):
        queue._complete_result(path, 509)


def test_queue_failure_isolation_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["queue", "--data-root", str(tmp_path / "data"),
        "--output-root", str(tmp_path), "--runner", "runner.py"])
    calls = []
    def run(command, check):
        seed = int(command[command.index("--seed") + 1])
        calls.append(seed)
        if seed != 509:
            path = tmp_path / queue.MODEL_KEY / f"seed_{seed}" / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result(seed)))
        return SimpleNamespace(returncode=1 if seed == 509 else 0)
    monkeypatch.setattr(queue.subprocess, "run", run)
    assert queue.main() == 1
    assert calls == [501, 509, 521]
    calls.clear()
    assert queue.main() == 1
    assert calls == [509]


def test_entrypoint_isolation_and_optimizations() -> None:
    source = (ROOT / "h200/run_baselines.sh").read_text()
    assert "refs/heads/control/imagenet1k-k64-mig1-lee" in source
    assert '/app/output/${OUTPUT_USER}/' in source
    assert '"${OUTPUT_USER}" != "Lee-Wonwoo1"' in source
    for flag in ("LNET_GPU_MIXUP=1", "LNET_YIELD_BEFORE_FETCH=1", "LNET_LOADER_CONTEXT=spawn",
                 "LNET_VALIDATION_PERSISTENT=1"):
        assert flag in source
    assert source.index("scripts/smoke_lnet_k64_mig1.py") < source.index("scripts/run_lnet_k64_mig1_imagenet1k_queue.py")
    assert "--batch-size 256" in source
