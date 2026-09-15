from __future__ import annotations

import json
from pathlib import Path

import validate_dense_transfer_runtime as readiness
import run_dense_transfer as training
import torch
from torch import nn
from dense_transfer.engine import OptimizerPlan, UpdateScheduler


CONTRACT = {"fixture": "readiness"}
REQUIRED = {
    "fullinput_train_finite": True,
    "eval_subset_finite": True,
    "restore_model_exact": True,
    "restore_optimizer_exact": True,
    "restore_scheduler_exact": True,
    "restore_rng_exact": True,
    "restore_eval_exact": True,
    "inspect_contract_hash": "fixture-inspect-hash",
}


def test_readiness_is_ready_only_when_every_gate_passes(tmp_path: Path) -> None:
    path = readiness.write_readiness(tmp_path, CONTRACT, dict(REQUIRED), True)
    payload = json.loads(path.read_text())
    assert payload["status"] == "ready"
    assert payload["contract_hash"] == readiness.contract_hash(CONTRACT)


def test_bad_restore_or_missing_eval_cannot_grant_readiness(tmp_path: Path) -> None:
    bad_restore = {**REQUIRED, "restore_rng_exact": False}
    path = readiness.write_readiness(tmp_path, CONTRACT, bad_restore, True)
    assert json.loads(path.read_text())["status"] == "needs_attention"
    no_evaluation = dict(REQUIRED)
    no_evaluation.pop("eval_subset_finite")
    path = readiness.write_readiness(tmp_path, CONTRACT, no_evaluation, True)
    assert json.loads(path.read_text())["status"] == "needs_attention"


def test_cpu_probe_is_bounded_and_does_not_change_schedule_budget(tmp_path: Path, monkeypatch) -> None:
    args = readiness.parser().parse_args([
        "--task", "ade20k", "--model", "va_k128", "--checkpoint", str(tmp_path / "source.pt"),
        "--data-root", str(tmp_path), "--output-root", str(tmp_path / "out"), "--device", "cpu",
        "--workers", "0", "--no-bf16", "--max-probe-updates", "2",
    ])
    # Dropout exposes RNG divergence; the evaluator below also consumes global
    # Torch RNG just like a loader iterator can.
    model = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(1, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = UpdateScheduler(optimizer, OptimizerPlan(epochs=0, learning_rate=0.1, weight_decay=0.0, warmup_updates=0), 100)
    batch = (torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    seen = {}

    def runtime(_settings, plan):
        seen["epochs"] = plan.epochs
        return model, [batch] * 16, [batch], optimizer, scheduler, CONTRACT, torch.device("cpu")

    monkeypatch.setattr(training, "build_runtime", runtime)
    def eval_consumes_rng(*_args):
        torch.rand(3)
        return {"score": 1.0}

    monkeypatch.setattr(training, "evaluate", eval_consumes_rng)
    monkeypatch.setattr(training, "runtime_contract", lambda *_args: {"inspect": "fixture"})
    monkeypatch.setattr(readiness.importlib, "import_module", lambda _name: object())
    result = readiness.run_validation(args)
    assert result["status"] == "ready", result["evidence"]
    assert seen["epochs"] == 0
    assert scheduler.total_updates == 100
    assert scheduler.update_count == 0  # disposable probe restores its initial state
