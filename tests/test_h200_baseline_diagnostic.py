from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import diagnose_h200_baseline_backlog as diagnostic


def test_diagnostic_reports_checkpoint_logs_and_redacts_secrets(tmp_path: Path) -> None:
    output = tmp_path / "app" / "output" / "daehwa00" / "campaign"
    control = tmp_path / "app" / "output" / "daehwa00" / "control"
    task = output / "run-data992f76a6fb09" / "full" / "emov2_1m" / "seed_501"
    telemetry = task / "telemetry"
    telemetry.mkdir(parents=True)
    control.mkdir(parents=True)
    checkpoint = {
        "completed_epochs": 7,
        "contract_sha256": "a" * 64,
        "global_step": 35028,
        "history": [{"epoch": 7, "validation": {"accuracy": 0.42}}],
        "parameters": 1_000_000,
        "task_sha256": "b" * 64,
        "training_seconds": 123.5,
    }
    torch.save(checkpoint, task / "checkpoint.pt")
    (task / "worker.log").write_text(
        "epoch seven\nWANDB_API_KEY=must-not-leak\nRuntimeError: stopped\n",
        encoding="utf-8",
    )
    task_name = "emov2_1m__full__seed501__lr0p003"
    (telemetry / f"{task_name}.jsonl").write_text(
        json.dumps({"id": "epoch-7", "metrics": {"epoch": 7}, "step": 35028})
        + "\n",
        encoding="utf-8",
    )
    (output / "run-data992f76a6fb09" / "queue-status.json").write_text(
        json.dumps({"schema": "queue", "jobs": {}}),
        encoding="utf-8",
    )
    (control / "stopped.json").write_text(
        json.dumps({"generation": 4, "action": "stop"}),
        encoding="utf-8",
    )

    payload = diagnostic.diagnose(
        output,
        control,
        allowed_root=tmp_path / "app" / "output" / "daehwa00",
    )

    model = payload["models"]["emov2_1m"]
    assert model["checkpoint"]["completed_epochs"] == 7
    assert model["telemetry"]["records"] == 1
    assert "must-not-leak" not in model["worker_log"]["tail"]
    assert "<redacted>" in model["worker_log"]["tail"]
    assert payload["owner_stop"]["generation"] == 4


def test_diagnostic_entrypoint_requires_an_immutable_commit() -> None:
    script = (Path(__file__).parents[1] / "h200/diagnose_baseline_backlog.sh").read_text()
    assert "H200_EXPECTED_DIAGNOSTIC_COMMIT" in script
    assert "git status --porcelain" in script
    assert "scripts/diagnose_h200_baseline_backlog.py" in script
