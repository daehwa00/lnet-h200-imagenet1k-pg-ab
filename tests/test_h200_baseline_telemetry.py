from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# ruff: noqa: ANN401, TC002, TC003
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import run_h200_baseline_telemetry as telemetry


def test_sidecar_replays_spool_with_strict_settings_and_marks_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spool = tmp_path / "telemetry.jsonl"
    spool.write_text(
        json.dumps({"id": "epoch:1", "step": 1, "metrics": {"epoch": 1.0}}) + "\n"
    )
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps({"final_validation": {"accuracy": 0.5, "top5_accuracy": 0.8}})
    )
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"schema": "test"}))
    stop = tmp_path / "stop.json"
    stop.write_text("{}")
    complete = tmp_path / "complete.json"
    cursor = tmp_path / "cursor.json"
    captured: dict[str, Any] = {}

    class Run:
        def __init__(self) -> None:
            self.summary: dict[str, float] = {}
            self.logged: list[tuple[int, dict[str, float]]] = []
            self.finished = False

        def log(self, metrics: dict[str, float], *, step: int) -> None:
            self.logged.append((step, metrics))

        def finish(self) -> None:
            self.finished = True

    run = Run()

    def settings(**kwargs: Any) -> dict[str, Any]:
        captured["settings"] = kwargs
        return kwargs

    def initialize(**kwargs: Any) -> Run:
        captured["init"] = kwargs
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Settings=settings, init=initialize))
    monkeypatch.setattr(
        telemetry,
        "_arguments",
        lambda: SimpleNamespace(
            spool=spool,
            result=result,
            stop=stop,
            cursor=cursor,
            complete_marker=complete,
            contract=contract,
            model_key="test_model",
            seed=501,
        ),
    )
    monkeypatch.setenv("H200_BASELINE_RUN_ID", "1" * 16)
    monkeypatch.setenv("H200_BASELINE_DISPLAY_NAME", "H200-BL-test-s501")
    monkeypatch.setenv("H200_BASELINE_TAGS_JSON", json.dumps(["ip-scoped-untrusted"]))
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_GROUP", "test-group")

    assert telemetry.main() == 0
    assert run.logged == [(1, {"epoch": 1.0})]
    assert run.finished is True
    assert complete.is_file()
    assert json.loads(cursor.read_text())["synced_ids"] == ["epoch:1"]
    assert captured["init"]["anonymous"] == "never"
    assert captured["settings"]["console"] == "off"
    assert captured["settings"]["disable_job_creation"] is True
    assert captured["settings"]["x_disable_viewer"] is True


def test_sidecar_cursor_prevents_duplicate_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spool = tmp_path / "telemetry.jsonl"
    spool.write_text(
        json.dumps({"id": "epoch:1", "step": 1, "metrics": {"epoch": 1.0}}) + "\n"
    )
    cursor = tmp_path / "cursor.json"
    cursor.write_text(json.dumps({"synced_ids": ["epoch:1"]}))
    monkeypatch.setattr(
        telemetry,
        "_arguments",
        lambda: SimpleNamespace(
            spool=spool,
            result=tmp_path / "result.json",
            stop=tmp_path / "stop.json",
            cursor=cursor,
            complete_marker=tmp_path / "complete.json",
            contract=tmp_path / "contract.json",
            model_key="test_model",
            seed=501,
        ),
    )
    assert telemetry.main() == 0
