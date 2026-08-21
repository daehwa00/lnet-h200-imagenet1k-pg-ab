from __future__ import annotations

# ruff: noqa: ARG002, SLF001, TC002, TC003
import argparse
import json
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts import remote_run_control
from scripts import run_h200_owner_controlled as supervisor

TARGET = "a" * 40


def _record(action: str = "run", generation: int = 1) -> dict[str, Any]:
    return {
        "schema": remote_run_control.SCHEMA,
        "campaign_id": "campaign",
        "target_commit": TARGET,
        "generation": generation,
        "action": action,
        "updated_at": "2026-08-21T12:00:00+09:00",
        "reason": "owner request",
    }


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=tmp_path,
        repo_url="https://github.com/owner/repo.git",
        ref="refs/heads/control/campaign",
        control_path="control.json",
        campaign_id="campaign",
        target_commit=TARGET,
        stop_marker=tmp_path / "state" / "stopped.json",
        fast_stop_marker=tmp_path / "fast" / "stopped.json",
        poll_seconds=1.0,
        grace_seconds=0.0,
        term_seconds=0.0,
        command=["command"],
    )


class _Control:
    def __init__(self, *, stop: bool) -> None:
        self.stop = stop

    def require_start(self) -> dict[str, Any]:
        return _record()

    def poll(self, *, force: bool = False) -> dict[str, Any]:
        assert force
        if self.stop:
            raise supervisor.StopRequestedError(_record("stop", 2))
        return _record()


class _Process:
    pid = 123

    def __init__(self, return_code: int | None) -> None:
        self.return_code = return_code
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.return_code is None:
            self.return_code = 0
        return self.return_code


def test_owner_stop_exits_zero_and_persists_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    process = _Process(None)
    monkeypatch.setattr(supervisor, "_parse_args", lambda: args)
    monkeypatch.setattr(supervisor, "_control", lambda *_args: _Control(stop=True))
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)

    assert supervisor.main() == 0
    payload = json.loads(args.stop_marker.read_text(encoding="utf-8"))
    assert payload["generation"] == 2
    assert payload["checkpoint_policy"] == "last completed cohort epoch remains authoritative"
    assert process.wait_calls == 1
    assert json.loads(args.fast_stop_marker.read_text(encoding="utf-8")) == payload


def test_ordinary_child_failure_is_not_reported_as_an_owner_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(supervisor, "_parse_args", lambda: args)
    monkeypatch.setattr(supervisor, "_control", lambda *_args: _Control(stop=False))
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: _Process(7))

    assert supervisor.main() == 7
    assert not args.stop_marker.exists()


def test_hung_process_group_escalates_term_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process(None)
    signals: list[signal.Signals] = []

    def wait(_process: object, _seconds: float) -> bool:
        return False

    def killpg(_pid: int, sent: signal.Signals) -> None:
        signals.append(sent)
        if sent == signal.SIGKILL:
            process.return_code = -int(signal.SIGKILL)

    monkeypatch.setattr(supervisor, "_wait", wait)
    monkeypatch.setattr(supervisor.os, "killpg", killpg)

    assert supervisor._terminate_process_group(process, term_seconds=0)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
