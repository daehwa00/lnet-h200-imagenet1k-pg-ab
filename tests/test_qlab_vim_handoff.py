from __future__ import annotations

import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qlab_vim_handoff as handoff

MANAGER_PID = 2619574
WORKER_PID = 2619577


def _args(tmp_path: Path) -> SimpleNamespace:
    old_root = tmp_path / "old"
    runtime_root = tmp_path / "runtime"
    new_root = tmp_path / "new"
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    for path in (old_root, runtime_root / "scripts", data_root, source_root):
        path.mkdir(parents=True)
    (runtime_root / "scripts" / "run_qlab_input_queue.py").write_text("# mocked queue\n")
    (old_root / "queue-state.json").write_text(
        json.dumps(
            {
                "jobs": {
                    handoff.OLD_JOB_ID: {
                        "status": "running",
                        "pid": WORKER_PID,
                        "models": [handoff.MODEL_KEY],
                        "seed": handoff.OLD_SEED,
                    }
                }
            }
        )
    )
    return SimpleNamespace(
        old_root=old_root,
        old_queue_pid=MANAGER_PID,
        old_worker_pid=WORKER_PID,
        runtime_root=runtime_root,
        new_root=new_root,
        data_root=data_root,
        source_root=source_root,
        python_bin=Path("/usr/bin/python3"),
        poll_interval=0.0,
        manager_timeout=2.0,
    )


def _install_proc_mocks(monkeypatch: pytest.MonkeyPatch, manager_gone: dict[str, bool]) -> None:
    command_lines = {
        MANAGER_PID: f"python -u scripts/run_qlab_input_queue.py --root {manager_gone['old_root']}",
        WORKER_PID: (
            "python -u scripts/run_qlab_input_single.py "
            f"--root {manager_gone['old_root']} --model-key {handoff.MODEL_KEY} --seed 501"
        ),
    }

    def read_cmdline(pid: int) -> str | None:
        return command_lines.get(pid)

    def process_gone(pid: int) -> bool:
        return bool(manager_gone.get(f"gone:{pid}", False))

    monkeypatch.setattr(handoff, "_read_proc_cmdline", read_cmdline)
    monkeypatch.setattr(handoff, "_process_gone", process_gone)


def _write_result(root: Path, seed: int) -> None:
    output = root / f"vision-mamba-tiny-s{seed}" / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "completed",
                "phase": "full",
                "model_key": handoff.MODEL_KEY,
                "seed": seed,
                "completed_epochs": 100,
                "requested_epochs": 100,
                "stopped_at_max_steps": False,
            }
        )
    )


def test_handoff_only_signals_manager_and_launches_seed509(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path)
    _write_result(args.old_root, handoff.OLD_SEED)
    flags: dict[str, bool | str] = {
        "gone:2619574": False,
        "gone:2619577": False,
        "old_root": str(args.old_root),
    }
    _install_proc_mocks(monkeypatch, flags)
    gone_calls = 0

    def worker_finishes_after_one_result_poll(pid: int) -> bool:
        nonlocal gone_calls
        if pid == WORKER_PID:
            gone_calls += 1
            return gone_calls > 3
        return bool(flags.get(f"gone:{pid}", False))

    monkeypatch.setattr(handoff, "_process_gone", worker_finishes_after_one_result_poll)
    sent: list[tuple[int, signal.Signals]] = []

    def fake_send(pid: int, sig: signal.Signals) -> None:
        sent.append((pid, sig))
        if pid == MANAGER_PID and sig is signal.SIGCONT:
            flags["gone:2619574"] = True

    monkeypatch.setattr(handoff, "_send", fake_send)
    launched: dict[str, object] = {}

    class Child:
        pid = 9911
        returncode = 0

        def __init__(self, command: list[str], **kwargs: object) -> None:
            launched["command"] = command
            launched["kwargs"] = kwargs

        def wait(self) -> int:
            _write_result(args.new_root, handoff.NEXT_SEED)
            return self.returncode

    monkeypatch.setattr(handoff.subprocess, "Popen", Child)

    assert handoff.run(args) == 0
    assert sent == [
        (MANAGER_PID, signal.SIGSTOP),
        (MANAGER_PID, signal.SIGTERM),
        (MANAGER_PID, signal.SIGCONT),
    ]
    assert all(pid != WORKER_PID for pid, _ in sent)
    assert not (args.old_root / handoff.OLD_JOB_ID / handoff.STOP_NAME).exists()
    assert (args.old_root / handoff.STOP_NAME).is_file()

    manifest = json.loads((args.new_root / "manifest.json").read_text())
    assert len(manifest["jobs"]) == 1
    job = manifest["jobs"][0]
    assert job["seed"] == handoff.NEXT_SEED
    assert job["models"] == [handoff.MODEL_KEY]
    assert job["wandb_id"] == handoff.WANDB_ID
    assert job["wandb_group"] == handoff.WANDB_GROUP
    assert "521" not in json.dumps(manifest)
    command = launched["command"]
    assert isinstance(command, list)
    assert "--resume" not in command
    assert command[command.index("--gpus") + 1] == "1"
    kwargs = launched["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["start_new_session"] is True
    state = json.loads((args.new_root / handoff.STATE_NAME).read_text())
    assert state["status"] == "completed"
    assert handoff.run(args) == 0
    assert sent == [
        (MANAGER_PID, signal.SIGSTOP),
        (MANAGER_PID, signal.SIGTERM),
        (MANAGER_PID, signal.SIGCONT),
    ]


def test_worker_exit_without_result_never_launches_or_signals_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path)
    flags: dict[str, bool | str] = {
        "gone:2619574": False,
        "gone:2619577": False,
        "old_root": str(args.old_root),
    }
    _install_proc_mocks(monkeypatch, flags)
    gone_calls = 0

    def worker_exits_after_validation(pid: int) -> bool:
        nonlocal gone_calls
        if pid == WORKER_PID:
            gone_calls += 1
            return gone_calls > 2
        return bool(flags.get(f"gone:{pid}", False))

    monkeypatch.setattr(handoff, "_process_gone", worker_exits_after_validation)
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(handoff, "_send", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(
        handoff.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("new queue must not launch"),
    )

    with pytest.raises(handoff.HandoffError, match="exited before a complete result"):
        handoff.run(args)
    assert sent == [(MANAGER_PID, signal.SIGSTOP)]
    assert all(pid != WORKER_PID for pid, _ in sent)
    state = json.loads((args.new_root / handoff.STATE_NAME).read_text())
    assert state["status"] == "failed"


def test_invalid_receipt_is_rejected_before_any_freeze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path)
    args.new_root.mkdir(parents=True)
    receipt = handoff._args_receipt(args)
    receipt["old_worker_pid"] = 999999
    (args.new_root / handoff.RECEIPT_NAME).write_text(json.dumps(receipt))
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(handoff, "_send", lambda pid, sig: sent.append((pid, sig)))

    with pytest.raises(handoff.HandoffError, match="receipt mismatch"):
        handoff.run(args)
    assert sent == []
    assert not (args.old_root / handoff.STOP_NAME).exists()
