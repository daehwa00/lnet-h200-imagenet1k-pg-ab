from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from scripts import run_lnet_k96_imagenet1k_queue as queue

if TYPE_CHECKING:
    from pathlib import Path


def test_queue_order_and_schema_are_frozen() -> None:
    assert queue.SEEDS == (501, 509, 521)
    assert queue.MODEL_KEY == "lnet_k96_p128x4_d2262_optimized_v2"
    assert queue.SCHEMA == "lnet.imagenet1k.lnet_k96_3seed_queue.v1"


def test_queue_rejects_status_from_another_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / "queue-status.json").write_text(
        json.dumps({"schema": "wrong", "order": [501, 509, 521], "jobs": {}})
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "queue",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(root),
            "--runner",
            str(tmp_path / "runner.py"),
        ],
    )
    with pytest.raises(RuntimeError, match="another contract"):
        queue.main()


def test_queue_launches_two_incomplete_seeds_before_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "output"
    completed = root / queue.MODEL_KEY / "seed_501" / "result.json"
    completed.parent.mkdir(parents=True)
    completed.write_text("{}")
    launched: list[int] = []
    observed_at_wait: list[int] = []

    class Process:
        def __init__(self, command: list[str], **_kwargs: object) -> None:
            self.command = command
            launched.append(int(command[command.index("--seed") + 1]))

        def wait(self) -> int:
            observed_at_wait.append(len(launched))
            seed = int(self.command[self.command.index("--seed") + 1])
            result = root / queue.MODEL_KEY / f"seed_{seed}" / "result.json"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text("{}")
            return 0

    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(root),
            "--runner",
            str(tmp_path / "runner.py"),
            "--max-parallel",
            "2",
        ],
    )
    assert queue.main() == 0
    assert launched == [509, 521]
    assert observed_at_wait == [2, 2]
