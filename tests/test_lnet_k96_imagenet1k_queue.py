from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts import run_lnet_k96_imagenet1k_queue as queue

if TYPE_CHECKING:
    from pathlib import Path


def test_queue_order_and_schema_are_frozen() -> None:
    assert queue.SEEDS == (501, 509, 521)
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
