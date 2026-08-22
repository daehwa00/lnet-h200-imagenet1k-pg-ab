from __future__ import annotations

# pyright: reportExplicitAny=false, reportMissingImports=false, reportUnknownLambdaType=false
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import nn


class _Loader:
    dataset = tuple(range(8))

    def __len__(self) -> int:
        return 2


class _FailingRun:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.summary: dict[str, float] = {}

    def log(self, _metrics: dict[str, float], *, step: int) -> None:
        self.events.append(f"wandb-log-{step}")
        message = "simulated asynchronous relay failure"
        raise RuntimeError(message)

    def finish(self) -> None:
        self.events.append("wandb-finish")
        message = "simulated finish failure"
        raise RuntimeError(message)


class _SuccessfulRun:
    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, float]]] = []
        self.summary: dict[str, float] = {}
        self.finished = False

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.logged.append((step, metrics))

    def finish(self) -> None:
        self.finished = True


def _contract() -> dict[str, Any]:
    return {
        "model": {},
        "recipe": {
            "batch_size": 4,
            "channels_last": False,
            "epochs": 1,
            "gradient_accumulation_steps": 1,
            "mixup_alpha": 0.8,
            "precision": "float32",
        },
        "schema": "durability-test",
        "variant_configs": {"variant": {}},
    }


def _bindings() -> harness.RunnerBindings:
    return harness.RunnerBindings(
        variants=("variant",),
        seeds=(501,),
        model_config=lambda **_kwargs: object(),
        build_model=lambda _variant, _config: nn.Linear(1, 1),
        contract=lambda _args: _contract(),
        build_optimizer=lambda model, _recipe: torch.optim.SGD(model.parameters(), lr=0.1),
        prepare_model=lambda model, _recipe: model,
        train_epoch=lambda *_args, **_kwargs: {"loss": 1.0, "mixed_accuracy": 0.5},
        evaluate=lambda *_args, **_kwargs: {"accuracy": 0.75, "cross_entropy": 0.5},
        wandb_model_metrics=lambda _model: {"model/diagnostic": 1.0},
        summarize=lambda _root, _contract_payload: None,
    )


def test_checkpoint_and_spool_precede_failing_wandb_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    run = _FailingRun(events)
    real_atomic_torch = harness._atomic_torch

    def record_checkpoint(path: Path, payload: object) -> None:
        events.append("checkpoint")
        real_atomic_torch(path, payload)

    def train_with_optimizer_step(
        _train_epoch: object,
        _model: object,
        _runtime: object,
        _loader: object,
        optimizer: torch.optim.Optimizer,
        **_kwargs: object,
    ) -> tuple[dict[str, float], int]:
        optimizer.step()
        return {"loss": 1.0, "mixed_accuracy": 0.5}, 1

    loader = _Loader()
    monkeypatch.setattr(harness, "_loaders", lambda *_args, **_kwargs: (loader, loader))
    monkeypatch.setattr(harness, "_build_runtime", lambda model, _recipe: model)
    monkeypatch.setattr(
        harness,
        "_train_epoch_with_step_count",
        train_with_optimizer_step,
    )
    monkeypatch.setattr(harness, "_initialize_wandb_run", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(harness, "_atomic_torch", record_checkpoint)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)

    harness._run_job(
        tmp_path,
        _contract(),
        variant="variant",
        seed=501,
        data_root=tmp_path / "data",
        workers=0,
        device=torch.device("cpu"),
        bindings=_bindings(),
    )

    result_path = tmp_path / "results" / "variant__seed501.json"
    checkpoint_path = tmp_path / "checkpoints" / "variant__seed501.pt"
    spool_path = tmp_path / "telemetry" / "variant__seed501.jsonl"
    assert result_path.exists()
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["contract_sha256"] == harness._contract_sha256(_contract())
    assert [record["kind"] for record in harness._read_telemetry_spool(spool_path)] == [
        "epoch",
        "final",
    ]
    assert events.index("checkpoint") < events.index("wandb-log-1")
    output = capsys.readouterr().out.splitlines()
    progress_line = next(line for line in output if line.startswith("H200_PROGRESS_JSON="))
    progress = json.loads(progress_line.removeprefix("H200_PROGRESS_JSON="))
    assert progress["epoch"] == 1
    assert progress["global_step"] == 1
    assert any(line.startswith("H200_WANDB_DEGRADED_JSON=") for line in output)


def test_missing_contract_cannot_adopt_existing_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "orphan.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"orphan")

    with pytest.raises(RuntimeError, match="artifacts exist without"):
        harness._initialize(tmp_path, _contract())


def test_completed_result_replays_durable_telemetry_without_retraining(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "results" / "variant__seed501.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "variant": "variant",
                "seed": 501,
                "contract_sha256": harness._contract_sha256(_contract()),
                "parameters": 2,
                "global_step": 1,
                "final_validation": {"accuracy": 0.75, "cross_entropy": 0.5},
                "training_seconds": 1.0,
                "history": [
                    {
                        "epoch": 1,
                        "learning_rate": 0.1,
                        "train_loss": 1.0,
                        "train_mixed_accuracy": 0.5,
                        "validation_accuracy": 0.75,
                        "validation_cross_entropy": 0.5,
                        "training_seconds": 1.0,
                        "global_step": 1,
                        "optimizer_steps": 1,
                    }
                ],
            }
        )
    )
    run = _SuccessfulRun()
    monkeypatch.setattr(harness, "_initialize_wandb_run", lambda *_args, **_kwargs: run)
    bindings = replace(
        _bindings(),
        build_model=lambda *_args: pytest.fail("completed result retrained"),
    )

    harness._run_job(
        tmp_path,
        _contract(),
        variant="variant",
        seed=501,
        data_root=tmp_path / "data",
        workers=0,
        device=torch.device("cpu"),
        bindings=bindings,
    )

    assert run.logged[0][0] == 1
    assert run.summary["final_validation_accuracy"] == 0.75
    assert run.finished is True


def test_completed_result_preserves_richer_existing_epoch_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = {
        "variant": "variant",
        "seed": 501,
        "contract_sha256": harness._contract_sha256(_contract()),
        "parameters": 2,
        "global_step": 1,
        "final_validation": {"accuracy": 0.75, "cross_entropy": 0.5},
        "training_seconds": 1.0,
        "history": [
            {
                "epoch": 1,
                "learning_rate": 0.1,
                "train_loss": 1.0,
                "train_mixed_accuracy": 0.5,
                "validation_accuracy": 0.75,
                "validation_cross_entropy": 0.5,
                "training_seconds": 1.0,
                "global_step": 1,
                "optimizer_steps": 1,
            }
        ],
    }
    result_path = tmp_path / "results" / "variant__seed501.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(result))
    spool_path = harness._telemetry_spool_path(
        tmp_path,
        variant="variant",
        seed=501,
    )
    richer = {
        "kind": "epoch",
        "metrics": {
            **harness._epoch_telemetry_metrics(result["history"][0]),
            "model/diagnostic": 1.0,
        },
        "step": 1,
    }
    harness._append_telemetry_record(spool_path, richer)
    run = _SuccessfulRun()
    monkeypatch.setattr(harness, "_initialize_wandb_run", lambda *_args, **_kwargs: run)

    harness._run_job(
        tmp_path,
        _contract(),
        variant="variant",
        seed=501,
        data_root=tmp_path / "data",
        workers=0,
        device=torch.device("cpu"),
        bindings=_bindings(),
    )

    assert harness._read_telemetry_spool(spool_path)[0] == richer
    assert run.logged[0][1]["model/diagnostic"] == 1.0


def test_torn_telemetry_tail_is_repaired_without_losing_valid_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text('{"kind":"epoch","metrics":{"epoch":1},"step":1}\n{"kind":')
    second = {"kind": "epoch", "metrics": {"epoch": 2}, "step": 2}

    assert harness._append_telemetry_record(path, second) is True

    records = harness._read_telemetry_spool(path)
    assert [record["step"] for record in records] == [1, 2]


def test_telemetry_replace_failure_is_degraded_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"

    def fail_replace(_self: Path, _target: Path) -> None:
        message = "simulated telemetry storage failure"
        raise OSError(message)

    monkeypatch.setattr(Path, "replace", fail_replace)
    assert (
        harness._append_telemetry_record(
            path,
            {"kind": "epoch", "metrics": {"epoch": 1}, "step": 1},
        )
        is False
    )


def test_persistent_workers_require_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LNET_PERSISTENT_WORKERS", raising=False)
    assert harness._persistent_loader_workers(8) is False
    monkeypatch.setenv("LNET_PERSISTENT_WORKERS", "1")
    assert harness._persistent_loader_workers(8) is True
    assert harness._persistent_loader_workers(0) is False


def test_wandb_retry_schedule_is_bounded() -> None:
    assert [epoch for epoch in range(1, 22) if harness._wandb_retry_due(epoch)] == [
        1,
        10,
        20,
    ]


def test_checkpoint_history_backfills_crash_window_without_overwriting_spool(
    tmp_path: Path,
) -> None:
    spool = tmp_path / "telemetry.jsonl"
    history = [
        {
            "epoch": epoch,
            "learning_rate": 0.1,
            "train_loss": 1.0,
            "train_mixed_accuracy": 0.5,
            "validation_accuracy": 0.75,
            "validation_cross_entropy": 0.5,
        }
        for epoch in (1, 2)
    ]
    first = {
        "kind": "epoch",
        "metrics": {
            **harness._epoch_telemetry_metrics(history[0]),
            "model/diagnostic": 123.0,
        },
        "step": 1,
    }
    harness._append_telemetry_record(spool, first)

    harness._backfill_checkpoint_telemetry(spool, history)

    records = harness._read_telemetry_spool(spool)
    assert [record["step"] for record in records] == [1, 2]
    assert records[0]["metrics"]["model/diagnostic"] == 123.0
