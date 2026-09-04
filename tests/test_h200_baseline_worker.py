from __future__ import annotations

import fcntl

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# ruff: noqa: ANN401, SLF001, TC003
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h200_baseline_registry as registry
import pytest
import run_h200_baseline_worker as worker
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

EXPECTED_KEYS = (
    "parc_net_xs",
    "parc_net_s",
    "mobilevitv2_050",
    "mobilevitv2_075",
    "mobilevitv2_100",
    "sret_tiny",
    "moganet_xt",
    "uniconvnet_a",
    "convnextv2_atto",
    "efficientmod_xxs",
    "emov2_1m",
    "emov2_2m",
    "mobileone_s0",
    "mobileone_s1",
    "efficientformerv2_s0",
    "swiftformer_xs",
    "fastvit_t8",
    "tinynext_t",
    "tinynext_s",
    "tinynext_m",
    "tinyvim_s",
    "efficientvim_m1",
    "mambaout_femto",
)


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(12, num_classes))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class _RandomMixup:
    def __call__(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        permutation = torch.randperm(inputs.shape[0])
        amount = torch.rand(())
        labels = torch.nn.functional.one_hot(targets, worker.NUM_CLASSES).float()
        mixed_inputs = amount * inputs + (1.0 - amount) * inputs[permutation]
        mixed_labels = amount * labels + (1.0 - amount) * labels[permutation]
        return mixed_inputs, mixed_labels


def test_compiled_runtime_is_explicit_and_fixed_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyClassifier(worker.NUM_CLASSES)
    captured: dict[str, object] = {}

    def compile_model(source: nn.Module, **kwargs: object) -> nn.Module:
        captured["source"] = source
        captured.update(kwargs)
        return source

    monkeypatch.setenv("H200_BASELINE_TORCH_COMPILE_MODE", "default")
    monkeypatch.setattr(worker.torch, "compile", compile_model)
    assert worker._compiled_runtime(model, torch.device("cuda")) is model
    assert captured == {
        "source": model,
        "mode": "default",
        "fullgraph": False,
        "dynamic": False,
    }


def _model_builder(_key: str, _source_root: str | Path | None, classes: int) -> nn.Module:
    return _TinyClassifier(classes)


def _loader_builder(task: worker.BaselineTask, _device: torch.device) -> worker.LoaderBundle:
    generator = torch.Generator().manual_seed(9)
    train_inputs = torch.randn(8, 3, 2, 2, generator=generator)
    train_targets = torch.arange(8) % 3
    validation_inputs = torch.randn(4, 3, 2, 2, generator=generator)
    validation_targets = torch.arange(4) % 3
    train_generator = torch.Generator().manual_seed(task.seed)
    validation_generator = torch.Generator().manual_seed(task.seed + 1)
    train = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=task.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    validation = DataLoader(
        TensorDataset(validation_inputs, validation_targets),
        batch_size=task.batch_size,
        shuffle=False,
        generator=validation_generator,
    )
    return worker.LoaderBundle(train, validation, train_generator, validation_generator)


def _task(root: Path, *, resume: bool = False) -> worker.BaselineTask:
    return worker.BaselineTask(
        phase="calibration",
        model_key="mobilevitv2_050",
        seed=501,
        learning_rate=1.0e-3,
        epochs=2,
        data_root=root / "data",
        output_dir=root,
        result_path=root / "results" / "task.json",
        checkpoint_path=root / "checkpoints" / "task.pt",
        batch_size=2,
        workers=0,
        resume=resume,
    )


def _scientific_history(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = json.loads(json.dumps(result["history"]))
    for row in rows:
        row.pop("training_seconds")
        row["train"].pop("images_per_second")
    return rows


def test_registry_declares_all_twenty_models_and_exact_timm_keys() -> None:
    assert registry.PUBLIC_MODEL_KEYS == EXPECTED_KEYS
    assert set(registry.MODEL_SPECS) == set(EXPECTED_KEYS)
    assert {
        spec.implementation_key for spec in registry.MODEL_SPECS.values() if spec.backend == "timm"
    } == {
        "mobilevitv2_050",
        "mobilevitv2_075",
        "mobilevitv2_100",
        "convnextv2_atto",
        "mobileone_s0",
        "mobileone_s1",
        "efficientformerv2_s0",
        "swiftformer_xs",
        "fastvit_t8",
    }
    assert registry.MODEL_SPECS["fastvit_t8"].create_kwargs == {"inference_mode": False}
    assert registry.MODEL_SPECS["efficientformerv2_s0"].single_head is True
    assert registry.MODEL_SPECS["swiftformer_xs"].single_head is True
    assert registry.MODEL_SPECS["uniconvnet_a"].precision == "float32"
    assert all(
        spec.precision == "bfloat16"
        for key, spec in registry.MODEL_SPECS.items()
        if key != "uniconvnet_a"
    )


def test_output_adapter_handles_tensor_tuple_and_emo_dict() -> None:
    logits = torch.randn(2, 3)
    assert worker._extract_logits(logits) is logits
    with pytest.raises(TypeError, match="ambiguous auxiliary"):
        worker._extract_logits((logits, torch.zeros_like(logits)))
    assert worker._extract_logits({"out": logits, "out_kd": torch.zeros_like(logits)}) is logits
    with pytest.raises(TypeError, match="ambiguous dict"):
        worker._extract_logits({"a": logits, "b": logits})
    with pytest.raises(RuntimeError, match="classifier logits"):
        worker._classification_logits(torch.randn(2, 999), 2)


def test_cli_accepts_inline_task_json_and_queue_contract(tmp_path: Path) -> None:
    payload = {
        "phase": "preflight",
        "model_key": "fastvit_t8",
        "seed": 509,
        "learning_rate": 3.0e-4,
        "epochs": 1,
        "data_root": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "run"),
        "result_path": str(tmp_path / "result.json"),
        "checkpoint_path": str(tmp_path / "checkpoint.pt"),
        "batch_size": 128,
        "workers": 0,
    }
    task = worker._parse_task(["--task-json", json.dumps(payload)])
    assert task.model_key == "fastvit_t8"
    assert task.max_steps == 100
    assert task.gradient_accumulation_steps == 2
    assert task.result_path == tmp_path / "result.json"


@pytest.mark.parametrize(
    ("device", "current_device", "expected_index"),
    [
        (torch.device("cuda"), 3, 3),
        (torch.device("cuda:5"), 3, 5),
    ],
)
def test_cuda_memory_limit_resolves_an_explicit_device_index(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    current_device: int,
    expected_index: int,
) -> None:
    configured: list[tuple[float, int]] = []
    reset: list[int] = []

    def configure_fraction(fraction: float, index: int) -> None:
        configured.append((fraction, index))

    def reset_peak_memory(index: int) -> None:
        reset.append(index)

    monkeypatch.setenv("H200_GPU_MEMORY_FRACTION", "0.5")
    monkeypatch.setattr(worker.torch.cuda, "current_device", lambda: current_device)
    monkeypatch.setattr(
        worker.torch.cuda,
        "set_per_process_memory_fraction",
        configure_fraction,
    )
    monkeypatch.setattr(
        worker.torch.cuda,
        "reset_peak_memory_stats",
        reset_peak_memory,
    )

    assert worker._configure_cuda_memory_limit(device) == 0.5  # pyright: ignore[reportPrivateUsage]
    assert configured == [(0.5, expected_index)]
    assert reset == [expected_index]


def test_epoch_checkpoint_resume_preserves_rng_and_scientific_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(worker, "_make_mixup", _RandomMixup)
    uninterrupted_task = _task(tmp_path / "uninterrupted")
    uninterrupted = worker.run_task(
        uninterrupted_task,
        device=torch.device("cpu"),
        model_builder=_model_builder,
        loader_builder=_loader_builder,
    )

    interrupted_task = _task(tmp_path / "resumed")
    real_emit = worker._emit_progress
    interrupted = False

    def interrupt_after_first_checkpoint(
        task: worker.BaselineTask,
        payload: dict[str, Any],
    ) -> None:
        nonlocal interrupted
        real_emit(task, payload)
        if payload["status"] == "running" and payload["epoch"] == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(worker, "_emit_progress", interrupt_after_first_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        worker.run_task(
            interrupted_task,
            device=torch.device("cpu"),
            model_builder=_model_builder,
            loader_builder=_loader_builder,
        )
    assert interrupted_task.checkpoint_path.exists()

    monkeypatch.setattr(worker, "_emit_progress", real_emit)
    resumed = worker.run_task(
        replace(interrupted_task, resume=True),
        device=torch.device("cpu"),
        model_builder=_model_builder,
        loader_builder=_loader_builder,
    )
    assert _scientific_history(resumed) == _scientific_history(uninterrupted)
    assert resumed["metrics"]["validation_top1"] == resumed["final_validation"]["accuracy"]
    assert resumed["metrics"]["images_per_second"] > 0

    uninterrupted_checkpoint = torch.load(
        uninterrupted_task.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    resumed_checkpoint = torch.load(
        interrupted_task.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    for name, parameter in uninterrupted_checkpoint["model"].items():
        assert torch.equal(parameter, resumed_checkpoint["model"][name])


def test_checkpoint_is_durable_before_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(worker, "_make_mixup", _RandomMixup)
    task = replace(_task(tmp_path), epochs=1)
    observed: list[bool] = []

    def observe_checkpoint(_task: worker.BaselineTask, payload: dict[str, Any]) -> None:
        if payload["status"] == "running":
            observed.append(task.checkpoint_path.exists())

    monkeypatch.setattr(worker, "_emit_progress", observe_checkpoint)
    result = worker.run_task(
        task,
        device=torch.device("cpu"),
        model_builder=_model_builder,
        loader_builder=_loader_builder,
    )
    assert observed == [True]
    assert json.loads(task.result_path.read_text())["status"] == "completed"
    assert result["contract_sha256"]
    assert result["task_sha256"]
    assert result["source_digest_sha256"]


def test_live_wandb_mirror_uses_campaign_identity_and_console_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class Process:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.waited = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waited = True
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    process = Process()

    def launch(command: list[str], **kwargs: Any) -> Process:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    monkeypatch.setenv("H200_BASELINE_RUN_ID", "1" * 16)
    monkeypatch.setenv("H200_BASELINE_DISPLAY_NAME", "H200-BL-test-s501")
    monkeypatch.setenv(
        "H200_BASELINE_TAGS_JSON",
        json.dumps(["H200", "ImageNet-1K", "matched-baseline"]),
    )
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_GROUP", "test-group")
    task = replace(_task(tmp_path), phase="full", wandb_mode="online")
    contract_path = task.output_dir / "contracts" / f"{task.task_name}.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps({"schema": "test"}))
    worker._append_telemetry(
        task,
        {"id": "epoch:1", "kind": "epoch", "step": 1, "metrics": {"epoch": 1.0}},
    )
    mirror = worker._WandbMirror(task, {"schema": "test"})
    mirror.sync()
    mirror.finish(
        {"final_validation": {"accuracy": 0.5, "top5_accuracy": 0.8}}
    )

    command = captured["command"]
    assert command[0] == worker.sys.executable
    assert "run_h200_baseline_telemetry.py" in command[2]
    assert command[command.index("--cursor") + 1].endswith(".cursor.json")
    assert process.waited is True
    assert mirror.stop_path.is_file()


def test_telemetry_spool_is_idempotent_and_repairs_torn_tail(tmp_path: Path) -> None:
    task = _task(tmp_path)
    first = {"id": "epoch:1", "kind": "epoch", "step": 1, "metrics": {"epoch": 1}}
    second = {"id": "epoch:2", "kind": "epoch", "step": 2, "metrics": {"epoch": 2}}
    assert worker._append_telemetry(task, first) is True
    assert worker._append_telemetry(task, first) is True
    path = worker._telemetry_spool_path(task)
    with path.open("ab") as stream:
        stream.write(b'{"id":')
    assert worker._append_telemetry(task, second) is True
    assert [record["id"] for record in worker._read_telemetry(task)] == [
        "epoch:1",
        "epoch:2",
    ]


def test_worker_main_refuses_a_duplicate_task_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = task.output_dir / ".worker.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(worker, "_parse_task", lambda _argv=None: task)
        monkeypatch.setattr(
            worker,
            "run_task",
            lambda _task: pytest.fail("duplicate worker entered run_task"),
        )
        with pytest.raises(RuntimeError, match="another worker owns"):
            worker.main([])
