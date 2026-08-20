from __future__ import annotations

# pyright: reportPrivateUsage=false
# ruff: noqa: S603, SLF001, TC002
import json
import subprocess
import sys
from pathlib import Path

import pytest
import run_h200_baseline_queue as queue

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "h200" / "baselines" / "campaign.json"


def _campaign() -> queue.Campaign:
    return queue.load_campaign(MANIFEST)


def _status(campaign: queue.Campaign, root: Path) -> dict[str, object]:
    return queue._new_status(campaign, queue._manifest_sha256(MANIFEST), root)


def _write_valid_result(task: queue.Task, score: float) -> None:
    assert task.learning_rate is not None
    lr_label = f"{task.learning_rate:.8g}".replace(".", "p")
    task_name = f"{task.model_key}__{task.phase}__seed{task.seed}__lr{lr_label}"
    contract = {
        "task_sha256": "a" * 64,
        "source_digest_sha256": "b" * 64,
        "dataset": {"identity_sha256": "c" * 64},
    }
    contract_path = task.output_dir / "contracts" / f"{task_name}.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract))
    task.result_path.parent.mkdir(parents=True, exist_ok=True)
    task.result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "phase": task.phase,
                "model_key": task.model_key,
                "seed": task.seed,
                "learning_rate": task.learning_rate,
                "completed_epochs": task.epochs,
                "requested_epochs": task.epochs,
                "contract_sha256": queue._json_sha256(contract),
                "task_sha256": contract["task_sha256"],
                "source_digest_sha256": contract["source_digest_sha256"],
                "metrics": {"validation_top1": score},
            }
        )
    )


def test_manifest_has_exact_models_recipe_and_seed_major_full_order(tmp_path: Path) -> None:
    campaign = _campaign()

    assert tuple(model.key for model in campaign.models) == queue.MODEL_KEYS
    assert len(campaign.models) == 20
    assert campaign.seeds == (501, 509, 521)
    assert campaign.learning_rates == (3e-4, 1e-3, 3e-3)
    assert campaign.calibration_epochs == 3
    assert campaign.full_epochs == 100
    assert campaign.batch_size == 256
    assert campaign.dataloader_workers == 2
    assert campaign.prefetch_factor == 1
    assert campaign.gpu_memory_fraction == 0.22
    assert campaign.preflight_trials == 3

    calibration = queue.calibration_tasks(campaign, tmp_path)
    assert len(calibration) == 60
    assert all(task.wandb_mode == "disabled" for task in calibration)

    selected = {model.key: 1e-3 for model in campaign.models}
    full = queue.full_tasks(campaign, tmp_path, selected)
    assert len(full) == 60
    assert [(task.seed, task.model_key) for task in full[:21]] == [
        *((501, model.key) for model in campaign.models),
        (509, campaign.models[0].key),
    ]
    assert all(task.epochs == 100 and task.wandb_mode == "online" for task in full)
    for parallelism in (1, 2, 4):
        preflight = queue.preflight_tasks(
            campaign,
            tmp_path,
            parallelism,
            mps_active=True,
        )
        assert len(preflight) == parallelism
        assert {task.model_key for task in preflight} == {"convnextv2_atto"}
        assert all("mps_pct" in task.task_id for task in preflight)


def test_worker_command_and_environment_enforce_resource_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    task = queue.calibration_tasks(campaign, tmp_path)[0]
    task.checkpoint_path.parent.mkdir(parents=True)
    task.checkpoint_path.write_bytes(b"checkpoint")
    command = queue.worker_command(
        task,
        python=Path("/python"),
        worker=Path("/worker.py"),
        data_root=Path("/imagenet"),
        source_root=Path("/source"),
        batch_size=campaign.batch_size,
        dataloader_workers=campaign.dataloader_workers,
    )

    assert command[:3] == ["/python", "-u", "/worker.py"]
    assert command[command.index("--source-root") + 1] == "/source"
    assert command[command.index("--batch-size") + 1] == "256"
    assert command[command.index("--workers") + 1] == "2"
    assert command[-1] == "--resume"
    assert command[command.index("--wandb-mode") + 1] == "disabled"

    monkeypatch.setenv("WANDB_API_KEY", "must-not-reach-calibration")
    environment = queue._child_environment(
        task,
        mps_active=True,
        mps_percentage=50,
        gpu_memory_fraction=0.22,
    )
    assert "WANDB_API_KEY" not in environment
    assert environment["WANDB_MODE"] == "disabled"
    assert environment["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "50"
    assert environment["H200_GPU_MEMORY_FRACTION"] == "0.22"

    full = queue.full_tasks(campaign, tmp_path, {campaign.models[0].key: 1e-3})[0]
    runtime_path = ROOT / "h200" / "baselines" / "wandb.runtime.json"
    monkeypatch.setenv("H200_BASELINE_WANDB_RUNTIME", str(runtime_path))
    full_environment = queue._child_environment(
        full,
        mps_active=True,
        mps_percentage=50,
        gpu_memory_fraction=0.22,
    )
    assert full_environment["WANDB_API_KEY"] == "must-not-reach-calibration"
    assert full_environment["WANDB_MODE"] == "online"
    runtime = json.loads(runtime_path.read_text())
    expected_run = runtime["runs"][full.model_key]["seeds"][str(full.seed)]
    assert full_environment["H200_BASELINE_RUN_ID"] == expected_run["id"]
    assert full_environment["H200_BASELINE_DISPLAY_NAME"] == expected_run["display_name"]


def test_result_and_checkpoint_reconciliation_selects_best_lr(tmp_path: Path) -> None:
    campaign = _campaign()
    tasks = queue.calibration_tasks(campaign, tmp_path)[:3]
    status = _status(campaign, tmp_path)
    scores = (70.0, 72.0, 72.0)
    for task, score in zip(tasks, scores, strict=True):
        _write_valid_result(task, score)
    resumable = queue.calibration_tasks(campaign, tmp_path)[3]
    resumable.checkpoint_path.parent.mkdir(parents=True)
    resumable.checkpoint_path.write_bytes(b"checkpoint")

    queue.reconcile_tasks([*tasks, resumable], status)
    jobs = queue._jobs(status)
    assert all(jobs[task.task_id]["status"] == "COMPLETED" for task in tasks)
    assert jobs[resumable.task_id]["status"] == "RESUMABLE"
    selected, evidence = queue.select_learning_rates(campaign, tasks)
    assert selected[campaign.models[0].key] == 1e-3
    assert evidence[campaign.models[0].key]["status"] == "SELECTED"


def test_partial_calibration_grid_blocks_full_training(tmp_path: Path) -> None:
    campaign = _campaign()
    tasks = queue.calibration_tasks(campaign, tmp_path)[:3]
    _write_valid_result(tasks[0], 0.5)
    _write_valid_result(tasks[1], 0.6)

    selected, evidence = queue.select_learning_rates(campaign, tasks)

    assert campaign.models[0].key not in selected
    assert evidence[campaign.models[0].key]["status"] == "INCOMPLETE_GRID"


def test_new_orchestrator_resets_failed_attempt_budget(tmp_path: Path) -> None:
    campaign = _campaign()
    manifest_sha = queue._manifest_sha256(MANIFEST)
    status = queue._new_status(campaign, manifest_sha, tmp_path)
    first_task = queue.calibration_tasks(campaign, tmp_path)[0]
    job = queue._jobs(status)[first_task.task_id]
    job["status"] = "FAILED"
    job["attempts"] = 2
    queue._atomic_json(tmp_path / "queue-status.json", status)

    reloaded = queue._load_or_create_status(campaign, manifest_sha, tmp_path)

    assert queue._jobs(reloaded)[first_task.task_id]["attempts"] == 0


def test_completed_full_result_replays_until_wandb_marker_exists(tmp_path: Path) -> None:
    campaign = _campaign()
    task = queue.full_tasks(
        campaign,
        tmp_path,
        {campaign.models[0].key: 1.0e-3},
    )[0]
    _write_valid_result(task, 0.5)
    assert queue._needs_telemetry_replay(task) is True
    lr_label = f"{task.learning_rate:.8g}".replace(".", "p")
    task_name = f"{task.model_key}__{task.phase}__seed{task.seed}__lr{lr_label}"
    marker = task.output_dir / "telemetry" / f"{task_name}.mirror-complete.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    assert queue._needs_telemetry_replay(task) is False


def test_result_is_rejected_when_active_dataset_identity_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    task = queue.calibration_tasks(campaign, tmp_path)[0]
    _write_valid_result(task, 0.5)
    assert queue._result_payload(task) is not None
    monkeypatch.setenv("LNET_DATASET_IDENTITY_SHA256", "different-dataset")
    assert queue._result_payload(task) is None


def test_dynamic_pool_isolates_worker_failure_and_persists_only_return_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    tasks = [
        queue.calibration_tasks(campaign, tmp_path)[0],
        queue.calibration_tasks(campaign, tmp_path)[3],
    ]
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--phase')
parser.add_argument('--model-key')
parser.add_argument('--seed', type=int)
parser.add_argument('--learning-rate', type=float)
parser.add_argument('--result-path', type=Path)
args, _ = parser.parse_known_args()
if args.model_key == 'parc_net_xs':
    raise SystemExit(7)
args.result_path.write_text(json.dumps({
    'status': 'completed',
    'phase': args.phase,
    'model_key': args.model_key,
    'seed': args.seed,
    'learning_rate': args.learning_rate,
    'metrics': {'validation_top1': 1.0},
}))
"""
    )
    status = _status(campaign, tmp_path)
    monkeypatch.setenv("QUEUE_TEST_SECRET", "do-not-persist")

    def accept_fake_result(task: queue.Task) -> dict[str, object] | None:
        if not task.result_path.is_file():
            return None
        return json.loads(task.result_path.read_text())

    monkeypatch.setattr(queue, "_result_payload", accept_fake_result)

    queue.run_task_pool(
        tasks,
        status=status,
        root=tmp_path,
        repo=tmp_path,
        python=Path(sys.executable),
        worker=fake_worker,
        data_root=tmp_path,
        max_parallel=2,
        max_attempts=1,
        mps_active=True,
        mps_percentage=50,
        batch_size=256,
        dataloader_workers=2,
        gpu_memory_fraction=0.22,
        poll_seconds=0.01,
    )

    jobs = queue._jobs(status)
    assert jobs[tasks[0].task_id]["status"] == "FAILED"
    assert jobs[tasks[0].task_id]["last_exit_code"] == 7
    assert jobs[tasks[1].task_id]["status"] == "COMPLETED"
    serialized = (tmp_path / "queue-status.json").read_text()
    assert "do-not-persist" not in serialized
    assert "QUEUE_TEST_SECRET" not in serialized


def test_mps_auto_has_safe_single_process_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(queue.shutil, "which", lambda _name: None)
    session = queue.start_mps(tmp_path, "auto")

    assert session.active is False
    assert session.started_by_queue is False
    assert session.reason == "control_binary_unavailable"


def test_active_worker_process_group_is_terminated_before_mps_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _campaign()
    queue_task = queue.calibration_tasks(task, tmp_path)[0]
    signals: list[tuple[int, int]] = []

    class Process:
        def __init__(self) -> None:
            self.alive = True

        def poll(self) -> int | None:
            return None if self.alive else -15

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.alive = False
            return -15

    process = Process()
    log = (tmp_path / "worker.log").open("w")
    record = queue.RunningProcess(
        queue_task,
        process,  # pyright: ignore[reportArgumentType]
        log,
    )
    queue._ACTIVE_PROCESSES[4321] = record
    monkeypatch.setattr(
        queue.os,
        "killpg",
        lambda pid, signal_number: signals.append((pid, signal_number)),
    )

    queue._terminate_active_processes()

    assert signals == [(4321, queue.signal.SIGTERM)]
    assert log.closed is True
    assert not queue._ACTIVE_PROCESSES


def test_list_mode_emits_all_planned_tasks_without_runtime_dependencies(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_h200_baseline_queue.py"),
            "--mode",
            "list",
            "--root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    tasks = json.loads(result.stdout)
    assert len(tasks) == 120
    assert [task["seed"] for task in tasks[60:80]] == [501] * 20
    assert [task["seed"] for task in tasks[80:100]] == [509] * 20
    assert [task["seed"] for task in tasks[100:120]] == [521] * 20
