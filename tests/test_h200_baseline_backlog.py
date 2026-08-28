# pyright: reportPrivateUsage=false
from pathlib import Path
from typing import cast

from scripts import run_h200_baseline_backlog as backlog
from scripts import run_h200_baseline_queue as queue

ROOT = Path(__file__).resolve().parents[1]


def test_backlog_contains_only_four_unstarted_seed501_models(tmp_path: Path) -> None:
    campaign = queue.load_campaign(ROOT / "h200/baselines/campaign.json")
    selected: dict[str, float] = dict.fromkeys(backlog.MODEL_KEYS, backlog.LEARNING_RATE)
    tasks = [
        task
        for task in queue.full_tasks(campaign, tmp_path, selected)
        if task.seed == backlog.SEED and task.model_key in backlog.MODEL_KEYS
    ]
    assert tuple(task.model_key for task in tasks) == backlog.MODEL_KEYS
    assert {task.seed for task in tasks} == {501}
    assert {task.learning_rate for task in tasks} == {3.0e-3}
    assert backlog.GPU_MEMORY_FRACTION == 0.5


def test_h200_entrypoint_routes_backlog_without_global_summary() -> None:
    script = (ROOT / "h200/run_baselines.sh").read_text(encoding="utf-8")
    assert 'H200_BASELINE_BACKLOG_ONLY:-0}" == "1"' in script
    assert 'QUEUE_SCRIPT="scripts/run_h200_baseline_backlog.py"' in script
    assert '"${QUEUE[@]}" || QUEUE_EXIT_CODE=$?' in script
    assert "H200_BASELINE_UNICONV_DISABLED=not_requested_by_backlog" in script


def test_preflight_jobs_are_added_to_the_authoritative_status_mapping(tmp_path: Path) -> None:
    campaign = queue.load_campaign(ROOT / "h200/baselines/campaign.json")
    status = queue._new_status(campaign, "0" * 64, tmp_path)
    task = queue.preflight_tasks(campaign, tmp_path, 1)[0]
    jobs = queue._jobs(status)
    jobs[task.task_id] = queue._new_job(task)
    authoritative = cast("dict[str, object]", status["jobs"])
    assert task.task_id in authoritative
