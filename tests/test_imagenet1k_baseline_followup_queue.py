from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
from pathlib import Path

from scripts import h200_baseline_registry as registry
from scripts import run_h200_baseline_queue as queue
from scripts import run_h200_baseline_worker as worker
from scripts import run_imagenet1k_baseline_followup_queue as followup

ROOT = Path(__file__).parents[1]


def test_followup_partition_is_exact_disjoint_and_balanced() -> None:
    tasks = [task for lane in followup.FOLLOWUP_TASKS.values() for task in lane]
    assert len(tasks) == 13
    assert len(set(tasks)) == 13
    expected = {
        (model, seed)
        for model in (
            "parc_net_xs",
            "parc_net_s",
            "convnextv2_atto",
            "tinynext_t",
            "emov2_1m",
        )
        for seed in (509, 521)
    }
    expected |= {("moganet_xt", seed) for seed in (501, 509, 521)}
    assert set(tasks) == expected
    assert followup.FOLLOWUP_TASKS["qlab0"][-1] == ("moganet_xt", 501)
    assert followup.FOLLOWUP_TASKS["h200"][-2:] == (
        ("moganet_xt", 509),
        ("moganet_xt", 521),
    )


def test_moganet_keeps_bf16_training_but_uses_fp32_validation() -> None:
    assert registry.model_spec("moganet_xt").precision == "bfloat16"
    assert worker._validation_uses_bfloat16("moganet_xt") is False
    assert worker._validation_uses_bfloat16("convnextv2_atto") is True


def test_h200_followup_is_selected_before_generic_queue() -> None:
    source = (ROOT / "h200/run_baselines.sh").read_text(encoding="utf-8")
    assert "H200_BASELINE_FOLLOWUP_ONLY" in source
    assert 'QUEUE+=(--lane h200)' in source
    assert "scripts/run_imagenet1k_baseline_followup_queue.py" in source


def test_followup_tasks_do_not_depend_on_campaign_seed_list(tmp_path: Path) -> None:
    campaign = queue.load_campaign(ROOT / "h200/baselines/campaign.json")
    object.__setattr__(campaign, "seeds", (501,))
    tasks = followup._selected_tasks(campaign, tmp_path, "qlab0")
    assert [(task.model_key, task.seed) for task in tasks] == list(
        followup.FOLLOWUP_TASKS["qlab0"]
    )
    assert all("followup-full" in task.output_dir.parts for task in tasks)


def test_qlab_followup_wandb_ids_are_stable_and_complete(tmp_path: Path) -> None:
    path = followup._configure_qlab_wandb(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [
        record
        for model in payload["runs"].values()
        for record in model["seeds"].values()
    ]
    assert len(records) == 8
    assert len({record["id"] for record in records}) == 8
    assert payload["group"] == followup.QLAB_WANDB_GROUP


def test_followup_job_registration_mutates_status_jobs(tmp_path: Path) -> None:
    campaign = queue.load_campaign(ROOT / "h200/baselines/campaign.json")
    status = queue._new_status(campaign, "a" * 64, tmp_path)
    task = followup._selected_tasks(campaign, tmp_path, "qlab0")[0]
    followup._ensure_job(status, task)
    jobs = status["jobs"]
    assert isinstance(jobs, dict)
    assert task.task_id in jobs
