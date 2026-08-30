from __future__ import annotations

# pyright: reportPrivateUsage=false
from pathlib import Path

from scripts import h200_baseline_registry as registry
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
