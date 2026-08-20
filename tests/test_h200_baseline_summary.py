from __future__ import annotations

# ruff: noqa: TC003
import json
from pathlib import Path

import pytest
import summarize_h200_baselines as summary


def test_summary_reports_three_seed_mean_and_incomplete_models(tmp_path: Path) -> None:
    campaign = {
        "campaign_id": "test",
        "seeds": [501, 509, 521],
        "full_training": {"epochs": 100},
        "models": [
            {"key": "complete", "display_name": "Complete"},
            {"key": "missing", "display_name": "Missing"},
        ],
    }
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign))
    for index, seed in enumerate(campaign["seeds"]):
        path = tmp_path / "full" / "complete" / f"seed_{seed}" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "model_key": "complete",
                    "seed": seed,
                    "completed_epochs": 100,
                    "learning_rate": 1.0e-3,
                    "parameters": 10,
                    "training_seconds": 10.0 + index,
                    "contract_sha256": str(index) * 64,
                    "metrics": {
                        "validation_top1": 0.7 + 0.01 * index,
                        "validation_top5": 0.9,
                        "images_per_second": 1000.0,
                    },
                }
            )
        )

    payload = summary.summarize(campaign_path, tmp_path)

    assert payload["complete"] is False
    assert payload["complete_models"] == 1
    assert payload["models"]["complete"]["validation_top1"]["mean"] == pytest.approx(0.71)
    assert payload["models"]["missing"]["missing_seeds"] == [501, 509, 521]
