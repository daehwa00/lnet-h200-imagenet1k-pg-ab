from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k_family_wave_a_h200_imagenet100 as runner
from scripts import run_a2d_r2k3_k_family_wave_a_imagenet100 as family
from scripts import run_h200_k_family_xl_queue as queue

if TYPE_CHECKING:
    import pytest

ROOT = Path(__file__).resolve().parents[1]
PARAMETER_COUNTS = {
    "XL-K96-U1": 2_313_892,
    "XL-K96-U125": 2_791_524,
    "XL-K96-Shaped": 2_996_740,
    "XL-K96-Rich": 3_200_068,
}
RUN_IDS = {
    "XL-K96-U1": "baf09e91f1b0cf25",
    "XL-K96-U125": "ea0a8a0e72d323c3",
    "XL-K96-Shaped": "b1328c66a789d2a1",
    "XL-K96-Rich": "14000aaadd786a84",
}


def test_xl_models_match_exact_parameter_and_identity_carry_contracts() -> None:
    assert runner.VARIANTS == family.XL_VARIANTS
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        family._assert_model(model, variant)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == PARAMETER_COUNTS[variant]
        )
        assert family.SPECS[variant].excitation_modes == (96, 96, 96, 96)
        assert family.SPECS[variant].depth == (2, 2, 6, 2)
        for name in family.STAGE_NAMES[:3]:
            transition = getattr(model, name).augmented
            assert type(transition) is GatedPoleExcitationS2DTransition
            assert transition.carry_projection is None


def test_xl_runtime_has_frozen_order_counts_and_run_ids() -> None:
    payload = json.loads(
        (ROOT / "h200/k_family_xl/campaign.runtime.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "lnet.h200.imagenet100.k_family_xl.runtime.v1"
    assert tuple(payload["training"]["variants"]) == runner.VARIANTS
    assert payload["parameter_counts"] == PARAMETER_COUNTS
    actual = {
        variant: payload["runs"][variant]["501"]["id"] for variant in runner.VARIANTS
    }
    assert actual == RUN_IDS


def test_xl_control_is_stopped_before_representative_smoke() -> None:
    payload = json.loads((ROOT / "h200/k_family_xl/control.json").read_text(encoding="utf-8"))
    assert payload["campaign_id"] == "h200-imagenet100-k-family-xl-s501-v1"
    assert payload["target_commit"] == "0" * 40
    assert payload["generation"] == 1
    assert payload["action"] == "stop"


def test_xl_queue_continues_after_one_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "schema": queue.RUNTIME_SCHEMA,
                "campaign_id": "test-xl",
                "parameter_counts": PARAMETER_COUNTS,
                "training": {
                    "variants": list(runner.VARIANTS),
                    "seed": 501,
                    "epochs": 100,
                    "batch_size": 128,
                    "precision": "bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "run"
    completed: set[str] = set()
    launched: list[str] = []

    def fake_complete(_root: Path, variant: str, _seed: int, _epochs: int) -> bool:
        return variant in completed

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        variant = command[command.index("--variants") + 1]
        launched.append(variant)
        if variant == "XL-K96-U125":
            return SimpleNamespace(returncode=17)
        completed.add(variant)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("H200_K_FAMILY_XL_WANDB_RUNTIME", str(runtime_path))
    monkeypatch.setenv("H200_EXPECTED_COMMIT", "a" * 40)
    monkeypatch.setenv("H200_CONTROL_STOP_MARKER", str(tmp_path / "stop.json"))
    monkeypatch.setattr(queue, "_complete_result", fake_complete)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_h200_k_family_xl_queue.py",
            "--root",
            str(root),
            "--data-root",
            str(tmp_path / "data"),
            "--workers",
            "8",
        ],
    )
    assert queue.main() == 1
    assert tuple(launched) == runner.VARIANTS
    status = json.loads((root / "k-family-xl-queue.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETE_WITH_FAILURES"
    assert status["jobs"]["XL-K96-U125"]["status"] == "FAILED"
    assert all(
        status["jobs"][variant]["status"] == "COMPLETED"
        for variant in runner.VARIANTS
        if variant != "XL-K96-U125"
    )
