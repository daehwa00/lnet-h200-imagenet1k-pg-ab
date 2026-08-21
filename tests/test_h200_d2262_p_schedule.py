from __future__ import annotations

# ruff: noqa: S603, S607, SLF001
# pyright: reportPrivateUsage=false
import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import run_a2d_r2k3_d2262_p_schedule_imagenet100 as experiment
import run_h200_imagenet100_d2262_p_schedule as h200_runner
from torch import nn

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "h200/d2262_p_schedule/campaign.runtime.json"


def _relay_generator() -> ModuleType:
    path = ROOT / "h200/stage_allocation/generate_contract.py"
    spec = importlib.util.spec_from_file_location("stage_relay_generator", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_campaign_freezes_six_sequential_runs() -> None:
    subprocess.run(
        ["python", "h200/d2262_p_schedule/generate_contract.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    runtime = json.loads(RUNTIME_PATH.read_text())

    assert runtime["schema"] == "lnet.h200.imagenet100.d2262_p_schedule.runtime.v1"
    assert runtime["campaign_id"] == "h200-imagenet100-d2262-p-schedule-s501-v1"
    assert runtime["training"] == {
        "batch_size": 128,
        "epochs": 100,
        "execution": "one_model_to_epoch_100_then_next",
        "precision": "bfloat16",
        "seed": 501,
        "variants": list(experiment.VARIANTS),
    }
    assert runtime["program"] == "h200/run_imagenet100_d2262_p_schedule.sh"
    records = [runtime["runs"][variant]["501"] for variant in experiment.VARIANTS]
    assert len({record["id"] for record in records}) == 6
    assert len({record["display_name"] for record in records}) == 6
    assert runtime["canary"]["id"] not in {record["id"] for record in records}


def test_h200_shell_runs_exactly_one_model_per_process() -> None:
    script_path = ROOT / "h200/run_imagenet100_d2262_p_schedule.sh"
    script = script_path.read_text()
    subprocess.run(["bash", "-n", str(script_path)], cwd=ROOT, check=True)

    assert "H200_EXPECTED_COMMIT" in script
    assert "git status --porcelain" in script
    assert "h200/d2262_p_schedule/generate_contract.py --check" in script
    assert "canary_d2262_p_schedule.py" in script
    assert 'for variant in "${P_SCHEDULE_VARIANTS[@]}"' in script
    assert '--variants "${variant}"' in script
    assert script.count("scripts/run_h200_imagenet100_d2262_p_schedule.py") == 1
    assert "shared_batch_cohort" not in script
    assert "benchmark_h200_stage_allocation_autotune.py" not in script
    assert "--batch-size 128" in script
    assert "--epochs 100" in script
    assert "--run-seeds 501" in script
    assert "--kill-after=5m 24h" in script
    assert "H200_D2262_P_RESULTS_COMPLETE" in script


def test_h200_wrapper_accepts_exactly_one_registered_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--variants", experiment.A, "--epochs", "100"],
    )
    assert h200_runner._selected_variant() == experiment.A

    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--variants", experiment.A, experiment.B, "--epochs", "100"],
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        h200_runner._selected_variant()


def test_scientific_contract_ignores_pod_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "model": {},
        "recipe": {"cpu_affinity": "0", "epochs": 100},
        "runtime": {"hostname": "ephemeral-pod"},
        "source_sha256": {},
    }
    monkeypatch.setattr(experiment.runtime, "base_contract", lambda _args: base)
    monkeypatch.setattr(experiment, "_build", lambda _variant, _config: nn.Linear(1, 1))
    args = argparse.Namespace()

    contract = experiment._contract(args)

    assert "hostname" not in contract["runtime"]
    assert "cpu_affinity" not in contract["recipe"]
    assert contract["variants"] == list(experiment.VARIANTS)
    assert contract["seeds"] == [501]
    assert set(contract["parameter_counts"]) == set(experiment.VARIANTS)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("training", "variants", [*experiment.VARIANTS[:-1], "unregistered"]),
        ("training", "precision", "float32"),
        ("training", "execution", "parallel"),
        ("wandb", "group", "h200-imagenet100-stage-allocation-s501-v3"),
    ],
)
def test_shared_relay_fails_closed_on_supplemental_identity_changes(
    section: str,
    key: str,
    value: object,
) -> None:
    generator = _relay_generator()
    primary = json.loads((ROOT / "h200/stage_allocation/campaign.json").read_text())
    supplemental = json.loads((ROOT / "h200/d2262_p_schedule/campaign.json").read_text())
    generator._validate_supplemental(primary, supplemental)
    mutated = copy.deepcopy(supplemental)
    mutated[section][key] = value

    with pytest.raises(ValueError, match="invalid supplemental"):
        generator._validate_supplemental(primary, mutated)
