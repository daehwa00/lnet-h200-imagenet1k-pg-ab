from __future__ import annotations

# ruff: noqa: S607
# pyright: reportPrivateUsage=false
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import run_a2d_r2k3_k64_p_allocation_d2262_imagenet100 as experiment
import run_h200_imagenet100_k64_p_small_factorial as h200_runner
import run_h200_k64_p_small_factorial_queue as queue

if TYPE_CHECKING:
    from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "h200/k64_p_small_factorial/campaign.runtime.json"


def _relay_generator() -> ModuleType:
    path = ROOT / "h200/stage_allocation/generate_contract.py"
    spec = importlib.util.spec_from_file_location(
        "stage_relay_generator_k64_p_small_factorial",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_campaign_freezes_two_k64_p_small_factorial_runs() -> None:
    subprocess.run(
        ["python", "h200/k64_p_small_factorial/generate_contract.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    runtime = json.loads(RUNTIME_PATH.read_text())
    assert runtime["schema"] == "lnet.h200.imagenet100.k64_p_small_factorial.runtime.v1"
    assert runtime["campaign_id"] == "h200-imagenet100-k64-p-small-factorial-d2262-s501-v1"
    assert runtime["output_namespace"] == "lnet-h200-imagenet100-k64-p-small-factorial-d2262-v1"
    assert runtime["training"] == {
        "batch_size": 128,
        "epochs": 100,
        "execution": "one_model_to_epoch_100_then_next",
        "precision": "bfloat16",
        "seed": 501,
        "variants": list(experiment.H200_VARIANTS),
    }
    assert runtime["project"] == "alphabet2d-imagenet100"
    assert runtime["group"] == "R2K3-K64-PSmallFactorial-D2262-H200-S501"
    assert runtime["program"] == "h200/run_imagenet100_k64_p_small_factorial.sh"
    records = [runtime["runs"][variant]["501"] for variant in experiment.H200_VARIANTS]
    assert len({record["id"] for record in records}) == 2
    assert runtime["canary"]["id"] not in {record["id"] for record in records}
    generated = (ROOT / "cloudflare/stage-allocation-relay/src/campaign.generated.ts").read_text()
    for record in [*records, runtime["canary"]]:
        assert record["id"] in generated
        assert record["display_name"] in generated


def test_h200_shell_runs_restart_safe_persistent_sequential_queue() -> None:
    script_path = ROOT / "h200/run_imagenet100_k64_p_small_factorial.sh"
    script = script_path.read_text()
    subprocess.run(["bash", "-n", str(script_path)], cwd=ROOT, check=True)
    assert "H200_EXPECTED_COMMIT" in script
    assert "h200/k64_p_small_factorial/generate_contract.py --check" in script
    assert "lnet.h200.imagenet100.k64_p_small_factorial.runtime.v1" in script
    assert "canary_k64_p_small_factorial.py" in script
    assert "scripts/run_h200_k64_p_small_factorial_queue.py" in script
    assert "scripts/smoke_h200_k64_p_small_factorial.py" in script
    assert "--batch-size 128" in script
    assert "WORKERS=8" in script
    assert "LNET_PERSISTENT_WORKERS=1" in script
    assert "--managed-canonical-receipt h200/imagenet1k_canonical_receipt.json" in script
    assert "H200_K64_P_SMALL_FACTORIAL_RESULTS_COMPLETE" in script
    assert script.index('CPU_COUNT="$(nproc)"') < script.index("export OMP_NUM_THREADS=1")


def test_h200_wrapper_accepts_exactly_one_registered_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = experiment.H200_VARIANTS
    monkeypatch.setattr(sys, "argv", ["runner", "--variants", first])
    assert h200_runner._selected_variant() == first
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--variants", first, second],
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        h200_runner._selected_variant()


def test_queue_continues_to_second_variant_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first, second = experiment.H200_VARIANTS
    runtime = {
        "schema": "lnet.h200.imagenet100.k64_p_small_factorial.runtime.v1",
        "campaign_id": "h200-imagenet100-k64-p-small-factorial-d2262-s501-v1",
        "training": {
            "batch_size": 128,
            "epochs": 100,
            "precision": "bfloat16",
            "seed": 501,
            "variants": [first, second],
        },
    }
    monkeypatch.setattr(queue, "_runtime", lambda: runtime)
    completions = iter((False, False, False, True))
    monkeypatch.setattr(queue, "_complete_result", lambda *_args: next(completions))
    calls: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        calls.append(command)
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(queue.subprocess, "run", run)
    monkeypatch.setenv("H200_EXPECTED_COMMIT", "a" * 40)
    monkeypatch.setenv("H200_CONTROL_STOP_MARKER", str(tmp_path / "stop.json"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "--root",
            str(tmp_path / "run"),
            "--data-root",
            str(tmp_path / "data"),
            "--workers",
            "8",
        ],
    )
    assert queue.main() == 1
    assert len(calls) == 2
    status = json.loads((tmp_path / "run/k64-p-small-factorial-queue.json").read_text())
    assert status["status"] == "COMPLETE_WITH_FAILURES"
    assert status["jobs"][first]["status"] == "FAILED"
    assert status["jobs"][second]["status"] == "COMPLETED"


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("training", "variants", list(reversed(experiment.H200_VARIANTS))),
        ("training", "precision", "float32"),
        ("training", "execution", "parallel"),
        ("wandb", "group", "h200-imagenet100-stage-allocation-s501-v3"),
    ],
)
def test_shared_relay_fails_closed_on_k64_campaign_changes(
    section: str,
    key: str,
    value: object,
) -> None:
    generator = _relay_generator()
    primary = json.loads((ROOT / "h200/stage_allocation/campaign.json").read_text())
    campaign = json.loads((ROOT / "h200/k64_p_small_factorial/campaign.json").read_text())
    generator._validate_k64_p_small_factorial(primary, campaign)
    mutated = copy.deepcopy(campaign)
    mutated[section][key] = value
    with pytest.raises(ValueError, match="K64 P-small-factorial"):
        generator._validate_k64_p_small_factorial(primary, mutated)
