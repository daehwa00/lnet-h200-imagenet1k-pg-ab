from __future__ import annotations

# ruff: noqa: S603, S607, SLF001
# pyright: reportPrivateUsage=false
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import run_a2d_r2k3_reader_wl_imagenet100 as experiment
import run_h200_imagenet100_reader_wl as h200_runner

if TYPE_CHECKING:
    from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "h200/reader_wl/campaign.runtime.json"


def _relay_generator() -> ModuleType:
    path = ROOT / "h200/stage_allocation/generate_contract.py"
    spec = importlib.util.spec_from_file_location("stage_relay_generator_reader_wl", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_campaign_freezes_two_reader_runs() -> None:
    subprocess.run(
        ["python", "h200/reader_wl/generate_contract.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    runtime = json.loads(RUNTIME_PATH.read_text())
    assert runtime["schema"] == "lnet.h200.imagenet100.reader_wl.runtime.v1"
    assert runtime["campaign_id"] == "h200-imagenet100-reader-wl-s501-v1"
    assert runtime["output_namespace"] == "lnet-h200-imagenet100-reader-wl-v1"
    assert runtime["training"] == {
        "batch_size": 128,
        "epochs": 100,
        "execution": "one_model_to_epoch_100_then_next",
        "precision": "bfloat16",
        "seed": 501,
        "variants": list(experiment.VARIANTS),
    }
    assert runtime["program"] == "h200/run_imagenet100_reader_wl.sh"
    records = [runtime["runs"][variant]["501"] for variant in experiment.VARIANTS]
    assert len({record["id"] for record in records}) == 2
    assert runtime["canary"]["id"] not in {record["id"] for record in records}
    generated = (ROOT / "cloudflare/stage-allocation-relay/src/campaign.generated.ts").read_text()
    for record in [*records, runtime["canary"]]:
        assert record["id"] in generated
        assert record["display_name"] in generated


def test_h200_shell_runs_one_reader_variant_per_process() -> None:
    script_path = ROOT / "h200/run_imagenet100_reader_wl.sh"
    script = script_path.read_text()
    subprocess.run(["bash", "-n", str(script_path)], cwd=ROOT, check=True)
    assert "H200_EXPECTED_COMMIT" in script
    assert "h200/reader_wl/generate_contract.py --check" in script
    assert "lnet.h200.imagenet100.reader_wl.runtime.v1" in script
    assert "canary_reader_wl.py" in script
    assert 'for variant in "${READER_VARIANTS[@]}"' in script
    assert '--variants "${variant}"' in script
    assert script.count("scripts/run_h200_imagenet100_reader_wl.py") == 1
    assert "--managed-canonical-receipt h200/imagenet1k_canonical_receipt.json" in script
    assert "H200_READER_WL_RESULTS_COMPLETE" in script
    assert script.index('CPU_COUNT="$(nproc)"') < script.index("export OMP_NUM_THREADS=1")


def test_h200_wrapper_accepts_exactly_one_reader_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "--variants", experiment.STRICT])
    assert h200_runner._selected_variant() == experiment.STRICT
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--variants", experiment.STRICT, experiment.WL],
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        h200_runner._selected_variant()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("training", "variants", [experiment.WL, experiment.STRICT]),
        ("training", "precision", "float32"),
        ("training", "execution", "parallel"),
        ("wandb", "group", "h200-imagenet100-stage-allocation-s501-v3"),
    ],
)
def test_shared_relay_fails_closed_on_reader_campaign_changes(
    section: str,
    key: str,
    value: object,
) -> None:
    generator = _relay_generator()
    primary = json.loads((ROOT / "h200/stage_allocation/campaign.json").read_text())
    reader = json.loads((ROOT / "h200/reader_wl/campaign.json").read_text())
    generator._validate_reader_wl(primary, reader)
    mutated = copy.deepcopy(reader)
    mutated[section][key] = value
    with pytest.raises(ValueError, match="strict-versus-WL"):
        generator._validate_reader_wl(primary, mutated)
