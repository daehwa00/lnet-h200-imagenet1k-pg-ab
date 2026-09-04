from __future__ import annotations

# ruff: noqa: S607
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_generated_wandb_contract_has_no_drift() -> None:
    subprocess.run(
        ["python", "h200/baselines/generate_wandb_contract.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_baseline_runtime_contains_baselines_and_lnet_k96_runs() -> None:
    campaign_path = ROOT / "h200" / "baselines" / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    runtime = json.loads((ROOT / "h200/baselines/wandb.runtime.json").read_text())
    assert runtime["campaign_manifest_sha256"] == hashlib.sha256(
        campaign_path.read_bytes()
    ).hexdigest()
    assert len(campaign["models"]) == 20
    assert campaign["seeds"] == [501, 509, 521]
    run_ids = {
        record["id"]
        for model in runtime["runs"].values()
        for record in model["seeds"].values()
    }
    assert len(run_ids) == 62
    assert set(runtime["runs"]["lnet_k96_p128x4_d2262_clean_restart_v3"]["seeds"]) == {
        "509",
        "521",
    }
    assert runtime["canary"]["id"] not in run_ids
    generated = (ROOT / "cloudflare/baseline-relay/src/campaign.generated.ts").read_text()
    for model in runtime["runs"].values():
        for record in model["seeds"].values():
            assert record["id"] in generated
            assert record["display_name"] in generated
    assert runtime["wandb_sdk_version"] == "0.22.3"
    assert runtime["console"] == "off"


def test_baseline_entrypoint_freezes_environment_sources_and_parallel_queue() -> None:
    script = (ROOT / "h200/run_baselines.sh").read_text()
    assert "H200_EXPECTED_COMMIT" in script
    assert "H200_ALLOW_NOASSERTION_SOURCES" in script
    assert "git status --porcelain" in script
    assert "${ACTUAL_COMMIT:0:12}" in script
    assert 'readonly PYTHON_VERSION="3.13.11"' in script
    assert 'export WANDB_API_KEY="${DUMMY_WANDB_API_KEY}"' in script
    assert "${WANDB_API_KEY:-" not in script
    assert 'export H200_BASELINE_SOURCE_ROOT="${SOURCE_ROOT}"' in script
    assert "bootstrap_h200_baseline_sources.py" in script
    assert "setup.py bdist_wheel" in script
    assert "DCNv3_build_failed" in script
    assert "DCNV3_WHEEL_SHA_FILE" in script
    assert "H200_DCNV3_WHEEL_SHA256" in script
    assert "--reuse-existing" in script
    assert '"${QUEUE[@]}" --mps auto --mode auto-run' in script


def test_baseline_entrypoint_is_always_owner_controlled() -> None:
    script = (ROOT / "h200/run_baselines.sh").read_text()
    guard = 'if [[ "${H200_OWNER_CONTROL_INNER:-0}" != "1" ]]; then'
    assert guard in script
    assert "scripts/run_h200_owner_controlled.py" in script
    assert 'CONTROL_REF="refs/heads/control/imagenet1k-baselines"' in script
    assert 'CONTROL_REF="refs/heads/control/imagenet1k-baselines-followup"' in script
    assert 'CONTROL_PATH="h200/baselines/control.json"' in script
    assert 'value != "h200-imagenet1k-moga-emo-100ep-s501-v2"' in script
    assert '-- env H200_OWNER_CONTROL_INNER=1 bash "$0"' in script
    assert script.index(guard) < script.index('ACTUAL_COMMIT="$(git rev-parse --verify HEAD)"')


def test_baseline_entrypoint_requires_wandb_canary_before_training() -> None:
    script = (ROOT / "h200/run_baselines.sh").read_text(encoding="utf-8")
    canary = '"${ENV_ROOT}/bin/python" cloudflare/baseline-relay/canary.py'
    dataset = '"${ENV_ROOT}/bin/python" h200/validate_imagenet1k.py'
    assert canary in script
    assert script.index(canary) < script.index(dataset)
    assert "H200_BASELINE_WANDB_CANARY_ONLY_COMPLETE=1" in script


def test_baseline_dependency_lock_is_exact_and_hashed() -> None:
    requirements = (ROOT / "h200/baselines/requirements.txt").read_text()
    lock = (ROOT / "h200/baselines/requirements.lock").read_text()
    for requirement in (
        "timm==1.0.26",
        "torch==2.9.1+cu128",
        "torchvision==0.24.1+cu128",
        "wandb==0.22.3",
        "wheel==0.46.2",
    ):
        assert requirement in requirements
        assert requirement in lock
    assert "--hash=sha256:" in lock
