from __future__ import annotations

# ruff: noqa: S607, SLF001
# pyright: reportOptionalMemberAccess=false, reportPrivateUsage=false
# pyright: reportUnknownLambdaType=false
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as stage
import run_h200_imagenet100_stage_allocation as h200_runner


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "h200/stage_allocation/campaign.runtime.json"


def _prepare_module() -> ModuleType:
    path = ROOT / "h200/stage_allocation/prepare_imagenet100.py"
    spec = importlib.util.spec_from_file_location("prepare_imagenet100", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_contract_contains_thirteen_scoped_runs() -> None:
    subprocess.run(
        ["python", "h200/stage_allocation/generate_contract.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    runtime = json.loads(RUNTIME_PATH.read_text())
    assert runtime["campaign_id"] == "h200-imagenet100-stage-allocation-s501-v2"
    assert runtime["output_namespace"] == "lnet-h200-imagenet100-stage-allocation-v2"
    assert runtime["relay_protocol_version"] == "wandb-0.22.3-h200-imagenet100-stage-v2"
    assert runtime["training"]["variants"] == list(stage.VARIANTS)
    assert runtime["training"] == {
        "batch_size": 128,
        "epochs": 100,
        "precision": "bfloat16",
        "seed": 501,
        "variants": list(stage.VARIANTS),
    }
    records = [record["501"] for record in runtime["runs"].values()]
    assert len(records) == 13
    assert len({record["id"] for record in records}) == 13
    assert runtime["canary"]["id"] not in {record["id"] for record in records}


def test_h200_entrypoint_is_commit_bound_and_runs_the_exact_screen() -> None:
    script = (ROOT / "h200/run_imagenet100_stage_allocation.sh").read_text()
    subprocess.run(
        ["bash", "-n", "h200/run_imagenet100_stage_allocation.sh"],
        cwd=ROOT,
        check=True,
    )
    assert "H200_EXPECTED_COMMIT" in script
    assert "git status --porcelain" in script
    assert "generate_contract.py --check" in script
    assert "prepare_imagenet100.py" in script
    assert "smoke_r2k3_campaign.py" in script
    assert "run_h200_imagenet100_stage_allocation.py" in script
    assert "cloudflare/stage-allocation-relay/canary.py" in script
    assert "--full-batch-size 128" in script
    assert "--epochs 100" in script
    assert 'export WANDB_API_KEY="${DUMMY_WANDB_API_KEY}"' in script
    assert "export LNET_COMPILE_MODE=reduce-overhead" in script
    assert "export LNET_DATALOADER_PREFETCH_FACTOR=2" in script
    assert "WORKERS=$((CPU_COUNT - 1))" in script
    assert "cp --archive --dereference --reflink=auto" in script
    assert "throughput < 300.0" in script
    assert "STAGED_DATA_AVAILABLE_BYTES" in script
    assert "--epochs 2" in script
    canary = (ROOT / "cloudflare/stage-allocation-relay/canary.py").read_text()
    assert '"stage_allocation" / "campaign.runtime.json"' in canary
    assert "run.status().sync_items_pending" in canary
    assert "time.sleep(35)" in canary
    assert '"filestream: fatal error"' in canary


def test_first100_view_is_zero_copy_and_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepare = _prepare_module()
    source = tmp_path / "imagenet"
    output = tmp_path / "view"
    names = prepare._desired_classes()
    source_names = (*names, *(f"unused-{index:04d}" for index in range(900)))
    for split in ("train", "val"):
        for name in source_names:
            directory = source / split / name
            directory.mkdir(parents=True)
            (directory / "sample.jpg").write_bytes(b"image")
    monkeypatch.setattr(
        prepare,
        "_count",
        lambda _source, split, _classes: 130000 if split == "train" else 5000,
    )

    first = prepare.prepare(source, output)
    second = prepare.prepare(source, output)

    assert first == second
    assert first["classes"] == list(names)
    assert (output / "train" / names[0]).is_symlink()
    assert (output / "val" / names[-1]).resolve() == (source / "val" / names[-1]).resolve()


def test_first100_view_maps_numeric_h200_class_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepare = _prepare_module()
    source = tmp_path / "imagenet"
    output = tmp_path / "view"
    for split in ("train", "val"):
        for index in range(1000):
            directory = source / split / str(index)
            directory.mkdir(parents=True)
            (directory / "sample.jpg").write_bytes(b"image")
    monkeypatch.setattr(
        prepare,
        "_count",
        lambda _source, split, _classes: 130000 if split == "train" else 5000,
    )

    payload = prepare.prepare(source, output)
    names = prepare._desired_classes()

    assert payload["source_class_by_synset"][names[0]] == "0"
    assert payload["source_class_by_synset"][names[-1]] == "99"
    assert (output / "train" / names[0]).resolve() == (source / "train/0").resolve()
    assert (output / "val" / names[-1]).resolve() == (source / "val/99").resolve()


def test_wandb_run_uses_variant_scoped_relay_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = json.loads(RUNTIME_PATH.read_text())
    variant = stage.VARIANTS[0]
    record = runtime["runs"][variant]["501"]
    captured: dict[str, object] = {}

    def settings(**kwargs: object) -> dict[str, object]:
        captured["settings"] = kwargs
        return kwargs

    def initialize(**kwargs: object) -> SimpleNamespace:
        captured["init"] = kwargs
        return SimpleNamespace(
            url=f"https://wandb.ai/{runtime['entity']}/{runtime['project']}/runs/{record['id']}"
        )

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Settings=settings, init=initialize))
    monkeypatch.setenv("H200_STAGE_ALLOCATION_WANDB_RUNTIME", str(RUNTIME_PATH))
    monkeypatch.setenv("WANDB_API_KEY", "0" * 40)
    monkeypatch.setenv("WANDB_APP_URL", runtime["wandb_app_url"])
    monkeypatch.setenv("WANDB_BASE_URL", runtime["wandb_base_url"])
    monkeypatch.setenv("WANDB_ENTITY", runtime["entity"])
    monkeypatch.setenv("WANDB_PROJECT", runtime["project"])
    monkeypatch.setenv("WANDB_GROUP", runtime["group"])
    monkeypatch.setenv("WANDB_CONSOLE", runtime["console"])

    run = h200_runner._initialize_required_wandb_run(
        tmp_path,
        {
            "model": {},
            "recipe": {},
            "schema": "test",
            "variant_configs": {variant: {}},
        },
        variant=variant,
        seed=501,
        parameters=1,
    )

    assert run is not None
    assert run.url.endswith(record["id"])
    init = cast("dict[str, object]", captured["init"])
    assert init["id"] == record["id"]
    assert init["name"] == record["display_name"]
    assert init["tags"] == record["tags"]
    settings_payload = cast("dict[str, object]", captured["settings"])
    assert settings_payload["disable_code"] is True
    assert settings_payload["disable_git"] is True
    assert settings_payload["x_disable_stats"] is True


def test_h200_prefetch_override_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_DATALOADER_PREFETCH_FACTOR", "4")
    assert h200_runner.harness._active_loader_prefetch_factor() == 4

    monkeypatch.setenv("LNET_DATALOADER_PREFETCH_FACTOR", "9")
    with pytest.raises(ValueError, match="between 1 and 8"):
        h200_runner.harness._active_loader_prefetch_factor()


def test_owner_stop_marker_is_validated_off_the_training_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = json.loads(RUNTIME_PATH.read_text())
    target_commit = "a" * 40
    marker = tmp_path / "stopped.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "lnet.h200.owner_stop.v1",
                "campaign_id": runtime["campaign_id"],
                "target_commit": target_commit,
                "generation": 2,
                "reason": "test stop",
                "control_updated_at": "2026-08-21T14:02:37+09:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("H200_STAGE_ALLOCATION_WANDB_RUNTIME", str(RUNTIME_PATH))
    monkeypatch.setenv("H200_EXPECTED_COMMIT", target_commit)

    record = h200_runner._read_owner_stop_marker(marker)

    assert record is not None
    assert record["generation"] == 2
    assert record["action"] == "stop"
