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
from typing import TYPE_CHECKING, cast

import run_a2d_r2k3_stage_allocation_screen_imagenet100 as stage
import run_h200_imagenet100_stage_allocation as h200_runner

if TYPE_CHECKING:
    import pytest


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


def test_generated_contract_contains_twelve_scoped_runs() -> None:
    subprocess.run(
        ["python", "h200/stage_allocation/generate_contract.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    runtime = json.loads(RUNTIME_PATH.read_text())
    assert runtime["training"]["variants"] == list(stage.VARIANTS)
    assert runtime["training"] == {
        "batch_size": 128,
        "epochs": 100,
        "precision": "bfloat16",
        "seed": 501,
        "variants": list(stage.VARIANTS),
    }
    records = [record["501"] for record in runtime["runs"].values()]
    assert len(records) == 12
    assert len({record["id"] for record in records}) == 12
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
    assert "--full-batch-size 128" in script
    assert "--epochs 100" in script
    assert 'export WANDB_API_KEY="${DUMMY_WANDB_API_KEY}"' in script


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
