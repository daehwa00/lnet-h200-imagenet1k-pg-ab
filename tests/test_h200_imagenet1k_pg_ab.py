from __future__ import annotations

# ruff: noqa: S108, SLF001
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import run_h200_imagenet1k_pg_ab as runner

from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN

if TYPE_CHECKING:
    import pytest


def _args(data_root: Path) -> Namespace:
    return Namespace(
        root=Path("/tmp/lnet-h200-contract-test"),
        data_root=data_root,
        variants=runner.VARIANTS,
        run_seeds=runner.SEEDS,
        epochs=100,
        batch_size=256,
        gradient_accumulation_steps=1,
        workers=8,
        precision="bfloat16",
        initialize_only=True,
    )


def test_imagenet1k_models_change_only_pg_topology() -> None:
    runner._configure()
    ramp = runner.base.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=runner.NUM_CLASSES, stem_strides=(2, 2))
    pg_model = runner._build(runner.PG_VARIANT, config)
    no_pg_model = runner._build(runner.NO_PG_VARIANT, config)

    assert pg_model.classifier.fusion.classifier.out_features == runner.NUM_CLASSES
    assert pg_model.classifier.affine.linear.out_features == runner.NUM_CLASSES
    assert no_pg_model.classifier.fusion.classifier.out_features == runner.NUM_CLASSES
    assert no_pg_model.classifier.affine.linear.out_features == runner.NUM_CLASSES
    assert sum(isinstance(module, PhaseGatedComplexFFN) for module in pg_model.modules()) == 3
    assert sum(isinstance(module, PhaseGatedComplexFFN) for module in no_pg_model.modules()) == 0


def test_contract_is_1000_way_and_independent_of_imagenet100_manifest(
    tmp_path: Path,
) -> None:
    payload = runner._contract(_args(tmp_path))
    assert payload["model"]["output_dim"] == runner.NUM_CLASSES
    assert payload["variants"] == list(runner.VARIANTS)
    assert payload["seeds"] == list(runner.SEEDS)
    assert set(payload["parameter_counts"]) == set(runner.VARIANTS)
    assert payload["comparison"]["controlled_factor"].startswith("Stage1-3")
    assert payload["data"] == {
        "classes": runner.NUM_CLASSES,
        "dataset": "ImageNet-1K",
        "layout": "ImageFolder train/val validated by h200/run.sh",
        "train_images": 1_281_167,
        "validation_images": 50_000,
    }
    assert json.loads(json.dumps(payload)) == payload


def test_summary_reports_paired_percentage_points(tmp_path: Path) -> None:
    result_root = tmp_path
    results = result_root / "results"
    results.mkdir()
    accuracies = {runner.PG_VARIANT: 0.721, runner.NO_PG_VARIANT: 0.706}
    for variant, accuracy in accuracies.items():
        path = results / f"{variant}__seed{runner.SEEDS[0]}.json"
        path.write_text(json.dumps({"final_validation": {"accuracy": accuracy}}))
    payload = runner._summarize(result_root, {})
    assert payload is not None
    assert abs(payload["pg_minus_no_pg_percentage_points"] - 1.5) < 1.0e-12
    assert (result_root / "summary.json").exists()


def test_wandb_initialization_uses_scoped_relay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_url = (
        "https://wandb.ai/daehwa/alphabet2d-imagenet1k-h200/runs/"
        "fb393834f69d59a2"
    )
    captured: dict[str, object] = {}

    def settings(**kwargs: object) -> dict[str, object]:
        captured["settings"] = kwargs
        return kwargs

    def initialize(**kwargs: object) -> SimpleNamespace:
        captured["init"] = kwargs
        return SimpleNamespace(url=expected_url)

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Settings=settings, init=initialize),
    )
    monkeypatch.setenv("WANDB_API_KEY", "0" * 40)
    monkeypatch.setenv("WANDB_ENTITY", "daehwa")
    monkeypatch.setenv("WANDB_PROJECT", "alphabet2d-imagenet1k-h200")
    monkeypatch.setenv("WANDB_GROUP", "h200-imagenet1k-k3-rmsmatch-pg-ab-v1")
    contract = {
        "model": {},
        "recipe": {},
        "schema": "test",
        "variant_configs": {runner.PG_VARIANT: {}},
    }

    run = runner._initialize_required_wandb_run(
        tmp_path,
        contract,
        variant=runner.PG_VARIANT,
        seed=runner.SEEDS[0],
        parameters=1,
    )

    assert run.url == expected_url
    init = captured["init"]
    assert isinstance(init, dict)
    assert init["anonymous"] == "never"
    assert init["entity"] == "daehwa"
    assert init["id"] == "fb393834f69d59a2"
    settings_payload = captured["settings"]
    assert isinstance(settings_payload, dict)
    assert settings_payload["disable_code"] is True
    assert settings_payload["disable_git"] is True
    assert settings_payload["disable_job_creation"] is True
    assert settings_payload["x_disable_meta"] is True
    assert settings_payload["x_disable_stats"] is True
    assert settings_payload["x_disable_viewer"] is True
    assert settings_payload["x_save_requirements"] is False
    assert settings_payload["x_extra_http_headers"] == {
        "User-Agent": "Mozilla/5.0 lnet-h200-wandb-client/1"
    }
