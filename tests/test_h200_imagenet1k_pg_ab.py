from __future__ import annotations

# ruff: noqa: S108, SLF001
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_h200_imagenet1k_pg_ab as runner
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lnet.a2d_q_heads import expected_calibration_error
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN


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


def _freeze_dataset_manifest(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> dict[str, object]:
    identity = "d" * 64
    payload = {
        "identity_sha256": identity,
        "classes": {"count": 1000, "sha256": "c" * 64},
        "splits": {
            "train": {
                "count": 1_281_167,
                "relpath_size_content_sha256": "a" * 64,
            },
            "val": {"count": 50_000, "relpath_size_content_sha256": "b" * 64},
        },
    }
    path = root / "dataset_manifest.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("LNET_DATASET_MANIFEST_PATH", str(path))
    monkeypatch.setenv("LNET_DATASET_IDENTITY_SHA256", identity)
    monkeypatch.setenv("H200_EXPECTED_COMMIT", "e" * 40)
    return payload


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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_manifest = _freeze_dataset_manifest(monkeypatch, tmp_path)
    payload = runner._contract(_args(tmp_path))
    assert payload["model"]["output_dim"] == runner.NUM_CLASSES
    assert payload["variants"] == list(runner.VARIANTS)
    assert payload["seeds"] == list(runner.SEEDS)
    assert set(payload["parameter_counts"]) == set(runner.VARIANTS)
    assert payload["comparison"]["controlled_factor"].startswith("Stage1-3")
    assert payload["schema"].endswith("h200.v3")
    assert payload["campaign"]["id"].endswith("-v3")
    assert payload["data"]["classes"] == runner.NUM_CLASSES
    assert payload["data"]["train_images"] == 1_281_167
    assert payload["data"]["validation_images"] == 50_000
    assert payload["data"]["identity_sha256"] == dataset_manifest["identity_sha256"]
    assert payload["telemetry"]["authority"].startswith("durable local")
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
        "56038182b4beb318"
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
    monkeypatch.setenv(
        "WANDB_BASE_URL",
        "https://lnet-h200-wandb-relay-v3.gpupulse-monitor.workers.dev",
    )
    monkeypatch.setenv("WANDB_APP_URL", "https://wandb.ai")
    monkeypatch.setenv("WANDB_ENTITY", "daehwa")
    monkeypatch.setenv("WANDB_PROJECT", "alphabet2d-imagenet1k-h200")
    monkeypatch.setenv("WANDB_GROUP", "h200-imagenet1k-k3-rmsmatch-pg-ab-v3")
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    contract = {
        "model": {},
        "recipe": {},
        "schema": "test",
        "data": {"identity_sha256": "d" * 64},
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
    assert init["id"] == "56038182b4beb318"
    assert init["name"] == "H200-I1K-PG-s501-v3"
    assert init["group"] == "h200-imagenet1k-k3-rmsmatch-pg-ab-v3"
    settings_payload = captured["settings"]
    assert isinstance(settings_payload, dict)
    assert settings_payload["disable_code"] is True
    assert settings_payload["disable_git"] is True
    assert settings_payload["disable_job_creation"] is True
    assert settings_payload["console"] == "off"
    assert settings_payload["x_disable_meta"] is True
    assert settings_payload["x_disable_stats"] is True
    assert settings_payload["x_disable_viewer"] is True
    assert settings_payload["x_save_requirements"] is False
    assert settings_payload["x_extra_http_headers"] == {
        "User-Agent": "Mozilla/5.0 lnet-h200-wandb-client/1"
    }


def test_streaming_evaluation_matches_full_tensor_metrics() -> None:
    runner._configure()
    head_type = runner.base.control.control.head_runner.A2DAffineQClassifier

    class Affine(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.standardizer = nn.Identity()
            self.linear = nn.Linear(3, 4, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(self.standardizer(inputs))

    affine = Affine()
    fusion = nn.Linear(3, 4, bias=False)
    classifier = head_type(
        3,
        4,
        main="fusion",
        affine=affine,
        fusion=fusion,
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=0.2,
    )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = classifier

        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return self.classifier(inputs)

    model = Model()
    inputs = torch.tensor(
        [
            [0.0, 0.5, 1.0],
            [1.0, -0.5, 0.25],
            [-1.0, 0.75, 0.5],
            [0.2, 0.1, -0.3],
            [0.9, 0.8, -0.7],
        ]
    )
    labels = torch.tensor([0, 1, 2, 3, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    metrics = runner._streaming_qhead_evaluate(
        model,
        model,
        loader,
        torch.device("cpu"),
        precision="float32",
    )
    with torch.inference_mode():
        joint, affine_logits, fusion_logits, _, _ = model(inputs)
    assert metrics["accuracy"] == pytest.approx(
        float(joint.argmax(dim=-1).eq(labels).float().mean())
    )
    assert metrics["nll"] == pytest.approx(
        float(torch.nn.functional.cross_entropy(joint, labels))
    )
    assert metrics["ece"] == pytest.approx(expected_calibration_error(joint, labels))
    assert metrics["affine_only_nll"] == pytest.approx(
        float(torch.nn.functional.cross_entropy(affine_logits, labels))
    )
    assert metrics["fusion_only_nll"] == pytest.approx(
        float(torch.nn.functional.cross_entropy(fusion_logits, labels))
    )
    assert metrics["affine_reconstruction_max_error"] == 0.0
