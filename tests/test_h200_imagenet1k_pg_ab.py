from __future__ import annotations

# ruff: noqa: S108, SLF001
import json
from argparse import Namespace
from pathlib import Path

import run_h200_imagenet1k_pg_ab as runner

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


def test_contract_is_1000_way_and_paired() -> None:
    data_root = Path("/home/qlab/data/ImageNet100")
    if not data_root.exists():
        return
    payload = runner._contract(_args(data_root))
    assert payload["model"]["output_dim"] == runner.NUM_CLASSES
    assert payload["variants"] == list(runner.VARIANTS)
    assert payload["seeds"] == list(runner.SEEDS)
    assert set(payload["parameter_counts"]) == set(runner.VARIANTS)
    assert payload["comparison"]["controlled_factor"].startswith("Stage1-3")


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
