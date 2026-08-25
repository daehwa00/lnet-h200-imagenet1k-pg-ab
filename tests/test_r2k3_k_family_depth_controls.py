from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import json
from pathlib import Path

import torch

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k_family_depth_controls_imagenet100 as runner
from scripts import run_h200_imagenet100_k_family_depth_controls as worker

ROOT = Path(__file__).resolve().parents[1]
PARAMETER_COUNTS = {
    runner.S_D2262: 399_492,
    runner.S_D2242: 337_732,
    runner.L_D2262: 1_273_860,
    runner.L_D2282: 1_479_236,
    runner.XL_D2282: 3_249_124,
}
RUN_IDS = {
    runner.S_D2262: "f3edf4e837a0551f",
    runner.S_D2242: "1976e239074f0a4c",
    runner.L_D2262: "bcfd6ba8ff117c7f",
    runner.L_D2282: "8359cfa6f1cfac97",
    runner.XL_D2282: "bffc6cc43cb29ada",
}


def test_depth_control_models_are_exact_and_terminal_depth_two() -> None:
    assert tuple(PARAMETER_COUNTS) == runner.VARIANTS
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        runner._assert_model(model, variant)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == PARAMETER_COUNTS[variant]
        )
        assert runner.SPECS[variant].depth[-1] == 2
        assert len(set(runner.SPECS[variant].pole_modes)) == 1
        for name in runner.STAGE_NAMES[:3]:
            transition = getattr(model, name).augmented
            assert type(transition) is GatedPoleExcitationS2DTransition
            assert transition.carry_projection is None


def test_depth_control_runtime_excludes_completed_xl_control() -> None:
    payload = json.loads(
        (ROOT / "h200/k_family_depth_controls/campaign.runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["schema"] == worker.RUNTIME_SCHEMA
    assert tuple(payload["training"]["variants"]) == runner.VARIANTS
    assert payload["parameter_counts"] == PARAMETER_COUNTS
    assert "XL-K96-P128x4-D2262" not in payload["runs"]
    reused = payload["reused_controls"]["XL-K96-P128x4-D2262"]
    assert reused["wandb_run_id"] == "ea0a8a0e72d323c3"
    actual = {
        variant: payload["runs"][variant]["501"]["id"] for variant in runner.VARIANTS
    }
    assert actual == RUN_IDS


def test_depth_control_launcher_and_smoke_bindings_are_frozen() -> None:
    source = (ROOT / "h200/run_imagenet100_k_family_depth_controls.sh").read_text(
        encoding="utf-8"
    )
    assert "H200_K_FAMILY_VARIANT_COUNT=5" in source
    assert "H200_K_FAMILY_DEPTH_CONTROLS_WANDB_RUNTIME" in source
    assert "control/imagenet100-k-family-depth-controls" in source
