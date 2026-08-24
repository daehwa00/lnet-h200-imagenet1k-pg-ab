from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import json
from pathlib import Path

import torch

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k_family_p_refinement_imagenet100 as runner
from scripts import run_a2d_r2k3_k_family_wave_a_imagenet100 as family
from scripts import run_h200_imagenet100_k_family_p_refinement as worker

ROOT = Path(__file__).resolve().parents[1]
PARAMETER_COUNTS = {
    runner.M_K48_P80: 857_124,
    runner.L_K64_P2: 1_290_308,
    runner.L_K64_P23: 1_339_652,
    runner.S_K32_P3: 430_404,
    runner.XL_K96_P2_LITE: 2_814_116,
}
RUN_IDS = {
    runner.M_K48_P80: "ad0212fc586e0ee1",
    runner.L_K64_P2: "7d9e3d591ea2081f",
    runner.L_K64_P23: "e37237cce614704b",
    runner.S_K32_P3: "30a262a513661e12",
    runner.XL_K96_P2_LITE: "5f4e05906b50dd3d",
}


def test_refinement_order_and_exact_model_contracts() -> None:
    assert tuple(PARAMETER_COUNTS) == runner.VARIANTS
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        runner._assert_model(model, variant)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == PARAMETER_COUNTS[variant]
        )
        width = runner.POLICIES[variant][0]
        assert runner.SPECS[variant].excitation_modes == (width,) * 4
        assert runner.SPECS[variant].depth == (2, 2, 6, 2)
        for name in family.STAGE_NAMES[:3]:
            transition = getattr(model, name).augmented
            assert type(transition) is GatedPoleExcitationS2DTransition
            assert transition.carry_projection is None


def test_refinement_runtime_and_run_ids_are_frozen() -> None:
    payload = json.loads(
        (ROOT / "h200/k_family_p_refinement/campaign.runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["schema"] == worker.RUNTIME_SCHEMA
    assert tuple(payload["training"]["variants"]) == runner.VARIANTS
    assert payload["parameter_counts"] == PARAMETER_COUNTS
    actual = {
        variant: payload["runs"][variant]["501"]["id"] for variant in runner.VARIANTS
    }
    assert actual == RUN_IDS


def test_refinement_launcher_binds_the_generic_safe_path() -> None:
    source = (ROOT / "h200/run_imagenet100_k_family_p_refinement.sh").read_text(
        encoding="utf-8"
    )
    assert "H200_K_FAMILY_VARIANT_COUNT=5" in source
    assert "H200_K_FAMILY_P_REFINEMENT_WANDB_RUNTIME" in source
    assert "control/imagenet100-k-family-p-refinement" in source
