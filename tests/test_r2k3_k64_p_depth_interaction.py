from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import torch

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k64_p_depth_interaction_imagenet100 as runner


def test_interaction_grid_contains_four_nonseed_variants() -> None:
    assert runner.VARIANTS == (
        runner.P3_D2282,
        runner.P3_D2283,
        runner.P3_D2263,
        runner.PFULL_D2283,
    )
    assert runner.JOBS_BY_GPU == {0: (runner.P3_D2282,), 1: (runner.P3_D2283,)}
    assert runner.H200_VARIANTS == (runner.P3_D2263, runner.PFULL_D2283)
    assert runner.SPECS[runner.P3_D2282].depth == (2, 2, 8, 2)
    assert runner.SPECS[runner.P3_D2283].depth == (2, 2, 8, 3)
    assert runner.SPECS[runner.P3_D2263].depth == (2, 2, 6, 3)


def test_interaction_models_preserve_k64_identity_carries() -> None:
    expected_parameters = {
        runner.P3_D2282: 1_630_180,
        runner.P3_D2283: 1_741_092,
        runner.P3_D2263: 1_519_268,
        runner.PFULL_D2283: 1_872_676,
    }
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        runner._assert_model(model, variant)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == expected_parameters[variant]
        )
        for name in runner.STAGE_NAMES[:3]:
            transition = getattr(model, name).augmented
            assert type(transition) is GatedPoleExcitationS2DTransition
            assert transition.carry_projection is None


def test_common_state_is_paired_to_reference() -> None:
    torch.manual_seed(501)
    reference = runner._build_spec(runner.REFERENCE_SPEC, runtime.model_config())
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        reference_state = reference.state_dict()
        model_state = model.state_dict()
        common = {
            name
            for name in reference_state.keys() & model_state.keys()
            if reference_state[name].shape == model_state[name].shape
        }
        assert common
        assert all(torch.equal(reference_state[name], model_state[name]) for name in common)
