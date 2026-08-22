from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import torch

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k64_p_allocation_d2262_imagenet100 as runner


def test_pole_allocation_campaign_partitions_qlab_and_h200() -> None:
    assert runner.JOBS_BY_GPU == {
        0: (runner.MAIN, runner.P2_128),
        1: (runner.P1_128, runner.P3_128),
    }
    assert runner.H200_VARIANTS == (runner.P4_128, runner.P3_192)
    assert set(runner.QLAB_VARIANTS).isdisjoint(runner.H200_VARIANTS)
    assert set(runner.VARIANTS) == set(runner.SPECS)


def test_every_candidate_fixes_k64_d2262_and_only_changes_p() -> None:
    expected_parameters = {
        runner.MAIN: 1_638_628,
        runner.P1_128: 1_671_524,
        runner.P2_128: 1_605_732,
        runner.P3_128: 1_539_940,
        runner.P4_128: 1_677_348,
        runner.P3_192: 1_737_316,
    }
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        runner._assert_model(model, variant)
        spec = runner.SPECS[variant]
        assert spec.excitation_modes == (64, 64, 64, 64)
        assert spec.depth == (2, 2, 6, 2)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == expected_parameters[variant]
        )
        assert model.classifier.q4_dim == 4 * spec.pole_modes[-1]
        for name in runner.STAGE_NAMES[:3]:
            transition = getattr(model, name).augmented
            assert type(transition) is GatedPoleExcitationS2DTransition
            assert transition.carry_projection is None


def test_pole_allocation_pairs_every_common_state_tensor() -> None:
    torch.manual_seed(501)
    reference = runner._build(runner.MAIN, runtime.model_config()).state_dict()
    for variant in runner.VARIANTS[1:]:
        torch.manual_seed(501)
        candidate = runner._build(variant, runtime.model_config()).state_dict()
        common = {
            name
            for name in reference.keys() & candidate.keys()
            if reference[name].shape == candidate[name].shape
        }
        assert common
        assert all(torch.equal(reference[name], candidate[name]) for name in common)
