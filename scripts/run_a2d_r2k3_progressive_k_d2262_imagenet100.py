#!/usr/bin/env python3
"""Train stage-wise excitation-width candidates on the D2262/P160 backbone."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# pyright: reportUnusedFunction=false
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_capacity_factory as capacity
import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as control
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as factorial
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import run_a2d_r2k3_uniform_prenorm_gated_postfusion_imagenet100 as uniform
import torch

from lnet.pac_complex_layers import ComplexLinear, identity_complex_linear_
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_gated_post_fusion import (
    GatedComplexPostFusion,
    GatedPoleExcitationS2DTransition,
)
from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.complex_scan_stage import ComplexScanStage


K14_160 = "K128-128-160-128-P160x4-D2262-WLPost15K-PathH4-LMP"
K14_192 = "K128-128-192-128-P160x4-D2262-WLPost15K-PathH4-LMP"
K28_14_160 = "K128-160-160-128-P160x4-D2262-WLPost15K-PathH4-LMP"
K56_96 = "K96-128-160-128-P160x4-D2262-WLPost15K-PathH4-LMP"
VARIANTS = (K14_160, K14_192, K28_14_160, K56_96)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {
    0: (K14_160, K28_14_160),
    1: (K14_192, K56_96),
}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = factorial.RESOLUTIONS
STAGE_NAMES = runtime.STAGE_NAMES
SameResolutionFactorialBackbone = factorial.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = 1.5
POLE_MODES = (160, 160, 160, 160)
EXTRA_BLOCKS = (0, 0, 4, 0)
SPECS = {
    K14_160: allocation.StageAllocationSpec(
        (128, 128, 160, 128),
        POLE_MODES,
        extra_blocks=EXTRA_BLOCKS,
        family="progressive_k",
    ),
    K14_192: allocation.StageAllocationSpec(
        (128, 128, 192, 128),
        POLE_MODES,
        extra_blocks=EXTRA_BLOCKS,
        family="progressive_k",
    ),
    K28_14_160: allocation.StageAllocationSpec(
        (128, 160, 160, 128),
        POLE_MODES,
        extra_blocks=EXTRA_BLOCKS,
        family="progressive_k",
    ),
    K56_96: allocation.StageAllocationSpec(
        (96, 128, 160, 128),
        POLE_MODES,
        extra_blocks=EXTRA_BLOCKS,
        family="progressive_k",
    ),
}


def _capacity_spec_for(
    spec: allocation.StageAllocationSpec,
) -> capacity.CapacitySpec:
    return capacity.CapacitySpec(
        spec.excitation_modes,
        spec.pole_modes,
        post_hidden_ratio=POST_HIDDEN_RATIO,
    )


def _capacity_spec(variant: str) -> capacity.CapacitySpec:
    return _capacity_spec_for(SPECS[variant])


def _templates(model: ComplexScanBackbone) -> tuple[ComplexScanStage, ...]:
    return tuple(cast("ComplexScanStage", getattr(model, name)) for name in STAGE_NAMES)


def _make_block(
    spec: allocation.StageAllocationSpec,
    capacity_spec: capacity.CapacitySpec,
    stage_index: int,
    template: ComplexScanStage,
    *,
    seed_offset: int = 0,
) -> SameResolutionPoleScanBlock:
    return factorial._make_block(
        width=spec.excitation_modes[stage_index],
        pole_modes=spec.pole_modes[stage_index],
        resolution=RESOLUTIONS[stage_index],
        pole_template=template,
        post_hidden=capacity_spec.post_hidden_modes[stage_index],
        seed_offset=seed_offset,
    )


def _materialize_square_memory_projection(
    owner: GatedPoleExcitationS2DTransition | SameResolutionPoleScanBlock,
) -> None:
    if owner.memory_projection is not None:
        return
    if isinstance(owner, GatedPoleExcitationS2DTransition):
        input_modes, output_modes = owner.input_modes, owner.output_modes
    else:
        input_modes, output_modes = owner.pole_modes, owner.modes
    if input_modes != output_modes:
        raise RuntimeError("only a square memory projection may be initialized as identity")
    reference = next(owner.parameters())
    with torch.random.fork_rng(devices=[]):
        projection = ComplexLinear(input_modes, output_modes).to(
            device=reference.device,
            dtype=reference.dtype,
        )
    identity_complex_linear_(projection)
    owner.memory_projection = projection


def _build_spec_model(
    spec: allocation.StageAllocationSpec,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    capacity_spec = _capacity_spec_for(spec)
    model = capacity._build_spec(capacity_spec, config)
    uniform._install_combined_transition(model)
    templates = _templates(model)
    blocks = {
        resolution: _make_block(spec, capacity_spec, index, templates[index])
        for index, resolution in enumerate(RESOLUTIONS)
    }
    model = factorial.SameResolutionFactorialBackbone(model, blocks)
    control._install_path_h4(model)
    extras: dict[int, tuple[SameResolutionPoleScanBlock, ...]] = {}
    for stage_index, (resolution, count) in enumerate(
        zip(RESOLUTIONS, spec.extra_blocks, strict=True)
    ):
        repeated = tuple(
            _make_block(
                spec,
                capacity_spec,
                stage_index,
                templates[stage_index],
                seed_offset=allocation.EXTRA_BLOCK_SEED_STRIDE * (repeat + 1),
            )
            for repeat in range(count)
        )
        for block in repeated:
            block.path_collapse = PathOnlyCollapse(
                spec.pole_modes[stage_index],
                path_hidden=control.PATH_HIDDEN,
            )
        if repeated:
            extras[resolution] = repeated
    model = allocation.RepeatedSameResolutionBackbone(model, extras)
    extra_blocks = tuple(
        block
        for group in model.extra_same_resolution_blocks.values()
        for block in cast("nn.ModuleList", group)
    )
    owners = (
        *(getattr(model, name).augmented for name in STAGE_NAMES[:3]),
        *model.same_resolution_blocks.values(),
        *extra_blocks,
    )
    for owner in owners:
        _materialize_square_memory_projection(owner)
    return model


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported progressive-K D2262 variant: {variant}") from error
    model = _build_spec_model(spec, config)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model_for_spec(model, spec, variant)
    return model


def _assert_path(collapse: nn.Module, modes: int, label: str) -> None:
    if (
        type(collapse) is not PathOnlyCollapse
        or collapse.modes != modes
        or collapse.path_input.output_paths != control.PATH_HIDDEN
        or collapse.path_output.input_paths != control.PATH_HIDDEN
        or collapse.output_paths != 1
    ):
        raise RuntimeError(f"{label} changed its PathH4 contract")


def _valid_memory_projection(
    projection: nn.Module | None,
    input_modes: int,
    output_modes: int,
) -> bool:
    return isinstance(projection, ComplexLinear) or (
        projection is None and input_modes == output_modes
    )


def _assert_block(
    block: nn.Module,
    *,
    excitation_modes: int,
    pole_modes: int,
    post_hidden: int,
    label: str,
) -> None:
    if (
        type(block) is not SameResolutionPoleScanBlock
        or block.modes != excitation_modes
        or block.pole_modes != pole_modes
        or block.reader.input_modes != excitation_modes
        or block.reader.output_modes != pole_modes
        or block.reader.rank != capacity.READER_RANK
        or block.reader.kernel_size != capacity.KERNEL_SIZE
        or not _valid_memory_projection(
            block.memory_projection,
            block.pole_modes,
            block.modes,
        )
        or type(block.post_fusion) is not GatedComplexPostFusion
        or block.post_fusion.modes != excitation_modes
        or block.post_fusion.hidden_modes != post_hidden
    ):
        raise RuntimeError(f"{label} changed its progressive-K full-block contract")
    _assert_path(block.path_collapse, pole_modes, label)


def _assert_model_for_spec(
    model: ComplexScanBackbone,
    spec: allocation.StageAllocationSpec,
    label: str,
) -> None:
    capacity_spec = _capacity_spec_for(spec)
    if (
        model.descriptor_dim != capacity_spec.descriptor_dim
        or model.classifier.input_dim != capacity_spec.q4_dim
    ):
        raise RuntimeError(f"{label} changed its Raw-Q contract")

    stages = tuple(getattr(model, name) for name in STAGE_NAMES)
    for index, (name, stage) in enumerate(zip(STAGE_NAMES, stages, strict=True)):
        reader = stage.pole_input_projection
        if (
            stage.input_modes != spec.excitation_modes[index]
            or stage.modes != spec.pole_modes[index]
            or type(reader) is not FactorizedComplexConv2dReader
            or reader.input_modes != spec.excitation_modes[index]
            or reader.output_modes != spec.pole_modes[index]
        ):
            raise RuntimeError(f"{label}/{name} changed its K-to-P160 reader")
        if name == "terminal":
            if stage.output_modes is not None:
                raise RuntimeError(f"{label} terminal unexpectedly emits an excitation")
            continue
        transition = stage.augmented
        if (
            not isinstance(transition, GatedPoleExcitationS2DTransition)
            or transition.excitation_modes != spec.excitation_modes[index]
            or transition.output_modes != spec.excitation_modes[index + 1]
            or transition.post_hidden != capacity_spec.post_hidden_modes[index + 1]
            or not _valid_memory_projection(
                transition.memory_projection,
                transition.input_modes,
                transition.output_modes,
            )
            or type(transition.post_fusion) is not GatedComplexPostFusion
        ):
            raise RuntimeError(f"{label}/{name} changed its PF1.5K transition")
        _assert_path(stage.quadrant_path_mode_combiner, spec.pole_modes[index], name)

    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    if set(blocks) != {str(resolution) for resolution in RESOLUTIONS}:
        raise RuntimeError(f"{label} lost a canonical same-resolution block")
    for index, resolution in enumerate(RESOLUTIONS):
        _assert_block(
            blocks[str(resolution)],
            excitation_modes=spec.excitation_modes[index],
            pole_modes=spec.pole_modes[index],
            post_hidden=capacity_spec.post_hidden_modes[index],
            label=f"{label}/SR{resolution}",
        )

    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    expected_extras = {
        str(resolution): count
        for resolution, count in zip(RESOLUTIONS, spec.extra_blocks, strict=True)
        if count
    }
    if {name: len(blocks) for name, blocks in extras.items()} != expected_extras:
        raise RuntimeError(f"{label} changed its repeated same-resolution depth")
    for stage_index, resolution in enumerate(RESOLUTIONS):
        key = str(resolution)
        if key not in extras:
            continue
        for repeat, block in enumerate(extras[key]):
            _assert_block(
                block,
                excitation_modes=spec.excitation_modes[stage_index],
                pole_modes=spec.pole_modes[stage_index],
                post_hidden=capacity_spec.post_hidden_modes[stage_index],
                label=f"{label}/ExtraSR{resolution}.{repeat}",
            )


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    _assert_model_for_spec(model, SPECS[variant], variant)


def _variant_config_for_spec(
    variant: str,
    spec: allocation.StageAllocationSpec,
) -> dict[str, Any]:
    capacity_spec = _capacity_spec_for(spec)
    payload = capacity._variant_config_for_spec(variant, capacity_spec)
    payload["backbone"]["pole_input"]["normalization"] = (
        "stage1-3 learned CRMSNorm before rank-2 K3; terminal token RMSMatch"
    )
    payload["backbone"]["transition"].update(
        {
            "path_collapse": "shared per-mode GWL 4-to-4-to-1 with Cartesian SiLU",
            "depth_by_resolution": dict(zip(map(str, RESOLUTIONS), spec.depth, strict=True)),
            "post_hidden_modes_by_resolution": dict(
                zip(map(str, RESOLUTIONS), capacity_spec.post_hidden_modes, strict=True)
            ),
        }
    )
    payload["backbone"]["same_resolution"] = {
        "resolutions": list(RESOLUTIONS),
        "extra_full_blocks_by_resolution": dict(
            zip(map(str, RESOLUTIONS), spec.extra_blocks, strict=True)
        ),
        "carry": "identity excitation",
        "post_hidden_ratio": POST_HIDDEN_RATIO,
    }
    payload["experiment"] = {
        "family": spec.family,
        "control": "K128-P160x4-D2262-FullSR14x4",
        "question": "which spatial stages need persistent excitation width beyond K128",
        "selection_role": "single-seed stage-localization and width-dose screen",
        "square_memory_projection": ("learned strict ComplexLinear initialized to exact identity"),
    }
    return payload


def _variant_config(variant: str) -> dict[str, Any]:
    return _variant_config_for_spec(variant, SPECS[variant])


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    variant_configs = {variant: _variant_config(variant) for variant in VARIANTS}
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload = campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="progressive_k_d2262_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.progressive_k_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required for all four variants",
        variant_configs=variant_configs,
        architectures={
            variant: (
                f"D2262/P160 PathH4 backbone with K={list(SPECS[variant].excitation_modes)} "
                "and stage-local Hpost=1.5K; all other model and recipe settings fixed."
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={
            "K128_P160_D2262": {
                "variant": "K128-P160x4-D2262-FullSR14x4",
                "parameters": 4_713_444,
            }
        },
    )
    payload["jobs_by_gpu"] = {str(gpu): list(variants) for gpu, variants in JOBS_BY_GPU.items()}
    return payload


def main() -> None:
    runtime.run(
        variants=VARIANTS,
        seeds=SEEDS,
        build_model=_build,
        contract=_contract,
    )


if __name__ == "__main__":
    main()
