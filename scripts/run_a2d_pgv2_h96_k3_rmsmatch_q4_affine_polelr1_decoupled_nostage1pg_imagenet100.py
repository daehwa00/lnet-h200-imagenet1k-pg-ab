#!/usr/bin/env python3
"""Train DecoupledInit after removing only the Stage-1 mode PG."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_pgv2_h96_k3_rmsmatch_q4_affine_polelr1_decoupled_init_imagenet100 as control
import torch
from torch import Tensor

from lnet.pac_grouped_path_cffn import grouped_cartesian_cffn
from lnet.pac_path_cffn import D4PathModeCombiner
from lnet.pac_phase_gated_transition import PhaseGatedModeResidualPathCollapse

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "PGv2-H96-K3-RMSMatch-Q4Affine-PoleLR1-DecoupledInit-NoStage1PG"
VARIANTS = (VARIANT,)
SEEDS = control.SEEDS


class PathOnlyCollapse(D4PathModeCombiner):
    """Retain the frozen runtime's trained path collapse without mode PG."""

    collapses_product_paths = True

    def __init__(self, source: PhaseGatedModeResidualPathCollapse) -> None:
        super().__init__()
        self.modes = source.modes
        self.path_count = source.path_count
        self.output_paths = source.output_paths
        self.input_modes = source.input_modes
        self.path_input = source.path_input
        self.path_output = source.path_output

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(
        self,
        source_real: Tensor,
        source_imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            message = "path-only Stage 1 requires NHW-path-mode inputs"
            raise ValueError(message)
        return grouped_cartesian_cffn(
            source_real,
            source_imag,
            input_projection=self.path_input,
            output_projection=self.path_output,
        )


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = control.base.local_reader.control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _remove_stage1_mode_pg(model: ComplexScanBackbone) -> None:
    mixer = model.stage1.quadrant_path_mode_combiner
    if not isinstance(mixer, PhaseGatedModeResidualPathCollapse):
        message = "DecoupledInit Stage 1 lost its phase-gated path-collapse contract"
        raise TypeError(message)
    model.stage1.quadrant_path_mode_combiner = PathOnlyCollapse(mixer)


def _assert_model(model: ComplexScanBackbone) -> None:
    control.base._assert_model(model)
    control._assert_initialization(model)
    if not isinstance(model.stage1.quadrant_path_mode_combiner, PathOnlyCollapse):
        message = "Stage 1 mode PG removal was not installed"
        raise TypeError(message)
    for name in ("stage2", "stage3"):
        mixer = getattr(model, name).quadrant_path_mode_combiner
        if not isinstance(mixer, PhaseGatedModeResidualPathCollapse):
            message = f"{name} mode PG changed in the Stage-1-only ablation"
            raise TypeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported DecoupledInit no-Stage1-PG variant: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    _remove_stage1_mode_pg(model)
    _configure_ramp()
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = "A2D-PGv2-H96-K3-RMSMatch-DecoupledPoleInit-NoStage1PG"
    payload["backbone"]["stage1_mode_processing"] = (
        "identity; PhaseGatedComplexFFN removed, established GWL 4-to-8-to-1 path collapse retained"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    ramp = control.base.local_reader.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = (
        "lnet.a2d.pgv2_h96.k3_rmsmatch.q4_affine.polelr1.decoupled_init.no_stage1_pg.imagenet100.v1"
    )
    payload["evidence_status"] = "untrained causal Stage-1 mode-PG removal"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact DecoupledInit control with only Stage-1 PhaseGatedComplexFFN "
            "deleted. The learned GWL 4-to-8-to-1 path collapse and every reader, "
            "pole, later PG, transition, Q4 descriptor, and affine head are unchanged."
        )
    }
    payload["source_sha256"]["no_stage1_pg_runner"] = ramp.heads.harness._digest(Path(__file__))
    return payload


def main() -> None:
    _configure_ramp()
    ramp = control.base.local_reader.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    harness = source.heads.harness
    source.heads.VARIANTS = VARIANTS
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ramp.PoleModelConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=control.base._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=(control.base.local_reader.control.control._wandb_model_metrics),
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
