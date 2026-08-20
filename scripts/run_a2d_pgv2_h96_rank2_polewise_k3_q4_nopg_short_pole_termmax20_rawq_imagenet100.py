#!/usr/bin/env python3
"""Train the H96 ShortDamp rank-2 reader with raw directional Q."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_pgv2_h96_k3_q4_nopg_short_pole_termmax20_imagenet100 as control
import torch

from lnet.pac_complex_scan_reader import PackedComplexConv2dReader
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "H96-R2PW-PoleDWK3-Q4-RawQ-OrthInit-NoPG-ShortDamp-TMax20"
VARIANTS = (VARIANT,)
SEEDS = control.SEEDS
READER_RANK = 2
STAGE_NAMES = control.STAGE_NAMES


def _ramp() -> Any:
    return control.no_pg.control.base.local_reader.control.control.control.stemres.uniform.base


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = _ramp()
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _replace_readers(model: ComplexScanBackbone) -> None:
    # Keep reader construction from perturbing the controlled data/augmentation
    # RNG while still giving different stages and experiment seeds distinct
    # dormant rank components.
    with torch.random.fork_rng(devices=[]):
        for stage_name in STAGE_NAMES:
            stage = getattr(model, stage_name)
            old_reader = stage.pole_input_projection
            if (
                type(old_reader) is not PackedComplexConv2dReader
                or old_reader.input_modes != old_reader.output_modes
                or old_reader.kernel_size != 3
                or not old_reader.match_input_rms
            ):
                message = f"{stage_name} lost the exact H96 K3 RMSMatch reader control"
                raise TypeError(message)
            reader = FactorizedComplexConv2dReader(
                old_reader.input_modes,
                old_reader.output_modes,
                rank=READER_RANK,
                kernel_size=old_reader.kernel_size,
                variance_epsilon=old_reader.variance_epsilon,
                match_input_rms=old_reader.match_input_rms,
            )
            reader.initialize_orthogonal_()
            stage.pole_input_projection = reader


def _assert_model(model: ComplexScanBackbone) -> None:
    for stage_name in STAGE_NAMES:
        stage = getattr(model, stage_name)
        reader = stage.pole_input_projection
        if (
            type(reader) is not FactorizedComplexConv2dReader
            or reader.input_modes != 96
            or reader.output_modes != 96
            or reader.rank != READER_RANK
            or reader.kernel_size != 3
            or not reader.match_input_rms
        ):
            message = f"{stage_name} changed the rank-2 reader contract"
            raise RuntimeError(message)
        if hasattr(stage, "product_descriptor_basis"):
            message = f"{stage_name} restored a selectable descriptor basis"
            raise RuntimeError(message)
    if model.descriptor_dim != 4 * 4 * 96:
        message = "rank-2 RawQ model must expose four raw directions per stage"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported factorized-reader variant: {variant}"
        raise ValueError(message)
    model = control._build(control.SHORT_DAMP_VARIANT, config)
    _replace_readers(model)
    _configure_ramp()
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config(control.SHORT_DAMP_VARIANT))
    payload["backbone"]["name"] = "A2D-H96-Rank2-PW-PoleDWK3-RawQ-OrthInit-NoPG-ShortDamp-TMax20"
    payload["backbone"]["pole_input"] = {
        "operator": "strict-complex pointwise analysis followed by pole-wise complex K3",
        "factorization": "W[p,k,dy,dx] = sum_r D[p,r,dy,dx] * A[p,r,k]",
        "rank": READER_RANK,
        "shape": "96-to-(2x96) pointwise; 2 filters per pole; sum rank; 96 output poles",
        "initialization": (
            "identity rank plus active complex-orthogonal point/spatial residual ranks; "
            "residual amplitude is derived as input_modes**-0.5"
        ),
        "normalization": "exact unit energy of the synthesized full kernel per output pole",
        "activation_gain": "per-token shared real RMS(U)=RMS(E) matching",
        "scope": (
            "rank-2 reader plus raw directional Q; pole atlas, scan recurrence, NoPG "
            "transition, Q4 head, and recipe unchanged"
        ),
    }
    payload["backbone"]["descriptor"] = {
        "operator": "direct raw directional log-energy",
        "shape": "4 directions x 96 modes per stage; Q4 head selects terminal 384",
        "basis_transform": "none",
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    config = _ramp().PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.h96.rank2_pw_poledwk3.rawq.shortdamp_tmax20.imagenet100.v1"
    payload["evidence_status"] = "untrained orthogonally initialized raw-Q rank-2 candidate"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact H96-K3-Q4-NoPG-ShortDamp-TermMax20 control with only each full strict "
            "complex K3 reader replaced by a rank-2 strict-complex pointwise analysis and "
            "pole-wise K3 synthesis. All four readers retain RMSMatch and begin with active, "
            "mutually orthogonal rank components; Q reads the four scan directions directly "
            "without a basis transform."
        )
    }
    digest = _ramp().heads.harness._digest
    payload["source_sha256"]["factorized_complex_scan_reader"] = digest(
        Path("src/lnet/pac_factorized_complex_scan_reader.py")
    )
    payload["source_sha256"]["rank2_reader_runner"] = digest(Path(__file__))
    return payload


def _build_optimizer(model: torch.nn.Module, recipe: dict[str, Any]) -> torch.optim.Optimizer:
    return control._build_optimizer(model, recipe)


def main() -> None:
    _configure_ramp()
    ramp = _ramp()
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
            build_optimizer=_build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=control.no_pg.control.base.local_reader.control.control._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
