#!/usr/bin/env python3
"""Train calibrated uniform-P96 Deep4 with the mode-scaled residual stem."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_imagenet100 as uniform
import torch
from torch import nn
from torch.nn.utils import parametrize

from lnet.image_layers import LayerNorm2d, ModeScaledTwoConvStem, ResidualPreComplexMixer

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "D4-Cal-U96-Stem32"
VARIANTS = (VARIANT,)
SEEDS = uniform.SEEDS
P = uniform.MODES
STEM_HIDDEN_WIDTH = 32
_RAMP_BUILD = uniform.base._build


def _configure_ramp() -> None:
    """Retain the uniform-P96 backbone while assigning a distinct run identity."""
    uniform._configure_base()
    ramp = uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _assert_stem(
    model: nn.Module,
    modes: int = P,
    hidden_width: int = STEM_HIDDEN_WIDTH,
) -> None:
    if not isinstance(model.stem, ModeScaledTwoConvStem):
        message = "uniform-P96 StemRes is missing its mode-scaled convolution stem"
        raise TypeError(message)
    first, first_norm, first_gelu, second, second_norm, second_gelu = model.stem.layers
    if (
        not isinstance(first, nn.Conv2d)
        or first.in_channels != 3
        or first.out_channels != hidden_width
        or first.stride != (2, 2)
        or not isinstance(first_norm, LayerNorm2d)
        or not isinstance(first_gelu, nn.GELU)
        or not isinstance(second, nn.Conv2d)
        or second.in_channels != hidden_width
        or second.out_channels != 2 * modes
        or second.stride != (2, 2)
        or not isinstance(second_norm, LayerNorm2d)
        or not isinstance(second_gelu, nn.GELU)
    ):
        message = "uniform-P96 StemRes convolution contract changed"
        raise RuntimeError(message)
    if not isinstance(model.precomplex_fc, ResidualPreComplexMixer):
        message = "uniform-P96 StemRes is missing its residual real mixer"
        raise TypeError(message)
    if model.precomplex_fc.width != 2 * modes:
        message = "uniform-P96 StemRes mixer width changed"
        raise RuntimeError(message)
    if not isinstance(model.input_norm, nn.RMSNorm) or model.input_norm.normalized_shape != (
        2 * modes,
    ):
        message = "uniform-P96 StemRes requires RMSNorm before modal analysis"
        raise TypeError(message)
    if (
        not isinstance(model.analysis, nn.Linear)
        or model.analysis.in_features != 2 * modes
        or model.analysis.out_features != 2 * modes
        or model.analysis.bias is not None
        or not parametrize.is_parametrized(model.analysis, "weight")
    ):
        message = "uniform-P96 StemRes requires the retained orthogonal interface"
        raise TypeError(message)


def _build(variant: str, config: uniform.base.PoleModelConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported uniform-P96 StemRes variant: {variant}"
        raise ValueError(message)
    _configure_ramp()
    model = _RAMP_BUILD(variant, config)
    model.stem = ModeScaledTwoConvStem(
        P,
        model.config.stem_strides,
        hidden_width=STEM_HIDDEN_WIDTH,
    )
    model.precomplex_fc = ResidualPreComplexMixer(model.precomplex_fc)
    _assert_stem(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(uniform._variant_config())
    payload["backbone"]["name"] = "A2D-Calibrated-Product4-Uniform96-Stem32-FullOpt"
    payload["backbone"]["stem_width"] = 2 * P
    payload["backbone"]["stem"] = {
        "P": P,
        "convolutions": f"3_to_{STEM_HIDDEN_WIDTH}_s2_then_{STEM_HIDDEN_WIDTH}_to_{2 * P}_s2",
        "normalization": "LayerNorm2d_after_each_convolution",
        "activation": "GELU_after_each_convolution",
        "precomplex_mixer": f"residual_Linear{2 * P}_GELU_Linear{2 * P}",
        "interface_norm": f"RMSNorm{2 * P}",
        "complex_projection": f"orthogonal_Linear{2 * P}_to_{2 * P}_bias_false_then_split",
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = uniform._contract(args)
    config = uniform.base.PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.deep4_calibrated_uniform_p96_stem32.imagenet100.v1"
    payload["evidence_status"] = "untrained uniform-P96 Stem32 FullOpt candidate"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "The unchanged calibrated product-only uniform-P96 Deep4 model with "
            "a 3-to-32-to-192 Conv-LN-GELU stem, residual 192-wide real mixer, "
            "RMSNorm, and the retained square orthogonal complex interface."
        )
    }
    payload["source_sha256"]["a2d_deep4_calibrated_uniform_p96_stem32_runner"] = (
        uniform.base.heads.harness._digest(Path(__file__))
    )
    return payload


def main() -> None:
    _configure_ramp()
    ramp = uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = ramp.backbone.a2d_base.residuals
    harness = source.heads.harness
    source.heads.VARIANTS = VARIANTS
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    runner_bindings = getattr(harness, "runner_bindings", None)
    if callable(runner_bindings):
        harness.main(
            runner_bindings(
                variants=VARIANTS,
                seeds=SEEDS,
                model_config=ramp.PoleModelConfig,
                build_model=_build,
                contract=_contract,
                build_optimizer=residuals.optimizer_source._build_optimizer,
                prepare_model=source._prepare_model,
                train_epoch=source.structured._train_epoch,
                evaluate=source.heads._evaluate,
                wandb_model_metrics=ramp.backbone._wandb_model_metrics,
                summarize=source.heads._summarize,
            )
        )
        return

    harness.VARIANTS = VARIANTS
    harness.SEEDS = SEEDS
    harness.ImageNetNanoConfig = ramp.PoleModelConfig
    harness.build_imagenet_nano = _build
    harness._contract = _contract
    harness._build_optimizer = residuals.optimizer_source._build_optimizer
    harness._prepare_model = source._prepare_model
    harness._train_epoch = source.structured._train_epoch
    harness._evaluate = source.heads._evaluate
    harness._wandb_model_metrics = ramp.backbone._wandb_model_metrics
    harness._summarize = source.heads._summarize
    harness.main()


if __name__ == "__main__":
    main()
