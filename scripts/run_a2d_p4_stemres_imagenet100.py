#!/usr/bin/env python3
"""Train P4-Joint128 with an activated Conv stem and residual real MLP."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_p4_joint_imagenet100 as joint
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from lnet.complex_scan import ComplexScanConfig
from lnet.image_layers import CifarConvStem

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "P4-StemRes"
VARIANTS = (VARIANT,)
SEEDS = (501,)
P = 64
STEM_WIDTH = 2 * P
heads = joint.heads


class ActivatedTwoConvStem(nn.Module):
    """Conv-LN-GELU-Conv-LN-GELU followed by NCHW-to-NHWC."""

    def __init__(self, source: CifarConvStem) -> None:
        super().__init__()
        layers = tuple(source.layers.children())
        if len(layers) != 5 or not isinstance(layers[-1], nn.Module):
            message = "P4-StemRes requires the established five-layer Conv stem"
            raise TypeError(message)
        self.layers = nn.Sequential(*layers, nn.GELU())

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs).permute(0, 2, 3, 1)


class ResidualStemMLP(nn.Module):
    """Apply x + Linear-GELU-Linear in the real NHWC feature space."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        if not isinstance(source, nn.Sequential) or len(source) != 4:
            message = "P4-StemRes requires the established two-linear precomplex FC"
            raise TypeError(message)
        input_projection, activation, output_projection, tail = tuple(source.children())
        if (
            not isinstance(input_projection, nn.Linear)
            or not isinstance(activation, nn.GELU)
            or not isinstance(output_projection, nn.Linear)
            or not isinstance(tail, nn.Identity)
            or input_projection.in_features != STEM_WIDTH
            or input_projection.out_features != STEM_WIDTH
            or output_projection.in_features != STEM_WIDTH
            or output_projection.out_features != STEM_WIDTH
        ):
            message = "P4-StemRes precomplex FC dimensions changed"
            raise TypeError(message)
        self.input_projection = input_projection
        self.activation = activation
        self.output_projection = output_projection

    def forward(self, inputs: Tensor) -> Tensor:
        update = self.output_projection(self.activation(self.input_projection(inputs)))
        return inputs + update


def _assert_stem(model: nn.Module) -> None:
    if not isinstance(model.stem, ActivatedTwoConvStem):
        message = "P4-StemRes is missing the activated two-convolution stem"
        raise TypeError(message)
    layers = tuple(model.stem.layers.children())
    first, _, _, second, _, final_activation = layers
    if (
        not isinstance(first, nn.Conv2d)
        or first.in_channels != 3
        or first.out_channels != P // 2
        or first.stride != (2, 2)
        or not isinstance(second, nn.Conv2d)
        or second.in_channels != P // 2
        or second.out_channels != 2 * P
        or second.stride != (2, 2)
        or not isinstance(final_activation, nn.GELU)
    ):
        message = "P4-StemRes convolution contract changed"
        raise RuntimeError(message)
    if not isinstance(model.precomplex_fc, ResidualStemMLP):
        message = "P4-StemRes is missing its residual real MLP"
        raise TypeError(message)
    if not isinstance(model.input_norm, nn.RMSNorm):
        message = "P4-StemRes requires RMSNorm before modal analysis"
        raise TypeError(message)
    if (
        not isinstance(model.analysis, nn.Linear)
        or model.analysis.in_features != 2 * P
        or model.analysis.out_features != 2 * P
        or model.analysis.bias is not None
        or not parametrize.is_parametrized(model.analysis, "weight")
    ):
        message = "P4-StemRes requires a square bias-free orthogonal analysis map"
        raise TypeError(message)


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported P4 stem-residual variant: {variant}"
        raise ValueError(message)
    model = joint._build(joint.VARIANT, config)
    if not isinstance(model.stem, CifarConvStem):
        message = "P4-StemRes requires the normalized convolution stem"
        raise TypeError(message)
    model.stem = ActivatedTwoConvStem(model.stem)
    model.precomplex_fc = ResidualStemMLP(model.precomplex_fc)
    _assert_stem(model)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = joint._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    variant_config = deepcopy(payload["variant_configs"][joint.VARIANT])
    variant_config["backbone"]["stem"] = {
        "P": P,
        "convolutions": "3_to_32_s2_then_32_to_128_s2",
        "nonlinearity": "LN_GELU_after_each_convolution",
        "layout_handoff": "NCHW_to_NHWC",
        "real_residual_mlp": "128_to_128_GELU_128_with_identity_skip",
        "post_mlp_norm": "RMSNorm128",
        "modal_analysis": "orthogonal_128_to_128_bias_false",
    }
    payload.update(
        {
            "schema": "lnet.a2d.p4_stemres.imagenet100.v1",
            "evidence_status": "paired P4-Joint128 stem residual ablation",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
            "variant_configs": {VARIANT: variant_config},
            "parameter_counts": {
                VARIANT: sum(parameter.numel() for parameter in model.parameters())
            },
            "architecture": {
                VARIANT: (
                    "P4-Joint128 with Conv-LN-GELU-Conv-LN-GELU, NHWC residual "
                    "128-to-128-to-128 real MLP, RMSNorm, and the retained square "
                    "orthogonally parametrized 128-to-128 modal analysis map."
                )
            },
        }
    )
    payload["source_sha256"]["a2d_p4_stemres_runner"] = heads.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def main() -> None:
    calibrated = joint.p4.calibrated
    source = resaux_base
    residuals = a2d_base.residuals
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
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=calibrated.canonical8.fair_init.backbone._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
