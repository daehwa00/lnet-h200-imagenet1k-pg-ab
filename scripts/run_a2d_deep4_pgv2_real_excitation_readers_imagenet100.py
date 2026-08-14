#!/usr/bin/env python3
"""Compare six real-carrier-to-complex scan readers on one fixed PGv2 backbone."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_local_reader_imagenet100 as local_reader
import torch
from torch import Tensor, nn

from lnet.pac_phase_gated_transition import (
    AveragePoolMagnitudeGateTransition,
    PathPhaseGatedCollapse,
)
from lnet.pac_real_excitation_reader import (
    READER_VARIANTS,
    ContentDepthwiseQuadratureReader,
    ContentPointwiseQuadratureReader,
    DualFullK3Reader,
    FixedContrastQuadratureReader,
    JustInTimeComplexK3Reader,
    ReaderVariant,
    RealOnlyK3Reader,
    RMSMatchedRealExcitationReader,
    build_real_excitation_reader,
)

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage
    from lnet.pac_mean_one_magnitude_gate import MeanOneMagnitudeGate
    from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN


VARIANTS = READER_VARIANTS
VARIANT = VARIANTS[0]
SEEDS = local_reader.SEEDS
P = local_reader.P
KERNEL_SIZE = 3
REAL_EXCITATION_WIDTH = 2 * P
DESCRIPTOR_DIM = 4 * 4 * P

_READER_TYPES: dict[str, type[RMSMatchedRealExcitationReader]] = {
    "R0_JIT_COMPLEX_K3": JustInTimeComplexK3Reader,
    "R1_REAL_U": RealOnlyK3Reader,
    "R2_DUAL_FULL_K3": DualFullK3Reader,
    "R3_CONTENT_DWQ": ContentDepthwiseQuadratureReader,
    "R4_FIXED_CONTRAST_Q": FixedContrastQuadratureReader,
    "R5_CONTENT_PWQ": ContentPointwiseQuadratureReader,
}

_READER_CONTRACTS: dict[str, dict[str, Any]] = {
    "R0_JIT_COMPLEX_K3": {
        "measurement": "reinterpret real carrier halves as complex, strict-complex K3 96-to-96",
        "initialization": "identity-center complex kernel; zero spatial residual",
        "learnable_parameters_per_stage": 2 * P * P * KERNEL_SIZE**2,
        "role": "persistent-complex-philosophy control with just-in-time complexification",
    },
    "R1_REAL_U": {
        "measurement": "real K3 192-to-96 content response, with an exactly zero imaginary drive",
        "initialization": "orthonormal-row center matrix; zero neighbors",
        "learnable_parameters_per_stage": REAL_EXCITATION_WIDTH * P * KERNEL_SIZE**2,
        "role": "lower bound where the pole recurrence creates the first complex dynamics",
    },
    "R2_DUAL_FULL_K3": {
        "measurement": "unconstrained real K3 192-to-192, split into real and imaginary drives",
        "initialization": "identity center matrix; zero neighbors",
        "learnable_parameters_per_stage": REAL_EXCITATION_WIDTH**2 * KERNEL_SIZE**2,
        "role": "free complex-measurement upper bound",
    },
    "R3_CONTENT_DWQ": {
        "measurement": "real K3 192-to-96 content plus depthwise K3 spatial quadrature",
        "initialization": "orthonormal-row content center; zero neighbors and zero quadrature",
        "learnable_parameters_per_stage": (
            REAL_EXCITATION_WIDTH * P * KERNEL_SIZE**2 + P * KERNEL_SIZE**2
        ),
        "role": "cheap learned spatial-phase candidate",
    },
    "R4_FIXED_CONTRAST_Q": {
        "measurement": "real K3 content plus fixed content-minus-3x3-average contrast",
        "initialization": "orthonormal-row content center; zero neighbors",
        "learnable_parameters_per_stage": REAL_EXCITATION_WIDTH * P * KERNEL_SIZE**2,
        "role": "minimal structured local-phase candidate",
    },
    "R5_CONTENT_PWQ": {
        "measurement": "real K3 content plus pointwise 96-to-96 channel quadrature",
        "initialization": "orthonormal-row content center; zero neighbors and zero quadrature",
        "learnable_parameters_per_stage": (
            REAL_EXCITATION_WIDTH * P * KERNEL_SIZE**2 + P * P
        ),
        "role": "control separating spatial quadrature from channel coding",
    },
}


def _configure_ramp() -> None:
    local_reader._configure_ramp()
    ramp = local_reader.control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _install_real_excitation_readers(
    model: ComplexScanBackbone,
    variant: ReaderVariant,
) -> None:
    # One fork preserves the experiment RNG while deliberately giving the four
    # orthogonally initialized stage readers distinct matrices.
    with torch.random.fork_rng(devices=[]):
        for name in ("stage1", "stage2", "stage3", "terminal"):
            stage = cast("ComplexScanStage", getattr(model, name))
            if stage.modes != P:
                message = f"{name} changed the uniform real-excitation width"
                raise RuntimeError(message)
            stage.pole_input_projection = build_real_excitation_reader(
                variant,
                P,
                kernel_size=KERNEL_SIZE,
            )


def _assert_model(  # noqa: C901, PLR0912
    model: ComplexScanBackbone,
    variant: str,
) -> None:
    if variant not in VARIANTS:
        message = f"unsupported real-excitation reader variant: {variant}"
        raise ValueError(message)
    if model.descriptor_dim != DESCRIPTOR_DIM:
        message = "real-excitation campaign changed the four-stage Q1536 descriptor"
        raise RuntimeError(message)
    if not isinstance(model.input_norm, nn.RMSNorm):
        message = "real-excitation campaign removed the locked stem RMSNorm"
        raise TypeError(message)
    if (
        model.analysis is None
        or model.analysis.in_features != REAL_EXCITATION_WIDTH
        or model.analysis.out_features != REAL_EXCITATION_WIDTH
    ):
        message = "real-excitation campaign changed the orthogonal 192-to-192 interface"
        raise RuntimeError(message)

    expected_reader = _READER_TYPES[variant]
    for name in ("stage1", "stage2", "stage3", "terminal"):
        stage = cast("ComplexScanStage", getattr(model, name))
        reader = stage.pole_input_projection
        if type(reader) is not expected_reader:
            message = f"{name} is missing the declared {variant} reader"
            raise TypeError(message)
        active = cast("RMSMatchedRealExcitationReader", reader)
        if (
            active.modes != P
            or active.input_modes != REAL_EXCITATION_WIDTH
            or active.output_modes != P
            or active.diagnostics_enabled
        ):
            message = f"{name} changed the shared RMS-matched reader contract"
            raise RuntimeError(message)

    for name in ("stage1", "stage2", "stage3"):
        stage = cast("ComplexScanStage", getattr(model, name))
        mixer = stage.quadrant_path_mode_combiner
        if not isinstance(mixer, PathPhaseGatedCollapse):
            message = f"{name} lost the locked GWL-PathPG-GWL transition"
            raise TypeError(message)
        if (
            mixer.path_input.input_paths != 4
            or mixer.path_input.output_paths != local_reader.control.PATH_HIDDEN
            or mixer.mode.modes != P
            or mixer.mode.hidden_modes != local_reader.control.MODE_HIDDEN
            or mixer.path_output.input_paths != local_reader.control.PATH_HIDDEN
            or mixer.path_output.output_paths != 1
            or mixer.apply_cartesian_silu
        ):
            message = f"{name} changed the shared Path-PG contract"
            raise RuntimeError(message)
        transition = stage.augmented
        if type(transition) is not AveragePoolMagnitudeGateTransition:
            message = f"{name} lost the locked average-carry pure magnitude gate"
            raise TypeError(message)
        if (
            transition.carry_input_modes != 4 * P
            or transition.norm.modes != P
            or transition.gate.modes != P
        ):
            message = f"{name} changed the real-carrier re-excitation contract"
            raise RuntimeError(message)

    terminal = model.terminal
    if terminal.output_modes is not None or terminal.quadrant_path_mode_combiner is not None:
        message = "real-excitation campaign changed the terminal direct-Q stage"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant not in VARIANTS:
        message = f"unsupported real-excitation reader variant: {variant}"
        raise ValueError(message)
    model = local_reader._build(local_reader.AVG_PURE_GATE_VARIANT, config)
    _install_real_excitation_readers(model, variant)
    _configure_ramp()
    _assert_model(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    payload = deepcopy(local_reader._variant_config(local_reader.AVG_PURE_GATE_VARIANT))
    backbone = payload["backbone"]
    backbone["name"] = f"A2D-PGv2-H96-RealExcitation-{variant}"
    backbone["excitation"] = {
        "storage": (
            f"one real {REAL_EXCITATION_WIDTH}-channel carrier stored as two "
            f"{P}-wide halves"
        ),
        "interface": "stem RMSNorm then orthogonal real Linear 192-to-192",
        "complex_semantics": "introduced only by the stage reader output",
    }
    backbone["pole_input"] = {
        **_READER_CONTRACTS[variant],
        "input": f"real {REAL_EXCITATION_WIDTH}",
        "output": f"complex {P}",
        "bias": False,
        "rms_contract": "per-token RMS(complex drive)=RMS(full real excitation)",
        "scope": "the same reader family is used at Stage 1-3 and terminal",
    }
    backbone["stage_transition"] = {
        "scan": "associative D4 product scan plus endpoint 2x coarsening",
        "direct_q": f"raw normalized D4 memory energy, {4 * P} coordinates per stage",
        "path": "GWL 4-to-8, shared residual PG96, identity activation, GWL 8-to-1",
        "memory_carrier": "lossless concatenation [Re(M), Im(M)]",
        "carry": "fixed AvgPool2x2 of the original real excitation carrier",
        "merge": "X=R(M)+C",
        "refinement": "CRMSNorm then mean-one magnitude gate; E_next=X*[g,g]",
    }
    return payload


def _selected_variants(args: Namespace) -> tuple[str, ...]:
    requested = tuple(getattr(args, "variants", ()))
    return requested or VARIANTS


def _contract(args: Namespace) -> dict[str, Any]:
    payload = local_reader.control._contract(args)
    selected = _selected_variants(args)
    ramp = local_reader.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    models = {variant: _build(variant, config) for variant in selected}
    payload["schema"] = "lnet.a2d.pgv2_h96.real_excitation_readers.imagenet100.v1"
    payload["evidence_status"] = "controlled real-excitation complex-reader comparison"
    payload["variants"] = list(selected)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {
        variant: _variant_config(variant) for variant in selected
    }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["reader_parameter_counts"] = {
        variant: {
            "per_stage": _READER_CONTRACTS[variant]["learnable_parameters_per_stage"],
            "four_stages": sum(
                parameter.numel()
                for name in ("stage1", "stage2", "stage3", "terminal")
                for parameter in getattr(models[variant], name).pole_input_projection.parameters()
            ),
        }
        for variant in selected
    }
    payload["controlled_architecture"] = {
        "stem": "unchanged Conv3-32-192, residual real mixer, RMSNorm192, orthogonal Linear192",
        "persistent_excitation": "real192 at 56, 28, 14, and 7 spatial resolutions",
        "scan_drive": "reader-specific complex96 with identical tokenwise RMSMatch",
        "memory": "D4 complex96 with direct Q384 at each of four stages",
        "transition": "Path GWL4-8, shared PG96, GWL8-1, AvgPool carry, pure MagGate",
        "descriptor_and_head": "Q1536 and the established Fusion plus affine auxiliary head",
        "recipe": "optimizer, scheduler, augmentation, precision, and checkpointing unchanged",
    }
    payload["required_diagnostics"] = {
        "reader": "per-stage RMS(Re u), RMS(Im u), ratio, and phase circular variance",
        "q": "one fixed validation batch variance for Q1-Q4",
        "transition": "memory/carry RMS ratio, Path-PG gate/update, final MagGate statistics",
    }
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["real_excitation_reader"] = digest(
        Path("src/lnet/pac_real_excitation_reader.py")
    )
    payload["source_sha256"]["real_excitation_reader_runner"] = digest(Path(__file__))
    return payload


def _set_reader_diagnostics(model: nn.Module, *, enabled: bool) -> None:
    for name in ("stage1", "stage2", "stage3", "terminal"):
        reader = getattr(model, name).pole_input_projection
        if not isinstance(reader, RMSMatchedRealExcitationReader):
            message = f"{name} lost its real-excitation reader before diagnostics"
            raise TypeError(message)
        reader.set_diagnostics_enabled(enabled=enabled)


def _sampled_real_carrier_rms(first: Tensor, second: Tensor) -> Tensor:
    sampled_first = first.detach().reshape(-1, first.shape[-1])[:4096].float()
    sampled_second = second.detach().reshape(-1, second.shape[-1])[:4096].float()
    energy = torch.cat((sampled_first, sampled_second), dim=-1).square().mean()
    return torch.sqrt(energy)


def _merge_hook(
    metrics: dict[str, float],
    stage_index: int,
) -> Callable[[nn.Module, tuple[object, ...], object], None]:
    @torch.no_grad()
    def record(module: nn.Module, arguments: tuple[object, ...], _output: object) -> None:
        if len(arguments) != 4 or not all(isinstance(value, Tensor) for value in arguments):
            message = "real-excitation merge diagnostic received an invalid call"
            raise RuntimeError(message)
        transition = cast("AveragePoolMagnitudeGateTransition", module)
        memory_real, memory_imag, carry_real, carry_imag = cast(
            "tuple[Tensor, Tensor, Tensor, Tensor]", arguments
        )
        reduced_carry = transition._carry(carry_real, carry_imag)
        memory_rms = _sampled_real_carrier_rms(memory_real, memory_imag)
        carry_rms = _sampled_real_carrier_rms(*reduced_carry)
        prefix = f"merge/stage{stage_index}"
        metrics[f"{prefix}/memory_rms"] = float(memory_rms)
        metrics[f"{prefix}/carry_rms"] = float(carry_rms)
        metrics[f"{prefix}/memory_over_carry_rms"] = float(
            memory_rms / carry_rms.clamp_min(1.0e-12)
        )

    return record


@torch.inference_mode()
def _diagnostic_validation_batch(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool,
) -> None:
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device, non_blocking=True)
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    merge_metrics: dict[str, float] = {}
    handles = [
        getattr(model, name).augmented.register_forward_hook(
            _merge_hook(merge_metrics, index)
        )
        for index, name in enumerate(("stage1", "stage2", "stage3"), start=1)
    ]
    _set_reader_diagnostics(model, enabled=True)
    try:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            output = model(inputs)
    finally:
        _set_reader_diagnostics(model, enabled=False)
        for handle in handles:
            handle.remove()
    if not isinstance(output, tuple) or len(output) != 5:
        message = "real-excitation diagnostic pass requires the five-output Q classifier"
        raise RuntimeError(message)
    descriptor = output[4].detach().float()
    if descriptor.shape[-1] != DESCRIPTOR_DIM:
        message = "real-excitation diagnostic pass lost the Q1536 layout"
        raise RuntimeError(message)
    metrics = dict(merge_metrics)
    for index, part in enumerate(descriptor.split(4 * P, dim=-1), start=1):
        metrics[f"q/stage{index}/variance"] = float(part.var(unbiased=False))
        metrics[f"q/stage{index}/rms"] = float(part.square().mean().sqrt())
    cast("Any", model)._latest_real_excitation_validation = metrics


def _evaluate(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    source = local_reader.control.control.control.stemres.uniform.base
    result = source.heads._evaluate(
        model,
        runtime,
        loader,
        device,
        precision=precision,
        channels_last=channels_last,
    )
    _diagnostic_validation_batch(
        model,
        loader,
        device,
        precision=precision,
        channels_last=channels_last,
    )
    return result


def _module_metrics(
    prefix: str,
    module: PhaseGatedComplexFFN | MeanOneMagnitudeGate,
) -> dict[str, float]:
    values = module.diagnostic_metrics()
    values.update(module.gradient_metrics())
    return {f"{prefix}/{name}": float(value) for name, value in values.items()}


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = local_reader.control.control._wandb_model_metrics(model)
    metrics.update(
        {
            f"validation/{name}": float(value)
            for name, value in getattr(
                model,
                "_latest_real_excitation_validation",
                {},
            ).items()
        }
    )
    for index, name in enumerate(("stage1", "stage2", "stage3", "terminal"), start=1):
        reader = getattr(model, name).pole_input_projection
        if not isinstance(reader, RMSMatchedRealExcitationReader):
            message = f"{name} lost its real-excitation reader before W&B logging"
            raise TypeError(message)
        metrics.update(
            {
                f"reader/stage{index}_reader_{metric}": value
                for metric, value in reader.diagnostic_metrics().items()
            }
        )
        metrics[f"reader/stage{index}_reader_updates"] = float(
            reader.diagnostic_updates
        )
        if name == "terminal":
            continue
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        if not isinstance(mixer, PathPhaseGatedCollapse):
            message = f"{name} lost its Path-PG before W&B logging"
            raise TypeError(message)
        if not isinstance(transition, AveragePoolMagnitudeGateTransition):
            message = f"{name} lost its final magnitude gate before W&B logging"
            raise TypeError(message)
        metrics.update(_module_metrics(f"path_pg/stage{index}", mixer.mode))
        metrics.update(_module_metrics(f"magnitude_gate/stage{index}", transition.gate))
    return metrics


def main() -> None:
    _configure_ramp()
    ramp = local_reader.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = ramp.backbone.a2d_base.residuals
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
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=_evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
