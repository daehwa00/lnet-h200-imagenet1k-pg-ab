#!/usr/bin/env python3
"""Post-training factorial knockout audit for the ImageNet-100 D4-M64 model.

This script never updates checkpoint parameters.  It restores a fresh model for
each condition, applies structured masks in memory, optionally recalibrates the
descriptor standardizers without labels, and evaluates the unchanged classifier.
The resulting effects measure reliance of the trained M64 network; they are not
a substitute for training the missing P64/H96 and P48/H128 factorial cells.
"""

# ruff: noqa: ANN401, EM101, SLF001, T201, TRY003

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import run_a2d_deep4_backbone_variants_imagenet100 as m64_runner
import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import datasets

from lnet.complex_scan import (
    ComplexScanConfig,
    ComplexScanStage,
    FactorizedQuadrantPathModeCFFNCombiner,
    S2DPostFusionCFFNTransition,
    WidelyLinear,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


Condition = Literal[
    "full",
    "pole48",
    "cffn96",
    "pole48_cffn96",
    "mode96",
    "transition96",
    "post96",
    "pole48_s1",
    "pole48_s2",
    "pole48_s3",
    "pole48_s4",
]

ALL_CONDITIONS: tuple[Condition, ...] = (
    "pole48",
    "cffn96",
    "pole48_cffn96",
    "mode96",
    "transition96",
    "post96",
    "pole48_s1",
    "pole48_s2",
    "pole48_s3",
    "pole48_s4",
)
STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
TRAINED_MODES = 64
RETAINED_MODES = 48
TRAINED_HIDDEN = 128
RETAINED_HIDDEN = 96
TRAINED_EXPANDED = 256
RETAINED_EXPANDED = 192
ORIENTATIONS = 4


@dataclass(frozen=True, slots=True)
class MaskSpec:
    """One paired random intervention shared by the factorial cells."""

    repeat: int
    pole_keep: tuple[int, ...]
    mode_hidden_keep: tuple[tuple[int, ...], ...]
    transition_keep: tuple[tuple[int, ...], ...]
    transition_expanded_keep: tuple[tuple[int, ...], ...]
    post_hidden_keep: tuple[tuple[int, ...], ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--mask-seed", type=int, default=20260807)
    parser.add_argument(
        "--pole-mask-strategy",
        choices=("random-radial", "even-radial"),
        default="random-radial",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=0,
        help="Label-free validation batches used to refresh descriptor BN statistics.",
    )
    parser.add_argument(
        "--conditions",
        choices=ALL_CONDITIONS,
        nargs="+",
        default=list(ALL_CONDITIONS),
    )
    return parser.parse_args()


def _keep_indices(total: int, retained: int, generator: torch.Generator) -> tuple[int, ...]:
    if not 0 < retained <= total:
        raise ValueError("retained width must be in (0, total]")
    return tuple(sorted(torch.randperm(total, generator=generator)[:retained].tolist()))


def make_mask_spec(
    repeat: int,
    seed: int,
    pole_strategy: str = "random-radial",
) -> MaskSpec:
    """Construct paired masks without privileging channel order.

    Pole masking keeps twelve complete four-orientation radial groups out of
    sixteen.  This preserves the Gabor-atlas orientation balance while varying
    which learned radial bands are removed.
    """
    generator = torch.Generator().manual_seed(seed + repeat)
    if pole_strategy == "random-radial":
        radial_keep = _keep_indices(
            TRAINED_MODES // ORIENTATIONS,
            RETAINED_MODES // ORIENTATIONS,
            generator,
        )
    elif pole_strategy == "even-radial":
        radial_keep = tuple(
            torch.linspace(
                0,
                TRAINED_MODES // ORIENTATIONS - 1,
                RETAINED_MODES // ORIENTATIONS,
            )
            .round()
            .int()
            .tolist()
        )
    else:
        raise ValueError("unsupported pole mask strategy")
    pole_keep = tuple(
        level * ORIENTATIONS + orientation
        for level in radial_keep
        for orientation in range(ORIENTATIONS)
    )

    def stage_masks(total: int, retained: int) -> tuple[tuple[int, ...], ...]:
        return tuple(_keep_indices(total, retained, generator) for _ in range(3))

    return MaskSpec(
        repeat=repeat,
        pole_keep=pole_keep,
        mode_hidden_keep=stage_masks(TRAINED_HIDDEN, RETAINED_HIDDEN),
        transition_keep=stage_masks(TRAINED_HIDDEN, RETAINED_HIDDEN),
        transition_expanded_keep=stage_masks(TRAINED_EXPANDED, RETAINED_EXPANDED),
        post_hidden_keep=stage_masks(TRAINED_HIDDEN, RETAINED_HIDDEN),
    )


def _drop_indices(total: int, keep: Sequence[int]) -> Tensor:
    keep_set = set(keep)
    if (
        len(keep_set) != len(keep)
        or min(keep_set, default=0) < 0
        or max(keep_set, default=0) >= total
    ):
        raise ValueError("invalid structured keep-index set")
    return torch.tensor([index for index in range(total) if index not in keep_set])


def _projection_tensors(projection: WidelyLinear) -> Iterable[Tensor]:
    return (
        projection.weight_real,
        projection.weight_imag,
        projection.conjugate_real,
        projection.conjugate_imag,
    )


def zero_projection_outputs(projection: WidelyLinear, keep: Sequence[int]) -> None:
    """Disable affine output coordinates while preserving tensor shapes."""
    drop = _drop_indices(projection.output_modes, keep).to(projection.weight_real.device)
    with torch.no_grad():
        for weight in _projection_tensors(projection):
            weight.index_fill_(0, drop, 0.0)
        if projection.bias_real is not None:
            projection.bias_real.index_fill_(0, drop, 0.0)
        if projection.bias_imag is not None:
            projection.bias_imag.index_fill_(0, drop, 0.0)


def zero_projection_inputs(projection: WidelyLinear, keep: Sequence[int]) -> None:
    """Disconnect affine input coordinates while preserving tensor shapes."""
    drop = _drop_indices(projection.input_modes, keep).to(projection.weight_real.device)
    with torch.no_grad():
        for weight in _projection_tensors(projection):
            weight.index_fill_(1, drop, 0.0)


def _prune_hidden_pair(
    input_projection: WidelyLinear,
    output_projection: WidelyLinear,
    keep: Sequence[int],
) -> None:
    if input_projection.output_modes != output_projection.input_modes:
        raise ValueError("CFFN projection pair has incompatible hidden widths")
    zero_projection_outputs(input_projection, keep)
    zero_projection_inputs(output_projection, keep)


def _stage_modules(model: nn.Module) -> tuple[ComplexScanStage, ...]:
    stages = tuple(cast("ComplexScanStage", getattr(model, name)) for name in STAGE_NAMES)
    if any(not isinstance(stage, ComplexScanStage) for stage in stages):
        raise TypeError("M64 audit expected four complex pole stages")
    return stages


def apply_cffn_knockout(
    model: nn.Module,
    masks: MaskSpec,
    components: frozenset[str],
) -> None:
    """Apply in-memory H128-to-H96 structured pruning to selected components."""
    for stage_index, stage in enumerate(_stage_modules(model)[:3]):
        combiner = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        if not isinstance(combiner, FactorizedQuadrantPathModeCFFNCombiner):
            raise TypeError("M64 stage lost its factorized ModeCFFN")
        if not isinstance(transition, S2DPostFusionCFFNTransition):
            raise TypeError("M64 stage lost its PostCarry/PostFFN transition")

        if "mode" in components:
            _prune_hidden_pair(
                combiner.mode_input,
                combiner.mode_output,
                masks.mode_hidden_keep[stage_index],
            )

        if "transition" in components:
            state_keep = masks.transition_keep[stage_index]
            expanded_keep = masks.transition_expanded_keep[stage_index]
            zero_projection_outputs(transition.direction_mixer, state_keep)
            zero_projection_inputs(transition.ffn_input, state_keep)
            _prune_hidden_pair(transition.ffn_input, transition.ffn_output, expanded_keep)
            zero_projection_outputs(transition.ffn_output, state_keep)
            zero_projection_inputs(transition.output_projection, state_keep)
            drop = _drop_indices(transition.hidden_modes, state_keep).to(
                transition.layer_scale.device
            )
            with torch.no_grad():
                transition.layer_scale[drop].zero_()

        if "post" in components:
            _prune_hidden_pair(
                transition.post_ffn_input,
                transition.post_ffn_output,
                masks.post_hidden_keep[stage_index],
            )


def factorial_effects(
    full: float,
    pole48: float,
    cffn96: float,
    both: float,
) -> dict[str, float]:
    """Return accuracy-scale factorial effects and the explicit interaction."""
    return {
        "pole_main_effect": 0.5 * ((full - pole48) + (cffn96 - both)),
        "cffn_main_effect": 0.5 * ((full - cffn96) + (pole48 - both)),
        "interaction": full - pole48 - cffn96 + both,
    }


def _model_from_checkpoint(checkpoint: Path, device: torch.device) -> nn.Module:
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = m64_runner._build(m64_runner.UNIFORM_M64, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("variant") != m64_runner.UNIFORM_M64:
        raise ValueError("checkpoint is not the trained D4-M64 variant")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).to(memory_format=torch.channels_last).eval()


def _validation_loader(data_root: Path, batch_size: int, workers: int) -> DataLoader[Any]:
    _, evaluation_transform = harness._transforms()
    dataset = datasets.ImageFolder(data_root / "val", evaluation_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=harness.PREFETCH_FACTOR if workers > 0 else None,
    )


def _pole_descriptor_mask(
    keep: Sequence[int],
    device: torch.device,
    stage_indices: frozenset[int],
) -> Tensor:
    stage_mask = torch.zeros(TRAINED_MODES, device=device)
    stage_mask[list(keep)] = 1.0
    full_mask = torch.ones(TRAINED_MODES, device=device)
    return torch.cat(
        tuple(
            stage_mask if stage_index in stage_indices else full_mask
            for stage_index in range(len(STAGE_NAMES))
            for _direction in range(4)
        )
    )


def _install_pole_hooks(
    stack: ExitStack,
    model: nn.Module,
    keep: Sequence[int],
    device: torch.device,
    stage_indices: frozenset[int],
) -> None:
    mode_gate = torch.zeros(TRAINED_MODES, device=device)
    mode_gate[list(keep)] = 1.0

    def stage_input_hook(
        _module: nn.Module,
        inputs: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        real, imag = inputs
        gate = mode_gate.to(dtype=real.dtype)
        return real * gate, imag * gate

    for stage_index, stage in enumerate(_stage_modules(model)):
        if stage_index not in stage_indices:
            continue
        handle = stage.register_forward_pre_hook(stage_input_hook)
        stack.callback(handle.remove)

    descriptor_gate = _pole_descriptor_mask(keep, device, stage_indices)

    def descriptor_hook(_module: nn.Module, _inputs: tuple[Tensor], output: Tensor) -> Tensor:
        return output * descriptor_gate.to(dtype=output.dtype)

    classifier = cast("Any", model.classifier)
    standardizers = (classifier.fusion.standardizer, classifier.affine.standardizer)
    for standardizer in standardizers:
        handle = standardizer.register_forward_hook(descriptor_hook)
        stack.callback(handle.remove)


def _descriptor_standardizers(model: nn.Module) -> tuple[nn.BatchNorm1d, ...]:
    classifier = cast("Any", model.classifier)
    standardizers = (classifier.fusion.standardizer, classifier.affine.standardizer)
    if any(not isinstance(module, nn.BatchNorm1d) for module in standardizers):
        raise TypeError("M64 audit expects BatchNorm descriptor standardizers")
    return cast("tuple[nn.BatchNorm1d, ...]", standardizers)


def _recalibrate_standardizers(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    batches: int,
) -> int:
    if batches <= 0:
        return 0
    total = 0
    sum_descriptor: Tensor | None = None
    sum_square: Tensor | None = None
    with torch.inference_mode():
        for batch_index, (inputs, _targets) in enumerate(loader):
            if batch_index >= batches:
                break
            device_inputs = inputs.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            descriptor = cast("Any", model).raw_descriptor(device_inputs).double()
            current_sum = descriptor.sum(0)
            current_square = descriptor.square().sum(0)
            sum_descriptor = current_sum if sum_descriptor is None else sum_descriptor + current_sum
            sum_square = current_square if sum_square is None else sum_square + current_square
            total += descriptor.shape[0]
    if total == 0 or sum_descriptor is None or sum_square is None:
        raise RuntimeError("descriptor calibration received no samples")
    mean = sum_descriptor / total
    variance = (sum_square / total - mean.square()).clamp_min(0.0)
    for standardizer in _descriptor_standardizers(model):
        standardizer.running_mean.copy_(mean.to(standardizer.running_mean))
        standardizer.running_var.copy_(variance.to(standardizer.running_var))
    return total


def _joint_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("M64 model returned an unsupported classifier output")


def _evaluate(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> dict[str, float]:
    correct = 0
    count = 0
    cross_entropy = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets in loader:
            device_inputs = inputs.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            device_targets = targets.to(device, non_blocking=True)
            logits = _joint_logits(model(device_inputs))
            cross_entropy += float(
                functional.cross_entropy(logits, device_targets, reduction="sum")
            )
            correct += int((logits.argmax(-1) == device_targets).sum())
            count += device_targets.numel()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "accuracy": correct / count,
        "cross_entropy": cross_entropy / count,
        "samples": float(count),
        "seconds": elapsed,
        "images_per_second": count / elapsed,
    }


def _condition_components(condition: Condition) -> frozenset[str]:
    if condition in {"cffn96", "pole48_cffn96"}:
        return frozenset(("mode", "transition", "post"))
    if condition == "mode96":
        return frozenset(("mode",))
    if condition == "transition96":
        return frozenset(("transition",))
    if condition == "post96":
        return frozenset(("post",))
    return frozenset()


def _pole_stage_indices(condition: Condition) -> frozenset[int]:
    if condition in {"pole48", "pole48_cffn96"}:
        return frozenset(range(len(STAGE_NAMES)))
    stage_conditions: dict[Condition, int] = {
        "pole48_s1": 0,
        "pole48_s2": 1,
        "pole48_s3": 2,
        "pole48_s4": 3,
    }
    index = stage_conditions.get(condition)
    return frozenset() if index is None else frozenset((index,))


def _run_condition(
    args: argparse.Namespace,
    loader: DataLoader[Any],
    device: torch.device,
    condition: Condition,
    masks: MaskSpec,
) -> dict[str, Any]:
    model = _model_from_checkpoint(args.checkpoint, device)
    components = _condition_components(condition)
    if components:
        apply_cffn_knockout(model, masks, components)
    pole_stages = _pole_stage_indices(condition)
    with ExitStack() as stack:
        if pole_stages:
            _install_pole_hooks(stack, model, masks.pole_keep, device, pole_stages)
        calibration_samples = _recalibrate_standardizers(
            model,
            loader,
            device,
            args.calibration_batches,
        )
        metrics = _evaluate(model, loader, device)
    del model
    torch.cuda.empty_cache()
    return {
        "repeat": masks.repeat,
        "condition": condition,
        "calibration_samples": calibration_samples,
        **metrics,
    }


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if args.repeats <= 0 or args.repeat_start < 0 or args.calibration_batches < 0:
        raise ValueError("repeat and calibration counts must be non-negative")
    random.seed(args.mask_seed)
    torch.manual_seed(args.mask_seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the M64 checkpoint audit requires an available CUDA device")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    loader = _validation_loader(args.data_root, args.batch_size, args.workers)
    payload: dict[str, Any] = {
        "schema": "lnet.d4_m64.factorial_knockout.v1",
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "mask_seed": args.mask_seed,
        "pole_mask_strategy": args.pole_mask_strategy,
        "repeat_start": args.repeat_start,
        "repeats": args.repeats,
        "calibration_batches": args.calibration_batches,
        "conditions": list(args.conditions),
        "rows": [],
    }

    full_masks = make_mask_spec(
        args.repeat_start,
        args.mask_seed,
        args.pole_mask_strategy,
    )
    full_row = _run_condition(args, loader, device, "full", full_masks)
    full_row["repeat"] = -1
    payload["rows"].append(full_row)
    _write_output(args.output, payload)
    print(json.dumps(full_row, sort_keys=True), flush=True)

    for repeat in range(args.repeat_start, args.repeat_start + args.repeats):
        masks = make_mask_spec(repeat, args.mask_seed, args.pole_mask_strategy)
        for raw_condition in args.conditions:
            condition = cast("Condition", raw_condition)
            row = _run_condition(args, loader, device, condition, masks)
            payload["rows"].append(row)
            _write_output(args.output, payload)
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
