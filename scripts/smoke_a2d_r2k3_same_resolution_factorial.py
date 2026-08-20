#!/usr/bin/env python3
"""Smoke the complete width-by-resolution same-stage factorial."""

from __future__ import annotations

# The experiment scripts are executable modules loaded through PYTHONPATH=scripts.
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportFunctionMemberAccess=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import argparse
import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_same_resolution_depth_imagenet100 as prior
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as runner
import torch

from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_directional import direction_aligned_cells
from lnet.pac_same_resolution_depth import (
    SameResolutionPoleScanBlock,
    _normalized_odd_directional_states,
    direction_relative_cells_to_full_resolution,
)

if TYPE_CHECKING:
    from lnet.complex_scan import ComplexScanBackbone
    from lnet.complex_scan_types import ComplexField


REPRESENTATIVE_PLACEMENTS = (
    (),
    (56,),
    (28,),
    (14,),
    (7,),
    runner.RESOLUTIONS,
)
COMPILE_VARIANTS = frozenset(
    runner.variant_name(width, placement)
    for width in runner.WIDTHS
    for placement in REPRESENTATIVE_PLACEMENTS
)
MAXIMAL_VARIANTS = tuple(runner.variant_name(width, runner.RESOLUTIONS) for width in runner.WIDTHS)


def _logits(output: Any) -> torch.Tensor:
    if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
        raise TypeError("model did not return the established affine-head tuple")
    return output[0]


def _compiled_cuda_step(model: torch.nn.Module) -> float:
    model.prepare_for_compiled_training_()
    compiled = torch.compile(model, mode="default")
    compiled.train()
    inputs = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = _logits(compiled(inputs)).float().square().mean()
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("compiled CUDA smoke produced a non-finite loss")
    return float(loss.detach())


def _cell_reconstruction_check(device: torch.device) -> float:
    real = torch.randn(2, 6, 8, 5, device=device)
    directions = ((1, 1), (-1, 1), (1, -1), (-1, -1))
    cells = torch.stack(
        tuple(
            direction_aligned_cells(
                real,
                real,
                direction_x=direction_x,
                direction_y=direction_y,
            )[0]
            for direction_x, direction_y in directions
        ),
        dim=-3,
    )
    expected = torch.stack((real, real, real, real), dim=-2)
    error = float((direction_relative_cells_to_full_resolution(cells) - expected).abs().max())
    if error != 0.0:
        raise RuntimeError(f"D4 full-cell reconstruction changed coordinates: {error}")
    return error


def _full_state_semantic_check(device: torch.device) -> float:
    modes = 8
    template = ComplexScanStage(
        modes,
        maximum_phase=1.0,
        output_modes=None,
        scan_memory_policy="recompute",
    ).to(device)
    block = SameResolutionPoleScanBlock(
        modes,
        reader_rank=2,
        kernel_size=3,
        pole_template=template,
        post_hidden=12,
    ).to(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    real = torch.randn(1, 6, 8, modes, device=device, dtype=dtype)
    imag = torch.randn_like(real)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        drive = block.reader(real, imag)
        shape = cast("tuple[int, int, int, int]", tuple(drive[0].shape))
        poles = block.pole_scan.pole_coefficients(shape)
        expected = _normalized_odd_directional_states(
            *poles,
            drive,
            gain_normalization=block.pole_scan.product_gain_normalization,
            epsilon=1.0e-8,
        )
        actual = block.directional_states(real, imag)
    error = max(
        float((actual[0].float() - expected[0].float()).abs().max()),
        float((actual[1].float() - expected[1].float()).abs().max()),
    )
    tolerance = 3.0e-2 if device.type == "cuda" else 2.0e-5
    if error > tolerance:
        raise RuntimeError(f"fused full16 reconstruction mismatches full D4 states: {error}")
    return error


def _transition_checks(model: torch.nn.Module, variant: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in runner.STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        device = transition.carry_weight.device
        memory = (
            torch.randn(2, 3, 3, transition.input_modes, device=device),
            torch.randn(2, 3, 3, transition.input_modes, device=device),
        )
        carry = (
            torch.randn(2, 3, 3, transition.carry_input_modes, device=device),
            torch.randn(2, 3, 3, transition.carry_input_modes, device=device),
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            projected_memory = transition._project(transition.memory_projection, memory)
            projected_carry = transition._project(
                transition.carry_projection,
                transition._carry(*carry),
            )
            merged = (
                projected_memory[0] + projected_carry[0],
                projected_memory[1] + projected_carry[1],
            )
            expected = transition.post_fusion(*merged)
            actual = transition(memory[0], memory[1], *carry)
        error = max(
            float((actual[0] - expected[0]).abs().max().detach()),
            float((actual[1] - expected[1]).abs().max().detach()),
        )
        if error != 0.0:
            raise RuntimeError(f"{variant}/{name} gated transition algebra changed")
        if not torch.equal(
            transition.carry_weight,
            torch.full_like(transition.carry_weight, 0.25),
        ):
            raise RuntimeError(f"{variant}/{name} S2D carry initialization changed")
        metrics[f"{name}_transition_error"] = error
    return metrics


def _blocks(
    model: ComplexScanBackbone,
    resolution: int,
) -> tuple[SameResolutionPoleScanBlock, ...]:
    if not isinstance(model, runner.SameResolutionFactorialBackbone):
        return ()
    key = str(resolution)
    blocks = []
    if key in model.same_resolution_blocks:
        blocks.append(model.same_resolution_blocks[key])
    extras = getattr(model, "extra_same_resolution_blocks", None)
    if extras is not None and key in extras:
        extra = extras[key]
        if isinstance(extra, torch.nn.ModuleList):
            blocks.extend(extra)
        else:
            blocks.append(extra)
    if any(not isinstance(block, SameResolutionPoleScanBlock) for block in blocks):
        raise TypeError(f"{resolution}x{resolution} contains a non-pole stage")
    return tuple(cast("list[SameResolutionPoleScanBlock]", blocks))


def _block_transition_check(
    block: SameResolutionPoleScanBlock | None,
    source: ComplexField,
    output: ComplexField,
) -> tuple[float, float]:
    if block is None:
        return 0.0, float("inf")
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=source[0].device.type,
            dtype=torch.bfloat16,
            enabled=source[0].is_cuda,
        ),
    ):
        paths = block.directional_states(*source)
        memory = block.collapse_paths(*paths)
        expected = block.post_fusion(source[0] + memory[0], source[1] + memory[1])
    error = max(
        float((output[0] - expected[0]).abs().max().detach()),
        float((output[1] - expected[1]).abs().max().detach()),
    )
    update_rms = float(
        (
            (output[0] - source[0]).float().square().mean()
            + (output[1] - source[1]).float().square().mean()
        )
        .sqrt()
        .detach()
    )
    return error, update_rms


def _validate_transition_error(device_type: str, error: float) -> None:
    """Enforce semantic parity where a deterministic FP32 reference is valid."""
    if device_type == "cpu" and error > 2.0e-5:
        raise RuntimeError(f"same-resolution algebra changed: {error}")


def _shape_and_stage_checks(
    model: ComplexScanBackbone,
    variant: str,
    device: torch.device,
) -> dict[str, Any]:
    spec = runner.SPECS[variant]
    state_width = int(getattr(spec, "excitation_width", spec.width))
    descriptor_dim = int(getattr(spec, "descriptor_dim", 16 * spec.width))
    batch = 2
    inputs = torch.randn(batch, 3, 32, 32, device=device)
    transition_error = 0.0
    minimum_update = float("inf")
    states: dict[str, list[int]] = {}
    descriptors = []
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        state = model._initial_excitation(inputs)
        for resolution, next_resolution, stage in (
            (56, 28, model.stage1),
            (28, 14, model.stage2),
            (14, 7, cast("Any", model).stage3),
        ):
            source = state
            state = source
            for block in _blocks(model, resolution):
                block_source = state
                state = cast("ComplexField", block(*state))
                error, update = _block_transition_check(block, block_source, state)
                transition_error = max(transition_error, error)
                minimum_update = min(minimum_update, update)
            states[f"sr{resolution}"] = list(state[0].shape)
            state, descriptor = stage(*state)
            state = model._require_state(state)
            states[f"stage_to_{next_resolution}"] = list(state[0].shape)
            descriptors.append(descriptor)
        source = state
        state = source
        for block in _blocks(model, 7):
            block_source = state
            state = cast("ComplexField", block(*state))
            error, update = _block_transition_check(block, block_source, state)
            transition_error = max(transition_error, error)
            minimum_update = min(minimum_update, update)
        states["sr7"] = list(state[0].shape)
        _, descriptor4 = model.terminal(*state)
        descriptors.append(descriptor4)
        descriptor = torch.cat(descriptors, dim=-1)

    spatial = {56: 8, 28: 4, 14: 2, 7: 1}
    expected_states = {
        f"sr{resolution}": [batch, size, size, state_width]
        for resolution, size in spatial.items()
    }
    expected_states.update(
        {
            "stage_to_28": [batch, 4, 4, state_width],
            "stage_to_14": [batch, 2, 2, state_width],
            "stage_to_7": [batch, 1, 1, state_width],
        }
    )
    if states != expected_states:
        raise RuntimeError(f"{variant} shape contract changed: {states} != {expected_states}")
    if list(descriptor.shape) != [batch, descriptor_dim]:
        raise RuntimeError(f"{variant} changed its established descriptor width")
    _validate_transition_error(device.type, transition_error)
    if spec.resolutions and minimum_update == 0.0:
        raise RuntimeError(f"{variant} contains a dormant full stage")
    return {
        "states": states,
        "descriptor": list(descriptor.shape),
        "transition_max_abs": transition_error,
        "transition_check_mode": (
            "diagnostic-bf16" if device.type == "cuda" else "enforced-fp32"
        ),
        "minimum_update_rms": None if minimum_update == float("inf") else minimum_update,
    }


def _train_step(model: ComplexScanBackbone, device: torch.device) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    inputs = torch.randn(2, 3, 32, 32, device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        loss = _logits(model(inputs)).float().square().mean()
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    if missing:
        raise RuntimeError(f"factorial smoke found inactive parameters: {missing[:8]}")
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    if nonfinite:
        raise RuntimeError(f"factorial smoke found non-finite gradients: {nonfinite[:8]}")
    dormant = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith(("same_resolution_blocks.", "extra_same_resolution_blocks."))
        and (parameter.grad is None or not bool(torch.count_nonzero(parameter.grad)))
    ]
    if dormant:
        raise RuntimeError(f"factorial smoke found dormant full-stage parameters: {dormant[:8]}")
    optimizer.step()
    return float(loss.detach())


def _checkpoint_roundtrip(
    model: ComplexScanBackbone,
    variant: str,
    device: torch.device,
    checkpoint: Path,
) -> float:
    model.eval()
    inputs = torch.randn(2, 3, 32, 32, device=device)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        reference = _logits(model(inputs)).float()
    torch.save(model.state_dict(), checkpoint)
    restored = runner._build(
        variant,
        runtime.model_config(),
    ).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    restored.eval()
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        candidate = _logits(restored(inputs)).float()
    error = float((reference - candidate).abs().max())
    if error != 0.0:
        raise RuntimeError(f"{variant} checkpoint round-trip changed logits: {error}")
    checkpoint.unlink(missing_ok=True)
    return error


def _validate_variant_selection(variants: tuple[str, ...]) -> None:
    selected = set(variants)
    for width in runner.WIDTHS:
        width_variants = {name for name in selected if runner.SPECS[name].width == width}
        base = runner.variant_name(width, ())
        if width_variants and base not in width_variants:
            raise ValueError(f"smoke selection for K{width} requires {base}")


def _reference_parity(device: torch.device, variants: tuple[str, ...]) -> dict[str, float]:
    config = runtime.model_config()
    inputs = torch.randn(2, 3, 32, 32, device=device)
    mapping = {
        (): prior.D1,
        (7,): prior.D2,
        (14,): prior.D3,
        (14, 7): prior.D4,
    }
    parity = {}
    for placement, prior_variant in mapping.items():
        factorial_variant = runner.variant_name(128, placement)
        if factorial_variant not in variants:
            continue
        torch.manual_seed(501)
        reference = prior._build(prior_variant, config).to(device).eval()
        torch.manual_seed(501)
        candidate = runner._build(factorial_variant, config).to(device).eval()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            expected = reference.raw_descriptor(inputs).float()
            actual = candidate.raw_descriptor(inputs).float()
        error = float((actual - expected).abs().max())
        tolerance = 1.0e-5 if device.type == "cuda" else 0.0
        if error > tolerance:
            raise RuntimeError(f"K128 reference parity changed for placement {placement}: {error}")
        parity[factorial_variant] = error
    return parity


def _run_candidate(
    variant: str,
    device: torch.device,
    root: Path,
    *,
    compile_model: bool,
) -> dict[str, Any]:
    torch.manual_seed(501)
    model = runner._build(
        variant,
        runtime.model_config(),
    ).to(device)
    runner._assert_model(model, variant)
    payload: dict[str, Any] = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "contract": _shape_and_stage_checks(model, variant, device),
        "transition": _transition_checks(model, variant),
    }
    payload["eager_loss"] = _train_step(model, device)
    payload["checkpoint_max_abs"] = _checkpoint_roundtrip(
        model,
        variant,
        device,
        root / "roundtrip.pt",
    )
    if compile_model:
        # Each candidate owns a fresh Dynamo specialization budget.  Without
        # this reset, compiling many independent model instances in one smoke
        # process can hit the global recompile limit and silently run eager.
        torch.compiler.reset()
        payload["compiled_loss"] = _compiled_cuda_step(model)
        torch.compiler.reset()
    else:
        payload["compiled_loss"] = None
    payload["status"] = "passed"
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _full_batch_cuda_step(
    variant: str,
    batch_size: int,
    repeat_steps: int,
) -> dict[str, float]:
    if repeat_steps < 2:
        message = "repeat_steps must exercise compiler cache stability"
        raise ValueError(message)
    torch.compiler.reset()
    torch.manual_seed(501)
    model = runner._build(
        variant,
        runtime.model_config(),
    ).cuda()
    model.prepare_for_compiled_training_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    inputs = torch.randn(batch_size, 3, 224, 224, device="cuda")
    loss = inputs.new_zeros(())
    samples = []
    torch.cuda.reset_peak_memory_stats()
    with cast("Any", torch)._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        compiled = torch.compile(model, mode="default", fullgraph=False, dynamic=False)
        for _ in range(repeat_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = _logits(compiled(inputs)).float().square().mean()
            loss.backward()
            optimizer.step()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"{variant} full-batch smoke produced a non-finite loss")
    steady_samples = samples[1:]
    median_step_ms = statistics.median(steady_samples)
    payload = {
        "loss": float(loss.detach()),
        "median_step_ms": median_step_ms,
        "images_per_second": 1000.0 * batch_size / median_step_ms,
        "peak_bytes": float(torch.cuda.max_memory_allocated()),
        "steps": float(repeat_steps),
    }
    del inputs, optimizer, compiled, model
    torch.cuda.empty_cache()
    torch.compiler.reset()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--full-batch", action="store_true")
    parser.add_argument("--full-batch-size", type=int, default=128)
    parser.add_argument("--repeat-steps", type=int, default=10)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=runner.VARIANTS,
        default=list(runner.VARIANTS),
    )
    args = parser.parse_args()
    variants = tuple(args.variants)
    _validate_variant_selection(variants)
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested without an available CUDA device")
    if (args.compile or args.full_batch) and device.type != "cuda":
        raise RuntimeError("compiled and full-batch smokes require CUDA")

    payload: dict[str, Any] = {
        "device": str(device),
        "candidate_count": len(variants),
        "expected_factorial_cells": len(runner.VARIANTS),
        "cell_reconstruction_max_abs": _cell_reconstruction_check(device),
        "full_state_semantic_max_abs": _full_state_semantic_check(device),
        "reference_parity": _reference_parity(device, variants),
        "compile_variants": sorted(COMPILE_VARIANTS.intersection(variants)),
        "variants": {},
    }
    for variant in variants:
        payload["variants"][variant] = _run_candidate(
            variant,
            device,
            args.root,
            compile_model=args.compile and variant in COMPILE_VARIANTS,
        )
        print(
            json.dumps(
                {
                    "event": "candidate_passed",
                    "variant": variant,
                    "compiled": variant in COMPILE_VARIANTS and args.compile,
                }
            ),
            flush=True,
        )

    block_counts: dict[int, set[int]] = {width: set() for width in runner.WIDTHS}
    for variant, evidence in payload["variants"].items():
        spec = runner.SPECS[variant]
        base_variant = runner.variant_name(spec.width, ())
        base_parameters = payload["variants"][base_variant]["parameters"]
        enabled = len(spec.resolutions)
        if enabled:
            delta = evidence["parameters"] - base_parameters
            if delta % enabled:
                raise RuntimeError(f"{variant} has non-uniform full-stage capacity")
            block_counts[spec.width].add(delta // enabled)
    active_counts = {width: counts for width, counts in block_counts.items() if counts}
    if any(len(counts) != 1 for counts in active_counts.values()):
        raise RuntimeError(f"full-stage capacity changed by placement: {block_counts}")
    payload["block_parameters_by_width"] = {
        str(width): counts.pop() for width, counts in active_counts.items()
    }

    payload["full_batch"] = {}
    if args.full_batch:
        for variant in MAXIMAL_VARIANTS:
            if variant not in variants:
                continue
            payload["full_batch"][variant] = _full_batch_cuda_step(
                variant,
                args.full_batch_size,
                args.repeat_steps,
            )
            print(
                json.dumps(
                    {
                        "event": "full_batch_passed",
                        "variant": variant,
                        **payload["full_batch"][variant],
                    }
                ),
                flush=True,
            )
    payload["status"] = "passed"
    (args.root / "smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
