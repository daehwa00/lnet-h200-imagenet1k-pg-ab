#!/usr/bin/env python3
# ruff: noqa: C901, PERF401, PLR0912, PLR0915, T201
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
"""Build, benchmark, and gate the frozen ComplexScanBackbone KAU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
from lnet.pac_capture_safe_orthogonal import prepare_capture_safe_orthogonal_

SCHEMA = "lnet.complex_scan.common_kau.training.v2"
FIXTURE_SCHEMA = "lnet.complex_scan.common_kau.fixture.v1"
EVALUATION_SCHEMA = "lnet.complex_scan.common_kau.evaluation.v1"
EXPECTED_PARAMETERS = 374_052
EXPECTED_BATCH_SIZE = int(os.environ.get("POLEPYRAMID_BENCHMARK_BATCH_SIZE", "256"))
SEED = 20260802


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fixture = commands.add_parser("fixture", help="write the shared state/input fixture")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    fixture.add_argument("--seed", type=int, default=SEED)

    benchmark = commands.add_parser("benchmark", help="benchmark one frozen runtime")
    benchmark.add_argument("--role", choices=("baseline", "candidate"), required=True)
    benchmark.add_argument("--runtime-root", type=Path, required=True)
    benchmark.add_argument("--fixture", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--evidence-output", type=Path)
    benchmark.add_argument("--reference-evidence", type=Path)
    benchmark.add_argument("--warmups", type=int, default=5)
    benchmark.add_argument("--iterations", type=int, default=10)
    benchmark.add_argument("--rounds", type=int, default=5)

    evaluate = commands.add_parser("evaluate", help="apply the performance contract")
    evaluate.add_argument("--baseline", type=Path, required=True)
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--tests-status", choices=("pass", "fail"), required=True)
    evaluate.add_argument("--minimum-speedup", type=float, default=1.15)
    evaluate.add_argument("--maximum-memory-ratio", type=float, default=1.0)
    evaluate.add_argument("--maximum-logits-error", type=float, default=2.0e-2)
    evaluate.add_argument("--maximum-loss-error", type=float, default=5.0e-3)
    evaluate.add_argument("--maximum-gradient-relative-rmse", type=float, default=2.0e-2)
    evaluate.add_argument("--maximum-update-relative-rmse", type=float, default=2.0e-2)
    return parser


def _config() -> ComplexScanConfig:
    """Return the fixed ImageNet-100 LRQ64 associative-scan configuration."""
    return ComplexScanConfig(
        output_dim=100,
        stem_width=64,
        stem="normalized",
        stem_strides=(2, 2),
        modes=(32, 32, 32),
        augmented_widths=(64, 64),
        carry_bases=("s2d", "s2d"),
        carry_merge="pole_main",
        carry_scale_initial=1.0e-2,
        quadratic_rank=64,
        fusion_width=None,
    )


def _config_payload() -> dict[str, Any]:
    """Return the config in the same tuple-free form used by JSON artifacts."""
    return cast("dict[str, Any]", json.loads(json.dumps(asdict(_config()))))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_model(
    state: dict[str, Tensor] | None = None,
) -> ComplexScanBackbone:
    model = ComplexScanBackbone(_config())
    replaced = prepare_capture_safe_orthogonal_(model)
    if not replaced:
        message = "expected at least one matrix-exp orthogonal parametrization"
        raise RuntimeError(message)
    if state is not None:
        model.load_state_dict(state, strict=True)
    model = model.cuda().train()
    model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    return model


def _optimizer(
    model: nn.Module,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    named_groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "decay": [],
        "no_decay": [],
        "modal": [],
        "geometry": [],
    }
    for name, parameter in model.named_parameters():
        if "damping_logits" in name or "phase_" in name:
            group = "geometry"
        elif (
            "analysis" in name
            or "augmented.direction_mixer" in name
            or "augmented.output_projection" in name
        ):
            group = "modal"
        elif parameter.ndim < 2 or "norm" in name or name.endswith(".bias"):
            group = "no_decay"
        else:
            group = "decay"
        named_groups[group].append((name, parameter))

    learning_rate = 3.0e-3
    settings = {
        "decay": (learning_rate, 0.05),
        "no_decay": (learning_rate, 0.0),
        "modal": (learning_rate / 3.0, 0.0),
        "geometry": (learning_rate * 0.1, 0.0),
    }
    groups: list[dict[str, Any]] = []
    signature: list[dict[str, Any]] = []
    for name in ("decay", "no_decay", "modal", "geometry"):
        learning_rate, weight_decay = settings[name]
        entries = named_groups[name]
        groups.append(
            {
                "params": [parameter for _, parameter in entries],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
        signature.append(
            {
                "name": name,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "parameters": [parameter_name for parameter_name, _ in entries],
            }
        )
    return torch.optim.AdamW(groups, fused=True), signature


def _compile(model: nn.Module) -> nn.Module:
    return cast(
        "nn.Module",
        torch.compile(model, mode="default", fullgraph=False, dynamic=False),
    )


def _fixture(args: argparse.Namespace) -> None:
    if args.batch_size != EXPECTED_BATCH_SIZE:
        message = f"contract requires batch size {EXPECTED_BATCH_SIZE}"
        raise ValueError(message)
    torch.manual_seed(args.seed)
    model = ComplexScanBackbone(_config())
    replaced = prepare_capture_safe_orthogonal_(model)
    if not replaced:
        message = "fixture model did not contain an orthogonal parametrization"
        raise RuntimeError(message)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    inputs = torch.randn(args.batch_size, 3, 224, 224, generator=generator)
    targets = torch.randint(100, (args.batch_size,), generator=generator)
    permutation = torch.randperm(args.batch_size, generator=generator)
    payload = {
        "schema": FIXTURE_SCHEMA,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "config": _config_payload(),
        "mixing": 0.375,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "inputs": inputs,
        "targets": targets,
        "permutation": permutation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"fixture": str(args.output), "sha256": _sha256(args.output)}, indent=2))


def _tensor_mapping(value: object, name: str) -> dict[str, Tensor]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()
    ):
        message = f"{name} must be a string-to-tensor mapping"
        raise TypeError(message)
    return cast("dict[str, Tensor]", value)


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != FIXTURE_SCHEMA:
        message = "invalid ComplexScanBackbone benchmark fixture"
        raise ValueError(message)
    fixture_config = json.loads(json.dumps(payload.get("config")))
    active_config = _config_payload()
    matching_config = isinstance(fixture_config, dict) and all(
        key in active_config and active_config[key] == value
        for key, value in fixture_config.items()
    )
    if payload.get("batch_size") != EXPECTED_BATCH_SIZE or not matching_config:
        message = "fixture does not match the fixed batch/configuration contract"
        raise ValueError(message)
    return cast("dict[str, Any]", payload)


def _training_step(
    runtime: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    permutation: Tensor,
    mixing: float,
    *,
    collect: bool,
) -> dict[str, Any] | None:
    optimizer.zero_grad(set_to_none=True)
    mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = runtime(mixed_inputs)
        loss = mixing * functional.cross_entropy(logits, targets, label_smoothing=0.1) + (
            1.0 - mixing
        ) * functional.cross_entropy(logits, targets[permutation], label_smoothing=0.1)
    loss.backward()
    if collect:
        gradients = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        collected: dict[str, Any] = {
            "logits": logits.detach().float().cpu().clone(),
            "loss": loss.detach().float().cpu().clone(),
            "gradients": gradients,
        }
    else:
        collected = {}
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if not collect:
        return None
    collected["gradient_norm"] = gradient_norm.detach().float().cpu()
    collected["updated_parameters"] = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    return collected


def _mapping_error(reference: dict[str, Tensor], candidate: dict[str, Tensor]) -> dict[str, Any]:
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    common = sorted(reference_keys & candidate_keys)
    maximum = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    for name in common:
        expected = reference[name].double()
        difference = candidate[name].double() - expected
        maximum = max(maximum, float(difference.abs().max()))
        squared_error += float(difference.square().sum())
        squared_reference += float(expected.square().sum())
    return {
        "keys_equal": reference_keys == candidate_keys,
        "missing_keys": sorted(reference_keys - candidate_keys),
        "unexpected_keys": sorted(candidate_keys - reference_keys),
        "max_abs": maximum,
        "relative_rmse": math.sqrt(squared_error / max(squared_reference, 1.0e-24)),
    }


def _parity(reference_path: Path, actual: dict[str, Any]) -> dict[str, Any]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if not isinstance(reference, dict):
        message = "reference evidence is not a mapping"
        raise TypeError(message)
    expected_logits = reference.get("logits")
    expected_loss = reference.get("loss")
    actual_logits = actual.get("logits")
    actual_loss = actual.get("loss")
    logit_and_loss_values = (expected_logits, expected_loss, actual_logits, actual_loss)
    if not all(isinstance(value, Tensor) for value in logit_and_loss_values):
        message = "reference or candidate logits/loss evidence is missing"
        raise TypeError(message)
    gradient_error = _mapping_error(
        _tensor_mapping(reference.get("gradients"), "reference gradients"),
        _tensor_mapping(actual.get("gradients"), "candidate gradients"),
    )
    update_error = _mapping_error(
        _tensor_mapping(reference.get("updated_parameters"), "reference updated parameters"),
        _tensor_mapping(actual.get("updated_parameters"), "candidate updated parameters"),
    )
    return {
        "logits_max_abs": float(
            (cast("Tensor", actual_logits) - cast("Tensor", expected_logits)).abs().max()
        ),
        "loss_abs": float(abs(cast("Tensor", actual_loss) - cast("Tensor", expected_loss))),
        "gradient_max_abs": gradient_error["max_abs"],
        "gradient_relative_rmse": gradient_error["relative_rmse"],
        "gradient_keys_equal": gradient_error["keys_equal"],
        "gradient_missing_keys": gradient_error["missing_keys"],
        "gradient_unexpected_keys": gradient_error["unexpected_keys"],
        "optimizer_parameter_max_abs": update_error["max_abs"],
        "optimizer_parameter_relative_rmse": update_error["relative_rmse"],
        "optimizer_parameter_keys_equal": update_error["keys_equal"],
        "optimizer_parameter_missing_keys": update_error["missing_keys"],
        "optimizer_parameter_unexpected_keys": update_error["unexpected_keys"],
    }


def _benchmark(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        message = "the KAU ComplexScanBackbone benchmark requires CUDA"
        raise RuntimeError(message)
    if min(args.warmups, args.iterations, args.rounds) <= 0:
        message = "warmups, iterations, and rounds must be positive"
        raise ValueError(message)
    if args.role == "baseline" and args.evidence_output is None:
        message = "baseline benchmark requires --evidence-output"
        raise ValueError(message)
    if args.role == "candidate" and args.reference_evidence is None:
        message = "candidate benchmark requires --reference-evidence"
        raise ValueError(message)

    fixture = _load_fixture(args.fixture)
    torch.manual_seed(int(fixture["seed"]))
    torch.cuda.manual_seed_all(int(fixture["seed"]))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    state = _tensor_mapping(fixture["model_state"], "fixture model state")
    model = _prepare_model(state)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        message = f"fixed LRQ64 model parameter count changed: {parameters}"
        raise RuntimeError(message)
    runtime = _compile(model)
    optimizer, optimizer_signature = _optimizer(model)
    inputs = cast("Tensor", fixture["inputs"]).cuda().contiguous(memory_format=torch.channels_last)
    actual_batch_size = int(inputs.shape[0])
    targets = cast("Tensor", fixture["targets"]).cuda()
    permutation = cast("Tensor", fixture["permutation"]).cuda()
    mixing = float(fixture["mixing"])

    evidence = _training_step(
        runtime,
        model,
        optimizer,
        inputs,
        targets,
        permutation,
        mixing,
        collect=True,
    )
    if evidence is None:
        message = "parity step failed to collect evidence"
        raise RuntimeError(message)
    if args.role == "baseline":
        args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(evidence, args.evidence_output)
        parity: dict[str, Any] = {"reference": True}
    else:
        parity = _parity(args.reference_evidence, evidence)

    for _ in range(args.warmups):
        _training_step(
            runtime,
            model,
            optimizer,
            inputs,
            targets,
            permutation,
            mixing,
            collect=False,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sample_seconds: list[float] = []
    for _ in range(args.rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            _training_step(
                runtime,
                model,
                optimizer,
                inputs,
                targets,
                permutation,
                mixing,
                collect=False,
            )
        end.record()
        end.synchronize()
        sample_seconds.append(float(start.elapsed_time(end)) / (1000.0 * args.iterations))

    median_step_seconds = statistics.median(sample_seconds)
    runtime_root = args.runtime_root.resolve()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "role": args.role,
        "device": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__,
        "batch_size": actual_batch_size,
        "precision": "bfloat16 autocast",
        "input_shape": list(inputs.shape),
        "input_memory_format": "channels_last",
        "compile_mode": "default",
        "fused_adamw": True,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "step_seconds_samples": sample_seconds,
        "median_step_seconds": median_step_seconds,
        "median_step_throughput_images_per_second": actual_batch_size / median_step_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "parameters": parameters,
        "config": _config_payload(),
        "model_fixture_sha256": _sha256(args.fixture),
        "optimizer_group_signature": optimizer_signature,
        "scan_pipeline": "associative_product",
        "last_training_loss": float(cast("Tensor", evidence["loss"])),
        "last_gradient_norm": float(cast("Tensor", evidence["gradient_norm"])),
        "parity": parity,
        "runtime_root": str(runtime_root),
        "source_sha256": {
            "model": _sha256(runtime_root / "src/lnet/complex_scan.py"),
            "product_scan_pipeline": _sha256(
                runtime_root / "src/lnet/pac_product_scan_pipeline.py"
            ),
            "bidirectional_product_scan": _sha256(
                runtime_root / "src/lnet/pac_triton_bidirectional_product_scan.py"
            ),
            "product_scan_coarse4": _sha256(
                runtime_root / "src/lnet/pac_triton_product_scan_coarse4.py"
            ),
        },
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"{key} must be numeric"
        raise TypeError(message)
    result = float(value)
    if not math.isfinite(result):
        message = f"{key} must be finite"
        raise ValueError(message)
    return result


def _config_extends(reference: object, actual: object) -> bool:
    """Return whether ``actual`` preserves every explicit reference setting."""
    return (
        isinstance(reference, dict)
        and isinstance(actual, dict)
        and all(key in actual and actual[key] == value for key, value in reference.items())
    )


def _evaluate(args: argparse.Namespace) -> int:
    baseline = cast("dict[str, Any]", json.loads(args.baseline.read_text()))
    candidate = cast("dict[str, Any]", json.loads(args.candidate.read_text()))
    failures: list[str] = []

    for name, payload, role in (
        ("baseline", baseline, "baseline"),
        ("candidate", candidate, "candidate"),
    ):
        if payload.get("schema") != SCHEMA:
            failures.append(f"{name} schema mismatch")
        if payload.get("role") != role:
            failures.append(f"{name} role mismatch")
        if payload.get("parameters") != EXPECTED_PARAMETERS:
            failures.append(f"{name} parameter count is not {EXPECTED_PARAMETERS}")
        if payload.get("batch_size") != EXPECTED_BATCH_SIZE:
            failures.append(f"{name} batch size is not {EXPECTED_BATCH_SIZE}")
        if not _config_extends(payload.get("config"), _config_payload()):
            failures.append(f"{name} model configuration changed")
        if payload.get("precision") != "bfloat16 autocast":
            failures.append(f"{name} did not use BF16 autocast")
        if payload.get("compile_mode") != "default" or not payload.get("fused_adamw"):
            failures.append(f"{name} production runtime bundle changed")
        if payload.get("scan_pipeline") != "associative_product":
            failures.append(f"{name} did not use the associative product scan pipeline")

    for key in (
        "device",
        "device_capability",
        "device_total_memory_bytes",
        "torch_version",
        "batch_size",
        "input_shape",
        "model_fixture_sha256",
        "optimizer_group_signature",
    ):
        if candidate.get(key) != baseline.get(key):
            failures.append(f"baseline/candidate {key} mismatch")
    if "RTX 4090" not in str(candidate.get("device", "")):
        failures.append("benchmark device is not an RTX 4090")

    baseline_speed = _number(baseline, "median_step_throughput_images_per_second")
    candidate_speed = _number(candidate, "median_step_throughput_images_per_second")
    speedup = candidate_speed / baseline_speed
    if speedup < args.minimum_speedup:
        failures.append(f"median training speedup {speedup:.4f}x < {args.minimum_speedup:.4f}x")

    baseline_memory = _number(baseline, "peak_allocated_bytes")
    candidate_memory = _number(candidate, "peak_allocated_bytes")
    memory_ratio = candidate_memory / baseline_memory
    if memory_ratio > args.maximum_memory_ratio:
        failures.append(
            f"peak allocated memory ratio {memory_ratio:.4f} > {args.maximum_memory_ratio:.4f}"
        )

    parity = candidate.get("parity")
    if not isinstance(parity, dict):
        failures.append("candidate parity evidence is missing")
        parity = {}
    parity_limits = {
        "logits_max_abs": args.maximum_logits_error,
        "loss_abs": args.maximum_loss_error,
        "gradient_relative_rmse": args.maximum_gradient_relative_rmse,
        "optimizer_parameter_relative_rmse": args.maximum_update_relative_rmse,
    }
    for key, limit in parity_limits.items():
        try:
            value = _number(parity, key)
        except (TypeError, ValueError) as error:
            failures.append(f"candidate parity {error}")
            continue
        if value > limit:
            failures.append(f"candidate parity {key} {value:.6g} > {limit:.6g}")
    for key in ("gradient_keys_equal", "optimizer_parameter_keys_equal"):
        if parity.get(key) is not True:
            failures.append(f"candidate parity {key} is not true")
    if args.tests_status != "pass":
        failures.append("focused candidate regression tests failed")

    payload: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "status": "pass" if not failures else "fail",
        "contract": {
            "minimum_speedup": args.minimum_speedup,
            "maximum_memory_ratio": args.maximum_memory_ratio,
            "maximum_logits_error": args.maximum_logits_error,
            "maximum_loss_error": args.maximum_loss_error,
            "maximum_gradient_relative_rmse": args.maximum_gradient_relative_rmse,
            "maximum_update_relative_rmse": args.maximum_update_relative_rmse,
            "focused_regression_tests_required": True,
        },
        "baseline_images_per_second": baseline_speed,
        "candidate_images_per_second": candidate_speed,
        "speedup": speedup,
        "baseline_peak_allocated_bytes": baseline_memory,
        "candidate_peak_allocated_bytes": candidate_memory,
        "memory_ratio": memory_ratio,
        "candidate_parity": parity,
        "focused_regression_tests": args.tests_status,
        "failures": failures,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


def main() -> int:
    args = _parser().parse_args()
    if args.command == "fixture":
        _fixture(args)
        return 0
    if args.command == "benchmark":
        _benchmark(args)
        return 0
    if args.command == "evaluate":
        return _evaluate(args)
    message = f"unsupported command: {args.command}"
    raise ValueError(message)


if __name__ == "__main__":
    raise SystemExit(main())
