"""Strict evaluator for the final EFP16 RTX 4090 appendix evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Final, cast

BUNDLE_FILE: Final = "bundle.json"
EVALUATION_FILE: Final = "evaluation.json"
BUNDLE_SCHEMA: Final = "pac_efp16_final_appendix_bundle.v1"
EVALUATION_SCHEMA: Final = "pac_efp16_final_appendix_evaluation.v1"

REQUIRED_ARTIFACTS: Final = {
    "hardware": "hardware.json",
    "efp16_inference": "efp16_inference.json",
    "efp16_training": "efp16_training.json",
    "model_inference": "model_inference.json",
    "model_training": "model_training.json",
    "profiler": "profiler.json",
    "candidate_registry": "candidate_registry.json",
    "reproducibility": "reproducibility.json",
    "test_report": "test_report.json",
}
ARTIFACT_SCHEMAS: Final = {
    "hardware": "pac_efp16_final_hardware.v1",
    "efp16_inference": "pac_efp16_inference_stages.v1",
    "efp16_training": "pac_efp16_training_stages.v1",
    "model_inference": "pac_10model_inference.v1",
    "model_training": "pac_10model_training.v1",
    "profiler": "pac_efp16_profiler.v1",
    "candidate_registry": "pac_efp16_candidate_registry.v1",
    "reproducibility": "pac_efp16_reproducibility.v1",
    "test_report": "pac_efp16_test_report.v1",
}

LENGTHS: Final = (128, 512, 2048)
BATCH_SIZES: Final = (1, 64)
SHAPES: Final = tuple((length, batch) for length in LENGTHS for batch in BATCH_SIZES)
EFP_STAGES: Final = ("eager", "previous_best", "final_best")
MODEL_STAGES: Final = ("eager", "best")
MODELS: Final = (
    "efp16",
    "gru",
    "minirocket",
    "cnn1d",
    "lstm",
    "mamba",
    "transformer",
    "tcn",
    "pa2wp",
    "s4d",
)
TRAINING_TIMINGS: Final = ("forward", "forward_backward", "full_step")
MAXIMUM_ERROR: Final = 2.0e-5
MINIMUM_PARITY_STEPS: Final = 75
REQUIRED_APPENDIX_SECTIONS: Final = (
    "Hardware and protocol",
    "EFP16 inference",
    "EFP16 training",
    "Ten-model inference",
    "Ten-model training",
    "Profiler and candidate registry",
    "Reproducibility and tests",
)
_COMMON_PROTOCOL_KEYS: Final = (
    "dtype",
    "tf32",
    "autocast",
    "synchronized",
    "compile_cost_excluded",
    "raw_samples",
    "normalized",
)
_ENVIRONMENT_KEYS: Final = ("device", "host", "torch", "cuda")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_MARKDOWN_LINK_PATTERN: Final = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def evaluate_final_appendix_bundle(  # noqa: C901, PLR0912, PLR0915
    artifact_root: Path,
    appendix_path: Path,
) -> dict[str, object]:
    """Fail closed over bundle contents, hashes, measurements, and appendix links."""
    root = artifact_root.resolve()
    failures: list[str] = []
    checked_hashes: dict[str, str] = {}
    counts = {
        "efp16_inference_rows": 0,
        "efp16_training_rows": 0,
        "model_inference_rows": 0,
        "model_training_rows": 0,
        "unsupported_inference_rows": 0,
        "unsupported_training_rows": 0,
    }
    if not root.is_dir():
        failures.append(f"artifact root does not exist: {artifact_root}")

    bundle = _load_json(root / BUNDLE_FILE, failures, BUNDLE_FILE)
    if bundle is None:
        return _result(failures, checked_hashes, counts, appendix_path)
    if bundle.get("schema") != BUNDLE_SCHEMA:
        failures.append(f"{BUNDLE_FILE}: schema must be {BUNDLE_SCHEMA}")

    artifacts = cast("dict[str, object]", bundle.get("artifacts", {}))
    if set(artifacts) != set(REQUIRED_ARTIFACTS):
        missing = sorted(set(REQUIRED_ARTIFACTS) - set(artifacts))
        extra = sorted(set(artifacts) - set(REQUIRED_ARTIFACTS))
        failures.append(f"bundle artifact keys mismatch; missing={missing}, extra={extra}")

    loaded: dict[str, dict[str, object]] = {}
    for key, expected_name in REQUIRED_ARTIFACTS.items():
        entry = cast("dict[str, object]", artifacts.get(key, {}))
        path_value = entry.get("path")
        if path_value != expected_name:
            failures.append(f"bundle.artifacts.{key}.path must be {expected_name}")
        digest = str(entry.get("sha256", ""))
        artifact_path = _checked_root_path(root, path_value, f"artifact {key}", failures)
        if artifact_path is None:
            continue
        if _verify_sha256(artifact_path, digest, f"artifact {key}", failures):
            checked_hashes[key] = digest
        payload = _load_json(artifact_path, failures, f"artifact {key}")
        if payload is not None:
            loaded[key] = payload
            if payload.get("schema") != ARTIFACT_SCHEMAS[key]:
                failures.append(
                    f"artifact {key}: schema must be {ARTIFACT_SCHEMAS[key]}"
                )

    appendix_digest = _validate_appendix(
        root,
        appendix_path,
        bundle,
        failures,
    )

    hardware = loaded.get("hardware")
    reference_environment: dict[str, object] = {}
    reference_protocol: dict[str, object] = {}
    if hardware is not None:
        reference_environment = cast("dict[str, object]", hardware.get("environment", {}))
        reference_protocol = cast("dict[str, object]", hardware.get("protocol", {}))
        _validate_environment(reference_environment, "hardware", failures)
        _validate_protocol(reference_protocol, "hardware", failures, measurement=False)

    for key in (
        "efp16_inference",
        "efp16_training",
        "model_inference",
        "model_training",
    ):
        payload = loaded.get(key)
        if payload is None:
            continue
        _validate_common_context(
            payload,
            key,
            reference_environment,
            reference_protocol,
            failures,
        )

    efp_inference = loaded.get("efp16_inference")
    if efp_inference is not None:
        counts["efp16_inference_rows"] = _validate_efp_inference(
            efp_inference, failures
        )
    efp_training = loaded.get("efp16_training")
    if efp_training is not None:
        counts["efp16_training_rows"] = _validate_efp_training(
            efp_training, failures
        )
    model_inference = loaded.get("model_inference")
    if model_inference is not None:
        measured, unsupported = _validate_model_inference(
            model_inference, root, failures
        )
        counts["model_inference_rows"] = measured + unsupported
        counts["unsupported_inference_rows"] = unsupported
    model_training = loaded.get("model_training")
    if model_training is not None:
        measured, unsupported = _validate_model_training(model_training, root, failures)
        counts["model_training_rows"] = measured + unsupported
        counts["unsupported_training_rows"] = unsupported

    profiler = loaded.get("profiler")
    if profiler is not None:
        _validate_profiler(profiler, root, failures)
    registry = loaded.get("candidate_registry")
    if registry is not None:
        _validate_candidate_registry(registry, root, failures)
    reproducibility = loaded.get("reproducibility")
    if reproducibility is not None:
        _validate_reproducibility(
            reproducibility,
            reference_environment,
            failures,
        )
    test_report = loaded.get("test_report")
    if test_report is not None:
        _validate_test_report(test_report, root, failures)

    return _result(
        failures,
        checked_hashes,
        counts,
        appendix_path,
        appendix_digest=appendix_digest,
    )


def _validate_efp_inference(payload: dict[str, object], failures: list[str]) -> int:
    rows = _indexed_rows(
        payload,
        ("stage", "length", "batch_size"),
        {
            (stage, length, batch)
            for stage in EFP_STAGES
            for length, batch in SHAPES
        },
        "efp16 inference",
        failures,
    )
    for key, row in rows.items():
        label = f"efp16 inference/{key}"
        if row.get("model") != "efp16" or row.get("status") != "measured":
            failures.append(f"{label}: must be a measured efp16 row")
        _validate_timing(cast("dict[str, object]", row.get("timing", {})), label, failures)
        _validate_inference_accuracy(
            cast("dict[str, object]", row.get("accuracy", {})), label, failures
        )
    return len(rows)


def _validate_efp_training(payload: dict[str, object], failures: list[str]) -> int:
    rows = _indexed_rows(
        payload,
        ("stage", "length", "batch_size"),
        {
            (stage, length, batch)
            for stage in EFP_STAGES
            for length, batch in SHAPES
        },
        "efp16 training",
        failures,
    )
    for key, row in rows.items():
        label = f"efp16 training/{key}"
        if row.get("model") != "efp16" or row.get("status") != "measured":
            failures.append(f"{label}: must be a measured efp16 row")
        _validate_training_measurement(row, label, failures, require_logit=True)
    return len(rows)


def _validate_model_inference(
    payload: dict[str, object], root: Path, failures: list[str]
) -> tuple[int, int]:
    rows = _indexed_rows(
        payload,
        ("model", "stage", "length", "batch_size"),
        {
            (model, stage, length, batch)
            for model in MODELS
            for stage in MODEL_STAGES
            for length, batch in SHAPES
        },
        "ten-model inference",
        failures,
    )
    measured = 0
    unsupported = 0
    for key, row in rows.items():
        label = f"ten-model inference/{key}"
        status = row.get("status")
        if status == "measured":
            measured += 1
            _validate_timing(
                cast("dict[str, object]", row.get("timing", {})), label, failures
            )
            _validate_inference_accuracy(
                cast("dict[str, object]", row.get("accuracy", {})), label, failures
            )
        elif status == "unsupported":
            unsupported += 1
            _validate_unsupported(row, root, label, failures)
        else:
            failures.append(f"{label}: status must be measured or unsupported")
    if measured == 0:
        failures.append("ten-model inference: no measured rows")
    return measured, unsupported


def _validate_model_training(
    payload: dict[str, object], root: Path, failures: list[str]
) -> tuple[int, int]:
    rows = _indexed_rows(
        payload,
        ("model", "stage", "length", "batch_size"),
        {
            (model, stage, length, batch)
            for model in MODELS
            for stage in MODEL_STAGES
            for length, batch in SHAPES
        },
        "ten-model training",
        failures,
    )
    measured = 0
    unsupported = 0
    for key, row in rows.items():
        label = f"ten-model training/{key}"
        status = row.get("status")
        if status == "measured":
            measured += 1
            _validate_training_measurement(row, label, failures)
        elif status == "unsupported":
            unsupported += 1
            _validate_unsupported(row, root, label, failures)
        else:
            failures.append(f"{label}: status must be measured or unsupported")
    if measured == 0:
        failures.append("ten-model training: no measured rows")
    return measured, unsupported


def _validate_training_measurement(
    row: dict[str, object],
    label: str,
    failures: list[str],
    *,
    require_logit: bool = False,
) -> None:
    timings = cast("dict[str, object]", row.get("timings", {}))
    if set(timings) != set(TRAINING_TIMINGS):
        failures.append(f"{label}: timings must contain {list(TRAINING_TIMINGS)}")
    for name in TRAINING_TIMINGS:
        _validate_timing(
            cast("dict[str, object]", timings.get(name, {})),
            f"{label}/{name}",
            failures,
        )
    full_step = cast("dict[str, object]", timings.get("full_step", {}))
    full_step_ms = _positive_float(full_step.get("wall_ms"))
    throughput = _positive_float(row.get("throughput_sequences_per_second"))
    batch_size = _as_int(row.get("batch_size"))
    if throughput is None:
        failures.append(f"{label}: missing positive throughput_sequences_per_second")
    elif full_step_ms is not None and batch_size > 0:
        expected = batch_size * 1000.0 / full_step_ms
        if not math.isclose(throughput, expected, rel_tol=0.01, abs_tol=1.0e-6):
            failures.append(f"{label}: throughput is inconsistent with full-step wall time")
    if _positive_float(row.get("peak_memory_mb")) is None:
        failures.append(f"{label}: missing positive peak_memory_mb")
    _validate_training_accuracy(
        cast("dict[str, object]", row.get("accuracy", {})),
        label,
        failures,
        require_logit=require_logit,
    )


def _validate_timing(
    timing: dict[str, object], label: str, failures: list[str]
) -> None:
    if timing.get("normalized") is not False:
        failures.append(f"{label}: timing.normalized must be false")
    for prefix in ("wall", "cuda_event"):
        samples = _positive_samples(timing.get(f"{prefix}_samples_ms"))
        if samples is None or len(samples) < 3:
            failures.append(f"{label}: {prefix}_samples_ms must contain >=3 raw samples")
            continue
        center = _positive_float(timing.get(f"{prefix}_ms"))
        if center is None:
            failures.append(f"{label}: missing positive {prefix}_ms")
            continue
        if not math.isclose(center, statistics.median(samples), rel_tol=1.0e-6, abs_tol=1.0e-9):
            failures.append(f"{label}: {prefix}_ms is not the median of raw samples")
    wall_samples = _positive_samples(timing.get("wall_samples_ms"))
    event_samples = _positive_samples(timing.get("cuda_event_samples_ms"))
    if wall_samples is not None and event_samples is not None and len(wall_samples) != len(
        event_samples
    ):
        failures.append(f"{label}: wall/event raw sample counts differ")


def _validate_inference_accuracy(
    accuracy: dict[str, object], label: str, failures: list[str]
) -> None:
    error = _finite_float(accuracy.get("max_abs_error"))
    if error is None or error < 0.0 or error > MAXIMUM_ERROR:
        failures.append(f"{label}: max_abs_error exceeds {MAXIMUM_ERROR}: {error}")
    agreement = _finite_float(accuracy.get("prediction_agreement"))
    if agreement != 1.0:
        failures.append(f"{label}: prediction_agreement must be 1.0")


def _validate_training_accuracy(
    accuracy: dict[str, object],
    label: str,
    failures: list[str],
    *,
    require_logit: bool,
) -> None:
    if _as_int(accuracy.get("parity_steps")) < MINIMUM_PARITY_STEPS:
        failures.append(f"{label}: parity_steps must be >= {MINIMUM_PARITY_STEPS}")
    metrics = [
        "loss_trajectory_max_abs_error",
        "final_gradient_max_abs_error",
        "final_parameter_max_abs_error",
    ]
    if require_logit:
        metrics.append("final_logit_max_abs_error")
    for metric in metrics:
        value = _finite_float(accuracy.get(metric))
        if value is None or value < 0.0 or value > MAXIMUM_ERROR:
            failures.append(f"{label}: {metric} exceeds {MAXIMUM_ERROR}: {value}")
    if _finite_float(accuracy.get("prediction_agreement")) != 1.0:
        failures.append(f"{label}: prediction_agreement must be 1.0")


def _validate_unsupported(
    row: dict[str, object], root: Path, label: str, failures: list[str]
) -> None:
    unsupported = cast("dict[str, object]", row.get("unsupported", {}))
    failures.extend(
        f"{label}: unsupported.{field} is required"
        for field in ("category", "reason", "expected_failure")
        if not str(unsupported.get(field, "")).strip()
    )
    command = unsupported.get("reproduction_command")
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part.strip() for part in command
    ):
        failures.append(f"{label}: unsupported.reproduction_command must be a nonempty argv list")
    _validate_evidence_reference(
        cast("dict[str, object]", unsupported.get("evidence", {})),
        root,
        f"{label} unsupported evidence",
        failures,
    )


def _validate_profiler(
    payload: dict[str, object], root: Path, failures: list[str]
) -> None:
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        failures.append("profiler: profiles must be a nonempty list")
        return
    kinds: set[str] = set()
    for index, value in enumerate(profiles):
        profile = cast("dict[str, object]", value if isinstance(value, dict) else {})
        label = f"profiler.profiles[{index}]"
        kind = str(profile.get("kind", ""))
        if kind not in {"inference", "training"}:
            failures.append(f"{label}: kind must be inference or training")
        else:
            kinds.add(kind)
        if profile.get("model") != "efp16":
            failures.append(f"{label}: model must be efp16")
        kernels = profile.get("dominant_kernels")
        if not isinstance(kernels, list) or not kernels or not all(
            isinstance(kernel, str) and kernel.strip() for kernel in kernels
        ):
            failures.append(f"{label}: dominant_kernels must be nonempty")
        _validate_evidence_reference(
            cast("dict[str, object]", profile.get("trace", {})),
            root,
            f"{label}.trace",
            failures,
        )
    if kinds != {"inference", "training"}:
        failures.append("profiler: both inference and training profiles are required")


def _validate_candidate_registry(
    payload: dict[str, object], root: Path, failures: list[str]
) -> None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        failures.append("candidate registry: candidates must be a nonempty list")
        return
    statuses: set[str] = set()
    for index, value in enumerate(candidates):
        candidate = cast("dict[str, object]", value if isinstance(value, dict) else {})
        label = f"candidate_registry.candidates[{index}]"
        failures.extend(
            f"{label}: {field} is required"
            for field in ("name", "scope", "reason")
            if not str(candidate.get(field, "")).strip()
        )
        status = str(candidate.get("status", ""))
        if status not in {"selected", "rejected"}:
            failures.append(f"{label}: status must be selected or rejected")
        else:
            statuses.add(status)
        _validate_evidence_reference(
            cast("dict[str, object]", candidate.get("evidence", {})),
            root,
            f"{label}.evidence",
            failures,
        )
    if statuses != {"selected", "rejected"}:
        failures.append("candidate registry: selected and rejected candidates are required")


def _validate_reproducibility(
    payload: dict[str, object],
    environment: dict[str, object],
    failures: list[str],
) -> None:
    if not str(payload.get("code_revision", "")).strip():
        failures.append("reproducibility: code_revision is required")
    dependencies = cast("dict[str, object]", payload.get("dependencies", {}))
    failures.extend(
        f"reproducibility: dependencies.{key} mismatches hardware"
        for key in ("torch", "cuda")
        if dependencies.get(key) != environment.get(key)
    )
    seeds = payload.get("random_seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        failures.append("reproducibility: random_seeds must be a nonempty integer list")
    commands = payload.get("commands")
    if not isinstance(commands, list) or not commands:
        failures.append("reproducibility: commands must be a nonempty list")
        return
    covered: set[str] = set()
    for index, value in enumerate(commands):
        command = cast("dict[str, object]", value if isinstance(value, dict) else {})
        label = f"reproducibility.commands[{index}]"
        key = str(command.get("artifact_key", ""))
        if key:
            covered.add(key)
        argv = command.get("argv")
        if not str(command.get("name", "")).strip():
            failures.append(f"{label}: name is required")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(part, str) and part.strip() for part in argv
        ):
            failures.append(f"{label}: argv must be a nonempty list")
    required_coverage = {
        "efp16_inference",
        "efp16_training",
        "model_inference",
        "model_training",
        "profiler",
        "test_report",
    }
    if not required_coverage.issubset(covered):
        missing = sorted(required_coverage - covered)
        failures.append(f"reproducibility: command coverage missing {missing}")


def _validate_test_report(
    payload: dict[str, object], root: Path, failures: list[str]
) -> None:
    if payload.get("status") != "PASS":
        failures.append("test report: status must be PASS")
    suites = payload.get("suites")
    if not isinstance(suites, list) or not suites:
        failures.append("test report: suites must be nonempty")
        return
    for index, value in enumerate(suites):
        suite = cast("dict[str, object]", value if isinstance(value, dict) else {})
        label = f"test_report.suites[{index}]"
        if not str(suite.get("name", "")).strip():
            failures.append(f"{label}: name is required")
        if _as_int(suite.get("passed")) < 1 or _as_int(suite.get("failed")) != 0:
            failures.append(f"{label}: must record passed>=1 and failed=0")
        command = suite.get("command")
        if not isinstance(command, list) or not command:
            failures.append(f"{label}: command argv is required")
        _validate_evidence_reference(
            cast("dict[str, object]", suite.get("log", {})),
            root,
            f"{label}.log",
            failures,
        )


def _validate_common_context(
    payload: dict[str, object],
    label: str,
    reference_environment: dict[str, object],
    reference_protocol: dict[str, object],
    failures: list[str],
) -> None:
    environment = cast("dict[str, object]", payload.get("environment", {}))
    protocol = cast("dict[str, object]", payload.get("protocol", {}))
    _validate_environment(environment, label, failures)
    _validate_protocol(protocol, label, failures, measurement=True)
    failures.extend(
        f"{label}: environment.{key} mismatches hardware"
        for key in _ENVIRONMENT_KEYS
        if environment.get(key) != reference_environment.get(key)
    )
    failures.extend(
        f"{label}: protocol.{key} mismatches hardware"
        for key in _COMMON_PROTOCOL_KEYS
        if protocol.get(key) != reference_protocol.get(key)
    )


def _validate_environment(
    environment: dict[str, object], label: str, failures: list[str]
) -> None:
    device = str(environment.get("device", "")).lower().replace(" ", "")
    if "rtx4090" not in device:
        failures.append(f"{label}: device must be an RTX 4090")
    failures.extend(
        f"{label}: environment.{key} is required"
        for key in ("host", "torch", "cuda")
        if not str(environment.get(key, "")).strip()
    )


def _validate_protocol(
    protocol: dict[str, object],
    label: str,
    failures: list[str],
    *,
    measurement: bool,
) -> None:
    if str(protocol.get("dtype", "")).lower() != "float32":
        failures.append(f"{label}: protocol.dtype must be float32")
    expected = {
        "tf32": False,
        "autocast": False,
        "synchronized": True,
        "compile_cost_excluded": True,
        "raw_samples": True,
        "normalized": False,
    }
    for key, value in expected.items():
        if protocol.get(key) is not value:
            failures.append(f"{label}: protocol.{key} must be {value}")
    if measurement:
        failures.extend(
            f"{label}: protocol.{key} must be positive"
            for key in ("warmup_steps", "groups", "iterations_per_group")
            if _as_int(protocol.get(key)) < 1
        )
        if _as_int(protocol.get("groups")) < 3:
            failures.append(f"{label}: protocol.groups must be >=3")


def _validate_appendix(
    root: Path,
    appendix_path: Path,
    bundle: dict[str, object],
    failures: list[str],
) -> str | None:
    appendix = appendix_path.resolve()
    if not appendix.is_file():
        failures.append(f"appendix does not exist: {appendix_path}")
        return None
    digest = hashlib.sha256(appendix.read_bytes()).hexdigest()
    expected = str(bundle.get("appendix_sha256", ""))
    if not _SHA256_PATTERN.fullmatch(expected) or digest != expected:
        failures.append("appendix sha256 mismatch")
    text = appendix.read_text(encoding="utf-8")
    for section in REQUIRED_APPENDIX_SECTIONS:
        pattern = re.compile(rf"^#{{1,6}}\s+{re.escape(section)}\s*$", re.IGNORECASE | re.MULTILINE)
        if not pattern.search(text):
            failures.append(f"appendix missing section: {section}")
    linked = _resolved_markdown_links(appendix, text)
    required_links = {root / BUNDLE_FILE} | {
        root / filename for filename in REQUIRED_ARTIFACTS.values()
    }
    failures.extend(
        f"appendix missing artifact link: {required.name}"
        for required in sorted(required_links)
        if required.resolve() not in linked
    )
    return digest


def _resolved_markdown_links(appendix: Path, text: str) -> set[Path]:
    resolved: set[Path] = set()
    for match in _MARKDOWN_LINK_PATTERN.finditer(text):
        target = match.group(1).strip().strip("<>").split("#", maxsplit=1)[0]
        if not target or "://" in target:
            continue
        path = Path(target)
        resolved.add((path if path.is_absolute() else appendix.parent / path).resolve())
    return resolved


def _indexed_rows(
    payload: dict[str, object],
    fields: tuple[str, ...],
    expected: set[tuple[object, ...]],
    label: str,
    failures: list[str],
) -> dict[tuple[object, ...], dict[str, object]]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        failures.append(f"{label}: rows must be a list")
        return {}
    rows: dict[tuple[object, ...], dict[str, object]] = {}
    for index, value in enumerate(raw_rows):
        if not isinstance(value, dict):
            failures.append(f"{label}: row {index} is not an object")
            continue
        row = cast("dict[str, object]", value)
        key = tuple(
            _as_int(row.get(field)) if field in {"length", "batch_size"} else row.get(field)
            for field in fields
        )
        if key in rows:
            failures.append(f"{label}: duplicate row {key}")
        rows[key] = row
    if set(rows) != expected:
        missing = sorted(expected - set(rows), key=str)
        extra = sorted(set(rows) - expected, key=str)
        failures.append(f"{label}: row coverage mismatch; missing={missing}, extra={extra}")
    return rows


def _validate_evidence_reference(
    evidence: dict[str, object],
    root: Path,
    label: str,
    failures: list[str],
) -> None:
    path = _checked_root_path(root, evidence.get("path"), label, failures)
    if path is not None:
        _verify_sha256(path, str(evidence.get("sha256", "")), label, failures)


def _checked_root_path(
    root: Path,
    value: object,
    label: str,
    failures: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        failures.append(f"{label}: path is required")
        return None
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        failures.append(f"{label}: path escapes artifact root")
        return None
    if not path.is_file():
        failures.append(f"{label}: file does not exist: {value}")
        return None
    return path


def _verify_sha256(
    path: Path, expected: str, label: str, failures: list[str]
) -> bool:
    if not _SHA256_PATTERN.fullmatch(expected):
        failures.append(f"{label}: sha256 must be 64 lowercase hex characters")
        return False
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        failures.append(f"{label}: sha256 mismatch")
        return False
    return True


def _load_json(
    path: Path, failures: list[str], label: str
) -> dict[str, object] | None:
    if not path.is_file():
        failures.append(f"{label}: JSON file does not exist")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        failures.append(f"{label}: invalid JSON: {error}")
        return None
    if not isinstance(value, dict):
        failures.append(f"{label}: JSON root must be an object")
        return None
    return cast("dict[str, object]", value)


def _positive_samples(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    converted = [_positive_float(item) for item in value]
    if any(item is None for item in converted):
        return None
    return cast("list[float]", converted)


def _positive_float(value: object) -> float | None:
    converted = _finite_float(value)
    return converted if converted is not None and converted > 0.0 else None


def _finite_float(value: object) -> float | None:
    try:
        converted = float(cast("float | int | str", value))
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _as_int(value: object) -> int:
    try:
        return int(cast("int | str", value))
    except (TypeError, ValueError):
        return 0


def _result(
    failures: list[str],
    checked_hashes: dict[str, str],
    counts: dict[str, int],
    appendix_path: Path,
    *,
    appendix_digest: str | None = None,
) -> dict[str, object]:
    return {
        "schema": EVALUATION_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_artifact_sha256": checked_hashes,
        "checked_counts": counts,
        "appendix": str(appendix_path),
        "appendix_sha256": appendix_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--appendix", type=Path, required=True)
    arguments = parser.parse_args()
    root = cast("Path", arguments.artifact_root)
    appendix = cast("Path", arguments.appendix)
    result = evaluate_final_appendix_bundle(root, appendix)
    root.mkdir(parents=True, exist_ok=True)
    (root / EVALUATION_FILE).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "ARTIFACT_SCHEMAS",
    "BATCH_SIZES",
    "BUNDLE_FILE",
    "BUNDLE_SCHEMA",
    "EFP_STAGES",
    "EVALUATION_FILE",
    "EVALUATION_SCHEMA",
    "LENGTHS",
    "MODELS",
    "MODEL_STAGES",
    "REQUIRED_APPENDIX_SECTIONS",
    "REQUIRED_ARTIFACTS",
    "evaluate_final_appendix_bundle",
]
