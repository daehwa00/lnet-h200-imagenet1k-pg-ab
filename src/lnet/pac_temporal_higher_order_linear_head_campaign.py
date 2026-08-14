"""Mechanism-fair P1-3 campaign with affine final heads.

The v1 P1-3 design is retained unchanged for provenance.  This separate v2
phase removes the learned nonlinear budget adapter from every final head:
capacity is allocated only inside the reader body, and every descriptor is
classified by one plain ``nn.Linear`` layer.
"""

# pyright: reportExplicitAny=false, reportImplicitStringConcatenation=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import torch
from scipy.stats import t as student_t
from torch import Tensor, nn
from torch.nn import functional

from . import pac_temporal_controls_campaign as v1
from .pac_campaign_utils import seed_everything, source_file_hashes, write_once

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_ROOT: Final = Path(".omx/results/pac-temporal-higher-order-linear-head-v2-local_gpu-20260725")
V1_ROOT: Final = Path(".omx/results/pac-temporal-controls-local_gpu-20260724")
DEFAULT_SEEDS: Final = (
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
)
VARIANTS: Final = v1.HIGHER_ORDER_VARIANTS
READER_VARIANTS: Final = (
    "full_writer_reader",
    "mlp_reader",
    "conv_reader",
)
READER_BODY_PARAMETER_TARGET: Final = 8192
PARAMETER_TOLERANCE: Final = 0.03
ONE_SCAN_CHANCE_INTERVAL: Final = (0.47, 0.53)

Variant = v1.HigherOrderVariant


@dataclass(frozen=True, slots=True)
class LinearHeadJob:
    seed: int
    family: Literal["p1_3_linear_head_v2"] = "p1_3_linear_head_v2"

    @property
    def key(self) -> str:
        return f"{self.family}__seed{self.seed}"


class PointwiseReaderBody(nn.Module):
    """Capacity-bearing pointwise body shared by full and MLP controls."""

    def __init__(self, state_dim: int, hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(state_dim, hidden)
        self.output = nn.Linear(hidden, state_dim)

    def forward(self, values: Tensor) -> Tensor:
        return self.output(functional.silu(self.input(values)))


class ConvolutionalReaderBody(nn.Module):
    """Capacity-bearing temporal convolution body for the matched control."""

    def __init__(self, state_dim: int, hidden: int) -> None:
        super().__init__()
        self.temporal = nn.Conv1d(
            state_dim,
            hidden,
            kernel_size=3,
            padding=1,
        )
        self.output = nn.Conv1d(hidden, state_dim, kernel_size=1)

    def forward(self, values: Tensor) -> Tensor:
        channels_first = values.transpose(1, 2)
        refined = self.output(functional.silu(self.temporal(channels_first)))
        return refined.transpose(1, 2)


class LinearHeadHigherOrderClassifier(nn.Module):
    """Fixed linear writer, capacity-bearing reader body, affine final head."""

    def __init__(
        self,
        variant: Variant,
        *,
        seed: int,
        modes: int = v1.FIXED_MODES,
        body_parameter_target: int = READER_BODY_PARAMETER_TARGET,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            message = f"unknown higher-order variant: {variant}"
            raise ValueError(message)
        self.variant = variant
        self.modes = modes
        self.state_dim = 2 * modes
        self.body_parameter_target = body_parameter_target
        self.writer = v1.FixedComplexPoleBank(torch.ones(modes, 1))
        generator = torch.Generator().manual_seed(seed * 1009 + 17)
        directions = torch.randn(
            modes,
            self.state_dim,
            generator=generator,
        )
        self.reader_bank = v1.FixedComplexPoleBank(directions)
        self.reader_body: nn.Module | None = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed * 2017 + 31)
            if variant in {"full_writer_reader", "mlp_reader"}:
                hidden = _nearest_hidden_width(
                    body_parameter_target,
                    fixed=self.state_dim,
                    coefficient=2 * self.state_dim + 1,
                )
                self.reader_body = PointwiseReaderBody(
                    self.state_dim,
                    hidden,
                )
            elif variant == "conv_reader":
                hidden = _nearest_hidden_width(
                    body_parameter_target,
                    fixed=self.state_dim,
                    coefficient=4 * self.state_dim + 1,
                )
                self.reader_body = ConvolutionalReaderBody(
                    self.state_dim,
                    hidden,
                )
            self.head = nn.Linear(self._descriptor_dim(), 2)

    def _descriptor_dim(self) -> int:
        writer_moments = 7 * self.modes
        if self.variant == "writer_energy_only":
            return self.modes
        if self.variant == "writer_energy_lag":
            return writer_moments
        if self.variant == "one_scan_full":
            return writer_moments + 2 * self.state_dim
        if self.variant == "full_writer_reader":
            return 2 * writer_moments
        return writer_moments + 5 * self.state_dim

    def descriptor(self, inputs: Tensor) -> Tensor:
        writer_real, writer_imag = self.writer(inputs)
        writer_moments = v1.complex_modal_moments(
            writer_real,
            writer_imag,
        )
        if self.variant == "writer_energy_only":
            return writer_moments[:, : self.modes]
        if self.variant == "writer_energy_lag":
            return writer_moments
        writer_path = torch.cat((writer_real, writer_imag), dim=-1)
        if self.variant == "one_scan_full":
            return torch.cat(
                (
                    writer_moments,
                    writer_path.mean(dim=1),
                    writer_path[:, -1],
                ),
                dim=-1,
            )
        if self.reader_body is None:
            message = "reader variant lost its capacity-bearing body"
            raise RuntimeError(message)
        refined = self.reader_body(writer_path)
        if self.variant == "full_writer_reader":
            reader_real, reader_imag = self.reader_bank(refined)
            reader_moments = v1.complex_modal_moments(
                reader_real,
                reader_imag,
            )
            return torch.cat(
                (writer_moments, reader_moments),
                dim=-1,
            )
        return torch.cat(
            (
                writer_moments,
                v1.real_temporal_moments(refined),
            ),
            dim=-1,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.descriptor(inputs))

    def mechanism_contract(self) -> dict[str, object]:
        body_parameters = _parameter_count(self.reader_body) if self.reader_body is not None else 0
        relative_error = (
            (body_parameters - self.body_parameter_target) / self.body_parameter_target
            if self.reader_body is not None
            else 0.0
        )
        return {
            "writer": "fixed linear complex pole bank",
            "reader_body": (
                type(self.reader_body).__name__ if self.reader_body is not None else None
            ),
            "reader_body_parameter_target": (
                self.body_parameter_target if self.reader_body is not None else None
            ),
            "reader_body_parameters": body_parameters,
            "reader_body_relative_error": relative_error,
            "reader_body_within_three_percent": (
                abs(relative_error) <= PARAMETER_TOLERANCE if self.reader_body is not None else True
            ),
            "final_head_type": type(self.head).__name__,
            "final_head_is_exact_nn_linear": (type(self.head) is nn.Linear),
            "nonlinear_head_adapter_present": False,
            "final_head_parameters": _parameter_count(self.head),
            "total_trainable_parameters": _parameter_count(self),
        }


def campaign_jobs(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[LinearHeadJob]:
    return [LinearHeadJob(seed=seed) for seed in seeds]


def prepare_campaign(
    root: Path = DEFAULT_ROOT,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    lane_count: int = 1,
) -> dict[str, object]:
    """Freeze the separate mechanism-fair P1-3 v2 phase."""
    if not seeds or len(set(seeds)) != len(seeds):
        message = "seeds must be a nonempty unique tuple"
        raise ValueError(message)
    if lane_count < 1:
        message = "lane_count must be positive"
        raise ValueError(message)
    if root.resolve() == V1_ROOT.resolve():
        message = "linear-head v2 must not write into the v1 root"
        raise ValueError(message)
    jobs = campaign_jobs(seeds)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("attempts", "completed", "failed", "manifests", "reports"):
        (root / name).mkdir(exist_ok=True)
    write_once(
        root / "queue.jsonl",
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
    )
    for lane in range(lane_count):
        write_once(
            root / "manifests" / f"worker-{lane:02d}.jsonl",
            "".join(
                json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs[lane::lane_count]
            ),
        )
    sources = _source_hashes()
    contract: dict[str, object] = {
        "schema": "pac_temporal_higher_order_linear_head_contract.v2",
        "campaign_family": "p1_3_linear_head_v2",
        "output_root": str(root),
        "paper_sources_modified": False,
        "hardware_target": "local_gpu CUDA GPU",
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "overlap_with_v1_seeds": sorted(set(seeds).intersection(v1.DEFAULT_SEEDS)),
        "confirmatory_seed_requirement": 10,
        "confirmatory_seed_requirement_met": len(seeds) >= 10,
        "jobs": len(jobs),
        "lane_count": lane_count,
        "predecessor": {
            "root": str(V1_ROOT),
            "contract_sha256": _hash_if_exists(V1_ROOT / "contract.json"),
            "disposition": ("v1 P1-3 design superseded and excluded from v2"),
            "launch_blocker": (
                "full/MLP/conv used ActiveBudgetHead learned SiLU "
                "adapters, confounding reader mechanism with head capacity"
            ),
            "rows_reused": 0,
        },
        "mechanism_contract": {
            "controlled_writer": "fixed linear complex pole writer",
            "variants": list(VARIANTS),
            "all_final_heads": "exact nn.Linear(descriptor_dim,2)",
            "head_nonlinearity": None,
            "head_budget_adapter": None,
            "reader_body_parameter_target": (READER_BODY_PARAMETER_TARGET),
            "reader_body_parameter_tolerance": PARAMETER_TOLERANCE,
            "budget_location": ("reader body hidden width only; never final head"),
            "full_and_mlp_share_identical_pointwise_body_family": True,
            "difference_under_test": (
                "second reader pole scan and its quadratic moments versus "
                "parameter-matched no-scan MLP/conv temporal summaries"
            ),
        },
        "process": {
            "law": ("common AR(1) phi=0.6 with unit Gaussian versus unit Rademacher innovations"),
            "population_second_order": ("identical Gamma(k)=0.6^|k|"),
            "counts": {
                "train": v1.HIGHER_ORDER_TRAIN_COUNT,
                "validation": v1.HIGHER_ORDER_VALIDATION_COUNT,
                "test": v1.HIGHER_ORDER_TEST_COUNT,
                "length": v1.HIGHER_ORDER_LENGTH,
            },
        },
        "criteria": {
            "second_order_equivalence_established_every_seed": True,
            "one_scan_full_mean_interval": list(ONE_SCAN_CHANCE_INTERVAL),
            "one_scan_full_student_t_90pct_ci_contained_in": list(ONE_SCAN_CHANCE_INTERVAL),
            "full_minus_one_scan_mean_minimum": 0.10,
            "full_minus_best_matched_reader_mean_strictly_above": 0.0,
            "all_final_heads_exact_nn_linear": True,
            "all_reader_bodies_within_three_percent": True,
        },
        "source_sha256": sources,
        "source_manifest_sha256": _mapping_sha256(sources),
    }
    write_once(
        root / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def run_job(
    job: LinearHeadJob,
    *,
    device: str,
) -> dict[str, object]:
    seed_everything(job.seed)
    train_inputs, train_labels = v1.higher_order_samples(
        v1.HIGHER_ORDER_TRAIN_COUNT,
        v1.HIGHER_ORDER_LENGTH,
        seed=job.seed * 211 + 1,
    )
    validation_inputs, validation_labels = v1.higher_order_samples(
        v1.HIGHER_ORDER_VALIDATION_COUNT,
        v1.HIGHER_ORDER_LENGTH,
        seed=job.seed * 211 + 2,
    )
    test_inputs, test_labels = v1.higher_order_samples(
        v1.HIGHER_ORDER_TEST_COUNT,
        v1.HIGHER_ORDER_LENGTH,
        seed=job.seed * 211 + 3,
    )
    second_order = v1.spectrum_equality_diagnostics(
        torch.cat((train_inputs, validation_inputs)),
        torch.cat((train_labels, validation_labels)),
        seed=job.seed * 211 + 4,
    )
    models: dict[str, object] = {}
    for index, variant in enumerate(VARIANTS):
        model = LinearHeadHigherOrderClassifier(
            variant,
            seed=job.seed,
        )
        training = v1.fit_classifier(
            model,
            train_inputs,
            train_labels,
            validation_inputs,
            validation_labels,
            seed=job.seed * 31 + index,
            device=device,
            epochs=v1.HIGHER_ORDER_EPOCHS,
            batch_size=64,
            learning_rate=3e-3,
            weight_decay=1e-4,
        )
        test_score = v1.evaluate_balanced_accuracy(
            model,
            test_inputs,
            test_labels,
            device=device,
            batch_size=128,
        )
        models[variant] = {
            **training,
            "test_balanced_accuracy": test_score,
            "mechanism_contract": model.mechanism_contract(),
        }
    return {
        "schema": ("pac_temporal_higher_order_linear_head_result.v2"),
        "job_key": job.key,
        "family": job.family,
        "seed": job.seed,
        "status": "done",
        "device": device,
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device()) if device == "cuda" else None
        ),
        "second_order_equality": second_order,
        "models": models,
    }


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    if device == "cuda" and not torch.cuda.is_available():
        message = "CUDA was requested but is unavailable"
        raise RuntimeError(message)
    contract, contract_hash = _load_contract(root)
    source_hash = str(contract["source_manifest_sha256"])
    for job in _read_jobs(manifest):
        output = root / "completed" / f"{job.key}.json"
        if output.exists():
            row = json.loads(output.read_text(encoding="utf-8"))
            error = _row_error(
                output,
                row,
                job,
                contract_hash,
                source_hash,
            )
            if error is not None:
                message = f"invalid completed row: {error}"
                raise RuntimeError(message)
            continue
        attempt_dir = root / "attempts" / job.key
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_id = f"{os.getpid()}-{int(perf_counter() * 1_000_000)}"
        _write_json(
            attempt_dir / f"{attempt_id}.started.json",
            {
                "schema": ("pac_temporal_higher_order_linear_head_attempt.v2"),
                "job": asdict(job),
                "status": "started",
                "device": device,
            },
        )
        started = perf_counter()
        try:
            row = run_job(job, device=device)
            row["elapsed_seconds"] = perf_counter() - started
            row["frozen_contract_sha256"] = contract_hash
            row["frozen_source_manifest_sha256"] = source_hash
            _write_json(output, row)
            _write_json(
                attempt_dir / f"{attempt_id}.succeeded.json",
                {
                    "schema": ("pac_temporal_higher_order_linear_head_attempt.v2"),
                    "job_key": job.key,
                    "status": "succeeded",
                    "elapsed_seconds": row["elapsed_seconds"],
                },
            )
            failure = root / "failed" / f"{job.key}.json"
            if failure.exists():
                failure.unlink()
        except Exception as error:  # noqa: BLE001
            failure_payload = {
                "schema": ("pac_temporal_higher_order_linear_head_failure.v2"),
                "job": asdict(job),
                "job_key": job.key,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": perf_counter() - started,
            }
            _write_json(
                root / "failed" / f"{job.key}.json",
                failure_payload,
            )
            _write_json(
                attempt_dir / f"{attempt_id}.failed.json",
                failure_payload,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return campaign_status(root)


def campaign_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    audit = _audit(root)
    expected = {job.key for job in audit["jobs"]}
    completed = {str(row["job_key"]) for row in audit["valid_rows"]}
    failed = {
        path.stem
        for path in (root / "failed").glob("*.json")
        if path.stem in expected and path.stem not in completed
    }
    return {
        "expected": len(expected),
        "completed": len(completed),
        "failed": len(failed),
        "invalid_completed": len(audit["invalid_rows"]),
        "invalid_completed_reasons": audit["invalid_rows"],
        "remaining": len(expected - completed),
        "contract_valid": audit["contract_valid"],
        "source_manifest_matches_current_files": audit["source_manifest_matches_current_files"],
        "done": (
            completed == expected
            and not failed
            and not audit["invalid_rows"]
            and bool(audit["contract_valid"])
            and bool(audit["source_manifest_matches_current_files"])
        ),
    }


def report_campaign(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    status = campaign_status(root)
    rows = cast("list[dict[str, Any]]", _audit(root)["valid_rows"])
    aggregate = _aggregate(rows)
    complete = bool(status["done"]) and len(rows) >= 10
    all_gates = bool(aggregate.get("all_prespecified_criteria_passed", False))
    payload: dict[str, object] = {
        "schema": ("pac_temporal_higher_order_linear_head_report.v2"),
        "status": status,
        "report_state": "complete" if complete else "partial",
        "complete_evidence": complete,
        "complete_claims_authorized": complete and all_gates,
        "p1_3_linear_head": aggregate,
        "interpretation_policy": (
            "v1 rows are excluded; incomplete evidence or any failed "
            "equivalence/mechanism/performance gate blocks claims"
        ),
    }
    _write_json(
        root / "reports" / "LINEAR_HEAD_P1_3_V2_REPORT.json",
        payload,
    )
    _write_markdown(
        root / "reports" / "LINEAR_HEAD_P1_3_V2_REPORT.md",
        payload,
    )
    return payload


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        return {
            "rows": 0,
            "all_prespecified_criteria_passed": False,
        }
    scores = {
        variant: [float(row["models"][variant]["test_balanced_accuracy"]) for row in rows]
        for variant in VARIANTS
    }
    summaries = {variant: _student_summary(active) for variant, active in scores.items()}
    one_scan = summaries["one_scan_full"]
    one_scan_gate = (
        ONE_SCAN_CHANCE_INTERVAL[0] <= one_scan["mean"] <= ONE_SCAN_CHANCE_INTERVAL[1]
        and one_scan["equivalence_ci90_low"] >= ONE_SCAN_CHANCE_INTERVAL[0]
        and one_scan["equivalence_ci90_high"] <= ONE_SCAN_CHANCE_INTERVAL[1]
    )
    full_minus_one = [
        full - one
        for full, one in zip(
            scores["full_writer_reader"],
            scores["one_scan_full"],
            strict=True,
        )
    ]
    full_minus_best = [
        full - max(mlp, conv)
        for full, mlp, conv in zip(
            scores["full_writer_reader"],
            scores["mlp_reader"],
            scores["conv_reader"],
            strict=True,
        )
    ]
    full_one_summary = _student_summary(full_minus_one)
    full_best_summary = _student_summary(full_minus_best)
    equivalence_flags = [
        bool(row["second_order_equality"]["equivalence"]["second_order_equivalence_established"])
        for row in rows
    ]
    mechanism_rows = [
        row["models"][variant]["mechanism_contract"] for row in rows for variant in VARIANTS
    ]
    all_linear = all(
        bool(active["final_head_is_exact_nn_linear"])
        and not bool(active["nonlinear_head_adapter_present"])
        for active in mechanism_rows
    )
    matched_rows = [
        row["models"][variant]["mechanism_contract"] for row in rows for variant in READER_VARIANTS
    ]
    all_matched = all(bool(active["reader_body_within_three_percent"]) for active in matched_rows)
    maximum_body_error = max(
        abs(float(active["reader_body_relative_error"])) for active in matched_rows
    )
    equivalence_gate = all(equivalence_flags)
    full_one_gate = full_one_summary["mean"] >= 0.10
    full_best_gate = full_best_summary["mean"] > 0.0
    return {
        "rows": len(rows),
        "model_test_balanced_accuracy": summaries,
        "one_scan_near_chance": {
            **one_scan,
            "prespecified_interval": list(ONE_SCAN_CHANCE_INTERVAL),
            "criterion_passed": one_scan_gate,
        },
        "paired_contrasts": {
            "full_minus_one_scan": {
                **full_one_summary,
                "mean_at_least_ten_points": full_one_gate,
            },
            "full_minus_best_matched_reader": {
                **full_best_summary,
                "mean_strictly_positive": full_best_gate,
            },
        },
        "second_order_equivalence": {
            "established_fraction": (sum(equivalence_flags) / len(equivalence_flags)),
            "established_every_seed": equivalence_gate,
            "non_rejection_used_as_equivalence": False,
        },
        "mechanism": {
            "all_final_heads_exact_nn_linear": all_linear,
            "all_reader_bodies_within_three_percent": all_matched,
            "maximum_absolute_reader_body_relative_error": (maximum_body_error),
            "budget_in_final_head_adapter": False,
        },
        "all_prespecified_criteria_passed": (
            one_scan_gate
            and full_one_gate
            and full_best_gate
            and equivalence_gate
            and all_linear
            and all_matched
        ),
    }


def _write_markdown(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    aggregate = cast(
        "dict[str, Any]",
        payload["p1_3_linear_head"],
    )
    status = cast("dict[str, Any]", payload["status"])
    lines = [
        "# P1-3 affine-head mechanism control (v2)",
        "",
        f"Report state: **{str(payload['report_state']).upper()}**.",
        f"Complete claims authorized: **{payload['complete_claims_authorized']}**.",
        f"Status: {status['completed']}/{status['expected']} complete; "
        f"{status['invalid_completed']} invalid.",
        "",
        "All final classifiers are exact nn.Linear layers. Capacity matching "
        "is confined to reader-body hidden width; v1 ActiveBudgetHead rows "
        "are excluded.",
        "",
    ]
    if aggregate.get("rows"):
        lines.extend(
            (
                "| Variant | TEST BA | Student-t 95% CI |",
                "|---|---:|---:|",
            )
        )
        for variant in VARIANTS:
            summary = aggregate["model_test_balanced_accuracy"][variant]
            lines.append(
                f"| {variant} | {summary['mean']:.3f} | "
                f"[{summary['ci95_low']:.3f},"
                f"{summary['ci95_high']:.3f}] |"
            )
        lines.extend(
            (
                "",
                "Second-order equivalence established every seed: "
                f"**{aggregate['second_order_equivalence']['established_every_seed']}**.",
                "One-scan chance-equivalence gate: "
                f"**{aggregate['one_scan_near_chance']['criterion_passed']}**.",
                "All heads affine and bodies parameter-matched: "
                f"**{aggregate['mechanism']['all_final_heads_exact_nn_linear']}** / "
                f"**{aggregate['mechanism']['all_reader_bodies_within_three_percent']}**.",
                "All prespecified gates passed: "
                f"**{aggregate['all_prespecified_criteria_passed']}**.",
            )
        )
    else:
        lines.append("No valid v2 rows are available.")
    if payload["report_state"] == "partial":
        lines.extend(
            (
                "",
                "**PARTIAL:** no complete mechanism claim is authorized.",
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _audit(root: Path) -> dict[str, Any]:
    invalid: list[str] = []
    try:
        jobs = _read_jobs(root / "queue.jsonl")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "jobs": [],
            "valid_rows": [],
            "invalid_rows": [f"queue: {type(error).__name__}: {error}"],
            "contract_valid": False,
            "source_manifest_matches_current_files": False,
        }
    by_key = {job.key: job for job in jobs}
    if len(by_key) != len(jobs):
        invalid.append("queue contains duplicate keys")
    try:
        contract, contract_hash = _load_contract(root)
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        contract = {}
        contract_hash = ""
        invalid.append(f"contract: {type(error).__name__}: {error}")
        contract_valid = False
        source_matches = False
    else:
        contract_keys = {LinearHeadJob(seed=int(seed)).key for seed in contract.get("seeds", [])}
        contract_valid = contract_keys == set(by_key) and int(contract.get("jobs", -1)) == len(jobs)
        if not contract_valid:
            invalid.append("contract grid differs from queue")
        source_matches = contract.get("source_sha256") == _source_hashes()
    source_hash = str(contract.get("source_manifest_sha256", ""))
    valid: list[dict[str, Any]] = []
    for path in sorted((root / "completed").glob("*.json")):
        job = by_key.get(path.stem)
        if job is None:
            invalid.append(f"{path.name}: key absent from queue")
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            invalid.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        error = _row_error(
            path,
            row,
            job,
            contract_hash,
            source_hash,
        )
        if error is None:
            valid.append(row)
        else:
            invalid.append(error)
    return {
        "jobs": jobs,
        "valid_rows": valid,
        "invalid_rows": invalid,
        "contract_valid": contract_valid,
        "source_manifest_matches_current_files": source_matches,
    }


def _load_contract(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "contract.json"
    encoded = path.read_bytes()
    contract = json.loads(encoded)
    if contract.get("schema") != "pac_temporal_higher_order_linear_head_contract.v2":
        message = "invalid linear-head v2 contract schema"
        raise RuntimeError(message)
    sources = _source_hashes()
    if contract.get("source_sha256") != sources or contract.get(
        "source_manifest_sha256"
    ) != _mapping_sha256(sources):
        message = "linear-head v2 source manifest mismatch"
        raise RuntimeError(message)
    return contract, hashlib.sha256(encoded).hexdigest()


def _row_error(
    path: Path,
    row: Mapping[str, Any],
    job: LinearHeadJob,
    contract_hash: str,
    source_hash: str,
) -> str | None:
    expected = {
        "schema": ("pac_temporal_higher_order_linear_head_result.v2"),
        "job_key": job.key,
        "family": job.family,
        "seed": job.seed,
        "status": "done",
        "frozen_contract_sha256": contract_hash,
        "frozen_source_manifest_sha256": source_hash,
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            return f"{path.name}: {field}={row.get(field)!r}, expected {expected_value!r}"
    return None


def _student_summary(values: list[float]) -> dict[str, float]:
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        standard_error = deviation / math.sqrt(len(values))
        critical_95 = float(student_t.ppf(0.975, len(values) - 1))
        critical_90 = float(student_t.ppf(0.95, len(values) - 1))
    else:
        standard_error = math.inf
        critical_95 = math.inf
        critical_90 = math.inf
    return {
        "mean": average,
        "sample_sd": deviation,
        "ci95_low": average - critical_95 * standard_error,
        "ci95_high": average + critical_95 * standard_error,
        "one_sided_95_lower": (average - critical_90 * standard_error),
        "equivalence_ci90_low": (average - critical_90 * standard_error),
        "equivalence_ci90_high": (average + critical_90 * standard_error),
    }


def _nearest_hidden_width(
    target: int,
    *,
    fixed: int,
    coefficient: int,
) -> int:
    candidates = {max(1, int((target - fixed) / coefficient) + offset) for offset in (-1, 0, 1, 2)}
    return min(
        candidates,
        key=lambda hidden: abs(fixed + coefficient * hidden - target),
    )


def _parameter_count(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _read_jobs(path: Path) -> list[LinearHeadJob]:
    return [
        LinearHeadJob(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_hashes() -> dict[str, str]:
    return source_file_hashes(
        (
            "src/lnet/pac_temporal_higher_order_linear_head_campaign.py",
            "src/lnet/pac_temporal_controls_campaign.py",
        ),
        project_root=Path(__file__).resolve().parents[2],
        missing="placeholder",
    )


def _mapping_sha256(values: Mapping[str, str]) -> str:
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_if_exists(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "unavailable"


def _write_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "DEFAULT_ROOT",
    "DEFAULT_SEEDS",
    "READER_BODY_PARAMETER_TARGET",
    "READER_VARIANTS",
    "VARIANTS",
    "LinearHeadHigherOrderClassifier",
    "LinearHeadJob",
    "campaign_jobs",
    "campaign_status",
    "prepare_campaign",
    "report_campaign",
    "run_job",
    "run_manifest",
]
