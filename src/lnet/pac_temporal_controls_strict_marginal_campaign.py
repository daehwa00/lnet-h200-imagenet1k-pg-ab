"""Strict finite-path marginal control for the failed temporal-controls pilot.

This is a new v2 campaign.  It never writes into the v1 root and never reuses
v1 rows.  Its sole design change is an explicit, label-blind finite-path rank
remarginalization to one fixed Gaussian-score multiset before training or
controls, with a disclosed repeat after the across-sample timestamp control.
"""

# pyright: reportExplicitAny=false, reportImplicitStringConcatenation=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

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
from torch import Tensor

from . import pac_temporal_controls_campaign as v1
from .pac_campaign_utils import seed_everything, source_file_hashes, write_once

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_ROOT: Final = Path(".omx/results/pac-temporal-controls-strict-marginal-v2-local_gpu-20260725")
V1_FAILED_PILOT_ROOT: Final = Path(".omx/results/pac-temporal-controls-local_gpu-20260724")
DEFAULT_SEEDS: Final = (71, 73, 79, 83, 89, 97, 101, 103, 107, 109)
VIEWS: Final = v1.AR1_VIEWS
CONTROLS: Final = (
    "original",
    "iid_within_path_permutation",
    "pooled_timestamp_reassembly_remarginalized",
)
CONTROL_TO_V1: Final = {
    "original": "original",
    "iid_within_path_permutation": "marginal_autocorrelation_destroyed",
    "pooled_timestamp_reassembly_remarginalized": ("timestamp_rearrangement"),
}

ORIGINAL_BALANCED_ACCURACY_MINIMUM: Final = 0.95
SHUFFLE_BALANCED_ACCURACY_INTERVAL: Final = (0.47, 0.53)
PATH_MEAN_TOLERANCE: Final = 1.0e-6
PATH_GAMMA0_TOLERANCE: Final = 1.0e-5
MODEL_TRAIN_COUNT: Final = 512
VALIDATION_COUNT: Final = 256
PROTOTYPE_CALIBRATION_COUNT: Final = 512
FINAL_TEST_COUNT: Final = 512

Control = Literal[
    "original",
    "iid_within_path_permutation",
    "pooled_timestamp_reassembly_remarginalized",
]

# Frozen numerical summary of the completed ten-seed v1 P1-1 pilot.  These
# values are provenance only; no v1 result row enters v2 estimation.
V1_FAILED_PILOT_MEAN_BALANCED_ACCURACY: Final = {
    "raw_fixed_bank": {
        "original": 0.99921875,
        "iid_within_path_permutation": 0.767578125,
        "pooled_timestamp_reassembly_remarginalized": 0.496875,
    },
    "frozen_orthogonal": {
        "original": 0.969921875,
        "iid_within_path_permutation": 0.694921875,
        "pooled_timestamp_reassembly_remarginalized": 0.501953125,
    },
    "learned_identity": {
        "original": 0.990625,
        "iid_within_path_permutation": 0.611328125,
        "pooled_timestamp_reassembly_remarginalized": 0.521484375,
    },
}


@dataclass(frozen=True, slots=True)
class StrictMarginalJob:
    seed: int
    family: Literal["p1_1_strict_marginal"] = "p1_1_strict_marginal"

    @property
    def key(self) -> str:
        return f"{self.family}__seed{self.seed}"


def campaign_jobs(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[StrictMarginalJob]:
    """Return the prospectively frozen strict-marginal seed grid."""
    return [StrictMarginalJob(seed=seed) for seed in seeds]


def strict_standardize_paths(values: Tensor) -> Tensor:
    """Map each path's ranks to one centered unit-Gamma(0) score multiset."""
    if values.ndim != 3 or values.shape[1] < 2:
        message = "values must have shape [batch,time,channels] with time >= 2"
        raise ValueError(message)
    if not bool(torch.isfinite(values).all()):
        message = "strict path standardization requires finite values"
        raise ValueError(message)
    active = values.to(torch.float64)
    centered = active - active.mean(dim=1, keepdim=True)
    if bool((centered.square().mean(dim=1, keepdim=True) <= torch.finfo(active.dtype).eps).any()):
        message = "strict path standardization requires nonconstant paths"
        raise ValueError(message)
    probability = (
        torch.arange(
            values.shape[1],
            device=values.device,
            dtype=torch.float64,
        )
        + 0.5
    ) / values.shape[1]
    template = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    template = template - template.mean()
    template = template / torch.sqrt(template.square().mean())
    template = template.view(1, -1, 1).expand_as(active)
    ranks_to_positions = torch.argsort(
        active,
        dim=1,
        stable=True,
    )
    remarginalized = torch.empty_like(active).scatter(
        1,
        ranks_to_positions,
        template,
    )
    return remarginalized.to(dtype=values.dtype)


def strict_path_diagnostics(values: Tensor) -> dict[str, float]:
    """Return the worst finite-path first- and zero-lag-moment errors."""
    active = values.to(torch.float64)
    means = active.mean(dim=1)
    centered = active - means.unsqueeze(1)
    gamma_zero = centered.square().mean(dim=1)
    return {
        "maximum_absolute_path_mean": float(means.abs().max()),
        "maximum_absolute_path_gamma0_error": float((gamma_zero - 1.0).abs().max()),
    }


def strict_matched_ar1_samples(
    count: int,
    length: int,
    *,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Draw the v1 AR(1) law, then apply the fixed label-blind v2 projection."""
    values, labels = v1.matched_ar1_samples(count, length, seed=seed)
    return strict_standardize_paths(values), labels


def apply_strict_control(
    values: Tensor,
    labels: Tensor,
    control: Control,
    *,
    seed: int,
) -> Tensor:
    """Apply a disclosed v2 shuffle while keeping strict finite-path moments."""
    _require_strict_paths(values)
    del labels
    generator = torch.Generator().manual_seed(seed)
    if control == "original":
        transformed = values.clone()
    elif control == "iid_within_path_permutation":
        transformed = values.clone()
        for sample in range(values.shape[0]):
            permutation = torch.randperm(
                values.shape[1],
                generator=generator,
            )
            transformed[sample] = values[sample, permutation]
    elif control == "pooled_timestamp_reassembly_remarginalized":
        transformed = values.clone()
        for timestamp in range(values.shape[1]):
            permutation = torch.randperm(
                values.shape[0],
                generator=generator,
            )
            transformed[:, timestamp] = values[
                permutation,
                timestamp,
            ]
        transformed = strict_standardize_paths(transformed)
    else:
        message = f"unknown strict-marginal control: {control}"
        raise ValueError(message)
    _require_strict_paths(transformed)
    return transformed


def prepare_campaign(
    root: Path = DEFAULT_ROOT,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    lane_count: int = 1,
) -> dict[str, object]:
    """Freeze a new v2 contract and write disjoint resumable manifests."""
    if not seeds:
        message = "at least one seed is required"
        raise ValueError(message)
    if len(set(seeds)) != len(seeds):
        message = "campaign seeds must be unique"
        raise ValueError(message)
    if lane_count < 1:
        message = "lane_count must be positive"
        raise ValueError(message)
    if root.resolve() == V1_FAILED_PILOT_ROOT.resolve():
        message = "v2 must not write into the immutable v1 failed-pilot root"
        raise ValueError(message)

    jobs = campaign_jobs(seeds)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("attempts", "completed", "failed", "manifests", "reports"):
        (root / name).mkdir(exist_ok=True)
    queue_text = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs)
    write_once(root / "queue.jsonl", queue_text)
    for lane in range(lane_count):
        manifest_text = "".join(
            json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs[lane::lane_count]
        )
        write_once(
            root / "manifests" / f"worker-{lane:02d}.jsonl",
            manifest_text,
        )

    source_hashes = _source_hashes()
    contract: dict[str, object] = {
        "schema": "pac_temporal_controls_strict_marginal_contract.v2",
        "campaign_family": "p1_1_strict_marginal",
        "output_root": str(root),
        "paper_sources_modified": False,
        "hardware_target": "local_gpu CUDA GPU",
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "overlap_with_v1_pilot_seeds": sorted(set(seeds).intersection(v1.DEFAULT_SEEDS)),
        "confirmatory_seed_requirement": 10,
        "confirmatory_seed_requirement_met": len(seeds) >= 10,
        "jobs": len(jobs),
        "lane_count": lane_count,
        "predecessor": {
            "schema": "pac_temporal_controls_contract.v1",
            "root": str(V1_FAILED_PILOT_ROOT),
            "disposition": "failed pilot; immutable and excluded from v2",
            "failure": (
                "shared and independent within-path shuffles retained "
                "substantial balanced accuracy in all A/B/C views"
            ),
            "completed_p1_1_seed_count": 10,
            "mean_balanced_accuracy": (V1_FAILED_PILOT_MEAN_BALANCED_ACCURACY),
            "retired_shared_permutation_mean_balanced_accuracy": {
                "raw_fixed_bank": 0.781640625,
                "frozen_orthogonal": 0.69921875,
                "learned_identity": 0.622265625,
            },
            "retired_shared_permutation_reason": (
                "one shared permutation is invertible and only reindexes "
                "covariance; it is not a valid destruction control"
            ),
            "rows_reused_in_v2": 0,
            "artifact_sha256_at_v2_freeze": _v1_artifact_hashes(),
        },
        "v1_to_v2_change": {
            "single_design_intent": (
                "make the entire unordered finite-path empirical marginal "
                "identical so only temporal rank order can carry class signal"
            ),
            "preprocessing": (
                "for every path/channel independently, preserve temporal "
                "ranks but replace sorted values by one fixed centered, "
                "unit-Gamma(0) Gaussian-score multiset"
            ),
            "strictly_stronger_than_z_score_only": True,
            "exact_empirical_marginal_shared_by_every_path": True,
            "label_blind": True,
            "cross_sample_statistics_used": False,
            "applied_before_training_validation_test_use": True,
            "applied_before_every_control": True,
            "timestamp_control_exception": (
                "label-blind pooled timestamp reassembly changes individual "
                "path marginals, so the same fixed projection is applied "
                "again afterward and disclosed in its name"
            ),
            "noncausal_offline_preprocessing": True,
            "scope": "mechanism control only; not an online/end-to-end claim",
        },
        "process": {
            "base_law": (
                "stationary Gaussian AR(1), phi in {+0.8,-0.8}, "
                "population mean zero and variance one"
            ),
            "finite_path_law": (
                "base-path temporal ranks after replacement by a common "
                "Gaussian-score multiset with exact per-path mean zero and "
                "unit Gamma(0)"
            ),
            "counts": {
                "model_train": MODEL_TRAIN_COUNT,
                "checkpoint_validation": VALIDATION_COUNT,
                "prototype_calibration": (PROTOTYPE_CALIBRATION_COUNT),
                "final_test": FINAL_TEST_COUNT,
                "length": v1.AR1_LENGTH,
            },
            "split_policy": (
                "model train, checkpoint validation, prototype calibration, "
                "and final TEST are independently generated; TEST is never "
                "used for training or prototype estimation"
            ),
        },
        "views": list(VIEWS),
        "controls": list(CONTROLS),
        "closest_v1_control_for_provenance_only": CONTROL_TO_V1,
        "criteria": {
            "per_view_original_np_ba_mean_strictly_above": (ORIGINAL_BALANCED_ACCURACY_MINIMUM),
            "per_view_original_np_ba_one_sided_95pct_lower_ci_strictly_above": (
                ORIGINAL_BALANCED_ACCURACY_MINIMUM
            ),
            "per_view_each_shuffle_np_ba_mean_interval": list(SHUFFLE_BALANCED_ACCURACY_INTERVAL),
            "per_view_each_shuffle_np_ba_90pct_equivalence_ci_contained_in": list(
                SHUFFLE_BALANCED_ACCURACY_INTERVAL
            ),
            "per_seed_original_pole_input_standardized_mean_difference_max": (0.10),
            "per_seed_original_pole_input_standardized_gamma0_difference_max": (0.10),
            "maximum_absolute_path_mean": PATH_MEAN_TOLERANCE,
            "maximum_absolute_path_gamma0_error": PATH_GAMMA0_TOLERANCE,
            "gate_scope": (
                "all three A/B/C views and every non-original control; "
                "no gate widening after execution"
            ),
        },
        "source_sha256": source_hashes,
        "source_manifest_sha256": _mapping_sha256(source_hashes),
    }
    write_once(
        root / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def campaign_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    """Return a fail-closed audit, not a filename count."""
    audit = _audit_completed_rows(root)
    jobs = audit["jobs"]
    expected = {job.key for job in jobs}
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


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    """Run a v2 manifest, skipping completed rows and preserving attempts."""
    if device == "cuda" and not torch.cuda.is_available():
        message = "CUDA was requested but is unavailable"
        raise RuntimeError(message)
    contract, contract_sha256 = _load_frozen_contract(root)
    source_manifest_sha256 = str(contract["source_manifest_sha256"])
    for job in _read_jobs(manifest):
        output = root / "completed" / f"{job.key}.json"
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            validation_error = _result_validation_error(
                output,
                existing,
                job,
                contract_sha256=contract_sha256,
                source_manifest_sha256=source_manifest_sha256,
            )
            if validation_error is not None:
                message = f"refusing to skip invalid completed row: {validation_error}"
                raise RuntimeError(message)
            continue
        attempt_dir = root / "attempts" / job.key
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_id = f"{os.getpid()}-{int(perf_counter() * 1_000_000)}"
        _write_json(
            attempt_dir / f"{attempt_id}.started.json",
            {
                "schema": ("pac_temporal_controls_strict_marginal_attempt.v2"),
                "job": asdict(job),
                "status": "started",
                "device": device,
                "pid": os.getpid(),
            },
        )
        started = perf_counter()
        try:
            row = run_job(job, device=device)
            row["elapsed_seconds"] = perf_counter() - started
            row["frozen_contract_sha256"] = contract_sha256
            row["frozen_source_manifest_sha256"] = source_manifest_sha256
            _write_json(output, row)
            _write_json(
                attempt_dir / f"{attempt_id}.succeeded.json",
                {
                    "schema": ("pac_temporal_controls_strict_marginal_attempt.v2"),
                    "job_key": job.key,
                    "status": "succeeded",
                    "elapsed_seconds": row["elapsed_seconds"],
                },
            )
            failure_path = root / "failed" / f"{job.key}.json"
            if failure_path.exists():
                failure_path.unlink()
        except Exception as error:  # noqa: BLE001
            failure = {
                "schema": ("pac_temporal_controls_strict_marginal_failure.v2"),
                "job": asdict(job),
                "job_key": job.key,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": perf_counter() - started,
            }
            _write_json(root / "failed" / f"{job.key}.json", failure)
            _write_json(
                attempt_dir / f"{attempt_id}.failed.json",
                failure,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return campaign_status(root)


def run_job(
    job: StrictMarginalJob,
    *,
    device: str,
) -> dict[str, object]:
    """Run one seed without consulting TEST during representation training."""
    seed_everything(job.seed)
    train_inputs, train_labels = strict_matched_ar1_samples(
        MODEL_TRAIN_COUNT,
        v1.AR1_LENGTH,
        seed=job.seed * 101 + 1,
    )
    validation_inputs, validation_labels = strict_matched_ar1_samples(
        VALIDATION_COUNT,
        v1.AR1_LENGTH,
        seed=job.seed * 101 + 2,
    )
    calibration_inputs, calibration_labels = strict_matched_ar1_samples(
        PROTOTYPE_CALIBRATION_COUNT,
        v1.AR1_LENGTH,
        seed=job.seed * 101 + 3,
    )
    test_inputs, test_labels = strict_matched_ar1_samples(
        FINAL_TEST_COUNT,
        v1.AR1_LENGTH,
        seed=job.seed * 101 + 4,
    )
    identity_model, identity_training = v1._train_identity_model(
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        seed=job.seed,
        device=device,
    )

    evaluations: dict[str, dict[str, object]] = {view: {} for view in VIEWS}
    for control_index, control in enumerate(CONTROLS):
        controlled_calibration = apply_strict_control(
            calibration_inputs,
            calibration_labels,
            control,
            seed=job.seed * 1009 + control_index * 17 + 1,
        )
        controlled_test = apply_strict_control(
            test_inputs,
            test_labels,
            control,
            seed=job.seed * 1009 + control_index * 17 + 2,
        )
        calibration_input_diagnostics = strict_path_diagnostics(controlled_calibration)
        test_input_diagnostics = strict_path_diagnostics(controlled_test)
        for view_index, view in enumerate(VIEWS):
            if view == "learned_identity":
                calibration_pole_input, calibration_energy = v1._identity_pole_features(
                    identity_model,
                    controlled_calibration,
                    device=device,
                )
                _, test_energy = v1._identity_pole_features(
                    identity_model,
                    controlled_test,
                    device=device,
                )
            else:
                calibration_pole_input, calibration_energy = v1._fixed_ar1_view(
                    controlled_calibration,
                    view=view,
                    seed=job.seed * 409 + view_index,
                    device=device,
                )
                _, test_energy = v1._fixed_ar1_view(
                    controlled_test,
                    view=view,
                    seed=job.seed * 409 + view_index,
                    device=device,
                )
            evaluations[view][control] = {
                "controlled_raw_input": {
                    "prototype_calibration": (calibration_input_diagnostics),
                    "test": test_input_diagnostics,
                },
                "pole_input": v1.temporal_statistics(
                    calibration_pole_input,
                    calibration_labels,
                ),
                "energy_nearest_prototype": v1.prototype_metrics(
                    calibration_energy,
                    calibration_labels,
                    test_energy,
                    test_labels,
                ),
            }

    return {
        "schema": ("pac_temporal_controls_strict_marginal_result.v2"),
        "job_key": job.key,
        "family": job.family,
        "seed": job.seed,
        "status": "done",
        "device": device,
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device()) if device == "cuda" else None
        ),
        "preprocessing": {
            "name": ("strict_finite_path_gaussian_score_remarginalization_v2"),
            "label_blind": True,
            "entire_empirical_marginal_identical_across_paths": True,
            "model_train": strict_path_diagnostics(train_inputs),
            "checkpoint_validation": strict_path_diagnostics(validation_inputs),
            "prototype_calibration": strict_path_diagnostics(calibration_inputs),
            "final_test": strict_path_diagnostics(test_inputs),
        },
        "identity_training": identity_training,
        "evaluations": evaluations,
    }


def report_campaign(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    """Aggregate immutable v2 rows and retain every failed gate."""
    status = campaign_status(root)
    audit = _audit_completed_rows(root)
    rows = cast("list[dict[str, Any]]", audit["valid_rows"])
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    aggregate = _aggregate_rows(rows)
    complete_evidence = bool(status["done"]) and len(rows) >= 10
    all_gates_passed = bool(aggregate.get("all_prespecified_criteria_passed", False))
    payload: dict[str, object] = {
        "schema": ("pac_temporal_controls_strict_marginal_report.v2"),
        "status": status,
        "report_state": ("complete" if complete_evidence else "partial"),
        "complete_evidence": complete_evidence,
        "complete_claims_authorized": (complete_evidence and all_gates_passed),
        "v1_to_v2": {
            "v1_disposition": contract["predecessor"],
            "v2_change": contract["v1_to_v2_change"],
            "v1_rows_reused": 0,
            "comparison_policy": (
                "v1 is a failed pilot shown for provenance only; v2 gates "
                "are evaluated solely on new v2 rows"
            ),
        },
        "p1_1_strict_marginal": aggregate,
        "interpretation_policy": (
            "no gate widening; incomplete execution or any failed gate "
            "prevents promotion of the strict-marginal claim"
        ),
    }
    _write_json(
        root / "reports" / "STRICT_MARGINAL_V2_REPORT.json",
        payload,
    )
    _write_markdown_report(
        root / "reports" / "STRICT_MARGINAL_V2_REPORT.md",
        payload,
    )
    return payload


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        return {
            "rows": 0,
            "views": {},
            "strict_path_moments": {},
            "all_prespecified_accuracy_criteria_passed": False,
            "all_strict_path_moment_criteria_passed": False,
            "all_prespecified_criteria_passed": False,
        }
    views: dict[str, object] = {}
    all_accuracy_gates = True
    all_pole_input_gates = True
    path_means: list[float] = []
    path_gamma_errors: list[float] = []
    for view in VIEWS:
        controls: dict[str, object] = {}
        for control in CONTROLS:
            scores = [
                float(
                    row["evaluations"][view][control]["energy_nearest_prototype"][
                        "balanced_accuracy"
                    ]
                )
                for row in rows
            ]
            summary = _mean_sd_ci(scores)
            if control == "original":
                mean_passed = summary["mean"] > ORIGINAL_BALANCED_ACCURACY_MINIMUM
                ci_passed = summary["one_sided_95_lower"] > ORIGINAL_BALANCED_ACCURACY_MINIMUM
            else:
                mean_passed = (
                    SHUFFLE_BALANCED_ACCURACY_INTERVAL[0]
                    <= summary["mean"]
                    <= SHUFFLE_BALANCED_ACCURACY_INTERVAL[1]
                )
                ci_passed = (
                    summary["equivalence_ci90_low"] >= SHUFFLE_BALANCED_ACCURACY_INTERVAL[0]
                    and summary["equivalence_ci90_high"] <= SHUFFLE_BALANCED_ACCURACY_INTERVAL[1]
                )
            passed = mean_passed and ci_passed
            all_accuracy_gates = all_accuracy_gates and passed
            controls[control] = {
                **summary,
                "mean_criterion": (">0.95" if control == "original" else "[0.47,0.53]"),
                "mean_criterion_passed": mean_passed,
                "student_t_ci_criterion": (
                    "one-sided 95% lower >0.95"
                    if control == "original"
                    else "two-sided 90% CI contained in [0.47,0.53]"
                ),
                "student_t_ci_criterion_passed": ci_passed,
                "criterion_passed": passed,
            }
            if view == VIEWS[0]:
                for row in rows:
                    diagnostics = row["evaluations"][view][control]["controlled_raw_input"]
                    for split in ("prototype_calibration", "test"):
                        path_means.append(float(diagnostics[split]["maximum_absolute_path_mean"]))
                        path_gamma_errors.append(
                            float(diagnostics[split]["maximum_absolute_path_gamma0_error"])
                        )
        original_pole_rows = [row["evaluations"][view]["original"]["pole_input"] for row in rows]
        maximum_pole_mean_difference = max(
            float(value["standardized_mean_difference"]) for value in original_pole_rows
        )
        maximum_pole_gamma0_difference = max(
            float(value["standardized_gamma0_difference"]) for value in original_pole_rows
        )
        pole_mean_passed = maximum_pole_mean_difference <= 0.10
        pole_gamma0_passed = maximum_pole_gamma0_difference <= 0.10
        all_pole_input_gates = all_pole_input_gates and pole_mean_passed and pole_gamma0_passed
        views[view] = {
            "controls": controls,
            "original_pole_input_per_seed_max_gates": {
                "maximum_standardized_mean_difference": (maximum_pole_mean_difference),
                "standardized_mean_difference_maximum": 0.10,
                "standardized_mean_difference_criterion_passed": (pole_mean_passed),
                "maximum_standardized_gamma0_difference": (maximum_pole_gamma0_difference),
                "standardized_gamma0_difference_maximum": 0.10,
                "standardized_gamma0_difference_criterion_passed": (pole_gamma0_passed),
            },
        }
    maximum_mean = max(path_means)
    maximum_gamma_error = max(path_gamma_errors)
    moment_gates = (
        maximum_mean <= PATH_MEAN_TOLERANCE and maximum_gamma_error <= PATH_GAMMA0_TOLERANCE
    )
    return {
        "rows": len(rows),
        "views": views,
        "strict_path_moments": {
            "maximum_absolute_path_mean": maximum_mean,
            "mean_tolerance": PATH_MEAN_TOLERANCE,
            "mean_criterion_passed": (maximum_mean <= PATH_MEAN_TOLERANCE),
            "maximum_absolute_path_gamma0_error": maximum_gamma_error,
            "gamma0_tolerance": PATH_GAMMA0_TOLERANCE,
            "gamma0_criterion_passed": (maximum_gamma_error <= PATH_GAMMA0_TOLERANCE),
        },
        "all_prespecified_accuracy_criteria_passed": all_accuracy_gates,
        "all_original_pole_input_per_seed_max_criteria_passed": (all_pole_input_gates),
        "all_strict_path_moment_criteria_passed": moment_gates,
        "all_prespecified_criteria_passed": (
            all_accuracy_gates and all_pole_input_gates and moment_gates
        ),
    }


def _write_markdown_report(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    aggregate = cast(
        "dict[str, Any]",
        payload["p1_1_strict_marginal"],
    )
    status = cast("dict[str, Any]", payload["status"])
    lines = [
        "# Strict finite-path marginal temporal control (v2)",
        "",
        f"Report state: **{str(payload['report_state']).upper()}**.",
        f"Complete claims authorized: **{payload['complete_claims_authorized']}**.",
        f"Campaign status: {status['completed']}/{status['expected']} "
        f"completed, {status['failed']} failed.",
        f"Invalid completed rows: {status['invalid_completed']}; "
        "frozen source manifest matches: "
        f"**{status['source_manifest_matches_current_files']}**.",
        "",
        "v1 is retained as a failed pilot. No v1 row is reused in v2.",
        "v2 preserves each path's temporal ranks but replaces its sorted "
        "values by one fixed, centered, unit-Gamma(0) Gaussian-score "
        "multiset before training and controls. The timestamp rearrangement "
        "is explicitly remarginalized afterward.",
        "",
        "## v1 failed pilot versus v2",
        "",
    ]
    if aggregate.get("rows"):
        lines.extend(
            (
                "| View | Control | v1 pilot BA | v2 BA | Student-t bound/CI | v2 gate |",
                "|---|---|---:|---:|---:|:---:|",
            )
        )
        for view in VIEWS:
            for control in CONTROLS:
                v1_score = V1_FAILED_PILOT_MEAN_BALANCED_ACCURACY[view][control]
                v2_control = aggregate["views"][view]["controls"][control]
                interval = (
                    f"lower={v2_control['one_sided_95_lower']:.3f}"
                    if control == "original"
                    else (
                        f"[{v2_control['equivalence_ci90_low']:.3f},"
                        f"{v2_control['equivalence_ci90_high']:.3f}]"
                    )
                )
                lines.append(
                    f"| {view} | {control} | {v1_score:.3f} | "
                    f"{v2_control['mean']:.3f} | {interval} | "
                    f"{v2_control['criterion_passed']} |"
                )
        lines.extend(
            (
                "",
                "| View | max seedwise standardized pole mean diff | "
                "<=0.10 | max seedwise standardized Gamma(0) diff | "
                "<=0.10 |",
                "|---|---:|:---:|---:|:---:|",
            )
        )
        for view in VIEWS:
            pole = aggregate["views"][view]["original_pole_input_per_seed_max_gates"]
            lines.append(
                f"| {view} | "
                f"{pole['maximum_standardized_mean_difference']:.3f} | "
                f"{pole['standardized_mean_difference_criterion_passed']} | "
                f"{pole['maximum_standardized_gamma0_difference']:.3f} | "
                f"{pole['standardized_gamma0_difference_criterion_passed']} |"
            )
        moments = aggregate["strict_path_moments"]
        lines.extend(
            (
                "",
                "Worst absolute path mean: "
                f"{moments['maximum_absolute_path_mean']:.3e} "
                f"(gate <= {moments['mean_tolerance']:.1e}): "
                f"**{moments['mean_criterion_passed']}**.",
                "",
                "Worst absolute path Gamma(0) error: "
                f"{moments['maximum_absolute_path_gamma0_error']:.3e} "
                f"(gate <= {moments['gamma0_tolerance']:.1e}): "
                f"**{moments['gamma0_criterion_passed']}**.",
                "",
                "All current-row prespecified v2 gates passed: "
                f"**{aggregate['all_prespecified_criteria_passed']}**.",
            )
        )
    else:
        lines.append("No v2 result rows are available.")
    if payload["report_state"] == "partial":
        lines.extend(
            (
                "",
                "**PARTIAL:** these rows cannot support a complete claim.",
            )
        )
    lines.extend(
        (
            "",
            "No threshold is widened and no v1 pilot result is pooled into the v2 estimate.",
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _require_strict_paths(values: Tensor) -> None:
    diagnostics = strict_path_diagnostics(values)
    if (
        diagnostics["maximum_absolute_path_mean"] > PATH_MEAN_TOLERANCE
        or diagnostics["maximum_absolute_path_gamma0_error"] > PATH_GAMMA0_TOLERANCE
    ):
        message = (
            "strict control received or produced a path outside the frozen mean/Gamma(0) tolerances"
        )
        raise ValueError(message)


def _load_frozen_contract(
    root: Path,
) -> tuple[dict[str, Any], str]:
    contract_path = root / "contract.json"
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if contract.get("schema") != "pac_temporal_controls_strict_marginal_contract.v2":
        message = "strict-marginal v2 contract schema is invalid"
        raise RuntimeError(message)
    source_hashes = _source_hashes()
    if contract.get("source_sha256") != source_hashes:
        message = "current sources do not match the frozen v2 source manifest"
        raise RuntimeError(message)
    if contract.get("source_manifest_sha256") != _mapping_sha256(source_hashes):
        message = "frozen v2 source-manifest digest is invalid"
        raise RuntimeError(message)
    return contract, hashlib.sha256(contract_bytes).hexdigest()


def _audit_completed_rows(root: Path) -> dict[str, Any]:
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
    expected_by_key = {job.key: job for job in jobs}
    if len(expected_by_key) != len(jobs):
        invalid.append("queue contains duplicate job keys")
    contract_valid = True
    source_matches = True
    try:
        contract, contract_sha256 = _load_frozen_contract(root)
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        contract_valid = False
        source_matches = False
        contract = {}
        contract_sha256 = ""
        invalid.append(f"contract: {type(error).__name__}: {error}")
    else:
        contract_keys = {
            StrictMarginalJob(seed=int(seed)).key for seed in contract.get("seeds", [])
        }
        if contract_keys != set(expected_by_key) or int(contract.get("jobs", -1)) != len(jobs):
            contract_valid = False
            invalid.append("contract seed grid does not match queue")
        source_matches = contract.get("source_sha256") == _source_hashes() and contract.get(
            "source_manifest_sha256"
        ) == _mapping_sha256(_source_hashes())
    source_manifest_sha256 = str(contract.get("source_manifest_sha256", ""))
    valid_rows: list[dict[str, Any]] = []
    for path in sorted((root / "completed").glob("*.json")):
        job = expected_by_key.get(path.stem)
        if job is None:
            invalid.append(f"{path.name}: filename key is absent from queue")
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            invalid.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        validation_error = _result_validation_error(
            path,
            row,
            job,
            contract_sha256=contract_sha256,
            source_manifest_sha256=source_manifest_sha256,
        )
        if validation_error is None:
            valid_rows.append(row)
        else:
            invalid.append(validation_error)
    return {
        "jobs": jobs,
        "valid_rows": valid_rows,
        "invalid_rows": invalid,
        "contract_valid": contract_valid,
        "source_manifest_matches_current_files": source_matches,
    }


def _result_validation_error(
    path: Path,
    row: Mapping[str, Any],
    job: StrictMarginalJob,
    *,
    contract_sha256: str,
    source_manifest_sha256: str,
) -> str | None:
    expected = {
        "schema": ("pac_temporal_controls_strict_marginal_result.v2"),
        "job_key": job.key,
        "family": job.family,
        "seed": job.seed,
        "status": "done",
        "frozen_contract_sha256": contract_sha256,
        "frozen_source_manifest_sha256": source_manifest_sha256,
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            return f"{path.name}: {field}={row.get(field)!r}, expected {expected_value!r}"
    if path.stem != job.key:
        return f"{path.name}: filename does not match queued job key {job.key}"
    return None


def _mean_sd_ci(values: list[float]) -> dict[str, float]:
    """Return Student-t two-sided 95%, 90%, and one-sided 95% bounds."""
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        standard_error = deviation / math.sqrt(len(values))
        two_sided_95 = float(student_t.ppf(0.975, len(values) - 1))
        equivalence_90 = float(student_t.ppf(0.95, len(values) - 1))
    else:
        standard_error = math.inf
        two_sided_95 = math.inf
        equivalence_90 = math.inf
    return {
        "mean": average,
        "sample_sd": deviation,
        "ci95_low": average - two_sided_95 * standard_error,
        "ci95_high": average + two_sided_95 * standard_error,
        "one_sided_95_lower": (average - equivalence_90 * standard_error),
        "equivalence_ci90_low": (average - equivalence_90 * standard_error),
        "equivalence_ci90_high": (average + equivalence_90 * standard_error),
    }


def _read_jobs(path: Path) -> list[StrictMarginalJob]:
    return [
        StrictMarginalJob(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_hashes() -> dict[str, str]:
    return source_file_hashes(
        (
            "src/lnet/pac_temporal_controls_strict_marginal_campaign.py",
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


def _v1_artifact_hashes() -> dict[str, str]:
    paths = (V1_FAILED_PILOT_ROOT / "contract.json",)
    return {
        str(path): (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "unavailable"
        )
        for path in paths
    }


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
    "CONTROLS",
    "DEFAULT_ROOT",
    "DEFAULT_SEEDS",
    "PATH_GAMMA0_TOLERANCE",
    "PATH_MEAN_TOLERANCE",
    "V1_FAILED_PILOT_ROOT",
    "VIEWS",
    "StrictMarginalJob",
    "apply_strict_control",
    "campaign_jobs",
    "campaign_status",
    "prepare_campaign",
    "report_campaign",
    "run_job",
    "run_manifest",
    "strict_matched_ar1_samples",
    "strict_path_diagnostics",
    "strict_standardize_paths",
]
