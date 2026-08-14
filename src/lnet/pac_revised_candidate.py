from __future__ import annotations

import csv
import fcntl
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Final, Literal, cast

from scipy.stats import wilcoxon

from .pac_overnight_io import append_csv_row
from .pac_stiefel_variants import REVISED_UNTIED_MODEL
from .pac_tf_evidence_eval import run_evidence_job
from .pac_tf_evidence_queue import (
    CAPACITY_SELECTION_PATH,
    PROTOCOL_PATH,
    EvidenceJob,
    file_sha256,
    full_evidence_config,
    load_protocol,
    load_selection_binding,
    paired_hierarchical_bootstrap,
)

if TYPE_CHECKING:
    from .pac_types import PACDevice, PACExperimentConfig
    from .tapped_prl_followup_schema import JsonRow

DEFAULT_ROOT: Final[Path] = Path(".omx/results/pac-tf-revised-untied-candidate-20260711")
CANONICAL_MODEL: Final[str] = "pac_stiefel_depth2_norm_autocorr_d64_m16"
REVISED_INTERVENTION: Final[str] = "revised_fixed_mean_nogate_untied"
RESULT_FILE: Final[str] = "revised_candidate.csv"
CandidateCollection = Literal["development", "untouched"]


def enqueue_jobs(
    root: Path = DEFAULT_ROOT,
    protocol_path: Path = PROTOCOL_PATH,
    capacity_selection: Path = CAPACITY_SELECTION_PATH,
    collection: CandidateCollection = "development",
) -> int:
    protocol = load_protocol(protocol_path)
    binding = load_selection_binding(capacity_selection, protocol_path)
    if (binding.model_dim, binding.modes) != (64, 16):
        message = (
            "the revised candidate is locked to D=64/M=16, but the capacity "
            f"binding selected D={binding.model_dim}/M={binding.modes}"
        )
        raise ValueError(message)
    dataset_key = (
        "development_datasets" if collection == "development" else "untouched_final_datasets"
    )
    datasets = protocol[dataset_key]
    jobs = tuple(
        EvidenceJob(
            key=f"revised_candidate__{intervention}__{dataset}__seed{seed}",
            protocol_id=str(protocol["protocol_id"]),
            kind="core_ablation",
            seed=int(seed),
            scope=str(dataset),
            intervention=intervention,
            model=CANONICAL_MODEL if intervention == "reference" else REVISED_UNTIED_MODEL,
            protocol_sha256=binding.protocol_sha256,
            capacity_artifact_sha256=binding.capacity_artifact_sha256,
            selected_model=binding.selected_model,
            selected_model_dim=binding.model_dim,
            selected_modes=binding.modes,
        )
        for seed in protocol["seeds"]
        for dataset in datasets
        for intervention in ("reference", REVISED_INTERVENTION)
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate_manifest.jsonl").write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    contract = {
        "schema_version": "pac_tf_revised_candidate.v1",
        "evidence_design": f"{collection}_validation",
        "official_test_read": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "capacity_artifact": str(capacity_selection),
        "capacity_artifact_sha256": file_sha256(capacity_selection),
        "selected_model": binding.selected_model,
        "selected_model_dim": binding.model_dim,
        "selected_modes": binding.modes,
        "candidate_model": REVISED_UNTIED_MODEL,
        "changes": [
            "fixed_damping",
            "global_mean_pooling",
            "no_modal_gate",
            "untied_synthesis",
        ],
        "datasets": list(datasets),
        "seeds": list(protocol["seeds"]),
        "jobs": len(jobs),
    }
    (root / "candidate_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _event(root, "enqueue", "done")
    return len(jobs)


def run_workers(
    root: Path = DEFAULT_ROOT,
    *,
    device: str = "cuda",
    workers: int = 8,
    total_slots: int = 16,
    max_jobs: int | None = None,
) -> None:
    contract = _read_contract(root)
    jobs = _read_jobs(root / "candidate_manifest.jsonl")
    completed = {key for key, status in _latest_statuses(root).items() if status == "done"}
    pending = [job for job in jobs if job.key not in completed]
    model_dim = contract["selected_model_dim"]
    modes = contract["selected_modes"]
    if not isinstance(model_dim, int) or not isinstance(modes, int):
        message = "candidate contract contains invalid capacity values"
        raise TypeError(message)
    config = full_evidence_config(
        root,
        model_dim=model_dim,
        modes=modes,
        device=cast("PACDevice", device),
    )
    active: dict[Future[tuple[EvidenceJob, JsonRow | None]], int] = {}
    launched = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            available = total_slots - sum(active.values())
            while pending and available > 0 and (max_jobs is None or launched < max_jobs):
                index = next(
                    (i for i, candidate in enumerate(pending) if candidate.slots <= available),
                    None,
                )
                if index is None:
                    break
                job = pending.pop(index)
                _event(root, job.key, "running")
                active[pool.submit(_execute, root, config, job)] = job.slots
                available -= job.slots
                launched += 1
            if not active:
                break
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                job, row = future.result()
                _event(root, job.key, "done" if row is not None else "failed")
            if max_jobs is not None and launched >= max_jobs and not active:
                break


def write_report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    jobs = _read_jobs(root / "candidate_manifest.jsonl")
    latest_rows: dict[str, dict[str, str]] = {}
    result_path = root / "results" / RESULT_FILE
    if result_path.exists():
        for row in csv.DictReader(result_path.open(newline="")):
            latest_rows[row["queue_key"]] = row
    expected = {job.key for job in jobs}
    complete = expected <= {
        key for key, row in latest_rows.items() if row.get("status") == "done"
    }
    scores: dict[str, dict[tuple[str, int], float]] = {"reference": {}, "revised": {}}
    params: dict[str, list[int]] = {"reference": [], "revised": []}
    for row in latest_rows.values():
        if row.get("status") != "done":
            continue
        label = "reference" if row["intervention"] == "reference" else "revised"
        key = (row["dataset_or_task"], int(row["seed"]))
        scores[label][key] = float(row["validation_balanced_accuracy"])
        params[label].append(int(row["params_trainable"]))
    paired_keys = sorted(set(scores["reference"]) & set(scores["revised"]))
    paired = {
        dataset: [
            scores["revised"][(dataset, seed)] - scores["reference"][(dataset, seed)]
            for candidate_dataset, seed in paired_keys
            if candidate_dataset == dataset
        ]
        for dataset in sorted({dataset for dataset, _ in paired_keys})
    }
    if paired and all(paired.values()):
        delta, ci_low, ci_high = paired_hierarchical_bootstrap(paired)
        dataset_effects = [mean(values) for values in paired.values()]
        p_value = (
            1.0
            if all(effect == 0.0 for effect in dataset_effects)
            else float(cast("float", wilcoxon(dataset_effects, alternative="two-sided")[1]))
        )
    else:
        delta = ci_low = ci_high = p_value = float("nan")
    payload: dict[str, object] = {
        "schema_version": "pac_tf_revised_candidate_report.v1",
        "status": "complete" if complete else "partial",
        "evaluation_split": "official_train_derived_validation",
        "official_test_read": False,
        "paired_runs": len(paired_keys),
        "datasets": len(paired),
        "reference_mean_balanced_accuracy": (
            mean(scores["reference"].values()) if scores["reference"] else None
        ),
        "revised_mean_balanced_accuracy": (
            mean(scores["revised"].values()) if scores["revised"] else None
        ),
        "paired_balanced_accuracy_delta": delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "dataset_level_wilcoxon_p": p_value,
        "reference_mean_params": round(mean(params["reference"])) if params["reference"] else None,
        "revised_mean_params": round(mean(params["revised"])) if params["revised"] else None,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "revised_candidate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parameter_summary = (
        f"- mean parameters (reference/revised): {payload['reference_mean_params']} / "
        f"{payload['revised_mean_params']}"
    )
    lines = [
        "# Revised PAC-TF Candidate",
        "",
        f"- status: {payload['status']}",
        "- split: official TRAIN-derived validation",
        "- official TEST read: no",
        f"- paired runs: {payload['paired_runs']}",
        f"- reference balanced accuracy: {payload['reference_mean_balanced_accuracy']}",
        f"- revised balanced accuracy: {payload['revised_mean_balanced_accuracy']}",
        f"- paired delta: {payload['paired_balanced_accuracy_delta']}",
        f"- 95% hierarchical bootstrap CI: [{payload['ci95_low']}, {payload['ci95_high']}]",
        f"- dataset-level Wilcoxon p: {payload['dataset_level_wilcoxon_p']}",
        parameter_summary,
    ]
    (reports / "revised_candidate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _execute(
    root: Path, config: PACExperimentConfig, job: EvidenceJob
) -> tuple[EvidenceJob, JsonRow | None]:
    try:
        row = run_evidence_job(root, config, job)
        append_csv_row(root / "results" / RESULT_FILE, row)
    except Exception as error:  # noqa: BLE001 - queue must preserve per-job failures
        failed: JsonRow = {
            "queue_key": job.key,
            "dataset_or_task": job.scope,
            "seed": job.seed,
            "model": job.model,
            "intervention": job.intervention,
            "status": "failed",
            "notes": f"{type(error).__name__}: {error}",
        }
        append_csv_row(root / "results" / RESULT_FILE, failed)
        return job, None
    return job, row


def _read_contract(root: Path) -> dict[str, object]:
    path = root / "candidate_contract.json"
    if not path.is_file():
        message = f"candidate contract is missing: {path}"
        raise FileNotFoundError(message)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jobs(path: Path) -> tuple[EvidenceJob, ...]:
    return tuple(
        EvidenceJob(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _latest_statuses(root: Path) -> dict[str, str]:
    path = root / "queue_state.jsonl"
    latest: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[str(row["key"])] = str(row["status"])
    return latest


def _event(root: Path, key: str, status: str) -> None:
    path = root / "queue_state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps({"key": key, "status": status}, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
