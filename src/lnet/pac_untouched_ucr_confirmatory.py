# The fail-closed contract uses explicit, diagnostic exceptions at every gate.
# ruff: noqa: EM101, EM102, PLR0911, TRY003

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Final
from zipfile import ZipFile

from .pac_campaign_utils import write_once
from .pac_confirmatory_baselines import (
    build_matched_confirmatory_classifier,
    confirmatory_implementation_metadata,
)
from .pac_pa2wp_official_test import validation_selected_refit_epochs
from .pac_real_data import UCR_ARCHIVE_PASSWORD
from .pac_recommended_low_data_runner import run_workers
from .pac_recommended_low_data_types import LowDataJob, LowDataQueueConfig
from .pac_types import PACExperimentConfig

DEFAULT_ROOT: Final = Path(".omx/results/pac-untouched-ucr12-confirmatory-pro6000-20260713")
ARCHIVE: Final = Path(".omx/data/ucr/UCRArchive_2018.zip")
DATA_ROOT: Final = Path(".omx/data/ucr")
SEEDS: Final = (7, 11, 19, 23, 31)
REFERENCE_MODEL: Final = "PA2WP"
PUBLIC_MODEL: Final = "ALPHABET"
BASELINES: Final = (
    "cnn1d",
    "s4d",
    "mamba",
    "tcn",
    "gru",
    "lstm",
    "transformer",
    "inception_time",
)
MODELS: Final = (REFERENCE_MODEL, *BASELINES)
BASELINE_SELECTION: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
SCAN_ROOTS: Final = (
    Path(".omx/results"),
    Path(".omx/protocols"),
    Path("src"),
    Path("tests"),
    Path("scripts"),
    Path("paper"),
)
TEXT_SUFFIXES: Final = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
CONTRACT_NAME: Final = "contract.json"
MANIFEST_NAME: Final = "queue_manifest.jsonl"
AUDIT_NAME: Final = "dataset_provenance_audit.json"
LOCK_NAME: Final = "official_test_access.lock.json"
COLLECTION: Final = "untouched_ucr_confirmatory"
SCHEMA: Final = "pac_untouched_ucr_confirmatory.v1"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_dataset_names(archive: Path) -> tuple[str, ...]:
    with ZipFile(archive) as handle:
        names = {
            member.split("/")[1]
            for member in handle.namelist()
            if member.startswith("UCRArchive_2018/")
            and member.endswith("_TRAIN.tsv")
            and len(member.split("/")) == 3
        }
    return tuple(sorted(names))


def provenance_scan(root: Path, datasets: tuple[str, ...]) -> dict[str, object]:
    if not shutil.which("rg"):
        message = "ripgrep is required for the bounded provenance audit"
        raise RuntimeError(message)
    hits: dict[str, set[str]] = {name: set() for name in datasets}
    relative_root = root.resolve().relative_to(Path.cwd().resolve())
    globs = [value for suffix in sorted(TEXT_SUFFIXES) for value in ("-g", f"*{suffix}")]
    exclusions = ["-g", f"!{relative_root}/**"]
    scan_roots = [str(path) for path in SCAN_ROOTS if path.exists()]
    file_command = [
        "rg",
        "--files",
        "--hidden",
        "--no-ignore",
        *globs,
        *exclusions,
        *scan_roots,
    ]
    file_result = subprocess.run(  # noqa: S603 - fixed executable and repository roots
        file_command,
        check=True,
        capture_output=True,
        text=True,
    )
    files = tuple(line for line in file_result.stdout.splitlines() if line)
    for path_text in files:
        for name in datasets:
            if name in path_text:
                hits[name].add(path_text)
    for name in datasets:
        command = [
            "rg",
            "-l",
            "--hidden",
            "--no-ignore",
            "--fixed-strings",
            "--word-regexp",
            *globs,
            *exclusions,
            name,
            *scan_roots,
        ]
        result = subprocess.run(  # noqa: S603 - archive names are data, not shell syntax
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in {0, 1}:
            message = f"ripgrep provenance scan failed for {name}: {result.stderr.strip()}"
            raise RuntimeError(message)
        hits[name].update(line for line in result.stdout.splitlines() if line)
    return {
        "scan_roots": [str(path) for path in SCAN_ROOTS],
        "active_output_root_excluded": "<local-path>",
        "scanned_text_files": len(files),
        "scan_engine": "ripgrep --fixed-strings --word-regexp, one bounded query per archive name",
        "exact_token_rule": (
            "dataset token bounded on both sides by non-[A-Za-z0-9_] characters; "
            "dataset tokens in file paths also count"
        ),
        "hits": {name: sorted(paths) for name, paths in hits.items() if paths},
    }


def _train_metadata(archive: Path, dataset: str) -> dict[str, object]:
    member = f"UCRArchive_2018/{dataset}/{dataset}_TRAIN.tsv"
    with ZipFile(archive) as handle:
        raw = handle.read(member, pwd=UCR_ARCHIVE_PASSWORD)
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8")), delimiter="\t"))
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        return {"eligible": False, "reason": "empty_or_ragged_train_table"}
    if len(rows[0]) < 3:
        return {"eligible": False, "reason": "not_univariate_sequence_classification"}
    try:
        numeric = [[float(value) for value in row] for row in rows]
    except ValueError:
        return {"eligible": False, "reason": "nonnumeric_train_value"}
    if not all(math.isfinite(value) for row in numeric for value in row):
        return {"eligible": False, "reason": "nonfinite_train_value"}
    class_counts = Counter(row[0] for row in numeric)
    if len(class_counts) < 2 or min(class_counts.values()) < 2:
        return {"eligible": False, "reason": "insufficient_train_classes_or_support"}
    if len(rows) < 20:
        return {"eligible": False, "reason": "fewer_than_20_train_examples"}
    return {
        "eligible": True,
        "train_examples": len(rows),
        "sequence_length": len(rows[0]) - 1,
        "class_count": len(class_counts),
        "minimum_class_support": min(class_counts.values()),
    }


def select_untouched_datasets(
    root: Path,
    *,
    archive: Path = ARCHIVE,
    data_root: Path = DATA_ROOT,
    count: int = 12,
) -> tuple[tuple[str, ...], dict[str, object]]:
    datasets = _archive_dataset_names(archive)
    scan = provenance_scan(root, datasets)
    prior_hits = scan["hits"]
    if not isinstance(prior_hits, dict):
        raise TypeError("provenance hit map must be an object")
    decisions: dict[str, dict[str, object]] = {}
    selected: list[str] = []
    for dataset in datasets:
        test_path = data_root / dataset / f"{dataset}_TEST.tsv"
        reasons: list[str] = []
        if dataset in prior_hits:
            reasons.append("prior_exact_token_occurrence")
        if test_path.exists():
            reasons.append("official_test_already_materialized")
        metadata = _train_metadata(archive, dataset)
        if metadata.get("eligible") is not True:
            reasons.append(str(metadata.get("reason")))
        decisions[dataset] = {
            "selected": not reasons and len(selected) < count,
            "exclusion_reasons": reasons,
            "train_only_metadata": metadata,
            "prior_hit_paths": prior_hits.get(dataset, []),
            "test_materialized_before_lock": test_path.exists(),
        }
        if not reasons and len(selected) < count:
            selected.append(dataset)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} untouched eligible UCR datasets were found")
    audit = {
        "schema": f"{SCHEMA}.dataset_audit",
        "selection_rule": (
            "lexicographically sort UCR2018 univariate dataset names; exclude any exact "
            "dataset token previously present in PAC/ALPHABET source, tests, scripts, "
            "protocols, paper, or result artifacts; exclude any materialized TEST; verify "
            "numeric finite fixed-length suitability from TRAIN only; take the first 12"
        ),
        "archive_path": "<local-path>",
        "archive_sha256": sha256_path(archive),
        "archive_already_present_before_selection": True,
        "test_member_content_read_during_selection": False,
        "train_members_read_during_selection": True,
        "selected": selected,
        "decisions_through_selection_boundary": decisions,
        "provenance_scan": scan,
    }
    return tuple(selected), audit


def _baseline_recipes() -> tuple[dict[str, dict[str, object]], str]:
    payload = json.loads(BASELINE_SELECTION.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "pac_confirmatory_baseline_selection.v1":
        raise ValueError("unsupported prior baseline-selection artifact")
    if payload.get("status") != "complete":
        raise ValueError("prior baseline recipes are not completely selected")
    selected = payload.get("selected_trials")
    if not isinstance(selected, dict) or not set(BASELINES).issubset(selected):
        raise ValueError("prior selection does not cover every confirmatory baseline")
    recipes: dict[str, dict[str, object]] = {}
    for family in BASELINES:
        raw = selected[family]
        if not isinstance(raw, dict):
            raise TypeError(f"invalid selected recipe for {family}")
        recipes[family] = {
            "trial": int(raw["trial"]),
            "refit_epochs": int(raw["refit_epochs"]),
            "learning_rate": float(raw["learning_rate"]),
            "weight_decay": float(raw["weight_decay"]),
            "architecture": raw["architecture"],
        }
    return recipes, sha256_path(BASELINE_SELECTION)


def experiment_config(root: Path, *, device: str = "cuda") -> PACExperimentConfig:
    return PACExperimentConfig(
        2048,
        512,
        512,
        64,
        raw_input_dim=1,
        model_dim=64,
        modes=16,
        epochs=88,
        batch_size=64,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        seeds=SEEDS,
        device=device,  # type: ignore[arg-type]
        output_dir=root,
    )


def _parameter_matches(
    root: Path,
    datasets: tuple[str, ...],
    audit: dict[str, object],
    recipes: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    decisions = audit["decisions_through_selection_boundary"]
    if not isinstance(decisions, dict):
        raise TypeError("audit decisions must be an object")
    config = experiment_config(root, device="cpu")
    matches: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        raw = decisions[dataset]
        if not isinstance(raw, dict):
            raise TypeError("dataset audit entry must be an object")
        metadata = raw["train_only_metadata"]
        if not isinstance(metadata, dict):
            raise TypeError("TRAIN metadata must be an object")
        class_count = int(metadata["class_count"])
        dataset_matches: dict[str, object] = {}
        for family in BASELINES:
            model, match = build_matched_confirmatory_classifier(
                family,  # type: ignore[arg-type]
                REFERENCE_MODEL,
                config,
                class_count,
                tolerance=0.05,
                validation_trial=int(recipes[family]["trial"]),
            )
            if match.relative_error > 0.05:
                raise ValueError(
                    f"{dataset}/{family} parameter error {match.relative_error:.6f} exceeds 5%"
                )
            match_payload = asdict(match)
            match_payload["functional_budget_adapter_params"] = int(
                getattr(model, "budget_parameter_count", 0)
            )
            dataset_matches[family] = match_payload
        matches[dataset] = dataset_matches
    return matches


def _jobs(
    datasets: tuple[str, ...],
    recipes: dict[str, dict[str, object]],
    contract_sha256: str,
) -> tuple[LowDataJob, ...]:
    refit_epochs, _ = validation_selected_refit_epochs()
    jobs: list[LowDataJob] = []
    for dataset in datasets:
        for seed in SEEDS:
            jobs.append(
                LowDataJob(
                    key=f"untouched_ucr:{dataset}:ALPHABET:seed{seed}",
                    seed=seed,
                    model=REFERENCE_MODEL,
                    dataset=dataset,
                    ratio=1.0,
                    evaluation_split="test",
                    refit_full_train=True,
                    data_protocol="clean_stratified",
                    restore_best_validation=False,
                    evaluation_collection=COLLECTION,
                    reference_model=REFERENCE_MODEL,
                    refit_epochs=refit_epochs,
                    learning_rate=3.0e-3,
                    weight_decay=1.0e-4,
                    official_test_contract_sha256=contract_sha256,
                )
            )
            for family in BASELINES:
                recipe = recipes[family]
                trial = int(recipe["trial"])
                jobs.append(
                    LowDataJob(
                        key=f"untouched_ucr:{dataset}:{family}:seed{seed}",
                        seed=seed,
                        model=family,
                        dataset=dataset,
                        ratio=1.0,
                        evaluation_split="test",
                        refit_full_train=True,
                        data_protocol="clean_stratified",
                        restore_best_validation=False,
                        evaluation_collection=COLLECTION,
                        baseline_family=family,  # type: ignore[arg-type]
                        reference_model=REFERENCE_MODEL,
                        validation_trial=trial,
                        architecture_metadata_json=json.dumps(
                            confirmatory_implementation_metadata(
                                family,  # type: ignore[arg-type]
                                trial,
                            ),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        refit_epochs=int(recipe["refit_epochs"]),
                        learning_rate=float(recipe["learning_rate"]),
                        weight_decay=float(recipe["weight_decay"]),
                        parameter_match_tolerance=0.05,
                        official_test_contract_sha256=contract_sha256,
                    )
                )
    return tuple(jobs)


def prepare(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    lock_path = root / LOCK_NAME
    if lock_path.exists():
        return verify_test_access_lock(root)
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / AUDIT_NAME
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        datasets = tuple(str(value) for value in audit["selected"])
    else:
        datasets, audit = select_untouched_datasets(root)
        write_once(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    recipes, recipe_sha256 = _baseline_recipes()
    parameter_matches = _parameter_matches(root, datasets, audit, recipes)
    alphabet_epochs, alphabet_validation_epochs = validation_selected_refit_epochs()
    contract = {
        "schema": SCHEMA,
        "purpose": "post-development untouched UCR2018 confirmatory comparison",
        "architecture_frozen": True,
        "public_model": PUBLIC_MODEL,
        "internal_model": REFERENCE_MODEL,
        "model_dim": 64,
        "modes": 16,
        "dual_origin_inference": True,
        "datasets": list(datasets),
        "dataset_count": len(datasets),
        "dataset_selection_rule": audit["selection_rule"],
        "dataset_audit": "<local-path>",
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "primary_baselines": list(BASELINES[:-1]),
        "additional_baseline": "inception_time",
        "jobs": len(datasets) * len(SEEDS) * len(MODELS),
        "parameter_match_relative_tolerance": 0.05,
        "task_specific_parameter_matches": parameter_matches,
        "alphabet_recipe": {
            "refit_epochs": alphabet_epochs,
            "learning_rate": 3.0e-3,
            "weight_decay": 1.0e-4,
            "epoch_selection": (
                "frozen global median from the completed 90-row TRAIN-derived PA2WP "
                "validation campaign"
            ),
            "source_epoch_count": len(alphabet_validation_epochs),
        },
        "baseline_recipes": recipes,
        "baseline_recipe_source": "<local-path>",
        "baseline_recipe_source_sha256": recipe_sha256,
        "training_protocol": "full official TRAIN refit at frozen epochs",
        "normalization": "fit on full official TRAIN only",
        "checkpoint_policy": "final fixed epoch; no TEST-selected checkpoint",
        "test_policy": "one official TEST evaluation after this contract and manifest lock",
        "official_test_accessed_at_enqueue": False,
        "test_access_authorized": True,
        "result_root_is_distinct_from_prior_official_test_roots": True,
    }
    audit_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    contract_text = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    write_once(root / AUDIT_NAME, audit_text)
    write_once(root / CONTRACT_NAME, contract_text)
    contract_sha256 = sha256_path(root / CONTRACT_NAME)
    jobs = _jobs(datasets, recipes, contract_sha256)
    manifest_text = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs)
    write_once(root / MANIFEST_NAME, manifest_text)
    lock = {
        "schema": f"{SCHEMA}.test_access_lock",
        "authorized": True,
        "contract": CONTRACT_NAME,
        "contract_sha256": contract_sha256,
        "manifest": MANIFEST_NAME,
        "manifest_sha256": sha256_path(root / MANIFEST_NAME),
        "audit": AUDIT_NAME,
        "audit_sha256": sha256_path(root / AUDIT_NAME),
        "jobs": len(jobs),
        "created_before_selected_test_member_read": True,
    }
    write_once(lock_path, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return verify_test_access_lock(root)


def verify_test_access_lock(  # noqa: C901 - explicit fail-closed integrity gate
    root: Path,
    *,
    expected_contract_sha256: str | None = None,
    expected_job: LowDataJob | None = None,
) -> dict[str, object]:
    lock_path = root / LOCK_NAME
    if not lock_path.exists():
        raise FileNotFoundError("official TEST access lock is absent")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != f"{SCHEMA}.test_access_lock" or lock.get("authorized") is not True:
        raise ValueError("official TEST access lock is invalid or unauthorized")
    contract_path = root / CONTRACT_NAME
    manifest_path = root / MANIFEST_NAME
    audit_path = root / AUDIT_NAME
    actual_contract_sha = sha256_path(contract_path)
    if actual_contract_sha != lock.get("contract_sha256"):
        raise ValueError("frozen untouched contract hash mismatch")
    if expected_contract_sha256 is not None and actual_contract_sha != expected_contract_sha256:
        raise ValueError("job is not bound to the active untouched contract")
    if sha256_path(manifest_path) != lock.get("manifest_sha256"):
        raise ValueError("frozen untouched manifest hash mismatch")
    if sha256_path(audit_path) != lock.get("audit_sha256"):
        raise ValueError("frozen untouched dataset audit hash mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("architecture_frozen") is not True
        or contract.get("test_access_authorized") is not True
        or contract.get("official_test_accessed_at_enqueue") is not False
    ):
        raise ValueError("untouched confirmatory contract is not a valid pre-TEST freeze")
    lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != int(contract["jobs"]) or len(lines) != int(lock["jobs"]):
        raise ValueError("locked manifest job count mismatch")
    if expected_job is not None:
        if expected_job.dataset not in contract["datasets"]:
            raise ValueError("job dataset is outside the locked untouched collection")
        if (
            expected_job.seed not in contract["seeds"]
            or expected_job.model not in contract["models"]
        ):
            raise ValueError("job seed/model is outside the locked untouched collection")
        expected_row = asdict(expected_job)
        locked_rows = [json.loads(line) for line in lines if expected_job.key in line]
        if locked_rows != [expected_row]:
            raise ValueError("job payload differs from its frozen manifest row")
    return {
        "verified": True,
        "root": "<local-path>",
        "contract_sha256": actual_contract_sha,
        "manifest_sha256": lock["manifest_sha256"],
        "audit_sha256": lock["audit_sha256"],
        "datasets": contract["datasets"],
        "models": contract["models"],
        "seeds": contract["seeds"],
        "jobs": contract["jobs"],
    }


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    verified = verify_test_access_lock(root)
    latest: dict[str, str] = {}
    state = root / "queue_state.jsonl"
    if state.exists():
        for line in state.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                latest[str(row["key"])] = str(row["status"])
    manifest = root / MANIFEST_NAME
    keys = {
        str(json.loads(line)["key"])
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    }
    counts = Counter(latest.get(key, "pending") for key in keys)
    verified.update(
        {
            "done": counts["done"],
            "running": counts["running"],
            "failed": counts["failed"],
            "pending": counts["pending"],
            "supervisor_pid": _read_pid(root / "supervisor.pid"),
        }
    )
    return verified


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None
    return pid


def workers(root: Path, *, device: str, worker_count: int) -> None:
    verify_test_access_lock(root)
    if not 1 <= worker_count <= 6:
        raise ValueError("confirmatory worker count must be in [1, 6]")
    queue_config = LowDataQueueConfig(
        output_root=root,
        preset="full",
        seeds=SEEDS,
        device=device,  # type: ignore[arg-type]
        workers=worker_count,
        total_slots=worker_count,
    )
    run_workers(
        queue_config,
        experiment_config=experiment_config(root, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "verify", "workers", "status"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        payload = prepare(args.output_root)
    elif args.stage == "verify":
        payload = verify_test_access_lock(args.output_root)
    elif args.stage == "status":
        payload = status(args.output_root)
    else:
        workers(args.output_root, device=args.device, worker_count=args.workers)
        payload = status(args.output_root)
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
