from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Final

from .pac_recommended_low_data_types import LowDataJob
from .pac_wp_evidence_campaign import SEEDS, UCR_DATASETS

DEFAULT_ROOT: Final = Path(
    ".omx/results/pac-pa2wp-official-ucr-test-pro6000-20260713"
)
MODEL: Final = "PA2WP"
REFERENCE_MODEL: Final = "pac_headroom_phase_augmented_ensemble_wp_d64_m16"
VALIDATION_ROOTS: Final = (
    Path(".omx/results/pac-pa2wp-nontop-ucr18-20260713/completed"),
    Path(".omx/results/pac-pa2wp-top-guard-20260713/completed"),
)


def validation_selected_refit_epochs() -> tuple[int, tuple[int, ...]]:
    rows: dict[str, dict[str, object]] = {}
    for root in VALIDATION_ROOTS:
        for path in root.glob("ucr_validation_*_PA2WP_seed*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = str(payload.get("job_key") or payload.get("key"))
            rows[key] = payload
    expected = {
        f"ucr_validation:{dataset}:PA2WP:seed{seed}"
        for dataset in UCR_DATASETS
        for seed in SEEDS
    }
    invalid = [
        key
        for key in expected
        if key not in rows
        or rows[key].get("status") != "done"
        or rows[key].get("official_test_accessed") is not False
    ]
    if invalid or set(rows) != expected:
        message = (
            "final ALPHABET official TEST enqueue requires exactly 90 successful, "
            f"TRAIN-derived validation rows; invalid={invalid[:3]} "
            f"missing={len(expected - set(rows))} extra={len(set(rows) - expected)}"
        )
        raise ValueError(message)
    epochs = tuple(int(rows[key]["best_epoch"]) for key in sorted(expected))
    return max(1, math.floor(median(epochs) + 0.5)), epochs


def official_test_jobs() -> tuple[LowDataJob, ...]:
    refit_epochs, _ = validation_selected_refit_epochs()
    return tuple(
        LowDataJob(
            key=f"pa2wp_official_test:{dataset}:seed{seed}",
            seed=seed,
            model=MODEL,
            dataset=dataset,
            ratio=1.0,
            evaluation_split="test",
            refit_full_train=True,
            data_protocol="clean_stratified",
            restore_best_validation=False,
            evaluation_collection="pa2wp_official_ucr_test",
            reference_model=REFERENCE_MODEL,
            refit_epochs=refit_epochs,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
        )
        for dataset in UCR_DATASETS
        for seed in SEEDS
    )


def enqueue(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    jobs = official_test_jobs()
    refit_epochs, validation_epochs = validation_selected_refit_epochs()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "queue_manifest.jsonl"
    if manifest.exists():
        current = manifest.read_text(encoding="utf-8")
        expected = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs)
        if current != expected:
            message = "refusing to replace a different final ALPHABET TEST manifest"
            raise ValueError(message)
    else:
        manifest.write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
            encoding="utf-8",
        )
    contract: dict[str, object] = {
        "schema": "pac_pa2wp_official_ucr_test.v1",
        "public_model": "ALPHABET",
        "internal_model": MODEL,
        "reference_model": REFERENCE_MODEL,
        "architecture_frozen": True,
        "selection_source": [str(path) for path in VALIDATION_ROOTS],
        "selection_rows": len(validation_epochs),
        "datasets": list(UCR_DATASETS),
        "seeds": list(SEEDS),
        "jobs": len(jobs),
        "refit_epochs": refit_epochs,
        "refit_epoch_policy": (
            "global median best_epoch over the frozen 90-row ALPHABET validation suite; "
            "round half upward before official TEST access"
        ),
        "optimizer": {"learning_rate": 3.0e-3, "weight_decay": 1.0e-4},
        "training_protocol": "all official TRAIN, final fixed epoch",
        "test_policy": "one official TEST evaluation per frozen dataset-seed job",
        "official_test_accessed_at_enqueue": False,
    }
    contract_path = root / "contract.json"
    encoded = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if contract_path.exists() and contract_path.read_text(encoding="utf-8") != encoded:
        message = "refusing to replace a different final ALPHABET TEST contract"
        raise ValueError(message)
    contract_path.write_text(encoded, encoding="utf-8")
    return contract


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    expected = {job.key for job in official_test_jobs()}
    latest: dict[str, str] = {}
    state = root / "queue_state.jsonl"
    if state.exists():
        for line in state.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[str(row["key"])] = str(row["status"])
    done = {key for key in expected if latest.get(key) == "done"}
    failed = {key for key in expected if latest.get(key) == "failed"}
    running = {key for key in expected if latest.get(key) == "running"}
    return {
        "expected": len(expected),
        "done": len(done),
        "running": len(running),
        "failed": len(failed),
        "pending": len(expected - done - failed - running),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("enqueue", "status"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    payload = enqueue(args.output_root) if args.stage == "enqueue" else status(args.output_root)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
