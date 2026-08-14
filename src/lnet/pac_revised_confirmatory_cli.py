from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Final, Literal

import typer

from .pac_confirmatory_baselines import confirmatory_implementation_metadata
from .pac_overnight_io import prepare_overnight_dirs
from .pac_recommended_low_data_runner import run_workers
from .pac_recommended_low_data_types import LowDataJob, LowDataQueueConfig
from .pac_stiefel_variants import REVISED_UNTIED_MODEL
from .pac_tf_evidence_queue import PROTOCOL_PATH, load_protocol
from .pac_types import PACDevice  # noqa: TC001 - Typer resolves annotations at runtime.

app = typer.Typer(add_completion=False)

DEFAULT_ROOT: Final = Path(".omx/results/pac-tf-revised-confirmatory-20260712")
SEEDS: Final = (7, 11, 19, 23, 31)
VALIDATION_TRIAL: Final = 4
REFIT_EPOCHS: Final = 75
LEARNING_RATE: Final = 3.0e-3
WEIGHT_DECAY: Final = 1.0e-4


def enqueue(root: Path, protocol_path: Path = PROTOCOL_PATH) -> int:
    protocol_bytes = protocol_path.read_bytes()
    protocol = load_protocol(protocol_path)
    datasets = tuple(
        dict.fromkeys(
            protocol["development_datasets"]
            + protocol["untouched_final_datasets"]
        )
    )
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    if seeds != SEEDS:
        message = f"revised confirmatory seeds changed: {seeds}"
        raise ValueError(message)
    architecture = json.dumps(
        confirmatory_implementation_metadata("pac_tf", VALIDATION_TRIAL),
        sort_keys=True,
        separators=(",", ":"),
    )
    jobs = tuple(
        LowDataJob(
            key=f"low_data:unseen_final:{seed}:pac_tf:{dataset}:1.0",
            seed=seed,
            model="pac_tf",
            dataset=dataset,
            ratio=1.0,
            evaluation_split="test",
            refit_full_train=True,
            data_protocol="clean_stratified",
            restore_best_validation=False,
            evaluation_collection="unseen_final_ucr",
            baseline_family="pac_tf",
            reference_model=REVISED_UNTIED_MODEL,
            validation_trial=VALIDATION_TRIAL,
            architecture_metadata_json=architecture,
            refit_epochs=REFIT_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        for seed in seeds
        for dataset in datasets
    )
    prepare_overnight_dirs(root)
    (root / "queue_manifest.jsonl").write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    contract = {
        "schema_version": "pac_revised_confirmatory.v1",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha256,
        "model": REVISED_UNTIED_MODEL,
        "datasets": list(datasets),
        "seeds": list(seeds),
        "jobs": len(jobs),
        "selection_source": "validation_on_18_train-derived_splits",
        "refit_epoch_policy": "median revised validation best_epoch (75)",
        "official_test_policy": "single full-TRAIN refit after revised architecture freeze",
    }
    (reports / "revised_confirmatory_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_p1p2_selection(root, protocol_sha256)
    return len(jobs)


def _write_p1p2_selection(root: Path, protocol_sha256: str) -> None:
    source = (
        Path(".omx/results/pac-tf-confirmatory-unseen-20260711")
        / "reports"
        / "confirmatory_baseline_selection.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["reference_model"] = REVISED_UNTIED_MODEL
    pac_trial = payload["selected_trials"]["pac_tf"]
    pac_trial.update(
        {
            "trial": VALIDATION_TRIAL,
            "refit_epochs": REFIT_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "architecture": confirmatory_implementation_metadata(
                "pac_tf", VALIDATION_TRIAL
            ),
        }
    )
    payload["protocol_sha256"] = protocol_sha256
    payload["selection_split"] = "revised_train-derived_validation"
    payload["refit_epoch_policy"] = "median revised validation best_epoch; round half upward"
    (root / "reports" / "confirmatory_baseline_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[Literal["enqueue", "workers"], typer.Option("--stage")] = "enqueue",
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_ROOT,
    protocol_path: Annotated[Path, typer.Option("--protocol-path")] = PROTOCOL_PATH,
    device: Annotated[PACDevice, typer.Option("--device")] = "auto",
    workers: Annotated[int, typer.Option("--workers")] = 8,
    total_slots: Annotated[int, typer.Option("--total-slots")] = 16,
    max_jobs: Annotated[int | None, typer.Option("--max-jobs")] = None,
) -> None:
    config = LowDataQueueConfig(
        output_root=output_root,
        preset="full",
        seeds=SEEDS,
        device=device,
        workers=workers,
        total_slots=total_slots,
        optimizer_mode="fused",
    )
    if stage == "enqueue":
        typer.echo(f"enqueued={enqueue(output_root, protocol_path)}")
    else:
        run_workers(config, max_jobs=max_jobs)


if __name__ == "__main__":
    app()
