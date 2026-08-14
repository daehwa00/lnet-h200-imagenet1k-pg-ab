"""Create data-identical GPU calibration and irregular-path preflight manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from lnet.pac_broad_benchmark_queue import (
    DEFAULT_ROOT,
    BenchmarkJob,
    stage1_jobs,
)


def _write_once(path: Path, jobs: tuple[BenchmarkJob, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(job.payload(), sort_keys=True) + "\n" for job in jobs)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        message = f"refusing to overwrite a different calibration manifest: {path}"
        raise FileExistsError(message)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", default="ECG200")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.repeats < 1:
        message = "epochs and repeats must be positive"
        raise ValueError(message)
    manifest_name = Path(arguments.manifest_name)
    if (
        manifest_name.name != arguments.manifest_name
        or manifest_name.suffix != ".jsonl"
    ):
        message = "manifest-name must be one .jsonl file name"
        raise ValueError(message)

    alphabet = next(
        job
        for job in stage1_jobs()
        if job.dataset == arguments.dataset
        and job.model == "alphabet"
        and job.candidate_id == "d64-m16-recipea"
    )
    mamba = next(
        job
        for job in stage1_jobs()
        if job.dataset == arguments.dataset
        and job.model == "mamba"
        and job.candidate_id == "w64-s16-c3-recipea"
    )
    jobs: list[BenchmarkJob] = []
    for repeat in range(arguments.repeats):
        suffix = "" if arguments.repeats == 1 else f":repeat-{repeat}"
        group_suffix = "" if arguments.repeats == 1 else f":{repeat}"
        jobs.extend(
            (
                replace(
                    alphabet,
                    key=f"broad:calibration:{arguments.dataset}:alphabet{suffix}",
                    epochs=arguments.epochs,
                    train_seed=alphabet.train_seed + repeat,
                    comparison_group=f"calibration:{arguments.dataset}:core{group_suffix}",
                ),
                replace(
                    mamba,
                    key=f"broad:calibration:{arguments.dataset}:mamba{suffix}",
                    epochs=arguments.epochs,
                    train_seed=mamba.train_seed + repeat,
                    comparison_group=f"calibration:{arguments.dataset}:mamba{group_suffix}",
                ),
            )
        )
    path = arguments.root / "calibration" / manifest_name
    _write_once(path, tuple(jobs))

    physionet = next(
        job
        for job in stage1_jobs()
        if job.dataset == "physionet-2012"
        and job.model == "alphabet"
        and job.candidate_id == "d64-m16-recipea"
    )
    physionet_path = arguments.root / "calibration" / "physionet2012-manifest.jsonl"
    _write_once(
        physionet_path,
        (
            replace(
                physionet,
                key="broad:preflight:physionet-2012:alphabet",
                epochs=1,
                comparison_group="preflight:physionet-2012:metadata",
            ),
        ),
    )
    print(json.dumps({"calibration": str(path), "physionet": str(physionet_path)}))  # noqa: T201


if __name__ == "__main__":
    main()
