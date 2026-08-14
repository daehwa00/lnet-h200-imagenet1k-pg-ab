# ruff: noqa: T201
"""Freeze the three new runnable datasets against the fixed 10-model set."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from lnet.pac_broad_benchmark_distributed import (
    audit_deployment,
    load_deployment_config,
    plan_jobs,
)
from lnet.pac_broad_benchmark_queue import stage1_jobs
from lnet.pac_broad_benchmark_worker import code_sha256
from lnet.pac_fast_completion import (
    FINAL_EPOCHS,
    FINAL_SEEDS,
    STAGE1_CANDIDATES_PER_CELL,
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
    STAGE2_SEEDS,
    STAGE2_TOP_K,
)

DEFAULT_ROOT = Path(".omx/results/alphabet-new3-10model-5gpu-20260727")
BASE_CONFIG = Path("optimization/hosts/broad_benchmark_3gpu.local.json")
DEFAULT_CONFIG = Path("optimization/hosts/new3_10model_5gpu.local.json")
DATASETS = {
    "physionet-2012",
    "physionet-2019",
    "pam",
}
MODELS = {
    "alphabet",
    "cnn1d",
    "tcn",
    "transformer",
    "mamba",
    "s4d",
    "s5",
    "lru",
    "gru",
    "lstm",
}
FAST_ARCHITECTURES = {
    "cnn1d": "d2-k3",
    "tcn": "d3-k3",
    "transformer": "d1-h2",
    "mamba": "s16-c3",
    "s4d": "d1-s16",
    "s5": "d1-s16",
    "lru": "d1-s16",
    "gru": "d1-s16",
    "lstm": "d1-s16",
}
FAST_WIDTHS = {32, 64}
FAST_ALPHABET_CAPACITIES = {(32, 8), (64, 16)}
DATA_SHARDS = (
    "irregular:physionet-2012:mortality",
    "irregular:physionet-2019:early-sepsis",
    "irregular:pam:activity-8",
)


def _write_once(path: Path, payload: object) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        message = f"refusing to overwrite a different artifact: {path}"
        raise FileExistsError(message)
    path.write_text(content, encoding="utf-8")


def write_config(source: Path, destination: Path) -> None:
    payload = cast("dict[str, object]", json.loads(source.read_text(encoding="utf-8")))
    hosts = cast("list[dict[str, object]]", payload["hosts"])
    for host in hosts:
        name = str(host["name"])
        host.pop("data_shard_set", None)
        host["data_shards"] = list(DATA_SHARDS)
        host["repo"] = (
            "<remote-home>/lnet-new3-10model-runtime-20260727"
            if name == "secondary_gpu"
            else "<remote-home>/lnet-new3-10model-runtime-20260727"
        )
        roots = cast("dict[str, object]", host["data_roots"])
        roots["physionet2012"] = (
            "<remote-home>/lnet-terminal-20260718/data/physionet2012"
            if name == "secondary_gpu"
            else "<remote-home>/lnet-data/physionet2012"
        )
        roots["raindrop"] = (
            "<remote-home>/lnet-data/irregular/raindrop"
            if name == "secondary_gpu"
            else "<remote-home>/lnet-data/irregular/raindrop"
        )
    hosts.append(
        {
            "name": "local_gpu",
            "enabled": False,
            "transport": "ssh",
            "ssh_host": "local_gpu",
            "repo": "<local-home>/lnet-new3-10model-runtime-20260727",
            "gpu_type": "rtx4090",
            "gpus": [0, 1],
            "memory_mb": 24_564,
            "profiles": [
                {
                    "name": "core",
                    "python": "<local-home>/miniconda3/envs/brelu/bin/python",
                },
                {
                    "name": "mamba",
                    "python": "<local-home>/miniconda3/envs/brelu/bin/python",
                },
            ],
            "profile_relative_speeds": {"core": 1.0, "mamba": 1.0},
            "data_shards": list(DATA_SHARDS),
            "data_roots": {
                "ucr": "<local-home>/lnet-external-20260718/.omx/data/ucr",
                "external": "<local-home>/lnet-external-20260718/data/external",
                "physionet2012": "<local-home>/lnet-external-20260718/data/physionet2012",
                "raindrop": "<local-home>/lnet-data/irregular/raindrop",
            },
        }
    )
    payload["schema"] = "alphabet.possible_10model.deployment_config.v1"
    payload["campaign"] = {
        "datasets": sorted(DATASETS),
        "models": sorted(MODELS),
        "protocol": "6 -> 2 -> 1; validation-only selection; three final seeds",
    }
    _write_once(destination, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    write_config(arguments.base_config, arguments.config)
    jobs = tuple(
        replace(
            job,
            comparison_group=f"{job.comparison_group}:shard-{job.candidate_rank % 3}",
            epochs=STAGE1_EPOCHS,
            estimated_seconds=job.estimated_seconds * STAGE1_EPOCHS / job.epochs,
            job_class=(
                "short"
                if job.estimated_seconds * STAGE1_EPOCHS / job.epochs < 180
                else "medium"
                if job.estimated_seconds * STAGE1_EPOCHS / job.epochs <= 1_200
                else "long"
            ),
        )
        for job in stage1_jobs()
        if (
            job.dataset in DATASETS
            and job.model in MODELS
            and (
                (job.model == "alphabet" and (job.width, job.modes) in FAST_ALPHABET_CAPACITIES)
                or (
                    job.model != "alphabet"
                    and job.architecture == FAST_ARCHITECTURES[job.model]
                    and job.width in FAST_WIDTHS
                )
            )
        )
    )
    expected_stage1 = len(DATASETS) * len(MODELS) * STAGE1_CANDIDATES_PER_CELL
    expected_stage2 = len(DATASETS) * len(MODELS) * STAGE2_TOP_K * len(STAGE2_SEEDS)
    expected_final = len(DATASETS) * len(MODELS) * len(FINAL_SEEDS)
    if len(jobs) != expected_stage1:
        message = f"expected {expected_stage1} Stage-1 jobs, found {len(jobs)}"
        raise RuntimeError(message)
    _write_once(
        arguments.root / "contract.json",
        {
            "schema": "alphabet.possible_10model_fast.contract.v1",
            "state": "stage1_frozen",
            "datasets": sorted(DATASETS),
            "models": sorted(MODELS),
            "baseline_architectures": FAST_ARCHITECTURES,
            "alphabet_capacities": sorted(FAST_ALPHABET_CAPACITIES),
            "widths": sorted(FAST_WIDTHS),
            "stage1_jobs": expected_stage1,
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_jobs": expected_stage2,
            "stage2_epochs": STAGE2_EPOCHS,
            "final_jobs": expected_final,
            "final_epochs": FINAL_EPOCHS,
            "code_sha256": code_sha256(),
            "test_policy": "official tests remain sealed until Stage-2 selection",
            "implementation": "frozen balanced-HPO 10-model implementations",
            "hardware": "secondary_gpu + two RTX 3080 Ti hosts + local_gpu dual RTX 4090",
            "pruning_policy": (
                "retain all 30 new dataset-model cells; use two capacities, "
                "one baseline architecture, and three optimizer recipes in Stage 1"
            ),
        },
    )
    deployment = plan_jobs(
        arguments.root,
        stage="stage1",
        jobs=jobs,
        config=load_deployment_config(arguments.config),
    )
    audit = audit_deployment(arguments.root, stage="stage1")
    if not audit["ok"] or deployment["blocked_jobs"]:
        message = f"irregular baseline deployment audit failed: {audit}"
        raise RuntimeError(message)
    print(
        json.dumps(
            {
                "root": str(arguments.root),
                "config": str(arguments.config),
                "deployment": deployment,
                "audit": audit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
