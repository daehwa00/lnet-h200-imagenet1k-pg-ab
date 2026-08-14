# ruff: noqa: T201
"""Freeze the new-dataset queue and its three-GPU deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from lnet.pac_broad_benchmark_distributed import (
    audit_deployment,
    load_deployment_config,
    plan_jobs,
)
from lnet.pac_broad_followup_queue import (
    EXTERNAL_TASKS,
    FOLLOWUP_ROOT,
    followup_datasets,
    stage1_jobs,
    write_campaign_contract,
)

BASE_CONFIG = Path("optimization/hosts/broad_benchmark_3gpu.local.json")
DEFAULT_CONFIG = Path("optimization/hosts/broad_followup_3gpu.local.json")
ACTIVE_ROOT = Path(".omx/results/alphabet-broad-benchmark-3gpu-20260727")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            message = f"refusing to overwrite a different artifact: {path}"
            raise FileExistsError(message)
        return
    path.write_text(content, encoding="utf-8")


def write_followup_config(source: Path, destination: Path) -> dict[str, object]:
    payload = cast("dict[str, object]", json.loads(source.read_text(encoding="utf-8")))
    hosts = cast("list[dict[str, object]]", payload["hosts"])
    all_ucr = [
        dataset.data_shard for dataset in followup_datasets() if dataset.suite == "regular"
    ]
    all_external = {
        dataset.key: dataset.data_shard
        for dataset in EXTERNAL_TASKS
    }
    for host in hosts:
        name = str(host["name"])
        host.pop("data_shard_set", None)
        host.pop("data_shards", None)
        host["repo"] = (
            "<remote-home>/lnet-broad-new-runtime-20260727"
            if name == "secondary_gpu"
            else "<remote-home>/lnet-broad-new-runtime-20260727"
        )
        external_shards = (
            ()
            if name == "secondary_gpu"
            else tuple(all_external[key] for key in sorted(all_external))
        )
        host["data_shards"] = sorted((*all_ucr, *external_shards))
    payload["schema"] = "alphabet.broad_new_datasets.deployment_config.v1"
    payload["predecessor"] = {
        "root": str(ACTIVE_ROOT),
        "release": str(ACTIVE_ROOT / "stage1" / "release.json"),
        "start_policy": "launch only after all three active Stage-1 worker PIDs exit",
    }
    payload["data_policy"] = {
        "ucr": "all 83 new UCR datasets synchronized to all hosts before launch",
        "external_cuda12_8_hosts": sorted(all_external),
        "reason": (
            "external exact-split ALPHABET requires CUDA 12.8 conditional-graph APIs; "
            "secondary_gpu has a CUDA 12.4 toolkit and runs UCR groups only"
        ),
    }
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_once(destination, content)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=FOLLOWUP_ROOT)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-audit",
        type=Path,
        default=FOLLOWUP_ROOT / "data-audit.json",
    )
    arguments = parser.parse_args()
    if not arguments.data_audit.is_file():
        message = (
            f"follow-up data audit is missing; run prepare_broad_followup_data.py: "
            f"{arguments.data_audit}"
        )
        raise FileNotFoundError(message)
    data_audit = cast(
        "dict[str, object]",
        json.loads(arguments.data_audit.read_text(encoding="utf-8")),
    )
    if data_audit.get("counts") != {"external": 8, "ucr": 83}:
        message = f"unexpected follow-up data audit counts: {data_audit.get('counts')}"
        raise RuntimeError(message)

    write_followup_config(arguments.base_config, arguments.config)
    contract = write_campaign_contract(arguments.root)
    deployment = plan_jobs(
        arguments.root,
        stage="stage1",
        jobs=stage1_jobs(),
        config=load_deployment_config(arguments.config),
    )
    audit = audit_deployment(arguments.root, stage="stage1")
    if not audit["ok"]:
        message = f"follow-up deployment audit failed: {audit}"
        raise RuntimeError(message)
    release = {
        "schema": "alphabet.broad_new_datasets.release.v1",
        "state": "queued_waiting_for_active_stage1",
        "predecessor_root": str(ACTIVE_ROOT),
        "contract_sha256": _sha256(arguments.root / "contract.json"),
        "data_audit_sha256": _sha256(arguments.data_audit),
        "deployment_sha256": _sha256(arguments.root / "stage1" / "deployment.json"),
        "assigned_jobs": deployment["assigned_jobs"],
        "blocked_jobs": deployment["blocked_jobs"],
        "comparison_groups": deployment["comparison_groups"],
        "start_policy": "automatic watcher; active Stage-1 PIDs must all terminate first",
    }
    _write_once(
        arguments.root / "stage1" / "release.json",
        json.dumps(release, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "root": str(arguments.root),
                "config": str(arguments.config),
                "expected": contract["expected"],
                "deployment": {
                    "assigned_jobs": deployment["assigned_jobs"],
                    "blocked_jobs": deployment["blocked_jobs"],
                    "units": len(cast("list[object]", deployment["units"])),
                    "estimated_wall_hours": (
                        float(cast("int | float", deployment["max_estimated_wall_seconds"]))
                        / 3_600.0
                    ),
                },
                "audit": audit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
