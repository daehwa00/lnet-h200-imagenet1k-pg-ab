# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false, reportPrivateUsage=false
"""Drive the three new runnable datasets against the fixed 10-model set."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import cast

import supervise_broad_campaign as engine

from lnet import pac_fast_completion as completion

engine.ROOT = Path(".omx/results/alphabet-new3-10model-5gpu-20260727")
engine.CONFIG = Path("optimization/hosts/new3_10model_5gpu.local.json")
engine.STATE = engine.ROOT / "supervisor-state.json"
engine.audit_campaign = completion.audit_campaign
engine.select_stage1 = completion.select_stage1
engine.select_stage2 = completion.select_stage2
engine.stage_jobs = completion.stage_jobs
engine.WORKERS_PER_HOST = {
    **engine.WORKERS_PER_HOST,
    "local_gpu": 2,
}


def _verify_frozen_host(host: dict[str, object]) -> None:
    """Verify the campaign snapshot without syncing the evolving Wave-1 tree."""
    repo = str(host["repo"])
    python = str(cast("list[dict[str, object]]", host["profiles"])[0]["python"])
    command = "from lnet.pac_broad_benchmark_worker import code_sha256;print(code_sha256())"
    observed = engine._ssh(  # noqa: SLF001
        str(host["ssh_host"]),
        (
            f"cd {shlex.quote(repo)} && PYTHONPATH=src {shlex.quote(python)} -c "
            f"{shlex.quote(command)}"
        ),
        capture=True,
    )
    contract = cast(
        "dict[str, object]",
        json.loads((engine.ROOT / "contract.json").read_text(encoding="utf-8")),
    )
    expected = str(contract["code_sha256"])
    if observed != expected:
        message = (
            f"frozen campaign code hash mismatch on {host['name']}: "
            f"{observed} != {expected}"
        )
        raise RuntimeError(message)


engine._stage_host = _verify_frozen_host  # noqa: SLF001


def _write_launch(path: Path, launched: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "alphabet.possible_10model.stage1_launch.v1",
                "launched": launched,
                "workers_per_host": engine.WORKERS_PER_HOST,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _launch_stage1(host_names: set[str] | None = None) -> None:
    launch_path = engine.ROOT / "stage1/launch.json"
    hosts = engine._hosts()  # noqa: SLF001
    hosts_by_name = {str(host["name"]): host for host in hosts}
    deployment = cast(
        "dict[str, object]",
        json.loads((engine.ROOT / "stage1/deployment.json").read_text(encoding="utf-8")),
    )
    launched = (
        cast(
            "list[dict[str, object]]",
            json.loads(launch_path.read_text(encoding="utf-8"))["launched"],
        )
        if launch_path.is_file()
        else []
    )
    existing = {(str(item["host"]), str(item["lane"])) for item in launched}
    staged_hosts: set[str] = set()
    for unit in cast("list[dict[str, object]]", deployment["units"]):
        lane = cast("dict[str, object]", unit["lane"])
        host = hosts_by_name.get(str(lane["host"]))
        if host is None:
            # Historical deployment manifests can retain lanes for a host that
            # has since been disabled.  Completed artifacts remain valid; only
            # new launches must obey the current enabled-host set.
            continue
        if host_names is not None and str(host["name"]) not in host_names:
            continue
        host_name = str(host["name"])
        if host_name not in staged_hosts:
            engine._stage_host(host)  # noqa: SLF001
            staged_hosts.add(host_name)
        source = Path(str(unit["manifest"]))
        for index in range(engine.WORKERS_PER_HOST[str(host["name"])]):
            worker_lane = f"{lane['name']}-w{index}"
            if (str(host["name"]), worker_lane) in existing:
                continue
            manifest = source.with_name(f"{worker_lane}.jsonl")
            manifest.write_bytes(source.read_bytes())
            engine._sync_manifest(host, manifest)  # noqa: SLF001
            pid = engine._launch(  # noqa: SLF001
                host,
                "stage1",
                worker_lane,
                gpu=int(cast("str | int", lane["gpu"])),
            )
            launched.append(
                {
                    "host": host["name"],
                    "ssh_host": host["ssh_host"],
                    "lane": worker_lane,
                    "manifest_lane": lane["name"],
                    "pid": pid,
                }
            )
            existing.add((str(host["name"]), worker_lane))
            _write_launch(launch_path, launched)


if __name__ == "__main__":
    _launch_stage1()
    engine.main()
