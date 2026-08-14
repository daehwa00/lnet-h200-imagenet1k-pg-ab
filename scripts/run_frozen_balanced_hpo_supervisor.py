#!/usr/bin/env python3
"""Load a frozen balanced-HPO supervisor with its execution imports intact.

The original supervisor keyed preflight manifests only by runtime profile.
That collides when several hosts use the same profile but own different model
subsets.  This launcher changes only that coordinator-side filename; remote
workers still execute the untouched hash-addressed source snapshot.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType


def _snapshot_argument() -> Path:
    try:
        index = sys.argv.index("--snapshot")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        message = "--snapshot is required"
        raise SystemExit(message) from error


def _load_supervisor(snapshot: Path) -> ModuleType:
    sys.path.insert(0, str(snapshot))
    sys.path.insert(0, str(snapshot / "src"))
    path = snapshot / "scripts" / "supervise_balanced_hpo_27task.py"
    spec = importlib.util.spec_from_file_location("_frozen_balanced_hpo_supervisor", path)
    if spec is None or spec.loader is None:
        message = f"cannot load frozen supervisor: {path}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_host_scoped_preflight(module: ModuleType) -> None:
    existing = module._profile_preflight_manifest  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]
    if len(inspect.signature(existing).parameters) >= 5:
        return

    def host_scoped_manifest(
        root: Path,
        stage: str,
        profile: str,
        units: list[dict[str, object]],
    ) -> Path:
        representatives: dict[str, object] = {}
        hosts: set[str] = set()
        for unit in units:
            lane = cast("dict[str, object]", unit["lane"])
            hosts.add(str(lane["host"]))
            for job in module._unit_jobs(unit):  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]
                representatives.setdefault(job.model, job)
        if len(hosts) != 1:
            message = f"preflight group spans multiple hosts: {sorted(hosts)}"
            raise RuntimeError(message)
        host = next(iter(hosts))
        path = root / stage / "preflight" / f"{host}-{profile}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(job.payload(), sort_keys=True) + "\n"
            for job in representatives.values()
        )
        if path.exists() and path.read_text(encoding="utf-8") != content:
            message = f"preflight manifest changed after creation: {path}"
            raise FileExistsError(message)
        path.write_text(content, encoding="utf-8")
        return path

    module._profile_preflight_manifest = host_scoped_manifest  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]


def main() -> None:
    module = _load_supervisor(_snapshot_argument())
    _install_host_scoped_preflight(module)
    module.main()  # pyright: ignore[reportAttributeAccessIssue]


if __name__ == "__main__":
    main()
