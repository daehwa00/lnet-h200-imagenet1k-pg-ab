"""Backfill selection hashes omitted by distributed Q1 workers.

Remote workers received stage-local directories but not the preceding selection
artifact, so otherwise valid rows recorded ``selection_artifact_sha256=null``.
This repair only fills null links after independently validating each row
against the selected config grid.  Existing non-null mismatches are rejected.
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

DEFAULT_ROOT = Path(".omx/results/pac-two-tap-q1-final-20260720")
CANDIDATE = "two_tap_h_only"


def _read(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _repair_row(
    path: Path,
    *,
    expected_hash: str,
    expected_configs: dict[str, set[str]],
    expected_seeds: set[int],
    apply: bool,
) -> dict[str, str] | None:
    row = _read(path)
    cell_key = str(row.get("cell_key"))
    config_key = str(row.get("config_key"))
    seed_value = row.get("train_seed")
    seed = seed_value if isinstance(seed_value, int) else -1
    if (
        row.get("model") != CANDIDATE
        or row.get("status") != "done"
        or row.get("official_test_accessed") not in {False, True}
        or config_key not in expected_configs.get(cell_key, set())
        or seed not in expected_seeds
    ):
        message = f"row cannot be linked to the selected grid: {path}"
        raise RuntimeError(message)
    recorded = row.get("selection_artifact_sha256")
    if recorded == expected_hash:
        return None
    if recorded is not None:
        message = f"refusing to replace a non-null selection hash: {path}"
        raise RuntimeError(message)
    original_sha256 = _sha256(path)
    row["selection_artifact_sha256"] = expected_hash
    encoded = json.dumps(row, indent=2, sort_keys=True) + "\n"
    repaired_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
    if apply:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    return {
        "path": str(path),
        "job_key": str(row["job_key"]),
        "original_sha256": original_sha256,
        "repaired_sha256": repaired_sha256,
        "selection_artifact_sha256": expected_hash,
    }


def repair(root: Path, *, apply: bool) -> dict[str, object]:
    stage1_path = root / "stage1/selection.json"
    stage2_path = root / "stage2/selection.json"
    stage1 = _read(stage1_path)
    stage2 = _read(stage2_path)
    stage1_selected = stage1.get("selected")
    stage2_selected = stage2.get("selected")
    if not isinstance(stage1_selected, dict) or not isinstance(stage2_selected, dict):
        message = "selection artifacts are missing their selected grids"
        raise TypeError(message)

    active_stage1 = cast("dict[str, list[object]]", stage1_selected)
    active_stage2 = cast("dict[str, dict[str, object]]", stage2_selected)
    stage2_expected = {
        str(cell_key): {str(config) for config in configs}
        for cell_key, configs in active_stage1.items()
    }
    final_expected = {
        str(cell_key): {str(selection["config_key"])}
        for cell_key, selection in active_stage2.items()
    }
    repairs: list[dict[str, str]] = []
    for path in sorted((root / "stage2/completed").glob("*.json")):
        row = _read(path)
        if row.get("model") != CANDIDATE:
            continue
        repaired = _repair_row(
            path,
            expected_hash=_sha256(stage1_path),
            expected_configs=stage2_expected,
            expected_seeds={11, 19},
            apply=apply,
        )
        if repaired is not None:
            repairs.append(repaired)
    for path in sorted((root / "final/completed").glob("*.json")):
        row = _read(path)
        if row.get("model") != CANDIDATE:
            continue
        repaired = _repair_row(
            path,
            expected_hash=_sha256(stage2_path),
            expected_configs=final_expected,
            expected_seeds={23, 31, 43, 47, 59},
            apply=apply,
        )
        if repaired is not None:
            repairs.append(repaired)

    payload: dict[str, object] = {
        "schema": "pac_two_tap_q1_selection_link_repair.v1",
        "status": "complete" if apply else "dry-run",
        "reason": (
            "distributed workers received the active stage directory but not the preceding "
            "selection artifact; null links were reconstructed only after exact "
            "config-grid validation"
        ),
        "stage1_selection_sha256": _sha256(stage1_path),
        "stage2_selection_sha256": _sha256(stage2_path),
        "repaired_rows": len(repairs),
        "repairs": repairs,
    }
    if apply:
        _write_json(root / "audit/selection_link_repair.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(args.campaign_root, apply=args.apply), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
