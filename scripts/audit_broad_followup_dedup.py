# ruff: noqa: T201
"""Prove that excluded follow-up tasks have compatible complete results."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Final, cast

from lnet.pac_broad_followup_queue import (
    FOLLOWUP_ROOT,
    REUSE_PROVENANCE,
    REUSED_EXTERNAL_DATASETS,
    followup_datasets,
    followup_models,
)

STAGES: Final = {
    "stage1": ((7,), 18, "validation"),
    "stage2": ((11, 19), 12, "validation"),
    "final": ((23, 31, 43, 47, 59), 5, "test"),
}
ALPHABET_ROOT = Path(str(REUSE_PROVENANCE["alphabet"]))
BASELINE_ROOT = Path(str(REUSE_PROVENANCE["baselines"]))


def _rows(root: Path, stage: str) -> list[tuple[Path, dict[str, object]]]:
    rows: list[tuple[Path, dict[str, object]]] = []
    for path in sorted((root / stage).rglob("*.json")):
        try:
            payload = cast(
                "dict[str, object]",
                json.loads(path.read_text(encoding="utf-8")),
            )
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("status") == "done":
            rows.append((path, payload))
    return rows


def _tree_sha256(rows: list[tuple[Path, dict[str, object]]]) -> str:
    digest = hashlib.sha256()
    for path, _ in rows:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    models = {model.key for model in followup_models()}
    baselines = models - {"alphabet"}
    pending = {dataset.key for dataset in followup_datasets()}
    problems: list[str] = []
    if pending.intersection(REUSED_EXTERNAL_DATASETS):
        problems.append("pending queue overlaps the reuse ledger")
    sources: dict[str, object] = {}
    for source_name, root, source_models in (
        ("alphabet", ALPHABET_ROOT, {"alphabet"}),
        ("baselines", BASELINE_ROOT, baselines),
    ):
        source_stages: dict[str, object] = {}
        for stage, (seeds, candidates_per_model, split) in STAGES.items():
            all_rows = _rows(root, stage)
            selected = [
                (path, row)
                for path, row in all_rows
                if row.get("dataset") in REUSED_EXTERNAL_DATASETS
                and row.get("model") in source_models
            ]
            expected = (
                len(REUSED_EXTERNAL_DATASETS)
                * len(source_models)
                * candidates_per_model
            )
            keys = [str(row.get("job_key")) for _, row in selected]
            invalid = [
                str(path)
                for path, row in selected
                if row.get("stage") != stage
                or row.get("evaluation_split") != split
                or bool(row.get("official_test_accessed")) != (stage == "final")
                or int(cast("int", row.get("split_seed", -1)))
                != int(cast("int", row.get("train_seed", -2)))
                or int(cast("int", row.get("train_seed", -1))) not in seeds
                or (
                    stage != "final"
                    and not math.isfinite(
                        float(cast("str | int | float", row["selection_score"]))
                    )
                )
            ]
            if len(selected) != expected:
                problems.append(
                    f"{source_name}/{stage}: expected {expected}, got {len(selected)}"
                )
            if len(keys) != len(set(keys)):
                problems.append(f"{source_name}/{stage}: duplicate logical job key")
            if invalid:
                problems.append(
                    f"{source_name}/{stage}: {len(invalid)} protocol-invalid rows"
                )
            if source_name == "alphabet":
                wrong_architecture = sum(
                    row.get("architecture") != "radial-log-r-affine"
                    for _, row in selected
                )
                if wrong_architecture:
                    problems.append(
                        f"alphabet/{stage}: {wrong_architecture} wrong architectures"
                    )
            source_stages[stage] = {
                "rows": len(selected),
                "expected": expected,
                "models": sorted({str(row["model"]) for _, row in selected}),
                "datasets": sorted({str(row["dataset"]) for _, row in selected}),
                "train_seeds": sorted(
                    {int(cast("int", row["train_seed"])) for _, row in selected}
                ),
                "code_sha256": sorted(
                    {str(row.get("code_sha256")) for _, row in selected}
                ),
                "result_bytes_sha256": _tree_sha256(selected),
                "protocol_invalid": len(invalid),
            }
        sources[source_name] = {
            "root": str(root),
            "stages": source_stages,
        }
    payload: dict[str, object] = {
        "schema": "alphabet.broad_new_datasets.dedup_audit.v1",
        "ok": not problems,
        "problems": problems,
        "policy": (
            "exclude a dataset only when corrected ALPHABET and all nine baselines "
            "have complete compatible 18 -> 6 -> 1 results"
        ),
        "reused_datasets": sorted(REUSED_EXTERNAL_DATASETS),
        "pending_dataset_count": len(pending),
        "pending_reuse_overlap": sorted(pending.intersection(REUSED_EXTERNAL_DATASETS)),
        "sources": sources,
    }
    output = FOLLOWUP_ROOT / "dedup-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if problems:
        message = f"follow-up de-duplication audit failed: {problems}"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
