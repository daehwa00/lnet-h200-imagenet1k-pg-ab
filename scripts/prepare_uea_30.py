from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from lnet.pac_uea_tasks import (
    UEA_30_DATASETS,
    download_uea_dataset,
    prepare_uea_task,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare original UEA-30 TRAIN/TEST tasks."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=UEA_30_DATASETS,
        default=list(UEA_30_DATASETS),
    )
    parser.add_argument("--source-root", type=Path, default=Path(".omx/data/uea-30"))
    parser.add_argument("--output-root", type=Path, default=Path("data/external"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require the original .ts files to exist locally.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    rows: list[dict[str, object]] = []
    for dataset in arguments.datasets:
        if not arguments.no_download:
            download_uea_dataset(dataset, arguments.source_root)
        row = prepare_uea_task(
            dataset,
            arguments.source_root,
            arguments.output_root,
            validation_fraction=arguments.validation_fraction,
            split_seed=arguments.split_seed,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)  # noqa: T201
    summary = {
        "schema": "lnet.uea-30-preparation-summary.v1",
        "datasets": [row["dataset"] for row in rows],
        "count": len(rows),
        "source_bytes": sum(
            sum(cast("dict[str, int]", row["source_bytes"]).values()) for row in rows
        ),
        "full_artifact_bytes": sum(
            cast("int", row["full_artifact_bytes"]) for row in rows
        ),
        "selection_artifact_bytes": sum(
            cast("int", row["selection_artifact_bytes"]) for row in rows
        ),
        "selection_full_identity_verified": all(
            cast("dict[str, object]", row["selection_full_identity"]).get("verified")
            is True
            for row in rows
        ),
    }
    path = arguments.output_root / "provenance" / "uea-30-summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
