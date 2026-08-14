# ruff: noqa: T201
"""Execute one restart-safe broad-benchmark GPU manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

from lnet.pac_broad_benchmark_queue import DEFAULT_ROOT
from lnet.pac_broad_benchmark_worker import BroadDataRoots, run_manifest

if TYPE_CHECKING:
    from lnet.pac_types import PACDevice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ucr-data-root", type=Path, required=True)
    parser.add_argument("--external-data-root", type=Path, required=True)
    parser.add_argument("--physionet2012-data-root", type=Path)
    parser.add_argument("--raindrop-data-root", type=Path)
    parser.add_argument("--max-attempts", type=int, default=3)
    arguments = parser.parse_args()

    summary = run_manifest(
        arguments.root,
        arguments.manifest,
        device=cast("PACDevice", arguments.device),
        data_roots=BroadDataRoots(
            ucr=arguments.ucr_data_root,
            external=arguments.external_data_root,
            physionet2012=arguments.physionet2012_data_root,
            raindrop=arguments.raindrop_data_root,
        ),
        max_attempts=arguments.max_attempts,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
