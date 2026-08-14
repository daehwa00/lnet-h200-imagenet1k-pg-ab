"""Fast UCR-16 validation screen for the square-modal H-compact D=8, M=4 cell."""

# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import pac_h_compact_lag124_m4_ucr16_fast as campaign

ROOT = Path(".omx/results/pac-h-compact-lag124-d8m4-ucr16-fast-v2-20260721")
VARIANT = "h_compact_lag124_d8_m4"
_base_design = campaign._design  # pyright: ignore[reportPrivateUsage]


def _design() -> dict[str, object]:
    body = _base_design()
    source = Path(__file__).resolve()
    body.update(
        {
            "schema": "pac_h_compact_lag124_d8m4_ucr16_fast_contract.v1",
            "purpose": "validation-only square-modal D=8,M=4 boundary screen",
            "model_dims": [8],
            "modes": 4,
        }
    )
    hashes = dict(body["source_sha256"])
    hashes[str(source.relative_to(source.parents[1]))] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    body["source_sha256"] = hashes
    return body


def main() -> None:
    campaign.ROOT = ROOT
    campaign.MODEL_DIMS = (8,)
    campaign.VARIANTS = (VARIANT,)
    campaign._design = _design  # pyright: ignore[reportPrivateUsage]
    campaign.main()


if __name__ == "__main__":
    main()
