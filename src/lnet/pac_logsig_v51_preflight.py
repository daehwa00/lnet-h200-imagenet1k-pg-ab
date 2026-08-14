# pyright: reportPrivateLocalImportUsage=false
"""Fail-closed runtime preflight for direct-stem LogSig V5.1."""

from __future__ import annotations

import contextlib
import io
import json
import sys

from optimization.direct_stem_three_stage_log_signature import (
    FullyPoleNativeDirectStemThreeStageALPHABET,
)

from . import pac_logsig_v5_preflight as base


def main() -> None:
    base.FullyPoleNativeThreeStageCausalALPHABET = (
        FullyPoleNativeDirectStemThreeStageALPHABET
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        base.main()
    payload = json.loads(output.getvalue())
    payload["schema"] = "pac_logsig_v51_preflight.v1"
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
