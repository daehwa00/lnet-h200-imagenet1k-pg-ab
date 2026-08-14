"""Return success when every logical key in one worker manifest is terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _result_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for result in path.glob("*.json"):
        try:
            row = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("job_key") is not None:
            keys.add(str(row["job_key"]))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-failed", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected = {str(row["key"]) for row in rows}
    stages = {str(row["job"]["stage"]) for row in rows}
    if len(stages) != 1 or not expected:
        return 2
    stage = next(iter(stages))
    terminal = _result_keys(args.output_root / stage / "completed")
    if args.allow_failed:
        terminal |= _result_keys(args.output_root / stage / "failed")
    return 0 if expected <= terminal else 1


if __name__ == "__main__":
    raise SystemExit(main())
